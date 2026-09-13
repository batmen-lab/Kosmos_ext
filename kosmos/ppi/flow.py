"""In-process PPI classification flow for Kosmos experiment execution.

This is the bridge that lets ResearchDirectorAgent/CLI treat an external
unlabeled dataset as PPI evidence:

    gold (train donor)  --80%--> gold_train / 20% gold_validation
    gold (test donor)   -------------------> final_test
    external unlabeled  --> pseudo-labels from frozen gold-only cross-fit
                         --> signed PPILoss correction

The flow never reads external source labels as training labels.
"""

import json
import time
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .schemas import ExternalEvidenceDataset, GoldDataset, PPITrainingConfig
from .trainer import evaluate_final_test, prepare_pseudo_labeler, run_ppi_experiment
from .split import split_gold


def _gold_from_frame(
    df: pd.DataFrame,
    feature_names,
    role: str,
    dataset_id: str,
    groups: bool,
):
    return GoldDataset(
        X=df[feature_names].astype("float32").to_numpy(),
        y=df["cell_type"].to_numpy(dtype=str),
        sample_ids=df["cell_id"].astype(str).to_numpy(),
        role=role,
        dataset_id=dataset_id,
        feature_names=feature_names,
        groups=df["DonorID"].astype(str).to_numpy() if groups else None,
    )


def _external_by_donor(
    path: str,
    donor_list,
    feature_names,
    per_donor_limit: int,
):
    keep = ["cell_id", "DonorID"] + feature_names
    datasets = []
    for donor in donor_list:
        chunks = []
        for chunk in pd.read_csv(
            path,
            usecols=keep,
            dtype={"cell_id": str, "DonorID": str},
            chunksize=200_000,
        ):
            sub = chunk[chunk["DonorID"] == donor]
            if len(sub):
                chunks.append(sub)
            if sum(len(c) for c in chunks) >= per_donor_limit:
                break
        if not chunks:
            continue
        df = pd.concat(chunks, ignore_index=True).head(per_donor_limit)
        datasets.append(
            ExternalEvidenceDataset(
                X=df[feature_names].astype("float32").to_numpy(),
                sample_ids=df["cell_id"].astype(str).to_numpy(),
                evidence_route="DIRECT_PPI",
                source_dataset_id=f"external-{donor}",
                feature_names=feature_names,
                groups=df["DonorID"].astype(str).to_numpy(),
                audit_metadata={"source": str(path)},
            )
        )
    return datasets


def run_cross_donor_ppi_classification(
    *,
    gold_test_csv: str,
    external_csv: str,
    output_dir: str,
    train_donor: str = "13272",
    test_donor: str = "19593",
    external_donors=None,
    seed: int = 42,
    validation_fraction: float = 0.2,
    per_donor_limit: int = 5000,
    max_external_samples: int = 20000,
    external_weight_budget: float = 0.5,
    max_epochs: int = 20,
    patience: int = 5,
    cross_fit_folds: int = 3,
    stage1_epochs: int = 4,
    stage2_epochs: int = 1,
    pseudo_mode: str = "cross_fit",
    model_factory: Optional[Callable] = None,
    model_design: Optional[dict] = None,
) -> dict:
    """Run the gold-only vs PPI comparison and evaluate final donor test."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    if any(out.iterdir()):
        raise ValueError(f"Output dir must be empty/new: {out}")

    gold_df = pd.read_csv(
        gold_test_csv, dtype={"cell_id": str, "DonorID": str, "cell_type": str}
    )
    feature_names = [c for c in gold_df.columns if c.startswith("ENSG")]

    train_df = gold_df[gold_df["DonorID"] == train_donor]
    test_df = gold_df[gold_df["DonorID"] == test_donor]
    if len(train_df) == 0 or len(test_df) == 0:
        raise ValueError("gold_test_csv must contain both train and test donors")

    gold_full = _gold_from_frame(
        train_df, feature_names, "gold_train", f"donor-{train_donor}", groups=False
    )
    gold_train, gold_validation = split_gold(
        gold_full, validation_fraction=validation_fraction, seed=seed
    )
    final_test = _gold_from_frame(
        test_df, feature_names, "final_test", f"donor-{test_donor}", groups=True
    )

    external = _external_by_donor(
        external_csv,
        list(external_donors or ["10886", "11466", "12710", "15078", "16710", "18303", "28045"]),
        feature_names,
        per_donor_limit=per_donor_limit,
    )
    if not external:
        raise ValueError("No external donor rows were loaded")

    config = PPITrainingConfig(
        seed=seed,
        max_epochs=max_epochs,
        patience=patience,
        cross_fit_folds=cross_fit_folds,
        max_external_samples=max_external_samples,
        external_weight_budget=external_weight_budget,
        max_rows_per_dataset=per_donor_limit,
        evaluation_metric="balanced_accuracy",
        stage1_epochs=stage1_epochs,
        stage2_epochs=stage2_epochs,
        pseudo_mode=pseudo_mode,
    )
    pseudo = make_pipeline(
        StandardScaler(), LogisticRegression(max_iter=1000, random_state=seed)
    )

    t0 = time.time()
    ppi_kwargs = dict(
        gold_train=gold_train,
        gold_validation=gold_validation,
        external_evidence=external,
        pseudo_labeler=pseudo,
        config=config,
        output_dir=str(out),
    )
    if model_factory is not None:
        ppi_kwargs["model_factory"] = model_factory
    result = run_ppi_experiment(**ppi_kwargs)
    final = evaluate_final_test(result, final_test)
    summary = {
        "mode": "ppi_external_augmented",
        "training_donor": train_donor,
        "test_donor": test_donor,
        "n_gold_train": len(gold_train.X),
        "n_gold_validation": len(gold_validation.X),
        "n_final_test": len(final_test.X),
        "external_sources": {d.source_dataset_id: len(d.X) for d in external},
        "external_used": len(result.reproducibility["external_ids"]),
        "validation_metrics": {
            "baseline": result.baseline_metrics,
            "ppi": result.validation_metrics,
            "delta": result.delta_metrics,
        },
        "final_test_metrics": final,
        "selected_epochs": result.selected_epochs,
        "stage_epochs": {"true_loss": stage1_epochs, "correction": stage2_epochs},
        "model_design": model_design or {},
        "training_seconds": round(time.time() - t0, 1),
        "config_sha256": result.reproducibility["config_sha256"],
        "artifact_dir": str(out),
    }
    (out / "ppi_summary.json").write_text(
        json.dumps(summary, indent=2, default=str), encoding="utf-8"
    )
    return summary
