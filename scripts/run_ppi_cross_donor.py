"""Cross-donor PPI test on the real gold/external data (offline).

Data roles:
  gold_train       = DonorID 13272 (80%, stratified cell-level split)
  gold_validation  = DonorID 13272 (20%, model selection / early stopping)
  final_test       = DonorID 19593 (explicit post-selection evaluation)
  external evidence= DonorID 10886/11466/12710/15078/16710/18303/28045
                     (unlabeled; source labels are NOT used)

Run (stable Python 3.11, offline, CPU):
    OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
    .venv/bin/python scripts/run_ppi_cross_donor.py

The PPI trainer materializes pseudo labels from a gold-only cross-fitted
classifier and compares gold-only vs PPI-augmented training on the same
validation protocol, then reports both on the final donor 19593.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from kosmos.ppi import (  # noqa: E402
    ExternalEvidenceDataset,
    GoldDataset,
    PPITrainingConfig,
    evaluate_final_test,
    prepare_pseudo_labeler,
    run_ppi_experiment,
    split_gold,
)


def read_gold_labeled(path: str, role: str, dataset_id: str, feature_names, include_groups: bool):
    df = pd.read_csv(path, dtype={"cell_id": str, "DonorID": str, "cell_type": str})
    X = df[feature_names].astype("float32").to_numpy()
    return GoldDataset(
        X=X,
        y=df["cell_type"].to_numpy().astype(str),
        sample_ids=df["cell_id"].to_numpy().astype(str),
        role=role,
        dataset_id=dataset_id,
        feature_names=feature_names,
        groups=df["DonorID"].astype(str).to_numpy() if include_groups else None,
    )


def read_external_by_donor(
    path: str,
    donor_list,
    feature_names,
    per_donor_limit: int,
):
    """Read up to per_donor_limit rows per donor from the unlabeled CSV."""
    keep = ["cell_id", "DonorID"] + feature_names
    datasets = []
    for donor in donor_list:
        chunks = []
        reader = pd.read_csv(
            path, usecols=keep, dtype={"cell_id": str, "DonorID": str}, chunksize=200_000
        )
        for chunk in reader:
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
                sample_ids=df["cell_id"].to_numpy().astype(str),
                evidence_route="DIRECT_PPI",
                source_dataset_id=f"external-{donor}",
                feature_names=feature_names,
                groups=df["DonorID"].astype(str).to_numpy(),
                audit_metadata={"source": "external_unlabeled.csv"},
            )
        )
    return datasets


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--gold-path",
        default=str(ROOT / "data/csv_donor_splits/13272_gold_train.csv"),
    )
    parser.add_argument(
        "--final-test-path",
        default=str(ROOT / "data/csv_donor_splits/19593_test.csv"),
    )
    parser.add_argument(
        "--external-path",
        default=str(ROOT / "data/csv_donor_splits/external_unlabeled.csv"),
    )
    parser.add_argument(
        "--output-dir",
        default=str(ROOT / f"artifacts/ppi/cross-donor-{int(time.time())}"),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--external-per-donor", type=int, default=5000)
    parser.add_argument("--max-external-samples", type=int, default=20000)
    parser.add_argument("--external-weight-budget", type=float, default=0.5)
    parser.add_argument("--max-epochs", type=int, default=20)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--cross-fit-folds", type=int, default=3)
    args = parser.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    if any(out.iterdir()):
        raise SystemExit(f"Output dir must be empty/new: {out}")

    print("Loading gold train header...", flush=True)
    header = pd.read_csv(args.gold_path, nrows=1)
    feature_names = [c for c in header.columns if c.startswith("ENSG")]
    print(f"Features: {len(feature_names)} ENSG genes", flush=True)

    print("Loading gold donor 13272...", flush=True)
    gold_full = read_gold_labeled(
        args.gold_path, "gold_train", "donor-13272", feature_names, include_groups=False
    )
    gold_train, gold_validation = split_gold(gold_full, validation_fraction=0.2, seed=args.seed)
    print(f"gold train={len(gold_train.X)} validation={len(gold_validation.X)}", flush=True)

    print("Loading final test donor 19593...", flush=True)
    final_test = read_gold_labeled(
        args.final_test_path,
        "final_test",
        "donor-19593",
        feature_names,
        include_groups=True,
    )
    print(f"final test={len(final_test.X)}", flush=True)

    external_donors = ["10886", "11466", "12710", "15078", "16710", "18303", "28045"]
    print("Loading external unlabeled donors (capped per donor)...", flush=True)
    external = read_external_by_donor(
        args.external_path,
        external_donors,
        feature_names,
        per_donor_limit=args.external_per_donor,
    )
    print(
        "external sources: "
        + ", ".join(f"{d.source_dataset_id}={len(d.X)}" for d in external),
        flush=True,
    )

    config = PPITrainingConfig(
        seed=args.seed,
        max_epochs=args.max_epochs,
        patience=args.patience,
        cross_fit_folds=args.cross_fit_folds,
        max_external_samples=args.max_external_samples,
        external_weight_budget=args.external_weight_budget,
        max_rows_per_dataset=args.external_per_donor,
        evaluation_metric="balanced_accuracy",
    )
    pseudo = make_pipeline(
        StandardScaler(), LogisticRegression(max_iter=1000, random_state=args.seed)
    )

    print("\nRunning PPI experiment (gold-only baseline vs augmented)...", flush=True)
    t0 = time.time()
    result = run_ppi_experiment(
        gold_train=gold_train,
        gold_validation=gold_validation,
        external_evidence=external,
        pseudo_labeler=pseudo,
        config=config,
        output_dir=str(out),
    )
    print(f"Training complete in {time.time() - t0:.1f}s", flush=True)

    print("\nEvaluating on final donor 19593...", flush=True)
    final = evaluate_final_test(result, final_test)
    summary = {
        "config": {
            "seed": args.seed,
            "max_epochs": args.max_epochs,
            "patience": args.patience,
            "cross_fit_folds": args.cross_fit_folds,
            "external_weight_budget": args.external_weight_budget,
            "max_external_samples": args.max_external_samples,
            "per_donor_cap": args.external_per_donor,
        },
        "n_gold_train": len(gold_train.X),
        "n_gold_validation": len(gold_validation.X),
        "n_final_test": len(final_test.X),
        "external_sources": {d.source_dataset_id: len(d.X) for d in external},
        "validation_metrics": {
            "baseline": result.baseline_metrics,
            "ppi": result.validation_metrics,
            "delta": result.delta_metrics,
        },
        "final_test_metrics": final,
        "selected_epochs": result.selected_epochs,
        "reproducibility": {
            "config_sha256": result.reproducibility["config_sha256"],
            "seed": result.reproducibility["seed"],
        },
    }
    (out / "cross_donor_summary.json").write_text(
        json.dumps(summary, indent=2, default=str), encoding="utf-8"
    )

    print("\n=== VALIDATION (held-out from 13272) ===")
    for name in ("accuracy", "balanced_accuracy", "macro_f1"):
        b = result.baseline_metrics[name]
        p = result.validation_metrics[name]
        d = result.delta_metrics[name]
        print(f"{name:18s} baseline={b:.4f}  ppi={p:.4f}  delta={d:+.4f}")

    print("\n=== FINAL TEST (donor 19593) ===")
    for name in ("accuracy", "balanced_accuracy", "macro_f1"):
        b = final["baseline"][name]
        p = final["ppi"][name]
        print(f"{name:18s} baseline={b:.4f}  ppi={p:.4f}  delta={p - b:+.4f}")

    print(f"\nArtifacts written to {out}", flush=True)


if __name__ == "__main__":
    main()
