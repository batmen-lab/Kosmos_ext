"""NPZ/JSON CLI and a deterministic synthetic software smoke experiment."""

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .schemas import ExternalEvidenceDataset, GoldDataset, PPITrainingConfig
from .trainer import run_ppi_experiment


def load_dataset(path, external=False):
    # JSON metadata and non-object arrays: no unpickling user-supplied datasets.
    with np.load(path, allow_pickle=False) as data:
        fields = {k: data[k] for k in data.files}
    for name in ("role", "dataset_id", "source_dataset_id", "evidence_route", "representation"):
        if name in fields:
            fields[name] = fields[name].item()
    for name in ("metadata", "audit_metadata"):
        if name in fields:
            fields[name] = json.loads(fields[name].item())
    if "feature_names" in fields:
        fields["feature_names"] = fields["feature_names"].tolist()
    return ExternalEvidenceDataset(**fields) if external else GoldDataset(**fields)


def synthetic_data(seed=42):
    X, y = make_classification(
        n_samples=480, n_features=12, n_informative=7, n_redundant=2, n_classes=3, random_state=seed
    )
    names = [f"feature-{i}" for i in range(X.shape[1])]
    train = GoldDataset(
        X[:90],
        y[:90],
        np.array([f"train-{i}" for i in range(90)]),
        "gold_train",
        "synthetic-gold",
        names,
    )
    validation = GoldDataset(
        X[90:180],
        y[90:180],
        np.array([f"val-{i}" for i in range(90)]),
        "gold_validation",
        "synthetic-gold",
        names,
    )
    external = ExternalEvidenceDataset(
        X[180:],
        np.array([f"external-{i}" for i in range(300)]),
        "DIRECT_PPI",
        "synthetic-external",
        names,
    )
    return train, validation, [external]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gold-train", type=Path)
    parser.add_argument("--gold-validation", type=Path)
    parser.add_argument("--external-evidence", nargs="*", type=Path, default=[])
    parser.add_argument("--config", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--synthetic", action="store_true")
    args = parser.parse_args()
    config = (
        PPITrainingConfig.model_validate_json(args.config.read_text())
        if args.config
        else PPITrainingConfig()
    )
    if config.pseudo_mode == "pretrained":
        parser.error(
            "Reviewed pretrained classifiers use the Python API; CLI never loads arbitrary pickle models"
        )
    if args.synthetic:
        if args.gold_train or args.gold_validation or args.external_evidence:
            parser.error("--synthetic cannot be mixed with real input files")
        train, validation, external = synthetic_data(config.seed)
    else:
        if not args.gold_train or not args.gold_validation:
            parser.error("Provide --gold-train and --gold-validation, or --synthetic")
        train, validation = load_dataset(args.gold_train), load_dataset(args.gold_validation)
        external = [load_dataset(p, external=True) for p in args.external_evidence]
    # Scaling is fit inside every cross-fit fold, never on validation or external data.
    pseudo = make_pipeline(
        StandardScaler(), LogisticRegression(max_iter=1000, random_state=config.seed)
    )
    result = run_ppi_experiment(
        gold_train=train,
        gold_validation=validation,
        external_evidence=external,
        pseudo_labeler=pseudo,
        config=config,
        output_dir=args.output_dir,
    )
    print(
        json.dumps(
            {
                "baseline": result.baseline_metrics,
                "ppi": result.validation_metrics,
                "delta": result.delta_metrics,
                "result": str(args.output_dir / "result.json"),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
