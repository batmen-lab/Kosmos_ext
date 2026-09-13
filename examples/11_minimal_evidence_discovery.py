"""Minimal end-to-end run using the extended Kosmos evidence/PPI path.

This script exercises the newest extension on top of the original framework:

    ResearchWorkflow (1 research cycle)
        -> plan/review (mock, offline)          # original loop
        -> external evidence routing            # kosmos.evidence
        -> frozen gold-only pseudo-labeling     # kosmos.ppi
        -> PPI training vs matched gold baseline # kosmos.ppi

Everything is synthetic and offline, so no API key is needed.  It is a
mechanism/smoke demonstration, not a real biological claim.  Swap the synthetic
CSV/GoldDataset construction for your real data when you are ready.

Run with the repository venv (stable Python 3.11 + PyTorch):

    .venv/bin/python examples/11_minimal_evidence_discovery.py

Expected result: cycle completes, evidence records 1 accepted DIRECT_PPI
dataset, and PPI baseline vs augmented delta is reported (typically 0 on this
toy data; that is expected and only validates the pipeline).

To turn on real LLM-driven research, pass an anthropic_client into
ResearchWorkflow(...) (for example the client object used by the rest of Kosmos)
and provide real CandidateDataset manifests.  See docs/evidence_routing.md and
docs/ppi_training.md before treating any result as scientific evidence.
"""

import asyncio
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from kosmos.evidence import (
    Assessment,
    CandidateDataset,
    EvidencePipeline,
    PPIConfig,
    TargetDataProfile,
)
from kosmos.ppi import (
    GoldDataset,
    PPIEvidenceTrainer,
    PPITrainingConfig,
    RoutingPredictor,
    prepare_pseudo_labeler,
)
from kosmos.workflow.research_loop import ResearchWorkflow


SEED = 42


def build_demo(root: Path):
    """Return (gold_train, gold_validation, candidate, target)."""
    X, y = make_classification(
        n_samples=480,
        n_features=12,
        n_informative=7,
        n_redundant=2,
        n_classes=3,
        random_state=SEED,
    )
    feature_names = [f"feature-{i}" for i in range(X.shape[1])]

    gold_train = GoldDataset(
        X[:90],
        y[:90],
        np.array([f"gold-train-{i}" for i in range(90)]),
        "gold_train",
        "demo-gold",
        feature_names,
    )
    gold_validation = GoldDataset(
        X[90:180],
        y[90:180],
        np.array([f"gold-val-{i}" for i in range(90)]),
        "gold_validation",
        "demo-gold",
        feature_names,
    )

    external_ids = [f"external-{i}" for i in range(300)]
    external = pd.DataFrame(X[180:], columns=feature_names)
    external.insert(0, "sample_id", external_ids)
    csv_path = root / "external.csv"
    external.to_csv(csv_path, index=False)

    target = TargetDataProfile(
        prediction_task="cell-state classification",
        label_ontology="demo-cell-states",
        observation_unit="cell",
        organism="homo-sapiens",
        biological_system="blood",
        condition="demo",
        experimental_context="synthetic demo",
        target_modality="rna",
        feature_names=feature_names,
        preprocessing="standardized",
        gold_dataset_id="demo-gold",
    )
    candidate = CandidateDataset(
        dataset_id="demo-external-batch",
        source="local-cache",
        file_path=str(csv_path),
        source_modality="rna",
        observation_unit="cell",
        feature_names=feature_names,
        preprocessing="standardized",
        biological_relevance=Assessment(
            status="accept",
            rationale="Rows are drawn from the same simulated population for demo purposes.",
            evidence=["demo-data-source"],
        ),
        provenance=["generated-by-minimal-demo"],
        sample_size=300,
        metadata={"sample_ids": external_ids},
    )
    return gold_train, gold_validation, candidate, target


def main():
    root = Path(tempfile.mkdtemp(prefix="kosmos-discovery-demo-"))

    gold_train, gold_validation, candidate, target = build_demo(root)

    training_config = PPITrainingConfig(seed=SEED)
    pseudo_model = make_pipeline(
        StandardScaler(), LogisticRegression(max_iter=1000, random_state=SEED)
    )
    prepared = prepare_pseudo_labeler(gold_train, pseudo_model, training_config)
    predictor = RoutingPredictor(prepared)
    trainer = PPIEvidenceTrainer(
        gold_train,
        gold_validation,
        prepared,
        training_config,
        output_dir=str(root / "ppi-trainer"),
    )
    ppi_config = PPIConfig(
        implementation="kosmos.ppi/minimal-demo",
        code_reference="kosmos@local",
        parameters={"note": "synthetic demo"},
        external_weight_budget=0.5,
        max_rows_per_dataset=10000,
        seed=SEED,
    )
    pipeline = EvidencePipeline(target=target)

    workflow = ResearchWorkflow(
        research_objective=(
            "Demo: does external non-gold RNA evidence improve prediction "
            "of demo cell states?"
        ),
        anthropic_client=None,  # offline mock planning; pass a client for live LLM research
        artifacts_dir=str(root / "artifacts"),
        seed=SEED,
        evidence_pipeline=pipeline,
        evidence_candidates=[candidate],
        evidence_predictor=predictor,
        evidence_trainer=trainer,
        evidence_config=ppi_config,
        evidence_output_dir=str(root / "workflow-evidence"),
    )

    result = asyncio.run(workflow.run(num_cycles=1, tasks_per_cycle=5))

    print("\n=== Minimal discovery run complete ===")
    print(f"cycles completed      : {result.get('cycles_completed')}")
    print(f"evidence enabled      : {result['evidence']['enabled']}")
    print(f"evidence runs         : {len(result['evidence']['runs'])}")
    if result["evidence"]["runs"]:
        ev = result["evidence"]["runs"][0]
        print(f"decision counts       : accepted={ev['accepted']} "
              f"deferred={ev['deferred']} rejected={ev['rejected']}")
        print(f"PPI evaluations       : {ev['evaluations']}")
        print(f"evidence artifacts    : {ev['artifact_dir']}")
    print(f"all artifacts         : {root}")


if __name__ == "__main__":
    main()
