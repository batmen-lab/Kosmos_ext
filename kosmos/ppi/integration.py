"""Narrow adapters for the existing evidence pipeline; training ignores source modality."""

from pathlib import Path

import numpy as np

from kosmos.evidence import Prediction, ValidationResult

from .schemas import ExternalEvidenceDataset
from .trainer import linear_model_factory, run_ppi_experiment


class RoutingPredictor:
    def __init__(self, prepared):
        self.prepared = prepared
        self.gold_dataset_id = prepared.gold_dataset_id
        self.model_reference = prepared.model_reference
        self.code_reference = prepared.code_reference
        self.prediction_provenance = prepared.provenance

    def predict(self, features):
        labels, probabilities = self.prepared.predict_with_probabilities(
            features.to_numpy(dtype=np.float32)
        )
        return Prediction(labels, probabilities, self.prepared.classes)


class PPIEvidenceTrainer:
    def __init__(
        self,
        gold_train,
        gold_validation,
        prepared,
        config,
        output_dir,
        model_factory=linear_model_factory,
    ):
        self.gold_train, self.gold_validation = gold_train, gold_validation
        self.prepared, self.config = prepared, config
        self.output_dir = Path(output_dir)
        self.model_factory = model_factory
        self.results = []

    def evaluate(self, target, evidence, config):
        if (
            target.gold_dataset_id != self.gold_train.dataset_id
            or target.feature_names != self.gold_train.feature_names
        ):
            raise ValueError("Routing target differs from PPI gold profile")
        if config.seed != self.config.seed:
            raise ValueError("Routing and PPI sampling seeds must agree")
        external = []
        for batch in evidence:
            record = batch.record
            if record.decision != "accept" or record.route == "REJECT":
                raise ValueError("PPI accepts only successfully prepared evidence")
            candidate = record.candidate
            indices = np.asarray(record.execution["selected_row_ids"], dtype=int)
            # Do not invent global biological identities from dataset-local row numbers.
            if "sample_ids" not in candidate.metadata:
                raise ValueError("PPI integration requires global sample_ids in candidate.metadata")
            sample_ids = np.asarray(candidate.metadata["sample_ids"], dtype=str)
            if len(sample_ids) != record.execution["original_rows"]:
                raise ValueError("Candidate sample_ids must describe all original rows")
            groups = candidate.metadata.get("groups")
            if groups is not None and len(groups) != len(sample_ids):
                raise ValueError("Candidate groups must describe all original rows")
            if record.execution["model_reference"] != self.prepared.model_reference:
                raise ValueError("Routing pseudo-labeler differs from PPI pseudo-labeler")
            predicted, probability = self.prepared.predict_with_probabilities(
                batch.features.to_numpy(dtype=np.float32)
            )
            if not np.array_equal(predicted, batch.pseudo_labels.to_numpy()):
                raise ValueError("Routing pseudo predictions differ from PPI pseudo predictions")
            if probability is not None and (
                batch.probabilities is None or not np.allclose(probability, batch.probabilities)
            ):
                raise ValueError("Routing probabilities differ from PPI pseudo probabilities")
            external.append(
                ExternalEvidenceDataset(
                    X=batch.features.to_numpy(),
                    sample_ids=sample_ids[indices],
                    evidence_route=record.route,
                    source_dataset_id=candidate.dataset_id,
                    feature_names=list(batch.features.columns),
                    evidence_weight=batch.weights,
                    groups=np.asarray(groups)[indices] if groups is not None else None,
                    audit_metadata=record.model_dump(mode="json"),
                    source_labels=(
                        batch.source_labels.to_numpy() if batch.source_labels is not None else None
                    ),
                )
            )
        training_config = self.config.model_copy(
            update={
                "external_weight_budget": min(
                    config.external_weight_budget, self.config.external_weight_budget
                ),
                "max_rows_per_dataset": min(
                    config.max_rows_per_dataset, self.config.max_rows_per_dataset
                ),
            }
        )
        result = run_ppi_experiment(
            gold_train=self.gold_train,
            gold_validation=self.gold_validation,
            external_evidence=external,
            pseudo_labeler=self.prepared,
            config=training_config,
            model_factory=self.model_factory,
            output_dir=self.output_dir / f"evaluation-{len(self.results):03d}",
        )
        self.results.append(result)
        metric = self.config.evaluation_metric
        if result.validation_metrics[metric] is None:
            metric = "balanced_accuracy"
        return ValidationResult(
            split_role="validation",
            split_id=result.reproducibility["validation_fingerprint"],
            protocol="fixed held-out gold_validation; gold-only pseudo-label cross-fitting or reviewed frozen model",
            metric=metric,
            baseline=result.baseline_metrics[metric],
            augmented=result.validation_metrics[metric],
        )
