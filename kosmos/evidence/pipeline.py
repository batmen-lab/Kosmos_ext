"""Explicit adapters connect retrieved files to an existing PPI implementation.

No translator, gold-label promotion, model training or PPI algorithm is invented
here. Adapters must implement the documented scientific contracts.
"""

import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from hashlib import sha256
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import pandas as pd

from kosmos.execution.data_provider import DataProvider

from .models import (
    CandidateDataset,
    PPIConfig,
    RoutingRecord,
    TargetDataProfile,
    TransformSpec,
    ValidationResult,
)


@dataclass
class Prediction:
    labels: Sequence[Any]
    probabilities: np.ndarray | None = None
    classes: Sequence[Any] | None = None
    calibration: dict = field(default_factory=dict)
    ood: Sequence[float] | None = None
    ensemble_disagreement: Sequence[float] | None = None


class Predictor(Protocol):
    """Frozen target predictor with reviewed lineage; no fitting on validation."""

    model_reference: str
    code_reference: str
    gold_dataset_id: str

    def predict(self, features: pd.DataFrame) -> Prediction: ...


@dataclass
class Transform:
    spec: TransformSpec
    # Must preserve observation index and return ordered target features.
    apply: Callable[[pd.DataFrame], pd.DataFrame]


@dataclass
class EvidenceBatch:
    record: RoutingRecord
    features: pd.DataFrame
    pseudo_labels: pd.Series
    source_labels: pd.Series | None
    probabilities: np.ndarray | None
    weights: np.ndarray


class PPITrainer(Protocol):
    """Existing PPI adapter: keep gold separate and honor batch weights.

    Evaluate gold-only and augmented models on the SAME validation protocol.
    Never use final test data, refit the pseudo-label predictor, or substitute
    source_labels for pseudo_labels. Weights are an aggregate influence budget,
    not a prescription for the underlying PPI estimator.
    """

    def evaluate(
        self, target: TargetDataProfile, evidence: Sequence[EvidenceBatch], config: PPIConfig
    ) -> ValidationResult: ...


class EvidencePipeline:
    def __init__(
        self,
        target: TargetDataProfile,
        transforms: Sequence[Transform] = (),
        data_provider: DataProvider | None = None,
    ):
        self.target = target
        self.transforms = list(transforms)
        ids = [t.spec.method_id for t in self.transforms]
        if len(ids) != len(set(ids)):
            raise ValueError("Transform IDs must be unique")
        self.provider = data_provider or DataProvider()

    def route(self, candidate: CandidateDataset, method_id: str | None = None) -> RoutingRecord:
        target = self.target
        translation = candidate.source_modality.casefold() != target.target_modality.casefold()
        exact = (
            candidate.feature_names == target.feature_names
            and candidate.preprocessing == target.preprocessing
            and candidate.observation_unit == target.observation_unit
        )
        record = RoutingRecord(
            candidate=candidate,
            target=target,
            route="REJECT",
            decision="defer",
            reason="Scientific relevance requires evidence-backed review",
            translation_required=translation,
            feature_compatibility={
                "status": "compatible" if exact and not translation else "review",
                "required_harmonization": [],
            },
            quality_components=dict(candidate.quality_components),
            uncertainties=["High pseudo-label confidence does not establish correctness"],
        )
        relevance = candidate.biological_relevance
        record.quality_components["biological_compatibility"] = relevance.model_dump()
        for component in (
            "domain_shift",
            "covariate_shift",
            "preprocessing_mismatch",
            "feature_mismatch",
            "dataset_quality",
            "provenance_quality",
            "pseudo_label_uncertainty",
        ):
            record.quality_components.setdefault(component, {"status": "unknown"})
        if translation:
            record.uncertainties.append("Translation may destroy task-relevant biological signal")
        if candidate.dataset_id == target.gold_dataset_id:
            record.decision = "reject"
            record.reason = "Gold dataset belongs to the existing gold path, not external evidence"
            return record
        if relevance.status == "reject":
            record.decision = "reject"
            record.reason = relevance.rationale
            return record
        if relevance.status != "accept" or not relevance.evidence:
            return record
        kind = "translate" if translation else "harmonize"
        matches = [
            t
            for t in self.transforms
            if (
                t.spec.kind == kind
                and t.spec.source_modality.casefold() == candidate.source_modality.casefold()
                and t.spec.target_modality.casefold() == target.target_modality.casefold()
                and candidate.dataset_id in t.spec.applicable_dataset_ids
                and t.spec.target_profile == target
                and (method_id is None or t.spec.method_id == method_id)
            )
        ]
        if translation or not exact or method_id is not None:
            if len(matches) != 1:
                record.reason = (
                    "No reviewed applicable transformation"
                    if not matches
                    else "Multiple applicable transformations; select a method explicitly"
                )
                return record
            record.transform = matches[0].spec
            record.feature_compatibility = {
                "status": "translated" if translation else "harmonized",
                "required_harmonization": [matches[0].spec.method_id],
            }
            record.quality_components["transformation"] = matches[0].spec.model_dump()
            record.uncertainties.extend(matches[0].spec.limitations)
        record.route = "TRANSLATE_PPI" if translation else "DIRECT_PPI"
        record.decision = "accept"
        record.reason = (
            "Relevant non-gold observations require reviewed cross-modal translation"
            if translation
            else "Relevant observed data can enter target feature space"
        )
        return record

    def prepare(
        self, record: RoutingRecord, predictor: Predictor, config: PPIConfig
    ) -> EvidenceBatch:
        # Recheck decisions rather than trusting a caller-edited routing record.
        checked = self.route(
            record.candidate, record.transform.method_id if record.transform else None
        )
        if checked.decision != "accept" or checked.model_dump() != record.model_dump():
            raise ValueError("Evidence requires an unmodified accepted routing record")
        if predictor.gold_dataset_id != self.target.gold_dataset_id:
            raise ValueError("Pseudo-label model must be bound to the declared gold dataset")
        if not predictor.model_reference or not predictor.code_reference:
            raise ValueError("Predictor model and code references are required")
        candidate = record.candidate
        path = Path(candidate.file_path)
        digest = self._digest(path)
        frame, source = self.provider.get_data(file_path=str(path), allow_synthetic=False)
        if self._digest(path) != digest:
            raise ValueError("Dataset changed during loading")
        if frame.empty or not frame.index.is_unique or not frame.columns.is_unique:
            raise ValueError("Dataset must be nonempty with unique row and column identifiers")
        if candidate.sample_size is not None and candidate.sample_size != len(frame):
            raise ValueError("Observed sample count disagrees with manifest")
        original_count = len(frame)
        if len(frame) > config.max_rows_per_dataset:
            frame = frame.sample(config.max_rows_per_dataset, random_state=config.seed)
        labels = None
        if candidate.source_label.available:
            labels = frame[candidate.source_label.column].copy()
        features = frame.loc[:, candidate.feature_names].copy()
        if record.transform:
            transform = next(
                t for t in self.transforms if t.spec.method_id == record.transform.method_id
            )
            features = transform.apply(features)
        if not isinstance(features, pd.DataFrame) or not features.index.equals(frame.index):
            raise ValueError("Transform must preserve observation identity and row order")
        if list(features.columns) != self.target.feature_names:
            raise ValueError("Transformed features must match the ordered target feature space")
        if not np.isfinite(features.to_numpy(dtype=float)).all():
            raise ValueError("Target features contain missing or non-finite values")
        prediction = predictor.predict(features.copy())
        y = np.asarray(prediction.labels)
        if y.ndim != 1 or len(y) != len(features) or pd.isna(y).any():
            raise ValueError("Predictor must return one non-missing pseudo-label per observation")
        if np.issubdtype(y.dtype, np.number) and not np.isfinite(y).all():
            raise ValueError("Numeric pseudo-labels must be finite")
        audit = {"calibration": prediction.calibration or {"status": "unknown"}}
        probabilities = None
        if prediction.probabilities is not None:
            probabilities = np.asarray(prediction.probabilities, dtype=float)
            classes = (
                np.asarray(prediction.classes).tolist() if prediction.classes is not None else []
            )
            if (
                probabilities.ndim != 2
                or probabilities.shape != (len(y), len(classes))
                or len(classes) < 2
                or len(set(classes)) != len(classes)
                or not np.isfinite(probabilities).all()
                or (probabilities < 0).any()
                or (probabilities > 1).any()
                or not np.allclose(probabilities.sum(axis=1), 1)
            ):
                raise ValueError("Invalid class probabilities or missing class ontology")
            if not set(y).issubset(set(classes)):
                raise ValueError("Predicted labels are outside probability class ontology")
            sorted_p = np.sort(probabilities, axis=1)
            audit.update(
                classes=classes,
                probabilities=probabilities.tolist(),
                entropy=(
                    -np.sum(probabilities * np.log(np.clip(probabilities, 1e-15, 1)), axis=1)
                ).tolist(),
                margin=(sorted_p[:, -1] - sorted_p[:, -2]).tolist(),
            )
        for name in ("ood", "ensemble_disagreement"):
            signal = getattr(prediction, name)
            if signal is not None:
                values = np.asarray(signal, dtype=float)
                if values.shape != (len(y),) or not np.isfinite(values).all():
                    raise ValueError(f"Invalid per-observation {name}")
                audit[name] = values.tolist()
            else:
                audit[name] = {"status": "unknown"}
        sl = candidate.source_label
        if sl.audit_compatible and sl.ontology == self.target.label_ontology and labels is not None:
            present = labels.notna().to_numpy()
            audit["source_label_agreement"] = {
                "n_compared": int(present.sum()),
                "fraction": (
                    float(np.mean(labels.to_numpy()[present] == y[present]))
                    if present.any()
                    else None
                ),
            }
        else:
            audit["source_label_agreement"] = {"status": "not_comparable"}
        # Preserve the routing record as a reusable immutable-in-practice input.
        record = record.model_copy(deep=True)
        record.audit_signals = audit
        prediction_provenance = getattr(predictor, "prediction_provenance", None)
        if prediction_provenance is not None:
            record.pseudo_label_strategy = prediction_provenance.get(
                "gold_prediction_mode", record.pseudo_label_strategy
            )
            record.audit_signals["pseudo_labeler_provenance"] = prediction_provenance
        record.execution = {
            "data_source": source,
            "sha256": digest,
            "original_rows": original_count,
            "selected_row_ids": frame.index.tolist(),
            "model_reference": predictor.model_reference,
            "code_reference": predictor.code_reference,
            "gold_dataset_id": predictor.gold_dataset_id,
            "representation": (
                "translated" if record.translation_required else "observed_or_harmonized"
            ),
            "seed": config.seed,
        }
        return EvidenceBatch(
            record,
            features.copy(),
            pd.Series(y, index=features.index),
            labels,
            probabilities,
            np.full(len(y), config.external_weight_budget / len(y)),
        )

    def run(
        self,
        candidates: Sequence[CandidateDataset],
        predictor: Predictor,
        trainer: PPITrainer,
        config: PPIConfig,
        output_dir: str,
        methods: dict[str, str] | None = None,
    ) -> dict:
        """Evaluate each source and their combination; persist even deferred candidates.

        Each candidate is explicit non-gold input. Automatic scientific relevance
        or translator selection is deliberately outside this initial prototype.
        """
        ids = [c.dataset_id for c in candidates]
        if len(ids) != len(set(ids)):
            raise ValueError("Duplicate dataset IDs would double-count evidence")
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        if any(out.iterdir()):
            raise ValueError("Use an empty output directory to preserve prior experiment artifacts")
        records, batches, evaluations = [], [], []
        for candidate in candidates:
            record = self.route(candidate, (methods or {}).get(candidate.dataset_id))
            if record.decision == "accept":
                try:
                    batch = self.prepare(record, predictor, config)
                    record = batch.record
                    number = len(batches)
                    batch.features.to_csv(out / f"evidence_{number}_features.csv", index=True)
                    batch.pseudo_labels.to_csv(
                        out / f"evidence_{number}_pseudo_labels.csv", index=True
                    )
                    if batch.source_labels is not None:
                        batch.source_labels.to_csv(
                            out / f"evidence_{number}_source_labels.csv", index=True
                        )
                    record.execution["artifact_prefix"] = f"evidence_{number}"
                    batches.append(batch)
                except Exception as exc:
                    record.decision = "defer"
                    record.reason = f"Evidence preparation failed: {type(exc).__name__}: {exc}"
            records.append(record)
        self._save(out, records, config, evaluations)
        groups = [[b] for b in batches]
        if len(batches) > 1:
            groups.append(batches)
        for group in groups:
            # Each source receives equal mass, regardless of its sample count.
            for batch in group:
                batch.weights = np.full(
                    len(batch.features),
                    config.external_weight_budget / len(group) / len(batch.features),
                )
            entry = {
                "datasets": [b.record.candidate.dataset_id for b in group],
                "weight_mass_per_dataset": [float(b.weights.sum()) for b in group],
            }
            try:
                result = ValidationResult.model_validate(
                    trainer.evaluate(self.target, group, config)
                )
                entry.update(
                    result=result.model_dump(),
                    delta=result.delta,
                    improvement=result.delta * (1 if result.higher_is_better else -1),
                )
            except Exception as exc:
                entry["error"] = f"{type(exc).__name__}: {exc}"
            evaluations.append(entry)
            self._save(out, records, config, evaluations)
        return {"records": records, "evaluations": evaluations}

    @staticmethod
    def _digest(path: Path) -> str:
        digest = sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    def _save(self, out, records, config, evaluations):
        payload = {
            "schema_version": 1,
            "target": self.target.model_dump(mode="json"),
            "ppi_config": config.model_dump(mode="json"),
            "records": [r.model_dump(mode="json") for r in records],
            "evaluations": evaluations,
        }
        temporary = out / "run.json.tmp"
        temporary.write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")
        temporary.replace(out / "run.json")
        rows = []
        for r in records:
            c = r.candidate
            rows.append(
                {
                    "Dataset": c.dataset_id,
                    "Scientific relevance": c.biological_relevance.status,
                    "Source modality": c.source_modality,
                    "Target modality": r.target.target_modality,
                    "Translation required?": r.translation_required,
                    "Translator": (
                        r.transform.method_id if r.transform and r.translation_required else None
                    ),
                    "Source labels available?": c.source_label.available,
                    "Source labels usable for auditing?": c.source_label.audit_compatible
                    and c.source_label.ontology == r.target.label_ontology,
                    "Route": r.route,
                    "Expected evidence quality": json.dumps(r.quality_components),
                    "Main uncertainty": "; ".join(r.uncertainties),
                    "Decision": r.decision,
                    "Reason": r.reason,
                }
            )
        pd.DataFrame(rows).to_csv(out / "evidence_summary.csv", index=False)
