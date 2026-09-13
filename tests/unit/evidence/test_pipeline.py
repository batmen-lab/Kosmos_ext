"""Scientific boundary and end-to-end adapter tests; no network or model downloads."""

import json

import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError

from kosmos.evidence import (
    Assessment,
    CandidateDataset,
    EvidencePipeline,
    PPIConfig,
    Prediction,
    SourceLabel,
    TargetDataProfile,
    Transform,
    TransformSpec,
    ValidationResult,
)


@pytest.fixture
def target():
    return TargetDataProfile(
        prediction_task="classification",
        label_ontology="outcome-v1",
        observation_unit="patient",
        organism="human",
        biological_system="blood",
        condition="example condition",
        experimental_context="observational",
        target_modality="RNA",
        feature_names=["g1", "g2"],
        preprocessing="normalized-v1",
        gold_dataset_id="gold-1",
    )


@pytest.fixture
def candidate(tmp_path):
    path = tmp_path / "observations.csv"
    pd.DataFrame(
        {
            "g1": [1.0, 2.0, 3.0],
            "g2": [2.0, 3.0, 4.0],
            "label": ["negative", "positive", "negative"],
        }
    ).to_csv(path, index=False)
    return CandidateDataset(
        dataset_id="external-1",
        source="retrieved repository",
        file_path=str(path),
        source_modality="RNA",
        observation_unit="patient",
        feature_names=["g1", "g2"],
        preprocessing="normalized-v1",
        sample_size=3,
        provenance=["repository:accession-1"],
        biological_relevance=Assessment(
            status="accept", rationale="Reviewed task relevance", evidence=["review:example"]
        ),
    )


class GoldPredictor:
    model_reference = "fixture:model-v1"
    code_reference = "fixture:predictor-v1"
    gold_dataset_id = "gold-1"

    def predict(self, features):
        assert list(features.columns) == ["g1", "g2"]
        return Prediction(
            ["positive"] * len(features),
            np.tile([0.1, 0.9], (len(features), 1)),
            ["negative", "positive"],
        )


class Trainer:
    def __init__(self):
        self.calls = []

    def evaluate(self, target, evidence, config):
        self.calls.append([b.record.candidate.dataset_id for b in evidence])
        assert sum(b.weights.sum() for b in evidence) == pytest.approx(
            config.external_weight_budget
        )
        assert all((b.pseudo_labels == "positive").all() for b in evidence)
        return ValidationResult(
            split_role="validation",
            split_id="validation-patients-v1",
            protocol="fixed patient-disjoint split",
            metric="accuracy",
            baseline=0.6,
            augmented=0.65,
        )


def spec(target, candidate, **updates):
    fields = {
        "method_id": "fixture-transform",
        "kind": "translate",
        "source_modality": "ATAC",
        "target_modality": "RNA",
        "applicable_dataset_ids": [candidate.dataset_id],
        "target_profile": target,
        "evidence_for_applicability": ["fixture:review"],
        "code_reference": "fixture:code",
        "model_reference": "fixture:model",
        "training_domain": "fixture domain",
        "pairing": "paired",
        "limitations": ["Fixture only; no scientific validation"],
    }
    fields.update(updates)
    return TransformSpec(**fields)


def test_labels_do_not_control_routing_or_replace_predictions(target, candidate):
    pipeline = EvidencePipeline(target)
    assert pipeline.route(candidate).route == "DIRECT_PPI"
    candidate.source_label = SourceLabel(
        available=True,
        column="label",
        ontology="outcome-v1",
        audit_compatible=True,
        compatibility_evidence=["review"],
    )
    record = pipeline.route(candidate)
    batch = pipeline.prepare(
        record, GoldPredictor(), PPIConfig(implementation="fixture", code_reference="v1")
    )
    assert record.route == "DIRECT_PPI"
    assert batch.source_labels.tolist() == ["negative", "positive", "negative"]
    assert batch.pseudo_labels.tolist() == ["positive"] * 3
    assert batch.record.audit_signals["source_label_agreement"]["fraction"] == pytest.approx(1 / 3)
    assert len(batch.record.audit_signals["entropy"]) == 3
    assert not record.execution  # preparation does not mutate route input


def test_translation_requires_applicability_and_tracks_representation(target, candidate):
    candidate.source_modality = "ATAC"
    assert EvidencePipeline(target).route(candidate).decision == "defer"
    transform = Transform(spec(target, candidate), lambda x: x.copy())
    pipeline = EvidencePipeline(target, [transform])
    record = pipeline.route(candidate)
    assert record.route == "TRANSLATE_PPI"
    batch = pipeline.prepare(
        record, GoldPredictor(), PPIConfig(implementation="fixture", code_reference="v1")
    )
    assert batch.record.execution["representation"] == "translated"
    transform.spec.applicable_dataset_ids = ["other"]
    assert pipeline.route(candidate).decision == "defer"


@pytest.mark.parametrize("status", ["reject", "uncertain"])
def test_relevance_precedes_transformation(target, candidate, status):
    candidate.biological_relevance.status = status
    record = EvidencePipeline(target).route(candidate)
    assert record.route == "REJECT"
    assert record.decision == ("reject" if status == "reject" else "defer")


def test_matching_metadata_does_not_establish_relevance(target, candidate):
    candidate.biological_relevance.evidence = []
    assert EvidencePipeline(target).route(candidate).decision == "defer"


@pytest.mark.parametrize(
    "field,value",
    [("preprocessing", "raw"), ("observation_unit", "cell"), ("feature_names", ["g2", "g1"])],
)
def test_same_modality_mismatch_needs_harmonizer(target, candidate, field, value):
    setattr(candidate, field, value)
    assert EvidencePipeline(target).route(candidate).decision == "defer"


def test_harmonizer_keeps_direct_route(target, candidate):
    candidate.feature_names = ["g2", "g1"]
    transform = Transform(
        spec(target, candidate, kind="harmonize", source_modality="RNA"), lambda x: x[["g1", "g2"]]
    )
    pipeline = EvidencePipeline(target, [transform])
    record = pipeline.route(candidate)
    assert record.route == "DIRECT_PPI"
    batch = pipeline.prepare(
        record, GoldPredictor(), PPIConfig(implementation="fixture", code_reference="v1")
    )
    assert list(batch.features.columns) == target.feature_names


def test_no_synthetic_fallback_and_persists_deferred(target, candidate, tmp_path):
    candidate.file_path = str(tmp_path / "missing.csv")
    trainer = Trainer()
    result = EvidencePipeline(target).run(
        [candidate],
        GoldPredictor(),
        trainer,
        PPIConfig(implementation="fixture", code_reference="v1"),
        str(tmp_path / "run"),
    )
    assert result["records"][0].decision == "defer"
    assert not trainer.calls
    assert (tmp_path / "run" / "run.json").exists()


def test_end_to_end_individual_and_combined_utility(target, candidate, tmp_path):
    second = candidate.model_copy(update={"dataset_id": "external-2"}, deep=True)
    trainer = Trainer()
    out = tmp_path / "run"
    EvidencePipeline(target).run(
        [candidate, second],
        GoldPredictor(),
        trainer,
        PPIConfig(implementation="fixture", code_reference="v1", max_rows_per_dataset=2),
        str(out),
    )
    data = json.loads((out / "run.json").read_text())
    assert len(trainer.calls) == 3
    assert data["evaluations"][0]["delta"] == pytest.approx(0.05)
    assert len(data["records"][0]["execution"]["selected_row_ids"]) == 2
    assert (out / "evidence_summary.csv").exists()
    assert pd.read_csv(out / "evidence_0_features.csv").shape[0] == 2


def test_final_test_is_not_a_validation_result():
    with pytest.raises(ValidationError):
        ValidationResult(
            split_role="test",
            split_id="final",
            protocol="test",
            metric="accuracy",
            baseline=0.5,
            augmented=0.9,
        )


def test_source_label_cannot_be_gold_or_feature(candidate):
    with pytest.raises(ValidationError):
        SourceLabel(usable_as_gold=True)
    fields = candidate.model_dump()
    fields["source_label"] = {"available": True, "column": "g1"}
    with pytest.raises(ValidationError):
        CandidateDataset(**fields)


def test_transformation_cannot_shuffle_rows(target, candidate):
    candidate.source_modality = "ATAC"
    pipeline = EvidencePipeline(
        target, [Transform(spec(target, candidate), lambda x: x.iloc[::-1])]
    )
    with pytest.raises(ValueError, match="observation identity"):
        pipeline.prepare(
            pipeline.route(candidate),
            GoldPredictor(),
            PPIConfig(implementation="fixture", code_reference="v1"),
        )


def test_invalid_probabilities_rejected(target, candidate):
    class InvalidPredictor(GoldPredictor):
        def predict(self, features):
            return Prediction(
                ["positive"] * len(features),
                np.tile([0.9, 0.9], (len(features), 1)),
                ["negative", "positive"],
            )

    pipeline = EvidencePipeline(target)
    with pytest.raises(ValueError, match="probabilities"):
        pipeline.prepare(
            pipeline.route(candidate),
            InvalidPredictor(),
            PPIConfig(implementation="fixture", code_reference="v1"),
        )


def test_gold_dataset_cannot_be_external(target, candidate):
    candidate.dataset_id = target.gold_dataset_id
    assert EvidencePipeline(target).route(candidate).decision == "reject"


def test_wrong_gold_predictor_lineage_rejected(target, candidate):
    predictor = GoldPredictor()
    predictor.gold_dataset_id = "untrusted"
    pipeline = EvidencePipeline(target)
    with pytest.raises(ValueError, match="declared gold"):
        pipeline.prepare(
            pipeline.route(candidate),
            predictor,
            PPIConfig(implementation="fixture", code_reference="v1"),
        )


def test_translator_for_different_target_profile_deferred(target, candidate):
    candidate.source_modality = "ATAC"
    other_target = target.model_copy(update={"condition": "another condition"})
    pipeline = EvidencePipeline(target, [Transform(spec(other_target, candidate), lambda x: x)])
    assert pipeline.route(candidate).decision == "defer"


def test_trainer_failure_remains_in_audit(target, candidate, tmp_path):
    class FailingTrainer:
        def evaluate(self, *args):
            raise RuntimeError("adapter unavailable")

    out = tmp_path / "failed-run"
    EvidencePipeline(target).run(
        [candidate],
        GoldPredictor(),
        FailingTrainer(),
        PPIConfig(implementation="fixture", code_reference="v1"),
        str(out),
    )
    payload = json.loads((out / "run.json").read_text())
    assert "adapter unavailable" in payload["evaluations"][0]["error"]
    assert "delta" not in payload["evaluations"][0]
