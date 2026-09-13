"""PPI objective, leakage controls and synthetic execution contracts."""

import copy
import json
from dataclasses import replace

import numpy as np
import pytest

pytest.importorskip("torch")
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from kosmos.ppi import (
    ExternalEvidenceDataset,
    PPILoss,
    PPITrainingConfig,
    PretrainedPseudoLabeler,
    evaluate_final_test,
    prepare_pseudo_labeler,
    run_ppi_experiment,
    split_gold,
)
from kosmos.ppi.datasets import prepare_external
from kosmos.ppi.metrics import classification_metrics
from kosmos.ppi.split import validate_roles
from kosmos.ppi.train import load_dataset, synthetic_data


@pytest.fixture(autouse=True)
def threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


@pytest.fixture
def data():
    return synthetic_data(7)


@pytest.fixture
def config():
    return PPITrainingConfig(
        max_epochs=6, patience=6, cross_fit_folds=3, gold_batch_size=40, max_external_samples=70
    )


def estimator():
    return make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))


def run(data, config, path, **kwargs):
    train, val, external = data
    return run_ppi_experiment(
        gold_train=train,
        gold_validation=val,
        external_evidence=external,
        pseudo_labeler=estimator(),
        config=config,
        output_dir=path,
        **kwargs,
    )


def test_gold_and_external_predictions_and_serialization(data, config, tmp_path):
    result = run(data, config, tmp_path)
    assert np.array_equal(result.pseudo_gold.y_true, data[0].y)
    assert result.pseudo_gold.y_pred.shape == data[0].y.shape
    assert result.pseudo_gold.y_prob.shape == (90, 3)
    assert result.pseudo_external[0].y_prob.shape == (70, 3)
    assert not hasattr(result.pseudo_external[0], "y_true")
    assert result.n_external_available == 300 and result.n_external_used == 70
    assert "correction" in {r["phase"] for r in result.training_history["ppi"]}
    for row in result.training_history["ppi"]:
        assert all(
            k in row
            for k in (
                "loss_true",
                "loss_pseudo_gold",
                "loss_correction",
                "loss_external",
                "loss_total",
            )
        )
    payload = json.loads((tmp_path / "result.json").read_text())
    assert payload["delta_metrics"]["accuracy"] == pytest.approx(
        payload["validation_metrics"]["accuracy"] - payload["baseline_metrics"]["accuracy"]
    )
    with np.load(tmp_path / "pseudo_external_0.npz", allow_pickle=False) as artifact:
        assert "y_true" not in artifact
        assert artifact["weights"].sum() <= config.external_weight_budget + 1e-8


def test_signed_correction_value_and_gradient():
    logits = torch.tensor([[1.0, 0.0], [0.0, 1.0]], requires_grad=True)
    ext = torch.tensor([[0.0, 1.0]], requires_grad=True)
    loss = PPILoss(coefficient=0.7)
    terms = loss.total_loss(
        logits,
        torch.tensor([0, 1]),
        torch.tensor([1, 0]),
        ext,
        torch.tensor([1]),
        torch.tensor([0.4]),
        1,
        0.4,
    )
    true = torch.nn.functional.cross_entropy(logits, torch.tensor([0, 1]))
    pg = torch.nn.functional.cross_entropy(logits, torch.tensor([1, 0]))
    pe = torch.nn.functional.cross_entropy(ext, torch.tensor([1]))
    expected = true + 0.7 * 0.4 * (pe - pg)
    assert terms["loss_total"].item() == pytest.approx(expected.item())
    actual_grad = torch.autograd.grad(terms["loss_total"], logits, retain_graph=True)[0]
    expected_grad = torch.autograd.grad(expected, logits, retain_graph=True)[0]
    assert torch.allclose(actual_grad, expected_grad)
    assert not torch.allclose(actual_grad, torch.autograd.grad(true, logits)[0])


def test_crossfit_never_trains_on_holdout_or_validation(data, config):
    train, val, _ = data
    prepared = prepare_pseudo_labeler(train, estimator(), config)
    holdouts = []
    for fold in prepared.provenance["folds"]:
        assert not set(fold["train_ids"]) & set(fold["heldout_ids"])
        assert not set(fold["train_ids"]) & set(val.sample_ids)
        holdouts.extend(fold["heldout_ids"])
    assert sorted(holdouts) == sorted(train.sample_ids)
    for model, fold in zip(prepared.models, prepared.provenance["folds"], strict=False):
        idx = [list(train.sample_ids).index(i) for i in fold["train_ids"]]
        assert np.allclose(model[0].mean_, train.X[idx].mean(0), atol=1e-6)


def test_source_labels_and_routes_do_not_change_training(data, config, tmp_path):
    first = run(data, config, tmp_path / "direct")
    translated = replace(
        data[2][0],
        evidence_route="TRANSLATE_PPI",
        source_labels=np.array(["WRONG"] * len(data[2][0].X)),
    )
    second = run((data[0], data[1], [translated]), config, tmp_path / "translated")
    assert first.validation_metrics == second.validation_metrics
    for name, tensor in first.model.state_dict().items():
        assert torch.equal(tensor, second.model.state_dict()[name])
    assert second.reproducibility["route_counts"] == {"TRANSLATE_PPI": 70}
    assert second.pseudo_external[0].evidence_metadata["source_labels"] == ["WRONG"] * 70


def test_zero_external_exactly_baseline(data, config, tmp_path):
    result = run((data[0], data[1], []), config, tmp_path)
    assert result.baseline_metrics == result.validation_metrics
    assert result.delta_metrics["accuracy"] == 0
    assert result.n_external_used == 0
    for key, value in result.model.state_dict().items():
        assert torch.equal(value, result.baseline_model.state_dict()[key])


def test_weights_and_budget_are_not_confidence(data, config):
    train, _, evidence = data
    prepared = prepare_pseudo_labeler(train, estimator(), config)
    evidence[0].alpha = np.zeros(len(evidence[0].X))
    unweighted = prepare_external(evidence, prepared, config)[0]
    weighted = prepare_external(
        evidence, prepared, config.model_copy(update={"use_evidence_weights": True})
    )[0]
    assert unweighted.weights.sum() == pytest.approx(config.external_weight_budget)
    assert weighted.weights.sum() == 0
    assert np.array_equal(unweighted.y_prob, weighted.y_prob)
    evidence[0].evidence_weight = np.full(len(evidence[0].X), 0.1 / len(evidence[0].X))
    assert prepare_external(evidence, prepared, config)[0].weights.sum() == pytest.approx(0.1)


def test_group_split_crossfit_and_overlap(data, config):
    train, _, _ = data
    grouped = replace(train, groups=np.repeat(np.arange(30), 3).astype(str))
    training, validation = split_gold(grouped, seed=10)
    assert not set(training.groups) & set(validation.groups)
    prepared = prepare_pseudo_labeler(training, estimator(), config)
    for fold in prepared.provenance["folds"]:
        mapping = dict(zip(training.sample_ids, training.groups, strict=False))
        assert not {mapping[i] for i in fold["train_ids"]} & {
            mapping[i] for i in fold["heldout_ids"]
        }
    with pytest.raises(ValueError, match="group overlap"):
        validate_roles(
            training,
            replace(validation, groups=np.repeat(training.groups[0], len(validation.X))),
            [],
        )


@pytest.mark.parametrize("role", ["gold_train", "gold_validation"])
def test_final_test_rejected_from_training(data, config, tmp_path, role):
    train, val, external = data
    if role == "gold_train":
        train = replace(train, role="final_test")
    else:
        val = replace(val, role="final_test")
    with pytest.raises(ValueError):
        run((train, val, external), config, tmp_path)


def test_observed_only_gold_and_external_type(data, config, tmp_path):
    with pytest.raises(ValueError, match="observed"):
        replace(data[0], representation="translated")
    with pytest.raises(ValueError, match="gold_train"):
        run((data[2][0], data[1], []), config, tmp_path)
    with pytest.raises(TypeError):
        ExternalEvidenceDataset(
            X=data[2][0].X,
            sample_ids=data[2][0].sample_ids,
            evidence_route="DIRECT_PPI",
            source_dataset_id="x",
            y_true=data[2][0].sample_ids,
        )


def test_pretrained_is_frozen_and_requires_review(data, config, tmp_path):
    train, val, ext = data
    frozen = estimator().fit(
        train.X, train.y
    )  # Fixture only; known IDs demonstrate overlap guard below.
    with pytest.raises(ValueError):
        PretrainedPseudoLabeler(
            frozen,
            model_reference="x",
            code_reference="x",
            feature_names=train.feature_names,
            applicability_evidence=[],
            leakage_review="fixture",
        )
    wrapper = PretrainedPseudoLabeler(
        frozen,
        model_reference="fixture-model",
        code_reference="fixture-code",
        feature_names=train.feature_names,
        applicability_evidence=["fixture review, not scientific evidence"],
        leakage_review="Fixture-only declared independent training",
        training_sample_ids=[val.sample_ids[0]],
    )
    config = config.model_copy(update={"pseudo_mode": "pretrained"})
    with pytest.raises(ValueError, match="overlap validation"):
        run_ppi_experiment(
            gold_train=train,
            gold_validation=val,
            external_evidence=ext,
            pseudo_labeler=wrapper,
            config=config,
            output_dir=tmp_path,
        )
    wrapper.training_sample_ids = set(train.sample_ids)
    with pytest.raises(ValueError, match="gold correction"):
        prepare_pseudo_labeler(train, wrapper, config)
    # Use genuinely separate synthetic fitting observations for the positive path.
    X, y = synthetic_data(22)[0].X, synthetic_data(22)[0].y
    wrapper.estimator = estimator().fit(X, y)
    wrapper.training_sample_ids = {"pretrain-independent"}
    before = copy.deepcopy(wrapper.estimator[1].coef_)
    prepared = prepare_pseudo_labeler(train, wrapper, config)
    assert np.array_equal(before, wrapper.estimator[1].coef_)
    assert prepared.pseudo_gold.y_prob.shape == (90, 3)


def test_undefined_metrics_reported():
    metrics = classification_metrics(
        np.array([0, 0]), np.array([[0.8, 0.2], [0.9, 0.1]]), np.array([0, 1])
    )
    assert metrics["macro_auroc"] is None and metrics["macro_auprc"] is None
    assert metrics["undefined"]


def test_binary_soft_targets_joint_training(data, config, tmp_path):
    train, val, ext = data
    train, val = replace(train, y=train.y % 2), replace(val, y=val.y % 2)
    result = run(
        (train, val, ext),
        config.model_copy(update={"pseudo_targets": "soft", "schedule": "joint"}),
        tmp_path,
    )
    assert result.pseudo_gold.y_prob.shape == (90, 2)
    assert result.validation_metrics["macro_auroc"] is not None


def test_validation_sample_leakage(data):
    train, val, external = data
    with pytest.raises(ValueError, match="sample overlap"):
        validate_roles(train, replace(val, sample_ids=train.sample_ids), [])
    external[0].sample_ids[0] = val.sample_ids[0]
    with pytest.raises(ValueError, match="External sample overlap"):
        validate_roles(train, val, external)


def test_final_evaluation_separate_and_overlap_checked(data, config, tmp_path):
    result = run(data, config, tmp_path)
    with pytest.raises(ValueError, match="overlaps"):
        evaluate_final_test(result, replace(data[1], role="final_test"))
    final = replace(
        data[1], role="final_test", sample_ids=np.array([f"final-{i}" for i in range(90)])
    )
    assert "baseline" in evaluate_final_test(result, final)


def test_cli_npz_roles_and_roundtrip(data, tmp_path):
    train = data[0]
    path = tmp_path / "train.npz"
    np.savez(
        path,
        X=train.X,
        y=train.y,
        sample_ids=train.sample_ids,
        role=train.role,
        dataset_id=train.dataset_id,
        feature_names=train.feature_names,
    )
    assert np.array_equal(load_dataset(path).X, train.X)


def test_routing_calls_real_trainer(data, config, tmp_path):
    import pandas as pd

    from kosmos.evidence import (
        Assessment,
        CandidateDataset,
        EvidencePipeline,
        PPIConfig,
        TargetDataProfile,
    )
    from kosmos.ppi import PPIEvidenceTrainer, RoutingPredictor

    gold, val, datasets = data
    ext = datasets[0]
    path = tmp_path / "retrieved.csv"
    pd.DataFrame(ext.X, columns=ext.feature_names).to_csv(path, index=False)
    target = TargetDataProfile(
        prediction_task="classification",
        label_ontology="synthetic-classes",
        observation_unit="sample",
        organism="synthetic",
        biological_system="synthetic",
        condition="synthetic",
        experimental_context="software test",
        target_modality="numeric",
        feature_names=gold.feature_names,
        preprocessing="synthetic",
        gold_dataset_id=gold.dataset_id,
    )
    candidate = CandidateDataset(
        dataset_id=ext.source_dataset_id,
        source="synthetic fixture",
        file_path=str(path),
        source_modality="numeric",
        observation_unit="sample",
        feature_names=gold.feature_names,
        preprocessing="synthetic",
        provenance=["fixture:v1"],
        biological_relevance=Assessment(
            status="accept", rationale="software fixture", evidence=["fixture"]
        ),
        metadata={"sample_ids": ext.sample_ids.tolist()},
    )
    prepared = prepare_pseudo_labeler(gold, estimator(), config)
    trainer = PPIEvidenceTrainer(gold, val, prepared, config, tmp_path / "ppi")
    result = EvidencePipeline(target).run(
        [candidate],
        RoutingPredictor(prepared),
        trainer,
        PPIConfig(implementation="kosmos.ppi", code_reference="v1", seed=config.seed),
        str(tmp_path / "routing"),
    )
    assert len(trainer.results) == 1, result
    assert "delta" in result["evaluations"][0]
    assert (tmp_path / "ppi/evaluation-000/result.json").exists()


def test_custom_downstream_factory_used_for_matched_pair(data, config, tmp_path):
    calls = []

    def custom(features, classes):
        calls.append((features, classes))
        return torch.nn.Sequential(
            torch.nn.Linear(features, 5), torch.nn.ReLU(), torch.nn.Linear(5, classes)
        )

    result = run(data, config, tmp_path, model_factory=custom)
    assert calls == [(12, 3)]  # one initialization, deep-copied for matched training
    assert str(result.model) == str(result.baseline_model)


def test_budget_handles_many_sources_and_zero_cap(data, config):
    train, _, evidence = data
    prepared = prepare_pseudo_labeler(train, estimator(), config)
    second = replace(
        evidence[0],
        source_dataset_id="second",
        sample_ids=np.array([f"second-{i}" for i in range(300)]),
    )
    selected = prepare_external(
        [*evidence, second], prepared, config.model_copy(update={"max_external_samples": 3})
    )
    assert sum(len(d.X) for d in selected) == 3
    assert sorted(len(d.X) for d in selected) == [1, 2]
    assert sum(d.weights.sum() for d in selected) <= config.external_weight_budget
    assert (
        prepare_external(evidence, prepared, config.model_copy(update={"max_external_samples": 0}))
        == []
    )


def test_very_large_external_population_is_capped(data, config):
    train, _, evidence = data
    original = evidence[0]
    large = replace(
        original,
        X=np.tile(original.X, (100, 1)),
        sample_ids=np.array([f"large-{i}" for i in range(30000)]),
    )
    prepared = prepare_pseudo_labeler(train, estimator(), config)
    selected = prepare_external([large], prepared, config)
    assert len(selected[0].X) == config.max_external_samples
    assert selected[0].weights.sum() == pytest.approx(config.external_weight_budget)


def test_zero_alpha_reduces_to_matched_baseline(data, config, tmp_path):
    train, validation, external = data
    external[0].alpha = np.zeros(len(external[0].X))
    result = run(
        (train, validation, external),
        config.model_copy(update={"use_evidence_weights": True}),
        tmp_path,
    )
    assert result.validation_metrics == result.baseline_metrics
    assert result.reproducibility["external_weight_mass"] == 0
