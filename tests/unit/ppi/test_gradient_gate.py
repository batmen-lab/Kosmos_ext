"""Gold-anchored gradient gating: the fallback, the clip, and what is logged."""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")
from torch import nn  # noqa: E402

from kosmos.ppi import PPITrainingConfig, run_ppi_experiment  # noqa: E402
from kosmos.ppi.gating import GateStats, GradientGate  # noqa: E402
from kosmos.ppi.losses import SupervisedLoss  # noqa: E402
from kosmos.ppi.train import synthetic_data  # noqa: E402


def _model(features=6, classes=2, seed=0):
    torch.manual_seed(seed)
    return nn.Linear(features, classes)


def _batch(features=6, rows=8, classes=2, seed=1):
    generator = torch.Generator().manual_seed(seed)
    X = torch.randn(rows, features, generator=generator)
    y = torch.randint(0, classes, (rows,), generator=generator)
    return X, y


def test_an_aligned_synthetic_gradient_is_added():
    model = _model()
    X, y = _batch()
    loss = SupervisedLoss(coefficient=1.0)
    gold_loss = loss.labeled_term(model(X), y)
    synthetic_loss = loss.labeled_term(model(X * 1.01), y)

    gate = GradientGate(kappa=1.0)
    stats = gate.combine(gold_loss, synthetic_loss, model.parameters())

    assert stats.cosine > 0.9  # nearly the same direction
    assert stats.weight > 0
    assert stats.active
    assert stats.synthetic_norm * stats.weight <= stats.gold_norm * 1.0001


def test_a_conflicting_synthetic_gradient_is_ignored_completely():
    """The property the whole design exists for: bad evidence cannot move gold."""
    model = _model()
    X, y = _batch()
    loss = SupervisedLoss(coefficient=1.0)
    gold_loss = loss.labeled_term(model(X), y)
    flipped = torch.where(y > 0, torch.zeros_like(y), torch.ones_like(y))
    synthetic_loss = loss.labeled_term(model(X * 5), flipped)

    gold_gradients = torch.autograd.grad(
        gold_loss, list(model.parameters()), retain_graph=True
    )
    gate = GradientGate(kappa=1.0)
    stats = gate.combine(gold_loss, synthetic_loss, model.parameters())

    assert stats.cosine < 0
    assert stats.weight == 0.0
    assert not stats.active
    for assigned, expected in zip(model.parameters(), gold_gradients, strict=True):
        assert torch.allclose(assigned.grad, expected)


def test_a_huge_synthetic_loss_cannot_dominate_the_gold_gradient():
    model = _model()
    X, y = _batch()
    loss = SupervisedLoss(coefficient=1.0)
    gold_loss = loss.labeled_term(model(X), y)
    # The same direction, but a hundred times the size (a mis-scaled cohort).
    synthetic_loss = loss.labeled_term(model(X), y) * 100.0

    gate = GradientGate(kappa=0.25)
    stats = gate.combine(gold_loss, synthetic_loss, model.parameters())

    assert stats.cosine > 0.9
    assert stats.scale < 1.0
    assert stats.weight * stats.synthetic_norm <= 0.25 * stats.gold_norm * 1.0001


def test_the_gate_is_a_controller_not_a_loss():
    """The gate's weight must be a detached number, not a term to optimise."""
    model = _model()
    X, y = _batch()
    loss = SupervisedLoss(coefficient=1.0)
    stats = GradientGate().combine(
        loss.labeled_term(model(X), y),
        loss.labeled_term(model(X), y),
        model.parameters(),
    )

    assert isinstance(stats, GateStats)
    assert isinstance(stats.weight, float)
    # Gradients are values written onto the parameters, so nothing above them
    # carries a graph the model could push on.
    for parameter in model.parameters():
        assert parameter.grad is not None
        assert not parameter.grad.requires_grad
        assert parameter.grad.grad_fn is None


def test_a_sample_scope_gate_keeps_only_the_rows_that_agree():
    model = _model()
    X, y = _batch(rows=8)
    loss = SupervisedLoss(coefficient=1.0)
    gold_loss = loss.labeled_term(model(X), y)
    # Half the synthetic rows carry their own label, half carry the other one.
    half = len(y) // 2
    flipped = torch.cat([y[:half], 1 - y[half:]])
    synthetic_logits = model(X)
    synthetic_loss = loss.labeled_term(synthetic_logits, flipped)

    gate = GradientGate(kappa=1.0, scope="sample")
    stats = gate.combine(
        gold_loss,
        synthetic_loss,
        model.parameters(),
        synthetic_per_row=loss.per_row(synthetic_logits, flipped),
    )

    assert 0.0 < stats.accepted_fraction < 1.0
    assert stats.weight >= 0.0


def test_gradient_gated_training_runs_and_records_its_diagnostics(tmp_path):
    train, validation, external = synthetic_data(7)
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    config = PPITrainingConfig(
        loss_mode="gradient_gated",
        max_epochs=4,
        patience=4,
        gold_batch_size=45,
        max_external_samples=120,
        cross_fit_folds=3,
    )
    result = run_ppi_experiment(
        gold_train=train,
        gold_validation=validation,
        external_evidence=external,
        pseudo_labeler=make_pipeline(
            StandardScaler(), LogisticRegression(max_iter=1000, random_state=7)
        ),
        config=config,
        output_dir=tmp_path,
    )

    history = result.training_history["ppi"]
    assert history and all(row["phase"] == "gated" for row in history)
    for row in history:
        for key in (
            "train_gold_loss",
            "train_synthetic_loss",
            "gold_grad_norm",
            "synthetic_grad_norm",
            "gradient_cosine",
            "gradient_cosine_std",
            "synthetic_weight",
            "synthetic_weight_std",
            "synthetic_active_fraction",
            "validation_accuracy",
            "validation_balanced_accuracy",
            "validation_macro_f1",
        ):
            assert key in row, key
            assert np.isfinite(float(row[key]))
    gate = result.training["gate"]
    assert gate["scope"] == "batch"
    assert len(gate["epochs"]) == len(history)
    assert -1.0 <= gate["gradient_cosine_mean"] <= 1.0
    assert 0.0 <= gate["synthetic_active_fraction"] <= 1.0


def _stress_module():
    """The stress script, loaded the way a person runs it."""
    import importlib.util
    from pathlib import Path

    root = Path(__file__).resolve().parents[3]
    spec = importlib.util.spec_from_file_location(
        "gradient_gate_stress", root / "scripts" / "gradient_gate_stress.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_the_stress_teacher_is_actually_wrong_when_it_says_it_is():
    """The corruption has to survive the pipeline, or the test measures nothing.

    The first version corrupted `predict` and left `predict_proba` clean, and the
    pipeline takes the argmax of the probabilities whenever every fold model has
    them -- so every "corrupted" run was silently uncorrupted and the gate looked
    unaffected. A stress test whose corruption does not arrive is worse than no
    stress test.
    """
    stress = _stress_module()
    train, _validation, _test, cohort = stress.build_data(0)
    clean = stress.CorruptedTeacher(base=stress._default_base(), rate=0.0, seed=0)
    clean.fit(train.X, train.y)
    truth = np.asarray(clean.predict(cohort.X))

    for rate in (0.0, 0.5, 1.0):
        teacher = stress.CorruptedTeacher(base=stress._default_base(), rate=rate, seed=0)
        teacher.fit(train.X, train.y)
        labels = np.asarray(teacher.predict(cohort.X))
        probabilities = np.asarray(teacher.predict_proba(cohort.X))
        # The pipeline reads hard labels off the probabilities when it can.
        assert np.array_equal(labels, np.asarray(teacher.classes_)[probabilities.argmax(1)])
        agreement = float(np.mean(labels == truth))
        if rate == 0.0:
            assert agreement == 1.0
        elif rate == 0.5:
            assert 0.3 < agreement < 0.8
        else:
            assert agreement < 0.4
