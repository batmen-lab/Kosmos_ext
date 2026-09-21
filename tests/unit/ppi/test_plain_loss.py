"""The plain (no negative term) loss: soft pseudo-labels, weighted mean, ramp."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")
import torch.nn.functional as F  # noqa: E402

from kosmos.ppi import PseudoLabelLoss  # noqa: E402
from kosmos.ppi.schemas import PPITrainingConfig  # noqa: E402


def _logits(n=4, k=3, seed=0):
    generator = torch.Generator().manual_seed(seed)
    return torch.randn(n, k, generator=generator)


def test_the_objective_is_true_plus_weighted_soft_external_and_nothing_else():
    torch.manual_seed(0)
    gold_logits, external_logits = _logits(seed=1), _logits(seed=2)
    y = torch.tensor([0, 1, 2, 1])
    probs = F.softmax(_logits(seed=3), dim=1)
    weights = torch.tensor([1.0, 1.0, 2.0, 1.0])

    loss = PseudoLabelLoss(coefficient=0.5)
    terms = loss.total_loss(
        gold_logits,
        y,
        y_pseudo=None,
        external_logits=external_logits,
        external_pseudo=probs,
        external_weights=weights,
        population_size=99,  # must not scale anything in this loss
        weight_mass=float(weights.sum()),
    )

    expected_true = F.cross_entropy(gold_logits, y)
    per_row = -(probs * F.log_softmax(external_logits, dim=1)).sum(dim=1)
    expected_external = (per_row * weights).sum() / weights.sum()

    assert torch.allclose(terms["loss_true"], expected_true, atol=1e-6)
    assert torch.allclose(terms["loss_external"], expected_external, atol=1e-6)
    assert torch.allclose(terms["loss_total"], expected_true + 0.5 * expected_external, atol=1e-6)
    # No subtraction anywhere: the total is larger than the labeled term alone.
    assert float(terms["loss_total"]) > float(terms["loss_true"])


def test_the_population_scale_does_not_leak_in():
    """The signed loss multiplies by the population size; this one must not."""
    torch.manual_seed(0)
    gold_logits, external_logits = _logits(seed=4), _logits(seed=5)
    y = torch.tensor([0, 1, 2, 1])
    probs = F.softmax(_logits(seed=6), dim=1)
    weights = torch.ones(4)
    loss = PseudoLabelLoss(coefficient=1.0)
    small = loss.total_loss(
        gold_logits, y, None, external_logits, probs, weights, 10, 4.0
    )["loss_external"]
    large = loss.total_loss(
        gold_logits, y, None, external_logits, probs, weights, 100_000, 4.0
    )["loss_external"]
    assert torch.allclose(small, large)


def test_soft_targets_are_required():
    loss = PseudoLabelLoss(coefficient=1.0)
    with pytest.raises(ValueError, match="soft"):
        loss.total_loss(
            _logits(), torch.tensor([0, 1, 2, 1]), None, _logits(seed=7), None, torch.ones(4)
        )


def test_the_coefficient_ramps_from_zero():
    loss = PseudoLabelLoss(coefficient=1.0, ramp_epochs=4)
    assert loss.schedule_epoch(0) == 0.0
    assert loss.schedule_epoch(1) == pytest.approx(0.25)
    assert loss.schedule_epoch(2) == pytest.approx(0.5)
    assert loss.schedule_epoch(4) == pytest.approx(1.0)
    assert loss.schedule_epoch(9) == pytest.approx(1.0)  # clamped

    constant = PseudoLabelLoss(coefficient=0.3, ramp_epochs=0)
    assert constant.schedule_epoch(1) == pytest.approx(0.3)


def test_config_refuses_plain_mode_with_hard_targets():
    with pytest.raises(ValueError, match="pseudo_targets must be 'soft'"):
        PPITrainingConfig(loss_mode="plain", pseudo_targets="hard")


def test_config_moves_plain_mode_to_a_joint_schedule():
    """Two-stage alternation means nothing when there is one objective."""
    config = PPITrainingConfig(loss_mode="plain", pseudo_targets="soft", schedule="two_stage")
    assert config.schedule == "joint"
    signed = PPITrainingConfig(loss_mode="signed", schedule="two_stage")
    assert signed.schedule == "two_stage"  # unchanged for the signed path


def _tables(tmp_path, rows=180, classes=3, seed=0):
    rng = np.random.default_rng(seed)
    labeled = pd.DataFrame(
        {
            "sample_id": [f"L{i}" for i in range(rows)],
            "f1": rng.normal(size=rows),
            "f2": rng.normal(size=rows),
            "f3": rng.normal(size=rows),
            "label": [f"c{i % classes}" for i in range(rows)],
        }
    )
    unlabeled = pd.DataFrame(
        {
            "sample_id": [f"U{i}" for i in range(rows)],
            "f1": rng.normal(size=rows) + 0.2,
            "f2": rng.normal(size=rows),
            "f3": rng.normal(size=rows),
        }
    )
    a, b = tmp_path / "labeled.csv", tmp_path / "unlabeled.csv"
    labeled.to_csv(a, index=False)
    unlabeled.to_csv(b, index=False)
    return a, b


def test_plain_mode_trains_end_to_end_and_records_its_loss(tmp_path):
    from kosmos.ppi import TaskSpec, run_training

    labeled, unlabeled = _tables(tmp_path)
    task = TaskSpec(
        target_column="label",
        feature_prefixes=("f",),
        sample_id_column="sample_id",
        seed=3,
    )
    config = PPITrainingConfig(
        seed=3,
        max_epochs=3,
        patience=3,
        cross_fit_folds=3,
        loss_mode="plain",
        pseudo_targets="soft",
        loss_ramp_epochs=2,
    )
    summary = run_training(
        labeled_path=labeled,
        task=task,
        supplementary_paths=[unlabeled],
        output_dir=tmp_path / "run-plain",
        config=config,
    )
    assert summary["mode"] == "ppi"
    assert summary["loss"]["mode"] == "plain"
    assert summary["loss"]["ramp_epochs"] == 2
    assert summary["loss"]["schedule"] == "joint"
    assert summary["loss"]["pseudo_targets"] == "soft"
    # The comparison the run exists for is still reported.
    assert set(summary["final_test_metrics"]) == {"baseline", "ppi"}
    assert "macro_f1" in summary["final_test_metrics"]["ppi"]
    # Every run also leaves behind the human-readable rendering.
    written = tmp_path / "run-plain" / "summary.md"
    assert written.exists()
    text = written.read_text(encoding="utf-8")
    assert "mode: `plain`" in text and "labeled.csv" in text
