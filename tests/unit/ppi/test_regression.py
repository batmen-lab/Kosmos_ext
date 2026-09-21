"""A measured target: the same PPI correction over squared error."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("torch")
from kosmos.ppi import TaskSpec, run_training
from kosmos.ppi.metrics import regression_metrics
from kosmos.ppi.schemas import PPITrainingConfig


def make_frame(rows: int, seed: int) -> pd.DataFrame:
    """`pce` is a linear function of the features plus small noise."""
    rng = np.random.default_rng(seed)
    frame = pd.DataFrame(
        {
            "x1": rng.normal(size=rows),
            "x2": rng.normal(size=rows),
            "x3": rng.normal(size=rows),
        }
    )
    frame["pce"] = 18 + 2.0 * frame["x1"] - 1.5 * frame["x2"] + rng.normal(scale=0.3, size=rows)
    return frame


def write_tables(tmp_path, rows=300, seed=0, supp_seed=1):
    """Gold and evidence from *different* draws: a copy of the gold's own rows
    is a mirror, and the run refuses mirrors on purpose."""
    gold = tmp_path / "gold.csv"
    make_frame(rows, seed).to_csv(gold, index=False)
    supplementary = tmp_path / "supplementary.csv"
    make_frame(rows, supp_seed).drop(columns=["pce"]).to_csv(supplementary, index=False)
    return gold, supplementary


def task() -> TaskSpec:
    return TaskSpec(
        target_column="pce",
        feature_prefixes=("x",),
        task_type="regression",
        seed=7,
        description="efficiency from fabrication parameters",
    )


def config() -> PPITrainingConfig:
    return PPITrainingConfig(
        seed=7, max_epochs=40, patience=8, cross_fit_folds=3, task_type="regression"
    )


def test_regression_metrics_are_in_the_targets_own_units():
    metrics = regression_metrics([1.0, 2.0, 3.0], [1.5, 2.0, 2.5])
    assert metrics["mse"] == pytest.approx(1 / 6)
    assert metrics["mae"] == pytest.approx(1 / 3)
    assert metrics["neg_mse"] == pytest.approx(-1 / 6)  # selectable form
    assert metrics["r2"] > 0
    assert metrics["pearson"] == pytest.approx(1.0, abs=1e-9)


def test_a_target_with_no_variance_reports_undefined_agreement():
    metrics = regression_metrics([2.0, 2.0], [2.0, 2.0])
    assert metrics["mse"] == 0.0
    assert metrics["r2"] is None and metrics["pearson"] is None
    assert "r2" in metrics["undefined"]


def test_the_default_learning_rate_matches_the_task(tmp_path):
    """Squared error on a standardized target needs a larger step than CE."""
    assert PPITrainingConfig(task_type="regression").learning_rate > (
        PPITrainingConfig(task_type="classification").learning_rate
    )
    assert PPITrainingConfig(task_type="regression").evaluation_metric == "r2"
    assert PPITrainingConfig(learning_rate=0.0005, task_type="regression").learning_rate == 0.0005


def test_training_a_regression_task_learns_the_target(tmp_path):
    gold, supplementary = write_tables(tmp_path)
    summary = run_training(
        labeled_path=gold,
        task=task(),
        supplementary_paths=[supplementary],
        output_dir=tmp_path / "run",
        config=config(),
    )

    assert summary["mode"] == "ppi"
    assert summary["task"]["task_type"] == "regression"
    assert summary["task"]["evaluation_metric"] == "r2"
    assert summary["task"]["classes"] == []
    test = summary["final_test_metrics"]["baseline"]
    # The target is standardized for training and mapped back for the metrics:
    # if it were not, the reported error would be in standard deviations (or
    # the model would never leave its initial value).
    assert test["r2"] > 0.9
    assert test["rmse"] < 1.0  # the noise is 0.3; unscaled units
    assert summary["inputs"]["encoding"]["width"] == 3


def test_without_supplementary_it_is_plain_supervised_training(tmp_path):
    """The option the caller asked for: no unlabeled table, no correction."""
    gold, _ = write_tables(tmp_path)
    summary = run_training(
        labeled_path=gold,
        task=task(),
        output_dir=tmp_path / "run-supervised",
        config=config(),
    )
    assert summary["mode"] == "supervised"
    assert summary["final_test_metrics"]["baseline"]["r2"] > 0.9
