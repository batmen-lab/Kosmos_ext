"""Tests for the three-stage IRF1 workflow and this session's fixes.

Two layers:

* **logic** — the arm selection, the split, the arm names. Fast, no data.
* **artifacts** — the files stage 2 and stage 3 actually wrote. They are
  skipped when the run has not been produced, so the file is safe to run
  anywhere.

    .venv/bin/python -m pytest -o addopts="" Perturb_middle/test_perturb_middle.py -q -p no:logging
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

HERE = Path(__file__).resolve().parent
STAGE2 = HERE / "output_irf1-run-mine"


def _latest_run() -> Path | None:
    import re

    runs = sorted(
        (path for path in STAGE2.glob("run*") if (path / "summary.md").exists()),
        key=lambda path: int(m.group(1)) if (m := re.search(r"run-(\d+)$", path.name)) else 0,
    )
    return runs[-1] if runs else None


# --- logic -----------------------------------------------------------------


def test_arms_env_selects_both_families(monkeypatch):
    """`base,signed` is 2 GEARS arms and their 2 graph-free MLP counterparts."""
    monkeypatch.setenv("KOSMOS_PERTURBATION_ARMS", "base,signed")
    from kosmos.ppi.discovery_bridge import _perturbation_arms_from_env

    assert _perturbation_arms_from_env() == (
        "gears_base",
        "mlp_base",
        "gears_augmented_ungated",
        "mlp_augmented_ungated",
    )


def test_a_qualified_arm_name_selects_exactly_one(monkeypatch):
    monkeypatch.setenv("KOSMOS_PERTURBATION_ARMS", "mlp_base")
    from kosmos.ppi.discovery_bridge import _perturbation_arms_from_env

    assert _perturbation_arms_from_env() == ("mlp_base",)


def test_no_arm_env_means_the_default():
    import os

    from kosmos.ppi.discovery_bridge import _perturbation_arms_from_env

    os.environ.pop("KOSMOS_PERTURBATION_ARMS", None)
    assert _perturbation_arms_from_env() is None


SCREEN = ["IRF1", "IFNGR1", "IFNGR2", "CD86", "ATF2"]


def test_holdout_pins_a_perturbation_into_test():
    from kosmos.ppi.perturbation.contract import perturbation_splits

    splits = perturbation_splits(SCREEN, mode="unseen_single", holdout=["IRF1"])

    assert "IRF1" in splits["test"]
    assert "IRF1" not in splits["train"]
    assert "IRF1" not in splits["validation"]


def test_a_small_screen_still_leaves_a_validation_perturbation():
    """`round(3 * 0.1) == 0` used to leave validation empty (NaN pearson)."""
    from kosmos.ppi.perturbation.contract import perturbation_splits

    splits = perturbation_splits(["IRF1", "IFNGR1", "IFNGR2"], mode="unseen_single")

    assert splits["train"] and splits["validation"] and splits["test"]


def test_per_perturbation_metrics_are_computed_per_label():
    """The metrics module can score a single held-out perturbation (IRF1)."""
    import numpy as np

    from kosmos.ppi.perturbation.metrics import perturbation_metrics

    predictions = {"IRF1": np.array([1.0, 2.0, 3.0, 4.0])}
    observations = {"IRF1": np.array([1.0, 2.0, 3.0, 5.0])}

    metrics = perturbation_metrics(predictions, observations, top_k=2)

    assert "IRF1" in metrics.per_perturbation
    assert not np.isnan(metrics.per_perturbation["IRF1"]["pearson"])


def test_summary_uses_readable_arm_names(tmp_path):
    from kosmos.ppi.perturbation.figures import write_summary

    def arm(mse_deg: float) -> dict:
        return {
            "test": {
                "summary": {
                    "mse": 0.03,
                    "mse_deg": mse_deg,
                    "pearson": 0.2,
                    "spearman": 0.05,
                    "direction_accuracy": 0.8,
                    "top_k_overlap": 0.1,
                }
            },
            "epochs_run": 3,
            "signal": {
                "across_perturbation_std": 0.02,
                "observed_across_perturbation_std": 0.07,
                "signal": 0.3,
            },
        }

    results = {
        "task": {"n_genes": 2000, "condition_column": "condition"},
        "split_sizes": {"train": 10, "validation": 2, "test": 4},
        "arms": {
            "gears_base": arm(0.35),
            "gears_augmented_ungated": arm(0.30),
            "mlp_base": arm(0.40),
            "mlp_augmented_ungated": arm(0.38),
        },
    }

    text = write_summary(results, tmp_path).read_text()

    assert "`base`" in text
    assert "`signed ppi`" in text
    assert "`mlp base`" in text
    assert "`mlp signed ppi`" in text
    assert "gears_augmented_ungated" not in text


# --- artifacts (skipped until the stages have run) --------------------------


@pytest.mark.skipif(not (STAGE2 / "plan.json").exists(), reason="stage 2 has not run")
def test_stage2_plan_is_trainable_and_keeps_the_perturbation_genes():
    from kosmos.cli.commands.run import resolve_data_plan

    plan = json.loads((STAGE2 / "plan.json").read_text())
    resolved = resolve_data_plan(STAGE2 / "plan.json")

    assert plan["task"]["backend"] == "perturbation"
    assert resolved.target_column == "condition"
    assert plan["supplementary"], "the plan must carry supplementary expression"

    gold = pd.read_csv(resolved.labeled_paths[0])
    supp_columns = set(pd.read_csv(plan["supplementary"][0]["path"], nrows=0).columns)
    perturbations = {
        value
        for value in gold["condition"].astype(str).unique()
        if not value.upper().startswith("NT")
    }

    # A perturbation whose gene is not a panel column cannot be encoded, and its
    # cells are silently dropped -- the subset must keep them all.
    assert perturbations, "the subset has no perturbations"
    assert perturbations <= supp_columns


@pytest.mark.skipif(not (STAGE2 / "plan.json").exists(), reason="stage 2 has not run")
def test_stage2_supplementary_keeps_every_cell():
    """No cell cap: the earlier `--supp-max-cells 5000` truncated evidence."""
    plan = json.loads((STAGE2 / "plan.json").read_text())
    sizes = {}
    for entry in plan["supplementary"]:
        frame = pd.read_csv(entry["path"], usecols=["barcode"])
        sizes[Path(entry["path"]).name] = len(frame)

    assert sizes, "no supplementary output"
    # The whole screen is 20,729 cells; the gold's own counts must keep them all.
    assert max(sizes.values()) == 20729, sizes


@pytest.mark.skipif(_latest_run() is None, reason="stage 3 has not run")
def test_latest_summary_has_both_families_and_a_non_nan_pearson():
    import re

    text = (_latest_run() / "summary.md").read_text()

    for label in ("`base`", "`signed ppi`", "`mlp base`", "`mlp signed ppi`"):
        assert label in text, f"{label} missing from the arms table"
    assert not re.search(r"\bnan\b", text.lower()), "a metric came out NaN"


@pytest.mark.skipif(_latest_run() is None, reason="stage 3 has not run")
def test_latest_run_split_holds_irf1_out():
    contract = json.loads((_latest_run() / "perturbation_contract.json").read_text())

    assert "IRF1" in contract["splits"]["test"]
    assert "IRF1" not in contract["splits"]["train"]
    assert contract["split_sizes"]["validation"] > 0
