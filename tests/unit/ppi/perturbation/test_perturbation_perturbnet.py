"""The PerturbNet arms' Kosmos side: the bundle, the alignment guard, scoring and
the skip rules. The runner itself needs the PerturbNet environment and is
exercised by the smoke run, not here."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from kosmos.ppi.discovery_bridge import _study_counts
from kosmos.ppi.perturbation import perturbnet_arm
from kosmos.ppi.perturbation.contract import PerturbationExample, PerturbationTask

GENES = ["A", "B", "C", "D"]
TARGET_SUM = 1e4


def _log1p_cpm(counts: np.ndarray, total: np.ndarray) -> np.ndarray:
    return np.log1p(counts / total[:, None] * TARGET_SUM).astype(np.float32)


@pytest.fixture
def screen(tmp_path):
    rng = np.random.default_rng(0)
    counts = rng.poisson(20, size=(10, len(GENES))).astype(np.float32)
    total = counts.sum(axis=1).astype(np.float64) * 1.25  # the panel is 80% of each cell
    values = _log1p_cpm(counts, total)
    labels = ["ctrl", "ctrl", "ctrl", "A", "A", "B", "B", "C", "C", "D"]
    control_rows = np.array([0, 1, 2])
    control_mean = values[control_rows].mean(axis=0)
    examples = {
        row: PerturbationExample(
            control=control_mean,
            perturbation=(labels[row],),
            delta=values[row] - control_mean,
            target=values[row],
            sample_id=str(row),
        )
        for row in range(3, 10)
    }
    splits = {
        "train": [examples[r] for r in (3, 4, 5, 6)],
        "validation": [examples[r] for r in (7, 8)],
        "test": [examples[9]],
    }
    table = pd.DataFrame(counts, columns=GENES)
    table["total_counts"] = total
    gold_counts = tmp_path / "gold_counts.parquet"
    table.to_parquet(gold_counts, index=False)
    supp = table.iloc[:4].copy()
    supp_counts = tmp_path / "supp_counts.parquet"
    supp.to_parquet(supp_counts, index=False)
    task = PerturbationTask(
        genes=tuple(GENES),
        gene_to_index={g: i for i, g in enumerate(GENES)},
        condition_column="condition",
        control_labels=("ctrl",),
    )
    counts_block = {
        "gold": gold_counts,
        "supplementary": supp_counts,
        "total_column": "total_counts",
        "target_sum": TARGET_SUM,
    }
    return {
        "splits": splits, "task": task, "counts": counts_block,
        "control_rows": control_rows, "values": values, "control_mean": control_mean,
    }


def test_bundle_holds_training_cells_and_controls_and_the_perturbations_to_predict(screen, tmp_path):
    bundle = tmp_path / "bundle"
    info = perturbnet_arm.write_bundle(
        bundle, arm=perturbnet_arm.PERTURBNET_SUPP, splits=screen["splits"], task=screen["task"],
        counts=screen["counts"], control_rows=screen["control_rows"], seed=1, settings={"scvi_epochs": 1},
    )
    labels = json.loads((bundle / "train_labels.json").read_text())
    assert labels == ["A", "A", "B", "B", "ctrl", "ctrl", "ctrl"]
    assert np.load(bundle / "train_counts.npy").shape == (7, len(GENES))
    assert np.load(bundle / "supp_counts.npy").shape == (4, len(GENES))
    predict = json.loads((bundle / "predict.json").read_text())
    assert predict == {"C": {"genes": ["C"], "n_cells": 2}, "D": {"genes": ["D"], "n_cells": 1}}
    config = json.loads((bundle / "config.json").read_text())
    assert config["use_supplementary"] is True and config["genes"] == GENES
    assert info["scvi_cells"] == {"gold_train": 7, "supplementary": 4}


def test_base_arm_bundle_has_no_supplementary_cells(screen, tmp_path):
    bundle = tmp_path / "bundle"
    perturbnet_arm.write_bundle(
        bundle, arm=perturbnet_arm.PERTURBNET_BASE, splits=screen["splits"], task=screen["task"],
        counts=screen["counts"], control_rows=screen["control_rows"], seed=1, settings={},
    )
    assert not (bundle / "supp_counts.npy").exists()
    assert json.loads((bundle / "config.json").read_text())["use_supplementary"] is False


def test_misaligned_count_rows_are_refused(screen, tmp_path):
    shuffled = pd.read_parquet(screen["counts"]["gold"]).iloc[::-1].reset_index(drop=True)
    path = tmp_path / "shuffled.parquet"
    shuffled.to_parquet(path, index=False)
    with pytest.raises(ValueError, match="not aligned"):
        perturbnet_arm.write_bundle(
            tmp_path / "bundle", arm=perturbnet_arm.PERTURBNET_BASE, splits=screen["splits"],
            task=screen["task"], counts={**screen["counts"], "gold": path},
            control_rows=screen["control_rows"], seed=1, settings={},
        )


def test_score_uses_the_gold_control_mean_and_the_observed_delta(screen):
    examples = screen["splits"]["validation"]
    # a perfect generator returns each perturbation's mean observed profile
    predicted = {"C": screen["values"][[7, 8]].mean(axis=0)}
    metrics, predictions, observations = perturbnet_arm._score(examples, predicted, screen["task"], top_k=2)
    np.testing.assert_allclose(predictions["C"], observations["C"], atol=1e-6)
    assert metrics["summary"]["mse"] == pytest.approx(0.0, abs=1e-10)


def test_unavailable_reasons(screen, monkeypatch, tmp_path):
    assert "raw counts" in perturbnet_arm.unavailable_reason(perturbnet_arm.PERTURBNET_BASE, None)
    no_supp = {**screen["counts"], "supplementary": None}
    assert "supplementary" in perturbnet_arm.unavailable_reason(perturbnet_arm.PERTURBNET_SUPP, no_supp)
    monkeypatch.setenv("KOSMOS_PERTURBNET_PYTHON", str(tmp_path / "missing-python"))
    assert "interpreter" in perturbnet_arm.unavailable_reason(perturbnet_arm.PERTURBNET_BASE, screen["counts"])


def test_study_counts_resolves_the_supplementary_table_by_path(tmp_path):
    supp = tmp_path / "supp.parquet"
    study = {
        "target_sum": 1e4,
        "counts": {
            "total_column": "total_counts",
            "gold": str(tmp_path / "gold_counts.parquet"),
            "supplementary": {str(supp): str(tmp_path / "supp_counts.parquet")},
        },
    }
    resolved = _study_counts(study, [supp])
    assert resolved["supplementary"] == tmp_path / "supp_counts.parquet"
    assert _study_counts(study, [tmp_path / "other.parquet"])["supplementary"] is None
    assert _study_counts({}, [supp]) is None


def test_external_arms_are_not_in_the_default_set():
    from kosmos.ppi.perturbation.train import ARMS, EXTERNAL_ARMS

    assert not set(ARMS) & set(EXTERNAL_ARMS)


def test_runner_gets_absolute_paths_for_a_relative_out(screen, tmp_path, monkeypatch):
    seen = {}

    def fake_runner(bundle, work):
        seen["bundle"], seen["work"] = bundle, work
        raise RuntimeError("stop after the call")

    monkeypatch.setattr(perturbnet_arm, "_run_runner", fake_runner)
    monkeypatch.chdir(tmp_path)
    with pytest.raises(RuntimeError, match="stop"):
        perturbnet_arm.run_perturbnet_arm(
            perturbnet_arm.PERTURBNET_BASE, splits=screen["splits"], task=screen["task"],
            counts=screen["counts"], control_rows=screen["control_rows"],
            out=Path("relative/run"), seed=1, top_k=2, settings={},
        )
    assert seen["bundle"].is_absolute() and seen["work"].is_absolute()
    assert (seen["bundle"] / "config.json").exists()


def test_supplementary_cap_is_seeded_sorted_and_off_below_the_cap():
    from kosmos.ppi.perturbation.contract import supplementary_rows

    assert supplementary_rows(100, 200, seed=1) is None
    assert supplementary_rows(100, 0, seed=1) is None
    kept = supplementary_rows(1000, 50, seed=1)
    assert len(kept) == 50 and np.all(np.diff(kept) > 0)
    np.testing.assert_array_equal(kept, supplementary_rows(1000, 50, seed=1))


def test_bundle_takes_the_capped_supplementary_rows(screen, tmp_path):
    bundle = tmp_path / "bundle"
    perturbnet_arm.write_bundle(
        bundle, arm=perturbnet_arm.PERTURBNET_SUPP, splits=screen["splits"], task=screen["task"],
        counts={**screen["counts"], "supplementary_rows": np.array([1, 3])},
        control_rows=screen["control_rows"], seed=1, settings={},
    )
    supp = np.load(bundle / "supp_counts.npy")
    full = pd.read_parquet(screen["counts"]["supplementary"])[GENES].to_numpy(dtype=np.float32)
    np.testing.assert_array_equal(supp, full[[1, 3]])


def test_prefer_large_changes_the_supplementary_prompts():
    from kosmos.discovery.preflight import choose_downloads
    from kosmos.discovery.search import PREFER_LARGE_EVIDENCE
    from kosmos.ppi.perturbation.run import supplementary_requirement_text

    prompts = []

    class Client:
        def generate_structured(self, prompt, **_):
            prompts.append(prompt)
            return {"downloads": []}

    listing = [{"reference": "geo://GSE1", "files": [{"reference": "geo://GSE1#a.h5ad", "path": "a.h5ad", "bytes": 10}]}]
    choose_downloads("q", listing, client=Client(), log=lambda *_: None)
    choose_downloads("q", listing, client=Client(), log=lambda *_: None, prefer_large=True)
    assert "choose the smallest" in prompts[0] and "most cells" not in prompts[0]
    assert "most cells" in prompts[1] and "choose the smallest" not in prompts[1]
    assert "MOST rows" in PREFER_LARGE_EVIDENCE
    # the first round looks for the screen only; size is the evidence round's job
    assert "screen(s) only" in supplementary_requirement_text(prefer_large=True)
    assert "screen(s) only" not in supplementary_requirement_text()
