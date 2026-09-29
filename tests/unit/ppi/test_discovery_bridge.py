"""The discovery loop's data path: fetch, plan, train, and report metrics."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from kosmos.ppi.discovery_bridge import fetch_and_plan, run_data_task
from kosmos.ppi.metrics import bootstrap_interval


def write_table(path, *, rows=120, columns=8, classes=3, seed=0, label="cell_type"):
    """A small labeled table plus whatever the task needs beside it."""
    rng = np.random.default_rng(seed)
    frame = pd.DataFrame(
        rng.normal(size=(rows, columns)), columns=[f"gene_{i}" for i in range(columns)]
    )
    frame[label] = [f"type_{i % classes}" for i in range(rows)]
    frame.to_csv(path, index=False)
    return path


def write_plan(path, gold, supplementary=()):
    payload = {
        "version": 1,
        "task": {
            "target_column": "cell_type",
            "task_type": "classification",
            "feature_columns": None,
            "feature_prefixes": [],
            "exclude_columns": [],
            "sample_id_column": None,
        },
        "gold": [{"path": str(gold), "features": None}],
        "supplementary": [{"path": str(entry)} for entry in supplementary],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_a_prediction_question_trains_with_the_gated_loss(tmp_path, capsys):
    gold = write_table(tmp_path / "gold.csv", seed=1)
    cohort = write_table(tmp_path / "cohort.csv", rows=90, seed=2)
    plan = write_plan(tmp_path / "plan.json", gold, [cohort])

    outcome = run_data_task(
        question=(
            "Can the cell type of a bone-marrow mononuclear cell be predicted "
            "from its expression, and do unlabeled cells improve the prediction?"
        ),
        out_dir=tmp_path / "out",
        plan_path=plan,
        max_epochs=3,
        patience=3,
        echo=True,
    )

    assert outcome.ok, outcome.error
    assert outcome.kind.kind == "prediction"
    assert outcome.metrics["loss_mode"] == "gradient_gated"
    assert outcome.summary_path.exists()
    # The loop used to print nothing between question and report.
    printed = capsys.readouterr().out
    assert "# metrics:" in printed
    assert "# task: prediction" in printed
    # A figure of the metrics, and the numbers next to it as a record.
    assert outcome.figures and outcome.figures[0].exists()
    record = json.loads((tmp_path / "out" / "run" / "metrics.json").read_text())
    assert record["kind"] == "prediction"
    assert record["value"] is not None
    # The summary carries the figure, so a reader of summary.md sees it.
    assert "metrics_overview.png" in outcome.summary_path.read_text()
    # And the returned dict is what the director stores: keys it already reads.
    returned = outcome.as_return_value()
    assert returned["data_source"] == "data_task"
    assert returned["loss_mode"] == "gradient_gated"
    assert "balanced_accuracy" in returned


def test_an_inference_question_trains_with_the_signed_correction(tmp_path):
    gold = write_table(tmp_path / "gold.csv", seed=3)
    plan = write_plan(tmp_path / "plan.json", gold)

    outcome = run_data_task(
        question="Is there an association between gene_0 and gene_1 across donors?",
        out_dir=tmp_path / "out",
        plan_path=plan,
        max_epochs=3,
        patience=3,
        echo=False,
    )

    assert outcome.ok, outcome.error
    assert outcome.metrics["loss_mode"] == "signed"
    summary = json.loads((tmp_path / "out" / "run" / "ppi_summary.json").read_text())
    assert summary["loss"]["mode"] == "signed"
    # An inference answer carries an uncertainty, not only a point estimate.
    assert "ci_low" in outcome.metrics and "ci_high" in outcome.metrics


def test_a_fetch_that_cannot_produce_a_plan_is_a_recorded_failure(tmp_path, monkeypatch, capsys):
    """No data behind the question is a finding, not something to invent."""
    import kosmos.ppi.discovery_bridge as bridge

    monkeypatch.setattr(
        bridge, "fetch_and_plan", lambda **kwargs: (None, "# nothing downloadable")
    )
    outcome = run_data_task(
        question=(
            "Is there an association between the compound and cell type in these "
            "cells across donors?"
        ),
        out_dir=tmp_path / "out",
        echo=True,
    )

    assert outcome.ok is False
    assert outcome.stage == "fetch"
    assert "no plan" in outcome.error
    returned = outcome.as_return_value()
    assert returned["ok"] is False and returned["stage"] == "fetch"


def test_the_bootstrap_interval_brackets_the_point_estimate():
    rng = np.random.default_rng(0)
    y = np.array(["a", "b"] * 60)
    # A model that is right 80% of the time, with the errors spread over classes.
    probability = np.where(
        (rng.random(len(y)) < 0.8)[:, None] == (np.array([0, 1])[None, :]),
        0.9,
        0.1,
    )
    interval = bootstrap_interval(y, probability, np.array(["a", "b"]))
    assert interval is not None
    low, high = interval
    assert low <= high
    assert 0.0 <= low and high <= 1.0


def test_the_fetcher_is_invoked_in_plan_mode(tmp_path, monkeypatch):
    """The bridge must not ask the fetcher to also launch a training run."""
    import kosmos.ppi.discovery_bridge as bridge

    seen = {}

    class FakeProcess:
        def __init__(self, command, **kwargs):
            seen["command"] = command

        stdout = iter(["# retrieval: proposal 1/1\n"])

        def wait(self):
            return 1  # no plan written

    monkeypatch.setattr(bridge.subprocess, "Popen", FakeProcess)
    plan, output = fetch_and_plan(question="anything", out_dir=tmp_path, echo=False)

    assert plan is None
    assert "--plan-out" in seen["command"]
    assert "--run" not in seen["command"]
    assert "--supp-limit" in seen["command"]  # every task asks for evidence
    assert "nothing" not in output.lower()


def test_a_non_data_question_is_skipped_before_any_retrieval(tmp_path, monkeypatch):
    """No retrieval round is spent on a question data cannot answer."""
    import kosmos.ppi.discovery_bridge as bridge

    def explode(**kwargs):  # pragma: no cover - the point is that it is not called
        raise AssertionError("the fetcher was called for a non-data question")

    monkeypatch.setattr(bridge, "fetch_and_plan", explode)
    outcome = bridge.run_data_task(
        question="Simulate the binding energy of this protein from first principles.",
        out_dir=tmp_path / "out",
        domain="biology",
        echo=False,
    )

    assert outcome.ok is False
    assert outcome.stage == "skipped"
    assert "simulation" in outcome.error
    # A caller can tell "not a data question" apart from "fetch failed".
    assert outcome.as_return_value()["stage"] == "skipped"


def test_a_patient_level_data_question_is_declined_by_this_backend(tmp_path, monkeypatch):
    """The pipeline is single cell: it says so instead of approximating."""
    import kosmos.ppi.discovery_bridge as bridge

    def explode(**kwargs):  # pragma: no cover - must not be reached
        raise AssertionError("retrieval ran for a question this backend declines")

    monkeypatch.setattr(bridge, "fetch_and_plan", explode)
    outcome = bridge.run_data_task(
        question=(
            "Is a patient's blood pressure associated with hospital readmission, "
            "using routine clinical records?"
        ),
        out_dir=tmp_path / "out",
        domain="biology",
        echo=False,
    )

    assert outcome.ok is False
    assert outcome.stage == "skipped"
    assert "single_cell" not in outcome.error  # the reason, not a stack trace
    assert outcome.error


def test_a_single_cell_question_passes_the_modality_gate(tmp_path):
    """A per-cell question is accepted; the run carries the verdict."""
    gold = write_table(tmp_path / "gold.csv", seed=5)
    plan = write_plan(tmp_path / "plan.json", gold)
    outcome = run_data_task(
        question=(
            "Can the cell type of a pancreatic islet cell be predicted from its "
            "expression profile across donors?"
        ),
        out_dir=tmp_path / "out",
        plan_path=plan,
        max_epochs=3,
        patience=3,
        echo=False,
    )

    assert outcome.ok, outcome.error
    assert outcome.metrics["modality"] == "single_cell"
