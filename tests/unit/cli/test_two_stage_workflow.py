"""The two-stage workflow: fetch-and-review, then experiment -- each on its own."""

from __future__ import annotations

import json

from kosmos.ppi import stages
from kosmos.ppi.discovery_bridge import DataTaskOutcome


def _plan(tmp_path, *, gold=("gold.csv",), supplementary=(), target="cell_type"):
    plan = {
        "version": 1,
        "task": {
            "objective": "predict cell type",
            "target_column": target,
            "target_source": "cli",
            "target_confidence": "declared",
            "task_type": "classification",
            "feature_columns": ["g1", "g2"],
        },
        "gold": [{"path": p, "reason": "target present"} for p in gold],
        "supplementary": [{"path": p, "reason": "no target"} for p in supplementary],
        "unusable": [],
    }
    path = tmp_path / "plan.json"
    path.write_text(json.dumps(plan), encoding="utf-8")
    return path


def test_review_document_names_the_task_and_the_next_command(tmp_path):
    path = _plan(tmp_path, supplementary=("evidence.csv",))
    document = stages.render_review_document(
        "predict cell type", path, json.loads(path.read_text())
    )

    assert "predict cell type" in document
    assert "`cell_type`" in document
    assert "gold.csv" in document
    assert "evidence.csv" in document
    # The handoff from stage 1 to stage 2 is the experiment command itself.
    assert "--data-plan" in document
    assert "experiment" in document


def test_fetch_stage_writes_data_md_and_never_trains(tmp_path, monkeypatch):
    def fake_fetch_and_plan(*, question, out_dir, **kwargs):
        return _plan(tmp_path), "fetch log"

    def no_training(*args, **kwargs):
        raise AssertionError("stage 1 must not train")

    monkeypatch.setattr(stages, "fetch_and_plan", fake_fetch_and_plan)
    monkeypatch.setattr(stages, "run_data_task", no_training)

    outcome = stages.fetch_stage(question="predict cell type", out_dir=tmp_path)

    assert outcome.ok
    assert outcome.stage == "fetched"
    assert outcome.plan_path == tmp_path / "plan.json"
    assert (tmp_path / "data.md").exists()
    assert outcome.metrics["data_md"] == str(tmp_path / "data.md")
    # Stage 1 stamps the profile so stage 2 does not have to re-read the wording.
    plan = json.loads((tmp_path / "plan.json").read_text())
    assert plan["task"]["backend"] == "per_cell"


def test_fetch_stage_reports_a_fetch_that_produced_no_plan(tmp_path, monkeypatch):
    monkeypatch.setattr(
        stages, "fetch_and_plan", lambda **kwargs: (None, "line one\nline two")
    )

    outcome = stages.fetch_stage(question="anything", out_dir=tmp_path)

    assert not outcome.ok
    assert outcome.stage == "fetch"
    assert "line two" in outcome.error


def test_fetch_stage_forwards_candidates(tmp_path, monkeypatch):
    seen: dict = {}

    def fake_fetch_and_plan(**kwargs):
        seen.update(kwargs)
        return _plan(tmp_path), "log"

    monkeypatch.setattr(stages, "fetch_and_plan", fake_fetch_and_plan)

    stages.fetch_stage(
        question="knock out JAK2",
        out_dir=tmp_path,
        backend="perturbation",
        candidates=("screen_metadata.tsv",),
    )

    assert seen["candidates"] == ("screen_metadata.tsv",)


def test_a_perturbation_fetch_carries_the_condition_hints(tmp_path, monkeypatch):
    seen: dict = {}

    def fake_fetch_and_plan(**kwargs):
        seen.update(kwargs)
        return _plan(tmp_path), "log"

    monkeypatch.setattr(stages, "fetch_and_plan", fake_fetch_and_plan)

    stages.fetch_stage(
        question="Does CRISPR knockout of JAK2 change macrophages?",
        out_dir=tmp_path,
        backend="perturbation",
    )

    # The connected path names a screen's columns too; a standalone stage 1
    # must not be weaker than it.
    assert "gene" in seen["hints"]
    # The profile's ask rides on `retrieval_ask`, not `intent`: it must not turn
    # the search on by itself (that is `--intent`'s job).
    assert seen["retrieval_ask"]
    assert seen["intent"] == ""


def test_named_candidates_do_not_force_a_search_round(tmp_path, monkeypatch):
    seen: dict = {}

    def fake_fetch_and_plan(**kwargs):
        seen.update(kwargs)
        return _plan(tmp_path), "log"

    monkeypatch.setattr(stages, "fetch_and_plan", fake_fetch_and_plan)

    stages.fetch_stage(
        question="Does CRISPR knockout of JAK2 change macrophages?",
        out_dir=tmp_path,
        backend="perturbation",
        candidates=("screen_metadata.tsv", "counts.tsv.gz"),
    )

    # A person who named the tables must not have retrieval forced back on by
    # our own ask for control cells.
    assert seen["intent"] == ""
    # The ask is still composed the same way -- it just cannot start a search.
    assert seen["retrieval_ask"]
    # The condition-column hints still help the review.
    assert "gene" in seen["hints"]


def test_fetch_stage_forwards_the_breadth_knobs(tmp_path, monkeypatch):
    seen: dict = {}

    def fake_fetch_and_plan(**kwargs):
        seen.update(kwargs)
        return _plan(tmp_path), "log"

    monkeypatch.setattr(stages, "fetch_and_plan", fake_fetch_and_plan)

    stages.fetch_stage(
        question="anything",
        out_dir=tmp_path,
        fetch_limit=6,
        supp_limit=4,
        search_hops=5,
        max_files_per_fetch=32,
        archive_members=64,
        fetch_retries=2,
        mechanical_search=True,
        refresh=True,
    )

    assert seen["fetch_limit"] == 6
    assert seen["supp_limit"] == 4
    assert seen["search_hops"] == 5
    assert seen["max_files_per_fetch"] == 32
    assert seen["archive_members"] == 64
    assert seen["fetch_retries"] == 2
    assert seen["mechanical_search"] is True
    assert seen["refresh"] is True


def test_experiment_stage_trains_the_plan_without_fetching(tmp_path, monkeypatch):
    plan = _plan(tmp_path)
    seen: dict = {}

    def fake_run_data_task(**kwargs):
        seen.update(kwargs)
        return DataTaskOutcome(
            ok=True, stage="done", plan_path=plan, run_dir=tmp_path / "run"
        )

    monkeypatch.setattr(stages, "run_data_task", fake_run_data_task)

    outcome = stages.experiment_stage(question="predict cell type", plan_path=plan)

    assert outcome.ok
    assert seen["fetch"] is False
    assert seen["plan_path"] == str(plan)
    # The plan is already reviewed; the stage-1 gates must not veto it.
    assert seen["require_data_question"] is False
    assert seen["require_single_cell"] is False
    # Without an --out the run lands beside the plan.
    assert seen["out_dir"] == tmp_path


def test_experiment_stage_refuses_a_missing_plan(tmp_path):
    outcome = stages.experiment_stage(
        question="predict cell type", plan_path=tmp_path / "nope.json"
    )

    assert not outcome.ok
    assert outcome.stage == "experiment"
    assert "dataprep" in outcome.error


def test_fetch_task_kind_pins_the_profile_or_reads_the_question():
    assert stages.fetch_task_kind("perturbation", "any words") == "perturbation"
    assert stages.fetch_task_kind("per_cell", "knock out STAT1") == "per_cell"
    # No explicit profile: a perturbation question asks for a screen.
    assert stages.fetch_task_kind(None, "knock out STAT1 with CRISPR") == "perturbation"


def test_experiment_stage_takes_its_backend_from_the_plan(tmp_path, monkeypatch):
    plan = _plan(tmp_path, target="gene")
    stages.stamp_plan_backend(plan, "perturbation")
    seen: dict = {}

    def fake_run_data_task(**kwargs):
        seen.update(kwargs)
        return DataTaskOutcome(ok=True, stage="done", plan_path=plan)

    monkeypatch.setattr(stages, "run_data_task", fake_run_data_task)

    stages.experiment_stage(question="any words at all", plan_path=plan)

    assert stages.plan_backend(plan) == "perturbation"
    assert seen["backend"] == "perturbation"


def test_an_explicit_backend_beats_the_plan(tmp_path, monkeypatch):
    plan = _plan(tmp_path)
    stages.stamp_plan_backend(plan, "per_cell")
    seen: dict = {}
    monkeypatch.setattr(
        stages,
        "run_data_task",
        lambda **kwargs: (seen.update(kwargs), DataTaskOutcome(ok=True, stage="done"))[1],
    )

    stages.experiment_stage(question="q", plan_path=plan, backend="perturbation")

    assert seen["backend"] == "perturbation"


def test_a_plan_without_a_backend_stamp_is_unreadable_not_wrong(tmp_path):
    plan = _plan(tmp_path)
    assert stages.plan_backend(plan) is None
