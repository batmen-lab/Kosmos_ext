"""The discovery loop's new branch: fetch real data, train, report metrics.

The branch sits between the plan-driven training path (untouched) and the
LLM-code path (the previous behaviour). These tests hold the seam: the switch,
the shape of the result the director stores, and the fallback when the fetch
cannot ground the question.
"""

from __future__ import annotations

import pytest

from kosmos.agents.research_director import ResearchDirectorAgent
from kosmos.ppi.discovery_bridge import DataTaskOutcome
from kosmos.ppi.task_ontology import classify_task


class FakeProtocol:
    id = "12345678-0000-0000-0000-000000000000"
    name = "Predict the cell type from expression"
    description = "Train a classifier and report its held-out accuracy."


@pytest.fixture
def director(monkeypatch):
    monkeypatch.setenv("KOSMOS_DISCOVERY_DATA_TASKS", "fallback")
    return ResearchDirectorAgent(
        research_question="Can the cell type be predicted from expression?",
        domain="biology",
        config={"max_iterations": 1},
    )


def test_the_switch_is_the_caller_s_word(director, monkeypatch):
    monkeypatch.setenv("KOSMOS_DISCOVERY_DATA_TASKS", "off")
    assert director.discovery_data_tasks_enabled() is False
    for value in ("fallback", "on", "data_tasks"):
        monkeypatch.setenv("KOSMOS_DISCOVERY_DATA_TASKS", value)
        assert director.discovery_data_tasks_enabled() is True


@pytest.mark.asyncio
async def test_a_data_task_returns_the_metrics_the_director_stores(director, monkeypatch, tmp_path):
    """The result carries numbers, which a training experiment did not before."""
    from kosmos.ppi import discovery_bridge

    def fake_run(**kwargs):
        assert kwargs["question"] == director.research_question
        assert "Predict the cell type" in kwargs["extra_text"]
        assert kwargs["domain"] == "biology"
        return DataTaskOutcome(
            ok=True,
            stage="done",
            kind=classify_task(kwargs["question"]),
            metrics={
                "metric": "balanced_accuracy",
                "value": 0.91,
                "baseline": 0.88,
                "delta": 0.03,
                "ci_low": 0.87,
                "ci_high": 0.94,
                "accuracy": 0.93,
                "balanced_accuracy": 0.91,
                "loss_mode": "gradient_gated",
                "kind": "prediction",
            },
            summary_path=tmp_path / "run" / "summary.md",
        )

    monkeypatch.setattr(discovery_bridge, "run_data_task", fake_run)

    result = await director._execute_data_task(FakeProtocol())

    assert result.success is True
    assert result.data_source == "data_task"
    payload = result.return_value
    assert payload["balanced_accuracy"] == 0.91
    assert payload["value"] == 0.91
    assert payload["effect_size"] == 0.03      # the delta the director stores
    assert payload["loss_mode"] == "gradient_gated"
    assert payload["data_source"] == "data_task"


@pytest.mark.asyncio
async def test_a_skipped_question_falls_through_to_the_code_path(director, monkeypatch):
    """A simulation is not a failure: the caller is told to use its code path."""
    from kosmos.ppi import discovery_bridge

    monkeypatch.setattr(
        discovery_bridge,
        "run_data_task",
        lambda **kwargs: DataTaskOutcome(
            ok=False, stage="skipped", error="answered by simulation, not data"
        ),
    )

    assert await director._execute_data_task(FakeProtocol()) is None


@pytest.mark.asyncio
async def test_a_failed_fetch_follows_the_configured_mode(director, monkeypatch):
    """`fallback` keeps the old path; `on` reports the failure instead."""
    from kosmos.ppi import discovery_bridge

    monkeypatch.setattr(
        discovery_bridge,
        "run_data_task",
        lambda **kwargs: DataTaskOutcome(
            ok=False, stage="fetch", error="the fetcher produced no plan"
        ),
    )

    monkeypatch.setenv("KOSMOS_DISCOVERY_DATA_TASKS", "fallback")
    assert await director._execute_data_task(FakeProtocol()) is None

    monkeypatch.setenv("KOSMOS_DISCOVERY_DATA_TASKS", "on")
    result = await director._execute_data_task(FakeProtocol())
    assert result.data_source == "data_task_failed"
    assert result.return_value["stage"] == "fetch"
