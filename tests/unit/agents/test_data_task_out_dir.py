"""The data-task output directory must not depend on the protocol carrying an id.

A protocol rebuilt from the stored JSON has no `id` (it lives on the DB row), so
slicing `protocol.id[:8]` raised `TypeError: 'NoneType' object is not
subscriptable` and the experiment produced no artifacts at all.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

from kosmos.agents.research_director import ResearchDirectorAgent
from kosmos.ppi.discovery_bridge import DataTaskOutcome


def _director(config):
    director = ResearchDirectorAgent.__new__(ResearchDirectorAgent)
    director.config = config
    director.research_question = "a question"
    director.domain = "biology"
    director._llm_client_for_tasks = lambda: None
    return director


def test_a_protocol_without_an_id_still_gets_an_output_dir(monkeypatch):
    import kosmos.ppi.discovery_bridge as bridge

    captured: dict = {}

    def fake_run_data_task(**kwargs):
        captured.update(kwargs)
        return DataTaskOutcome(ok=False, stage="skipped", error="not a data question")

    monkeypatch.setattr(bridge, "run_data_task", fake_run_data_task)

    protocol = SimpleNamespace(id=None, name="an experiment", description="")
    asyncio.run(_director({})._execute_data_task(protocol))

    out_dir = str(captured["out_dir"])
    assert "artifacts/ppi/discovery-" in out_dir
    assert "None" not in out_dir


def test_an_explicit_output_dir_wins(monkeypatch, tmp_path):
    import kosmos.ppi.discovery_bridge as bridge

    captured: dict = {}

    def fake_run_data_task(**kwargs):
        captured.update(kwargs)
        return DataTaskOutcome(ok=False, stage="skipped", error="not a data question")

    monkeypatch.setattr(bridge, "run_data_task", fake_run_data_task)

    protocol = SimpleNamespace(id=None, name="an experiment", description="")
    asyncio.run(_director({"ppi_output_dir": str(tmp_path)})._execute_data_task(protocol))

    assert str(captured["out_dir"]) == str(tmp_path)
