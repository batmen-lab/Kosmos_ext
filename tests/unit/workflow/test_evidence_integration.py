"""The evidence extension is an explicit stage of ResearchWorkflow."""

from types import SimpleNamespace

import pytest

from kosmos.workflow.research_loop import ResearchWorkflow


class FakeEvidencePipeline:
    def __init__(self):
        self.calls = []

    def run(self, **kwargs):
        self.calls.append(kwargs)
        return {
            "records": [
                SimpleNamespace(decision="accept"),
                SimpleNamespace(decision="defer"),
                SimpleNamespace(decision="reject"),
            ],
            "evaluations": [{"datasets": ["external-1"], "delta": 0.02}],
        }


def bare_workflow(tmp_path, pipeline, candidates=None, discoverer=None):
    workflow = ResearchWorkflow.__new__(ResearchWorkflow)
    workflow.research_objective = "test evidence integration"
    workflow.evidence_pipeline = pipeline
    workflow.evidence_candidates = list(candidates or [])
    workflow.evidence_discoverer = discoverer
    workflow.evidence_predictor = object()
    workflow.evidence_trainer = object()
    workflow.evidence_config = object()
    workflow.evidence_output_dir = tmp_path / "evidence"
    workflow.evidence_results = []
    workflow.evidence_enabled = True
    return workflow


@pytest.mark.asyncio
async def test_evidence_stage_is_explicit_and_persisted(tmp_path):
    pipeline = FakeEvidencePipeline()
    workflow = bare_workflow(tmp_path, pipeline, ["candidate"])
    result = await workflow._execute_evidence_stage(1)
    assert result["accepted"] == 1
    assert result["deferred"] == 1
    assert result["rejected"] == 1
    assert workflow.evidence_results == [result]
    assert pipeline.calls[0]["candidates"] == ["candidate"]
    assert pipeline.calls[0]["output_dir"].endswith("cycle-001")


def test_evidence_constructor_parameters_are_opt_in():
    import inspect

    parameters = inspect.signature(ResearchWorkflow.__init__).parameters
    assert "evidence_pipeline" in parameters
    assert "evidence_discoverer" in parameters


def test_discoverer_is_called_by_evidence_stage(tmp_path):
    pipeline = FakeEvidencePipeline()
    calls = []

    def discoverer(objective, cycle, context):
        calls.append((objective, cycle, context))
        return ["discovered-candidate"]

    workflow = bare_workflow(tmp_path, pipeline, discoverer=discoverer)
    workflow.research_objective = "discover evidence"

    import asyncio

    result = asyncio.run(workflow._execute_evidence_stage(3, {"findings_count": 2}))
    assert result["datasets"] == 3
    assert calls == [("discover evidence", 3, {"findings_count": 2})]
    assert pipeline.calls[0]["candidates"] == ["discovered-candidate"]
