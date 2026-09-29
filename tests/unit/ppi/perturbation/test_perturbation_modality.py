"""Judging what *kind* of perturbation a question asks for and a screen provides.

The pipeline is modality-blind (it learns Delta either way), so a "knock out
CBL" question used to be answered with Norman's CRISPRa (over-expression)
screen without a word. These tests pin the three judgements and the wiring.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")

from kosmos.ppi.perturbation.modality import (  # noqa: E402
    check_modality,
    infer_requested_modality,
    modality_of,
)


@pytest.mark.parametrize(
    "text,expected",
    [
        ("CRISPRa (sgRNA)", "crispra"),
        ("CRISPRi (genome-wide, filtered)", "crispri"),
        ("CRISPRi (sgRNA)", "crispri"),
        ("CRISPRi / CRISPRa", "mixed"),
        ("Perturb-seq with a transcription-factor library", "unknown"),
        ("", "unknown"),
    ],
)
def test_a_screen_description_is_normalised_to_a_modality(text, expected):
    assert modality_of(text) == expected


@pytest.mark.parametrize(
    "question,expected",
    [
        ("Does knocking out CBL with CRISPR change the transcriptome?", "crisprko"),
        ("Can the response to a double knockout be predicted?", "crisprko"),
        ("Does over-expressing CBL change the transcriptome?", "crispra"),
        ("Does CRISPRi knockdown of CBL reduce proliferation?", "crispri"),
        ("Does CRISPR change these cells?", "unknown"),
    ],
)
def test_the_question_names_the_modality_it_wants(question, expected):
    verdict = infer_requested_modality(question)
    assert verdict["modality"] == expected
    assert verdict["hits"]


def test_a_mismatch_is_a_mismatch_and_a_mixed_screen_is_compatible():
    assert check_modality("crisprko", "crispra")["status"] == "mismatch"
    assert check_modality("crisprko", "crispri")["status"] == "mismatch"
    assert check_modality("crisprko", "mixed")["status"] == "compatible"
    assert check_modality("crispra", "crispra")["status"] == "match"
    assert check_modality("crisprko", "unknown")["status"] == "unchecked"
    assert check_modality("unknown", "crispra")["status"] == "unchecked"


def _stub_run(tmp_path, monkeypatch, kind="perturbation_response"):
    """Registry staging + a stubbed trainer, so the wiring is what is tested."""
    import kosmos.ppi.discovery_bridge as bridge
    from kosmos.ppi.perturbation import run as run_module
    from kosmos.ppi.perturbation import stage as stage_module
    from kosmos.ppi.task_ontology import TaskKind

    gold = tmp_path / "gold.csv"
    pd.DataFrame({"condition": ["ctrl", "A"], "G0": [0.0, 1.0]}).to_csv(gold, index=False)
    seen: dict = {}

    def fake_stage(*, gold, supplementary, condition_column="condition", echo=True):
        return gold_path, [], {"sources": {}}

    # the source object the registry hands back carries the modality string
    from kosmos.ppi.perturbation.sources import SOURCES

    class FakeStage:
        pass

    def stage_patch(*, gold, supplementary, condition_column="condition", echo=True):
        seen["gold_perturbation"] = gold.perturbation
        return gold_path, [], {"sources": {}}

    gold_path = gold

    def fake_run(**kwargs):
        seen["modality"] = kwargs.get("modality")
        arm = {
            "test": {
                "summary": {
                    "mse": 0.5, "mse_deg": 0.4, "pearson": 0.2,
                    "spearman": 0.1, "direction_accuracy": 0.8, "top_k_overlap": 0.2,
                },
                "n_perturbations": 2,
            }
        }
        return {
            "task": {"n_genes": 2, "condition_column": "condition", "control_labels": ["ctrl"]},
            "split_sizes": {"train": 1, "validation": 1, "test": 1},
            "graph_edge_overlap": {"shared": 0},
            "sources": {},
            "figures": [],
            "arms": {"gears_base": arm, "gears_augmented_ungated": arm, "gears_augmented": arm},
        }

    monkeypatch.setattr(stage_module, "stage_sources", stage_patch)
    monkeypatch.setattr(run_module, "run_perturbation_task", fake_run)
    kind_obj = TaskKind(
        kind=kind, loss_mode="graph_gated", rationale="t", decided_by="config",
        backend="perturbation",
    )
    return bridge, kind_obj, seen, gold_path


def test_a_knockout_question_on_a_crispra_screen_is_flagged(tmp_path, monkeypatch, capsys):
    bridge, kind, seen, gold_path = _stub_run(tmp_path, monkeypatch)
    monkeypatch.setenv("KOSMOS_PERTURBATION_MODALITY", "warn")

    outcome = bridge.run_perturbation_outcome(
        question="Does knocking out CBL change the transcriptome?",
        gold_table=gold_path,   # skips the registry so no network is needed
        out_dir=tmp_path / "out", kind=kind, seed=0, echo=True,
    )

    assert outcome.ok, outcome.error
    # with a caller-supplied table the data declares no modality -> unchecked
    assert outcome.metrics["modality_requested"] == "crisprko"
    assert outcome.metrics["modality_provided"] == "unknown"
    assert outcome.metrics["modality_verdict"] == "unchecked"
    assert (tmp_path / "out" / "data_report.md").exists()


def test_the_registry_source_declares_its_modality(tmp_path, monkeypatch):
    """The registry path records what the screen actually is."""
    import kosmos.ppi.discovery_bridge as bridge
    from kosmos.ppi.perturbation import run as run_module
    from kosmos.ppi.perturbation import stage as stage_module
    from kosmos.ppi.perturbation.sources import SOURCES
    from kosmos.ppi.task_ontology import TaskKind

    seen: dict = {}
    gold_path = tmp_path / "gold.csv"
    pd.DataFrame({"condition": ["ctrl"], "G0": [0.0]}).to_csv(gold_path, index=False)

    def stage_patch(*, gold, supplementary, condition_column="condition", echo=True):
        seen["perturbation_string"] = gold.perturbation
        return gold_path, [], {"sources": {}}

    def fake_run(**kwargs):
        seen["modality"] = kwargs.get("modality")
        arm = {"test": {"summary": {"mse": 1.0, "mse_deg": 1.0, "pearson": 0.0,
                                    "spearman": 0.0, "direction_accuracy": 0.5,
                                    "top_k_overlap": 0.0}, "n_perturbations": 1}}
        return {"task": {"n_genes": 2, "condition_column": "condition", "control_labels": ["ctrl"]},
                "split_sizes": {"train": 1, "validation": 1, "test": 1},
                "graph_edge_overlap": {}, "sources": {}, "figures": [],
                "arms": {"gears_base": arm, "gears_augmented_ungated": arm, "gears_augmented": arm}}

    monkeypatch.setattr(stage_module, "stage_sources", stage_patch)
    monkeypatch.setattr(run_module, "run_perturbation_task", fake_run)

    kind = TaskKind(kind="perturbation_response", loss_mode="graph_gated",
                    rationale="t", decided_by="config", backend="perturbation")
    outcome = bridge.run_perturbation_outcome(
        question="Does knocking out CBL change the transcriptome?",
        out_dir=tmp_path / "out", kind=kind, seed=0, echo=False,
    )

    assert outcome.ok, outcome.error
    assert seen["perturbation_string"].startswith("CRISPRa")
    assert seen["modality"]["provided"]["gold"] == "crispra"
    assert outcome.metrics["modality_verdict"] == "mismatch"
    assert outcome.metrics["modality_provided"] == "crispra"


def test_strict_mode_refuses_a_mismatch(tmp_path, monkeypatch):
    import kosmos.ppi.discovery_bridge as bridge
    from kosmos.ppi.perturbation import run as run_module
    from kosmos.ppi.perturbation import stage as stage_module
    from kosmos.ppi.task_ontology import TaskKind

    gold_path = tmp_path / "gold.csv"
    pd.DataFrame({"condition": ["ctrl"], "G0": [0.0]}).to_csv(gold_path, index=False)

    def stage_patch(*, gold, supplementary, condition_column="condition", echo=True):
        return gold_path, [], {"sources": {}}

    def fake_run(**kwargs):  # pragma: no cover - strict mode must never get here
        raise AssertionError("training ran despite a strict modality refusal")

    monkeypatch.setattr(stage_module, "stage_sources", stage_patch)
    monkeypatch.setattr(run_module, "run_perturbation_task", fake_run)
    monkeypatch.setenv("KOSMOS_PERTURBATION_MODALITY", "strict")

    kind = TaskKind(kind="perturbation_response", loss_mode="graph_gated",
                    rationale="t", decided_by="config", backend="perturbation")
    with pytest.raises(ValueError, match="asks for crisprko"):
        bridge.run_perturbation_outcome(
            question="Does knocking out CBL change the transcriptome?",
            out_dir=tmp_path / "out", kind=kind, seed=0, echo=False,
        )
