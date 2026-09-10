"""A run must not report more than it did.

Three defects, one theme: the report described work that did not happen.

  1. An experiment that executed and returned nothing was stored COMPLETED,
     counted as successful, and printed under "Research completed
     successfully!" over `data: {}`.
  2. When code generation fell back to a single-table template, the report
     still carried the multi-dataset Design text -- so a correlation between
     two columns of one table was narrated as a colocalisation study.
  3. Experiments designed but never started were dropped from the export
     entirely, making a run that ran out of iterations look like a run that
     designed nothing.

The distinction each test protects is between "no effect was found" (a finding)
and "nothing ran" (not one).
"""

from __future__ import annotations

import pytest

from kosmos.cli.commands.run import (
    _experiment_produced_output,
    _n_successful_experiments,
)
from kosmos.cli.views.results_viewer import ResultsViewer
from kosmos.execution.code_generator import ExperimentCodeGenerator


# --- 1. empty output is not success -----------------------------------------

def _exp(status="completed", **result):
    return {"status": status, "results": [result] if result else []}


def test_a_completed_experiment_that_returned_nothing_is_not_successful():
    """The observed failure: status COMPLETED, `data: {}`, counted as a win."""
    assert _n_successful_experiments([_exp(data={})]) == 0


def test_an_experiment_with_no_result_row_at_all_is_not_successful():
    assert _n_successful_experiments([_exp()]) == 0


def test_a_null_result_is_still_a_success():
    """A non-significant p-value is a finding; only silence is not."""
    exp = _exp(data={"n_samples": 400}, p_value=0.87, effect_size=0.01)
    assert _n_successful_experiments([exp]) == 1


def test_a_p_value_alone_counts_as_output():
    assert _experiment_produced_output(_exp(data={}, p_value=0.5))


def test_statistical_tests_alone_count_as_output():
    assert _experiment_produced_output(_exp(data={}, statistical_tests={"t": 1.0}))


def test_the_fallback_note_alone_is_not_output():
    """Our own annotation must not make an empty experiment look productive."""
    assert not _experiment_produced_output(_exp(data={"analysis_note": "fell back"}))


def test_a_failed_experiment_with_output_is_still_not_successful():
    exp = _exp(status="failed", data={"n_samples": 10}, p_value=0.01)
    assert _n_successful_experiments([exp]) == 0


# --- 2. a fallback must announce itself -------------------------------------

def _gen(datasets, last):
    gen = ExperimentCodeGenerator.__new__(ExperimentCodeGenerator)
    gen.datasets = datasets
    gen.last_generation = last
    return gen


def test_no_note_when_the_designed_multi_dataset_code_ran():
    gen = _gen({"a": "/a.csv", "b": "/b.csv"},
               {"path": "multi_dataset_llm", "template": None, "fallback_reason": None})
    assert gen.generation_note() is None


def test_the_note_names_the_template_and_the_reason():
    gen = _gen({"a": "/a.csv", "b": "/b.csv"},
               {"path": "template", "template": "correlation_analysis",
                "fallback_reason": "ValueError: provider down"})
    note = gen.generation_note()

    assert "NOT the designed analysis" in note
    assert "correlation_analysis" in note
    assert "2 datasets" in note
    assert "provider down" in note


def test_a_single_dataset_template_run_is_disclosed_not_accused():
    """One dataset means a template is the designed path -- but the reader
    still has to be told the numbers came from a generic template rather than
    from code written for this protocol. This asserted `is None` until a run
    printed a nested-logistic-regression design over the ml_experiment
    template's accuracy and CV, with nothing marking the difference."""
    gen = _gen({"only": "/a.csv"},
               {"path": "template", "template": "ttest_comparison", "fallback_reason": None})

    note = gen.generation_note()

    assert "ttest_comparison" in note
    assert "not by code written for this protocol" in note
    assert "NOT the designed analysis" not in note, "a template is not a fallback here"


def test_the_basic_fallback_also_produces_a_note():
    gen = _gen({"a": "/a.csv", "b": "/b.csv"},
               {"path": "basic_template", "template": None, "fallback_reason": None})
    assert "generic single-table fallback" in gen.generation_note()


# --- 3. the report must distinguish truncated from empty --------------------

def _md(tmp_path, data):
    out = tmp_path / "r.md"
    ResultsViewer().export_to_markdown(data, out)
    return out.read_text()


def test_a_truncated_run_says_so_and_lists_what_was_designed(tmp_path):
    """The observed report: an empty Experiments section, work invisible."""
    text = _md(tmp_path, {
        "question": "q", "experiments": [], "current_iteration": 4, "max_iterations": 4,
        "pending_experiments": [{
            "experiment_type": "computational", "status": "CREATED",
            "description": "Two-sample Mendelian Randomization of Plasma Proteins",
        }],
    })

    assert "None executed" in text
    assert "cut short, not empty" in text
    assert "## Designed but not run" in text
    assert "Mendelian Randomization" in text


def test_a_run_that_designed_nothing_says_that_instead(tmp_path):
    text = _md(tmp_path, {"question": "q", "experiments": [], "pending_experiments": []})

    assert "No experiment was designed or executed" in text
    assert "## Designed but not run" not in text


def test_the_fallback_note_is_rendered_above_the_numbers(tmp_path):
    """Order matters: the caveat has to arrive before the reader does."""
    text = _md(tmp_path, {
        "question": "q",
        "experiments": [{
            "type": "computational", "status": "completed", "duration_seconds": 60,
            "description": "Intersect cis-pQTLs with T1 hits and run coloc.abf",
            "results": [{"data": {"n_samples": 3791847, "analysis_note": "NOT the designed analysis: fell back"},
                         "p_value": 0.0}],
        }],
    })

    assert "> NOT the designed analysis" in text
    assert text.index("NOT the designed analysis") < text.index("Sample size")


def test_an_ordinary_result_gains_no_note(tmp_path):
    text = _md(tmp_path, {
        "question": "q",
        "experiments": [{
            "type": "computational", "status": "completed", "duration_seconds": 1,
            "results": [{"data": {"n_samples": 100}, "p_value": 0.02}],
        }],
    })

    assert "NOT the designed analysis" not in text
    assert "Sample size: 100" in text


# --- 4. a quarantined failure must not vanish -------------------------------

def test_a_failed_experiment_is_neither_queued_nor_completed():
    """The list that has to exist for a quarantined protocol to be reportable.

    On failure the director drops the protocol from `experiment_queue` so it
    cannot block the head, and it never reaches `completed_experiments`. With
    only those two lists, a run whose experiment died in the sandbox exported
    an empty Experiments section and the report said no experiment was
    designed -- which is what a Docker outage looked like from the outside.
    """
    from kosmos.core.workflow import ResearchPlan

    plan = ResearchPlan(research_question="q")
    assert plan.failed_experiments == []

    plan.experiment_queue.append("p1")
    plan.failed_experiments.append("p1")
    plan.experiment_queue.remove("p1")

    assert plan.failed_experiments == ["p1"]
    assert "p1" not in plan.experiment_queue
    assert "p1" not in plan.completed_experiments


def test_a_failed_experiment_renders_with_its_error(tmp_path):
    text = _md(tmp_path, {
        "question": "q",
        "experiments": [{
            "type": "computational", "status": "FAILED", "duration_seconds": 0,
            "description": "Two-sample Mendelian Randomization of Plasma Proteins",
            "error_message": "Docker not available: Connection refused",
            "results": [],
        }],
        "pending_experiments": [],
    })

    assert "Docker not available" in text
    assert "No experiment was designed" not in text


def test_a_failed_experiment_is_not_counted_as_successful():
    exp = {"status": "FAILED", "error_message": "Docker not available", "results": []}
    assert _n_successful_experiments([exp]) == 0


# --- 5. a second draft for silence, not only for exceptions -----------------

def test_the_empty_payload_helper_treats_empty_as_no_output():
    """`data: {}` after 16 seconds of work is silence, not a result."""
    def _payload(res):
        rv = getattr(res, "return_value", None)
        return rv if isinstance(rv, dict) and rv else {}

    class _R:
        def __init__(self, rv):
            self.return_value = rv

    assert _payload(_R({})) == {}
    assert _payload(_R(None)) == {}
    assert _payload(_R([1, 2])) == {}
    assert _payload(_R({"n": 0})) == {"n": 0}, "a zero COUNT is output; an empty dict is not"


def test_the_prompt_requires_a_module_level_non_empty_results():
    from kosmos.execution.code_generator import ExperimentCodeGenerator

    gen = ExperimentCodeGenerator.__new__(ExperimentCodeGenerator)
    gen.dataset_context = None
    text = gen._data_access_instructions({"a": "/a.csv", "b": "/b.csv"})

    # Normalise: the instruction is wrapped across lines in the prompt.
    flat = " ".join(text.split())
    assert "AT MODULE LEVEL" in flat
    assert "a null finding is a finding, silence is not" in flat


def test_the_regeneration_message_explains_what_to_do_with_a_null_result(tmp_path):
    """The correction has to say where the numbers go, not just that they are missing."""
    from pathlib import Path

    src = Path("kosmos/agents/research_director.py").read_text()
    i = src.index("assigned no non-empty")
    message = src[i - 200:i + 700]

    assert "MODULE level" in message
    assert "that is a FINDING" in message


# --- 6. the payload IS the answer; the report cannot pick from it ------------

class TestPayloadRendering:
    """A successful run exported an EMPTY Findings section.

    The renderer knew four keys -- n_samples, p_value, effect_size,
    statistical_tests -- and dropped everything else. The experiment had
    returned `causal_proteins`, `all_ranked` (14 proteins with IVW / weighted
    median / MR-Egger estimates), `coloc_summary`, `mr_summary` and `notes`,
    all of which sat in the database unreported.
    """

    def test_a_ranked_list_of_dicts_becomes_a_table(self):
        from kosmos.cli.views.results_viewer import _render_payload

        out = "\n".join(_render_payload({
            "all_ranked": [
                {"gene": "SOD2", "ivw_beta": -0.2815186, "ivw_p_adj": 0.0},
                {"gene": "CSF2RB", "ivw_beta": 0.7061848, "ivw_p_adj": 0.0},
            ],
        }))

        assert "**all_ranked** (2 rows)" in out
        assert "| gene | ivw_beta | ivw_p_adj |" in out
        assert "| SOD2 | -0.2815 | 0 |" in out

    def test_a_long_table_says_what_it_dropped(self):
        """A silently truncated table reads as the whole result."""
        from kosmos.cli.views.results_viewer import _render_payload

        rows = [{"gene": f"G{i}", "beta": i / 10} for i in range(30)]
        out = "\n".join(_render_payload({"all_ranked": rows}))

        assert "_18 further rows not shown._" in out

    def test_an_empty_list_is_reported_as_none_not_omitted(self):
        """`causal_proteins: []` is a finding: nothing passed the threshold."""
        from kosmos.cli.views.results_viewer import _render_payload

        assert "- causal_proteins: none" in "\n".join(
            _render_payload({"causal_proteins": []})
        )

    def test_notes_are_carried_through(self):
        from kosmos.cli.views.results_viewer import _render_payload

        out = "\n".join(_render_payload({
            "notes": ["heart_sqtl and heart_eqtl could not be linked: different genome builds."],
        }))

        assert "could not be linked" in out

    def test_already_rendered_keys_are_not_repeated(self):
        from kosmos.cli.views.results_viewer import _render_payload

        out = "\n".join(_render_payload({
            "n_samples": 100, "p_value": 0.01, "effect_size": -0.3,
            "statistical_tests": {"t": 1.0}, "analysis_note": "fell back",
            "extra": 7,
        }))

        assert out.strip() == "- extra: 7"

    def test_nan_does_not_break_the_report(self):
        from kosmos.cli.views.results_viewer import _render_payload

        assert "NaN" in "\n".join(_render_payload({"beta": float("nan")}))

    def test_the_real_payload_reaches_the_exported_markdown(self, tmp_path):
        text = _md(tmp_path, {
            "question": "q",
            "experiments": [{
                "type": "computational", "status": "completed", "duration_seconds": 13,
                "results": [{"data": {
                    "causal_proteins": [],
                    "all_ranked": [{"gene": "SOD2", "ivw_beta": -0.2815}],
                    "notes": ["coloc not computable from a hits-only release"],
                }}],
            }],
        })

        assert "SOD2" in text
        assert "causal_proteins: none" in text
        assert "coloc not computable" in text


# --- 7. an error message is not a result ------------------------------------

def test_an_error_only_payload_is_not_a_successful_experiment():
    """Observed: a "completed" experiment whose entire output was one error line.

    `{"error": "Gene symbol-to-Ensembl mapping required (mygene) not
    available."}` was counted as a success, reported with no p-value and no
    finding -- the failure dressed as an answer.
    """
    exp = _exp(data={"error": "Gene symbol-to-Ensembl mapping required (mygene) not available."})

    assert not _experiment_produced_output(exp)
    assert _n_successful_experiments([exp]) == 0


def test_an_error_alongside_real_numbers_still_counts():
    """A run that reports a caveat AND results has produced output."""
    exp = _exp(data={"error": "coloc skipped", "n_proteins": 14, "top_beta": -0.28})

    assert _experiment_produced_output(exp)


def test_a_traceback_only_payload_is_not_output():
    assert not _experiment_produced_output(_exp(data={"traceback": "Traceback ..."}))
