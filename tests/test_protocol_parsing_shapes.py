"""A protocol the designer could not parse became one built from boilerplate.

Observed in a single run:

    Skipping malformed protocol step: 1 validation error for ProtocolStep
      description: String should have at least 10 characters (input_value='')
    Skipping malformed protocol step: 'str' object has no attribute 'get'
    Skipping malformed protocol step: 'int' object has no attribute 'get'
    Skipping malformed protocol step: 'str' object has no attribute 'get'
    Skipping malformed protocol step: 'str' object has no attribute 'get'
    LLM returned empty variables, generating defaults from hypothesis

Every one of those lines is content the model produced and the parser threw
away. The experiment was then generated from generic defaults -- so the run
looked like a model that planned nothing, when it had planned in a shape the
parser did not accept.
"""

from __future__ import annotations

import pytest

from kosmos.agents.experiment_designer import ExperimentDesignerAgent
from kosmos.models.experiment import ExperimentType
from kosmos.models.hypothesis import Hypothesis


def _hypothesis():
    return Hypothesis(
        research_question="Which proteins causally affect myocardial fibrosis?",
        statement="Circulating SOD2 lowers native myocardial T1.",
        rationale="cis-pQTL instruments are available for SOD2 in the released data.",
        domain="biology",
    )


def _parse(data):
    agent = ExperimentDesignerAgent.__new__(ExperimentDesignerAgent)
    return ExperimentDesignerAgent._parse_claude_protocol(
        agent, data, _hypothesis(), ExperimentType.DATA_ANALYSIS
    )


# --- steps ------------------------------------------------------------------

def test_steps_given_as_plain_strings_are_kept():
    """`steps: ["...", "..."]` is a reasonable emission, and was discarded."""
    protocol = _parse({
        "name": "MR of plasma proteins on T1",
        "description": "Two-sample MR using cis-pQTL instruments against T1.",
        "objective": "Identify proteins with a causal effect on native T1.",
        "steps": [
            "Load the cis-pQTL and T1 GWAS tables",
            "Harmonise alleles to the T1 effect allele",
            "Compute the Wald ratio per protein",
        ],
    })

    assert len(protocol.steps) == 3
    assert "Harmonise alleles" in protocol.steps[1].description


def test_a_contentless_step_is_dropped_without_taking_its_neighbours():
    """`3` is not a step; a sentence is.

    The old parser dropped BOTH of these -- the integer raised, and so did
    every string beside it. Keeping the real step is the point; manufacturing
    one out of `3` would only put noise in the protocol.
    """
    protocol = _parse({
        "name": "Protein MR protocol", "description": "d" * 20, "objective": "o" * 20,
        "steps": ["Load the data tables", 3],
    })

    assert len(protocol.steps) == 1
    assert protocol.steps[0].description == "Load the data tables"


def test_a_step_carrying_its_text_only_in_action_survives():
    """`description` is required >= 10 chars and had no fallback."""
    protocol = _parse({
        "name": "Protein MR protocol", "description": "d" * 20, "objective": "o" * 20,
        "steps": [{"title": "Merge", "description": "", "action": "Merge on variant id"}],
    })

    assert len(protocol.steps) == 1
    assert "Merge on variant id" in protocol.steps[0].description


def test_a_short_description_is_expanded_not_dropped():
    protocol = _parse({
        "name": "Protein MR protocol", "description": "d" * 20, "objective": "o" * 20,
        "steps": [{"step_number": 2, "title": "Merge", "description": "Merge"}],
    })

    assert len(protocol.steps) == 1
    assert len(protocol.steps[0].description) >= 10
    assert "Merge" in protocol.steps[0].description


def test_a_real_step_dict_is_unchanged():
    protocol = _parse({
        "name": "Protein MR protocol", "description": "d" * 20, "objective": "o" * 20,
        "steps": [{
            "step_number": 1, "title": "Wald ratio",
            "description": "Compute the Wald ratio for each protein instrument.",
            "action": "Divide the T1 beta by the pQTL beta.",
        }],
    })

    step = protocol.steps[0]
    assert step.title == "Wald ratio"
    assert step.description.startswith("Compute the Wald ratio")


# --- variables --------------------------------------------------------------

def test_variables_given_as_a_list_do_not_empty_the_protocol():
    """`.items()` on a list raised AttributeError and took the protocol with it."""
    protocol = _parse({
        "name": "Protein MR protocol", "description": "d" * 20, "objective": "o" * 20,
        "steps": ["Load the data tables"],
        "variables": [
            {"name": "protein", "type": "independent", "description": "Assayed protein."},
            {"name": "t1_beta", "type": "dependent", "description": "Effect on native T1."},
        ],
    })

    assert set(protocol.variables) == {"protein", "t1_beta"}


def test_variables_given_as_name_to_description_strings_are_kept():
    protocol = _parse({
        "name": "Protein MR protocol", "description": "d" * 20, "objective": "o" * 20,
        "steps": ["Load the data tables"],
        "variables": {"protein": "The assayed plasma protein"},
    })

    assert "protein" in protocol.variables
    assert protocol.variables["protein"].description == "The assayed plasma protein"


def test_genuinely_absent_variables_still_fall_back():
    """The default path stays -- it just stops firing on parseable input."""
    protocol = _parse({
        "name": "Protein MR protocol", "description": "d" * 20, "objective": "o" * 20,
        "steps": ["Load the data tables"],
    })

    assert protocol.variables, "a protocol with no variables at all should get defaults"


def test_a_string_where_variables_should_be_is_ignored_not_fatal():
    protocol = _parse({
        "name": "Protein MR protocol", "description": "d" * 20, "objective": "o" * 20,
        "steps": ["Load the data tables"],
        "variables": "protein and t1",
    })

    assert protocol.variables


# --- one bad item must not cost the protocol --------------------------------

def test_a_short_variable_description_does_not_abort_the_protocol():
    """`description` is required at >= 10 chars; "Age" is a plausible answer.

    Unguarded, one such variable raised ValidationError out of the parse and
    took every step and every other variable with it.
    """
    protocol = _parse({
        "name": "Protein MR protocol", "description": "d" * 20, "objective": "o" * 20,
        "steps": ["Load the cis-pQTL and T1 tables"],
        "variables": {
            "age": {"type": "independent", "description": "Age"},
            "t1": {"type": "dependent", "description": "Native myocardial T1 time."},
        },
    })

    assert set(protocol.variables) == {"age", "t1"}
    assert len(protocol.variables["age"].description) >= 10
    assert protocol.steps[0].description == "Load the cis-pQTL and T1 tables"


def test_an_unknown_variable_type_falls_back_rather_than_aborting():
    protocol = _parse({
        "name": "Protein MR protocol", "description": "d" * 20, "objective": "o" * 20,
        "steps": ["Load the cis-pQTL and T1 tables"],
        "variables": {"x": {"type": "instrumental_variable", "description": "A cis-pQTL instrument."}},
    })

    assert "x" in protocol.variables


def test_a_truly_unparseable_variable_is_skipped_not_fatal():
    protocol = _parse({
        "name": "Protein MR protocol", "description": "d" * 20, "objective": "o" * 20,
        "steps": ["Load the cis-pQTL and T1 tables"],
        "variables": {
            "good": {"type": "dependent", "description": "Native myocardial T1 time."},
            "bad": {"type": "dependent", "description": "Fine.", "values": object()},
        },
    })

    assert "good" in protocol.variables
