"""Rules and model disagree: who wins, and when a person has to decide."""

from __future__ import annotations

from kosmos.discovery import Disagreement, parse_answer, resolve


def test_agreement_passes_through():
    assert resolve("gold", "gold") == "gold"
    assert resolve("supplementary", "unlabeled") == "supplementary"
    assert resolve("unusable", "unusable") == "unusable"


def test_a_model_that_reads_nothing_usable_cannot_be_overruled_by_rules():
    assert resolve("gold", "bad_label") == "supplementary"
    assert resolve("gold", "unusable") == "unusable"
    assert resolve("supplementary", "unusable") == "unusable"


def test_rules_refusing_and_the_model_accepting_goes_to_a_person():
    """The one case the model argues for more access than the rules allow."""
    assert resolve("unusable", "gold") == "pending"
    assert resolve("unusable", "unlabeled") == "pending"
    assert resolve("unusable", "bad_label") == "pending"


def test_no_review_means_the_rules_stand():
    assert resolve("gold", None) == "gold"
    assert resolve("unusable", None) == "unusable"


def test_answers_map_to_roles_and_anything_unknown_defers():
    assert parse_answer("g") == "gold"
    assert parse_answer("evidence") == "supplementary"
    assert parse_answer("R") == "unusable"
    assert parse_answer("skip") == "pending"
    assert parse_answer("what?") == "pending"


def test_the_question_shows_both_verdicts_and_the_evidence():
    case = Disagreement(
        path="/data/pima-indians-diabetes.csv",
        mechanical_role="unusable",
        mechanical_reason="columns are named 6, 148, 72 -- not a target column",
        model_role="gold",
        model_reason="headerless copy of the Pima dataset; the target is column 9",
        raw_head=["6,148,72,35,0,33.6,0.627,50,1", "1,85,66,29,0,26.6,0.351,31,0"],
    )
    text = case.question()
    assert case.needs_human is True
    assert "6,148,72,35,0,33.6,0.627,50,1" in text  # the evidence, not a summary of it
    assert "rules say: unusable" in text
    assert "model says: gold" in text
    assert "[g]old" in text and "[e]vidence" in text and "[r]eject" in text


def test_a_case_the_rules_already_reject_does_not_need_a_person():
    case = Disagreement(
        path="x.csv",
        mechanical_role="gold",
        mechanical_reason="target present",
        model_role="bad_label",
        model_reason="labels are a different ontology",
    )
    assert case.needs_human is False
    assert resolve(case.mechanical_role, case.model_role) == "supplementary"
