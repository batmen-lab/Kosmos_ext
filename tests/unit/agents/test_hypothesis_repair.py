"""A hypothesis the model wrote too long is used, not dropped.

The live run logged `Error parsing hypothesis 2: ... String should have at most
500 characters` and then converged with "no testable hypotheses" -- a parsing
failure that reads like the model having nothing to say.
"""

from __future__ import annotations

import pytest

from kosmos.agents.hypothesis_generator import _build_hypothesis, _shorten
from kosmos.models.hypothesis import Hypothesis, HypothesisStatus


def _build(**overrides):
    payload = {
        "statement": "Adding unlabeled cells from another assay improves cell-type accuracy.",
        "rationale": "The extra cells constrain the decision boundary.",
        "confidence_score": 0.6,
        "testability_score": 0.8,
    }
    payload.update(overrides)
    return _build_hypothesis(
        payload,
        hypothesis_cls=Hypothesis,
        status=HypothesisStatus.GENERATED,
        research_question="does it help?",
        domain="biology",
        exp_types=[],
        context_papers=[],
        generated_by="test",
    )


def test_a_short_statement_is_kept_verbatim():
    hypothesis = _build()
    assert hypothesis.statement.startswith("Adding unlabeled cells")


def test_a_long_statement_is_shortened_to_the_limit_and_kept():
    long_statement = " ".join(["cell type"] * 200)  # ~1,800 characters
    assert len(long_statement) > 500

    hypothesis = _build(statement=long_statement)

    assert len(hypothesis.statement) <= 500
    assert hypothesis.statement.endswith("…")
    # It is still a hypothesis, not a rejection.
    assert hypothesis.rationale


def test_shorten_cuts_at_a_word_and_marks_it():
    # Never longer than the limit, and never in the middle of a word.
    assert _shorten("alpha beta gamma", 10) == "alpha…"
    assert _shorten("alpha beta gamma", 15) == "alpha beta…"
    assert _shorten("short", 10) == "short"


def test_a_payload_that_is_broken_another_way_still_raises():
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        _build(statement="tiny")  # below the model's 10-character minimum
