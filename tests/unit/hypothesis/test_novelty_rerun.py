"""A re-run of a question must not be blocked by its own past hypotheses.

The deadlock: the novelty check compared a fresh hypothesis against *every*
stored hypothesis in the domain, including the ones an earlier run of the same
question wrote. Every regeneration matched and was filtered, the pool stayed
empty, and the research loop regenerated forever. These tests pin the two fixes:
the same question is not compared against itself, and a filter that removes
everything still yields a hypothesis.
"""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import MagicMock

import pytest

from kosmos.models.hypothesis import Hypothesis, NoveltyReport
from kosmos.hypothesis import novelty_checker as nc


QUESTION = "Does knocking out CBL with CRISPR change the transcriptome?"
STATEMENT = "CRISPR knockout of CBL changes the transcriptome"


def _stored_row(**overrides):
    defaults = dict(
        id="old",
        research_question="some other question",
        statement=STATEMENT,
        rationale="because the module is co-regulated",
        domain="single_cell",
        created_at=datetime.now(timezone.utc),
        updated_at=datetime.now(timezone.utc),
    )
    defaults.update(overrides)
    return MagicMock(**defaults)


def _checker_with_rows(monkeypatch, rows):
    checker = nc.NoveltyChecker(similarity_threshold=0.5, use_vector_db=False)
    session = MagicMock()
    session.query.return_value.filter.return_value.all.return_value = rows
    context = MagicMock()
    context.__enter__.return_value = session
    monkeypatch.setattr(nc, "get_session", lambda: context)
    return checker


def test_a_question_is_not_compared_to_its_own_past_hypotheses(monkeypatch):
    same_question = _stored_row(id="old-1", research_question=QUESTION)
    other_question = _stored_row(id="old-2", research_question="a different question")
    checker = _checker_with_rows(monkeypatch, [same_question, other_question])

    fresh = Hypothesis(
        research_question=QUESTION, statement=STATEMENT,
        rationale="the module is co-regulated", domain="single_cell",
    )
    found = checker._check_existing_hypotheses(fresh)

    assert [row.research_question for row in found] == ["a different question"]


def test_stored_hypotheses_do_not_lower_the_novelty_score(monkeypatch):
    """Novelty is scored on literature; the system's own memory is advisory."""
    checker = nc.NoveltyChecker(similarity_threshold=0.5, use_vector_db=False)
    checker._search_similar_literature = lambda hypothesis: []          # no prior art
    checker._check_existing_hypotheses = lambda hypothesis: [            # identical stored one
        Hypothesis(research_question="other", statement=STATEMENT,
                   rationale="the module is co-regulated", domain="single_cell")
    ]
    fresh = Hypothesis(
        research_question=QUESTION, statement=STATEMENT,
        rationale="the module is co-regulated", domain="single_cell",
    )
    report = checker.check_novelty(fresh)
    assert report.novelty_score == pytest.approx(1.0)
    assert report.similar_hypotheses  # still reported, just not scored


def test_novelty_filter_that_removes_everything_still_yields_a_hypothesis(monkeypatch):
    """The safety net: never return an empty set for a produced hypothesis."""
    from kosmos.agents.hypothesis_generator import HypothesisGeneratorAgent

    agent = HypothesisGeneratorAgent.__new__(HypothesisGeneratorAgent)
    agent.num_hypotheses = 1
    agent.use_literature_context = False
    agent.literature_search = None
    agent.require_novelty_check = True
    agent.min_novelty_score = 0.5
    agent.agent_id = "test"

    produced = Hypothesis(
        research_question=QUESTION, statement=STATEMENT,
        rationale="the module is co-regulated by the same enhancer", domain="single_cell",
    )
    agent._generate_with_claude = lambda **kwargs: [produced]
    agent._validate_hypothesis = lambda hypothesis: True
    agent._store_hypothesis = lambda hypothesis: hypothesis.id

    class AlwaysUnoriginal:
        def __init__(self, **kwargs):
            pass

        def check_novelty(self, hypothesis):
            return NoveltyReport(
                hypothesis_id=hypothesis.id or "x",
                novelty_score=0.0,
                similar_papers=[],
                similar_hypotheses=[],
                max_similarity=0.95,
                prior_art_detected=True,
                is_novel=False,
                novelty_threshold_used=0.5,
                summary="duplicate",
            )

    monkeypatch.setattr(nc, "NoveltyChecker", AlwaysUnoriginal)

    response = agent.generate_hypotheses(
        research_question=QUESTION, num_hypotheses=1, domain="single_cell",
        store_in_db=False,
    )

    assert len(response.hypotheses) == 1
    assert response.hypotheses[0].novelty_score == 0.0
