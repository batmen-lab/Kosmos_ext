"""Novelty checking may cost a moment, never the run.

The stack that ended in a stuck run went: `generate_hypotheses` ->
`check_novelty` -> `embed_query` -> BERT forward, on the research loop's own
thread. One `encode()` per stored hypothesis, on CPU, blocks everything and
looks like a hang.
"""

from __future__ import annotations

import time

import numpy as np
import pytest

from kosmos.hypothesis.novelty_checker import (
    NOVELTY_BUDGET_ENV,
    NoveltyChecker,
    _comparison_budget_seconds,
)
from kosmos.models.hypothesis import Hypothesis, HypothesisStatus


class FakeEmbedder:
    """An embedder that records the calls it was given."""

    def __init__(self, dim: int = 4, delay: float = 0.0):
        self.dim = dim
        self.delay = delay
        self.calls: list[list[str]] = []

    def embed_texts(self, texts, batch_size: int = 32):
        self.calls.append(list(texts))
        if self.delay:
            time.sleep(self.delay)
        return np.ones((len(texts), self.dim), dtype=np.float32)

    def embed_query(self, text: str):
        self.calls.append([text])
        return np.ones(self.dim, dtype=np.float32)


def _hypothesis(statement: str = "Unlabeled cells from another assay improve accuracy."):
    return Hypothesis(
        research_question="does it help?",
        statement=statement,
        rationale="because the boundary is better constrained",
        domain="biology",
        status=HypothesisStatus.GENERATED,
    )


def _checker(embedder):
    checker = NoveltyChecker.__new__(NoveltyChecker)  # no API clients, no vector db
    checker.embedder = embedder
    return checker


def test_all_comparisons_go_through_the_model_in_one_call():
    embedder = FakeEmbedder()
    checker = _checker(embedder)
    existing = [_hypothesis(f"hypothesis {i}") for i in range(12)]

    score = checker._max_hypothesis_similarity(
        _hypothesis(), existing, deadline=time.monotonic() + 30
    )

    assert score == pytest.approx(1.0, abs=1e-6)  # identical fake vectors
    assert len(embedder.calls) == 1  # one batch, not 2 x 12 forwards
    assert len(embedder.calls[0]) == 13  # the query plus every comparison


def test_an_expired_deadline_skips_the_comparison_instead_of_running_it():
    embedder = FakeEmbedder()
    checker = _checker(embedder)

    score = checker._max_hypothesis_similarity(
        _hypothesis(),
        [_hypothesis(f"hypothesis {i}") for i in range(5)],
        deadline=time.monotonic() - 1,  # already over budget
    )

    assert score == 0.0
    assert embedder.calls == []  # the model was never asked


def test_the_budget_comes_from_the_environment(monkeypatch):
    monkeypatch.delenv(NOVELTY_BUDGET_ENV, raising=False)
    assert _comparison_budget_seconds() == 20.0
    monkeypatch.setenv(NOVELTY_BUDGET_ENV, "5")
    assert _comparison_budget_seconds() == 5.0
    monkeypatch.setenv(NOVELTY_BUDGET_ENV, "nonsense")
    assert _comparison_budget_seconds() == 20.0
