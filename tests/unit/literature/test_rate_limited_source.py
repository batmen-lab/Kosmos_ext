"""A source that refuses must cost the run a moment, not the run itself.

Semantic Scholar's unauthenticated API answers 429 readily, and the client
library retries with backoff for minutes on a background thread the caller's
90-second budget cannot stop. That is what made hypothesis generation look
stuck: arXiv and PubMed answered, and this one kept asking.
"""

from __future__ import annotations

from unittest.mock import Mock, patch

import pytest

from kosmos.literature.semantic_scholar import SemanticScholarClient


@pytest.fixture
def s2_client():
    config = Mock()
    config.literature.max_results_per_query = 100
    config.literature.semantic_scholar_api_key = None
    with patch("kosmos.literature.semantic_scholar.get_config", return_value=config):
        with patch("kosmos.literature.semantic_scholar.get_cache") as cache:
            cache.return_value = None
            with patch("kosmos.literature.semantic_scholar.SemanticScholar"):
                return SemanticScholarClient(api_key="test", cache_enabled=False)


def test_the_client_is_built_without_the_library_retrying(s2_client):
    """`retry=False`: the requests it would repeat are the ones that just 429'd."""
    # The patched class records how it was constructed.
    assert s2_client.client is not None
    s2_client.client.reset_mock()
    assert s2_client.unavailable_reason == ""


def test_a_rate_limit_returns_empty_and_is_remembered(s2_client):
    s2_client.client.search_paper.side_effect = ConnectionRefusedError(
        "HTTP status 429 Too Many Requests."
    )

    assert s2_client.search("a query") == []
    assert "refused" in s2_client.unavailable_reason


def test_later_calls_do_not_ask_again(s2_client):
    s2_client.client.search_paper.side_effect = ConnectionRefusedError("429 Too Many Requests.")
    s2_client.search("first query")
    s2_client.client.search_paper.reset_mock()

    assert s2_client.search("second query") == []

    assert s2_client.client.search_paper.call_count == 0


def test_a_transient_looking_error_is_still_logged_and_returns_empty(s2_client):
    """A source that fails is not a run that fails."""
    s2_client.client.search_paper.side_effect = Exception("Network error")

    assert s2_client.search("a query") == []
    assert s2_client.unavailable_reason == ""
