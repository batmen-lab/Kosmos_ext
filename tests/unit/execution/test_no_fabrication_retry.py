"""A missing dataset is terminal, and the repair prompt says so too.

This is a correctness bug independent of dataset search, and it is the reason
`kosmos run` now refuses a question with no data rather than warning about one.

The generated experiment's data-loading preamble raises
`RuntimeError('No dataset available at data_path; refusing to fabricate
synthetic data (real data only).')` -- a refusal, correctly made. That
RuntimeError was not in the executor's non-retryable list, so the failure went
to `_repair_with_llm` three times, with a prompt whose entire instruction was
"fix the following Python code that produced an error" and which said nothing
about where data may come from. The cheapest fix available to a model for a
missing file is to write the file, at which point the experiment reports a
result computed from numbers nothing measured.

These tests pin both halves of the fix: the failure never reaches the repair
loop, and if some other phrasing ever does, the prompt forbids the fix.
"""

from __future__ import annotations

import pytest

from kosmos.execution.executor import (
    FABRICATION_REFUSAL_MARKERS,
    RetryStrategy,
    is_fabrication_refusal,
)


# The exact strings `execution/code_generator.py` writes into every generated
# experiment. Duplicated here rather than imported so that rewording the
# generator without rewording the marker list fails a test instead of silently
# reopening the repair path.
GENERATOR_REFUSALS = [
    "No dataset available at data_path; refusing to fabricate synthetic data "
    "(real data only).",
    "Failed to load required dataset from /x/data.csv: [Errno 2]. "
    "Real data only; no synthetic fallback.",
]


@pytest.mark.parametrize("message", GENERATOR_REFUSALS)
def test_the_generators_refusals_are_recognised(message):
    assert is_fabrication_refusal(message) is True


def test_ordinary_failures_are_not_mistaken_for_a_refusal():
    assert is_fabrication_refusal("KeyError: 'age'") is False
    assert is_fabrication_refusal("") is False
    assert is_fabrication_refusal(None) is False


@pytest.mark.parametrize("message", GENERATOR_REFUSALS)
def test_a_refusal_is_not_retryable_at_any_attempt(message):
    """No attempt number makes an absent dataset repairable."""
    strategy = RetryStrategy(max_retries=3)

    for attempt in (1, 2, 3):
        assert strategy.should_retry(attempt, "RuntimeError", message) is False


def test_an_unrelated_runtime_error_is_still_retryable():
    """RuntimeError in general must stay repairable.

    This is why the check is on the message and not the type: making every
    RuntimeError terminal would stop repairing the large class of errors the
    retry strategy exists for.
    """
    strategy = RetryStrategy(max_retries=3)

    assert strategy.should_retry(1, "RuntimeError", "something broke") is True


def test_existing_two_argument_calls_are_unchanged():
    """`error_message` defaults to empty, so no existing caller changes answer."""
    strategy = RetryStrategy(max_retries=3)

    assert strategy.should_retry(1, "ValueError") is True
    assert strategy.should_retry(1, "SyntaxError") is False
    assert strategy.should_retry(3, "ValueError") is False


def test_the_repair_prompt_forbids_inventing_data():
    """Asserted on the built prompt, not on a comment about it.

    The marker list is a substring match against sentences another module
    writes; this is the second line of defence for every phrasing it misses.
    """
    strategy = RetryStrategy()
    captured = {}

    class _RecordingLLM:
        def generate(self, prompt, max_tokens=None):
            captured["prompt"] = prompt
            return "```python\npass\n```"

    strategy._repair_with_llm(
        code="print(1)",
        error="FileNotFoundError: data.csv",
        traceback_str="",
        llm_client=_RecordingLLM(),
    )

    prompt = captured["prompt"]
    assert "do not invent, synthesise, simulate" in prompt
    assert "there is\nno fix -- return the code unchanged" in prompt


def test_the_marker_list_is_not_empty():
    """A guard against a refactor that empties it and quietly reopens the path."""
    assert FABRICATION_REFUSAL_MARKERS
    assert all(isinstance(m, str) and m for m in FABRICATION_REFUSAL_MARKERS)
