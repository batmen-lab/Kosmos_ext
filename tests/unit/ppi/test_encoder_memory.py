"""Wide tables must not be cast through text one column at a time.

The run that died with `Segmentation fault (core dumped)` was inside
`FeatureEncoder.fit` -> `_as_text` -> pandas `astype`, on a single-cell table
with ~130,000 numeric columns: `astype(str)` then `to_numeric` for every one of
them is 130,000 string allocations per row, and that is what ran the process out
of memory.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from kosmos.ppi import features as features_module
from kosmos.ppi.features import FeatureEncoder


def _frame(rows: int = 40, genes: int = 6) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    frame = pd.DataFrame(
        {f"gene{i}": rng.random(rows).astype(np.float32) for i in range(genes)}
    )
    frame["batch"] = ["s1", "s2", "s1", "s2"] * (rows // 4)
    return frame


def test_numeric_columns_are_not_cast_through_text(monkeypatch):
    """The crash was the cast itself; numeric columns must skip it."""
    calls: list[str] = []
    original = features_module._as_text

    def spy(series, _original=original):
        calls.append(str(series.name))
        return _original(series)

    monkeypatch.setattr(features_module, "_as_text", spy)
    frame = _frame()

    encoder = FeatureEncoder.fit(frame, list(frame.columns))

    assert calls == ["batch"]  # only the categorical column needed the text path
    assert set(encoder.numeric) == {f"gene{i}" for i in range(6)}
    assert list(encoder.categorical) == ["batch"]


def test_the_encoding_is_the_same_as_before_the_fast_path():
    """Same numbers: the fast path changes the route, not the result."""
    frame = _frame()
    frame["mixed"] = ["1", "2", "x", "4"] * 10  # numeric written as text

    encoder = FeatureEncoder.fit(frame, list(frame.columns))

    assert encoder.numeric["gene0"].median == pytest.approx(
        float(frame["gene0"].median())
    )
    # `mixed` is numbers with one stray value, so it is categorical -- the same
    # call the text path made before the fast path existed.
    assert set(encoder.categorical) == {"batch", "mixed"}


def test_a_table_with_130k_columns_can_be_fitted_without_strings():
    """The shape that used to crash, at a size a test can afford."""
    rng = np.random.default_rng(0)
    frame = pd.DataFrame(
        rng.random((20, 2000), dtype=np.float32),
        columns=[f"g{i}" for i in range(2000)],
    )

    encoder = FeatureEncoder.fit(frame, list(frame.columns))

    assert len(encoder.numeric) == 2000
    assert encoder.categorical == {}


def test_running_out_of_memory_says_what_to_do(monkeypatch):
    def explode(*_args, **_kwargs):
        raise MemoryError()

    monkeypatch.setattr(features_module, "_fit_columns", explode)

    with pytest.raises(ValueError) as caught:
        FeatureEncoder.fit(_frame(), list(_frame().columns))

    message = str(caught.value)
    assert "ran out of memory" in message
    assert "KOSMOS_SINGLE_CELL_MAX_CELLS" in message


def test_a_wide_frame_fits_in_bounded_time():
    """A `set(...)` inside a comprehension is a hang, not a slow step.

    `[c for c in features if c not in set(frame.columns)]` rebuilds a
    129,923-element set once per feature: 1.5 billion string hashes on a
    single-cell table, which showed up as "stuck in Executing experiments" for
    minutes. The same shape appeared in four other places.
    """
    import time

    columns = 20000
    frame = pd.DataFrame(
        np.ones((3, columns), dtype=np.float32),
        columns=[f"g{i}" for i in range(columns)],
    )

    started = time.monotonic()
    encoder = FeatureEncoder.fit(frame, list(frame.columns))
    elapsed = time.monotonic() - started

    assert encoder.width == columns
    assert elapsed < 20, f"fitting {columns:,} columns took {elapsed:.0f}s"
