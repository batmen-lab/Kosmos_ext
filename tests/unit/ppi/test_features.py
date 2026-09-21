"""Encoding a real table: categorical columns, missing markers, one vocabulary."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from kosmos.ppi.features import FeatureEncoder, UnencodableColumn, max_cardinality


def gold_frame() -> pd.DataFrame:
    """A table shaped like the ones that used to be refused outright."""
    return pd.DataFrame(
        {
            "age": [50.0, 60.0, 70.0, 80.0],
            "chol": [200.0, 240.0, np.nan, 280.0],
            "sex": ["M", "F", "M", "F"],
            "thal": ["fixed", "normal", "reversible", "normal"],
        }
    )


def test_numeric_columns_are_centred_and_scaled_on_gold():
    encoder = FeatureEncoder.fit(gold_frame(), ["age", "chol"])
    matrix = encoder.transform(gold_frame())
    assert encoder.feature_names == ["age", "chol"]
    assert matrix.dtype == np.float32
    assert matrix[:, 0].mean() == pytest.approx(0.0, abs=1e-6)
    # The missing cell takes the gold median, so the column still has no NaN.
    assert np.isfinite(matrix).all()


def test_categorical_columns_become_one_hot_with_an_unknown_bucket():
    encoder = FeatureEncoder.fit(gold_frame(), ["sex", "thal"])
    assert encoder.feature_names == [
        "sex__F",
        "sex__M",
        "sex__<unknown>",
        "thal__fixed",
        "thal__normal",
        "thal__reversible",
        "thal__<unknown>",
    ]
    matrix = encoder.transform(pd.DataFrame({"sex": ["F"], "thal": ["normal"]}))
    assert matrix.tolist() == [[1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0]]


def test_a_level_the_gold_never_had_lands_in_the_unknown_bucket():
    """A second cohort is allowed to carry its own categories."""
    encoder = FeatureEncoder.fit(gold_frame(), ["thal"])
    matrix = encoder.transform(pd.DataFrame({"thal": ["something new"]}))
    assert matrix.tolist() == [[0.0, 0.0, 0.0, 1.0]]


def test_missing_markers_are_missing_not_values():
    """`?` is how the UCI cohort files write "no measurement"."""
    frame = pd.DataFrame({"trestbps": ["120", "?", "130", "?"]})
    encoder = FeatureEncoder.fit(frame, ["trestbps"])
    assert encoder.raw_features == ["trestbps"]
    assert np.isfinite(encoder.transform(frame)).all()


def test_a_column_with_too_many_levels_is_dropped_by_name():
    """A full-table level count can beat the sample the rules profiled.

    `medical_specialty` has 72 levels in 101,766 rows and looked small in a
    500-row sample. Raising here failed the whole run; the column is dropped
    instead, by name, and every other column is still used.
    """
    frame = pd.DataFrame({"note": [f"row {i}" for i in range(80)]})
    encoder = FeatureEncoder.fit(frame, ["note"], max_levels=5)
    assert encoder.dropped_wide == {"note": 80}
    assert encoder.raw_features == [] and encoder.width == 0
    # The refusal is still available for a caller that wants it.
    with pytest.raises(UnencodableColumn) as caught:
        FeatureEncoder.fit(frame, ["note"], max_levels=5, drop_wide=False)
    assert caught.value.column == "note"
    assert "PPI_MAX_CATEGORICAL_CARDINALITY" in str(caught.value)


def test_the_cap_comes_from_the_environment(monkeypatch):
    monkeypatch.setenv("PPI_MAX_CATEGORICAL_CARDINALITY", "7")
    assert max_cardinality() == 7
    monkeypatch.setenv("PPI_MAX_CATEGORICAL_CARDINALITY", "nonsense")
    assert max_cardinality() == 50


def test_transforming_a_table_for_another_task_is_refused():
    encoder = FeatureEncoder.fit(gold_frame(), ["age", "thal"])
    with pytest.raises(ValueError, match="missing"):
        encoder.transform(pd.DataFrame({"age": [1.0]}))


def test_the_encoding_is_recorded_for_provenance():
    encoder = FeatureEncoder.fit(gold_frame(), ["age", "thal"])
    record = encoder.to_dict()
    assert record["width"] == encoder.width
    assert record["categorical"]["thal"]["n_levels"] == 3
    assert record["numeric"]["age"]["median"] == pytest.approx(65.0)
