"""A table with categories and missing markers trains, instead of being refused.

This is the case that stopped a whole run: every clinical table carries a column
like `thal` (fixed/normal/reversible) or writes "no measurement" as `?`, and the
flow used to require features that already were numbers.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("torch")
from kosmos.ppi import TaskSpec, run_training
from kosmos.ppi.schemas import PPITrainingConfig

LEVELS = ["typical", "atypical", "non-anginal"]


def table(rows: int, *, with_label: bool, extra_level: bool = False):
    """Age and cholesterol as numbers-with-missing, chest pain as a category."""
    chest_pain = [LEVELS[i % len(LEVELS)] for i in range(rows)]
    if extra_level:
        # A level this cohort has and the gold does not: it belongs in the
        # unknown bucket rather than off the end of the matrix.
        chest_pain = [
            "silent-ischemia" if i % 17 == 0 else cp
            for i, cp in enumerate(chest_pain)
        ]
    cholesterol = [200.0 + 10.0 * i for i in range(rows)]
    if rows > 20:
        cholesterol[7] = np.nan  # written as `?` below
    frame = pd.DataFrame(
        {
            "age": 40.0 + (np.arange(rows) % 45),
            "chol": cholesterol,
            "chest_pain": chest_pain,
        }
    )
    if with_label:
        frame["target"] = [1 if cp == "typical" else 0 for cp in chest_pain]
    text = frame.astype(object)
    text["chol"] = ["?" if pd.isna(value) else value for value in frame["chol"]]
    return text


def task() -> TaskSpec:
    return TaskSpec(
        target_column="target",
        seed=5,
        description="chest pain category from clinical measurements",
    )


def test_a_table_with_categories_and_missing_markers_trains(tmp_path):
    labeled_path = tmp_path / "labeled.csv"
    table(240, with_label=True).to_csv(labeled_path, index=False)
    supplementary_path = tmp_path / "other-cohort.csv"
    table(120, with_label=False, extra_level=True).to_csv(
        supplementary_path, index=False
    )

    summary = run_training(
        labeled_path=labeled_path,
        task=task(),
        supplementary_paths=[supplementary_path],
        output_dir=tmp_path / "run",
        config=PPITrainingConfig(seed=5, max_epochs=2, patience=2, cross_fit_folds=2),
    )

    assert summary["mode"] == "ppi"
    # age, chol, three chest-pain levels and the unknown bucket.
    assert summary["task"]["n_features"] == 6
    assert summary["supplementary"]["available_rows"] == {"other-cohort": 120}
    assert "balanced_accuracy" in summary["validation_metrics"]["baseline"]
    assert "balanced_accuracy" in summary["final_test_metrics"]["ppi"]
    encoding = summary["inputs"]["encoding"]
    assert encoding["categorical"]["chest_pain"]["levels"] == [
        "atypical",
        "non-anginal",
        "typical",
    ]
    assert encoding["numeric"]["chol"]["median"] > 0


def test_a_supplementary_table_missing_a_raw_column_is_refused(tmp_path):
    labeled_path = tmp_path / "labeled.csv"
    table(60, with_label=True).to_csv(labeled_path, index=False)
    broken_path = tmp_path / "broken.csv"
    table(40, with_label=False).drop(columns=["chol"]).to_csv(
        broken_path, index=False
    )

    with pytest.raises(ValueError, match="missing"):
        run_training(
            labeled_path=labeled_path,
            task=task(),
            supplementary_paths=[broken_path],
            output_dir=tmp_path / "run-broken",
            config=PPITrainingConfig(seed=5, max_epochs=1, patience=1),
        )
