"""The flow trains any task: labeled table in, optional unlabeled table beside it."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("torch")

from kosmos.ppi import TaskSpec, run_training
from kosmos.ppi.schemas import PPITrainingConfig


def write_tables(
    tmp_path, rows=300, classes=3, seed=0, with_label=True, supplementary=0, prefix="L"
):
    tmp_path.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    labeled = pd.DataFrame(
        {
            "sample_id": [f"{prefix}{i}" for i in range(rows)],
            "feat_a": rng.normal(size=rows),
            "feat_b": rng.normal(size=rows),
            "feat_c": rng.normal(size=rows),
            "condition": [f"g{i % 2}" for i in range(rows)],
            "label": [f"c{i % classes}" for i in range(rows)],
        }
    )
    labeled_path = tmp_path / "labeled.csv"
    labeled.to_csv(labeled_path, index=False)
    paths = []
    for index in range(supplementary):
        fresh = pd.DataFrame(
            {
                "sample_id": [f"U{index}-{i}" for i in range(rows)],
                "feat_a": rng.normal(size=rows) + 0.3,
                "feat_b": rng.normal(size=rows),
                "feat_c": rng.normal(size=rows),
                "condition": [f"g{i % 2}" for i in range(rows)],
            }
        )
        if with_label:
            fresh["label"] = [f"c{i % classes}" for i in range(rows)]
        path = tmp_path / f"supplementary-{index}.csv"
        fresh.to_csv(path, index=False)
        paths.append(path)
    return labeled_path, paths


def task(**overrides) -> TaskSpec:
    base = {
        "target_column": "label",
        "feature_prefixes": ("feat_",),
        "sample_id_column": "sample_id",
        "seed": 5,
        "description": "demo task",
    }
    base.update(overrides)
    return TaskSpec(**base)


def fast_config() -> PPITrainingConfig:
    return PPITrainingConfig(seed=5, max_epochs=2, patience=2, cross_fit_folds=3)


def test_labeled_data_alone_runs_supervised_training(tmp_path):
    labeled, _ = write_tables(tmp_path)
    out = tmp_path / "run-supervised"
    summary = run_training(
        labeled_path=labeled, task=task(), output_dir=out, config=fast_config()
    )

    assert summary["mode"] == "supervised"
    assert summary["task"]["target_column"] == "label"
    assert summary["task"]["n_features"] == 3
    assert summary["task"]["classes"] == ["c0", "c1", "c2"]
    assert summary["n_labeled"]["total"] == 300
    # test and validation both come out of the labeled table
    assert summary["n_labeled"]["test"] == 60
    assert summary["n_labeled"]["validation"] == 48
    assert summary["n_labeled"]["train"] == 192
    assert summary["external_used"] == 0
    assert "accuracy" in summary["final_test_metrics"]["baseline"]
    # The comparison is reported either way, with a zero delta when nothing was added.
    assert summary["final_test_metrics"]["ppi"] == summary["final_test_metrics"]["baseline"]
    assert json.loads((out / "ppi_summary.json").read_text())["mode"] == "supervised"


def test_unlabeled_supplementary_data_switches_on_the_ppi_correction(tmp_path):
    labeled, supplementary = write_tables(tmp_path, supplementary=2, with_label=False)
    summary = run_training(
        labeled_path=labeled,
        task=task(),
        supplementary_paths=supplementary,
        output_dir=tmp_path / "run-ppi",
        config=fast_config(),
    )

    assert summary["mode"] == "ppi"
    assert summary["external_used"] > 0
    assert set(summary["external_sources"]) == {"supplementary-0", "supplementary-1"}
    # Both arms are scored on the same held-out labels.
    assert set(summary["final_test_metrics"]) == {"baseline", "ppi"}
    assert summary["validation_metrics"]["delta"]["balanced_accuracy"] is not None


def test_a_table_with_labels_we_do_not_want_is_still_usable_as_evidence(tmp_path):
    """Some other label column present is not a reason to refuse the table."""
    labeled, supplementary = write_tables(tmp_path, supplementary=1, with_label=True)
    summary = run_training(
        labeled_path=labeled,
        task=task(),
        supplementary_paths=supplementary,
        output_dir=tmp_path / "run-ppi-labeled",
        config=fast_config(),
    )
    assert summary["mode"] == "ppi"
    assert summary["external_used"] > 0


def test_unlabeled_data_passed_as_gold_is_refused_with_a_pointer(tmp_path):
    _, supplementary = write_tables(tmp_path, supplementary=1, with_label=False)
    with pytest.raises(ValueError, match="supplementary"):
        run_training(
            labeled_path=supplementary[0],
            task=task(),
            output_dir=tmp_path / "run-bad",
            config=fast_config(),
        )


def test_supplementary_data_must_share_the_gold_feature_schema(tmp_path):
    labeled, supplementary = write_tables(tmp_path, supplementary=1, with_label=False)
    frame = pd.read_csv(supplementary[0]).drop(columns=["feat_c"])
    frame.to_csv(supplementary[0], index=False)
    with pytest.raises(ValueError, match="missing"):
        run_training(
            labeled_path=labeled,
            task=task(),
            supplementary_paths=supplementary,
            output_dir=tmp_path / "run-schema",
            config=fast_config(),
        )


def test_an_explicit_test_table_is_evaluated_instead_of_a_random_slice(tmp_path):
    labeled, _ = write_tables(tmp_path, rows=300)
    holdout_dir = tmp_path / "holdout"
    holdout_dir.mkdir()
    # Distinct identifiers: the engine refuses a final test that overlaps the
    # development observations, and a reused id column would trip that check.
    held_out, _ = write_tables(holdout_dir, rows=80, seed=99, prefix="T")
    summary = run_training(
        labeled_path=labeled,
        task=task(),
        test_path=held_out,
        output_dir=tmp_path / "run-explicit-test",
        config=fast_config(),
    )
    assert summary["n_labeled"]["test"] == 80
    assert summary["n_labeled"]["train"] + summary["n_labeled"]["validation"] == 300


def test_several_labeled_tables_are_pooled_when_they_share_features(tmp_path):
    """More data is the point: same features means concatenate, not choose."""
    first, _ = write_tables(tmp_path / "a", rows=120, prefix="A")
    second, _ = write_tables(tmp_path / "b", rows=80, prefix="B", seed=11)
    summary = run_training(
        labeled_paths=[first, second],
        task=task(),
        output_dir=tmp_path / "run-pooled",
        config=fast_config(),
    )
    assert summary["n_labeled"]["total"] == 200
    # 20% of the pooled 200, rounded per class rather than exactly: the split
    # keeps at least one row of every class in training.
    assert summary["n_labeled"]["test"] == pytest.approx(40, abs=2)


def test_pooling_refuses_a_table_with_a_different_feature_schema(tmp_path):
    first, _ = write_tables(tmp_path / "a", rows=60)
    second_dir = tmp_path / "b"
    second_dir.mkdir()
    second, _ = write_tables(second_dir, rows=60, seed=3)
    frame = pd.read_csv(second).rename(columns={"feat_c": "something_else"})
    frame.to_csv(second, index=False)
    with pytest.raises(ValueError, match="does not share the pooled feature schema"):
        run_training(
            labeled_paths=[first, second],
            task=task(),
            output_dir=tmp_path / "run-bad-pool",
            config=fast_config(),
        )


def test_pooling_refuses_colliding_row_identifiers(tmp_path):
    first, _ = write_tables(tmp_path / "a", rows=60, prefix="X")
    second_dir = tmp_path / "b"
    second_dir.mkdir()
    second, _ = write_tables(second_dir, rows=60, prefix="X")  # same ids
    with pytest.raises(ValueError, match="share sample identifiers"):
        run_training(
            labeled_paths=[first, second],
            task=task(),
            output_dir=tmp_path / "run-dup-ids",
            config=fast_config(),
        )


def test_a_supplementary_table_with_a_different_spelling_is_aligned(tmp_path):
    """The plan matched `concave points_mean` to `concave_points_mean`; the run
    has to apply that before it loads the table."""
    labeled, _ = write_tables(tmp_path, rows=200)
    mirror = tmp_path / "mirror.csv"
    # Other rows, not a copy: a table of the same rows is a mirror, and a mirror
    # is not evidence (the correction would cancel against itself).
    other_dir = tmp_path / "supp"
    other_dir.mkdir()
    other, _ = write_tables(other_dir, rows=200, seed=11)
    frame = pd.read_csv(other).drop(columns=["label"])
    frame = frame.rename(
        columns={"feat_a": "feat a", "condition": "Condition"}  # spelling only
    )
    frame["sample_id"] = [f"M{i}" for i in range(len(frame))]
    frame.to_csv(mirror, index=False)

    summary = run_training(
        labeled_path=labeled,
        task=task(),
        supplementary_paths=[mirror],
        supplementary_renames={str(mirror): {"feat a": "feat_a", "Condition": "condition"}},
        output_dir=tmp_path / "run-aligned",
        config=fast_config(),
    )

    assert summary["mode"] == "ppi"
    assert summary["supplementary"]["available_rows"] == {"mirror": 200}


def test_a_table_holding_the_gold_rows_is_refused_as_evidence(tmp_path):
    """Evidence has to be *other* rows, however the file was written.

    A republished copy of the labeled set would enter the correction and cancel
    against itself, so the run would report a semi-supervised result that did
    nothing. The check is on the encoded rows, so a different spelling or a
    different format does not get around it.
    """
    labeled, _ = write_tables(tmp_path, rows=200)
    copy = tmp_path / "republished.csv"
    frame = pd.read_csv(labeled).drop(columns=["label"])
    frame["sample_id"] = [f"R{i}" for i in range(len(frame))]  # new ids, same rows
    frame.to_csv(copy, index=False)

    summary = run_training(
        labeled_path=labeled,
        task=task(),
        supplementary_paths=[copy],
        output_dir=tmp_path / "run-mirror",
        config=fast_config(),
    )

    assert summary["mode"] == "supervised"
    assert summary["supplementary"]["available_rows"] == {}
    assert summary["supplementary"]["mirrors_dropped"] == ["republished"]
