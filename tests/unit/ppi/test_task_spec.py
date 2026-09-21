"""A task is a label column, features and a random split -- nothing else."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from kosmos.ppi import TaskSpec, task_spec_from_config
from kosmos.ppi.schemas import GoldDataset


def frame(rows=120, classes=3) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    data = {
        "sample_id": [f"s{i}" for i in range(rows)],
        "gene_A": rng.normal(size=rows),
        "gene_B": rng.normal(size=rows),
        "other_C": rng.normal(size=rows),
        "label": np.array([f"c{i % classes}" for i in range(rows)]),
        "junk": rng.normal(size=rows),
    }
    return pd.DataFrame(data)


def dataset(rows=120, classes=3) -> GoldDataset:
    df = frame(rows, classes)
    return GoldDataset(
        X=df[["gene_A", "gene_B"]].to_numpy(),
        y=np.asarray(df["label"], dtype=str),
        sample_ids=df["sample_id"].to_numpy(),
        role="gold_train",
        dataset_id="demo",
        feature_names=["gene_A", "gene_B"],
    )


def test_features_come_from_prefixes_exclusions_and_the_target_is_never_one():
    task = TaskSpec(
        target_column="label",
        feature_prefixes=("gene_", "other_"),
        exclude_columns=("junk",),
        sample_id_column="sample_id",
    )
    assert task.feature_names(frame().columns) == ["gene_A", "gene_B", "other_C"]
    assert task.has_target(frame().columns) is True
    assert TaskSpec(target_column="label").has_target(["label"]) is True
    assert TaskSpec(target_column="label").has_target(["other"]) is False
    assert TaskSpec().has_target(["label"]) is False


def test_explicit_feature_columns_win_and_are_validated():
    task = TaskSpec(target_column="label", feature_columns=("gene_A", "gene_B"))
    assert task.feature_names(frame().columns) == ["gene_A", "gene_B"]
    with pytest.raises(ValueError, match="does not have"):
        TaskSpec(target_column="label", feature_columns=("gene_A", "nope")).feature_names(
            frame().columns
        )


def test_selecting_no_features_is_an_error_not_an_empty_matrix():
    with pytest.raises(ValueError, match="no feature columns"):
        TaskSpec(target_column="label", feature_prefixes=("ENSG",)).feature_names(
            frame().columns
        )


def test_sample_ids_are_dataset_qualified_unless_a_column_is_named():
    df = frame()
    assert TaskSpec().sample_ids(df, "tbl")[:2].tolist() == ["tbl:0", "tbl:1"]
    assert TaskSpec(sample_id_column="sample_id").sample_ids(df, "tbl")[:2].tolist() == [
        "s0",
        "s1",
    ]
    with pytest.raises(ValueError, match="not in the table"):
        TaskSpec(sample_id_column="nope").sample_ids(df, "tbl")


def test_split_is_random_from_gold_and_carries_no_groups():
    task = TaskSpec(target_column="label", test_fraction=0.2, validation_fraction=0.2, seed=7)
    train, validation, test = task.split_labeled(dataset(rows=100))

    assert (train.role, validation.role, test.role) == (
        "gold_train",
        "gold_validation",
        "final_test",
    )
    # Proportions, rounded per class: every class gives up rows in proportion,
    # which lands a row or two away from an exact 64/16/20.
    assert (len(train.X), len(validation.X), len(test.X)) == (64, 15, 21)
    for part in (train, validation, test):
        assert part.groups is None  # leakage prevention is not modelled here
        assert part.feature_names == ["gene_A", "gene_B"]
    # Every row appears exactly once across the three parts.
    seen = set(train.sample_ids) | set(validation.sample_ids) | set(test.sample_ids)
    assert len(seen) == 100


def test_split_keeps_every_class_in_every_part_when_it_can():
    task = TaskSpec(target_column="label", seed=3)
    train, validation, test = task.split_labeled(dataset(rows=150, classes=3))
    for part in (train, validation, test):
        assert set(np.unique(part.y)) == {"c0", "c1", "c2"}


def test_a_tiny_labeled_table_is_refused_with_a_clear_message():
    with pytest.raises(ValueError, match="at least 3"):
        TaskSpec(target_column="label").split_labeled(dataset(rows=2, classes=2))


def test_config_becomes_a_task_and_nothing_becomes_none():
    """The CLI flags -> TaskSpec mapping, in one place and testable."""
    assert task_spec_from_config({}) is None
    assert task_spec_from_config({"data_path": "x.csv"}) is None

    task = task_spec_from_config(
        {
            "task_target_column": "cell_type",
            "task_feature_prefixes": ["ENSG"],
            "task_exclude_columns": ["DonorID"],
            "task_sample_id_column": "cell_id",
            "ppi_seed": 7,
            "task_test_fraction": 0.3,
            "task_validation_fraction": 0.1,
            "ppi_evaluation_metric": "macro_f1",
        },
        description="demo",
    )
    assert task is not None
    assert task.target_column == "cell_type"
    assert task.feature_prefixes == ("ENSG",)
    assert task.exclude_columns == ("DonorID",)
    assert task.sample_id_column == "cell_id"
    assert (task.seed, task.test_fraction, task.validation_fraction) == (7, 0.3, 0.1)
    assert task.evaluation_metric == "macro_f1"
    assert task.description == "demo"
    # Defaults come out of the same place, so a missing key is not a crash.
    bare = task_spec_from_config({"task_target_column": "y"})
    assert bare is not None and bare.seed == 42 and bare.feature_prefixes == ()


def test_a_classification_task_with_a_continuous_label_is_refused():
    """A plan named a gene (`WAS`) as the label; the teacher said so obscurely.

    The training side only found out inside the pseudo-label teacher, as
    `Unknown label type: continuous` -- a message that names neither the column
    nor the plan.
    """
    import numpy as np
    import pandas as pd
    import pytest

    from kosmos.ppi.flow import check_task_against_data
    from kosmos.ppi.task_spec import TaskSpec

    frame = pd.DataFrame(
        {
            "WAS": np.linspace(0.0, 5.0, 400),
            "GENE1": np.arange(400.0),
            "compound": ["drug" if index % 2 else "vehicle" for index in range(400)],
        }
    )
    task = TaskSpec(target_column="WAS", feature_columns=("GENE1",))

    with pytest.raises(ValueError) as caught:
        check_task_against_data(frame, task)

    message = str(caught.value)
    assert "WAS" in message and "continuous" in message
    assert "--task-type regression" in message


def test_a_categorical_label_is_left_alone_and_a_feature_target_is_refused():
    import numpy as np
    import pandas as pd

    from kosmos.ppi.flow import check_task_against_data
    from kosmos.ppi.task_spec import TaskSpec

    frame = pd.DataFrame(
        {
            "compound": ["drug", "vehicle", "drug", "vehicle"] * 50,
            "GENE1": np.arange(200.0),
            "GENE2": np.arange(200.0) * 2,
            "GENE3": ["a", "b"] * 100,
        }
    )
    task = TaskSpec(target_column="compound", feature_columns=("GENE1", "GENE2"))
    assert check_task_against_data(frame, task) == ""

    # A column used as both the target and a feature is the model reading the
    # answer off its own input. The number it produces is meaningless, so the
    # run stops here with a message that names the column and the way out.
    odd = TaskSpec(target_column="GENE3", feature_columns=("GENE1", "GENE3"))
    with pytest.raises(ValueError) as caught:
        check_task_against_data(frame, odd)

    message = str(caught.value)
    assert "GENE3" in message and "also one of the" in message
