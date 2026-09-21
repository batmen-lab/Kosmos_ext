"""Reading a fetcher plan: the one JSON contract Kosmos knows about."""

from __future__ import annotations

import json

import pytest

from kosmos.cli.commands.run import resolve_data_plan


def write_plan(tmp_path, gold=("a.csv",), supplementary=("s.csv",), unusable=()):
    plan = {
        "version": 1,
        "task": {
            "objective": "predict cell type",
            "target_column": "cell_type",
            "target_source": "cli",
            "target_confidence": "declared",
            "feature_prefixes": ["ENSG"],
            "exclude_columns": ["DonorID"],
            "sample_id_column": "cell_id",
        },
        "gold": [{"path": p, "role": "gold", "reason": "target present"} for p in gold],
        "supplementary": [
            {"path": p, "role": "supplementary", "reason": "no target"}
            for p in supplementary
        ],
        "unusable": [{"path": p, "role": "unusable", "reason": "no features"} for p in unusable],
    }
    path = tmp_path / "plan.json"
    path.write_text(json.dumps(plan), encoding="utf-8")
    return path


def test_plan_gives_the_task_the_tables_and_the_reasons(tmp_path):
    resolved = resolve_data_plan(write_plan(tmp_path, supplementary=("s1.csv", "s2.csv")))
    assert resolved.target_column == "cell_type"
    assert resolved.feature_prefixes == ["ENSG"]
    assert resolved.exclude_columns == ["DonorID"]
    assert resolved.sample_id_column == "cell_id"
    assert resolved.labeled_paths == ["a.csv"]
    assert resolved.supplementary_paths == ["s1.csv", "s2.csv"]
    assert any("target column" in line for line in resolved.summary_lines)
    assert any("supplementary : 2 table(s)" in line for line in resolved.summary_lines)


def test_explicit_arguments_beat_the_file(tmp_path):
    resolved = resolve_data_plan(
        write_plan(tmp_path),
        target_column="other_label",
        feature_prefixes=["gene_"],
        sample_id_column="row_id",
        extra_supplementary=["extra.csv"],
    )
    assert resolved.target_column == "other_label"
    assert resolved.feature_prefixes == ["gene_"]
    assert resolved.sample_id_column == "row_id"
    # An extra table given on the command line is added, not replacing the plan's.
    assert resolved.supplementary_paths == ["extra.csv", "s.csv"]


def test_the_plans_own_feature_list_reaches_the_run(tmp_path):
    """The plan decides which columns are features; the run must use them.

    Without this the trainer takes every column that is not the target, which
    puts back the free-text columns the plan left out -- and is what would have
    refused the perovskite table outright.
    """
    path = write_plan(tmp_path)
    plan = json.loads(path.read_text())
    plan["task"].update(
        {"task_type": "regression", "feature_columns": ["ff", "jsc", "voc", "bandgap"]}
    )
    path.write_text(json.dumps(plan), encoding="utf-8")

    resolved = resolve_data_plan(path)

    assert resolved.task_type == "regression"
    assert resolved.feature_columns == ["ff", "jsc", "voc", "bandgap"]
    assert any("4 column(s)" in line for line in resolved.summary_lines)


def test_a_plan_without_gold_is_refused_with_a_pointer(tmp_path):
    with pytest.raises(ValueError, match="no gold tables"):
        resolve_data_plan(write_plan(tmp_path, gold=(), supplementary=("s.csv",)))


def test_extra_supplementary_is_deduplicated(tmp_path):
    resolved = resolve_data_plan(write_plan(tmp_path), extra_supplementary=["s.csv"])
    assert resolved.supplementary_paths == ["s.csv"]


def test_unusable_tables_are_counted_in_the_summary(tmp_path):
    resolved = resolve_data_plan(write_plan(tmp_path, unusable=("broken.csv",)))
    assert any("unusable      : 1 table(s) excluded" in line for line in resolved.summary_lines)
