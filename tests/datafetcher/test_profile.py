"""Profiling a table, and deciding which role it can play for a task."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd
import pytest

from datafetcher import TaskShape, classify_role, plan_roles, profile_table
from datafetcher.cli import main


def write_labeled(path, rows=60):
    frame = pd.DataFrame(
        {
            "sample_id": [f"s{i}" for i in range(rows)],
            "ENSG00000000001": [float(i) for i in range(rows)],
            "ENSG00000000002": [float(i % 7) for i in range(rows)],
            "cell_type": ["B", "T", "NK"] * (rows // 3),
            "batch_note": ["x"] * rows,
        }
    )
    frame.to_csv(path, index=False)
    return path


def write_unlabeled(path, rows=40):
    frame = pd.DataFrame(
        {
            "sample_id": [f"u{i}" for i in range(rows)],
            "ENSG00000000001": [float(i) + 0.5 for i in range(rows)],
            "ENSG00000000002": [float(i % 5) for i in range(rows)],
        }
    )
    frame.to_csv(path, index=False)
    return path


def task(**overrides) -> TaskShape:
    base = {
        "target_column": "cell_type",
        "feature_prefixes": ("ENSG",),
        "sample_id_column": "sample_id",
    }
    base.update(overrides)
    return TaskShape(**base)


def test_profile_reads_kinds_and_counts_rows(tmp_path):
    path = write_labeled(tmp_path / "t.csv")
    profile = profile_table(path, count_rows=True)
    assert profile.delimiter == ","
    assert profile.n_rows == 60
    assert profile.kind_of("ENSG00000000001") == "numeric"
    assert profile.kind_of("cell_type") == "text"
    assert profile.kind_of("sample_id") == "text"
    assert profile.numeric_columns() == ["ENSG00000000001", "ENSG00000000002"]


def test_profile_survives_a_file_that_is_not_a_plain_table(tmp_path):
    """A GEO series matrix parses as one column; say so, do not pretend."""
    path = tmp_path / "matrix.txt"
    path.write_text('!Series_title\t"x"\n!Sample_geo_accession\t"GSM1"\n')
    profile = profile_table(path)
    assert any("GEO's metadata preamble" in note for note in profile.notes)


def test_profile_flags_a_single_column_parse(tmp_path):
    path = tmp_path / "blob.bin"
    path.write_text("just some text\nand more\n")
    profile = profile_table(path)
    assert any("single column" in note for note in profile.notes)


def test_a_parquet_file_is_a_table_not_an_unreadable_blob(tmp_path):
    """HuggingFace ships parquet; reading it as text made every such fetch unusable."""
    pytest.importorskip("pyarrow")
    path = tmp_path / "train-00000-of-00001.parquet"
    pd.DataFrame(
        {
            "age": [50, 60, 70],
            "sex": [1, 0, 1],
            "num": [0, 1, 0],
        }
    ).to_parquet(path)
    profile = profile_table(path, count_rows=True)
    assert profile.n_rows == 3
    assert [c.name for c in profile.columns] == ["age", "sex", "num"]
    assert profile.kind_of("age") == "numeric"
    assert any("read as parquet" in note for note in profile.notes)


def test_an_arrow_file_is_read_too(tmp_path):
    pa = pytest.importorskip("pyarrow")
    path = tmp_path / "dataset.arrow"
    table = pa.Table.from_pandas(pd.DataFrame({"x": [1, 2], "y": ["a", "b"]}))
    with pa.OSFile(str(path), "wb") as sink:
        with pa.ipc.new_file(sink, table.schema) as writer:
            writer.write_table(table)
    profile = profile_table(path)
    assert [c.name for c in profile.columns] == ["x", "y"]
    assert any("read as arrow" in note for note in profile.notes)


def test_a_table_with_the_target_is_gold(tmp_path):
    path = write_labeled(tmp_path / "t.csv")
    decision = classify_role(profile_table(path), task())
    assert decision.role == "gold"
    assert decision.has_target is True
    assert decision.features == ["ENSG00000000001", "ENSG00000000002"]
    assert "can train" in decision.reason


def test_a_table_without_the_target_is_supplementary(tmp_path):
    path = write_unlabeled(tmp_path / "u.csv")
    decision = classify_role(profile_table(path), task())
    assert decision.role == "supplementary"
    assert decision.has_target is False
    assert "correction term" in decision.reason


def test_a_handful_of_text_values_is_a_categorical_feature(tmp_path):
    """`thal: fixed/normal/reversible` is a column, not a reason to refuse."""
    path = write_labeled(tmp_path / "t.csv")
    decision = classify_role(profile_table(path), task(feature_prefixes=()))
    assert decision.role == "gold"
    assert decision.categorical_features == ["batch_note"]
    assert "1 categorical" in decision.reason


def test_a_distinct_value_per_row_is_dropped_not_trained_on(tmp_path):
    """One free-text column is left out; the table keeps the columns that work."""
    frame = pd.read_csv(write_labeled(tmp_path / "t.csv"))
    frame["note"] = [f"free text row {i}" for i in range(len(frame))]
    path = tmp_path / "text.csv"
    frame.to_csv(path, index=False)
    decision = classify_role(profile_table(path), task(feature_prefixes=()))
    assert decision.role == "gold"
    assert "note" not in decision.features
    assert decision.dropped_features == ["note"]
    assert decision.high_cardinality_features == ["note"]
    assert "not features" in decision.reason


def test_a_table_of_nothing_but_free_text_is_unusable(tmp_path):
    frame = pd.read_csv(write_labeled(tmp_path / "t.csv"))
    frame["note"] = [f"free text row {i}" for i in range(len(frame))]
    path = tmp_path / "text-only.csv"
    frame[["note", "cell_type"]].to_csv(path, index=False)
    decision = classify_role(profile_table(path), task(feature_prefixes=()))
    assert decision.role == "unusable"
    assert "no encodable feature column remains" in decision.reason


def test_the_level_cap_follows_the_trainers_variable(tmp_path, monkeypatch):
    """The fetcher and the encoder state one rule; the variable keeps them equal."""
    monkeypatch.setenv("PPI_MAX_CATEGORICAL_CARDINALITY", "3")
    frame = pd.read_csv(write_labeled(tmp_path / "t.csv"))
    frame["severity"] = ["mild", "moderate", "severe", "critical"] * (len(frame) // 4)
    path = tmp_path / "capped.csv"
    frame.to_csv(path, index=False)
    decision = classify_role(profile_table(path), task(feature_prefixes=()))
    assert decision.role == "gold"
    assert decision.high_cardinality_features == ["severity"]
    assert "severity" not in decision.features


def test_a_row_index_is_not_a_feature(tmp_path):
    """HuggingFace parquet exports carry `__index_level_0__`; it is a row number."""
    frame = pd.read_csv(write_labeled(tmp_path / "t.csv"))
    frame["__index_level_0__"] = range(len(frame))
    path = tmp_path / "indexed.csv"
    frame.to_csv(path, index=False)
    decision = classify_role(profile_table(path), task(feature_prefixes=()))
    assert decision.role == "gold"
    assert "__index_level_0__" in decision.dropped_features
    assert "__index_level_0__" not in decision.features


def test_a_single_column_file_is_not_a_table(tmp_path):
    """A README is one long column; it used to be called supplementary evidence."""
    path = tmp_path / "README.md"
    path.write_text("---\ntitle: dataset\n---\n")
    decision = classify_role(profile_table(path), task())
    assert decision.role == "unusable"
    assert "not a table" in decision.reason


def test_a_parquet_review_packet_shows_rows_not_bytes(tmp_path):
    """The model reviewed parquet files blind: `raw_head` was binary or empty."""
    pytest.importorskip("pyarrow")
    from datafetcher.sample import compact, sample_packet

    path = tmp_path / "train-00000-of-00001.parquet"
    pd.DataFrame({"pce": [14.9, 15.9], "stack": ["ITO | NiO | MAPbI3", "FTO | TiO2"]}).to_parquet(
        path
    )
    packet = sample_packet(path, rows=2)
    assert packet["raw_head_kind"] == "table"
    assert packet["raw_head"][0] == "pce | stack"
    assert "15.9" in packet["raw_head"][2]
    rendered = compact(packet)
    assert "read from the file's own table format" in rendered
    assert "MAPbI3" in rendered


def test_a_wide_table_shows_the_model_both_ends(tmp_path):
    """A single-cell table has a column per gene and the label at the end.

    Rendering the first 60 column names and truncating each row at 300
    characters hid `cell_type`, and the review reported "no cell-type label
    column" for a table that has one -- which took the table out of the run.
    """
    from datafetcher.sample import compact, sample_packet

    genes = [f"GENE{i}" for i in range(200)]
    path = tmp_path / "cells.csv"
    pd.DataFrame(
        [
            [float(i)] * len(genes) + ["CD14+ Mono", "donor-1"]
            for i in range(5)
        ],
        columns=[*genes, "cell_type", "batch"],
    ).to_csv(path, index=False)

    rendered = compact(sample_packet(path, rows=2), max_columns=40)

    assert "cell_type" in rendered
    assert "batch" in rendered
    assert "more" in rendered  # the middle is elided, not the tail


def test_the_raw_head_of_a_wide_table_is_not_empty(tmp_path):
    """A single-cell row is longer than the whole byte budget.

    The budget was applied to the line as a whole, so the first line did not fit
    and the packet carried no rows at all -- for exactly the tables whose
    meaning sits in the columns at the far end.
    """
    from datafetcher.sample import sample_packet

    genes = [f"GENE{i}" for i in range(200)]
    path = tmp_path / "cells.csv"
    pd.DataFrame(
        [[float(i)] * len(genes) + ["CD14+ Mono"] for i in range(3)],
        columns=[*genes, "cell_type"],
    ).to_csv(path, index=False)

    packet = sample_packet(path, rows=3)

    assert packet["raw_head"]
    assert "cell_type" in packet["raw_head"][0]
    assert "more]" in packet["raw_head"][0]  # the middle is elided, not the tail


def test_missing_named_features_are_reported(tmp_path):
    path = write_labeled(tmp_path / "t.csv")
    decision = classify_role(
        profile_table(path),
        task(feature_columns=("ENSG00000000001", "ENSG99999999999")),
    )
    assert decision.role == "unusable"
    assert decision.missing_features == ["ENSG99999999999"]


def test_no_feature_columns_matching_is_unusable(tmp_path):
    path = write_labeled(tmp_path / "t.csv")
    decision = classify_role(profile_table(path), task(feature_prefixes=("nope_",)))
    assert decision.role == "unusable"
    assert "no feature columns match" in decision.reason


def test_plan_roles_groups_tables_and_keeps_reasons(tmp_path):
    labeled = write_labeled(tmp_path / "labeled.csv")
    unlabeled = write_unlabeled(tmp_path / "unlabeled.csv")
    broken = tmp_path / "broken.csv"
    broken.write_text("a,b\n1,2\n")
    plan = plan_roles([labeled, unlabeled, broken], task())
    assert [e["path"] for e in plan["gold"]] == [str(labeled)]
    assert [e["path"] for e in plan["supplementary"]] == [str(unlabeled)]
    assert [e["path"] for e in plan["unusable"]] == [str(broken)]


def test_cli_profile_and_roles(tmp_path, capsys):
    labeled = write_labeled(tmp_path / "labeled.csv")
    unlabeled = write_unlabeled(tmp_path / "unlabeled.csv")

    assert main(["profile", str(labeled), "--json"]) == 0
    profile = json.loads(capsys.readouterr().out)
    assert profile["n_rows"] == 60
    assert any(c["name"] == "cell_type" for c in profile["columns"])

    args = [
        "roles",
        str(labeled),
        str(unlabeled),
        "--target-column",
        "cell_type",
        "--feature-prefix",
        "ENSG",
        "--sample-id-column",
        "sample_id",
        "--json",
    ]
    assert main(args) == 0
    plan = json.loads(capsys.readouterr().out)["plan"]
    assert len(plan["gold"]) == 1 and len(plan["supplementary"]) == 1


def test_the_board_still_does_not_import_kosmos():
    """A static check, because the two packages must stay independent.

    `sys.modules` is not usable here: the repository's own test conftest imports
    kosmos for the rest of the suite. What matters is that no file in this board
    *imports* it -- mentioning it in prose is fine, and this board mentions it a
    lot while explaining what it deliberately does not do.
    """
    import datafetcher

    root = Path(datafetcher.__file__).parent
    pattern = re.compile(r"^\s*(?:import|from)\s+kosmos\b", re.MULTILINE)
    offenders = [
        str(file)
        for file in root.rglob("*.py")
        if pattern.search(file.read_text(encoding="utf-8"))
    ]
    assert offenders == []


def test_a_very_wide_table_is_profiled_from_both_ends(tmp_path):
    """One column per gene: 130,000 of them take ~110s to characterise, per call.

    The names are what decide features and the label, so they stay complete; the
    kinds are computed for the first and last columns, which is where a
    single-cell table keeps its identifiers and its label.
    """
    import numpy as np

    from datafetcher.profile import WIDE_TABLE_COLUMNS, profile_table

    genes = [f"GENE{i}" for i in range(WIDE_TABLE_COLUMNS + 200)]
    rows = {
        name: np.zeros(5)
        for name in genes
    }
    path = tmp_path / "cells.csv"
    frame = pd.DataFrame(rows)
    frame["batch"] = ["s1"] * 5
    frame["cell_type"] = ["B", "T", "NK", "B", "T"]
    frame.to_csv(path, index=False)

    profile = profile_table(path, sample_rows=500)

    assert len(profile.names) == len(genes) + 2  # every name is kept
    assert profile.n_columns == len(genes) + 2
    assert len(profile.columns) < profile.n_columns  # kinds are a bounded slice
    # The label and the id column sit at the end, and are still characterised.
    assert profile.kind_of("cell_type") == "text"
    assert profile.kind_of("batch") == "text"
    assert any("kinds computed for the first and last" in note for note in profile.notes)
