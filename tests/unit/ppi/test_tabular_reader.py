"""Reading a fetched table fast enough that it is not mistaken for a hang.

pandas' C parser on a 1 GB, 129,923-column single-cell table took over five
minutes; pyarrow reads the 12,062 columns the encoder needs in under two
seconds. These tests pin the behaviours the pandas path had.
"""

from __future__ import annotations

import gzip

import pandas as pd
import pytest

from kosmos.ppi.tabular import detect_separator, read_table


def test_a_comma_table_reads_the_same(tmp_path):
    path = tmp_path / "t.csv"
    pd.DataFrame({"a": [1, 2], "b": ["x", "y"]}).to_csv(path, index=False)
    frame = read_table(path)
    assert frame.to_dict("list") == {"a": [1, 2], "b": ["x", "y"]}


def test_a_semicolon_table_is_sniffed(tmp_path):
    """The separator is not the extension: UCI ships semicolons."""
    path = tmp_path / "t.csv"
    path.write_text("a;b\n1;2\n3;4\n")
    assert detect_separator(path) == ";"
    frame = read_table(path)
    assert list(frame.columns) == ["a", "b"]
    assert frame["b"].tolist() == [2, 4]


def test_a_tab_table_is_read(tmp_path):
    path = tmp_path / "t.tsv"
    pd.DataFrame({"a": [1], "b": [2]}).to_csv(path, sep="\t", index=False)
    frame = read_table(path)
    assert frame.to_dict("list") == {"a": [1], "b": [2]}


def test_a_gzipped_table_is_read(tmp_path):
    path = tmp_path / "t.csv.gz"
    with gzip.open(path, "wt") as handle:
        handle.write("a,b\n1,2\n")
    frame = read_table(path)
    assert frame.to_dict("list") == {"a": [1], "b": [2]}


def test_only_the_requested_columns_are_read(tmp_path):
    """This is the difference between two seconds and five minutes."""
    path = tmp_path / "wide.csv"
    columns = [f"g{i}" for i in range(3000)]
    pd.DataFrame([[float(i)] * len(columns) for i in range(5)], columns=columns).to_csv(
        path, index=False
    )

    frame = read_table(path, columns=["g0", "g2999"])

    assert list(frame.columns) == ["g0", "g2999"]
    assert frame.shape == (5, 2)


def test_a_requested_column_that_is_absent_is_a_clear_error(tmp_path):
    path = tmp_path / "t.csv"
    pd.DataFrame({"a": [1]}).to_csv(path, index=False)
    # pyarrow raises its own type, pandas raises ValueError; both are the
    # parser's message rather than a silent column of NaN.
    with pytest.raises((ValueError, Exception)):  # noqa: B017 - either parser
        read_table(path, columns=["a", "missing"])


def test_the_row_limit_still_stops_early(tmp_path):
    path = tmp_path / "t.csv"
    pd.DataFrame({"a": list(range(100))}).to_csv(path, index=False)
    assert len(read_table(path, nrows=5)) == 5


def test_a_parquet_table_still_reads(tmp_path):
    pytest.importorskip("pyarrow")
    path = tmp_path / "t.parquet"
    pd.DataFrame({"a": [1, 2]}).to_parquet(path)
    assert read_table(path)["a"].tolist() == [1, 2]
