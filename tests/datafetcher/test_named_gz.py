"""A `.gz` name is not evidence that a file is gzip.

The plan stage's header normalisation writes plain text beside a `.gz` source,
and GEO serves some `.gz` files uncompressed. Deciding by suffix sent those to
`gzip.open`, which raised and aborted the whole plan -- the celltype-full run
produced no plan, so nothing trained.
"""

from __future__ import annotations

import gzip
import importlib.util
from pathlib import Path

from datafetcher.profile import is_gzip, profile_table
from datafetcher.sample import raw_head

ROOT = Path(__file__).resolve().parents[2]


def test_a_plain_table_named_gz_is_profiled_not_crashed(tmp_path):
    path = tmp_path / "table.csv.gz"
    path.write_text("gene,count\nA,1\nB,2\n", encoding="utf-8")

    assert is_gzip(path) is False
    profile = profile_table(path)
    assert [column.name for column in profile.columns] == ["gene", "count"]
    assert profile.sampled_rows == 2
    assert len(raw_head(path, rows=2)) == 2


def test_a_real_gzip_is_still_read_as_gzip(tmp_path):
    path = tmp_path / "real.csv.gz"
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        handle.write("gene,count\nA,1\nB,2\n")

    assert is_gzip(path) is True
    profile = profile_table(path)
    assert [column.name for column in profile.columns] == ["gene", "count"]


def test_a_corrupt_file_is_a_profile_result_not_a_crash(tmp_path):
    path = tmp_path / "broken.csv.gz"
    path.write_bytes(b"\x1f\x8b" + b"not really gzip")  # gzip magic, broken body

    profile = profile_table(path)          # must not raise
    assert profile.columns == []
    assert profile.notes


def test_header_normalisation_does_not_name_plain_text_gz(tmp_path):
    spec = importlib.util.spec_from_file_location(
        "auto_task_run", ROOT / "scripts" / "auto_task_run.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    source = tmp_path / "raw.csv.gz"          # the source is plain text too
    source.write_text("1,2\n3,4\n", encoding="utf-8")
    target = tmp_path / "normalized" / source.name

    assert module.write_with_header(str(source), ["a", "b"], target) is True
    written = sorted(p.name for p in (tmp_path / "normalized").iterdir())
    assert written == ["raw.csv"], written    # not `raw.csv.gz`
