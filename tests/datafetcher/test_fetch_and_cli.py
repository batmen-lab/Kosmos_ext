"""End to end: manifest, ledger, and the CLI a script would call."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from datafetcher import DataFetcherConfig, fetch
from datafetcher.cli import main
from datafetcher.store import list_records, read_ledger, read_manifest

CSV_URL = "https://example.test/evidence.csv"
STAGED = "example.test__evidence.csv"


def test_fetch_writes_a_manifest_a_ledger_line_and_a_usable_path(tmp_path, web):
    web.add(CSV_URL, "sample_id,a\ns1,1\n")
    config = DataFetcherConfig(root=tmp_path / "out")

    result = fetch(CSV_URL, config=config, query="bmmc cell types")
    record = result.primary
    assert record is not None

    records = read_manifest(tmp_path / "out" / "http" / STAGED)
    assert len(records) == 1
    assert records[0].query == "bmmc cell types"
    assert records[0].files[0].sha256 == record.sha256

    ledger = read_ledger(config)
    # The attempt is recorded before the request leaves; the result after it
    # returns, so the ledger answers both "what did we ask for" and "what did we
    # get".
    assert [row["kind"] for row in ledger] == ["fetch", "fetch_result"]
    assert ledger[0]["reference"] == CSV_URL
    result_row = ledger[1]
    assert result_row["ok"] is True
    assert result_row["files"][0]["sha256"] == record.sha256
    assert result_row["files"][0]["path"] == record.path

    # The path handed to Kosmos exists and is the fetched bytes.
    assert open(result.directory + "/" + record.path).read() == "sample_id,a\ns1,1\n"


def test_refetching_the_same_bytes_reuses_them_instead_of_downloading(tmp_path, web):
    """The ledger shows the same file fetched nine times: nothing checked disk.

    A second fetch of a reference whose bytes are staged and unchanged makes no
    request at all; `--refresh` (here `reuse=False`) is how a caller asks for the
    transfer again.
    """
    web.add(CSV_URL, "sample_id,a\ns1,1\n")
    config = DataFetcherConfig(root=tmp_path / "out")
    fetch(CSV_URL, config=config)
    again = fetch(CSV_URL, config=config, query="again")

    assert web.requested == [CSV_URL]  # one request for two fetches
    assert any("reused" in note for note in again.notes)
    assert len(list_records(config)) == 1

    fetch(CSV_URL, config=config, query="again", reuse=False)
    records = list_records(config)
    assert len(records) == 2
    assert records[-1].query == "again"


def test_the_ledger_records_which_search_a_fetch_came_from(tmp_path, web):
    """The query behind a download is part of the record, not a convention.

    Searches are the model's now and live in `search_trace.json`; what the
    ledger still has to carry is the query a fetched file was chosen for.
    """
    web.add(CSV_URL, "sample_id,a\ns1,1\n")
    config = DataFetcherConfig(root=tmp_path / "out")

    fetch(CSV_URL, config=config, query="bmmc")

    rows = read_ledger(config)
    assert [row["kind"] for row in rows] == ["fetch", "fetch_result"]
    assert rows[0]["query"] == "bmmc"


def test_cli_fetch_list_and_ledger(tmp_path, web, capsys):
    web.index("https://example.test/", ["ignored"])
    web.add(CSV_URL, "sample_id,a\ns1,1\n")
    staging = str(tmp_path / "out")

    assert main(["--root", staging, "fetch", CSV_URL, "--query", "bmmc", "--json"]) == 0
    fetched = json.loads(capsys.readouterr().out)
    assert fetched["files"][0]["path"] == "evidence.csv"

    assert main(["--root", staging, "list", "--json"]) == 0
    assert len(json.loads(capsys.readouterr().out)) == 1

    assert main(["--root", staging, "ledger", "--json"]) == 0
    ledger = json.loads(capsys.readouterr().out)
    assert [row["kind"] for row in ledger] == ["fetch", "fetch_result"]


def test_cli_reports_a_refusal_with_exit_code_2(tmp_path, capsys):
    assert main(["--root", str(tmp_path), "fetch", "s3://bucket/key"]) == 2
    assert "unsupported scheme" in capsys.readouterr().err


def test_a_zip_is_unpacked_and_its_table_becomes_the_artifact(tmp_path):
    """UCI ships its datasets as a zip holding one CSV.

    A staged archive is not a table: without unpacking, the run downloads the
    data, finds nothing to train on, and stops.
    """
    import zipfile

    from datafetcher.fetch import fetch

    source = tmp_path / "kidney.csv"
    source.write_text("age,bp,class\n48,80,ckd\n53,90,notckd\n")
    archive = tmp_path / "dataset_ckd.zip"
    with zipfile.ZipFile(archive, "w") as zipped:
        zipped.write(source, "kidney_disease.csv")
    config = DataFetcherConfig(root=tmp_path / "staged")

    result = fetch(f"file://{archive}", config=config)

    derived = [f for f in result.files if f.derived_from == archive.name]
    assert [Path(f.path).name for f in derived] == ["kidney_disease.csv"]
    assert any("unpacked to 1 file" in note for note in result.notes)
    extracted = Path(result.directory) / derived[0].path
    assert extracted.exists() and "class" in extracted.read_text()


def test_a_rar_is_reported_not_silently_staged(tmp_path):
    from datafetcher.fetch import fetch

    archive = tmp_path / "Chronic_Kidney_Disease.rar"
    archive.write_bytes(b"Rar!\x1a\x07\x00not really a rar")
    config = DataFetcherConfig(root=tmp_path / "staged-rar")

    result = fetch(f"file://{archive}", config=config)

    assert any("RAR archive" in note for note in result.notes)


def test_a_hub_url_fragment_is_accepted_as_a_repo_id():
    """A model that has seen a Hub page writes `hf://datasets/owner/name`."""
    from datafetcher.sources.hf import repo_id_of

    assert repo_id_of("datasets/healthcare/readmission") == "healthcare/readmission"
    assert repo_id_of("allisonhorst/palmerpenguins") == "allisonhorst/palmerpenguins"
    assert repo_id_of("/owner/name/") == "owner/name"


def test_a_failed_fetch_is_recorded_as_a_failure(tmp_path, web):
    """A refused download has to be visible afterwards, not just on screen."""
    from datafetcher import ReferenceError, SourceError
    from datafetcher import fetch as run_fetch

    config = DataFetcherConfig(root=tmp_path / "out")
    with pytest.raises(ReferenceError):
        run_fetch("s3://bucket/key.csv", config=config)
    assert read_ledger(config) == []  # refused before the ledger stage

    with pytest.raises(SourceError):
        run_fetch("https://example.test/missing.csv", config=config)  # 404 from the fake web
    rows = read_ledger(config)
    assert [row["kind"] for row in rows] == ["fetch", "fetch_error"]
    assert rows[1]["ok"] is False and "404" in rows[1]["error"]


def test_a_tooluniverse_fetch_records_the_tool_and_its_arguments(tmp_path, monkeypatch):
    """`tu://` is a tool call; the ledger keeps the tool and args as fields."""
    import datafetcher.tooluniverse as tu
    from datafetcher import DataFetcherConfig
    from datafetcher import fetch as run_fetch
    from datafetcher.store import sha256_file

    class StubDownloader:
        """Stands in for the Worker client; writes one file like download_file does."""

        detail = "stub interpreter"

        def __init__(self, python=None, timeout_s=None):
            pass

        def run(self, tool, arguments, out_dir):
            target = Path(out_dir) / "payload.csv"
            target.write_text("a,b\n1,2\n")
            return {
                "ok": True,
                "files": [
                    {
                        "path": str(target),
                        "bytes": target.stat().st_size,
                        "sha256": sha256_file(target),
                    }
                ],
            }

    monkeypatch.setattr(tu, "ToolUniverseDownloader", StubDownloader)
    config = DataFetcherConfig(root=tmp_path / "out")
    run_fetch('tu://download_file#{"url":"https://example.test/x.csv"}', config=config)
    row = read_ledger(config)[-1]
    assert row["kind"] == "fetch_result" and row["tool"] == "download_file"
    assert row["arguments"] == {"url": "https://example.test/x.csv"}
    assert row["files"][0]["sha256"]


def test_a_multi_file_repo_names_the_candidates_instead_of_picking_one(tmp_path, web):
    """A dataset card must never be handed to a run as `--data-path`."""
    from datafetcher.models import FetchResult, FileRecord

    result = FetchResult(
        reference="hf://owner/name",
        scheme="hf",
        locator="owner/name",
        directory=str(tmp_path),
        files=[
            FileRecord(path=".gitattributes", bytes=1, sha256="a" * 64),
            FileRecord(path="README.md", bytes=1, sha256="b" * 64),
            FileRecord(path="train.csv", bytes=1, sha256="c" * 64),
            FileRecord(path="test.csv", bytes=1, sha256="d" * 64),
        ],
    )
    assert result.primary is None
    assert [f.path for f in result.data_files] == ["train.csv", "test.csv"]

    single = result.model_copy(
        update={"files": [result.files[2]], "reference": "hf://owner/one"}
    )
    assert single.primary is not None and single.primary.path == "train.csv"


def test_the_handoff_artifact_is_the_csv_when_to_csv_ran(tmp_path):
    """Kosmos reads one row per observation, so the transpose is what to hand over."""
    from datafetcher.models import FetchResult, FileRecord

    raw = FileRecord(path="GSE2034_series_matrix.txt.gz", bytes=1, sha256="a" * 64)
    csv = FileRecord(
        path="GSE2034_series_matrix.csv",
        bytes=1,
        sha256="b" * 64,
        derived_from=raw.path,
    )
    result = FetchResult(
        reference="geo://GSE2034",
        scheme="geo",
        locator="GSE2034",
        directory=str(tmp_path),
        files=[raw, csv],
    )
    assert result.primary is not None and result.primary.path == raw.path
    assert result.handoff is not None and result.handoff.path == csv.path


def test_list_reports_a_multi_file_repo_by_count_not_as_no_files(
    tmp_path, web, capsys, monkeypatch
):
    """A repo with two tables is not 'no files'; say how many and how big."""
    web.add("https://hub.test/train.csv", "a\n1\n")
    web.add("https://hub.test/test.csv", "a\n2\n")
    staging = str(tmp_path / "out")
    assert main(["--root", staging, "fetch", "https://hub.test/train.csv"]) == 0
    capsys.readouterr()
    # Both files land in one record only for hf://; emulate by listing two rows.
    from datafetcher.config import DataFetcherConfig
    from datafetcher.models import FetchResult, FileRecord
    from datafetcher.store import write_manifest

    directory = tmp_path / "out" / "hf" / "owner__name"
    directory.mkdir(parents=True)
    write_manifest(
        directory,
        FetchResult(
            reference="hf://owner/name",
            scheme="hf",
            locator="owner/name",
            directory=str(directory),
            files=[
                FileRecord(path="train.csv", bytes=10, sha256="a" * 64),
                FileRecord(path="test.csv", bytes=20, sha256="b" * 64),
            ],
        ),
    )
    assert DataFetcherConfig(root=tmp_path / "out").root.exists()

    assert main(["--root", staging, "list"]) == 0
    out = capsys.readouterr().out
    assert "2 file(s), 30 bytes" in out
    assert "no files" not in out
