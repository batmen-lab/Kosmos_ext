"""What a run should not download: staged bytes, wrong choices, whole archives.

The ledger from the first real runs reads: the same 616 MB file fetched nine
times, the same 314 MB archive seven times, and two unrelated multi-gigabyte
repositories pulled before anything had read them. Each test here is one of
those, turned into a rule.
"""

from __future__ import annotations

import json
import tarfile

import pytest

from datafetcher import DataFetcherConfig, SourceError, fetch
from datafetcher.fetch import list_reference_files
from datafetcher.sources.geo import format_size, parse_size
from datafetcher.sources.hf import choose_bounded
from datafetcher.sources.net import head_size, looks_like_a_web_page


def _config(tmp_path, **overrides) -> DataFetcherConfig:
    return DataFetcherConfig(
        root=tmp_path / "out", geo_ftp_root="https://geo.test/series", **overrides
    )


def test_a_reference_already_staged_is_reused_without_a_request(tmp_path, web):
    url = web.add("https://example.test/evidence.csv", "sample_id,a\ns1,1\n")
    config = _config(tmp_path)

    first = fetch(url, config=config)
    second = fetch(url, config=config)

    assert web.requested == [url]
    assert second.directory == first.directory
    assert any("reused" in note for note in second.notes)
    assert any("no request was made" in note for note in second.notes)


def test_reuse_is_refused_when_the_staged_bytes_changed(tmp_path, web):
    url = web.add("https://example.test/evidence.csv", "sample_id,a\ns1,1\n")
    config = _config(tmp_path)
    result = fetch(url, config=config)
    staged = tmp_path / "out" / "http" / "example.test__evidence.csv" / "evidence.csv"
    staged.write_text("sample_id,a\ns9,9\n")

    again = fetch(url, config=config)

    assert web.requested == [url, url]  # the changed copy is not trusted
    assert again.directory == result.directory
    assert not any("reused" in note for note in again.notes)


def test_reuse_is_off_when_the_caller_asks_for_a_fresh_copy(tmp_path, web):
    url = web.add("https://example.test/evidence.csv", "sample_id,a\ns1,1\n")
    config = _config(tmp_path)
    fetch(url, config=config)
    fetch(url, config=config, reuse=False)

    assert web.requested == [url, url]


def test_a_converted_table_is_rebuilt_from_the_staged_bytes(tmp_path, web):
    """A better reader must not send the reference back to the network.

    The table beside a single-cell file is derived, not fetched: it is rebuilt
    from the bytes already on disk when the reader that writes it changes. The
    alternative is what happened live -- a stale `.h5ad-table.csv` kept being
    reused, so the obs column that labels a perturbation was missing from it no
    matter how many times the question was asked.
    """
    h5py = pytest.importorskip("h5py")
    import numpy as np

    source = tmp_path / "sample.h5ad"
    with h5py.File(source, "w") as handle:
        handle.create_dataset("X", data=np.zeros((6, 2), dtype=np.float32))
        obs = handle.create_group("obs")
        for name, values in (
            ("compound_1", ["drugA", "drugB"] * 3),
            ("dose", np.arange(6, dtype=np.float32)),
        ):
            if isinstance(values, np.ndarray):
                obs.create_dataset(name, data=values)
                continue
            levels = sorted(set(values))
            group = obs.create_group(name)
            group.create_dataset("categories", data=np.array([v.encode() for v in levels]))
            index = {level: i for i, level in enumerate(levels)}
            group.create_dataset("codes", data=np.array([index[v] for v in values], dtype=np.int32))
        handle.create_group("var").create_dataset(
            "_index", data=np.array([b"G0", b"G1"])
        )
    url = web.add("https://example.test/sample.h5ad", source.read_bytes())
    config = _config(tmp_path)
    fetch(url, config=config)
    derived = (
        tmp_path / "out" / "http" / "example.test__sample.h5ad" / "sample.h5ad-table.csv"
    )
    assert derived.exists()

    # An older reader's table: the genes and `batch`, and no label column.
    derived.write_text("G0,G1\n0.0,1.0\n")

    again = fetch(url, config=config)

    assert web.requested == [url]  # the bytes on disk were trusted
    assert any("reused" in note for note in again.notes)
    assert any("rebuilt from the staged file" in note for note in again.notes)
    assert "compound_1" in derived.read_text().splitlines()[0]
    # A measurement is still a measurement: only columns that could label a row
    # are carried out of `obs`.
    assert "dose" not in derived.read_text().splitlines()[0]


def _tar_bytes(members: dict[str, str]) -> bytes:
    import io
    import tarfile

    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w") as archive:
        for name, text in members.items():
            payload = text.encode()
            info = tarfile.TarInfo(name)
            info.size = len(payload)
            archive.addfile(info, io.BytesIO(payload))
    return buffer.getvalue()


def test_reusing_an_archive_keeps_the_members_it_was_unpacked_into(tmp_path, web):
    """A reuse must not shrink a fetch down to the archive.

    The members of a `.tar` are "derived" artifacts too, and a reuse that
    rebuilt every derived record emptied an islet fetch down to the archive
    itself -- which is not a table -- and the run then stopped with "none of the
    fetched files is a table". Only the converted tables may be rebuilt.
    """
    blob = _tar_bytes(
        {
            "human1_counts.csv": "cell,assigned_cluster,G1\nc1,alpha,3\n",
            "human2_counts.csv": "cell,assigned_cluster,G1\nc2,beta,5\n",
        }
    )
    url = web.add("https://example.test/GSE_RAW.tar", blob)
    config = _config(tmp_path)

    first = fetch(url, config=config)
    second = fetch(url, config=config)

    members = [file.path for file in second.files if file.derived_from is not None]
    assert len(members) == 2
    assert members == [file.path for file in first.files if file.derived_from is not None]
    assert web.requested == [url]


def test_reuse_prefers_the_record_that_describes_the_most_files(tmp_path, web):
    """A record that lost its artifacts does not get to hide the ones on disk."""
    blob = _tar_bytes({"human1_counts.csv": "cell,assigned_cluster,G1\nc1,alpha,3\n"})
    url = web.add("https://example.test/GSE_RAW.tar", blob)
    config = _config(tmp_path)
    complete = fetch(url, config=config)
    manifest = tmp_path / "out" / "http" / "example.test__GSE_RAW.tar" / "manifest.json"
    payload = json.loads(manifest.read_text())
    # A later reuse that recorded only the archive, as the broken one did.
    damaged = json.loads(json.dumps(complete.model_dump(mode="json")))
    damaged["files"] = [f for f in damaged["files"] if f.get("derived_from") is None]
    payload["records"].append(damaged)
    manifest.write_text(json.dumps(payload))

    from datafetcher.store import find_reusable

    found = find_reusable(config, url)

    assert found is not None
    assert len(found[0].files) == len(complete.files)


def test_geo_listings_carry_the_size_the_index_prints(tmp_path, web):
    accession = "GSE194122"
    root = "https://geo.test/series/GSE194nnn/" + accession
    web.listing(f"{root}/matrix/", [(f"{accession}_series_matrix.txt.gz", "1.2M")])
    web.listing(
        f"{root}/suppl/",
        [
            (f"{accession}_multiome_BMMC.h5ad.gz", "2.7G"),
            (f"{accession}_cite_BMMC.h5ad.gz", "587M"),
        ],
    )

    payload = list_reference_files(f"geo://{accession}", _config(tmp_path))

    sizes = {entry["path"]: entry["bytes"] for entry in payload["files"]}
    cite = sizes["suppl/GSE194122_cite_BMMC.h5ad.gz"]
    multiome = sizes["suppl/GSE194122_multiome_BMMC.h5ad.gz"]
    assert cite < multiome
    # Smallest first: the cheap artifact is the one a caller reads about first.
    assert payload["files"][0]["path"] == "matrix/GSE194122_series_matrix.txt.gz"
    assert payload["bytes"] == sum(sizes.values())


def test_a_geo_refusal_names_the_sizes_and_the_smallest_file(tmp_path, web):
    accession = "GSE120221"
    root = "https://geo.test/series/GSE120nnn/" + accession
    web.listing(f"{root}/matrix/", [])
    web.listing(
        f"{root}/suppl/",
        [(f"{accession}_RAW.tar", "300M"), (f"{accession}_counts.txt.gz", "2.1M")],
    )

    with pytest.raises(SourceError) as caught:
        fetch(f"geo://{accession}", config=_config(tmp_path))

    message = str(caught.value)
    assert "counts.txt.gz (2.1 MB)" in message
    assert "smallest" in message
    # Smallest first, so the reader sees the cheap option before the archive.
    assert message.index("counts.txt.gz") < message.index("RAW.tar")


def test_size_parsing_matches_what_the_index_prints():
    assert parse_size("587M") == 587 * 1024**2
    assert parse_size("2.7G") == int(2.7 * 1024**3)
    assert parse_size("-") is None
    assert format_size(587 * 1024**2) == "587.0 MB"
    assert format_size(None) == "size unknown"


def test_head_size_reads_the_header_the_server_sent(tmp_path, web):
    url = web.add("https://example.test/big.zip", b"x" * 4096)
    assert head_size(url, _config(tmp_path)) == 4096


def test_listing_a_url_says_when_it_is_a_web_page(tmp_path, web):
    """A moved dataset path answers 200 with the site's own "not found" page.

    The size and the status code both look fine; four kilobytes of the body are
    what say the URL is wrong -- and `list-files` is what a run reads before it
    downloads anything.
    """
    url = web.add(
        "https://wwwn.cdc.gov/Nchs/Nhanes/2017-2018/P_BMX.csv",
        "<!DOCTYPE html><html><head><title>Page Not Found | CDC</title></head>"
        "<body>404</body></html>",
    )

    payload = list_reference_files(url, _config(tmp_path))

    assert payload["web_page"] == "Page Not Found | CDC"


def test_listing_a_real_table_is_not_a_web_page(tmp_path, web):
    url = web.add("https://example.test/table.csv", "age,weight\n44,71.2\n")
    assert list_reference_files(url, _config(tmp_path))["web_page"] == ""


def test_a_url_that_returns_a_web_page_is_refused_as_a_wrong_url(tmp_path, web):
    """A moved dataset path answers 200 with the site's own 404 page.

    The download used to succeed, and the failure surfaced much later as "this
    is not a table" -- which reads like a problem with the data instead of with
    the identifier. Refusing it here is what lets the retrieval step correct it.
    """
    page = b"<!DOCTYPE html>\n<html><head><title>Page Not Found | CDC</title></head></html>\n"
    url = web.add("https://wwwn.cdc.gov/Nchs/Nhanes/2017-2018/P_BMX.csv", page)

    with pytest.raises(SourceError) as caught:
        fetch(url, config=_config(tmp_path))

    message = str(caught.value)
    assert "web page" in message
    assert "Page Not Found | CDC" in message
    # Nothing was staged: a page is not a dataset.
    assert list((tmp_path / "out").rglob("P_BMX.csv")) == []


def test_ordinary_csv_bytes_are_not_mistaken_for_a_page():
    assert looks_like_a_web_page(b"<!DOCTYPE html><title>x</title>") != ""
    assert looks_like_a_web_page(b"<html>", "text/html; charset=utf-8") != ""
    assert looks_like_a_web_page(b"age,weight\n44,71.2\n") == ""
    assert looks_like_a_web_page(b"\x1f\x8b\x08\x00") == ""


def test_a_local_file_is_not_held_to_the_download_cap(tmp_path):
    """`file://` copies; a 3 GB local `.h5ad` is not a mistake."""
    source = tmp_path / "local.h5ad"
    source.write_bytes(b"x" * 4096)

    result = fetch(
        f"file://{source}",
        config=_config(tmp_path, max_bytes=1024, local_max_bytes=10_000),
    )

    assert result.primary is not None and result.primary.bytes == 4096


def test_a_repository_without_a_named_file_yields_one_small_table():
    """Two unrelated multi-gigabyte repositories arrived this way."""
    wanted, why = choose_bounded(
        {
            "README.md": 2000,
            "data/train-00000-of-00005.parquet": 300_000_000,
            "data/test.parquet": 12_000_000,
            "data/notes.txt": 900,
        },
        cap=2 * 1024**3,
    )

    # The training split, even though `test.parquet` is smaller.
    assert wanted == ["data/train-00000-of-00005.parquet"]
    assert "smallest data file" in why


def test_without_a_train_split_the_smallest_table_is_taken():
    wanted, _ = choose_bounded(
        {"README.md": 2000, "data/eval.parquet": 9_000, "data/held_out.parquet": 4_000},
        cap=2 * 1024**3,
    )

    assert wanted == ["data/held_out.parquet"]


def test_a_repository_whose_smallest_table_is_over_the_cap_is_refused():
    wanted, why = choose_bounded({"big.parquet": 5 * 1024**3}, cap=2 * 1024**3)

    assert wanted == []
    assert "over the" in why


def _tar_of_tables(tmp_path, name: str, count: int, suffix: str = "_barcodes_A.tsv"):
    archive = tmp_path / name
    with tarfile.open(archive, "w") as tar:
        for index in range(count):
            member = tmp_path / f"{name}-part-{index}"
            member.write_text(f"barcode\tcell_type\nbc{index}\tT cell\n")
            tar.add(member, arcname=f"GSM{index}{suffix}")
    return archive


def test_an_archive_is_unpacked_only_up_to_the_member_cap(tmp_path):
    """A 25-sample `RAW.tar` holds 76 files; a bounded read needs a few."""
    archive = _tar_of_tables(tmp_path, "GSE120221_RAW.tar", 20)

    result = fetch(f"file://{archive}", config=_config(tmp_path, archive_max_members=5))

    extracted = [f for f in result.files if f.derived_from == archive.name]
    assert len(extracted) == 5
    assert any(
        "15 further file(s) were left inside the archive" in note for note in result.notes
    )
    assert any("5 members" in note for note in result.notes)


def test_no_member_cap_unpacks_everything(tmp_path):
    archive = _tar_of_tables(tmp_path, "small.tar", 3)

    result = fetch(f"file://{archive}", config=_config(tmp_path, archive_max_members=0))

    assert len([f for f in result.files if f.derived_from == archive.name]) == 3
    assert not any("left inside the archive" in note for note in result.notes)
