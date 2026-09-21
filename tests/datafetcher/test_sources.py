"""Connectors: http, file and geo, against the fake web."""

from __future__ import annotations

import pytest
from conftest import empty_series_matrix_text, gz, series_matrix_text

from datafetcher import DataFetcherConfig, SourceError, fetch
from datafetcher.references import parse_reference
from datafetcher.sources.geo import (
    GeoSource,
    matrix_has_rows,
    parse_series_matrix,
    write_samples_by_features_csv,
)

GEO_ROOT = "https://geo.test/series"
SERIES = "GSE55296"
SERIES_DIR_URL = f"{GEO_ROOT}/GSE55nnn/{SERIES}"


def _config(tmp_path, **overrides) -> DataFetcherConfig:
    return DataFetcherConfig(root=tmp_path / "out", geo_ftp_root=GEO_ROOT, **overrides)


def _publish_series(web, matrix_name="GSE55296_series_matrix.txt.gz", suppl=("counts.txt.gz",)):
    web.index(f"{SERIES_DIR_URL}/matrix/", [matrix_name])
    web.add(f"{SERIES_DIR_URL}/matrix/{matrix_name}", gz(series_matrix_text()))
    web.index(f"{SERIES_DIR_URL}/suppl/", list(suppl))
    for name in suppl:
        web.add(f"{SERIES_DIR_URL}/suppl/{name}", gz(series_matrix_text()))
    return SERIES_DIR_URL


def test_http_download_hashes_stages_and_names(tmp_path, web):
    url = web.add("https://example.test/tables/evidence.csv", "sample_id,a\ns1,1\n")
    result = fetch(url, config=_config(tmp_path))

    record = result.primary
    assert record is not None and record.path == "evidence.csv"
    assert record.bytes == len("sample_id,a\ns1,1\n")
    assert record.source_url == url
    staged = tmp_path / "out" / "http" / "example.test__tables__evidence.csv" / "evidence.csv"
    assert staged.read_text() == "sample_id,a\ns1,1\n"
    assert web.requested == [url]


def test_http_download_refuses_over_the_size_cap_and_leaves_no_partial(tmp_path, web):
    url = web.add("https://example.test/big.csv", b"x" * 5000)
    with pytest.raises(SourceError, match="exceeds the"):
        fetch(url, config=_config(tmp_path, max_bytes=1000))
    # `.partial` is the partial-download directory; no leftover *file* in it.
    assert [p for p in (tmp_path / "out").rglob("*.partial") if p.is_file()] == []
    assert list((tmp_path / "out").rglob("big.csv")) == []


def test_offline_mode_refuses_before_any_request(tmp_path, web):
    url = web.add("https://example.test/table.csv", "a\n1\n")
    with pytest.raises(SourceError, match="offline mode"):
        fetch(url, config=_config(tmp_path, offline=True))
    assert web.requested == []


def test_file_source_copies_and_records_the_origin(tmp_path):
    source = tmp_path / "local.csv"
    source.write_text("sample_id,a\ns1,1\n")
    result = fetch(f"file://{source}", config=_config(tmp_path))
    record = result.primary
    assert record is not None and record.source_url == source.as_uri()
    staged = list((tmp_path / "out" / "file").rglob("local.csv"))
    assert len(staged) == 1 and staged[0].read_text() == source.read_text()


def test_file_source_refuses_a_directory(tmp_path):
    with pytest.raises(SourceError, match="is a directory"):
        fetch(f"file://{tmp_path}", config=_config(tmp_path))


def test_geo_fetches_the_series_matrix_and_says_it_has_no_revision(tmp_path, web):
    _publish_series(web)
    result = fetch(f"geo://{SERIES}", config=_config(tmp_path))
    record = result.primary
    assert record is not None and record.path == f"{SERIES}_series_matrix.txt.gz"
    assert result.revision is None
    assert any("no revision id" in note for note in result.notes)
    assert f"{SERIES_DIR_URL}/matrix/{SERIES}_series_matrix.txt.gz" in web.requested


def test_geo_selector_fetches_one_supplementary_file(tmp_path, web):
    _publish_series(web, suppl=("counts.txt.gz", "other.txt.gz"))
    result = fetch(f"geo://{SERIES}#suppl/counts.txt.gz", config=_config(tmp_path))
    assert result.primary is not None and result.primary.path == "counts.txt.gz"
    assert f"{SERIES_DIR_URL}/suppl/counts.txt.gz" in web.requested


def test_geo_selector_names_the_files_it_does_have(tmp_path, web):
    _publish_series(web, suppl=("counts.txt.gz",))
    with pytest.raises(SourceError, match="counts.txt.gz"):
        fetch(f"geo://{SERIES}#suppl/missing.txt.gz", config=_config(tmp_path))


def test_geo_refuses_an_empty_matrix_and_points_at_suppl(tmp_path, web):
    """The RNA-seq case: landing an empty table would read as 'no rows'."""
    web.index(f"{SERIES_DIR_URL}/matrix/", [f"{SERIES}_series_matrix.txt.gz"])
    web.add(
        f"{SERIES_DIR_URL}/matrix/{SERIES}_series_matrix.txt.gz",
        gz(empty_series_matrix_text()),
    )
    web.index(f"{SERIES_DIR_URL}/suppl/", ["counts.txt.gz"])
    with pytest.raises(SourceError, match="counts.txt.gz"):
        fetch(f"geo://{SERIES}", config=_config(tmp_path))


def test_geo_refuses_to_pick_between_platforms(tmp_path, web):
    _publish_series(web, matrix_name="GSE55296_series_matrix.txt.gz")
    web.index(
        f"{SERIES_DIR_URL}/matrix/",
        [f"{SERIES}_series_matrix.txt.gz", f"{SERIES}-GPL97_series_matrix.txt.gz"],
    )
    with pytest.raises(SourceError, match="one per platform"):
        fetch(f"geo://{SERIES}", config=_config(tmp_path))


def test_geo_refuses_a_sample_accession_and_a_revision(tmp_path, web):
    source = GeoSource()
    config = _config(tmp_path)
    with pytest.raises(SourceError, match="GEO SERIES accession"):
        source.fetch(parse_reference("geo://GSM123"), config)
    with pytest.raises(SourceError, match="no revision"):
        source.fetch(parse_reference(f"geo://{SERIES}@r1"), config)


def test_geo_selector_grammar_is_checked(tmp_path, web):
    with pytest.raises(SourceError, match="selector must name a file"):
        fetch(f"geo://{SERIES}#pheno", config=_config(tmp_path))


def test_series_matrix_transposes_to_samples_by_features(tmp_path):
    text = series_matrix_text()
    assert matrix_has_rows(text) is True
    samples, features, values = parse_series_matrix(text)
    assert samples == ["GSM1", "GSM2"]
    assert features == ["ENSG00000000001", "ENSG00000000002"]
    assert values == [["1.0", "2.0"], ["2.0", "4.0"]]

    out = tmp_path / "out.csv"
    rows, columns = write_samples_by_features_csv(text, out, max_features=1)
    assert (rows, columns) == (2, 1)
    assert out.read_text().splitlines() == [
        "sample_id,ENSG00000000001",
        "GSM1,1.0",
        "GSM2,2.0",
    ]


def test_geo_to_csv_writes_a_derived_artifact(tmp_path, web):
    _publish_series(web)
    result = fetch(f"geo://{SERIES}", config=_config(tmp_path), to_csv=True)
    derived = [f for f in result.files if f.derived_from]
    assert len(derived) == 1
    assert derived[0].derived_from == f"{SERIES}_series_matrix.txt.gz"
    assert derived[0].path.endswith(".csv")
