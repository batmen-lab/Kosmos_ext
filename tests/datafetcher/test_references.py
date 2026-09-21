"""Reference parsing is the one place a third-party string becomes a path."""

from __future__ import annotations

import pytest

from datafetcher import ReferenceError, parse_reference
from datafetcher.sources.base import name_for


def test_each_scheme_parses_into_its_parts():
    geo = parse_reference("geo://GSE55296#suppl/counts.txt.gz")
    assert (geo.scheme, geo.locator, geo.selector) == (
        "geo",
        "GSE55296",
        "suppl/counts.txt.gz",
    )

    hf = parse_reference("hf://owner/name@abc123#data/train.csv")
    assert (hf.scheme, hf.locator, hf.revision, hf.selector) == (
        "hf",
        "owner/name",
        "abc123",
        "data/train.csv",
    )

    http = parse_reference("https://example.org/path/file.csv")
    assert (http.scheme, http.locator) == ("https", "example.org/path/file.csv")

    local = parse_reference("file:///tmp/data.csv")
    assert (local.scheme, local.locator) == ("file", "/tmp/data.csv")


def test_str_round_trips():
    for text in (
        "geo://GSE55296",
        "geo://GSE55296#suppl/a.txt.gz",
        "hf://owner/name",
        "hf://owner/name@rev#file.csv",
        "https://example.org/x.csv",
        "file:///tmp/x.csv",
    ):
        assert str(parse_reference(text)) == text


@pytest.mark.parametrize(
    "bad,why",
    [
        ("", "empty"),
        ("GSE55296", "no scheme"),
        ("s3://bucket/key", "unknown scheme"),
        ("geo://", "names nothing"),
        ("https://", "no host"),
        ("file://relative/path.csv", "not absolute"),
        ("geo://GSE1#../etc/passwd", "traversal"),
    ],
)
def test_bad_references_refuse(bad, why):
    with pytest.raises(ReferenceError):
        parse_reference(bad)


def test_locator_becomes_one_directory_component():
    """A locator that could escape the staging root refuses; it is not rewritten."""
    assert name_for(parse_reference("geo://GSE55296")) == "GSE55296"
    assert name_for(parse_reference("hf://owner/name")) == "owner__name"
    assert name_for(parse_reference("https://example.org/a/b.csv")) == "example.org__a__b.csv"
    # file:// names are the stem plus a digest, so two same-named files in
    # different directories do not land in one place.
    first = name_for(parse_reference("file:///tmp/a/data.csv"))
    second = name_for(parse_reference("file:///tmp/b/data.csv"))
    assert first != second
    assert first.startswith("data-") and second.startswith("data-")


def test_a_traversing_locator_is_refused_not_sanitised():
    ref = parse_reference("geo://GSE123")
    object.__setattr__(ref, "locator", "../escape")
    with pytest.raises(ReferenceError):
        name_for(ref)
