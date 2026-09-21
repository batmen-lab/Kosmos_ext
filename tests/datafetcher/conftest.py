"""Fixtures for the datafetcher suite.

Two deliberate choices:

  * the package is not installed (it is not in `packages.find`), so the
    repository root goes on `sys.path` here rather than in the developer's
    environment; and
  * the network is replaced, not simulated with a socket. `net._open` is the
    one call site for a request, so a fake web here asserts the exact URLs a
    connector builds, runs anywhere, and never depends on an upstream that may
    change under the suite.
"""

from __future__ import annotations

import gzip
import sys
import urllib.error
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


class FakeResponse:
    def __init__(self, blob: bytes, method: str = "GET"):
        self._blob = blob
        self._offset = 0
        # What a real response carries, so `head_size` reads a fake the same way
        # it reads a server.
        self.headers = {"Content-Length": str(len(blob))}
        self.method = method

    def read(self, size: int = -1) -> bytes:
        if size is None or size < 0:
            size = len(self._blob) - self._offset
        if self.method == "HEAD":
            return b""
        chunk = self._blob[self._offset : self._offset + size]
        self._offset += len(chunk)
        return chunk

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class FakeWeb:
    """A dict of URL -> bytes, with the HTTP errors the connectors must handle."""

    def __init__(self):
        self.resources: dict[str, bytes] = {}
        self.requested: list[str] = []

    def add(self, url: str, blob: bytes | str) -> str:
        self.resources[url] = blob.encode() if isinstance(blob, str) else blob
        return url

    def open(self, url: str, _config=None, method: str = "GET"):
        self.requested.append(url)
        if url not in self.resources:
            raise urllib.error.HTTPError(url, 404, "Not Found", None, None)
        return FakeResponse(self.resources[url], method=method)

    def index(self, url: str, names: list[str]) -> str:
        """A directory listing shaped the way GEO's own index pages are."""
        links = "".join(f'<a href="{name}">{name}</a>' for name in names)
        return self.add(url, f"<html>{links}<a href=\"https://policy.example\">p</a></html>")

    def listing(self, url: str, entries: list[tuple[str, str]]) -> str:
        """The same page with the size column NCBI prints (`name`, `2.7G`)."""
        rows = "".join(
            f'<a href="{name}">{name}</a>  2022-02-01 15:14  {size}\n'
            for name, size in entries
        )
        return self.add(url, f"<html><pre>{rows}<hr></pre></html>")


@pytest.fixture
def web(monkeypatch) -> FakeWeb:
    fake = FakeWeb()
    monkeypatch.setattr("datafetcher.sources.net._open", fake.open)
    return fake


def series_matrix_text(
    samples=("GSM1", "GSM2"), features=("ENSG00000000001", "ENSG00000000002")
) -> str:
    header = "\t".join(['"ID_REF"', *(f'"{s}"' for s in samples)])
    rows = []
    for position, feature in enumerate(features):
        cells = [f'"{(position + 1) * (index + 1)}.0"' for index in range(len(samples))]
        rows.append("\t".join([f'"{feature}"', *cells]))
    return "\n".join(
        [
            '!Series_title\t"demo"',
            "\t".join(['!Sample_geo_accession', *(f'"{s}"' for s in samples)]),
            "!series_matrix_table_begin",
            header,
            *rows,
            "!series_matrix_table_end",
            "",
        ]
    )


def empty_series_matrix_text(samples=("GSM1",)) -> str:
    """The RNA-seq shape: a header row and nothing under it."""
    return "\n".join(
        [
            "\t".join(['!Sample_geo_accession', *(f'"{s}"' for s in samples)]),
            "!series_matrix_table_begin",
            "\t".join(['"ID_REF"', *(f'"{s}"' for s in samples)]),
            "!series_matrix_table_end",
            "",
        ]
    )


def gz(text: str) -> bytes:
    return gzip.compress(text.encode())
