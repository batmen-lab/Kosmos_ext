"""The connector contract, and the registry that finds one by scheme."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

from ..config import DataFetcherConfig
from ..errors import ReferenceError
from ..models import FileRecord
from ..references import Reference


@dataclass
class Resolved:
    """What a connector produced: files on disk, plus how it got them."""

    scheme: str
    locator: str
    directory: Path
    files: list[FileRecord] = field(default_factory=list)
    revision: str | None = None
    notes: list[str] = field(default_factory=list)


class Source(Protocol):
    scheme: str

    def fetch(self, ref: Reference, config: DataFetcherConfig) -> Resolved: ...


SOURCES: dict[str, Source] = {}


def register(source: Source) -> None:
    SOURCES[source.scheme] = source


def source_for(scheme: str) -> Source:
    try:
        return SOURCES[scheme]
    except KeyError:
        raise ReferenceError(
            f"no connector registered for scheme {scheme!r}; known schemes: "
            f"{', '.join(sorted(SOURCES))}"
        ) from None


def name_for(ref: Reference) -> str:
    """The directory name a reference stages into.

    Each scheme decides its own shape because the locator grammars differ: a
    GEO accession is already a single component, a Hub repo id is `owner/name`,
    and an HTTP locator is `host/path/file`. All of them are checked here, and a
    bad one refuses before any request is made.
    """
    if ref.scheme == "file":
        filename = Path(ref.locator).name
        if not filename:
            raise ReferenceError(f"file:// reference {ref.locator!r} names no file")
        # Absolute paths are not usable as a directory name, and their basename
        # alone collides across directories (`experiments/data.csv` vs
        # `reports/data.csv`), so the name is the stem plus a locator digest.
        stem = Path(filename).stem or "file"
        digest = hashlib.sha1(ref.locator.encode("utf-8")).hexdigest()[:8]
        return f"{stem}-{digest}"
    return ref.safe_locator
