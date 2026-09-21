"""Typed records: a search hit, a fetched file, a fetch result.

Every field that carries third-party text is bounded, on the same principle
AutoEvidence's `schema/candidates.py` uses: a repository description is
arbitrary text written by a stranger, and it ends up in a manifest, in a
terminal and in an agent's context. Truncating at the type means a future
adapter that forgets cannot widen what a search pushes downstream.

There is no `verified` flag and no `columns` field here, for the same reason
they are absent upstream: a hit is a lead. Nothing in this package has resolved,
downloaded or inspected a dataset at the moment a hit is created, so there is no
value such a field could honestly take.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import PurePosixPath

from pydantic import BaseModel, ConfigDict, Field

MAX_NOTE_CHARS = 500

#: Repository furniture: never "the data path" a caller should be handed.
_FURNITURE = {
    ".gitattributes",
    ".gitignore",
    "manifest.json",
}
_FURNITURE_SUFFIXES = (".md", ".json", ".yaml", ".yml", ".toml", ".ini", ".cfg", ".lock")


def _looks_like_data(path: str) -> bool:
    name = PurePosixPath(path).name
    if name.lower() in _FURNITURE or name.startswith("."):
        return False
    if name.lower().startswith(("readme", "license", "changelog", "citation")):
        return False
    return not name.lower().endswith(_FURNITURE_SUFFIXES)


def now_iso() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


class FileRecord(BaseModel):
    """One artifact on disk, with the hash that makes it re-checkable."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    #: Relative to the fetch directory, so the manifest stays portable.
    path: str = Field(min_length=1)
    bytes: int = Field(ge=0)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_url: str | None = None
    #: Set on artifacts this package produced from another artifact (--to-csv).
    derived_from: str | None = None
    #: For a derived table: what the reader saw in the file it came from and what
    #: it kept (cells on disk against cells written, genes, the label column).
    #: A converted single-cell table is a *sample* of a much larger file, and a
    #: reviewer who is not told so reads the sample as the whole dataset.
    conversion: dict | None = None


class FetchResult(BaseModel):
    """One resolved reference, staged on disk and written down."""

    model_config = ConfigDict(extra="forbid")

    reference: str
    scheme: str
    locator: str
    directory: str
    files: list[FileRecord] = Field(default_factory=list)
    #: Upstream revision, when the scheme has one (HuggingFace commit sha).
    revision: str | None = None
    retrieved_at: str = Field(default_factory=now_iso)
    notes: list[str] = Field(default_factory=list)
    #: The search intent that led here, when the fetch came from a hit.
    query: str | None = None
    #: Shape of the staged tables (columns, kinds, rows) so a reader can decide
    #: what role each artifact can play without opening it again. Optional:
    #: manifests written before profiling existed still validate.
    profiles: list[dict] = Field(default_factory=list)

    @property
    def primary(self) -> FileRecord | None:
        """The one file a caller should pass to a run, when there is one.

        A repository ships metadata beside its tables (`.gitattributes`, a
        README, a dataset card), and calling one of those "the data path" hands
        a caller a file that fails at the far end with a confusing error. So:
        candidates are the non-derived files that are not repository furniture,
        and when there is not exactly one, this returns None rather than picking.
        """
        plain = [f for f in self.files if f.derived_from is None]
        candidates = [f for f in plain if _looks_like_data(f.path)]
        if len(candidates) == 1:
            return candidates[0]
        if len(plain) == 1:
            return plain[0]
        return None

    @property
    def data_files(self) -> list[FileRecord]:
        """Every fetched file that could reasonably be read as data."""
        return [
            f for f in self.files if f.derived_from is None and _looks_like_data(f.path)
        ]

    @property
    def handoff(self) -> FileRecord | None:
        """The artifact a run should read.

        With `--to-csv` the derived table is the one Kosmos can use (GEO writes
        features x samples, Kosmos reads one row per observation), so it is
        preferred over the raw download it came from.
        """
        derived = [
            f
            for f in self.files
            if f.derived_from and f.path.lower().endswith((".csv", ".tsv"))
        ]
        if len(derived) == 1:
            return derived[0]
        return self.primary

    def path_of(self, record: FileRecord) -> str:
        from pathlib import Path

        return str(Path(self.directory) / record.path)

    def note(self, text: str) -> None:
        self.notes.append(text[:MAX_NOTE_CHARS])
