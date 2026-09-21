"""The on-disk layout: one directory per fetch, a manifest, and a ledger.

    data/fetched/
    ├── ledger.jsonl              every search and every fetch, appended
    ├── geo/GSE55296/
    │   ├── GSE55296_series_matrix.txt.gz
    │   └── manifest.json
    └── hf/owner__name/...

The ledger is written **before** a query leaves the machine, which is the one
ordering property worth keeping from AutoEvidence's finder: an outbound call
that cannot be attributed afterwards is the thing an operator can never check.
Here it is a plain append-only log, not a gate -- nothing reads it to decide
whether a request is allowed.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from .config import DataFetcherConfig
from .models import FetchResult

MANIFEST_NAME = "manifest.json"
MANIFEST_VERSION = 1


def sha256_file(path: str | Path, chunk: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(chunk), b""):
            digest.update(block)
    return digest.hexdigest()


def append_ledger(config: DataFetcherConfig, entry: dict[str, Any]) -> Path:
    """Append one line to the ledger. Never overwrites, never reorders."""
    config.ensure_dirs()
    line = json.dumps(entry, default=str, sort_keys=True)
    with config.ledger_path.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    return config.ledger_path


def read_ledger(config: DataFetcherConfig, limit: int | None = None) -> list[dict]:
    if not config.ledger_path.exists():
        return []
    rows = [
        json.loads(line)
        for line in config.ledger_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    return rows[-limit:] if limit else rows


def stage_dir(config: DataFetcherConfig, scheme: str, name: str) -> Path:
    """`<root>/<scheme>/<name>` with the name kept a single path component."""
    directory = config.root / scheme / name
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def write_manifest(directory: Path, result: FetchResult) -> Path:
    """Append a fetch result to the directory's manifest, atomically.

    Re-fetching the same reference adds a record rather than replacing one, so
    the file keeps the history of what was retrieved and when.
    """
    path = directory / MANIFEST_NAME
    existing: list[dict] = []
    if path.exists():
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            existing = list(payload.get("records", []))
        except json.JSONDecodeError:
            # A truncated manifest is worse than an extra file: keep it.
            backup = path.with_suffix(".json.corrupt")
            path.replace(backup)
    payload = {
        "version": MANIFEST_VERSION,
        "records": existing + [result.model_dump(mode="json")],
    }
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=directory, delete=False
    ) as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.flush()
        os.fsync(handle.fileno())
        tmp = Path(handle.name)
    tmp.replace(path)
    return path


def read_manifest(directory: str | Path) -> list[FetchResult]:
    path = Path(directory) / MANIFEST_NAME
    if not path.exists():
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    return [FetchResult.model_validate(record) for record in payload.get("records", [])]


#: Files above this size are reused on their recorded size alone: the manifest is
#: only written after a download completed, so a wrong size is what a truncated
#: or replaced file looks like, and re-hashing 3 GB on every run costs more than
#: it tells us.
REUSE_HASH_MAX_BYTES = 512 * 1024**2


def find_reusable(
    config: DataFetcherConfig, reference: str, *, verify: bool = True
) -> tuple[FetchResult, str] | None:
    """A fetch of this reference whose bytes are still on disk, if there is one.

    Returns the record and how it was checked (`sha256` or `size`). The ledger
    here shows the same 616 MB file fetched nine times and the same 314 MB
    archive seven times, all of it re-downloading bytes that were already
    staged; nothing consulted what was on disk before opening a connection.

    When several records of this reference are still on disk, the one that
    describes the most files wins. A later record may legitimately be smaller
    (`--archive-members 2`), but a record that has *lost* artifacts describes
    less than what is staged: reusing it once emptied a `.tar` fetch down to the
    archive itself, and the next run found "no table" where there were six.
    """
    if not config.root.exists():
        return None
    newest: tuple[FetchResult, str] | None = None
    richest = 0
    for manifest in sorted(config.root.rglob(MANIFEST_NAME)):
        try:
            payload = json.loads(manifest.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        for raw in payload.get("records", []):
            if str(raw.get("reference")) != reference:
                continue
            try:
                record = FetchResult.model_validate(raw)
            except Exception:  # noqa: BLE001 - an older manifest is not a hit
                continue
            check = _still_on_disk(record, verify=verify)
            if check is not None and (newest is None or len(record.files) >= richest):
                newest = (record, check)
                richest = len(record.files)
    return newest


def _still_on_disk(record: FetchResult, *, verify: bool) -> str | None:
    """`sha256`, `size`, or None when the staged files are gone or changed."""
    if not record.files:
        return None
    directory = Path(record.directory)
    checked = "size"
    # A derived table is rebuilt from its source, not part of what was fetched:
    # judging the staged bytes by it would send a reference back to the network
    # because a reader improved the table it writes beside the file.
    wanted = [file for file in record.files if file.derived_from is None]
    if not wanted:
        return None
    for file in wanted:
        path = directory / file.path
        try:
            if path.stat().st_size != file.bytes:
                return None
        except OSError:
            return None
        if not verify or file.bytes > REUSE_HASH_MAX_BYTES:
            continue
        if sha256_file(path) != file.sha256:
            return None
        checked = "sha256"
    return checked


def list_records(
    config: DataFetcherConfig, scheme: str | None = None
) -> list[FetchResult]:
    """Every fetch recorded under the root, newest last."""
    if not config.root.exists():
        return []
    directories: Iterable[Path] = (
        [config.root / scheme] if scheme else sorted(p for p in config.root.iterdir() if p.is_dir())
    )
    records: list[FetchResult] = []
    for directory in directories:
        if not directory.is_dir() or directory.name == ".partial":
            continue
        for child in sorted(p for p in directory.iterdir() if p.is_dir()):
            records.extend(read_manifest(child))
    return records
