"""Resolve a reference to bytes, stage them, and write down what happened.

The whole fetch is: parse -> connector -> hash -> manifest -> ledger. No
request is ever admitted or refused on policy grounds; the only reasons this
raises are technical (unknown scheme, unreachable host, over the size cap,
ambiguous accession).

`to_csv` is the one convenience offered, because GEO publishes its matrices
features x samples while Kosmos reads tables one row per observation. It is
opt-in, it writes a *derived* artifact beside the raw one, and the manifest
records what it was derived from -- the download itself is never transformed.
"""

from __future__ import annotations

from pathlib import Path

from .config import DataFetcherConfig
from .errors import FetchError
from .models import FetchResult, FileRecord, now_iso
from .profile import profile_table
from .references import parse_reference
from .sources import source_for
from .sources.geo import _read_table, write_samples_by_features_csv
from .store import append_ledger, find_reusable, sha256_file, write_manifest

#: Row counting reads the file, so it is only done for tables small enough that
#: a fetch does not visibly stall. Larger artifacts are profiled from a sample.
PROFILE_COUNT_ROWS_MAX_BYTES = 200 * 1024**2


def _to_csv(resolved, record: FileRecord, max_features: int | None) -> FileRecord:
    """Transpose a GEO series matrix to samples x features."""
    if not record.path.endswith("_series_matrix.txt.gz"):
        raise FetchError(
            f"--to-csv understands a GEO series matrix; {record.path} is not "
            f"one. The raw artifact was still written."
        )
    source_path = Path(resolved.directory) / record.path
    out = source_path.with_name(record.path.replace(".txt.gz", "") + ".csv")
    rows, columns = write_samples_by_features_csv(
        _read_table(source_path, limit=1 << 30), out, max_features=max_features
    )
    return FileRecord(
        path=out.name,
        bytes=out.stat().st_size,
        sha256=sha256_file(out),
        source_url=record.source_url,
        derived_from=record.path,
    )


def fetch(
    reference: str,
    *,
    config: DataFetcherConfig | None = None,
    query: str | None = None,
    to_csv: bool = False,
    max_features: int | None = None,
    profile: bool = True,
    reuse: bool | None = None,
) -> FetchResult:
    """Fetch `reference` into the staging root and return its record.

    A reference whose bytes are already staged is reused instead of downloaded
    again (say so with `reuse=False`, which is what `--refresh` passes).

    Raises `ReferenceError`, `SourceError` or `FetchError`; nothing else is
    swallowed, because a fetch that half-worked must not look like one that did.
    """
    config = config or DataFetcherConfig.from_env()
    ref = parse_reference(reference)
    source = source_for(ref.scheme)
    if reuse is None:
        reuse = config.reuse

    if reuse:
        found = find_reusable(config, str(ref))
        if found is not None:
            record, checked = found
            # The reader that turns a single-cell file into a table gets better
            # (a `.h5ad` now carries the obs column that labels a perturbation),
            # and a table built by an older one is stale rather than fetched.
            # Rebuild it from the bytes already on disk; the network is not
            # asked for anything, and the original file is untouched.
            rebuilt = _rebuild_derived(record, config)
            if rebuilt:
                write_manifest(Path(record.directory), record)
                record.notes.append(
                    "the table converted from this reference was rebuilt from "
                    "the staged file (the reader changed)"
                )
            append_ledger(
                config,
                {
                    "at": now_iso(),
                    "kind": "fetch_reuse",
                    "reference": str(ref),
                    "scheme": ref.scheme,
                    "directory": record.directory,
                    "checked": checked,
                    "bytes": sum(file.bytes for file in record.files),
                },
            )
            record.notes.append(
                f"reused: already staged at {record.directory} "
                f"({len(record.files)} file(s), verified by {checked}); "
                f"no request was made"
            )
            return record

    append_ledger(
        config,
        {
            "at": now_iso(),
            "kind": "fetch",
            "reference": str(ref),
            "scheme": ref.scheme,
            "query": query,
            "root": str(config.root),
        },
    )

    try:
        resolved = source.fetch(ref, config)
    except Exception as e:  # noqa: BLE001 - recorded, then re-raised unchanged
        append_ledger(
            config,
            {
                "at": now_iso(),
                "kind": "fetch_error",
                "reference": str(ref),
                "scheme": ref.scheme,
                "ok": False,
                "error": f"{type(e).__name__}: {e}"[:500],
                **(
                    {"tool": ref.locator, "arguments": _selector_arguments(ref)}
                    if ref.scheme == "tu"
                    else {}
                ),
            },
        )
        raise
    # The attempt was logged before the request went out; this records what came
    # back, so the ledger answers "what did we actually get" and not only "what
    # did we ask for".
    append_ledger(
        config,
        {
            "at": now_iso(),
            "kind": "fetch_result",
            "reference": str(ref),
            "scheme": ref.scheme,
            "ok": True,
            "directory": str(resolved.directory),
            "revision": resolved.revision,
            "files": [
                {"path": record.path, "bytes": record.bytes, "sha256": record.sha256}
                for record in resolved.files
            ],
            # A ToolUniverse fetch is a tool call, not a URL: keep the tool and
            # its arguments as fields so the record is readable without parsing
            # the reference string back apart.
            **(
                {"tool": ref.locator, "arguments": _selector_arguments(ref)}
                if ref.scheme == "tu"
                else {}
            ),
        },
    )
    result = FetchResult(
        reference=str(ref),
        scheme=resolved.scheme,
        locator=resolved.locator,
        directory=str(resolved.directory),
        files=list(resolved.files),
        revision=resolved.revision,
        notes=list(resolved.notes),
        query=query,
    )
    if resolved.revision is None and ref.scheme == "geo":
        result.note("no revision id: identity is the sha256 of the artifact")
    _unpack_archives(
        result,
        max_bytes=config.max_bytes,
        max_members=config.archive_max_members,
    )
    _convert_single_cell(result, config)
    if to_csv:
        primary = result.primary
        if primary is None:
            raise FetchError("nothing was fetched, so there is nothing to convert")
        result.files.append(_to_csv(resolved, primary, max_features))
    if profile:
        # The shape of every staged table, recorded beside the hashes. A table
        # that failed to parse is reported in `notes` rather than failing the
        # fetch: the bytes are already on disk and are still the artifact.
        for record in result.files:
            try:
                result.profiles.append(
                    profile_table(
                        Path(result.directory) / record.path,
                        count_rows=record.bytes < PROFILE_COUNT_ROWS_MAX_BYTES,
                    ).model_dump(mode="json")
                )
            except Exception as e:  # noqa: BLE001 - profiling is advisory
                result.note(f"profile skipped for {record.path}: {e}")
    write_manifest(resolved.directory, result)
    return result


def _selector_arguments(ref) -> dict:
    """The JSON arguments of a `tu://` selector, or {} when it has none."""
    import json

    if not ref.selector:
        return {}
    try:
        parsed = json.loads(ref.selector)
    except json.JSONDecodeError:
        return {"_raw": ref.selector}
    return parsed if isinstance(parsed, dict) else {"_raw": parsed}


#: How archives are recognised. UCI ships most of its datasets as a `.zip`
#: holding one CSV, and a staged archive is not a table: without this the run
#: downloads the data, finds no table, and stops.
ARCHIVE_SUFFIXES = (".zip", ".tar", ".tar.gz", ".tgz", ".tar.bz2", ".rar")


def _archive_suffix(name: str) -> str | None:
    lowered = name.lower()
    for suffix in sorted(ARCHIVE_SUFFIXES, key=len, reverse=True):
        if lowered.endswith(suffix):
            return suffix
    return None


def _unpack_archives(
    result: FetchResult,
    *,
    max_bytes: int | None = None,
    max_members: int | None = None,
) -> None:
    """Unpack the archives a fetch staged, bounded and zip-slip safe.

    Each extracted file is recorded as a derived artifact, which is how the
    fetcher already describes "this file came out of that one". A `.rar` is
    named and left alone: no extractor ships with Python, and pretending the
    bytes are a table is worse than saying so.

    `max_members` bounds how many members are unpacked. A GEO `RAW.tar` is one
    triplet per sample and a series can hold 25 of them, so unpacking all 76
    files costs disk and time for samples nothing downstream will read; the
    members that were left are counted in the note, and the archive itself stays
    staged, so nothing is lost that another run cannot get by naming it.
    """
    import tarfile
    import zipfile

    directory = Path(result.directory)
    for record in list(result.files):
        suffix = _archive_suffix(record.path)
        if suffix is None:
            continue
        source = directory / record.path
        if not source.exists():
            continue
        if suffix == ".rar":
            result.note(
                f"{record.path} is a RAR archive; this fetcher cannot unpack it "
                f"(no unrar available), so its contents are not usable"
            )
            continue
        target = directory / f"{source.name}-contents"
        try:
            if suffix == ".zip":
                with zipfile.ZipFile(source) as archive:
                    members = [m for m in archive.infolist() if not m.is_dir()]
                    members, skipped = _bounded_members(
                        members, max_members, max_bytes, lambda m: getattr(m, "file_size", 0)
                    )
                    for member in members:
                        _guard_member(member.filename, directory)
                    archive.extractall(
                        target, members=[member.filename for member in members]
                    )
                    extracted = [member.filename for member in members]
            else:
                with tarfile.open(source) as archive:
                    members = [m for m in archive.getmembers() if m.isfile()]
                    members, skipped = _bounded_members(
                        members, max_members, max_bytes, lambda m: m.size
                    )
                    for member in members:
                        _guard_member(member.name, directory)
                    archive.extractall(target, members=members)
                    extracted = [member.name for member in members]
        except FetchError:
            raise
        except Exception as e:  # noqa: BLE001 - a broken archive is a finding
            result.note(f"could not unpack {record.path}: {type(e).__name__}: {e}")
            continue
        for name in extracted:
            path = target / name
            if not path.is_file():
                continue
            result.files.append(
                FileRecord(
                    path=str(path.relative_to(directory)),
                    bytes=path.stat().st_size,
                    sha256=sha256_file(path),
                    derived_from=record.path,
                )
            )
        result.note(
            f"{record.path} unpacked to {len(extracted)} file(s); they are "
            f"candidates in their own right"
            + (
                f". {skipped} further file(s) were left inside the archive "
                f"(caps: {_unpack_caps(max_members, max_bytes)}); raise "
                f"--archive-members or name one directly to take more"
                if skipped
                else ""
            )
        )


def _unpack_caps(max_members: int | None, max_bytes: int | None) -> str:
    return ", ".join(
        part
        for part in (
            f"{max_members} members" if max_members else "",
            f"{max_bytes:,} bytes" if max_bytes else "",
        )
        if part
    ) or "none"


def _bounded_members(
    members: list,
    max_members: int | None,
    max_bytes: int | None = None,
    size_of=lambda member: 0,
) -> tuple[list, int]:
    """The members to unpack, and how many were left inside the archive.

    Two bounds, both about not turning a fetched archive into an unbounded
    unpack: a count (one triplet per sample adds up) and a byte budget. Whatever
    is left stays in the archive, which is still staged.
    """
    keep = members[:max_members] if max_members else list(members)
    if max_bytes:
        used = 0
        bounded = []
        for member in keep:
            size = size_of(member)
            if bounded and used + size > max_bytes:
                break
            bounded.append(member)
            used += size
        keep = bounded
    return keep, len(members) - len(keep)


def _guard_member(name: str, root: Path) -> None:
    """Refuse an archive entry that would write outside the staging directory."""
    candidate = (root / name).resolve()
    if not str(candidate).startswith(str(root.resolve())):
        raise FetchError(f"archive member {name!r} would be written outside {root}")


def _rebuild_derived(result: FetchResult, config: DataFetcherConfig) -> bool:
    """Re-convert this record's single-cell tables; say whether anything changed.

    Returns False for an ordinary record (nothing derived, or the tables came
    out byte-identical to the files already on disk), so a reuse of plain CSV
    bytes stays a pure read. The comparison is against the disk, not against the
    manifest: a table left behind by an older reader is stale on disk even when
    the manifest records it as current.

    Only the tables the converter writes (`<source>-table.csv`) are rebuilt. An
    archive's members are also "derived" artifacts, and dropping them here — the
    first version of this did — silently emptied a `.tar` fetch down to the
    archive itself, which is not a table: the islet run then stopped with "none
    of the fetched files is a table".
    """
    directory = Path(result.directory)

    def is_converted_table(file: FileRecord) -> bool:
        return file.derived_from is not None and file.path.endswith("-table.csv")

    before = {
        file.path: (
            sha256_file(directory / file.path)
            if (directory / file.path).exists()
            else None
        )
        for file in result.files
        if is_converted_table(file)
    }
    result.files = [file for file in result.files if not is_converted_table(file)]
    _convert_single_cell(result, config)
    after = {
        file.path: file.sha256
        for file in result.files
        if is_converted_table(file)
    }
    return after != before


def _convert_single_cell(result: FetchResult, config: DataFetcherConfig) -> None:
    """Turn any staged single-cell file into a bounded table beside it.

    The table is a derived artifact, like the contents of an unpacked archive:
    the original file stays exactly as it was fetched, and the table records
    what it came from plus the selection used to make it (cells, genes, seed).
    A file whose format this reader does not know is left alone with a note.
    """
    from .singlecell import (
        convert_single_cell,
        is_single_cell,
        selection_from_config,
    )

    directory = Path(result.directory)
    for record in list(result.files):
        if record.derived_from is not None or not is_single_cell(record.path):
            continue
        path = directory / record.path
        if not path.exists():
            continue
        selection = selection_from_config(config)
        # A barcode -> label table beside an mtx triplet is the only way a
        # CellRanger download can be labeled; look for one before giving up.
        metadata = _metadata_beside(path)
        try:
            table, selection = convert_single_cell(
                path, selection=selection, metadata=metadata
            )
        except Exception as e:  # noqa: BLE001 - an unreadable file is a finding
            result.note(f"{record.path} could not be read as single-cell data: {type(e).__name__}: {e}")
            continue
        target = directory / f"{path.name}-table.csv"
        table.to_csv(target, index=False)
        facts = selection.facts
        result.files.append(
            FileRecord(
                path=str(target.relative_to(directory)),
                bytes=target.stat().st_size,
                sha256=sha256_file(target),
                derived_from=record.path,
                conversion=facts.to_dict() if facts else None,
            )
        )
        result.note(
            f"{record.path} converted to {target.name}: "
            + (
                facts.describe()
                if facts
                else f"{len(table)} rows x {len(table.columns)} columns"
            )
            + "; "
            + "; ".join(selection.notes)
            + _bounds_note(selection)
        )


def _bounds_note(selection) -> str:
    """The pre-check's own rule, in words: cells are capped, genes are not.

    A reviewer who sees a sample of a file without being told it is a sample
    reads it as the whole file, and the gene panel is the dimension that must not
    be silently trimmed -- so both are stated on every converted table.
    """
    genes = (
        f"genes capped at {selection.max_genes:,}"
        if selection.max_genes
        else "genes not capped"
    )
    return f"; pre-check bounds: cells capped at {selection.max_cells:,}, {genes}"


def _metadata_beside(path: Path) -> Path | None:
    """A barcode/x/label table next to a single-cell file, when there is one."""
    for candidate in sorted(path.parent.iterdir()):
        if candidate == path or not candidate.is_file():
            continue
        lowered = candidate.name.lower()
        if lowered.endswith((".csv", ".tsv", ".txt")) and "metadata" in lowered:
            return candidate
    return None


def list_reference_files(
    reference: str, config: DataFetcherConfig | None = None
) -> dict:
    """What a reference contains, without downloading it.

    Used after the labeled table is known: a repository's other files, or a GEO
    series' supplementary files, are the likeliest sources of same-schema
    evidence, and listing them costs one API call instead of a download.
    """
    config = config or DataFetcherConfig.from_env()
    ref = parse_reference(reference)
    if ref.scheme == "hf":
        from .sources.hf import HuggingFaceSource

        source = HuggingFaceSource()
        files = source.listing(ref, config)
        return {
            "reference": str(ref),
            "scheme": ref.scheme,
            "files": files,
            "selector": ref.selector,
            "bytes": sum(int(entry.get("bytes") or 0) for entry in files),
            "about": source.about(ref, config),
        }
    if ref.scheme == "geo":
        from .sources.geo import list_files_detailed, series_about

        listed = list_files_detailed(ref.locator, config)
        base = f"geo://{ref.locator}"
        files = [
            {
                "path": entry["path"],
                "bytes": entry["bytes"],
                "size": entry["size"],
                "reference": (
                    str(ref) if ref.selector else f"{base}#{entry['path']}"
                ),
            }
            for entry in listed
        ]
        return {
            "reference": str(ref),
            "scheme": ref.scheme,
            "files": files,
            "selector": ref.selector,
            "bytes": sum(int(entry.get("bytes") or 0) for entry in files),
            "about": series_about(ref.locator, config),
        }
    # One file per reference: there is nothing to choose between, but the size
    # still decides whether a run wants to spend its budget on it.
    size = None
    page = ""
    error = ""
    if ref.scheme in {"http", "https"}:
        from .sources.net import get_prefix, head_size, looks_like_a_web_page

        url = f"{ref.scheme}://{ref.locator}"
        size = head_size(url, config)
        # A moved dataset path answers 200 with the site's own "not found" page,
        # so the size alone says nothing. Four kilobytes of the body decide it,
        # before a run spends anything on the file.
        try:
            page = looks_like_a_web_page(get_prefix(url, config, limit=4096))
        except Exception as e:  # noqa: BLE001 - reported, not raised: a probe
            error = str(e)
    return {
        "reference": str(ref),
        "scheme": ref.scheme,
        "files": (
            [{"path": ref.selector or ref.locator, "bytes": size}] if size else []
        ),
        "selector": ref.selector,
        "bytes": size or 0,
        "about": page,
        "web_page": page,
        "error": error,
        "note": (
            "this scheme is one file per reference, so there is nothing to list: "
            "the reference itself is the whole dataset"
        ),
    }
