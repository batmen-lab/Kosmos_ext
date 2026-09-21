"""The review packet: what a table looks like, before anyone decides its role.

Role assignment used to be purely mechanical -- "is the target column present,
are the features numeric" -- and that is fine arithmetic, but it cannot tell
that the first row of a file is *data* rather than column names, that `Outcome`
is the label the question calls "diabetes", or that a column is a sample id.
Those are judgements about meaning, and they need the actual rows.

So this collects a bounded packet: the raw first lines as text (not our parse of
them), the columns we parsed out, their dtypes, null rates, the delimiter and
the row count. Both the raw text and our reading are included on purpose -- the
gap between them is exactly how a headerless file is spotted.

The packet is bounded: a few rows and a few kilobytes, never the table.
"""

from __future__ import annotations

import gzip
import json
from pathlib import Path
from typing import Any

from .profile import binary_table_sample, profile_table
from .store import MANIFEST_NAME

DEFAULT_ROWS = 5
MAX_RAW_BYTES = 4096
#: Fields kept at each end of a wide row before the middle is elided.
RENDER_FIELDS = 25


def _render_rows(path: str | Path, rows: int) -> list[str] | None:
    """The first rows of a parquet/arrow file, rendered as text, or None.

    Reading those files as text gives the model binary noise or nothing at all,
    and a review of nothing is how a table with a `pce` column came back
    "unusable ... the parser misread it as text". The file carries its own
    column names and values; show those.
    """
    import pandas as pd

    frame = binary_table_sample(path, rows)
    if frame is None:
        return None
    # " | " rather than a comma: a cell like "SLG, ITO, NiO-np, ..." is a device
    # stack, and joining on commas would make it read as several columns.
    lines = [" | ".join(str(name) for name in frame.columns)]
    for _, row in frame.iterrows():
        cells = ["" if pd.isna(value) else str(value) for value in row.tolist()]
        lines.append(" | ".join(cells))
    return lines


def raw_head_with_kind(
    path: str | Path,
    rows: int = DEFAULT_ROWS,
    max_bytes: int = MAX_RAW_BYTES,
    *,
    delimiter: str = "",
) -> tuple[list[str], str]:
    """The first `rows` of the file, and whether they are text or a table."""
    rendered = _render_rows(path, rows)
    if rendered is not None:
        return rendered, "table"
    return raw_head(path, rows=rows, max_bytes=max_bytes, delimiter=delimiter), "text"


def raw_head(
    path: str | Path,
    rows: int = DEFAULT_ROWS,
    max_bytes: int = MAX_RAW_BYTES,
    *,
    delimiter: str = "",
) -> list[str]:
    """The first `rows` lines of the file as text, bounded by `max_bytes`.

    `delimiter` binds each line field-wise before the byte budget is applied.
    A single-cell table's first line is tens of kilobytes long and the budget
    refused it whole, so the packet carried no rows at all for exactly the tables
    a reviewer most needs to see; keeping both ends of the line keeps the columns
    that say what a row *is* (an id, a label) visible at the far end.
    """
    path = Path(path)
    opener = gzip.open if path.suffix.lower() == ".gz" else open
    lines: list[str] = []
    used = 0
    try:
        with opener(path, "rt", encoding="utf-8", errors="replace") as handle:
            for line in handle:
                if len(lines) >= rows:
                    break
                line = _bounded_line(line.rstrip("\n"), delimiter, RENDER_FIELDS)
                if used + len(line) > max_bytes:
                    # One line wider than the whole budget still gets in, cut to
                    # the budget: a packet with lines beats a packet with none.
                    if not lines:
                        lines.append(line[:max_bytes])
                    break
                lines.append(line)
                used += len(line)
    except OSError:
        return []
    return lines


def _bounded_line(line: str, delimiter: str, keep: int) -> str:
    """Head and tail of a line, so neither end hides the other."""
    if not delimiter:
        return line
    fields = line.split(delimiter)
    if len(fields) <= 2 * keep:
        return line
    head = delimiter.join(fields[:keep])
    tail = delimiter.join(fields[-keep:])
    return f"{head}{delimiter} ... [{len(fields) - 2 * keep} more]{delimiter}{tail}"


def sample_packet(
    path: str | Path, *, rows: int = DEFAULT_ROWS, max_bytes: int = MAX_RAW_BYTES
) -> dict[str, Any]:
    """Enough of a table for a model to say what it is, and nothing more."""
    path = Path(path)
    profile = profile_table(path, sample_rows=max(rows, DEFAULT_ROWS))
    # A table wider than the render cap has rows longer than this packet; bind
    # them field-wise, because the label is as likely to be the last column as
    # the first (a converted single-cell table puts it after 14,089 genes).
    wide = len(profile.columns) > 2 * RENDER_FIELDS
    head, kind = raw_head_with_kind(
        path,
        rows=rows,
        max_bytes=max_bytes,
        delimiter=profile.delimiter if wide else "",
    )
    provenance = conversion_facts(path)
    notes = list(profile.notes)
    if provenance and provenance.get("conversion"):
        notes.append(precheck_note(provenance["conversion"]))
    packet = {
        "path": str(path),
        "raw_head": head,
        # "table" means these rows were read from the file's own table format
        # (parquet, arrow), not guessed from bytes.
        "raw_head_kind": kind,
        "columns": profile.names,
        "kinds": {column.name: column.kind for column in profile.columns},
        "null_fraction": {column.name: round(column.null_fraction, 3) for column in profile.columns},
        "delimiter": profile.delimiter,
        "rows_sampled": profile.sampled_rows,
        "notes": notes,
        # What the fetcher recorded when it produced this file, when it did:
        # which artifact it came from and how much of it was kept.
        "provenance": provenance,
    }
    # Rendering belongs with the readers that know a wide single-cell table from
    # a wide flat one, so the caller (or a model) can print the packet without
    # re-implementing the head-and-tail rules.
    packet["rendered"] = compact(packet)
    return packet


def conversion_facts(path: str | Path) -> dict[str, Any] | None:
    """What the fetcher wrote down about this file, when it is a derived table.

    The manifest sits with the staged files, so the record is found by walking
    up from the file -- an archive's extracted contents live one level below it.
    """
    target = Path(path).resolve()
    for directory in [target.parent, *target.parents]:
        manifest = directory / MANIFEST_NAME
        if not manifest.is_file():
            continue
        try:
            payload = json.loads(manifest.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        for record in payload.get("records", []):
            staged = record.get("directory")
            if not staged:
                continue
            for entry in record.get("files", []):
                if (Path(staged) / str(entry.get("path", ""))).resolve() != target:
                    continue
                return {
                    "reference": record.get("reference"),
                    "scheme": record.get("scheme"),
                    "directory": staged,
                    "derived_from": entry.get("derived_from"),
                    "source_url": entry.get("source_url"),
                    "conversion": entry.get("conversion"),
                }
        return None
    return None


def precheck_note(conversion: dict[str, Any]) -> str:
    """The provenance sentence for a converted table, in words."""
    from .singlecell import SingleCellFacts

    try:
        return SingleCellFacts(**conversion).precheck_note()
    except TypeError:  # a manifest written by an older version
        return f"this table was converted from {conversion.get('source') or 'another file'}"


def _row_ends(line: str, delimiter: str, keep: int = 25) -> str:
    """A wide row, rendered so both ends are visible.

    A single-cell table has a column per gene: 200 of them, with `cell_type`
    last. Truncating the line at N characters showed the model the first genes
    and hid the label, and it then reported "no cell-type label column" for a
    table that has one. The head and the tail are what carry the meaning here.
    """
    if not delimiter:
        return line[:300]
    fields = line.split(delimiter)
    if len(fields) <= 2 * keep:
        rendered = delimiter.join(fields)
        return rendered[:300] + (" ..." if len(rendered) > 300 else "")
    head = delimiter.join(fields[:keep])
    tail = delimiter.join(fields[-keep:])
    return f"{head}{delimiter} ... [{len(fields) - 2 * keep} more]{delimiter}{tail}"


def compact(packet: dict[str, Any], max_columns: int = 60) -> str:
    """A readable rendering for a prompt: raw lines, then how we parsed them."""
    if packet.get("raw_head_kind") == "table":
        # These rows were read from the file's own table format, so the model
        # is looking at the data rather than at the bytes it is stored as.
        heading = "first rows (read from the file's own table format):"
    else:
        heading = "first lines as text:"
    lines = [f"file: {Path(packet['path']).name}", heading]
    delimiter = str(packet.get("delimiter") or "")
    for line in packet["raw_head"]:
        lines.append(f"  {_row_ends(line, delimiter)}")
    columns = packet["columns"]
    if len(columns) > max_columns:
        # The first *and* the last: a table's label and its identifiers are as
        # likely to be at one end as the other.
        half = max_columns // 2
        shown = [*columns[:half], f"... {len(columns) - 2 * half} more ...", *columns[-half:]]
        lines.append(f"columns we parsed it into ({len(columns)}):")
    else:
        shown = columns
        lines.append("columns we parsed it into:")
    for name in shown:
        if name.startswith("... "):
            lines.append(f"  {name}")
            continue
        lines.append(
            f"  {name} | {packet['kinds'].get(name)} | "
            f"null={packet['null_fraction'].get(name)}"
        )
    lines.append(
        f"delimiter={packet['delimiter']!r}, rows_sampled={packet['rows_sampled']}"
    )
    for note in packet.get("notes") or []:
        lines.append(f"note: {note}")
    return "\n".join(lines)


def dumps(packet: dict[str, Any]) -> str:
    return json.dumps(packet, indent=2, default=str)
