"""Render a datafetcher sample packet for a prompt.

Kosmos does not import the fetcher, so the packet arrives as a dict (its shape is
part of the same JSON contract as the plan) and is rendered here. Both halves are
shown on purpose: the raw lines as they are in the file, and the columns our
parser produced from them. A headerless file is visible precisely in the gap
between the two.

The fetcher renders wide tables itself (`packet["rendered"]`, head and tail of
both the rows and the columns) and that rendering is used verbatim when it is
there: it keeps the label column of a 14,089-gene table visible, where showing
"the first 60 columns" showed the model genes and nothing else and it concluded
the table had no label. The fallback below is for packets built by hand.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any


def render_packet(packet: dict[str, Any], max_columns: int = 60) -> str:
    rendered = packet.get("rendered")
    if isinstance(rendered, str) and rendered.strip():
        return rendered
    path = str(packet.get("path", "?"))
    lines = [f"file: {Path(path).name}", "first lines, exactly as in the file:"]
    for line in packet.get("raw_head") or []:
        lines.append(f"  {str(line)[:1000]}")
    if not packet.get("raw_head"):
        lines.append("  (could not read the file as text)")

    columns = [str(c) for c in (packet.get("columns") or [])]
    kinds = packet.get("kinds") or {}
    nulls = packet.get("null_fraction") or {}
    if len(columns) > max_columns:
        # Both ends: a table's label and its identifiers are as likely to sit at
        # one end as the other, and a wide table's first `max_columns` are all
        # measurements.
        half = max_columns // 2
        shown = [*columns[:half], f"... {len(columns) - 2 * half} more ...", *columns[-half:]]
    else:
        shown = columns
    header = "columns our parser produced"
    if len(columns) > max_columns:
        header += f" (first and last {max_columns // 2} of {len(columns)})"
    lines.append(f"{header}:")
    for name in shown:
        if name.startswith("... ") and name.endswith(" more ..."):
            lines.append(f"  {name}")
            continue
        lines.append(f"  {name} | {kinds.get(name, '?')} | null={nulls.get(name, '?')}")
    lines.append(
        f"delimiter={packet.get('delimiter')!r}, rows sampled={packet.get('rows_sampled')}"
    )
    for note in packet.get("notes") or []:
        lines.append(f"note from the parser: {note}")
    return "\n".join(lines)
