"""What each proposal would cost, decided before anything is transferred.

Retrieval names datasets from what the model knows; the cost of what it named is
a fact only the repository can state, and it is knowable *before* the transfer:
GEO prints a size beside every supplementary file, the Hub returns sizes with a
repository listing, and an `https://` reference answers a HEAD request. Two
runs' worth of that amounts to the same 616 MB file fetched nine times and two
unrelated multi-gigabyte repositories pulled before anything had read them.

So the retrieval step gets a second, cheaper question: here is what each
candidate is and what it would cost -- which of them still look worth
downloading, and which file of the ones on offer? The answer is a set of
references; the download itself stays mechanical.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from typing import Any


def _schema() -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "downloads": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "reference": {
                            "type": "string",
                            "description": (
                                "one reference exactly as offered in the listing "
                                "above, including any #selector"
                            ),
                        },
                        "why": {
                            "type": "string",
                            "description": "one sentence: what this file gives the question",
                        },
                    },
                    "required": ["reference"],
                },
            }
        },
        "required": ["downloads"],
    }


def plan_prompt(
    question: str, listings: Sequence[dict[str, Any]], *, used: set[str] | None = None
) -> str:
    """The listing as a model reads it: what it is, and what it costs."""
    used = used or set()
    blocks = []
    for index, listing in enumerate(listings, start=1):
        reference = str(listing.get("reference"))
        lines = [f"[{index}] {_mark_used(reference, used)}"]
        if listing.get("about"):
            lines.append(f"    about: {_clip(str(listing['about']), 400)}")
        if listing.get("why"):
            lines.append(f"    proposed because: {_clip(str(listing['why']), 240)}")
        files = list(listing.get("files") or [])
        if files:
            lines.append(f"    files on offer ({len(files)}, smallest first):")
            for entry in files[:12]:
                size = entry.get("size") or f"{int(entry.get('bytes') or 0):,} bytes"
                lines.append(
                    f"      {_mark_used(str(entry.get('reference')), used)}  {size}"
                )
            if len(files) > 12:
                lines.append(f"      ... ({len(files) - 12} more)")
        elif listing.get("note"):
            lines.append(f"    one file per reference: {listing['note']}")
        if listing.get("bytes"):
            lines.append(f"    total on offer: {listing['bytes']:,} bytes")
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks)


def _mark_used(text: str, used: set[str]) -> str:
    """`person.csv` -> `person.csv [already in this run]` when it is."""
    name = text.rsplit("/", 1)[-1]
    return f"{text} [already in this run]" if text in used or name in used else text


def choose_downloads(
    question: str,
    listings: Sequence[dict[str, Any]],
    *,
    client: Any,
    log: Callable[[str], None] = print,
    max_downloads: int | None = None,
    used: Sequence[str] = (),
) -> list[dict[str, str]]:
    """Which offered references to download, as the model reads the listing.

    Returns `[{"reference": ..., "why": ...}]`, filtered so that every reference
    was actually on offer. An empty list is a valid answer: a candidate that
    turns out to be a different dataset is worth nothing, and skipping it costs
    nothing but the call.
    """
    offered: dict[str, dict[str, Any]] = {}
    for listing in listings:
        offered[str(listing.get("reference"))] = listing
        for entry in listing.get("files") or []:
            if entry.get("reference"):
                offered[str(entry["reference"])] = entry
    if not offered:
        return []

    allowed = sorted(offered)[:400]
    used_set = {str(item) for item in used}
    prompt = (
        "You are about to spend a run's bandwidth on datasets for an analysis.\n\n"
        f"Research question:\n{question}\n\n"
        f"What the repositories say each candidate is, and what it costs:\n"
        f"{plan_prompt(question, listings, used=used_set)}\n\n"
        "Rules for your answer:\n"
        "- Choose the *measurement* file: the one whose rows are the observations "
        "the question is about (an `.h5ad`, `.parquet`, a `.csv`/`.tsv` of cells "
        "with their genes).\n"
        "- Never choose an index of the data instead of the data. A `manifest/`, "
        "a `README`, a `*_files.tsv`, a `*_summary.tsv`, a `SHA256SUMS` or a "
        "`filelist` *points at* the measurements and contains none; picking one "
        "of those is how a run ends with nothing to train on.\n"
        "- Among files that could answer the question, choose the smallest. A "
        "0.6 GB file and a 2.7 GB file from the same series often carry the same "
        "thing.\n"
        "- Anything marked `[already in this run]` is already here: the labeled "
        "table and whatever was fetched for it. Choosing one of those is always "
        "wrong -- it adds nothing and it is refused. Pick a *different* file, "
        "even when it is the larger one.\n"
        "- Drop a candidate whose description says it is a different dataset, a "
        "different organism, or a different measurement, however confident the "
        "earlier proposal was.\n"
        f"- At most {max_downloads or len(listings)} download(s).\n"
        "- `reference` must be copied exactly from the listing, including a "
        "`#selector` when you are choosing one file from a repository or series.\n"
        "- An empty list is a valid answer when nothing on offer fits.\n\n"
        "Answer as JSON with exactly the fields in the schema."
    )
    log(f"# preflight: asking the model which of {len(listings)} candidate(s) to download")
    try:
        response = client.generate_structured(
            prompt=prompt, schema=_schema(), max_tokens=900, temperature=0
        )
    except Exception as e:  # noqa: BLE001 - a failed call must not fail the run
        log(f"# preflight: the model call failed ({e}); keeping the candidates")
        return [{"reference": str(listing.get("reference")), "why": ""} for listing in listings]
    if isinstance(response, str):
        try:
            response = json.loads(response)
        except json.JSONDecodeError:
            log("# preflight: the model returned non-JSON; keeping the candidates")
            return [{"reference": str(listing.get("reference")), "why": ""} for listing in listings]
    if not isinstance(response, dict):
        return [{"reference": str(listing.get("reference")), "why": ""} for listing in listings]

    chosen: list[dict[str, str]] = []
    seen: set[str] = set()
    for item in response.get("downloads") or []:
        reference = str((item or {}).get("reference") or "").strip()
        if not reference:
            continue
        match = _match_offered(reference, allowed)
        if match is None:
            log(f"# preflight: {reference!r} was not on offer; ignoring it")
            continue
        if match in seen:
            continue
        seen.add(match)
        chosen.append({"reference": match, "why": str((item or {}).get("why") or "")})
    return chosen[: max_downloads or len(chosen)]


def _match_offered(reference: str, allowed: Sequence[str]) -> str | None:
    """The offered string the model meant, tolerating `#selector` spacing."""
    for candidate in allowed:
        if candidate == reference:
            return candidate
    squashed = reference.replace(" #", "#").replace("# ", "#").strip()
    for candidate in allowed:
        if candidate.replace(" #", "#").replace("# ", "#").strip() == squashed:
            return candidate
    return None


def _clip(text: str, limit: int) -> str:
    return text if len(text) <= limit else text[: limit - 1] + "…"
