"""Retrieval by the model: it proposes datasets, the fetcher downloads them.

The division of labour is deliberate:

  * **Deciding what to get is a model's job.** It reads the question, names
    concrete dataset identifiers (``geo://GSE194122``, ``hf://owner/name``, a
    URL) and says what it expects each to contain. Keyword search over repository
    metadata was the previous mechanism; it could only return what a repository
    happened to index, and it could not read the question.
  * **Getting it is mechanical.** Downloading, hashing and recording provenance
    is `datafetcher`'s job (its connectors, or ToolUniverse's download tools --
    anything that takes an identifier and returns bytes). Nothing in the
    retrieval step touches data.

The model can name a dataset that does not exist. That is not a failure mode to
hide: the fetch is the verification, and a proposal that cannot be downloaded is
printed with its reason and skipped. A model that is unavailable is a hard stop
for this step rather than a silent empty search -- there is no query to fall
back to when the model is the one choosing the datasets.

The client is Kosmos's own (`kosmos.core.llm.get_client`), so this uses the same
provider, key and model as the rest of the run.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

#: Schemes the mechanical layer can fetch. Kept as a shape check here so an
#: obviously unusable proposal is dropped before a subprocess is spawned; the
#: real validation is the fetch itself.
FALLBACK_SCHEMES = ("geo", "hf", "tu", "https", "http", "file")
_REFERENCE = re.compile(r"^[a-z][a-z0-9+.-]*://\S+$")


def _log(log: Callable[[str], None], message: str) -> None:
    log(message)


@dataclass
class Proposal:
    """One dataset the model wants fetched."""

    identifier: str
    why: str = ""
    contains: str = ""
    confidence: float = 0.0
    source: str = "llm"
    raw: dict[str, Any] = field(default_factory=dict)

    @property
    def fetchable_shape(self) -> bool:
        return bool(_REFERENCE.match(self.identifier.strip()))


def _search_tool_name(identifier: str) -> str | None:
    """The tool name when this identifier is a search call, else None.

    The model is offered ToolUniverse's tools, and it sometimes uses one the way
    it was meant to be used -- `tu://HuggingFace_search_datasets#{...}` -- rather
    than naming a dataset. Dropping those left the best lead of the run on the
    floor: the model had asked exactly the question that would have found the
    data.
    """
    text = str(identifier or "").strip()
    if not text.lower().startswith("tu://"):
        return None
    name = text[5:].split("#", 1)[0].strip()
    if not name:
        return None
    lowered = name.lower()
    if "search" in lowered or lowered.startswith(("find_", "list_")):
        return name
    return None


def _ask(client: Any, prompt: str, schema: dict[str, Any], log) -> Any:
    response = client.generate_structured(
        prompt=prompt, schema=schema, max_tokens=1200, temperature=0
    )
    if isinstance(response, str):
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            _log(log, "# retrieval: model did not return JSON")
            return None
    return response


def _harvest(
    entries: Any,
    *,
    log,
    seen: set[str],
    max_items: int,
    found: int = 0,
) -> tuple[list[Proposal], list[str]]:
    """Split what the model returned into proposals and search calls."""
    if not isinstance(entries, list):
        entries = []
    proposals: list[Proposal] = []
    searches: list[str] = []
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        identifier = str(entry.get("identifier", "")).strip()
        proposal = Proposal(
            identifier=identifier,
            why=str(entry.get("why", "")),
            contains=str(entry.get("contains", "")),
            confidence=float(entry.get("confidence") or 0.0),
            raw=entry,
        )
        if not identifier or identifier in seen:
            continue
        seen.add(identifier)
        if _search_tool_name(identifier):
            searches.append(identifier)
            continue
        if not proposal.fetchable_shape:
            _log(
                log,
                f"# retrieval: dropped {identifier!r} "
                f"(not a fetchable identifier; expected e.g. geo://GSE194122)",
            )
            continue
        scheme = identifier.split("://", 1)[0].lower()
        if scheme not in FALLBACK_SCHEMES:
            _log(log, f"# retrieval: dropped {identifier!r} (scheme {scheme!r} has no downloader)")
            continue
        proposals.append(proposal)
        _log(
            log,
            f"# retrieval: proposal {found + len(proposals)}/{max_items} {identifier} "
            f"(confidence {proposal.confidence:.2f})",
        )
        if proposal.contains:
            _log(log, f"#   contains: {proposal.contains[:160]}")
        if proposal.why:
            _log(log, f"#   why     : {proposal.why[:160]}")
    return proposals, searches


_DATASETS_SCHEMA = {
    "type": "object",
    "properties": {
        "datasets": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "identifier": {"type": "string"},
                    "contains": {"type": "string"},
                    "why": {"type": "string"},
                    "confidence": {"type": "number"},
                },
                "required": ["identifier", "why"],
            },
        }
    },
    "required": ["datasets"],
}


def _tool_block(tools: Sequence[dict[str, Any]]) -> str:
    """The tools the model may call, split by what they are for."""
    if not tools:
        return ""

    def listing(kind: str) -> str:
        return "\n".join(
            f"  tu://{tool['name']}#<json arguments>  {tool.get('description', '')[:120]}"
            for tool in tools
            if str(tool.get("kind") or "download") == kind
        )

    searchable = listing("search")
    downloadable = listing("download")
    return (
        "You may call tools, writing the identifier as\n"
        "`tu://<tool name>#<a JSON object of that tool's arguments>`. The "
        "arguments must be the tool's own parameter names.\n\n"
        + (
            "Read-only SEARCH tools. Call one when you know the kind of data "
            "you want but not which dataset holds it; the hits come back to "
            "you and you then name datasets to download:\n"
            f"{searchable}\n\n"
            if searchable
            else ""
        )
        + (
            "DOWNLOAD tools. Call one to fetch a specific file (for example "
            "`tu://download_file#{\"url\": \"https://example.org/f.csv\"}`), "
            "or name a repository identifier directly:\n"
            f"{downloadable}\n\n"
            if downloadable
            else ""
        )
    )


def _identifiers_block() -> str:
    return (
        "Use identifiers a downloader can act on directly:\n"
        "  geo://GSE194122            an NCBI GEO series\n"
        "  hf://owner/name            a HuggingFace dataset repository\n"
        "  hf://owner/name#test.csv   one file inside that repository\n"
        "  https://example.org/f.csv  a direct file URL\n"
        "  tu://download_file#{\"url\": \"https://example.org/f.csv\"}   via ToolUniverse\n"
    )


def _run_retrieval(
    prompt: str,
    *,
    client: Any,
    max_items: int,
    log: Callable[[str], None],
    run_search: Callable[[str], str] | None,
    max_hops: int,
) -> list[Proposal]:
    """One conversation: the model looks things up, then names datasets.

    After this many hops it has to answer. Results of the searches it asked for
    are appended to the same prompt, so the next hop sees them.
    """
    seen: set[str] = set()
    proposals: list[Proposal] = []
    found: list[str] = []
    conversation = prompt
    for hop in range(1, max_hops + 1):
        try:
            response = _ask(client, conversation, _DATASETS_SCHEMA, log)
        except Exception as e:  # noqa: BLE001 - surfaced, not swallowed
            _log(log, f"# retrieval: model call failed: {e}")
            if hop == 1:
                # A provider that is down is a hard stop: there is no query to
                # fall back to when the model is the one choosing the datasets.
                raise
            break
        more, searches = _harvest(
            (response or {}).get("datasets"),
            log=log,
            seen=seen,
            max_items=max_items,
            found=len(proposals),
        )
        proposals += more
        if len(proposals) >= max_items or not searches:
            break
        if run_search is None:
            for call in searches:
                _log(
                    log,
                    f"# retrieval: dropped {call!r} (nothing can run a search "
                    f"in this run)",
                )
            break
        _log(
            log,
            f"# retrieval: hop {hop}/{max_hops} — the model asked for "
            f"{len(searches)} search(es)",
        )
        blocks: list[str] = []
        for call in searches[:3]:
            _log(log, f"# retrieval: running {call}")
            try:
                text = run_search(call)
            except Exception as e:  # noqa: BLE001 - a failed search is a finding
                _log(log, f"# retrieval: that search failed: {e}")
                continue
            if str(text).strip():
                blocks.append(f"{call}\n{text}")
            else:
                _log(log, f"# retrieval: {call} found nothing")
        if not blocks:
            break
        found.append("\n\n".join(blocks))
        conversation = (
            f"{prompt}\n\nWhat you have already looked up:\n\n"
            + "\n\n".join(found)
            + "\n\nNow name the datasets to DOWNLOAD, using the identifiers "
            "above where they fit. If you need a different search first, call "
            "a search tool again. Answer with the same JSON shape."
        )
    if not proposals:
        _log(log, "# retrieval: the model proposed no fetchable dataset")
    return proposals[:max_items]


def propose_datasets(
    objective: str,
    *,
    client: Any,
    max_items: int = 4,
    hints: Sequence[str] = (),
    context: str = "",
    tools: Sequence[dict[str, Any]] = (),
    log: Callable[[str], None] = print,
    run_search: Callable[[str], str] | None = None,
    max_hops: int = 3,
    retry: str = "",
) -> list[Proposal]:
    """Let the model find and choose the datasets, in one conversation.

    One round of "the model names datasets" was not enough -- it cannot name a
    dataset it does not know, and it cannot see what a keyword search found --
    and two independent rounds meant the second one knew nothing about the
    first. So the model gets the search tools and a bounded number of hops:

        hop 0   question + hints + tool list
        hop n   the model either asks for another search, or names datasets
                (results of the searches it asked for are in the same prompt)

    Every step prints: what was asked, what came back, and which proposals were
    dropped and why. `log` is injectable so a test can capture the same stream.
    """
    if client is None:
        raise ValueError(
            "retrieval needs the model (it is the component that chooses the "
            "datasets); no client was supplied"
        )
    if client is None:
        raise ValueError(
            "retrieval needs the model (it is the component that chooses the "
            "datasets); no client was supplied"
        )
    hints_line = f"Terms the caller says matter: {', '.join(hints)}\n" if hints else ""
    context_line = f"Context from the experiment protocol:\n{context}\n" if context else ""
    retry_block = (
        "Your previous identifiers were refused. The repositories answered:\n"
        f"{retry}\n\n"
        "Name corrected identifiers that act on those answers -- a file inside "
        "the repository it names, or the platform-specific one it asks for. Do "
        "not repeat an identifier that already failed, and do not name a "
        "container format this analysis cannot read (an `.h5ad`, `.mtx`, or a "
        "`.tar` of them is not a table; say so instead of proposing it again).\n"
        "A URL answered by the site's own 'page not found' document means the "
        "*path* is wrong: every file under that directory fails the same way, so "
        "look the current path up with a search tool instead of changing the "
        "extension. Sources this pipeline can read are CSV, TSV and parquet: a "
        "SAS `.xpt`, a Stata `.dta`, a spreadsheet, or a page of HTML needs to "
        "be named as one of those if a readable copy exists.\n\n"
        if retry
        else ""
    )
    prompt = (
        "You are choosing datasets to DOWNLOAD for a scientific analysis.\n\n"
        f"Research question:\n{objective}\n\n"
        f"{hints_line}{context_line}"
        f"{retry_block}"
        f"{_tool_block(tools)}"
        f"Name at most {max_items} concrete datasets that plausibly contain the "
        "observations this question needs. "
        f"{_identifiers_block()}"
        "Only name a dataset you have real reason to believe exists. A wrong "
        "identifier costs a failed download, so prefer well-known accessions over "
        "plausible-looking ones, and prefer ones whose content you can describe "
        "specifically. Do not name a paper, a model, or software. Before naming a "
        "file inside a repository or series (`hf://owner/name#file.csv`, "
        "`geo://GSE1#suppl/file`), check with a search tool that it exists: the "
        "repository alone is always safe to name, and the fetcher lists what it "
        "holds.\n\n"
        "Answer as JSON with key 'datasets': a list of objects with\n"
        "  'identifier' (as above), 'contains' (what data it holds),\n"
        "  'why' (why it fits this question), 'confidence' (0.0-1.0)."
    )
    _log(log, f"# retrieval: asking the model for candidate datasets "
              f"(max {max_items}, up to {max_hops} search hop(s), "
              f"client={type(client).__name__})")
    return _run_retrieval(
        prompt,
        client=client,
        max_items=max_items,
        log=log,
        run_search=run_search,
        max_hops=max_hops,
    )


def propose_evidence(
    objective: str,
    *,
    columns: Sequence[str],
    gold_reference: str = "",
    sibling_files: Sequence[dict[str, Any]] = (),
    search_hits: str = "",
    exclude: Sequence[str] = (),
    client: Any,
    max_items: int = 3,
    tools: Sequence[dict[str, Any]] = (),
    log: Callable[[str], None] = print,
    run_search: Callable[[str], str] | None = None,
    max_hops: int = 3,
) -> list[Proposal]:
    """Round two: other tables that carry the gold's columns, with other rows.

    This is the search the first round could not do. Until a labeled table is
    chosen there is nothing to aim at: an "external" table has to supply the
    *same measured columns*, and no query can be written for a schema that does
    not exist yet. With the schema in hand the model can look for the same
    measurement protocol on a different cohort, a different cohort file, or a
    differently-named copy of the label -- which this analysis ignores anyway.

    Candidates are handed everything cheap that was already gathered: the
    repository's other files (listed without downloading) and the hits of a
    mechanical search on the gold's own distinctive column names.
    """
    if client is None:
        raise ValueError("the evidence round needs the model to choose datasets")
    columns_line = ", ".join(columns[:40])
    exclude_line = ""
    if exclude:
        listed = "\n".join(f"  {item}" for item in list(exclude)[:40])
        exclude_line = (
            "Already in this run -- the labeled table, and anything already "
            "fetched. Do not name these again, in any spelling, and do not name "
            "a container and then take one of these out of it:\n"
            f"{listed}\n\n"
        )
    siblings = "\n".join(
        f"  {record.get('path')}"
        + (f"  ({record['bytes']:,} bytes)" if record.get("bytes") else "")
        for record in sibling_files[:40]
    )
    prompt = (
        "You are looking for SUPPLEMENTARY data for an analysis that already "
        "has its labeled table.\n\n"
        f"Research question:\n{objective}\n\n"
        f"The labeled table is {gold_reference or 'the primary gold table'}.\n"
        f"Its measured columns are:\n  {columns_line}\n\n"
        f"{exclude_line}"
        + (
            "Other files in the same repository (listed, not downloaded):\n"
            f"{siblings}\n\n"
            if siblings
            else ""
        )
        + (
            "A keyword search on those column names returned:\n"
            f"{search_hits[:4000]}\n\n"
            if search_hits
            else ""
        )
        + f"{_tool_block(tools)}"
        "Name at most "
        f"{max_items} tables that could supply these SAME measurements for "
        "OTHER rows -- another cohort, another site, another split of the same "
        "study. Rules:\n"
        "- the columns have to line up with the list above; a table that measures "
        "something else is of no use here, however relevant it looks;\n"
        "- it does not need a column for the label, and if its label has a "
        "different name that is fine: this analysis ignores it on purpose;\n"
        "- do NOT name a mirror of the labeled table itself (the same rows, "
        "republished). A copy of the same patients adds nothing;\n"
        "- the list under `Already in this run` is exactly that: naming one of "
        "those files again (or a series that holds it, with that file selected) "
        "costs a round trip and adds nothing;\n"
        "- check a file exists before naming it. `hf://owner/name` is always "
        "safe (the fetcher lists what is in it), but `hf://owner/name#train.csv` "
        "is a claim that the repository holds that file, and guessing the usual "
        "`train.csv`/`test.csv` layout for a repository that has one `person.csv` "
        "costs a round trip. Search first, or name the repository.\n"
        f"{_identifiers_block()}"
        "Answer as JSON with key 'datasets': a list of objects with\n"
        "  'identifier' (as above), 'contains' (what data it holds),\n"
        "  'why' (why it fits -- say which columns it shares), "
        "'confidence' (0.0-1.0)."
    )
    _log(
        log,
        f"# supplementary: asking the model for tables with these "
        f"{len(columns)} column(s), max {max_items}, up to {max_hops} hop(s)",
    )
    return _run_retrieval(
        prompt,
        client=client,
        max_items=max_items,
        log=log,
        run_search=run_search,
        max_hops=max_hops,
    )
