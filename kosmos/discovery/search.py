"""Retrieval by the model: it proposes datasets, the fetcher downloads them.

The division of labour is deliberate:

  * **Deciding what to get is a model's job.** It reads the question, names
    concrete dataset identifiers (``geo://GSE<number>``, ``hf://owner/name``, a
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
                f"(not a fetchable identifier; expected e.g. geo://GSE<number>)",
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
        # A placeholder, never a real accession: the model copied the real one
        # that used to stand here into its answers (GSE194122, a bone-marrow
        # atlas, proposed twice as "a CRISPR screen in T cells").
        "Use identifiers a downloader can act on directly (the GEO number below "
        "is a placeholder, not a suggestion):\n"
        "  geo://GSE<number>          an NCBI GEO series\n"
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


_QUERIES_SCHEMA = {
    "type": "object",
    "properties": {
        "topic": {"type": "string"},
        "assay": {"type": "string"},
        "cell_type": {"type": "string"},
        "queries": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["queries"],
}

#: What each search-first purpose asks the model to look for.
_SEARCH_PURPOSES = {
    "gold": (
        "the labeled data this question is about. Its most specific biological "
        "topic (a pathway, process or gene family, e.g. `unfolded protein "
        "response`) is what separates the right study from the many generic ones "
        "on the same cell type: name it as 'topic'"
    ),
    "evidence": (
        "LARGE single-cell datasets of the same cell type or tissue as the "
        "question, with any perturbation or none -- unlabeled evidence for a model "
        "trained on a small labeled screen. The topic does not matter here; the "
        "cell type does: name it as 'cell_type'"
    ),
}


def search_repositories(
    objective: str,
    hints: Sequence[str],
    *,
    client: Any,
    log: Callable[[str], None],
    run_search: Callable[[str], str],
    most: int = 4,
    purpose: str = "gold",
) -> str:
    """Ask the model for repository search queries, run them, return the hits.

    The model names the question's topic, assay and cell type alongside its
    queries, and the code adds the combinations it must not skip: told to keep
    queries short, it dropped the topic (`unfolded protein response`) and the
    generic queries it kept (`Perturb-seq K562 CRISPRi`) never returned the
    study the question is about. Hits are then ranked by how many searches
    returned them, so a study matching several angles rises above one that
    only shares the cell type.
    """
    hint_block = "\n".join(f"  - {hint}" for hint in hints)
    prompt = (
        "Before choosing datasets, search the public repositories (GEO, ArrayExpress, "
        "OmicsDI, HuggingFace).\n\n"
        f"Research question:\n{objective}\n\n"
        + (f"What the data has to be:\n{hint_block}\n\n" if hint_block else "")
        + f"Look for {_SEARCH_PURPOSES.get(purpose, _SEARCH_PURPOSES['gold'])}.\n\n"
        "Answer as JSON with:\n"
        "  'topic'     the question's most specific biological subject, 1-4 words\n"
        "  'assay'     the measurement, e.g. `Perturb-seq`, `CROP-seq`, `scRNA-seq`\n"
        "  'cell_type' the cell line, cell type or tissue, e.g. `K562`\n"
        f"  'queries'   2-{most} keyword queries worded like a repository title, "
        "3-6 words each, no sentences or boolean syntax. Repository search "
        "matches every word, so do not pile up qualifiers -- but never drop the "
        "words that make this question specific."
    )
    try:
        response = _ask(client, prompt, _QUERIES_SCHEMA, log)
    except Exception as e:  # noqa: BLE001 - a failed step falls back to the plain prompt
        _log(log, f"# retrieval: the search-first step failed ({e}); continuing without it")
        return ""
    response = response or {}
    topic = str(response.get("topic") or "").strip()
    assay = str(response.get("assay") or "").strip()
    cell_type = str(response.get("cell_type") or "").strip()
    queries = [str(q).strip() for q in (response.get("queries") or []) if str(q).strip()][:most]
    # The anchors the model's own queries may have dropped.
    anchors = (
        [f"{topic} {assay}", f"{topic} {cell_type}"]
        if purpose == "gold" and topic
        else [f"{cell_type} {assay}", f"{cell_type} single-cell"]
        if purpose == "evidence" and cell_type
        else []
    )
    for anchor in anchors:
        anchor = " ".join(anchor.split())
        if anchor and not any(anchor.lower() == q.lower() for q in queries):
            queries.append(anchor)
    if topic or cell_type:
        _log(log, f"# retrieval: topic {topic!r}, assay {assay!r}, cell type {cell_type!r}")
    blocks: list[str] = []
    seen_in: dict[str, list[str]] = {}
    titles: dict[str, str] = {}
    for query in queries:
        _log(log, f"# retrieval: search first: {query!r}")
        try:
            text = run_search("tu://search#" + json.dumps({"query": query}))
        except Exception as e:  # noqa: BLE001 - one failed search is not fatal
            _log(log, f"# retrieval: that search failed: {e}")
            continue
        if not str(text).strip():
            continue
        blocks.append(f"search {query!r}:\n{text}")
        for line in str(text).splitlines():
            parts = line.strip().split(None, 1)
            if not parts:
                continue
            accession = parts[0]
            seen_in.setdefault(accession, [])
            if query not in seen_in[accession]:
                seen_in[accession].append(query)
            titles.setdefault(accession, line.strip())
    # A title that names the topic outranks one that only matched the cell
    # type: HuggingFace returns the same K562 repositories for every query that
    # says "K562", which tied them with the UPR screen on count alone.
    key_words = [
        word
        for word in re.findall(r"[a-z0-9]+", (topic if purpose == "gold" else cell_type).lower())
        if len(word) > 3 or word.isupper()
    ]

    def on_topic(accession: str) -> bool:
        title = titles[accession].lower()
        # most of the words, not one: `unfolded-veil` is not the unfolded protein response
        return bool(key_words) and sum(word in title for word in key_words) >= max(1, len(key_words) - 1)

    ranked = sorted(
        (accession for accession in seen_in if len(seen_in[accession]) > 1 or on_topic(accession)),
        key=lambda accession: (not on_topic(accession), -len(seen_in[accession])),
    )
    summary = ""
    if ranked:
        summary = (
            "Strongest candidates -- titles naming the "
            + ("topic" if purpose == "gold" else "cell type")
            + " first, then by how many searches returned them:\n"
            + "\n".join(
                f"  [{len(seen_in[a])} searches{', on topic' if on_topic(a) else ''}] {titles[a]}"
                for a in ranked[:12]
            )
            + "\n\n"
        )
    return summary + "\n\n".join(blocks)


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
    search_first: bool = False,
) -> list[Proposal]:
    """Let the model find and choose the datasets, in one conversation.

    `search_first` (with `run_search`): before naming anything, the model writes
    search queries, they are run, and it names datasets from the hits. Offered
    the search tools as an option, it never used them and named accessions from
    memory -- three T-cell runs proposed COVID atlases as "CRISPR screens".

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
        + (
            "A search tool is available: look a series up before naming it. An "
            "accession recalled from memory is often a different study with a "
            "similar topic, and the run then downloads the wrong data.\n\n"
            if run_search is not None
            else ""
        )
        + "Answer as JSON with key 'datasets': a list of objects with\n"
        "  'identifier' (as above), 'contains' (what data it holds),\n"
        "  'why' (why it fits this question), 'confidence' (0.0-1.0)."
    )
    _log(log, f"# retrieval: asking the model for candidate datasets "
              f"(max {max_items}, up to {max_hops} search hop(s), "
              f"client={type(client).__name__})")
    if search_first and run_search is not None:
        looked_up = search_repositories(objective, hints, client=client, log=log, run_search=run_search)
        if looked_up:
            prompt = (
                f"{prompt}\n\nRepository search results for this question (run before "
                "you answer; prefer accessions from these hits, and do not name an "
                "accession that is not in them unless you are certain of it):\n\n"
                f"{looked_up}"
            )
    return _run_retrieval(
        prompt,
        client=client,
        max_items=max_items,
        log=log,
        run_search=run_search,
        max_hops=max_hops,
    )


#: The `--supp-size large` rule, shared by the evidence proposal and preflight.
PREFER_LARGE_EVIDENCE = (
    "this analysis is set up as a small labeled table plus as much unlabeled evidence as possible, so among sources that fit, prefer the one with the MOST rows (cells) -- a large atlas or a large screen of the same cell type beats a small one; relevance (same cell type, same measurement) still comes first"
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
    prefer_large: bool = False,
) -> list[Proposal]:
    """Round two: other tables that carry the gold's columns, with other rows.

    `prefer_large` asks for the sources with the most rows (`--supp-size large`).

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
            "Repository searches for this evidence returned:\n"
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
        + (f"- {PREFER_LARGE_EVIDENCE};\n" if prefer_large else "")
        + f"{_identifiers_block()}"
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
