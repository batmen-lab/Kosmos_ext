"""Keyword search over the repositories, mechanically.

No model is involved. The query is built by rule from the question, the
repositories are asked with their own search tools through ToolUniverse, and the
answers are adapted into download references. Whatever comes back exists, which
is the property that matters: a lead that turns out not to exist would waste the
whole run.

It is **off by default** in the orchestrator. A rule-built query matches on
single words ("chronic kidney disease" found four commit-chronicle
repositories), and everything it returns is downloaded and reviewed, so the
model is the better retriever. This stays for callers who want a first pass with
no model involved, and as the tool the model's own search calls run through.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

from .config import DataFetcherConfig
from .query import keyword_query
from .search_tools import DEFAULT_REPOSITORIES, REPOSITORIES, Hit, hydrate


@dataclass
class MechanicalSearch:
    query: str
    hits: list[Hit] = field(default_factory=list)
    failures: dict[str, str] = field(default_factory=dict)
    queried: list[str] = field(default_factory=list)
    log: list[str] = field(default_factory=list)

    @property
    def references(self) -> list[str]:
        """The hits that can actually be downloaded, in the order found."""
        seen: dict[str, None] = {}
        for hit in self.hits:
            if hit.reference:
                seen.setdefault(hit.reference, None)
        return list(seen)

    def to_dict(self) -> dict[str, Any]:
        return {
            "query": self.query,
            "queried": self.queried,
            "hits": [hit.to_dict() for hit in self.hits],
            "references": self.references,
            "failures": self.failures,
            "log": self.log,
        }


def mechanical_search(
    question: str,
    *,
    config: DataFetcherConfig | None = None,
    repositories: Sequence[str] | None = None,
    limit: int = 5,
    query: str | None = None,
    downloader=None,
    emit: Callable[[str], None] | None = None,
) -> MechanicalSearch:
    """Search the configured repositories with a rule-built query.

    `downloader` is a `ToolUniverseDownloader`; when the ToolUniverse
    interpreter is not configured, every repository is reported as a failure
    rather than silently returning nothing.
    """
    config = config or DataFetcherConfig.from_env()
    say = emit or (lambda message: None)
    search = MechanicalSearch(query=query or keyword_query(question))
    say(f"# round 1 (mechanical): query={search.query!r}")

    labels = list(repositories or DEFAULT_REPOSITORIES)
    if downloader is None:
        from .tooluniverse import ToolUniverseDownloader, ToolUniverseUnavailable

        try:
            downloader = ToolUniverseDownloader(
                python=config.tooluniverse_python, timeout_s=config.tooluniverse_timeout_s
            )
        except ToolUniverseUnavailable as e:
            search.failures["tooluniverse"] = str(e)
            say(f"# round 1: no ToolUniverse interpreter ({e})")
            return search

    for label in labels:
        spec = REPOSITORIES.get(label)
        if spec is None:
            search.failures[label] = f"unknown repository label {label!r}"
            continue
        arguments: dict[str, Any] = {spec["query_param"]: search.query}
        if spec.get("limit_param"):
            arguments[spec["limit_param"]] = limit
        search.queried.append(label)
        say(f"# round 1: {label} -> {spec['tool']}({arguments})")
        try:
            payload = downloader.search(spec["tool"], arguments)["result"]
        except Exception as e:  # noqa: BLE001 - one repository must not sink the round
            search.failures[label] = f"{type(e).__name__}: {e}"
            say(f"# round 1: {label} failed: {search.failures[label][:160]}")
            continue
        try:
            hits = hydrate(payload, spec["tool"])[:limit]
        except Exception as e:  # noqa: BLE001 - an unrecognised answer is a refusal
            search.failures[label] = f"unreadable answer: {e}"
            say(f"# round 1: {label} answer not understood: {e}")
            continue
        if not hits and len(search.query.split()) > 1:
            # Repository search is name-based and a phrase often matches nothing
            # (probed: the Hub finds `diabetes` and not `clinical measurements
            # predicted diabetes`). One retry with the most specific word, not a
            # rewrite: the phrase's results are used whenever it has any.
            # The first token is the most specific one the query builder found;
            # the longest word is not (it picked `predicted` before).
            single = search.query.split()[0]
            arguments = dict(arguments)
            arguments[spec["query_param"]] = single
            say(f"# round 1: {label} returned nothing for the phrase; retrying {single!r}")
            try:
                payload = downloader.search(spec["tool"], arguments)["result"]
                hits = hydrate(payload, spec["tool"])[:limit]
            except Exception as e:  # noqa: BLE001
                search.failures[label] = f"retry failed: {e}"
                continue
        search.hits.extend(hits)
        fetchable = sum(1 for hit in hits if hit.reference)
        say(
            f"# round 1: {label} -> {len(hits)} hit(s), {fetchable} fetchable"
        )
        for hit in hits:
            say(
                f"#   {hit.accession}  {hit.title[:70]!r}  "
                f"{'-> ' + hit.reference if hit.reference else '(no connector: a lead)'}"
            )
    return search
