"""Two decisions that used to be a human's: what to search for, and what to fetch.

    research question --propose_intents--> 1-3 keyword queries
    search hits       --rank_candidates--> the ones worth downloading, with reasons

Both are LLM-assisted and both fall back to a deterministic rule when no model is
available or the model answers something unusable. The fallback is not a
formality: a sweep must still run when the provider is down, and a ranking that
silently became empty would look like "no data exists".

The model is *constrained* the way `task_inference` constrains it: it may only
choose among the accessions it was shown, and inventing one is discarded rather
than repaired. Both functions return plain dataclasses, so the caller (a script
that may also shell out to the fetcher) needs no shared types with either side.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any

#: Words that make a query worse in a repository that matches on dataset names.
_STOPWORDS = {
    "a", "an", "the", "does", "do", "is", "are", "can", "could", "would", "should",
    "improve", "improves", "improving", "adding", "add", "use", "using", "used",
    "help", "helps", "better", "worse", "than", "versus", "vs", "compare",
    "comparison", "whether", "when", "what", "which", "how", "why", "and", "or",
    "of", "for", "from", "with", "without", "in", "on", "to", "by", "as", "at",
    "be", "been", "being", "it", "its", "this", "that", "these", "those", "we",
    "our", "my", "their", "there", "here", "also", "more", "most", "less",
    "objective", "objective:", "research", "question", "study", "analysis",
    "predict", "predicting", "prediction", "classify", "classifying",
    "classification", "model", "models", "training", "train", "test", "testing",
    "data", "dataset", "datasets", "labels", "labeled", "labelled", "unlabeled",
    "unlabelled", "pseudo", "loss", "results", "report",
}
_WORD = re.compile(r"[A-Za-z][A-Za-z0-9._+-]*")
MAX_QUERY_WORDS = 5
MAX_QUERY_CHARS = 60


def _words(text: str) -> list[str]:
    return _WORD.findall(text or "")


def usable_queries(queries: Iterable[str], *, max_words: int = MAX_QUERY_WORDS) -> list[str]:
    """Keep the queries a repository can actually match on.

    Short, keyword-shaped, de-duplicated: a repository searches dataset names and
    tags, so a sentence is the wrong unit and a six-word question tail is noise.
    A query that survives here is still just a string -- this validates shape, it
    does not promise the search will return anything.
    """
    kept: list[str] = []
    seen: set[str] = set()
    for raw in queries:
        query = " ".join(_words(str(raw))).strip()
        if not query:
            continue
        words = query.split()
        if len(words) > max_words:
            query = " ".join(words[:max_words])
        content = [w for w in query.split() if w.lower() not in _STOPWORDS]
        if not content:
            continue
        query = " ".join(content)[:MAX_QUERY_CHARS].strip()
        key = query.lower()
        if key and key not in seen:
            seen.add(key)
            kept.append(query)
    return kept


def keyword_query(objective: str, extra_terms: Sequence[str] = ()) -> str:
    """A deterministic query built from the question's content words."""
    terms = [t for t in extra_terms if t]
    if terms:
        return " ".join(usable_queries([" ".join(terms)]) or terms[:3])
    words = [w for w in _words(objective) if w.lower() not in _STOPWORDS]
    # Longest words are the most specific ones in a sentence like this one.
    ranked = sorted(dict.fromkeys(words), key=lambda w: (-len(w), w.lower()))
    return " ".join(usable_queries([" ".join(ranked[:3])]) or ranked[:3])


@dataclass
class Intent:
    query: str
    why: str = ""
    source: str = "llm"  # llm | fallback | human


@dataclass
class Candidate:
    accession: str
    repository: str = ""
    title: str = ""
    reference: str | None = None
    score: float = 0.0
    reason: str = ""
    source: str = "llm"  # llm | fallback
    raw: dict[str, Any] = field(default_factory=dict)


def propose_intents(
    objective: str,
    *,
    client: Any = None,
    max_intents: int = 3,
    extra_terms: Sequence[str] = (),
    hints: Sequence[str] = (),
) -> list[Intent]:
    """1-3 search queries for a research question, from the model or from rules.

    `hints` are terms the caller believes matter (a protocol's required dataset,
    a domain word). They are passed to the model *and* used by the fallback,
    because a hint is a fact the question may not state.
    """
    fallback = keyword_query(objective, [*hints, *extra_terms])
    if client is None:
        return [Intent(query=fallback, why="no model available; content words of the question",
                       source="fallback")] if fallback else []

    schema = {
        "type": "object",
        "properties": {
            "intents": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {"query": {"type": "string"}, "why": {"type": "string"}},
                    "required": ["query", "why"],
                },
            }
        },
        "required": ["intents"],
    }
    hint_line = f"\nTerms that matter here: {', '.join(hints)}\n" if hints else ""
    prompt = (
        "A scientist wants data for this research question:\n\n"
        f"{objective}\n"
        f"{hint_line}\n"
        "Propose at most "
        f"{max_intents} SHORT search queries (one to four words each) for a dataset "
        "repository. Repositories match queries against dataset NAMES and TAGS, not "
        "against sentences, so name the measurement, the tissue, the organism or the "
        "assay -- never the analysis. Do not use words like improve, does, predict, "
        "prediction, model, classification: those match no dataset name. Include at "
        "least one query that is a single distinctive word, because some repositories "
        "return nothing at all for a phrase.\n\n"
        "Answer as JSON with key 'intents': a list of objects with 'query' and 'why' "
        "(one sentence explaining what the query should find)."
    )
    try:
        response = client.generate_structured(
            prompt=prompt, schema=schema, max_tokens=600, temperature=0
        )
    except Exception as e:  # noqa: BLE001 - a failed call is "no suggestion"
        return [Intent(query=fallback, why=f"model call failed: {e}", source="fallback")]
    if isinstance(response, str):
        try:
            response = json.loads(response)
        except json.JSONDecodeError:
            response = None

    proposed = (response or {}).get("intents") or []
    queries = usable_queries(
        [item.get("query", "") for item in proposed if isinstance(item, dict)]
    )[:max_intents]
    reasons = {
        item.get("query", ""): item.get("why", "")
        for item in proposed
        if isinstance(item, dict)
    }
    if not queries:
        return [
            Intent(
                query=fallback,
                why="the model proposed nothing usable; content words of the question",
                source="fallback",
            )
        ] if fallback else []
    return [Intent(query=q, why=reasons.get(q, ""), source="llm") for q in queries]


def score_by_keywords(objective: str, candidates: Sequence[Candidate]) -> list[Candidate]:
    """Deterministic relevance: how many question words a candidate's text carries."""
    words = {w.lower() for w in _words(objective) if w.lower() not in _STOPWORDS}
    scored = []
    for candidate in candidates:
        text = " ".join(
            [candidate.accession, candidate.repository, candidate.title or ""]
        ).lower()
        hits = {w for w in words if w in text}
        score = (len(hits) / len(words)) if words else 0.0
        if candidate.reference:
            score += 0.05  # routable: it can be downloaded, not just looked at
        scored.append(
            Candidate(
                accession=candidate.accession,
                repository=candidate.repository,
                title=candidate.title,
                reference=candidate.reference,
                score=min(score, 1.0),
                reason=(
                    f"shares {sorted(hits)} with the question"
                    if hits
                    else "no keyword overlap with the question"
                ),
                source="fallback",
                raw=candidate.raw,
            )
        )
    return sorted(scored, key=lambda c: (-c.score, c.accession))


def rank_candidates(
    objective: str,
    candidates: Sequence[Candidate],
    *,
    client: Any = None,
    top_k: int = 3,
) -> list[Candidate]:
    """Which search hits are worth downloading, best first.

    Only hits that carry a `reference` can be fetched; the rest are leads a human
    must route, and they are ranked but never selected for download.
    """
    candidates = list(candidates)
    if not candidates:
        return []
    fallback = score_by_keywords(objective, candidates)
    if client is None:
        return fallback[:top_k]

    routable = [c for c in candidates if c.reference]
    if not routable:
        return fallback[:top_k]
    listing = "\n".join(
        f"- {c.accession} ({c.repository}) {c.title or ''}"[:200] for c in routable
    )
    schema = {
        "type": "object",
        "properties": {
            "ranking": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "accession": {"type": "string"},
                        "relevant": {"type": "boolean"},
                        "why": {"type": "string"},
                    },
                    "required": ["accession", "relevant", "why"],
                },
            }
        },
        "required": ["ranking"],
    }
    prompt = (
        "A scientist wants data for this research question:\n\n"
        f"{objective}\n\n"
        "These datasets were found. Which would plausibly contain data useful for "
        "that question -- measured observations of the right kind, not a paper, a "
        "model or a software package?\n\n"
        f"{listing}\n\n"
        "Answer as JSON with key 'ranking': the accessions above that are worth "
        "looking at, most useful first, each with 'relevant' and a one-sentence "
        "'why'. Use only the accessions listed."
    )
    try:
        response = client.generate_structured(
            prompt=prompt, schema=schema, max_tokens=800, temperature=0
        )
    except Exception as e:  # noqa: BLE001
        fallback[0].reason = f"{fallback[0].reason} (model call failed: {e})"
        return fallback[:top_k]
    if isinstance(response, str):
        try:
            response = json.loads(response)
        except json.JSONDecodeError:
            response = None

    by_accession = {c.accession: c for c in routable}
    ranked: list[Candidate] = []
    for rank, item in enumerate((response or {}).get("ranking") or []):
        if not isinstance(item, dict):
            continue
        chosen = by_accession.get(str(item.get("accession")))
        if chosen is None:
            # A name the model invented is a signal, not a typo to repair.
            continue
        if not item.get("relevant", True):
            continue
        ranked.append(
            Candidate(
                accession=chosen.accession,
                repository=chosen.repository,
                title=chosen.title,
                reference=chosen.reference,
                score=max(0.1, 1.0 - rank * 0.1),
                reason=str(item.get("why") or "chosen by the model"),
                source="llm",
                raw=chosen.raw,
            )
        )
    if not ranked:
        fallback[0].reason = f"{fallback[0].reason} (model ranked nothing usable)"
        return fallback[:top_k]
    # Keep the model's order, then append rule-ranked routable hits it did not name.
    named = {c.accession for c in ranked}
    ranked.extend(c for c in fallback if c.reference and c.accession not in named)
    return ranked[:top_k]
