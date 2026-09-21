"""Turning a research question into a search term, without a model.

Round one of retrieval is deliberately mechanical: it must be reproducible, it
must not spend tokens, and it must never invent a dataset that does not exist.
So the query is built by rule from the question's own words -- the content words,
ranked by how specific they are -- and the search itself is performed by the
repository tools. Whatever comes back exists; what it is for is decided later.

The rules are the same as any keyword query: drop the words that every question
has ("does", "improve", "predict", "dataset"), keep the ones that name a
measurement, an organism, a tissue or an assay.
"""

from __future__ import annotations

import re
from collections.abc import Iterable

STOPWORDS = {
    "a", "an", "the", "does", "do", "is", "are", "was", "were", "can", "could",
    "would", "should", "will", "improve", "improves", "improving", "add", "adding",
    "use", "using", "used", "help", "helps", "better", "worse", "than", "versus",
    "vs", "compare", "comparison", "whether", "when", "what", "which", "how",
    "why", "and", "or", "of", "for", "from", "with", "without", "in", "on", "to",
    "by", "as", "at", "be", "been", "being", "it", "its", "this", "that", "these",
    "those", "we", "our", "my", "their", "there", "here", "also", "more", "most",
    "during", "after", "before", "between", "among", "across", "under", "over",
    "through", "via", "per", "along", "while", "where", "into", "onto", "upon",
    "within", "toward", "towards", "if", "then", "such", "may", "might", "must",
    "less", "research", "question", "study", "analysis", "predict", "predicting",
    "prediction", "classify", "classifying", "classification", "model", "models",
    "training", "train", "test", "testing", "data", "dataset", "datasets", "set",
    "label", "labels", "labeled", "labelled", "unlabeled", "unlabelled", "pseudo",
    "loss", "result", "results", "report", "improved", "quality",
    # Generic scientific vocabulary: every question has these, so they match
    # every dataset name and narrow nothing.
    "measurement", "measurements", "measure", "measured", "variable", "variables",
    "feature", "features", "value", "values", "sample", "samples", "patient",
    "patients", "cohort", "cohorts", "subject", "subjects", "clinical", "routine",
    "status", "outcome", "outcomes", "condition", "conditions", "group", "groups",
    "level", "levels", "effect", "effects", "association", "associated", "related",
    "correlation", "difference", "differences", "change", "changes", "including",
    "based", "given", "present", "absent", "different", "other", "another", "same",
    "first", "second", "single", "multiple", "various", "several", "large", "small",
    "each", "every", "any", "all", "some",
    "high", "low", "normal", "abnormal", "increase", "decrease", "expert", "predictor", "predictors", "target", "input", "inputs", "output",
    "experiment", "experiments", "task", "tasks", "method", "methods", "approach",
    # Words that look specific because they are long, but name nothing: every
    # question about a measurement mentions the environment it was taken in and
    # the parameters that were varied. Ranking by length put these first and
    # dropped "spin coating" -- the terms the question is actually about.
    "environmental", "environment", "parameter", "parameters", "factor",
    "factors", "affect", "affects", "affected", "affecting", "determine",
    "determines", "determining", "influence", "influences", "impact", "impacts",
    "relationship", "relationships", "relation", "relations", "assess",
    "assessing", "evaluate", "evaluating", "investigate", "investigating",
    "examine", "examining", "quantify", "quantifying",
    "contribution", "contributions", "role", "roles",
}

_WORD = re.compile(r"[A-Za-z][A-Za-z0-9._+-]*")
_EDGE = re.compile(r"^[^A-Za-z0-9]+|[^A-Za-z0-9]+$")
_PUNCTUATION = set(",;:()[]{}<>\"'")


def words(text: str) -> list[str]:
    return _WORD.findall(text or "")


def _clean(word: str) -> str:
    """A token without the punctuation it was written next to.

    `efficiency.` at the end of a sentence used to reach the query with its
    period attached, which no repository search matches.
    """
    return _EDGE.sub("", str(word))


def _is_generic(word: str) -> bool:
    """Is this a word every question has?

    Checked against the word and its crude stems, because enumerating every
    inflection is a losing game: `predict`, `predicts`, `predicted`,
    `predicting` and `prediction` are one entry plus four suffixes, and the
    first version of this let `predicted` into the query it was meant to keep
    out.
    """
    lowered = word.lower()
    candidates = {lowered}
    for suffix in ("ing", "ed", "es", "s"):
        if lowered.endswith(suffix) and len(lowered) > len(suffix) + 2:
            candidates.add(lowered[: -len(suffix)])
    return bool(candidates & STOPWORDS)


def keyword_query(question: str, extra_terms: Iterable[str] = (), limit: int = 4) -> str:
    """The deterministically chosen search term for a question.

    Two words that were written next to each other ("spin coating") name
    something more specific than either word alone, so they are kept as one
    term and ranked above single words. Words joined by punctuation are not:
    "(glucose, blood pressure, insulin, BMI)" is a list, and gluing its items
    together produced phrases like "glucose blood".
    """
    terms = [str(t).strip() for t in extra_terms if str(t).strip()]
    if terms:
        return " ".join(terms[:limit])

    kept: list[tuple[int, int, str]] = []
    seen: set[str] = set()
    for match in _WORD.finditer(question or ""):
        token = match.group(0)
        if _is_generic(token):
            continue
        cleaned = _clean(token)
        if not cleaned or cleaned.lower() in seen:
            continue
        seen.add(cleaned.lower())
        kept.append((match.start(), match.end(), cleaned))

    phrases: list[str] = []
    for (_, end, first), (start, _, second) in zip(kept, kept[1:], strict=False):
        gap = (question or "")[end:start]
        if gap and not gap.strip() and not (_PUNCTUATION & set(gap)):
            phrases.append(f"{first} {second}")

    ranked = sorted(phrases, key=lambda p: (-len(p), p.lower()))
    ranked += sorted((token for _, _, token in kept), key=lambda w: (-len(w), w.lower()))
    chosen: list[str] = []
    used: set[str] = set()
    for term in ranked:
        if len(chosen) >= limit:
            break
        # A term whose words are already carried by a chosen phrase adds nothing
        # to the query, and makes it redundant ("solar-cell efficiency
        # perovskite solar-cell" is one idea written twice).
        words_in_term = set(term.split())
        if words_in_term & used:
            continue
        chosen.append(term)
        used |= words_in_term
    return " ".join(chosen)
