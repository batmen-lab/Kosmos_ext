"""Does the staged data actually contain what the question asks about?

Kosmos will generate hypotheses against whatever it is handed. When the data
cannot address the question, nothing stops it: it picks the columns it has and
proposes something testable about those instead. That failure is quiet and it
has already happened -- a run asked about nucleotide salvage tested a
fatty-acid metabolite, because that was what sat at the top of the table.

The check here is deliberately modest, for two reasons. It is WARN-ONLY: a
legitimate exploratory run ("what is interesting in this dataset?") has no
question terms to match and must not be blocked. And it is LEXICAL FIRST: if
any term from the question already appears in the data description, the answer
is settled for free and no model is consulted. A model is asked only in the one
case where its judgement adds something -- zero overlap, where the question
might still be answerable through a synonym the string match cannot see.

Every failure path returns `unknown`, which renders as nothing at all. A
relevance check that breaks a run would be worse than the problem it detects.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any, List, Optional

logger = logging.getLogger(__name__)

# Words that appear in nearly every research question and discriminate nothing.
_STOPWORDS = {
    "about", "across", "after", "against", "analysis", "analyse", "analyze",
    "associated", "association", "before", "between", "cause", "causal",
    "compare", "compared", "data", "dataset", "datasets", "determine",
    "difference", "different", "does", "effect", "effects", "examine",
    "from", "identify", "investigate", "levels", "measure", "mechanism",
    "more", "propose", "relationship", "samples", "show", "significant",
    "study", "test", "testing", "than", "that", "their", "these", "this",
    "using", "well", "were", "what", "when", "where", "whether", "which",
    "with", "within", "would",
}

_MAX_TERMS = 24
_HAYSTACK_CHARS = 12_000


@dataclass(frozen=True)
class RelevanceVerdict:
    """One judgement about whether a dataset can address a question."""

    level: str  # relevant | partial | unrelated | unknown
    reason: str = ""
    matched: List[str] = field(default_factory=list)
    unmatched: List[str] = field(default_factory=list)
    decided_by: str = "lexical"

    def render(self) -> str:
        """The warning block to prefix onto the data context, or ''.

        `relevant` and `unknown` render as nothing: the first needs no comment
        and the second has nothing to say. Only a real mismatch earns space in
        the prompt, and when it does it carries an instruction, because naming
        a problem without saying what to do about it invites the model to
        invent a proxy -- which is the failure this exists to prevent.
        """
        if self.level in ("relevant", "unknown"):
            return ""
        absent = ", ".join(self.unmatched[:10]) or "the terms in the question"
        head = (
            "WARNING -- THE DATA MAY NOT ANSWER THE QUESTION AS ASKED."
            if self.level == "unrelated"
            else "NOTE -- THE DATA ONLY PARTLY COVERS THE QUESTION."
        )
        return (
            f"{head} {self.reason}\n"
            f"Not found anywhere in the served variables: {absent}.\n"
            f"Do NOT invent a stand-in for a variable that is absent. Frame "
            f"every hypothesis over what IS measured here, and say plainly in "
            f"each rationale which part of the question this data cannot reach."
        )


def question_terms(question: str) -> List[str]:
    """The content words of a question, as things to look for in the data.

    Two token classes survive: ALL-CAPS runs of two or more characters, which
    is how gene and protein symbols are written (SOD2, IMP, BMI), and ordinary
    words of four or more characters outside the stopword list. The caps rule
    matters -- lowercasing everything first would lose exactly the identifiers
    most likely to appear verbatim as a column name.
    """
    if not question:
        return []
    terms: List[str] = []
    seen = set()
    for token in re.findall(r"[A-Za-z][A-Za-z0-9_\-]*", question):
        if token.isupper() and len(token) >= 2:
            candidate = token
        elif len(token) >= 4 and token.lower() not in _STOPWORDS:
            candidate = token.lower()
        else:
            continue
        if candidate.lower() in seen:
            continue
        seen.add(candidate.lower())
        terms.append(candidate)
        if len(terms) >= _MAX_TERMS:
            break
    return terms


def _normalise(token: str) -> str:
    """One token reduced to the form both sides are compared in.

    Case, hyphens and underscores are dropped, so the question's `IL-6` finds
    the column `IL6` and `TNF_alpha` finds `TNF-alpha`. Without this the check
    warned that a variable was absent while it sat in the very first column.
    """
    return re.sub(r"[^a-z0-9]", "", token.lower())


def _haystack_tokens(haystack: str) -> set:
    """The data context as a SET OF TOKENS, not as one long string.

    Substring matching against the rendered prose made the check useless in
    the direction that matters: a two-letter symbol like AD, MI, ER or EF
    appears inside Kosmos's own boilerplate ("READ THIS FILE WITH", "after"),
    so every dataset matched every such question and nothing was ever flagged.
    Matching whole tokens means a term has to appear as a name, not as three
    letters inside an unrelated word.
    """
    out = set()
    for token in re.findall(r"[A-Za-z0-9_\-]+", haystack or ""):
        norm = _normalise(token)
        if norm:
            out.add(norm)
        # A column called `SOD2_beta` should also answer a question about SOD2.
        for part in re.split(r"[_\-]+", token):
            norm_part = _normalise(part)
            if norm_part:
                out.add(norm_part)
    return out


def assess(
    question: str,
    haystack: str,
    client: Any = None,
    mode: str = "warn",
) -> RelevanceVerdict:
    """Judge whether `haystack` (the data context) can address `question`.

    Never raises. Returns `unknown` for every condition it cannot decide,
    including every failure of the optional model call.
    """
    if mode == "off" or not question or not haystack:
        return RelevanceVerdict("unknown", "not assessed")

    terms = question_terms(question)
    if not terms:
        # A data-driven run with no real question. There is nothing to be
        # irrelevant to.
        return RelevanceVerdict("unknown", "the question names no specific variables")

    tokens = _haystack_tokens(haystack)
    matched = [t for t in terms if _normalise(t) in tokens]
    if matched:
        return RelevanceVerdict(
            "relevant",
            f"{len(matched)} of {len(terms)} question terms appear in the data",
            matched=matched,
            unmatched=[t for t in terms if t not in matched],
        )

    if client is None:
        return RelevanceVerdict(
            "unrelated",
            "none of the question's terms appear among the served variables",
            unmatched=terms,
        )

    # Zero lexical overlap is the only case worth a model call: the question
    # may still be answerable through a synonym ("fibrosis" via "native T1").
    try:
        prompt = (
            "A research question is to be answered using ONLY the dataset "
            "described below. Judge whether the dataset contains variables "
            "that can address it, allowing for synonyms and standard proxies.\n\n"
            f"QUESTION: {question}\n\n"
            f"DATASET:\n{haystack[:_HAYSTACK_CHARS]}\n\n"
            'Reply with JSON only: {"level": "relevant" | "partial" | '
            '"unrelated", "reason": "<one sentence>", "missing": ["<variable '
            'the question needs that is absent>", ...]}'
        )
        response = client.generate(prompt=prompt, max_tokens=300, temperature=0.0)
        text = getattr(response, "content", response) or ""
        match = re.search(r"\{.*\}", text, re.S)
        if not match:
            return RelevanceVerdict("unknown", "the relevance check returned no verdict")
        data = json.loads(match.group(0))
        level = str(data.get("level", "")).strip().lower()
        if level not in ("relevant", "partial", "unrelated"):
            return RelevanceVerdict("unknown", "unrecognised relevance verdict")
        raw_missing = data.get("missing") or []
        # A model that answers "missing": "ejection fraction" instead of a list
        # would otherwise be iterated one CHARACTER at a time into the warning.
        if isinstance(raw_missing, (str, bytes)):
            raw_missing = [raw_missing]
        elif not isinstance(raw_missing, (list, tuple, set)):
            raw_missing = [raw_missing]
        missing = [str(m) for m in raw_missing][:10]
        return RelevanceVerdict(
            level,
            str(data.get("reason", "")).strip() or "judged by the model",
            matched=[],
            unmatched=missing or terms,
            decided_by="model",
        )
    except Exception as e:
        logger.warning(f"Relevance check failed, continuing without it: {e}")
        return RelevanceVerdict("unknown", "the relevance check could not be completed")
