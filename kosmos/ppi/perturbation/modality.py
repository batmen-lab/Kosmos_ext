"""Which *kind* of perturbation the data provides, and which the question asked for.

A perturbation screen is not just "genes and a condition column": the same gene
can be over-expressed (CRISPRa), knocked down (CRISPRi) or knocked out
(CRISPRko), and those are different experiments with different expected signs.
The rest of the perturbation pipeline is modality-blind -- it learns `Delta`
either way -- so nothing stopped a "knock out CBL" question from being answered
with Norman's CRISPRa screen until this module existed.

Three judgements, deliberately kept as plain word-scanning (no model call):

  * `modality_of(text)`            -- normalise a screen's own description, e.g.
    the registry's `PerturbationSource.perturbation` ("CRISPRa (sgRNA)").
  * `infer_requested_modality(text)` -- what the *question* asked for, recorded
    with how it was decided, the way `task_ontology` records its kind.
  * `check_modality(requested, provided)` -- do the two agree? A mismatch is the
    finding this module exists to surface.

The vocabulary is intentionally explicit and conservative: a bare "CRISPR" or a
bare "knock" decides nothing, because guessing a modality is worse than saying
"unknown".
"""

from __future__ import annotations

import re
from typing import Any

#: Normalised tags, and what they mean in words.
MODALITY_LABELS: dict[str, str] = {
    "crispra": "CRISPR activation / over-expression",
    "crispri": "CRISPR interference / knockdown (incl. shRNA/siRNA/RNAi)",
    "crisprko": "CRISPR knockout / gene deletion",
    "mixed": "more than one modality in the same screen",
    "unknown": "not determined",
}

#: Surface forms per modality, matched after lowercasing and turning `-`/`_`
#: into spaces. Tokens of 5+ characters are matched as word *prefixes* (so
#: `overexpress` covers `overexpression`/`overexpressing`); shorter ones must be
#: whole words, so `rnai` cannot fire inside a longer token.
MODALITY_TOKENS: dict[str, tuple[str, ...]] = {
    "crispra": (
        "crispra",
        "crispr activation",
        "activation screen",
        "overexpress",
        "over express",       # after `-`/`_` normalisation: "over-expressing"
        "gain of function",
        "sun tag",
    ),
    "crispri": (
        "crispri",
        "crispr interference",
        "interference",
        "knockdown",
        "knock down",
        "knocking down",
        "knocked down",
        "silencing",
        "gene silencing",
        "shrna",
        "sirna",
        "rnai",
        "rna interference",
        "crispr off",
        "crisproff",
        "repression",
    ),
    "crisprko": (
        "crisprko",
        "knockout",
        "knock out",
        "knocking out",
        "knocked out",
        "gene deletion",
        "deletion",
        "loss of function",
    ),
}

#: Short tokens that must match as whole words (prefix matching would over-fire).
_WHOLE_WORD = {"rnai", "shrna", "sirna"}


def _normalise(text: str) -> str:
    return re.sub(r"[-_/]", " ", str(text or "").lower())


def _count(text: str, token: str) -> int:
    if len(token) >= 5 and token not in _WHOLE_WORD:
        pattern = rf"(?<![a-z0-9]){re.escape(token)}"
    else:
        pattern = rf"(?<![a-z0-9]){re.escape(token)}(?![a-z0-9])"
    return len(re.findall(pattern, text))


def modality_hits(text: str) -> dict[str, int]:
    """How many times each modality's vocabulary appears, per normalised tag."""
    normalised = _normalise(text)
    return {
        modality: sum(_count(normalised, token) for token in tokens)
        for modality, tokens in MODALITY_TOKENS.items()
    }


def modality_of(text: str) -> str:
    """Normalise a screen/assay description to a tag.

    `"CRISPRa (sgRNA)"` -> `crispra`; `"CRISPRi / CRISPRa"` -> `mixed`; anything
    that names no modality -> `unknown`.
    """
    present = {modality for modality, count in modality_hits(text).items() if count}
    if not present:
        return "unknown"
    if len(present) == 1:
        return present.pop()
    return "mixed"


def infer_requested_modality(text: str) -> dict[str, Any]:
    """What modality the question asked for, and how that was decided."""
    hits = modality_hits(text)
    present = sorted(modality for modality, count in hits.items() if count)
    if not present:
        modality, decided_by = "unknown", "none"
    elif len(present) == 1:
        modality, decided_by = present[0], "words"
    else:
        modality, decided_by = "mixed", "words"
    total = sum(hits.values())
    rationale = (
        f"the question names no perturbation modality"
        if decided_by == "none"
        else f"the question's words name {modality!r} ({total} hit(s): "
        + ", ".join(f"{name} x{hits[name]}" for name in present)
        + ")"
    )
    return {
        "modality": modality,
        "decided_by": decided_by,
        "rationale": rationale,
        "hits": hits,
    }


def check_modality(requested: str, provided: str) -> dict[str, Any]:
    """Do the requested and provided modalities agree?

    Statuses:
      * `match`      -- same tag
      * `compatible` -- the screen is `mixed` (it covers both), or one side
                        names several modalities and the other is among them
      * `mismatch`   -- two concrete, different modalities
      * `unchecked`  -- one side is `unknown`, so no judgement is possible
    """
    requested = str(requested or "unknown")
    provided = str(provided or "unknown")
    if requested == "unknown" or provided == "unknown":
        status = "unchecked"
        message = (
            "perturbation modality was not compared: "
            + (
                "the question does not name one"
                if requested == "unknown"
                else "the data does not declare one"
            )
        )
    elif requested == provided:
        status = "match"
        message = f"the question and the data agree: {provided}"
    elif provided == "mixed":
        status = "compatible"
        message = (
            f"the question asks for {requested} and the screen publishes more "
            f"than one modality, so it may cover it"
        )
    elif requested == "mixed":
        status = "compatible"
        message = (
            f"the question names more than one modality and the data provides "
            f"{provided}"
        )
    else:
        status = "mismatch"
        message = (
            f"the question asks for {requested} "
            f"({MODALITY_LABELS.get(requested, requested)}) but the data provides "
            f"{provided} ({MODALITY_LABELS.get(provided, provided)})"
        )
    return {"status": status, "message": message, "requested": requested, "provided": provided}


def describe(modality: str) -> str:
    return MODALITY_LABELS.get(str(modality or "unknown"), str(modality))

def retrieval_requirement(modality: str) -> str:
    """What the retrieval stage must look for, given the question's modality.

    A screen is not interchangeable with one of a different assay: a knockout
    question answered on an over-expression screen measures the wrong thing. The
    sentence below is carried into the fetcher's intent so the search itself is
    steered, rather than the mistake being caught only afterwards.
    """
    modality = str(modality or "unknown")
    if modality in ("unknown", "mixed"):
        return (
            "the labeled screen must be a single-cell perturbation screen of the "
            "assay the question names (activation / interference / knockout are "
            "not interchangeable); say which assay a candidate screen used."
        )
    label = MODALITY_LABELS.get(modality, modality)
    return (
        f"the labeled screen must be a {label} screen: prefer a dataset that "
        f"perturbs genes by {modality}, and do not substitute a screen of a "
        f"different assay (activation, interference and knockout are not "
        f"interchangeable). State the assay each candidate screen used."
    )
