"""What a perturbation question needs from the data, as structured fields."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any

#: The modality words a spec uses, and what counts as the same thing.
MODALITIES = ("CRISPRko", "CRISPRi", "CRISPRa", "any")
_MODALITY_WORDS = {
    "CRISPRko": ("crisprko", "knockout", "knock-out", "cas9", "crispr ko"),
    "CRISPRi": ("crispri", "interference", "knockdown", "knock-down", "dcas9-krab", "krab"),
    "CRISPRa": ("crispra", "activation", "overexpression", "vp64", "dcas9-vp"),
}

_SPEC_SCHEMA = {
    "type": "object",
    "properties": {
        "cell_type": {"type": "string"},
        "cell_type_aliases": {"type": "array", "items": {"type": "string"}},
        "organism": {"type": "string"},
        "modality": {"type": "string", "enum": list(MODALITIES)},
        "topic": {"type": "string"},
        "topic_aliases": {"type": "array", "items": {"type": "string"}},
        "assays": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["cell_type", "modality", "topic"],
}


#: Words that say nothing about which study: a "topic" made only of these
#: ("transcriptional response prediction", invented for a question with no
#: topic) turned every topic-anchored query into noise.
_GENERIC_TOPIC_WORDS = frozenset(
    "transcriptional transcription response responses prediction predict predicting "
    "perturbation perturbations gene genes expression knockdown knockout knockouts "
    "screen screens cell cells regulator regulators regulatory factor factors "
    "effect effects change changes profile profiles of the and in to".split()
)


#: Repository search matches ASCII words: `IFN-γ` finds nothing, `IFN-gamma` does.
_GREEK = {"α": "alpha", "β": "beta", "γ": "gamma", "δ": "delta", "κ": "kappa", "λ": "lambda", "μ": "mu", "ω": "omega"}


def ascii_terms(text: str) -> str:
    for letter, word in _GREEK.items():
        text = text.replace(letter, word)
    return text


def specific_topic(topic: str) -> str:
    """The topic, or "" when every word of it is generic."""
    words = [w for w in str(topic or "").lower().replace("-", " ").split() if w]
    return topic.strip() if any(w not in _GENERIC_TOPIC_WORDS for w in words) else ""


@dataclass
class ScreenSpec:
    question: str
    cell_type: str = ""
    cell_type_aliases: list[str] = field(default_factory=list)
    organism: str = "human"
    modality: str = "any"
    topic: str = ""
    #: other ways a study title names the topic: Papalexi's IFN-γ / PD-L1 screen is
    #: titled "...regulation of inhibitory immune checkpoints...", which only a
    #: query on `immune checkpoint` finds
    topic_aliases: list[str] = field(default_factory=list)
    assays: list[str] = field(default_factory=lambda: ["Perturb-seq", "CROP-seq"])

    def topic_terms(self) -> list[str]:
        return [t for t in [self.topic, *self.topic_aliases] if t]

    def cell_type_terms(self) -> list[str]:
        return [term for term in [self.cell_type, *self.cell_type_aliases] if term]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def modality_of(text: str) -> str:
    """The modality a free-text description names, or `any`."""
    lowered = str(text or "").lower()
    for modality, words in _MODALITY_WORDS.items():
        if any(word in lowered for word in words):
            return modality
    return "any"


def modality_compatible(wanted: str, found: str) -> bool:
    return wanted == "any" or found in ("any", "", None) or wanted == found


def extract_spec(question: str, client: Any = None) -> ScreenSpec:
    """The question's cell type, organism, modality, topic and assays.

    One model call; without a model the modality is read from the wording and
    the rest is left empty, which the survey reports rather than guesses.
    """
    spec = ScreenSpec(question=question, modality=modality_of(question))
    if client is None:
        return spec
    prompt = (
        "Read this research question about a genetic perturbation screen and fill "
        "in what the data must be.\n\n"
        f"Question:\n{question}\n\n"
        "- cell_type: the cell line, cell type or tissue the screen must be in "
        "(e.g. `K562`, `Jurkat`, `primary CD4+ T cells`)\n"
        "- cell_type_aliases: other names a repository may use for it (e.g. "
        "`chronic myelogenous leukemia cells` for K562); at most 4\n"
        "- organism: `human` or `mouse`\n"
        "- modality: CRISPRko (knockout), CRISPRi (knockdown/interference), CRISPRa "
        "(activation), or any\n"
        "- topic: the question's most specific biological subject, 1-4 words "
        "(a pathway, process or gene family, e.g. `unfolded protein response`); "
        "empty if the question names none\n"
        "- topic_aliases: up to 3 other ways a study TITLE could name that subject "
        "(broader or narrower terms, e.g. `immune checkpoint` and `PD-L1` for "
        "regulators of PD-L1 expression); ASCII words, no Greek letters\n"
        "- assays: single-cell screen assays that would answer it (e.g. "
        "`Perturb-seq`, `CROP-seq`)\n"
        "Answer as JSON."
    )
    try:
        response = client.generate_structured(
            prompt=prompt, schema=_SPEC_SCHEMA, max_tokens=500, temperature=0
        )
        if isinstance(response, str):
            response = json.loads(response)
    except Exception:  # noqa: BLE001 - the wording-only spec still works
        return spec
    response = response or {}
    spec.cell_type = str(response.get("cell_type") or "").strip()
    spec.cell_type_aliases = [str(a).strip() for a in (response.get("cell_type_aliases") or []) if str(a).strip()][:4]
    spec.organism = str(response.get("organism") or "human").strip().lower() or "human"
    modality = str(response.get("modality") or "").strip()
    spec.modality = modality if modality in MODALITIES else spec.modality
    spec.topic = specific_topic(ascii_terms(str(response.get("topic") or "")))
    spec.topic_aliases = [
        alias
        for alias in (specific_topic(ascii_terms(str(a))) for a in (response.get("topic_aliases") or []))
        if alias and alias.lower() != spec.topic.lower()
    ][:3]
    assays = [str(a).strip() for a in (response.get("assays") or []) if str(a).strip()]
    spec.assays = assays[:3] or spec.assays
    return spec
