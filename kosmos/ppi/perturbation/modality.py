"""Compatibility shim: the perturbation-type vocabulary lives in the domain.

`kosmos.ppi.perturbation` imports torch (the trainer), so anything the *fetcher*
needs must not live under it. The judgement itself is torch-free and now belongs
to the single-cell domain: `kosmos.domains.single_cell.modality`.
"""

from kosmos.domains.single_cell.modality import (  # noqa: F401
    CONDITION_HINTS,
    MODALITY_LABELS,
    MODALITY_TOKENS,
    check_modality,
    describe,
    infer_requested_modality,
    modality_hits,
    modality_of,
    retrieval_requirement,
)

__all__ = [
    "CONDITION_HINTS",
    "MODALITY_LABELS",
    "MODALITY_TOKENS",
    "check_modality",
    "describe",
    "infer_requested_modality",
    "modality_hits",
    "modality_of",
    "retrieval_requirement",
]
