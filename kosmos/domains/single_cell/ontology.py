"""What a single-cell question looks like, and what it needs to be answered.

The data pipeline behind this repository -- h5ad/mtx ingestion, per-source
highly-variable-gene selection, library-size normalisation, gene-panel
intersection, the PPI correction -- is a *single-cell* pipeline. Applying it to
a clinical table or a bulk assay is not a small mistake: it changes what the
features mean, and a per-cell split is not a split of patients.

So the modality is a first-class judgement, made here rather than inferred from
the research domain. `biology` is far wider than single cell (clinical records,
bulk transcriptomics, genetics, epidemiology), and a question that names a
cohort of *people* is not a question about *cells*.

Two vocabularies decide it:

  * per-cell language -- cells, cell type, donor, scRNA-seq, h5ad, clusters,
    multiome, "gene expression *per cell*";
  * assays that are explicitly *not* per-cell -- bulk RNA-seq, microarrays,
    tissue-level sequencing -- which win when no per-cell word appears, because
    "gene expression" alone is bulk as often as it is single cell.

With no word either way the declared domain decides: `single_cell` means the
caller said so (and is accepted for thin questions like "cluster these cells"),
`biology` does not, and anything else is not this domain's business.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

#: Language that says the observation is a *cell*.
PER_CELL_WORDS = (
    "single-cell",
    "single cell",
    "singlecell",
    "scrna",
    "sc-rna",
    "sc rna",
    "10x",
    "10x genomics",
    "drop-seq",
    "dropseq",
    "indrop",
    "cite-seq",
    "citeseq",
    "multiome",
    "multioome",
    "h5ad",
    "anndata",
    "cell type",
    "celltype",
    "cell-type",
    "cell types",
    "per cell",
    "per-cell",
    "cell state",
    "cells",
    "cell ",
    "cluster",
    "donor",
    "donors",
    "barcode",
    "umap",
    "hvg",
    "marker gene",
    "pseudobulk",
    "pseudo-bulk",
    "annotate",
    "cellranger",
    "cell ranger",
    "smart-seq",
    "seurat",
    "scanpy",
)
#: Assays that are explicitly not per-cell observations.
BULK_WORDS = (
    "bulk rna",
    "bulk transcriptom",
    "bulk sequencing",
    "bulk tissue",
    "microarray",
    "rnaseq of tissue",
    "tissue-level",
    "tissue level",
    "whole tissue",
    "deseq2",
)

#: Where a single-cell table keeps the things the pipeline reads.
LABEL_COLUMNS = (
    "cell_type",
    "celltype",
    "cell_type_annotation",
    "assigned_cluster",
    "cluster",
    "annotation",
    "condition",
    "compound_1",
    "perturbation",
)
#: Columns that identify a row rather than describe it.
ID_COLUMNS = ("cell_id", "barcode", "donor_id", "donor", "batch", "batch_id", "sample_id")
#: File forms this pipeline can read, and what it does with them.
FILE_FORMATS = {
    ".h5ad": "AnnData: genes in `var`, cells in `obs`; converted to a table on fetch",
    ".h5mu": "MuData: several modalities over the same cells",
    ".mtx": "MatrixMarket count matrix, joined to barcodes/features and any metadata",
}
#: The recipe the pipeline applies, per source, and where it is implemented.
PREPROCESSING = {
    "recipe": [
        "drop genes seen in fewer than `min_cells` cells",
        "select top-n highly variable genes on that source's own counts",
        "library-size normalise each cell",
        "log1p",
        "per-gene z-score within the source",
        "clip negatives to 0",
        "intersect sources on the genes every source measured",
    ],
    "module": "kosmos.ppi.singlecell",
    "applies_when": "the table is per-cell counts, or was converted from .h5ad/.h5mu/.mtx",
}
#: How a single-cell task is judged rather than how it is trained.
EVALUATION = {
    "split": "random rows of the labeled table, unless a test table is given",
    "headline": "per-class recall first: rare types are the reason these questions are asked",
    "supplementary": "other donors/assays/platforms, admitted by the PPI correction or the gradient gate",
    "labels_to_ignore": "cell identifiers, barcodes, batches",
}


@dataclass(frozen=True)
class ModalityVerdict:
    """Is this question about individual cells, and why (or why not)?"""

    single_cell: bool
    rationale: str
    decided_by: str
    per_cell_hits: int = 0
    bulk_hits: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "modality": "single_cell" if self.single_cell else "not_single_cell",
            "modality_rationale": self.rationale,
            "modality_decided_by": self.decided_by,
        }


def _count(text: str, words) -> int:
    return sum(1 for word in words if word in text)


def is_single_cell_question(
    question: str,
    *,
    extra_text: str = "",
    domain: str = "",
    client: Any = None,
) -> ModalityVerdict:
    """Decide whether this question belongs to the single-cell pipeline.

    Words first: per-cell language wins, an explicitly bulk assay loses when no
    per-cell word appears. A tie or silence goes to the model, then to the
    declared domain -- `single_cell` is the caller having already said so, and
    `biology` is not a claim about modality.
    """
    text = f"{question} {extra_text}".lower()
    per_cell = _count(text, PER_CELL_WORDS)
    bulk = _count(text, BULK_WORDS)
    if per_cell > bulk:
        return ModalityVerdict(
            single_cell=True,
            rationale=(
                f"the question is about individual cells ({per_cell} per-cell "
                f"word(s) against {bulk} bulk)"
            ),
            decided_by="words",
            per_cell_hits=per_cell,
            bulk_hits=bulk,
        )
    if bulk > per_cell:
        return ModalityVerdict(
            single_cell=False,
            rationale=(
                f"the question names a bulk assay rather than per-cell "
                f"observations ({bulk} bulk word(s) against {per_cell})"
            ),
            decided_by="words",
            per_cell_hits=per_cell,
            bulk_hits=bulk,
        )
    if client is not None:
        asked = _ask_model(question, extra_text, client)
        if asked is not None:
            return ModalityVerdict(
                single_cell=asked,
                rationale=f"the model read the question and called it single-cell={asked}",
                decided_by="model",
                per_cell_hits=per_cell,
                bulk_hits=bulk,
            )
    declared = str(domain or "").lower().replace("-", "_") in {
        "single_cell",
        "singlecell",
        "single_cell_biology",
    }
    return ModalityVerdict(
        single_cell=declared,
        rationale=(
            "no word decides it, so the declared domain does: "
            + (
                "the caller said single_cell"
                if declared
                else f"{domain or 'no domain'} is not a claim about the observation unit"
            )
        ),
        decided_by="domain",
        per_cell_hits=per_cell,
        bulk_hits=bulk,
    )


_SCHEMA = {
    "type": "object",
    "properties": {
        "single_cell": {"type": "boolean"},
        "reason": {"type": "string"},
    },
    "required": ["single_cell", "reason"],
}


def _ask_model(question: str, extra_text: str, client: Any) -> bool | None:
    """One constrained call, or None when there is no usable answer."""
    prompt = (
        "A researcher asks:\n\n"
        f"{question}\n"
        + (f"\nMore detail given with it:\n{extra_text}\n" if extra_text else "")
        + "\nAre the observations in the data that answers this individual "
        "cells (single-cell / per-cell measurements), or something else "
        "(patients, samples of tissue, bulk assays, simulations)?\n\n"
        "Answer with a JSON object with keys 'single_cell' (boolean) and 'reason'."
    )
    try:
        response = client.generate_structured(
            prompt=prompt, schema=_SCHEMA, max_tokens=200, temperature=0
        )
    except Exception:  # noqa: BLE001 - a failed call is "no answer"
        return None
    if isinstance(response, str):
        try:
            response = json.loads(response)
        except json.JSONDecodeError:
            return None
    value = (response or {}).get("single_cell")
    return value if isinstance(value, bool) else None
