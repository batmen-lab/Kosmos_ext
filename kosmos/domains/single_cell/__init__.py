"""Single-cell domain: per-cell observations, and the pipeline that serves them.

Most of this repository's data machinery is built for one observation unit --
the cell: `.h5ad`/`.mtx` ingestion, per-source HVG selection and
log-normalisation, gene-panel intersection across donors, the PPI correction for
unlabeled cells. This domain makes that explicit, so a question about patients
or bulk tissue is not silently answered with a single-cell pipeline.

The judgement itself is `is_single_cell_question`; the conventions the pipeline
relies on (label columns, id columns, file formats, the recipe, how a task is
evaluated) are recorded beside it so the pipeline and the domain cannot drift
apart.
"""

from kosmos.domains.single_cell.ontology import (
    BULK_WORDS,
    EVALUATION,
    FILE_FORMATS,
    ID_COLUMNS,
    LABEL_COLUMNS,
    PER_CELL_WORDS,
    PREPROCESSING,
    ModalityVerdict,
    is_single_cell_question,
)

__all__ = [
    "ModalityVerdict",
    "is_single_cell_question",
    "PER_CELL_WORDS",
    "BULK_WORDS",
    "LABEL_COLUMNS",
    "ID_COLUMNS",
    "FILE_FORMATS",
    "PREPROCESSING",
    "EVALUATION",
]
