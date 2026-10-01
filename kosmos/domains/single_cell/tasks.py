"""One profile per single-cell data task: what to *find*, and how to judge it.

The data layer grew up around one task -- "find a table with a label column and
predict it". A perturbation question needs a different object: a per-cell
**screen**, with a column naming the perturbation each cell carries, the values
that mean "nothing was perturbed", and the assay. Same sample, different
question, so the fetcher, the review and the target inference all have to know
which question is being asked.

Rather than sprinkling `if task_kind == ...` through those modules, each kind is
one `DataTaskProfile`. Adding a third kind (dose-response, time-course) is a new
instance, not a new branch in five files.

The profiles live in the domain package because these are the domain's
conventions; the implementations they point at stay in `kosmos/ppi`.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

from kosmos.domains.single_cell.modality import CONDITION_HINTS

TaskKindName = Literal["per_cell", "perturbation"]

PER_CELL_ROLE_HELP = """\
gold       -- carries the label this question asks about, at the granularity the
              question asks for, and you would trust it as training labels.
bad_label  -- has a label-like column, but it is NOT the target this question
              wants (different ontology, coarser or finer grouping, a proxy) or
              it is visibly unreliable. Never train on these labels; the rows
              may still be used as unlabeled evidence.
unlabeled  -- no label column for this question at all. Usable as evidence.
unusable   -- not a per-observation table (a matrix, an archive, a listing), or
              it lacks the measurements the question needs.
"""

PERTURBATION_ROLE_HELP = """\
gold       -- a per-cell perturbation screen: one row per cell, a column that \
records which gene(s) that cell's perturbation targeted (a guide, sgRNA, target \
gene or condition value), cells that were not perturbed, AND the transcriptome \
itself (columns named after genes). A metadata table with the perturbations but \
no gene columns is not trainable on its own -- its expression is in another file.
bad_label  -- has a column that looks like a perturbation identity, but the rows \
are not single perturbed cells (a sample-level or summary table), or the column \
is not the perturbation itself. Never train on it.
unlabeled  -- per-cell measurements with no perturbation identity at all (an \
unperturbed atlas). Useful as control cells for the auxiliary graph only.
unusable   -- not a per-cell screen: a genes-by-cells matrix with no condition \
column, a cell-metadata table, an archive, a listing.
"""


def per_cell_schema() -> dict[str, Any]:
    """The label-column schema: what the x->y task has always asked for."""
    return {
        "type": "object",
        "properties": {
            "header_present": {"type": "boolean"},
            "header_names": {
                "type": "array",
                "items": {"type": "string"},
                "description": (
                    "when header_present is false: the real column names, in "
                    "order, if you can name them; otherwise []"
                ),
            },
            "role": {"type": "string", "enum": ["gold", "unlabeled", "bad_label", "unusable"]},
            "target_column": {"type": ["string", "null"]},
            "same_dataset_as": {"type": ["string", "null"]},
            "id_columns": {"type": "array", "items": {"type": "string"}},
            "label_problem": {"type": "string"},
            "reason": {"type": "string"},
            "blocker": {
                "type": "string",
                "enum": ["no_label_column", "not_a_table", "wrong_measurements", "other"],
                "description": (
                    "when the role is unusable, which blocker applies: "
                    "no_label_column (the table is fine but has no column for this "
                    "task's label), not_a_table, wrong_measurements, other. Use an "
                    "empty string when the role is not unusable."
                ),
            },
            "report": {"type": "string"},
            "salvage": {"type": "string"},
            "confidence": {"type": "number"},
        },
        "required": ["header_present", "role", "reason"],
    }


def perturbation_schema() -> dict[str, Any]:
    """The screen schema: a condition column, its controls, and the assay."""
    schema = per_cell_schema()
    schema["properties"].update(
        {
            "is_screen": {
                "type": "boolean",
                "description": (
                    "true only when each row is a single cell and the table records "
                    "which perturbation it carries (or that it is a control). A "
                    "genes-by-cells matrix, a cell metadata table or a sample-level "
                    "table is false."
                ),
            },
            "condition_column": {
                "type": ["string", "null"],
                "description": (
                    "the column holding the perturbation identity (gene target, "
                    "guide, sgRNA, condition); null when the table has none, which "
                    "means it is not a screen."
                ),
            },
            "has_expression": {
                "type": "boolean",
                "description": (
                    "true only when the table also carries the transcriptome -- "
                    "columns named after genes holding counts or normalised values. "
                    "A cell metadata table (barcodes with QC columns such as "
                    "nCount_RNA / nFeature_RNA / percent.mt) has the perturbations "
                    "but not the measurements, and is NOT trainable on its own."
                ),
            },
            "control_labels": {
                "type": "array",
                "items": {"type": "string"},
                "description": (
                    "the values of `condition_column` that mean nothing was "
                    "perturbed, as they appear in the table."
                ),
            },
            "modality": {
                "type": "string",
                "enum": ["crispra", "crispri", "crisprko", "mixed", "unknown"],
                "description": (
                    "how the screen perturbs genes: crisprko = knockout, crispri = "
                    "interference/knockdown, crispra = activation/over-expression, "
                    "mixed, unknown."
                ),
            },
        }
    )
    schema["required"] = ["header_present", "role", "reason", "is_screen", "has_expression"]
    return schema


@dataclass(frozen=True)
class DataTaskProfile:
    """What one kind of single-cell data task asks the data layer for."""

    name: TaskKindName
    roles_help: str
    schema: Callable[[], dict[str, Any]]
    #: Column names the *name-matching* inference should consider for the target.
    #: Empty means "any column that looks like a label" (the per-cell default).
    label_column_hints: tuple[str, ...] = ()
    #: A sentence steered into the retrieval prompt, from the modality asked for.
    retrieval_requirement: Callable[[str], str] | None = None
    #: Kind-specific fields + the mechanical guard, from the model's answer.
    extract: Callable[[dict[str, Any], Sequence[str]], dict[str, Any]] | None = None
    #: Name markers of a file that *carries the labels* (annotated/processed) and
    #: of one that is raw measurements without them. A prior, not a fact: the
    #: names only decide what to keep, the sample decides what it is.
    annotation_markers: tuple[str, ...] = ()
    raw_markers: tuple[str, ...] = ()
    #: Markers of the *measurements* half of a screen (the counts matrix).
    expression_markers: tuple[str, ...] = ()
    #: How many files to unpack from an archive, and how many candidates one
    #: fetch may contribute. A screen ships one archive with a file per sample
    #: (GEO's `RAW.tar`), and the counts matrix is the fifteenth member while the
    #: default cap is twelve -- so the measurements of a screen the model had
    #: already found were never even unpacked. `None` keeps the generic default.
    archive_members: int | None = None
    files_per_fetch: int | None = None

    def review_schema(self) -> dict[str, Any]:
        return self.schema()

    def file_prior(self, name: str) -> int:
        """How promising a file is for this task, from its name alone.

        2 = annotated/processed (a screen's labels usually live here),
        1 = neutral, 0 = raw measurements that carry no labels by themselves.
        """
        text = str(name).lower()
        if any(marker in text for marker in self.annotation_markers):
            return 2
        if any(marker in text for marker in self.raw_markers):
            return 0
        return 1


def _per_cell_extract(response: dict[str, Any], visible: Sequence[str]) -> dict[str, Any]:
    return {}


def _perturbation_extract(response: dict[str, Any], visible: Sequence[str]) -> dict[str, Any]:
    """Read the screen fields, and refuse a "gold" that is not a screen.

    This is the mechanical guard the mode exists for: the model can call a
    metadata table gold, but a table without a perturbation column cannot
    supervise a response model, whatever it looks like.
    """
    condition = response.get("condition_column")
    condition = str(condition) if condition else None
    if condition and condition not in visible:
        condition = None
    control_labels = [
        str(value) for value in (response.get("control_labels") or []) if str(value).strip()
    ]
    modality = str(response.get("modality") or "unknown").strip().lower()
    if modality not in {"crispra", "crispri", "crisprko", "mixed", "unknown"}:
        modality = "unknown"
    is_screen = bool(response.get("is_screen", False))
    has_expression = bool(response.get("has_expression", False))
    return {
        "is_screen": is_screen,
        "has_expression": has_expression,
        "condition_column": condition,
        "control_labels": control_labels,
        "modality": modality,
        # the plan's target is the condition column, so the rest of the pipeline
        # (which speaks in terms of a target column) keeps working
        "target_column": condition,
        # A screen needs all three: a condition column, controls, and the
        # measurements. The metadata table that passed `is_screen` here had the
        # first two and none of the third, so nothing could be trained.
        "force_role": (
            "unusable"
            if not is_screen or not has_expression or condition is None
            else None
        ),
    }


def _perturbation_retrieval_requirement(modality: str) -> str:
    from kosmos.domains.single_cell.modality import retrieval_requirement

    return retrieval_requirement(modality)


PER_CELL = DataTaskProfile(
    name="per_cell",
    roles_help=PER_CELL_ROLE_HELP,
    schema=per_cell_schema,
    extract=_per_cell_extract,
)

#: The markers a *perturbation* run looks for. The gold has to carry the
#: perturbation identity, and a published screen usually keeps it in a sibling
#: annotation (guide / cell metadata / a processed h5ad) rather than the matrix.
PERTURBATION_ANNOTATION_MARKERS = (
    "processed",
    "annotat",
    "metadata",
    "cell_identities",
    "cell_meta",
    "guide",
    "sgrna",
    "grna",
    "condition",
    "obs",
    "h5ad",
    "anndata",
)
#: Markers of a file that carries the *measurements* (the other half of a screen).
PERTURBATION_EXPRESSION_MARKERS = (
    "counts",
    "count_matrix",
    "umi",
    "matrix",
    ".mtx",
    "expression",
    "tpm",
    "lognorm",
    "raw",          # GEO ships the matrices as `GSE..._RAW.tar`
    ".tar",
)

PERTURBATION_RAW_MARKERS = (
    "umi",
    "counts",
    "count_matrix",
    "matrix",
    ".mtx",
    "raw",
    "tpm",
)

PERTURBATION = DataTaskProfile(
    name="perturbation",
    roles_help=PERTURBATION_ROLE_HELP,
    schema=perturbation_schema,
    annotation_markers=PERTURBATION_ANNOTATION_MARKERS,
    raw_markers=PERTURBATION_RAW_MARKERS,
    expression_markers=PERTURBATION_EXPRESSION_MARKERS,
    # A RAW.tar holds the counts matrix deep in the archive (GSM4633614 is the
    # fifteenth of twenty-two files in GSE153056), so the caps have to be wide
    # enough to reach it: unpacking more is cheap, and the screening step is what
    # costs a model call per table.
    archive_members=48,
    files_per_fetch=24,
    label_column_hints=CONDITION_HINTS,
    retrieval_requirement=_perturbation_retrieval_requirement,
    extract=_perturbation_extract,
)

PROFILES: dict[str, DataTaskProfile] = {
    "per_cell": PER_CELL,
    "perturbation": PERTURBATION,
}


def profile_for(task_kind: str | None) -> DataTaskProfile:
    """The profile for a task kind; unknown kinds fall back to the per-cell one."""
    return PROFILES.get(str(task_kind or "per_cell").lower(), PER_CELL)
