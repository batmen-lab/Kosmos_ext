"""Review a fetched table's first rows: what is it, and what role can it play.

The mechanical rules answer "is the target column present and are the features
numeric". They cannot answer four things that decide whether that arithmetic
means anything:

  * is the first line a header, or is it data (the header then gets eaten and
    every column is named after a value);
  * is the label here under a different name (`Outcome` for a question about
    diabetes);
  * which columns are identifiers, timestamps or free text rather than
    measurements;
  * are the labels present but unusable -- a different ontology, a coarser
    grouping, or visibly unreliable -- which makes the table evidence rather
    than gold.

So the model gets a bounded packet (raw first lines plus how we parsed them) and
returns a constrained verdict. **The verdict proposes; the mechanical layer
still decides.** A table it calls `gold` without the target column is refused,
and one it calls `bad_label` is only ever used as evidence, never as labels --
the model can take a table out of the training set but cannot put one in.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, field
from typing import Any, Literal

Role = Literal["gold", "unlabeled", "bad_label", "unusable"]

from kosmos.domains.single_cell.tasks import profile_for

ROLE_HELP = """\
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


@dataclass
class TableReview:
    path: str
    role: Role | None
    header_present: bool = True
    header_names: list[str] = field(default_factory=list)
    same_dataset_as: str | None = None
    target_column: str | None = None
    id_columns: list[str] = field(default_factory=list)
    label_problem: str = ""
    reason: str = ""
    #: Why the model refused the table, as a machine-checkable label:
    #:   no_label_column | not_a_table | wrong_measurements | other
    #: A refusal on grounds the rules can check is not simply obeyed -- if the
    #: rules can see the label column the model says is missing, the two
    #: disagree about a fact, and a person decides.
    blocker: str = ""
    #: A few sentences for a person reading the run afterwards: what this table
    #: is, what it can contribute to *this* question, and -- when it cannot be
    #: used -- what specifically blocks it and what would make it usable. The
    #: `reason` above is the machine's one line; this is the account.
    report: str = ""
    #: What a person could do to make this table usable, in one concrete step
    #: ("transpose it: genes are rows", "join the labels on `barcode`, which the
    #: metadata table beside it has", "fetch the file named in `relative_path`").
    #: A run that ends with no gold is not always a run with no data -- this is
    #: the difference between the two.
    salvage: str = ""
    confidence: float = 0.0
    source: str = "llm"  # llm | fallback | none
    #: Perturbation tasks only (see `review_table(task_kind="perturbation")`):
    #: the column holding the perturbation identity, the values that mean
    #: "nothing was perturbed", the assay, and whether the table is a screen at
    #: all. The single-cell x->y task has no use for these -- its label is a
    #: measured phenotype, a screen's label is *which perturbation*.
    is_screen: bool = False
    #: A screen needs the measurements too: a metadata table with the
    #: perturbations but no gene columns cannot supervise a response model.
    has_expression: bool = False
    condition_column: str | None = None
    control_labels: list[str] = field(default_factory=list)
    modality: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _schema() -> dict[str, Any]:
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
                "enum": [
                    "no_label_column",
                    "not_a_table",
                    "wrong_measurements",
                    "other",
                ],
                "description": (
                    "when the role is unusable, which blocker applies: "
                    "no_label_column (the table is fine but has no column for "
                    "this task's label), not_a_table, wrong_measurements, other. "
                    "Use an empty string when the role is not unusable."
                ),
            },
            "report": {
                "type": "string",
                "description": (
                    "3-5 sentences for a person: what this table is, what it can "
                    "contribute to this question, and if it cannot be used, what "
                    "blocks it and what would make it usable"
                ),
            },
            "salvage": {
                "type": "string",
                "description": (
                    "one concrete manual step that would make this table usable, "
                    "e.g. 'transpose it: genes are rows and cells are columns', "
                    "'join the labels in <other file> on barcode', 'fetch the file "
                    "named in column relative_path', 'unzip it and use the counts "
                    "matrix'. Empty string when no manual step would help, or when "
                    "the table is usable as it is."
                ),
            },
            "confidence": {"type": "number"},
        },
        "required": ["header_present", "role", "reason"],
    }


def review_table(
    question: str,
    packet: dict[str, Any],
    *,
    client: Any = None,
    known_paths: Sequence[str] = (),
    log: Callable[[str], None] = print,
    target_hint: str = "",
    task_kind: str = "per_cell",
) -> TableReview | None:
    """Ask the model what this table is. None means "no review was possible".

    `task_kind="perturbation"` switches the question: the target is not a
    measured phenotype but *which perturbation* a cell carries, so the review
    looks for a screen's condition column, its control values and its assay
    instead of a label column. A table with no condition column cannot be gold
    for such a task, however table-like it is.
    """
    from .sample_prompt import render_packet  # local import: keeps prompt text testable

    path = str(packet.get("path", ""))
    if client is None:
        log(f"# review: no model available for {_name(path)}; mechanical rules only")
        return None

    known = [p for p in known_paths if p != path]
    target_line = (
        f"The run has already decided that this task's label column is named "
        f"{target_hint!r}. If you can see that column in the listing above, the "
        f"table is labelled for this task -- a wide table (one column per gene, "
        f"say) is still a per-observation table when it has a label column.\n"
        if target_hint
        else ""
    )
    known_line = (
        "Other tables already fetched (for `same_dataset_as`):\n"
        + "\n".join(f"  {p}" for p in known)
        if known
        else "No other table has been fetched, so `same_dataset_as` must be null."
    )
    # One profile per task kind: the roles to explain, the schema to ask for,
    # the extra fields to read back and the guard to apply. The x->y task and the
    # perturbation task ask different questions of the same sample.
    profile = profile_for(task_kind)
    perturbation = profile.name == "perturbation"
    roles_help = profile.roles_help
    schema = profile.review_schema()
    if perturbation:
        task_block = (
            "This run answers a **perturbation-response** question: what does "
            "perturbing some gene(s) do to a cell's transcriptome. The thing to "
            "find is therefore a per-cell perturbation **screen**, not a labeled "
            "phenotype:\n"
            "- `is_screen` is true only when each row is a single cell and the "
            "table records which perturbation it carries (or that it is a "
            "control). A genes-by-cells matrix with no condition column, a cell "
            "metadata table, or a sample-level table is false.\n"
            "- `condition_column` is the column naming the targeted gene(s) / "
            "guide / sgRNA; it must be a column you can see. If there is none, the "
            "role cannot be `gold`.\n"
            "- `control_labels` are the values of that column that mean 'nothing "
            "was perturbed', exactly as they appear in the table (commonly "
            "`ctrl`, `control`, `non-targeting`, `NT`). A value like `GENE+ctrl` "
            "means the second guide is a control, i.e. a single-gene "
            "perturbation.\n"
            "- `modality` says how the screen perturbs: `crisprko` (knockout / "
            "deletion), `crispri` (interference / knockdown), `crispra` "
            "(activation / over-expression), `mixed`, or `unknown` when the table "
            "does not say. Use your knowledge of the dataset when the header "
            "alone is silent.\n"
            "- `target_column` must equal `condition_column` when the role is "
            "`gold`.\n"
            "- `bad_label` is for a table that has a perturbation-looking column "
            "but whose rows are not single perturbed cells.\n\n"
        )
    else:
        task_block = ""
    prompt = (
        "You are reviewing a dataset someone just downloaded, to decide what role "
        "it can play in an analysis.\n\n"
        f"Research question:\n{question}\n\n"
        f"What was downloaded:\n{render_packet(packet)}\n\n"
        "Roles:\n"
        f"{roles_help}\n"
        f"{known_line}\n\n"
        f"{task_block}"
        f"{target_line}"
        "Rules for your answer:\n"
        "- `header_present` is false when the first line is data rather than column "
        "names. The parsed columns above are the strongest clue: if they are values "
        "('6', '148', '33.6'), the file has no header. When it is false, give the "
        "real column names in `header_names` **only if you are confident** (for a "
        "well-known dataset you may know them); otherwise leave it empty.\n"
        + (
            ""
            if perturbation
            else "- `target_column` must be a column you actually see (from the "
            "parsed columns, or from `header_names` when you supplied them). Use "
            "null if the question's target is not in this table.\n"
        )
        +
        "- `bad_label` is for labels that exist but that this analysis must not "
        "train on. Say why in `label_problem`.\n"
        "- `id_columns` are columns that identify a row rather than measure it.\n\n"
        "- `blocker` says why a table is unusable, in one of the four machine-"
        "checkable values. Use `no_label_column` only when you have looked for "
        f"{target_hint or 'the label'} and it is genuinely absent.\n\n"
        "- `report` is written for a person reading this run later: what the table "
        "is, what it contributes to this question, and -- when it cannot be used -- "
        "what blocks it and what would make it usable (a column to drop, a column "
        "to exclude, a table it should be paired with). Keep it to 3-5 sentences "
        "and do not repeat the schema fields.\n\n"
        "- `salvage` is the one step a person would take by hand to make this "
        "table usable, when there is one: `transpose it, genes are rows`, `join "
        "the labels in <file> on barcode`, `fetch the file named in column "
        "relative_path`, `unzip it and use the counts matrix`. A run that ends "
        "with nothing to train on is not the same as a run with no usable data, "
        "and this field is the difference. Leave it empty when no manual step "
        "would help.\n\n"
        "Answer as JSON with exactly the fields in the schema."
    )
    log(f"# review: asking the model about {_name(path)}")
    try:
        response = client.generate_structured(
            prompt=prompt, schema=schema, max_tokens=1100, temperature=0
        )
    except Exception as e:  # noqa: BLE001 - a failed call means "no review"
        log(f"# review: model call failed for {_name(path)}: {e}")
        return None
    if isinstance(response, str):
        try:
            response = json.loads(response)
        except json.JSONDecodeError:
            log(f"# review: model returned non-JSON for {_name(path)}")
            return None
    if not isinstance(response, dict):
        return None

    role = str(response.get("role", ""))
    if role not in {"gold", "unlabeled", "bad_label", "unusable"}:
        log(f"# review: model gave an unknown role {role!r}; ignoring the review")
        return None

    header_present = bool(response.get("header_present", True))
    header_names = [str(n) for n in (response.get("header_names") or []) if str(n).strip()]
    columns = [str(c) for c in (packet.get("columns") or [])]
    if not header_present and header_names and len(header_names) != len(columns):
        log(
            f"# review: {_name(path)} proposed {len(header_names)} names for "
            f"{len(columns)} columns; the name list is discarded"
        )
        header_names = []

    target = response.get("target_column")
    target = str(target) if target else None
    visible = header_names or columns
    if target and target not in visible:
        log(
            f"# review: {_name(path)} named a target {target!r} that is not in "
            f"the table; ignoring it"
        )
        target = None

    same_as = response.get("same_dataset_as") or None
    if same_as and str(same_as) not in known:
        same_as = None

    review = TableReview(
        path=path,
        role=role,  # type: ignore[arg-type]
        header_present=header_present,
        header_names=header_names,
        same_dataset_as=str(same_as) if same_as else None,
        target_column=target,
        id_columns=[str(c) for c in (response.get("id_columns") or [])],
        label_problem=str(response.get("label_problem") or ""),
        reason=str(response.get("reason") or ""),
        blocker=str(response.get("blocker") or ""),
        report=str(response.get("report") or ""),
        salvage=str(response.get("salvage") or ""),
        confidence=float(response.get("confidence") or 0.0),
        source="llm",
    )
    if profile.extract is not None:
        fields = dict(profile.extract(response, visible))
        forced = fields.pop("force_role", None)
        if forced and review.role == "gold":
            # The guard the profile exists for: the model called it gold, but a
            # table without the column this task needs cannot supervise it.
            log(
                f"# review: {_name(path)} was proposed gold but {forced} for this "
                f"task ({ {k: fields.get(k) for k in ('is_screen', 'condition_column')} }); "
                f"refusing it as gold"
            )
            review.role = forced
            review.blocker = review.blocker or "no_label_column"
        for key, value in fields.items():
            setattr(review, key, value)
        if perturbation:
            log(
                f"# review: {_name(path)} screen={review.is_screen} "
                f"expression={getattr(review, 'has_expression', None)} "
                f"condition={review.condition_column!r} "
                f"controls={review.control_labels[:4]} modality={review.modality}"
            )

    log(
        f"# review: {_name(path)} -> {review.role} "
        f"(header_present={review.header_present}, target={review.target_column})"
    )
    log(f"#   why: {review.reason[:180]}")
    return review


def _name(path: str) -> str:
    from pathlib import Path

    return Path(path).name
