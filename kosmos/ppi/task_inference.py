"""Decide which column a research question asks us to predict.

This is the only inference in the training path, and the only place a model
makes a decision rather than a measurement. Everything else -- is the label
present, do the features match, is a column numeric -- is a deterministic rule
in `datafetcher.profile`.

So the design is defensive, in three descending steps, each recording how it
decided:

  1. **Name matching.** A hint (the protocol's dependent variable, or a
     `--hint`) against the table's real column names, normalised. Exact and
     auditable.
  2. **A constrained model choice.** The model sees the question, the eligible
     column names, their kinds and a few sample values, and must return *one of
     those names*. A name outside the list is discarded, never repaired.
  3. **Nothing.** Returns `None` with the rejected candidates and the reasons,
     so the caller asks a human instead of guessing.

Guards run on every candidate regardless of which step proposed it, because a
name that matches the question can still be structurally wrong: an identifier
is unique in every row, and a class with one member cannot be split or scored.
Those are the two ways a confident-sounding guess produces a run that "worked"
and means nothing.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import pandas as pd

from .tabular import read_table as _read_table

InferenceSource = Literal["hint_exact", "hint_match", "objective_match", "llm", "none"]
Confidence = Literal["exact_match", "inferred", "none"]

DEFAULT_MIN_CLASSES = 2
DEFAULT_MIN_PER_CLASS = 2
#: Below this, a numeric column is a small set of categories, not a measurement.
DEFAULT_MIN_REGRESSION_LEVELS = 10


def normalize(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(name).strip().lower()).strip("_")


def _tokens(name: str) -> set[str]:
    return {token for token in normalize(name).split("_") if token}


#: Words that carry a sentence rather than name its label. A column called
#: after one of them is a coincidence: the McFarland question asked about "the
#: drug a cancer cell *was* treated with", the gene `WAS` matched, and every
#: single-cell table has a column for it -- so an unlabeled perturbation table
#: looked labeled and a run trained on its own expression.
FUNCTION_WORDS = frozenset(
    """
    a an and are as at be been being but by can could did do does doing for from
    had has have having how if in into is it its may might must no nor not of off
    on onto or our should so some such than that the their them then there these
    they this those to under until up was we were what when where whether which
    while who whom whose why will with within would you your
    """.split()
)
#: A bare name this short, matched only by the prose of the question, is a
#: coincidence rather than a label: `WAS`, `SET` and `MET` are gene symbols and
#: ordinary words at once. An explicit hint still wins -- the protocol named the
#: column -- and the model step is unaffected, so a real three-letter label
#: (`age`) is still found.
MIN_PROSE_MATCH_CHARS = 4


def _prose_plausible(name: str) -> bool:
    """Can the question's own wording promote this column to a label?

    Mentions of real columns live in the question too ("predict the **cell
    type**"), so the rule cannot be "never". It is: the name has to be long
    enough to be a name, and a one-word name must not be a word that carries the
    sentence.
    """
    tokens = list(_tokens(name))
    if not tokens:
        return False
    if len(tokens) == 1:
        token = tokens[0]
        if token in FUNCTION_WORDS:
            return False
        if len(token) < MIN_PROSE_MATCH_CHARS and not token.isdigit():
            return False
    return True


@dataclass
class ColumnFacts:
    """What the table says about one column, measured rather than guessed."""

    name: str
    kind: str
    n_rows: int
    n_unique: int
    null_fraction: float
    samples: list[str] = field(default_factory=list)
    #: The size of the smallest value group, over the WHOLE column.
    min_class_count: int = 0
    #: Display only, capped: value counts of a continuous column are thousands
    #: of entries long and the question is only ever "how small is the smallest".
    value_counts: dict[str, int] = field(default_factory=dict)

    @property
    def is_unique_text(self) -> bool:
        return self.kind in {"text", "other"} and self.n_unique == self.n_rows

    @property
    def is_unique_numeric(self) -> bool:
        """A numeric column where every value differs is an index, not a label."""
        return self.kind == "numeric" and self.n_rows > 3 and self.n_unique == self.n_rows


@dataclass
class TargetCandidate:
    column: str
    score: float
    reason: str
    #: Where the score came from: an explicit hint names a column, the question
    #: is prose that may merely mention one. The two are scored apart because
    #: only the first is evidence that this column is the label.
    matched_by: Literal["hint", "objective"] = "hint"


@dataclass
class TargetInference:
    column: str | None
    source: InferenceSource
    confidence: Confidence
    reason: str
    #: What kind of label the column is: a category, or a measured value.
    task_type: str = "classification"
    candidates: list[TargetCandidate] = field(default_factory=list)
    rejected: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        # A wide table rejects hundreds of columns (every gene that is unique
        # per row looks like an index). The reason for four of them is
        # informative; the reason for six hundred is noise.
        shown = dict(list(self.rejected.items())[:10])
        return {
            "target_column": self.column,
            "task_type": self.task_type,
            "source": self.source,
            "confidence": self.confidence,
            "reason": self.reason,
            "candidates": [c.__dict__ for c in self.candidates],
            "rejected": shown,
            "rejected_total": len(self.rejected),
        }


def _kind_of(series: pd.Series) -> str:
    if series.isna().all():
        return "empty"
    kind = series.dtype.kind
    if kind in {"i", "u", "f"}:
        return "numeric"
    if kind == "b":
        return "bool"
    if kind in {"M", "m"}:
        return "datetime"
    return "text"


def column_facts(
    path: str | Path,
    *,
    columns: Sequence[str] | None = None,
    sample_values: int = 3,
    max_rows: int | None = None,
) -> dict[str, ColumnFacts]:
    """Measure each column: kind, cardinality, nulls, a few example values.

    `max_rows` bounds the read. Choosing a label column is a question about
    column *names* and kinds, so scanning 78,967 unlabeled rows to answer it is
    work that changes no decision -- and it is the slowest part of a sweep.
    """
    frame = _read_table(path, nrows=max_rows)
    if columns is not None:
        missing = [c for c in columns if c not in frame.columns]
        if missing:
            raise ValueError(f"{path} has no column(s) {missing[:3]}")
        frame = frame[list(columns)]
    facts: dict[str, ColumnFacts] = {}
    for name in frame.columns:
        series = frame[name]
        counts = series.astype(str).value_counts()
        facts[str(name)] = ColumnFacts(
            name=str(name),
            kind=_kind_of(series),
            n_rows=int(len(series)),
            n_unique=_n_unique(series),
            null_fraction=float(series.isna().mean()),
            samples=_samples(series, sample_values),
            min_class_count=int(counts.min()) if len(counts) else 0,
            value_counts={str(k): int(v) for k, v in counts.head(50).items()},
        )
    return facts


def _n_unique(series: pd.Series) -> int:
    """Distinct values, tolerating cells that cannot be hashed.

    A dataset prepared for machine learning carries an embedding per row -- a
    numpy array, which `nunique` refuses. That made the whole candidate
    unreadable ("could not read: unhashable type: 'numpy.ndarray'") and hid the
    columns that could actually serve as a label.
    """
    try:
        return int(series.nunique(dropna=True))
    except TypeError:
        return int(series.astype(str).nunique(dropna=True))


def _samples(series: pd.Series, count: int) -> list[str]:
    """A few example values, by the same rule as `_n_unique`."""
    present = series.dropna()
    try:
        values = present.unique()[:count]
    except TypeError:
        values = present.head(count).to_numpy()
    return [str(value) for value in values]


def eligible_columns(
    facts: dict[str, ColumnFacts],
    *,
    exclude: Iterable[str] = (),
    feature_prefixes: Iterable[str] = (),
    min_classes: int = DEFAULT_MIN_CLASSES,
    min_per_class: int = DEFAULT_MIN_PER_CLASS,
    min_regression_levels: int = DEFAULT_MIN_REGRESSION_LEVELS,
    task_type: str = "classification",
    max_cardinality_fraction: float = 0.1,
    max_cardinality_floor: int = 50,
) -> tuple[list[str], dict[str, str]]:
    """Which columns could be a label at all, and why the rest could not."""
    excluded = set(exclude)
    prefixes = tuple(feature_prefixes)
    eligible: list[str] = []
    rejected: dict[str, str] = {}
    for name, column in facts.items():
        if name in excluded:
            rejected[name] = "excluded: it is a feature or an identifier for this task"
            continue
        if prefixes and name.startswith(prefixes):
            # A column the task already declared to be a feature cannot also be
            # the label. Without this, a question mentioning a gene name could
            # nominate that gene as the thing to predict.
            rejected[name] = (
                f"excluded: its name starts with {', '.join(repr(p) for p in prefixes)}, "
                f"which this task treats as features"
            )
            continue
        if column.kind == "empty":
            rejected[name] = "every value is missing"
            continue
        if column.is_unique_text:
            rejected[name] = (
                f"unique text in every row ({column.n_unique}/{column.n_rows}), "
                f"so it is an identifier rather than a label"
            )
            continue
        if column.is_unique_numeric:
            rejected[name] = (
                f"unique number in every row ({column.n_unique}/{column.n_rows}), "
                f"so it looks like an index"
            )
            continue
        if task_type == "regression":
            # A regression label is a measurement: numbers, and enough distinct
            # values that predicting a value means something.
            if column.kind != "numeric":
                rejected[name] = (
                    f"a regression label has to be a measured number; {name} is "
                    f"{column.kind}"
                )
                continue
            if column.n_unique < min_regression_levels:
                rejected[name] = (
                    f"only {column.n_unique} distinct value(s); a regression label "
                    f"needs at least {min_regression_levels} to be worth predicting "
                    f"(this looks like a category)"
                )
                continue
            eligible.append(name)
            continue
        if task_type == "classification":
            if column.n_unique < min_classes:
                rejected[name] = (
                    f"only {column.n_unique} distinct value(s); a classification "
                    f"label needs at least {min_classes}"
                )
                continue
            cardinality_cap = max(max_cardinality_floor, int(max_cardinality_fraction * column.n_rows))
            if column.n_unique > cardinality_cap:
                rejected[name] = (
                    f"{column.n_unique} distinct values across {column.n_rows} rows "
                    f"is a continuous measurement, not a classification label "
                    f"(more than {cardinality_cap} would be)"
                )
                continue
            # A class with a single member is a fact about the sample, not a
            # reason to refuse the label: BMMC cell types include populations
            # that a 2,000-cell sample catches once, and the split already
            # degrades gracefully for them (`_stratify` skips stratification
            # when a class is too small). Refusing here lost the whole table.
            # The cardinality cap above is what keeps ids and measurements out.
        eligible.append(name)
    return eligible, rejected


def _score(column: str, hints: Sequence[str], *, prose: bool = False) -> tuple[float, str]:
    """How well a column name matches the hints, 0..1, with the reason.

    `prose=True` says the terms are the question itself rather than names. A
    column mentioned in a sentence is a weaker signal than a column named by a
    hint, so the loose rules -- matching part of a longer string -- are only
    available to names that could carry a label (`_prose_plausible`). Without
    that, "was treated with" nominated the gene `WAS`, and `_score` is what made
    the rules call a gene-expression matrix "labeled".
    """
    if not hints:
        return 0.0, "no hint to match against"
    column_norm = normalize(column)
    column_tokens = _tokens(column)
    best = 0.0
    best_reason = "no overlap with the hints"
    for hint in hints:
        hint_norm = normalize(hint)
        if not hint_norm:
            continue
        if column_norm == hint_norm:
            return 1.0, f"column name equals the hint {hint!r}"
        overlap = column_tokens & _tokens(hint)
        if not overlap:
            continue
        score = len(overlap) / max(len(_tokens(hint)), 1)
        if (hint_norm in column_norm or column_norm in hint_norm) and (
            not prose or _prose_plausible(column)
        ):
            score = max(score, 0.75)
        if score > best:
            best = score
            best_reason = f"shares {sorted(overlap)} with the hint {hint!r}"
    return best, best_reason


def rank_candidates(
    hints: Sequence[str],
    facts: dict[str, ColumnFacts],
    *,
    objective: str = "",
) -> list[TargetCandidate]:
    """Every column a name or the question's wording could point at, best first.

    The two sources are scored separately so that the origin travels with the
    score: a column found only because the question's prose mentions it is
    reported as `objective_match`, not as if a hint had named it.
    """
    ranked: list[TargetCandidate] = []
    for name in facts:
        score, reason = _score(name, hints)
        matched_by: Literal["hint", "objective"] = "hint"
        if objective:
            prose_score, prose_reason = _score(name, [objective], prose=True)
            if prose_score > score:
                score, reason, matched_by = prose_score, prose_reason, "objective"
        if score > 0:
            ranked.append(
                TargetCandidate(
                    column=name, score=score, reason=reason, matched_by=matched_by
                )
            )
    return sorted(ranked, key=lambda c: (-c.score, c.column))


def _llm_choice(
    objective: str,
    eligible: list[str],
    facts: dict[str, ColumnFacts],
    client: Any,
    *,
    max_columns: int = 30,
    decide_task_type: bool = False,
) -> tuple[str | None, str, str]:
    """Ask a model, constrained to the eligible names. Discards anything else.

    A wide table makes a wide prompt, so the shortlist is bounded: columns that
    scored against the question first, then the least-cardinal columns, which is
    where labels live. The model can only answer from what it is shown, so the
    bound is also what keeps the answer checkable.

    With `decide_task_type`, the same call also answers whether the column is a
    category or a measurement -- the model reads the question, so it knows
    whether "how do these parameters affect efficiency" asks for a value or a
    class. The mechanical guards then check that answer.
    """
    ranked = rank_candidates(
        (), {name: facts[name] for name in eligible}, objective=objective
    )
    ordered = [c.column for c in ranked if c.score > 0]
    ordered_set = set(ordered)
    ordered += sorted(
        [name for name in eligible if name not in ordered_set],
        key=lambda name: (facts[name].n_unique, name),
    )
    shortlist = ordered[:max_columns]
    listing = "\n".join(
        f"- {name} ({facts[name].kind}, {facts[name].n_unique} distinct, "
        f"e.g. {facts[name].samples[:3]})"
        for name in shortlist
    )
    type_block = (
        "Also say what kind of quantity it is:\n"
        "  classification -- a category or label (disease yes/no, cell type, "
        "grade), predicted by a classifier\n"
        "  regression     -- a measured value (efficiency, concentration, "
        "expression level), predicted by a regressor\n"
        if decide_task_type
        else ""
    )
    keys = "'column', 'reason'" + (", 'task_type'" if decide_task_type else "")
    header = (
        "A researcher wants this analysis:\n\n"
        f"{objective}\n\n"
        "Which ONE of the columns below is the quantity being predicted?\n\n"
        f"{listing}\n\n"
        f"{type_block}"
        f"Answer with a JSON object with keys {keys}. "
        "'column' must be one of the names listed above, exactly; "
        "if none is plausible, make it null."
    )
    schema = {
        "type": "object",
        "properties": {
            "column": {"type": ["string", "null"]},
            "reason": {"type": "string"},
            **(
                {"task_type": {"type": "string", "enum": ["classification", "regression"]}}
                if decide_task_type
                else {}
            ),
        },
        "required": ["column", "reason"] + (["task_type"] if decide_task_type else []),
    }
    try:
        response = client.generate_structured(
            prompt=header, schema=schema, max_tokens=500, temperature=0
        )
    except Exception as e:  # noqa: BLE001 - a failed model call is "no answer"
        return None, f"model call failed: {e}", "classification"
    if isinstance(response, str):
        try:
            response = json.loads(response)
        except json.JSONDecodeError:
            return None, "model did not return JSON", "classification"
    column = (response or {}).get("column")
    reason = str((response or {}).get("reason") or "")
    task_type = str((response or {}).get("task_type") or "classification")
    if task_type not in {"classification", "regression"}:
        task_type = "classification"
    if column not in shortlist:
        # Deliberately not repaired into the closest name: a name the model
        # invented is a signal that the question does not match this table.
        return None, f"model named {column!r}, which is not an eligible column", task_type
    return column, reason or "chosen by the model", task_type


def infer_target_column(
    path: str | Path,
    *,
    objective: str = "",
    hints: Sequence[str] = (),
    exclude: Iterable[str] = (),
    feature_prefixes: Iterable[str] = (),
    client: Any = None,
    facts: dict[str, ColumnFacts] | None = None,
    min_classes: int = DEFAULT_MIN_CLASSES,
    min_per_class: int = DEFAULT_MIN_PER_CLASS,
    min_regression_levels: int = DEFAULT_MIN_REGRESSION_LEVELS,
    task_type: str = "classification",
) -> TargetInference:
    """Name the column this question predicts, or say nothing with reasons.

    `task_type="auto"` lets the model answer a second question at the same
    time -- is this column a category or a measured value -- and the mechanical
    guards check the answer: a regression label has to be numeric, and a
    classification label has to have few enough distinct values. The model
    decides what the analysis is; the rules decide whether the column can carry
    it.
    """
    facts = facts if facts is not None else column_facts(path)
    auto = str(task_type) == "auto"
    eligible_by_type: dict[str, list[str]] = {}
    if auto:
        rejected: dict[str, str] = {}
        for kind in ("classification", "regression"):
            names, why = eligible_columns(
                facts,
                exclude=exclude,
                feature_prefixes=feature_prefixes,
                min_classes=min_classes,
                min_per_class=min_per_class,
                min_regression_levels=min_regression_levels,
                task_type=kind,
            )
            eligible_by_type[kind] = names
            for name, reason in why.items():
                rejected.setdefault(name, f"[{kind}] {reason}")
        eligible = list(
            dict.fromkeys(eligible_by_type["classification"] + eligible_by_type["regression"])
        )
        resolved_type = "classification"
    else:
        eligible, rejected = eligible_columns(
            facts,
            exclude=exclude,
            feature_prefixes=feature_prefixes,
            min_classes=min_classes,
            min_per_class=min_per_class,
            min_regression_levels=min_regression_levels,
            task_type=task_type,
        )
        resolved_type = str(task_type)
    if not eligible:
        return TargetInference(
            column=None,
            source="none",
            confidence="none",
            reason="no column in this table can serve as a label for the task",
            task_type=resolved_type,
            rejected=rejected,
        )

    def usable_type(column: str, preferred: str) -> str | None:
        """The task type this column can actually carry."""
        eligible_set = set(eligible)
        if not auto:
            return preferred if column in eligible_set else None
        eligible_by_kind = {
            kind: set(names or []) for kind, names in eligible_by_type.items()
        }
        order = [preferred, "regression" if preferred == "classification" else "classification"]
        for kind in order:
            if column in eligible_by_kind.get(kind, set()):
                return kind
        return None

    def ambiguous(column: str) -> bool:
        """True when both task types are mechanically possible.

        A numeric column with a dozen distinct values can be read either way
        (a coarse measurement, or a small set of categories). The rules cannot
        settle that, so the shortcut does not either: the model is asked, and
        only if there is no model does classification -- the previous behaviour
        -- stand.
        """
        if not auto:
            return False
        eligible_by_kind = {
            kind: set(names or []) for kind, names in eligible_by_type.items()
        }
        return all(
            column in eligible_by_kind.get(kind, set())
            for kind in ("classification", "regression")
        )

    # Hints and the question's prose are ranked together but not the same: only
    # a hint is a name, so only a hint can be the confident shortcut below.
    ranked = [
        c
        for c in rank_candidates(hints, facts, objective=objective)
        if c.column in eligible
    ]

    # A hint that names a column exactly is a lookup, not an inference: the
    # protocol said "predict cell_type" and the table has `cell_type`.
    for candidate in ranked:
        if (
            candidate.score >= 1.0
            and hints
            and candidate.matched_by == "hint"
            and not (ambiguous(candidate.column) and client)
        ):
            return TargetInference(
                column=candidate.column,
                source="hint_exact",
                confidence="exact_match",
                reason=candidate.reason,
                task_type=usable_type(candidate.column, resolved_type) or resolved_type,
                candidates=ranked[:5],
                rejected=rejected,
            )
    if ranked and ranked[0].score >= 0.6 and (
        not ranked[1:] or ranked[1].score < ranked[0].score
    ) and not (ambiguous(ranked[0].column) and client):
        return TargetInference(
            column=ranked[0].column,
            source="hint_match" if ranked[0].matched_by == "hint" else "objective_match",
            confidence="inferred",
            reason=ranked[0].reason,
            task_type=usable_type(ranked[0].column, resolved_type) or resolved_type,
            candidates=ranked[:5],
            rejected=rejected,
        )

    if client is not None:
        column, reason, model_type = _llm_choice(
            objective, eligible, facts, client, decide_task_type=auto
        )
        if column:
            kind = usable_type(column, model_type if auto else resolved_type)
            if kind is None:
                rejected["__model__"] = (
                    f"the model chose {column!r} as a {model_type} label, and it "
                    f"cannot carry one"
                )
                return TargetInference(
                    column=None,
                    source="none",
                    confidence="none",
                    reason=rejected["__model__"],
                    task_type=model_type if auto else resolved_type,
                    candidates=ranked[:5],
                    rejected=rejected,
                )
            if auto and kind != model_type:
                reason = (
                    f"{reason} (the model called it {model_type}; the column's "
                    f"values make it {kind})"
                )
            return TargetInference(
                column=column,
                source="llm",
                confidence="inferred",
                reason=reason,
                task_type=kind,
                candidates=ranked[:5],
                rejected=rejected,
            )
        rejected["__model__"] = reason

    return TargetInference(
        column=None,
        source="none",
        confidence="none",
        reason=(
            "no column matched the question or hints closely enough"
            if ranked
            else "no column name appears in the question or the hints"
        ),
        task_type=resolved_type,
        candidates=ranked[:5],
        rejected=rejected,
    )
