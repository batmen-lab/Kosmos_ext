"""The perturbation-response contract: control, perturbation, delta, splits.

One example is a *cell that was perturbed*: its control expectation, the gene(s)
that were perturbed, and what happened. The target is the expression change

    Δy = y_pert − μ_ctrl

rather than the absolute post-perturbation expression, because that is the
quantity the perturbation caused; `μ_ctrl` is the mean of the control cells in
the same context group (cell type / donor / batch when those columns exist),
and every example records which group its control mean came from.

Splitting is by **perturbation identity**, never by cell. Cells of one
perturbation are near-duplicates of each other; a random cell split would put
some of them in training and score the rest, which measures memorisation of a
condition the model has already seen. Three evaluation regimes are supported:

  * `unseen_single`      -- held-out single-gene perturbations
  * `unseen_combination` -- held-out multi-gene perturbations
  * `mixed` (default)    -- held out from both, which is the harder and more
                            honest one for a first number

The gene panel and `gene_to_index` are built once and reused by every dataset
and every graph: a graph whose nodes are not the panel's nodes is not a graph
of the same genes.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

#: Values that mean "nothing was perturbed".
DEFAULT_CONTROL_LABELS = (
    "ctrl",
    "control",
    "control_ctrl",
    "vehicle",
    "none",
    "non-targeting",
    "non_targeting",
    "nt",
    "wt",
    "unperturbed",
)
#: How conditions are written when several genes are perturbed together.
CONDITION_SEPARATORS = ("+", "_", ";")


@dataclass(frozen=True)
class PerturbationTask:
    """The panel every model and every graph is defined on."""

    genes: tuple[str, ...]
    gene_to_index: dict[str, int]
    condition_column: str
    control_labels: tuple[str, ...]
    delta_scale: float = 1.0

    @property
    def n_genes(self) -> int:
        return len(self.genes)

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_genes": self.n_genes,
            "condition_column": self.condition_column,
            "control_labels": list(self.control_labels),
            "delta_scale": self.delta_scale,
            "genes_sha256": _digest(self.genes),
        }


@dataclass
class PerturbationExample:
    """One perturbed cell: where it started, what was perturbed, what changed."""

    control: np.ndarray          # (G,) control expectation for this cell's context
    perturbation: tuple[str, ...]
    delta: np.ndarray            # (G,) y_pert − control
    target: np.ndarray           # (G,) y_pert
    context: dict[str, Any] = field(default_factory=dict)
    sample_id: str = ""
    control_group: str = "global"

    @property
    def label(self) -> str:
        return "+".join(self.perturbation)


def parse_condition(value: Any, *, separators: Sequence[str] = CONDITION_SEPARATORS) -> tuple[str, ...]:
    """`'CBL+CNN1'` (or `'CBL_CNN1'`) -> `('CBL', 'CNN1')`; control -> `()`.

    Everything that is not a separator is kept as the gene's own name, so names
    with hyphens (`ARMCX5-GPRASP2`) survive; only the separators above split.
    """
    text = str(value).strip()
    if not text:
        return ()
    for separator in separators:
        if separator in text:
            return tuple(part for part in text.split(separator) if part)
    return (text,)


def is_control(value: Any, control_labels: Sequence[str] = DEFAULT_CONTROL_LABELS) -> bool:
    text = str(value).strip().lower()
    return text in {label.lower() for label in control_labels}


def _digest(values: Iterable[str]) -> str:
    import hashlib

    digest = hashlib.sha256()
    for value in values:
        digest.update(str(value).encode())
        digest.update(b"\x00")
    return digest.hexdigest()


def build_task(
    frame: pd.DataFrame,
    *,
    condition_column: str,
    gene_columns: Sequence[str],
    control_labels: Sequence[str] = DEFAULT_CONTROL_LABELS,
    delta_scale: float = 1.0,
) -> PerturbationTask:
    """The panel and the vocabularies, from the table that has the labels."""
    genes = tuple(str(gene) for gene in gene_columns)
    if not genes:
        raise ValueError("a perturbation task needs the measured genes as columns")
    return PerturbationTask(
        genes=genes,
        gene_to_index={gene: index for index, gene in enumerate(genes)},
        condition_column=str(condition_column),
        control_labels=tuple(control_labels),
        delta_scale=float(delta_scale or 1.0),
    )


def build_examples(
    frame: pd.DataFrame,
    task: PerturbationTask,
    *,
    context_columns: Sequence[str] = (),
    min_cells_per_perturbation: int = 2,
    gene_reference: np.ndarray | None = None,
) -> tuple[list[PerturbationExample], dict[str, Any]]:
    """Turn a per-cell table into (control, perturbation, Δ) examples.

    `gene_reference` is the control mean profile to subtract when the table has
    no control cells of its own (a supplementary dataset, for instance): the
    caller passes the gold's `μ_ctrl` so the deltas are on one scale.
    """
    genes = list(task.genes)
    missing = [gene for gene in genes if gene not in frame.columns]
    if missing:
        raise ValueError(
            f"the table is missing {len(missing)} panel gene(s), e.g. {missing[:3]}"
        )
    values = frame[genes].to_numpy(dtype=np.float32, na_value=0.0)
    conditions = frame[task.condition_column]
    context = {column: frame[column].to_numpy() for column in context_columns if column in frame.columns}

    control_mask = conditions.map(lambda value: is_control(value, task.control_labels)).to_numpy()
    group_key = None
    if context:
        columns = [column for column in ("cell_type", "donor", "batch") if column in context]
        if columns:
            group_key = np.array(
                [
                    "|".join(str(context[column][index]) for column in columns)
                    for index in range(len(frame))
                ]
            )
    groups: dict[str, np.ndarray] = {}
    if control_mask.any():
        if group_key is not None:
            for key in np.unique(group_key[control_mask]):
                rows = control_mask & (group_key == key)
                groups[str(key)] = values[rows].mean(axis=0)
        groups["global"] = values[control_mask].mean(axis=0)
    elif gene_reference is not None:
        groups["global"] = np.asarray(gene_reference, dtype=np.float32)
    else:
        raise ValueError(
            "no control cells and no gene_reference: Δ cannot be defined for this table"
        )
    global_control = groups["global"]

    examples: list[PerturbationExample] = []
    counts: dict[str, int] = {}
    fallbacks = 0
    for index in range(len(frame)):
        if control_mask[index]:
            continue
        perturbation = parse_condition(conditions.iloc[index])
        if not perturbation:
            continue
        unknown = [gene for gene in perturbation if gene not in task.gene_to_index]
        if unknown:
            continue  # a perturbation whose gene is not in the panel cannot be encoded
        key = str(group_key[index]) if group_key is not None else "global"
        baseline = groups.get(key, global_control)
        if key != "global" and key not in groups:
            fallbacks += 1
        observed = values[index]
        delta = (observed - baseline) / task.delta_scale
        label = "+".join(perturbation)
        counts[label] = counts.get(label, 0) + 1
        examples.append(
            PerturbationExample(
                control=baseline.astype(np.float32),
                perturbation=tuple(perturbation),
                delta=delta.astype(np.float32),
                target=observed.astype(np.float32),
                context={
                    column: (values_.iloc[index] if hasattr(values_, "iloc") else values_[index])
                    for column, values_ in context.items()
                },
                sample_id=str(frame.index[index]),
                control_group=key,
            )
        )
    kept = [example for example in examples if counts[example.label] >= min_cells_per_perturbation]
    dropped = sorted(
        label for label, count in counts.items() if count < min_cells_per_perturbation
    )
    report = {
        "n_examples": len(kept),
        "n_candidates": len(examples),
        "n_perturbations": len({example.label for example in kept}),
        "control_cells": int(control_mask.sum()),
        "control_groups": sorted(groups),
        "global_fallbacks": int(fallbacks),
        "dropped_perturbations": dropped[:20],
        "dropped_count": len(dropped),
        "min_cells_per_perturbation": int(min_cells_per_perturbation),
    }
    return kept, report


def perturbation_splits(
    perturbations: Sequence[str],
    *,
    mode: str = "mixed",
    test_fraction: float = 0.2,
    validation_fraction: float = 0.1,
    seed: int = 42,
) -> dict[str, list[str]]:
    """Split **perturbation identities**, not cells.

    Singles and combinations are separated first, because the two regimes ask
    different questions: can the model transfer a single gene's effect from
    other perturbations, and can it compose two effects it has seen apart.
    """
    singles = sorted({p for p in perturbations if len(parse_condition(p)) == 1})
    combos = sorted({p for p in perturbations if len(parse_condition(p)) > 1})
    rng = np.random.default_rng(seed)

    def carve(pool: list[str], share: float) -> tuple[list[str], list[str]]:
        pool = list(pool)
        rng.shuffle(pool)
        take = int(round(len(pool) * share))
        return pool[:take], pool[take:]

    mode = str(mode)
    result: dict[str, list[str]] = {"splits_by": "perturbation"}
    if mode == "unseen_single":
        test, singles = carve(singles, test_fraction)
        validation, singles = carve(singles, validation_fraction)
        result.update(train=singles + combos, validation=validation, test=test)
    elif mode == "unseen_combination":
        test, combos = carve(combos, test_fraction)
        validation, combos = carve(combos, validation_fraction)
        result.update(train=singles + combos, validation=validation, test=test)
    elif mode == "mixed":
        test_singles, singles = carve(singles, test_fraction)
        test_combos, combos = carve(combos, test_fraction)
        validation_singles, singles = carve(singles, validation_fraction)
        validation_combos, combos = carve(combos, validation_fraction)
        result.update(
            train=singles + combos,
            validation=validation_singles + validation_combos,
            test=test_singles + test_combos,
        )
    else:
        raise ValueError(
            f"unknown split mode {mode!r}; use unseen_single, unseen_combination or mixed"
        )
    return result


def split_examples(
    examples: Sequence[PerturbationExample], splits: dict[str, list[str]]
) -> dict[str, list[PerturbationExample]]:
    """Assign every example to the split its perturbation was placed in."""
    where = {
        label: name
        for name in ("train", "validation", "test")
        for label in splits.get(name, [])
    }
    out: dict[str, list[PerturbationExample]] = {"train": [], "validation": [], "test": []}
    for example in examples:
        name = where.get(example.label)
        if name:
            out[name].append(example)
    return out


def write_contract(
    out_dir: str | Path,
    *,
    task: PerturbationTask,
    splits: dict[str, list[str]],
    report: dict[str, Any],
    sources: dict[str, Any] | None = None,
) -> Path:
    """The reproducible record of the contract: panel, mapping, splits, provenance."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    payload = {
        "task": task.to_dict(),
        "panel": list(task.genes),
        "gene_to_index": task.gene_to_index,
        "splits": splits,
        "split_sizes": {name: len(splits.get(name, [])) for name in ("train", "validation", "test")},
        "detail": report,
        "sources": sources or {},
    }
    path = out / "perturbation_contract.json"
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    return path
