"""Teacher-generated labels on supplementary cells -- model-agnostic.

The mechanism the perturbation backends share:

  1. train a **teacher** on the gold split only (any architecture: the caller
     passes a `predict` callable);
  2. take control cells from the supplementary screen, assign each one a
     perturbation query drawn from the gold **training** perturbations, and let
     the teacher predict the response -- the synthetic label;
  3. the student then trains on gold (measured targets) *and* supplementary
     (synthetic targets), combined by the caller's objective (the signed PPI
     correction, or the gradient gate).

Nothing here knows about GEARS, GNNs, graphs or MLPs: `predict(control, index)`
is the only thing bound to a model, so the same code drives every arm.

The supplementary screen's own measured post-perturbation expression is never
read -- every supplementary target is the teacher's prediction (the design's
rule that synthetic labels, never raw measurements, supervise the extra rows).
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from .contract import PerturbationExample, PerturbationTask

PredictFn = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]


def perturbation_index(
    perturbations: Sequence[Sequence[str]], task: PerturbationTask
) -> torch.Tensor:
    """`(N, P)` padded gene indices for a list of perturbation tuples."""
    width = max((len(perturbation) for perturbation in perturbations), default=1)
    width = max(1, width)
    index = np.zeros((len(perturbations), width), dtype=np.int64)
    for row, perturbation in enumerate(perturbations):
        for slot, gene in enumerate(perturbation):
            position = task.gene_to_index.get(str(gene))
            if position is not None:
                index[row, slot] = position
    return torch.as_tensor(index, dtype=torch.long)


def _predict_chunks(
    predict: PredictFn, control: torch.Tensor, index: torch.Tensor, chunk: int = 1024
) -> torch.Tensor:
    pieces = [
        predict(control[start : start + chunk], index[start : start + chunk])
        for start in range(0, len(control), chunk)
    ]
    if not pieces:
        return torch.zeros((0, 0))
    return torch.cat(pieces, dim=0)


@dataclass
class SyntheticSet:
    """The synthetic training rows, plus the control variate on the gold rows.

    `gold_delta` is the teacher's prediction on the *labeled* training rows: the
    signed correction subtracts its loss so the teacher's bias cancels, and the
    gradient gate ignores it. Both live here because both come from the same
    frozen teacher.
    """

    control: torch.Tensor          # (N, G) supplementary control cells
    index: torch.Tensor            # (N, P) query gene indices
    labels: list[str]              # per-row query perturbation label
    delta: torch.Tensor            # (N, G) teacher's predicted response
    gold_delta: torch.Tensor       # (n_gold, G) teacher's response on gold rows
    n_rows: int

    def summary(self) -> dict[str, Any]:
        return {
            "n_rows": self.n_rows,
            "query_rule": "one gold-training perturbation per supplementary control row, cycled",
            "query_labels": sorted(set(self.labels)),
            "label_source": "teacher-generated synthetic labels (never the source's measured expression)",
        }


def build_synthetic(
    *,
    predict: PredictFn,
    train_examples: Sequence[PerturbationExample],
    task: PerturbationTask,
    controls: np.ndarray | None,
    max_rows: int = 20000,
    seed: int = 42,
) -> SyntheticSet | None:
    """Teacher-label the supplementary control cells for gold-train queries.

    Returns None when there is nothing to label (no control cells), which is how
    an arm falls back to the gold-only update instead of failing.
    """
    if controls is None:
        return None
    controls = np.asarray(controls, dtype=np.float32)
    if controls.size == 0 or len(controls) == 0:
        return None
    labels = sorted({example.label for example in train_examples})
    if not labels:
        return None

    controls = controls[:max_rows]
    n_rows = int(controls.shape[0])
    queries = [tuple(labels[index % len(labels)].split("+")) for index in range(n_rows)]
    supp_control = torch.as_tensor(controls, dtype=torch.float32)
    supp_index = perturbation_index(queries, task)
    supp_delta = _predict_chunks(predict, supp_control, supp_index)

    gold_control = torch.as_tensor(
        np.stack([example.control for example in train_examples]), dtype=torch.float32
    )
    gold_index = perturbation_index(
        [example.perturbation for example in train_examples], task
    )
    gold_delta = _predict_chunks(predict, gold_control, gold_index)

    return SyntheticSet(
        control=supp_control,
        index=supp_index,
        labels=["+".join(query) for query in queries],
        delta=supp_delta,
        gold_delta=gold_delta,
        n_rows=n_rows,
    )
