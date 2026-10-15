"""GEARS' objective: a quartic error plus a direction-aware term.

Mirrors `gears/utils.py::loss_fct`: per perturbation group, the error is
`mean (pred − y)^{2+γ}` with γ=2 over the group's cells and its retained genes,
plus

    λ · mean (sign(y − ctrl) − sign(pred − ctrl))²

over the same genes, averaged over groups.

GEARS' `pred` and `y` are expressions: `pred = x + Δ̂` for the paired control
cell `x`, and `ctrl` is the control *mean*. Here everything is in Δ-space
(`prediction = Δ̂`, `target = y − x`), so the error term is unchanged and the
direction term is written with `offset = x − ctrl`:
`sign(target + offset) = sign(y − ctrl)`,
`sign(prediction + offset) = sign(x + Δ̂ − ctrl)`. `offset=None` means the
input was the control mean itself (offset 0).

`retained` is GEARS' `dict_filter` (`non_zeros_gene_idx`): the genes whose mean
expression over that perturbation's cells is non-zero -- nearly the whole
panel, **not** a top-K DEG set. `retained_genes` builds it.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import torch

#: GEARS' quartic exponent (`2 + gamma`, gamma = 2) and its direction weight.
ERROR_EXPONENT = 4
DEFAULT_DIRECTION_LAMBDA = 1e-1


@dataclass
class LossBreakdown:
    total: torch.Tensor
    error: torch.Tensor
    direction: torch.Tensor
    n_groups: int


def deg_indices(
    delta: np.ndarray, *, top_k: int | None = None, threshold: float | None = None
) -> np.ndarray:
    """The genes that moved: top-K by |Δ|, or everything above `threshold`."""
    magnitude = np.abs(np.asarray(delta, dtype=np.float64))
    if top_k is not None:
        k = max(1, min(int(top_k), magnitude.size))
        return np.sort(np.argpartition(-magnitude, k - 1)[:k])
    if threshold is not None:
        chosen = np.nonzero(magnitude >= float(threshold))[0]
        if chosen.size:
            return chosen
    return np.nonzero(magnitude > 0)[0] if magnitude.size else np.array([], dtype=int)


def retained_genes(expression_by_perturbation: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """GEARS' `non_zeros_gene_idx`: genes with non-zero mean expression, per perturbation.

    `expression_by_perturbation` maps a perturbation to its cells' expression
    `(n, G)` (y, not Δ).
    """
    return {
        label: np.nonzero(np.asarray(cells, dtype=np.float64).mean(axis=0) != 0)[0]
        for label, cells in expression_by_perturbation.items()
    }


def gears_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    perturbations: Sequence[str],
    *,
    genes_by_perturbation: dict[str, np.ndarray] | None = None,
    offset: torch.Tensor | None = None,
    direction_lambda: float = DEFAULT_DIRECTION_LAMBDA,
    error_exponent: int = ERROR_EXPONENT,
) -> LossBreakdown:
    """Per-perturbation quartic error + direction loss, averaged over groups."""
    if prediction.shape != target.shape:
        raise ValueError("prediction and target must have the same shape")
    device = prediction.device
    zero = torch.zeros((), device=device)
    error_total = zero.clone()
    direction_total = zero.clone()
    groups = 0
    labels = np.asarray(list(perturbations))
    for label in sorted(set(labels.tolist())):
        rows = np.where(labels == label)[0]
        index = torch.as_tensor(rows, dtype=torch.long, device=device)
        predicted = prediction[index]
        observed = target[index]
        shift = offset[index] if offset is not None else torch.zeros_like(observed)
        retained = None
        if genes_by_perturbation and label in genes_by_perturbation:
            retained = genes_by_perturbation[label]
        if retained is not None and len(retained):
            keep = torch.as_tensor(retained, dtype=torch.long, device=device)
            predicted = predicted[:, keep]
            observed = observed[:, keep]
            shift = shift[:, keep]
        scale = predicted.shape[0] * max(1, predicted.shape[1])
        error_total = error_total + ((predicted - observed).abs() ** error_exponent).sum() / scale
        direction_total = direction_total + (
            direction_lambda
            * ((torch.sign(predicted + shift) - torch.sign(observed + shift)) ** 2).sum()
            / scale
        )
        groups += 1
    if groups == 0:
        raise ValueError("no perturbation group in this batch, so no loss to compute")
    total = (error_total + direction_total) / groups
    return LossBreakdown(
        total=total,
        error=error_total / groups,
        direction=direction_total / groups,
        n_groups=groups,
    )
