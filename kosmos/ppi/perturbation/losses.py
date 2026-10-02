"""GEARS' objective: a quartic error plus a direction-aware term.

Mirrors `gears/utils.py::loss_fct`: per perturbation group, the error is
`Σ (pred − y)^{2+γ}` with γ=2 (a quartic, so large misses dominate) restricted to
the genes that actually moved (`retained`), plus

    λ · Σ (sign(y − ctrl) − sign(pred − ctrl))²

over the same genes, which pays the model for getting the *direction* of a
change right even when the magnitude is off. Everything here is written in
Δ-space: `y` and `pred` are changes, `ctrl` is zero, and the direction term
compares `sign(Δy)` with `sign(Δpred)`.

`retained` is the DEG set of each perturbation in the batch: the genes whose
observed change is large (or the top-K), which is what GEARS calls `de_idx`.
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
    # The signed-correction vocabulary (`kosmos/ppi/training.py`), filled only
    # by `ppi_signed_correction`; `error`/`direction` above are the historical
    # names for `gold`/`correction`.
    gold: torch.Tensor | None = None
    correction: torch.Tensor | None = None
    pseudo_gold: torch.Tensor | None = None
    extension: torch.Tensor | None = None


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


def gears_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    perturbations: Sequence[str],
    *,
    deg_by_perturbation: dict[str, np.ndarray] | None = None,
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
        retained = None
        if deg_by_perturbation and label in deg_by_perturbation:
            retained = deg_by_perturbation[label]
        if retained is not None and len(retained):
            keep = torch.as_tensor(retained, dtype=torch.long, device=device)
            predicted = predicted[:, keep]
            observed = observed[:, keep]
        scale = predicted.shape[0] * max(1, predicted.shape[1])
        error_total = error_total + ((predicted - observed).abs() ** error_exponent).sum() / scale
        direction_total = direction_total + (
            direction_lambda
            * ((torch.sign(predicted) - torch.sign(observed)) ** 2).sum()
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


def ppi_signed_correction(
    *,
    gold: torch.Tensor,
    pseudo_gold: torch.Tensor,
    extension: torch.Tensor,
    coefficient: float,
    mass: float,
) -> "LossBreakdown":
    """The single-cell signed correction, on the perturbation objective.

    `losses.PPILoss` in the column task is

        L = L_true + λ · (L_ext − mass · L_pseudo)

    where `L_true` is the model against the *measured* labels, `L_pseudo` is the
    same model against the **teacher's** labels on the same labeled rows (the
    control variate), and `L_ext` is the model against the teacher's labels on
    the unlabeled rows. The two teacher terms cancel in expectation, so the
    estimate stays the labeled one while the unlabeled rows shrink its variance.

    Nothing here is specific to a model or a graph: `gold`, `pseudo_gold` and
    `extension` are three values of the same loss function, computed on three
    different (input, target) pairs. That is what makes the mechanism reusable
    for the GEARS arms and the graph-free MLP arms alike.
    """
    from ..training import signed_correction

    terms = signed_correction(
        gold=gold,
        pseudo_gold=pseudo_gold,
        extension=extension,
        coefficient=coefficient,
        mass=mass,
    )
    return LossBreakdown(
        total=terms["loss_total"],
        error=terms["loss_true"],
        direction=terms["loss_correction"],
        n_groups=1,
        gold=terms["loss_true"],
        correction=terms["loss_correction"],
        pseudo_gold=terms["loss_pseudo_gold"],
        extension=terms["loss_external"],
    )
