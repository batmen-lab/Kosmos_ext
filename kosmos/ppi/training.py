"""The training policy every task shares, written once.

A data task differs only in *what* it models -- how its examples are built,
what its base loss is, what its model looks like. How the model is trained does
not differ: the signed PPI correction, the two-stage schedule that warms the
model up on the labeled rows before the correction is switched on, and the two
optimisers that keep the correction's step small are the same for a cell-type
classifier and a perturbation-response model.

This module is deliberately torch-level and task-free. `PPILoss` (the column
task) and `ppi_signed_correction` (the perturbation backend) are both thin
callers of `signed_correction`; both trainers ask the same `correction_stage`
predicate.
"""

from __future__ import annotations

import torch


def correction_stage(
    epoch: int,
    *,
    use_external: bool,
    schedule: str,
    stage1_epochs: int,
    stage2_epochs: int,
) -> bool:
    """Is this epoch the correction stage of a two-stage schedule?

    Mirrors `kosmos/ppi/trainer.py`: stage one trains on the labeled objective
    alone, stage two optimises the correction term on its own, and the two
    alternate so the model never drifts far from what the labels alone say.
    """
    return (
        use_external
        and schedule == "two_stage"
        and (epoch - 1) % (stage1_epochs + stage2_epochs) >= stage1_epochs
    )


def signed_correction(
    *,
    gold: torch.Tensor,
    pseudo_gold: torch.Tensor,
    extension: torch.Tensor,
    coefficient: float = 1.0,
    mass: float = 1.0,
) -> dict[str, torch.Tensor]:
    """The signed PPI correction, the one algebra every task shares.

        L = L_gold + coefficient * (L_extension - mass * L_pseudo_gold)

    `L_pseudo_gold` is the same model scored against the teacher's labels on the
    *labeled* rows, so the two teacher terms cancel in expectation and the
    estimate stays the labeled one while the unlabeled rows shrink its variance.
    Returns the five named terms so callers and the report speak one vocabulary.
    """
    correction = float(coefficient) * (extension - float(mass) * pseudo_gold)
    return {
        "loss_true": gold,
        "loss_pseudo_gold": pseudo_gold,
        "loss_external": extension,
        "loss_correction": correction,
        "loss_total": gold + correction,
    }
