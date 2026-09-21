"""Gold-anchored, gradient-gated synthetic training.

The labeled rows decide the direction; the synthetic rows may only accelerate
it. Every step computes two gradients -- one from the gold minibatch, one from
the synthetic minibatch -- and combines them by how well they agree:

    g_final = g_G + λ_t · g_S,   λ_t = max(0, cos(g_G, g_S)) · min(1, κ‖g_G‖ / ‖g_S‖)

The properties that follow from that definition, and that the tests hold the
implementation to:

  * **Gold alone is the fallback.** `cos ≤ 0` gives `λ_t = 0`, so `g_final` is
    exactly the gold gradient. Misleading synthetic data cannot push the model
    away from the supervised direction, it can only be ignored.
  * **Synthetic never dominates.** `min(1, κ‖g_G‖/‖g_S‖)` caps its magnitude at
    κ times the gold gradient, so a synthetic batch with huge loss (a wrongly
    scaled cohort, a few extreme rows) cannot swamp the labeled signal. κ = 1
    means "at most as loud as gold".
  * **The gate is not differentiable.** Cosine and the scale are computed from
    *detached* gradient vectors and combined by arithmetic, not by summing
    losses with a coefficient that stays in the graph. A coefficient the model
    could push on through higher-order derivatives would be an objective, not a
    controller -- the model would learn to make synthetic data *look*
    compatible.

Two scopes:

  * `batch` (default) -- one synthetic gradient per step, gated against the
    gold one. Two backward passes, no per-sample loops, and the behaviour above
    holds exactly.
  * `sample` -- each synthetic row is gated on its own agreement with the gold
    gradient, `w_j = max(0, cos(g_j, g_G))^γ`, and the accepted rows are
    averaged before the same magnitude clip. It costs one backward per row and
    is the honest reading of "ignore the rows that disagree", so it is available
    when a caller asks for it rather than applied by default.

This module knows nothing about tasks, tables or models: it takes two losses
and a parameter list, and returns the gradient to apply.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Literal

import torch

#: Keeps a zero-norm gradient from turning a cosine into a division by zero.
EPS = 1e-12

GateScope = Literal["batch", "sample"]


@dataclass
class GateStats:
    """What one gated step saw, so the behaviour can be inspected afterwards."""

    cosine: float = 0.0
    reliability: float = 0.0
    scale: float = 0.0
    weight: float = 0.0
    gold_norm: float = 0.0
    synthetic_norm: float = 0.0
    accepted_norm: float = 0.0
    gold_loss: float = 0.0
    synthetic_loss: float = 0.0
    accepted_fraction: float = 1.0
    rows: int = 0

    @property
    def active(self) -> bool:
        return self.weight > 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _flat(gradients, parameters) -> torch.Tensor:
    """Every parameter's gradient as one vector, missing ones as zeros."""
    pieces = []
    for gradient, parameter in zip(gradients, parameters, strict=True):
        pieces.append(
            torch.zeros_like(parameter).reshape(-1)
            if gradient is None
            else gradient.detach().reshape(-1)
        )
    if not pieces:
        return torch.zeros(0)
    return torch.cat(pieces)


def _assign(vector: torch.Tensor, parameters) -> None:
    """Write a flat gradient back into `p.grad`, view by view."""
    offset = 0
    for parameter in parameters:
        size = parameter.numel()
        piece = vector[offset : offset + size].reshape(parameter.shape)
        parameter.grad = piece.clone()
        offset += size


class GradientGate:
    """Combines the gold and synthetic gradients, with the gold one in charge."""

    def __init__(
        self,
        *,
        kappa: float = 1.0,
        scope: GateScope = "batch",
        gamma: float = 1.0,
        coefficient: float = 1.0,
        eps: float = EPS,
    ):
        self.kappa = float(kappa)
        self.scope = "sample" if str(scope) == "sample" else "batch"
        self.gamma = float(gamma)
        self.coefficient = float(coefficient)
        self.eps = float(eps)

    # -- the gate itself ----------------------------------------------------

    def combine(
        self,
        gold_loss: torch.Tensor,
        synthetic_loss: torch.Tensor | None,
        parameters,
        *,
        synthetic_per_row=None,
    ) -> GateStats:
        """Set `p.grad` to `g_G + λ_t·g_S` and report what the gate did.

        `synthetic_loss` is None when the step has no synthetic rows at all: the
        gold gradient is then applied unchanged and the stats say so (`active`
        False, cosine 0).
        """
        parameters = [parameter for parameter in parameters if parameter.requires_grad]
        gold_gradients = torch.autograd.grad(
            gold_loss, parameters, retain_graph=True, allow_unused=True
        )
        gold_vector = _flat(gold_gradients, parameters)
        gold_norm = float(gold_vector.norm())
        stats = GateStats(
            gold_norm=gold_norm,
            gold_loss=float(gold_loss.detach()),
        )
        if synthetic_loss is None:
            _assign(gold_vector, parameters)
            stats.weight = 0.0
            stats.accepted_fraction = 0.0
            return stats

        stats.synthetic_loss = float(synthetic_loss.detach())
        if self.scope == "sample" and synthetic_per_row is not None:
            accepted, synthetic_norm, fraction, cosine = self._sample_gradient(
                synthetic_per_row, parameters, gold_vector
            )
            stats.cosine = cosine
            stats.reliability = max(0.0, cosine)
            stats.synthetic_norm = synthetic_norm
            stats.accepted_fraction = fraction
        else:
            synthetic_gradients = torch.autograd.grad(
                synthetic_loss, parameters, retain_graph=False, allow_unused=True
            )
            synthetic_vector = _flat(synthetic_gradients, parameters)
            accepted = synthetic_vector
            stats.synthetic_norm = float(synthetic_vector.norm())
            stats.cosine = self._cosine(gold_vector, synthetic_vector)
            stats.reliability = max(0.0, stats.cosine)

        # Magnitude control: the accepted synthetic gradient is never louder than
        # κ times the gold one, however large the synthetic loss is.
        accepted_norm = float(accepted.norm())
        stats.accepted_norm = accepted_norm
        limit = self.kappa * gold_norm
        stats.scale = (
            min(1.0, limit / (accepted_norm + self.eps)) if accepted_norm > 0 else 0.0
        )
        stats.weight = (
            stats.reliability * stats.scale * self.coefficient
            if self.scope == "batch"
            else (stats.scale * self.coefficient if accepted_norm > 0 else 0.0)
        )
        final = gold_vector + stats.weight * accepted.to(gold_vector.dtype)
        _assign(final, parameters)
        return stats

    # -- helpers ------------------------------------------------------------

    def _cosine(self, gold: torch.Tensor, synthetic: torch.Tensor) -> float:
        denominator = float(gold.norm()) * float(synthetic.norm())
        if denominator <= self.eps:
            # No synthetic signal to be compatible or incompatible with.
            return 0.0
        return float(torch.dot(gold, synthetic)) / denominator

    def _sample_gradient(self, per_row, parameters, gold_vector):
        """Gate each synthetic row on its own agreement with the gold gradient.

        Rows that disagree contribute nothing; the rest are averaged, weighted
        by `cos^γ`, and then clipped as a whole against the gold gradient.
        """
        cosines: list[float] = []
        vectors: list[torch.Tensor] = []
        rows = len(per_row)
        for index in range(rows):
            gradients = torch.autograd.grad(
                per_row[index], parameters, retain_graph=index < rows - 1, allow_unused=True
            )
            vector = _flat(gradients, parameters)
            vectors.append(vector)
            cosines.append(self._cosine(gold_vector, vector))
        weights = [max(0.0, cosine) ** self.gamma for cosine in cosines]
        total = sum(weights)
        fraction = float(sum(1 for weight in weights if weight > 0) / rows) if rows else 0.0
        cosine = float(sum(cosines) / rows) if rows else 0.0
        if total <= self.eps or not vectors:
            return torch.zeros_like(gold_vector), 0.0, 0.0, cosine
        stacked = torch.stack(vectors)
        weight_tensor = torch.tensor(weights, dtype=stacked.dtype)
        accepted = (stacked * weight_tensor.unsqueeze(1)).sum(dim=0) / (total + self.eps)
        return accepted, float(accepted.norm()), fraction, cosine


def summarise(stats: list[GateStats]) -> dict[str, Any]:
    """Mean and spread of what the gate did over one epoch.

    This is the diagnostic the stress test reads: if the mechanism works, the
    cosine and the synthetic weight fall as the synthetic labels get worse, and
    `active_fraction` -- the share of steps that used synthetic data at all --
    falls with them.
    """
    if not stats:
        return {"steps": 0}
    import numpy as np

    cosine = np.array([s.cosine for s in stats], dtype=float)
    weight = np.array([s.weight for s in stats], dtype=float)
    active = np.array([s.active for s in stats], dtype=float)
    return {
        "steps": len(stats),
        "gold_grad_norm": float(np.mean([s.gold_norm for s in stats])),
        "synthetic_grad_norm": float(np.mean([s.synthetic_norm for s in stats])),
        "accepted_grad_norm": float(np.mean([s.accepted_norm for s in stats])),
        "gradient_cosine_mean": float(cosine.mean()),
        "gradient_cosine_std": float(cosine.std()),
        "synthetic_weight_mean": float(weight.mean()),
        "synthetic_weight_std": float(weight.std()),
        "synthetic_active_fraction": float(active.mean()),
        "train_gold_loss": float(np.mean([s.gold_loss for s in stats])),
        "train_synthetic_loss": float(np.mean([s.synthetic_loss for s in stats])),
    }
