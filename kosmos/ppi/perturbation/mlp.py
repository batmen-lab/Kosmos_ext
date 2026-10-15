"""Graph-free MLP for perturbation response: the baseline architecture.

Two separate ReLU encoders (control expression -> hidden, perturbed-gene
indicator -> hidden) are **added**, then a linear head predicts Δ. No graph, GO
term or GNN is consumed, so the GEARS and MLP arms differ only by architecture:
both are trained by `train.py` on the same batches, losses and schedule.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import torch
from torch import nn

MLP_ARMS = ("mlp_base", "mlp_augmented_ungated", "mlp_gated")


@dataclass
class MlpConfig:
    """Shape of the baseline. Defaults are the design's benchmark settings."""

    n_genes: int
    hidden_size: int = 128
    #: size of the perturbation vocabulary (one-hot width); None = n_genes
    n_perturbations: int | None = None
    context_categorical: Mapping[str, int] | None = None
    context_numeric: int = 0


class MlpModel(nn.Module):
    """`Δ = W_o (ReLU(W_c x_ctrl + b_c) + ReLU(W_p p + b_p)) + b_o`."""

    def __init__(self, config: MlpConfig):
        super().__init__()
        self.config = config
        self.control_encoder = nn.Linear(config.n_genes, config.hidden_size)
        self.perturbation_encoder = nn.Linear(
            config.n_perturbations or config.n_genes, config.hidden_size
        )
        self.output = nn.Linear(config.hidden_size, config.n_genes)
        self.context_embeddings = nn.ModuleDict(
            {
                name: nn.Embedding(int(cardinality), config.hidden_size)
                for name, cardinality in (config.context_categorical or {}).items()
            }
        )
        self.context_numeric = (
            nn.Linear(config.context_numeric, config.hidden_size)
            if config.context_numeric
            else None
        )

    def forward(
        self,
        control: torch.Tensor,                 # (B, G) control expression
        indicator: torch.Tensor,               # (B, G) perturbed-gene indicator
        context: dict[str, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        hidden = torch.relu(self.control_encoder(control)) + torch.relu(
            self.perturbation_encoder(indicator)
        )
        if context:
            for name, values in context.items():
                if name in self.context_embeddings:
                    hidden = hidden + self.context_embeddings[name](values)
            numeric = [
                values for name, values in context.items() if name not in self.context_embeddings
            ]
            if numeric and self.context_numeric is not None:
                hidden = hidden + self.context_numeric(torch.cat(numeric, dim=-1).float())
        return self.output(hidden)


def indicator_from_index(index: torch.Tensor, n_genes: int) -> torch.Tensor:
    """`(B, V)` 0/1 indicator from the `(B, P)` vocabulary-index form (-1 = padding)."""
    indicator = torch.zeros((index.shape[0], n_genes + 1), dtype=torch.float32, device=index.device)
    # padding (-1) lands in the extra last column, which is dropped
    indicator.scatter_(1, torch.where(index < 0, n_genes, index), 1.0)
    return indicator[:, :n_genes]
