"""A plain-torch GEARS-shaped model: gene embeddings, GNN, compositional encoder.

The pieces are GEARS' (`gears/model.py`), re-implemented without PyTorch
Geometric so it runs on CPU in this repo's environment:

  * a learnable **gene embedding** `E`, the same for every graph;
  * **GNN** message passing over a gene graph (co-expression, and optionally an
    auxiliary co-expression graph), which is how a gene's representation borrows
    from the genes it covaries with. The gold and auxiliary branches **share the
    GNN parameters** -- they are the same relation type measured on different
    cells -- and the augmented representation is `H_A = H_C + η·H_S`;
  * a **compositional module**: the perturbation is encoded from the sum of its
    perturbed genes' embeddings, so a two-gene perturbation is a composition of
    the two rather than a new vocabulary item;
  * a **cross-gene decoder** over the per-gene scalar (the shape GEARS uses:
    `Linear(n_genes -> hidden)` over the gene axis), and **gene-specific output
    layers** (`gene_out`, `gene_bias`) producing one number per gene;
  * optional **context** (cell type, dose, time, donor, batch) added to the
    perturbation encoding.

The model predicts the **change**, and the post-perturbation expression is
`control + Δ`, so the control profile is always part of the prediction.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import torch
from torch import nn


def sparse_adjacency(
    edge_index: torch.Tensor, edge_weight: torch.Tensor, n_nodes: int
) -> torch.Tensor:
    """Symmetric normalised adjacency `D^-1/2 (A+I) D^-1/2` as a sparse tensor."""
    loops = torch.arange(n_nodes, dtype=torch.long)
    rows = torch.cat([edge_index[0], loops])
    columns = torch.cat([edge_index[1], loops])
    weights = torch.cat([edge_weight, torch.ones(n_nodes, dtype=edge_weight.dtype)])
    degrees = torch.zeros(n_nodes, dtype=weights.dtype)
    degrees.index_add_(0, rows, weights)
    degrees = degrees.clamp_min(1e-12)
    scale = degrees.pow(-0.5)
    normalised = weights * scale[rows] * scale[columns]
    indices = torch.stack([rows, columns])
    return torch.sparse_coo_tensor(indices, normalised, (n_nodes, n_nodes)).coalesce()


class _GcnLayer(nn.Module):
    """One message-passing step with a residual and a layer norm."""

    def __init__(self, dim: int):
        super().__init__()
        self.linear = nn.Linear(dim, dim)
        self.norm = nn.LayerNorm(dim)

    def forward(self, hidden: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        message = torch.sparse.mm(adjacency, hidden)
        return self.norm(hidden + torch.relu(self.linear(message)))


@dataclass
class ModelConfig:
    n_genes: int
    embedding_dim: int = 64
    hidden_dim: int = 64
    gnn_layers: int = 2
    eta: float = 1.0
    context_categorical: Mapping[str, int] | None = None
    context_numeric: int = 0
    dropout: float = 0.1


class GearsModel(nn.Module):
    """GEARS-shaped predictor of Δy, with one or two co-expression graphs."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        n = config.n_genes
        self.gene_emb = nn.Embedding(n, config.embedding_dim)
        nn.init.normal_(self.gene_emb.weight, std=0.02)
        self.gnn = nn.ModuleList(
            [_GcnLayer(config.embedding_dim) for _ in range(config.gnn_layers)]
        )
        self.perturbation_encoder = nn.Sequential(
            nn.Linear(config.embedding_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, config.hidden_dim),
        )
        # `states` keeps the embedding width (the GNN does not change it) while
        # the perturbation encoder outputs `hidden_dim`; the two need not be the
        # same size, so the fusion layer takes their sum rather than 2*hidden.
        self.fuse = nn.Sequential(
            nn.Linear(config.embedding_dim + config.hidden_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout),
        )
        self.cross_gene = nn.Sequential(
            nn.Linear(n, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, config.hidden_dim),
        )
        # gene-specific output layers: one weight vector and one bias per gene
        self.gene_in = nn.Parameter(torch.randn(n, config.hidden_dim) * 0.02)
        self.gene_out = nn.Parameter(torch.randn(n, config.hidden_dim) * 0.02)
        self.gene_bias = nn.Parameter(torch.zeros(n))
        self.context_embeddings = nn.ModuleDict(
            {
                name: nn.Embedding(int(cardinality), config.hidden_dim)
                for name, cardinality in (config.context_categorical or {}).items()
            }
        )
        self.context_numeric = (
            nn.Linear(config.context_numeric, config.hidden_dim)
            if config.context_numeric
            else None
        )

    # -- graph branches -----------------------------------------------------

    def gene_states(self, gold_adjacency: torch.Tensor, auxiliary_adjacency: torch.Tensor | None = None):
        """`H_C` (gold graph alone) or `H_A = H_C + η·H_S` when a second graph is given."""
        hidden = self.gene_emb.weight
        for layer in self.gnn:
            hidden = layer(hidden, gold_adjacency)
        if auxiliary_adjacency is None:
            return hidden
        supplementary = self.gene_emb.weight
        for layer in self.gnn:
            supplementary = layer(supplementary, auxiliary_adjacency)
        return hidden + self.config.eta * supplementary

    # -- forward ------------------------------------------------------------

    def forward(
        self,
        control: torch.Tensor,                 # (B, G) control expression
        perturbation_index: torch.Tensor,      # (B, P) perturbed gene indices
        gold_adjacency: torch.Tensor,
        auxiliary_adjacency: torch.Tensor | None = None,
        go_adjacency: torch.Tensor | None = None,
        context: dict[str, torch.Tensor] | None = None,
        gene_states: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns `(post_perturbation_expression, delta)` for the batch.

        The GO graph feeds the **perturbation branch**, the way GEARS uses it:
        the perturbed gene's representation is pooled with its GO-similar genes,
        so `G_C + G_GO` (base) and `G_C + G_S + G_GO` (augmented) are the two
        graphs each pass sees.
        """
        states = gene_states if gene_states is not None else self.gene_states(
            gold_adjacency, auxiliary_adjacency
        )
        batch = control.shape[0]
        pooled = states
        if go_adjacency is not None:
            pooled = states + torch.sparse.mm(go_adjacency, states)
        # compositional encoding: sum of the perturbed genes' states, then MLP
        chosen = pooled[perturbation_index.reshape(-1)].reshape(
            batch, perturbation_index.shape[1], -1
        )
        encoded = self.perturbation_encoder(chosen.sum(dim=1))
        if context:
            for name, values in context.items():
                if name in self.context_embeddings:
                    encoded = encoded + self.context_embeddings[name](values)
            numeric = [values for name, values in context.items() if name not in self.context_embeddings]
            if numeric and self.context_numeric is not None:
                encoded = encoded + self.context_numeric(torch.stack(numeric, dim=1))
        fused = self.fuse(torch.cat([states.unsqueeze(0).expand(batch, -1, -1), encoded.unsqueeze(1).expand(-1, states.shape[0], -1)], dim=-1))
        per_gene = (self.gene_in.unsqueeze(0) * fused).sum(dim=-1)          # (B, G)
        cross = self.cross_gene(per_gene)                                   # (B, hidden)
        delta = (self.gene_out.unsqueeze(0) * cross.unsqueeze(1)).sum(dim=-1) + self.gene_bias
        return control + delta, delta
