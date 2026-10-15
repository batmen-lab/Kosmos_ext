"""GEARS (`gears/model.py::GEARS_Model`) in plain torch, plus two optional extensions.

The port follows the original layer for layer (no PyTorch Geometric: `SGConv`
with K=1 is `Linear(Â x)` with PyG's GCN normalisation, see `sparse_adjacency`):

  * gene embedding -> BatchNorm -> ReLU, plus `0.2 ×` a positional embedding
    propagated over the **co-expression graph** (`G_C`), then an MLP;
  * a **perturbation embedding** over its own vocabulary (the GO graph's genes,
    not the measured panel), propagated over the **GO graph** (`G_GO`); a cell's
    perturbation is the sum of its genes' embeddings, through `pert_fuse`, added
    to every gene of that cell;
  * BatchNorm -> ReLU -> `recovery_w` MLP -> gene-specific weights (`indv_w1`,
    `indv_b1`) -> a cross-gene MLP over all genes -> gene-specific output
    (`indv_w2`, `indv_b2`) = Δ̂, and the prediction is `control + Δ̂`.

Extensions, both off in the original:

  * `auxiliary_adjacency` (`G_S`, the signed PPI arm): the positional embedding
    is also propagated over the supplementary co-expression graph with the same
    layer, `pos = SG(G_C) + η·SG(G_S)`;
  * `condition_on_control`: each gene's control expression is embedded
    (`x_g · w_g`) and added to that gene's state, so Δ̂ depends on the input
    cell. In the original, Δ̂ is a function of the perturbation alone.

Optional context (cell type, ...) embeddings are added to every gene's state.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import torch
from torch import nn


def sparse_adjacency(
    edge_index: torch.Tensor, edge_weight: torch.Tensor, n_nodes: int
) -> torch.Tensor:
    """PyG's `gcn_norm` as a sparse matrix `M` with `M @ x` = one SGConv step.

    `edge_index[0]` is the source and `edge_index[1]` the target (GEARS' edge
    lists: the top-k sources of every target). Self loops are added where a node
    has none (`add_remaining_self_loops`, weight 1), the degree is the weighted
    in-degree of the target, and messages flow source -> target:
    `M[target, source] = w / sqrt(deg[source] · deg[target])`.
    """
    sources, targets = edge_index[0].long(), edge_index[1].long()
    weights = edge_weight.float()
    has_loop = torch.zeros(n_nodes, dtype=torch.bool)
    has_loop[sources[sources == targets]] = True
    missing = torch.nonzero(~has_loop).reshape(-1)
    sources = torch.cat([sources, missing])
    targets = torch.cat([targets, missing])
    weights = torch.cat([weights, torch.ones(len(missing))])
    degree = torch.zeros(n_nodes).index_add_(0, targets, weights)
    scale = degree.clamp_min(1e-12).pow(-0.5)
    values = scale[sources] * weights * scale[targets]
    indices = torch.stack([targets, sources])
    return torch.sparse_coo_tensor(indices, values, (n_nodes, n_nodes)).coalesce()


class SGConv(nn.Module):
    """`torch_geometric.nn.SGConv(dim, dim, K=1)`: propagate once, then a linear map."""

    def __init__(self, dim: int):
        super().__init__()
        self.lin = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        return self.lin(torch.sparse.mm(adjacency, x))


def _mlp(sizes: list[int]) -> nn.Sequential:
    """GEARS' `MLP(sizes, batch_norm=True)`: Linear-BN-ReLU blocks minus the last ReLU."""
    layers: list[nn.Module] = []
    for index in range(len(sizes) - 1):
        layers += [nn.Linear(sizes[index], sizes[index + 1]), nn.BatchNorm1d(sizes[index + 1]), nn.ReLU()]
    return nn.Sequential(*layers[:-1])


def _batched(module: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """Apply a BatchNorm-containing module; a batch of one is duplicated (GEARS does)."""
    if x.shape[0] == 1 and module.training:
        return module(torch.cat([x, x]))[:1]
    return module(x)


@dataclass
class ModelConfig:
    n_genes: int
    #: size of the perturbation vocabulary; None = the panel (n_genes)
    n_perturbations: int | None = None
    hidden_dim: int = 64
    go_layers: int = 1
    gene_layers: int = 1
    eta: float = 1.0
    context_categorical: Mapping[str, int] | None = None
    context_numeric: int = 0
    condition_on_control: bool = True


class GearsModel(nn.Module):
    """GEARS' Δ predictor; see the module docstring."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        n, hidden = config.n_genes, config.hidden_dim
        n_perts = config.n_perturbations or n
        self.gene_emb = nn.Embedding(n, hidden, max_norm=True)
        self.emb_pos = nn.Embedding(n, hidden, max_norm=True)
        self.pert_emb = nn.Embedding(n_perts, hidden, max_norm=True)
        self.bn_emb = nn.BatchNorm1d(hidden)
        self.emb_trans_v2 = _mlp([hidden, hidden, hidden])
        self.layers_emb_pos = nn.ModuleList(SGConv(hidden) for _ in range(config.gene_layers))
        self.sim_layers = nn.ModuleList(SGConv(hidden) for _ in range(config.go_layers))
        self.pert_fuse = _mlp([hidden, hidden, hidden])
        self.bn_pert_base = nn.BatchNorm1d(hidden)
        self.recovery_w = _mlp([hidden, hidden * 2, hidden])
        self.indv_w1 = nn.Parameter(torch.empty(n, hidden, 1))
        self.indv_b1 = nn.Parameter(torch.empty(n, 1))
        self.cross_gene_state = _mlp([n, hidden, hidden])
        self.indv_w2 = nn.Parameter(torch.empty(1, n, hidden + 1))
        self.indv_b2 = nn.Parameter(torch.empty(1, n))
        for parameter in (self.indv_w1, self.indv_b1, self.indv_w2, self.indv_b2):
            nn.init.xavier_normal_(parameter)
        self.expression_weight = (
            nn.Parameter(torch.randn(n, hidden) * 0.02) if config.condition_on_control else None
        )
        self.context_embeddings = nn.ModuleDict(
            {
                name: nn.Embedding(int(cardinality), hidden)
                for name, cardinality in (config.context_categorical or {}).items()
            }
        )
        self.context_numeric = (
            nn.Linear(config.context_numeric, hidden) if config.context_numeric else None
        )

    def _propagate(self, x: torch.Tensor, layers: nn.ModuleList, adjacency: torch.Tensor) -> torch.Tensor:
        for index, layer in enumerate(layers):
            x = layer(x, adjacency)
            if index < len(layers) - 1:
                x = x.relu()
        return x

    def forward(
        self,
        control: torch.Tensor,                 # (B, G) control expression
        perturbation_index: torch.Tensor,      # (B, P) vocabulary indices, -1 = padding
        gold_adjacency: torch.Tensor,
        auxiliary_adjacency: torch.Tensor | None = None,
        go_adjacency: torch.Tensor | None = None,
        context: dict[str, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns `(control + Δ̂, Δ̂)`."""
        batch, n = control.shape
        device = control.device
        genes = torch.arange(n, device=device)
        base = torch.relu(self.bn_emb(self.gene_emb(genes)))                  # (G, H)
        positions = self.emb_pos(genes)
        pos = self._propagate(positions, self.layers_emb_pos, gold_adjacency)
        if auxiliary_adjacency is not None:
            pos = pos + self.config.eta * self._propagate(positions, self.layers_emb_pos, auxiliary_adjacency)
        base = self.emb_trans_v2(base + 0.2 * pos)                           # (G, H)

        perts = self.pert_emb(torch.arange(self.pert_emb.num_embeddings, device=device))
        if go_adjacency is not None:
            perts = self._propagate(perts, self.sim_layers, go_adjacency)
        valid = perturbation_index >= 0
        chosen = perts[perturbation_index.clamp_min(0)] * valid.unsqueeze(-1)
        total = chosen.sum(dim=1)                                             # (B, H)
        has_perturbation = valid.any(dim=1)
        added = torch.zeros_like(total)
        if has_perturbation.any():
            added[has_perturbation] = _batched(self.pert_fuse, total[has_perturbation])

        states = base.unsqueeze(0) + added.unsqueeze(1)                       # (B, G, H)
        if self.expression_weight is not None:
            states = states + control.unsqueeze(-1) * self.expression_weight
        if context:
            extra = torch.zeros_like(total)
            for name, values in context.items():
                if name in self.context_embeddings:
                    extra = extra + self.context_embeddings[name](values)
            numeric = [values for name, values in context.items() if name not in self.context_embeddings]
            if numeric and self.context_numeric is not None:
                extra = extra + self.context_numeric(torch.cat(numeric, dim=-1).float())
            states = states + extra.unsqueeze(1)

        hidden = states.reshape(batch * n, -1)
        hidden = torch.relu(self.bn_pert_base(hidden))
        out = self.recovery_w(hidden).reshape(batch, n, -1)                  # (B, G, H)
        out = (out.unsqueeze(-1) * self.indv_w1).sum(dim=2) + self.indv_b1    # (B, G, 1)
        cross = _batched(self.cross_gene_state, out.squeeze(-1))              # (B, H)
        cross = cross.unsqueeze(1).expand(-1, n, -1)
        delta = (torch.cat([out, cross], dim=2) * self.indv_w2).sum(dim=2) + self.indv_b2
        return control + delta, delta
