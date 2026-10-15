"""Three arms per architecture: gold-only, two-stage signed PPI, gradient-gated.

    arm                        architecture  objective
    gears_base                 GEARS         L_true                    (also the teacher)
    gears_augmented_ungated    GEARS + G_S   two-stage signed PPI, from scratch
    gears_gated                GEARS + G_S   L_true, plus L_u where its gradient agrees
    gears_finetune_ppi         GEARS         two-stage signed PPI, starting from the
                                             trained base (not in the default set)
    mlp_base                   MLP           L_true                    (also the teacher)
    mlp_augmented_ungated      MLP           two-stage signed PPI
    mlp_gated                  MLP           L_true, plus L_u where its gradient agrees

**The gated arms** (`gatedlossdesign.md`, batch level). Every step computes the
gold gradient `g_G = ∇L_true` and the synthetic gradient `g_S = ∇L_u` (the
supplementary cells against the frozen base arm's Δ, as in stage 2 below), then

    r = max(0, cos(g_G, g_S))          gold compatibility (0 when they conflict)
    c = min(1, κ‖g_G‖ / ‖g_S‖)         g_S never larger than κ × g_G
    g = g_G + r·c·g_S                  then value clipping and the Adam step

`r` and `c` are computed from detached gradients, so the gate is a controller,
not something the model can optimise. With `r = 0` the step is exactly the
gold-only step. There is no stage cycle and no L_pg; κ = `gated_kappa` (1).

Every arm runs through the same loop (`_train_arm`); the architecture is hidden
behind `_Net.delta(model, control, index, context) -> Δ`.

**The losses.** All three are `gears_loss`, the GEARS paper's objective: a
quartic error plus a direction term, over every gene with non-zero mean
expression in that perturbation's training cells (GEARS' `non_zeros_gene_idx`,
`losses.retained_genes`), with the direction measured against the control
mean. They differ only in the (input, target) pair:

    L_true  gold cells,          measured Δ
    L_pg    the same gold cells, the teacher's Δ
    L_u     supplementary cells, the teacher's Δ

**The schedule.** Epochs cycle `stage1_epochs` (5) of stage 1, then
`stage2_epochs` (1) of stage 2 (scDesign3 `src/train_ppi.py`):

    stage 1   minimise L_true                Adam(model, lr)
    stage 2   minimise λ·(L_u − L_pg)        model weights: SGD, lr × stage2_lr_multiplier (÷10)
                                             λ:             Adam, lr × lambda_lr_multiplier (×10)

λ is learned: `λ = λ_min + (λ_max − λ_min)·sigmoid(raw)`, initialised at
`lambda_init` (1 in [0, 2]). Its gradient is `L_u − L_pg`.

**The stage-2 step is projected.** On its own, `λ·(L_u − L_pg)` has no lower
bound: the student can blow up both L_u and L_pg as long as L_pg grows faster
(an Adamson MLP run reached L_pg ≈ 9e12 with λ pinned at its ceiling). So the
model-weight gradient `g = ∇λ(L_u − L_pg)` is replaced by the closest `d` with
`⟨d, ∇L_u⟩ ≥ 0` and `⟨d, ∇L_true⟩ ≥ 0` (`project_to_descent`; L_true is
computed from the same gold forward as L_pg): to first order the step never
raises L_u or L_true, and the −L_pg correction only acts in directions that
do not conflict with them. The step is plain SGD without weight decay so the
guarantee survives the optimiser (Adam rescales each coordinate and would not
preserve it); value clipping is applied to the three gradients before the
projection, not after.

**Inputs.** A gold row is a perturbed cell `y_i` paired with one real control
cell `x_c`, drawn at random (per epoch) from the same control group; its
target is `Δ_i = y_i − x_c` and the model predicts `Δ̂(x_c, perturbation)`, so
the prediction of `y_i` is `x_c + Δ̂`. A supplementary row is one supplementary
cell used as `x`, with the teacher's `Δ̂(x, perturbation)` as its target. Both
sides therefore feed the model single cells. Evaluation averages Δ̂ per
perturbation (controls drawn with a fixed seed) and compares it with the
observed mean Δ (`y − control mean`). With `sample_controls=False` every gold
row uses its group's control mean and target `y − mean` (the old behaviour).

**Batch allocation.** Each epoch shuffles every gold training cell once and cuts
the order into batches of `batch_size`; no cell is skipped. Stage 1 reads gold
only. In stage 2 each gold batch is paired with `supplementary_ratio × B`
supplementary control cells, drawn without replacement until the pool is spent
(then reshuffled), and supplementary row *i* is queried with the perturbation of
gold row *i mod B*. L_u and L_pg therefore cover the same perturbations in the
same proportions, which is what makes L_pg the control variate for L_u. Teacher
labels are computed on the fly for exactly those pairs.

Early stopping and best-weight selection read `selection_metric` on the
validation perturbations after every epoch, in either stage. The default is
GEARS' rule, validation `mse_deg`. (`pearson_specific` was the default until
2026-10-08: it ignores scale, so it picked stage-2 epochs whose validation
mse_deg had exploded, e.g. 8.8e6 on Adamson.)
"""

from __future__ import annotations

import copy
import json
import time
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from ..training import correction_stage
from .contract import PerturbationExample, PerturbationTask
from .figures import write_perturbation_figures, write_summary
from .graphs import GraphArtifact, edge_overlap
from .losses import gears_loss, retained_genes
from .metrics import baseline_metrics, perturbation_metrics, perturbation_signal
from .mlp import MlpConfig, MlpModel, indicator_from_index
from .model import GearsModel, ModelConfig, sparse_adjacency
from .perturbnet_arm import PERTURBNET_ARMS, default_settings, run_perturbnet_arm, unavailable_reason

GEARS_BASE = "gears_base"
GEARS_SIGNED = "gears_augmented_ungated"
MLP_BASE = "mlp_base"
MLP_SIGNED = "mlp_augmented_ungated"
GEARS_FINETUNE = "gears_finetune_ppi"
GEARS_GATED = "gears_gated"
MLP_GATED = "mlp_gated"
#: the default set: base, signed PPI and gated for both architectures
ARMS = (GEARS_BASE, GEARS_SIGNED, GEARS_GATED, MLP_BASE, MLP_SIGNED, MLP_GATED)
#: every in-process arm a caller may name (the fine-tune arm is opt-in)
ALL_ARMS = ARMS + (GEARS_FINETUNE,)
#: family -> (base arm, signed arm); the base arm is the signed arm's teacher.
FAMILIES = {"gears": (GEARS_BASE, GEARS_SIGNED), "mlp": (MLP_BASE, MLP_SIGNED)}
#: family -> its gated arm; same teacher, same architecture as the signed arm.
GATED = {"gears": GEARS_GATED, "mlp": MLP_GATED}
#: family -> its fine-tune arm: the trained base, fine-tuned with the PPI schedule.
FINETUNE = {"gears": GEARS_FINETUNE}
#: arms trained outside this process (`perturbnet_arm.py`); never in the default.
EXTERNAL_ARMS = PERTURBNET_ARMS


@dataclass
class PerturbationTrainingConfig:
    arms: Sequence[str] = ARMS
    #: GEARS' `train(epochs=20, lr=1e-3, weight_decay=5e-4)` with
    #: `StepLR(step_size=1, gamma=0.5)`, `clip_grad_value_(1.0)`, batch 32 and
    #: no early stopping (`patience=None`); the best validation epoch is kept.
    epochs: int = 20
    patience: int | None = None
    batch_size: int = 32
    learning_rate: float = 1e-3
    weight_decay: float = 5e-4
    #: StepLR gamma per epoch for the GEARS arms. GEARS' own schedule halves the
    #: lr every epoch (0.5); turned off (1.0) on 2026-10-14 at the user's request:
    #: with it, the signed arm's stage-2 steps ran at lr ≈ 3e-6 and below (B7).
    #: The MLP is our own baseline, not GEARS: it keeps a constant learning rate.
    lr_decay: float = 1.0
    mlp_lr_decay: float = 1.0
    grad_clip: float | None = 1.0
    direction_lambda: float = 1e-1
    top_k_deg: int = 20
    #: GEARS' `hidden_size=64`, `num_go_gnn_layers=1`, `num_gene_gnn_layers=1`
    hidden_dim: int = 64
    go_layers: int = 1
    gene_layers: int = 1
    eta: float = 1.0
    #: "auto" = cuda when available
    device: str = "auto"
    mlp_hidden_size: int = 128
    seed: int = 42
    context_categorical: dict[str, int] = field(default_factory=dict)
    context_numeric: int = 0
    #: the learnable λ of stage 2, `λ·(L_u − L_pg)`, bounded to
    #: [lambda_min, lambda_max] (scDesign3 `train_ppi.py`: 1 in [0, 2]).
    lambda_init: float = 1.0
    lambda_min: float = 0.0
    lambda_max: float = 2.0
    stage1_epochs: int = 5
    stage2_epochs: int = 1
    #: the fine-tune arm: epochs and starting lr (× learning_rate). It runs the
    #: same stage cycle, so it begins with stage 1 (see `_finetune`).
    finetune_epochs: int = 12
    finetune_lr_multiplier: float = 0.1
    #: stage-2 learning rates: λ at lr × 10, the model weights at lr ÷ 10.
    lambda_lr_multiplier: float = 10.0
    stage2_lr_multiplier: float = 0.1
    #: supplementary rows per gold row in a stage-2 (or gated) batch.
    supplementary_ratio: int = 1
    #: gated arms: the largest synthetic gradient norm, as a multiple of the gold one.
    gated_kappa: float = 1.0
    #: pair each gold cell with a real control cell (True) or the group mean.
    sample_controls: bool = True
    #: GEARS draws each cell's control once (fixed pairing); True redraws it
    #: every epoch.
    resample_controls: bool = False
    #: GEARS reads the control cell (see model.ModelConfig.condition_on_control).
    condition_on_control: bool = True
    #: validation metric for early stopping / best weights: `pearson_specific`,
    #: `pearson` (higher is better) or `mse_deg`, `mse` (lower is better).
    selection_metric: str = "mse_deg"


# -- batches ----------------------------------------------------------------


#: control group -> `(n, G)` real control cells (`contract.control_cells`).
ControlPool = dict[str, np.ndarray]


@dataclass
class GoldBatch:
    labels: list[str]
    control: torch.Tensor   # (B, G) the input cell: a sampled control, or the mean
    index: torch.Tensor     # (B, P) perturbed-gene indices
    target: torch.Tensor    # (B, G) training target: y − control input
    observed: torch.Tensor  # (B, G) y − control mean, what evaluation scores
    offset: torch.Tensor    # (B, G) control input − control mean (direction term)
    context: dict[str, torch.Tensor] | None


def _context(
    examples: Sequence[PerturbationExample] | None,
    rows: Sequence[int],
    config: PerturbationTrainingConfig,
) -> dict[str, torch.Tensor] | None:
    """Context tensors; `examples=None` is the supplementary rows' "unknown"."""
    if not config.context_categorical and not config.context_numeric:
        return None
    out: dict[str, torch.Tensor] = {}
    for name, cardinality in config.context_categorical.items():
        values = [
            str(examples[row].context.get(name, "unknown")) if examples is not None else "unknown"
            for row in rows
        ]
        out[name] = torch.as_tensor([hash(value) % cardinality for value in values], dtype=torch.long)
    if config.context_numeric:
        out["numeric"] = torch.as_tensor(
            [
                [
                    float(examples[row].context.get(f"numeric_{slot}", 0.0)) if examples is not None else 0.0
                    for slot in range(config.context_numeric)
                ]
                for row in rows
            ],
            dtype=torch.float32,
        )
    return out


def _perturbation_index(
    examples: Sequence[PerturbationExample], rows: Sequence[int], task: PerturbationTask
) -> torch.Tensor:
    """`(B, P)` perturbation-vocabulary indices, padded with -1 (GEARS' convention)."""
    vocabulary = task.perturbation_to_index
    width = max(1, max(len(examples[row].perturbation) for row in rows))
    index = np.full((len(rows), width), -1, dtype=np.int64)
    for position, row in enumerate(rows):
        for slot, gene in enumerate(examples[row].perturbation):
            index[position, slot] = vocabulary[gene]
    return torch.as_tensor(index, dtype=torch.long)


def _group_cells(example: PerturbationExample, pool: ControlPool) -> np.ndarray:
    cells = pool.get(example.control_group)
    return cells if cells is not None and len(cells) else pool["global"]


def draw_pairing(
    examples: Sequence[PerturbationExample], pool: ControlPool, rng: np.random.Generator
) -> np.ndarray:
    """One control-cell index per example (into its group's cells), drawn once."""
    return np.array(
        [rng.integers(len(_group_cells(example, pool))) for example in examples], dtype=np.int64
    )


def _control_inputs(
    examples: Sequence[PerturbationExample],
    rows: Sequence[int],
    pool: ControlPool | None,
    rng: np.random.Generator,
    pairing: np.ndarray | None = None,
) -> np.ndarray:
    """One real control cell per row, from the row's group (else every control).

    With `pairing`, row `r` always gets the same cell (GEARS' fixed pairing);
    without it a fresh cell is drawn from `rng`.
    """
    if not pool:
        return np.stack([examples[row].control for row in rows])
    picked = []
    for row in rows:
        cells = _group_cells(examples[row], pool)
        choice = pairing[row] if pairing is not None else rng.integers(len(cells))
        picked.append(cells[choice])
    return np.stack(picked)


def _make_batch(
    examples: Sequence[PerturbationExample],
    rows: Sequence[int],
    task: PerturbationTask,
    config: PerturbationTrainingConfig,
    pool: ControlPool | None = None,
    rng: np.random.Generator | None = None,
    pairing: np.ndarray | None = None,
) -> GoldBatch:
    observed = np.stack([examples[row].delta for row in rows])
    mean = np.stack([examples[row].control for row in rows])
    if pool and rng is not None and config.sample_controls:
        control = _control_inputs(examples, rows, pool, rng, pairing)
        target = np.stack([examples[row].target for row in rows]) - control
    else:
        control = mean
        target = observed
    return GoldBatch(
        labels=[examples[row].label for row in rows],
        control=torch.as_tensor(control, dtype=torch.float32),
        index=_perturbation_index(examples, rows, task),
        target=torch.as_tensor(target, dtype=torch.float32),
        observed=torch.as_tensor(observed, dtype=torch.float32),
        offset=torch.as_tensor(control - mean, dtype=torch.float32),
        context=_context(examples, rows, config),
    )


def gold_batches(
    examples: Sequence[PerturbationExample],
    task: PerturbationTask,
    config: PerturbationTrainingConfig,
    rng: np.random.Generator,
    pool: ControlPool | None = None,
    pairing: np.ndarray | None = None,
) -> Iterator[GoldBatch]:
    """One epoch: every gold training cell exactly once, in a fresh order, each
    with its control cell (the fixed `pairing`, or a fresh draw)."""
    order = rng.permutation(len(examples))
    for start in range(0, len(order), config.batch_size):
        yield _make_batch(
            examples, order[start : start + config.batch_size], task, config, pool, rng, pairing
        )


def _to_device(batch: GoldBatch, device: torch.device) -> GoldBatch:
    return GoldBatch(
        labels=batch.labels,
        control=batch.control.to(device),
        index=batch.index.to(device),
        target=batch.target.to(device),
        observed=batch.observed.to(device),
        offset=batch.offset.to(device),
        context={key: value.to(device) for key, value in batch.context.items()} if batch.context else None,
    )


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


class SupplementarySampler:
    """Supplementary control cells, drawn without replacement pass by pass."""

    def __init__(self, controls: np.ndarray, rng: np.random.Generator):
        self.controls = torch.as_tensor(np.asarray(controls, dtype=np.float32))
        self.rng = rng
        self._order = rng.permutation(len(self.controls))
        self._cursor = 0

    def __len__(self) -> int:
        return len(self.controls)

    def draw(self, count: int) -> torch.Tensor:
        picked: list[np.ndarray] = []
        need = int(count)
        while need > 0:
            if self._cursor >= len(self._order):
                self._order, self._cursor = self.rng.permutation(len(self.controls)), 0
            take = min(need, len(self._order) - self._cursor)
            picked.append(self._order[self._cursor : self._cursor + take])
            self._cursor += take
            need -= take
        return self.controls[torch.as_tensor(np.concatenate(picked), dtype=torch.long)]


# -- architectures ----------------------------------------------------------


class _Net:
    """One architecture behind one call: `delta(model, control, index, context)`."""

    def __init__(
        self,
        family: str,
        task: PerturbationTask,
        config: PerturbationTrainingConfig,
        adjacency: dict[str, torch.Tensor],
        *,
        augmented: bool,
    ):
        self.family = family
        self.task = task
        self.config = config
        self.adjacency = adjacency
        # Only the GEARS signed arm reads the supplementary co-expression graph.
        self.augmented = augmented and family == "gears" and "G_S" in adjacency

    def new_model(self) -> nn.Module:
        config = self.config
        if self.family == "gears":
            return GearsModel(
                ModelConfig(
                    n_genes=self.task.n_genes,
                    n_perturbations=self.task.n_perturbation_genes,
                    hidden_dim=config.hidden_dim,
                    go_layers=config.go_layers,
                    gene_layers=config.gene_layers,
                    eta=config.eta,
                    context_categorical=config.context_categorical,
                    context_numeric=config.context_numeric,
                    condition_on_control=config.condition_on_control,
                )
            )
        return MlpModel(
            MlpConfig(
                n_genes=self.task.n_genes,
                n_perturbations=self.task.n_perturbation_genes,
                hidden_size=config.mlp_hidden_size,
                context_categorical=config.context_categorical,
                context_numeric=config.context_numeric,
            )
        )

    def delta(
        self,
        model: nn.Module,
        control: torch.Tensor,
        index: torch.Tensor,
        context: dict[str, torch.Tensor] | None,
    ) -> torch.Tensor:
        if self.family == "gears":
            _, delta = model(
                control,
                index,
                self.adjacency["G_C"],
                self.adjacency["G_S"] if self.augmented else None,
                self.adjacency["G_GO"],
                context,
            )
            return delta
        return model(control, indicator_from_index(index, self.task.n_perturbation_genes), context)


@dataclass
class _Teacher:
    net: _Net
    model: nn.Module

    def delta(self, control, index, context) -> torch.Tensor:
        with torch.no_grad():
            return self.net.delta(self.model, control, index, context)


# -- training ---------------------------------------------------------------


def _loss(prediction, target, labels, offset, genes, config) -> torch.Tensor:
    return gears_loss(
        prediction, target, labels,
        genes_by_perturbation=genes, offset=offset,
        direction_lambda=config.direction_lambda,
    ).total


_LOWER_IS_BETTER = {"mse", "mse_deg"}


def _selection_score(summary: dict[str, Any], metric: str) -> float:
    """Higher is better; NaN (e.g. a constant model's pearson_specific) is -inf."""
    value = float(summary.get(metric, float("nan")))
    if not np.isfinite(value):
        return -np.inf
    return -value if metric in _LOWER_IS_BETTER else value


def project_to_descent(
    gradient: torch.Tensor, constraints: Sequence[torch.Tensor], eps: float = 1e-12
) -> torch.Tensor:
    """The closest `d` to `gradient` with `⟨d, c⟩ ≥ 0` for every constraint `c`.

    Stepping `θ − η·d` then does not increase (to first order) any loss whose
    gradient is a constraint. Exact for one or two constraints: try every active
    set (none, each alone, both) and keep the closest feasible candidate.
    """
    rows = [c for c in constraints if float(c.dot(c)) > eps]
    if all(float(gradient.dot(c)) >= 0 for c in rows):
        return gradient
    candidates = []
    for c in rows:
        candidates.append(gradient - min(0.0, float(gradient.dot(c))) / float(c.dot(c)) * c)
    if len(rows) == 2:
        a = torch.stack(rows)                                     # (2, n)
        gram = a @ a.T
        if float(torch.linalg.det(gram)) > eps:
            # d = g + aᵀμ with ⟨d, c_i⟩ = 0 for both
            mu = torch.linalg.solve(gram, -(a @ gradient))
            if bool((mu >= 0).all()):
                candidates.append(gradient + a.T @ mu)
    feasible = [d for d in candidates if all(float(d.dot(c)) >= -1e-6 * float(c.norm()) * float(d.norm()) for c in rows)]
    if not feasible:
        return torch.zeros_like(gradient)
    return min(feasible, key=lambda d: float((d - gradient).norm()))


def _flat(grads: Sequence[torch.Tensor | None], params: Sequence[nn.Parameter], clip: float | None) -> torch.Tensor:
    parts = []
    for grad, param in zip(grads, params):
        value = torch.zeros_like(param) if grad is None else grad
        if clip:
            value = value.clamp(-clip, clip)
        parts.append(value.reshape(-1))
    return torch.cat(parts)


def _supplementary_loss(net, model, batch, teacher, sampler, ratio, reference, genes, config, device):
    """L_u: supplementary cells, each queried with a perturbation of this gold
    batch (row i gets gold row i mod B's), against the frozen teacher's Δ."""
    size = len(batch.labels) * ratio
    supp_control = sampler.draw(size).to(device)
    supp_index = batch.index.repeat(ratio, 1)
    supp_labels = batch.labels * ratio
    supp_context = _context(None, range(size), config)
    if supp_context:
        supp_context = {key: value.to(device) for key, value in supp_context.items()}
    u_target = teacher.delta(supp_control, supp_index, supp_context)
    supp_offset = supp_control - reference if reference is not None else None
    return _loss(
        net.delta(model, supp_control, supp_index, supp_context),
        u_target, supp_labels, supp_offset, genes, config,
    )


def gated_gradient(
    g_gold: torch.Tensor, g_syn: torch.Tensor, kappa: float = 1.0, eps: float = 1e-12
) -> tuple[torch.Tensor, dict[str, float]]:
    """`g_G + r·c·g_S` (gatedlossdesign.md §3-5), from detached gradients.

    r = max(0, cos(g_G, g_S)); c = min(1, κ‖g_G‖ / (‖g_S‖ + ε)). A conflicting
    synthetic gradient (cos ≤ 0) gives exactly g_G.
    """
    g_gold, g_syn = g_gold.detach(), g_syn.detach()
    gold_norm, syn_norm = float(g_gold.norm()), float(g_syn.norm())
    cosine = float(g_gold.dot(g_syn)) / (gold_norm * syn_norm + eps)
    reliability = max(0.0, cosine)
    scale = min(1.0, kappa * gold_norm / (syn_norm + eps))
    weight = reliability * scale
    final = g_gold + weight * g_syn if weight > 0 else g_gold
    return final, {
        "gold_grad_norm": gold_norm,
        "synthetic_grad_norm": syn_norm,
        "gradient_cosine": cosine,
        "synthetic_weight": weight,
        "synthetic_active": float(weight > 0),
    }


def _assign(params: Sequence[nn.Parameter], flat: torch.Tensor) -> None:
    offset = 0
    for param in params:
        n = param.numel()
        param.grad = flat[offset : offset + n].view_as(param).clone()
        offset += n


class LearnableLambda(nn.Module):
    """One learnable scalar in `[min_value, max_value]` (scDesign3 `train_ppi.py`)."""

    def __init__(self, init_value: float, min_value: float, max_value: float):
        super().__init__()
        if not min_value < init_value < max_value:
            raise ValueError("need lambda_min < lambda_init < lambda_max")
        self.min_value, self.max_value = float(min_value), float(max_value)
        p = (init_value - min_value) / (max_value - min_value)
        self.raw = nn.Parameter(torch.tensor(float(np.log(p / (1.0 - p)))))

    def forward(self) -> torch.Tensor:
        return self.min_value + (self.max_value - self.min_value) * torch.sigmoid(self.raw)

    def value(self) -> float:
        return float(self().detach().cpu())


def _evaluate(
    net: _Net,
    model: nn.Module,
    examples: Sequence[PerturbationExample],
    task: PerturbationTask,
    config: PerturbationTrainingConfig,
    pool: ControlPool | None = None,
) -> tuple[dict[str, Any], dict[str, np.ndarray], dict[str, np.ndarray]]:
    model.eval()
    device = next(model.parameters()).device
    # A fixed seed: every evaluation of every arm sees the same control draws.
    rng = np.random.default_rng(config.seed + 2)
    # per-perturbation running sums: a 40k-cell test split never sits in memory
    sums: dict[str, list] = {}
    size = max(config.batch_size, 256)
    with torch.no_grad():
        for start in range(0, len(examples), size):
            rows = list(range(start, min(start + size, len(examples))))
            batch = _to_device(_make_batch(examples, rows, task, config, pool, rng), device)
            delta = net.delta(model, batch.control, batch.index, batch.context).cpu().numpy()
            observed = batch.observed.cpu().numpy()
            for position, label in enumerate(batch.labels):
                entry = sums.setdefault(label, [0.0, 0.0, 0])
                entry[0] = entry[0] + delta[position]
                entry[1] = entry[1] + observed[position]
                entry[2] += 1
    mean_predictions = {label: np.asarray(v[0] / v[2]) for label, v in sums.items()}
    mean_observations = {label: np.asarray(v[1] / v[2]) for label, v in sums.items()}
    metrics = perturbation_metrics(mean_predictions, mean_observations, top_k=config.top_k_deg)
    return metrics.to_dict(), mean_predictions, mean_observations


def _train_arm(
    arm: str,
    net: _Net,
    model: nn.Module,
    splits: dict[str, list[PerturbationExample]],
    task: PerturbationTask,
    config: PerturbationTrainingConfig,
    genes: dict[str, np.ndarray],
    *,
    reference: torch.Tensor | None = None,
    pool: ControlPool | None = None,
    teacher: _Teacher | None = None,
    sampler: SupplementarySampler | None = None,
    epochs: int | None = None,
    learning_rate: float | None = None,
    gated: bool = False,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Train one arm; returns its history and where its best weights came from.

    `gated`: every step is the gradient-gated update (`gated_gradient`) with the
    teacher's Δ on supplementary cells as the synthetic loss; no stage cycle.
    """
    lr = config.learning_rate if learning_rate is None else learning_rate
    epochs = config.epochs if epochs is None else epochs
    params = [param for param in model.parameters() if param.requires_grad]
    optimizer = torch.optim.Adam(params, lr=lr, weight_decay=config.weight_decay)
    has_teacher = teacher is not None and sampler is not None and len(sampler) > 0
    gated = gated and has_teacher
    signed = has_teacher and not gated
    device = next(model.parameters()).device
    learnable_lambda = None
    if signed:
        learnable_lambda = LearnableLambda(
            config.lambda_init, config.lambda_min, config.lambda_max
        ).to(device)
        # SGD: the projected direction is the step (Adam would rescale it)
        optimizer_stage2 = torch.optim.SGD(params, lr=lr * config.stage2_lr_multiplier)
        optimizer_lambda = torch.optim.Adam(
            learnable_lambda.parameters(), lr=lr * config.lambda_lr_multiplier
        )
    optimizers = [optimizer] + ([optimizer_stage2, optimizer_lambda] if signed else [])
    rng = np.random.default_rng(config.seed)
    # GEARS pairs every training cell with one control cell, once
    pairing = (
        draw_pairing(splits["train"], pool, np.random.default_rng(config.seed + 3))
        if pool and config.sample_controls and not config.resample_controls
        else None
    )
    ratio = max(1, int(config.supplementary_ratio))
    history: list[dict[str, Any]] = []
    best_state, best_score, best = None, -np.inf, {"epoch": 0, "stage": None}
    metric = config.selection_metric
    stale = 0
    for epoch in range(1, epochs + 1):
        model.train()
        stage2 = signed and correction_stage(
            epoch,
            use_external=True,
            schedule="two_stage",
            stage1_epochs=config.stage1_epochs,
            stage2_epochs=config.stage2_epochs,
        )
        losses: dict[str, list[float]] = {
            "train_loss": [], "loss_true": [], "loss_u": [], "loss_pg": [], "correction": [],
            "projected": [],
        }
        gate_log: dict[str, list[float]] = {}
        for batch in gold_batches(splits["train"], task, config, rng, pool, pairing):
            batch = _to_device(batch, device)
            delta = net.delta(model, batch.control, batch.index, batch.context)
            if gated:
                loss_true = _loss(delta, batch.target, batch.labels, batch.offset, genes, config)
                loss_u = _supplementary_loss(
                    net, model, batch, teacher, sampler, ratio, reference, genes, config, device
                )
                g_gold = _flat(torch.autograd.grad(loss_true, params, allow_unused=True), params, None)
                g_syn = _flat(torch.autograd.grad(loss_u, params, allow_unused=True), params, None)
                final, stats = gated_gradient(g_gold, g_syn, config.gated_kappa)
                optimizer.zero_grad(set_to_none=True)
                _assign(params, final)
                if config.grad_clip:
                    nn.utils.clip_grad_value_(params, config.grad_clip)
                optimizer.step()
                for key, value in stats.items():
                    gate_log.setdefault(key, []).append(value)
                losses["train_loss"].append(float(loss_true.detach()))
                losses["loss_true"].append(float(loss_true.detach()))
                losses["loss_u"].append(float(loss_u.detach()))
                continue
            if not stage2:
                loss_true = _loss(delta, batch.target, batch.labels, batch.offset, genes, config)
                optimizer.zero_grad(set_to_none=True)
                loss_true.backward()
                if config.grad_clip:
                    nn.utils.clip_grad_value_(params, config.grad_clip)
                optimizer.step()
                losses["train_loss"].append(float(loss_true.detach()))
                losses["loss_true"].append(float(loss_true.detach()))
                continue

            # L_pg: the gold forward against the teacher's Δ for the same cells.
            pg_target = teacher.delta(batch.control, batch.index, batch.context)
            loss_pg = _loss(delta, pg_target, batch.labels, batch.offset, genes, config)
            # L_u: supplementary controls, paired with this batch's perturbations.
            loss_u = _supplementary_loss(
                net, model, batch, teacher, sampler, ratio, reference, genes, config, device
            )
            loss_true = _loss(delta, batch.target, batch.labels, batch.offset, genes, config)
            correction = loss_u - loss_pg
            lam = learnable_lambda()
            loss = lam * correction
            clip = config.grad_clip
            grads = torch.autograd.grad(loss, params + [learnable_lambda.raw], retain_graph=True, allow_unused=True)
            g = _flat(grads[:-1], params, clip)
            g_u = _flat(torch.autograd.grad(loss_u, params, retain_graph=True, allow_unused=True), params, clip)
            g_true = _flat(torch.autograd.grad(loss_true, params, allow_unused=True), params, clip)
            d = project_to_descent(g, [g_u, g_true])
            offset = 0
            for param in params:
                n = param.numel()
                param.grad = d[offset : offset + n].view_as(param)
                offset += n
            learnable_lambda.raw.grad = grads[-1]
            optimizer_stage2.step()
            optimizer_lambda.step()
            for param in params:
                param.grad = None
            learnable_lambda.raw.grad = None
            losses["projected"].append(float(not torch.equal(d, g)))
            losses["loss_true"].append(float(loss_true.detach()))
            losses["train_loss"].append(float(loss.detach()))
            losses["correction"].append(float(correction.detach()))
            losses["loss_u"].append(float(loss_u.detach()))
            losses["loss_pg"].append(float(loss_pg.detach()))

        # StepLR(step_size=1, gamma) on every optimiser; GEARS halves, the MLP does not decay
        gamma = config.lr_decay if net.family == "gears" else config.mlp_lr_decay
        for each in optimizers:
            for group in each.param_groups:
                group["lr"] *= gamma
        if splits["validation"]:
            validation, _, _ = _evaluate(net, model, splits["validation"], task, config, pool)
            summary = validation["summary"]
            score = _selection_score(summary, metric)
        else:
            # No validation perturbation: select on the training objective.
            fallback = float(np.mean(losses["train_loss"])) if losses["train_loss"] else float("inf")
            summary = {"mse_deg": float("nan"), "pearson": float("nan"),
                       "pearson_specific": float("nan"), "direction_accuracy": float("nan")}
            score = -fallback
        gate_summary: dict[str, float] = {}
        if gate_log:
            # gatedlossdesign.md §9: per-epoch means, and the spread of cosine / weight
            gate_summary = {
                "train_gold_loss": float(np.mean(losses["loss_true"])),
                "train_synthetic_loss": float(np.mean(losses["loss_u"])),
                "gold_grad_norm": float(np.mean(gate_log["gold_grad_norm"])),
                "synthetic_grad_norm": float(np.mean(gate_log["synthetic_grad_norm"])),
                "gradient_cosine": float(np.mean(gate_log["gradient_cosine"])),
                "gradient_cosine_std": float(np.std(gate_log["gradient_cosine"])),
                "synthetic_weight": float(np.mean(gate_log["synthetic_weight"])),
                "synthetic_weight_std": float(np.std(gate_log["synthetic_weight"])),
                "synthetic_active_fraction": float(np.mean(gate_log["synthetic_active"])),
            }
        entry: dict[str, Any] = {
            "epoch": epoch,
            "phase": "gated" if gated else "stage2" if stage2 else "stage1",
            **{key: float(np.mean(values)) for key, values in losses.items() if values},
            **gate_summary,
            **({"lambda": learnable_lambda.value()} if learnable_lambda is not None else {}),
            "validation_mse_deg": summary.get("mse_deg", float("nan")),
            "validation_pearson": summary.get("pearson", float("nan")),
            "validation_pearson_specific": summary.get("pearson_specific", float("nan")),
            "validation_direction_accuracy": summary.get("direction_accuracy", float("nan")),
        }
        history.append(entry)
        print(
            f"# {arm} epoch {epoch}/{epochs} [{entry['phase']}]: "
            f"validation mse_deg {entry['validation_mse_deg']:.4f}, "
            f"pearson {entry['validation_pearson']:.3f}, "
            f"specific pearson {entry['validation_pearson_specific']:.3f}"
            + (f", lambda {entry['lambda']:.3f}" if "lambda" in entry else "")
            + (
                f", cos {entry['gradient_cosine']:+.3f}, weight {entry['synthetic_weight']:.3f}, "
                f"active {entry['synthetic_active_fraction']:.0%}"
                if "gradient_cosine" in entry
                else ""
            ),
            flush=True,
        )
        if best_state is None or score > best_score + 1e-6:
            best_score = score
            best_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
            best = {
                "epoch": epoch,
                "stage": entry["phase"],
                **({"lambda": entry["lambda"]} if "lambda" in entry else {}),
                "selection_metric": metric,
                f"validation_{metric}": summary.get(metric, float("nan")),
            }
            stale = 0
        else:
            stale += 1
            if config.patience and stale >= config.patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return history, best


def _tensor_graphs(
    graphs: dict[str, GraphArtifact], n_genes: int, n_perturbations: int | None = None
) -> dict[str, torch.Tensor]:
    """Sparse propagation matrices; `G_GO` lives on the perturbation vocabulary."""
    return {
        name: sparse_adjacency(
            torch.as_tensor(graph.edge_index, dtype=torch.long),
            torch.as_tensor(graph.edge_weight, dtype=torch.float32),
            (n_perturbations or n_genes) if name == "G_GO" else n_genes,
        )
        for name, graph in graphs.items()
    }


def _retained_genes(examples: Sequence[PerturbationExample]) -> dict[str, np.ndarray]:
    """GEARS' per-perturbation loss genes, from the training cells' expression."""
    grouped: dict[str, list[np.ndarray]] = {}
    for example in examples:
        grouped.setdefault(example.label, []).append(example.target)
    return retained_genes({label: np.stack(cells) for label, cells in grouped.items()})


def run_perturbation_experiment(
    *,
    splits: dict[str, list[PerturbationExample]],
    task: PerturbationTask,
    graphs: dict[str, GraphArtifact],
    out_dir: str | Path,
    config: PerturbationTrainingConfig | None = None,
    supplementary_controls: np.ndarray | None = None,
    control_cells: ControlPool | None = None,
    modality: dict[str, Any] | None = None,
    counts: dict[str, Any] | None = None,
    control_rows: np.ndarray | None = None,
) -> dict[str, Any]:
    """Train the requested arms on identical splits and report the comparison.

    `control_cells` is the gold's control-cell pool; without it (or with
    `sample_controls=False`) gold rows fall back to the group control mean.
    `counts` (the stage-2 raw count tables) and `control_rows` (the gold rows
    of the control cells) are what the PerturbNet arms need.
    """
    config = config or PerturbationTrainingConfig()
    unknown = [arm for arm in config.arms if arm not in ALL_ARMS + EXTERNAL_ARMS]
    if unknown:
        raise ValueError(
            f"unknown perturbation arm(s) {unknown}; choose from {ALL_ARMS + EXTERNAL_ARMS}"
        )
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    device = resolve_device(config.device)
    adjacency = {
        name: matrix.to(device)
        for name, matrix in _tensor_graphs(graphs, task.n_genes, task.n_perturbation_genes).items()
    }
    genes = _retained_genes(splits["train"])
    # The control mean the direction term of supplementary rows is measured from.
    reference = torch.as_tensor(
        control_cells["global"].mean(axis=0)
        if control_cells
        else np.mean([example.control for example in splits["train"]], axis=0),
        dtype=torch.float32,
    ).to(device)
    has_supp = supplementary_controls is not None and len(supplementary_controls) > 0

    test_sums: dict[str, list] = {}
    for example in splits["test"]:
        entry = test_sums.setdefault(example.label, [0.0, 0])
        entry[0] = entry[0] + np.asarray(example.delta, dtype=np.float64)
        entry[1] += 1

    results: dict[str, Any] = {
        "task": task.to_dict(),
        "modality": modality or {},
        "arms": {},
        "split_sizes": {name: len(values) for name, values in splits.items()},
        "graph_edge_overlap": edge_overlap(graphs["G_C"], graphs["G_S"]) if "G_S" in graphs else {},
        "config": {
            "epochs": config.epochs,
            "patience": config.patience,
            "batch_size": config.batch_size,
            "learning_rate": config.learning_rate,
            "weight_decay": config.weight_decay,
            "lr_decay_per_epoch": {"gears": config.lr_decay, "mlp": config.mlp_lr_decay},
            "grad_clip": config.grad_clip,
            "hidden_dim": config.hidden_dim,
            "device": str(device),
            "resample_controls": config.resample_controls,
            "perturbation_vocabulary": task.n_perturbation_genes,
            "direction_lambda": config.direction_lambda,
            "top_k_deg": config.top_k_deg,
            "seed": config.seed,
            "stage2_objective": (
                "lambda * (L_u - L_pg), lambda learnable (Adam); model step = SGD on the "
                "gradient projected so that L_u and L_true do not increase"
            ),
            "lambda_init": config.lambda_init,
            "lambda_bounds": [config.lambda_min, config.lambda_max],
            "stage1_epochs": config.stage1_epochs,
            "stage2_epochs": config.stage2_epochs,
            "finetune_epochs": config.finetune_epochs,
            "finetune_lr_multiplier": config.finetune_lr_multiplier,
            "lambda_lr_multiplier": config.lambda_lr_multiplier,
            "stage2_lr_multiplier": config.stage2_lr_multiplier,
            "supplementary_ratio": config.supplementary_ratio,
            "gated_update": "g_G + max(0, cos(g_G, g_S)) * min(1, kappa |g_G| / |g_S|) * g_S, Adam",
            "gated_kappa": config.gated_kappa,
            "sample_controls": bool(config.sample_controls and control_cells),
            "condition_on_control": config.condition_on_control,
            "selection_metric": config.selection_metric,
            "loss_genes": "GEARS non_zeros_gene_idx (non-zero mean expression per perturbation)",
        },
        "loss_genes_per_perturbation": {label: int(len(index)) for label, index in genes.items()},
        "control_input": (
            f"a real control cell per row ({len(control_cells['global'])} gold control "
            f"cells); target y − x_c"
            if config.sample_controls and control_cells
            else "the control group mean; target y − mean"
        ),
        "stage_epochs": {"stage1": config.stage1_epochs, "stage2": config.stage2_epochs},
        "supplementary": {
            "control_cells": int(len(supplementary_controls)) if has_supp else 0,
            "pairing": "supplementary row i is queried with gold row i's perturbation",
            "labels": "the frozen base arm's Δ, computed per batch",
        },
        "baselines": baseline_metrics(
            {label: total / count for label, (total, count) in test_sums.items()},
            top_k=config.top_k_deg,
        ),
    }

    def _record(arm: str, net: _Net, model: nn.Module, history, best, started) -> None:
        validation = (
            _evaluate(net, model, splits["validation"], task, config, control_cells)[0]
            if splits["validation"]
            else None
        )
        test, predictions, observations = _evaluate(
            net, model, splits["test"], task, config, control_cells
        )
        torch.save(model.state_dict(), out / f"{arm}.pt")
        labels = sorted(predictions)
        np.savez_compressed(
            out / f"{arm}_test_predictions.npz",
            labels=np.array(labels),
            predictions=np.stack([predictions[label] for label in labels]),
            observations=np.stack([observations[label] for label in labels]),
        )
        gate_keys = ("gradient_cosine", "synthetic_weight", "synthetic_active_fraction",
                     "gold_grad_norm", "synthetic_grad_norm")
        gated_epochs = [entry for entry in history if "gradient_cosine" in entry]
        results["arms"][arm] = {
            **(
                {"gating": {key: float(np.mean([e[key] for e in gated_epochs])) for key in gate_keys}}
                if gated_epochs
                else {}
            ),
            "validation": validation,
            "test": test,
            "history": history,
            "epochs_run": len(history),
            "best": best,
            "seconds": round(time.time() - started, 1),
            "predictions": str(out / f"{arm}_test_predictions.npz"),
            "signal": perturbation_signal(predictions, observations),
        }
        print(
            f"# {arm} done in {results['arms'][arm]['seconds']:.0f}s: "
            f"test mse_deg {test['summary']['mse_deg']:.4f}, "
            f"pearson {test['summary']['pearson']:.3f}, "
            f"direction {test['summary']['direction_accuracy']:.3f}",
            flush=True,
        )
        try:
            results["partial"] = True
            write_summary(results, out)
        except Exception as error:  # noqa: BLE001 - a report never fails a run
            print(f"# could not write the partial summary: {error}", flush=True)

    for family, (base_arm, signed_arm) in FAMILIES.items():
        finetune_arm = FINETUNE.get(family)
        gated_arm = GATED.get(family)
        wanted = [
            arm for arm in (base_arm, signed_arm, gated_arm, finetune_arm) if arm and arm in config.arms
        ]
        if not wanted:
            continue
        # The base arm is always trained: it is the signed arm's teacher.
        base_net = _Net(family, task, config, adjacency, augmented=False)
        torch.manual_seed(config.seed)
        base = base_net.new_model().to(device)
        started = time.time()
        history, best = _train_arm(
            base_arm, base_net, base, splits, task, config, genes,
            reference=reference, pool=control_cells,
        )
        if base_arm in wanted:
            _record(base_arm, base_net, base, history, best, started)
        if not has_supp and (signed_arm in wanted or gated_arm in wanted or finetune_arm in wanted):
            print(f"# {family}: no supplementary control cells, so stage 2 never runs", flush=True)
        base.eval()
        if signed_arm in wanted:
            signed_net = _Net(family, task, config, adjacency, augmented=True)
            torch.manual_seed(config.seed)
            model = signed_net.new_model().to(device)
            started = time.time()
            history, best = _train_arm(
                signed_arm, signed_net, model, splits, task, config, genes,
                reference=reference, pool=control_cells,
                teacher=_Teacher(base_net, base),
                sampler=(
                    SupplementarySampler(supplementary_controls, np.random.default_rng(config.seed + 1))
                    if has_supp
                    else None
                ),
            )
            _record(signed_arm, signed_net, model, history, best, started)
        if gated_arm and gated_arm in wanted:
            # Same architecture, teacher and supplementary cells as the signed
            # arm: only the objective differs.
            gated_net = _Net(family, task, config, adjacency, augmented=True)
            torch.manual_seed(config.seed)
            model = gated_net.new_model().to(device)
            started = time.time()
            history, best = _train_arm(
                gated_arm, gated_net, model, splits, task, config, genes,
                reference=reference, pool=control_cells,
                teacher=_Teacher(base_net, base),
                sampler=(
                    SupplementarySampler(supplementary_controls, np.random.default_rng(config.seed + 1))
                    if has_supp
                    else None
                ),
                gated=True,
            )
            _record(gated_arm, gated_net, model, history, best, started)
        if finetune_arm and finetune_arm in wanted:
            # The student starts as an exact copy of the teacher (the base arm's
            # best weights, BatchNorm statistics included) and keeps the base
            # architecture (no G_S), so the only change is the PPI objective.
            # It starts with stage 1: at student == teacher both L_u and L_pg
            # sit at their minimum (zero residual), so a stage-2 step there
            # would have zero gradient.
            student = copy.deepcopy(base)
            started = time.time()
            history, best = _train_arm(
                finetune_arm, base_net, student, splits, task, config, genes,
                reference=reference, pool=control_cells,
                teacher=_Teacher(base_net, base),
                sampler=(
                    SupplementarySampler(supplementary_controls, np.random.default_rng(config.seed + 1))
                    if has_supp
                    else None
                ),
                epochs=config.finetune_epochs,
                learning_rate=config.learning_rate * config.finetune_lr_multiplier,
            )
            _record(finetune_arm, base_net, student, history, best, started)

    for arm in EXTERNAL_ARMS:
        if arm not in config.arms:
            continue
        reason = unavailable_reason(arm, counts)
        if reason is None and control_rows is None:
            reason = "the gold control rows were not passed"
        if reason:
            print(f"# WARNING: {arm} skipped: {reason}", flush=True)
            results.setdefault("skipped_arms", {})[arm] = reason
            continue
        try:
            results["arms"][arm] = run_perturbnet_arm(
                arm, splits=splits, task=task, counts=counts or {}, control_rows=control_rows,
                out=out, seed=config.seed, top_k=config.top_k_deg, settings=default_settings(),
            )
        except Exception as error:  # noqa: BLE001 - an external arm never sinks the trained arms
            print(f"# WARNING: {arm} failed: {error}", flush=True)
            results.setdefault("failed_arms", {})[arm] = str(error)
            continue
        try:
            results["partial"] = True
            write_summary(results, out)
        except Exception as error:  # noqa: BLE001 - a report never fails a run
            print(f"# could not write the partial summary: {error}", flush=True)

    results.pop("partial", None)
    figures = write_perturbation_figures(results, out)
    results["figures"] = [str(path) for path in figures]
    write_summary(results, out)
    (out / "perturbation_results.json").write_text(
        json.dumps(results, indent=2, default=str), encoding="utf-8"
    )
    return results
