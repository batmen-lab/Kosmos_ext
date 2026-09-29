"""Graph-free MLP baselines for perturbation response (Base and Augmented).

Two arms the perturbation report compares against the GEARS-shaped ones:

  * ``mlp_base``      -- gold only.
  * ``mlp_augmented`` -- the *same* architecture and the same gold objective,
                         plus supplementary rows whose targets are
                         **teacher-generated synthetic labels**, combined by the
                         existing detached gradient gate in
                         ``kosmos/ppi/gating.py``.

**One objective for every arm.** A comparison of architectures is only valid if
the training signal is the same, so by default the MLP arms optimise exactly the
GEARS objective (``kosmos/ppi/perturbation/losses.py::gears_loss``): the per
perturbation group ``Σ(pred-y)^(2+γ)`` with γ=2 restricted to that perturbation's
DEG set, plus the direction term, on **Δ**. That means:

  * the MLP predicts Δ (post-perturbation expression is ``control + Δ``), the
    same quantity the GEARS arms predict -- *not* post-perturbation expression
    with a plain MSE;
  * the DEG sets are the ones computed once, from the gold training split, and
    shared with the GEARS arms;
  * the direction weight is the run's ``direction_lambda``;
  * the augmented arm's synthetic term uses the same objective against the
    teacher's Δ labels.

The design document's benchmark form (post-perturbation expression, gene-averaged
MSE) is still available with ``PerturbationTrainingConfig.mlp_objective="mse"``;
it is recorded in the artifacts either way, so a run says which signal it used.

The architecture is still graph-free: two separate ReLU encoders
(control -> hidden, perturbation indicator -> hidden) **added**, then a linear
head. No graph, GO term, co-expression matrix or GNN is consumed -- that is what
makes it a baseline, and it is why the arms differ only by the supplementary
term.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
from torch import nn

from ..gating import GradientGate
from .contract import PerturbationExample, PerturbationTask
from .losses import gears_loss
from .losses import ppi_signed_correction
from .metrics import perturbation_metrics
from .synthetic import SyntheticSet, build_synthetic

if TYPE_CHECKING:  # avoids the train.py <-> mlp.py import cycle
    from .train import PerturbationTrainingConfig

MLP_ARMS = ("mlp_base", "mlp_augmented_ungated", "mlp_augmented")


@dataclass
class MlpConfig:
    """Shape of the baseline. Defaults are the design's benchmark settings."""

    n_genes: int
    hidden_size: int = 128
    context_categorical: Mapping[str, int] | None = None
    context_numeric: int = 0


class MlpModel(nn.Module):
    """`out = W_o (ReLU(W_c x_ctrl + b_c) + ReLU(W_p p + b_p)) + b_o`.

    What ``out`` *is* depends on the objective: Δ when the arm optimises the
    GEARS objective, post-perturbation expression when it optimises MSE. Both
    readings go through `_predict`, so the rest of the code never branches on it.
    """

    def __init__(self, config: MlpConfig):
        super().__init__()
        self.config = config
        self.control_encoder = nn.Linear(config.n_genes, config.hidden_size)
        self.perturbation_encoder = nn.Linear(config.n_genes, config.hidden_size)
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
                hidden = hidden + self.context_numeric(torch.stack(numeric, dim=1))
        return self.output(hidden)


# -- tensor construction ----------------------------------------------------


def _indicator(
    examples: Sequence[PerturbationExample], rows: np.ndarray, task: PerturbationTask
) -> torch.Tensor:
    """`(len(rows), G)` with a 1 at every perturbed gene's position."""
    matrix = np.zeros((len(rows), task.n_genes), dtype=np.float32)
    for position, row in enumerate(rows):
        for gene in examples[row].perturbation:
            index = task.gene_to_index.get(gene)
            if index is not None:
                matrix[position, index] = 1.0
    return torch.as_tensor(matrix, dtype=torch.float32)


def _context(
    examples: Sequence[PerturbationExample],
    rows: np.ndarray,
    config: "PerturbationTrainingConfig",
) -> dict[str, torch.Tensor] | None:
    if not config.context_categorical and not config.context_numeric:
        return None
    out: dict[str, torch.Tensor] = {}
    for name, cardinality in config.context_categorical.items():
        values = [
            hash(str(examples[row].context.get(name, "unknown"))) % cardinality
            for row in rows
        ]
        out[name] = torch.as_tensor(values, dtype=torch.long)
    if config.context_numeric:
        out["numeric"] = torch.zeros((len(rows), config.context_numeric), dtype=torch.float32)
    return out


def _supplementary_context(
    size: int, config: "PerturbationTrainingConfig"
) -> dict[str, torch.Tensor] | None:
    """Supplementary rows carry no context metadata: use the "unknown" bucket."""
    if not config.context_categorical and not config.context_numeric:
        return None
    out: dict[str, torch.Tensor] = {}
    for name, cardinality in config.context_categorical.items():
        out[name] = torch.full((size,), hash("unknown") % cardinality, dtype=torch.long)
    if config.context_numeric:
        out["numeric"] = torch.zeros((size, config.context_numeric), dtype=torch.float32)
    return out


def _batches(examples: Sequence[PerturbationExample], size: int, rng: np.random.Generator):
    """Same batching rule as the GEARS arms: a batch needs >= 2 perturbations."""
    order = rng.permutation(len(examples))
    for start in range(0, len(order), size):
        rows = order[start : start + size]
        labels = [examples[index].label for index in rows]
        if len(set(labels)) < 2:
            continue
        yield rows, labels


def _gold_batch(
    examples: Sequence[PerturbationExample],
    rows: np.ndarray,
    task: PerturbationTask,
    config: "PerturbationTrainingConfig",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Any]:
    control = torch.as_tensor(
        np.stack([examples[row].control for row in rows]), dtype=torch.float32
    )
    indicator = _indicator(examples, rows, task)
    delta = torch.as_tensor(
        np.stack([examples[row].delta for row in rows]), dtype=torch.float32
    )
    target = torch.as_tensor(
        np.stack([examples[row].target for row in rows]), dtype=torch.float32
    )
    return control, indicator, delta, target, _context(examples, rows, config)


def _predict(
    model: MlpModel,
    control: torch.Tensor,
    indicator: torch.Tensor,
    context: dict[str, torch.Tensor] | None,
    *,
    target_kind: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """`(post_perturbation_expression, delta)` from the model's raw output.

    ``target_kind="delta"`` (the unified objective) reads the head as Δ;
    ``target_kind="expression"`` reads it as y. Both are returned so the caller
    never has to know which one the arm is trained on.
    """
    raw = model(control, indicator, context)
    if target_kind == "delta":
        return control + raw, raw
    return raw, raw - control


def _target_kind(config: "PerturbationTrainingConfig") -> str:
    return "delta" if config.mlp_objective == "gears" else "expression"


def _arm_loss(
    prediction_y: torch.Tensor,
    prediction_delta: torch.Tensor,
    target_y: torch.Tensor,
    target_delta: torch.Tensor,
    labels: Sequence[str],
    config: "PerturbationTrainingConfig",
    deg_map: Mapping[str, np.ndarray] | None,
) -> torch.Tensor:
    if config.mlp_objective == "gears":
        return gears_loss(
            prediction_delta,
            target_delta,
            labels,
            deg_by_perturbation=deg_map,
            direction_lambda=config.direction_lambda,
        ).total
    return ((prediction_y - target_y) ** 2).mean()


def evaluate_mlp(
    model: MlpModel,
    examples: Sequence[PerturbationExample],
    task: PerturbationTask,
    config: "PerturbationTrainingConfig",
) -> dict[str, Any]:
    """Perturbation-level metrics on Δ, identical to the GEARS arms' evaluation."""
    model.eval()
    target_kind = _target_kind(config)
    predictions: dict[str, list[np.ndarray]] = {}
    observations: dict[str, list[np.ndarray]] = {}
    with torch.no_grad():
        for start in range(0, len(examples), config.batch_size):
            rows = np.arange(start, min(start + config.batch_size, len(examples)))
            control, indicator, delta, _, context = _gold_batch(examples, rows, task, config)
            _, delta_pred = _predict(
                model, control, indicator, context, target_kind=target_kind
            )
            for position, row in enumerate(rows):
                label = examples[row].label
                predictions.setdefault(label, []).append(delta_pred[position].numpy())
                observations.setdefault(label, []).append(delta[position].numpy())
    mean_predictions = {
        label: np.mean(np.stack(values), axis=0) for label, values in predictions.items()
    }
    mean_observations = {
        label: np.mean(np.stack(values), axis=0) for label, values in observations.items()
    }
    metrics = perturbation_metrics(
        mean_predictions, mean_observations, top_k=config.top_k_deg
    )
    return {
        "metrics": metrics.to_dict(),
        "predictions": mean_predictions,
        "observations": mean_observations,
    }


# -- training ---------------------------------------------------------------


def _indicator_from_index(index: torch.Tensor, n_genes: int) -> torch.Tensor:
    """`(B, G)` 0/1 indicator from the `(B, P)` gene-index form the shared
    synthetic builder speaks (so the MLP and the GEARS arms share one builder)."""
    indicator = torch.zeros((index.shape[0], n_genes), dtype=torch.float32)
    for slot in range(index.shape[1]):
        indicator.scatter_(1, index[:, slot : slot + 1], 1.0)
    return indicator


def _train_arm(
    model: MlpModel,
    train: Sequence[PerturbationExample],
    validation: Sequence[PerturbationExample],
    task: PerturbationTask,
    config: "PerturbationTrainingConfig",
    rng: np.random.Generator,
    *,
    mode: str,                                  # "gold" | "signed" | "gated"
    deg_map: Mapping[str, np.ndarray] | None = None,
    synthetic: SyntheticSet | None = None,
    gate: GradientGate | None = None,
) -> list[dict[str, Any]]:
    """Train one MLP arm under the shared synthetic-data mechanism.

    * `gold`   -- gold rows only.
    * `signed` -- gold + teacher-labeled supplementary rows, combined by the
      column task's signed PPI correction: `L_G + λ(L_S − L_pseudo_gold)`.
    * `gated`  -- the same two quantities, combined by the detached controller.

    The architecture is identical in all three; only the objective changes.
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    history: list[dict[str, Any]] = []
    target_kind = _target_kind(config)
    best_state: dict[str, torch.Tensor] | None = None
    best_score = float("inf")
    stale = 0
    supp_rng = np.random.default_rng(config.seed + 1)

    for epoch in range(1, config.epochs + 1):
        model.train()
        gold_losses: list[float] = []
        synthetic_losses: list[float] = []
        gate_stats_list = []
        supp_perm = (
            supp_rng.permutation(synthetic.n_rows) if synthetic is not None else np.zeros(0, dtype=int)
        )
        supp_cursor = 0
        for batch_rows, labels in _batches(train, config.batch_size, rng):
            control, indicator, delta, target, context = _gold_batch(
                train, batch_rows, task, config
            )
            gold_y, gold_delta = _predict(
                model, control, indicator, context, target_kind=target_kind
            )
            gold_loss = _arm_loss(
                gold_y, gold_delta, target, delta, labels, config, deg_map
            )
            optimizer.zero_grad(set_to_none=True)

            if mode == "gold" or synthetic is None:
                gold_loss.backward()
                optimizer.step()
                gold_losses.append(float(gold_loss.detach()))
                continue

            # Control variate: the same forward against the teacher's labels.
            pseudo_delta = synthetic.gold_delta[batch_rows]
            pseudo_y = control + pseudo_delta if target_kind == "delta" else pseudo_delta
            pseudo_loss = _arm_loss(
                gold_y, gold_delta, pseudo_y, pseudo_delta, labels, config, deg_map
            )

            take = min(len(batch_rows), len(supp_perm) - supp_cursor)
            if take <= 0:
                supp_cursor = 0
                supp_perm = supp_rng.permutation(synthetic.n_rows)
                take = min(len(batch_rows), len(supp_perm))
            chosen = np.sort(supp_perm[supp_cursor : supp_cursor + take])
            supp_cursor += take
            chosen_t = torch.as_tensor(chosen, dtype=torch.long)
            sc = synthetic.control[chosen_t]
            si = synthetic.index[chosen_t]
            supp_indicator = _indicator_from_index(si, task.n_genes)
            st_delta = synthetic.delta[chosen_t]
            st_y = sc + st_delta if target_kind == "delta" else st_delta
            supp_labels = [synthetic.labels[int(i)] for i in chosen]
            supp_y, supp_delta = _predict(
                model, sc, supp_indicator, None, target_kind=target_kind
            )
            extension_loss = _arm_loss(
                supp_y, supp_delta, st_y, st_delta, supp_labels, config, deg_map
            )

            if mode == "signed":
                terms = ppi_signed_correction(
                    gold=gold_loss,
                    pseudo_gold=pseudo_loss,
                    extension=extension_loss,
                    coefficient=config.ppi_lambda,
                    mass=1.0,
                )
                terms.total.backward()
                optimizer.step()
                gold_losses.append(float(terms.total.detach()))
                synthetic_losses.append(float(extension_loss.detach()))
            else:  # gated
                stats = gate.combine(gold_loss, extension_loss, model.parameters())
                optimizer.step()
                gold_losses.append(float(gold_loss.detach()))
                synthetic_losses.append(float(extension_loss.detach()))
                gate_stats_list.append(stats)

        validation_metrics = evaluate_mlp(model, validation, task, config)["metrics"]
        entry: dict[str, Any] = {
            "epoch": epoch,
            "train_gold_loss": float(np.mean(gold_losses)) if gold_losses else float("nan"),
            "train_loss": float(np.mean(gold_losses)) if gold_losses else float("nan"),
            "validation_mse_deg": validation_metrics["summary"]["mse_deg"],
            "validation_pearson": validation_metrics["summary"]["pearson"],
            "validation_direction_accuracy": validation_metrics["summary"]["direction_accuracy"],
        }
        if synthetic_losses:
            entry["train_synthetic_loss"] = float(np.mean(synthetic_losses))
        if gate_stats_list:
            entry["gate"] = {
                "cosine": float(np.mean([s.cosine for s in gate_stats_list])),
                "weight": float(np.mean([s.weight for s in gate_stats_list])),
                "gold_norm": float(np.mean([s.gold_norm for s in gate_stats_list])),
                "synthetic_norm": float(np.mean([s.synthetic_norm for s in gate_stats_list])),
            }
            entry["gate_active_fraction"] = float(
                np.mean([s.active for s in gate_stats_list])
            )
        history.append(entry)

        score = validation_metrics["summary"]["mse_deg"]
        if score < best_score - 1e-6:
            best_score = score
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= config.patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return history


def run_mlp_arms(
    *,
    train_examples: Sequence[PerturbationExample],
    validation_examples: Sequence[PerturbationExample],
    test_examples: Sequence[PerturbationExample],
    task: PerturbationTask,
    config: "PerturbationTrainingConfig",
    out_dir: str | Path,
    supplementary_controls: np.ndarray | None = None,
    deg_by_perturbation: Mapping[str, np.ndarray] | None = None,
) -> dict[str, dict[str, Any]]:
    """Train ``mlp_base``, ``mlp_augmented_ungated`` and ``mlp_augmented``.

    Same recipe as the GEARS arms -- a gold-only teacher labels the supplementary
    control cells, and the student sees gold (measured) plus supplementary
    (synthetic) rows under either the signed PPI correction or the gradient gate
    -- but with a graph-free MLP, so the two backends differ only by architecture.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    results: dict[str, dict[str, Any]] = {}
    mlp_config = MlpConfig(
        n_genes=task.n_genes,
        hidden_size=int(getattr(config, "mlp_hidden_size", 128)),
        context_categorical=config.context_categorical,
        context_numeric=config.context_numeric,
    )
    target_kind = _target_kind(config)

    import time

    def _make_model() -> MlpModel:
        return MlpModel(mlp_config)

    def _save_predictions(arm: str, evaluation: dict[str, Any]) -> str:
        labels = np.array(sorted(evaluation["predictions"]))
        path = out / f"{arm}_test_predictions.npz"
        np.savez_compressed(
            path,
            labels=labels,
            predictions=np.stack([evaluation["predictions"][label] for label in labels]),
            observations=np.stack([evaluation["observations"][label] for label in labels]),
        )
        return str(path)

    def _finish(arm: str, model: MlpModel, history, started, evaluation, validation) -> None:
        torch.save(model.state_dict(), out / f"{arm}.pt")
        results[arm] = {
            "validation": validation,
            "test": evaluation["metrics"],
            "history": history,
            "epochs_run": len(history),
            "seconds": round(time.time() - started, 1),
            "predictions": _save_predictions(arm, evaluation),
        }

    # -- Base MLP: gold only, and the teacher for the augmented arms --------
    torch.manual_seed(config.seed)
    base = _make_model()
    started = time.time()
    base_history = _train_arm(
        base, train_examples, validation_examples, task, config,
        np.random.default_rng(config.seed), mode="gold", deg_map=deg_by_perturbation,
    )
    base_validation = evaluate_mlp(base, validation_examples, task, config)["metrics"]
    base_test = evaluate_mlp(base, test_examples, task, config)
    _finish("mlp_base", base, base_history, started, base_test, base_validation)

    teacher = _make_model()
    teacher.load_state_dict(base.state_dict())
    teacher.eval()
    torch.save(teacher.state_dict(), out / "teacher_checkpoint.pt")

    synthetic: SyntheticSet | None = None
    if supplementary_controls is not None and len(supplementary_controls) > 0:
        def _teacher_predict(control, index, _teacher=teacher):
            with torch.no_grad():
                indicator = _indicator_from_index(index, task.n_genes)
                _, delta = _predict(
                    _teacher, control, indicator, None, target_kind=target_kind
                )
            return delta

        synthetic = build_synthetic(
            predict=_teacher_predict,
            train_examples=train_examples,
            task=task,
            controls=supplementary_controls,
            seed=config.seed,
        )
        if synthetic is not None:
            _write_supplementary_metadata(out, synthetic.summary(), config, target_kind)

    for arm, mode in (("mlp_augmented_ungated", "signed"), ("mlp_augmented", "gated")):
        model = _make_model()
        started = time.time()
        gate = (
            GradientGate(kappa=config.gate_kappa, scope="batch", coefficient=config.gate_lambda)
            if mode == "gated"
            else None
        )
        history = _train_arm(
            model, train_examples, validation_examples, task, config,
            np.random.default_rng(config.seed), mode=mode,
            deg_map=deg_by_perturbation, synthetic=synthetic, gate=gate,
        )
        validation = evaluate_mlp(model, validation_examples, task, config)["metrics"]
        test = evaluate_mlp(model, test_examples, task, config)
        _finish(arm, model, history, started, test, validation)
    return results


def _write_supplementary_metadata(
    out: Path,
    summary: Mapping[str, Any],
    config: "PerturbationTrainingConfig",
    target_kind: str,
) -> None:
    import json

    payload = {
        "objective": config.mlp_objective,
        "direction_lambda": config.direction_lambda,
        **dict(summary),
        # the convention the loss uses (delta), not the label source
        "target": target_kind,
        "evidence_weight": "uniform 1.0",
        "sampling": "with replacement",
        "gate_scope": "batch",
        "gate_kappa": config.gate_kappa,
        "gate_lambda": config.gate_lambda,
        "ppi_lambda": config.ppi_lambda,
        "seed": config.seed,
    }
    (Path(out) / "synthetic_label_metadata.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
