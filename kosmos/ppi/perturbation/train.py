"""Three arms on one contract: garde-base, ungated three-graph, gated augmented.

  * `gears_base`             -- gold co-expression graph + GO graph (gold only)
  * `gears_augmented_ungated` -- adds the supplementary co-expression graph and
                                trains on the augmented loss directly
  * `gears_augmented`         -- the same two passes, but the update is

        g_final = g_G + λ_t · g_ΔG,   g_ΔG = ∇(L_A − L_G)

    gated by the detached controller in `kosmos/ppi/gating.py` (batch scope
    only: a graph built from an entire supplementary source has no per-row
    reading). If the graph increment conflicts with the gold gradient,
    `λ_t = 0` and the update is exactly `g_G`.

All three see identical perturbation-level splits, and every arm is evaluated on
the same held-out perturbations, so the comparison is between models rather than
between data.
"""

from __future__ import annotations

import json
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch

from ..gating import GradientGate
from .contract import PerturbationExample, PerturbationTask
from .figures import write_perturbation_figures, write_summary
from .graphs import GraphArtifact, edge_overlap
from .losses import deg_indices, gears_loss, ppi_signed_correction
from .synthetic import SyntheticSet, build_synthetic
from .metrics import perturbation_metrics
from .model import GearsModel, ModelConfig, sparse_adjacency

ARMS = ("gears_base", "gears_augmented_ungated", "gears_augmented")


@dataclass
class PerturbationTrainingConfig:
    arms: Sequence[str] = ARMS
    epochs: int = 40
    patience: int = 8
    batch_size: int = 32
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    direction_lambda: float = 1e-1
    top_k_deg: int = 20
    embedding_dim: int = 32
    hidden_dim: int = 32
    gnn_layers: int = 2
    eta: float = 1.0
    gate_kappa: float = 1.0
    gate_lambda: float = 1.0
    seed: int = 42
    context_categorical: dict[str, int] = field(default_factory=dict)
    context_numeric: int = 0
    #: Graph-free MLP baselines (mlp_base / mlp_augmented). Off by default so a
    #: caller that does not ask for them gets exactly the three GEARS arms.
    include_mlp_baselines: bool = False
    #: Coefficient of the signed PPI correction in the ungated augmented arm
    #: (the column task's `ppi_lambda`).
    ppi_lambda: float = 0.5
    mlp_hidden_size: int = 128
    #: The MLP arms optimise the *same* objective as the GEARS arms by default:
    #: "gears" = the quartic DEG loss + direction term, on Δ (see losses.py).
    #: "mse" restores the design document's benchmark form (gene-averaged MSE
    #: on post-perturbation expression); it is recorded, but it breaks the
    #: like-for-like comparison.
    mlp_objective: str = "gears"


def _tensor_graphs(
    graphs: dict[str, GraphArtifact], n_genes: int, device: str
) -> dict[str, torch.Tensor]:
    out: dict[str, torch.Tensor] = {}
    for name, graph in graphs.items():
        edge_index = torch.as_tensor(graph.edge_index, dtype=torch.long, device=device)
        edge_weight = torch.as_tensor(graph.edge_weight, dtype=torch.float32, device=device)
        out[name] = sparse_adjacency(edge_index, edge_weight, n_genes)
    return out


def _batches(examples: Sequence[PerturbationExample], size: int, rng: np.random.Generator):
    order = rng.permutation(len(examples))
    for start in range(0, len(order), size):
        rows = order[start : start + size]
        labels = [examples[index].label for index in rows]
        if len(set(labels)) < 2:
            continue  # a one-perturbation batch makes the direction term degenerate
        control = torch.as_tensor(
            np.stack([examples[index].control for index in rows]), dtype=torch.float32
        )
        target = torch.as_tensor(
            np.stack([examples[index].delta for index in rows]), dtype=torch.float32
        )
        yield rows, labels, control, target


def _perturbation_index(
    examples: Sequence[PerturbationExample], rows: np.ndarray, task: PerturbationTask
) -> torch.Tensor:
    width = max(len(examples[index].perturbation) for index in rows)
    index = np.zeros((len(rows), width), dtype=np.int64)
    for position, row in enumerate(rows):
        for slot, gene in enumerate(examples[row].perturbation):
            index[position, slot] = task.gene_to_index[gene]
    return torch.as_tensor(index, dtype=torch.long)


def _context_tensors(
    examples: Sequence[PerturbationExample],
    rows: np.ndarray,
    config: PerturbationTrainingConfig,
) -> dict[str, torch.Tensor]:
    out: dict[str, torch.Tensor] = {}
    for name in config.context_categorical:
        values = [str(examples[row].context.get(name, "unknown")) for row in rows]
        out[name] = torch.as_tensor([hash(value) % config.context_categorical[name] for value in values], dtype=torch.long)
    if config.context_numeric:
        numeric = [
            [float(examples[row].context.get(f"numeric_{position}", 0.0)) for position in range(config.context_numeric)]
            for row in rows
        ]
        out["numeric"] = torch.as_tensor(numeric, dtype=torch.float32)
    return out


def _deg_map(examples: Sequence[PerturbationExample], top_k: int) -> dict[str, np.ndarray]:
    grouped: dict[str, list[np.ndarray]] = {}
    for example in examples:
        grouped.setdefault(example.label, []).append(example.delta)
    return {
        label: deg_indices(np.mean(np.stack(values), axis=0), top_k=top_k)
        for label, values in grouped.items()
    }


def _evaluate(
    model: GearsModel,
    examples: Sequence[PerturbationExample],
    task: PerturbationTask,
    graphs: dict[str, torch.Tensor],
    config: PerturbationTrainingConfig,
    *,
    augmented: bool,
) -> dict[str, Any]:
    model.eval()
    predictions: dict[str, list[np.ndarray]] = {}
    observations: dict[str, list[np.ndarray]] = {}
    with torch.no_grad():
        for start in range(0, len(examples), config.batch_size):
            chunk = examples[start : start + config.batch_size]
            rows = np.arange(start, start + len(chunk))
            control = torch.as_tensor(np.stack([e.control for e in chunk]), dtype=torch.float32)
            index = _perturbation_index(examples, rows, task)
            context = _context_tensors(examples, rows, config)
            _, delta = model(
                control,
                index,
                graphs["G_C"],
                graphs["G_S"] if augmented else None,
                graphs["G_GO"],
                context or None,
            )
            for position, example in enumerate(chunk):
                predictions.setdefault(example.label, []).append(delta[position].numpy())
                observations.setdefault(example.label, []).append(example.delta)
    mean_predictions = {label: np.mean(np.stack(values), axis=0) for label, values in predictions.items()}
    mean_observations = {label: np.mean(np.stack(values), axis=0) for label, values in observations.items()}
    metrics = perturbation_metrics(mean_predictions, mean_observations, top_k=config.top_k_deg)
    return metrics.to_dict(), mean_predictions, mean_observations


def run_perturbation_experiment(
    *,
    splits: dict[str, list[PerturbationExample]],
    task: PerturbationTask,
    graphs: dict[str, GraphArtifact],
    out_dir: str | Path,
    config: PerturbationTrainingConfig | None = None,
    supplementary_controls: np.ndarray | None = None,
    modality: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Train the requested arms on identical splits and report the comparison."""
    config = config or PerturbationTrainingConfig()
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(config.seed)
    rng = np.random.default_rng(config.seed)
    device = "cpu"
    adjacency = _tensor_graphs(graphs, task.n_genes, device)
    deg_map = _deg_map(splits["train"], config.top_k_deg)
    overlap = edge_overlap(graphs["G_C"], graphs["G_S"]) if "G_S" in graphs else {}

    results: dict[str, Any] = {
        "task": task.to_dict(),
        # Which kind of perturbation this is (knockout / knockdown / activation),
        # and whether it is what the question asked for. Recorded, not guessed:
        # see kosmos/ppi/perturbation/modality.py.
        "modality": modality or {},
        "arms": {},
        "split_sizes": {name: len(values) for name, values in splits.items()},
        "graph_edge_overlap": overlap,
        "config": {
            "epochs": config.epochs,
            "batch_size": config.batch_size,
            "learning_rate": config.learning_rate,
            "direction_lambda": config.direction_lambda,
            "top_k_deg": config.top_k_deg,
            "eta": config.eta,
            "gate_kappa": config.gate_kappa,
            "gate_lambda": config.gate_lambda,
            "seed": config.seed,
            "include_mlp_baselines": config.include_mlp_baselines,
            "mlp_objective": config.mlp_objective,
            "ppi_lambda": config.ppi_lambda,
        },
    }

    gears_arms = [arm for arm in config.arms if arm in ARMS]
    augmented_arms = [arm for arm in gears_arms if arm != "gears_base"]

    def _new_model() -> GearsModel:
        return GearsModel(
            ModelConfig(
                n_genes=task.n_genes,
                embedding_dim=config.embedding_dim,
                hidden_dim=config.hidden_dim,
                gnn_layers=config.gnn_layers,
                eta=config.eta,
                context_categorical=config.context_categorical,
                context_numeric=config.context_numeric,
            )
        ).to(device)

    # The teacher is the gold-only model. `gears_base` trains first, is frozen,
    # and then *generates* the supplementary labels -- the mechanism is "train on
    # gold, label supp, retrain", not a graph trick.
    synthetic: "SyntheticSet | None" = None
    supp_rng = np.random.default_rng(config.seed)

    for arm in gears_arms:
        model = _new_model()
        optimizer = torch.optim.Adam(
            model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
        )
        gate = (
            GradientGate(kappa=config.gate_kappa, scope="batch", coefficient=config.gate_lambda)
            if arm == "gears_augmented"
            else None
        )
        # The augmented arms keep the supplementary co-expression branch (the
        # panel is already the shared gene intersection), so they evaluate on
        # `H_A`; the base arm stays on `H_C`.
        augmented = arm != "gears_base"
        history: list[dict[str, Any]] = []
        best = None
        best_score = np.inf
        stale = 0
        started = time.time()
        for epoch in range(1, config.epochs + 1):
            model.train()
            epoch_losses: list[float] = []
            epoch_synthetic: list[float] = []
            gate_stats: list[dict[str, float]] = []
            supp_perm = (
                supp_rng.permutation(synthetic.n_rows)
                if synthetic is not None
                else np.zeros(0, dtype=int)
            )
            supp_cursor = 0
            for rows, labels, control, target in _batches(splits["train"], config.batch_size, rng):
                index = _perturbation_index(splits["train"], rows, task)
                context = _context_tensors(splits["train"], rows, config)
                if arm == "gears_base":
                    _, delta = model(
                        control, index, adjacency["G_C"], None, adjacency["G_GO"], context or None
                    )
                    loss = gears_loss(
                        delta, target, labels,
                        deg_by_perturbation=deg_map, direction_lambda=config.direction_lambda,
                    )
                    optimizer.zero_grad()
                    loss.total.backward()
                    optimizer.step()
                    epoch_losses.append(float(loss.total.detach()))
                    continue

                # Augmented arms: the same architecture, with the supplementary
                # graph *and* the teacher's synthetic labels on supp cells.
                optimizer.zero_grad(set_to_none=True)
                _, delta = model(
                    control, index, adjacency["G_C"], adjacency["G_S"],
                    adjacency["G_GO"], context or None,
                )
                loss_gold = gears_loss(
                    delta, target, labels,
                    deg_by_perturbation=deg_map, direction_lambda=config.direction_lambda,
                )
                if synthetic is None:
                    # No supplementary evidence: this reduces to the gold update.
                    loss_gold.total.backward()
                    optimizer.step()
                    epoch_losses.append(float(loss_gold.total.detach()))
                    continue

                # Control variate: the *same* forward, scored against the
                # teacher's labels on the gold rows.
                loss_pseudo_gold = gears_loss(
                    delta, synthetic.gold_delta[rows], labels,
                    deg_by_perturbation=deg_map, direction_lambda=config.direction_lambda,
                )
                take = min(len(rows), len(supp_perm) - supp_cursor)
                if take <= 0:
                    supp_cursor = 0
                    supp_perm = supp_rng.permutation(synthetic.n_rows)
                    take = min(len(rows), len(supp_perm))
                srows = np.sort(supp_perm[supp_cursor : supp_cursor + take])
                supp_cursor += take
                supp_labels = [synthetic.labels[int(i)] for i in srows]
                _, delta_supp = model(
                    synthetic.control[srows], synthetic.index[srows],
                    adjacency["G_C"], adjacency["G_S"], adjacency["G_GO"], None,
                )
                loss_extension = gears_loss(
                    delta_supp, synthetic.delta[srows], supp_labels,
                    deg_by_perturbation=deg_map, direction_lambda=config.direction_lambda,
                )

                if gate is None:
                    # Ungated = the column task's signed PPI correction:
                    #   L = L_gold + λ·(L_ext − L_pseudo_gold)
                    terms = ppi_signed_correction(
                        gold=loss_gold.total,
                        pseudo_gold=loss_pseudo_gold.total,
                        extension=loss_extension.total,
                        coefficient=config.ppi_lambda,
                        mass=1.0,
                    )
                    terms.total.backward()
                    optimizer.step()
                    epoch_losses.append(float(terms.total.detach()))
                    epoch_synthetic.append(float(loss_extension.total.detach()))
                else:
                    # Gated: gold sets the direction, the synthetic gradient only
                    # accelerates where it agrees.
                    stats = gate.combine(
                        loss_gold.total, loss_extension.total, model.parameters()
                    )
                    optimizer.step()
                    epoch_losses.append(float(loss_gold.total.detach()))
                    epoch_synthetic.append(float(loss_extension.total.detach()))
                    gate_stats.append(
                        {
                            "cosine": stats.cosine,
                            "weight": stats.weight,
                            "gold_norm": stats.gold_norm,
                            "increment_norm": stats.synthetic_norm,
                            "active": float(stats.active),
                        }
                    )
            validation, _, _ = _evaluate(
                model, splits["validation"], task, adjacency, config, augmented=augmented
            )
            score = validation["summary"]["mse_deg"]
            entry = {
                "epoch": epoch,
                "train_loss": float(np.mean(epoch_losses)) if epoch_losses else float("nan"),
                "validation_mse_deg": score,
                "validation_pearson": validation["summary"]["pearson"],
                "validation_direction_accuracy": validation["summary"]["direction_accuracy"],
            }
            if epoch_synthetic:
                entry["train_synthetic_loss"] = float(np.mean(epoch_synthetic))
            if gate_stats:
                entry["gate"] = {
                    key: float(np.mean([stats[key] for stats in gate_stats]))
                    for key in ("cosine", "weight", "gold_norm", "increment_norm", "active")
                }
                entry["gate_active_fraction"] = float(np.mean([stats["active"] for stats in gate_stats]))
            history.append(entry)
            if score < best_score - 1e-6:
                best_score = score
                best = {key: value.detach().clone() for key, value in model.state_dict().items()}
                stale = 0
            else:
                stale += 1
                if stale >= config.patience:
                    break
        if best is not None:
            model.load_state_dict(best)
        if arm == "gears_base" and augmented_arms:
            teacher = _new_model()
            teacher.load_state_dict(model.state_dict())
            teacher.eval()
            if supplementary_controls is not None and len(supplementary_controls) > 0:
                def _teacher_predict(control, index, _teacher=teacher):
                    with torch.no_grad():
                        _, delta = _teacher(
                            control, index, adjacency["G_C"], None, adjacency["G_GO"]
                        )
                    return delta

                synthetic = build_synthetic(
                    predict=_teacher_predict,
                    train_examples=splits["train"],
                    task=task,
                    controls=supplementary_controls,
                    seed=config.seed,
                )
        test, predictions, observations = _evaluate(
            model, splits["test"], task, adjacency, config, augmented=augmented
        )
        torch.save(model.state_dict(), out / f"{arm}.pt")
        # The per-perturbation means are what a figure needs, and they are small
        # next to the model: kept as an artifact so a plot never has to re-train.
        np.savez_compressed(
            out / f"{arm}_test_predictions.npz",
            labels=np.array(sorted(predictions)),
            predictions=np.stack([predictions[label] for label in sorted(predictions)]),
            observations=np.stack([observations[label] for label in sorted(predictions)]),
        )
        results["arms"][arm] = {
            "validation": validation,
            "test": test,
            "history": history,
            "epochs_run": len(history),
            "seconds": round(time.time() - started, 1),
            "predictions": str(out / f"{arm}_test_predictions.npz"),
        }
    if config.include_mlp_baselines:
        # Graph-free baselines. A local import keeps the GEARS path free of the
        # MLP module (and costs nothing when the baselines are off).
        from .mlp import run_mlp_arms

        results["arms"].update(
            run_mlp_arms(
                train_examples=splits["train"],
                validation_examples=splits["validation"],
                test_examples=splits["test"],
                task=task,
                config=config,
                out_dir=out,
                supplementary_controls=supplementary_controls,
                # the same DEG sets the GEARS arms train on, so the objective is
                # identical across every arm
                deg_by_perturbation=deg_map,
            )
        )
    figures = write_perturbation_figures(results, out)
    results["figures"] = [str(path) for path in figures]
    write_summary(results, out)
    (out / "perturbation_results.json").write_text(
        json.dumps(results, indent=2, default=str), encoding="utf-8"
    )
    return results
