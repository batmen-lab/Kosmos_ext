"""CPU-deterministic, classification-first PPI runner with a matched baseline."""

import copy
import json
import random
import tempfile
from dataclasses import replace
from hashlib import sha256
from pathlib import Path

import numpy as np
import sklearn
import torch
from torch import nn

from .datasets import prepare_external
from .gating import GateStats, GradientGate
from .gating import summarise as summarise_gate
from .losses import PPILoss, PseudoLabelLoss, SupervisedLoss
from .metrics import classification_metrics, regression_metrics
from .pseudo_labeler import (
    PreparedPseudoLabeler,
    PretrainedPseudoLabeler,
    fingerprint,
    prepare_pseudo_labeler,
)
from .schemas import PPITrainingConfig, PPITrainingResult
from .split import validate_roles


def check_matrix(matrix, *, what: str, max_abs: float = 1e12) -> None:
    """Refuse a matrix that a forward pass would answer with a crash.

    A NaN, an infinity, or a value large enough to overflow a float32 is not
    something the loss can learn from: it poisons every gradient, and in a C
    extension it can take the whole process down (the run that ended in
    `Segmentation fault (core dumped)` inside `torch.nn.Module._call_impl`).
    Cheaper and clearer to say which table produced it.
    """
    array = np.asarray(matrix, dtype=np.float32)
    if array.size == 0:
        raise ValueError(f"{what} is empty, so there is nothing to train on")
    bad = ~np.isfinite(array)
    if bad.any():
        rows, columns = np.argwhere(bad)[0]
        fraction = bad.mean()
        raise ValueError(
            f"{what} contains {int(bad.sum()):,} non-finite value(s) "
            f"({fraction:.1%} of the matrix), first at row {rows}, column "
            f"{columns}; the encoder produced them, so check for an empty or "
            f"constant column being standardised"
        )
    largest = float(np.abs(array).max())
    if largest > max_abs:
        raise ValueError(
            f"{what} has a value of {largest:g}, which overflows what the model "
            f"can use; scale the column or drop it"
        )
    if largest == 0.0:
        raise ValueError(
            f"{what} is all zeros: every feature was empty or dropped, so the "
            f"model would learn nothing"
        )


def linear_model_factory(n_features, n_classes):
    return nn.Linear(n_features, n_classes)


def predict_model(model, X, n_classes, batch_size=1024):
    model.eval()
    probabilities = []
    with torch.no_grad():
        for start in range(0, len(X), batch_size):
            logits = model(torch.as_tensor(X[start : start + batch_size], dtype=torch.float32))
            if (
                logits.shape != (min(batch_size, len(X) - start), n_classes)
                or not torch.isfinite(logits).all()
            ):
                raise ValueError("Downstream model must return finite [batch, class] logits")
            probabilities.append(torch.softmax(logits, dim=1).cpu().numpy())
    return np.concatenate(probabilities)


def predict_values(model, X, batch_size=1024):
    """The model's value prediction for a regression task: one number per row."""
    model.eval()
    values = []
    with torch.no_grad():
        for start in range(0, len(X), batch_size):
            output = model(torch.as_tensor(X[start : start + batch_size], dtype=torch.float32))
            if output.dim() > 1 and output.shape[-1] == 1:
                output = output.squeeze(-1)
            expected = (min(batch_size, len(X) - start),)
            if tuple(output.shape) != expected or not torch.isfinite(output).all():
                raise ValueError(
                    "A regression model must return one finite value per row "
                    f"(expected {expected}, got {tuple(output.shape)})"
                )
            values.append(output.cpu().numpy())
    return np.concatenate(values) if values else np.zeros(0, dtype=float)


def encode(labels, classes):
    mapping = {value: i for i, value in enumerate(classes)}
    return np.array([mapping[value] for value in labels], dtype=np.int64)


def target_scaler(y) -> tuple[float, float]:
    """Mean and spread of a measured target, so the model can learn its scale.

    A regressor starts at roughly zero, and Adam moves a parameter by about the
    learning rate per step: asking it to reach a target centred on 18 takes tens
    of thousands of steps, which is why an unscaled regression target trains to
    a flat line. The scaler is fitted on the labeled rows and inverted before
    any metric is computed, so every reported number stays in the target's own
    units.
    """
    values = np.asarray(y, dtype=float).reshape(-1)
    mean = float(values.mean())
    std = float(values.std())
    return mean, (std if std > 0 else 1.0)


def _fit(
    initial,
    gold,
    validation,
    external,
    classes,
    config,
    *,
    ppi,
    loss_factory,
    scaler: tuple[float, float] = (0.0, 1.0),
):
    regression = str(config.task_type) == "regression"
    mean, std = scaler
    torch.manual_seed(config.seed)
    model = copy.deepcopy(initial)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    correction_optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate * config.stage2_lr_multiplier,
        weight_decay=config.weight_decay,
    )
    loss_fn = loss_factory(config.ppi_lambda, config.label_smoothing)
    if not hasattr(loss_fn, "task_type"):
        if regression:
            raise ValueError(
                "this loss_factory cannot train a regression task: it has no "
                "task_type, so it would compare class indices against measured "
                "values"
            )
    else:
        loss_fn.task_type = config.task_type
    if hasattr(loss_fn, "ramp_epochs"):
        # The ramp is a property of the loss, but the config decides it; wiring
        # it here keeps every loss's constructor signature at (λ, smoothing).
        loss_fn.ramp_epochs = max(0, int(getattr(config, "loss_ramp_epochs", 0) or 0))
        loss_fn.schedule_epoch(0, config.max_epochs)
    X = torch.as_tensor(gold.X)
    if regression:
        y = torch.as_tensor(np.asarray(gold.y_true, dtype=float), dtype=torch.float32)
        yp = torch.as_tensor(np.asarray(gold.y_pred, dtype=float), dtype=torch.float32)
    else:
        y = torch.as_tensor(encode(gold.y_true, classes))
        yp = (
            torch.as_tensor(gold.y_prob, dtype=torch.float32)
            if config.pseudo_targets == "soft"
            else torch.as_tensor(encode(gold.y_pred, classes))
        )
    mass = sum(d.weights.sum() for d in external)
    # The gate has its own coefficient: a caller who sets `--lambda 0` on the
    # signed objective means "no correction at all", but the same number must not
    # silently switch the gate off -- its default 1.0 is the gate's own weight.
    external_coefficient = (
        config.gate_lambda if config.loss_mode == "gradient_gated" else config.ppi_lambda
    )
    use_external = ppi and bool(external) and mass > 0 and external_coefficient > 0
    gate = (
        GradientGate(
            kappa=config.gate_kappa,
            scope=config.gate_scope,
            gamma=config.gate_gamma,
            coefficient=config.gate_lambda,
        )
        if use_external and config.loss_mode == "gradient_gated"
        else None
    )
    if use_external:
        Xe = torch.as_tensor(np.concatenate([d.X for d in external]))
        ye = (
            torch.as_tensor(
                np.asarray(np.concatenate([d.y_pred for d in external]), dtype=float),
                dtype=torch.float32,
            )
            if regression
            else (
                torch.as_tensor(np.concatenate([d.y_prob for d in external]), dtype=torch.float32)
                if config.pseudo_targets == "soft"
                else torch.as_tensor(
                    encode(np.concatenate([d.y_pred for d in external]), classes)
                )
            )
        )
        weights = torch.as_tensor(
            np.concatenate([d.weights for d in external]), dtype=torch.float32
        )
    gold_rng = np.random.default_rng(config.seed)
    external_rng = np.random.default_rng(config.seed + 1)
    history, best_score, best_epoch, stale = [], -np.inf, 0, 0
    best_state = copy.deepcopy(model.state_dict())
    for epoch in range(1, config.max_epochs + 1):
        if hasattr(loss_fn, "schedule_epoch"):
            loss_fn.schedule_epoch(epoch, config.max_epochs)
        correction_stage = (
            use_external
            and config.schedule == "two_stage"
            and (epoch - 1) % (config.stage1_epochs + config.stage2_epochs) >= config.stage1_epochs
        )
        selected_optimizer = correction_optimizer if correction_stage else optimizer
        model.train()
        totals = dict.fromkeys(
            [
                "loss_true",
                "loss_pseudo_gold",
                "loss_correction",
                "loss_external",
                "loss_total",
                "loss_optimized",
            ],
            0.0,
        )
        gate_stats: list[GateStats] = []
        order = gold_rng.permutation(len(X))
        for start in range(0, len(X), config.gold_batch_size):
            idx = order[start : start + config.gold_batch_size]
            gold_logits = model(X[idx])
            if gate is not None:
                # The labeled rows are the anchor and the synthetic rows are
                # evidence: two losses, two gradients, and the gate decides how
                # much of the second may be applied.
                ei = external_rng.integers(
                    len(Xe), size=min(config.external_batch_size, len(Xe))
                )
                external_logits = model(Xe[ei])
                gold_loss = loss_fn.labeled_term(gold_logits, y[idx])
                synthetic_loss = loss_fn.external_term(
                    external_logits, ye[ei], weights[ei], len(Xe)
                )
                stats = gate.combine(
                    gold_loss,
                    synthetic_loss,
                    model.parameters(),
                    synthetic_per_row=(
                        loss_fn.per_row(external_logits, ye[ei])
                        if config.gate_scope == "sample"
                        else None
                    ),
                )
                if not np.isfinite(stats.gold_norm) or not np.isfinite(stats.weight):
                    raise ValueError("Non-finite gradient gate statistics")
                selected_optimizer.step()
                selected_optimizer.zero_grad(set_to_none=True)
                gate_stats.append(stats)
                weight = len(idx) / len(X)
                totals["loss_true"] += stats.gold_loss * weight
                totals["loss_external"] += stats.synthetic_loss * weight
                totals["loss_correction"] += (
                    stats.synthetic_loss * stats.weight * weight
                )
                totals["loss_total"] += (
                    stats.gold_loss + stats.synthetic_loss * stats.weight
                ) * weight
                totals["loss_optimized"] += (
                    stats.gold_loss + stats.synthetic_loss * stats.weight
                ) * weight
                continue
            if use_external:
                ei = external_rng.integers(len(Xe), size=min(config.external_batch_size, len(Xe)))
                terms = loss_fn.total_loss(
                    gold_logits,
                    y[idx],
                    yp[idx],
                    model(Xe[ei]),
                    ye[ei],
                    weights[ei],
                    len(Xe),
                    float(mass),
                )
            else:
                terms = loss_fn.total_loss(gold_logits, y[idx], yp[idx])
            objective = (
                terms["loss_correction"]
                if correction_stage
                else terms[
                    "loss_total" if use_external and config.schedule == "joint" else "loss_true"
                ]
            )
            if not torch.isfinite(objective):
                raise ValueError("Non-finite PPI objective")
            selected_optimizer.zero_grad()
            objective.backward()
            selected_optimizer.step()
            for name, value in terms.items():
                totals[name] += float(value.detach()) * len(idx) / len(X)
            totals["loss_optimized"] += float(objective.detach()) * len(idx) / len(X)
        metrics = (
            regression_metrics(
                validation.y, predict_values(model, validation.X) * std + mean
            )
            if regression
            else classification_metrics(
                validation.y, predict_model(model, validation.X, len(classes)), classes
            )
        )
        metric = config.evaluation_metric
        # Explicit fallback for folds on which the requested ranking metric is
        # undefined. Every metric in the fallback chain is "higher is better",
        # so the selection below stays a max() for both task types.
        if metrics.get(metric) is None:
            for fallback in (("pearson", "neg_mse") if regression else ("balanced_accuracy",)):
                if metrics.get(fallback) is not None:
                    metric = fallback
                    break
        score = metrics[metric]
        entry = {
            "epoch": epoch,
            "phase": (
                "gated"
                if gate is not None
                else (
                    "correction"
                    if correction_stage
                    else ("joint" if use_external and config.schedule == "joint" else "true")
                )
            ),
            **totals,
            "validation": metrics,
            "selection_metric": metric,
        }
        if gate_stats:
            # The gate's own record: how compatible the synthetic gradient was,
            # how much of it survived the magnitude clip, and how often any of it
            # was applied at all. A stress test reads exactly these numbers.
            summary = summarise_gate(gate_stats)
            entry["gate"] = summary
            entry.update(
                {
                    "train_gold_loss": summary["train_gold_loss"],
                    "train_synthetic_loss": summary["train_synthetic_loss"],
                    "gold_grad_norm": summary["gold_grad_norm"],
                    "synthetic_grad_norm": summary["synthetic_grad_norm"],
                    "gradient_cosine": summary["gradient_cosine_mean"],
                    "gradient_cosine_std": summary["gradient_cosine_std"],
                    "synthetic_weight": summary["synthetic_weight_mean"],
                    "synthetic_weight_std": summary["synthetic_weight_std"],
                    "synthetic_active_fraction": summary["synthetic_active_fraction"],
                }
            )
            for name in (
                "accuracy",
                "balanced_accuracy",
                "macro_f1",
                "mse",
                "mae",
                "r2",
            ):
                if name in metrics:
                    entry[f"validation_{name}"] = metrics[name]
        history.append(entry)
        if score > best_score + config.min_delta:
            best_score, best_epoch, stale = score, epoch, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            stale += 1
        if stale >= config.patience:
            break
    model.load_state_dict(best_state)
    model.eval()
    return model, history, best_epoch


def _distribution(values):
    if not len(values):
        return {"n": 0}
    return {
        "n": len(values),
        "min": float(np.min(values)),
        "mean": float(np.mean(values)),
        "max": float(np.max(values)),
        "quantiles": np.quantile(values, [0.1, 0.5, 0.9]).tolist(),
    }


def run_ppi_experiment(
    *,
    gold_train,
    external_evidence,
    gold_validation,
    pseudo_labeler,
    model_factory=linear_model_factory,
    config=None,
    output_dir=None,
    loss_factory=None,
):
    """No final_test argument. Fits neither pseudo-labeler nor preprocessing on validation.

    CPU is intentional in this first implementation. A model factory receives
    (feature_count, class_count) and returns a fresh torch module producing logits.
    """
    config = PPITrainingConfig.model_validate((config or PPITrainingConfig()).model_dump())
    regression = str(config.task_type) == "regression"
    if loss_factory is None:
        # `plain` is a different objective, not a different schedule: it trains
        # on the external rows' soft pseudo labels and subtracts nothing.
        # `gold_plus_synthetic` is ordinary supervision on both batches -- the
        # naive baseline the gradient gate is measured against -- and
        # `gradient_gated` uses the same two terms, but combines their
        # *gradients* instead of their values.
        if config.loss_mode == "plain":
            loss_factory = PseudoLabelLoss
        elif config.loss_mode in {"gold_plus_synthetic", "gradient_gated"}:
            loss_factory = SupervisedLoss
        else:
            loss_factory = PPILoss
    external_evidence = list(external_evidence)
    validate_roles(gold_train, gold_validation, external_evidence)
    classes = np.array([], dtype=float) if regression else np.unique(gold_train.y)
    scaler = (
        target_scaler(gold_train.y) if regression else (0.0, 1.0)
    )
    if regression:
        # Train on the scaled target: the teacher, the pseudo-labels and the
        # loss all speak the same units, and the held-out labels are untouched
        # because they are only ever used to score an unscaled prediction.
        scaled = (np.asarray(gold_train.y, dtype=float) - scaler[0]) / scaler[1]
        gold_train = replace(gold_train, y=scaled)
    if not regression and not set(gold_validation.y).issubset(set(classes)):
        raise ValueError("Validation contains classes absent from gold training")
    known_ids = set()
    if isinstance(pseudo_labeler, PretrainedPseudoLabeler):
        known_ids = pseudo_labeler.training_sample_ids
    elif isinstance(pseudo_labeler, PreparedPseudoLabeler):
        known_ids = set(pseudo_labeler.provenance.get("known_training_sample_ids", []))
    if known_ids.intersection(gold_validation.sample_ids):
        raise ValueError("Pretrained training samples overlap validation")
    out = (
        Path(output_dir) if output_dir is not None else Path(tempfile.mkdtemp(prefix="kosmos-ppi-"))
    )
    out.mkdir(parents=True, exist_ok=True)
    if any(out.iterdir()):
        raise ValueError(
            "Use an empty output directory; existing experiments are never overwritten"
        )
    numpy_state, python_state = np.random.get_state(), random.getstate()
    try:
        with torch.random.fork_rng(devices=[]):
            np.random.seed(config.seed)
            random.seed(config.seed)
            torch.manual_seed(config.seed)
            labeler = prepare_pseudo_labeler(gold_train, pseudo_labeler, config)
            external = prepare_external(external_evidence, labeler, config)
            if (
                not regression
                and config.pseudo_targets == "soft"
                and any(d.y_prob is None for d in external)
            ):
                raise ValueError("Soft external pseudo targets require probabilities")
            # Before anything reaches torch: a matrix it cannot use is a clear
            # error here and a segfault in there.
            check_matrix(gold_train.X, what="the labeled training matrix")
            if external_evidence:
                for dataset in external_evidence:
                    check_matrix(
                        dataset.X,
                        what=f"the evidence matrix from {dataset.source_dataset_id!r}",
                    )
            initial = model_factory(
                gold_train.X.shape[1], 1 if regression else len(classes)
            ).cpu()
            baseline, baseline_history, baseline_epoch = _fit(
                initial,
                labeler.pseudo_gold,
                gold_validation,
                [],
                classes,
                config,
                ppi=False,
                loss_factory=loss_factory,
                scaler=scaler,
            )
            if (
                not external
                or sum(d.weights.sum() for d in external) == 0
                or (
                    config.gate_lambda
                    if config.loss_mode == "gradient_gated"
                    else config.ppi_lambda
                )
                == 0
            ):
                model, history, epoch = (
                    copy.deepcopy(baseline),
                    copy.deepcopy(baseline_history),
                    baseline_epoch,
                )
            else:
                model, history, epoch = _fit(
                    initial,
                    labeler.pseudo_gold,
                    gold_validation,
                    external,
                    classes,
                    config,
                    ppi=True,
                    loss_factory=loss_factory,
                    scaler=scaler,
                )
    finally:
        np.random.set_state(numpy_state)
        random.setstate(python_state)
    if regression:
        # The model was trained on the scaled target; a metric has to be in the
        # target's own units.
        baseline_probability = predict_values(baseline, gold_validation.X) * scaler[1] + scaler[0]
        probability = predict_values(model, gold_validation.X) * scaler[1] + scaler[0]
        baseline_metrics = regression_metrics(gold_validation.y, baseline_probability)
        metrics = regression_metrics(gold_validation.y, probability)
    else:
        baseline_probability = predict_model(baseline, gold_validation.X, len(classes))
        probability = predict_model(model, gold_validation.X, len(classes))
        baseline_metrics = classification_metrics(gold_validation.y, baseline_probability, classes)
        metrics = classification_metrics(gold_validation.y, probability, classes)
    delta = {
        k: (
            metrics[k] - baseline_metrics[k]
            if metrics[k] is not None and baseline_metrics[k] is not None
            else None
        )
        for k in metrics
        if k != "undefined"
    }
    route_counts = {}
    for data in external:
        route = data.evidence_metadata["evidence_route"]
        route_counts[route] = route_counts.get(route, 0) + len(data.X)
    config_json = config.model_dump_json()
    pseudo_counts = {}
    all_pred = np.concatenate([d.y_pred for d in external]) if external else np.array([])
    if not regression:
        for c in classes:
            pseudo_counts[str(c)] = int(np.sum(all_pred == c))
    provenance = {
        "config": config.model_dump(),
        "config_sha256": sha256(config_json.encode()).hexdigest(),
        # Regression only: the model predicts (y - mean) / std, so every
        # prediction has to be mapped back before it means anything.
        "target_scaler": {"mean": scaler[0], "std": scaler[1]},
        "seed": config.seed,
        "device": "cpu",
        "numpy_version": np.__version__,
        "torch_version": torch.__version__,
        "sklearn_version": sklearn.__version__,
        "gold_fingerprint": fingerprint(gold_train),
        "feature_names": gold_train.feature_names,
        "validation_fingerprint": fingerprint(gold_validation),
        "split_ids": {
            "gold_train": gold_train.sample_ids.tolist(),
            "gold_validation": gold_validation.sample_ids.tolist(),
        },
        "development_groups": (
            sorted(set(gold_train.groups) | set(gold_validation.groups))
            if gold_train.groups is not None
            else []
        ),
        "external_ids": [i for d in external for i in d.sample_ids.tolist()],
        "external_available_ids": [i for d in external_evidence for i in d.sample_ids.tolist()],
        "external_sources_available": [
            {
                "dataset_id": d.source_dataset_id,
                "n": len(d.X),
                "route": d.evidence_route,
                "sha256": sha256(d.X.tobytes()).hexdigest(),
            }
            for d in external_evidence
        ],
        "external_groups": sorted(
            {str(g) for d in external_evidence if d.groups is not None for g in d.groups}
        ),
        "pseudo_labeler": labeler.provenance,
        "model_factory": f"{model_factory.__module__}.{model_factory.__qualname__}",
        "model_structure": str(initial),
        "loss": f"{loss_factory.__module__}.{loss_factory.__qualname__}",
        "route_counts": route_counts,
        "pseudo_label_distribution": pseudo_counts,
        "gold_pseudo_agreement": (
            float(np.mean(labeler.pseudo_gold.y_pred == gold_train.y))
            if not regression
            else None
        ),
        "gold_pseudo_mae": (
            float(
                np.mean(
                    np.abs(
                        np.asarray(labeler.pseudo_gold.y_pred, dtype=float)
                        - np.asarray(gold_train.y, dtype=float)
                    )
                )
            )
            if regression
            else None
        ),
        "pseudo_value_distribution": (
            _distribution(np.asarray(all_pred, dtype=float)) if regression else None
        ),
        "confidence_distribution": (
            _distribution(
                np.concatenate([d.y_prob.max(1) for d in external if d.y_prob is not None])
            )
            if any(d.y_prob is not None for d in external)
            else {"status": "unavailable"}
        ),
        "weight_distribution": (
            _distribution(np.concatenate([d.weights for d in external])) if external else {"n": 0}
        ),
        "external_weight_mass": float(sum(d.weights.sum() for d in external)),
        "source_audits": [d.evidence_metadata for d in external],
    }
    model_path, baseline_path = out / "ppi_model.pt", out / "baseline_model.pt"
    for path, fitted in ((model_path, model), (baseline_path, baseline)):
        torch.save(
            {
                "state_dict": fitted.state_dict(),
                "classes": classes.tolist(),
                "feature_names": gold_train.feature_names,
                "config": config.model_dump(),
            },
            path,
        )
    training = _write_training_log(out, baseline_history, history, config)
    result = PPITrainingResult(
        model,
        baseline,
        classes,
        str(model_path),
        str(baseline_path),
        {"baseline": baseline_history, "ppi": history},
        metrics,
        baseline_metrics,
        delta,
        len(gold_train.X),
        sum(len(d.X) for d in external_evidence),
        sum(len(d.X) for d in external),
        [d.evidence_metadata["source_dataset_id"] for d in external],
        provenance,
        {"baseline": baseline_epoch, "ppi": epoch},
        labeler.pseudo_gold,
        external,
        training,
    )
    gold = labeler.pseudo_gold

    def save_pseudo(path, data, **extra):
        payload = {"X": data.X, "sample_ids": data.sample_ids, "y_pred": data.y_pred, **extra}
        if data.y_prob is not None:
            payload["y_prob"] = data.y_prob
        np.savez_compressed(path, **payload)

    save_pseudo(out / "pseudo_gold.npz", gold, y_true=gold.y_true)
    for i, data in enumerate(external):
        save_pseudo(out / f"pseudo_external_{i}.npz", data, weights=data.weights)
    np.savez_compressed(
        out / "validation_predictions.npz",
        sample_ids=gold_validation.sample_ids,
        y_true=gold_validation.y,
        baseline_probability=baseline_probability,
        ppi_probability=probability,
    )
    (out / "result.json").write_text(
        json.dumps(result.summary(), indent=2, allow_nan=False), encoding="utf-8"
    )
    return result


def _write_training_log(
    out: Path,
    baseline_history: list[dict],
    ppi_history: list[dict],
    config: PPITrainingConfig,
) -> dict:
    """Persist the training process: one line per epoch per arm, plus a summary."""
    lines: list[str] = []
    for arm, history in (("baseline", baseline_history), ("ppi", ppi_history)):
        for record in history:
            lines.append(json.dumps({"arm": arm, **record}, default=str))
    (out / "training_log.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")

    def arm_summary(history: list[dict]) -> dict:
        if not history:
            return {"epochs_run": 0, "best_epoch": None, "stopped": "no epochs"}

        def score(row: dict) -> float:
            value = row["validation"].get(row["selection_metric"])
            return -np.inf if value is None else float(value)

        best = max(history, key=score)
        stopped = (
            "patience"
            if len(history) < config.max_epochs
            else "max_epochs"
        )
        return {
            "epochs_run": len(history),
            "best_epoch": best["epoch"],
            "stopped": stopped,
            "best_validation": best["validation"][best["selection_metric"]],
            "selection_metric": best["selection_metric"],
            "phases": sorted({row["phase"] for row in history}),
            "first_epoch_loss": history[0].get("loss_total"),
            "last_epoch_loss": history[-1].get("loss_total"),
        }

    gate_epochs = [
        {
            key: row.get(key)
            for key in (
                "epoch",
                "gradient_cosine",
                "gradient_cosine_std",
                "synthetic_weight",
                "synthetic_weight_std",
                "synthetic_active_fraction",
                "gold_grad_norm",
                "synthetic_grad_norm",
                "train_gold_loss",
                "train_synthetic_loss",
                "validation_accuracy",
                "validation_balanced_accuracy",
                "validation_macro_f1",
            )
        }
        for row in ppi_history
        if row.get("gate")
    ]
    training = {
        "max_epochs": config.max_epochs,
        "patience": config.patience,
        "arms": {
            "baseline": arm_summary(baseline_history),
            "ppi": arm_summary(ppi_history),
        },
    }
    if gate_epochs:
        # The gate's per-epoch record travels with the run: whether the synthetic
        # gradient agreed, how loud it was allowed to be, and how often it was
        # used at all. `mean`/`std` over epochs answer "did the gate adapt?".
        import numpy as np

        cosine = np.array([row["gradient_cosine"] or 0.0 for row in gate_epochs], dtype=float)
        weight = np.array([row["synthetic_weight"] or 0.0 for row in gate_epochs], dtype=float)
        active = np.array(
            [row["synthetic_active_fraction"] or 0.0 for row in gate_epochs], dtype=float
        )
        training["gate"] = {
            "scope": config.gate_scope,
            "kappa": config.gate_kappa,
            "gamma": config.gate_gamma,
            "lambda": config.gate_lambda,
            "epochs": gate_epochs,
            "gradient_cosine_mean": float(cosine.mean()),
            "gradient_cosine_std": float(cosine.std()),
            "synthetic_weight_mean": float(weight.mean()),
            "synthetic_weight_std": float(weight.std()),
            "synthetic_active_fraction": float(active.mean()),
        }
    return training


train_ppi = run_ppi_experiment


def evaluate_final_test(result, final_test):
    """Explicit post-selection operation; never modifies selection or training history."""
    if final_test.role != "final_test":
        raise ValueError("Explicit final evaluation requires final_test role")
    final_test.__post_init__()
    used = set(result.reproducibility["external_ids"])
    used.update(result.reproducibility["external_available_ids"])
    for ids in result.reproducibility["split_ids"].values():
        used.update(ids)
    used.update(result.reproducibility["pseudo_labeler"].get("known_training_sample_ids", []))
    if used.intersection(final_test.sample_ids):
        raise ValueError("Final test overlaps development/pretraining observations")
    groups = set(result.reproducibility["development_groups"]) | set(
        result.reproducibility["external_groups"]
    )
    if groups and final_test.groups is None:
        raise ValueError("Final test group metadata is required")
    if final_test.groups is not None and groups.intersection(final_test.groups):
        raise ValueError("Final test overlaps development groups")
    if final_test.feature_names != result.reproducibility["feature_names"]:
        raise ValueError("Final test ordered feature schema mismatch")
    if final_test.X.shape[1] != result.pseudo_gold.X.shape[1]:
        raise ValueError("Final test feature/label schema mismatch")
    regression = (
        str((result.reproducibility.get("config") or {}).get("task_type", "classification"))
        == "regression"
    )
    if regression:
        if not np.isfinite(np.asarray(final_test.y, dtype=float)).all():
            raise ValueError("A regression test set must carry finite numbers")
        mean = float(result.reproducibility["target_scaler"]["mean"])
        std = float(result.reproducibility["target_scaler"]["std"])
        return {
            "ppi": regression_metrics(
                final_test.y, predict_values(result.model, final_test.X) * std + mean
            ),
            "baseline": regression_metrics(
                final_test.y,
                predict_values(result.baseline_model, final_test.X) * std + mean,
            ),
        }
    if not set(final_test.y).issubset(result.classes):
        raise ValueError("Final test feature/label schema mismatch")
    return {
        "ppi": classification_metrics(
            final_test.y,
            predict_model(result.model, final_test.X, len(result.classes)),
            result.classes,
        ),
        "baseline": classification_metrics(
            final_test.y,
            predict_model(result.baseline_model, final_test.X, len(result.classes)),
            result.classes,
        ),
    }
