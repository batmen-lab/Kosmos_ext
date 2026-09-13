"""CPU-deterministic, classification-first PPI runner with a matched baseline."""

import copy
import json
import random
import tempfile
from hashlib import sha256
from pathlib import Path

import numpy as np
import sklearn
import torch
from torch import nn

from .datasets import prepare_external
from .losses import PPILoss
from .metrics import classification_metrics
from .pseudo_labeler import (
    PreparedPseudoLabeler,
    PretrainedPseudoLabeler,
    fingerprint,
    prepare_pseudo_labeler,
)
from .schemas import PPITrainingConfig, PPITrainingResult
from .split import validate_roles


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


def encode(labels, classes):
    mapping = {value: i for i, value in enumerate(classes)}
    return np.array([mapping[value] for value in labels], dtype=np.int64)


def _fit(initial, gold, validation, external, classes, config, *, ppi, loss_factory):
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
    X = torch.as_tensor(gold.X)
    y = torch.as_tensor(encode(gold.y_true, classes))
    yp = (
        torch.as_tensor(gold.y_prob, dtype=torch.float32)
        if config.pseudo_targets == "soft"
        else torch.as_tensor(encode(gold.y_pred, classes))
    )
    mass = sum(d.weights.sum() for d in external)
    use_external = ppi and bool(external) and mass > 0 and config.ppi_lambda > 0
    if use_external:
        Xe = torch.as_tensor(np.concatenate([d.X for d in external]))
        ye = (
            torch.as_tensor(np.concatenate([d.y_prob for d in external]), dtype=torch.float32)
            if config.pseudo_targets == "soft"
            else torch.as_tensor(encode(np.concatenate([d.y_pred for d in external]), classes))
        )
        weights = torch.as_tensor(
            np.concatenate([d.weights for d in external]), dtype=torch.float32
        )
    gold_rng = np.random.default_rng(config.seed)
    external_rng = np.random.default_rng(config.seed + 1)
    history, best_score, best_epoch, stale = [], -np.inf, 0, 0
    best_state = copy.deepcopy(model.state_dict())
    for epoch in range(1, config.max_epochs + 1):
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
        order = gold_rng.permutation(len(X))
        for start in range(0, len(X), config.gold_batch_size):
            idx = order[start : start + config.gold_batch_size]
            gold_logits = model(X[idx])
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
        metrics = classification_metrics(
            validation.y, predict_model(model, validation.X, len(classes)), classes
        )
        metric = config.evaluation_metric
        # Explicit fallback for folds on which requested macro ranking metric is undefined.
        if metrics[metric] is None:
            metric = "balanced_accuracy"
        score = metrics[metric]
        history.append(
            {
                "epoch": epoch,
                "phase": (
                    "correction"
                    if correction_stage
                    else ("joint" if use_external and config.schedule == "joint" else "true")
                ),
                **totals,
                "validation": metrics,
                "selection_metric": metric,
            }
        )
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
    loss_factory=PPILoss,
):
    """No final_test argument. Fits neither pseudo-labeler nor preprocessing on validation.

    CPU is intentional in this first implementation. A model factory receives
    (feature_count, class_count) and returns a fresh torch module producing logits.
    """
    config = PPITrainingConfig.model_validate((config or PPITrainingConfig()).model_dump())
    external_evidence = list(external_evidence)
    validate_roles(gold_train, gold_validation, external_evidence)
    classes = np.unique(gold_train.y)
    if not set(gold_validation.y).issubset(set(classes)):
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
            if config.pseudo_targets == "soft" and any(d.y_prob is None for d in external):
                raise ValueError("Soft external pseudo targets require probabilities")
            initial = model_factory(gold_train.X.shape[1], len(classes)).cpu()
            baseline, baseline_history, baseline_epoch = _fit(
                initial,
                labeler.pseudo_gold,
                gold_validation,
                [],
                classes,
                config,
                ppi=False,
                loss_factory=loss_factory,
            )
            if (
                not external
                or sum(d.weights.sum() for d in external) == 0
                or config.ppi_lambda == 0
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
                )
    finally:
        np.random.set_state(numpy_state)
        random.setstate(python_state)
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
    for c in classes:
        pseudo_counts[str(c)] = int(np.sum(all_pred == c))
    provenance = {
        "config": config.model_dump(),
        "config_sha256": sha256(config_json.encode()).hexdigest(),
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
        "gold_pseudo_agreement": float(np.mean(labeler.pseudo_gold.y_pred == gold_train.y)),
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
    if final_test.X.shape[1] != result.pseudo_gold.X.shape[1] or not set(final_test.y).issubset(
        result.classes
    ):
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
