"""Metrics report undefined class-specific quantities rather than fabricate scores."""

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    r2_score,
    roc_auc_score,
)


def classification_metrics(y, probability, classes):
    prediction = classes[probability.argmax(1)]
    result = {
        "accuracy": float(accuracy_score(y, prediction)),
        "balanced_accuracy": float(balanced_accuracy_score(y, prediction)),
        "macro_f1": float(
            f1_score(y, prediction, labels=classes, average="macro", zero_division=0)
        ),
    }
    auc, ap, undefined = [], [], {}
    for i, label in enumerate(classes):
        binary = (y == label).astype(int)
        if len(np.unique(binary)) < 2:
            undefined[str(label)] = "AUROC/AUPRC undefined: validation lacks positives or negatives"
        else:
            auc.append(roc_auc_score(binary, probability[:, i]))
            ap.append(average_precision_score(binary, probability[:, i]))
    # Don't silently change the macro population when a class is missing.
    result["macro_auroc"] = float(np.mean(auc)) if len(auc) == len(classes) else None
    result["macro_auprc"] = float(np.mean(ap)) if len(ap) == len(classes) else None
    result["undefined"] = undefined
    return result


def regression_metrics(y, prediction):
    """How far off a predicted value is, for a continuous target.

    The error metrics are `mse`/`mae`/`rmse` (lower is better), and the
    agreement metrics are `r2`/`pearson` (higher is better). Selection among
    epochs uses `max`, so the two error metrics are also reported negated
    (`neg_mse`, `neg_mae`): one task type, one direction, no special case in the
    training loop.
    """
    y = np.asarray(y, dtype=float).reshape(-1)
    prediction = np.asarray(prediction, dtype=float).reshape(-1)
    if prediction.shape != y.shape:
        raise ValueError(
            f"a regression prediction must have one value per row: got "
            f"{prediction.shape} for {y.shape} labels"
        )
    residual = y - prediction
    mse = float(np.mean(residual**2))
    mae = float(np.mean(np.abs(residual)))
    result = {
        "mse": mse,
        "mae": mae,
        "rmse": float(np.sqrt(mse)),
        "neg_mse": -mse,
        "neg_mae": -mae,
        "r2": None,
        "pearson": None,
        "undefined": {},
    }
    if len(y) < 2 or float(np.ptp(y)) == 0.0:
        result["undefined"]["r2"] = (
            "r2/pearson undefined: the labels have no variance"
        )
        return result
    result["r2"] = float(r2_score(y, prediction))
    with np.errstate(invalid="ignore", divide="ignore"):
        correlation = float(np.corrcoef(y, prediction)[0, 1])
    result["pearson"] = correlation if np.isfinite(correlation) else None
    return result


def recall_by_class(y, prediction, classes) -> dict[str, float]:
    """Per-class recall, skipping classes the split does not contain."""
    y = np.asarray(y)
    prediction = np.asarray(prediction)
    recalls: dict[str, float] = {}
    for label in np.asarray(classes):
        mask = y == label
        if mask.any():
            recalls[str(label)] = float(np.mean(prediction[mask] == label))
    return recalls


def bootstrap_interval(
    y,
    probability,
    classes,
    *,
    metric: str = "balanced_accuracy",
    n_boot: int = 500,
    alpha: float = 0.05,
    seed: int = 42,
) -> tuple[float, float] | None:
    """A percentile interval for a held-out metric, by resampling rows.

    An *inference* answer is an estimate plus an uncertainty, and one number on
    one split is neither: with a few hundred validation rows the difference
    between two runs is routinely inside the resampling noise. The interval is
    over the rows of the split the model was scored on, so it says how much of
    the number is sampling; seed-to-seed variance is a different, larger
    quantity and is not what this reports.

    Returns None when the interval is undefined: too few rows, fewer than two
    classes present, or a resample on which the metric cannot be computed.
    """
    y = np.asarray(y)
    probability = np.asarray(probability, dtype=float)
    classes = np.asarray(classes)
    if len(y) < 2 or len(classes) < 2 or probability.ndim != 2:
        return None
    prediction = classes[probability.argmax(1)]
    rng = np.random.default_rng(seed)
    values: list[float] = []
    for _ in range(max(2, int(n_boot))):
        rows = rng.integers(0, len(y), len(y))
        sample_y, sample_prediction = y[rows], prediction[rows]
        present = np.unique(sample_y)
        if len(present) < 2:
            continue
        if metric == "balanced_accuracy":
            recalls = recall_by_class(sample_y, sample_prediction, present)
            values.append(float(np.mean(list(recalls.values()))))
        else:
            values.append(float(np.mean(sample_prediction == sample_y)))
    if len(values) < 2:
        return None
    low, high = np.quantile(values, [alpha / 2.0, 1.0 - alpha / 2.0])
    return float(low), float(high)
