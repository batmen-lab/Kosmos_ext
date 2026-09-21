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
