"""Metrics report undefined class-specific quantities rather than fabricate scores."""

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
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
