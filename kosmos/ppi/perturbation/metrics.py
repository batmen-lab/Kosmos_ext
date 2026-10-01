"""Perturbation-level metrics: the ones a perturbation paper reports.

A cell-level number cannot answer "did the model predict this perturbation":
cells of one perturbation are near-duplicates. So each perturbation is
summarised by its **mean change** over held-out cells, and the comparison is
between two vectors of length G:

  * `mse`               -- all genes
  * `mse_deg`           -- the top-K genes by |observed Δ| (GEARS' `mse_de`)
  * `pearson`/`spearman` -- correlation of the Δ vectors across genes
  * `direction_accuracy` -- share of the top-K genes whose sign was predicted
  * `top_k_overlap`     -- |predicted top-K ∩ observed top-K| / K

Averaged over the held-out perturbations, with per-perturbation detail kept.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class PerturbationMetrics:
    per_perturbation: dict[str, dict[str, float]]
    summary: dict[str, float]
    n_perturbations: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_perturbations": self.n_perturbations,
            "summary": self.summary,
            "per_perturbation": self.per_perturbation,
        }


def _top_k(values: np.ndarray, k: int) -> np.ndarray:
    magnitude = np.abs(values)
    k = max(1, min(int(k), magnitude.size))
    return np.sort(np.argpartition(-magnitude, k - 1)[:k])


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    from scipy.stats import spearmanr

    value = spearmanr(a, b).statistic
    return float(value) if np.isfinite(value) else float("nan")


def _pearson(a: np.ndarray, b: np.ndarray) -> float:
    if a.size < 2:
        return float("nan")
    a_centred, b_centred = a - a.mean(), b - b.mean()
    denominator = np.sqrt((a_centred**2).sum() * (b_centred**2).sum())
    if denominator == 0:
        return float("nan")
    return float((a_centred * b_centred).sum() / denominator)


def perturbation_signal(
    predictions: dict[str, np.ndarray], observations: dict[str, np.ndarray]
) -> dict[str, float]:
    """How much of the *observed* between-perturbation variation an arm reproduces.

    Every arm predicts one profile per perturbation. An arm that has not learned
    the identity predicts (nearly) the same profile for all of them -- its
    per-gene output bias -- and the spread of those profiles across perturbations
    collapses to the spread of that bias. Dividing by the observed spread says
    what share of the real difference between perturbations the model carries:

      * 1.0 -- the profiles differ as much as the data's do;
      * ~0  -- the prediction is a constant, so its `top_k_overlap` and `pearson`
              describe that constant rather than a ranking of the perturbation's
              genes (on the ECCITE screen the GEARS arms sat at 14-28% and the
              MLP arms at 6-7%, which is what "top-K = 0" was made of).
    """
    labels = sorted(set(predictions) & set(observations))
    if len(labels) < 2:
        return {
            "across_perturbation_std": 0.0,
            "observed_across_perturbation_std": 0.0,
            "signal": 0.0,
        }
    predicted = np.stack([np.asarray(predictions[label], dtype=np.float64) for label in labels])
    observed = np.stack([np.asarray(observations[label], dtype=np.float64) for label in labels])
    across = float(np.mean(predicted.std(axis=0)))
    observed_across = float(np.mean(observed.std(axis=0)))
    return {
        "across_perturbation_std": across,
        "observed_across_perturbation_std": observed_across,
        "signal": across / observed_across if observed_across > 0 else 0.0,
    }


def baseline_metrics(
    observations: dict[str, np.ndarray], *, top_k: int = 20
) -> dict[str, dict[str, float]]:
    """What a reader needs to judge the table: two references on the same metric.

    `predict 0` is "no change anywhere" and `mean response (leave-one-out)` is
    "the average response of the *other* held-out perturbations" -- the honest
    version of "predict the average", since the test perturbations are not used
    to predict themselves. Without them a collapsed arm that scores well on the
    DEG-restricted quartic looks like a result.
    """
    import warnings

    labels = sorted(observations)
    zero = {label: np.zeros_like(np.asarray(observations[label], dtype=np.float64)) for label in labels}
    leave_one_out = {}
    for label in labels:
        others = [np.asarray(observations[other], dtype=np.float64) for other in labels if other != label]
        leave_one_out[label] = (
            np.mean(np.stack(others), axis=0) if others else np.zeros_like(np.asarray(observations[label], dtype=np.float64))
        )
    with warnings.catch_warnings():
        # A constant prediction has no correlation with anything, by definition.
        warnings.simplefilter("ignore")
        return {
            "predict 0": perturbation_metrics(zero, observations, top_k=top_k).summary,
            "mean response (leave-one-out)": perturbation_metrics(
                leave_one_out, observations, top_k=top_k
            ).summary,
        }


def perturbation_metrics(
    predictions: dict[str, np.ndarray],
    observations: dict[str, np.ndarray],
    *,
    top_k: int = 20,
) -> PerturbationMetrics:
    """Per-perturbation metrics for the held-out perturbations."""
    shared = sorted(set(predictions) & set(observations))
    if not shared:
        raise ValueError("no perturbation is in both the predictions and the observations")
    detail: dict[str, dict[str, float]] = {}
    for label in shared:
        predicted = np.asarray(predictions[label], dtype=np.float64).reshape(-1)
        observed = np.asarray(observations[label], dtype=np.float64).reshape(-1)
        if predicted.shape != observed.shape:
            raise ValueError(f"{label}: prediction and observation shapes differ")
        deg = _top_k(observed, top_k)
        predicted_top = _top_k(predicted, top_k)
        detail[label] = {
            "mse": float(np.mean((predicted - observed) ** 2)),
            "mse_deg": float(np.mean((predicted[deg] - observed[deg]) ** 2)),
            "pearson": _pearson(predicted, observed),
            "pearson_deg": _pearson(predicted[deg], observed[deg]),
            "spearman": _spearman(predicted, observed),
            "direction_accuracy": float(
                np.mean(np.sign(predicted[deg]) == np.sign(observed[deg]))
            ),
            "top_k_overlap": float(len(set(deg.tolist()) & set(predicted_top.tolist())) / len(deg)),
            "n_deg": int(len(deg)),
        }
    keys = sorted(next(iter(detail.values())).keys())
    summary = {
        key: float(np.nanmean([detail[label][key] for label in shared]))
        for key in keys
    }
    return PerturbationMetrics(per_perturbation=detail, summary=summary, n_perturbations=len(shared))
