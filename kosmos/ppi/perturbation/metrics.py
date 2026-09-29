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
