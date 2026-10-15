"""How reliable a gold screen's observed responses are, before anything is trained.

Split-half reliability: a perturbation's cells are split at random into two
halves, each half's Δ (mean log1p-CPM minus the controls' mean) is computed, and
the two Δs are correlated. It bounds what any model can score against that
observed Δ (Jurkat/Datlinger: 0.10, all arms at the noise floor; Adamson UPR:
0.79).

Two scores per perturbation, as in GEARS' evaluation:
    all     Pearson of the halves over the panel genes (expressed in controls)
    de20    Pearson over the 20 genes with the largest |Δ| in half A, scored on
            half A vs half B (choosing them on both halves would inflate it)

The controls are split as well, so the halves share no cells.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
from scipy import sparse

from .load import Sample


def _log_cpm(counts: sparse.spmatrix, target_sum: float = 1e4) -> sparse.csr_matrix:
    counts = sparse.csr_matrix(counts, dtype=np.float64)
    totals = np.asarray(counts.sum(axis=1)).ravel()
    scaled = sparse.diags(target_sum / np.where(totals > 0, totals, 1.0)) @ counts
    scaled.data = np.log1p(scaled.data)
    return scaled.tocsr()


def _mean(matrix: sparse.csr_matrix, rows: np.ndarray) -> np.ndarray:
    return np.asarray(matrix[rows].mean(axis=0)).ravel()


def _pearson(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 3 or a.std() == 0 or b.std() == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def split_half(
    sample: Sample,
    labels: Sequence[str],
    controls: Sequence[str],
    *,
    min_cells: int = 20,
    n_genes: int = 5000,
    seed: int = 0,
    skip: Sequence[str] = ("no_guide", "ambiguous", "multiple", "unassigned", ""),
) -> dict[str, Any]:
    """Per-perturbation split-half reliability of the observed Δ.

    `labels`: one perturbation label per cell (same order as `sample.counts`);
    cells labelled with a member of `controls` are the controls; perturbations
    with fewer than `min_cells` cells, and the `skip` labels (no guide, several
    guides), are not scored. Panel = the `n_genes` genes
    with the highest mean in the controls.
    """
    rng = np.random.default_rng(seed)
    labels = np.asarray(labels, dtype=object)
    data = _log_cpm(sample.counts)
    control_rows = np.flatnonzero(np.isin(labels, list(controls)))
    if len(control_rows) < 2 * min_cells:
        raise ValueError(f"{len(control_rows)} control cells: too few to split")
    control_mean = _mean(data, control_rows)
    panel = np.argsort(-control_mean)[:n_genes]
    shuffled = rng.permutation(control_rows)
    ctrl_a, ctrl_b = shuffled[: len(shuffled) // 2], shuffled[len(shuffled) // 2 :]
    base_a, base_b = _mean(data, ctrl_a)[panel], _mean(data, ctrl_b)[panel]

    per: dict[str, dict[str, float]] = {}
    for label in sorted(set(labels) - set(controls) - set(skip)):
        rows = np.flatnonzero(labels == label)
        if len(rows) < min_cells:
            continue
        rows = rng.permutation(rows)
        half_a, half_b = rows[: len(rows) // 2], rows[len(rows) // 2 :]
        delta_a = _mean(data, half_a)[panel] - base_a
        delta_b = _mean(data, half_b)[panel] - base_b
        top = np.argsort(-np.abs(delta_a))[:20]
        per[str(label)] = {
            "cells": int(len(rows)),
            "all": _pearson(delta_a, delta_b),
            "de20": _pearson(delta_a[top], delta_b[top]),
            # how large the response is: mean |Δ| on the full cells' top-20 genes
            "effect": float(np.abs((_mean(data, rows) - control_mean)[panel]).max()),
        }
    scores = lambda key: np.array([v[key] for v in per.values() if np.isfinite(v[key])])
    summary = {
        "perturbations": len(per),
        "control_cells": int(len(control_rows)),
        "cells": int(sum(v["cells"] for v in per.values())),
        "median_cells": float(np.median([v["cells"] for v in per.values()])) if per else 0.0,
        "median_all": float(np.median(scores("all"))) if per else float("nan"),
        "median_de20": float(np.median(scores("de20"))) if per else float("nan"),
        "share_de20_over_0.5": float(np.mean(scores("de20") > 0.5)) if per else 0.0,
    }
    return {"summary": summary, "per_perturbation": per}


def render(result: dict[str, Any], top: int = 40) -> str:
    s = result["summary"]
    lines = [
        f"- perturbations scored: {s['perturbations']} (cells {s['cells']:,}, median "
        f"{s['median_cells']:.0f} per perturbation), control cells {s['control_cells']:,}",
        f"- split-half Pearson, median over perturbations: all panel genes **{s['median_all']:.2f}**, "
        f"top-20 DE **{s['median_de20']:.2f}**; share with top-20 DE > 0.5: {s['share_de20_over_0.5']:.0%}",
        "",
        "| perturbation | cells | all genes | top-20 DE | max abs Δ |",
        "|---|---|---|---|---|",
    ]
    ranked = sorted(result["per_perturbation"].items(), key=lambda kv: -np.nan_to_num(kv[1]["de20"]))
    for label, v in ranked[:top]:
        lines.append(f"| {label} | {v['cells']} | {v['all']:.2f} | {v['de20']:.2f} | {v['effect']:.2f} |")
    if len(ranked) > top:
        lines.append(f"| ... {len(ranked) - top} more | | | | |")
    return "\n".join(lines) + "\n"
