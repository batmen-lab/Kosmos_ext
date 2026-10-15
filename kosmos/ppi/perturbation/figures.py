"""What a perturbation run looks like: four panels and a summary.

A table of numbers cannot answer the questions this design is about, so the run
draws them:

  * **arms compared** on the perturbation-level metrics (does the auxiliary
    graph help, and does gating change that?);
  * **Δ predicted against Δ observed**, per perturbation, for each arm -- the
    diagonal is a perfect prediction, and the spread is the honest part;
  * **the gate**, per epoch: the cosine between the gold and graph-increment
    gradients and the weight it produced;
  * **training loss** per epoch per arm.

`summary.md` beside them lists the numbers, the contract and the graph facts.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

METRICS = ("mse_deg", "pearson", "direction_accuracy", "top_k_overlap")

#: The word a reader sees for an arm: the internal keys are implementation
#: names, not the comparison the report is about.
_ARM_LABELS = {
    "gears_base": "base",
    "gears_augmented_ungated": "signed ppi",
    "mlp_base": "mlp base",
    "mlp_augmented_ungated": "mlp signed ppi",
    "gears_gated": "gated",
    "mlp_gated": "mlp gated",
    "gears_finetune_ppi": "finetune ppi",
    "perturbnet_base": "perturbnet base",
    "perturbnet_supp": "perturbnet supp",
}


def _number(value: Any, digits: int = 3) -> str:
    """A metric cell; `-` for a missing or NaN value (e.g. a constant prediction)."""
    try:
        value = float(value)
    except (TypeError, ValueError):
        return "-"
    return f"{value:.{digits}f}" if np.isfinite(value) else "-"


def _arm_label(arm: str) -> str:
    return _ARM_LABELS.get(str(arm), str(arm).replace("_", " "))


def _load(path: str | Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        return np.asarray(data["labels"]), np.asarray(data["predictions"]), np.asarray(data["observations"])


def write_perturbation_figures(results: dict[str, Any], out_dir: str | Path) -> list[Path]:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:  # noqa: BLE001 - figures are a convenience
        return []
    out = Path(out_dir)
    figures = out / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    arms = list((results.get("arms") or {}).keys())
    if not arms:
        return []

    figure, axes = plt.subplots(2, 2, figsize=(13, 9))

    # 1. the comparison the design asks for
    width = 0.8 / max(1, len(arms))
    for position, arm in enumerate(arms):
        summary = results["arms"][arm]["test"]["summary"]
        axes[0, 0].bar(
            [index + position * width for index in range(len(METRICS))],
            [summary[metric] for metric in METRICS],
            width=width,
            label=arm,
        )
    axes[0, 0].set_xticks([index + 0.4 - width / 2 for index in range(len(METRICS))], METRICS, fontsize=8)
    axes[0, 0].set_title("perturbation-level metrics (test)")
    axes[0, 0].legend(fontsize=7)
    axes[0, 0].grid(axis="y", alpha=0.3)

    # 2. Δ predicted vs observed, per perturbation
    for arm in arms:
        path = results["arms"][arm].get("predictions")
        if not path or not Path(path).exists():
            continue
        _, predictions, observations = _load(path)
        axes[0, 1].scatter(
            observations.reshape(-1), predictions.reshape(-1), s=4, alpha=0.35, label=arm
        )
    axes[0, 1].set_xlabel("observed Δ (held-out perturbations)")
    axes[0, 1].set_ylabel("predicted Δ")
    axes[0, 1].set_title("Δ predicted against Δ observed")
    axes[0, 1].legend(fontsize=7)
    axes[0, 1].grid(alpha=0.3)

    # 3. held-out error per epoch, per arm (the early-stopping signal)
    for arm in arms:
        history = results["arms"][arm].get("history") or []
        axes[1, 0].plot(
            [entry["epoch"] for entry in history],
            [entry.get("validation_mse_deg", float("nan")) for entry in history],
            marker="o",
            label=arm,
        )
    axes[1, 0].set_xlabel("epoch")
    axes[1, 0].set_ylabel("validation mse_deg")
    axes[1, 0].set_title("held-out error (the early-stopping signal)")
    axes[1, 0].legend(fontsize=7)
    axes[1, 0].grid(alpha=0.3)

    # 4. training loss
    for arm in arms:
        history = results["arms"][arm].get("history") or []
        axes[1, 1].plot(
            [entry["epoch"] for entry in history],
            [entry["train_loss"] for entry in history],
            marker="o",
            label=arm,
        )
    axes[1, 1].set_xlabel("epoch")
    axes[1, 1].set_ylabel("train loss (per arm's own objective)")
    axes[1, 1].set_title("training loss")
    axes[1, 1].legend(fontsize=7)
    axes[1, 1].grid(alpha=0.3)

    figure.tight_layout()
    target = figures / "perturbation_overview.png"
    figure.savefig(target, dpi=150)
    plt.close(figure)
    return [target]


def write_summary(results: dict[str, Any], out_dir: str | Path) -> Path:
    """`summary.md`: the arms, the contract, the graphs, the warning."""
    out = Path(out_dir)
    task = results.get("task") or {}
    lines = [
        "# Perturbation response run",
        "",
        f"- panel: {task.get('n_genes')} gene(s), condition column `{task.get('condition_column')}`",
        f"- splits (by perturbation): {results.get('split_sizes')}",
        f"- arms: {', '.join(_arm_label(arm) for arm in (results.get('arms') or []))}",
    ]
    if results.get("partial"):
        lines += [
            "",
            "> **Partial run.** The arms above have finished; the run was still "
            "training (or was stopped) when this file was written. Read it again "
            "after the run ends for the complete table.",
        ]
    modality = results.get("modality") or {}
    verdict = modality.get("verdict") or {}
    if verdict:
        requested = modality.get("requested") or {}
        lines += [
            "",
            "## Perturbation type",
            "",
            f"- requested by the question: `{verdict.get('requested')}`"
            + (f" (decided by {requested.get('decided_by')})" if requested else ""),
            f"- provided by the data: `{verdict.get('provided')}`",
            f"- verdict: **{verdict.get('status')}** - {verdict.get('message')}",
        ]
    lines += [
        "",
        "## Test metrics (held-out perturbations)",
        "",
        "| arm | mse | mse_deg | pearson | specific pearson | spearman | direction | top-K overlap | epochs |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for arm, payload in (results.get("arms") or {}).items():
        summary = payload["test"]["summary"]
        lines.append(
            f"| `{_arm_label(arm)}` | {summary['mse']:.4f} | {summary['mse_deg']:.4f} | "
            f"{summary['pearson']:.3f} | {_number(summary.get('pearson_specific'))} | "
            f"{summary['spearman']:.3f} | "
            f"{summary['direction_accuracy']:.3f} | {summary['top_k_overlap']:.3f} | "
            f"{payload.get('epochs_run')} |"
        )
    baselines = results.get("baselines") or {}
    if baselines:
        lines += [
            "",
            "## Reference baselines (same metrics, same held-out perturbations)",
            "",
            "| baseline | mse | mse_deg | pearson | specific pearson | direction | top-K |",
            "|---|---|---|---|---|---|---|",
        ]
        for name, summary in baselines.items():
            lines.append(
                f"| {name} | {summary['mse']:.4f} | {summary['mse_deg']:.4f} | "
                f"{_number(summary.get('pearson'))} | {_number(summary.get('pearson_specific'))} | "
                f"{summary['direction_accuracy']:.3f} | {summary['top_k_overlap']:.3f} |"
            )
        lines += [
            "",
            "`predict 0` is the no-change reference; `mean response (leave-one-out)` "
            "predicts each held-out perturbation with the average response of the "
            "others. An arm that does not beat both is not predicting the "
            "perturbation. `specific pearson` correlates each perturbation's "
            "Δ minus the mean Δ of the held-out perturbations, so it only credits "
            "what tells perturbations apart (`-` = the prediction is the same for "
            "all of them). The best epoch is chosen by validation `mse_deg` "
            "(GEARS' rule) unless `KOSMOS_PERTURBATION_SELECTION` says otherwise.",
        ]

    signals = [
        (arm, payload.get("signal") or {})
        for arm, payload in (results.get("arms") or {}).items()
        if payload.get("signal")
    ]
    if signals:
        lines += [
            "",
            "## Per-perturbation signal",
            "",
            "How much of the *observed* difference between perturbations the arm "
            "reproduces (spread of its per-perturbation profiles / spread of the "
            "observed ones). 0 means a per-gene constant.",
            "",
            "| arm | across-perturbation std | of the data's |",
            "|---|---|---|",
        ]
        for arm, signal in signals:
            lines.append(
                f"| `{_arm_label(arm)}` | {signal['across_perturbation_std']:.4f} | "
                f"{signal['signal']:.1%} |"
            )
        faded = [_arm_label(arm) for arm, signal in signals if signal.get("signal", 1.0) < 0.1]
        if faded:
            lines += [
                "",
                "> **No perturbation signal**: "
                + ", ".join(f"`{arm}`" for arm in faded)
                + " reproduce less than a tenth of the observed between-perturbation "
                "variation. Their prediction is a per-gene constant, so their "
                "`top_k_overlap` (and `pearson`) describe that constant rather than "
                "a ranking of the perturbation's genes -- read those rows as "
                "'did not learn the perturbation', not as a ranking result.",
            ]

    gated = [(arm, payload["gating"]) for arm, payload in (results.get("arms") or {}).items() if payload.get("gating")]
    if gated:
        lines += [
            "",
            "## Gradient gate (gated arms, mean over epochs)",
            "",
            "Each step uses `g_G + max(0, cos)·min(1, κ‖g_G‖/‖g_S‖)·g_S`; weight 0 = the "
            "supplementary gradient conflicted with the gold one and the step was gold-only.",
            "",
            "| arm | cos(g_G, g_S) | synthetic weight | steps with weight > 0 | ‖g_G‖ | ‖g_S‖ |",
            "|---|---|---|---|---|---|",
        ]
        for arm, gate in gated:
            lines.append(
                f"| `{_arm_label(arm)}` | {gate['gradient_cosine']:+.3f} | {gate['synthetic_weight']:.3f} | "
                f"{gate['synthetic_active_fraction']:.0%} | {gate['gold_grad_norm']:.3g} | "
                f"{gate['synthetic_grad_norm']:.3g} |"
            )

    overlap = results.get("graph_edge_overlap") or {}
    if overlap:
        lines += [
            "",
            "## Graphs",
            "",
            f"- gold co-expression vs supplementary: {overlap.get('shared')} shared edge(s), "
            f"Jaccard {overlap.get('jaccard', 0):.3f} "
            f"(first {overlap.get('edges_first')}, second {overlap.get('edges_second')})",
        ]
    panel_warning = (results.get("sources") or {}).get("panel_warning")
    if panel_warning:
        lines += ["", "## Warnings", "", f"- {panel_warning}"]
    figures = results.get("figures") or []
    if figures:
        lines += ["", "## Figures", ""]
        for path in figures:
            lines.append(f"- [`{Path(path).name}`](figures/{Path(path).name})")
    path = out / "summary.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path
