"""`summary.md`: the run's result written the way a person reads it.

`ppi_summary.json` is the machine record and stays the source of truth; this
renders the same numbers plus the per-class breakdown, so a run can be read
without knowing which npz key holds what. Written next to the JSON by
`flow.run_experiment`, and regenerable for older runs with
`scripts/show_run.py --write-md`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

CLASSIFICATION_METRICS = (
    "accuracy",
    "balanced_accuracy",
    "macro_f1",
    "macro_auroc",
    "macro_auprc",
)
REGRESSION_METRICS = ("r2", "pearson", "rmse", "mae", "mse")
#: The ones that move first when the gain is in rare classes.
HEADLINE = ("accuracy", "balanced_accuracy", "macro_f1")


def metrics_for(summary: dict) -> tuple[str, ...]:
    task = summary.get("task") or {}
    return (
        REGRESSION_METRICS
        if str(task.get("task_type")) == "regression"
        else CLASSIFICATION_METRICS
    )


def _number(value: Any, digits: int = 4) -> str:
    return "n/a" if value is None else f"{float(value):.{digits}f}"


def _delta(baseline: Any, augmented: Any) -> str:
    if baseline is None or augmented is None:
        return "n/a"
    return f"{float(augmented) - float(baseline):+.4f}"


def metric_table(summary: dict, split: str) -> list[str]:
    data = summary.get(split) or {}
    baseline = data.get("baseline") or {}
    augmented = data.get("ppi") or {}
    lines = ["| metric | baseline | augmented | delta |", "|---|---|---|---|"]
    for name in metrics_for(summary):
        lines.append(
            f"| {name} | {_number(baseline.get(name))} | {_number(augmented.get(name))} "
            f"| {_delta(baseline.get(name), augmented.get(name))} |"
        )
    return lines


def per_class_rows(summary: dict, directory: str | Path) -> list[dict]:
    """Per-class recall on validation, ordered by class frequency.

    Only the validation split keeps per-row predictions; the final test is
    aggregated by the trainer, and this says so rather than pretending.
    """
    if str((summary.get("task") or {}).get("task_type")) == "regression":
        # There are no classes: the prediction is a value, and "recall" is not
        # a word that applies to it.
        return []
    path = Path(directory) / "validation_predictions.npz"
    if not path.exists():
        return []
    try:
        import numpy as np
    except ImportError:  # pragma: no cover - numpy is a core dependency
        return []
    classes = (summary.get("task") or {}).get("classes") or []
    with np.load(path, allow_pickle=False) as data:
        y_true = np.asarray(data["y_true"]).astype(str)
        baseline_prob = data["baseline_probability"]
        augmented_prob = data["ppi_probability"]
    # Labels may be strings (cell types) or numbers (a quality score). The npz
    # stores them with their original dtype, so both sides are compared as text
    # -- indexing an int-keyed map with the string "5" silently matched nothing
    # and produced an empty table, which then divided by zero.
    index = {str(name): i for i, name in enumerate(classes)}
    known = np.array([str(value) in index for value in y_true])
    y = np.array([index[str(value)] for value in y_true[known]])
    base_pred = baseline_prob[known].argmax(1)
    aug_pred = augmented_prob[known].argmax(1)
    rows: list[dict] = []
    for i, name in enumerate(classes):
        mask = y == i
        if not mask.any():
            continue
        base = float((base_pred[mask] == i).mean())
        aug = float((aug_pred[mask] == i).mean())
        rows.append(
            {"class": name, "n": int(mask.sum()), "base": base, "aug": aug, "delta": aug - base}
        )
    rows.sort(key=lambda row: row["n"])
    return rows


def _per_class_section(summary: dict, directory: Path, top: int) -> list[str]:
    if str((summary.get("task") or {}).get("task_type")) == "regression":
        return [
            "_Not applicable: the target is a measured value, so there is no "
            "per-class breakdown. The error metrics above are the breakdown._"
        ]
    rows = per_class_rows(summary, directory)
    if not rows:
        return ["_No per-class breakdown: `validation_predictions.npz` is absent._"]
    lines = [
        f"Rarest {min(top, len(rows))} of {len(rows)} classes, by how often they occur "
        "in the validation split:",
        "",
        "| class | n | baseline recall | augmented recall | delta |",
        "|---|---|---|---|---|",
    ]
    for row in rows[:top]:
        lines.append(
            f"| {row['class']} | {row['n']} | {row['base']:.3f} | {row['aug']:.3f} "
            f"| {row['delta']:+.3f} |"
        )
    worst = sorted(rows, key=lambda row: row["delta"])[:3]
    lines += ["", "Largest regressions anywhere:", ""]
    lines += ["| class | n | baseline recall | augmented recall | delta |", "|---|---|---|---|---|"]
    for row in worst:
        lines.append(
            f"| {row['class']} | {row['n']} | {row['base']:.3f} | {row['aug']:.3f} "
            f"| {row['delta']:+.3f} |"
        )
    rarest, common = rows[:10], rows[-10:]
    mean_rare = sum(r["delta"] for r in rarest) / len(rarest)
    mean_common = sum(r["delta"] for r in common) / len(common)
    lines += [
        "",
        f"Mean delta — 10 rarest: **{mean_rare:+.4f}** | 10 most common: **{mean_common:+.4f}**. "
        "The hypothesis is that the first is larger than the second; if both are near zero, "
        "the augmentation changed nothing measurable on this split.",
    ]
    return lines


def render_markdown(summary: dict, directory: str | Path, *, top: int = 10) -> str:
    directory = Path(directory)
    task = summary.get("task") or {}
    loss = summary.get("loss") or {}
    labeled = summary.get("n_labeled") or {}
    supplementary = summary.get("supplementary") or {}
    inputs = summary.get("inputs") or {}

    lines = [
        "# Training run summary",
        "",
        f"- **mode:** `{summary.get('mode')}` "
        f"(augmented arm: `{loss.get('mode', 'signed?')}` loss)",
        f"- **target column:** `{task.get('target_column')}` "
        "({}, {}, {} features)".format(
            task.get("task_type"),
            (
                "a measured value"
                if str(task.get("task_type")) == "regression"
                else f"{len(task.get('classes') or [])} classes"
            ),
            task.get("n_features"),
        ),
        f"- **evaluation metric:** `{task.get('evaluation_metric')}`",
    ]
    if task.get("description"):
        lines.append(f"- **task:** {task['description']}")
    lines += ["", "## Data", ""]
    labeled_paths = inputs.get("labeled_paths") or []
    lines.append(
        f"- labeled: {labeled.get('total')} rows "
        f"(train {labeled.get('train')} / validation {labeled.get('validation')} / "
        f"test {labeled.get('test')})"
    )
    for path in labeled_paths:
        lines.append(f"  - `{path}`")
    preprocessing = inputs.get("preprocessing") or {}
    if preprocessing.get("kind") == "single_cell":
        settings = preprocessing.get("per_source") or {}
        lines.append(
            f"- **single-cell preprocessing, per source:** "
            f"{settings.get('n_top_genes')} HVGs -> normalize to "
            f"{settings.get('target_sum'):g} -> log1p -> per-gene z-score -> "
            f"negatives clipped to 0; shared panel: "
            f"{len(preprocessing.get('panel') or []):,} gene(s)"
        )
        for source in preprocessing.get("sources") or []:
            detail = (
                f"  - `{source.get('name')}`: {source.get('cells'):,} cells, "
                f"{source.get('genes_selected'):,} of "
                f"{source.get('genes_in_file'):,} gene(s) selected"
            )
            if source.get("median_counts_per_cell") is not None:
                detail += (
                    f"; median {source['median_counts_per_cell']:,.0f} counts/cell, "
                    f"{source.get('sparsity_raw', 0):.1%} zeros before, "
                    f"{source.get('sparsity_processed', 0):.1%} after"
                )
            lines.append(detail)
        if preprocessing.get("report"):
            lines.append(
                f"- **what was done to each gene:** "
                f"[preprocessing.md]({Path(preprocessing['report']).name})"
            )
    available = supplementary.get("available_rows") or {}
    if available:
        lines.append(
            f"- supplementary: {summary.get('external_used')} rows used of "
            f"{sum(available.values())} available, from "
            f"{len(available)} source(s); weight mass {supplementary.get('weight_mass')}"
        )
        for source, rows in available.items():
            detail = (supplementary.get("sources") or {}).get(source) or {}
            note = ""
            if detail.get("had_target_column"):
                note = (
                    " — **it carries the label column, and those labels were "
                    "ignored by policy**: its rows entered as unlabeled evidence"
                )
            lines.append(f"  - `{source}`: {rows} rows available{note}")
    else:
        lines.append("- supplementary: none — this is plain supervised training")
    for path in inputs.get("supplementary_paths") or []:
        lines.append(f"  - `{path}`")
    if inputs.get("test_path"):
        lines.append(f"- held-out test table: `{inputs['test_path']}`")
    encoding = inputs.get("encoding") or {}
    dropped_wide = encoding.get("dropped_wide") or {}
    if dropped_wide:
        # The role rules profile a sample; the encoder sees every row, and a
        # column that looked like a category in the sample can turn out to be
        # free text. Named here so it does not disappear quietly.
        named = ", ".join(f"{name} ({levels} levels)" for name, levels in dropped_wide.items())
        lines.append(
            f"- encoding: {len(dropped_wide)} column(s) dropped for having more "
            f"levels than the cap allows: {named}"
        )

    figures = preprocessing.get("figures") or []
    if figures:
        lines += ["", "## Figures", ""]
        for path in figures:
            name = Path(path).name
            lines.append(f"- [`{name}`](figures/{name})")

    lines += ["", "## Loss", ""]
    if loss:
        mode = loss.get("mode")
        lines += [
            f"- mode: `{loss.get('mode')}`, lambda: `{loss.get('lambda')}`, "
            f"ramp: `{loss.get('ramp_epochs')}` epochs, schedule: `{loss.get('schedule')}`",
            f"- pseudo labels: `{loss.get('pseudo_mode')}` / `{loss.get('pseudo_targets')}` targets",
            "",
            (
                "- `signed` is the PPI correction: it subtracts the pseudo-label loss "
                "on the labeled rows, which is what makes the estimator unbiased."
                if mode == "signed"
                else (
                    "- `gradient_gated` keeps the labeled rows as the anchor and admits the "
                    "synthetic gradient only where it agrees with them: "
                    "`g = g_G + max(0, cos(g_G, g_S)) · min(1, κ‖g_G‖/‖g_S‖) · g_S`. "
                    "Conflicting synthetic data contributes nothing, so the worst case is "
                    "gold-only training."
                    if mode == "gradient_gated"
                    else (
                        "- `gold_plus_synthetic` is the naive baseline `L_G + λ·L_S`: every "
                        "synthetic step is applied in full, whether or not it agrees with "
                        "the labeled rows."
                        if mode == "gold_plus_synthetic"
                        else "- `plain` is pseudo-label distillation: no negative term, so it is "
                        "**biased** by construction — the evidence for it is the held-out "
                        "comparison below, not estimator theory."
                    )
                )
            ),
        ]
        if mode == "gradient_gated":
            lines += [
                f"- gate: κ=`{loss.get('gate_kappa')}`, scope=`{loss.get('gate_scope')}`, "
                f"γ=`{loss.get('gate_gamma')}`, λ=`{loss.get('gate_lambda')}`",
                "",
                *gate_table(summary),
            ]
    else:
        lines.append("- not recorded: this run predates the loss field (it used `signed`).")

    lines += ["", "## Validation (used for model selection)", ""]
    lines += metric_table(summary, "validation_metrics")
    lines += ["", "## Final test (untouched by selection)", ""]
    lines += metric_table(summary, "final_test_metrics")
    lines += ["", "## Training process", ""]
    lines += _training_section(summary)
    heading = (
        "## Prediction error (validation)"
        if str(task.get("task_type")) == "regression"
        else f"## Per-class recall (validation, top {top} rarest)"
    )
    lines += ["", heading, ""]
    lines += _per_class_section(summary, directory, top)

    lines += ["", "## Reproducibility", ""]
    lines += [
        f"- selected epochs: {summary.get('selected_epochs')}",
        f"- stage epochs: {summary.get('stage_epochs')}",
        f"- training seconds: {summary.get('training_seconds')}",
        f"- config sha256: `{summary.get('config_sha256')}`",
        f"- artifacts: `{summary.get('artifact_dir')}`",
        "",
        "Machine-readable record: `ppi_summary.json` (same directory). Per-class "
        "numbers come from `validation_predictions.npz`; the final test is aggregated.",
    ]
    return "\n".join(lines) + "\n"


def gate_table(summary: dict) -> list[str]:
    """The gate's own diagnostics, per epoch.

    The mechanism is only interesting if these numbers move: the cosine should
    fall as the synthetic labels get worse, the weight should follow it down,
    and `active` -- the share of steps that used synthetic data at all -- should
    reach zero when the synthetic gradient stops agreeing with the gold one.
    """
    gate = (summary.get("training") or {}).get("gate") or {}
    epochs = gate.get("epochs") or []
    if not epochs:
        return ["", "_No gate diagnostics recorded for this run._"]
    lines = [
        "| epoch | cos(g_G, g_S) | ‖g_G‖ | ‖g_S‖ | λ_t | active | gold loss | synthetic loss | val balanced acc |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for row in epochs:
        lines.append(
            "| {epoch} | {cos:.3f} ± {cos_std:.2f} | {gold:.4g} | {syn:.4g} | "
            "{weight:.3f} ± {weight_std:.2f} | {active:.0%} | {gold_loss:.4f} | "
            "{syn_loss:.4f} | {val} |".format(
                epoch=row.get("epoch"),
                cos=float(row.get("gradient_cosine") or 0.0),
                cos_std=float(row.get("gradient_cosine_std") or 0.0),
                gold=float(row.get("gold_grad_norm") or 0.0),
                syn=float(row.get("synthetic_grad_norm") or 0.0),
                weight=float(row.get("synthetic_weight") or 0.0),
                weight_std=float(row.get("synthetic_weight_std") or 0.0),
                active=float(row.get("synthetic_active_fraction") or 0.0),
                gold_loss=float(row.get("train_gold_loss") or 0.0),
                syn_loss=float(row.get("train_synthetic_loss") or 0.0),
                val=(
                    f"{float(row['validation_balanced_accuracy']):.4f}"
                    if row.get("validation_balanced_accuracy") is not None
                    else "n/a"
                ),
            )
        )
    lines.append("")
    lines.append(
        f"- over the run: cosine {gate.get('gradient_cosine_mean', 0):.3f} "
        f"± {gate.get('gradient_cosine_std', 0):.3f}, weight "
        f"{gate.get('synthetic_weight_mean', 0):.3f} "
        f"± {gate.get('synthetic_weight_std', 0):.3f}, active in "
        f"{gate.get('synthetic_active_fraction', 0):.0%} of epochs"
    )
    return lines


def _training_section(summary: dict) -> list[str]:
    """How the two arms were fitted, and why they stopped when they did."""
    training = summary.get("training") or {}
    arms = training.get("arms") or {}
    if not arms:
        lines = [
            f"- selected epochs: {summary.get('selected_epochs')}",
            "- per-epoch detail: `training_log.jsonl` (not recorded for this run)",
        ]
        return lines
    lines = [
        f"- budget: up to {training.get('max_epochs')} epochs, patience "
        f"{training.get('patience')}",
        "",
        "| arm | epochs run | best epoch | stopped by | best validation | loss first → last |",
        "|---|---|---|---|---|---|",
    ]
    for arm in ("baseline", "ppi"):
        entry = arms.get(arm) or {}
        first, last = entry.get("first_epoch_loss"), entry.get("last_epoch_loss")
        best = entry.get("best_validation")
        loss = (
            f"{float(first):.4f} → {float(last):.4f}"
            if first is not None and last is not None
            else "n/a"
        )
        best_text = (
            f"{float(best):.4f} {entry.get('selection_metric', '')}"
            if best is not None
            else "n/a"
        )
        lines.append(
            f"| {arm} | {entry.get('epochs_run')} | {entry.get('best_epoch')} | "
            f"{entry.get('stopped')} | {best_text} | {loss} |"
        )
    lines.append("")
    lines.append(
        "Per-epoch losses and validation metrics for both arms are in "
        "`training_log.jsonl` in the same directory."
    )
    return lines


def write_markdown(summary: dict, directory: str | Path, *, top: int = 10) -> Path:
    path = Path(directory) / "summary.md"
    path.write_text(render_markdown(summary, directory, top=top), encoding="utf-8")
    return path
