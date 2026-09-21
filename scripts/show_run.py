"""Read a training run: baseline vs augmented, and whether the gain is in rare classes.

    python scripts/show_run.py artifacts/auto/bmmc_plain_full
    python scripts/show_run.py artifacts/auto/bmmc_plain_full artifacts/auto/bmmc_signed_full

Read-only. It looks at `<run>/ppi_summary.json` and, when present,
`<run>/validation_predictions.npz` (the only place per-class predictions are
kept -- the final test is aggregated by the trainer).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

CLASSIFICATION_METRICS = (
    "accuracy",
    "balanced_accuracy",
    "macro_f1",
    "macro_auroc",
    "macro_auprc",
)
REGRESSION_METRICS = ("r2", "pearson", "rmse", "mae", "mse")


def metrics_for(summary: dict) -> tuple[str, ...]:
    from kosmos.ppi.report import metrics_for as report_metrics_for

    return report_metrics_for(summary)

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def load(argument: str) -> tuple[dict, Path]:
    path = Path(argument)
    if path.is_dir():
        path = path / "ppi_summary.json"
    return json.loads(path.read_text(encoding="utf-8")), path.parent


def headline(summary: dict) -> str:
    task = summary.get("task", {})
    loss = summary.get("loss") or {}
    supplementary = (summary.get("supplementary") or {}).get("available_rows", {})
    loss_text = (
        f"loss={loss.get('mode')} lambda={loss.get('lambda')} ramp={loss.get('ramp_epochs')} "
        f"pseudo={loss.get('pseudo_mode')}/{loss.get('pseudo_targets')}"
        if loss
        else "loss=(not recorded: run predates the loss field)"
    )
    return (
        f"{loss_text}\n"
        f"  target={task.get('target_column')} features={task.get('n_features')} "
        f"classes={len(task.get('classes') or [])}\n"
        f"  labeled={(summary.get('n_labeled') or {}).get('total')} "
        f"supplementary used={summary.get('external_used')}/{sum(supplementary.values())}"
    )


def metric_block(summary: dict, split: str) -> str:
    data = summary.get(split) or {}
    baseline, augmented = data.get("baseline") or {}, data.get("ppi") or {}
    lines = [f"    {'metric':<20}{'baseline':>10}{'augmented':>10}{'delta':>10}"]
    for name in metrics_for(summary):
        base, aug = baseline.get(name), augmented.get(name)
        if base is None or aug is None:
            lines.append(f"    {name:<20}{'-':>10}{'-':>10}{'-':>10}")
        else:
            lines.append(f"    {name:<20}{base:>10.4f}{aug:>10.4f}{aug - base:>+10.4f}")
    return "\n".join(lines)


def rare_class_report(directory: Path, summary: dict, top: int) -> str:
    """Per-class recall on the validation split, ordered by class frequency."""
    from kosmos.ppi.report import per_class_rows

    measured = per_class_rows(summary, directory)
    if not measured:
        if str((summary.get("task") or {}).get("task_type")) == "regression":
            return (
                "  (no per-class breakdown: the target is a measured value, "
                "so the error metrics above are the breakdown)"
            )
        return "  (no per-class breakdown: validation_predictions.npz absent or unreadable)"
    rows = [
        {
            "class": row["class"],
            "n": row["n"],
            "recall_base": row["base"],
            "recall_aug": row["aug"],
            "delta": row["delta"],
        }
        for row in measured
    ]
    lines = [f"  per-class recall on validation (rarest {min(top, len(rows))} of {len(rows)}):"]
    lines.append(
        f"    {'class':<34}{'n':>6}{'base':>8}{'aug':>8}{'delta':>8}"
    )
    for row in rows[:top]:
        lines.append(
            f"    {str(row['class'])[:33]:<34}{row['n']:>6}{row['recall_base']:>8.3f}"
            f"{row['recall_aug']:>8.3f}{row['delta']:>+8.3f}"
        )
    if len(rows) > top:
        worst = sorted(rows, key=lambda row: row["delta"])[:3]
        lines.append("  largest regressions anywhere:")
        for row in worst:
            lines.append(
                f"    {str(row['class'])[:33]:<34}{row['n']:>6}{row['recall_base']:>8.3f}"
                f"{row['recall_aug']:>8.3f}{row['delta']:>+8.3f}"
            )
    rarest = rows[:10]
    common = rows[-10:]
    lines.append(
        f"  mean delta | 10 rarest: {sum(r['delta'] for r in rarest) / len(rarest):+.4f}"
        f" | 10 most common: {sum(r['delta'] for r in common) / len(common):+.4f}"
    )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", nargs="+", help="run directory or ppi_summary.json")
    parser.add_argument("--top", type=int, default=8, help="rarest classes to list")
    parser.add_argument(
        "--write-md",
        action="store_true",
        help="also write summary.md next to each run's ppi_summary.json",
    )
    args = parser.parse_args()
    for argument in args.runs:
        summary, directory = load(argument)
        if args.write_md:
            from kosmos.ppi.report import write_markdown

            written = write_markdown(summary, directory, top=max(args.top, 10))
            print(f"# wrote {written}")
        print(f"\n=== {argument} ===")
        for line in headline(summary).splitlines():
            print(f"  {line}")
        print("  validation (selected epochs: "
              f"{(summary.get('selected_epochs') or {})}):")
        print(metric_block(summary, "validation_metrics"))
        print("  final test:")
        print(metric_block(summary, "final_test_metrics"))
        print(rare_class_report(directory, summary, args.top))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
