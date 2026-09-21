"""Corrupt the synthetic labels and watch the gate close.

The claim behind gold-anchored gating is not "synthetic data helps" -- it is
that the mechanism *knows when it does not*, and falls back to the labeled rows
on its own. That claim is testable: hold the model, the split, the optimizer,
the seed and the metrics fixed, and degrade only the quality of the synthetic
labels. Three arms see the same corrupted cohort:

    gold                 -- the labeled rows only (the floor)
    gold_plus_synthetic  -- L_G + λ·L_S, applied in full (the naive baseline)
    gradient_gated       -- g_G + λ_t·g_S, λ_t from agreement with g_G

and the question is whether the gated arm's synthetic weight falls as the
corruption rises, and whether it avoids the naive baseline's damage.

    .venv/bin/python scripts/gradient_gate_stress.py --out artifacts/runs/gate-stress

Writes `results.json`, `report.md` and `figures/gate_stress.png`.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from kosmos.ppi import PPITrainingConfig, evaluate_final_test, run_ppi_experiment  # noqa: E402
from kosmos.ppi.schemas import ExternalEvidenceDataset, GoldDataset  # noqa: E402

DEFAULT_LEVELS = (0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0)
MODES = ("gold", "gold_plus_synthetic", "gradient_gated")


class CorruptedTeacher(BaseEstimator, ClassifierMixin):
    """The ordinary cross-fit teacher, with a share of its labels replaced.

    This is how "synthetic label quality" is varied without touching anything
    else: the unlabeled cohort, the features, the split and the model are
    identical across runs, and only the labels the synthetic rows carry change.
    A replacement that lands on the same class is not counted, so `rate` is the
    fraction of labels that are actually wrong.
    """

    def __init__(self, base=None, rate=0.0, seed=0):
        self.base = base
        self.rate = rate
        self.seed = seed

    def fit(self, X, y):
        self.base_ = clone(self.base if self.base is not None else _default_base())
        self.base_.fit(X, y)
        self.classes_ = np.asarray(self.base_.classes_)
        return self

    def _plan(self, rows):
        """Which rows the teacher is wrong about, and what it says instead."""
        rng = np.random.default_rng(self.seed)
        values = self.classes_
        replacement = values[rng.integers(0, len(values), rows)]
        chosen = rng.random(rows) < float(self.rate)
        return chosen, replacement

    def predict(self, X):
        predictions = np.asarray(self.base_.predict(X))
        if not self.rate:
            return predictions
        chosen, replacement = self._plan(len(predictions))
        out = predictions.copy()
        # A "corruption" that repeats the original label is not a corruption.
        change = chosen & (replacement != predictions)
        out[change] = replacement[change]
        return out

    def predict_proba(self, X):
        probabilities = np.asarray(self.base_.predict_proba(X), dtype=float)
        if not self.rate:
            return probabilities
        # The teacher has to be wrong in its distribution too. The pipeline takes
        # the argmax of these probabilities when every fold model has them, so a
        # corrupted `predict` with clean probabilities would be silently undone.
        predictions = np.asarray(self.base_.predict(X))
        chosen, replacement = self._plan(len(predictions))
        change = np.where(chosen & (replacement != predictions))[0]
        if not len(change):
            return probabilities
        order = list(self.classes_)
        probabilities = probabilities.copy()
        probabilities[change] = 0.05 / max(1, len(order) - 1)
        probabilities[change, [order.index(value) for value in replacement[change]]] = 0.95
        return probabilities


def _default_base():
    return make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, random_state=0))


def build_data(seed: int, n_classes: int = 3):
    """A labeled pool, an unlabeled cohort, and a test set nobody trains on."""
    X, y = make_classification(
        n_samples=1200,
        n_features=20,
        n_informative=10,
        n_redundant=4,
        n_classes=n_classes,
        class_sep=1.5,
        flip_y=0.01,
        random_state=seed,
    )
    names = [f"feature-{i}" for i in range(X.shape[1])]
    pool = np.arange(len(X))
    labelled, rest = train_test_split(pool, test_size=0.6, random_state=seed, stratify=y)
    train, validation = train_test_split(
        labelled, test_size=0.4, random_state=seed, stratify=y[labelled]
    )
    cohort, test = train_test_split(rest, test_size=0.4, random_state=seed, stratify=y[rest])

    def gold(rows, role, tag):
        return GoldDataset(
            X[rows],
            y[rows],
            np.array([f"{tag}-{i}" for i in rows]),
            role,
            "stress-gold",
            names,
        )

    external = ExternalEvidenceDataset(
        X[cohort],
        np.array([f"cohort-{i}" for i in cohort]),
        "DIRECT_PPI",
        "stress-cohort",
        names,
    )
    return (
        gold(train, "gold_train", "train"),
        gold(validation, "gold_validation", "val"),
        gold(test, "final_test", "test"),
        external,
    )


def run_one(
    mode: str,
    level: float,
    seed: int,
    output_dir: Path,
    *,
    max_epochs: int,
    patience: int,
    gate_scope: str,
    kappa: float,
):
    train, validation, test, cohort = build_data(seed)
    config = PPITrainingConfig(
        loss_mode="gold_plus_synthetic" if mode == "gold_plus_synthetic" else (
            "gradient_gated" if mode == "gradient_gated" else "gradient_gated"
        ),
        max_epochs=max_epochs,
        patience=patience,
        gold_batch_size=45,
        external_batch_size=64,
        max_external_samples=len(cohort.X),
        cross_fit_folds=3,
        ppi_lambda=1.0,
        gate_scope=gate_scope,
        gate_kappa=kappa,
        gate_lambda=1.0,
        seed=seed,
    )
    teacher = CorruptedTeacher(base=_default_base(), rate=level, seed=seed)
    result = run_ppi_experiment(
        gold_train=train,
        gold_validation=validation,
        # The "gold" arm is the same run with no synthetic rows at all: the
        # pipeline's own baseline is exactly that, and it is fitted identically.
        external_evidence=[] if mode == "gold" else [cohort],
        pseudo_labeler=teacher,
        config=config,
        output_dir=output_dir,
    )
    # `evaluate_final_test` scores both arms on the same held-out rows. For the
    # `gold` arm the two are the same model, so one call serves all three modes.
    final = evaluate_final_test(result, test)
    final_ppi = (final or {}).get("ppi") or {}
    final_baseline = (final or {}).get("baseline") or {}
    gate = (result.training or {}).get("gate") or {}
    return {
        "mode": mode,
        "corruption": level,
        "seed": seed,
        "validation_balanced_accuracy": result.validation_metrics.get("balanced_accuracy"),
        "validation_accuracy": result.validation_metrics.get("accuracy"),
        "validation_macro_f1": result.validation_metrics.get("macro_f1"),
        "baseline_validation_balanced_accuracy": result.baseline_metrics.get(
            "balanced_accuracy"
        ),
        "test_balanced_accuracy": final_ppi.get("balanced_accuracy"),
        "test_accuracy": final_ppi.get("accuracy"),
        "test_macro_f1": final_ppi.get("macro_f1"),
        "baseline_test_balanced_accuracy": final_baseline.get("balanced_accuracy"),
        "gradient_cosine_mean": gate.get("gradient_cosine_mean"),
        "gradient_cosine_std": gate.get("gradient_cosine_std"),
        "synthetic_weight_mean": gate.get("synthetic_weight_mean"),
        "synthetic_weight_std": gate.get("synthetic_weight_std"),
        "synthetic_active_fraction": gate.get("synthetic_active_fraction"),
        "epochs": gate.get("epochs") or [],
        "result_dir": str(output_dir),
    }


def summarise(records: list[dict]) -> list[dict]:
    """Mean and spread over seeds, per (mode, corruption)."""
    rows = []
    keys = (
        "validation_balanced_accuracy",
        "test_balanced_accuracy",
        "test_macro_f1",
        "gradient_cosine_mean",
        "synthetic_weight_mean",
        "synthetic_active_fraction",
    )
    for mode in MODES:
        for level in sorted({r["corruption"] for r in records}):
            group = [
                r
                for r in records
                if r["mode"] == mode and abs(r["corruption"] - level) < 1e-9
            ]
            if not group:
                continue
            row = {"mode": mode, "corruption": level, "runs": len(group)}
            for key in keys:
                values = [r[key] for r in group if r[key] is not None]
                row[key] = float(np.mean(values)) if values else None
                row[f"{key}_std"] = float(np.std(values)) if values else None
            rows.append(row)
    return rows


def render_report(summary: list[dict], records: list[dict], *, kappa: float, scope: str) -> str:
    lines = [
        "# Gradient-gate stress test",
        "",
        "The same unlabeled cohort, the same split, model, optimizer and seed; only the",
        "quality of the synthetic labels changes. `gold` sees the labeled rows alone,",
        "`gold_plus_synthetic` applies `L_G + λ·L_S` in full, and `gradient_gated`",
        "admits the synthetic gradient only where it agrees with the gold gradient",
        f"(κ={kappa}, scope={scope}).",
        "",
        "## Balanced accuracy on the held-out test set",
        "",
        "| corruption | gold | gold_plus_synthetic | gradient_gated | gated vs naive |",
        "|---|---|---|---|---|",
    ]
    by_level: dict[float, dict[str, dict]] = {}
    for row in summary:
        by_level.setdefault(row["corruption"], {})[row["mode"]] = row

    def cell(row, key="test_balanced_accuracy"):
        if row is None or row.get(key) is None:
            return "n/a"
        value = row[key]
        spread = row.get(f"{key}_std")
        return f"{value:.4f}" + (f" ± {spread:.4f}" if spread else "")

    for level in sorted(by_level):
        group = by_level[level]
        gated = group.get("gradient_gated")
        naive = group.get("gold_plus_synthetic")
        delta = (
            f"{gated['test_balanced_accuracy'] - naive['test_balanced_accuracy']:+.4f}"
            if gated
            and naive
            and gated.get("test_balanced_accuracy") is not None
            and naive.get("test_balanced_accuracy") is not None
            else "n/a"
        )
        lines.append(
            f"| {level:.0%} | {cell(group.get('gold'))} | {cell(naive)} | "
            f"{cell(gated)} | {delta} |"
        )

    lines += [
        "",
        "## What the gate did",
        "",
        "| corruption | cos(g_G, g_S) | synthetic weight | active fraction |",
        "|---|---|---|---|",
    ]
    for level in sorted(by_level):
        row = by_level[level].get("gradient_gated")
        if row is None:
            continue
        lines.append(
            f"| {level:.0%} | {cell(row, 'gradient_cosine_mean')} | "
            f"{cell(row, 'synthetic_weight_mean')} | "
            f"{(row.get('synthetic_active_fraction') or 0):.0%} |"
        )

    lines += [
        "",
        "## What to look for",
        "",
        "1. Does the synthetic weight fall as the corruption rises, and does the",
        "   active fraction reach zero at total corruption? That is the adaptive",
        "   behaviour, and it is the mechanism's own claim.",
        "2. Does the gated arm stay at or above the gold-only arm when the",
        "   synthetic labels are bad? The fallback is exact (`λ_t = 0` ⇒",
        "   `g_final = g_G`), so a drop here is noise in the run, not the gate",
        "   leaking.",
        "3. Does the naive baseline degrade faster than the gated one as corruption",
        "   rises? That is the comparison the method has to earn.",
        "4. Is the cosine predictive of usefulness? If high-cosine runs are not",
        "   the good ones, the gate is a safe default but not a selector.",
        "",
        "Per-run detail (every epoch's gate diagnostics) is in `results.json`.",
    ]
    return "\n".join(lines) + "\n"


def write_figure(summary: list[dict], out: Path) -> str | None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:  # noqa: BLE001 - a figure is a convenience, not the result
        return None
    levels = sorted({row["corruption"] for row in summary})
    figure, axes = plt.subplots(1, 2, figsize=(11, 4))
    for mode in MODES:
        rows = {row["corruption"]: row for row in summary if row["mode"] == mode}
        values = [rows.get(level, {}).get("test_balanced_accuracy") for level in levels]
        axes[0].plot(levels, values, marker="o", label=mode)
    axes[0].set_xlabel("synthetic label corruption")
    axes[0].set_ylabel("test balanced accuracy")
    axes[0].set_title("Does the gate limit the damage?")
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.3)

    gated = {row["corruption"]: row for row in summary if row["mode"] == "gradient_gated"}
    axes[1].plot(
        levels,
        [gated.get(level, {}).get("gradient_cosine_mean") for level in levels],
        marker="o",
        label="cos(g_G, g_S)",
    )
    axes[1].plot(
        levels,
        [gated.get(level, {}).get("synthetic_weight_mean") for level in levels],
        marker="s",
        label="synthetic weight λ_t",
    )
    axes[1].plot(
        levels,
        [gated.get(level, {}).get("synthetic_active_fraction") for level in levels],
        marker="^",
        label="active fraction",
    )
    axes[1].set_xlabel("synthetic label corruption")
    axes[1].set_title("Does the gate close?")
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.3)
    figure.tight_layout()
    target = out / "figures" / "gate_stress.png"
    target.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(target, dpi=150)
    plt.close(figure)
    return str(target)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("artifacts/runs/gate-stress"))
    parser.add_argument("--seeds", type=int, nargs="*", default=[0, 1, 2])
    parser.add_argument("--levels", type=float, nargs="*", default=list(DEFAULT_LEVELS))
    parser.add_argument("--max-epochs", type=int, default=40)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--gate-scope", default="batch", choices=("batch", "sample"))
    parser.add_argument("--gate-kappa", type=float, default=1.0)
    args = parser.parse_args()

    if args.out.exists() and any(args.out.iterdir()):
        parser.error(f"{args.out} is not empty; the stress test never overwrites a run")
    args.out.mkdir(parents=True, exist_ok=True)

    records: list[dict] = []
    started = time.time()
    for seed in args.seeds:
        for level in args.levels:
            for mode in MODES:
                run_dir = args.out / "runs" / f"{mode}-c{int(round(level * 100)):03d}-s{seed}"
                print(f"# {mode:20s} corruption={level:.0%} seed={seed}", flush=True)
                record = run_one(
                    mode,
                    level,
                    seed,
                    run_dir,
                    max_epochs=args.max_epochs,
                    patience=args.patience,
                    gate_scope=args.gate_scope,
                    kappa=args.gate_kappa,
                )
                records.append(record)
                print(
                    "    val balanced acc "
                    f"{record['validation_balanced_accuracy']:.4f}"
                    + (
                        f" | test {record['test_balanced_accuracy']:.4f}"
                        if record["test_balanced_accuracy"] is not None
                        else ""
                    )
                    + (
                        f" | cosine {record['gradient_cosine_mean']:.3f}"
                        f" | weight {record['synthetic_weight_mean']:.3f}"
                        f" | active {record['synthetic_active_fraction']:.0%}"
                        if record["gradient_cosine_mean"] is not None
                        else ""
                    ),
                    flush=True,
                )
    summary = summarise(records)
    (args.out / "results.json").write_text(
        json.dumps(
            {
                "levels": list(args.levels),
                "seeds": list(args.seeds),
                "modes": list(MODES),
                "gate": {"scope": args.gate_scope, "kappa": args.gate_kappa},
                "max_epochs": args.max_epochs,
                "patience": args.patience,
                "seconds": round(time.time() - started, 1),
                "summary": summary,
                "runs": records,
            },
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )
    (args.out / "report.md").write_text(
        render_report(summary, records, kappa=args.gate_kappa, scope=args.gate_scope),
        encoding="utf-8",
    )
    figure = write_figure(summary, args.out)
    print(f"\n# stress test written to {args.out}")
    print(f"#   report : {args.out / 'report.md'}")
    print(f"#   results: {args.out / 'results.json'}")
    if figure:
        print(f"#   figure : {figure}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
