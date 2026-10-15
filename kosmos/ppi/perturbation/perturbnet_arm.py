"""PerturbNet arms: a VAE-based model of unseen genetic perturbations, run in its
own environment and scored with the same metrics as every other arm.

PerturbNet (Yu & Welch) maps a perturbation's GO-annotation vector, through a
pretrained GenotypeVAE, to a conditional distribution over scVI cell latents
(a cINN). Its cell representation (scVI) is trained without labels, which is
where supplementary cells can enter:

  * `perturbnet_base`: scVI on the gold training cells (training perturbations
    plus every control);
  * `perturbnet_supp`: scVI on those plus the supplementary cells, with the
    source as scVI's batch covariate. cINN, GenotypeVAE, split and seed are
    identical; only the cell representation sees more cells.

PerturbNet pins Python 3.7 / scvi-tools 0.7.1 / torch 1.13, so it cannot share
Kosmos' environment. This module writes a bundle (raw counts of the training
cells, the perturbations to predict), runs `scripts/perturbnet_runner.py` with
`KOSMOS_PERTURBNET_PYTHON` (default: the `perturbnet` conda env), and reads back
the mean predicted log1p-CPM profile per perturbation.

Scoring: Δ̂_p = mean predicted profile − the gold control mean, against the
observed Δ_p = mean over p's cells of (y − control mean) -- the same
observations and metric code the GEARS / MLP arms are scored with. There is no
epoch selection (PerturbNet / scPerturbBench train fixed epochs); validation
metrics are reported, not used.

Raw counts come from the stage-2 study block (`study.counts`): count tables with
the same rows and panel as the normalised tables, plus each cell's
whole-transcriptome total.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .contract import PerturbationExample, PerturbationTask
from .metrics import perturbation_metrics, perturbation_signal

PERTURBNET_BASE = "perturbnet_base"
PERTURBNET_SUPP = "perturbnet_supp"
PERTURBNET_ARMS = (PERTURBNET_BASE, PERTURBNET_SUPP)

ROOT = Path(__file__).resolve().parents[3]
RUNNER = ROOT / "scripts" / "perturbnet_runner.py"
GENOTYPE_VAE_DIR = ROOT / "data" / "perturbation" / "perturbnet" / "genotypeVAE"
DEFAULT_PYTHON = Path.home() / "miniforge3" / "envs" / "perturbnet" / "bin" / "python"


def perturbnet_python() -> Path:
    return Path(os.getenv("KOSMOS_PERTURBNET_PYTHON", "") or DEFAULT_PYTHON)


def unavailable_reason(arm: str, counts: dict[str, Any] | None) -> str | None:
    """Why `arm` cannot run here, or None when it can."""
    if not counts or not counts.get("gold"):
        return "the plan's study block has no raw counts (`study.counts`); PerturbNet's scVI needs counts"
    if arm == PERTURBNET_SUPP and not counts.get("supplementary"):
        return "no supplementary count table in `study.counts`"
    if not perturbnet_python().exists():
        return f"no PerturbNet interpreter at {perturbnet_python()} (set KOSMOS_PERTURBNET_PYTHON)"
    if not (GENOTYPE_VAE_DIR / "model_params.pt").exists():
        return f"no pretrained GenotypeVAE in {GENOTYPE_VAE_DIR}"
    return None


def _row(example: PerturbationExample) -> int:
    return int(example.sample_id)


def _read_counts(path: str | Path, genes: Sequence[str], total_column: str) -> tuple[np.ndarray, np.ndarray]:
    table = pd.read_parquet(path, columns=[*genes, total_column])
    return table[list(genes)].to_numpy(dtype=np.float32), table[total_column].to_numpy(dtype=np.float64)


def _check_alignment(
    counts: np.ndarray,
    total: np.ndarray,
    examples: Sequence[PerturbationExample],
    rows: np.ndarray,
    target_sum: float,
) -> None:
    """The count rows must be the gold rows: log1p(CPM) of a count row has to
    reproduce the normalised row stage 3 trained on."""
    for position in range(min(50, len(examples))):
        rebuilt = np.log1p(counts[position] / max(total[position], 1.0) * target_sum)
        if not np.allclose(rebuilt, examples[position].target, atol=1e-3):
            raise ValueError(
                f"gold count row {int(rows[position])} does not reproduce the normalised gold "
                "row; the count table is not aligned with the gold table"
            )


def write_bundle(
    bundle: Path,
    *,
    arm: str,
    splits: dict[str, list[PerturbationExample]],
    task: PerturbationTask,
    counts: dict[str, Any],
    control_rows: np.ndarray,
    seed: int,
    settings: dict[str, Any],
) -> dict[str, Any]:
    bundle.mkdir(parents=True, exist_ok=True)
    genes = list(task.genes)
    total_column = counts.get("total_column", "total_counts")
    target_sum = float(counts.get("target_sum", 1e4))
    gold_counts, gold_total = _read_counts(counts["gold"], genes, total_column)
    train = splits["train"]
    train_rows = np.array([_row(example) for example in train], dtype=np.int64)
    _check_alignment(gold_counts[train_rows], gold_total[train_rows], train, train_rows, target_sum)
    control_rows = np.asarray(control_rows, dtype=np.int64)
    rebuilt = np.log1p(
        gold_counts[control_rows] / np.maximum(gold_total[control_rows], 1.0)[:, None] * target_sum
    ).mean(axis=0)
    if not np.allclose(rebuilt, train[0].control, atol=1e-3):
        raise ValueError("the gold control rows' counts do not reproduce the gold control mean")
    rows =np.concatenate([train_rows, np.asarray(control_rows, dtype=np.int64)])
    labels = [example.label for example in train] + ["ctrl"] * len(control_rows)
    np.save(bundle / "train_counts.npy", gold_counts[rows])
    np.save(bundle / "train_total.npy", gold_total[rows])
    (bundle / "train_labels.json").write_text(json.dumps(labels), encoding="utf-8")
    use_supp = arm == PERTURBNET_SUPP
    supp_cells = 0
    if use_supp:
        supp_counts, supp_total = _read_counts(counts["supplementary"], genes, total_column)
        kept = counts.get("supplementary_rows")
        if kept is not None:
            # the stage-3 supplementary cap: the cells the normalised table kept
            supp_counts, supp_total = supp_counts[kept], supp_total[kept]
        np.save(bundle / "supp_counts.npy", supp_counts)
        np.save(bundle / "supp_total.npy", supp_total)
        supp_cells = len(supp_counts)
    # every validation / test perturbation, sampled with as many cells as it has
    predict: dict[str, dict[str, Any]] = {}
    for name in ("validation", "test"):
        for example in splits.get(name) or []:
            entry = predict.setdefault(example.label, {"genes": list(example.perturbation), "n_cells": 0})
            entry["n_cells"] += 1
    (bundle / "predict.json").write_text(json.dumps(predict, indent=1), encoding="utf-8")
    config = {
        "arm": arm,
        "seed": int(seed),
        "genes": genes,
        "use_supplementary": use_supp,
        "target_sum": target_sum,
        "genotype_vae_dir": str(GENOTYPE_VAE_DIR),
        **settings,
    }
    (bundle / "config.json").write_text(json.dumps(config, indent=1), encoding="utf-8")
    return {
        "scvi_cells": {"gold_train": int(len(rows)), "supplementary": int(supp_cells)},
        "gold_control_cells": int(len(control_rows)),
        "perturbations_to_predict": len(predict),
    }


def _run_runner(bundle: Path, out: Path) -> None:
    command = [str(perturbnet_python()), "-u", str(RUNNER), "--bundle", str(bundle), "--out", str(out)]
    print(f"# perturbnet: {' '.join(command)}", flush=True)
    process = subprocess.Popen(
        command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, cwd=str(out)
    )
    tail: list[str] = []
    assert process.stdout is not None
    for line in process.stdout:
        line = line.rstrip()
        tail = (tail + [line])[-40:]
        if line.startswith("[perturbnet]") or "Epoch" in line or "Error" in line:
            print(f"#   {line}", flush=True)
    if process.wait() != 0:
        raise RuntimeError("PerturbNet runner failed:\n" + "\n".join(tail))


def _score(
    examples: Sequence[PerturbationExample],
    predicted: dict[str, np.ndarray],
    task: PerturbationTask,
    top_k: int,
) -> tuple[dict[str, Any], dict[str, np.ndarray], dict[str, np.ndarray]]:
    sums: dict[str, list] = {}
    for example in examples:
        entry = sums.setdefault(example.label, [0.0, 0.0, 0])
        entry[0] = entry[0] + np.asarray(example.delta, dtype=np.float64)
        entry[1] = entry[1] + np.asarray(example.control, dtype=np.float64)
        entry[2] += 1
    observations = {label: np.asarray(v[0] / v[2]) for label, v in sums.items()}
    # Δ̂ against the control each example's Δ was measured from (the gold mean)
    predictions = {
        label: np.asarray((predicted[label] - v[1] / v[2]) / task.delta_scale)
        for label, v in sums.items()
    }
    metrics = perturbation_metrics(predictions, observations, top_k=top_k)
    return metrics.to_dict(), predictions, observations


def run_perturbnet_arm(
    arm: str,
    *,
    splits: dict[str, list[PerturbationExample]],
    task: PerturbationTask,
    counts: dict[str, Any],
    control_rows: np.ndarray,
    out: Path,
    seed: int,
    top_k: int,
    settings: dict[str, Any],
) -> dict[str, Any]:
    started = time.time()
    # absolute: the runner starts in its own work dir, so a relative --out breaks it
    work = (out / arm).resolve()
    bundle_info = write_bundle(
        work / "bundle", arm=arm, splits=splits, task=task, counts=counts,
        control_rows=control_rows, seed=seed, settings=settings,
    )
    _run_runner(work / "bundle", work)
    if os.getenv("KOSMOS_PERTURBNET_KEEP_BUNDLE", "").strip().lower() not in ("1", "true", "yes"):
        # ~600 MB of exported counts per arm; the results do not need it
        shutil.rmtree(work / "bundle", ignore_errors=True)
    with np.load(work / "predictions.npz", allow_pickle=False) as data:
        predicted = {str(label): np.asarray(mean, dtype=np.float64) for label, mean in zip(data["labels"], data["means"])}
    runner_report = json.loads((work / "report.json").read_text(encoding="utf-8"))
    runner_history = json.loads((work / "history.json").read_text(encoding="utf-8"))
    validation = (
        _score(splits["validation"], predicted, task, top_k)[0] if splits.get("validation") else None
    )
    test, predictions, observations = _score(splits["test"], predicted, task, top_k)
    labels = sorted(predictions)
    np.savez_compressed(
        out / f"{arm}_test_predictions.npz",
        labels=np.array(labels),
        predictions=np.stack([predictions[label] for label in labels]),
        observations=np.stack([observations[label] for label in labels]),
    )
    # the decoder's own control: how far a generated ctrl profile sits from the
    # real control mean (a calibration check, not a metric)
    control_mean = np.mean([example.control for example in splits["test"]], axis=0)
    control_gap = predicted["ctrl"] - control_mean
    train_loss = runner_history.get("cinn_train_loss") or []
    heldout = runner_history.get("cinn_heldout_cell_loss") or []
    history = [
        {"epoch": epoch + 1, "stage": "cinn", "train_loss": loss,
         "heldout_cell_loss": heldout[epoch] if epoch < len(heldout) else None}
        for epoch, loss in enumerate(train_loss)
    ]
    payload = {
        "validation": validation,
        "test": test,
        "history": history,
        "epochs_run": len(history),
        "best": {"epoch": len(history), "selection": "fixed epochs (PerturbNet); no selection"},
        "seconds": round(time.time() - started, 1),
        "predictions": str(out / f"{arm}_test_predictions.npz"),
        "signal": perturbation_signal(predictions, observations),
        "perturbnet": {
            **bundle_info,
            "settings": settings,
            "runner": runner_report,
            "scvi_final_elbo": {
                key: (values[-1] if values else None)
                for key, values in runner_history.items()
                if key.startswith("elbo")
            },
            "generated_control_vs_real": {
                "mse": float(np.mean(control_gap**2)),
                "mean_abs": float(np.mean(np.abs(control_gap))),
            },
            "work_dir": str(work),
        },
    }
    print(
        f"# {arm} done in {payload['seconds']:.0f}s: "
        f"test mse_deg {test['summary']['mse_deg']:.4f}, "
        f"pearson {test['summary']['pearson']:.3f}, "
        f"direction {test['summary']['direction_accuracy']:.3f}",
        flush=True,
    )
    return payload


def default_settings() -> dict[str, Any]:
    """PerturbNet tutorial / scPerturbBench settings; env overrides for smoke runs."""
    return {
        "scvi_latent": 10,
        "scvi_epochs": int(os.getenv("KOSMOS_PERTURBNET_SCVI_EPOCHS", "") or 500),
        "cinn_epochs": int(os.getenv("KOSMOS_PERTURBNET_CINN_EPOCHS", "") or 45),
        "cinn_lr": 4.5e-6,
        "control_cells_to_sample": 2000,
    }


__all__ = [
    "PERTURBNET_ARMS",
    "PERTURBNET_BASE",
    "PERTURBNET_SUPP",
    "default_settings",
    "run_perturbnet_arm",
    "unavailable_reason",
]
