#!/usr/bin/env python
"""Stage 2 (prepare): turn a stage-1 data package into training tables + a study spec.

    stage 1  kosmos dataprep "<question>" --dataset <screen> [--supplementary-dataset <screen>]
             -> plan.json (h5ad tables) + data.md                       [human review]
    stage 2  scripts/prepare_perturbation.py --stage1 <dir> --out <dir>
             -> gold.parquet, supp_<name>.parquet, plan.json (+ study), prepare_report.md
                                                                        [human review]
    stage 3  kosmos experiment "<question>" --data-plan <stage-2 plan> --task perturbation

What it does, and the defaults (the GEARS paper's setting):

  * gold: every cell of the screen, `condition` (GEARS' `GENE+ctrl` / `ctrl`),
    `cell_type`, and **all** measured genes as columns (symbols). No extra
    normalisation: GEARS' release is already log-normalised (`--normalize
    zscore_clip` applies the per-gene z-score + clip used for the IRF1 screen).
  * supplementary: aligned to the gold's genes **by gene id**; a gene it does not
    measure is filled with the gold's control mean (a constant, so it gets no
    edge in the supplementary co-expression graph). `--supp-cells all` (default)
    drops its condition column so every cell is used as an unlabeled input;
    `controls` keeps only its control cells.
  * the study spec written into `plan.json` -- the only handoff to stage 3:
    split by perturbation (`unseen_single`: GEARS' simulation split, 25% of the
    perturbed genes held out for test, 10% of the rest for validation), and the
    panel size (0 = every gene, as GEARS).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def log(message: str) -> None:
    print(f"[prepare] {message}", flush=True)


def dense(matrix) -> np.ndarray:
    return np.asarray(matrix.toarray() if hasattr(matrix, "toarray") else matrix, dtype=np.float32)


def zscore_clip(values: np.ndarray) -> np.ndarray:
    mean = values.mean(axis=0, keepdims=True)
    std = values.std(axis=0, keepdims=True)
    std[std == 0] = 1.0
    return np.clip((values - mean) / std, 0.0, None).astype(np.float32)


def symbols(adata) -> list[str]:
    return [str(name) for name in (adata.var["gene_name"] if "gene_name" in adata.var else adata.var_names)]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--stage1", required=True, help="stage-1 directory (plan.json from dataprep --dataset)")
    parser.add_argument("--out", required=True, help="where the training tables and the stage-2 plan go")
    parser.add_argument("--split-mode", default="unseen_single", choices=("unseen_single", "unseen_combination", "mixed"))
    parser.add_argument("--test-fraction", type=float, default=0.25, help="GEARS: 1 - train_gene_set_size (0.75)")
    parser.add_argument("--validation-fraction", type=float, default=0.1, help="GEARS: 10%% of the training genes")
    parser.add_argument("--max-panel-genes", type=int, default=0, help="0 = every measured gene (GEARS)")
    parser.add_argument("--normalize", choices=("none", "zscore_clip"), default="none")
    parser.add_argument("--supp-cells", choices=("all", "controls"), default="all")
    args = parser.parse_args(argv)

    import anndata

    from kosmos.ppi.perturbation.graphs import GO_REFERENCE, go_vocabulary

    stage1 = Path(args.stage1)
    plan_in = json.loads((stage1 / "plan.json").read_text(encoding="utf-8"))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    report: dict = {"stage1": str(stage1), "objective": plan_in["task"]["objective"], "args": vars(args)}

    gold_entry = plan_in["gold"][0]
    log(f"gold: {gold_entry['path']}")
    gold = anndata.read_h5ad(gold_entry["path"])
    genes = symbols(gold)
    keep_genes = ~pd.Index(genes).duplicated()
    if (~keep_genes).any():
        log(f"  {int((~keep_genes).sum())} duplicated gene symbol(s) dropped")
    gene_ids = [str(value) for value, kept in zip(gold.var_names, keep_genes) if kept]
    genes = [gene for gene, kept in zip(genes, keep_genes) if kept]
    values = dense(gold.X)[:, keep_genes]
    conditions = gold.obs["condition"].astype(str).to_numpy()
    control = conditions == "ctrl"
    control_mean = values[control].mean(axis=0)
    if args.normalize == "zscore_clip":
        values = zscore_clip(values)
        control_mean = values[control].mean(axis=0)
    vocabulary = set(go_vocabulary(GO_REFERENCE)) if GO_REFERENCE.exists() else set()
    perturbed = sorted({g for c in set(conditions) for g in c.split("+") if g and g != "ctrl"})
    outside = [gene for gene in perturbed if vocabulary and gene not in vocabulary]
    frame = pd.DataFrame(values, columns=genes)
    frame.insert(0, "cell_type", gold.obs["cell_type"].astype(str).to_numpy() if "cell_type" in gold.obs else "unknown")
    frame.insert(0, "condition", conditions)
    gold_out = out / "gold.parquet"
    frame.to_parquet(gold_out, index=False)
    report["gold"] = {
        "dataset": gold_entry.get("dataset"),
        "source": gold_entry["path"],
        "written": str(gold_out),
        "cells": int(len(frame)),
        "genes": len(genes),
        "control_cells": int(control.sum()),
        "perturbed_genes": len(perturbed),
        "perturbed_genes_outside_go_vocabulary": outside,
        "normalize": args.normalize,
    }
    log(f"  -> {gold_out.name}: {len(frame):,} cells x {len(genes):,} genes, {len(perturbed):,} perturbed genes "
        f"({len(outside)} outside the GO vocabulary, dropped at training)")
    del frame, values

    supp_entries = []
    report["supplementary"] = []
    for entry in plan_in.get("supplementary") or []:
        log(f"supplementary: {entry['path']}")
        supp = anndata.read_h5ad(entry["path"])
        supp_values = dense(supp.X)
        supp_conditions = supp.obs["condition"].astype(str).to_numpy() if "condition" in supp.obs else None
        if args.supp_cells == "controls" and supp_conditions is not None:
            mask = supp_conditions == "ctrl"
            supp_values, supp_conditions = supp_values[mask], supp_conditions[mask]
        position = {str(gene_id): index for index, gene_id in enumerate(supp.var_names)}
        aligned = np.tile(control_mean, (len(supp_values), 1)).astype(np.float32)
        shared = [(column, position[gene_id]) for column, gene_id in enumerate(gene_ids) if gene_id in position]
        if shared:
            columns, source_columns = zip(*shared)
            measured = supp_values[:, list(source_columns)]
            if args.normalize == "zscore_clip":
                measured = zscore_clip(measured)
            aligned[:, list(columns)] = measured
        supp_frame = pd.DataFrame(aligned, columns=genes)
        if args.supp_cells == "controls" and supp_conditions is not None:
            supp_frame.insert(0, "condition", supp_conditions)
        name = entry.get("dataset") or Path(entry["path"]).parent.name
        written = out / f"supp_{name}.parquet"
        supp_frame.to_parquet(written, index=False)
        info = {
            "dataset": name,
            "source": entry["path"],
            "written": str(written),
            "cells": int(len(supp_frame)),
            "cells_used": args.supp_cells,
            "genes_measured": len(shared),
            "genes_filled_with_gold_control_mean": len(genes) - len(shared),
        }
        report["supplementary"].append(info)
        supp_entries.append(
            {
                "path": str(written),
                "role": "supplementary",
                "reason": (
                    f"{name}: {len(supp_frame):,} cells ({args.supp_cells}), {len(shared):,} of "
                    f"{len(genes):,} genes measured, the rest = gold control mean"
                ),
            }
        )
        log(f"  -> {written.name}: {len(supp_frame):,} cells; {len(shared):,}/{len(genes):,} genes measured")

    study = {
        "question": plan_in["task"]["objective"],
        "dataset": gold_entry.get("dataset"),
        "held_out_unit": {
            "unseen_single": "perturbed genes never seen in training",
            "unseen_combination": "combinations never seen in training",
            "mixed": "single genes and combinations",
        }[args.split_mode],
        "condition_column": "condition",
        "split_mode": args.split_mode,
        "test_fraction": args.test_fraction,
        "validation_fraction": args.validation_fraction,
        "max_panel_genes": args.max_panel_genes,
        "normalize": args.normalize,
    }
    plan = {
        "version": 1,
        "task": {**plan_in["task"], "target_column": "condition", "backend": "perturbation"},
        "study": study,
        "gold": [{"path": str(gold_out), "role": "gold", "reason": f"{gold_entry.get('dataset')}: every cell, all genes"}],
        "supplementary": supp_entries,
        "unusable": [],
    }
    (out / "plan.json").write_text(json.dumps(plan, indent=2), encoding="utf-8")
    report["study"] = study
    (out / "prepare_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (out / "prepare_report.md").write_text(render(report), encoding="utf-8")
    log(f"done -> {out / 'plan.json'}")
    return 0


def render(report: dict) -> str:
    gold, study = report["gold"], report["study"]
    lines = [
        f"# Prepare report — {gold['dataset']}",
        "",
        f"**Question:** {report['objective']}",
        "",
        "## Study (stage 3 reads this from plan.json)",
        "",
        f"- held-out unit: {study['held_out_unit']} (`{study['split_mode']}`)",
        f"- test {study['test_fraction']:.0%} of the perturbations, validation "
        f"{study['validation_fraction']:.0%} of the rest",
        f"- panel: {'every measured gene' if not study['max_panel_genes'] else study['max_panel_genes']}",
        f"- normalisation: {study['normalize']}",
        "",
        "## Gold",
        "",
        f"- `{gold['written']}` ← `{gold['source']}`",
        f"- {gold['cells']:,} cells ({gold['control_cells']:,} controls) × {gold['genes']:,} genes",
        f"- {gold['perturbed_genes']:,} perturbed genes; outside the GO vocabulary (not encodable, "
        f"dropped at training): {len(gold['perturbed_genes_outside_go_vocabulary'])} "
        f"{gold['perturbed_genes_outside_go_vocabulary'][:10]}",
        "",
        "## Supplementary",
        "",
    ]
    for info in report["supplementary"]:
        lines += [
            f"- `{info['written']}` ← `{info['source']}`",
            f"  - {info['cells']:,} cells ({info['cells_used']}); {info['genes_measured']:,} genes measured, "
            f"{info['genes_filled_with_gold_control_mean']:,} filled with the gold control mean",
        ]
    if not report["supplementary"]:
        lines.append("- none")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    raise SystemExit(main())
