#!/usr/bin/env python
"""Middle stage for the IRF1 screen: subset the stage-1 plan's gold *and* supp.

Stage 1 already ran and wrote a plan:

    artifacts/dataprep/irf1-run-mine/plan.json
        gold         GSE153056_ECCITE_metadata.tsv.gz        (labels: barcode -> gene)
        supplementary GSM4633605/3608/3612/3613/3614_*counts  (expression: genes x cells)

This script reads *that* plan, not hard-coded paths, and:

  1. cuts the gold down to the target perturbation(s) + controls (IRF1 + NT);
  2. cuts every supplementary table to those same cells, keeping the gene panel;
  3. writes a **subset report**;
  4. picks HVG once (on the measurement half over the subset cells);
  5. for every table: log1p-CPM (optional) -> per gene, z-score across all cells
     -> clip negatives to zero;
  6. writes the training inputs stage 2 consumes:

        gold_subset.csv.gz          barcode + condition            (labels)
        supp_<i>_<name>.csv.gz      barcode + <panel genes>        (expression)
        plan.json                   gold = the labels, supplementary = every
                                    expression table, so the backend joins them
                                    on the barcode and the augmented arm gets its
                                    PPI correction from gold + supp.

Run:

    python Perturb_middle/subset_process_irf1.py
    python Perturb_middle/subset_process_irf1.py --stage1 artifacts/dataprep/irf1-run-mine

Then:

    kosmos experiment "<question>" --data-plan <out>/plan.json --task perturbation
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

STAGE1 = "artifacts/dataprep/irf1-run-mine"
OUT = "Perturb_middle/output_irf1-run-mine"
TARGETS = "IRF1"
CONTROLS = "NT"
BARCODE_HINTS = ("barcode", "cell_barcode", "cellbarcode", "cell_id", "orig.ident", "index")


def log(message: str) -> None:
    print(f"[subset] {message}", flush=True)


def sep_for(path: Path) -> str:
    return "\t" if str(path).lower().endswith((".tsv", ".tsv.gz", ".txt", ".txt.gz")) else ","


def norm_barcode(values) -> pd.Series:
    """A barcode without its `-1` / `.1` suffix, the way the backend join reads it."""
    text = pd.Series(values).astype(str).str.strip()
    return text.str.replace(r"([.\-])\d+$", "", regex=True)


def pick_barcode_column(df: pd.DataFrame) -> str:
    """The column that identifies a cell: the unique first column, else a hint."""
    rows = max(1, len(df))

    def unique_ratio(column) -> float:
        return df[column].astype(str).nunique() / rows

    if unique_ratio(df.columns[0]) >= 0.9:
        return str(df.columns[0])
    names = {str(c).strip().lower(): c for c in df.columns}
    for hint in BARCODE_HINTS:
        column = names.get(hint)
        if column is not None and unique_ratio(column) >= 0.9:
            return str(column)
    return str(df.columns[0])


def is_control_name(value: str, controls: list[str]) -> bool:
    """`NT` and `NTg5` are the same control; `IRF1` is not."""
    low = str(value).strip().lower()
    return any(low == c.lower() or low.startswith(c.lower()) for c in controls)


def read_plan(stage1: Path) -> dict:
    plan_path = stage1 / "plan.json"
    if not plan_path.exists():
        raise SystemExit(f"no plan.json under {stage1}; point --stage1 at a stage-1 run dir")
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    if not plan.get("gold"):
        raise SystemExit(f"{plan_path} has no gold table")
    return plan


def subset_labels(
    path: Path, targets: list[str], controls: list[str], condition_column: str
) -> tuple[pd.DataFrame, dict]:
    frame = pd.read_csv(path, sep=sep_for(path), low_memory=False)
    barcode_col = pick_barcode_column(frame)
    condition = frame[condition_column].astype(str).str.strip()
    if any(str(t).lower() == "all" for t in targets):
        # Every perturbed cell plus the controls: a perturbation-response model
        # is trained across perturbations, so a one-perturbation subset cannot
        # be split (validation/test would be empty).
        mask = pd.Series(True, index=frame.index)
    else:
        mask = condition.isin([*targets, *controls])
    kept = frame[mask].copy()
    kept["barcode"] = norm_barcode(frame.loc[kept.index, barcode_col]).to_numpy()
    kept["condition"] = condition.loc[kept.index].to_numpy()
    return kept, {
        "path": str(path),
        "barcode_column": barcode_col,
        "condition_column": condition_column,
        "cells_total": int(len(frame)),
        "cells_kept": int(len(kept)),
        "condition_counts": kept["condition"].value_counts().to_dict(),
    }


def read_counts_for_cells(path: Path, wanted: set[str]) -> pd.DataFrame:
    """A genes x cells matrix cut to `wanted` barcodes (GEO's orientation)."""
    sep = sep_for(path)
    header = pd.read_csv(path, sep=sep, nrows=0)
    gene_col = str(header.columns[0])
    by_barcode = {norm_barcode([str(c)]).iloc[0]: c for c in header.columns[1:]}
    keep = [by_barcode[b] for b in wanted if b in by_barcode]
    if not keep:
        raise SystemExit(f"no barcode in {path} matches the label subset")
    genes = pd.read_csv(path, sep=sep, usecols=[gene_col])[gene_col].astype(str)
    matrix = pd.read_csv(path, sep=sep, usecols=keep, dtype=np.float32)
    matrix.index = genes.to_numpy()
    matrix.columns = norm_barcode(list(matrix.columns)).to_numpy()
    return matrix


def read_counts_panel(path: Path, panel: list[str], max_cells: int = 0) -> pd.DataFrame | None:
    """A genes x cells matrix cut to `panel` genes, keeping its own cells.

    Not every supplementary table measures the gold's cells -- a second arm of
    the same screen (CITE vs ECCITE) has its own barcodes. Such a table is still
    evidence through the genes it shares, so it is cut on the gene panel and
    **all** of its cells are kept (a cell cap would throw evidence away for no
    reason; `max_cells > 0` only exists for a deliberately small run).
    """
    sep = sep_for(path)
    header = pd.read_csv(path, sep=sep, nrows=0)
    gene_col = str(header.columns[0])
    cells = list(header.columns[1:])
    if max_cells:
        cells = cells[:max_cells]
    genes = pd.read_csv(path, sep=sep, usecols=[gene_col])[gene_col].astype(str)
    matrix = pd.read_csv(path, sep=sep, usecols=cells, dtype=np.float32)
    matrix.index = genes.to_numpy()
    wanted = set(panel)
    keep = [gene for gene in matrix.index if gene in wanted]
    if len(keep) < 10:
        return None
    frame = matrix.loc[keep].T  # cells x genes
    return frame.reindex(columns=panel).fillna(0.0)


def log1p_cpm(frame: pd.DataFrame, target_sum: float) -> pd.DataFrame:
    totals = frame.sum(axis=1).replace(0.0, np.nan)
    return np.log1p((frame.div(totals, axis=0) * float(target_sum)).fillna(0.0))


def zscore_clip(frame: pd.DataFrame) -> pd.DataFrame:
    """Per gene across cells: (x - mean) / sd, then negatives -> 0."""
    mean = frame.mean(axis=0)
    std = frame.std(axis=0, ddof=0).replace(0.0, np.nan)
    return (frame - mean).div(std, axis=1).fillna(0.0).clip(lower=0.0)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--stage1", default=STAGE1, help="stage-1 run dir containing plan.json")
    parser.add_argument("--out", default=OUT, help="where the subset + plan are written")
    parser.add_argument("--targets", default=TARGETS, help="perturbed gene(s), comma-separated")
    parser.add_argument("--controls", default=CONTROLS, help="control label(s), comma-separated")
    parser.add_argument("--condition-column", default="gene", help="column holding the perturbation identity")
    parser.add_argument("--n-hvg", type=int, default=2000, help="highly variable genes to keep")
    parser.add_argument(
        "--supp-max-cells",
        type=int,
        default=0,
        help="cells to keep per supplementary table (0 = all; a cap discards evidence)",
    )
    parser.add_argument("--normalize", choices=("log1p_cpm", "none"), default="log1p_cpm")
    parser.add_argument("--target-sum", type=float, default=1e4)
    parser.add_argument("--name", default=None, help="output stem (default: the stage-1 run name)")
    args = parser.parse_args(argv)

    stage1 = Path(args.stage1)
    plan_in = read_plan(stage1)
    name = args.name or stage1.name
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    targets = [t.strip() for t in args.targets.split(",") if t.strip()]
    controls = [c.strip() for c in args.controls.split(",") if c.strip()]
    gold_path = Path(plan_in["gold"][0]["path"])
    supp_paths = [Path(e["path"]) for e in (plan_in.get("supplementary") or []) if e.get("path")]
    log(f"stage 1 plan: {stage1 / 'plan.json'}")
    log(f"  gold: {gold_path.name} | supplementary: {len(supp_paths)} table(s)")

    report: dict = {
        "task": name,
        "objective": (plan_in.get("task") or {}).get("objective", ""),
        "labels": None,
        "supplementary": [],
        "preprocess": {"normalize": args.normalize, "target_sum": args.target_sum, "n_hvg": args.n_hvg},
        "outputs": {},
        "warnings": [],
    }

    log("1/5 subset the gold (labels)")
    labels, label_shape = subset_labels(gold_path, targets, controls, args.condition_column)
    report["labels"] = label_shape
    log(f"  kept {len(labels):,} cells: {label_shape['condition_counts']}")
    if not len(labels):
        raise SystemExit("the subset is empty; check --targets/--controls/--condition-column")
    if not any(str(t).lower() == "all" for t in targets) and any(
        t not in label_shape["condition_counts"] for t in targets
    ):
        report["warnings"].append(
            f"target not present in the labels: {[t for t in targets if t not in label_shape['condition_counts']]}"
        )
    perturbations = [
        name for name in label_shape["condition_counts"] if not is_control_name(name, controls)
    ]
    report["n_perturbations"] = len(perturbations)
    if len(perturbations) < 2:
        report["warnings"].append(
            f"only {len(perturbations)} perturbation(s) in the subset ({perturbations}). "
            f"The backend splits *by perturbation*, so validation/test would be empty and "
            f"training fails -- pass --targets all (every perturbation + controls) to make "
            f"a trainable screen."
        )
        log(
            f"WARNING: only {len(perturbations)} perturbation(s); stage 2 will fail at the "
            f"split. Re-run with --targets all."
        )
    wanted = set(labels["barcode"])

    log("2/5 subset each supplementary table to those cells, keep the gene panel")
    matched: list[tuple[Path, pd.DataFrame]] = []
    separate: list[Path] = []
    for path in supp_paths:
        try:
            matrix = read_counts_for_cells(path, wanted)  # genes x cells_subset
            matched.append((path, matrix))
            log(f"  {path.name}: gold's cells -> {matrix.shape[1]:,} cells")
        except SystemExit:
            # A different arm of the screen (its own barcodes): keep its cells,
            # cut it on the shared gene panel instead.
            separate.append(path)
            log(f"  {path.name}: no shared barcode -> keep its own cells, cut on the panel")
    if not matched:
        raise SystemExit("no supplementary table measures the gold's cells")

    log("3/5 HVG on the measurement half (the table that shares the most cells)")
    primary = max(matched, key=lambda item: item[1].shape[1])
    primary_frame = primary[1].T.astype(np.float64)  # cells x genes
    if args.normalize == "log1p_cpm":
        primary_frame = log1p_cpm(primary_frame, args.target_sum)
    variance = primary_frame.var(axis=0).sort_values(ascending=False)
    panel = list(variance.head(args.n_hvg).index)
    # A perturbation is only trainable if the gene it names is a panel column:
    # the backend encodes a perturbation by looking it up in the panel and drops
    # one it cannot find, which silently removes cells. HVG is a variance rule
    # and does not care -- `CD86` and `ATF2` were dropped here and their cells
    # vanished from the screen. Keep every perturbed gene.
    forced_genes = [gene for gene in perturbations if gene not in set(panel)]
    panel = list(dict.fromkeys([*panel, *forced_genes]))
    report["hvg"] = {
        "source": str(primary[0]),
        "n_selected": len(panel),
        "forced_perturbation_genes": forced_genes,
        "top_genes": list(variance.head(25).index),
    }
    log(
        f"  panel: {len(panel)} genes from {primary[0].name}"
        + (f" (+{len(forced_genes)} perturbed gene(s) HVG dropped: {forced_genes})" if forced_genes else "")
    )

    log("4/5 z-score per gene across cells, clip negatives")
    supp_outputs: list[dict] = []
    prepared_tables: list[tuple[Path, pd.DataFrame]] = [
        (path, matrix.T.astype(np.float64)) for path, matrix in matched
    ]
    for path in separate:
        frame = read_counts_panel(path, panel, args.supp_max_cells)
        if frame is None:
            report["warnings"].append(f"{path.name}: shares <10 panel genes; skipped")
            continue
        prepared_tables.append((path, frame.astype(np.float64)))
    for index, (path, frame) in enumerate(prepared_tables, start=1):
        frame = frame.reindex(columns=panel)
        if args.normalize == "log1p_cpm":
            frame = log1p_cpm(frame, args.target_sum)
        prepared = zscore_clip(frame).copy()
        prepared.insert(0, "barcode", prepared.index)
        written = out_dir / f"supp_{index}_{path.name.split('.')[0]}.csv.gz"
        prepared.to_csv(written, index=False, compression="gzip")
        values = prepared.iloc[:, 1:].to_numpy(dtype=np.float64, copy=False)
        supp_outputs.append(
            {
                "source": str(path),
                "written": str(written),
                "cells": int(prepared.shape[0]),
                "genes": int(prepared.shape[1] - 1),
                "value_max": float(values.max()),
                "value_mean": float(values.mean()),
            }
        )
        log(f"  -> {written.name} ({prepared.shape[0]:,} cells x {prepared.shape[1] - 1} genes)")
    report["supplementary"] = supp_outputs

    log("5/5 write the labels and the stage-2 plan")
    gold_out = out_dir / "gold_subset.csv.gz"
    labels[["barcode", "condition"]].to_csv(gold_out, index=False, compression="gzip")
    plan = {
        "version": 1,
        "task": {
            "objective": report["objective"],
            "target_column": "condition",
            "target_source": "human",
            "target_confidence": "declared",
            "task_type": "classification",
            "backend": "perturbation",
        },
        "gold": [
            {
                "path": str(gold_out),
                "role": "gold",
                "reason": f"{name}: {', '.join(targets)} + {', '.join(controls)} labels ({len(labels):,} cells)",
            }
        ],
        "supplementary": [
            {
                "path": entry["written"],
                "role": "supplementary",
                "reason": (
                    f"{entry['genes']}-gene panel, log1p-CPM + per-gene z-score + "
                    f"clip, {entry['cells']:,} cells (from {Path(entry['source']).name})"
                ),
            }
            for entry in supp_outputs
        ],
        "unusable": [],
    }
    (out_dir / "plan.json").write_text(json.dumps(plan, indent=2), encoding="utf-8")
    report["outputs"] = {
        "gold_subset": str(gold_out),
        "plan": str(out_dir / "plan.json"),
        "supplementary": [entry["written"] for entry in supp_outputs],
    }
    (out_dir / "subset_report.json").write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    (out_dir / "subset_report.md").write_text(render_report(report), encoding="utf-8")
    log(f"done -> {out_dir}")
    return 0


def render_report(report: dict) -> str:
    lines = [
        f"# Subset report — {report['task']}",
        "",
        f"- objective: {report['objective']}",
        f"- stage-1 gold: `{report['labels']['path']}`",
        f"- condition column: `{report['labels']['condition_column']}` "
        f"(barcodes in `{report['labels']['barcode_column']}`)",
        "",
        "## Subset (gold)",
        "",
        f"- cells in the label table: {report['labels']['cells_total']:,}",
        f"- cells kept: {report['labels']['cells_kept']:,}",
    ]
    for name, count in sorted(report["labels"]["condition_counts"].items()):
        lines.append(f"  - `{name}`: {count:,}")
    lines += [
        "",
        "## Pre-processing",
        "",
        f"1. normalise `{report['preprocess']['normalize']}` (target_sum {report['preprocess']['target_sum']:g})",
        f"2. HVG: top {report['preprocess']['n_hvg']} by variance "
        f"-> {report['hvg']['n_selected']} (from `{Path(report['hvg']['source']).name}`)",
        "3. z-score each gene across all cells",
        "4. clip negatives to zero",
        "",
        "Top HVG: " + ", ".join(report["hvg"]["top_genes"][:15]),
        "",
        "## Training inputs (stage 2)",
        "",
        f"- `{report['outputs']['gold_subset']}` — labels",
    ]
    for entry in report["supplementary"]:
        lines.append(
            f"- `{entry['written']}` — {entry['cells']:,} cells x {entry['genes']} genes "
            f"(from `{Path(entry['source']).name}`)"
        )
    lines += [
        f"- `{report['outputs']['plan']}` — gold + supplementary, backend joins on barcode",
        "",
    ]
    if report.get("warnings"):
        lines += ["## Warnings", ""] + [f"- {w}" for w in report["warnings"]] + [""]
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
