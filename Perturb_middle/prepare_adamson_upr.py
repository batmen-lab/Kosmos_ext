#!/usr/bin/env python
"""Stage 2 for the Adamson UPR question: the stage-1 package -> training tables.

Question: *In K562 cells, can the transcriptional response to CRISPRi knockdown
of unfolded protein response (UPR) genes not seen in training be predicted from
the other knockdowns, using single-cell profiles of K562 cells from independent
Perturb-seq experiments as supplementary evidence?*

Stage 1 (`artifacts/dataprep/adamson-auto`) fetched GSE90546 (Adamson et al.
2016) and, as supplementary evidence, GSE90063's K562 wild-type cells (Dixit et
al. 2016). This script makes the human decisions the stage-1 review cannot,
and writes the stage-3 inputs:

    gold.parquet                   condition (`GENE+ctrl` / `ctrl`) + genes
    supp_dixit_k562_wt.parquet     the same genes, unperturbed K562 cells
    *_counts.parquet               raw counts of both tables (same rows/panel) +
                                   `total_counts`, for count-based arms (PerturbNet)
    plan.json                      gold + supplementary + study (split, panel)
    prepare_report.md / .json      what was kept and why

Decisions (each is a flag):

  * gold = **GSM2406681_10X010 only** ("UPR Perturb-seq experiment"). The series'
    other samples are different experiments -- 10X001 a pilot TF screen, 10X005
    an epistasis screen -- and are not mixed into the gold.
  * a cell is kept when its guide call is a named guide (not `*`, an unassigned
    call), has `good coverage` and exactly one guide (`number of cells == 1`,
    which counts guides). A multi-guide cell's table row names only the
    dominant guide, so its label is ambiguous.
  * `63(mod)_pBA580` and `Gal4-4(mod)_pBA582` are the non-targeting controls ->
    `ctrl`; `GENE_pDSxxx` -> `GENE+ctrl` (GEARS' format; a gene's two guides are
    one perturbation, as in GEARS).
  * counts -> log1p(CPM, 1e4) for both tables; panel = the top `--n-hvg` (5,000,
    GEARS) genes by variance over the kept gold cells, plus every measured
    perturbed gene.
  * supplementary = Dixit's K562 wild-type cells (independent lab and run):
    aligned to the panel by Ensembl id, an unmeasured gene filled with the gold
    control mean; no condition column, so stage 3 uses every cell.
  * study: GEARS' simulation split -- 25% of the perturbed genes held out for
    test, 20% of the rest for validation (GEARS: 10%; with 58 non-test genes
    that is 6 perturbations, too few to select an epoch on); the panel is used
    whole; training batch size 128 (GEARS: 32).

Run:

    .venv/bin/python Perturb_middle/prepare_adamson_upr.py
    # then stage 3:
    .venv/bin/python -m kosmos.cli.main experiment "<question>" \\
      --data-plan Perturb_middle/output_adamson_upr/plan.json \\
      --task perturbation --max-epochs 20 --patience 0 --yes
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

STAGE1 = "artifacts/dataprep/adamson-auto"
OUT = "Perturb_middle/output_adamson_upr"
SAMPLE = "GSM2406681_10X010"
CONTROLS = ("63(mod)_pBA580", "Gal4-4(mod)_pBA582")
SUPP_NAME = "GSE90063_k562_umi_wt.txt.gz"
TOTAL_COLUMN = "total_counts"


def log(message: str) -> None:
    print(f"[adamson-upr] {message}", flush=True)


def log1p_cpm(counts: np.ndarray, target_sum: float) -> np.ndarray:
    totals = counts.sum(axis=1, keepdims=True)
    totals[totals == 0] = 1.0
    return np.log1p(counts / totals * target_sum).astype(np.float32)


def stage1_paths(stage1: Path) -> tuple[Path, Path]:
    """The sample directory of the gold and the supplementary file, from the plan."""
    plan = json.loads((stage1 / "plan.json").read_text(encoding="utf-8"))
    paths = [Path(e["path"]) for role in ("gold", "supplementary", "unusable") for e in plan.get(role) or []]
    sample_dir = next((p.parent for p in paths if p.name.startswith(SAMPLE)), None)
    supp = next((p for p in paths if p.name == SUPP_NAME), None)
    if sample_dir is None or supp is None:
        raise SystemExit(f"{stage1}/plan.json does not name {SAMPLE}* and {SUPP_NAME}")
    return sample_dir, supp


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--stage1", default=STAGE1)
    parser.add_argument("--out", default=OUT)
    parser.add_argument("--sample", default=SAMPLE, help="the GEO sample that is the UPR screen")
    parser.add_argument("--n-hvg", type=int, default=5000)
    parser.add_argument("--target-sum", type=float, default=1e4)
    parser.add_argument("--keep-multi-guide", action="store_true", help="keep cells with >1 guide (dominant guide label)")
    parser.add_argument("--test-fraction", type=float, default=0.25)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--batch-size", type=int, default=128, help="stage-3 training batch size")
    args = parser.parse_args(argv)

    from datafetcher.singlecell import _names, _read_matrix_market

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    sample_dir, supp_path = stage1_paths(Path(args.stage1))
    prefix = sample_dir / args.sample
    report: dict = {"stage1": args.stage1, "sample": args.sample, "steps": []}

    # -- gold: matrix + labels ------------------------------------------------
    log(f"reading {prefix.name}_matrix.mtx.txt.gz")
    matrix = _read_matrix_market(Path(f"{prefix}_matrix.mtx.txt.gz"))
    barcodes, _ = _names(Path(f"{prefix}_barcodes.tsv.gz"))
    gene_ids, symbols = _names(Path(f"{prefix}_genes.tsv.gz"))
    if matrix.shape == (len(gene_ids), len(barcodes)):
        matrix = matrix.T.tocsr()
    log(f"  {matrix.shape[0]:,} cells x {matrix.shape[1]:,} genes")

    identities = pd.read_csv(f"{prefix}_cell_identities.csv.gz")
    identities = identities.set_index("cell BC").reindex(barcodes)
    steps = report["steps"]
    steps.append(("cells in the matrix", len(barcodes)))
    keep = identities["guide identity"].notna().to_numpy()
    steps.append(("with a guide call", int(keep.sum())))
    keep &= identities["good coverage"].astype("boolean").fillna(False).to_numpy(dtype=bool)
    steps.append(("with good coverage", int(keep.sum())))
    if not args.keep_multi_guide:
        keep &= (identities["number of cells"] == 1).to_numpy()
        steps.append(("with exactly one guide", int(keep.sum())))

    guides = identities["guide identity"].astype(str).to_numpy()
    # A guide is `GENE_pXXnnn` or one of the two non-targeting controls; `*`
    # (an unassigned call, 27 cells) is not a perturbation identity.
    named = pd.Series(guides).str.match(r"^.+_p[A-Z]{2}\d+$").to_numpy() | np.isin(guides, CONTROLS)
    keep &= named
    steps.append(("with a named guide (not `*`)", int(keep.sum())))
    condition = np.where(
        np.isin(guides, CONTROLS),
        "ctrl",
        np.char.add(np.array([g.rsplit("_", 1)[0] for g in guides], dtype=object).astype(str), "+ctrl"),
    )
    rows = np.flatnonzero(keep)
    counts = matrix[rows].toarray().astype(np.float32)
    condition = condition[rows]
    values = log1p_cpm(counts, args.target_sum)
    del counts

    # -- panel ------------------------------------------------------------------
    unique = ~pd.Index(symbols).duplicated()
    perturbed = sorted({c.split("+")[0] for c in condition if c != "ctrl"})
    variance = values.var(axis=0)
    variance[~unique] = -1.0
    ranked = [int(i) for i in np.argsort(-variance) if unique[i]][: args.n_hvg]
    by_symbol = {symbols[i]: i for i in range(len(symbols)) if unique[i]}
    forced = [by_symbol[g] for g in perturbed if g in by_symbol and by_symbol[g] not in set(ranked)]
    panel = sorted(set(ranked) | set(forced))
    genes = [symbols[i] for i in panel]
    panel_ids = [gene_ids[i] for i in panel]
    unmeasured = [g for g in perturbed if g not in by_symbol]
    log(f"panel: {len(panel):,} genes ({len(forced)} perturbed gene(s) added beyond the top {args.n_hvg})")

    gold = pd.DataFrame(values[:, panel], columns=genes)
    gold.insert(0, "cell_type", "K562")
    gold.insert(0, "condition", condition)
    gold_out = out / "gold.parquet"
    gold.to_parquet(gold_out, index=False)
    control = condition == "ctrl"
    control_mean = gold.loc[control, genes].to_numpy().mean(axis=0)
    per_condition = pd.Series(condition).value_counts()
    log(f"  -> {gold_out.name}: {len(gold):,} cells ({int(control.sum()):,} controls), "
        f"{len(perturbed)} perturbed genes")
    # Raw counts for count-based arms (PerturbNet's scVI): the same rows and panel
    # as gold.parquet, plus each cell's whole-transcriptome total -- the CPM
    # denominator -- so a predicted cell can be normalised exactly as the gold was.
    kept = matrix[rows]
    gold_counts = pd.DataFrame(kept[:, panel].toarray().astype(np.float32), columns=genes)
    gold_counts[TOTAL_COLUMN] = np.asarray(kept.sum(axis=1), dtype=np.float64).ravel()
    gold_counts_out = out / "gold_counts.parquet"
    gold_counts.to_parquet(gold_counts_out, index=False)
    del kept, gold_counts
    log(f"  -> {gold_counts_out.name}: raw counts on the panel + `{TOTAL_COLUMN}`")

    # -- supplementary: Dixit K562 wild type ----------------------------------------
    log(f"reading {supp_path.name}")
    supp = pd.read_csv(supp_path, sep="\t", index_col=0)
    supp_ids = [str(name).split("_")[1] if str(name).startswith("hg19_") else str(name) for name in supp.index]
    supp_counts = supp.to_numpy(dtype=np.float32).T                 # cells x genes
    supp_values = log1p_cpm(supp_counts, args.target_sum)
    position = {gene_id: index for index, gene_id in enumerate(supp_ids)}
    aligned = np.tile(control_mean, (supp_values.shape[0], 1)).astype(np.float32)
    shared = [(column, position[gene_id]) for column, gene_id in enumerate(panel_ids) if gene_id in position]
    if shared:
        columns, source = zip(*shared)
        aligned[:, list(columns)] = supp_values[:, list(source)]
    supp_out = out / "supp_dixit_k562_wt.parquet"
    pd.DataFrame(aligned, columns=genes).to_parquet(supp_out, index=False)
    log(f"  -> {supp_out.name}: {aligned.shape[0]:,} cells; {len(shared):,}/{len(genes):,} panel genes measured")
    # an unmeasured panel gene has no count; 0 here, and the report says how many
    supp_raw = np.zeros((supp_counts.shape[0], len(genes)), dtype=np.float32)
    if shared:
        supp_raw[:, list(columns)] = supp_counts[:, list(source)]
    supp_raw = pd.DataFrame(supp_raw, columns=genes)
    supp_raw[TOTAL_COLUMN] = supp_counts.sum(axis=1, dtype=np.float64)
    supp_counts_out = out / "supp_dixit_k562_wt_counts.parquet"
    supp_raw.to_parquet(supp_counts_out, index=False)
    del supp_raw
    log(f"  -> {supp_counts_out.name}: raw counts on the panel + `{TOTAL_COLUMN}`")

    # -- plan + report ---------------------------------------------------------------
    stage1_plan = json.loads((Path(args.stage1) / "plan.json").read_text(encoding="utf-8"))
    study = {
        "question": stage1_plan["task"]["objective"],
        "dataset": f"GSE90546 {args.sample} (Adamson 2016 UPR Perturb-seq)",
        "held_out_unit": "perturbed genes never seen in training",
        "condition_column": "condition",
        "split_mode": "unseen_single",
        "test_fraction": args.test_fraction,
        "validation_fraction": args.validation_fraction,
        "batch_size": args.batch_size,
        "max_panel_genes": 0,
        "normalize": f"log1p(CPM {args.target_sum:g})",
        "target_sum": args.target_sum,
        # raw counts beside each normalised table (same rows, same panel)
        "counts": {
            "total_column": TOTAL_COLUMN,
            "gold": str(gold_counts_out),
            "supplementary": {str(supp_out): str(supp_counts_out)},
        },
    }
    plan = {
        "version": 1,
        "task": {**stage1_plan["task"], "target_column": "condition", "backend": "perturbation"},
        "study": study,
        "gold": [{"path": str(gold_out), "role": "gold", "reason": f"{args.sample}: single-guide, good-coverage cells"}],
        "supplementary": [
            {
                "path": str(supp_out),
                "role": "supplementary",
                "reason": f"GSE90063 K562 wild type (Dixit 2016): {aligned.shape[0]:,} cells, "
                f"{len(shared):,} of {len(genes):,} panel genes measured",
            }
        ],
        "unusable": [],
    }
    (out / "plan.json").write_text(json.dumps(plan, indent=2), encoding="utf-8")
    report.update(
        {
            "study": study,
            "gold": {
                "written": str(gold_out),
                "cells": len(gold),
                "control_cells": int(control.sum()),
                "perturbed_genes": perturbed,
                "perturbed_genes_not_measured": unmeasured,
                "cells_per_perturbation": {"min": int(per_condition.drop("ctrl").min()), "median": float(per_condition.drop("ctrl").median())},
                "panel_genes": len(genes),
                "panel_forced_perturbed_genes": len(forced),
            },
            "supplementary": {
                "source": str(supp_path),
                "written": str(supp_out),
                "cells": int(aligned.shape[0]),
                "panel_genes_measured": len(shared),
            },
        }
    )
    (out / "prepare_report.json").write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    (out / "prepare_report.md").write_text(render(report), encoding="utf-8")
    log(f"done -> {out / 'plan.json'}")
    return 0


def render(report: dict) -> str:
    gold, supp, study = report["gold"], report["supplementary"], report["study"]
    lines = [
        "# Prepare report — Adamson UPR (GSE90546)",
        "",
        f"**Question:** {study['question']}",
        "",
        "## Gold cell filter",
        "",
        "| step | cells |",
        "|---|---|",
        *[f"| {name} | {count:,} |" for name, count in report["steps"]],
        "",
        f"- {gold['cells']:,} cells, {gold['control_cells']:,} controls (`63(mod)`, `Gal4-4(mod)`)",
        f"- {len(gold['perturbed_genes'])} perturbed genes; cells per perturbation: min "
        f"{gold['cells_per_perturbation']['min']}, median {gold['cells_per_perturbation']['median']:.0f}",
        f"- perturbed genes not in the measured genes: {gold['perturbed_genes_not_measured'] or 'none'}",
        f"- panel: {gold['panel_genes']:,} genes (top HVG + {gold['panel_forced_perturbed_genes']} perturbed genes); "
        f"{study['normalize']}",
        "",
        "## Supplementary",
        "",
        f"- `{supp['written']}` ← `{supp['source']}`",
        f"- {supp['cells']:,} unperturbed K562 cells (independent lab/run); {supp['panel_genes_measured']:,} of "
        f"{gold['panel_genes']:,} panel genes measured" + (", the rest = gold control mean" if supp['panel_genes_measured'] < gold['panel_genes'] else ""),
        "",
        "## Study",
        "",
        f"- held-out unit: {study['held_out_unit']} (`{study['split_mode']}`), test "
        f"{study['test_fraction']:.0%}, validation {study['validation_fraction']:.0%} of the rest",
        f"- training batch size {study.get('batch_size', 32)}",
        "",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
