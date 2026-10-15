#!/usr/bin/env python
"""Stage 2 for the Jurkat TCR question: the survey's datasets -> training tables.

Question: *In Jurkat T cells, can the transcriptional response to CRISPR knockout
of T cell receptor signaling genes not seen in training be predicted from a small
labeled screen, using large single-cell profiles of Jurkat cells from other
CRISPR screens, with their labels ignored, as supplementary evidence?*

Stage 1 (`artifacts/dataprep/jurkat-tcr/survey.json`, nothing downloaded) chose:

  gold  GSE92872 -- Datlinger 2017 CROP-seq, Jurkat TCR stimulation (Drop-seq).
        One dense genes x cells CSV whose first five rows are per-cell metadata:
        condition, replicate, cell, grna, gene. `gene == CTRL` = non-targeting.
  supp  GSE249595 -- genome-scale CRISPRi Perturb-seq in Jurkat, per channel
        (`GSM...` transcriptome mtx). Unfiltered: 6.8M barcodes per channel, so
        cells are called by UMI count. Its labels are not used.

Decisions (each is a flag):

  * gold condition(s): `stimulated` only by default -- TCR knockouts act through
    stimulation; `--conditions stimulated,unstimulated` keeps both, as context.
  * `DHODH`, `MVD`, `TUBB` are dropped: the paper's essential-gene efficacy
    controls (7-28 cells each), not TCR signaling genes.
  * controls: `CTRL` guides -> `ctrl`; a target gene `G` -> `G+ctrl` (GEARS).
  * supplementary: every survey candidate (its supplementary samples, and the gold
    candidates not used as gold, labels ignored) is downloaded within a budget,
    cells are called at >= 1,000 UMIs (GEO matrices here are unfiltered:
    GSE249595 has 6.8M barcodes per channel), and its expression is compared with
    the gold controls (Pearson of gene means, log1p-CPM). Samples at or above
    `--min-similarity` are pooled into one supplementary table with a
    `supp_source` column; the rest are reported with why. A series is judged on
    its first sample, and at most `--max-samples-per-series` of it are pooled.
  * counts -> log1p(CPM 1e4); panel = the top `--n-hvg` genes by variance over
    the kept gold cells *among the genes the supplementary measures*, plus every
    perturbed gene; an unmeasured panel gene of the supp is the gold control mean.
  * study: unseen-gene split, test 25%, validation 20% of the rest, batch 32
    (GEARS' default; the gold is small).

Writes `gold.parquet`, `gold_counts.parquet`, `supp_*.parquet`, `supp_*_counts.parquet`,
`plan.json` (+ `study`) and `prepare_report.md/.json`, whose table puts the
survey's estimates next to what the files actually hold.

    .venv/bin/python Perturb_middle/prepare_jurkat_tcr.py
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

SURVEY = "artifacts/dataprep/jurkat-tcr/survey.json"
OUT = "Perturb_middle/output_jurkat_tcr"
GOLD_URL = (
    "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE92nnn/GSE92872/suppl/"
    "GSE92872_CROP-seq_Jurkat_TCR.digital_expression.csv.gz"
)
GOLD_FILE = "data/fetched/geo/GSE92872/GSE92872_CROP-seq_Jurkat_TCR.digital_expression.csv.gz"
GOLD_SAMPLE = "GSE92872"
#: one directory per series and sample, files as GEO names them
SUPP_DIR = "data/fetched/samples"
EXCLUDED_TARGETS = ("DHODH", "MVD", "TUBB")
CONTROL_GENE = "CTRL"
TOTAL_COLUMN = "total_counts"


def log(message: str) -> None:
    print(f"[jurkat-tcr] {message}", flush=True)


def log1p_cpm(counts: np.ndarray, totals: np.ndarray, target_sum: float) -> np.ndarray:
    totals = np.where(totals > 0, totals, 1.0)
    return np.log1p(counts / totals[:, None] * target_sum).astype(np.float32)


def download(url: str, path: Path) -> None:
    if path.exists() and path.stat().st_size > 0:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    log(f"downloading {url.rsplit('/', 1)[-1]}")
    urllib.request.urlretrieve(url, path)


# -- gold: Datlinger's digital expression ------------------------------------------


def read_gold(path: Path) -> tuple[pd.DataFrame, np.ndarray, list[str]]:
    """Per-cell metadata (5 header rows) and a cells x genes count matrix."""
    raw = pd.read_csv(path, header=None, low_memory=False)
    meta = raw.iloc[:5, 1:].T.reset_index(drop=True)
    meta.columns = [str(name) for name in raw.iloc[:5, 0]]
    body = raw.iloc[5:]
    body = body[body.iloc[:, 0].astype(str) != "GENE"]  # the separator row
    genes = body.iloc[:, 0].astype(str).tolist()
    counts = body.iloc[:, 1:].to_numpy(dtype=np.float32).T
    return meta, counts, genes


# -- supplementary: the survey's candidates, scored against the gold ------------------

#: Matrices that are not expression: guide/label/hashtag counts of a screen.
_NOT_EXPRESSION = ("guide", "label", "hto", "adt", "crispr", "gdo", "hash")


def candidate_files(row: dict) -> list[tuple[str, str]]:
    """(file name, url) of a survey row's expression files."""
    if row["series"].startswith("hf://"):
        repo = row["series"].split("://", 1)[1]
        return [
            (f["name"], f"https://huggingface.co/datasets/{repo}/resolve/main/{f['name']}")
            for f in row["files"]
            if str(f["name"]).lower().endswith((".h5", ".h5ad"))
        ]
    gsm = row["sample"]
    base = f"https://ftp.ncbi.nlm.nih.gov/geo/samples/{gsm[:-3]}nnn/{gsm}/suppl/"
    names = [f["name"] for f in row["files"]]
    expression = [n for n in names if not any(w in n.lower() for w in _NOT_EXPRESSION)]
    # a sample with several triplets: prefer the one named transcriptome/GEX
    preferred = [n for n in expression if re.search(r"transcriptome|gex|mrna|rna", n, re.I)]
    chosen = preferred if any(_is_matrix(n) for n in preferred) else expression
    return [(n, base + n) for n in chosen if re.search(r"matrix|barcode|feature|gene|\.h5", n, re.I)]


def _is_matrix(name: str) -> bool:
    return bool(re.search(r"matrix\.mtx|\.h5", name, re.I))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--survey", default=SURVEY)
    parser.add_argument("--out", default=OUT)
    parser.add_argument("--conditions", default="stimulated", help="comma-separated gold condition(s) to keep")
    parser.add_argument("--exclude-targets", default=",".join(EXCLUDED_TARGETS))
    parser.add_argument("--min-umis", type=int, default=1000, help="supp barcode -> cell threshold")
    parser.add_argument("--min-similarity", type=float, default=0.6,
                        help="pool a supp sample when its gene means correlate with the gold controls' at least this much")
    parser.add_argument("--max-samples-per-series", type=int, default=2)
    parser.add_argument("--supp-budget-gb", type=float, default=3.0, help="download budget for supp candidates")
    parser.add_argument("--n-hvg", type=int, default=5000)
    parser.add_argument("--target-sum", type=float, default=1e4)
    parser.add_argument("--test-fraction", type=float, default=0.25)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args(argv)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    survey = json.loads(Path(args.survey).read_text(encoding="utf-8"))
    report: dict = {"survey": args.survey, "decisions": vars(args), "steps": []}
    steps = report["steps"]

    # -- gold ------------------------------------------------------------------------
    download(GOLD_URL, Path(GOLD_FILE))
    meta, counts, genes = read_gold(Path(GOLD_FILE))
    steps.append(("gold cells in the file", len(meta)))
    conditions = [c.strip() for c in args.conditions.split(",") if c.strip()]
    keep = meta["condition"].isin(conditions).to_numpy()
    steps.append((f"in condition(s) {', '.join(conditions)}", int(keep.sum())))
    excluded = [g.strip() for g in args.exclude_targets.split(",") if g.strip()]
    keep &= ~meta["gene"].isin(excluded).to_numpy()
    steps.append((f"without {', '.join(excluded)}", int(keep.sum())))
    meta, counts = meta[keep].reset_index(drop=True), counts[keep]
    condition = np.where(meta["gene"] == CONTROL_GENE, "ctrl", meta["gene"].astype(str) + "+ctrl")
    totals = counts.sum(axis=1).astype(np.float64)

    # -- supplementary: download, score against the gold controls, pool -----------------
    from kosmos.ppi.perturbation.dataprep.load import call_cells, gene_means, load_sample, similarity

    is_control = condition == "ctrl"
    reference: dict[str, float] = {}
    for gene, value in zip(genes, gene_means(counts[is_control], args.target_sum)):
        reference.setdefault(gene, float(value))
    candidates = [r for r in survey["supplementary"] + survey["gold"] if r["sample"] != GOLD_SAMPLE]
    budget = args.supp_budget_gb * 1e9
    spent, per_series_kept, judged_series = 0.0, {}, {}
    kept, scored = [], []
    for row in candidates:
        series = row["series"]
        entry = {"sample": row["sample"], "series": series, "title": row["sample_title"][:70]}
        if judged_series.get(series) is False:
            scored.append({**entry, "decision": "skipped: its series failed on an earlier sample"})
            continue
        if per_series_kept.get(series, 0) >= args.max_samples_per_series:
            scored.append({**entry, "decision": f"skipped: {args.max_samples_per_series} sample(s) of this series already pooled"})
            continue
        files = candidate_files(row)
        size = sum(int(f.get("bytes") or 0) for f in row["files"] if f["name"] in {n for n, _ in files})
        if not files:
            scored.append({**entry, "decision": "skipped: no expression files"})
            continue
        if spent + size > budget:
            scored.append({**entry, "decision": f"skipped: over the {args.supp_budget_gb:g} GB budget ({size / 1e9:.2f} GB)"})
            continue
        directory = Path(SUPP_DIR) / series.split("://", 1)[1].replace("/", "__") / row["sample"].replace("/", "__")
        paths = []
        for name, url in files:
            download(url, directory / name)
            paths.append(directory / name)
        spent += size
        try:
            sample, barcodes_before = call_cells(load_sample(paths), args.min_umis)
        except Exception as error:  # noqa: BLE001 - an unreadable candidate is reported, not fatal
            scored.append({**entry, "decision": f"unreadable: {type(error).__name__}: {str(error)[:120]}"})
            continue
        stats = similarity(reference, sample, args.target_sum)
        accept = stats["pearson"] >= args.min_similarity and sample.n_cells > 0
        judged_series.setdefault(series, accept)
        scored.append(
            {
                **entry,
                "barcodes": barcodes_before,
                "cells": sample.n_cells,
                **{k: round(v, 3) if isinstance(v, float) else v for k, v in stats.items()},
                "decision": "pooled" if accept else f"dropped: Pearson {stats['pearson']:.2f} < {args.min_similarity}",
            }
        )
        log(f"{row['sample']}: {sample.n_cells:,} cells, Pearson {stats['pearson']:.2f}, "
            f"ribosomal {stats['ribosomal_fraction']:.1%} -> {scored[-1]['decision']}")
        if accept:
            kept.append(sample)
            per_series_kept[series] = per_series_kept.get(series, 0) + 1
    if not kept:
        raise SystemExit("no supplementary candidate is similar enough to the gold; see the log")
    supp_cells = sum(sample.n_cells for sample in kept)

    # -- panel ---------------------------------------------------------------------------
    gene_index = pd.Index(genes)
    unique = ~gene_index.duplicated()
    measured_by_supp = set().union(*(set(sample.genes) for sample in kept))
    values = log1p_cpm(counts, totals, args.target_sum)
    variance = values.var(axis=0)
    eligible = unique & gene_index.isin(list(measured_by_supp))
    ranked = [int(i) for i in np.argsort(-variance) if eligible[i]][: args.n_hvg]
    perturbed = sorted({c.split("+")[0] for c in condition if c != "ctrl"})
    by_symbol = {genes[i]: i for i in range(len(genes)) if unique[i]}
    forced = [by_symbol[g] for g in perturbed if g in by_symbol and by_symbol[g] not in set(ranked)]
    panel = sorted(set(ranked) | set(forced))
    panel_genes = [genes[i] for i in panel]
    log(f"panel: {len(panel):,} genes ({len(forced)} perturbed gene(s) added beyond the top {args.n_hvg})")

    gold = pd.DataFrame(values[:, panel], columns=panel_genes)
    gold.insert(0, "replicate", meta["replicate"].astype(str).to_numpy())
    gold.insert(0, "cell_type", "Jurkat_" + meta["condition"].astype(str).to_numpy())
    gold.insert(0, "condition", condition)
    gold.to_parquet(out / "gold.parquet", index=False)
    gold_counts = pd.DataFrame(counts[:, panel], columns=panel_genes)
    gold_counts[TOTAL_COLUMN] = totals
    gold_counts.to_parquet(out / "gold_counts.parquet", index=False)
    control_mean = values[condition == "ctrl"][:, panel].mean(axis=0)

    supp_name = "supp_jurkat_pooled"
    value_blocks, count_blocks, sources, coverage = [], [], [], {}
    for sample in kept:
        position: dict[str, int] = {}
        for index, symbol in enumerate(sample.genes):
            position.setdefault(symbol, index)
        shared = [(column, position[g]) for column, g in enumerate(panel_genes) if g in position]
        columns, source = (list(x) for x in zip(*shared))
        sample_totals = np.asarray(sample.counts.sum(axis=1)).ravel().astype(np.float64)
        picked = sample.counts[:, source].toarray().astype(np.float32)
        block = np.tile(control_mean, (sample.n_cells, 1)).astype(np.float32)
        block[:, columns] = log1p_cpm(picked, sample_totals, args.target_sum)
        raw = np.zeros((sample.n_cells, len(panel_genes)), dtype=np.float32)
        raw[:, columns] = picked
        value_blocks.append(block)
        count_blocks.append((raw, sample_totals))
        sources += [sample.source] * sample.n_cells
        coverage[sample.source] = len(shared)
    supp_table = pd.DataFrame(np.vstack(value_blocks), columns=panel_genes)
    supp_table["supp_source"] = sources
    supp_table.to_parquet(out / f"{supp_name}.parquet", index=False)
    supp_raw = pd.DataFrame(np.vstack([raw for raw, _ in count_blocks]), columns=panel_genes)
    supp_raw[TOTAL_COLUMN] = np.concatenate([t for _, t in count_blocks])
    supp_raw.to_parquet(out / f"{supp_name}_counts.parquet", index=False)

    # -- plan + report -------------------------------------------------------------------
    study = {
        "question": survey["question"],
        "dataset": "GSE92872 (Datlinger 2017 CROP-seq, Jurkat TCR)",
        "held_out_unit": "perturbed genes never seen in training",
        "condition_column": "condition",
        "split_mode": "unseen_single",
        "test_fraction": args.test_fraction,
        "validation_fraction": args.validation_fraction,
        "batch_size": args.batch_size,
        "max_panel_genes": 0,
        "normalize": f"log1p(CPM {args.target_sum:g})",
        "target_sum": args.target_sum,
        "counts": {
            "total_column": TOTAL_COLUMN,
            "gold": str(out / "gold_counts.parquet"),
            "supplementary": {str(out / f"{supp_name}.parquet"): str(out / f"{supp_name}_counts.parquet")},
        },
    }
    plan = {
        "version": 1,
        "task": {"objective": survey["question"], "target_column": "condition", "backend": "perturbation"},
        "study": study,
        "gold": [{"path": str(out / "gold.parquet"), "role": "gold", "reason": "GSE92872 CROP-seq Jurkat TCR"}],
        "supplementary": [
            {
                "path": str(out / f"{supp_name}.parquet"),
                "role": "supplementary",
                "reason": f"{len(kept)} Jurkat sample(s) pooled by similarity to the gold controls "
                f"({', '.join(sample.source for sample in kept)}), labels ignored",
            }
        ],
        "unusable": [],
    }
    (out / "plan.json").write_text(json.dumps(plan, indent=2), encoding="utf-8")

    per_target = pd.Series(condition).value_counts()
    estimate = {row["sample"]: row for row in survey["gold"] + survey["supplementary"]}
    gold_stats = similarity(reference, _as_sample(counts[is_control], genes), args.target_sum)
    report.update(
        {
            "study": study,
            "gold": {
                "sample": GOLD_SAMPLE,
                "estimated_bytes": estimate.get(GOLD_SAMPLE, {}).get("bytes"),
                "estimated_cells": estimate.get(GOLD_SAMPLE, {}).get("cells_estimate"),
                "actual_bytes": Path(GOLD_FILE).stat().st_size,
                "cells_in_file": int(report["steps"][0][1]),
                "cells": int(len(gold)),
                "controls": int(is_control.sum()),
                "perturbations": len(perturbed),
                "cells_per_perturbation": {
                    "min": int(per_target.drop("ctrl").min()),
                    "median": float(per_target.drop("ctrl").median()),
                    "max": int(per_target.drop("ctrl").max()),
                },
                "genes_in_file": len(genes),
                "median_umis": gold_stats["median_umis"],
                "ribosomal_fraction": gold_stats["ribosomal_fraction"],
            },
            "candidates": [
                {
                    **row,
                    "estimated_bytes": estimate.get(row["sample"], {}).get("bytes"),
                    "estimated_cells": estimate.get(row["sample"], {}).get("cells_estimate"),
                }
                for row in scored
            ],
            "supplementary": {
                "sources": [sample.source for sample in kept],
                "cells": int(supp_cells),
                "panel_coverage": coverage,
            },
            "panel": {"genes": len(panel_genes), "forced_perturbed": len(forced)},
            "download_gb": round(spent / 1e9, 2),
        }
    )
    (out / "prepare_report.json").write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    (out / "prepare_report.md").write_text(render(report, per_target), encoding="utf-8")
    log(f"done -> {out / 'plan.json'}")
    return 0


def _size(value) -> str:
    if not value:
        return "?"
    return f"{value / 1e9:.2f} GB" if value >= 1e9 else f"{value / 1e6:.1f} MB"


def _as_sample(counts: np.ndarray, genes: list[str]):
    from scipy import sparse

    from kosmos.ppi.perturbation.dataprep.load import Sample

    return Sample(sparse.csr_matrix(counts), [], genes, GOLD_SAMPLE)


def render(report: dict, per_target: pd.Series) -> str:
    gold, supp, panel, study = report["gold"], report["supplementary"], report["panel"], report["study"]
    est = lambda v: f"~{v:,}" if v else "?"
    lines = [
        "# Prepare report — Jurkat TCR",
        "",
        f"**Question:** {study['question']}",
        "",
        "## Gold: GSE92872 (Datlinger 2017 CROP-seq)",
        "",
        "| | survey (before download) | actual |",
        "|---|---|---|",
        f"| download | {_size(gold['estimated_bytes'])} | {_size(gold['actual_bytes'])} |",
        f"| cells | {est(gold['estimated_cells'])} | {gold['cells_in_file']:,} in the file, {gold['cells']:,} used |",
        "",
        "| step | cells |",
        "|---|---|",
        *[f"| {name} | {count:,} |" for name, count in report["steps"]],
        "",
        f"- {gold['controls']:,} non-targeting controls (`CTRL`), {gold['perturbations']} perturbed genes, "
        f"{gold['cells_per_perturbation']['min']}-{gold['cells_per_perturbation']['max']} cells each "
        f"(median {gold['cells_per_perturbation']['median']:.0f}); {gold['genes_in_file']:,} genes; median "
        f"{gold['median_umis']:,.0f} UMIs, ribosomal {gold['ribosomal_fraction']:.1%}",
        "",
        "## Supplementary candidates: survey vs downloaded, and similarity to the gold controls",
        "",
        f"Pooled when the Pearson correlation of mean log1p-CPM (shared expressed genes) with the gold "
        f"controls is >= {report['decisions']['min_similarity']}; a series is judged on its first sample and "
        f"at most {report['decisions']['max_samples_per_series']} of its samples are pooled. Download "
        f"{report['download_gb']:.2f} GB of the {report['decisions']['supp_budget_gb']:g} GB budget.",
        "",
        "| sample | series | download (survey) | cells (survey) | barcodes | cells (>= "
        f"{report['decisions']['min_umis']} UMIs) | Pearson | Spearman | median UMIs | ribosomal | decision |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for row in report["candidates"]:
        number = lambda key, fmt: (fmt.format(row[key]) if row.get(key) is not None else "-")
        lines.append(
            f"| {row['sample']} | {row['series'].split('://', 1)[1]} | {_size(row['estimated_bytes'])} | "
            f"{est(row['estimated_cells'])} | {number('barcodes', '{:,}')} | {number('cells', '{:,}')} | "
            f"{number('pearson', '{:.2f}')} | {number('spearman', '{:.2f}')} | {number('median_umis', '{:,.0f}')} | "
            f"{number('ribosomal_fraction', '{:.1%}')} | {row['decision']} |"
        )
    lines += [
        "",
        f"**Pooled:** {supp['cells']:,} cells from {len(supp['sources'])} sample(s): "
        + ", ".join(f"{name} ({covered:,} panel genes measured)" for name, covered in supp["panel_coverage"].items()),
        "",
        "## Panel and study",
        "",
        f"- {panel['genes']:,} genes (top HVG among genes a pooled supp measures + {panel['forced_perturbed']} "
        f"perturbed genes); {study['normalize']}; a panel gene a source does not measure is the gold control mean",
        f"- split `{study['split_mode']}`: test {study['test_fraction']:.0%}, validation "
        f"{study['validation_fraction']:.0%} of the rest; batch {study['batch_size']}",
        "",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
