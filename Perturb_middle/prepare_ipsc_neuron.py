#!/usr/bin/env python
"""Stage 2 for the iPSC-neuron question: the survey's datasets -> training tables.

Question: *In human iPSC-derived neurons, can the transcriptional response to
CRISPRi knockdown of genes not seen in training be predicted from a small labeled
screen, using large single-cell profiles of iPSC-derived neurons from other
experiments, with their labels ignored, as supplementary evidence?*

Stage 1: `artifacts/dataprep/ipsc-neuron-v3/survey.json` (survey, nothing
downloaded) and `gold_check.md` (golds downloaded, split-half reliability).

  gold  GSE289235 / GSM8787485 -- CRISPRi CROP-seq in day-21 iPSC neurons (WTC11),
        one h5ad, unfiltered (1.5M barcodes). Labels in `obs.guide`
        (`GENE_i1`/`GENE_i2`, `non-targeting_*`). Gold check: 41 genes, median 171
        cells each, 522 non-targeting cells, top-20 DE reliability 0.76.
  supp  the survey's other candidates, labels ignored: the unused gold candidates
        first (CRISPRi screens in iPSC neurons: GSE124703 Tian 2019, GSE152988
        Tian 2021), then the supplementary list (GEO by size, then scBaseCount).

Decisions (each is a flag):

  * gold cells: barcodes with >= `--min-umis` UMIs (the gold check used 1,000).
  * labels: both guides of a gene pool to the gene (named as the guide names it:
    `H2AFZ`, which GEARS' GO vocabulary knows; the matrix calls it `H2AZ1`);
    `non-targeting_*` -> `ctrl`;
    a gene `G` -> `G+ctrl` (GEARS); cells with no guide are dropped (not used as
    controls: non-targeting cells exist here).
  * supplementary: candidates are downloaded in order within `--supp-budget-gb`,
    cells called at >= `--min-umis`, compared with the gold controls (Pearson of
    gene means, log1p-CPM), pooled at >= `--min-similarity`; a series is judged
    on its first sample, at most `--max-samples-per-series` of it pooled. Stops
    once the pooled cells reach `--supp-ratio` x the gold cells (the
    demonstration needs supp 2-5x+ the gold).
  * counts -> log1p(CPM 1e4); panel = top `--n-hvg` genes by variance over the
    gold cells among the genes some pooled supp measures (as in the Jurkat
    script), plus every perturbed gene; a panel gene a source does not measure
    is the gold control mean.
  * study: unseen-gene split, test 25%, validation 20% of the rest, batch 32.

Writes `gold.parquet`, `gold_counts.parquet`, `supp_ipsc_neuron_pooled(.parquet,
_counts.parquet)`, `plan.json` (+ `study`), `prepare_report.md/.json`.

    .venv/bin/python Perturb_middle/prepare_ipsc_neuron.py
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
from scipy import sparse

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

SURVEY = "artifacts/dataprep/ipsc-neuron-v3/survey.json"
OUT = "Perturb_middle/output_ipsc_neuron"
GOLD_SERIES = "geo://GSE289235"
GOLD_SAMPLE = "GSM8787485"
GOLD_NAME = "GSM8787485_EXP224.h5ad"
GOLD_URL = f"https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM8787nnn/{GOLD_SAMPLE}/suppl/{GOLD_NAME}"
SUPP_DIR = "data/fetched/samples"
GOLD_FILE = f"{SUPP_DIR}/GSE289235/{GOLD_SAMPLE}/{GOLD_NAME}"
CELL_TYPE = "iPSC_neuron_d21"
#: Guides named with a retired HGNC symbol -> the matrix's current one. Empty: GEARS'
#: GO vocabulary has the old `H2AFZ` and not `H2AZ1`, and a perturbation outside it
#: is dropped in stage 3, so the guide's name is kept.
GUIDE_RENAMES: dict[str, str] = {}
TOTAL_COLUMN = "total_counts"
SUPP_NAME = "supp_ipsc_neuron_pooled"


def log(message: str) -> None:
    print(f"[ipsc-neuron] {message}", flush=True)


def download(url: str, path: Path) -> None:
    if path.exists() and path.stat().st_size > 0:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    log(f"downloading {url.rsplit('/', 1)[-1]}")
    tmp = path.with_name(path.name + ".part")
    urllib.request.urlretrieve(url, tmp)
    tmp.rename(path)


def log1p_cpm(counts: sparse.spmatrix, target_sum: float) -> sparse.csr_matrix:
    counts = sparse.csr_matrix(counts, dtype=np.float32)
    totals = np.asarray(counts.sum(axis=1)).ravel()
    scaled = sparse.diags((target_sum / np.where(totals > 0, totals, 1.0)).astype(np.float32)) @ counts
    scaled.data = np.log1p(scaled.data)
    return scaled.tocsr()


# -- gold -----------------------------------------------------------------------------


def read_gold(path: Path, min_umis: int) -> tuple[sparse.csr_matrix, list[str], np.ndarray, np.ndarray, list]:
    """Called cells x genes counts, gene symbols, guide per cell, condition per cell."""
    import anndata

    adata = anndata.read_h5ad(path)
    steps = [("barcodes in the file", int(adata.n_obs))]
    umis = np.asarray(adata.X.sum(axis=1)).ravel()
    called = np.flatnonzero(umis >= min_umis)
    steps.append((f"cells (>= {min_umis} UMIs)", len(called)))
    guides = adata.obs["guide"].astype(str).to_numpy()[called]
    has_guide = guides != "nan"
    steps.append(("with a guide call", int(has_guide.sum())))
    keep = called[has_guide]
    guides = guides[has_guide]
    symbols = set(adata.var_names)
    target = lambda g: (lambda gene: GUIDE_RENAMES.get(gene, gene) if GUIDE_RENAMES.get(gene) in symbols else gene)(
        re.sub(r"_i\d+$", "", g)
    )
    condition = np.array(
        ["ctrl" if g.startswith("non-targeting") else target(g) + "+ctrl" for g in guides], dtype=object
    )
    counts = sparse.csr_matrix(adata.X[keep], dtype=np.float32)
    return counts, [str(g) for g in adata.var_names], guides, condition, steps


# -- supplementary candidates -----------------------------------------------------------

#: Matrices that are not expression: guide/label/hashtag counts of a screen.
_NOT_EXPRESSION = ("guide", "label", "hto", "adt", "gdo", "hash", "sgrna")


def candidate_files(row: dict) -> list[tuple[str, str, int]]:
    """(file name, url, bytes) of a survey row's expression files."""
    if row.get("source") == "scbasecount":
        url = row["files"][0]["name"]
        return [(url.rsplit("/", 1)[-1], url, int(row.get("bytes") or 0))]
    if row["series"].startswith("hf://"):
        repo = row["series"].split("://", 1)[1]
        return [
            (f["name"], f"https://huggingface.co/datasets/{repo}/resolve/main/{f['name']}", int(f.get("bytes") or 0))
            for f in row["files"]
            if str(f["name"]).lower().endswith((".h5", ".h5ad"))
        ]
    gsm = row["sample"]
    base = f"https://ftp.ncbi.nlm.nih.gov/geo/samples/{gsm[:-3]}nnn/{gsm}/suppl/"
    files = [f for f in row["files"] if not any(w in str(f["name"]).lower() for w in _NOT_EXPRESSION)]
    files = [f for f in files if re.search(r"matrix|barcode|feature|genes|\.h5", str(f["name"]), re.I)]
    # a sample with two layouts of the same data (mtx triplet + counts txt): the triplet
    files = [f for f in files if not re.search(r"counts_matrix\.txt", str(f["name"]), re.I)]
    return [(f["name"], base + f["name"], int(f.get("bytes") or 0)) for f in files]


def remote_size(url: str) -> int:
    try:
        request = urllib.request.Request(url, method="HEAD")
        with urllib.request.urlopen(request, timeout=60) as response:
            return int(response.headers.get("Content-Length") or 0)
    except Exception:  # noqa: BLE001 - unknown size: the budget check sees 0
        return 0


def ordered_candidates(survey: dict) -> list[dict]:
    """Unused gold candidates first (CRISPRi screens in iPSC neurons, labels
    ignored), then the supplementary list in the survey's order."""
    gold_others = [r for r in survey["gold"] if r["series"] != GOLD_SERIES]
    supp = [r for r in survey["supplementary"] if r["series"] != GOLD_SERIES]
    seen, out = set(), []
    for row in gold_others + supp:
        if row["sample"] not in seen:
            seen.add(row["sample"])
            out.append(row)
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--survey", default=SURVEY)
    parser.add_argument("--out", default=OUT)
    parser.add_argument("--min-umis", type=int, default=1000, help="barcode -> cell threshold (gold and supp)")
    parser.add_argument("--min-similarity", type=float, default=0.6,
                        help="pool a supp sample when its gene means correlate with the gold controls' at least this much")
    parser.add_argument("--max-samples-per-series", type=int, default=2)
    parser.add_argument("--supp-budget-gb", type=float, default=8.0, help="download budget for supp candidates")
    parser.add_argument("--supp-ratio", type=float, default=5.0,
                        help="stop pooling once supp cells >= this x gold cells (0 = use the whole budget)")
    parser.add_argument("--n-hvg", type=int, default=5000)
    parser.add_argument("--target-sum", type=float, default=1e4)
    parser.add_argument("--test-fraction", type=float, default=0.25)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args(argv)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    survey = json.loads(Path(args.survey).read_text(encoding="utf-8"))
    report: dict = {"survey": args.survey, "decisions": vars(args)}

    # -- gold ------------------------------------------------------------------------
    download(GOLD_URL, ROOT / GOLD_FILE)
    counts, genes, guides, condition, steps = read_gold(ROOT / GOLD_FILE, args.min_umis)
    report["steps"] = steps
    is_control = condition == "ctrl"
    totals = np.asarray(counts.sum(axis=1)).ravel().astype(np.float64)
    log(f"gold: {counts.shape[0]:,} cells ({int(is_control.sum()):,} ctrl), "
        f"{len(set(condition)) - 1} perturbed genes")

    # -- supplementary: download, score against the gold controls, pool -----------------
    from kosmos.ppi.perturbation.dataprep.load import Sample, call_cells, gene_means, load_sample, similarity

    reference: dict[str, float] = {}
    for gene, value in zip(genes, gene_means(counts[is_control], args.target_sum)):
        reference.setdefault(gene, float(value))
    target_cells = args.supp_ratio * counts.shape[0] if args.supp_ratio > 0 else float("inf")
    budget = args.supp_budget_gb * 1e9
    spent, per_series_kept, judged_series = 0.0, {}, {}
    kept: list[Sample] = []
    scored: list[dict] = []
    for row in ordered_candidates(survey):
        series = row["series"]
        source = series if row.get("source") != "scbasecount" else "scbasecount"
        entry = {"sample": row["sample"], "series": series, "title": str(row["sample_title"])[:70],
                 "estimated_bytes": row.get("bytes"), "estimated_cells": row.get("cells_estimate")}
        pooled = sum(sample.n_cells for sample in kept)
        if pooled >= target_cells:
            scored.append({**entry, "decision": f"not needed: {pooled:,} cells pooled (>= {args.supp_ratio:g}x gold)"})
            continue
        # scBaseCount samples are separate studies: the per-series rules do not apply
        if source != "scbasecount":
            if judged_series.get(series) is False:
                scored.append({**entry, "decision": "skipped: its series failed on an earlier sample"})
                continue
            if per_series_kept.get(series, 0) >= args.max_samples_per_series:
                scored.append({**entry, "decision": f"skipped: {args.max_samples_per_series} sample(s) of this series already pooled"})
                continue
        files = candidate_files(row)
        if not files:
            scored.append({**entry, "decision": "skipped: no expression files"})
            continue
        directory = ROOT / SUPP_DIR / series.split("://", 1)[-1].replace("/", "__") / row["sample"]
        missing = [(n, u, b) for n, u, b in files if not (directory / n).exists()]
        size = sum(b or remote_size(u) for n, u, b in missing)
        if spent + size > budget:
            scored.append({**entry, "decision": f"skipped: over the {args.supp_budget_gb:g} GB budget ({size / 1e9:.2f} GB)"})
            continue
        try:
            for name, url, _ in files:
                download(url, directory / name)
        except Exception as error:  # noqa: BLE001
            scored.append({**entry, "decision": f"download failed: {str(error)[:100]}"})
            continue
        spent += size
        try:
            sample, barcodes_before = call_cells(load_sample([directory / n for n, _, _ in files]), args.min_umis)
        except Exception as error:  # noqa: BLE001 - an unreadable candidate is reported, not fatal
            scored.append({**entry, "decision": f"unreadable: {type(error).__name__}: {str(error)[:120]}"})
            continue
        sample.source = f"{row['sample']}"
        stats = similarity(reference, sample, args.target_sum)
        accept = stats["pearson"] >= args.min_similarity and sample.n_cells > 0
        judged_series.setdefault(series, accept)
        scored.append(
            {
                **entry,
                "downloaded_bytes": size,
                "barcodes": barcodes_before,
                "cells": sample.n_cells,
                **{k: round(v, 3) if isinstance(v, float) else v for k, v in stats.items()},
                "decision": "pooled" if accept else f"dropped: Pearson {stats['pearson']:.2f} < {args.min_similarity}",
            }
        )
        log(f"{row['sample']} ({series}): {sample.n_cells:,} cells, Pearson {stats['pearson']:.2f}, "
            f"{stats['shared_genes']:,} shared genes -> {scored[-1]['decision']}")
        if accept:
            kept.append(sample)
            per_series_kept[series] = per_series_kept.get(series, 0) + 1
    if not kept:
        raise SystemExit("no supplementary candidate is similar enough to the gold; see the log")
    supp_cells = sum(sample.n_cells for sample in kept)
    log(f"supp: {supp_cells:,} cells from {len(kept)} sample(s) ({supp_cells / counts.shape[0]:.1f}x gold), "
        f"{spent / 1e9:.2f} GB downloaded")

    # -- panel ---------------------------------------------------------------------------
    gene_index = pd.Index(genes)
    unique = ~gene_index.duplicated()
    measured_by_supp = set().union(*(set(sample.genes) for sample in kept))
    values = log1p_cpm(counts, args.target_sum)
    mean = np.asarray(values.mean(axis=0)).ravel()
    variance = np.asarray(values.multiply(values).mean(axis=0)).ravel() - mean**2
    eligible = unique & gene_index.isin(list(measured_by_supp))
    ranked = [int(i) for i in np.argsort(-variance) if eligible[i]][: args.n_hvg]
    perturbed = sorted({c.split("+")[0] for c in condition if c != "ctrl"})
    by_symbol = {genes[i]: i for i in range(len(genes)) if unique[i]}
    missing_perturbed = [g for g in perturbed if g not in by_symbol]
    forced = [by_symbol[g] for g in perturbed if g in by_symbol and by_symbol[g] not in set(ranked)]
    panel = sorted(set(ranked) | set(forced))
    panel_genes = [genes[i] for i in panel]
    log(f"panel: {len(panel):,} genes ({len(forced)} perturbed gene(s) added beyond the top {args.n_hvg}"
        + (f"; not in the gold's genes: {', '.join(missing_perturbed)}" if missing_perturbed else "") + ")")

    gold_values = values[:, panel].toarray()
    gold = pd.DataFrame(gold_values, columns=panel_genes)
    gold.insert(0, "guide", guides)
    gold.insert(0, "cell_type", CELL_TYPE)
    gold.insert(0, "condition", condition)
    gold.to_parquet(out / "gold.parquet", index=False)
    gold_counts = pd.DataFrame(counts[:, panel].toarray(), columns=panel_genes)
    gold_counts[TOTAL_COLUMN] = totals
    gold_counts.to_parquet(out / "gold_counts.parquet", index=False)
    control_mean = gold_values[is_control].mean(axis=0)

    value_blocks, count_blocks, sources, coverage = [], [], [], {}
    for sample in kept:
        position: dict[str, int] = {}
        for index, symbol in enumerate(sample.genes):
            position.setdefault(symbol, index)
        shared = [(column, position[g]) for column, g in enumerate(panel_genes) if g in position]
        columns, source_columns = (list(x) for x in zip(*shared))
        sample_totals = np.asarray(sample.counts.sum(axis=1)).ravel().astype(np.float64)
        picked = sample.counts[:, source_columns].toarray().astype(np.float32)
        block = np.tile(control_mean, (sample.n_cells, 1)).astype(np.float32)
        scale = (args.target_sum / np.where(sample_totals > 0, sample_totals, 1.0)).astype(np.float32)
        block[:, columns] = np.log1p(picked * scale[:, None])
        raw = np.zeros((sample.n_cells, len(panel_genes)), dtype=np.float32)
        raw[:, columns] = picked
        value_blocks.append(block)
        count_blocks.append((raw, sample_totals))
        sources += [sample.source] * sample.n_cells
        coverage[sample.source] = len(shared)
    supp_table = pd.DataFrame(np.vstack(value_blocks), columns=panel_genes)
    supp_table["supp_source"] = sources
    supp_table.to_parquet(out / f"{SUPP_NAME}.parquet", index=False)
    supp_raw = pd.DataFrame(np.vstack([raw for raw, _ in count_blocks]), columns=panel_genes)
    supp_raw[TOTAL_COLUMN] = np.concatenate([t for _, t in count_blocks])
    supp_raw.to_parquet(out / f"{SUPP_NAME}_counts.parquet", index=False)

    # -- plan + report -------------------------------------------------------------------
    study = {
        "question": survey["question"],
        "dataset": "GSE289235 (CRISPRi CROP-seq, day-21 iPSC-derived neurons)",
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
            "supplementary": {str(out / f"{SUPP_NAME}.parquet"): str(out / f"{SUPP_NAME}_counts.parquet")},
        },
    }
    plan = {
        "version": 1,
        "task": {"objective": survey["question"], "target_column": "condition", "backend": "perturbation"},
        "study": study,
        "gold": [{"path": str(out / "gold.parquet"), "role": "gold",
                  "reason": "GSE289235 CRISPRi CROP-seq, day-21 iPSC-derived neurons"}],
        "supplementary": [
            {
                "path": str(out / f"{SUPP_NAME}.parquet"),
                "role": "supplementary",
                "reason": f"{len(kept)} iPSC-neuron sample(s) pooled by similarity to the gold controls "
                f"({', '.join(sample.source for sample in kept)}), labels ignored",
            }
        ],
        "unusable": [],
    }
    (out / "plan.json").write_text(json.dumps(plan, indent=2), encoding="utf-8")

    per_target = pd.Series(condition).value_counts()
    gold_row = next((r for r in survey["gold"] if r["sample"] == GOLD_SAMPLE), {})
    gold_stats = similarity(reference, Sample(counts[is_control], [], genes, GOLD_SAMPLE), args.target_sum)
    report.update(
        {
            "study": study,
            "gold": {
                "sample": GOLD_SAMPLE,
                "estimated_bytes": gold_row.get("bytes"),
                "estimated_cells": gold_row.get("cells_estimate"),
                "actual_bytes": (ROOT / GOLD_FILE).stat().st_size,
                "cells": int(counts.shape[0]),
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
            "candidates": scored,
            "supplementary": {
                "sources": [sample.source for sample in kept],
                "cells": int(supp_cells),
                "ratio": round(supp_cells / counts.shape[0], 2),
                "panel_coverage": coverage,
            },
            "panel": {"genes": len(panel_genes), "forced_perturbed": len(forced), "missing_perturbed": missing_perturbed},
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


def render(report: dict, per_target: pd.Series) -> str:
    gold, supp, panel, study = report["gold"], report["supplementary"], report["panel"], report["study"]
    est = lambda v: f"~{v:,}" if v else "?"
    lines = [
        "# Prepare report — iPSC-derived neurons, CRISPRi",
        "",
        f"**Question:** {study['question']}",
        "",
        f"## Gold: {GOLD_SERIES.split('://')[1]} / {GOLD_SAMPLE}",
        "",
        "| | survey (before download) | actual |",
        "|---|---|---|",
        f"| download | {_size(gold['estimated_bytes'])} | {_size(gold['actual_bytes'])} |",
        f"| cells | {est(gold['estimated_cells'])} | {gold['cells']:,} used |",
        "",
        "| step | cells |",
        "|---|---|",
        *[f"| {name} | {count:,} |" for name, count in report["steps"]],
        "",
        f"- {gold['controls']:,} non-targeting controls, {gold['perturbations']} perturbed genes, "
        f"{gold['cells_per_perturbation']['min']}-{gold['cells_per_perturbation']['max']} cells each "
        f"(median {gold['cells_per_perturbation']['median']:.0f}); {gold['genes_in_file']:,} genes; median "
        f"{gold['median_umis']:,.0f} UMIs, ribosomal {gold['ribosomal_fraction']:.1%}",
        "",
        "| condition | cells |",
        "|---|---|",
        *[f"| {name} | {count:,} |" for name, count in per_target.items()],
        "",
        "## Supplementary candidates: survey vs downloaded, and similarity to the gold controls",
        "",
        f"Pooled when the Pearson correlation of mean log1p-CPM (shared expressed genes) with the gold "
        f"controls is >= {report['decisions']['min_similarity']}; a GEO series is judged on its first sample and "
        f"at most {report['decisions']['max_samples_per_series']} of its samples are pooled; pooling stops at "
        f"{report['decisions']['supp_ratio']:g}x the gold cells. Downloaded {report['download_gb']:.2f} GB of the "
        f"{report['decisions']['supp_budget_gb']:g} GB budget.",
        "",
        "| sample | series | title | download (survey) | cells (survey) | barcodes | cells (>= "
        f"{report['decisions']['min_umis']} UMIs) | Pearson | Spearman | median UMIs | ribosomal | decision |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for row in report["candidates"]:
        number = lambda key, fmt: (fmt.format(row[key]) if row.get(key) is not None else "-")
        lines.append(
            f"| {row['sample']} | {row['series'].split('://', 1)[-1]} | {row['title'][:40]} | "
            f"{_size(row.get('estimated_bytes'))} | {est(row.get('estimated_cells'))} | {number('barcodes', '{:,}')} | "
            f"{number('cells', '{:,}')} | {number('pearson', '{:.2f}')} | {number('spearman', '{:.2f}')} | "
            f"{number('median_umis', '{:,.0f}')} | {number('ribosomal_fraction', '{:.1%}')} | {row['decision']} |"
        )
    lines += [
        "",
        f"**Pooled:** {supp['cells']:,} cells ({supp['ratio']:g}x the gold) from {len(supp['sources'])} sample(s): "
        + ", ".join(f"{name} ({covered:,} panel genes measured)" for name, covered in supp["panel_coverage"].items()),
        "",
        "## Panel and study",
        "",
        f"- {panel['genes']:,} genes (top HVG among genes a pooled supp measures + {panel['forced_perturbed']} "
        f"perturbed genes); {study['normalize']}; a panel gene a source does not measure is the gold control mean"
        + (f"; perturbed genes absent from the gold's genes: {', '.join(panel['missing_perturbed'])}"
           if panel["missing_perturbed"] else ""),
        f"- split `{study['split_mode']}`: test {study['test_fraction']:.0%}, validation "
        f"{study['validation_fraction']:.0%} of the rest; batch {study['batch_size']}",
        "",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
