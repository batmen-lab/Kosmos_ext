"""Stage 1 for a registered screen: download it and write the data package.

`kosmos dataprep --dataset replogle_k562_essential --supplementary-dataset adamson`
pins the data instead of searching for it (retrieval is not deterministic, and a
training question needs the same data every time). The registered screens are
GEARS' processed releases (`sources.py`): an AnnData `perturb_processed.h5ad`
with `obs['condition']` (`GENE+ctrl`, `ctrl`), log-normalised expression over
GEARS' gene selection, and the per-perturbation gene sets in `uns`.

The output is the same package the searching path writes -- `plan.json` and a
human-readable `data.md` -- with the h5ad files as the plan's tables. Stage 2
(`scripts/prepare_perturbation.py`) turns them into training tables.
"""

from __future__ import annotations

import json
import urllib.request
import zipfile
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from .graphs import GO_REFERENCE, go_vocabulary
from .sources import SOURCES

DATA_ROOT = Path("data/perturbation/gears")


def stage_registered(name: str, root: Path = DATA_ROOT) -> Path:
    """Download and unpack a registered screen once; return its h5ad."""
    if name not in SOURCES:
        raise ValueError(f"unknown dataset {name!r}; registered: {sorted(SOURCES)}")
    directory = root / name
    found = sorted(directory.rglob("perturb_processed.h5ad"))
    if found:
        return found[0]
    directory.mkdir(parents=True, exist_ok=True)
    archive = directory / "download.zip"
    print(f"# downloading {name} from {SOURCES[name].url}", flush=True)
    urllib.request.urlretrieve(SOURCES[name].url, archive)
    with zipfile.ZipFile(archive) as bundle:
        bundle.extractall(directory)
    found = sorted(directory.rglob("perturb_processed.h5ad"))
    if not found:
        raise FileNotFoundError(f"{archive} has no perturb_processed.h5ad")
    return found[0]


def _perturbed_genes(conditions: Sequence[str]) -> list[str]:
    return sorted(
        {gene for value in conditions for gene in str(value).split("+") if gene and gene != "ctrl"}
    )


def describe(path: Path, vocabulary: set[str]) -> dict[str, Any]:
    """What a reviewer needs to know about one screen, read from the h5ad."""
    import anndata

    adata = anndata.read_h5ad(path, backed="r")
    conditions = adata.obs["condition"].astype(str)
    counts = conditions.value_counts()
    perturbed = _perturbed_genes(counts.index)
    measured = set(adata.var["gene_name"].astype(str)) if "gene_name" in adata.var else set(adata.var_names)
    singles = [value for value in counts.index if value != "ctrl" and value.count("+") == 1 and "ctrl" in value.split("+")]
    return {
        "path": str(path),
        "cells": int(adata.n_obs),
        "genes": int(adata.n_vars),
        "cell_types": sorted(adata.obs["cell_type"].astype(str).unique().tolist()) if "cell_type" in adata.obs else [],
        "control_cells": int(counts.get("ctrl", 0)),
        "conditions": int(len(counts)),
        "single_perturbations": len(singles),
        "combinations": int(sum(1 for value in counts.index if value != "ctrl" and value not in singles)),
        "perturbed_genes": len(perturbed),
        "perturbed_genes_in_go_vocabulary": int(sum(gene in vocabulary for gene in perturbed)),
        "perturbed_genes_measured": int(sum(gene in measured for gene in perturbed)),
        "cells_per_perturbation": {
            "median": float(np.median(counts.drop("ctrl", errors="ignore"))),
            "min": int(counts.drop("ctrl", errors="ignore").min()),
        },
        "gene_ids": [str(value) for value in adata.var_names],
    }


def write_registered_plan(
    *,
    question: str,
    dataset: str,
    supplementary: Sequence[str],
    out_dir: Path,
    go_reference: Path = GO_REFERENCE,
) -> dict[str, Any]:
    """Stage the screens and write `plan.json` + `data.md` + `data_report.json`."""
    out_dir.mkdir(parents=True, exist_ok=True)
    vocabulary = set(go_vocabulary(go_reference)) if go_reference.exists() else set()
    gold_path = stage_registered(dataset)
    gold = describe(gold_path, vocabulary)
    gold_ids = set(gold.pop("gene_ids"))
    supp: list[dict[str, Any]] = []
    for name in supplementary:
        info = describe(stage_registered(name), vocabulary)
        ids = set(info.pop("gene_ids"))
        info["name"] = name
        info["genes_shared_with_gold"] = len(ids & gold_ids)
        supp.append(info)
    source = SOURCES[dataset]
    plan = {
        "version": 1,
        "task": {
            "objective": question,
            "target_column": "condition",
            "target_source": "registry",
            "target_confidence": "declared",
            "task_type": "perturbation",
            "backend": "perturbation",
        },
        "gold": [
            {
                "path": gold["path"],
                "role": "gold",
                "dataset": dataset,
                "reason": f"registered screen {dataset}: {source.cell_line}, {source.perturbation}",
            }
        ],
        "supplementary": [
            {
                "path": info["path"],
                "role": "supplementary",
                "dataset": info["name"],
                "reason": (
                    f"registered screen {info['name']}: {SOURCES[info['name']].cell_line}, "
                    f"{SOURCES[info['name']].perturbation}; {info['genes_shared_with_gold']} "
                    f"of the gold's {gold['genes']} genes"
                ),
            }
            for info in supp
        ],
        "unusable": [],
        "stage": "dataprep (registered dataset; run stage 2 to prepare training tables)",
    }
    (out_dir / "plan.json").write_text(json.dumps(plan, indent=2), encoding="utf-8")
    report = {"question": question, "gold": {"name": dataset, **gold}, "supplementary": supp}
    (out_dir / "data_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (out_dir / "data.md").write_text(_render(question, dataset, gold, supp), encoding="utf-8")
    return {"plan_path": out_dir / "plan.json", "data_md": out_dir / "data.md", "report": report}


def _render(question: str, dataset: str, gold: dict[str, Any], supp: list[dict[str, Any]]) -> str:
    source = SOURCES[dataset]
    lines = [
        f"# Data package — {dataset}",
        "",
        f"**Question:** {question}",
        "",
        "## Gold",
        "",
        f"- `{gold['path']}`",
        f"- {source.cell_line}, {source.perturbation} ({source.notes})",
        f"- {gold['cells']:,} cells × {gold['genes']:,} genes (GEARS-processed, log-normalised)",
        f"- {gold['control_cells']:,} control cells; {gold['single_perturbations']:,} single "
        f"perturbations, {gold['combinations']:,} combinations",
        f"- cells per perturbation: median {gold['cells_per_perturbation']['median']:.0f}, "
        f"min {gold['cells_per_perturbation']['min']}",
        f"- perturbed genes: {gold['perturbed_genes']:,}; in GEARS' GO vocabulary "
        f"{gold['perturbed_genes_in_go_vocabulary']:,} (the rest cannot be encoded and are "
        f"dropped); measured in the panel {gold['perturbed_genes_measured']:,}",
        "",
        "## Supplementary",
        "",
    ]
    if not supp:
        lines.append("- none (the signed PPI arm then has no unlabeled cells)")
    for info in supp:
        lines += [
            f"- **{info['name']}** `{info['path']}`",
            f"  - {SOURCES[info['name']].cell_line}, {SOURCES[info['name']].perturbation}; "
            f"{info['cells']:,} cells ({info['control_cells']:,} controls) × {info['genes']:,} genes",
            f"  - shares {info['genes_shared_with_gold']:,} of the gold's {gold['genes']:,} genes "
            f"(by gene id); stage 2 aligns it to the gold panel and fills the rest with the "
            f"gold control mean",
        ]
    lines += [
        "",
        "## Review",
        "",
        "- Is the gold the screen the question is about (cell line, perturbation type)?",
        "- Is the supplementary source the same cell line? Its cells are the",
        "  signed PPI arm's unlabeled inputs and its co-expression graph is `G_S`.",
        "",
        "Next: `scripts/prepare_perturbation.py --stage1 <this dir> --out <dir>` (stage 2).",
        "",
    ]
    return "\n".join(lines)
