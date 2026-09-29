"""Fetch perturbation sources the way every other dataset is fetched.

`datafetcher` already knows how to download a reference, unpack an archive (by
name or, now, by its bytes) and convert a single-cell file into a table. This
module points it at the registered screens and hands back the tables, so the
perturbation backend gets its gold and supplementary data without anybody
running commands by hand.

One correction is applied afterwards: the panel must contain the genes that
were perturbed, or the contract has nothing to encode. Variance selection over
4,000 genes can drop a perturbed gene, so the gold is re-converted with those
genes required when that happens.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from .sources import PerturbationSource

ROOT = Path(__file__).resolve().parents[3]


def _fetch(url: str, *, env: dict[str, str], echo: bool = True) -> dict[str, Any]:
    command = [sys.executable, "-m", "datafetcher", "fetch", url, "--json"]
    process = subprocess.run(
        command, cwd=str(ROOT), env=env, capture_output=True, text=True
    )
    if process.returncode != 0:
        raise RuntimeError(
            f"fetching {url} failed: {(process.stderr or process.stdout).strip()[:300]}"
        )
    if echo:
        print(f"# staged: {url}", flush=True)
    payload = json.loads(process.stdout)
    return payload


def _file_path(payload: dict[str, Any], relative: str) -> Path:
    """A fetch record's path is relative to the fetch's staging directory."""
    candidate = Path(relative)
    if candidate.is_absolute():
        return candidate
    directory = Path(str(payload.get("directory") or ""))
    return (directory / candidate) if directory.is_absolute() else (ROOT / directory / candidate)


def _table_of(payload: dict[str, Any]) -> Path | None:
    """The converted table this fetch produced, if it produced one."""
    for record in payload.get("files") or []:
        relative = str(record.get("path") or "")
        if relative.endswith("-table.csv"):
            return _file_path(payload, relative)
    return None


def _h5ad_of(payload: dict[str, Any]) -> Path | None:
    for record in payload.get("files") or []:
        relative = str(record.get("path") or "")
        if relative.endswith((".h5ad", ".h5ad.gz")):
            return _file_path(payload, relative)
    return None


def _perturbed_genes_from(table: Path, condition_column: str) -> set[str]:
    import pandas as pd

    frame = pd.read_csv(table, usecols=[condition_column])
    genes: set[str] = set()
    for value in frame[condition_column].astype(str).unique():
        for part in value.split("+"):
            if part and part.lower() not in {"ctrl", "control"}:
                genes.add(part)
    return genes


def ensure_panel(
    table: Path,
    h5ad: Path,
    condition_column: str,
    *,
    candidate_genes: Sequence[str] = (),
) -> tuple[Path, dict[str, Any]]:
    """Re-convert the screen so the panel covers the perturbations *and* the sources.

    `candidate_genes` are the genes the supplementary sources measured: the
    variance selection then happens inside the intersection, which is what keeps
    the panel wide. Without it, the gold's own top-N is chosen first and the
    intersection afterwards throws most of it away.
    """

    from datafetcher.singlecell import SingleCellSelection, convert_single_cell

    required = _perturbed_genes_from(table, condition_column)
    columns = set(__import__("pandas").read_csv(table, nrows=0).columns)
    missing = sorted(required - columns)
    if not missing:
        return table, {"required_genes": len(required), "missing": 0, "reconverted": False}
    selection = SingleCellSelection(
        max_cells=int(os.environ.get("KOSMOS_SINGLE_CELL_MAX_CELLS", "8000")),
        max_genes=int(os.environ.get("KOSMOS_SINGLE_CELL_MAX_GENES", "4000")),
        scan_cells=int(os.environ.get("KOSMOS_SINGLE_CELL_SCAN_CELLS", "5000")),
        label_column=condition_column,
        required_genes=tuple(sorted(required)),
        candidate_genes=tuple(sorted(set(candidate_genes) | required)),
    )
    frame, selection = convert_single_cell(h5ad, selection=selection)
    table.write_text("", encoding="utf-8")
    frame.to_csv(table, index=False)
    return table, {
        "required_genes": len(required),
        "missing": len(missing),
        "reconverted": True,
        "missing_examples": missing[:5],
        "candidate_genes": len(set(candidate_genes)),
        "panel": int(frame.shape[1] - 1),
    }


def stage_sources(
    *,
    gold: PerturbationSource,
    supplementary: Sequence[PerturbationSource],
    condition_column: str = "condition",
    echo: bool = True,
) -> tuple[Path, list[Path], dict[str, Any]]:
    """Fetch, unpack and convert the gold screen and its supplementary sources."""
    env = dict(os.environ)
    env.setdefault("KOSMOS_SINGLE_CELL_LABEL", condition_column)
    env.setdefault("KOSMOS_SINGLE_CELL_MAX_CELLS", "8000")
    env.setdefault("KOSMOS_SINGLE_CELL_MAX_GENES", "4000")
    env.setdefault("MPLCONFIGDIR", "/tmp/kosmos-mpl")
    # A processed screen is gigabytes: the archive member cap has to be above it
    # or the fetch stages the small files beside the matrix and nothing else.
    env.setdefault("KOSMOS_DATAFETCHER_MAX_BYTES", str(4_000_000_000))
    env.setdefault("KOSMOS_DATAFETCHER_ARCHIVE_MEMBERS", "12")

    report: dict[str, Any] = {"sources": {}}

    # Supplementary sources first: their gene set is what the gold's panel is
    # selected *inside*, so the intersection is decided before the selection.
    tables: list[Path] = []
    measured: set[str] = set()
    for source in supplementary:
        try:
            payload = _fetch(source.url, env=env, echo=echo)
            table = _table_of(payload)
            if table is None:
                report["sources"][source.name] = {
                    "url": source.url,
                    "error": "the fetch produced no table",
                }
                continue
            import pandas as pd

            columns = {str(column) for column in pd.read_csv(table, nrows=0).columns}
            gene_like = {
                column
                for column in columns
                if column not in {"condition", "cell_type", "cell_id", "batch", "donor"}
            }
            if measured:
                measured &= gene_like
            else:
                measured = set(gene_like)
            tables.append(table)
            report["sources"][source.name] = {
                "url": source.url,
                "cell_line": source.cell_line,
                "table": str(table),
                "genes": len(gene_like),
            }
        except Exception as error:  # noqa: BLE001 - a failed source is reported
            report["sources"][source.name] = {
                "url": source.url,
                "error": f"{type(error).__name__}: {error}",
            }

    gold_payload = _fetch(gold.url, env=env, echo=echo)
    gold_table = _table_of(gold_payload)
    gold_h5ad = _h5ad_of(gold_payload)
    if gold_table is None:
        raise RuntimeError(
            f"{gold.name} produced no table; the fetch staged "
            f"{[r.get('path') for r in gold_payload.get('files') or []]}"
        )
    if gold_h5ad is not None:
        gold_table, correction = ensure_panel(
            gold_table, gold_h5ad, condition_column, candidate_genes=sorted(measured)
        )
        correction["selected_within"] = len(measured)
        report["sources"][gold.name] = {"url": gold.url, **correction}
    report["gold"] = str(gold_table)
    report["panel_policy"] = "intersect-then-select"
    return gold_table, tables, report
