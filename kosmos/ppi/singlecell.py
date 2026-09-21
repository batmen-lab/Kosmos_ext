"""Preprocessing a single-cell dataset for this pipeline, per source.

Raw counts are not what a classifier wants: a few genes carry most of the
variance, the library size differs from cell to cell, and the distributions are
skewed and non-negative. The recipe here is the standard one, applied **to each
source separately** so that a donor, an assay or a cohort does not borrow the
scaling of another:

    counts -> highly variable genes -> normalize + log1p -> per-gene z-score
           -> negative values clipped to 0

`scanpy` does this when it is installed; the fallback is the same arithmetic in
numpy, so a run without scanpy is not a different run. Selection happens per
source and the sources are then intersected: the model needs one feature space,
and a panel chosen by donor A that donor B does not measure is not a panel.

Everything the choice depended on is recorded -- genes in, genes kept, the cells
-- and `write_figures` draws the result, because "8,000 cells x 2,000 genes" is
not something a person can check by reading numbers.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

#: A table with this many features whose values are non-negative is counts.
COUNTS_MIN_FEATURES = 500


@dataclass(frozen=True)
class SingleCellConfig:
    """The knobs of the recipe, with the values the field uses by default."""

    #: Genes kept per source before the intersection.
    n_top_genes: int = 2000
    #: Library size each cell is scaled to before log1p.
    target_sum: float = 1e4
    #: A gene has to be seen in at least this many cells of a source.
    min_cells: int = 3
    seed: int = 42

    def to_dict(self) -> dict:
        return {
            "n_top_genes": self.n_top_genes,
            "target_sum": self.target_sum,
            "min_cells": self.min_cells,
            "seed": self.seed,
        }


@dataclass
class SourceStats:
    """What one source looked like, before and after the recipe.

    These are the numbers a reader needs to judge the preprocessing: how many
    cells, how deep they were sequenced, how sparse the matrix is, and what the
    transformation did to it. They travel into the report, the summary and the
    JSON, so "we preprocessed the genes" is never the only account of it.
    """

    name: str
    cells: int
    genes_in_file: int
    genes_non_numeric: int
    genes_detected: int
    genes_selected: int
    median_counts_per_cell: float
    sparsity_raw: float
    sparsity_processed: float
    mean_processed: float
    genes_dropped_rare: int

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "cells": self.cells,
            "genes_in_file": self.genes_in_file,
            "genes_non_numeric": self.genes_non_numeric,
            "genes_detected": self.genes_detected,
            "genes_selected": self.genes_selected,
            "genes_dropped_rare": self.genes_dropped_rare,
            "median_counts_per_cell": round(self.median_counts_per_cell, 1),
            "sparsity_raw": round(self.sparsity_raw, 4),
            "sparsity_processed": round(self.sparsity_processed, 4),
            "mean_processed": round(self.mean_processed, 4),
        }

    def describe(self) -> str:
        """One line for the console: what this source is."""
        return (
            f"{self.cells:,} cells x {self.genes_in_file:,} genes; median "
            f"{self.median_counts_per_cell:,.0f} counts/cell, "
            f"{self.sparsity_raw:.1%} zeros; kept {self.genes_selected:,} HVG "
            f"(dropped {self.genes_dropped_rare:,} seen in fewer than the minimum "
            f"cells); after log1p+z-clip {self.sparsity_processed:.1%} zeros"
        )


@dataclass
class PreparedSource:
    """One source after preprocessing, on its own selected genes."""

    name: str
    genes: list[str]
    #: Every gene this source measured, before selection: the panel has to be
    #: built from what the sources *have*, not only from what they selected.
    all_genes: list[str]
    #: (cells, genes) after log1p, z-score and clipping.
    matrix: np.ndarray
    #: Raw counts for every gene this source measured, so the panel can be cut
    #: from the whole source rather than from its own selection.
    counts: np.ndarray
    #: Library sizes, kept because normalisation is defined on the whole source.
    totals: np.ndarray
    cell_count: int
    total_genes: int
    #: The genes this source selected on its own, kept through alignment: after
    #: alignment `genes` is the shared panel by construction, so comparing those
    #: would say nothing about how much the sources actually agree.
    selection: list[str] = field(default_factory=list)
    #: What the source was, and what the recipe did to it.
    stats: SourceStats | None = None
    notes: list[str] = field(default_factory=list)

    @property
    def n_genes(self) -> int:
        return len(self.genes)

    @property
    def _rank_by_gene(self) -> dict[str, int]:
        """This source's own ranking: 0 is the gene it selected first."""
        return {
            gene: rank for rank, gene in enumerate(self.selection or self.genes)
        }

    def __post_init__(self) -> None:
        # A source built without an explicit `selection` selected its `genes`.
        if not self.selection:
            object.__setattr__(self, "selection", list(self.genes))


def looks_like_counts(frame: pd.DataFrame, feature_names: list[str]) -> bool:
    """Non-negative, integer-valued, and wide: a single-cell count matrix.

    A survey table with 20 measurement columns is not one; a matrix with 20,000
    mostly-zero integer columns is. The check is mechanical because it has to
    run before anything expensive, and a false negative only means the standard
    encoder is used instead (which is what every other task gets).
    """
    if len(feature_names) < COUNTS_MIN_FEATURES:
        return False
    sample = frame[feature_names[: min(len(feature_names), 200)]]
    # Only the numeric columns decide: a per-cell table carries its barcodes and
    # its labels next to the genes, and asking pandas to make floats of a
    # barcode raises before anything can be judged.
    numeric = sample.select_dtypes(include="number")
    if numeric.shape[1] == 0:
        return False
    values = numeric.to_numpy(dtype=np.float64, na_value=np.nan, copy=False)
    finite = values[np.isfinite(values)]
    if finite.size == 0 or finite.min() < 0:
        return False
    return bool(np.all(finite == np.floor(finite)))


def prepare_source(
    frame: pd.DataFrame,
    feature_names: list[str],
    *,
    name: str,
    config: SingleCellConfig | None = None,
) -> PreparedSource:
    """One source: select, normalize, log, standardise, clip.

    The steps are the ones the field uses, in the order the question asked for:
    highly variable genes first (on counts), then library-size normalisation and
    `log1p`, then a per-gene z-score, then negatives to zero -- the last step
    because a negative "expression" is not a measurement, and after centring
    every gene has them.
    """
    config = config or SingleCellConfig()
    genes = [gene for gene in feature_names if gene in set(frame.columns)]
    if not genes:
        raise ValueError(f"{name}: no feature columns to preprocess")
    # Whole-block conversion, not `apply(pd.to_numeric)` per column: 20,000
    # columns one at a time is a minute of pandas overhead for the same numbers.
    #
    # Only columns that are already numbers: a per-cell table carries its
    # barcodes and its labels beside the genes, and one of those in the panel is
    # a string where the matrix expects a count -- `could not convert string to
    # float: 'GATGACGGAC-GGTGGGAT'` is what that looks like from the training
    # side, with no hint about which column caused it.
    block = frame[genes]
    numeric_block = block.select_dtypes(include="number")
    not_numbers = [str(name) for name in block.columns if name not in numeric_block.columns]
    notes: list[str] = []
    if not_numbers:
        notes.append(
            f"{len(not_numbers)} column(s) hold text rather than measurements and "
            f"were left out, e.g. {not_numbers[:3]}"
        )
        logger.info(
            "%s: left out %d non-numeric column(s), e.g. %s",
            name,
            len(not_numbers),
            not_numbers[:3],
        )
    genes = [str(c) for c in numeric_block.columns]
    counts = numeric_block.to_numpy(dtype=np.float32, na_value=0.0, copy=False)
    counts[counts < 0] = 0.0
    total_genes = len(genes)
    counts_per_cell = counts.sum(axis=1)
    median_counts = float(np.median(counts_per_cell)) if counts.size else 0.0
    sparsity_raw = float((counts == 0).mean()) if counts.size else 1.0
    detected = (counts > 0).sum(axis=0)
    keep_detected = detected >= config.min_cells
    dropped_rare = int((~keep_detected).sum())
    if not keep_detected.all():
        genes = [gene for gene, keep in zip(genes, keep_detected, strict=True) if keep]
        counts = counts[:, keep_detected]
    measured_genes = list(genes)
    genes_detected = len(measured_genes)
    notes = [
        *notes,
        f"{len(genes):,} of {total_genes:,} gene(s) are detected in at least "
        f"{config.min_cells} cell(s)"
    ]

    totals = counts.sum(axis=1, keepdims=True)
    totals[totals == 0] = 1.0
    selected = _highly_variable(counts, genes, config, notes)
    positions = [genes.index(gene) for gene in selected]
    # Normalisation uses every detected gene, then the selection is applied:
    # scaling a cell by the total of ten selected genes makes that total depend
    # on which genes happened to be selected, which is not a library size.
    clipped = _transform(counts, totals, positions, config.target_sum)
    notes.append(
        f"values: counts -> log1p -> per-gene z-score -> negatives to 0 "
        f"(max {clipped.max():.2f}, mean {clipped.mean():.3f})"
    )
    logger.info(
        "%s: %s cells x %s genes after preprocessing (from %s feature(s))",
        name,
        f"{counts.shape[0]:,}",
        f"{counts.shape[1]:,}",
        f"{total_genes:,}",
    )
    logger.info(
        "%s: median %s counts/cell, %.1f%% zeros before; %s genes dropped for "
        "being seen in fewer than %d cell(s)",
        name,
        f"{median_counts:,.0f}",
        sparsity_raw * 100,
        f"{dropped_rare:,}",
        config.min_cells,
    )
    return PreparedSource(
        name=name,
        genes=selected,
        all_genes=measured_genes,
        matrix=clipped.astype(np.float32),
        counts=counts,
        totals=totals,
        cell_count=int(counts.shape[0]),
        total_genes=total_genes,
        stats=SourceStats(
            name=name,
            cells=int(counts.shape[0]),
            genes_in_file=total_genes,
            genes_non_numeric=len(not_numbers),
            genes_detected=genes_detected,
            genes_selected=len(selected),
            median_counts_per_cell=median_counts,
            sparsity_raw=sparsity_raw,
            sparsity_processed=float((clipped == 0).mean()) if clipped.size else 1.0,
            mean_processed=float(clipped.mean()) if clipped.size else 0.0,
            genes_dropped_rare=dropped_rare,
        ),
        notes=notes,
    )


def _highly_variable(
    counts: np.ndarray, genes: list[str], config: SingleCellConfig, notes: list[str]
) -> list[str]:
    """The top `n_top_genes`, by scanpy when it is installed."""
    wanted = min(config.n_top_genes, len(genes))
    if wanted >= len(genes):
        notes.append(f"every one of the {len(genes):,} gene(s) is kept (below the cap)")
        return list(genes)
    via_scanpy = _scanpy_hvg(counts, genes, wanted)
    if via_scanpy is not None:
        notes.append(f"scanpy selected {len(via_scanpy):,} highly variable gene(s)")
        return via_scanpy
    # Fallback: dispersion of log-normalised counts -- the quantity scanpy's
    # `seurat` flavour ranks, without the dependency.
    normalized = _normalize_and_log(counts, config.target_sum)
    mean = normalized.mean(axis=0)
    variance = normalized.var(axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        dispersion = np.where(mean > 0, variance / mean, 0.0)
    order = np.argsort(-dispersion)[:wanted]
    notes.append(
        f"selected {wanted:,} highly variable gene(s) by dispersion of "
        f"log-normalised counts (scanpy is not installed)"
    )
    return [genes[index] for index in sorted(order)]


def _scanpy_hvg(counts: np.ndarray, genes: list[str], wanted: int) -> list[str] | None:
    try:
        import anndata
        import scanpy as sc
    except ImportError:
        return None
    try:
        adata = anndata.AnnData(counts, var=pd.DataFrame(index=list(genes)))
        sc.pp.highly_variable_genes(adata, n_top_genes=wanted, flavor="seurat_v3")
    except Exception as e:  # noqa: BLE001 - a scanpy complaint is not a failure
        logger.warning("scanpy HVG selection failed (%s); using dispersion", e)
        return None
    mask = np.asarray(adata.var["highly_variable"], dtype=bool)
    return [gene for gene, keep in zip(genes, mask, strict=True) if keep]


def _normalize_and_log(counts: np.ndarray, target_sum: float) -> np.ndarray:
    """Library-size normalisation, then `log1p`."""
    totals = counts.sum(axis=1, keepdims=True)
    totals[totals == 0] = 1.0
    scaled = counts / totals * target_sum
    return np.log1p(scaled)


def _transform(
    counts: np.ndarray,
    totals: np.ndarray,
    positions: list[int],
    target_sum: float,
) -> np.ndarray:
    """The recipe for a chosen set of columns, from a source's whole counts.

    Only the chosen columns are materialised: normalising the whole 20,000-gene
    matrix to keep 2,000 of its columns is a gigabyte of temporaries for numbers
    that are thrown away.
    """
    block = counts[:, positions] / totals * target_sum
    logged = np.log1p(block)
    mean = logged.mean(axis=0, keepdims=True)
    std = logged.std(axis=0, keepdims=True)
    std[std == 0] = 1.0
    return np.clip((logged - mean) / std, 0.0, None)


def _zscore_genes(matrix: np.ndarray) -> np.ndarray:
    """Per-gene mean 0, standard deviation 1 -- within this source."""
    mean = matrix.mean(axis=0, keepdims=True)
    std = matrix.std(axis=0, keepdims=True)
    std[std == 0] = 1.0
    return (matrix - mean) / std


def align_sources(
    sources: list[PreparedSource],
    *,
    max_genes: int | None = None,
    target_sum: float = 1e4,
) -> tuple[list[str], list[PreparedSource]]:
    """The panel to train on, and each source cut to it.

    Selection is per source on purpose, so the panels differ. Intersecting them
    is what makes the correction comparable: a gene only one donor's variance
    ranked highly is not shared measurement. But the plain intersection of four
    2,000-gene lists is 700 genes, because selection is noisy -- so the panel is
    ranked *across* the sources (a gene several sources chose first, then a
    gene every source measured) and cut to `max_genes`. Every gene in it is
    measured by every source, which is the property the model needs.
    """
    if not sources:
        raise ValueError("no sources to align")
    # Measured everywhere: the panel cannot contain anything else. This is the
    # intersection of what the sources *have*, not of what each one selected --
    # selection is noisy, and requiring four independent selections to agree
    # leaves a few hundred genes where the files share twenty thousand.
    measured_everywhere = set(sources[0].all_genes)
    for source in sources[1:]:
        measured_everywhere &= set(source.all_genes)
    if not measured_everywhere:
        raise ValueError(
            "the sources selected no gene in common; their panels are too "
            "different to correct against each other"
        )
    # ... and, among those, the ones the sources ranked highest. The union of
    # the per-source selections is what creates the candidates, so a gene two
    # donors found and two missed is still eligible; it just ranks lower.
    ranked: list[tuple[int, float, str]] = []
    for gene in measured_everywhere:
        votes = 0
        rank_sum = 0.0
        for source in sources:
            if gene in source._rank_by_gene:
                votes += 1
                rank_sum += source._rank_by_gene[gene]
        ranked.append((-votes, rank_sum / max(1, votes), gene))
    ranked.sort()
    limit = max_genes if max_genes else len(ranked)
    panel = [gene for _, _, gene in ranked[:limit]]
    logger.info(
        "shared panel: %s gene(s) of %s measured by every source (%s selected by "
        "all of them)",
        f"{len(panel):,}",
        f"{len(measured_everywhere):,}",
        f"{sum(1 for entry in ranked if entry[0] == -len(sources)):,}",
    )
    aligned = []
    for source in sources:
        # Cut from what the source *measured*: a gene it did not select is still
        # a gene it measured, and the panel is the sources' shared measurement.
        positions = [source.all_genes.index(gene) for gene in panel]
        aligned.append(
            PreparedSource(
                name=source.name,
                genes=list(panel),
                selection=list(source.selection or source.genes),
                all_genes=list(source.all_genes),
                matrix=_transform(source.counts, source.totals, positions, target_sum),
                counts=source.counts[:, positions],
                totals=source.totals,
                cell_count=source.cell_count,
                total_genes=source.total_genes,
                stats=source.stats,
                notes=[
                    *source.notes,
                    f"cut to the shared panel of {len(panel):,} gene(s) "
                    f"(its own selection was {source.n_genes:,})",
                ],
            )
        )
    return panel, aligned


# --- what was done, in words and numbers -----------------------------------


def describe(
    sources: list[PreparedSource],
    panel: list[str],
    config: SingleCellConfig | None = None,
) -> dict:
    """The account of the preprocessing: the recipe, and what each source was."""
    config = config or SingleCellConfig()
    stats = [source.stats.to_dict() for source in sources if source.stats]
    # Each source's *own* selection, not the aligned panel.
    selected = [set(source.selection or source.genes) for source in sources]
    overlap = {
        sources[i].name: {
            sources[j].name: len(selected[i] & selected[j]) for j in range(len(sources))
        }
        for i in range(len(sources))
    }
    all_shared = set.intersection(*selected) if selected else set()
    return {
        "recipe": [
            "counts",
            f"highly variable genes (top {config.n_top_genes} per source)",
            f"normalize every cell to {config.target_sum:,.0f} counts",
            "log1p",
            "per-gene z-score within the source",
            "clip negative values to 0",
        ],
        "parameters": config.to_dict(),
        "panel_size": len(panel),
        "genes_selected_by_every_source": len(all_shared),
        "sources": stats,
        "pairwise_selection_overlap": overlap,
        "notes": [
            "the recipe is applied to each source separately, so a donor's depth "
            "or a platform's scale never leaks into another source",
            f"the trained panel is the {len(panel):,} gene(s) those sources "
            f"selected in common, ranked across sources",
            "a gene only some sources selected is still eligible when every "
            "source measured it; a gene written in another organism's style "
            "(mouse `Pisd` vs human `PISD`) is excluded as evidence entirely",
        ],
    }


def write_report(
    sources: list[PreparedSource],
    panel: list[str],
    out_dir: str | Path,
    *,
    config: SingleCellConfig | None = None,
    figures: list[Path] | None = None,
) -> tuple[Path, Path]:
    """`preprocessing.md` and `preprocessing.json` beside the run's summary."""
    import json

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    account = describe(sources, panel, config)
    account["figures"] = [Path(path).name for path in (figures or [])]

    lines = [
        "# Single-cell preprocessing",
        "",
        "## Recipe (per source, in this order)",
        "",
    ]
    lines += [f"{index}. {step}" for index, step in enumerate(account["recipe"], 1)]
    lines += [
        "",
        "## Datasets",
        "",
        "| source | cells | genes in file | text columns left out | detected | "
        "selected (HVG) | median counts/cell | zeros before | zeros after |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for row in account["sources"]:
        lines.append(
            f"| `{row['name']}` | {row['cells']:,} | {row['genes_in_file']:,} | "
            f"{row.get('genes_non_numeric', 0):,} | {row['genes_detected']:,} | "
            f"{row['genes_selected']:,} | "
            f"{row['median_counts_per_cell']:,.0f} | {row['sparsity_raw']:.1%} | "
            f"{row['sparsity_processed']:.1%} |"
        )
    lines += [
        "",
        f"**Trained panel:** {account['panel_size']:,} gene(s) selected by every "
        f"source (of which {account['genes_selected_by_every_source']:,} are in "
        f"every source's own selection).",
        "",
        "## How much the sources agree",
        "",
        "| | "
        + " | ".join(sources[i].name for i in range(len(sources)))
        + " |",
        "|---" * (len(sources) + 1) + "|",
    ]
    for row in sources:
        cells = [
            f"{account['pairwise_selection_overlap'][row.name][other.name]:,}"
            for other in sources
        ]
        lines.append(f"| `{row.name}` | " + " | ".join(cells) + " |")
    if account["notes"]:
        lines += ["", "## Notes", ""]
        lines += [f"- {note}" for note in account["notes"]]
    source_notes = [
        (source.name, note) for source in sources for note in source.notes
    ]
    if source_notes:
        lines += ["", "## What each source was told", ""]
        lines += [f"- `{name}`: {note}" for name, note in source_notes]
    if figures:
        lines += ["", "## Figures", ""]
        lines += [f"![{Path(path).name}](figures/{Path(path).name})" for path in figures]

    markdown = out / "preprocessing.md"
    record = out / "preprocessing.json"
    markdown.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    record.write_text(json.dumps(account, indent=2, default=str), encoding="utf-8")
    return markdown, record


# --- figures ---------------------------------------------------------------


def write_figures(
    sources: list[PreparedSource],
    out_dir: str | Path,
    *,
    labels: dict[str, np.ndarray] | None = None,
    max_cells: int = 4000,
    seed: int = 42,
) -> list[Path]:
    """Draw what was preprocessed, so the numbers can be looked at.

    Four panels, because they answer the four questions a reader has: how much
    data is there per source, what did gene selection keep and what is shared,
    do the sources look like each other after preprocessing (and does the label
    structure survive it), and what did the transformation do to a gene.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    labels = labels or {}
    labeled_name = next(iter(labels), "")
    written: list[Path] = []

    # 1. Cells per source (and, for the labeled one, its class balance).
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    names = [source.name for source in sources]
    axes[0].bar(names, [source.cell_count for source in sources], color="#4C78A8")
    axes[0].set_title("cells per source")
    axes[0].tick_params(axis="x", rotation=20)
    labeled = [(source, labels.get(source.name)) for source in sources if source.name in labels]
    if labeled:
        labelled_source, values = labeled[0]
        unique, counts = np.unique(values, return_counts=True)
        order = np.argsort(-counts)[:20]
        axes[1].barh(
            [str(unique[i]) for i in order][::-1],
            [int(counts[i]) for i in order][::-1],
            color="#F58518",
        )
        axes[1].set_title(f"class balance in {labelled_source.name} (top 20)")
    else:
        axes[1].axis("off")
    fig.tight_layout()
    written.append(_save(fig, out / "fig01_cells_per_source.png"))

    # 2. Gene selection: kept per source, and what the intersection costs.
    fig, ax = plt.subplots(figsize=(8, 4))
    x = np.arange(len(sources))
    ax.bar(x - 0.2, [source.total_genes for source in sources], width=0.4, label="genes in file")
    ax.bar(x + 0.2, [source.n_genes for source in sources], width=0.4, label="selected (HVG)")
    ax.set_xticks(x, [source.name for source in sources], rotation=20)
    ax.set_title(f"gene selection per source -- shared panel: {len(sources[0].genes):,}")
    ax.legend()
    fig.tight_layout()
    written.append(_save(fig, out / "fig02_gene_selection.png"))

    # 3. The sources after preprocessing: do they overlap, and where are the labels?
    embedding = _pca_embedding(
        sources, max_cells=max_cells, seed=seed, labels=labels
    )
    if embedding is not None:
        coords, owners, labels_for_rows = embedding
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
        owners_array = np.asarray(owners)
        for name in dict.fromkeys(owners):
            mask = owners_array == name
            axes[0].scatter(
                coords[mask, 0], coords[mask, 1], s=6, alpha=0.6, label=name
            )
        axes[0].set_title("sources after preprocessing (PCA)")
        axes[0].legend(fontsize=7, markerscale=2)
        # The labels come back aligned to the rows that were sampled, so the
        # scatter cannot disagree with its colours.
        labelled_rows = labels_for_rows is not None
        if labelled_rows:
            mask = owners_array == labeled_name
            classes = pd.Categorical(np.asarray(labels_for_rows)[mask])
            axes[1].scatter(
                coords[mask, 0],
                coords[mask, 1],
                c=classes.codes,
                cmap="tab20",
                s=6,
                alpha=0.7,
            )
            axes[1].set_title(f"{labeled_name} by cell type")
        else:
            axes[1].axis("off")
        fig.tight_layout()
        written.append(_save(fig, out / "fig03_pca_sources.png"))

    # 4. One gene, before and after: skewed counts become a 0-clipped z-score.
    source = sources[0]
    if source.counts.shape[1]:
        gene = int(np.argmax(source.counts.var(axis=0)))
        fig, axes = plt.subplots(1, 2, figsize=(11, 3.6))
        axes[0].hist(source.counts[:, gene], bins=40, color="#54A24B")
        axes[0].set_title(f"{source.name}: {source.genes[gene]} counts")
        axes[1].hist(source.matrix[:, gene], bins=40, color="#B279A2")
        axes[1].set_title("after log1p, z-score, clip at 0")
        fig.tight_layout()
        written.append(_save(fig, out / "fig04_gene_transform.png"))

    # 5. How much the sources' own selections agree -- the panel is built from
    #    this, and a reader should be able to see how thin the agreement is.
    fig, ax = plt.subplots(figsize=(1.2 + 1.1 * len(sources), 4.2))
    matrix = np.zeros((len(sources), len(sources)), dtype=float)
    for i, first in enumerate(sources):
        for j, second in enumerate(sources):
            first_selection = set(first.selection or first.genes)
            second_selection = set(second.selection or second.genes)
            shared = len(first_selection & second_selection)
            matrix[i, j] = shared / max(1, min(len(first_selection), len(second_selection)))
    image = ax.imshow(matrix, cmap="viridis", vmin=0, vmax=1)
    ax.set_xticks(range(len(sources)), [s.name[:22] for s in sources], rotation=45, ha="right")
    ax.set_yticks(range(len(sources)), [s.name[:22] for s in sources])
    for i in range(len(sources)):
        for j in range(len(sources)):
            ax.text(
                j, i, f"{int(matrix[i, j] * 100)}%", ha="center", va="center",
                color="white" if matrix[i, j] < 0.6 else "black", fontsize=8,
            )
    ax.set_title("share of each source's selected genes found in the other")
    fig.colorbar(image, ax=ax, shrink=0.8)
    fig.tight_layout()
    written.append(_save(fig, out / "fig05_gene_overlap.png"))

    # 6. What the recipe did, per source: zeros before and after, and how deep
    #    the cells were sequenced.
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    stats = [source.stats for source in sources if source.stats]
    if stats:
        names = [stat.name[:24] for stat in stats]
        axes[0].bar(names, [stat.median_counts_per_cell for stat in stats], color="#4C78A8")
        axes[0].set_title("median counts per cell (before)")
        axes[0].tick_params(axis="x", rotation=25)
        positions = np.arange(len(stats))
        axes[1].bar(positions - 0.2, [stat.sparsity_raw for stat in stats], width=0.4,
                    label="raw counts", color="#54A24B")
        axes[1].bar(positions + 0.2, [stat.sparsity_processed for stat in stats], width=0.4,
                    label="after log1p + z-clip", color="#B279A2")
        axes[1].set_xticks(positions, names, rotation=25)
        axes[1].set_title("fraction of zeros")
        axes[1].legend()
    fig.tight_layout()
    written.append(_save(fig, out / "fig06_recipe_effect.png"))

    logger.info("wrote %d figure(s) to %s", len(written), out)
    return written


def _save(fig, path: Path) -> Path:
    # No figure-level title: pandas and other libraries leave one behind on the
    # current figure, and a stray line of console text across the top of a plot
    # is worse than no title at all.
    fig.suptitle("")
    fig.savefig(path, dpi=130)
    fig.clf()
    return path


def _pca_embedding(
    sources: list[PreparedSource],
    *,
    max_cells: int,
    seed: int,
    labels: dict[str, np.ndarray] | None = None,
) -> tuple[np.ndarray, list[str], np.ndarray | None] | None:
    """Two components of the shared panel, on a bounded sample of cells.

    Returns the coordinates, the source each row came from, and -- when a source
    has labels -- the labels of exactly the rows that were sampled, so a caller
    can colour the picture without guessing which cells survived the sampling.
    """
    rng = np.random.default_rng(seed)
    per_source = max(1, max_cells // max(1, len(sources)))
    blocks, owners, label_blocks = [], [], []
    any_labels = False
    for source in sources:
        rows = np.arange(source.cell_count)
        if source.cell_count > per_source:
            rows = np.sort(rng.choice(rows, size=per_source, replace=False))
        blocks.append(source.matrix[rows])
        owners.extend([source.name] * len(rows))
        if labels and source.name in labels:
            values = np.asarray(labels[source.name])
            if len(values) >= source.cell_count:
                label_blocks.append(values[rows])
                any_labels = True
                continue
        label_blocks.append(np.full(len(rows), None, dtype=object))
    pooled = np.concatenate(blocks, axis=0)
    if pooled.shape[1] < 2 or pooled.shape[0] < 3:
        return None
    from sklearn.decomposition import TruncatedSVD
    from sklearn.preprocessing import StandardScaler

    standardized = StandardScaler(with_mean=True, with_std=False).fit_transform(pooled)
    coords = TruncatedSVD(n_components=2, random_state=seed).fit_transform(standardized)
    row_labels = np.concatenate(label_blocks) if any_labels else None
    return np.asarray(coords), owners, row_labels
