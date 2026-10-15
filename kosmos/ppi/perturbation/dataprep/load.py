"""Read a sample's files into a full cells x genes count matrix.

The fetcher's single-cell reader writes a bounded preview table for review;
stage 2 needs every cell. One function per layout, one entry point:

    load_sample(files) -> Sample(counts, barcodes, genes)

    10x mtx triplet   `<prefix>matrix.mtx(.gz)` + barcodes + features/genes
    10x h5            CellRanger v2 (genome group) or v3 (`matrix/features`)
    h5ad              AnnData (`X`, or `raw/X`, or a `counts` layer)

`call_cells` keeps barcodes with enough UMIs: GEO matrices are often unfiltered
(GSE249595: 6.8M barcodes per channel for ~55k cells).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import io as sio
from scipy import sparse


@dataclass
class Sample:
    counts: sparse.csr_matrix  # cells x genes, raw counts
    barcodes: list[str]
    genes: list[str]  # symbols
    source: str

    @property
    def n_cells(self) -> int:
        return self.counts.shape[0]


def _decode(value) -> str:
    return value.decode() if isinstance(value, bytes) else str(value)


def _names(path: Path, column: int) -> list[str]:
    table = pd.read_csv(path, sep="\t", header=None, dtype=str)
    column = min(column, table.shape[1] - 1)
    return table[column].astype(str).tolist()


def load_mtx(matrix: Path, barcodes: Path, features: Path) -> Sample:
    counts = sio.mmread(matrix).tocsr()
    cells, genes = _names(barcodes, 0), _names(features, 1)  # features: id, symbol, type
    if counts.shape == (len(genes), len(cells)):
        counts = counts.T.tocsr()
    elif counts.shape != (len(cells), len(genes)):
        raise ValueError(f"{matrix.name} is {counts.shape}, not {len(cells)} cells x {len(genes)} genes")
    return Sample(counts.astype(np.float32), cells, genes, matrix.name)


def load_h5(path: Path) -> Sample:
    """CellRanger HDF5 (v2/v3), or an AnnData file under any suffix."""
    import h5py

    with h5py.File(path, "r") as handle:
        group = next(
            (g for g in handle.values() if isinstance(g, h5py.Group) and {"data", "indices", "indptr", "shape"} <= set(g)),
            None,
        )
        if group is not None:
            n_genes, n_cells = (int(v) for v in group["shape"][:])
            counts = sparse.csc_matrix(
                (group["data"][:], group["indices"][:], group["indptr"][:]), shape=(n_genes, n_cells)
            ).T.tocsr()
            names = group["features"]["name"][:] if "features" in group else group["gene_names"][:]
            return Sample(
                counts.astype(np.float32),
                [_decode(v) for v in group["barcodes"][:]],
                [_decode(v) for v in names],
                path.name,
            )
    import anndata

    adata = anndata.read_h5ad(path)
    matrix = adata.layers["counts"] if "counts" in adata.layers else (adata.raw.X if adata.raw is not None else adata.X)
    var = adata.raw.var if adata.raw is not None and "counts" not in adata.layers else adata.var
    symbol = next((c for c in ("gene_name", "gene_symbols", "feature_name", "symbol") if c in var.columns), None)
    genes = var[symbol].astype(str).tolist() if symbol else var.index.astype(str).tolist()
    return Sample(sparse.csr_matrix(matrix, dtype=np.float32), adata.obs_names.astype(str).tolist(), genes, path.name)


_MATRIX = re.compile(r"matrix\.mtx(\.gz)?$", re.IGNORECASE)


def load_sample(files: list[Path]) -> Sample:
    """The sample in `files`: an mtx triplet (matched by prefix) or an h5 file."""
    files = [Path(f) for f in files]
    matrices = [f for f in files if _MATRIX.search(f.name)]
    if matrices:
        matrix = matrices[0]
        prefix = _MATRIX.sub("", matrix.name).lower()
        siblings = [f for f in files if f != matrix and f.name.lower().startswith(prefix)] or files

        def pick(*words: str) -> Path:
            found = [f for f in siblings if any(w in f.name.lower() for w in words)]
            if not found:
                raise ValueError(f"{matrix.name}: no {'/'.join(words)} file beside it")
            return found[0]

        return load_mtx(matrix, pick("barcode"), pick("feature", "gene"))
    h5 = [f for f in files if f.suffix.lower() in (".h5", ".h5ad")]
    if h5:
        return load_h5(h5[0])
    raise ValueError(f"no matrix in {[f.name for f in files]}")


def call_cells(sample: Sample, min_umis: int) -> tuple[Sample, int]:
    """Keep barcodes with >= `min_umis` counts; the number of barcodes before."""
    umis = np.asarray(sample.counts.sum(axis=1)).ravel()
    keep = np.flatnonzero(umis >= min_umis)
    before = sample.n_cells
    return Sample(sample.counts[keep], [sample.barcodes[i] for i in keep], sample.genes, sample.source), before


def gene_means(counts: sparse.spmatrix | np.ndarray, target_sum: float = 1e4) -> np.ndarray:
    """Mean log1p(CPM) per gene."""
    counts = sparse.csr_matrix(counts, dtype=np.float64)
    totals = np.asarray(counts.sum(axis=1)).ravel()
    scaled = sparse.diags(target_sum / np.where(totals > 0, totals, 1.0)) @ counts
    scaled.data = np.log1p(scaled.data)
    return np.asarray(scaled.mean(axis=0)).ravel()


def similarity(reference: dict[str, float], sample: Sample, target_sum: float = 1e4) -> dict[str, float]:
    """How alike a sample's expression is to a reference profile (gene -> mean
    log1p-CPM): Pearson / Spearman of gene means over the genes both measure and
    the reference expresses, plus the facts that betray a different assay."""
    means = gene_means(sample.counts, target_sum)
    own: dict[str, float] = {}
    for gene, value in zip(sample.genes, means):
        own.setdefault(gene, float(value))
    shared = [g for g in reference if g in own and reference[g] > 0.1]
    a = np.array([reference[g] for g in shared])
    b = np.array([own[g] for g in shared])
    ribosomal = [i for i, g in enumerate(sample.genes) if re.match(r"^RP[SL]\d", g)]
    totals = np.asarray(sample.counts.sum(axis=1)).ravel()
    ribo = np.asarray(sample.counts[:, ribosomal].sum(axis=1)).ravel() if ribosomal else np.zeros_like(totals)
    return {
        "shared_genes": len(shared),
        "pearson": float(np.corrcoef(a, b)[0, 1]) if len(shared) > 2 else float("nan"),
        "spearman": float(pd.Series(a).rank().corr(pd.Series(b).rank())) if len(shared) > 2 else float("nan"),
        "median_umis": float(np.median(totals)) if len(totals) else 0.0,
        "median_genes": float(np.median(np.diff(sample.counts.indptr))) if sample.n_cells else 0.0,
        "ribosomal_fraction": float(np.median(ribo / np.where(totals > 0, totals, 1))) if len(totals) else 0.0,
    }
