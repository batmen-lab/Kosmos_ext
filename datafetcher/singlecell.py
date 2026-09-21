"""Turning single-cell files into a table the rest of the pipeline can read.

Single-cell data is not tabular on disk. It arrives as an HDF5 `.h5ad`
(AnnData), or as the CellRanger triplet (`matrix.mtx` + `barcodes.tsv` +
`features.tsv`), usually inside a `.tar`. Everything downstream of the fetcher
speaks one language -- one row per observation, one column per measurement, a
label column -- so this module translates, and does it by *selecting first and
reading second*:

  * the label column and any id columns are read from `obs`, never the matrix;
  * cells are chosen by a bounded, stratified sample -- the one dimension that
    is capped, because a 90,000-cell file is not a table anyone trains on;
  * genes are *not* cut: the panel is part of the data for a single-cell
    question, and 2,000 x 14,089 is an ordinary table (a cap exists for a file
    that genuinely would not fit, and the file's own `highly_variable` flag is
    honoured when it has one);
  * only that slab of `X` is read.

A 50,000 x 20,000 file therefore costs a few thousand cells' worth of values in
memory, not 10^9. The output is a derived CSV, which is how the fetcher already
describes "this file came out of that one", and what was kept -- cells written
against cells on disk, genes written against genes on disk -- is recorded as
`SingleCellFacts` so a reviewer of the derived table can see how much of the file
it is.

`anndata` itself is deliberately not a dependency: an `.h5ad` is an HDF5 file
with a documented layout, and reading it through `h5py` is both lighter and the
only way to keep the reads selective.
"""

from __future__ import annotations

import gzip
import json
import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

#: Suffixes that say "this is a single-cell container".
H5AD_SUFFIXES = (".h5ad", ".h5ad.gz")
MTX_SUFFIXES = (".mtx", ".mtx.gz")
#: Columns worth carrying into the table besides the label: they identify a row
#: and they let a reader trace a cell back to its donor.
DEFAULT_ID_COLUMNS = ("donor_id", "donor", "sample", "sample_id", "batch")
#: A perturbation table's label is rarely called `cell_type`: the McFarland
#: files carry the drug in `compound_1`/`condition`. So the id list is not the
#: only thing worth carrying -- any obs column whose values could label a row
#: travels with the table too. A column with more levels than this is an
#: identifier or a measurement, not a label.
MAX_METADATA_LEVELS = 50
#: How many such columns to carry. A file with hundreds of metadata columns
#: would make a wide table wider than the genes justify.
MAX_METADATA_COLUMNS = 25
#: Groups inside `obs` that are not columns: the row index and the level table
#: AnnData keeps for the columns written as codes.
OBS_CONTAINERS = frozenset({"_index", "__categories"})


@dataclass
class SingleCellFacts:
    """What the pre-check saw in one file, and what it kept.

    The derived table looks like any other CSV downstream, and it is one -- but
    it is a *sample* of a file that may be forty times its size. A reviewer
    looking at the table's head cannot tell that, so the numbers travel with the
    table: cells on disk against cells written, genes on disk against genes
    written, and how each was chosen.
    """

    #: The file as it is on disk.
    source: str
    source_cells: int
    source_genes: int
    #: The table that was written from it.
    cells_written: int
    genes_written: int
    #: The label column, and whether the file actually had it.
    label_column: str
    label_present: bool
    #: How the cells were chosen ("all cells", "stratified sample of 2,000...").
    cell_sampling: str
    #: How the genes were chosen ("every gene", "the file's highly_variable...").
    gene_selection: str
    #: Other obs columns carried with the table: the ones that could label a
    #: row, when the task's label is not the file's `cell_type` but a treatment
    #: or a condition.
    metadata_columns: tuple[str, ...] = ()
    seed: int = 42

    @property
    def cells_were_capped(self) -> bool:
        return self.cells_written < self.source_cells

    @property
    def genes_were_capped(self) -> bool:
        return self.genes_written < self.source_genes

    def to_dict(self) -> dict:
        from dataclasses import asdict

        return asdict(self)

    def describe(self) -> str:
        """One readable line a person or a model can act on."""
        return (
            f"{self.cells_written:,} of {self.source_cells:,} cells "
            f"({self.cell_sampling}), {self.genes_written:,} of "
            f"{self.source_genes:,} genes ({self.gene_selection}), "
            + (
                f"label column {self.label_column!r} is present"
                if self.label_present
                else f"no {self.label_column!r} column"
            )
            + (
                f", {len(self.metadata_columns)} metadata column(s): "
                f"{', '.join(self.metadata_columns[:6])}"
                if self.metadata_columns
                else ""
            )
        )

    def precheck_note(self) -> str:
        """The line the review packet carries about this table's provenance."""
        cells = (
            f"kept {self.cells_written:,} of {self.source_cells:,} cells "
            f"({self.cell_sampling})"
            if self.cells_were_capped
            else f"kept all {self.cells_written:,} cells"
        )
        genes = (
            f"kept {self.genes_written:,} of {self.source_genes:,} genes "
            f"({self.gene_selection})"
            if self.genes_were_capped
            else f"kept all {self.genes_written:,} genes -- the panel is not cut"
        )
        return (
            f"this table was converted from {self.source}: {cells}, {genes}"
            + (
                f", label column {self.label_column!r} is present"
                if self.label_present
                else f", there is no {self.label_column!r} column to label it with"
            )
        )


@dataclass
class SingleCellSelection:
    """How much of a file to turn into a table."""

    #: Cells in the output table.
    max_cells: int = 2000
    #: Genes in the output table. 0 -- the default -- means every gene the file
    #: has: for a single-cell question the gene panel is part of the data, not a
    #: knob, and a capped panel silently changes what the model can learn from.
    max_genes: int = 0
    #: Cells read to rank genes by variance.
    scan_cells: int = 5000
    #: The column that makes this table trainable, when the file has it.
    label_column: str = "cell_type"
    seed: int = 42
    #: Genes to keep whatever their variance (a marker panel, when one is given).
    required_genes: tuple[str, ...] = ()
    notes: list[str] = field(default_factory=list)
    #: Filled in by the reader: what the file held, what the table kept.
    facts: SingleCellFacts | None = None

    def to_dict(self) -> dict:
        return {
            "max_cells": self.max_cells,
            "max_genes": self.max_genes,
            "scan_cells": self.scan_cells,
            "label_column": self.label_column,
            "seed": self.seed,
            "required_genes": list(self.required_genes),
        }


def is_single_cell(path: str | Path) -> bool:
    lowered = str(path).lower()
    return lowered.endswith(H5AD_SUFFIXES) or lowered.endswith(MTX_SUFFIXES)


def _open_text(path: Path):
    return gzip.open(path, "rt") if path.suffix == ".gz" else path.open("rt")


# --- h5ad ----------------------------------------------------------------x


def _h5_column(group, name: str) -> np.ndarray:
    """One column of an AnnData dataframe group (`obs`/`var`).

    AnnData has written categoricals two ways: older files keep
    `categories`/`codes` inside the column's own group, newer ones store the
    codes in `obs/<name>` and the levels in `obs/__categories/<name>`. Both are
    read here, because a real file is whichever version wrote it.
    """
    if name not in group:
        raise KeyError(name)
    node = group[name]
    if (
        "__categories" in group
        and name in group["__categories"]
        and not hasattr(node, "keys")
    ):
        categories = np.asarray(
            [_decode(value) for value in group["__categories"][name][:]], dtype=object
        )
        return _from_codes(categories, np.asarray(node[:]))
    if hasattr(node, "keys") and "categories" in node and "codes" in node:
        categories = np.asarray([_decode(value) for value in node["categories"][:]])
        return _from_codes(categories, np.asarray(node["codes"][:]))
    return np.asarray([_decode(value) for value in node[:]], dtype=object)


def _from_codes(categories: np.ndarray, codes: np.ndarray) -> np.ndarray:
    values = np.empty(len(codes), dtype=object)
    values[:] = None
    present = codes >= 0
    values[present] = categories[codes[present]]
    return values


def _decode(value) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8", "replace")
    return str(value)


def _obs_columns(handle, wanted: list[str]) -> tuple[dict[str, np.ndarray], list[str]]:
    """Only the columns asked for, read straight out of the HDF5 group."""
    obs = handle["obs"]
    available = set(obs.keys()) - OBS_CONTAINERS
    read: dict[str, np.ndarray] = {}
    for name in wanted:
        if name in available:
            read[name] = _h5_column(obs, name)
    return read, sorted(available)


def _obs_kind(group, name: str) -> str:
    """What kind of quantity an obs column holds, without reading it all."""
    node = group[name]
    if hasattr(node, "keys") and "categories" in node:
        return "category"
    if "__categories" in group and name in group["__categories"]:
        return "category"
    if not hasattr(node, "dtype"):
        return "unknown"
    if node.dtype.kind in {"i", "u", "f", "b", "c"}:
        return "number"
    return "text"


def _metadata_columns(
    group,
    wanted: Sequence[str],
    *,
    max_levels: int = MAX_METADATA_LEVELS,
    max_columns: int = MAX_METADATA_COLUMNS,
) -> tuple[dict[str, np.ndarray], list[str], list[str]]:
    """The obs columns that could label a row, besides the ones already wanted.

    The label of a task is often not `cell_type`: a chemical-perturbation table
    keeps the treatment in `compound_1`/`condition` and the cell line in
    `cell_context`. Carrying only the fixed id list threw those away at
    conversion time, so `--hint compound_1` could never match anything -- the
    column was gone before the question was asked.

    Returns `(columns, kept names, names left out)`. Numbers stay in the file:
    a dose or a timepoint is a measurement, and a measurement beside the gene
    panel would be read as one more feature.
    """
    kept: dict[str, np.ndarray] = {}
    kept_names: list[str] = []
    left_out: list[str] = []
    already = set(wanted)
    for name in sorted(set(group.keys()) - OBS_CONTAINERS):
        if name in already:
            continue
        if _obs_kind(group, name) in {"number", "unknown"}:
            continue
        values = _h5_column(group, name)
        levels = int(pd.Series(values).dropna().nunique())
        # One level per row is a barcode or a file name, and no level at all is
        # an empty column: neither can label anything.
        if levels < 2 or levels > max_levels or levels == len(values):
            continue
        if len(kept_names) >= max_columns:
            left_out.append(name)
            continue
        kept[name] = values
        kept_names.append(name)
    return kept, kept_names, left_out


def _var_names(handle) -> np.ndarray:
    var = handle["var"]
    if "feature_name" in var:
        return np.asarray([_decode(value) for value in var["feature_name"][:]], dtype=object)
    if "_index" in var:
        return np.asarray([_decode(value) for value in var["_index"][:]], dtype=object)
    return np.asarray([_decode(value) for value in var[list(var.keys())[0]][:]], dtype=object)


def _x_shape(handle) -> tuple[int, int]:
    node = handle["X"] if "X" in handle else handle["raw/X"]
    if hasattr(node, "shape"):
        return int(node.shape[0]), int(node.shape[1])
    # A sparse X is a group whose shape lives in its attributes.
    shape = node.attrs.get("shape")
    if shape is None:
        shape = node["shape"][:]
    shape = np.asarray(shape).reshape(-1)
    return int(shape[0]), int(shape[1])


def _x_dense(node) -> bool:
    return hasattr(node, "shape") and not (hasattr(node, "keys") and "data" in node)


def _read_rows(handle, rows: np.ndarray, columns: np.ndarray) -> np.ndarray:
    """`X[rows][:, columns]` without reading the rest of `X`."""
    node = handle["X"] if "X" in handle else handle["raw/X"]
    if _x_dense(node):
        # h5py wants increasing indices; the caller keeps them sorted. Read a
        # band of rows at a time: the whole selection never has to be in memory
        # at once, which is the difference between 20 MB and 160 MB on a
        # 2,000 x 20,000 slice.
        out = np.empty((len(rows), len(columns)), dtype=np.float32)
        band = max(1, min(len(rows), 256))
        for start in range(0, len(rows), band):
            block = node[rows[start : start + band], :][:, columns]
            out[start : start + band] = np.asarray(block, dtype=np.float32)
        return out
    encoding = node.attrs.get("encoding-type", "")
    encoding = _decode(encoding) if isinstance(encoding, bytes) else str(encoding)
    indptr = node["indptr"][:]
    data = node["data"]
    indices = node["indices"]
    out = np.zeros((len(rows), len(columns)), dtype=np.float32)
    positions = {int(column): i for i, column in enumerate(columns)}
    if "csc" in encoding:
        raise ValueError(
            "CSC-encoded X is not supported by the selective reader; "
            "transpose the file or export CSR"
        )
    for out_row, source_row in enumerate(rows):
        start, stop = int(indptr[source_row]), int(indptr[source_row + 1])
        row_columns = indices[start:stop]
        row_values = data[start:stop]
        for column, value in zip(row_columns, row_values, strict=False):
            target = positions.get(int(column))
            if target is not None:
                out[out_row, target] = float(value)
    return out


def _gene_variance(handle, columns: np.ndarray, scan_cells: int, seed: int) -> np.ndarray:
    """Per-gene variance from a bounded sample of cells."""
    n_cells, _ = _x_shape(handle)
    rng = np.random.default_rng(seed)
    rows = (
        np.arange(n_cells)
        if n_cells <= scan_cells
        else np.sort(rng.choice(n_cells, size=scan_cells, replace=False))
    )
    slab = _read_rows(handle, rows, columns)
    return slab.var(axis=0)


def read_h5ad(path: str | Path, selection: SingleCellSelection) -> pd.DataFrame:
    """A bounded table from an `.h5ad`: chosen cells, chosen genes, chosen obs."""
    path = Path(path)
    plain = _decompressed_copy(path)
    try:
        return _read_h5ad_file(plain, path, selection)
    finally:
        if plain != path and plain.exists():
            # A GEO `.h5ad.gz` is 600 MB compressed and 625 MB expanded; the
            # expanded copy exists only long enough to be read selectively, and
            # the `.gz` stays the artifact of record.
            plain.unlink(missing_ok=True)


def _decompressed_copy(path: Path) -> Path:
    """h5py cannot read a gzipped HDF5 file; give it one it can, briefly."""
    if not path.name.lower().endswith(".gz"):
        return path
    target = path.with_suffix("")
    if target.exists() and target.stat().st_size:
        return target
    with gzip.open(path, "rb") as source, target.open("wb") as sink:
        while chunk := source.read(8 * 1024 * 1024):
            sink.write(chunk)
    return target


def _read_h5ad_file(path: Path, original: Path, selection: SingleCellSelection) -> pd.DataFrame:
    import h5py

    with h5py.File(path, "r") as handle:
        n_cells, n_genes = _x_shape(handle)
        names = _var_names(handle)
        if len(names) != n_genes:
            names = np.asarray([f"gene_{i}" for i in range(n_genes)], dtype=object)

        wanted_obs = [selection.label_column, *DEFAULT_ID_COLUMNS]
        obs, available_obs = _obs_columns(handle, wanted_obs)
        # Beside the label and the ids, carry the obs columns that could label a
        # row: a drug-perturbation table keeps its treatment there.
        metadata, metadata_names, metadata_left_out = _metadata_columns(
            handle["obs"], wanted_obs
        )
        obs.update(metadata)
        if selection.label_column not in obs:
            # An unlabeled single-cell file is not a failure: it is exactly the
            # evidence the supplementary round looks for. Write the table and
            # say the label is missing.
            selection.notes.append(
                f"{original.name} has no {selection.label_column!r} column in obs "
                f"(it has {available_obs[:12]}), so the table has no label column"
            )
        labels = obs.get(selection.label_column)
        if labels is None:
            labels = np.asarray(["unlabeled"] * n_cells, dtype=object)

        # Genes: the file's own flag when it has one; otherwise the whole panel,
        # unless a cap was set (0 means no cap, and that is the default).
        candidates = np.arange(n_genes)
        highly_variable = None
        if "highly_variable" in handle["var"]:
            highly_variable = np.asarray(handle["var"]["highly_variable"][:], dtype=bool)
            candidates = candidates[highly_variable]
        if selection.max_genes and len(candidates) > selection.max_genes:
            variance = _gene_variance(handle, candidates, selection.scan_cells, selection.seed)
            order = np.argsort(-variance)
            chosen = list(candidates[order[: selection.max_genes]])
            gene_selection = (
                f"the {selection.max_genes:,} highest-variance genes over "
                f"{min(n_cells, selection.scan_cells):,} sampled cells"
            )
        elif highly_variable is not None:
            chosen = list(candidates)
            gene_selection = "every gene the file's own highly_variable flag kept"
        else:
            chosen = list(candidates)
            gene_selection = "every gene"
        by_name = {str(name): index for index, name in enumerate(names)}
        for gene in selection.required_genes:
            index = by_name.get(str(gene))
            if index is not None and index not in chosen:
                chosen.append(index)
        columns = np.sort(np.asarray(chosen, dtype=int))

        # Cells: a stratified sample, so a rare type is not sampled away.
        rng = np.random.default_rng(selection.seed)
        rows = _stratified_rows(labels, selection.max_cells, rng)
        slab = _read_rows(handle, rows, columns)

    table = pd.DataFrame(slab, columns=[str(names[index]) for index in columns])
    for name, values in obs.items():
        table[name] = values[rows]
    cell_sampling = (
        "all cells in the file"
        if len(rows) >= n_cells
        else f"a stratified sample of {len(rows):,} cells (seed {selection.seed})"
    )
    selection.facts = SingleCellFacts(
        source=original.name,
        source_cells=n_cells,
        source_genes=n_genes,
        cells_written=len(rows),
        genes_written=len(columns),
        label_column=selection.label_column,
        label_present=selection.label_column in obs,
        cell_sampling=cell_sampling,
        gene_selection=gene_selection,
        metadata_columns=tuple(metadata_names),
        seed=selection.seed,
    )
    if metadata_names:
        selection.notes.append(
            f"{original.name}: also carried {len(metadata_names)} obs column(s) "
            f"that could label a row ({', '.join(metadata_names[:8])}"
            + (", ..." if len(metadata_names) > 8 else "")
            + ")"
            + (
                f"; {len(metadata_left_out)} more were past the "
                f"{MAX_METADATA_COLUMNS}-column cap"
                if metadata_left_out
                else ""
            )
        )
    selection.notes.append(
        f"{original.name}: {n_cells:,} cells x {n_genes:,} genes on disk; wrote "
        f"{len(rows):,} cells x {len(columns):,} genes"
        + f" ({cell_sampling}; {gene_selection})"
    )
    return table


def _stratified_rows(labels: np.ndarray, max_cells: int, rng) -> np.ndarray:
    """A bounded sample that keeps every label, proportional to its frequency."""
    values = pd.Series(labels).astype(str)
    if len(values) <= max_cells:
        return np.arange(len(values))
    groups = values.groupby(values).groups
    total = len(values)
    picked: list[int] = []
    for _, members in groups.items():
        share = max(1, int(round(max_cells * len(members) / total)))
        indices = np.asarray(members)
        picked.extend(
            rng.choice(indices, size=min(share, len(indices)), replace=False).tolist()
        )
    if len(picked) < max_cells:
        # Rounding gives or takes a few; spend the rest of the budget so the
        # table is the size the caller asked for.
        remaining = np.setdiff1d(np.arange(total), np.asarray(picked), assume_unique=False)
        top_up = rng.choice(remaining, size=max_cells - len(picked), replace=False)
        picked.extend(int(index) for index in top_up)
    return np.sort(np.asarray(picked[:max_cells], dtype=int))


# --- CellRanger triplet ----------------------------------------------------


_MTX_HEADER = re.compile(r"^\s*(\d+)\s+(\d+)\s+(\d+)\s*$")


def _mtx_components(path: Path) -> dict[str, Path] | None:
    """The barcodes/features/metadata files that belong to an `.mtx`."""
    directory = path.parent
    stem = path.name
    for suffix in MTX_SUFFIXES:
        if stem.lower().endswith(suffix):
            stem = stem[: -len(suffix)]
            break
    found: dict[str, Path] = {"matrix": path}
    for key, patterns in (
        ("barcodes", ("barcodes.tsv", "barcodes.tsv.gz", "barcodes.txt")),
        ("features", ("features.tsv", "features.tsv.gz", "genes.tsv", "genes.tsv.gz")),
    ):
        for pattern in patterns:
            candidate = directory / pattern
            if candidate.exists():
                found[key] = candidate
                break
        else:
            # CellRanger v2 names them `<matrix>_barcodes.tsv` style.
            for pattern in patterns:
                candidate = directory / f"{stem}_{pattern}"
                if candidate.exists():
                    found[key] = candidate
                    break
    if "barcodes" not in found or "features" not in found:
        return None
    return found


def _read_lines(path: Path) -> list[str]:
    with _open_text(path) as handle:
        return [line.rstrip("\n") for line in handle if line.strip()]


def read_mtx(
    path: str | Path,
    selection: SingleCellSelection,
    *,
    metadata: str | Path | None = None,
) -> pd.DataFrame:
    """A bounded table from a CellRanger triplet.

    The triplet has no labels; the label comes from a metadata table beside it
    (barcode -> column). Without one the table is still written -- it is exactly
    the unlabeled evidence the supplementary round wants -- and the caller sees
    no label column.
    """
    path = Path(path)
    components = _mtx_components(path)
    if components is None:
        raise ValueError(
            f"{path.name} is a matrix-market file without its barcodes/features "
            f"beside it; a bare .mtx is not a table"
        )
    barcodes = [line.split("\t")[0] for line in _read_lines(components["barcodes"])]
    feature_lines = _read_lines(components["features"])
    features = [line.split("\t")[0] for line in feature_lines]
    names = [
        line.split("\t")[1] if "\t" in line else line.split("\t")[0]
        for line in feature_lines
    ]

    with _open_text(path) as handle:
        header = None
        for line in handle:
            if line.startswith("%"):
                continue
            header = _MTX_HEADER.match(line)
            break
        if header is None:
            raise ValueError(f"{path.name} has no matrix-market size line")
        n_rows, n_cols, _ = (int(part) for part in header.groups())

    cells_axis_is_rows = n_rows == len(barcodes) and n_cols == len(features)
    if not cells_axis_is_rows and not (n_rows == len(features) and n_cols == len(barcodes)):
        raise ValueError(
            f"{path.name} is {n_rows}x{n_cols} but has {len(barcodes)} barcodes and "
            f"{len(features)} features; the triplet does not line up"
        )
    cell_count = len(barcodes)
    gene_count = len(features)

    labels = _metadata_labels(metadata, barcodes, selection.label_column)
    rng = np.random.default_rng(selection.seed)
    rows = _stratified_rows(
        labels if labels is not None else np.asarray(["unknown"] * cell_count),
        selection.max_cells,
        rng,
    )
    if labels is None and cell_count > selection.max_cells:
        rows = np.sort(rng.choice(cell_count, size=selection.max_cells, replace=False))

    # Pass one: per-gene sums over a bounded sample of cells. Nothing is stored
    # but the two accumulators, so a 10-million-nonzero triplet costs 16 MB of
    # arrays rather than a list of tuples.
    scanned = (
        rows
        if len(rows) >= min(cell_count, selection.scan_cells)
        else np.sort(rng.choice(cell_count, size=min(cell_count, selection.scan_cells), replace=False))
    )
    sums = np.zeros(gene_count, dtype=np.float64)
    sumsq = np.zeros(gene_count, dtype=np.float64)
    scanned_lookup = {int(cell) for cell in scanned}
    _stream_triplets(path, n_rows, n_cols, _accumulate(scanned_lookup, cells_axis_is_rows, sums, sumsq))
    n_scanned = max(1, len(scanned))
    variance = sumsq / n_scanned - (sums / n_scanned) ** 2
    order = np.argsort(-variance)
    chosen = sorted(
        int(index)
        for index in (order[: selection.max_genes] if selection.max_genes else order)
    )
    by_name = {name: index for index, name in enumerate(names)}
    for gene in selection.required_genes:
        index = by_name.get(str(gene))
        if index is not None and index not in chosen:
            chosen.append(index)
    chosen = sorted(set(chosen))

    matrix = np.zeros((len(rows), len(chosen)), dtype=np.float32)
    position = {gene: index for index, gene in enumerate(chosen)}
    row_of = {int(cell): index for index, cell in enumerate(rows)}
    _stream_triplets(
        path,
        n_rows,
        n_cols,
        _fill(matrix, row_of, position, cells_axis_is_rows),
    )

    table = pd.DataFrame(matrix, columns=[names[index] for index in chosen])
    table["barcode"] = [barcodes[index] for index in rows]
    if labels is not None:
        table[selection.label_column] = [labels[index] for index in rows]
    selection.notes.append(
        f"{path.name}: {cell_count:,} cells x {gene_count:,} genes in the triplet; "
        f"wrote {len(rows):,} cells x {len(chosen):,} genes"
        + (
            f"; labels from {Path(metadata).name}"
            if labels is not None
            else "; no metadata table, so the output has no label column"
        )
    )
    selection.facts = SingleCellFacts(
        source=path.name,
        source_cells=cell_count,
        source_genes=gene_count,
        cells_written=len(rows),
        genes_written=len(chosen),
        label_column=selection.label_column,
        label_present=labels is not None,
        cell_sampling=(
            "all cells in the triplet"
            if len(rows) >= cell_count
            else f"a stratified sample of {len(rows):,} cells (seed {selection.seed})"
        ),
        gene_selection=(
            f"the {len(chosen):,} highest-variance genes over "
            f"{len(scanned):,} sampled cells"
            if len(chosen) < gene_count
            else "every gene in the triplet"
        ),
        seed=selection.seed,
    )
    return table


def _stream_triplets(path: Path, n_rows: int, n_cols: int, visit) -> None:
    """Walk a matrix-market file, handing each (row, column, value) to `visit`."""
    with _open_text(path) as handle:
        seen_size = False
        for line in handle:
            if not line.strip() or line.startswith("%"):
                continue
            if not seen_size:
                seen_size = True
                continue
            parts = line.split()
            if len(parts) < 3:
                continue
            row, column, value = int(parts[0]) - 1, int(parts[1]) - 1, float(parts[2])
            if 0 <= row < n_rows and 0 <= column < n_cols:
                visit(row, column, value)


def _accumulate(scanned: set[int], cells_are_rows: bool, sums, sumsq):
    def visit(row: int, column: int, value: float) -> None:
        cell, gene = (row, column) if cells_are_rows else (column, row)
        if cell in scanned:
            sums[gene] += value
            sumsq[gene] += value * value

    return visit


def _fill(matrix, row_of: dict, position: dict, cells_are_rows: bool):
    def visit(row: int, column: int, value: float) -> None:
        cell, gene = (row, column) if cells_are_rows else (column, row)
        out_row = row_of.get(cell)
        out_col = position.get(gene)
        if out_row is not None and out_col is not None:
            matrix[out_row, out_col] = value

    return visit


def _metadata_labels(
    metadata: str | Path | None, barcodes: list[str], label_column: str
) -> np.ndarray | None:
    """Barcode -> label, when a metadata table is there to supply it."""
    if metadata is None:
        return None
    frame = pd.read_csv(metadata, sep=None, engine="python")
    if label_column not in frame.columns:
        return None
    barcode_column = next(
        (name for name in frame.columns if name.lower().strip() in {"barcode", "cell", "cell_id"}),
        frame.columns[0],
    )
    lookup = dict(zip(frame[barcode_column].astype(str), frame[label_column].astype(str), strict=False))
    labels = np.asarray([lookup.get(str(barcode), "") for barcode in barcodes], dtype=object)
    if not any(str(value).strip() for value in labels):
        return None
    return labels


# --- the entry point the fetcher calls -------------------------------------


def convert_single_cell(
    path: str | Path,
    *,
    selection: SingleCellSelection | None = None,
    metadata: str | Path | None = None,
) -> tuple[pd.DataFrame, SingleCellSelection]:
    """Read any supported single-cell file into a bounded table."""
    selection = selection or SingleCellSelection()
    lowered = str(path).lower()
    if lowered.endswith(H5AD_SUFFIXES):
        return read_h5ad(path, selection), selection
    if lowered.endswith(MTX_SUFFIXES):
        return read_mtx(path, selection, metadata=metadata), selection
    raise ValueError(f"{path} is not a single-cell file this reader knows")


def selection_from_config(config) -> SingleCellSelection:
    """The knobs, from the fetcher's config (which reads the environment)."""
    return SingleCellSelection(
        max_cells=int(getattr(config, "single_cell_max_cells", 2000)),
        max_genes=int(getattr(config, "single_cell_max_genes", 0)),
        scan_cells=int(getattr(config, "single_cell_scan_cells", 5000)),
        label_column=str(getattr(config, "single_cell_label_column", "cell_type")),
        seed=int(getattr(config, "single_cell_seed", 42)),
    )


def describe(table: pd.DataFrame) -> str:
    return json.dumps(
        {"rows": int(len(table)), "columns": list(table.columns[:8])}, default=str
    )
