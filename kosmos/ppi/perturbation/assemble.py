"""A screen's labels and its expression often arrive in two files.

A publication is usually one table of *cells* (barcodes, the perturbation each
carried, QC columns) and one table of *measurements* (barcodes by genes), or an
archive of 10x matrices. Neither alone is a training table: the first has the
label and no features, the second the features and no label. `kosmos/ppi` already
speaks in terms of a single table with a condition column and gene columns, so
the two are joined here, once, before anything else looks at them.

The join is on a shared barcode: a column the two tables share, or their index.
Nothing about the perturbation is decided here -- that stays with the contract.

Two things about published screens shape the code:

  * **Orientation is not guaranteed.** GEO ships most count matrices with genes
    in rows and cell barcodes in the header, so the barcodes read as 20,729
    *column names* while the label table keeps them as a column. The overlap is
    the same either way, so the matrix is transposed when that is what joins.
  * **The two halves may not be named alike.** The label half is chosen by the
    review (it holds the perturbation), the measurement half by *which table
    actually shares its barcodes*. `pick_expression_partner` tests the barcodes
    rather than trusting a file name.
"""

from __future__ import annotations

import gzip
import re
from collections.abc import Sequence
from functools import lru_cache
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from ..tabular import header_columns, read_table

#: Column names a cell barcode is published under, most specific first.
KEY_HINTS = (
    "barcode",
    "cell_barcode",
    "cellbarcode",
    "cell_id",
    "cellid",
    "cell",
    "iid",
    "orig.ident",
    "index",
    "sample_id",
    "well",
    "rowname",
)

#: The name `_match` uses for the row index, which is not a column.
INDEX_KEY = "__index__"

#: How many unnamed/identifier columns to test for a barcode before giving up.
#: A per-cell table has a handful; a wide matrix has thousands of *measurement*
#: columns and testing each of them is O(columns) work for no answer.
KEY_COLUMN_LIMIT = 8


def _normalise_one(value: object) -> str:
    """One barcode as a string, with a `-1` / `.1` suffix tolerated.

    10x writes `AAAC-1`; a metadata file sometimes keeps it and sometimes drops
    it, and a counts matrix exported through a second tool writes `.1` instead
    of `-1`. A join that fails on that difference looks like "no overlapping
    cells" rather than what it is.
    """
    return re.sub(r"[-.]\d+$", "", str(value))


def _normalise_barcodes(values: pd.Series | pd.Index) -> pd.Series:
    """The values with the platform's suffix removed, keeping their own index.

    The index is kept because the result is assigned back onto a frame
    (`frame["__key__"] = ...`): a fresh 0..n-1 index would align against a
    barcode index and produce an all-NaN key, which reads as "no overlapping
    cells" rather than as the bug it is.
    """
    index = values.index if isinstance(values, pd.Series) else pd.Index(values)
    return pd.Series(
        [_normalise_one(value) for value in values], index=index, dtype="object"
    )


def _key_series(frame: pd.DataFrame) -> list[tuple[str, pd.Series]]:
    """`(name, values)` for every column-or-index that could hold a barcode.

    A delimited file read without an index keeps its row key as a column
    (`Unnamed: 0`, or the empty name GEO writes); a table read from an h5ad or a
    parsed series matrix may keep it in the index. Both are candidates, so the
    overlap test does not depend on which reader ran.
    """
    keys: list[tuple[str, pd.Series]] = []
    for column in frame.columns:
        series = frame[column]
        if pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series):
            keys.append((str(column), series))
            if len(keys) >= KEY_COLUMN_LIMIT:
                break
    if not isinstance(frame.index, pd.RangeIndex):
        keys.append((INDEX_KEY, pd.Series(frame.index, index=frame.index)))
    return keys


def _key_set(values: pd.Series) -> set[str]:
    return set(_normalise_barcodes(values))


@dataclass(frozen=True)
class BarcodeMatch:
    """The best reading of "the two tables describe the same cells"."""

    label_key: str
    expression_key: str
    shared: pd.Index
    fraction: float


def _rank(match: BarcodeMatch) -> tuple[float, int]:
    """How good a match is: how completely the two overlap, then how many cells.

    The tie-break matters. A perturbation screen's label table has a column
    called `gene` holding gene symbols, and a genes-by-cells matrix has a column
    (or a row index) whose *values* are those same symbols -- so a naive "which
    columns have values in common" test happily joins the labels to the gene
    names, keeping the sixth of the cells whose perturbation happens to be a
    measured gene. Both readings score well on the fraction alone (seven values
    matching six of them is 86%); only the number of shared cells tells the
    barcode join (every cell) from the coincidence.
    """
    return (round(match.fraction, 6), len(match.shared))


def _match(
    labels: pd.DataFrame,
    expression: pd.DataFrame,
    keys: Sequence[str] | None = None,
) -> BarcodeMatch | None:
    """The key pair that overlaps best between the two tables, and how well."""
    best: BarcodeMatch | None = None
    for label_name, label_values in _key_series(labels):
        left = _key_set(label_values)
        if not left:
            continue
        for expression_name, expression_values in _key_series(expression):
            if keys and label_name not in keys and expression_name not in keys:
                continue
            right = _key_set(expression_values)
            if not right:
                continue
            shared = pd.Index(sorted(left & right))
            fraction = len(shared) / max(1, min(len(left), len(right)))
            candidate = BarcodeMatch(label_name, expression_name, shared, fraction)
            if best is None or _rank(candidate) > _rank(best):
                best = candidate
    return best


def find_overlap(
    labels: pd.DataFrame, expression: pd.DataFrame, keys: Sequence[str] | None = None
) -> tuple[str | None, pd.Index | None, float]:
    """The key that overlaps best between the two tables, and how well.

    Returns `(key, shared_index, fraction)`. `key` is the name a caller can use
    on the label table: a column the two tables share when there is one (the
    common case), otherwise the other table's own row key, which the caller
    joins against the index.
    """
    match = _match(labels, expression, keys)
    if match is None:
        return None, None, 0.0
    return match.label_key, match.shared, match.fraction


def transpose_expression(frame: pd.DataFrame) -> pd.DataFrame:
    """A genes-by-cells matrix as cells-by-genes: rows are cells, columns genes.

    GEO ships count matrices with genes in rows and cell barcodes in the header
    (`"", "r_AAAC...", ...`). Read as a table that is a matrix whose *columns*
    are cells, which no part of this pipeline expects. Flipping it restores the
    layout a per-cell table has, so the join and the contract see one shape.
    """
    columns = list(frame.columns)
    if len(columns) < 2:
        return frame
    head, rest = columns[0], columns[1:]
    sampled = rest[:64]
    numeric = [c for c in sampled if pd.api.types.is_numeric_dtype(frame[c])]
    # Genes in the first column, counts everywhere else: the shape of every
    # genes-by-cells export, and never of a per-cell table (whose first column
    # is the barcode and whose gene columns are the numeric ones *after* it).
    looks_transposed = (
        not pd.api.types.is_numeric_dtype(frame[head])
        and len(numeric) >= max(1, int(0.8 * len(sampled)))
    )
    if not looks_transposed:
        return frame.T
    genes = frame[head].astype(str)
    matrix = frame[rest].apply(pd.to_numeric, errors="coerce")
    matrix.index = pd.Index(genes, name=str(head or "gene"))
    return matrix.T


def join_screen(
    labels: pd.DataFrame,
    expression: pd.DataFrame,
    *,
    condition_column: str,
    key: str | None = None,
    min_overlap: float = 0.05,
) -> pd.DataFrame:
    """One table per cell: the label columns plus the measured genes.

    The result keeps every column of the label table (so context travels with
    the row) and every gene column of the expression table. Rows whose barcode is
    in only one of the two tables are dropped -- the training set is the
    intersection, and the caller sees how big that was.

    The expression table is transposed when that is the orientation that joins;
    both readings are tried and the better overlap wins, so a genes-by-cells
    matrix and a cells-by-genes one are handled by the same call.
    """
    if condition_column not in labels.columns:
        raise ValueError(f"the label table has no condition column {condition_column!r}")

    wanted = [key] if key else None
    match = _match(labels, expression, wanted)
    # Both orientations are always tried, and the better reading wins: a
    # genes-by-cells matrix read the wrong way round can still find *a* key in
    # common (the gene symbols), and accepting that because it clears the
    # threshold is what silently trains on a sixth of the cells.
    flipped = transpose_expression(expression)
    flipped_match = _match(labels, flipped, wanted)
    if flipped_match is not None and (
        match is None or _rank(flipped_match) > _rank(match)
    ):
        expression, match = flipped, flipped_match

    if match is None or match.shared.size == 0 or match.fraction < min_overlap:
        best = 0.0 if match is None else match.fraction
        raise ValueError(
            "the labels and the expression share no barcode: the two tables are "
            f"not the same cells (best overlap {best:.1%}). Check that one is not "
            "a sample-level table, and that the barcodes have not been renamed"
        )

    left = labels.copy()
    right = expression.copy()
    left["__key__"] = (
        _normalise_barcodes(left.index)
        if match.label_key == INDEX_KEY
        else _normalise_barcodes(left[match.label_key])
    )
    if match.expression_key == INDEX_KEY:
        right["__key__"] = _normalise_barcodes(right.index)
    else:
        right["__key__"] = _normalise_barcodes(right[match.expression_key])
        right = right.drop(columns=[match.expression_key])

    gene_columns = [
        str(column)
        for column in right.columns
        if column != "__key__" and pd.api.types.is_numeric_dtype(right[column])
    ]
    if not gene_columns:
        raise ValueError("the expression table has no numeric gene columns to join")
    joined = left.merge(
        right[["__key__", *gene_columns]], on="__key__", how="inner", suffixes=("", "_expr")
    )
    joined = joined.drop(columns=["__key__"])
    # Which columns are the transcriptome, as opposed to the label table's own
    # QC columns: the caller cannot tell them apart afterwards, and treating
    # `percent.mito` as a measured gene would put it in the panel.
    joined.attrs["expression_columns"] = gene_columns
    joined.attrs["barcode_overlap"] = float(match.fraction)
    joined.attrs["joined_on"] = match.label_key
    return joined


@lru_cache(maxsize=256)
def _shape(path: str) -> tuple[int | None, int]:
    """`(rows, columns)` of a delimited table, without loading it.

    The row count is the piece a profile does not have when it skips counting
    (a wide table can be a gigabyte), and it is the piece that tells a
    transcriptome from an antibody panel: 18,649 genes in rows against 5.
    """
    file = Path(path)
    names = header_columns(file) or []
    opener = gzip.open if file.suffix.lower() == ".gz" else open
    rows = 0
    try:
        with opener(file, "rt", encoding="utf-8", errors="replace") as handle:
            for _ in handle:
                rows += 1
    except OSError:
        return None, len(names)
    return max(0, rows - 1), len(names)


def table_shape(path: str | Path) -> tuple[int | None, int]:
    """The cached shape of a table."""
    return _shape(str(path))


def measurement_features(path: str | Path) -> set[str]:
    """The gene names a measurements table offers, cheaply.

    A cells-by-genes export names them in the header; the genes-by-cells
    orientation GEO publishes lists them in the first column. Both are read
    without touching the matrix.
    """
    names = header_columns(path) or []
    features = {str(name) for name in names[1:]}
    if names:
        try:
            first = read_table(path, columns=[names[0]])
            features |= {str(value) for value in first.iloc[:, 0].tolist()}
        except Exception:  # noqa: BLE001 - an unreadable first column is not fatal
            pass
    return features


def is_measurement_table(path: str | Path, *, min_expression_features: int = 500) -> bool:
    """Does this table look like a transcriptome rather than a feature panel?

    A count matrix measures thousands of genes whichever way round it is
    written, so *both* axes are wide. A CITE-seq antibody panel (5 features),
    a hashtag panel (13) or a guide-count matrix (112) has one wide axis -- the
    cells -- and a short one, and a screen's supplementary graph built on
    `CD86`/`HTO22`/`eGFPg1` shares no gene with the gold.
    """
    rows, columns = table_shape(path)
    if rows is None:
        return False
    if columns < 2:
        # An archive or a binary blob: not a table at all, and counting its
        # "rows" means decoding a gigabyte of bytes as text.
        return False
    return min(rows, max(0, columns - 1)) >= min_expression_features


def numeric_columns(path: str | Path, *, rows: int = 50) -> int:
    """How many numeric columns a table has, reading only its head."""
    try:
        frame = pd.read_csv(path, nrows=rows, low_memory=False)
    except Exception:  # noqa: BLE001 - an unreadable table is not a candidate
        return 0
    return int(sum(1 for column in frame.columns if pd.api.types.is_numeric_dtype(frame[column])))


def pick_expression_table(
    gold_path: str | Path,
    candidates: Sequence[str | Path],
    *,
    max_label_columns: int = 200,
    min_expression_columns: int = 50,
    ratio: float = 3.0,
) -> Path | None:
    """Which candidate holds the measurements the gold's labels lack.

    A screen's label table is metadata: barcodes plus a handful of QC columns, so
    it has few numeric columns. The measurements have many more (a real one has
    thousands). The rule is therefore *relative* to the gold as well as absolute
    -- "many more numeric columns than the labels have" -- because a small test
    screen and a real one differ by orders of magnitude.
    """
    gold_numeric = numeric_columns(gold_path)
    if gold_numeric > max_label_columns:
        # The gold already looks like a measurement table: the two halves are in
        # one file and there is nothing to join.
        return None
    threshold = max(min_expression_columns, ratio * gold_numeric)
    best: tuple[int, Path] | None = None
    for candidate in candidates:
        count = numeric_columns(candidate)
        if count < threshold:
            continue
        if best is None or count > best[0]:
            best = (count, Path(candidate))
    return best[1] if best else None


def _published_barcodes(path: str | Path) -> set[str]:
    """Every string a table could be keyed on, without loading the matrix.

    Two places carry them and both are cheap to read: the header (a
    genes-by-cells matrix puts the barcodes there) and the first column (a
    per-cell table puts them there).
    """
    names = header_columns(path) or []
    if not names:
        return set()
    values = {str(name) for name in names[1:]}
    try:
        first = read_table(path, columns=[names[0]])
        values |= {str(value) for value in first.iloc[:, 0].tolist()}
    except Exception:  # noqa: BLE001 - an unreadable first column is not fatal
        pass
    return {_normalise_one(value) for value in values}


def pick_expression_partner(
    gold_path: str | Path,
    candidates: Sequence[str | Path],
    *,
    min_overlap: float = 0.3,
    max_label_columns: int = 200,
    min_expression_columns: int = 50,
    min_expression_features: int = 500,
) -> Path | None:
    """The table that measures the cells `gold_path` labels.

    Name matching cannot do this -- a screen's counts file is
    `GSM4633614_ECCITE_cDNA_counts.tsv.gz` and its label table is
    `GSE153056_ECCITE_metadata.tsv.gz`, with nothing in common but the series --
    so the decision is made on the data: which candidate shares the label
    table's barcodes. A filter step that drops cells lowers the overlap, so the
    bar is a fraction rather than exact equality.
    """
    gold = Path(gold_path)
    try:
        labels = read_table(gold)
    except Exception:  # noqa: BLE001 - an unreadable gold is the caller's problem
        return None
    gold_keys = [_key_set(values) for _, values in _key_series(labels)]
    gold_keys = [values for values in gold_keys if values]
    if not gold_keys:
        return None
    # A label table is narrow and mostly categorical; a table with hundreds of
    # numeric columns already carries its measurements.
    if numeric_columns(gold) > max_label_columns:
        return None

    best: tuple[float, Path] | None = None
    measured_cells = float(len(labels))
    for candidate in candidates:
        path = Path(candidate)
        if path == gold:
            continue
        names = header_columns(path) or []
        # "Wide enough to be measurements": a label table has a handful of QC
        # columns, a count matrix has thousands, so the bar is relative to the
        # label table as well as absolute.
        threshold = max(min_expression_columns, 3 * max(1, numeric_columns(gold)))
        if max(len(names), numeric_columns(path)) < threshold:
            continue
        # And the *other* axis has to be wide too, or this is a panel of
        # antibodies or guides that happens to be measured on the same cells.
        if not is_measurement_table(path, min_expression_features=min_expression_features):
            continue
        values = _published_barcodes(path)
        if not values:
            continue
        for gold_set in gold_keys:
            # The score is the share of the *label table's* cells this candidate
            # measures, not the share of whatever happened to overlap. A label
            # table whose condition column holds gene symbols overlaps a counts
            # matrix's gene names by construction -- a handful of values against
            # a handful of values is a high ratio and a meaningless join, where
            # the same handful against every row of the screen is 5%.
            coverage = len(gold_set & values) / max(1.0, measured_cells)
            if best is None or coverage > best[0]:
                best = (coverage, path)
    if best is None or best[0] < min_overlap:
        return None
    return best[1]
