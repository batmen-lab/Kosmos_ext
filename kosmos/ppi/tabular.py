"""Reading a table someone else wrote, including its separator.

The separator is sniffed from the file, not assumed from the extension. Fetched
data is not under our control: UCI ships semicolon-separated CSVs, GEO series
matrices are tab-separated with a metadata preamble, and a `.csv` that is
actually semicolon-separated parses as one giant column if you assume a comma --
which reads downstream as "this table has no label column", the most confusing
possible failure.

Sniffing is a convenience with a defined fallback: when it fails (a one-column
file, an exotic quoting style), the extension rule applies and the caller sees
whatever pandas makes of it.
"""

from __future__ import annotations

import csv
import gzip
import logging
from collections.abc import Sequence
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

DEFAULT_HEAD_BYTES = 64 * 1024


def _head(path: Path, limit: int = DEFAULT_HEAD_BYTES) -> str:
    opener = gzip.open if path.suffix.lower() == ".gz" else open
    try:
        with opener(path, "rt", encoding="utf-8", errors="replace") as handle:
            return handle.read(limit)
    except OSError:
        return ""


def detect_separator(path: str | Path, head: str | None = None) -> str:
    """Sniff the delimiter, falling back to the file extension's convention."""
    path = Path(path)
    extension_default = "\t" if path.suffix.lower() in {".tsv", ".tab"} else ","
    sample = head if head is not None else _head(path)
    if not sample:
        return extension_default
    try:
        return csv.Sniffer().sniff(sample, delimiters=",\t;|").delimiter
    except csv.Error:
        return extension_default


def header_columns(path: str | Path) -> list[str] | None:
    """The column names, from the first line only.

    Asking a parser for columns a file does not have is an error from the
    parser ("Usecols do not match…") and a much worse one than the answer the
    caller can give: which feature columns this table is missing. So a caller
    that wants a subset checks the header first.
    """
    path = Path(path)
    opener = gzip.open if path.suffix.lower() == ".gz" else open
    try:
        with opener(path, "rt", encoding="utf-8", errors="replace") as handle:
            first = handle.readline()
    except OSError:
        return None
    if not first.strip():
        return None
    try:
        return next(csv.reader([first], delimiter=detect_separator(path)))
    except (csv.Error, StopIteration):  # pragma: no cover - malformed header
        return None


def read_table(
    path: str | Path,
    *,
    columns: Sequence[str] | None = None,
    nrows: int | None = None,
) -> pd.DataFrame:
    """Load a table: delimited text (separator sniffed), parquet, or arrow.

    HuggingFace datasets are shipped as parquet or arrow, and a fetched file
    keeps whatever name the repository used, so the extension is a hint and the
    magic bytes decide when the extension says nothing useful.

    A wide delimited file goes through pyarrow when it is available. That is not
    a micro-optimisation: pandas' C parser on a 1 GB single-cell table with
    129,923 columns took **more than five minutes** (and gigabytes), where
    pyarrow with a large block size and only the columns the caller needs reads
    it in **under two seconds**. `columns` is what makes the difference -- it is
    passed through to the parser instead of being applied afterwards.
    """
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix in {".parquet", ".pq"} or _magic_is(path, b"PAR1"):
        frame = pd.read_parquet(path)
        return _select(frame, columns, nrows)
    if suffix in {".feather"}:
        return _select(pd.read_feather(path), columns, nrows)
    if suffix in {".arrow", ".ipc"} or _magic_is(path, b"ARROW1"):
        return _select(_read_arrow(path), columns, nrows)
    arrow = _read_delimited_with_arrow(path, columns=columns, nrows=nrows)
    if arrow is not None:
        return arrow
    return pd.read_csv(
        path,
        sep=detect_separator(path),
        usecols=list(columns) if columns else None,
        nrows=nrows,
    )


#: pyarrow's CSV reader infers the column count per block, so a block has to
#: hold a whole row: one row of a 129,923-column table is most of a megabyte.
ARROW_CSV_BLOCK_BYTES = 256 * 1024**2


def _read_delimited_with_arrow(
    path: Path, *, columns: Sequence[str] | None, nrows: int | None
) -> pd.DataFrame | None:
    """pyarrow's CSV reader, or None to let pandas do it.

    `nrows` keeps pandas: it stops reading early, which is what a probe of the
    first few rows wants, and pyarrow has no equivalent.
    """
    if nrows is not None:
        return None
    try:
        import pyarrow.csv as pacsv
    except ImportError:  # pragma: no cover - pyarrow is a declared dependency
        return None
    read_options = pacsv.ReadOptions(
        use_threads=True,
        block_size=ARROW_CSV_BLOCK_BYTES,
        autogenerate_column_names=False,
    )
    # Fetched data is not ours: UCI ships semicolons, GEO ships tabs. The
    # separator is sniffed the same way the pandas path sniffs it.
    parse_options = pacsv.ParseOptions(delimiter=detect_separator(path))
    convert_options = (
        pacsv.ConvertOptions(include_columns=list(columns)) if columns else None
    )
    try:
        table = pacsv.read_csv(
            path,
            read_options=read_options,
            parse_options=parse_options,
            convert_options=convert_options,
        )
    except Exception as e:  # noqa: BLE001 - any parser complaint: pandas tries
        logger.debug("pyarrow could not read %s (%s); falling back to pandas", path, e)
        return None
    if table.num_columns == 0:
        return None
    return table.to_pandas()


def _magic_is(path: Path, magic: bytes) -> bool:
    """True when the file starts with `magic`, for files named something else."""
    try:
        with path.open("rb") as handle:
            return handle.read(len(magic)) == magic
    except OSError:
        return False


def _read_arrow(path: Path) -> pd.DataFrame:
    """An Arrow IPC file, which is what a HuggingFace `dataset.arrow` is."""
    import pyarrow as pa

    with pa.memory_map(str(path), "rb") as source:
        try:
            return pa.ipc.open_file(source).read_all().to_pandas()
        except pa.ArrowInvalid:
            source.seek(0)
            return pa.ipc.open_stream(source).read_all().to_pandas()


def _select(
    frame: pd.DataFrame,
    columns: Sequence[str] | None,
    nrows: int | None,
) -> pd.DataFrame:
    if columns:
        frame = frame[list(columns)]
    return frame.iloc[:nrows] if nrows else frame
