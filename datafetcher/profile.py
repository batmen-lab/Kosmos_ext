"""What is actually inside a fetched file, and what role it can play.

Search answers "this dataset may exist". Fetch answers "here are the bytes".
Neither answers the question the training side has to ask before anything runs:
**does this table carry the label the task needs, and are its features usable?**
That is this module.

The role rules are deliberately two-valued, because the meaningful axis is
label availability and nothing else:

  * the task's label column is present  -> `gold`          (trains, selects, tests)
  * it is absent                        -> `supplementary` (PPI correction only)
  * features cannot be encoded          -> `unusable`      (with the reason)

What "can be encoded" means is the trainer's rule, restated here so a table is
not called gold and then refused at load time: a column of numbers is a column;
anything else is one-hot encoded if it has a handful of distinct values
(`thal`: fixed/normal/reversible) and refused if it has more than the cap, since
a distinct value per row is an identifier or free text. The cap lives in one
environment variable, `PPI_MAX_CATEGORICAL_CARDINALITY`, read by both this module
and `kosmos.ppi.features` -- they cannot import each other, so the rule is stated
twice and the variable keeps the two statements equal.

No grouping column is consulted. Where the rows came from is provenance, not a
role: a table is not "external" because of its donor, batch or repository, it is
supplementary because it has no label for this task.

This module does not import `kosmos` and `kosmos` does not import it: the task
shape is restated here as plain fields (`TaskShape`) that map onto Kosmos's
`TaskSpec` with `as_dict()`.
"""

from __future__ import annotations

import csv
import gzip
import math
import os
import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

DEFAULT_SAMPLE_ROWS = 500

#: Above this many columns, the per-column work (kind, nulls, cardinality) is
#: computed for a bounded slice instead of every column. A single-cell table has
#: one column per gene, and characterising 130,000 of them takes ~110 seconds --
#: per call, and the pipeline profiles the same file several times. The names
#: stay complete, because the names are what decide features and the label.
WIDE_TABLE_COLUMNS = 2000
WIDE_TABLE_SAMPLE_COLUMNS = 500
WIDE_TABLE_ROWS = 20
#: Counting rows means reading the file. Past this size the profile reports the
#: sample instead of a number it would have to spend minutes producing.
MAX_COUNT_BYTES = 2 * 1024**3

ColumnKind = Literal["numeric", "bool", "datetime", "text", "empty", "other"]
Role = Literal["gold", "supplementary", "unusable"]

DEFAULT_CATEGORICAL_LIMIT = 50
CATEGORICAL_LIMIT_ENV = "PPI_MAX_CATEGORICAL_CARDINALITY"

#: Cells that mean "no value" rather than a value. Mirrors
#: `kosmos.ppi.features.MISSING_MARKERS`: the fetcher and the trainer have to
#: agree on what counts as a missing measurement, or a table the rules accept is
#: refused by the encoder.
MISSING_MARKERS = frozenset({"", "?", "na", "n/a", "nan", "none", "null"})

#: Column names that are a row number, not a measurement. HuggingFace's parquet
#: exports carry `__index_level_0__`; a CSV exported from pandas carries
#: `Unnamed: 0`. A numeric column like this survives the "is it numbers" test
#: and would otherwise be trained on as a feature.
INDEX_LIKE_NAMES = frozenset(
    {
        "index",
        "id",
        "row_id",
        "row_index",
        "row",
        "unnamed: 0",
        "unnamed_0",
        "unnamed",
        "__index_level_0__",
    }
)


def is_index_like(name: str) -> bool:
    """True for a column name that is a row number rather than a measurement."""
    lowered = str(name).strip().lower()
    if lowered in INDEX_LIKE_NAMES:
        return True
    return bool(re.fullmatch(r"__index_level_\d+__", lowered))


def categorical_limit() -> int:
    """The level cap, from the environment, defaulting when unset or nonsense."""
    raw = os.environ.get(CATEGORICAL_LIMIT_ENV, "").strip()
    if not raw:
        return DEFAULT_CATEGORICAL_LIMIT
    try:
        value = int(raw)
    except ValueError:
        return DEFAULT_CATEGORICAL_LIMIT
    return value if value > 0 else DEFAULT_CATEGORICAL_LIMIT


#: How much of a feature schema an evidence table must share. Two columns is a
#: model; one is a coincidence. The fraction guards the other end: a 12-gene
#: overlap with a 14,000-gene gold would train the correction on almost none of
#: what the gold arm sees.
MIN_SHARED_FEATURES = 2
MIN_SHARED_FRACTION = 0.5
MIN_SHARED_ENV = "KOSMOS_MIN_SHARED_FEATURES"
MIN_SHARED_FRACTION_ENV = "KOSMOS_MIN_SHARED_FRACTION"


def normalize_column_name(name: str) -> str:
    """A column name without the punctuation a different export happened to use.

    `concave points_mean` (the UCI spelling) and `concave_points_mean` (the
    sklearn export) are the same measurement of the same patients. Comparing
    raw strings called one of them missing and kept a 569-row mirror out of the
    run, which is a naming difference, not a data one.
    """
    text = str(name).strip().strip("\"'").lower()
    text = re.sub(r"[\s\-.]+", "_", text)
    text = re.sub(r"[^0-9a-z_]+", "", text)
    return re.sub(r"_+", "_", text).strip("_")


#: How a table writes its gene symbols: human annotations are `PISD`, mouse
#: annotations are `Pisd`, a probe table is neither. `normalize_column_name`
#: lowercases both, so without this check a mouse table "shares" thousands of
#: genes with a human one (12,476 of them in the Baron pancreas series) and an
#: ortholog enters the correction as if it were the same measurement.
HUMAN_STYLE, MOUSE_STYLE, MIXED_STYLE = "human", "mouse", "mixed"
CROSS_SPECIES_ENV = "KOSMOS_ALLOW_CROSS_SPECIES"


def name_style(names: Sequence[str]) -> str:
    """`human` (`PISD`), `mouse` (`Pisd`), or `mixed` when neither dominates."""
    upper = sum(1 for name in names if len(str(name)) > 1 and str(name).isupper())
    title = sum(
        1
        for name in names
        if str(name)[:1].isupper() and str(name)[1:].islower()
    )
    total = upper + title
    if total < 5:
        return MIXED_STYLE
    if upper / total >= 0.6:
        return HUMAN_STYLE
    if title / total >= 0.6:
        return MOUSE_STYLE
    return MIXED_STYLE


def cross_species_reason(named: Sequence[str], columns: Sequence[str]) -> str:
    """Why these two tables are different organisms, or "" when they are not."""
    if os.environ.get(CROSS_SPECIES_ENV):
        return ""
    gold_style, table_style = name_style(named), name_style(columns)
    if MIXED_STYLE in (gold_style, table_style) or gold_style == table_style:
        return ""
    example = next(
        (str(name) for name in columns if str(name) not in {"barcode", "assigned_cluster"}),
        "",
    )
    return (
        f"its feature names are written the way a {table_style} annotation is "
        f"written where the gold's are {gold_style} (e.g. {example}): the spelling "
        f"matches but the organism does not, and an ortholog is not the same "
        f"measurement. Set {CROSS_SPECIES_ENV}=1 to use it anyway"
    )


def required_shared(features: Sequence[str]) -> int:
    """How many of `features` an evidence table has to share to be usable.

    Overridable from the environment so a run can deliberately accept thinner
    overlap (`KOSMOS_MIN_SHARED_FEATURES`, `KOSMOS_MIN_SHARED_FRACTION`)
    without a code change -- and a thinner overlap is a decision, because it
    narrows the model that both arms are trained and compared on.
    """
    if not features:
        return 0
    count = MIN_SHARED_FEATURES
    raw_count = os.environ.get(MIN_SHARED_ENV, "").strip()
    if raw_count:
        try:
            count = max(1, int(raw_count))
        except ValueError:
            count = MIN_SHARED_FEATURES
    fraction = MIN_SHARED_FRACTION
    raw_fraction = os.environ.get(MIN_SHARED_FRACTION_ENV, "").strip()
    if raw_fraction:
        try:
            fraction = min(1.0, max(0.0, float(raw_fraction)))
        except ValueError:
            fraction = MIN_SHARED_FRACTION
    return max(count, math.ceil(fraction * len(features)))


class ColumnProfile(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str
    kind: ColumnKind
    #: Filled from the sample; a null-rate in the sample, not in the file.
    null_fraction: float = Field(ge=0.0, le=1.0)
    #: Distinct non-null values in the sample: what decides whether a
    #: non-numeric column is a category or an identifier.
    n_unique: int | None = None
    #: True when the column is numbers, however they were written: a cohort
    #: file that spells one measurement `?` still has a numeric column, not a
    #: categorical one with a hundred levels.
    numeric_like: bool = False


class TableProfile(BaseModel):
    """The shape of one table: columns, their kinds, and how many rows."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    path: str
    delimiter: str
    file_bytes: int
    sampled_rows: int
    #: Exact row count when it was cheap to compute, else None.
    n_rows: int | None = None
    columns: list[ColumnProfile] = Field(default_factory=list)
    #: Every column name, when only a bounded slice of the columns was profiled.
    #: A single-cell table has one column per gene; the names decide which are
    #: features, and 130,000 of them cost two minutes *per profile call* to
    #: characterise. The names are cheap (one header line); the kinds are not.
    all_names: list[str] = Field(default_factory=list)
    #: How many columns the file has, when `columns` is a bounded slice.
    n_columns: int = 0
    notes: list[str] = Field(default_factory=list)

    @property
    def names(self) -> list[str]:
        return self.all_names or [c.name for c in self.columns]

    def kind_of(self, name: str) -> ColumnKind | None:
        for column in self.columns:
            if column.name == name:
                return column.kind
        return None

    def n_unique_of(self, name: str) -> int | None:
        for column in self.columns:
            if column.name == name:
                return column.n_unique
        return None

    def is_numeric_like(self, name: str) -> bool:
        for column in self.columns:
            if column.name == name:
                return column.kind in {"numeric", "bool"} or column.numeric_like
        return False

    def numeric_columns(self) -> list[str]:
        return [c.name for c in self.columns if c.kind in {"numeric", "bool"}]

    def text_columns(self) -> list[str]:
        return [c.name for c in self.columns if c.kind in {"text", "datetime", "other"}]


@dataclass(frozen=True)
class TaskShape:
    """The task, as far as role assignment needs to know it.

    Mirrors `kosmos.ppi.TaskSpec`'s selection fields. `as_dict()` produces the
    keyword arguments for that class, so a caller can hand the shape over
    without either package importing the other.
    """

    target_column: str | None = None
    feature_columns: tuple[str, ...] | None = None
    feature_prefixes: tuple[str, ...] = ()
    exclude_columns: tuple[str, ...] = ()
    sample_id_column: str | None = None
    description: str = ""

    def selected_features(self, profile: TableProfile) -> list[str]:
        present = profile.names
        if self.feature_columns is not None:
            # Matched the way the plan matches -- by normalised name -- and
            # returned in this table's own spelling, so `rename` can line it up
            # with the gold later. An exact-only match called a table that
            # spells `PISD` as `Pisd` a table with no features at all.
            by_normalised: dict[str, str] = {}
            for name in present:
                by_normalised.setdefault(normalize_column_name(name), name)
            matched: list[str] = []
            for wanted in self.feature_columns:
                found = by_normalised.get(normalize_column_name(wanted))
                if found is not None:
                    matched.append(found)
            return matched
        excluded = set(self.exclude_columns)
        if self.target_column:
            excluded.add(self.target_column)
        if self.sample_id_column:
            excluded.add(self.sample_id_column)
        names = [c for c in present if c not in excluded]
        if self.feature_prefixes:
            names = [c for c in names if c.startswith(tuple(self.feature_prefixes))]
        return names

    def as_dict(self) -> dict[str, Any]:
        return {
            "target_column": self.target_column,
            "feature_columns": self.feature_columns,
            "feature_prefixes": self.feature_prefixes,
            "exclude_columns": self.exclude_columns,
            "sample_id_column": self.sample_id_column,
            "description": self.description,
        }


@dataclass
class RoleDecision:
    role: Role
    reason: str
    features: list[str] = field(default_factory=list)
    #: Selected feature columns that are not numeric but are encodable, with the
    #: number of levels the encoder will one-hot.
    categorical_features: list[str] = field(default_factory=list)
    #: Selected feature columns with more levels than the encoder allows: the
    #: reason this table is `unusable`.
    high_cardinality_features: list[str] = field(default_factory=list)
    #: Columns that were dropped from `features` so the table could still be
    #: used. A table is not refused because one of its columns is free text.
    dropped_features: list[str] = field(default_factory=list)
    #: Feature columns the task named but the table does not have.
    missing_features: list[str] = field(default_factory=list)
    has_target: bool = False
    task: TaskShape | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "role": self.role,
            "reason": self.reason,
            "features": self.features,
            "n_features": len(self.features),
            "categorical_features": self.categorical_features,
            "high_cardinality_features": self.high_cardinality_features,
            "dropped_features": self.dropped_features,
            "missing_features": self.missing_features,
            "has_target": self.has_target,
        }


# --- reading the shape -----------------------------------------------------


_PARQUET_MAGIC = b"PAR1"
_ARROW_MAGIC = b"ARROW1"


def _binary_table_kind(path: Path) -> str | None:
    """`parquet`, `arrow`, or None -- by extension, then by magic bytes.

    HuggingFace datasets ship `.parquet` and `.arrow`, and a fetch keeps
    whatever name the repository used, so the extension is a hint rather than a
    contract.
    """
    if path.suffix.lower() in {".parquet", ".pq"}:
        return "parquet"
    if path.suffix.lower() in {".arrow", ".feather", ".ipc"}:
        return "arrow"
    try:
        with path.open("rb") as handle:
            head = handle.read(8)
    except OSError:
        return None
    if head.startswith(_PARQUET_MAGIC):
        return "parquet"
    if head.startswith(_ARROW_MAGIC):
        return "arrow"
    return None


def _sample_binary_table(path: Path, kind: str, sample_rows: int) -> pd.DataFrame:
    """The first `sample_rows` of a parquet/arrow file, without reading it all."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    if kind == "parquet":
        parquet = pq.ParquetFile(path)
        pieces = []
        seen = 0
        for index in range(parquet.num_row_groups):
            group = parquet.read_row_group(index).to_pandas()
            pieces.append(group)
            seen += len(group)
            if seen >= sample_rows:
                break
        frame = pd.concat(pieces, ignore_index=True) if pieces else pd.DataFrame()
        return frame.iloc[:sample_rows]

    with pa.memory_map(str(path), "rb") as source:
        try:
            reader = pa.ipc.open_file(source)
            table = reader.read_all()
        except pa.ArrowInvalid:
            source.seek(0)
            table = pa.ipc.open_stream(source).read_all()
    return table.to_pandas().iloc[:sample_rows]


def _count_binary_rows(path: Path, kind: str) -> int | None:
    """A parquet footer knows its row count for free; an arrow file may not."""
    if kind != "parquet":
        return None
    try:
        import pyarrow.parquet as pq

        return int(pq.ParquetFile(path).metadata.num_rows)
    except Exception:  # noqa: BLE001 - an unreadable footer is not fatal here
        return None


def binary_table_sample(path: str | Path, rows: int) -> pd.DataFrame | None:
    """The first `rows` of a parquet/arrow file, or None when it is neither."""
    path = Path(path)
    kind = _binary_table_kind(path)
    if kind is None:
        return None
    return _sample_binary_table(path, kind, rows)


def _delimiter_for(path: Path, head: bytes) -> str:
    if path.suffix.lower() in {".tsv", ".tab"}:
        return "\t"
    try:
        return csv.Sniffer().sniff(head.decode("utf-8", "replace"), delimiters=",\t;|").delimiter
    except Exception:  # noqa: BLE001 - sniffing is a convenience, not a contract
        return ","


def _hinted_columns(head: bytes, delimiter: str) -> int:
    """How many columns the first line has, from the header alone.

    A lower bound when the head is truncated (the header of a 130,000-column
    table does not fit in 64 kB), which is all this needs: it decides whether to
    read the per-column work in full or over a bounded slice.
    """
    if not head:
        return 0
    first = head.split(b"\n", 1)[0].decode("utf-8", "replace")
    if not first:
        return 0
    return first.count(delimiter) + 1


def _kind_of(series: pd.Series) -> ColumnKind:
    if series.isna().all():
        return "empty"
    kind = series.dtype.kind
    if kind in {"i", "u", "f"}:
        return "numeric"
    if kind == "b":
        return "bool"
    if kind in {"M", "m"}:
        return "datetime"
    if kind in {"U", "S", "O"}:
        return "text"
    return "other"


def _numeric_like(series: pd.Series) -> bool:
    """Numbers written as text, with missing markers: still a numeric column.

    This is the case that made the UCI cohorts unusable: two rows say `?`, so
    pandas calls `chol` a text column, and counting its distinct values gives
    ~150 -- which reads as free text. Every non-marker value parses as a number,
    so it is a numeric column with a couple of missing measurements.
    """
    text = series.astype(str).str.strip()
    text = text.mask(text.str.lower().isin(MISSING_MARKERS))
    present = int(text.notna().sum())
    if present == 0:
        return True  # nothing says otherwise; the encoder makes it a zero column
    parsed = pd.to_numeric(text, errors="coerce")
    return int(parsed.notna().sum()) == present


def _count_rows(path: Path) -> int | None:
    size = path.stat().st_size
    if size > MAX_COUNT_BYTES:
        return None
    opener = gzip.open if path.suffix.lower() == ".gz" else open
    try:
        with opener(path, "rt", encoding="utf-8", errors="replace") as handle:
            return sum(1 for line in handle if line.strip()) - 1  # minus the header
    except OSError:
        return None


def _columns_of(frame: pd.DataFrame) -> list[ColumnProfile]:
    """The shape of every column, however the frame was read."""
    return [
        ColumnProfile(
            name=str(name),
            kind=_kind_of(frame[name]),
            null_fraction=float(frame[name].isna().mean()) if len(frame) else 1.0,
            n_unique=_n_unique(frame[name]),
            numeric_like=_numeric_like(frame[name]),
        )
        for name in frame.columns
    ]


def _n_unique(series: pd.Series) -> int | None:
    """Distinct values, or None when the cells are not hashable.

    An audio dataset's column holds dicts, and `nunique` raises on those. That
    is a fact about the file, not a reason to fail the whole profile.
    """
    try:
        return int(series.nunique(dropna=True))
    except TypeError:
        try:
            return int(series.astype(str).nunique(dropna=True))
        except Exception:  # noqa: BLE001 - unhashable either way
            return None


def _profile_binary_table(
    path: Path,
    kind: str,
    *,
    sample_rows: int,
    count_rows: bool,
) -> TableProfile:
    """A parquet or arrow file is a table already: it carries its own names.

    HuggingFace datasets are shipped this way, and reading them as delimited
    text made every one of them unusable -- which is how a whole run can fetch
    the right repository and still end with nothing to train on.
    """
    try:
        frame = _sample_binary_table(path, kind, sample_rows)
    except Exception as e:  # noqa: BLE001 - a broken file is a finding, not a crash
        return TableProfile(
            path=str(path),
            delimiter="",
            file_bytes=path.stat().st_size,
            sampled_rows=0,
            columns=[],
            notes=[f"{kind} file could not be read: {type(e).__name__}: {e}"],
        )
    notes = [f"read as {kind}: the column names and types are in the file"]
    if frame.shape[1] <= 1:
        notes.append(
            "parsed as a single column; this file is probably not a plain table"
        )
    n_rows = _count_binary_rows(path, kind) if count_rows else None
    return TableProfile(
        path=str(path),
        delimiter="",
        file_bytes=path.stat().st_size,
        sampled_rows=len(frame),
        n_rows=n_rows,
        columns=_columns_of(frame),
        notes=notes,
    )


def profile_table(
    path: str | Path,
    *,
    sample_rows: int = DEFAULT_SAMPLE_ROWS,
    count_rows: bool = False,
) -> TableProfile:
    """Read a table's header and a sample of its rows. Never the whole file."""
    path = Path(path)
    binary = _binary_table_kind(path)
    if binary is not None:
        return _profile_binary_table(
            path,
            binary,
            sample_rows=sample_rows,
            count_rows=count_rows,
        )
    raw_head = path.open("rb").read(64 * 1024) if path.suffix.lower() != ".gz" else b""
    if path.suffix.lower() == ".gz":
        with gzip.open(path, "rb") as handle:
            raw_head = handle.read(64 * 1024)
    delimiter = _delimiter_for(path, raw_head)
    hinted_columns = _hinted_columns(raw_head, delimiter)
    wide = hinted_columns > WIDE_TABLE_COLUMNS
    try:
        frame = pd.read_csv(
            path,
            sep=delimiter,
            nrows=min(sample_rows, WIDE_TABLE_ROWS) if wide else sample_rows,
            low_memory=False,
            on_bad_lines="skip",
        )
    except (UnicodeDecodeError, pd.errors.ParserError, OSError, ValueError) as e:
        # A zip, an archive, a parquet, a binary blob: whatever it is, it is not
        # a delimited text table. Reporting that is the profile's job -- raising
        # here crashed the whole plan, which is how a single `.zip` candidate
        # took down an otherwise fine run.
        return TableProfile(
            path=str(path),
            delimiter=delimiter,
            file_bytes=path.stat().st_size,
            sampled_rows=0,
            columns=[],
            notes=[f"not readable as a delimited text table: {type(e).__name__}: {e}"],
        )
    all_names = [str(name) for name in frame.columns]
    notes: list[str] = []
    if wide and len(all_names) > 2 * WIDE_TABLE_SAMPLE_COLUMNS:
        # Keep the two ends: a single-cell table puts its identifiers and its
        # label after the measurements (a column per gene), and the middle is
        # the measurement block. Profiling all of it is what made every step of
        # the pipeline take minutes on this file.
        head = list(range(WIDE_TABLE_SAMPLE_COLUMNS))
        tail = list(range(len(all_names) - WIDE_TABLE_SAMPLE_COLUMNS, len(all_names)))
        columns = _columns_of(frame.iloc[:, head + tail])
        notes.append(
            f"{len(all_names):,} columns: kinds computed for the first and "
            f"last {WIDE_TABLE_SAMPLE_COLUMNS} of them, which is where the label "
            f"and the identifiers sit; every column name is kept"
        )
    else:
        columns = _columns_of(frame)
    if frame.shape[1] <= 1:
        # A GEO series matrix, an archive, or anything that is not a plain
        # table: say so instead of reporting a one-column dataset.
        notes.append(
            "parsed as a single column; this file is probably not a plain "
            "table (a matrix with a preamble, or a non-tabular artifact)"
        )
    elif any(str(name).startswith("!") for name in frame.columns):
        # GEO's series matrix: `!Series_*` / `!Sample_*` lines above the table.
        # Parsing it as a table reads those as column names, so the profile
        # would describe the preamble rather than the data.
        notes.append(
            "column names start with '!', which is GEO's metadata preamble; "
            "this is a series matrix, not a plain table -- use the derived CSV "
            "or select a supplementary file"
        )
    n_rows = None
    if count_rows:
        n_rows = _count_rows(path)
        if n_rows is None:
            notes.append(
                f"row count skipped: file is larger than {MAX_COUNT_BYTES:,} bytes"
            )
    return TableProfile(
        path=str(path),
        delimiter=delimiter,
        file_bytes=path.stat().st_size,
        sampled_rows=len(frame),
        n_rows=n_rows,
        columns=columns,
        all_names=all_names if len(columns) != len(all_names) else [],
        n_columns=len(all_names),
        notes=notes,
    )


# --- deciding the role -----------------------------------------------------


def classify_role(profile: TableProfile, task: TaskShape) -> RoleDecision:
    """Gold, supplementary or unusable -- and the reason, always."""
    if len(profile.columns) < 2:
        # A README, a `.gitattributes`, a series matrix that parsed as one
        # column: not a table. Without this a low-cardinality single column
        # became "supplementary evidence".
        return RoleDecision(
            role="unusable",
            reason=(
                f"not a table: {len(profile.columns)} parsed column(s). A table "
                f"needs at least a feature column and something else"
            ),
            task=task,
        )
    features = task.selected_features(profile)
    has_target = bool(task.target_column and task.target_column in set(profile.names))
    named = list(task.feature_columns or ())
    # Matched the way the plan matches: by normalised name, so `PISD`/`Pisd` and
    # `concave points_mean`/`concave_points_mean` are the same column here too.
    present_normalised = {normalize_column_name(name) for name in profile.names}
    missing = [c for c in named if normalize_column_name(c) not in present_normalised]
    shared = [c for c in named if c not in set(missing)]

    if missing and has_target:
        # The table carries the label, so it is a gold candidate: a named feature
        # it does not have means the task and the table disagree about what was
        # measured, and the run cannot take the caller's word for the schema.
        return RoleDecision(
            role="unusable",
            reason=(
                f"the table is missing {len(missing)} feature column(s) the task "
                f"names, e.g. {missing[:3]}"
            ),
            features=features,
            missing_features=missing,
            has_target=has_target,
            task=task,
        )
    if named and not has_target:
        # Evidence that is written in another organism's style is not evidence,
        # however many names it appears to share: this is the gate that keeps a
        # mouse table out of a human gold's correction.
        species = cross_species_reason(named, profile.names)
        if species:
            return RoleDecision(
                role="unusable",
                reason=species,
                features=features,
                missing_features=missing,
                has_target=has_target,
                task=task,
            )
    if missing and len(shared) < required_shared(named):
        # Evidence is matched by intersection: what a table shares with the named
        # feature columns is what the correction is trained on, and refusing it
        # for the columns it does not have is why so many runs ended with no
        # supplementary table at all. Too little overlap is still too little.
        return RoleDecision(
            role="unusable",
            reason=(
                f"it shares only {len(shared)} of the {len(named)} feature "
                f"column(s) the task names, and {required_shared(named)} are "
                f"needed to correct the same model"
            ),
            features=features,
            missing_features=missing,
            has_target=has_target,
            task=task,
        )
    if not features:
        return RoleDecision(
            role="unusable",
            reason=(
                "no feature columns match the task; check its feature prefixes "
                "and exclusions against the table's columns"
            ),
            has_target=has_target,
            task=task,
        )
    # A column is usable when it is numbers (however they were written) or a
    # category -- a handful of distinct values, one-hot encoded by the trainer.
    # A column that is neither is an identifier or free text, and it is left
    # out: one such column used to make the whole table unusable, which threw
    # away every other column in it (a perovskite table with a `pce` target and
    # four free-text columns was refused as a whole).
    limit = categorical_limit()
    high_cardinality = [
        c
        for c in features
        if not profile.is_numeric_like(c)
        and (profile.n_unique_of(c) or 0) > limit
    ]
    index_like = [c for c in features if is_index_like(c)]
    # A column where every value is missing is not a feature either: it carries
    # no information, and keeping it means every other table that does not have
    # it looks incomplete (`Unnamed: 32` is the classic one).
    empty = [c for c in features if profile.kind_of(c) == "empty"]
    dropped = list(dict.fromkeys([*high_cardinality, *index_like, *empty]))
    usable = [c for c in features if c not in set(dropped)]
    if not usable:
        return RoleDecision(
            role="unusable",
            reason=(
                f"no encodable feature column remains: all {len(features)} selected "
                f"column(s) look like free text or a row index, e.g. {dropped[:3]}; "
                f"exclude them, or raise {CATEGORICAL_LIMIT_ENV}"
            ),
            features=features,
            high_cardinality_features=high_cardinality,
            dropped_features=dropped,
            has_target=has_target,
            task=task,
        )
    categorical = [
        c for c in usable if not profile.is_numeric_like(c)
    ]
    dropped_note = (
        f"; {len(dropped)} column(s) are not features (free text, a row index, or "
        f"empty), "
        f"e.g. {dropped[:3]}"
        if dropped
        else ""
    )

    if has_target:
        return RoleDecision(
            role="gold",
            reason=(
                f"target column {task.target_column!r} is present with "
                f"{len(usable)} feature(s)"
                + (f" ({len(categorical)} categorical)" if categorical else "")
                + "; this table can train, select and test"
                + dropped_note
            ),
            features=usable,
            categorical_features=categorical,
            high_cardinality_features=high_cardinality,
            dropped_features=dropped,
            has_target=True,
            task=task,
        )
    return RoleDecision(
        role="supplementary",
        reason=(
            f"no target column for this task, so its rows can only enter through "
            f"the PPI correction term ({len(usable)} feature(s)"
            + (f", {len(categorical)} categorical)" if categorical else ")")
            + dropped_note
        ),
        features=usable,
        categorical_features=categorical,
        high_cardinality_features=high_cardinality,
        dropped_features=dropped,
        has_target=False,
        task=task,
    )


def classify_path(
    path: str | Path, task: TaskShape, *, sample_rows: int = DEFAULT_SAMPLE_ROWS
) -> tuple[TableProfile, RoleDecision]:
    """Convenience wrapper: profile a file and decide its role in one call."""
    profile = profile_table(path, sample_rows=sample_rows)
    return profile, classify_role(profile, task)


def plan_roles(
    paths: Iterable[str | Path], task: TaskShape
) -> dict[str, list[dict[str, Any]]]:
    """Group several tables by the role they can play.

    This is the answer the training side needs: one labeled table is enough to
    train; every unlabeled one beside it extends the run through PPI; and
    anything unusable is reported with its reason rather than dropped.
    """
    plan: dict[str, list[dict[str, Any]]] = {
        "gold": [],
        "supplementary": [],
        "unusable": [],
    }
    for path in paths:
        profile, decision = classify_path(path, task)
        entry = {"path": str(path), "profile": profile.model_dump(mode="json"), **decision.to_dict()}
        plan[decision.role].append(entry)
    return plan
