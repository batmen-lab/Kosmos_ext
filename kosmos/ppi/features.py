"""Turning whatever a fetched table holds into the numeric matrix a model needs.

Real tables are not matrices. A clinical table carries `thal` as
`fixed/normal/reversible` and a missing cell as `?`; another cohort file leaves
nine columns unparsed because a few of its rows say `?`. The PPI path is a model
over numbers, so those columns have to be encoded rather than refused -- asking
for "numeric features only" is what made every real table unusable.

One encoder is fitted per run, on the labeled (gold) rows alone, and then applied
unchanged to the supplementary tables and to the held-out test set, so a single
vocabulary covers the whole run:

  * a column whose non-missing values all parse as numbers is **numeric**:
    missing cells take the gold median, and the column is centred and scaled by
    the gold mean and standard deviation (a constant column keeps scale 1 rather
    than dividing by zero);
  * every other column is **categorical**: one-hot over the gold's levels, sorted
    so the width is deterministic, plus one trailing column for a level the gold
    never carried -- a supplementary cohort is allowed to have its own -- and for
    missing cells.

A categorical column with more than `max_cardinality()` levels is refused: a
column with a distinct value per row is an identifier or free text, and encoding
it would add one column per row. That rule is stated twice on purpose --
`datafetcher.profile` decides a table's role before anything runs, and the
fetcher neither imports Kosmos nor the other way round. Both sides read
`PPI_MAX_CATEGORICAL_CARDINALITY`, and when they disagree the run fails loudly at
load time with the column named rather than quietly training on something else.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

#: Cells that mean "no value" rather than a value. Kept short and literal: a
#: marker like `-` or `unknown` is a real category in some tables.
MISSING_MARKERS = frozenset({"", "?", "na", "n/a", "nan", "none", "null"})

DEFAULT_MAX_CARDINALITY = 50
CARDINALITY_ENV = "PPI_MAX_CATEGORICAL_CARDINALITY"

#: The one-hot column that collects levels the gold never carried, and missing
#: cells. It is written this way so the encoded name is stable across runs.
UNKNOWN_LEVEL = "<unknown>"


def max_cardinality() -> int:
    """The level cap, from the environment, defaulting when unset or nonsense."""
    raw = os.environ.get(CARDINALITY_ENV, "").strip()
    if not raw:
        return DEFAULT_MAX_CARDINALITY
    try:
        value = int(raw)
    except ValueError:
        return DEFAULT_MAX_CARDINALITY
    return value if value > 0 else DEFAULT_MAX_CARDINALITY


class UnencodableColumn(ValueError):
    """A column that cannot become numbers without inventing one per row."""

    def __init__(self, column: str, n_levels: int, limit: int):
        super().__init__(
            f"feature column {column!r} has {n_levels} distinct value(s), more "
            f"than the {limit} this encoder allows: it looks like an identifier "
            f"or free text. Exclude it, or raise {CARDINALITY_ENV}."
        )
        self.column = column
        self.n_levels = n_levels
        self.limit = limit


def _as_text(series: pd.Series) -> pd.Series:
    """The column as stripped text, with the missing markers turned into NaN."""
    text = series.astype(str).str.strip()
    return text.mask(text.str.lower().isin(MISSING_MARKERS))


def _is_numeric_dtype(series: pd.Series) -> bool:
    """True when pandas already parsed this column as numbers."""
    return series.dtype.kind in "iufb"


def _peak_memory_mb() -> float:
    """Peak RSS of this process, in MB: what an out-of-memory crash is made of."""
    try:
        import resource

        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
    except Exception:  # noqa: BLE001 - a diagnostic, never a failure
        return 0.0


@dataclass(frozen=True)
class _Numeric:
    median: float
    mean: float
    std: float


@dataclass(frozen=True)
class FeatureEncoder:
    """A fitted fit-on-gold, apply-everywhere transformation."""

    #: Raw columns this encoder was fitted on, in order.
    raw_features: list[str]
    #: The columns it produces, in order.
    feature_names: list[str]
    numeric: dict[str, _Numeric]
    categorical: dict[str, list[str]]
    #: Columns left out because their level count exceeded the cap: the rules
    #: judged them on a sample, this encoder sees every row, and a column that
    #: looked like a category in the sample can turn out to be free text.
    dropped_wide: dict[str, int] = field(default_factory=dict)
    #: Names that appear more than once in the table it was fitted on, with how
    #: many columns share them. Gene symbols are re-used in real annotations (a
    #: human BMMC table has 36 such names), and `frame[name]` with a duplicated
    #: label is a DataFrame, not a column -- which crashed inside pandas' own
    #: machinery rather than raising. The first column of each name is used.
    duplicate_columns: dict[str, int] = field(default_factory=dict)

    def _column(self, frame: pd.DataFrame, name: str, positions: dict[str, list[int]]):
        """The one column called `name`, even when the table has several.

        `frame[name]` returns a DataFrame when the label is duplicated, and
        every caller here assumes a Series. Positional selection is the same
        thing with the ambiguity resolved: the first column of that name, which
        is the one the file's own order puts there.
        """
        index = positions.get(name) or []
        if not index:
            raise ValueError(f"this table has no column {name!r}")
        return frame.iloc[:, index[0]]

    @property
    def width(self) -> int:
        return len(self.feature_names)

    @property
    def categorical_features(self) -> list[str]:
        return list(self.categorical)

    @classmethod
    def fit(
        cls,
        frame: pd.DataFrame,
        feature_names: list[str],
        *,
        max_levels: int | None = None,
        drop_wide: bool = True,
    ) -> FeatureEncoder:
        """Fit on the labeled rows: the levels and scales every table shares.

        A column with more levels than the cap is dropped (`drop_wide=True`,
        the default, naming it in `dropped_wide`) or raises
        (`drop_wide=False`), for a caller who would rather hear about it than
        have a feature quietly go missing.
        """
        limit = max_levels if max_levels is not None else max_cardinality()
        # `set(...)` outside the comprehension: rebuilt per item it is
        # 12,000 x 130,000 string hashes on a single-cell table, which is
        # minutes of work that looks exactly like a hang.
        present_columns = set(frame.columns)
        missing = [c for c in feature_names if c not in present_columns]
        if missing:
            raise ValueError(
                f"cannot encode: the table has no column(s) {missing[:3]}"
            )

        numeric: dict[str, _Numeric] = {}
        categorical: dict[str, list[str]] = {}
        dropped_wide: dict[str, int] = {}
        positions: dict[str, list[int]] = {}
        for index, column in enumerate(frame.columns):
            positions.setdefault(str(column), []).append(index)
        duplicate_columns = {
            name: len(index)
            for name, index in positions.items()
            if len(index) > 1 and name in set(feature_names)
        }
        if duplicate_columns:
            logger.warning(
                "%d feature name(s) appear more than once in the table, e.g. %s; "
                "the first column of each is used",
                len(duplicate_columns),
                list(duplicate_columns)[:3],
            )
        # A name repeated in the *feature list* (it happens after a rename) is
        # one feature, not two.
        feature_names = list(dict.fromkeys(feature_names))
        kept: list[str] = []
        encoded: list[str] = []
        total = len(feature_names)
        logger.debug("fitting the encoder on %s column(s)", f"{total:,}")
        try:
            _fit_columns(
                frame,
                feature_names,
                limit=limit,
                drop_wide=drop_wide,
                numeric=numeric,
                categorical=categorical,
                dropped_wide=dropped_wide,
                kept=kept,
                encoded=encoded,
                positions=positions,
            )
        except MemoryError as e:
            # A table this wide is read into memory twice (the frame, then the
            # encoded matrix), and running out is a fact about the data rather
            # than a bug in the model. Naming the size is the difference between
            # that being clear and it being a segfault.
            raise ValueError(
                f"ran out of memory fitting {total:,} feature column(s) over "
                f"{len(frame):,} row(s) "
                f"({frame.memory_usage(deep=True).sum() / 1e9:.1f} GB in the "
                f"table, peak {_peak_memory_mb():.0f} MB). Use fewer cells per "
                f"table (KOSMOS_SINGLE_CELL_MAX_CELLS) or fewer features."
            ) from e
        return cls(
            # A dropped column is not a feature: `raw_features` is what the
            # other tables have to supply, and `transform` has nothing to look
            # up for a column it never encoded.
            raw_features=kept,
            feature_names=encoded,
            numeric=numeric,
            categorical=categorical,
            dropped_wide=dropped_wide,
            duplicate_columns=duplicate_columns,
        )

    def transform(self, frame: pd.DataFrame) -> np.ndarray:
        """Apply the fitted encoding to any table with the same raw columns."""
        present_columns = set(frame.columns)
        missing = [c for c in self.raw_features if c not in present_columns]
        if missing:
            raise ValueError(
                f"this table is missing {len(missing)} feature column(s) the "
                f"encoder was fitted on, e.g. {missing[:3]}; it is data for a "
                f"different task"
            )
        positions: dict[str, list[int]] = {}
        for index, column in enumerate(frame.columns):
            positions.setdefault(str(column), []).append(index)
        # One output array filled column by column. Building 12,000 little
        # (n, 1) arrays and concatenating them costs the same numbers and a lot
        # more allocator churn -- on a single-cell table that is hundreds of
        # megabytes of temporaries around a 100 MB result.
        width = sum(
            len(self.categorical.get(name) or []) + 1
            if name in self.categorical
            else 1
            for name in self.raw_features
        )
        matrix = np.zeros((len(frame), width), dtype="float32")
        cursor = 0
        for name in self.raw_features:
            text = _as_text(self._column(frame, name, positions))
            spec = self.numeric.get(name)
            if spec is not None:
                values = pd.to_numeric(text, errors="coerce").fillna(spec.median)
                scaled = (values.astype("float64") - spec.mean) / spec.std
                matrix[:, cursor] = np.asarray(scaled, dtype="float32")
                cursor += 1
                continue
            levels = self.categorical[name]
            codes = pd.Categorical(
                text.fillna(UNKNOWN_LEVEL), categories=[*levels, UNKNOWN_LEVEL]
            ).codes
            codes = np.where(codes < 0, len(levels), codes)
            block = matrix[:, cursor : cursor + len(levels) + 1]
            if len(frame):
                block[np.arange(len(frame)), codes] = 1.0
            cursor += len(levels) + 1
        return matrix

    def to_dict(self) -> dict[str, Any]:
        """The provenance record: what was encoded, and how wide it came out."""
        return {
            "width": self.width,
            "feature_names": list(self.feature_names),
            "numeric": {
                name: {"median": s.median, "mean": s.mean, "std": s.std}
                for name, s in self.numeric.items()
            },
            "categorical": {
                name: {
                    "levels": list(levels),
                    "unknown": UNKNOWN_LEVEL,
                    "n_levels": len(levels),
                }
                for name, levels in self.categorical.items()
            },
            "dropped_wide": dict(self.dropped_wide),
            "duplicate_columns": dict(self.duplicate_columns),
        }


def _fit_columns(
    frame,
    feature_names,
    *,
    limit,
    drop_wide,
    numeric,
    categorical,
    dropped_wide,
    kept,
    encoded,
    positions,
) -> None:
    """One pass over the columns, filling the maps `fit` returns."""
    total = len(feature_names)
    for position, name in enumerate(feature_names, start=1):
        if position % 20000 == 0:
            logger.info(
                "encoder: %s of %s column(s) fitted (peak memory %s MB)",
                f"{position:,}",
                f"{total:,}",
                _peak_memory_mb(),
            )
        series = (
            frame.iloc[:, positions[name][0]]
            if positions.get(name)
            else frame[name]
        )
        if _is_numeric_dtype(series):
            # A numeric column is already numbers: the text round-trip
            # (`astype(str)` then `to_numeric`) is wasted work per column,
            # and on a single-cell table -- 130,000 columns -- it is what
            # made every step slow and, in the run that died with a
            # segfault inside pandas' `astype`, what made it run out of
            # memory. Same result, without the strings.
            values = series.astype("float64", copy=False)
            present = int(values.notna().sum())
            if present == 0:
                numeric[name] = _Numeric(median=0.0, mean=0.0, std=1.0)
                kept.append(name)
                encoded.append(name)
                continue
            filled = values.fillna(float(values.median()))
            std = float(filled.std(ddof=0))
            numeric[name] = _Numeric(
                median=float(values.median()),
                mean=float(filled.mean()),
                std=std if std > 0 else 1.0,
            )
            kept.append(name)
            encoded.append(name)
            continue

        text = _as_text(series)
        values = pd.to_numeric(text, errors="coerce")
        present = int(text.notna().sum())
        if present == 0:
            # Nothing to learn from: a column of zeros carries no signal and
            # cannot break the model, which refusing it would.
            numeric[name] = _Numeric(median=0.0, mean=0.0, std=1.0)
            kept.append(name)
            encoded.append(name)
            continue
        if int(values.notna().sum()) == present:
            filled = values.fillna(float(values.median()))
            std = float(filled.std(ddof=0))
            numeric[name] = _Numeric(
                median=float(values.median()),
                mean=float(filled.mean()),
                std=std if std > 0 else 1.0,
            )
            kept.append(name)
            encoded.append(name)
            continue

        levels = sorted(set(text.dropna().unique()))
        if len(levels) > limit:
            # Not a reason to fail the run. The role rules profile a sample
            # and can accept a column that a full pass shows to be free
            # text; the column is dropped here, named in the record, and
            # every other column in the table is still used.
            if not drop_wide:
                raise UnencodableColumn(name, len(levels), limit)
            dropped_wide[name] = len(levels)
            continue
        categorical[name] = levels
        kept.append(name)
        encoded.extend([f"{name}__{level}" for level in levels])
        encoded.append(f"{name}__{UNKNOWN_LEVEL}")
