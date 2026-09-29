"""Lift the statistics an experiment computed into the fields that are read.

Generated code returns everything in one free-form `results` dict:
`pearson_r`, `pearson_p`, `spearman_rho_cis_vs_t1`, `{'r':..,'p':..}`. The
Result row has dedicated `primary_p_value`, `primary_effect_size` and
`statistical_tests` columns, and NOTHING filled them -- so a run that computed
a p-value of 1.14e-41 stored it in `data` and left `p_value` empty.

That matters because `DataAnalystAgent._extract_result_summary` reads ONLY
those three columns. It never looks at `data`. So the analyst's prompt said

    Primary P-value: None
    Primary Effect Size: None
    Statistical Tests:

and the analyst -- correctly, for what it was shown -- reported that the
hypothesis was untested and scored the experiment 1/5. Every experiment in the
myocardial-fibrosis run was written off that way while its statistics sat in
the same row of the same table.

The pairing is done per SCOPE rather than by scanning for the smallest p in
the payload, because a nuisance test's p-value is not the experiment's answer.
"""

from __future__ import annotations

import logging
import math
import re
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Bounded because a payload may nest per-tissue or per-protein blocks, and an
# unbounded walk over a 20,000-row table of results would cost more than it
# returns.
_MAX_DEPTH = 3
_MAX_TESTS = 25

# Names that ARE a p-value.
_P_TOKENS = {"p", "pval", "pvalue", "pvals", "pvalues"}

# ...unless one of these qualifies it, in which case the number is a SETTING,
# not a result. `binomial_test.p_null = 0.05` is the null proportion the test
# was run against; it was picked as the experiment's primary p-value, and the
# analyst rightly called it out -- "likely a threshold or borderline value
# rather than a complete test result" -- while the real p-value, 1.75e-216,
# sat in the same dict under `pvalue`.
_P_QUALIFIERS = {
    "null", "threshold", "thresh", "alpha", "cutoff", "cut", "expected",
    "prior", "min", "max", "target", "nominal", "crit", "critical", "assumed",
}

# Names that carry an effect size or test statistic, most specific first so
# `rho` is preferred over a bare `r` when both are present.
_STAT_STEMS = (
    "rho", "r", "beta", "coef", "slope", "estimate", "statistic", "stat",
    "t", "u", "z", "d", "f", "chi2", "or", "auc", "diff", "ratio",
)

# Of those, the ones that are an EFFECT SIZE -- a magnitude on the data's own
# scale. The rest (t, U, z, chi2, F) are test statistics: they scale with
# sample size and say nothing about how large an effect is. Reporting one as
# an effect size produced `effect_size = 2425.81` from a chi-square, which the
# analyst could only describe as "extremely large, but its metric is
# unspecified -- it may be an odds ratio, chi-square statistic, or another
# quantity". An effect size nobody can interpret is worse than none.
_EFFECT_STEMS = frozenset({
    "rho", "r", "beta", "coef", "slope", "estimate", "d", "g",
    "or", "rr", "hr", "auc", "diff", "ratio",
})


def _is_effect_size(stat_key: Optional[str]) -> bool:
    """Is the paired statistic a magnitude, or just a test statistic?"""
    if not stat_key:
        return False
    return any(t in _EFFECT_STEMS for t in _tokens(stat_key))

# Never an effect size, whatever else the name looks like.
_NOT_A_STAT = re.compile(r"^(n|n_\w+|\w*_n|count|\w*_count|df|seed|random_seed)$")


def _tokens(name: str) -> List[str]:
    return [t for t in re.split(r"[_\W]+", name.lower()) if t]


def _is_p_value(name: str) -> bool:
    toks = _tokens(name)
    if not toks or not any(t in _P_TOKENS for t in toks):
        return False
    return not any(t in _P_QUALIFIERS for t in toks)


def _is_number(value: Any) -> bool:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    return not (math.isnan(value) or math.isinf(value))


def _plausible_p(value: Any) -> bool:
    """A p-value lies in [0, 1]. A key called `p_threshold` holding 12 does not."""
    return _is_number(value) and 0.0 <= float(value) <= 1.0


def _find_statistic(p_name: str, scope: Dict[str, Any]) -> Optional[Tuple[str, float]]:
    """The statistic `p_name` is the p-value OF, within the same scope.

    Three passes, narrowing: substitute the p-token for each known stat stem
    (`spearman_p_cis_vs_t1` -> `spearman_rho_cis_vs_t1`), then try the prefix
    plus a stem (`pearson_p` -> `pearson_r`), then settle for any numeric key
    sharing the prefix that is not itself a p-value or a count
    (`slope_pvalue` -> `slope_per_kb`).
    """
    toks = _tokens(p_name)
    p_at = next((i for i, t in enumerate(toks) if t in _P_TOKENS), None)
    if p_at is None:
        return None
    lowered = {k.lower(): k for k in scope}

    for stem in _STAT_STEMS:
        candidate = "_".join(toks[:p_at] + [stem] + toks[p_at + 1:])
        key = lowered.get(candidate)
        if key and _is_number(scope[key]):
            return key, float(scope[key])

    prefix = toks[:p_at]
    for stem in _STAT_STEMS:
        key = lowered.get("_".join(prefix + [stem]) if prefix else stem)
        if key and _is_number(scope[key]):
            return key, float(scope[key])

    head = "_".join(prefix)
    for key, value in scope.items():
        low = key.lower()
        if not _is_number(value) or _is_p_value(key) or _NOT_A_STAT.match(low):
            continue
        if head and low.startswith(head):
            return key, float(value)
    return None


def _test_name(p_name: str, path: List[str]) -> str:
    toks = _tokens(p_name)
    p_at = next((i for i, t in enumerate(toks) if t in _P_TOKENS), len(toks))
    label = "_".join(toks[:p_at] + toks[p_at + 1:])
    if not label:
        label = "_".join(path[-1:]) or "test"
    return ".".join(path[:-1] + [label]) if len(path) > 1 else label


# A table row usually names WHICH entity it is; use it to label the test so the
# analyst reads "top_hits[SOD2]" rather than a bare number. Most specific first.
_ROW_ID_KEYS = ("gene", "symbol", "protein", "name", "feature", "id", "label", "term")

# Bounds for table (list-of-dicts) handling: how many rows to scan for the most
# significant, and how many of those to keep. A per-protein/per-gene table's
# answer is its top hits, not all 20,000 rows.
_MAX_TABLE_SCAN = 2000
_MAX_TABLE_ROWS = 3


def _row_identifier(row: Dict[str, Any]) -> Optional[str]:
    """The name of the entity a table row describes (a gene, protein, ...)."""
    lowered = {k.lower(): k for k in row}
    for cand in _ROW_ID_KEYS:
        key = lowered.get(cand)
        if key and isinstance(row[key], str) and row[key].strip():
            return row[key].strip()
    return None


def _row_test(row: Dict[str, Any], path: List[str]) -> Optional[Dict[str, Any]]:
    """Build a test record from one table row, or None.

    Uses the FIRST p-value column in the row, in document order, as the row's
    p-value. For an MR table that is `p_mr` (the causal test), not a later
    directionality column such as `steiger_p` that happens to be smaller.
    """
    if not isinstance(row, dict):
        return None
    p_key = next(
        (k for k in row if _is_p_value(str(k)) and _plausible_p(row[k])), None
    )
    if p_key is None:
        return None
    found = _find_statistic(str(p_key), row)
    label = _test_name(str(p_key), path + [str(p_key)])
    ident = _row_identifier(row)
    if ident:
        label = f"{label}[{ident}]" if label else ident
    return {
        "test_name": label,
        "p_value": float(row[p_key]),
        "statistic": found[1] if found else None,
        "statistic_type": found[0] if found else None,
        "effect_size": found[1] if found and _is_effect_size(found[0]) else None,
        "effect_size_type": found[0] if found and _is_effect_size(found[0]) else None,
        "sample_size": next(
            (int(v) for k, v in row.items()
             if _tokens(str(k))[:1] == ["n"] and _is_number(v)),
            None,
        ),
    }


def extract_statistics(payload: Any) -> Dict[str, Any]:
    """Walk `payload` and return {primary_p_value, primary_effect_size, tests}.

    Never raises: an experiment that computed something must not be lost to a
    shape this did not anticipate. An empty return leaves the columns exactly
    as they were.
    """
    tests: List[Dict[str, Any]] = []
    from_table: List[bool] = []          # parallel to `tests`
    explicit: List[Dict[str, Any]] = []

    def _add(record: Dict[str, Any], table: bool) -> None:
        if len(tests) < _MAX_TESTS:
            tests.append(record)
            from_table.append(table)

    def walk(node: Any, path: List[str], depth: int) -> None:
        if depth > _MAX_DEPTH or not isinstance(node, dict) or len(tests) >= _MAX_TESTS:
            return
        for key, value in node.items():
            if isinstance(value, dict):
                walk(value, path + [str(key)], depth + 1)
                continue
            # A TABLE (list of records) -- e.g. per-protein MR `top_hits` or a
            # per-gene DE table. The dict-only walk above skipped these, so the
            # stats inside every such table were invisible: the MR result stored
            # SOD2 (p=2.4e-11) in `top_hits` and the analyst was handed an empty
            # form. Take each table's most significant rows as its tests.
            if isinstance(value, list) and any(isinstance(x, dict) for x in value):
                rows = [
                    _row_test(r, path + [str(key)])
                    for r in value[:_MAX_TABLE_SCAN] if isinstance(r, dict)
                ]
                rows = [r for r in rows if r]
                rows.sort(key=lambda t: t["p_value"])  # most significant first
                for r in rows[:_MAX_TABLE_ROWS]:
                    _add(r, table=True)
                continue
            if not (_is_p_value(str(key)) and _plausible_p(value)):
                continue
            found = _find_statistic(str(key), node)
            record = {
                "test_name": _test_name(str(key), path + [str(key)]),
                "p_value": float(value),
                "statistic": found[1] if found else None,
                "statistic_type": found[0] if found else None,
                "effect_size": (
                    found[1] if found and _is_effect_size(found[0]) else None
                ),
                "effect_size_type": (
                    found[0] if found and _is_effect_size(found[0]) else None
                ),
                "sample_size": next(
                    (int(v) for k, v in node.items()
                     if _tokens(str(k))[:1] == ["n"] and _is_number(v)),
                    None,
                ),
            }
            _add(record, table=False)
            # An explicitly named primary wins over document order.
            if _tokens(str(key)) in (["p", "value"], ["primary", "p", "value"], ["p"]) and depth == 1:
                explicit.append(record)

    try:
        walk(payload, [], 1)
    except Exception as e:  # pragma: no cover - a shape we did not foresee
        logger.warning(f"Could not extract statistics from the result payload: {e}")
        return {"primary_test": None, "primary_p_value": None,
                "primary_effect_size": None, "tests": []}

    # Primary selection, preferring the scalar/dict path as the backup: an
    # explicitly named p_value wins; else the first scalar/dict test in document
    # order (prior behaviour); else the most significant table row -- which is
    # what makes the pure-table MR result resolve to SOD2 instead of nothing.
    scalar = [t for t, tab in zip(tests, from_table) if not tab]
    table = [t for t, tab in zip(tests, from_table) if tab]
    if explicit:
        primary = explicit[0]
    elif scalar:
        primary = scalar[0]
    elif table:
        primary = min(table, key=lambda t: t["p_value"])
    else:
        primary = None
    return {
        "primary_test": primary["test_name"] if primary else None,
        "primary_p_value": primary["p_value"] if primary else None,
        "primary_effect_size": primary["effect_size"] if primary else None,
        "tests": tests,
    }
