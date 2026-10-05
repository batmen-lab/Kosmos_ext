"""Add canonical, build/orientation-tolerant variant keys to a table.

GENERAL and convention-driven -- nothing here knows about any specific dataset,
file, or study. It detects, by common genomics naming/value conventions, a
variant's chromosome / position / alleles and/or a packed variant-id column, and
adds a canonical key column so that cross-table joins stop silently failing on
cosmetic differences:

  * allele orientation  (A/B vs B/A)              -> alleles upper-cased + sorted
  * chr prefix / zero-pad  ('chr06' vs '6')       -> stripped
  * build suffix in a packed id  ('..._b38')      -> ignored for membership

The canonical form is ``<chrom>:<pos>:<alleleA>:<alleleB>``.

Why TWO keys sometimes. A table can carry a variant's position in two places that
disagree -- most often separate coordinate columns in one genome build and a
packed ``id`` column whose embedded position is a *different* build (a lifted-over
file). When that happens we emit BOTH ``variant_key`` (from the coordinate
columns) and ``variant_key_alt`` (from the id). A table that carries both builds
is then its own offline crosswalk: another table keyed in either build can be
joined to it without any external chain file. This is derived purely from what is
in the row -- it is NOT a hard-coded build map, and tables with a single
consistent source get a single ``variant_key``.

Membership only: the key deliberately discards allele order, so effect-direction
(e.g. a beta sign) must still be read from the ORIGINAL allele columns, which are
never dropped or renamed.
"""
from __future__ import annotations

import re
from typing import List, Optional, Tuple

# --- column detection by convention (normalised: lower-case, alnum only) -------

def _norm(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(name).lower())


_CHROM_NAMES = {"chrom", "chromosome", "chr", "chrname", "chrn"}
_POS_NAMES = {"pos", "position", "bp", "basepair", "basepairlocation",
              "genpos", "basepos", "poshg", "coordinate"}
# Allele columns, in rough preference order. Two distinct matches make a pair;
# order does not matter for the key because alleles are sorted.
_ALLELE_NAMES = [
    "ref", "alt", "reference", "effectallele", "otherallele", "ea", "oa",
    "a1", "a2", "a0", "allele0", "allele1", "allele2",
    "major", "minor", "nea", "allelea", "alleleb",
]
_ALLELE_SET = set(_ALLELE_NAMES)

# A packed variant id: chrom [sep] pos [sep] allele [sep] allele, optionally
# 'chr'-prefixed and with trailing junk (build tag, imputation flag, ...).
_VARID_RE = re.compile(r"^(?:chr)?([0-9xymt]+)[:_\-]([0-9]+)[:_\-]([acgtn]+)[:_\-]([acgtn]+)", re.IGNORECASE)


def _find_one(columns, candidates) -> Optional[str]:
    for c in columns:
        if _norm(c) in candidates:
            return c
    return None


def _find_allele_pair(columns, used) -> Optional[Tuple[str, str]]:
    hits = [c for c in columns if c not in used and _norm(c) in _ALLELE_SET]
    if len(hits) >= 2:
        return hits[0], hits[1]
    return None


def _detect_id_column(df, used, sample: int = 200, min_frac: float = 0.8) -> Optional[str]:
    """A column whose values look like packed variant ids. Highest match wins."""
    best, best_frac = None, min_frac
    for c in df.columns:
        if c in used:
            continue
        s = df[c]
        if s.dtype.kind not in ("O", "U", "S"):  # must be string-like
            continue
        vals = s.dropna().astype(str).head(sample)
        if len(vals) == 0:
            continue
        frac = vals.map(lambda v: bool(_VARID_RE.match(v))).mean()
        if frac >= best_frac:
            best, best_frac = c, frac
    return best


# --- canonical key construction (vectorised) -----------------------------------

def _canon_chrom(s):
    return (s.astype(str)
             .str.replace(r"(?i)^chr", "", regex=True)
             .str.replace(r"^0+(?=\d)", "", regex=True)
             .str.upper())


def _canon_pos(s):
    import pandas as pd
    return pd.to_numeric(s, errors="coerce").astype("Int64").astype(str)


def _sorted_key(chrom, pos, a1, a2):
    a = a1.astype(str).str.upper().str.strip()
    b = a2.astype(str).str.upper().str.strip()
    lo = a.where(a <= b, b)
    hi = b.where(a <= b, a)
    return chrom.str.cat([pos, lo, hi], sep=":")


def _key_from_coords(df, chrom_col, pos_col, al1, al2):
    return _sorted_key(_canon_chrom(df[chrom_col]), _canon_pos(df[pos_col]),
                       df[al1], df[al2])


def _key_from_id(df, id_col):
    ext = df[id_col].astype(str).str.extract(_VARID_RE)
    if ext.isna().all().all():
        return None
    chrom = ext[0].str.replace(r"^0+(?=\d)", "", regex=True).str.upper()
    pos = _canon_pos(ext[1])
    return _sorted_key(chrom, pos, ext[2], ext[3])


def add_variant_keys(df) -> Tuple["object", List[str]]:
    """Add canonical variant key column(s) to a dataframe. Returns (df, added).

    Non-destructive: only appends columns, never drops/renames. Adds nothing (and
    returns []) when no variant columns are detectable, so it is safe to call on
    any table. Idempotent: re-running does not add duplicate keys.
    """
    cols = list(df.columns)
    added: List[str] = []
    if "variant_key" in cols:  # already harmonised
        return df, added

    chrom_col = _find_one(cols, _CHROM_NAMES)
    pos_col = _find_one(cols, _POS_NAMES)
    used = {c for c in (chrom_col, pos_col) if c}
    allele_pair = _find_allele_pair(cols, used)
    if allele_pair:
        used.update(allele_pair)

    coord_key = None
    if chrom_col and pos_col and allele_pair:
        coord_key = _key_from_coords(df, chrom_col, pos_col, *allele_pair)

    id_col = _detect_id_column(df, used)
    id_key = _key_from_id(df, id_col) if id_col else None

    # Primary = coordinate-derived when available (the table's declared build),
    # else the id-derived key. The other becomes _alt only if it actually differs.
    if coord_key is not None:
        df["variant_key"] = coord_key
        added.append("variant_key")
        if id_key is not None and (id_key.fillna("") != coord_key.fillna("")).any():
            df["variant_key_alt"] = id_key
            added.append("variant_key_alt")
    elif id_key is not None:
        df["variant_key"] = id_key
        added.append("variant_key")

    return df, added


# --- genome build awareness ----------------------------------------------------
#
# The membership keys above let a join survive cosmetic differences, but NOT a
# genome-build difference: a variant at GRCh38 chr6:159169344 and the SAME variant
# at GRCh37 chr6:159590376 are different coordinates and produce different keys, so
# a GRCh38 table and a GRCh37 table silently share almost nothing. `add_unified_
# variant_key` lifts every table into ONE build so a single `variant_key` joins
# them all. Build is inferred from conventions in the row -- a build token embedded
# in a packed id (``..._b38``), or, for a table carrying two disagreeing coordinate
# sources, a liftOver self-consistency check -- never from a per-file hard-coding.

# Canonical build labels. Keys are tokens as they appear in ids/filenames and the
# CLI's --build arg; values are the names pyliftover uses. Bare "37"/"38" are
# deliberately absent from the embedded-token regex below (they collide with
# positions); they appear here only to normalise an explicit --build 38.
_BUILD_CANON = {
    "hg38": "hg38", "grch38": "hg38", "b38": "hg38", "38": "hg38",
    "hg19": "hg19", "grch37": "hg19", "b37": "hg19", "37": "hg19",
    "hg18": "hg18", "grch36": "hg18", "b36": "hg18", "36": "hg18",
}
# A build token embedded in a value, bounded so it is not a stray substring of a
# longer alnum run (so "hg19" matches but the "38" of a position does not).
_BUILD_TOKEN_RE = re.compile(
    r"(?<![a-z0-9])(grch38|grch37|grch36|hg38|hg19|hg18|b38|b37|b36)(?![a-z0-9])",
    re.IGNORECASE,
)
_DEFAULT_TARGET_BUILD = "hg19"  # GRCh37: the predominant build for GWAS/MR summary stats


def _canon_build(build: Optional[str]) -> Optional[str]:
    if build is None:
        return None
    return _BUILD_CANON.get(str(build).strip().lower())


def _detect_build_token(values, sample: int = 500) -> Optional[str]:
    """Majority genome-build token across a sample of string values, or None."""
    import collections

    counts: "collections.Counter" = collections.Counter()
    seen = 0
    for v in values:
        m = _BUILD_TOKEN_RE.search(str(v))
        if m:
            counts[_BUILD_CANON[m.group(1).lower()]] += 1
        seen += 1
        if seen >= sample:
            break
    if not counts or seen == 0:
        return None
    build, n = counts.most_common(1)[0]
    # Require a majority of the sample so a lone stray token in a large table does
    # not relabel it, while a small table whose every id is labelled still counts.
    return build if n >= max(1, 0.5 * seen) else None


class _PositionLifter:
    """pyliftover-backed 1-based position lift between human builds, memoised.

    pyliftover is imported lazily and only constructed when a lift is actually
    needed, so a table set already in one build pulls in no optional dependency
    and fetches no chain file.
    """

    def __init__(self, source: str, target: str):
        from pyliftover import LiftOver  # optional dep; only on the lift path

        self._lo = LiftOver(source, target)
        self._cache: dict = {}

    def lift(self, chrom, pos):
        key = (chrom, pos)
        if key in self._cache:
            return self._cache[key]
        c = str(chrom).upper()
        ucsc = "chrM" if c in ("M", "MT") else "chr" + c
        res = self._lo.convert_coordinate(ucsc, int(pos) - 1)  # pyliftover is 0-based
        if not res:
            out = None
        else:
            nc, np0 = res[0][0], res[0][1]
            ncc = re.sub(r"(?i)^chr", "", str(nc)).upper()
            if ncc == "M":
                ncc = "MT"
            out = (ncc, np0 + 1)  # back to 1-based
        self._cache[key] = out
        return out


def _default_lifter_factory(source: str, target: str) -> _PositionLifter:
    return _PositionLifter(source, target)


def _representations(df) -> dict:
    """Variant representations present in ``df``, keyed 'coord' and/or 'id'.

    Each value carries the pieces a lift and a key both need: canonical chrom,
    numeric pos, allele-sorted lo/hi, a ready membership key, and the build token
    found in the id (None for coordinate columns, which rarely self-declare).
    """
    import pandas as pd

    cols = list(df.columns)
    reps: dict = {}

    def _pack(chrom, pos_num, a1, a2, build):
        a = a1.astype(str).str.upper().str.strip()
        b = a2.astype(str).str.upper().str.strip()
        lo = a.where(a <= b, b)
        hi = b.where(a <= b, a)
        key = _sorted_key(chrom, pos_num.astype("Int64").astype(str), lo, hi)
        return dict(chrom=chrom, pos=pos_num, lo=lo, hi=hi, key=key, build=build)

    chrom_col = _find_one(cols, _CHROM_NAMES)
    pos_col = _find_one(cols, _POS_NAMES)
    used = {c for c in (chrom_col, pos_col) if c}
    allele_pair = _find_allele_pair(cols, used)
    if allele_pair:
        used.update(allele_pair)
    if chrom_col and pos_col and allele_pair:
        reps["coord"] = _pack(
            _canon_chrom(df[chrom_col]),
            pd.to_numeric(df[pos_col], errors="coerce"),
            df[allele_pair[0]], df[allele_pair[1]],
            None,
        )

    id_col = _detect_id_column(df, used)
    if id_col:
        ext = df[id_col].astype(str).str.extract(_VARID_RE)
        if not ext.isna().all().all():
            chrom = ext[0].str.replace(r"^0+(?=\d)", "", regex=True).str.upper()
            reps["id"] = _pack(
                chrom,
                pd.to_numeric(ext[1], errors="coerce"),
                ext[2], ext[3],
                _detect_build_token(df[id_col].dropna().astype(str).head(500)),
            )
    return reps


def _reps_agree(reps: dict, min_frac: float = 0.9) -> bool:
    """Do the available representations describe the same coordinates?

    Two reps that agree are one effective source (e.g. a varId that merely repeats
    the coordinate columns), not a two-build crosswalk.
    """
    if len(reps) < 2:
        return True
    a, b = (r["key"] for r in reps.values())
    both = a.notna() & b.notna()
    n = int(both.sum())
    if n == 0:
        return False
    return float((a[both] == b[both]).mean()) >= min_frac


def _same_row_match(x: dict, y: dict, lifter, sample: int = 3000) -> float:
    """Fraction of rows where lifting x's position lands on y's position.

    x and y are two representations of the SAME rows, so this is a per-row check
    with no join: if lifting x from a candidate build reproduces y, x was in that
    build and y is in the build it was lifted to.
    """
    xc, xp, yc, yp = x["chrom"], x["pos"], y["chrom"], y["pos"]
    idx = xp.dropna().index.intersection(yp.dropna().index)
    if len(idx) == 0:
        return 0.0
    idx = idx[:sample]
    match = tot = 0
    for i in idx:
        tot += 1
        try:
            r = lifter.lift(xc[i], int(xp[i]))
        except Exception:
            r = None
        if r is not None and r[0] == str(yc[i]).upper() and r[1] == int(yp[i]):
            match += 1
    return match / tot if tot else 0.0


# Adjacent human-build pairs to try when resolving an unlabeled crosswalk table.
_LIFT_PROBE_PAIRS = [
    ("hg38", "hg19"), ("hg19", "hg38"),
    ("hg38", "hg18"), ("hg18", "hg38"),
    ("hg19", "hg18"), ("hg18", "hg19"),
]


def _lift_rep_key(rep: dict, lifter):
    """Membership key for ``rep`` after lifting its positions, alleles unchanged.

    Lifts only the UNIQUE (chrom, pos) pairs then maps back, so a million-row table
    costs as many liftOver lookups as it has distinct loci, not as it has rows.
    """
    import pandas as pd

    chrom, pos = rep["chrom"], rep["pos"]
    pos_i = pos.astype("Int64")
    ks = chrom.astype("string") + "|" + pos_i.astype("string")
    pairs = pd.DataFrame({"c": chrom, "p": pos_i}).dropna().drop_duplicates()
    cmap: dict = {}
    pmap: dict = {}
    for c, p in zip(pairs["c"].astype(str), pairs["p"].astype("int64")):
        r = lifter.lift(c, int(p))
        if r is not None:
            k = f"{c}|{p}"
            cmap[k], pmap[k] = r[0], str(r[1])
    new_c = ks.map(cmap)
    new_p = ks.map(pmap)
    return _sorted_key(new_c.fillna("NA"), new_p.fillna("NA"), rep["lo"], rep["hi"])


def _resolve_dual(reps: dict, target: str, lifter_factory):
    """Label the two builds of a crosswalk table, then key it in ``target``."""
    names = list(reps.keys())
    for s, t in _LIFT_PROBE_PAIRS:
        try:
            lifter = lifter_factory(s, t)
        except Exception as e:  # pyliftover missing / chain unavailable
            return None, f"liftOver unavailable ({e})"
        for xn in names:
            yn = next(n for n in names if n != xn)
            if _same_row_match(reps[xn], reps[yn], lifter) >= 0.8:
                reps[xn]["build"], reps[yn]["build"] = s, t
                if t == target:
                    return reps[yn]["key"], f"dual crosswalk: {yn}={t} (target), {xn}={s}"
                if s == target:
                    return reps[xn]["key"], f"dual crosswalk: {xn}={s} (target), {yn}={t}"
                try:
                    lf = lifter_factory(t, target)
                except Exception as e:
                    return None, f"liftOver {t}->{target} unavailable ({e})"
                return _lift_rep_key(reps[yn], lf), (
                    f"dual crosswalk: {yn}={t}->{target}; {xn}={s}"
                )
    return None, "two coordinate sources disagree but no build pair reconciles them"


def _choose_target_key(reps: dict, target: str, lifter_factory):
    """The single ``target``-build membership key for a table, and a note why."""
    # 1. A representation already labelled in the target build: use it as-is.
    for name, r in reps.items():
        if r["build"] == target:
            return r["key"], f"{name} already {target}"
    # 2. A representation with a KNOWN build != target: lift it.
    for name, r in reps.items():
        if r["build"] is not None and r["build"] != target:
            try:
                lifter = lifter_factory(r["build"], target)
            except Exception as e:
                return None, f"liftOver {r['build']}->{target} unavailable ({e})"
            return _lift_rep_key(r, lifter), f"lifted {name} {r['build']}->{target}"
    # 3. No build label anywhere.
    if len(reps) == 1 or _reps_agree(reps):
        # One effective source: with nothing to pin the build, assume it is the
        # target. (Lifting is impossible without a source build, so this is the
        # only coherent choice; a wrong assumption shows up as an empty join, not
        # a silent mis-key, because the lifted tables simply will not meet it.)
        any_rep = next(iter(reps.values()))
        return any_rep["key"], f"assumed {target} (no build label, single source)"
    # 4. Two disagreeing coordinate sources: a crosswalk -- resolve by liftOver.
    return _resolve_dual(reps, target, lifter_factory)


def add_unified_variant_key(df, target_build: str = _DEFAULT_TARGET_BUILD,
                            lifter_factory=None) -> Tuple["object", List[str], str]:
    """Add ONE canonical ``variant_key`` to ``df`` in ``target_build``.

    Build-aware counterpart of :func:`add_variant_keys`: where a table's variant
    is in a different build than ``target_build``, its positions are lifted so a
    single key joins every table in one build. General and convention-driven --
    build is read from the row (an id build token, or liftOver self-consistency for
    a two-source crosswalk table), never from a per-file map. Non-destructive and
    idempotent; returns ``(df, added, note)``. ``added`` is ``[]`` (and the frame
    is untouched) when there is no detectable variant or the build cannot be
    resolved -- callers then keep whatever membership key they already had.
    """
    if "variant_key" in df.columns:
        return df, [], "already has variant_key"
    target = _canon_build(target_build) or _DEFAULT_TARGET_BUILD
    if lifter_factory is None:
        lifter_factory = _default_lifter_factory
    reps = _representations(df)
    if not reps:
        return df, [], "no variant columns detected"
    key, note = _choose_target_key(reps, target, lifter_factory)
    if key is None:
        return df, [], note
    df["variant_key"] = key
    return df, ["variant_key"], note


# --- file-level entry points ---------------------------------------------------

def _atomic_write_csv(df, target: str) -> None:
    import os
    import tempfile

    d = os.path.dirname(os.path.abspath(target)) or "."
    fd, tmp = tempfile.mkstemp(suffix=".csv", dir=d)
    os.close(fd)
    try:
        df.to_csv(tmp, index=False)
        os.replace(tmp, target)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


def _read_table(path: str):
    import pandas as pd

    from kosmos.data.table_head import sniff_head  # correct header/skiprows

    kwargs = {}
    try:
        kwargs = sniff_head(path).pandas_kwargs()
    except Exception:
        pass
    return pd.read_csv(path, **kwargs)


def harmonize_csv(path: str, out_path: Optional[str] = None,
                  target_build: Optional[str] = None,
                  lifter_factory=None) -> List[str]:
    """Read a CSV, add variant key column(s), write it back (or to out_path).

    With ``target_build`` set, adds one build-unified ``variant_key`` (lifting as
    needed); otherwise adds the build-agnostic membership key(s). Returns the
    columns added ([] if none). Writes atomically (temp file + rename).
    """
    df = _read_table(path)
    if target_build is not None:
        df, added, _ = add_unified_variant_key(df, target_build, lifter_factory)
    else:
        df, added = add_variant_keys(df)
    if not added:
        return []
    _atomic_write_csv(df, out_path or path)
    return added


def harmonize_tables(paths, target_build: Optional[str] = None,
                     lifter_factory=None, log=None) -> dict:
    """Stamp ONE build-unified ``variant_key`` across several tables, in place.

    The general multi-table entry point. Each file is keyed independently in a
    common build (``target_build``, defaulting to GRCh37/hg19 -- the predominant
    build for GWAS/MR summary statistics), lifting only tables whose build differs.
    Best-effort: a table with no variant columns is skipped, and a failure on one
    table never stops the others. Returns ``{path: note}`` for the tables keyed.
    """
    target = _canon_build(target_build) or _DEFAULT_TARGET_BUILD

    def _say(msg: str) -> None:
        if log is not None:
            log(msg)

    out: dict = {}
    for path in paths:
        try:
            df = _read_table(path)
        except Exception as e:  # unreadable / not a table
            _say(f"  [variant-key] {path}: skipped ({e})")
            continue
        # Drop any pre-existing derived keys so runtime unification is authoritative
        # and idempotent: a file keyed by an earlier (non-build-aware) pass, or a
        # previous run, is re-keyed fresh in the target build rather than skipped.
        stale = [c for c in ("variant_key", "variant_key_alt") if c in df.columns]
        if stale:
            df = df.drop(columns=stale)
        try:
            df, added, note = add_unified_variant_key(df, target, lifter_factory)
        except Exception as e:  # lift failure, malformed coords, ...
            _say(f"  [variant-key] {path}: no key ({e})")
            continue
        if not added:
            _say(f"  [variant-key] {path}: no variant_key ({note})")
            continue
        try:
            _atomic_write_csv(df, path)
        except Exception as e:
            _say(f"  [variant-key] {path}: computed but could not write ({e})")
            continue
        out[path] = note
        _say(f"  [variant-key] {path}: variant_key in {target} -- {note}")
    return out


if __name__ == "__main__":  # python -m kosmos.data.variant_keys [--build hg19] FILE [...]
    import sys

    argv = sys.argv[1:]
    build = None
    files = []
    i = 0
    while i < len(argv):
        if argv[i] == "--build" and i + 1 < len(argv):
            build = argv[i + 1]
            i += 2
        else:
            files.append(argv[i])
            i += 1
    if build is not None:
        notes = harmonize_tables(files, target_build=build, log=print)
        for p in files:
            print(f"{p}: {'added variant_key -- ' + notes[p] if p in notes else 'no variant_key'}")
    else:
        for p in files:
            try:
                cols = harmonize_csv(p)
                print(f"{p}: added {cols}" if cols else f"{p}: no variant columns detected")
            except Exception as e:  # noqa: BLE001
                print(f"{p}: FAILED -- {e}")
