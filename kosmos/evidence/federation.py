"""Several AutoEvidence gateways in one run, kept straight.

`evidence/client.py` speaks to ONE gateway: it opens a session, negotiates, and
hands back one capsule. That is the right shape for it and does not change here.
This module is the layer above -- it reads the operator's list of sources, calls
that client once per source, and returns one materialised record per source for
the research loop to ground itself in.

**What this does and does not do.** It does not join anything and does not merge
capsules: each source is materialised to its own file in its own directory, and
whatever relating happens is done by the experiment's own code over those files.
What it adds is *grounding* -- hypothesis generation seeing every dataset at
once, which is what an AI scientist needs in order to propose a relationship
that spans them.

Whether an experiment may then OPEN several of those files is a separate
decision, and it is made here by `mountable_together()` on the join key rather
than on the dataset count. Feature-keyed sources (gene, variant, cell line) may
be co-mounted: joining them relates features and re-identifies nobody. Two
subject-keyed sources may not, unless a steward has declared their subjects
`disjoint` -- two individual-level tables in one container can be joined on
their subject key whatever the code was asked to do, so the gate is on the
mount, not on the analysis. When the gate allows it the executor mounts every
file (`execution/executor.py`, the `data_files` dict it builds); when it does
not, the run falls back to one file per experiment.

**Why the sources live in a file rather than repeated flags.** A source is four
or five coupled values (where, which dataset, which credential, what the steward
says about subject overlap). Repeated single-valued flags would have to be
paired positionally, and a mis-pairing would silently authenticate to one
gateway with another's key. Same reasoning as AutoEvidence's own
`sources.yaml`: coupled configuration belongs in one reviewable document.

**Credentials are named, never written.** A source names an environment variable
holding its key. The value is read at load, kept on the source object, passed to
the client per call, and never logged, printed, or written into a report -- the
same discipline the gateway applies to its own source credentials one layer
down.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Optional

CONFIG_ENV_VAR = "KOSMOS_EVIDENCE_CONFIG"
DEFAULT_CONFIG_NAMES = ("evidence.yaml",)
USER_CONFIG_PATH = Path.home() / ".kosmos" / "evidence.yaml"

# What a steward says about whether two datasets describe the same subjects.
# Three values, and the default is the one that assumes the most: a run that was
# never told cannot claim independence. Kosmos only RECORDS this -- it is the
# gateway's job to enforce anything, and a declaration living in the caller's
# own config could not bind the caller anyway. It is here so that the report can
# say what was assumed, and so a later steward-side population file has
# something to be checked against.
OVERLAP_VALUES = ("shared", "disjoint", "unknown")


class EvidenceConfigError(RuntimeError):
    """The evidence source list is missing, unreadable, or malformed.

    A startup error, never a mid-run degradation: a run that silently proceeded
    on three of four sources would report as though it had seen four.
    """


@dataclass(frozen=True)
class EvidenceSource:
    """One gateway this run will ask.

    `name` is the identity used for the staging directory and the container-side
    filename, NOT the dataset id. Two stewards may both call their dataset
    `cohort`, and keying the filesystem on the dataset id would have one
    overwrite the other -- a collision the operator never created and could not
    see. `name` is unique by construction (the loader refuses duplicates).
    """

    name: str
    server: str
    dataset: Optional[str] = None
    api_key: Optional[str] = None
    subject_overlap: str = "unknown"
    primary: bool = False
    # The column, if any, that identifies a PERSON (or other protected unit) in
    # this dataset. Declaring one is what makes a dataset subject-keyed; leaving
    # it out says the rows are keyed on something that is not a subject -- a
    # gene, a variant, a transcript, a cell line.
    #
    # This is the distinction that decides whether two datasets may be opened by
    # one experiment, and it is NOT the same question as how many datasets there
    # are. Joining two perturbation screens on target gene, or two GWAS on SNP,
    # links features and re-identifies nobody -- it is also most of the science
    # anyone wants from several omics datasets. Joining two clinical tables on
    # patient id is the thing every disclosure control in AutoEvidence exists to
    # prevent. Gating on dataset count would block the first to prevent the
    # second; gating on the key class blocks only the second.
    subject_key: Optional[str] = None
    # The source this one was PUBLISHED WITH and split from -- an expression
    # matrix's own sample annotation, named by that matrix's source `name`.
    #
    # It exists because splitting one published object into two files must not
    # make its halves unjoinable. A GEO series matrix carries the values and the
    # per-sample labels in ONE file; the connector separates them because they
    # are two schemas, and without this the gate would then refuse to let an
    # experiment see both -- leaving a run with expression values and no way to
    # tell which column is which group. Observed exactly so: the generated
    # analysis invented its groups from column order.
    #
    # This grants no authority the operator did not already have (they can
    # declare `subject_overlap: disjoint` and get the same mount). It only names
    # the relationship accurately, so the reason appears in the decision rather
    # than as an unexplained override.
    companion_of: Optional[str] = None

    @property
    def subject_keyed(self) -> bool:
        return bool(self.subject_key)

    def redacted(self) -> dict[str, Any]:
        """Everything about this source except the secret. For logs and reports."""
        return {
            "name": self.name,
            "server": self.server,
            "dataset": self.dataset,
            "subject_overlap": self.subject_overlap,
            "subject_key": self.subject_key,
            "companion_of": self.companion_of,
            "primary": self.primary,
            "authenticated": bool(self.api_key),
        }


@dataclass
class MaterializedSource:
    """What one source actually yielded.

    Three shapes arrive, and the difference matters to the model, not just to the
    plumbing: a gated capsule of banded statistics, a full release of exact rows,
    and a schema with no rows at all. `release_regime` is what turns that
    difference into a sentence the hypothesis prompt can carry -- see
    `describe_release`.
    """

    source: EvidenceSource
    dataset_id: str
    kind: str  # "evidence" | "open_data" | "schema" | "failed"
    path: Optional[Path] = None
    menu: Optional[dict] = None
    body: dict = field(default_factory=dict)
    error: Optional[str] = None

    @property
    def has_rows(self) -> bool:
        return self.path is not None

    def describe_release(self) -> str:
        """One sentence stating what this source actually released.

        The distinction this draws is load-bearing and was previously got wrong:
        a policy-granted BULK release carries the same menu as a gated one, and
        describing it as "banded aggregates, never raw rows" tells the model the
        opposite of the truth about data it is holding in full. A model told its
        exact values are bands will hedge conclusions it is entitled to draw, and
        will read band labels as measurements.
        """
        if self.kind == "open_data":
            n_rows = self.body.get("n_rows")
            n_cols = self.body.get("n_columns")
            size = (
                f" ({n_rows:,} rows x {n_cols} columns)"
                if isinstance(n_rows, int) and n_cols
                else ""
            )
            return (
                f"released IN FULL{size}: exact values, one row per record, no "
                f"banding or suppression applied to the released columns"
            )
        if self.kind == "schema":
            return (
                "schema only: column names and types, NO rows and no statistics "
                "released. Hypotheses may be framed over these variables but "
                "cannot be tested against this source in this run"
            )
        if self.kind == "failed":
            return f"UNAVAILABLE: {self.error or 'no capsule was released'}"
        return (
            "released ONLY as banded aggregates (summary statistics -- banded "
            "counts, banded means), never raw rows. Do NOT treat band labels "
            "(e.g. 'n_band') as measurements"
        )


# --- what may be opened together -------------------------------------------

@dataclass(frozen=True)
class MountDecision:
    """Whether several sources may be handed to ONE experiment, and why."""

    allowed: bool
    reason: str
    #: The sources that may be mounted together. On refusal, just the primary.
    names: tuple[str, ...] = ()


def mountable_together(materialized: list["MaterializedSource"]) -> MountDecision:
    """May these sources be opened by a single experiment?

    The rule is about the JOIN KEY, not the dataset count, and that distinction
    is the whole point:

      * A source with no `subject_key` is keyed on something that is not a
        person -- a gene, a variant, a cell line, a summary statistic. Two such
        sources joined on that key relate features and re-identify nobody. This
        is also most of what anyone wants from several omics datasets: does the
        effect measured in screen A reproduce in screen B, does an eQTL agree
        with a pQTL, does an exposure GWAS line up with an outcome GWAS on the
        same SNP. Refusing it protects no one and blocks the science.

      * Two sources that BOTH declare a `subject_key` are individual-level
        tables about people. Joining them on that key is precisely the linkage
        every gate in AutoEvidence exists to prevent, and it becomes reachable
        the moment both files sit in one container -- whatever the code was
        asked to do, and with no gate in a position to see it.

    So subject-keyed sources are mounted together only when a steward has said
    the subjects are `disjoint`. `shared` refuses, and so does `unknown`: a run
    that was never told cannot claim there is nothing to link. That is the same
    fail-closed default the overlap declaration carries everywhere else.

    A refusal is not a failure -- it falls back to one dataset per experiment,
    which is the Phase 1a behaviour and still supports comparing findings across
    datasets in the report. Nothing is lost except the ability to open two
    person-level tables in one script.
    """
    usable = [m for m in materialized if m.has_rows]
    if len(usable) <= 1:
        return MountDecision(
            allowed=False,
            reason="only one source released rows; nothing to mount together",
            names=tuple(m.source.name for m in usable),
        )

    keyed = [m for m in usable if m.source.subject_keyed]
    if len(keyed) > 1:
        # A COMPANION and its parent are one published object, not two tables
        # about different people. A GEO series matrix ships values and
        # per-sample labels in a single file; the connector splits them because
        # they are two schemas, and rejoining them inside the container reveals
        # nothing the source did not already publish as one download. Refusing
        # that is not caution -- it leaves a run holding an expression matrix
        # with no way to say which column is which group, which is how a report
        # came to be written over groups invented from column order.
        #
        # Kept deliberately narrow: the exemption applies only to a source that
        # NAMES its parent and whose parent is in this same mount set. It does
        # not generalise to two datasets that merely share a subject key, and it
        # is not reachable by a source declaring itself a companion of something
        # absent.
        present = {m.source.name for m in usable}
        paired = {
            m.source.name
            for m in keyed
            if m.source.companion_of and m.source.companion_of in present
        }
        paired |= {
            m.source.companion_of
            for m in keyed
            if m.source.companion_of and m.source.companion_of in present
        }
        risky = [
            m for m in keyed
            if m.source.subject_overlap != "disjoint" and m.source.name not in paired
        ]
        if risky:
            names = ", ".join(
                f"{m.source.name} (subject_key={m.source.subject_key!r}, "
                f"overlap={m.source.subject_overlap})"
                for m in risky
            )
            return MountDecision(
                allowed=False,
                reason=(
                    f"refusing to mount subject-keyed datasets together: {names}. "
                    f"Two individual-level tables in one container can be joined "
                    f"on their subject key regardless of what the code was asked "
                    f"to do. Declare subject_overlap: disjoint if the subjects "
                    f"genuinely cannot overlap, or leave these to separate "
                    f"experiments and compare their findings in the report."
                ),
                names=(),
            )

    # Say which rule allowed it. A companion pair is permitted for a different
    # reason than a pair of feature-keyed sources, and a decision that gave the
    # feature-keyed reason for a subject-keyed pair would misdescribe itself to
    # whoever audits the run.
    companions = sorted(
        f"{m.source.name} -> {m.source.companion_of}"
        for m in usable
        if m.source.companion_of and m.source.companion_of in {
            u.source.name for u in usable
        }
    )
    if companions:
        reason = (
            "co-mounted as published: " + ", ".join(companions) + ". A companion "
            "and its parent are one published object split into two schemas, so "
            "rejoining them exposes nothing the source did not publish together"
        )
        if any(m.source.subject_keyed for m in usable if not m.source.companion_of
               and m.source.name not in {c.split(" -> ")[1] for c in companions}):
            reason += "; every other co-mounted source is feature-keyed or disjoint"
    else:
        reason = (
            "all co-mounted sources are feature-keyed (or declared disjoint), so "
            "any join between them relates features rather than subjects"
        )
    return MountDecision(
        allowed=True,
        reason=reason,
        names=tuple(m.source.name for m in usable),
    )


# --- loading ---------------------------------------------------------------

def find_config(explicit: Optional[str | Path] = None) -> Optional[Path]:
    """The evidence config to use, or None. Mirrors AutoEvidence's search order.

    An explicit path that does not exist is an error rather than a fallback: a
    run that silently used a different source list than the operator named would
    be reasoning over data nobody chose.
    """
    if explicit is not None:
        path = Path(explicit).expanduser()
        if not path.exists():
            raise EvidenceConfigError(f"evidence config not found: {path}")
        return path

    from_env = os.environ.get(CONFIG_ENV_VAR)
    if from_env:
        path = Path(from_env).expanduser()
        if not path.exists():
            raise EvidenceConfigError(
                f"${CONFIG_ENV_VAR} points at {path}, which does not exist"
            )
        return path

    for name in DEFAULT_CONFIG_NAMES:
        candidate = Path.cwd() / name
        if candidate.exists():
            return candidate
    if USER_CONFIG_PATH.exists():
        return USER_CONFIG_PATH
    return None


def load_sources(explicit: Optional[str | Path] = None) -> list[EvidenceSource]:
    """Read the source list. Raises EvidenceConfigError on anything malformed."""
    path = find_config(explicit)
    if path is None:
        raise EvidenceConfigError(
            "no evidence config found (searched --evidence-config, "
            f"${CONFIG_ENV_VAR}, ./evidence.yaml, {USER_CONFIG_PATH})"
        )
    try:
        import yaml

        raw = yaml.safe_load(Path(path).read_text()) or {}
    except OSError as e:
        raise EvidenceConfigError(f"cannot read evidence config {path}: {e}") from e
    except Exception as e:  # yaml.YAMLError and friends
        raise EvidenceConfigError(f"invalid YAML in evidence config {path}: {e}") from e

    if not isinstance(raw, dict):
        raise EvidenceConfigError(f"evidence config {path} must be a mapping")
    entries = raw.get("sources")
    if not isinstance(entries, list) or not entries:
        raise EvidenceConfigError(
            f"evidence config {path} must define a non-empty `sources:` list"
        )

    sources = _parse_entries(entries, path)
    _check_names(sources, path)
    return _assign_primary(sources, path)


def _parse_entries(entries: Iterable[Any], path: Path) -> list[EvidenceSource]:
    known = {
        "name", "server", "dataset", "key_env", "subject_overlap", "primary",
        "subject_key", "companion_of",
    }
    out: list[EvidenceSource] = []
    for i, entry in enumerate(entries):
        where = f"{path} sources[{i}]"
        if not isinstance(entry, dict):
            raise EvidenceConfigError(f"{where} must be a mapping")
        # Refuse unknown keys rather than ignoring them: a misspelled `key_env`
        # would otherwise mean an unauthenticated call whose 401 surfaces much
        # later as a transport failure.
        unknown = set(entry) - known
        if unknown:
            raise EvidenceConfigError(
                f"{where} has unknown key(s) {sorted(unknown)}; "
                f"known keys: {sorted(known)}"
            )
        server = entry.get("server")
        if not server or not isinstance(server, str):
            raise EvidenceConfigError(f"{where} must name a `server`")
        name = entry.get("name") or entry.get("dataset")
        if not name or not isinstance(name, str):
            raise EvidenceConfigError(
                f"{where} needs a `name` (used for its staging directory) "
                f"or a `dataset` to borrow one from"
            )
        overlap = entry.get("subject_overlap", "unknown")
        if overlap not in OVERLAP_VALUES:
            raise EvidenceConfigError(
                f"{where}: subject_overlap must be one of {list(OVERLAP_VALUES)}, "
                f"got {overlap!r}. Absent means 'unknown', which assumes the most."
            )
        key_env = entry.get("key_env")
        api_key = None
        if key_env:
            api_key = os.environ.get(key_env)
            if not api_key:
                raise EvidenceConfigError(
                    f"{where}: key_env names ${key_env}, which is unset or empty. "
                    f"Export it, or drop key_env for an unauthenticated gateway."
                )
        subject_key = entry.get("subject_key")
        if subject_key is not None and (
            not isinstance(subject_key, str) or not subject_key.strip()
        ):
            raise EvidenceConfigError(
                f"{where}: subject_key must be the NAME of the column that "
                f"identifies a subject, or absent if this dataset is not keyed "
                f"on subjects (e.g. keyed on gene or variant)."
            )
        companion_of = entry.get("companion_of")
        if companion_of is not None and (
            not isinstance(companion_of, str) or not companion_of.strip()
        ):
            raise EvidenceConfigError(
                f"{where}: companion_of must be the NAME of the source this one "
                f"was published with and split from, or absent."
            )
        out.append(EvidenceSource(
            name=name,
            server=server,
            dataset=entry.get("dataset"),
            api_key=api_key,
            subject_overlap=overlap,
            primary=bool(entry.get("primary", False)),
            subject_key=subject_key,
            companion_of=companion_of,
        ))
    return out


def _check_names(sources: list[EvidenceSource], path: Path) -> None:
    seen: set[str] = set()
    for s in sources:
        if s.name in seen:
            raise EvidenceConfigError(
                f"{path}: duplicate source name {s.name!r}. Names key the staging "
                f"directory, so a duplicate would have one source overwrite another."
            )
        seen.add(s.name)


def _assign_primary(sources: list[EvidenceSource], path: Path) -> list[EvidenceSource]:
    """Exactly one source is the primary: the file `data_path` binds to.

    Every experiment in this phase runs against one dataset's file, and the
    primary is which one that is by default. Declared rather than inferred where
    the operator cares; the first source otherwise, which is the least surprising
    reading of an ordered list.
    """
    declared = [s for s in sources if s.primary]
    if len(declared) > 1:
        raise EvidenceConfigError(
            f"{path}: {len(declared)} sources are marked primary "
            f"({', '.join(s.name for s in declared)}); exactly one may be"
        )
    if declared:
        return sources
    first, *rest = sources
    return [
        EvidenceSource(**{**first.__dict__, "primary": True}), *rest
    ]


# --- materialising ---------------------------------------------------------

def materialize_sources(
    sources: list[EvidenceSource],
    run_dir: str | Path,
    *,
    fetch: Optional[Callable[..., dict]] = None,
    stage: Optional[Callable[..., Path]] = None,
    log: Optional[Callable[[str], None]] = None,
) -> list[MaterializedSource]:
    """Fetch every source and stage each into its own directory under `run_dir`.

    One subdirectory per source, named for the source. This is not tidiness: the
    previous single-source path wrote `$TMPDIR/evidence_<dataset>.csv` and never
    removed it, so one run's data sat in a world-readable location where the next
    run's in-process execution fallback could read it -- a cross-dataset channel
    that opens by itself once N files are written per run.

    A source that fails is recorded as `kind="failed"` and carried, not dropped:
    the caller decides whether a run may proceed without it, and the report must
    be able to say which sources were missing. Losing that quietly is how a run
    reports on four datasets having seen three.

    `fetch` and `stage` are injectable so the pure staging logic is testable
    without a gateway; they default to the real client.
    """
    if fetch is None or stage is None:
        from kosmos.evidence.client import fetch_source, materialize

        fetch = fetch or fetch_source
        stage = stage or materialize

    def _say(msg: str) -> None:
        if log is not None:
            log(msg)

    run_root = Path(run_dir)
    out: list[MaterializedSource] = []
    for source in sources:
        out.append(_materialize_one(source, run_root, fetch, stage, _say))
    return out


def _materialize_one(
    source: EvidenceSource,
    run_root: Path,
    fetch: Callable[..., dict],
    stage: Callable[..., Path],
    say: Callable[[str], None],
) -> MaterializedSource:
    try:
        result = fetch(source.server, source.dataset, api_key=source.api_key)
    except Exception as e:
        # The gateway's own reason, never the credential that reached it.
        say(f"  [evidence] {source.name}: FAILED -- {e}")
        return MaterializedSource(
            source=source, dataset_id=source.dataset or source.name,
            kind="failed", error=str(e),
        )

    kind = result.get("kind", "evidence")
    signed = result.get("signed") or {}
    body = signed.get("capsule", signed) if isinstance(signed, dict) else {}
    dataset_id = source.dataset or body.get("dataset") or source.name
    menu = result.get("menu")

    if kind == "schema":
        say(
            f"  [evidence] {source.name} ({dataset_id}): SCHEMA ONLY -- "
            f"{len(body.get('columns') or [])} columns, no rows released"
        )
        return MaterializedSource(
            source=source, dataset_id=dataset_id, kind="schema", body=body, menu=menu,
        )

    target_dir = run_root / source.name
    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / f"{source.name}.csv"
    try:
        stage(signed, target)
    except Exception as e:
        say(f"  [evidence] {source.name}: staging FAILED -- {e}")
        return MaterializedSource(
            source=source, dataset_id=dataset_id, kind="failed", error=str(e),
            body=body, menu=menu,
        )

    mat = MaterializedSource(
        source=source, dataset_id=dataset_id, kind=kind, path=target,
        menu=menu, body=body,
    )
    say(f"  [evidence] {source.name} ({dataset_id}): {mat.describe_release()}")
    say(f"  [evidence] {source.name}: staged to {target}")
    return mat
