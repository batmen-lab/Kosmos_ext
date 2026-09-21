"""A DataPlan: which fetched tables are gold, which are supplementary, why.

This is the contract between the two packages, and the only one. `datafetcher`
writes it; Kosmos reads it; neither imports the other. It is a file rather than
a function call for three reasons: it can be read before anything runs, it is
auditable afterwards, and it pins the exact bytes (sha256) a run consumed.

A plan is *not* a permission document -- there is no gating in this board. It
records a decision that was made, so that a wrong task mapping is visible
immediately instead of surfacing as an unexplained metric.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from .models import now_iso
from .profile import (
    TableProfile,
    TaskShape,
    classify_role,
    cross_species_reason,
    normalize_column_name,
    profile_table,
    required_shared,
)
from .store import MANIFEST_NAME

PLAN_VERSION = 1

Role = Literal["gold", "supplementary", "unusable"]


class PlanTask(BaseModel):
    """The task half of the plan: what Kosmos builds its TaskSpec from."""

    model_config = ConfigDict(extra="forbid")

    objective: str = ""
    #: None means "not decided yet" -- the plan then carries no roles either.
    target_column: str | None = None
    #: Who decided the target: cli | protocol | llm | human.
    target_source: str = "cli"
    #: declared | exact_match | inferred
    target_confidence: str = "declared"
    feature_columns: list[str] | None = None
    feature_prefixes: list[str] = Field(default_factory=list)
    exclude_columns: list[str] = Field(default_factory=list)
    sample_id_column: str | None = None
    task_type: str = "classification"
    evaluation_metric: str = "balanced_accuracy"
    #: What to do when more than one table carries the label column.
    #:   external -- one becomes gold, the rest become supplementary evidence:
    #:               their features are used and their labels are ignored, so
    #:               they enter through the PPI correction
    #:   pool     -- concatenate them into one gold table (labels used)
    #:   error    -- refuse, and make the caller choose
    multi_gold_policy: Literal["external", "pool", "error"] = "external"
    test_fraction: float = 0.2
    validation_fraction: float = 0.2
    seed: int = 42

    def to_shape(self) -> TaskShape:
        return TaskShape(
            target_column=self.target_column,
            feature_columns=tuple(self.feature_columns) if self.feature_columns else None,
            feature_prefixes=tuple(self.feature_prefixes),
            exclude_columns=tuple(self.exclude_columns),
            sample_id_column=self.sample_id_column,
            description=self.objective,
        )


def align_to_gold(
    entry: PlanEntry, gold_features: list[str], *, min_shared: int | None = None
) -> str | None:
    """Match a table's columns to the gold's, by name. None means it fits.

    A supplementary table needs the gold's *features* -- its own labels are
    ignored, and any column it carries that the task does not use is not a
    reason to refuse it. It does not need all of them: what it shares is the
    intersection the two tables can be compared on, and the run trains both arms
    on exactly that intersection, so a table that measured sex, height and
    weight is still evidence for a question whose gold also has age.

    Returns the reason when it cannot be used, and records the shared columns
    and the renames on the entry when it can (`concave points_mean` ->
    `concave_points_mean`).
    """
    own = list(entry.features)
    # The gold's naming convention against this table's. This is the gate that
    # keeps a mouse table out of a human gold's correction: `Pisd` and `PISD`
    # normalise to the same string, so without it the Baron mouse samples
    # "share" 12,473 genes with the human ones and enter the correction as if an
    # ortholog were the same measurement. Checked here as well as in the role
    # rules, which only see a naming convention when the caller names columns.
    species = cross_species_reason(gold_features, own)
    if species:
        return species
    by_norm: dict[str, list[str]] = {}
    for name in own:
        by_norm.setdefault(normalize_column_name(name), []).append(name)
    renames: dict[str, str] = {}
    shared: list[str] = []
    ambiguous: list[str] = []
    #: This table's column -> the one gold feature it can serve. A single column
    #: cannot be two features: the gold may carry `CD1D` and `CD1d` as separate
    #: genes, and a table with one `CD1d` column can only supply one of them. The
    #: second is left out, because a rename that maps one column to two names
    #: leaves the missing one to be reported at training time.
    claims: dict[str, str] = {}
    for name in gold_features:
        candidates = by_norm.get(normalize_column_name(name)) or []
        if not candidates:
            continue
        if len(candidates) > 1:
            ambiguous.append(name)
            continue
        column = candidates[0]
        if column in claims:
            ambiguous.append(name)
            continue
        claims[column] = name
        shared.append(name)
        if column != name:
            renames[column] = name
    if ambiguous:
        # Two of a table's columns normalising to the same gold feature (a
        # `CD1D` and a `CD1D.1` from two annotation passes) is a naming
        # collision, not a reason to throw the table away: those columns are
        # left out of the intersection and the ones that are unambiguous carry
        # the evidence. Refusing the whole table over 2 of 129,922 gold columns
        # is how a usable second cohort is lost.
        ambiguous_set = set(ambiguous)
        shared = [name for name in shared if name not in ambiguous_set]
        entry.dropped_features = list(
            dict.fromkeys([*entry.dropped_features, *ambiguous])
        )
    threshold = required_shared(gold_features) if min_shared is None else min_shared
    if len(shared) < threshold:
        return (
            f"it shares only {len(shared)} measured column(s) with the gold, and "
            f"{threshold} are needed to correct the same model"
        )
    if ambiguous:
        entry.reason += (
            f"; {len(ambiguous)} column(s) were left out of the intersection "
            f"because more than one of the table's columns normalises to the "
            f"gold's name, e.g. {ambiguous[:3]}"
        )
    supplied = set(renames) | set(shared)
    extra = [name for name in own if name not in supplied]
    # Merged, not replaced: the demotion path and the schema check can both see
    # the table, and the second pass used to blank the first pass's mapping.
    entry.column_renames = {**entry.column_renames, **renames}
    entry.dropped_features = list(dict.fromkeys([*entry.dropped_features, *extra]))
    # What this table contributes is the intersection, in the gold's column
    # order. The columns it does not have are not a rejection; they are the
    # reason the correction is trained on fewer of the gold's inputs.
    entry.shared_features = list(shared)
    entry.features = list(shared)
    if extra:
        entry.reason += (
            f"; {len(extra)} of its column(s) are not part of this task and are "
            f"ignored, e.g. {extra[:3]}"
        )
    if renames:
        entry.reason += (
            f"; {len(renames)} column(s) matched the gold after normalising the "
            f"name, e.g. {list(renames.items())[:2]}"
        )
    if len(shared) < len(gold_features):
        entry.reason += (
            f"; it shares {len(shared)} of the gold's {len(gold_features)} "
            f"feature(s) and the run trains on that intersection"
        )
    return None


class PlanEntry(BaseModel):
    """One table, the role it can play, and the reason it has that role."""

    model_config = ConfigDict(extra="forbid")

    path: str
    role: Role
    reason: str
    n_features: int = 0
    features_sample: list[str] = Field(default_factory=list)
    #: The full ordered feature list a role decision used. Kept so a demoted
    #: table can be checked against the primary's schema here, rather than
    #: failing later inside the run.
    features: list[str] = Field(default_factory=list)
    #: For supplementary tables: the gold's columns this table actually has.
    #: The run trains both arms on the intersection of these, so what a table
    #: cannot supply is not a refusal -- it is the width the correction runs at.
    shared_features: list[str] = Field(default_factory=list)
    #: Selected columns that are not features: free text, or a row index. Named
    #: here so the plan explains what was left out rather than only what is in.
    dropped_features: list[str] = Field(default_factory=list)
    #: This table's column name -> the gold's, for columns that name the same
    #: measurement in a different spelling. The run has to apply these before it
    #: can feed the table to the model.
    column_renames: dict[str, str] = Field(default_factory=dict)
    rows: int | None = None
    columns: int = 0
    #: From the fetch manifest when this file was staged by the fetcher.
    provenance: dict[str, Any] = Field(default_factory=dict)
    #: The LLM review of this table, recorded verbatim: what it said the table
    #: is, and the role it proposed. A proposal, not a verdict -- the mechanical
    #: checks below still decide (see `_apply_reviews`).
    review: dict[str, Any] = Field(default_factory=dict)
    #: For supplementary tables: how their labels were handled.
    #:   unlabeled  -- there was no label column
    #:   bad_label  -- there was one, and the review said not to train on it
    label_provenance: str = ""

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump(mode="json")


class DataPlan(BaseModel):
    """The whole file: a task, three role buckets, and where the data came from."""

    model_config = ConfigDict(extra="forbid")

    version: int = PLAN_VERSION
    created_at: str = Field(default_factory=now_iso)
    task: PlanTask = Field(default_factory=PlanTask)
    gold: list[PlanEntry] = Field(default_factory=list)
    supplementary: list[PlanEntry] = Field(default_factory=list)
    unusable: list[PlanEntry] = Field(default_factory=list)
    #: Search/adapter provenance, so the plan says how the files were found.
    evidence: dict[str, Any] = Field(default_factory=dict)
    notes: list[str] = Field(default_factory=list)

    def summary(self) -> str:
        target = self.task.target_column
        parts = [
            f"task: predict {target!r} "
            f"({self.task.target_source}, {self.task.target_confidence})"
            if target
            else "task: no target column decided yet",
            f"features: prefixes={list(self.task.feature_prefixes)} "
            f"exclude={list(self.task.exclude_columns)}",
            f"gold: {len(self.gold)} table(s)",
        ]
        for entry in self.gold:
            parts.append(f"  + {entry.path}  ({entry.rows or '?'} rows, {entry.n_features} features)")
        parts.append(f"supplementary: {len(self.supplementary)} table(s)")
        for entry in self.supplementary:
            parts.append(f"  + {entry.path}  ({entry.rows or '?'} rows)")
        if self.unusable:
            parts.append(f"unusable: {len(self.unusable)} table(s)")
            for entry in self.unusable:
                parts.append(f"  - {entry.path}: {entry.reason}")
        for note in self.notes:
            parts.append(f"note: {note}")
        return "\n".join(parts)


def _provenance_index(root: str | Path | None) -> dict[str, dict[str, Any]]:
    """Map every manifest file path back to the record that staged it."""
    if root is None:
        return {}
    root = Path(root)
    index: dict[str, dict[str, Any]] = {}
    if not root.exists():
        return index
    for manifest in root.rglob(MANIFEST_NAME):
        try:
            payload = json.loads(manifest.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        for record in payload.get("records", []):
            directory = record.get("directory")
            if not directory:
                continue
            for file in record.get("files", []):
                full = str(Path(directory) / file.get("path", ""))
                entry = {
                    "reference": record.get("reference"),
                    "scheme": record.get("scheme"),
                    "revision": record.get("revision"),
                    "retrieved_at": record.get("retrieved_at"),
                    "sha256": file.get("sha256"),
                    "bytes": file.get("bytes"),
                    "source_url": file.get("source_url"),
                }
                # A derived table is a sample of the file it came from; the plan
                # carries how much of it was kept so a report can say so.
                if file.get("derived_from"):
                    entry["derived_from"] = file.get("derived_from")
                if file.get("conversion"):
                    entry["conversion"] = file.get("conversion")
                index[full] = entry
    return index


def _entry(profile: TableProfile, decision, provenance: dict[str, Any]) -> PlanEntry:
    return PlanEntry(
        path=profile.path,
        role=decision.role,
        reason=decision.reason,
        n_features=len(decision.features),
        features_sample=decision.features[:8],
        features=list(decision.features),
        dropped_features=list(getattr(decision, "dropped_features", []) or []),
        rows=profile.n_rows if profile.n_rows is not None else profile.sampled_rows,
        columns=profile.n_columns or len(profile.columns),
        provenance=provenance,
    )


def build_plan(
    paths: Iterable[str | Path],
    task: PlanTask,
    *,
    root: str | Path | None = None,
    evidence: dict[str, Any] | None = None,
    count_rows: bool = False,
    primary_labeled: str | None = None,
    reviews: dict[str, dict[str, Any]] | None = None,
) -> DataPlan:
    """Profile every table and sort it into gold / supplementary / unusable.

    Without a target column there is no role to assign -- the rules are "is the
    label there" -- so the plan keeps the profiles and says so, rather than
    guessing a role.

    More than one table can carry the label. Then `task.multi_gold_policy`
    decides what the extras are, and the default is deliberately not "use every
    label available": a fetched cohort that happens to have the label is the
    case where supervision *could* be pooled, but the case this pipeline is
    built for is one trusted labeled source and everything else as evidence.
    The choice is recorded per table so a reader can tell which happened.
    """
    plan = DataPlan(task=task, evidence=evidence or {})
    provenance = _provenance_index(root)
    reviews = reviews or {}
    if not task.target_column:
        plan.notes.append(
            "no target column decided, so no table has a role yet; decide the "
            "target and rebuild the plan"
        )
    for path in paths:
        # The same table twice is one table. Two retrieval rounds naming the
        # same repository is ordinary, and a duplicate used to become a
        # "supplementary" copy of the gold itself, which the run then refused.
        if str(path) in {entry.path for entry in (*plan.gold, *plan.supplementary, *plan.unusable)}:
            continue
        profile = profile_table(path, count_rows=count_rows)
        if not task.target_column:
            plan.notes.append(
                f"profiled {profile.path}: "
                f"{profile.n_columns or len(profile.columns)} columns"
            )
            continue
        decision = classify_role(profile, task.to_shape())
        entry = _entry(profile, decision, provenance.get(str(Path(profile.path)), {}))
        entry.review = reviews.get(str(profile.path)) or {}
        if entry.review:
            decision = _apply_review(entry, decision, task, plan)
        # The review may move a table to a different role than the rules gave
        # it, and `_entry` was built before that. Recording the decision back on
        # the entry keeps the file self-consistent: a table in `unusable` used
        # to still carry the role and the reason it had before the review, so
        # the plan explained something other than what it did.
        entry.role = decision.role
        entry.reason = decision.reason
        getattr(plan, decision.role).append(entry)
    if len(plan.gold) > 1:
        _apply_multi_gold_policy(plan, task, primary_labeled)
    if plan.gold:
        _enforce_supplementary_schema(plan)
        # What the run trains on is the primary gold's own columns. Writing the
        # list into the plan is what lets a table with a few unusable columns
        # still be usable: the trainer takes exactly these, and every other
        # table is compared against the same list, so its own free-text columns
        # are simply not features.
        features = list(plan.gold[0].features)
        if features and task.feature_columns is None:
            task.feature_columns = features
            plan.notes.append(
                f"training features taken from the primary gold "
                f"({len(features)} column(s): {', '.join(features[:8])}"
                + (", ..." if len(features) > 8 else "")
                + ")"
            )
    return plan


def _apply_review(entry: PlanEntry, decision, task: PlanTask, plan: DataPlan) -> Any:
    """Let a review change the reading, and only within mechanical limits.

    The review may take a table *out* of the role the rules gave it, never put it
    into one they cannot support:

      * `gold` when the target column is missing -> the rules keep their role
        (the table is still evidence), the proposal is recorded, and the
        disagreement is what the human gate in the orchestrator is for. Calling
        it unusable here used to *lose* a usable table because a model wanted to
        train on it -- the opposite of the rule this file exists to enforce;
      * `bad_label` / `unlabeled` over a table that mechanically looks labeled
        -> supplementary, with the reason recorded, so its labels are ignored;
      * `unusable` -> unusable, with the review's reason.
    """
    from .profile import RoleDecision

    proposed = str(entry.review.get("role", ""))
    reason = str(entry.review.get("reason") or "")
    if proposed == "unusable":
        return RoleDecision(
            role="unusable",
            reason=f"review: {reason or 'the model judged this table unusable'}",
            features=decision.features,
            has_target=decision.has_target,
        )
    if proposed in {"unlabeled", "bad_label"} and decision.role == "gold":
        label_note = (
            f" (labels not trainable: {entry.review.get('label_problem')})"
            if proposed == "bad_label" and entry.review.get("label_problem")
            else ""
        )
        entry.label_provenance = proposed
        return RoleDecision(
            role="supplementary",
            reason=(
                f"the review says its labels must not be trained on{label_note}; "
                f"its rows enter as unlabeled evidence"
            ),
            features=decision.features,
            has_target=decision.has_target,
        )
    if proposed in {"unlabeled", "bad_label"}:
        entry.label_provenance = proposed
    plan.notes.append(
        f"{Path(entry.path).name}: review proposed {proposed!r}, rules kept "
        f"{decision.role!r} ({reason[:100]})"
    )
    return decision


def _enforce_supplementary_schema(plan: DataPlan) -> None:
    """Evidence is matched to the gold by intersection, not by identity.

    A supplementary table needs some of the gold's *features* -- its own labels
    are ignored, and a spare column (`Unnamed: 32`, or the same label under
    another name) is not a reason to throw its rows away. What it does not need
    is all of them: the correction is trained on the intersection the tables
    share, and both arms are trained on that same intersection so the comparison
    between "gold only" and "gold plus evidence" stays a comparison of the
    evidence rather than of two different models.

    The intersection is widened greedily, widest table first, and a table that
    would narrow it below what a correction needs is dropped with that as its
    reason -- a 12-gene overlap with a 14,000-gene gold is not evidence, it is a
    different measurement of a different thing.
    """
    gold_features = list(plan.gold[0].features)
    gold_digest = str((plan.gold[0].provenance or {}).get("sha256") or "")
    kept: list[PlanEntry] = []
    rejected: list[tuple[PlanEntry, str]] = []
    order = {id(entry): index for index, entry in enumerate(plan.supplementary)}
    for entry in plan.supplementary:
        digest = str((entry.provenance or {}).get("sha256") or "")
        if gold_digest and digest == gold_digest:
            # The same bytes, fetched from somewhere else. A mirror of the gold
            # is not external evidence: its rows are already in the labeled set,
            # so the correction term would cancel against itself and the report
            # would claim a semi-supervised run that did nothing.
            entry.role = "unusable"
            entry.reason = (
                "a mirror of the primary gold (identical sha256), so its rows "
                "are already in the labeled set and it is not external evidence"
            )
            plan.unusable.append(entry)
            continue
        reason = align_to_gold(entry, gold_features)
        if reason is not None:
            rejected.append((entry, reason))
            continue
        kept.append(entry)

    # Widest first: the table that shares the most with the gold sets the width,
    # and a table that would cut into it is the one that loses its place.
    threshold = required_shared(gold_features)
    ordered = sorted(kept, key=lambda entry: (-len(entry.shared_features), entry.path))
    accepted: list[PlanEntry] = []
    intersection: list[str] = []
    for entry in ordered:
        candidate = (
            list(entry.shared_features)
            if not intersection
            else [name for name in intersection if name in set(entry.shared_features)]
        )
        if len(candidate) < threshold:
            rejected.append(
                (
                    entry,
                    f"taking its rows would narrow the model to {len(candidate)} "
                    f"column(s) shared with the other evidence, and {threshold} "
                    f"are needed",
                )
            )
            continue
        accepted.append(entry)
        intersection = candidate

    for entry, reason in rejected:
        entry.role = "unusable"
        entry.reason = (
            f"it cannot be evidence for the same model: {reason} (the gold has "
            f"{len(gold_features)} feature(s))"
        )
        plan.unusable.append(entry)
    # Back to the order the tables were planned in: the widest-first order is a
    # rule for deciding, not the order a reader wants to see them.
    plan.supplementary = sorted(accepted, key=lambda entry: order.get(id(entry), 0))
    if accepted and intersection and len(intersection) < len(gold_features):
        # The gold's own column order, because that is what the plan and the
        # reports already read, and what a person compares against the gold.
        chosen = [name for name in gold_features if name in set(intersection)]
        if plan.task.feature_columns:
            # A caller who named the feature columns keeps that choice, narrowed
            # to what the evidence can actually supply.
            narrowed = [name for name in plan.task.feature_columns if name in set(chosen)]
            if narrowed:
                chosen = narrowed
        plan.task.feature_columns = chosen
        plan.notes.append(
            f"the evidence shares {len(intersection)} of the gold's "
            f"{len(gold_features)} feature(s), so both arms train on those "
            f"({', '.join(intersection[:8])}"
            + (", ..." if len(intersection) > 8 else "")
            + "); the comparison is like-for-like, not the full gold schema "
            "against a narrower one"
        )


def _apply_multi_gold_policy(
    plan: DataPlan, task: PlanTask, primary_labeled: str | None
) -> None:
    """Resolve several labeled tables into one gold and the rest as evidence."""
    policy = task.multi_gold_policy
    if policy == "pool":
        plan.notes.append(
            f"{len(plan.gold)} labeled tables will be pooled; their labels are all "
            f"used (policy: pool)"
        )
        return
    if policy == "error":
        names = ", ".join(entry.path for entry in plan.gold)
        raise ValueError(
            f"{len(plan.gold)} tables carry the label column ({names}) and the "
            f"policy is 'error'; choose a primary with --labeled-path, or set "
            f"--multi-gold external|pool"
        )

    if primary_labeled:
        chosen = next(
            (e for e in plan.gold if Path(e.path) == Path(primary_labeled)), None
        )
        if chosen is None:
            raise ValueError(
                f"--labeled-path {primary_labeled} is not one of the labeled "
                f"tables: {', '.join(e.path for e in plan.gold)}"
            )
        why = "named as the primary labeled table"
    else:
        # Deterministic and stated: the largest labeled table holds the most
        # supervision, and every other labeled table is still usable as evidence.
        # On equal rows the narrower table wins: the primary's columns become
        # the schema every other table is matched against, and a 129,922-column
        # matrix in that role is a worse fit for a 14,088-column cohort than the
        # other way round.
        chosen = max(
            plan.gold,
            key=lambda entry: (entry.rows or 0, -entry.n_features, entry.path),
        )
        why = (
            f"largest labeled table ({chosen.rows or 0} rows, "
            f"{chosen.n_features} features)"
        )

    demoted = [entry for entry in plan.gold if entry is not chosen]
    plan.gold = [chosen]
    for entry in demoted:
        if entry.path == chosen.path:
            # The same file arrived twice (two rounds named one repository).
            # It is not evidence for itself.
            plan.notes.append(
                f"{entry.path} was listed twice; the duplicate is dropped"
            )
            continue
        # Whether it can actually supply the primary's features is decided once,
        # by `_enforce_supplementary_schema`, after every role is assigned.
        entry.role = "supplementary"
        entry.reason = (
            f"labeled table, demoted to supplementary evidence (the primary gold "
            f"is {chosen.path}, {why}); its labels are ignored and its rows enter "
            f"the semi-supervised correction"
        )
        plan.supplementary.append(entry)
    plan.notes.append(
        f"{len(demoted)} extra labeled table(s) treated as supplementary evidence "
        f"(policy: external); primary gold = {chosen.path} ({why}). "
        f"Use --multi-gold pool to train on all their labels instead."
    )


def load_plan(path: str | Path) -> DataPlan:
    return DataPlan.model_validate(json.loads(Path(path).read_text(encoding="utf-8")))


def write_plan(plan: DataPlan, path: str | Path) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(plan.model_dump(mode="json"), indent=2, sort_keys=False), encoding="utf-8")
    return target


def data_files_under(root: str | Path) -> list[Path]:
    """Every fetched table under a root, one per fetch, latest record wins.

    A derived CSV is preferred over the artifact it came from (GEO writes
    features x samples; the transpose is what a training task can read), so a
    series matrix and its CSV do not both end up as candidates.
    """
    root = Path(root)
    files: list[Path] = []
    for manifest in sorted(root.rglob(MANIFEST_NAME)):
        try:
            payload = json.loads(manifest.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        records = payload.get("records", [])
        if not records:
            continue
        directory = Path(records[-1]["directory"])
        derived = [
            directory / f["path"]
            for f in records[-1].get("files", [])
            if f.get("derived_from") and str(f.get("path", "")).lower().endswith((".csv", ".tsv"))
        ]
        candidates = derived or [
            directory / f["path"]
            for f in records[-1].get("files", [])
            if not f.get("derived_from")
        ]
        files.extend(path for path in candidates if path.exists())
    return files
