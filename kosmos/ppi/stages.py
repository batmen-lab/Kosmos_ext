"""Two stages, run and verified on their own: fetch-and-review, then experiment.

`kosmos run` connects the whole thing -- one question in, a fetcher round, a
plan, and a training run out. This module exposes the same seam as two
functions a caller can run separately, which is what a human-in-the-loop
workflow needs:

    stage 1  fetch_stage(...)       question      -> data package + `data.md`
    review   (a person reads `data.md` and edits `plan.json`)
    stage 2  experiment_stage(...)  reviewed plan -> `run/`

Nothing about the data contract changes. Stage 1 writes the same `plan.json`
that `kosmos run --data-plan` already reads, and stage 2 is `run_data_task` with
fetching disabled. The point is that each stage leaves artifacts the other does
not need, so either can be run, inspected and re-run without the other -- and
the two can be re-connected later because the plan is the only handoff.
"""

from __future__ import annotations

import json
import re
import time
from collections.abc import Sequence
from pathlib import Path

from .discovery_bridge import DataTaskOutcome, fetch_and_plan, run_data_task
from .task_ontology import classify_task


def slug(text: str, words: int = 6) -> str:
    """A short, filesystem-safe name for a question."""
    cleaned = re.sub(r"[^a-z0-9\s-]", "", str(text).lower()).split()
    return "-".join(cleaned[:words]) or "data"


def default_data_dir(question: str, root: str | Path = "artifacts/dataprep") -> Path:
    """Where stage 1 writes when the caller names no directory."""
    return Path(root) / f"{slug(question)}-{time.strftime('%Y%m%d-%H%M%S')}"


def fetch_task_kind(backend: str | None, question: str) -> str:
    """Which fetcher profile stage 1 should use: a screen or a labeled table.

    `--task perturbation` pins it; otherwise the question's own wording decides
    (`classify_task` without a model, so stage 1 never spends an LLM call just
    to learn which profile to fetch with)."""
    name = str(backend or "").strip().lower().replace("-", "_")
    if name in {"perturbation", "perturbation_response", "perturb"}:
        return "perturbation"
    if name in {"per_cell", "percell", "simple"}:
        return "per_cell"
    return "perturbation" if classify_task(question).backend == "perturbation" else "per_cell"


def _plan_of(plan: dict, bucket: str) -> list[dict]:
    return [entry for entry in (plan.get(bucket) or []) if entry.get("path")]


def plan_backend(plan_path: str | Path) -> str | None:
    """The backend stage 1 recorded in the plan, if it recorded one.

    The wording of a question is not a reliable router for a *reviewed* plan: a
    person may reword the question, or the plan may be handed on, and the plan
    would then train as the wrong task. Stage 1 stamps the profile it fetched
    with, and stage 2 reads it back."""
    try:
        task = json.loads(Path(plan_path).read_text(encoding="utf-8")).get("task") or {}
    except (OSError, json.JSONDecodeError):
        return None
    backend = str(task.get("backend") or "").strip().lower()
    return backend or None


def stamp_plan_backend(plan_path: str | Path, backend: str) -> None:
    """Record the fetcher profile in `plan.json` so stage 2 does not guess."""
    path = Path(plan_path)
    plan = json.loads(path.read_text(encoding="utf-8"))
    task = plan.setdefault("task", {})
    task.setdefault("backend", backend)
    path.write_text(json.dumps(plan, indent=2, default=str), encoding="utf-8")


def render_review_document(question: str, plan_path: Path, plan: dict) -> str:
    """The human-facing `data.md`: what was fetched, and what to check.

    The full reasoning chain (searches, listings, why each table was refused)
    stays in `data_report.md`; this is the shorter document a person opens
    first to decide whether the plan is the right task before anything trains.
    """
    task = plan.get("task") or {}
    gold = _plan_of(plan, "gold")
    supplementary = _plan_of(plan, "supplementary")
    unusable = _plan_of(plan, "unusable")
    lines = [
        f"# Data package — {question}",
        "",
        "Stage 1 of the **fetch → review → experiment** workflow. "
        "No model has been trained yet.",
        "",
        f"- plan to review: `{plan_path.name}`",
        "- full reasoning: `data_report.md`",
        "",
        "## The task the fetcher read",
        "",
        f"- objective           : {task.get('objective') or question}",
        f"- target / condition  : `{task.get('target_column')}` "
        f"(source: {task.get('target_source')}, confidence: {task.get('target_confidence')})",
        f"- task type           : {task.get('task_type')}",
        f"- feature columns     : {len(task.get('feature_columns') or [])}",
        "",
        f"## Gold — trainable ({len(gold)})",
        "",
    ]
    lines += [
        f"- `{entry['path']}` — {entry.get('reason', '')}" for entry in gold
    ] or ["- (none: nothing to train on — see the rescue steps below)"]
    lines += ["", f"## Supplementary — evidence only, never labels ({len(supplementary)})", ""]
    lines += [
        f"- `{entry['path']}` — {entry.get('reason', '')}" for entry in supplementary
    ] or ["- (none: the run will be plain supervised training)"]
    lines += ["", f"## Unusable — refused ({len(unusable)})", ""]
    shown = unusable[:12]
    lines += [f"- `{entry['path']}` — {entry.get('reason', '')}" for entry in shown]
    if len(unusable) > len(shown):
        lines.append(f"- … and {len(unusable) - len(shown)} more in `data_report.md`")
    if not shown:
        lines.append("- (none)")
    lines += [
        "",
        "## What to review",
        "",
        "1. Is the gold the table this question needs, and are its rows the right "
        "observation unit?",
        f"2. Is `{task.get('target_column')}` the right column to supervise on?",
        "3. For a perturbation task: is that same column the perturbation "
        "identity, and are the control labels right?",
        "4. Is any `unusable` table actually the data, one manual step away?",
        "",
        "Fix the plan by editing `plan.json` in place (the fields above), or by "
        "overriding them on the stage-2 command line. Nothing else in this "
        "directory needs to change.",
        "",
        "## Then run stage 2",
        "",
        "```bash",
        f'.venv/bin/python -m kosmos.cli.main experiment "{question}" '
        f"--data-plan {plan_path} --yes",
        "```",
        "",
        "Stage 2 writes `run/summary.md` and `run/metrics.json` beside this "
        "file; re-running it never overwrites a finished run (it lands in "
        "`run-2`, ...).",
        "",
    ]
    return "\n".join(lines) + "\n"


def write_review_document(
    out_dir: str | Path, question: str, plan_path: str | Path
) -> Path:
    """Write `<out_dir>/data.md` from `plan.json`. Returns its path."""
    out = Path(out_dir)
    plan = json.loads(Path(plan_path).read_text(encoding="utf-8"))
    document = out / "data.md"
    document.write_text(
        render_review_document(question, Path(plan_path), plan), encoding="utf-8"
    )
    return document


def fetch_stage(
    *,
    question: str,
    out_dir: str | Path,
    domain: str = "biology",
    backend: str | None = None,
    intent: str = "",
    candidates: Sequence[str] = (),
    hints: Sequence[str] = (),
    exclude_columns: Sequence[str] = (),
    fetch_limit: int = 2,
    supp_limit: int = 3,
    supp_size: str = "any",
    search_hops: int | None = None,
    max_files_per_fetch: int | None = None,
    archive_members: int | None = None,
    fetch_retries: int | None = None,
    mechanical_search: bool = False,
    refresh: bool = False,
    max_bytes: int | None = None,
    max_download_mb: float | None = None,
    no_llm: bool = False,
    adjudicate: str | None = None,
    echo: bool = True,
) -> DataTaskOutcome:
    """Stage 1: fetch and plan only. No training, ever.

    On success the outcome carries `plan_path` and, in `metrics`, the paths of
    `data.md` and the data directory. The caller reviews those, then runs
    `experiment_stage` on the same directory.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    task_kind = fetch_task_kind(backend, question)
    if echo:
        print(f"# stage 1 (fetch): {task_kind} profile -> {out}", flush=True)
    # A perturbation question gets the same augmentation the connected path
    # uses: name the columns a screen carries (so a review can find the
    # condition column) and ask for control cells for the auxiliary graph. The
    # ask rides on `retrieval_ask`, not `intent`: `--intent` turns the search on,
    # and a caller who named `--candidate` has decided what to judge. Composing
    # this the same way either way is what keeps one review path, not two.
    retrieval_ask = ""
    effective_hints = tuple(hints)
    if task_kind == "perturbation":
        from .perturbation.run import CONDITION_HINTS

        from kosmos.domains.single_cell.modality import (
            infer_requested_modality,
            retrieval_requirement,
        )

        from .perturbation.run import supplementary_requirement_text

        asked = infer_requested_modality(question)
        retrieval_ask = "\n".join(
            part
            for part in (
                supplementary_requirement_text(prefer_large=supp_size == "large"),
                retrieval_requirement(asked["modality"]),
            )
            if part
        )
        effective_hints = tuple(hints) + tuple(CONDITION_HINTS)
    plan_path, output = fetch_and_plan(
        question=question,
        out_dir=out,
        domain=domain,
        intent=intent,
        retrieval_ask=retrieval_ask,
        candidates=candidates,
        hints=effective_hints,
        exclude_columns=exclude_columns,
        fetch_limit=fetch_limit,
        supp_limit=supp_limit,
        supp_size=supp_size,
        search_hops=search_hops,
        max_files_per_fetch=max_files_per_fetch,
        archive_members=archive_members,
        fetch_retries=fetch_retries,
        mechanical_search=mechanical_search,
        refresh=refresh,
        max_bytes=max_bytes,
        max_download_mb=max_download_mb,
        no_llm=no_llm,
        adjudicate=adjudicate,
        task_kind=task_kind,
        echo=echo,
    )
    if plan_path is None:
        tail = "\n".join(output.splitlines()[-12:])
        return DataTaskOutcome(
            ok=False,
            stage="fetch",
            error=f"the fetcher produced no plan; its output ended with:\n{tail}",
            metrics={"data_dir": str(out)},
        )
    plan_path = Path(plan_path)
    # The profile is part of the contract now: a reviewed plan must route to the
    # same backend even if nobody re-types the question's perturbation words.
    stamp_plan_backend(plan_path, task_kind)
    document = write_review_document(out, question, plan_path)
    return DataTaskOutcome(
        ok=True,
        stage="fetched",
        plan_path=plan_path,
        metrics={
            "data_dir": str(out),
            "data_md": str(document),
            "data_report": str(out / "data_report.md"),
        },
    )


def experiment_stage(
    *,
    question: str,
    plan_path: str | Path,
    out_dir: str | Path | None = None,
    backend: str | None = None,
    max_epochs: int = 20,
    patience: int = 5,
    seed: int = 42,
    loss_mode: str | None = None,
    ppi_lambda: float | None = None,
    condition_column: str | None = None,
    control_labels: Sequence[str] = (),
    split_mode: str = "mixed",
    test_fraction: float = 0.2,
    validation_fraction: float = 0.1,
    min_cells_per_perturbation: int = 2,
    go_graph: str | Path | None = None,
    go_k: int = 20,
    coexpress_threshold: float = 0.4,
    coexpress_k: int = 20,
    eta: float = 1.0,
    echo: bool = True,
) -> DataTaskOutcome:
    """Stage 2: train a reviewed plan. No fetching, ever.

    The plan is the only handoff from stage 1, so whatever a person edited into
    it (or overrode here) is exactly what trains. `fetch=False` is what makes
    this stage offline once the data is staged: a plan that cannot be trained
    fails here with the reason, rather than quietly re-running retrieval.
    """
    plan = Path(plan_path)
    if not plan.exists():
        return DataTaskOutcome(
            ok=False,
            stage="experiment",
            error=f"no plan at {plan}; run stage 1 (`kosmos dataprep`) first",
        )
    out = Path(out_dir) if out_dir else plan.parent
    if echo:
        print(f"# stage 2 (experiment): training {plan} -> {out}/run", flush=True)
    if backend is None:
        # A plan stage 1 wrote carries its own backend; fall back to the
        # question's wording only for a plan that predates the stamp.
        backend = plan_backend(plan)
    return run_data_task(
        question=question,
        out_dir=out,
        plan_path=str(plan),
        fetch=False,
        backend=backend,
        # The plan is already reviewed: the question-classification gates that
        # guard stage 1 must not veto a decision a person has already made.
        require_data_question=False,
        require_single_cell=False,
        max_epochs=max_epochs,
        patience=patience,
        seed=seed,
        loss_mode=loss_mode,
        ppi_lambda=ppi_lambda,
        condition_column=condition_column,
        control_labels=tuple(control_labels or ()),
        split_mode=split_mode,
        test_fraction=test_fraction,
        validation_fraction=validation_fraction,
        min_cells_per_perturbation=min_cells_per_perturbation,
        go_graph=go_graph,
        go_k=go_k,
        coexpress_threshold=coexpress_threshold,
        coexpress_k=coexpress_k,
        eta=eta,
        echo=echo,
    )


__all__ = [
    "slug",
    "default_data_dir",
    "fetch_task_kind",
    "plan_backend",
    "stamp_plan_backend",
    "render_review_document",
    "write_review_document",
    "fetch_stage",
    "experiment_stage",
]
