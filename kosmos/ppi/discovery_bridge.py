"""A research question turns into a real data task, inside the discovery loop.

The discovery loop could already generate a hypothesis, design an experiment and
write code. What it could not do was *get data*: `data_provider` accepts
`csv/tsv/parquet/json/xlsx` and otherwise **generates synthetic data**
(`Issue #51`), so a biology question came back as an analysis of numbers the
run invented. This module is the seam that changes that, without touching the
pipeline that already works:

  1. **Fetch and plan.** The same `scripts/auto_task_run.py` that `run.py` uses
     -- retrieval, download, mechanical + model review, human adjudication,
     gold/supplementary roles -- invoked in plan mode, so the discovery loop gets
     the identical data contract and the same reports. Every task tries the
     supplementary round: unlabeled rows are evidence, and whether they are used
     is decided later by the objective, not by whether we looked for them.
  2. **Resolve the plan.** `kosmos.cli.commands.run.resolve_data_plan` reads the
     plan into paths, a target column, a task type and the exact feature list.
  3. **Train with the objective the question asks for.** `task_ontology`
     classifies the question: an *inference* question ("is this association
     real?") trains with the signed PPI correction, whose estimate stays
     unbiased; a *prediction* question ("how well can this be predicted?")
     trains with the gold-anchored gradient gate, which ignores synthetic rows
     that disagree with the labeled gradient.
  4. **Report metrics, not a narrative.** The held-out metrics, a bootstrap
     interval and the gold-only baseline are printed, written into
     `metrics.json`, drawn into a figure, and returned as a dict the research
     director stores with the experiment result.

And it is a **single-cell** backend: the pipeline behind it converts `.h5ad`,
selects genes per donor, intersects the panel and corrects with unlabeled cells.
A data question about patients or bulk tissue is declined here (`stage="skipped"`)
rather than approximated with the wrong observation unit -- `state="skipped"`
tells the caller to use its own path, which is the code path for everything this
backend is not built for.

Everything here is additive: if the fetch stage fails (no fetcher, no network,
no data), the caller falls back to what it did before, and the failure is
recorded with the stage it happened in.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from kosmos.domains.single_cell import is_single_cell_question

from .metrics import bootstrap_interval
from .task_ontology import TaskKind, apply_backend_override, classify_task, needs_external_data

ROOT = Path(__file__).resolve().parents[2]
AUTO_TASK_RUN = ROOT / "scripts" / "auto_task_run.py"

#: What a caller may write in `KOSMOS_PERTURBATION_ARMS` -> the arm key(s) it
#: selects. A bare variant (`base`, `signed`) selects it in **both**
#: families -- GEARS and the graph-free MLP -- because the comparison between
#: the two architectures is the point; a fully qualified name selects exactly
#: one arm.
_ARM_VARIANTS = {
    "base": ("gears_base", "mlp_base"),
    "gold": ("gears_base", "mlp_base"),
    "signed": ("gears_augmented_ungated", "mlp_augmented_ungated"),
    "signed_ppi": ("gears_augmented_ungated", "mlp_augmented_ungated"),
    "ungated": ("gears_augmented_ungated", "mlp_augmented_ungated"),
    "gears_base": ("gears_base",),
    "gears_augmented_ungated": ("gears_augmented_ungated",),
    "mlp_base": ("mlp_base",),
    "mlp_augmented_ungated": ("mlp_augmented_ungated",),
    "mlp_signed": ("mlp_augmented_ungated",),
    "gated": ("gears_gated", "mlp_gated"),
    "gears_gated": ("gears_gated",),
    "mlp_gated": ("mlp_gated",),
    # the 8-arm comparison: base / signed PPI / gated x {GEARS, MLP} + PerturbNet x 2
    "all": (
        "gears_base", "mlp_base", "gears_augmented_ungated", "mlp_augmented_ungated",
        "gears_gated", "mlp_gated", "perturbnet_base", "perturbnet_supp",
    ),
    "finetune": ("gears_finetune_ppi",),
    "finetune_ppi": ("gears_finetune_ppi",),
    "gears_finetune_ppi": ("gears_finetune_ppi",),
    # PerturbNet runs in its own environment and is never in the default set
    "perturbnet": ("perturbnet_base", "perturbnet_supp"),
    "perturbnet_base": ("perturbnet_base",),
    "perturbnet_supp": ("perturbnet_supp",),
}


def _arm_key(arm: str) -> str:
    """The metrics.json prefix of an arm: its `summary.md` name, snake_cased."""
    from .perturbation.figures import _arm_label

    return _arm_label(arm).replace(" ", "_")


def _study_counts(study: dict[str, Any], supplementary_paths: Sequence[Any]) -> dict[str, Any] | None:
    """The raw count tables a stage-2 study block names, resolved for stage 3.

    The supplementary count table is the one written beside the first
    supplementary table (the source stage 3 uses), matched by resolved path.
    """
    block = study.get("counts") or {}
    if not block.get("gold"):
        return None
    by_table = {
        Path(str(table)).resolve(): Path(str(path))
        for table, path in (block.get("supplementary") or {}).items()
    }
    first = Path(str(supplementary_paths[0])).resolve() if supplementary_paths else None
    return {
        "gold": Path(str(block["gold"])),
        "supplementary": by_table.get(first) if first is not None else None,
        "total_column": block.get("total_column", "total_counts"),
        "target_sum": float(study.get("target_sum", 1e4)),
    }


def _perturbation_arms_from_env() -> tuple[str, ...] | None:
    """The arms named by `KOSMOS_PERTURBATION_ARMS`, or None for the default.

    `KOSMOS_PERTURBATION_ARMS=base,signed` trains only the gold-only arm and the
    gold + supp signed PPI arm -- the two rows the comparison needs.
    """
    raw = str(os.getenv("KOSMOS_PERTURBATION_ARMS", "")).strip()
    if not raw:
        return None
    names = [part.strip().lower().replace("-", "_") for part in raw.replace(",", " ").split()]
    arms: list[str] = []
    for name in names:
        arms.extend(_ARM_VARIANTS.get(name, ()))
    return tuple(dict.fromkeys(arms)) or None


@dataclass
class DataTaskOutcome:
    """What happened, stage by stage, so a caller can fall back deliberately."""

    ok: bool
    stage: str
    kind: TaskKind | None = None
    plan_path: Path | None = None
    run_dir: Path | None = None
    summary_path: Path | None = None
    figures: list[Path] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)
    error: str = ""

    def as_return_value(self) -> dict[str, Any]:
        """The dict the research director stores as the experiment's result.

        The keys the director already knows (`accuracy`, `balanced_accuracy`,
        `effect_size`, `p_value`) are filled in, so a training task stops being
        an experiment whose result is empty in the database.
        """
        payload = {
            "stage": self.stage,
            "ok": self.ok,
            "error": self.error,
            **(self.kind.to_dict() if self.kind else {}),
            **self.metrics,
            # `data_source` is the director's marker for "a data task produced
            # this"; where the *data* came from (registry / plan / tables) is
            # `staging`, so a backend's own metric cannot shadow the marker.
            "data_source": "data_task",
            "staging": self.metrics.get("data_source"),
            "plan_path": str(self.plan_path) if self.plan_path else None,
            "artifact_dir": str(self.run_dir) if self.run_dir else None,
            "summary_path": str(self.summary_path) if self.summary_path else None,
            "figures": [str(path) for path in self.figures],
        }
        # The two keys the director stores next to `statistical_tests`. The
        # effect of supplementary evidence is the metric's change; a training
        # experiment used to leave both empty, so its result reported nothing.
        payload.setdefault("effect_size", self.metrics.get("delta"))
        payload.setdefault("p_value", None)
        return payload


def fetch_and_plan(
    *,
    question: str,
    out_dir: str | Path,
    domain: str = "biology",
    intent: str = "",
    retrieval_ask: str = "",
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
    max_download_mb: float | None = None,
    max_bytes: int | None = None,
    min_shared_fraction: float | None = None,
    tooluniverse_python: str | None = None,
    no_llm: bool = False,
    adjudicate: str | None = None,
    task_kind: str = "per_cell",
    echo: bool = True,
) -> tuple[Path | None, str]:
    """Run the fetcher in plan mode for this question. Returns `(plan, output)`.

    `plan` is None when the fetcher could not produce one -- the caller then
    knows this question has no grounded data behind it and must say so rather
    than invent some.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    plan_path = out / "plan.json"
    command = [
        sys.executable,
        str(AUTO_TASK_RUN),
        "--objective",
        question,
        "--domain",
        domain,
        "--task-kind",
        str(task_kind),
        "--fetch-limit",
        str(fetch_limit),
        "--supp-limit",
        str(supp_limit),
        "--supp-size",
        str(supp_size),
        "--plan-out",
        str(plan_path),
        "--yes",
    ]
    if intent:
        command += ["--intent", intent]
    if retrieval_ask:
        # Wording for the retrieval prompt that does not turn the search on --
        # so the same question is asked the same thing whether or not the caller
        # named the tables to judge.
        command += ["--retrieval-ask", retrieval_ask]
    # Breadth knobs. `--fetch-limit`/`--supp-limit` say *how many* datasets; these
    # say how hard to look and how much of each dataset to keep: more search
    # hops and retries give retrieval more chances to name a screen, and the
    # wider file/archive caps are what reach the counts matrix deep inside a
    # GEO RAW.tar.
    if search_hops is not None:
        command += ["--search-hops", str(search_hops)]
    if fetch_retries is not None:
        command += ["--fetch-retries", str(fetch_retries)]
    if max_files_per_fetch is not None:
        command += ["--max-files-per-fetch", str(max_files_per_fetch)]
    if archive_members is not None:
        command += ["--archive-members", str(archive_members)]
    if mechanical_search:
        command += ["--mechanical-search"]
    if refresh:
        # Reuse is the default (matching bytes are not fetched twice); refresh
        # turns that off so a re-run actually re-downloads.
        command += ["--refresh"]
    for candidate in candidates:
        # A table the caller already has (or a screen a person named) skips
        # retrieval entirely: `auto_task_run` treats `--candidate` as "these are
        # the files to judge", which is the human-in-the-loop rescue path.
        command += ["--candidate", str(candidate)]
    for hint in hints:
        command += ["--hint", str(hint)]
    for column in exclude_columns:
        command += ["--exclude-col", str(column)]
    if max_download_mb is not None:
        command += ["--max-download-mb", str(max_download_mb)]
    if max_bytes is not None:
        command += ["--max-bytes", str(max_bytes)]
    if min_shared_fraction is not None:
        command += ["--min-shared-fraction", str(min_shared_fraction)]
    if no_llm:
        command += ["--no-llm"]
    if adjudicate:
        command += ["--adjudicate", str(adjudicate)]

    environment = dict(os.environ)
    if tooluniverse_python:
        environment["KOSMOS_TOOLUNIVERSE_PYTHON"] = str(tooluniverse_python)

    collected: list[str] = []
    process = subprocess.Popen(
        command,
        cwd=str(ROOT),
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    assert process.stdout is not None
    for line in process.stdout:
        collected.append(line.rstrip("\n"))
        if echo:
            # The discovery loop used to swallow this: the question went in, a
            # report came out, and nothing in between was visible.
            print(f"# fetch | {line.rstrip()}", flush=True)
    code = process.wait()
    output = "\n".join(collected)
    if code != 0 or not plan_path.exists():
        return None, output
    return plan_path, output


def run_data_task(
    *,
    question: str,
    out_dir: str | Path,
    domain: str = "biology",
    intent: str = "",
    hints: Sequence[str] = (),
    exclude_columns: Sequence[str] = (),
    client: Any = None,
    extra_text: str = "",
    fetch: bool = True,
    plan_path: str | Path | None = None,
    require_data_question: bool = True,
    require_single_cell: bool = True,
    backend: str | None = None,
    max_epochs: int = 20,
    patience: int = 5,
    ppi_lambda: float | None = None,
    gate_kappa: float = 1.0,
    gate_scope: str = "batch",
    single_cell: str = "auto",
    loss_mode: str | None = None,
    max_bytes: int | None = None,
    seed: int = 42,
    echo: bool = True,
    # perturbation backend: own tables, or the screen registry when none given
    gold_table: str | Path | None = None,
    supplementary_tables: Sequence[str | Path] = (),
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
    **fetch_kwargs: Any,
) -> DataTaskOutcome:
    """Classify, fetch, plan, train, and report -- for one research question.

    `plan_path` skips the fetch stage, which is how a caller reuses data that is
    already staged (and how the tests run without a network). `backend` forces a
    single-cell backend by name (`per_cell` or `perturbation`); left unset, the
    question's wording decides.

    Both backends leave the same contract behind: a `run/` directory with
    `summary.md`, `metrics.json`, the objective's figures, and a
    `data_report.md` beside it describing where the data came from. The caller
    reads a `DataTaskOutcome` either way, so the director does not branch on the
    backend.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    if ppi_lambda is None:
        ppi_lambda = float(os.getenv("PPI_LAMBDA", "0.5") or 0.5)
    kind = apply_backend_override(
        classify_task(question, extra_text=extra_text, client=client), backend
    )
    if echo:
        print(
            f"# task: {kind.kind} -> loss {kind.loss_mode} "
            f"({kind.decided_by}: {kind.rationale})",
            flush=True,
        )

    if kind.backend == "perturbation":
        plan_from_retrieval = False
        # A perturbation screen is single-cell by construction, so the modality
        # gate does not apply -- but the *data* is found the same way the
        # per-cell task finds its table: the fetcher searches for a screen that
        # matches the question. The registry is a fallback, not the default, so a
        # "knock out gene X" question is not silently answered on an
        # over-expression screen.
        if plan_path is None and gold_table is None and fetch:
            from kosmos.domains.single_cell.modality import (
                infer_requested_modality,
                retrieval_requirement,
            )

            from .perturbation.run import CONDITION_HINTS, supplementary_requirement_text

            asked = infer_requested_modality(question)
            # The auxiliary graph only needs control cells; the labeled screen
            # has to be the assay the question names.
            # The profile's ask steers the retrieval prompt but must not be what
            # turns the search on: `intent` (the caller's) does that. Passing it
            # as `retrieval_ask` keeps one retrieval question across every path.
            fetch_ask = "\n".join(
                part
                for part in (
                    supplementary_requirement_text(),
                    retrieval_requirement(asked["modality"]),
                )
                if part
            )
            if echo:
                print(
                    f"# retrieval ask (perturbation): {asked['modality']} screen + "
                    f"control cells for the auxiliary graph",
                    flush=True,
                )
            # A perturbation screen's "label" is its condition column, and the
            # gating is written for the per-cell task, which infers a label from
            # names in the question. Naming the columns a screen uses lets a
            # genuine screen (`condition` / `guide` / `sgRNA` / ...) be gated as
            # gold instead of being refused for "no label column".
            fetch_hints = tuple(hints) + tuple(CONDITION_HINTS)
            # Screens ship as multi-gigabyte archives (a GSE RAW.tar was refused
            # at the 2 GiB per-file default, which is how a run that had found
            # the right dataset ended up with nothing).
            fetch_bytes = max_bytes if max_bytes is not None else int(
                os.getenv("KOSMOS_DATAFETCHER_MAX_BYTES", str(8 * 1024 ** 3))
            )
            try:
                resolved_plan, output = fetch_and_plan(
                    question=question,
                    out_dir=out,
                    domain=domain,
                    intent=intent,
                    retrieval_ask=fetch_ask,
                    hints=fetch_hints,
                    task_kind="perturbation",
                    exclude_columns=exclude_columns,
                    max_bytes=fetch_bytes,
                    echo=echo,
                    **fetch_kwargs,
                )
            except Exception as e:  # noqa: BLE001 - a failed fetch falls back
                resolved_plan, output = None, f"{type(e).__name__}: {e}"
            if resolved_plan is not None:
                plan_path = resolved_plan
                plan_from_retrieval = True
            elif echo:
                print(
                    "# perturbation: the fetcher named no usable screen "
                    "(see the fetch log above); falling back to the registered "
                    "screens",
                    flush=True,
                )
        try:
            return run_perturbation_outcome(
                question=question,
                plan_path=plan_path,
                gold_table=gold_table,
                supplementary_tables=supplementary_tables,
                out_dir=out,
                kind=kind,
                seed=seed,
                echo=echo,
                max_epochs=max_epochs,
                patience=patience,
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
                ppi_lambda=ppi_lambda,
            )
        except Exception as e:  # noqa: BLE001 - a failed backend is a finding
            message = f"{type(e).__name__}: {e}"
            # The gating is written for the per-cell task: it judges a table by
            # whether it has a *label* column, so a metadata file can be gated as
            # gold and a real screen refused. A retrieved table that carries no
            # perturbation column is therefore not a screen -- say so and let the
            # registered screens answer, rather than training on the wrong table.
            if plan_from_retrieval and gold_table is None and "no perturbation column" in message:
                if echo:
                    print(
                        f"# perturbation: the retrieved gold is not a screen ({message}); "
                        f"falling back to the registered screens",
                        flush=True,
                    )
                try:
                    return run_perturbation_outcome(
                        question=question,
                        plan_path=None,
                        gold_table=None,
                        supplementary_tables=supplementary_tables,
                        out_dir=out,
                        kind=kind,
                        seed=seed,
                        echo=echo,
                        max_epochs=max_epochs,
                        patience=patience,
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
                        ppi_lambda=ppi_lambda,
                    )
                except Exception as fallback_error:  # noqa: BLE001
                    message = f"{type(fallback_error).__name__}: {fallback_error}"
            return DataTaskOutcome(
                ok=False,
                stage="perturbation",
                kind=kind,
                plan_path=Path(plan_path) if plan_path else None,
                error=message,
            )

    # Before spending a retrieval round: is this a question data can answer?
    # A simulation or a derivation belongs on the code path, and sending it to a
    # repository burns an LLM call and a fetch to find nothing.
    needs_data, why_data = needs_external_data(
        question, extra_text=extra_text, domain=domain, client=client
    )
    if require_data_question and not needs_data:
        if echo:
            print(f"# task: not a data question ({why_data}) -> code path", flush=True)
        return DataTaskOutcome(
            ok=False, stage="skipped", kind=kind, error=why_data
        )
    # And the second judgement: this backend is a *single-cell* pipeline (h5ad
    # ingestion, per-source HVG selection, gene-panel intersection, the PPI
    # correction). A question about patients or bulk tissue is a data question
    # that this backend would answer with the wrong features and the wrong
    # split, so it is declined rather than approximated.
    modality = is_single_cell_question(
        question, extra_text=extra_text, domain=domain, client=client
    )
    if echo:
        print(
            f"# task: modality {modality.to_dict()['modality']} "
            f"({modality.decided_by}: {modality.rationale})",
            flush=True,
        )
    if require_single_cell and not modality.single_cell:
        return DataTaskOutcome(
            ok=False, stage="skipped", kind=kind, error=modality.rationale
        )

    if plan_path is None:
        if not fetch:
            return DataTaskOutcome(
                ok=False,
                stage="fetch",
                kind=kind,
                error="fetching is disabled and no plan was given",
            )
        try:
            if max_bytes is not None:
                fetch_kwargs.setdefault("max_bytes", max_bytes)
            resolved_plan, output = fetch_and_plan(
                question=question,
                out_dir=out,
                domain=domain,
                intent=intent,
                hints=hints,
                exclude_columns=exclude_columns,
                echo=echo,
                **fetch_kwargs,
            )
        except Exception as e:  # noqa: BLE001 - a failed fetch is a fallback
            return DataTaskOutcome(
                ok=False, stage="fetch", kind=kind, error=f"{type(e).__name__}: {e}"
            )
        if resolved_plan is None:
            tail = "\n".join(output.splitlines()[-12:])
            return DataTaskOutcome(
                ok=False,
                stage="fetch",
                kind=kind,
                error=f"the fetcher produced no plan; its output ended with:\n{tail}",
            )
    else:
        resolved_plan = Path(plan_path)

    try:
        summary = train_from_plan(
            plan_path=resolved_plan,
            run_dir=out / "run",
            kind=kind,
            question=question,
            max_epochs=max_epochs,
            patience=patience,
            ppi_lambda=ppi_lambda,
            gate_kappa=gate_kappa,
            gate_scope=gate_scope,
            single_cell=single_cell,
            loss_mode=loss_mode,
            seed=seed,
            echo=echo,
        )
    except Exception as e:  # noqa: BLE001 - training failure is a finding
        return DataTaskOutcome(
            ok=False,
            stage="train",
            kind=kind,
            plan_path=resolved_plan,
            error=f"{type(e).__name__}: {e}",
        )

    metrics, figures = summarise_run(summary, out / "run", kind, echo=echo)
    metrics.update(modality.to_dict())
    # The same keys the perturbation block writes, so a reader of either
    # metrics.json sees which backend ran and where its data came from.
    metrics.setdefault("backend", kind.backend)
    metrics.setdefault("data_source", "fetch" if plan_path is None else "plan")
    (out / "run" / "metrics.json").write_text(
        json.dumps(metrics, indent=2, default=str), encoding="utf-8"
    )
    # The fetcher writes `data_report.md` when it ran. A plan given to us
    # directly (a test, or a caller reusing staged tables) did not go through
    # the fetcher, so the same report is written here: both backends then leave
    # a data report beside the run.
    if not (out / "data_report.md").exists():
        write_plan_data_report(
            out=out,
            question=question,
            plan_path=resolved_plan,
            kind=kind,
            metrics=metrics,
        )
    return DataTaskOutcome(
        ok=True,
        stage="done",
        kind=kind,
        plan_path=resolved_plan,
        run_dir=out / "run",
        summary_path=out / "run" / "summary.md",
        figures=figures,
        metrics=metrics,
    )


def train_from_plan(
    *,
    plan_path: str | Path,
    run_dir: str | Path,
    kind: TaskKind,
    question: str,
    max_epochs: int,
    patience: int,
    ppi_lambda: float,
    gate_kappa: float,
    gate_scope: str,
    single_cell: str,
    loss_mode: str | None,
    seed: int,
    echo: bool,
) -> dict:
    """Resolve the plan and train it with this kind's objective.

    `loss_mode` overrides the objective the question's classification picked, so
    a caller can pin it (`kosmos run --loss signed`) the way `run.py` always
    could.
    """
    from kosmos.cli.commands.run import resolve_data_plan

    from .flow import next_output_dir, run_training
    from .report import write_metrics_figures
    from .schemas import PPITrainingConfig
    from .task_spec import TaskSpec

    resolved = resolve_data_plan(Path(plan_path))
    if not resolved.target_column:
        raise ValueError(
            f"{plan_path} names no target column, so there is nothing to "
            f"supervise: this question has no labeled data behind it yet"
        )
    task = TaskSpec(
        target_column=str(resolved.target_column),
        feature_columns=(
            tuple(resolved.feature_columns) if resolved.feature_columns else None
        ),
        feature_prefixes=tuple(resolved.feature_prefixes or ()),
        exclude_columns=tuple(resolved.exclude_columns or ()),
        sample_id_column=resolved.sample_id_column,
        task_type=str(resolved.task_type or "classification"),
        seed=seed,
        description=question,
    )
    config = PPITrainingConfig(
        task_type=task.task_type,
        seed=seed,
        max_epochs=max_epochs,
        patience=patience,
        # The objective is the difference between the two kinds of task: an
        # inference question wants the unbiased correction, a prediction
        # question wants the gate that can ignore bad evidence. An explicit
        # `loss_mode` wins over that classification.
        loss_mode=(loss_mode or kind.loss_mode),
        ppi_lambda=ppi_lambda,
        gate_kappa=gate_kappa,
        gate_scope=gate_scope,
        single_cell_preprocess=single_cell,
    )
    output_dir = next_output_dir(run_dir)
    summary = run_training(
        labeled_paths=list(resolved.labeled_paths),
        task=task,
        supplementary_paths=list(resolved.supplementary_paths),
        output_dir=output_dir,
        config=config,
        supplementary_renames=resolved.supplementary_renames,
    )
    if echo:
        print(f"# trained: {output_dir}", flush=True)
    figures = write_metrics_figures(summary, output_dir)
    if figures:
        summary["metric_figures"] = [str(path) for path in figures]
        # The markdown was rendered before the figure existed; render it again so
        # the report links the figure it now has.
        from .report import write_markdown

        write_markdown(summary, output_dir)
    return summary


def summarise_run(
    summary: dict,
    run_dir: Path,
    kind: TaskKind,
    *,
    echo: bool = True,
) -> tuple[dict[str, Any], list[Path]]:
    """The metric block: held-out numbers, an interval, and the gold baseline."""
    import numpy as np

    test = summary.get("final_test_metrics") or {}
    validation = summary.get("validation_metrics") or {}
    model = test.get("ppi") or validation.get("ppi") or {}
    baseline = test.get("baseline") or validation.get("baseline") or {}
    task = summary.get("task") or {}
    regression = str(task.get("task_type")) == "regression"
    metric = "r2" if regression else "balanced_accuracy"

    interval: tuple[float, float] | None = None
    predictions = run_dir / "validation_predictions.npz"
    if not regression and predictions.exists():
        try:
            with np.load(predictions, allow_pickle=False) as data:
                interval = bootstrap_interval(
                    np.asarray(data["y_true"]),
                    np.asarray(data["ppi_probability"]),
                    np.asarray(task.get("classes") or []),
                )
        except Exception:  # noqa: BLE001 - an interval is a bonus, not the result
            interval = None

    metrics: dict[str, Any] = {
        "kind": kind.kind,
        "loss_mode": kind.loss_mode,
        "metric": metric,
        "value": (model or {}).get(metric),
        "baseline": (baseline or {}).get(metric),
        "n_gold": summary.get("n_gold_train"),
        "n_supplementary": summary.get("external_used"),
        "preprocessing": summary.get("preprocessing"),
    }
    if interval:
        metrics["ci_low"], metrics["ci_high"] = interval
    for name, value in (model or {}).items():
        if isinstance(value, (int, float)):
            metrics[str(name)] = float(value)
    # The headline metric under its own name as well: the director's result
    # extraction looks for `balanced_accuracy` / `accuracy` / `macro_f1`, and a
    # result whose only key was `value` would store nothing.
    if metrics.get("value") is not None:
        metrics.setdefault(metric, float(metrics["value"]))
    if metrics["value"] is not None and metrics["baseline"] is not None:
        metrics["delta"] = float(metrics["value"]) - float(metrics["baseline"])
    # The director stores these two keys; a training task used to leave both
    # empty in the database, which is why its result reported nothing.
    metrics.setdefault("effect_size", metrics.get("delta"))
    metrics.setdefault("p_value", None)

    figures = [Path(path) for path in (summary.get("metric_figures") or [])]
    if echo:
        interval_text = (
            f" [{metrics['ci_low']:.4f}, {metrics['ci_high']:.4f}]"
            if "ci_low" in metrics and metrics.get("ci_low") is not None
            else ""
        )
        value = metrics.get("value")
        line = (
            f"# metrics: {metric} "
            + (f"{value:.4f}{interval_text}" if isinstance(value, float) else "n/a")
        )
        if metrics.get("baseline") is not None:
            line += f"  (gold-only {float(metrics['baseline']):.4f}"
            if metrics.get("delta") is not None:
                line += f", delta {float(metrics['delta']):+.4f}"
            line += ")"
        print(line, flush=True)
        for name in ("accuracy", "macro_f1", "r2", "pearson", "mae", "rmse"):
            if isinstance(metrics.get(name), float):
                print(f"#          {name} {metrics[name]:.4f}", flush=True)
        print(f"#   gold {metrics['n_gold']} rows, supplementary {metrics['n_supplementary']}", flush=True)
        for path in figures:
            print(f"#   figure {path}", flush=True)
    return metrics, figures




def write_plan_data_report(
    *,
    out: Path,
    question: str,
    plan_path: str | Path,
    kind: TaskKind,
    metrics: dict[str, Any],
) -> tuple[Path, Path]:
    """`data_report.md`/`.json` for a per-cell run that skipped the fetcher.

    The fetcher's own report is richer (it has the search chain); this is the
    equivalent account of a plan handed in directly: the roles the plan assigned
    and the target the run trained on.
    """
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    try:
        plan = json.loads(Path(plan_path).read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001 - a missing/odd plan still deserves a stub
        plan = {}
    gold = plan.get("gold") or []
    supplementary = plan.get("supplementary") or []
    unusable = plan.get("unusable") or []
    task = plan.get("task") or {}
    record = {
        "question": question,
        "backend": kind.backend,
        "kind": kind.kind,
        "target_column": task.get("target_column"),
        "task_type": task.get("task_type"),
        "gold": [entry.get("path") for entry in gold],
        "supplementary": [entry.get("path") for entry in supplementary],
        "unusable": [entry.get("path") for entry in unusable],
        "metrics": metrics,
    }
    lines = [
        f"# data report: {question or 'per-cell task'}",
        "",
        f"- backend: `{kind.backend}` ({kind.kind})",
        f"- target column: `{record['target_column']}` ({record['task_type']})",
        f"- gold tables: {len(gold)}",
        *[f"  - `{entry.get('path')}`" for entry in gold],
        f"- supplementary tables: {len(supplementary)}",
        *[f"  - `{entry.get('path')}`" for entry in supplementary],
    ]
    if unusable:
        lines.append(f"- unusable tables: {len(unusable)}")
    value = metrics.get("value")
    if value is not None:
        lines += [
            "",
            f"- held-out {metrics.get('metric')}: {float(value):.4f}"
            + (
                f" (gold-only {float(metrics['baseline']):.4f})"
                if metrics.get("baseline") is not None
                else ""
            ),
        ]
    markdown = out / "data_report.md"
    record_path = out / "data_report.json"
    markdown.write_text("\n".join(lines) + "\n", encoding="utf-8")
    record_path.write_text(json.dumps(record, indent=2, default=str), encoding="utf-8")
    return markdown, record_path


def _screen_partners(
    *,
    gold_path: Path,
    used: Sequence[Path],
    plan_tables: Sequence[str],
    gold_genes: set[str] | None = None,
    echo: bool = True,
) -> tuple[list[Any], list[dict[str, Any]]]:
    """Other screens in the plan, joined and ready to be the unlabeled source.

    A perturbation run wants a second measurement of the same panel with cells
    that were not perturbed. A published study usually has one -- the same
    series often holds a second arm, a second donor, or the arrayed version of
    the pooled screen -- and the plan lists it as `unusable`, because on its own
    a label table has no measurements and a counts matrix has no labels. Put
    back together, that pair is exactly the evidence the augmented arms need.

    The pairing is decided by barcodes and reported either way; nothing here
    invents a source that the fetch did not download.
    """
    from .perturbation.assemble import (
        is_measurement_table,
        join_screen,
        measurement_features,
        pick_expression_partner,
    )
    from .perturbation.contract import is_control as is_control_value
    from .perturbation.run import guess_condition_column
    from .tabular import read_table

    skip = {Path(str(path)) for path in [gold_path, *used]}
    # Which tables could be a screen's measurements is decided once, not per
    # candidate: a series' `RAW.tar` holds one count matrix per sample *and* the
    # antibody, hashtag and guide panels measured on the same cells, and every
    # one of them has as many columns as there are cells. Only the transcriptome
    # has genes in rows too.
    measurements = [
        str(path)
        for path in plan_tables
        if Path(str(path)) not in skip and is_measurement_table(path)
    ]
    frames: list[Any] = []
    records: list[dict[str, Any]] = []
    for candidate in plan_tables:
        label_path = Path(str(candidate))
        if label_path in skip:
            continue
        if is_measurement_table(candidate):
            # A transcriptome is not the label half of a screen, and reading a
            # head of one costs a 20,730-column parse per candidate.
            continue
        try:
            head = read_table(label_path, nrows=500)
        except Exception:  # noqa: BLE001 - an unreadable table is not a screen half
            continue
        condition = guess_condition_column(head)
        if condition is None or condition not in head.columns:
            continue
        partner = pick_expression_partner(label_path, measurements)
        if partner is None or Path(partner) in skip or Path(partner) == label_path:
            continue
        try:
            frame = join_screen(
                read_table(label_path), read_table(partner), condition_column=condition
            )
        except Exception as error:  # noqa: BLE001 - a failed join is not a source
            records.append(
                {
                    "label": str(label_path),
                    "measurements": str(partner),
                    "error": f"{type(error).__name__}: {error}",
                }
            )
            continue
        genes = [str(gene) for gene in frame.attrs.get("expression_columns") or []]
        shared = len(set(genes) & gold_genes) if gold_genes else 0
        if gold_genes and shared < 0.3 * max(1, len(genes)):
            # The same cells, a different measurement: CITE-seq's antibody panel
            # and ECCITE's guide counts join on barcodes exactly as well as the
            # transcriptome does, and a graph built on `CD86` shares no gene with
            # the gold's panel. It is not supplementary *evidence for this model*.
            records.append(
                {
                    "label": str(label_path),
                    "measurements": str(partner),
                    "error": (
                        f"only {shared} of its {len(genes)} measured feature(s) are "
                        f"genes the gold measures: the same cells, a different "
                        f"measurement"
                    ),
                }
            )
            continue
        frame.attrs["assembled_from"] = [str(label_path), str(partner)]
        frame.attrs["assembled_from_label"] = str(label_path)
        frames.append(frame)
        records.append(
            {
                "label": str(label_path),
                "measurements": str(partner),
                "condition_column": condition,
                "cells": int(len(frame)),
                "control_cells": int(
                    sum(1 for value in frame[condition].astype(str) if is_control_value(value))
                ),
                "genes": len(genes),
                "shared_with_gold": shared,
                "barcode_overlap": float(frame.attrs.get("barcode_overlap") or 0.0),
            }
        )
        skip.add(Path(partner))
        if echo:
            print(
                f"# perturbation: {label_path.name} + {Path(partner).name} is another "
                f"screen in this panel ({len(frame):,} cells, {len(genes):,} genes, "
                f"{shared:,} shared with the gold); it supplies the unlabeled cells "
                f"for the augmented arms",
                flush=True,
            )
    return frames, records


def run_perturbation_outcome(
    *,
    question: str = "",
    plan_path: str | Path | None = None,
    gold_table: str | Path | None = None,
    supplementary_tables: Sequence[str | Path] = (),
    out_dir: Path,
    kind: TaskKind,
    seed: int,
    echo: bool,
    max_epochs: int = 40,
    patience: int = 8,
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
    #: λ's initial value and the two-stage schedule (see perturbation/train.py).
    #: `None` reads `PPI_LAMBDA`, `PPI_STAGE1_EPOCHS`, `PPI_STAGE2_EPOCHS`,
    #: `PPI_LAMBDA_LR_MULTIPLIER`, `PPI_STAGE2_LR_MULTIPLIER`, and otherwise
    #: the training config's defaults (1, 5, 1, ×10, ÷10).
    ppi_lambda: float | None = None,
    direction_lambda: float | None = None,
    stage1_epochs: int | None = None,
    stage2_epochs: int | None = None,
) -> DataTaskOutcome:
    """Run the perturbation backend and report it the way the column task is.

    Data comes from, in order of precedence: the caller's own tables, a plan
    the caller supplied, or the registered screens. Only the last one is the
    default, and it is why a perturbation question does not need a retrieval
    round: the registry records which screens have paired perturbations (gold)
    and which same-cell-line screens supply control cells (the auxiliary graph).

    The artifacts are the column task's contract as well: `run/summary.md`,
    `run/metrics.json`, `run/figures/`, plus `data_report.md` and
    `data_report.json` beside the run, describing the sources and the contract.
    """
    from kosmos.cli.commands.run import resolve_data_plan

    from .perturbation.contract import DEFAULT_CONTROL_LABELS
    from .perturbation.graphs import GO_REFERENCE
    from .perturbation.modality import (
        check_modality,
        infer_requested_modality,
        modality_of,
    )
    from .perturbation.run import run_perturbation_task
    from .perturbation.train import ARMS as PERTURBATION_ARMS
    from .perturbation.train import PerturbationTrainingConfig

    from .flow import next_output_dir

    staged: dict[str, Any] = {}
    max_panel_genes: int | None = None
    expression_path: Path | None = None
    supplementary_frames: list[Any] = []
    assembled_screens: list[dict[str, Any]] = []
    gold_source = None
    supplementary_source_names: dict[str, str] = {}
    #: training settings a stage-2 `study` block may carry (e.g. batch_size)
    study_training: dict[str, Any] = {}
    #: the stage-2 raw count tables (`study.counts`), for the PerturbNet arms
    count_tables: dict[str, Any] | None = None
    if gold_table:
        gold_path = Path(str(gold_table))
        supplementary_paths = [Path(str(entry)) for entry in supplementary_tables]
        staged = {"how": "tables", "gold": str(gold_path)}
    elif plan_path is not None and Path(plan_path).exists():
        resolved = resolve_data_plan(Path(plan_path))
        if not resolved.labeled_paths:
            raise ValueError(
                "the plan carries no gold table, so there is nothing to perturb"
            )
        gold_path = resolved.labeled_paths[0]
        supplementary_paths = list(resolved.supplementary_paths)
        if condition_column is None:
            # The plan's target *is* the condition column for this task: the
            # review named the column holding the perturbation, and dropping it
            # here is what made a screen whose column is `gene` reach the
            # backend with no condition column at all.
            condition_column = resolved.target_column or None
        # A screen is often published as two files: the label table (barcodes
        # with the perturbation each cell carried) and the measurements. Neither
        # is trainable alone, so the pair is joined before training. The
        # measurements are chosen by *which table shares the label table's
        # barcodes* -- a plan cannot say that, because the two files share
        # nothing but their series accession.
        from kosmos.core.diagnostics import stage

        payload = json.loads(Path(plan_path).read_text(encoding="utf-8"))
        plan_tables = [
            str(entry["path"])
            for role in ("gold", "supplementary", "unusable")
            for entry in (payload.get(role) or [])
            if entry.get("path")
        ]
        from .perturbation.assemble import pick_expression_partner

        # A plan written by the prepare stage carries a `study` block: the split
        # and the panel are decided there, and its tables are already the
        # training tables (no label/measurement join, no partner screens).
        study = payload.get("study") or {}
        if study:
            split_mode = str(study.get("split_mode", split_mode))
            test_fraction = float(study.get("test_fraction", test_fraction))
            validation_fraction = float(study.get("validation_fraction", validation_fraction))
            if study.get("max_panel_genes") is not None:
                max_panel_genes = int(study["max_panel_genes"])
            condition_column = condition_column or study.get("condition_column")
            if study.get("batch_size"):
                study_training["batch_size"] = int(study["batch_size"])
            count_tables = _study_counts(study, supplementary_paths)
        expression_path = None if study else pick_expression_partner(gold_path, plan_tables)
        if expression_path is not None:
            supplementary_paths = [
                path for path in supplementary_paths if Path(path) != Path(expression_path)
            ]
            if echo:
                print(
                    f"# perturbation: the gold carries the labels and not the "
                    f"measurements; joining it with {Path(str(expression_path)).name}",
                    flush=True,
                )
        gold_genes: set[str] = set()
        if expression_path is not None:
            # What the gold's own measurements offer, so a second screen can be
            # judged on the genes it shares rather than on its file name.
            from .perturbation.assemble import measurement_features

            gold_genes = measurement_features(expression_path)
        if not study:
            with stage(f"looking for another screen among the plan's {len(plan_tables)} table(s)"):
                supplementary_frames, assembled_screens = _screen_partners(
                    gold_path=Path(str(gold_path)),
                    used=[Path(str(path)) for path in (expression_path,) if path is not None],
                    plan_tables=plan_tables,
                    gold_genes=gold_genes,
                    echo=echo,
                )
        staged = {
            "how": "plan",
            "plan": str(plan_path),
            "assembled_screens": assembled_screens,
            **({"study": study} if study else {}),
        }
    else:
        from .perturbation.sources import default_pair
        from .perturbation.stage import stage_sources

        gold_source, supplementary_sources = default_pair()
        for source in supplementary_sources[:1]:
            supplementary_source_names[source.name] = modality_of(source.perturbation)
        gold_path, supplementary_paths, report = stage_sources(
            gold=gold_source, supplementary=supplementary_sources[:1], echo=echo
        )
        staged = {"how": "registry", "sources": report}
    if echo:
        print(
            f"# perturbation backend: gold {Path(str(gold_path)).name}, "
            f"{len(supplementary_paths)} supplementary + "
            f"{len(supplementary_frames)} assembled screen(s) "
            f"({staged.get('how')})",
            flush=True,
        )

    # Which kind of perturbation is this, and is it what the question asked?
    # The registry declares a screen's assay ("CRISPRa (sgRNA)"); the question
    # names one in words; nothing compared them before this.
    requested = infer_requested_modality(question)
    provided = modality_of(gold_source.perturbation) if gold_source is not None else "unknown"
    verdict = check_modality(requested["modality"], provided)
    modality = {
        "requested": requested,
        "provided": {
            "gold": provided,
            "gold_source": getattr(gold_source, "name", staged.get("how")),
            "gold_description": getattr(gold_source, "perturbation", ""),
            "supplementary": supplementary_source_names,
        },
        "verdict": verdict,
    }
    policy = str(os.getenv("KOSMOS_PERTURBATION_MODALITY", "warn")).strip().lower()
    if policy == "off":
        if echo:
            print("# modality: check disabled (KOSMOS_PERTURBATION_MODALITY=off)", flush=True)
    elif verdict["status"] == "mismatch":
        message = f"# WARNING: perturbation type mismatch - {verdict['message']}"
        if policy in ("strict", "error", "raise"):
            raise ValueError(verdict["message"])
        if echo:
            print(message, flush=True)
    elif echo:
        print(f"# modality: {verdict['status']} - {verdict['message']}", flush=True)

    run_dir = next_output_dir(Path(out_dir) / "run")
    # Which arms to train: `KOSMOS_PERTURBATION_ARMS` if set (`all` = the 8-arm
    # comparison), else base / signed / gated for GEARS and MLP
    # (`PPI_MLP_BASELINES=0` drops the MLP arms).
    arms = _perturbation_arms_from_env()
    if arms is None:
        include_mlp = str(os.getenv("PPI_MLP_BASELINES", "1")).strip().lower() not in (
            "0", "false", "no", "off",
        )
        arms = tuple(
            arm for arm in PERTURBATION_ARMS if include_mlp or not arm.startswith("mlp_")
        )

    def _env(name: str, cast, given):
        if given is not None:
            return cast(given)
        raw = os.getenv(name, "").strip()
        return cast(raw) if raw else None

    overrides = {
        # the learnable λ's initial value (perturbation/train.py: 1 in [0, 2])
        "lambda_init": _env("PPI_LAMBDA", float, ppi_lambda),
        "direction_lambda": _env("PPI_DIRECTION_LAMBDA", float, direction_lambda),
        "stage1_epochs": _env("PPI_STAGE1_EPOCHS", int, stage1_epochs),
        "stage2_epochs": _env("PPI_STAGE2_EPOCHS", int, stage2_epochs),
        "lambda_lr_multiplier": _env("PPI_LAMBDA_LR_MULTIPLIER", float, None),
        "stage2_lr_multiplier": _env("PPI_STAGE2_LR_MULTIPLIER", float, None),
        # 0 = the original GEARS (Δ ignores the control cell), 1 = cell-conditioned
        "condition_on_control": _env(
            "KOSMOS_PERTURBATION_CONDITION_ON_CONTROL",
            lambda raw: str(raw).strip().lower() not in ("0", "false", "no", "off"),
            None,
        ),
        "selection_metric": _env("KOSMOS_PERTURBATION_SELECTION", str, None),
        # env first, then the plan's study block, then the config default (32)
        "batch_size": _env("KOSMOS_PERTURBATION_BATCH_SIZE", int, None)
        or study_training.get("batch_size"),
        "learning_rate": _env("KOSMOS_PERTURBATION_LR", float, None),
        "lr_decay": _env("KOSMOS_PERTURBATION_LR_DECAY", float, None),
        "gated_kappa": _env("PPI_GATED_KAPPA", float, None),
        "finetune_epochs": _env("PPI_FINETUNE_EPOCHS", int, None),
        "finetune_lr_multiplier": _env("PPI_FINETUNE_LR_MULTIPLIER", float, None),
    }
    config_kwargs: dict[str, Any] = {
        "arms": arms,
        "epochs": int(max_epochs),
        "patience": int(patience),
        "seed": int(seed),
        "eta": float(eta),
        **{key: value for key, value in overrides.items() if value is not None},
    }
    config = PerturbationTrainingConfig(**config_kwargs)
    results = run_perturbation_task(
        gold_path=gold_path,
        supplementary_paths=supplementary_paths,
        out_dir=run_dir,
        condition_column=condition_column,
        control_labels=tuple(control_labels) or DEFAULT_CONTROL_LABELS,
        split_mode=split_mode,
        test_fraction=test_fraction,
        validation_fraction=validation_fraction,
        min_cells_per_perturbation=min_cells_per_perturbation,
        go_reference=go_graph or GO_REFERENCE,
        go_k=go_k,
        coexpress_threshold=coexpress_threshold,
        coexpress_k=coexpress_k,
        seed=seed,
        config=config,
        modality=modality,
        expression_path=expression_path,
        supplementary_frames=supplementary_frames,
        max_panel_genes=max_panel_genes,
        counts=count_tables,
    )
    results["staged"] = staged
    arms = results["arms"]
    # The comparison is signed PPI against base within one family: GEARS when
    # it was trained, else the MLP pair (an MLP-only run has no `gears_base`).
    base_arm = next((arm for arm in ("gears_base", "mlp_base") if arm in arms), next(iter(arms)))
    base = arms[base_arm]["test"]["summary"]
    # The headline is the GEARS signed PPI arm against the GEARS base. A caller
    # who trained only `base` gets that, rather than a KeyError.
    headline_arm = next(
        (arm for arm in ("gears_augmented_ungated", "mlp_augmented_ungated") if arm in arms),
        base_arm,
    )
    headline = arms[headline_arm]["test"]
    signed = headline["summary"]
    metrics: dict[str, Any] = {
        "kind": kind.kind,
        "loss_mode": kind.loss_mode,
        "backend": "perturbation",
        "metric": "mse_deg",
        "data_source": staged.get("how", "unknown"),
        "value": float(signed["mse_deg"]),
        "baseline": float(base["mse_deg"]),
        "delta": float(signed["mse_deg"] - base["mse_deg"]),
        "effect_size": float(signed["mse_deg"] - base["mse_deg"]),
        "p_value": None,
        "n_perturbations_test": int(headline["n_perturbations"]),
        "graph_edge_overlap": results.get("graph_edge_overlap", {}),
        "modality_requested": requested["modality"],
        "modality_provided": provided,
        "modality_verdict": verdict["status"],
    }
    # Every trained arm is keyed by the name `summary.md` shows for it
    # (`base`, `signed_ppi`, `mlp_base`, ...), so the two reports agree.
    for arm, payload in arms.items():
        summary = (payload.get("test") or {}).get("summary") or {}
        if not summary:
            continue
        for key in (
            "mse",
            "mse_deg",
            "pearson",
            "spearman",
            "direction_accuracy",
            "top_k_overlap",
        ):
            metrics[f"{_arm_key(arm)}_{key}"] = float(summary[key])
    metrics["headline_arm"] = _arm_key(headline_arm)
    if "mlp_augmented_ungated" in arms and "mlp_base" in arms:
        metrics["mlp_delta_mse_deg"] = (
            metrics["mlp_signed_ppi_mse_deg"] - metrics["mlp_base_mse_deg"]
        )
    if "mlp_base_mse_deg" in metrics:
        metrics["headline_minus_mlp_base_mse_deg"] = (
            float(signed["mse_deg"]) - metrics["mlp_base_mse_deg"]
        )
    figures = [Path(path) for path in (results.get("figures") or [])]
    metrics["artifact_dir"] = str(run_dir)
    metrics["figures"] = [str(path) for path in figures]
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2, default=str), encoding="utf-8"
    )
    write_perturbation_data_report(
        out=Path(out_dir),
        question=question,
        kind=kind,
        staged=staged,
        gold_path=Path(str(gold_path)),
        supplementary_paths=supplementary_paths,
        results=results,
        modality=modality,
    )
    if echo:
        from .perturbation.figures import _arm_label

        trained = " | ".join(
            f"{_arm_label(arm)} {payload['test']['summary']['mse_deg']:.4f}"
            for arm, payload in arms.items()
        )
        print(
            f"# perturbation: mse_deg {trained} ({_arm_label(base_arm)} {float(base['mse_deg']):.4f})",
            flush=True,
        )
        print(
            f"#   {_arm_label(headline_arm)}: direction accuracy {signed['direction_accuracy']:.3f}, "
            f"pearson {signed['pearson']:.3f}, top-K overlap {signed['top_k_overlap']:.3f}",
            flush=True,
        )
        print(f"#   summary {run_dir / 'summary.md'}", flush=True)
        print(f"#   data report {Path(out_dir) / 'data_report.md'}", flush=True)
        for path in figures:
            print(f"#   figure {path}", flush=True)
    return DataTaskOutcome(
        ok=True,
        stage="done",
        kind=kind,
        plan_path=Path(plan_path) if plan_path else None,
        run_dir=run_dir,
        summary_path=run_dir / "summary.md",
        figures=figures,
        metrics=metrics,
    )


def write_perturbation_data_report(
    *,
    out: Path,
    question: str,
    kind: TaskKind,
    staged: dict[str, Any],
    gold_path: Path,
    supplementary_paths: Sequence[str | Path],
    results: dict[str, Any],
    modality: dict[str, Any] | None = None,
) -> tuple[Path, Path]:
    """`data_report.md`/`.json` for the perturbation backend.

    The column task's data report is written by the fetcher, because the fetcher
    is what found the tables. A perturbation run's tables come from the screen
    registry (or the caller), so the equivalent account is written here: which
    screens were used, how each supplementary source was read (control cells,
    an unperturbed atlas, or residualised), the contract's panel and splits, and
    what the auxiliary graph was built from.
    """
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    sources = results.get("sources") or {}
    contract = results.get("task") or {}
    split_sizes = results.get("split_sizes") or {}
    overlap = results.get("graph_edge_overlap") or {}
    record = {
        "question": question,
        "backend": "perturbation",
        "kind": kind.kind,
        "data_source": staged.get("how", "unknown"),
        "staged": staged,
        "gold": str(gold_path),
        "supplementary": [str(path) for path in supplementary_paths],
        "contract": contract,
        "split_sizes": split_sizes,
        "sources": sources,
        "graph_edge_overlap": overlap,
        "modality": modality or {},
    }
    lines = [
        f"# data report: {question or 'perturbation response'}",
        "",
        f"- backend: `perturbation_response`",
        f"- data source: `{record['data_source']}`",
        f"- gold: `{gold_path}`",
    ]
    verdict = (modality or {}).get("verdict") or {}
    if verdict:
        provided = (modality or {}).get("provided") or {}
        lines += [
            f"- requested perturbation type: `{verdict.get('requested')}` (from the question)",
            f"- provided perturbation type: `{verdict.get('provided')}`"
            + (
                f" (screen `{provided.get('gold_source')}`: {provided.get('gold_description')})"
                if provided.get("gold_description")
                else ""
            ),
            f"- type verdict: **{verdict.get('status')}** - {verdict.get('message')}",
        ]
    for path in supplementary_paths:
        lines.append(f"- supplementary: `{path}`")
    lines += [
        "",
        "## contract",
        "",
        f"- measured genes (the shared panel): {contract.get('n_genes', 'n/a')}",
        f"- condition column: `{contract.get('condition_column', 'n/a')}`",
        f"- control labels: {', '.join(contract.get('control_labels') or []) or 'n/a'}",
        f"- perturbations per split: "
        + ", ".join(f"{name} {count}" for name, count in split_sizes.items()),
        "",
        "## sources",
        "",
        "| source | path | cells | used | how | edges |",
        "|---|---|---|---|---|---|",
    ]
    examples = results.get("detail") or {}
    for name, detail in sources.items():
        if not isinstance(detail, dict):
            continue
        used, how = detail.get("used_cells", ""), detail.get("how", detail.get("error", ""))
        if name == "gold" and "n_examples" in examples:
            # the gold's used cells are its examples plus its controls (the
            # controls alone were shown here before: "522 used" of 10,693)
            used = int(examples["n_examples"]) + int(examples.get("control_cells") or 0)
            how = (
                f"{examples['n_examples']:,} perturbed cells ({examples.get('n_perturbations')} perturbations) "
                f"+ {int(examples.get('control_cells') or 0):,} controls"
            )
            if examples.get("dropped_count"):
                how += f"; {examples['dropped_count']} perturbation(s) under {examples.get('min_cells_per_perturbation')} cells dropped"
            if examples.get("outside_vocabulary"):
                how += (
                    f"; {examples['outside_vocabulary_cells']:,} cells of {len(examples['outside_vocabulary'])} "
                    f"perturbation(s) outside the GO vocabulary dropped: {', '.join(examples['outside_vocabulary'])}"
                )
        elif used == "":
            used = detail.get("control_cells", "")
        lines.append(
            "| {name} | {path} | {cells} | {used} | {how} | {edges} |".format(
                name=name,
                path=detail.get("path", ""),
                cells=detail.get("cells", ""),
                used=used,
                how=how,
                edges=detail.get("edges", ""),
            )
        )
    if overlap:
        lines += [
            "",
            "## auxiliary graph",
            "",
            f"- G_C/G_S edge overlap: {overlap.get('shared', 0)} shared "
            f"(jaccard {overlap.get('jaccard', 0):.3f})",
        ]
    if results.get("panel_share_of_gold") is not None:
        lines += [
            "",
            f"- shared panel covers {results.get('panel_share_of_gold'):.0%} of the "
            f"gold's gene columns",
        ]
    markdown = out / "data_report.md"
    record_path = out / "data_report.json"
    markdown.write_text("\n".join(lines) + "\n", encoding="utf-8")
    record_path.write_text(json.dumps(record, indent=2, default=str), encoding="utf-8")
    return markdown, record_path
