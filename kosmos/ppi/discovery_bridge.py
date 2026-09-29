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
    hints: Sequence[str] = (),
    exclude_columns: Sequence[str] = (),
    fetch_limit: int = 2,
    supp_limit: int = 3,
    max_download_mb: float | None = None,
    max_bytes: int | None = None,
    min_shared_fraction: float | None = None,
    tooluniverse_python: str | None = None,
    no_llm: bool = False,
    adjudicate: str | None = None,
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
        "--fetch-limit",
        str(fetch_limit),
        "--supp-limit",
        str(supp_limit),
        "--plan-out",
        str(plan_path),
        "--yes",
    ]
    if intent:
        command += ["--intent", intent]
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
    ppi_lambda: float = 0.5,
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
        # A perturbation screen is single-cell by construction, so the modality
        # gate does not apply -- but the *data* is found the same way the
        # per-cell task finds its table: the fetcher searches for a screen that
        # matches the question. The registry is a fallback, not the default, so a
        # "knock out gene X" question is not silently answered on an
        # over-expression screen.
        if plan_path is None and gold_table is None and fetch:
            from .perturbation.modality import infer_requested_modality, retrieval_requirement
            from .perturbation.run import supplementary_requirement_text

            asked = infer_requested_modality(question)
            # The auxiliary graph only needs control cells; the labeled screen
            # has to be the assay the question names.
            fetch_intent = "\n".join(
                part
                for part in (
                    intent,
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
            try:
                resolved_plan, output = fetch_and_plan(
                    question=question,
                    out_dir=out,
                    domain=domain,
                    intent=fetch_intent,
                    hints=hints,
                    exclude_columns=exclude_columns,
                    echo=echo,
                    **fetch_kwargs,
                )
            except Exception as e:  # noqa: BLE001 - a failed fetch falls back
                resolved_plan, output = None, f"{type(e).__name__}: {e}"
            if resolved_plan is not None:
                plan_path = resolved_plan
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
                gate_kappa=gate_kappa,
            )
        except Exception as e:  # noqa: BLE001 - a failed backend is a finding
            return DataTaskOutcome(
                ok=False,
                stage="perturbation",
                kind=kind,
                plan_path=Path(plan_path) if plan_path else None,
                error=f"{type(e).__name__}: {e}",
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
    gate_kappa: float = 1.0,
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
    from .perturbation.train import PerturbationTrainingConfig

    from .flow import next_output_dir

    staged: dict[str, Any] = {}
    gold_source = None
    supplementary_source_names: dict[str, str] = {}
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
        staged = {"how": "plan", "plan": str(plan_path)}
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
            f"{len(supplementary_paths)} supplementary ({staged.get('how')})",
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
    # The graph-free MLP baselines run in the autoresearch flow (set
    # PPI_MLP_BASELINES=0 to turn them off). Library callers that build their own
    # config are unaffected: the flag defaults to False there.
    include_mlp = str(os.getenv("PPI_MLP_BASELINES", "1")).strip().lower() not in (
        "0", "false", "no", "off",
    )
    config = PerturbationTrainingConfig(
        epochs=int(max_epochs),
        patience=int(patience),
        seed=int(seed),
        eta=float(eta),
        gate_kappa=float(gate_kappa),
        include_mlp_baselines=include_mlp,
    )
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
    )
    results["staged"] = staged
    arms = results["arms"]
    base = arms["gears_base"]["test"]["summary"]
    gated = arms["gears_augmented"]["test"]["summary"]
    metrics: dict[str, Any] = {
        "kind": kind.kind,
        "loss_mode": kind.loss_mode,
        "backend": "perturbation",
        "metric": "mse_deg",
        "data_source": staged.get("how", "unknown"),
        "value": float(gated["mse_deg"]),
        "baseline": float(base["mse_deg"]),
        "delta": float(gated["mse_deg"] - base["mse_deg"]),
        "effect_size": float(gated["mse_deg"] - base["mse_deg"]),
        "p_value": None,
        "n_perturbations_test": int(
            arms["gears_augmented"]["test"]["n_perturbations"]
        ),
        "graph_edge_overlap": results.get("graph_edge_overlap", {}),
        "modality_requested": requested["modality"],
        "modality_provided": provided,
        "modality_verdict": verdict["status"],
    }
    for name, summary in (
        ("base", base),
        ("ungated", arms["gears_augmented_ungated"]["test"]["summary"]),
        ("gated", gated),
        ("mlp_base", (arms.get("mlp_base") or {}).get("test", {}).get("summary") or {}),
        ("mlp_augmented_ungated", (arms.get("mlp_augmented_ungated") or {}).get("test", {}).get("summary") or {}),
        ("mlp_augmented", (arms.get("mlp_augmented") or {}).get("test", {}).get("summary") or {}),
    ):
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
            metrics[f"{name}_{key}"] = float(summary[key])
    if "mlp_base" in arms and "mlp_augmented" in arms:
        metrics["mlp_delta_mse_deg"] = (
            metrics["mlp_augmented_mse_deg"] - metrics["mlp_base_mse_deg"]
        )
        metrics["gated_minus_mlp_base_mse_deg"] = (
            metrics["gated_mse_deg"] - metrics["mlp_base_mse_deg"]
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
        print(
            f"# perturbation: mse_deg base {base['mse_deg']:.4f} | ungated "
            f"{metrics['ungated_mse_deg']:.4f} | gated {gated['mse_deg']:.4f} "
            f"(gold {metrics['base_mse_deg']:.4f})",
            flush=True,
        )
        print(
            f"#   direction accuracy {gated['direction_accuracy']:.3f}, "
            f"pearson {gated['pearson']:.3f}, top-K overlap {gated['top_k_overlap']:.3f}",
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
    for name, detail in sources.items():
        if not isinstance(detail, dict):
            continue
        lines.append(
            "| {name} | {path} | {cells} | {used} | {how} | {edges} |".format(
                name=name,
                path=detail.get("path", ""),
                cells=detail.get("cells", ""),
                used=detail.get("used_cells", detail.get("control_cells", "")),
                how=detail.get("how", detail.get("error", "")),
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
