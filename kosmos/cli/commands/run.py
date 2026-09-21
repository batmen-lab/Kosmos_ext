"""
Run command for Kosmos CLI.

Executes autonomous research with live progress visualization.

Async Architecture (Issue #66 fix):
- run_with_progress() is now async
- Uses asyncio.run() at CLI entry point
- ResearchDirector.execute() is now async
"""

import sys
import time
import logging
import asyncio
import json
import os
from dataclasses import dataclass
from typing import List, Optional
from datetime import datetime, timezone
from pathlib import Path

import typer
from rich.live import Live
from rich.table import Table
from rich.panel import Panel
from rich.progress import (
    Progress,
    SpinnerColumn,
    TextColumn,
    BarColumn,
    TaskProgressColumn,
    TimeRemainingColumn,
)
from rich.layout import Layout
from rich.text import Text

from kosmos.cli.utils import (
    console,
    print_success,
    print_error,
    print_info,
    get_icon,
    format_timestamp,
    create_status_text,
)
from kosmos.cli.interactive import run_interactive_mode
from kosmos.cli.views.results_viewer import ResultsViewer
from kosmos.core.stage_tracker import get_stage_tracker

logger = logging.getLogger(__name__)


@dataclass
class ResolvedPlan:
    """A data plan, read into the values a run needs."""

    target_column: Optional[str]
    #: classification | regression -- what kind of label `target_column` is.
    task_type: str
    #: The exact feature columns the run trains on. The plan writes the primary
    #: gold's own list here, which is what lets a table with a few unusable
    #: columns still be used.
    feature_columns: Optional[List[str]]
    #: Per supplementary table: its column name -> the gold's, for columns that
    #: name the same measurement in a different spelling. The plan matched them.
    supplementary_renames: dict
    feature_prefixes: Optional[List[str]]
    exclude_columns: Optional[List[str]]
    sample_id_column: Optional[str]
    labeled_paths: List[str]
    supplementary_paths: List[str]
    summary_lines: List[str]


def resolve_data_plan(
    plan_path: Path,
    *,
    target_column: Optional[str] = None,
    feature_prefixes: Optional[List[str]] = None,
    exclude_columns: Optional[List[str]] = None,
    sample_id_column: Optional[str] = None,
    extra_supplementary: Optional[List[str]] = None,
) -> ResolvedPlan:
    """Read a `datafetcher plan` file. Explicit arguments win over the file.

    Kosmos does not import the fetcher: the plan is a JSON contract, and this
    function is the only place its shape is known. A plan with no gold table is
    refused rather than run -- a run with nothing labeled would quietly become
    "train on pseudo-labels", which is the one thing this design rules out.
    """
    plan = json.loads(Path(plan_path).read_text(encoding="utf-8"))
    plan_task = plan.get("task") or {}
    gold = [entry["path"] for entry in plan.get("gold", []) if entry.get("path")]
    supplementary = [
        entry["path"] for entry in plan.get("supplementary", []) if entry.get("path")
    ]
    if not gold:
        raise ValueError(
            f"{plan_path} has no gold tables, so there is nothing to train on. "
            f"Run `python -m datafetcher roles` on the candidates to see why, "
            f"then rebuild the plan."
        )
    merged_supplementary = list(
        dict.fromkeys(list(extra_supplementary or []) + supplementary)
    )
    prefixes = list(feature_prefixes or plan_task.get("feature_prefixes") or [])
    excludes = list(exclude_columns or plan_task.get("exclude_columns") or [])
    features = list(plan_task.get("feature_columns") or [])
    renames = {
        str(entry["path"]): dict(entry.get("column_renames") or {})
        for entry in plan.get("supplementary", [])
        if entry.get("column_renames")
    }
    resolved_target = target_column or plan_task.get("target_column")
    gold_features = list(((plan.get("gold") or [{}])[0]).get("features") or [])
    feature_line = (
        f"  features      : {len(features)} column(s) "
        f"{features[:8]}{' ...' if len(features) > 8 else ''}"
        if features
        else f"  features      : prefixes={prefixes} exclude={excludes}"
    )
    if features and len(features) < len(gold_features):
        # Both arms train on this narrower list: it is the intersection with the
        # evidence, and saying so is what keeps the two-arm comparison readable.
        feature_line += (
            f" [dim](the intersection with the evidence; the gold has "
            f"{len(gold_features)})[/dim]"
        )
    lines = [
        f"  target column : [cyan]{resolved_target}[/cyan] "
        f"(source: {plan_task.get('target_source')}, "
        f"confidence: {plan_task.get('target_confidence')})",
        feature_line,
        f"  labeled       : {len(gold)} table(s)",
        *[f"      {path}" for path in gold],
        f"  supplementary : {len(merged_supplementary)} table(s)",
        *[f"      {path}" for path in merged_supplementary],
    ]
    if plan.get("unusable"):
        lines.append(f"  unusable      : {len(plan['unusable'])} table(s) excluded")
    return ResolvedPlan(
        target_column=resolved_target,
        task_type=str(plan_task.get("task_type") or "classification"),
        feature_columns=features or None,
        supplementary_renames=renames,
        feature_prefixes=prefixes,
        exclude_columns=excludes,
        sample_id_column=sample_id_column or plan_task.get("sample_id_column"),
        labeled_paths=gold,
        supplementary_paths=merged_supplementary,
        summary_lines=lines,
    )


def run_research(
    question: Optional[str] = typer.Argument(None, help="Research question to investigate"),
    domain: Optional[str] = typer.Option(None, "--domain", "-d", help="Research domain (biology, neuroscience, materials, etc.)"),
    max_iterations: int = typer.Option(10, "--max-iterations", "-i", help="Maximum number of research iterations"),
    budget: Optional[float] = typer.Option(None, "--budget", "-b", help="Budget limit in USD"),
    data_path: Optional[Path] = typer.Option(None, "--data-path", "-D", help="Path to CSV dataset for experiments"),
    external_data_path: Optional[Path] = typer.Option(
        None,
        "--external-data-path",
        help="Optional unlabeled CSV used as PPI supplementary evidence",
    ),
    task_target_column: Optional[str] = typer.Option(
        None,
        "--task-target-column",
        help="Label column to predict. Declaring it turns the run into a training task",
    ),
    task_type: Optional[str] = typer.Option(
        None,
        "--task-type",
        help=(
            "classification or regression. A plan carries its own; this overrides it"
        ),
    ),
    task_feature_prefix: Optional[List[str]] = typer.Option(
        None,
        "--task-feature-prefix",
        help="Keep feature columns starting with this (repeatable, e.g. ENSG)",
    ),
    task_exclude_column: Optional[List[str]] = typer.Option(
        None,
        "--task-exclude-column",
        help="Never a feature: identifiers, metadata, other labels (repeatable)",
    ),
    task_sample_id_column: Optional[str] = typer.Option(
        None,
        "--task-sample-id-column",
        help="Column holding per-row identifiers (default: <table>:<row>)",
    ),
    task_test_path: Optional[Path] = typer.Option(
        None,
        "--task-test-path",
        help="Labeled table to use as the final test set instead of a random slice",
    ),
    data_plan: Optional[Path] = typer.Option(
        None,
        "--data-plan",
        help="Plan JSON from `python -m datafetcher plan`: supplies the task, the "
        "labeled tables and the supplementary tables",
    ),
    assume_yes: bool = typer.Option(
        False,
        "--yes",
        help="Do not ask before running a task read from a plan file",
    ),
    no_cache: bool = typer.Option(False, "--no-cache", help="Disable caching"),
    interactive: bool = typer.Option(False, "--interactive", help="Use interactive mode"),
    output: Optional[Path] = typer.Option(None, "--output", "-o", help="Save results to file (JSON or Markdown)"),
    stream: bool = typer.Option(False, "--stream", "-s", help="Enable real-time event streaming display"),
    stream_tokens: bool = typer.Option(True, "--stream-tokens/--no-stream-tokens", help="Show LLM token streaming (with --stream)"),
):
    """
    Run autonomous research on a scientific question.

    Examples:

        # Interactive mode (recommended for first time)
        kosmos run --interactive

        # Direct command
        kosmos run "What metabolic pathways differ between cancer and normal cells?" --domain biology

        # With budget limit
        kosmos run "How do perovskites optimize efficiency?" --domain materials --budget 50

        # Save results
        kosmos run "Question" --output results.json

        # With real-time streaming (shows LLM tokens and events)
        kosmos run "Question" --stream

        # Streaming without token display (just progress events)
        kosmos run "Question" --stream --no-stream-tokens
    """
    # Defaults for settings only available via interactive mode
    auto_model_selection = True
    parallel_execution = False

    # Use interactive mode if requested or no question provided
    if interactive or not question:
        config = run_interactive_mode()

        if not config:
            console.print("[warning]Research cancelled.[/warning]")
            raise typer.Exit(0)

        # Extract config
        question = config["question"]
        domain = config["domain"]
        max_iterations = config["max_iterations"]
        budget = config.get("budget_usd")
        no_cache = not config.get("enable_cache", True)
        auto_model_selection = config.get("auto_model_selection", True)
        parallel_execution = config.get("parallel_execution", False)

    # Validate inputs
    if not question:
        print_error("No research question provided. Use --interactive or provide a question.")
        raise typer.Exit(1)

    # Validate data_path if provided
    if data_path and not data_path.exists():
        print_error(f"Data file not found: {data_path}")
        raise typer.Exit(1)
    if external_data_path and not external_data_path.exists():
        print_error(f"External data file not found: {external_data_path}")
        raise typer.Exit(1)
    if task_test_path and not task_test_path.exists():
        print_error(f"Test data file not found: {task_test_path}")
        raise typer.Exit(1)
    if data_plan and not data_plan.exists():
        print_error(f"Data plan not found: {data_plan}")
        raise typer.Exit(1)

    # A plan supplies the task and the tables. Explicit flags still win, so a
    # plan can be corrected on the command line without editing the file.
    task_labeled_paths: List[str] = []
    task_supplementary_paths: List[str] = list(
        [str(external_data_path.resolve())] if external_data_path else []
    )
    # `--task-feature-prefix` / `--exclude-column` still apply when there is no
    # plan; a plan's own feature list replaces them.
    plan_feature_columns: List[str] = []
    plan_supplementary_renames: dict = {}
    if data_plan:
        try:
            resolved = resolve_data_plan(
                data_plan,
                target_column=task_target_column,
                feature_prefixes=task_feature_prefix,
                exclude_columns=task_exclude_column,
                sample_id_column=task_sample_id_column,
                extra_supplementary=task_supplementary_paths,
            )
        except ValueError as e:
            print_error(str(e))
            raise typer.Exit(1) from e
        task_target_column = resolved.target_column
        # The plan says what kind of label it found; a caller can still override.
        task_type = task_type or resolved.task_type
        plan_feature_columns = resolved.feature_columns
        plan_supplementary_renames = resolved.supplementary_renames
        task_feature_prefix = resolved.feature_prefixes
        task_exclude_column = resolved.exclude_columns
        task_sample_id_column = resolved.sample_id_column
        task_labeled_paths = resolved.labeled_paths
        task_supplementary_paths = resolved.supplementary_paths
        if not data_path:
            data_path = Path(resolved.labeled_paths[0])

        # The confirmation gate. Nothing here is about data access: it is about
        # the one decision that can be wrong -- which column is the label.
        console.print()
        console.print("[bold]Task from plan[/bold]")
        for line in resolved.summary_lines:
            console.print(line)
        if not assume_yes:
            if not sys.stdin.isatty():
                print_error(
                    "a plan file needs confirmation before training; re-run with "
                    "--yes to accept it non-interactively"
                )
                raise typer.Exit(1)
            if not typer.confirm("Train on this task?", default=True):
                console.print("[warning]Cancelled.[/warning]")
                raise typer.Exit(0)

    # The default has to be applied after the plan is read, or it overwrites the
    # plan's own answer: a regression plan read as classification sends a
    # continuous label through the class-based trainer, which refuses it as
    # "classes absent from gold training" and leaves the run with no artifacts.
    # Without a plan there is nobody to say otherwise, and classification is what
    # every flag-based run has always been.
    task_type = task_type or "classification"

    # Show starting message
    console.print()
    console.print(
        Panel(
            f"[cyan]Starting autonomous research...[/cyan]\n\n"
            f"**Question:** {question}\n"
            f"**Domain:** {domain or 'auto-detect'}\n"
            f"**Max Iterations:** {max_iterations}\n"
            f"**Budget:** ${budget} USD" if budget else "**Budget:** No limit",
            title=f"[bright_blue]{get_icon('rocket')} Kosmos Research[/bright_blue]",
            border_style="bright_blue",
        )
    )
    console.print()

    # Initialize research
    try:
        from kosmos.agents.research_director import ResearchDirectorAgent
        from kosmos.config import get_config

        # Get configuration
        config_obj = get_config()

        # Override with CLI parameters
        if domain:
            config_obj.research.enabled_domains = [domain]
        config_obj.research.max_iterations = max_iterations
        if budget:
            config_obj.research.budget_usd = budget

        # Handle cache setting - claude config may be None for non-Anthropic providers
        cache_enabled = not no_cache
        if config_obj.claude:
            config_obj.claude.enable_cache = cache_enabled

        # Create flattened config dict for agents
        # Agents expect flat keys, not nested KosmosConfig structure
        flat_config = {
            # Research settings
            "max_iterations": config_obj.research.max_iterations,
            "enabled_domains": config_obj.research.enabled_domains,
            "enabled_experiment_types": config_obj.research.enabled_experiment_types,
            "min_novelty_score": config_obj.research.min_novelty_score,
            "enable_autonomous_iteration": config_obj.research.enable_autonomous_iteration,
            "budget_usd": config_obj.research.budget_usd,

            # Performance/concurrent operations settings
            "enable_concurrent_operations": config_obj.performance.enable_concurrent_operations,
            "max_parallel_hypotheses": config_obj.performance.max_parallel_hypotheses,
            "max_concurrent_experiments": config_obj.performance.max_concurrent_experiments,
            "max_concurrent_llm_calls": config_obj.performance.max_concurrent_llm_calls,
            "llm_rate_limit_per_minute": config_obj.performance.llm_rate_limit_per_minute,

            # LLM provider settings
            "llm_provider": config_obj.llm_provider,
            "enable_cache": cache_enabled,

            # Optional behavior switches (kept off by default to preserve
            # framework behavior; set env vars to override for deterministic runs)
            "use_literature_context": os.getenv(
                "USE_LITERATURE_CONTEXT", "true"
            ).lower() not in ("0", "false", "no"),
            "use_experiment_templates": os.getenv(
                "USE_EXPERIMENT_TEMPLATES", "true"
            ).lower() not in ("0", "false", "no"),
            "require_novelty_check": os.getenv(
                "REQUIRE_NOVELTY_CHECK", "true"
            ).lower() not in ("0", "false", "no"),
            "num_hypotheses": int(os.getenv("NUM_HYPOTHESES", "3")),

            # Dataset path
            "data_path": str(data_path.resolve()) if data_path else None,
            # Labeled tables to pool (from a plan); data_path stays the first
            # one so every existing reader keeps working.
            "task_labeled_paths": task_labeled_paths,
            "ppi_supplementary_paths": task_supplementary_paths,
            "ppi_external_data_path": (
                str(external_data_path.resolve()) if external_data_path else None
            ),
            # Training task: the label column plus how to pick features. With
            # this declared, the run trains a predictor (supervised, or
            # PPI-augmented when supplementary data is supplied).
            "task_target_column": task_target_column,
            # classification | regression: the plan decided it (the model read
            # the question and the rules checked the column), unless the caller
            # overrode it on the command line.
            "task_type": task_type,
            "task_feature_prefixes": list(task_feature_prefix or []),
            # The plan's own feature list: the primary gold's columns. Without
            # it the trainer would take every column that is not the target,
            # including the free-text ones the plan decided to leave out.
            "task_feature_columns": list(plan_feature_columns or []),
            "ppi_supplementary_renames": dict(plan_supplementary_renames or {}),
            "task_exclude_columns": list(task_exclude_column or []),
            "task_sample_id_column": task_sample_id_column,
            "ppi_test_path": str(task_test_path.resolve()) if task_test_path else None,
            "ppi_output_dir": os.getenv("PPI_OUTPUT_DIR"),
            "ppi_seed": int(os.getenv("PPI_SEED", "42")),
            "ppi_external_per_donor": int(os.getenv("PPI_EXTERNAL_PER_DONOR", "5000")),
            "ppi_max_external_samples": int(os.getenv("PPI_MAX_EXTERNAL_SAMPLES", "20000")),
            "ppi_external_weight_budget": float(os.getenv("PPI_EXTERNAL_WEIGHT_BUDGET", "0.5")),
            "ppi_max_epochs": int(os.getenv("PPI_MAX_EPOCHS", "20")),
            "ppi_patience": int(os.getenv("PPI_PATIENCE", "5")),
            "ppi_cross_fit_folds": int(os.getenv("PPI_CROSS_FIT_FOLDS", "3")),
            "ppi_model_design": os.getenv("PPI_MODEL_DESIGN", "deepseek"),
            "ppi_stage1_epochs": int(os.getenv("PPI_STAGE1_EPOCHS", "4")),
            "ppi_stage2_epochs": int(os.getenv("PPI_STAGE2_EPOCHS", "1")),
            "ppi_pseudo_mode": os.getenv("PPI_PSEUDO_MODE", "cross_fit"),
            "ppi_pseudo_targets": os.getenv("PPI_PSEUDO_TARGETS", "hard"),
            "ppi_loss_mode": os.getenv("PPI_LOSS_MODE", "signed"),
            "ppi_loss_ramp_epochs": int(os.getenv("PPI_LOSS_RAMP_EPOCHS", "0")),
            "ppi_lambda": float(os.getenv("PPI_LAMBDA", "1")),
            # The gradient gate's own knobs (loss_mode="gradient_gated").
            "ppi_gate_kappa": float(os.getenv("PPI_GATE_KAPPA", "1.0")),
            "ppi_gate_scope": os.getenv("PPI_GATE_SCOPE", "batch"),
            "ppi_gate_gamma": float(os.getenv("PPI_GATE_GAMMA", "1.0")),
            "ppi_gate_lambda": float(os.getenv("PPI_GATE_LAMBDA", "1.0")),
            "ppi_train_donor": os.getenv("PPI_TRAIN_DONOR", "13272"),

            # Interactive mode settings
            "auto_model_selection": auto_model_selection,
            "parallel_execution": parallel_execution,
        }

        # Create research director
        director = ResearchDirectorAgent(
            research_question=question,
            domain=domain,
            config=flat_config
        )

        # Register director with AgentRegistry for message routing (Issue #66 fix)
        from kosmos.agents.registry import get_registry
        registry = get_registry()
        registry.register(director)
        logger.info(f"Registered ResearchDirector with AgentRegistry")

        # Run research with live progress (async)
        results = asyncio.run(run_with_progress_async(
            director,
            question,
            max_iterations,
            enable_streaming=stream,
            show_tokens=stream_tokens
        ))

        # Display results
        viewer = ResultsViewer()
        viewer.display_research_overview(results)
        viewer.display_hypotheses_table(results.get("hypotheses", []))
        viewer.display_experiments_table(results.get("experiments", []))

        if "metrics" in results:
            viewer.display_metrics_summary(results["metrics"])

        # Export if requested
        if output:
            if output.suffix == ".json":
                viewer.export_to_json(results, output)
            elif output.suffix in [".md", ".markdown"]:
                viewer.export_to_markdown(results, output)
            else:
                print_error(f"Unsupported output format: {output.suffix}")

        print_success("Research completed successfully!", title="Complete")

    except KeyboardInterrupt:
        console.print("\n[warning]Research interrupted by user[/warning]")
        raise typer.Exit(130)

    except Exception as e:
        print_error(f"Research failed: {str(e)}", title="Error")
        if "--debug" in sys.argv:
            raise
        raise typer.Exit(1)


async def run_with_progress_async(
    director,
    question: str,
    max_iterations: int,
    enable_streaming: bool = False,
    show_tokens: bool = True
) -> dict:
    """
    Run research with live progress display asynchronously.

    Args:
        director: ResearchDirectorAgent instance
        question: Research question
        max_iterations: Maximum iterations
        enable_streaming: Enable real-time event streaming
        show_tokens: Show LLM token streaming (with enable_streaming)

    Returns:
        Research results dictionary
    """
    # Initialize streaming display if enabled
    streaming_display = None
    if enable_streaming:
        try:
            from kosmos.cli.streaming import create_streaming_display
            streaming_display = create_streaming_display(
                console=console,
                process_id=None,  # Will receive all events
                show_tokens=show_tokens
            )
            streaming_display.start()
            logger.info("Streaming display enabled")
        except Exception as e:
            logger.warning(f"Could not enable streaming display: {e}")
    # Create progress bars
    progress = Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TimeRemainingColumn(),
        console=console,
    )

    # Create tasks for each phase
    hypothesis_task = progress.add_task("[cyan]Generating hypotheses...", total=100)
    experiment_task = progress.add_task("[yellow]Designing experiments...", total=100)
    execution_task = progress.add_task("[green]Executing experiments...", total=100)
    analysis_task = progress.add_task("[magenta]Analyzing results...", total=100)
    iteration_task = progress.add_task("[bright_blue]Research progress...", total=max_iterations)

    # Create current hypothesis table
    def create_status_table():
        table = Table(title="Current Status", box=None, show_header=True)
        table.add_column("Phase", style="cyan")
        table.add_column("Status", style="white")

        # Get current state from director
        state = getattr(director.workflow, "current_state", "INITIALIZING")
        iteration = getattr(director.research_plan, "iteration_count", 0)

        table.add_row("Workflow State", create_status_text(state))
        table.add_row("Iteration", f"{iteration}/{max_iterations}")
        table.add_row("Started", format_timestamp(datetime.now(timezone.utc)))

        return table

    # Run with live display
    with Live(progress, console=console, refresh_per_second=4):
        try:
            # Start research (async)
            await director.execute({"action": "start_research"})

            # Research loop - execute until convergence or max iterations
            iteration = 0
            loop_start_time = time.time()
            max_loop_duration = 7200  # 2 hours max runtime

            # Initialize stage tracker for this research run
            tracker = get_stage_tracker(process_id=f"research_{int(time.time())}")

            logger.info(f"Starting research loop (max_iterations={max_iterations})")

            while iteration < max_iterations:
                loop_iteration_start = time.time()

                # Check for timeout
                elapsed_time = time.time() - loop_start_time
                if elapsed_time > max_loop_duration:
                    logger.error(f"Research loop timed out after {elapsed_time:.0f}s")
                    print_error(
                        f"Research loop exceeded maximum runtime of {max_loop_duration}s. "
                        "This may indicate an infinite loop or hanging operation.",
                        title="Timeout"
                    )
                    break

                # Update tracker iteration
                tracker.set_iteration(iteration)

                # Get current status
                status = director.get_research_status()

                # Log current state with enhanced debug info
                logger.info(
                    f"Loop iteration {iteration}: workflow_state={status.get('workflow_state')}, "
                    f"iteration_count={status.get('iteration')}, "
                    f"has_converged={status.get('has_converged')}"
                )

                # Update iteration progress
                progress.update(iteration_task, completed=iteration + 1)

                # Update phase-specific progress based on workflow state
                from kosmos.core.workflow import WorkflowState
                workflow_state = status.get("workflow_state", WorkflowState.INITIALIZING.value)

                if workflow_state == WorkflowState.GENERATING_HYPOTHESES.value:
                    logger.debug("Phase: Generating hypotheses")
                    progress.update(hypothesis_task, completed=50)
                elif workflow_state == WorkflowState.DESIGNING_EXPERIMENTS.value:
                    logger.debug("Phase: Designing experiments")
                    progress.update(hypothesis_task, completed=100)
                    progress.update(experiment_task, completed=50)
                elif workflow_state == WorkflowState.EXECUTING.value:
                    logger.debug("Phase: Executing experiments")
                    progress.update(experiment_task, completed=100)
                    progress.update(execution_task, completed=50)
                elif workflow_state == WorkflowState.ANALYZING.value:
                    logger.debug("Phase: Analyzing results")
                    progress.update(execution_task, completed=100)
                    progress.update(analysis_task, completed=50)
                elif workflow_state in [WorkflowState.REFINING.value, WorkflowState.CONVERGED.value]:
                    logger.debug(f"Phase: {workflow_state}")
                    progress.update(analysis_task, completed=100)

                # Check for convergence
                if status.get("has_converged", False):
                    logger.info(f"Research converged: {status.get('convergence_reason')}")
                    progress.update(iteration_task, completed=max_iterations)
                    break

                # Execute next research step (async)
                logger.debug("Executing next research step")
                await director.execute({"action": "step"})

                # Update iteration counter from status
                new_iteration = status.get("iteration", iteration)
                if new_iteration != iteration:
                    logger.info(f"Iteration advanced from {iteration} to {new_iteration}")
                iteration = new_iteration

                # Log iteration timing and state at end of loop
                loop_duration = time.time() - loop_iteration_start
                logger.info(
                    "[ITER %d/%d] state=%s, hyps=%d, exps=%d, duration=%.2fs",
                    iteration, max_iterations,
                    status.get('workflow_state'),
                    status.get('hypothesis_pool_size', 0),
                    status.get('experiments_completed', 0),
                    loop_duration
                )

                # Small delay to allow UI updates (async)
                await asyncio.sleep(0.05)

            logger.info(f"Research loop completed after {iteration} iterations")

            # Mark all tasks as complete
            progress.update(hypothesis_task, completed=100)
            progress.update(experiment_task, completed=100)
            progress.update(execution_task, completed=100)
            progress.update(analysis_task, completed=100)

            # Get final research status
            final_status = director.get_research_status()

            # Provider usage: OpenAI/DeepSeek providers expose usage via
            # get_usage_stats(); legacy ClaudeClient exposes attributes.
            usage_stats = {}
            try:
                if hasattr(director.llm_client, "get_usage_stats"):
                    usage_stats = director.llm_client.get_usage_stats() or {}
            except Exception as e:
                logger.warning(f"Could not read provider usage stats: {e}")

            # Build results from actual research
            # Fetch actual hypothesis and experiment objects from database
            from kosmos.db import get_session
            from kosmos.db.operations import get_hypothesis, get_experiment

            hypotheses_data = []
            experiments_data = []

            try:
                # Check if research_plan exists
                if not director.research_plan:
                    logger.warning("No research plan available")
                    hypotheses_data = []
                    experiments_data = []
                else:
                    with get_session() as session:
                        # Fetch hypotheses from database using IDs
                        if hasattr(director.research_plan, 'hypothesis_pool') and director.research_plan.hypothesis_pool:
                            for h_id in director.research_plan.hypothesis_pool:
                                hypothesis = get_hypothesis(session, h_id)
                                if hypothesis:
                                    hypotheses_data.append(hypothesis.to_dict() if hasattr(hypothesis, 'to_dict') else str(hypothesis))

                        # Fetch experiments from database using IDs
                        if hasattr(director.research_plan, 'completed_experiments') and director.research_plan.completed_experiments:
                            for e_id in director.research_plan.completed_experiments:
                                experiment = get_experiment(session, e_id)
                                if experiment:
                                    experiments_data.append(experiment.to_dict() if hasattr(experiment, 'to_dict') else str(experiment))
            except Exception as e:
                logger.warning(f"Could not fetch all objects from database: {e}")
                # Fallback: use IDs as strings
                hypotheses_data = list(director.research_plan.hypothesis_pool)
                experiments_data = list(director.research_plan.completed_experiments)

            results = {
                "id": f"research_{int(time.time())}",
                "question": question,
                "domain": final_status.get("domain", "auto"),
                "state": final_status.get("workflow_state", "COMPLETED"),
                "current_iteration": final_status.get("iteration", 0),
                "max_iterations": max_iterations,
                "has_converged": final_status.get("has_converged", False),
                "convergence_reason": final_status.get("convergence_reason"),
                "hypotheses": hypotheses_data,
                "experiments": experiments_data,
                "metrics": {
                    "api_calls": usage_stats.get(
                        "total_requests",
                        getattr(director.llm_client, "total_requests", 0),
                    ),
                    "cache_hits": getattr(director.llm_client, 'cache_hits', 0),
                    "cache_misses": getattr(director.llm_client, 'cache_misses', 0),
                    "total_input_tokens": usage_stats.get("total_input_tokens", 0),
                    "total_output_tokens": usage_stats.get("total_output_tokens", 0),
                    "total_cost_usd": usage_stats.get("total_cost_usd"),
                    "hypotheses_generated": final_status.get("hypothesis_pool_size", 0),
                    "hypotheses_tested": final_status.get("hypotheses_tested", 0),
                    "hypotheses_supported": final_status.get("hypotheses_supported", 0),
                    "hypotheses_rejected": final_status.get("hypotheses_rejected", 0),
                    "experiments_executed": final_status.get("experiments_completed", 0),
                },
            }

            # Stop streaming display if enabled
            if streaming_display:
                streaming_display.stop()

            return results

        except Exception as e:
            # Stop streaming display on error
            if streaming_display:
                streaming_display.stop()
            console.print(f"\n[error]Error during research: {str(e)}[/error]")
            raise


if __name__ == "__main__":
    # Allow standalone testing
    typer.run(run_research)
