"""`kosmos experiment` — stage 2 of the fetch → review → experiment workflow.

Train **only** from a plan a person has reviewed. No retrieval, no hypothesis
generation, no code generation: the plan written by `kosmos dataprep` is the
whole input, and `run/summary.md` + `run/metrics.json` beside it are the whole
output. Re-running trains again into `run-2`, `run-3`, ... (a finished run is
never overwritten).
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import typer
from rich.panel import Panel

from kosmos.cli.utils import console, print_error, print_info, print_success
from kosmos.ppi.stages import experiment_stage


def experiment_command(
    question: str = typer.Argument(..., help="The question the plan answers"),
    data_plan: Path = typer.Option(
        ...,
        "--data-plan",
        help="The plan.json a `kosmos dataprep` run wrote (reviewed or edited)",
    ),
    task: Optional[str] = typer.Option(
        None,
        "--task",
        help="Pin the backend: auto | per_cell | perturbation (default: the plan's own task)",
    ),
    out: Optional[Path] = typer.Option(
        None, "--out", "-o", help="Output directory (default: the plan's directory)"
    ),
    max_epochs: int = typer.Option(20, "--max-epochs", help="Training epochs"),
    patience: int = typer.Option(5, "--patience", help="Early-stopping patience"),
    seed: int = typer.Option(42, "--seed", help="Random seed"),
    loss: Optional[str] = typer.Option(
        None, "--loss", help="Objective override: signed | gradient_gated | plain"
    ),
    condition_column: Optional[str] = typer.Option(
        None,
        "--condition-column",
        help="perturbation: the column holding the perturbation identity",
    ),
    control_label: Optional[List[str]] = typer.Option(
        None, "--control-label", help="perturbation: a value meaning 'nothing perturbed'"
    ),
    split_mode: str = typer.Option(
        "mixed", "--split-mode", help="perturbation: mixed | unseen_single | unseen_combination"
    ),
    go_graph: Optional[Path] = typer.Option(
        None, "--go-graph", help="perturbation: GO similarity edge list"
    ),
    eta: float = typer.Option(
        1.0, "--eta", help="perturbation: how far the auxiliary graph may move the state"
    ),
    assume_yes: bool = typer.Option(
        False, "--yes", help="Do not ask before training the plan"
    ),
) -> None:
    """Train the plan DATA_PLAN. Fetching is stage 1 and is never done here."""
    import sys

    if not data_plan.exists():
        print_error(f"Data plan not found: {data_plan}")
        raise typer.Exit(1)
    if go_graph and not go_graph.exists():
        print_error(f"GO graph not found: {go_graph}")
        raise typer.Exit(1)
    if task and task.strip().lower().replace("-", "_") not in {
        "auto",
        "per_cell",
        "percell",
        "simple",
        "perturbation",
        "perturbation_response",
        "perturb",
    }:
        print_error(f"Unknown --task {task!r}; use auto, per_cell or perturbation")
        raise typer.Exit(1)
    if split_mode not in {"mixed", "unseen_single", "unseen_combination"}:
        print_error(f"Unknown --split-mode {split_mode!r}")
        raise typer.Exit(1)
    if loss and loss.replace("-", "_") not in {"signed", "gradient_gated", "plain"}:
        print_error(f"Unknown --loss {loss!r}; use signed, gradient_gated or plain")
        raise typer.Exit(1)

    console.print()
    console.print(
        Panel(
            f"[cyan]Stage 2 — experiment[/cyan]\n\n"
            f"**Question:** {question}\n"
            f"**Plan:** {data_plan}",
            title="[bright_blue]Kosmos · Experiment[/bright_blue]",
            border_style="bright_blue",
        )
    )
    console.print()

    if not assume_yes:
        if not sys.stdin.isatty():
            print_error(
                "training a plan needs confirmation; re-run with --yes to accept "
                "it non-interactively"
            )
            raise typer.Exit(1)
        if not typer.confirm(f"Train {data_plan}?", default=True):
            console.print("[warning]Cancelled.[/warning]")
            raise typer.Exit(0)

    try:
        outcome = experiment_stage(
            question=question,
            plan_path=data_plan,
            out_dir=out,
            backend=task,
            max_epochs=max_epochs,
            patience=patience,
            seed=seed,
            loss_mode=loss.replace("-", "_") if loss else None,
            condition_column=condition_column,
            control_labels=tuple(control_label or ()),
            split_mode=split_mode,
            go_graph=go_graph,
            eta=eta,
        )
    except Exception as e:  # noqa: BLE001 - report the stage, do not crash the CLI
        print_error(f"Stage 2 failed: {type(e).__name__}: {e}", title="Experiment error")
        raise typer.Exit(1) from e

    if not outcome.ok:
        print_error(
            (outcome.error or "the plan could not be trained").splitlines()[0],
            title=f"Stage 2 failed at '{outcome.stage}'",
        )
        raise typer.Exit(1)

    print_success(f"Run complete: {outcome.summary_path}", title="Stage 2 complete")
    if outcome.run_dir:
        print_info(f"Artifacts: {outcome.run_dir}")
    if outcome.metrics:
        console.print()
        console.print("[bold]Headline metrics[/bold]")
        for key in ("mse_deg", "pearson", "spearman", "direction", "top_k", "accuracy"):
            if key in outcome.metrics:
                console.print(f"  {key}: {outcome.metrics[key]}")


__all__ = ["experiment_command"]
