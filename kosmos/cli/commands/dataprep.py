"""`kosmos dataprep` — stage 1 of the fetch → review → experiment workflow.

Fetch data for a question, write the reviewable data package (`data.md`,
`plan.json`, `data_report.md`), and **stop**. Nothing trains here; the point of
the command is that a person can read `data.md`, fix the plan, and only then run
`kosmos experiment` on it.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import typer
from rich.panel import Panel

from kosmos.cli.utils import console, print_error, print_info, print_success
from kosmos.ppi.stages import default_data_dir, fetch_stage


def dataprep_command(
    question: str = typer.Argument(..., help="The research question, in plain words"),
    domain: str = typer.Option("biology", "--domain", "-d", help="Research domain"),
    task: Optional[str] = typer.Option(
        None,
        "--task",
        help=(
            "Which fetcher profile: auto | per_cell | perturbation. auto (default) "
            "lets the question's wording decide. perturbation looks for a screen "
            "with a condition column; per_cell looks for a labeled table."
        ),
    ),
    out: Optional[Path] = typer.Option(
        None,
        "--out",
        "-o",
        help="Directory for the data package (default: artifacts/dataprep/<slug>-<stamp>)",
    ),
    intent: Optional[str] = typer.Option(
        None, "--intent", help="A term the retrieval step should consider"
    ),
    candidate: Optional[List[str]] = typer.Option(
        None,
        "--candidate",
        help=(
            "A table to consider instead of searching (repeatable). Naming a "
            "candidate skips retrieval and judges exactly what you give it -- "
            "the path to hand stage 1 a screen you already have."
        ),
    ),
    hint: Optional[List[str]] = typer.Option(
        None, "--hint", help="A name the label or condition might have (repeatable)"
    ),
    exclude_column: Optional[List[str]] = typer.Option(
        None, "--exclude-column", help="A column that is never the label (repeatable)"
    ),
    fetch_limit: int = typer.Option(
        2, "--fetch-limit", help="Datasets the gold round may download at most"
    ),
    supp_limit: int = typer.Option(
        3, "--supp-limit", help="Supplementary tables the evidence round may download"
    ),
    supp_size: str = typer.Option(
        "any",
        "--supp-size",
        help="'large': prefer supplementary sources with the most cells (a small "
        "gold plus large unlabeled evidence) and order them largest first",
    ),
    search_hops: int = typer.Option(
        3,
        "--search-hops",
        help="How many times the model may look something up before naming datasets",
    ),
    max_files_per_fetch: Optional[int] = typer.Option(
        None,
        "--max-files-per-fetch",
        help="Files one fetch may contribute (default: 8; wider for a screen)",
    ),
    archive_members: Optional[int] = typer.Option(
        None,
        "--archive-members",
        help="Files to unpack from one archive (default: 12; wider for a screen)",
    ),
    fetch_retries: int = typer.Option(
        1,
        "--fetch-retries",
        help="Times the model may correct an identifier a repository refused",
    ),
    mechanical_search: bool = typer.Option(
        False,
        "--mechanical-search",
        help="Also run the rule-built keyword search before the model chooses",
    ),
    refresh: bool = typer.Option(
        False,
        "--refresh",
        help="Download again even when matching bytes are already staged",
    ),
    max_bytes: Optional[int] = typer.Option(
        None, "--max-bytes", help="Size cap per downloaded dataset, in bytes"
    ),
    max_download_mb: Optional[float] = typer.Option(
        None, "--max-download-mb", help="Size cap for the whole fetch, in megabytes"
    ),
    no_llm: bool = typer.Option(
        False, "--no-llm", help="Match names mechanically, without a model"
    ),
    adjudicate: Optional[Path] = typer.Option(
        None,
        "--adjudicate",
        help="JSON file of human answers to a previous run's disagreements",
    ),
    dataset: Optional[str] = typer.Option(
        None,
        "--dataset",
        help=(
            "A registered perturbation screen to use instead of searching "
            "(e.g. replogle_k562_essential, norman, adamson). Pins the data so "
            "the question always trains on the same screen."
        ),
    ),
    supplementary_dataset: Optional[List[str]] = typer.Option(
        None,
        "--supplementary-dataset",
        help="A registered screen to use as supplementary data (repeatable; with --dataset)",
    ),
) -> None:
    """Fetch and plan for QUESTION, then stop for human review."""
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
    if adjudicate and not adjudicate.exists():
        print_error(f"Adjudication file not found: {adjudicate}")
        raise typer.Exit(1)

    out_dir = Path(out) if out else default_data_dir(question)
    if dataset:
        from kosmos.ppi.perturbation.registry import write_registered_plan

        try:
            staged = write_registered_plan(
                question=question,
                dataset=dataset,
                supplementary=tuple(supplementary_dataset or ()),
                out_dir=out_dir,
            )
        except Exception as e:  # noqa: BLE001 - report the stage, do not crash the CLI
            print_error(f"Stage 1 failed: {type(e).__name__}: {e}", title="Data prep error")
            raise typer.Exit(1) from e
        print_success(f"Data package ready for review: {staged['data_md']}", title="Stage 1 complete")
        console.print("[bold]Then run stage 2 (prepare):[/bold]")
        console.print(
            f"  .venv/bin/python scripts/prepare_perturbation.py --stage1 {out_dir} --out <dir>"
        )
        return
    console.print()
    console.print(
        Panel(
            f"[cyan]Stage 1 — fetch and plan[/cyan]\n\n"
            f"**Question:** {question}\n"
            f"**Profile:** {task or 'auto'}\n"
            f"**Output:** {out_dir}",
            title="[bright_blue]Kosmos · Data prep[/bright_blue]",
            border_style="bright_blue",
        )
    )
    console.print()

    try:
        outcome = fetch_stage(
            question=question,
            out_dir=out_dir,
            domain=domain,
            backend=task,
            intent=intent or "",
            candidates=tuple(candidate or ()),
            hints=tuple(hint or ()),
            exclude_columns=tuple(exclude_column or ()),
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
            adjudicate=str(adjudicate) if adjudicate else None,
        )
    except Exception as e:  # noqa: BLE001 - report the stage, do not crash the CLI
        print_error(f"Stage 1 failed: {type(e).__name__}: {e}", title="Data prep error")
        raise typer.Exit(1) from e

    if not outcome.ok:
        print_error(outcome.error or "the fetcher produced no plan", title="No data package")
        # The fetcher still wrote a report describing why; point at it rather
        # than sending the reader to look for it.
        report = out_dir / "data_report.md"
        if report.exists():
            print_info(f"What it found and refused is in {report}")
        raise typer.Exit(2)

    document = Path(str(outcome.metrics.get("data_md")))
    print_success(
        f"Data package ready for review: {document}", title="Stage 1 complete"
    )
    console.print()
    console.print("[bold]Review it:[/bold]")
    console.print(f"  {document}")
    console.print("  then edit the plan if the task or the tables are wrong:")
    console.print(f"  {outcome.plan_path}")
    console.print()
    console.print("[bold]Then run stage 2:[/bold]")
    console.print(
        f'  .venv/bin/python -m kosmos.cli.main experiment "{question}" '
        f"--data-plan {outcome.plan_path} --yes"
    )


__all__ = ["dataprep_command"]
