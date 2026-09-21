"""`kosmos infer-task`: name the column a question asks us to predict.

Prints the decision and how it was made. Exit code 2 when nothing was chosen,
so a script can tell "no answer" from "an error".
"""

from __future__ import annotations

import json as json_module
from pathlib import Path

import typer
from rich.table import Table

from kosmos.cli.utils import console, print_error, print_info


def infer_task_command(
    table: Path = typer.Option(..., "--table", help="Table to choose a label column from"),
    objective: str = typer.Option("", "--objective", help="The research question"),
    hint: list[str] | None = typer.Option(
        None, "--hint", help="A name the target might have (repeatable)"
    ),
    exclude: list[str] | None = typer.Option(
        None, "--exclude-col", help="Columns that cannot be the target (repeatable)"
    ),
    feature_prefix: list[str] | None = typer.Option(
        None,
        "--feature-prefix",
        help="Name prefixes that are features, never the label (repeatable)",
    ),
    use_llm: bool = typer.Option(
        True, "--llm/--no-llm", help="Ask the configured model when names do not match"
    ),
    min_per_class: int = typer.Option(2, "--min-per-class"),
    as_json: bool = typer.Option(False, "--json", help="Machine-readable output"),
) -> None:
    """Decide which column is the label, or explain why none can be."""
    from kosmos.ppi.task_inference import column_facts, infer_target_column

    path = Path(table)
    if not path.exists():
        print_error(f"Table not found: {path}")
        raise typer.Exit(1)

    client = None
    if use_llm:
        try:
            from kosmos.core.llm import get_client

            client = get_client()
        except Exception as e:  # noqa: BLE001 - inference still works without a model
            print_info(f"No model available ({e}); matching names only.")

    facts = column_facts(path)
    result = infer_target_column(
        path,
        objective=objective,
        hints=list(hint or []),
        exclude=list(exclude or []),
        feature_prefixes=list(feature_prefix or []),
        client=client,
        facts=facts,
        min_per_class=min_per_class,
    )
    if as_json:
        console.print_json(json_module.dumps(result.to_dict(), default=str))
    else:
        table_view = Table(title=f"Target column inference for {path.name}")
        table_view.add_column("field")
        table_view.add_column("value")
        table_view.add_row("target column", str(result.column))
        table_view.add_row("decided by", f"{result.source} ({result.confidence})")
        table_view.add_row("reason", result.reason)
        if result.candidates:
            table_view.add_row(
                "other candidates",
                ", ".join(f"{c.column} ({c.score:.2f})" for c in result.candidates[:5]),
            )
        if result.rejected:
            shown = [k for k in result.rejected if not k.startswith("__")][:4]
            table_view.add_row(
                "rejected", "; ".join(f"{k}: {result.rejected[k]}" for k in shown) or "-"
            )
        console.print(table_view)
    if result.column is None:
        # 2 = "asked and got no answer", which a script can handle without
        # treating it as a crash.
        raise typer.Exit(2)


__all__ = ["infer_task_command"]
