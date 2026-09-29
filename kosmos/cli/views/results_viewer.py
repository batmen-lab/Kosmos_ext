"""
Results viewer for Kosmos CLI.

Provides beautiful visualization of research results, hypotheses, experiments,
and analysis using Rich library components.
"""

import json
import os
from datetime import datetime
from typing import Optional, List, Dict, Any
from pathlib import Path

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.tree import Tree
from rich.syntax import Syntax
from rich.markdown import Markdown
from rich.text import Text
from rich.columns import Columns

from kosmos.cli.utils import (
    console,
    create_table,
    format_timestamp,
    format_duration,
    format_currency,
    truncate_text,
    create_status_text,
    create_domain_text,
    create_metric_text,
    get_icon,
)
from kosmos.cli.themes import get_domain_color, get_state_color, get_box_style


class ResultsViewer:
    """Viewer for displaying research results in various formats."""

    def __init__(self, console_instance: Optional[Console] = None):
        """
        Initialize results viewer.

        Args:
            console_instance: Optional Rich Console instance
        """
        self.console = console_instance or console

    def display_research_overview(self, research_data: Dict[str, Any]):
        """
        Display overview of a research run.

        Args:
            research_data: Research run data with metadata
        """
        run_id = research_data.get("id", "Unknown")
        question = research_data.get("question", "Unknown")
        # `or` (not a get-default): the key is often PRESENT with value None
        # (no --domain, auto-detect not recorded), and None crashed .title() /
        # get_domain_color() and killed the whole results display + export.
        domain = research_data.get("domain") or "general"
        state = research_data.get("state", "Unknown")
        iteration = research_data.get("current_iteration", 0)
        max_iterations = research_data.get("max_iterations", 10)

        # Create overview panel
        overview_text = [
            f"**Run ID:** {run_id}",
            f"**Domain:** {domain.title()}",
            f"**State:** {state}",
            f"**Progress:** Iteration {iteration}/{max_iterations} ({iteration/max_iterations*100:.1f}%)",
            "",
            f"**Question:** {question}",
        ]

        self.console.print()
        self.console.print(
            Panel(
                "\n".join(overview_text),
                title=f"[h2]{get_icon('flask')} Research Overview[/h2]",
                border_style=get_domain_color(domain),
                box=get_box_style("default"),
            )
        )
        self.console.print()

    def display_hypotheses_table(self, hypotheses: List[Dict[str, Any]]):
        """
        Display table of hypotheses.

        Args:
            hypotheses: List of hypothesis dictionaries
        """
        if not hypotheses:
            self.console.print("[muted]No hypotheses yet.[/muted]")
            return

        table = create_table(
            title=f"{get_icon('magnifying_glass')} Hypotheses",
            columns=["#", "Claim", "Novelty", "Priority", "Status"],
            show_lines=False,
        )

        for i, hyp in enumerate(hypotheses, 1):
            claim = truncate_text(hyp.get("claim", "Unknown"), 50)
            novelty = hyp.get("novelty_score", 0.0)
            priority = hyp.get("priority_score", 0.0)
            status = hyp.get("status", "pending")

            table.add_row(
                str(i),
                claim,
                create_metric_text(novelty, format_type="number"),
                create_metric_text(priority, format_type="number"),
                create_status_text(status),
            )

        self.console.print(table)
        self.console.print()

    def display_hypothesis_tree(self, hypotheses: List[Dict[str, Any]]):
        """
        Display hypothesis evolution as a tree.

        Args:
            hypotheses: List of hypothesis dictionaries with parent relationships
        """
        if not hypotheses:
            self.console.print("[muted]No hypothesis tree available.[/muted]")
            return

        # Build tree structure
        tree = Tree(
            f"[h2]{get_icon('brain')} Hypothesis Evolution[/h2]",
            guide_style="bright_black"
        )

        # Group by parent
        root_hypotheses = [h for h in hypotheses if not h.get("parent_id")]
        children_map = {}

        for hyp in hypotheses:
            parent_id = hyp.get("parent_id")
            if parent_id:
                if parent_id not in children_map:
                    children_map[parent_id] = []
                children_map[parent_id].append(hyp)

        def add_hypothesis_node(parent_node, hypothesis):
            """Recursively add hypothesis nodes."""
            claim = truncate_text(hypothesis.get("claim", "Unknown"), 60)
            novelty = hypothesis.get("novelty_score", 0.0)
            status = hypothesis.get("status", "pending")

            node_label = (
                f"{claim}\n"
                f"[muted]Novelty: {novelty:.2f} | Status: {status}[/muted]"
            )

            node = parent_node.add(node_label)

            # Add children
            hyp_id = hypothesis.get("id")
            if hyp_id in children_map:
                for child in children_map[hyp_id]:
                    add_hypothesis_node(node, child)

        # Add root hypotheses
        for hyp in root_hypotheses:
            add_hypothesis_node(tree, hyp)

        self.console.print(tree)
        self.console.print()

    def display_experiments_table(self, experiments: List[Dict[str, Any]]):
        """
        Display table of experiments.

        Args:
            experiments: List of experiment dictionaries
        """
        if not experiments:
            self.console.print("[muted]No experiments yet.[/muted]")
            return

        table = create_table(
            title=f"{get_icon('flask')} Experiments",
            columns=["#", "Type", "Status", "Duration", "Timestamp"],
            show_lines=False,
        )

        for i, exp in enumerate(experiments, 1):
            exp_type = exp.get("type", "Unknown")
            status = exp.get("status", "pending")
            duration = exp.get("duration_seconds", 0)
            timestamp = exp.get("created_at")

            # Parse timestamp if string
            if isinstance(timestamp, str):
                try:
                    timestamp = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
                except (ValueError, TypeError):
                    timestamp = None

            table.add_row(
                str(i),
                exp_type,
                create_status_text(status),
                format_duration(duration),
                format_timestamp(timestamp) if timestamp else "[muted]Unknown[/muted]",
            )

        self.console.print(table)
        self.console.print()

    def display_experiment_details(self, experiment: Dict[str, Any]):
        """
        Display detailed view of a single experiment.

        Args:
            experiment: Experiment dictionary with full details
        """
        exp_id = experiment.get("id", "Unknown")
        exp_type = experiment.get("type", "Unknown")
        status = experiment.get("status", "Unknown")

        # Header panel
        header = [
            f"**Experiment ID:** {exp_id}",
            f"**Type:** {exp_type}",
            f"**Status:** {status}",
        ]

        self.console.print()
        self.console.print(
            Panel(
                "\n".join(header),
                title=f"[cyan]{get_icon('flask')} Experiment Details[/cyan]",
                border_style="cyan",
            )
        )

        # Parameters
        if "parameters" in experiment:
            self.console.print("\n[h3]Parameters:[/h3]")
            params_json = json.dumps(experiment["parameters"], indent=2)
            syntax = Syntax(params_json, "json", theme="monokai", line_numbers=False)
            self.console.print(syntax)

        # Results
        if "results" in experiment:
            self.console.print("\n[h3]Results:[/h3]")
            results_json = json.dumps(experiment["results"], indent=2)
            syntax = Syntax(results_json, "json", theme="monokai", line_numbers=False)
            self.console.print(syntax)

        # Code (if available)
        if "code" in experiment:
            self.console.print("\n[h3]Generated Code:[/h3]")
            syntax = Syntax(
                experiment["code"],
                "python",
                theme="monokai",
                line_numbers=True,
            )
            self.console.print(syntax)

        self.console.print()

    def display_metrics_summary(self, metrics: Dict[str, Any]):
        """
        Display research metrics summary.

        Args:
            metrics: Metrics dictionary
        """
        # API metrics
        api_table = create_table(
            title=f"{get_icon('info')} API Usage",
            columns=["Metric", "Value"],
            show_lines=True,
        )

        api_calls = metrics.get("api_calls", 0)
        cache_hits = metrics.get("cache_hits", 0)
        cache_misses = metrics.get("cache_misses", 0)
        total_cache = cache_hits + cache_misses
        hit_rate = (cache_hits / total_cache * 100) if total_cache > 0 else 0

        api_table.add_row("Total API Calls", str(api_calls))
        api_table.add_row("Cache Hits", f"{cache_hits} ({hit_rate:.1f}%)")
        api_table.add_row("Cache Misses", str(cache_misses))

        if "total_cost_usd" in metrics:
            api_table.add_row("Total Cost", format_currency(metrics["total_cost_usd"]))

        self.console.print(api_table)
        self.console.print()

        # Research metrics
        research_table = create_table(
            title=f"{get_icon('brain')} Research Progress",
            columns=["Metric", "Value"],
            show_lines=True,
        )

        research_table.add_row("Hypotheses Generated", str(metrics.get("hypotheses_generated", 0)))
        research_table.add_row("Experiments Executed", str(metrics.get("experiments_executed", 0)))
        research_table.add_row("Successful Experiments", str(metrics.get("successful_experiments", 0)))
        research_table.add_row("Failed Experiments", str(metrics.get("failed_experiments", 0)))

        self.console.print(research_table)
        self.console.print()

    def export_to_json(self, data: Dict[str, Any], output_path: Path):
        """
        Export results to JSON file.

        Args:
            data: Data to export
            output_path: Output file path
        """
        try:
            # Ensure the target directory exists so a completed run never loses
            # its report to a missing parent dir (the path may be relative to a
            # cwd that has no such folder).
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "w") as f:
                json.dump(data, f, indent=2, default=str)

            self.console.print(f"[success]Exported to {output_path}[/success]")
        except Exception as e:
            self.console.print(f"[error]Export failed: {str(e)}[/error]")

    def export_to_markdown(self, data: Dict[str, Any], output_path: Path):
        """
        Export results to Markdown file.

        Args:
            data: Data to export
            output_path: Output file path
        """
        try:
            lines = [
                f"# Research Results: {data.get('question', 'Unknown')}",
                "",
                f"**Run ID:** {data.get('id', 'Unknown')}",
                f"**Domain:** {data.get('domain', 'Unknown')}",
                f"**Status:** {data.get('state', 'Unknown')}",
                "",
                "## Hypotheses",
                "",
            ]

            for i, hyp in enumerate(data.get("hypotheses", []), 1):
                lines.extend([
                    f"### {i}. {hyp.get('claim', 'Unknown')}",
                    f"",
                    f"- **Novelty:** {_fmt_scalar(hyp.get('novelty_score') or 0, '.2f')}",
                    f"- **Priority:** {_fmt_scalar(hyp.get('priority_score') or 0, '.2f')}",
                    f"- **Status:** {hyp.get('status', 'Unknown')}",
                    "",
                ])

            lines.extend([
                "## Experiments",
                "",
            ])

            experiments = data.get("experiments", []) or []
            pending = data.get("pending_experiments", []) or []
            if not experiments:
                # An empty section reads as "the run designed nothing", which
                # is a different failure from "the run ran out of iterations
                # with experiments still queued" -- and the second one hides
                # real work. Name which one happened.
                if pending:
                    lines.extend([
                        f"**None executed.** {len(pending)} experiment(s) were "
                        f"designed and still queued when the run ended at "
                        f"iteration {data.get('current_iteration', '?')} of "
                        f"{data.get('max_iterations', '?')}. The run was cut "
                        f"short, not empty.",
                        "",
                    ])
                else:
                    lines.extend([
                        "**None.** No experiment was designed or executed, so "
                        "no hypothesis above was tested.",
                        "",
                    ])

            for i, exp in enumerate(experiments, 1):
                lines.extend([
                    f"### {i}. {exp.get('type', 'Unknown')}",
                    f"",
                    f"- **Status:** {exp.get('status', 'Unknown')}",
                    f"- **Duration:** {format_duration(exp.get('duration_seconds', 0))}",
                ])
                if exp.get("description"):
                    lines.append(f"- **Design:** {exp['description']}")
                if exp.get("error_message"):
                    lines.append(f"- **Error:** {exp['error_message']}")
                lines.append("")

                # The findings themselves. Without this the report says an
                # experiment ran and how long it took, but not what it found --
                # making a completed run indistinguishable from a no-op, and
                # silently hiding null results (a non-significant p-value is a
                # finding, not the absence of one).
                for result in exp.get("results") or []:
                    if not isinstance(result, dict):
                        continue
                    lines.extend(["**Findings**", ""])

                    payload = result.get("data")

                    # First, before any number: if code generation fell back,
                    # the Design text above describes an experiment that did
                    # not run, and every figure below it belongs to a narrower
                    # analysis. Reading the numbers under that design without
                    # this line is how a template's output gets taken for the
                    # multi-dataset result it replaced.
                    note = payload.get("analysis_note") if isinstance(payload, dict) else None
                    if note:
                        lines.extend([f"> {note}", ""])

                    # These three are numbers by contract but not by fact: a
                    # multi-cohort experiment returned `n_samples` as
                    # {'wilk': 14, 'stephenson': 0, 'ren': 0}, and `f"{n:,}"` on
                    # a dict raised "unsupported format string passed to
                    # dict.__format__" -- which aborted the WHOLE export, so a
                    # run that had produced five results wrote no report at all.
                    # Format numbers as numbers and render anything else as-is.
                    n = payload.get("n_samples") if isinstance(payload, dict) else None
                    if n is not None:
                        lines.append(f"- Sample size: {_fmt_scalar(n, ',')}")

                    p = result.get("p_value")
                    if p is not None:
                        if isinstance(p, (int, float)) and not isinstance(p, bool):
                            verdict = "significant" if p < 0.05 else "NOT significant"
                            lines.append(f"- p-value: {p:.4g} ({verdict} at alpha=0.05)")
                        else:
                            lines.append(f"- p-value: {_fmt_scalar(p, '.4g')}")
                    effect = result.get("effect_size")
                    if effect is not None:
                        lines.append(f"- Effect size: {_fmt_scalar(effect, '.4g')}")
                    ci = result.get("confidence_interval")
                    if ci:
                        lines.append(f"- 95% CI: {ci}")
                    supports = result.get("supports_hypothesis")
                    if supports is not None:
                        lines.append(f"- Supports hypothesis: {supports}")

                    tests = result.get("statistical_tests")
                    if isinstance(tests, dict):
                        for test_name, stats in tests.items():
                            if isinstance(stats, dict):
                                inner = ", ".join(
                                    f"{k}={v:.4g}" if isinstance(v, (int, float)) else f"{k}={v}"
                                    for k, v in stats.items()
                                )
                                lines.append(f"- {test_name}: {inner}")

                    # These arrive from a JSON column, so an absent value can
                    # reach here as the literal string "null"/"None" rather than
                    # Python None. Rendering that verbatim would print
                    # "Key findings: null", which reads as a finding rather than
                    # as the absence of one.
                    def _present(value):
                        if value is None:
                            return None
                        text = str(value).strip()
                        return text if text and text.lower() not in {"null", "none", "{}", "[]"} else None

                    # Everything else the experiment returned.
                    #
                    # The renderer knew four keys -- n_samples, p_value,
                    # effect_size, statistical_tests -- and silently dropped the
                    # rest. A run that returned `causal_proteins`, `all_ranked`
                    # (14 proteins with IVW/weighted-median/Egger estimates),
                    # `coloc_summary` and `notes` therefore exported an EMPTY
                    # Findings section while its whole result sat in the
                    # database. An experiment's payload is the experiment's
                    # answer; the report cannot pick which parts of it count.
                    if isinstance(payload, dict):
                        lines.extend(_render_payload(payload))

                    figs = result.get("figures")
                    if not figs and isinstance(payload, dict):
                        # Nothing was collected, but the code said it wrote
                        # something. That happens when it saved outside the
                        # figure directory -- and since figure_path/figures are
                        # filtered out of the payload above, suppressing them
                        # here too would delete the only record of where the
                        # file went. Fall back to what the payload claims.
                        figs = payload.get("figures") or payload.get("figure_path")
                        if isinstance(figs, str):
                            figs = [figs]
                    if figs:
                        rendered = _render_figures(figs, Path(output_path).parent)
                        if rendered:
                            lines.extend(["", "**Figures**", ""])
                            lines.extend(rendered)

                    findings = _present(result.get("key_findings"))
                    if findings:
                        lines.append(f"- Key findings: {findings}")
                    interpretation = _present(result.get("interpretation"))
                    if interpretation:
                        lines.extend(["", interpretation])
                    lines.append("")

            if pending:
                lines.extend(["## Designed but not run", ""])
                for i, exp in enumerate(pending, 1):
                    lines.append(
                        f"### {i}. {exp.get('experiment_type', 'experiment')} "
                        f"({exp.get('status', 'queued')})"
                    )
                    if exp.get("description"):
                        lines.extend(["", f"- **Design:** {exp['description']}"])
                    lines.append("")

            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "w") as f:
                f.write("\n".join(lines))

            self.console.print(f"[success]Exported to {output_path}[/success]")
        except Exception as e:
            self.console.print(f"[error]Export failed: {str(e)}[/error]")


_RENDERED_ELSEWHERE = frozenset({
    # Already rendered above by name; repeating them would double-report.
    "n_samples", "p_value", "effect_size", "statistical_tests",
    # Our own annotation, rendered as a caveat before the numbers.
    "analysis_note",
    # Rendered as images in their own block; the raw paths would be noise.
    "figures", "figure_path", "figure_saved",
})

_MAX_ROWS = 12
_MAX_COLS = 8


def _fmt_scalar(value, spec: str) -> str:
    """Apply a numeric format spec only to a real number; render the rest as-is.

    An experiment may hand back a dict or list where the renderer expected a
    scalar (per-cohort sample sizes, a p-value per arm). Those are findings
    too; they must not take the export down with a format error.
    """
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, (int, float)):
        try:
            return format(value, spec)
        except (ValueError, TypeError):
            return str(value)
    if isinstance(value, dict):
        return ", ".join(f"{k}={_fmt_scalar(v, spec)}" for k, v in value.items())
    if isinstance(value, (list, tuple)):
        return "[" + ", ".join(_fmt_scalar(v, spec) for v in value) + "]"
    return str(value)


def _fmt(value):
    """A scalar, formatted for a report. Small floats keep their exponent."""
    if isinstance(value, bool) or value is None:
        return str(value)
    if isinstance(value, float):
        if value != value:  # NaN
            return "NaN"
        return f"{value:.4g}"
    return str(value)


def _render_figures(figures, report_dir) -> list:
    """Figures as markdown image references, relative to the report.

    Relative because the report is a file someone opens from wherever it sits:
    an absolute /private/var/... path resolves on exactly one machine. A figure
    whose file is missing is NAMED rather than linked -- a broken image in a
    report reads as a rendering bug, while a line saying the file is gone is a
    fact about the run.

    Never raises: a figure block must not cost the report, which has already
    happened once in this file.
    """
    out: list = []
    try:
        if not isinstance(figures, (list, tuple)) or not figures:
            return out
        shown = 0
        for entry in figures:
            if isinstance(entry, str):
                path, caption = entry, ""
            elif isinstance(entry, dict):
                path, caption = entry.get("path"), (entry.get("caption") or "")
            else:
                continue
            if not isinstance(path, str) or not path.strip():
                continue
            name = os.path.basename(path)
            caption = str(caption).strip() or os.path.splitext(name)[0].replace("_", " ")
            if not os.path.isfile(path):
                out.append(f"- Figure recorded but no longer on disk: `{name}`")
                continue
            if shown >= 8:
                out.append(f"- Also written: `{name}`")
                continue
            try:
                ref = os.path.relpath(path, str(report_dir))
            except (ValueError, OSError):
                ref = path
            ref = ref.replace(os.sep, "/").replace(" ", "%20")
            if name.lower().endswith((".png", ".svg")):
                out.append(f"![{caption}]({ref})")
                out.append("")
                out.append(f"*{caption}*")
            else:
                out.append(f"[{caption}]({ref})")
            out.append("")
            shown += 1
    except Exception:  # pragma: no cover - a figure must never cost the report
        return out
    return out


def _render_payload(payload: dict) -> list:
    """Render an experiment's result dict as markdown, whatever shape it has.

    Bounded, and it SAYS when it bounds: a table cut at `_MAX_ROWS` prints how
    many rows it dropped, because a silently truncated table reads as the whole
    result.
    """
    lines: list = []
    for key, value in payload.items():
        if key in _RENDERED_ELSEWHERE:
            continue

        # A list of uniform dicts is a table -- the shape a ranked result takes.
        if (
            isinstance(value, list)
            and value
            and all(isinstance(v, dict) for v in value)
        ):
            columns = list(value[0].keys())[:_MAX_COLS]
            lines.extend([f"", f"**{key}** ({len(value)} rows)", ""])
            lines.append("| " + " | ".join(columns) + " |")
            lines.append("|" + "|".join(["---"] * len(columns)) + "|")
            for row in value[:_MAX_ROWS]:
                lines.append(
                    "| " + " | ".join(_fmt(row.get(c)) for c in columns) + " |"
                )
            if len(value) > _MAX_ROWS:
                lines.append("")
                lines.append(f"_{len(value) - _MAX_ROWS} further rows not shown._")
            lines.append("")
        elif isinstance(value, list) and not value:
            lines.append(f"- {key}: none")
        elif isinstance(value, list):
            shown = [_fmt(v) for v in value[:_MAX_ROWS]]
            suffix = "" if len(value) <= _MAX_ROWS else f" (+{len(value) - _MAX_ROWS} more)"
            lines.append(f"- {key}: {'; '.join(shown)}{suffix}")
        elif isinstance(value, dict):
            inner = ", ".join(
                f"{k}={_fmt(v)}" for k, v in list(value.items())[:_MAX_COLS]
                if not isinstance(v, (dict, list))
            )
            lines.append(f"- {key}: {inner}" if inner else f"- {key}: {len(value)} entries")
        else:
            lines.append(f"- {key}: {_fmt(value)}")
    return lines


# Convenience functions
def view_research_results(research_data: Dict[str, Any]):
    """Display complete research results."""
    viewer = ResultsViewer()

    viewer.display_research_overview(research_data)
    viewer.display_hypotheses_table(research_data.get("hypotheses", []))
    viewer.display_experiments_table(research_data.get("experiments", []))

    if "metrics" in research_data:
        viewer.display_metrics_summary(research_data["metrics"])


def view_hypothesis_evolution(hypotheses: List[Dict[str, Any]]):
    """Display hypothesis evolution tree."""
    viewer = ResultsViewer()
    viewer.display_hypothesis_tree(hypotheses)


def view_experiment_details(experiment: Dict[str, Any]):
    """Display detailed experiment view."""
    viewer = ResultsViewer()
    viewer.display_experiment_details(experiment)
