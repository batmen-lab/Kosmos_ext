"""
Run command for Kosmos CLI.

Executes autonomous research with live progress visualization.

Async Architecture (Issue #66 fix):
- run_with_progress() is now async
- Uses asyncio.run() at CLI entry point
- ResearchDirector.execute() is now async
"""

import os
import sys
import time
import logging
import asyncio
from typing import Optional
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
from rich.markup import escape
from rich.text import Text

from kosmos.cli.utils import (
    console,
    print_success,
    print_error,
    print_warning,
    print_info,
    get_icon,
    format_timestamp,
    create_status_text,
)
from kosmos.cli.interactive import run_interactive_mode
from kosmos.cli.views.results_viewer import ResultsViewer
from kosmos.core.stage_tracker import get_stage_tracker

logger = logging.getLogger(__name__)


def _serialize_result(result) -> dict:
    """Turn a Result ORM row into the plain dict the report renderer expects.

    Written out by hand because the Result model, unlike Experiment and
    Hypothesis, does not define `to_dict()`. JSON-typed columns can come back
    either already-decoded or as raw strings depending on how they were
    written, so decode defensively -- a string that fails to parse is passed
    through rather than dropped, since partial findings beat none.
    """
    import json

    def _maybe_json(value):
        if isinstance(value, str):
            try:
                return json.loads(value)
            except (ValueError, TypeError):
                return value
        return value

    return {
        "id": getattr(result, "id", None),
        "data": _maybe_json(getattr(result, "data", None)),
        "statistical_tests": _maybe_json(getattr(result, "statistical_tests", None)),
        "interpretation": getattr(result, "interpretation", None),
        "key_findings": _maybe_json(getattr(result, "key_findings", None)),
        "supports_hypothesis": getattr(result, "supports_hypothesis", None),
        "p_value": getattr(result, "p_value", None),
        "effect_size": getattr(result, "effect_size", None),
        "confidence_interval": _maybe_json(getattr(result, "confidence_interval", None)),
    }


def _experiment_produced_output(experiment) -> bool:
    """Did this experiment return anything a reader could act on?

    An empty payload is not a null result. A null result has a p-value and a
    test statistic saying the effect was not detected; this asks the weaker
    question of whether ANY output came back, because a run that produced none
    was still being counted as a successful experiment.
    """
    # An error message is not output. A payload of exactly
    # `{"error": "Gene symbol-to-Ensembl mapping required (mygene) not
    # available."}` was counted as a successful experiment -- a failure dressed
    # as an answer, with no p-value and nothing to report.
    _not_output = {"error", "errors", "message", "traceback", "analysis_note"}

    for result in experiment.get("results") or []:
        if not isinstance(result, dict):
            continue
        data = result.get("data")
        if isinstance(data, dict):
            if {k: v for k, v in data.items() if k not in _not_output}:
                return True
        elif data:
            return True
        if result.get("p_value") is not None or result.get("effect_size") is not None:
            return True
        if result.get("statistical_tests") or result.get("key_findings"):
            return True
    return False


def _n_successful_experiments(experiments) -> int:
    """Count experiments that reached a success status AND produced output.

    `experiments` is the list of experiment dicts (or id-strings on the DB
    fallback). Anything not clearly completed/succeeded counts as
    not-successful, so a FAILED experiment is never miscounted as a success --
    and neither is one that "completed" having returned nothing, which is how a
    run with `data: {}` came to report "Successful Experiments: 1".
    """
    ok = {"completed", "success", "succeeded"}
    return sum(
        1 for e in experiments
        if isinstance(e, dict)
        and str(e.get("status", "")).lower() in ok
        and _experiment_produced_output(e)
    )


def run_research(
    question: Optional[str] = typer.Argument(None, help="Research question to investigate"),
    domain: Optional[str] = typer.Option(None, "--domain", "-d", help="Research domain (biology, neuroscience, materials, etc.)"),
    max_iterations: int = typer.Option(10, "--max-iterations", "-i", help="Maximum number of research iterations"),
    budget: Optional[float] = typer.Option(None, "--budget", "-b", help="Budget limit in USD"),
    data_path: Optional[Path] = typer.Option(None, "--data-path", "-D", help="Path to CSV dataset for experiments"),
    evidence_server: Optional[str] = typer.Option(
        None, "--evidence-server",
        help="AutoEvidence gateway to source data from instead of --data-path: "
             "either a shell command that spawns one on stdio, or an https:// URL "
             "for a gateway a data steward runs. A URL needs --evidence-key.",
    ),
    evidence_key: Optional[str] = typer.Option(
        # Deliberately NO envvar= here: typer would fill this parameter from
        # $AUTOEVIDENCE_API_KEY, and the mutual-exclusion check below cannot
        # tell an env-supplied value from a typed flag -- so a key exported in
        # a shell profile (as the help itself recommends) would abort every
        # plain --data-path run with an error naming a flag the user never
        # passed. The env var still works: `kosmos.evidence.client` reads
        # $AUTOEVIDENCE_API_KEY itself as the per-call fallback.
        None, "--evidence-key",
        help="API key this Kosmos presents to an https:// evidence gateway. The "
             "steward issues it; it names no role -- the gateway looks the role "
             "up server-side. Prefer $AUTOEVIDENCE_API_KEY over the flag, so it "
             "does not land in shell history.",
    ),
    evidence_dataset: Optional[str] = typer.Option(None, "--evidence-dataset", help="Dataset id to request from the evidence server"),
    evidence_config: Optional[Path] = typer.Option(
        None, "--evidence-config",
        help="YAML listing SEVERAL AutoEvidence gateways to run over at once. "
             "Every source is described to hypothesis generation together, so "
             "hypotheses can span datasets, and one experiment may open more "
             "than one of them -- but only where the join key permits it: two "
             "subject-keyed sources are never co-mounted unless the steward "
             "declared their subjects disjoint. Alternative to "
             "--evidence-server.",
    ),
    find_data: bool = typer.Option(
        False, "--find-data",
        help="This run has a question and no data: search public repositories "
             "for candidate datasets and write an evidence.yaml for one, then "
             "STOP. It does not run over what it finds -- the emitted config's "
             "`server:` line is a command this machine will execute, and reading "
             "it is the whole of admission control on found data. Needs a "
             "discovery server started with --finder-config; see `kosmos "
             "find-data --help`.",
    ),
    find_data_out: Path = typer.Option(
        Path("./found"), "--find-data-out", metavar="DIR",
        help="Where --find-data writes evidence.yaml and found_datasets.json.",
    ),
    find_data_intent: Optional[str] = typer.Option(
        None, "--find-data-intent", metavar="WORDS",
        help="The words --find-data actually searches for, when the research "
             "question is not them. Repositories match dataset NAMES and "
             "topics, so a whole question ('Which genes drive single-cell "
             "heterogeneity?') matches nothing while 'single-cell' does. "
             "Defaults to the question. Sent verbatim and recorded verbatim in "
             "the gateway's ledger; nothing here rewrites it.",
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

    # A dataset (local CSV or evidence server) can seed a data-driven run with
    # no question: the hypothesis generator is grounded in the dataset schema
    # instead. So only fall into interactive mode when there is truly nothing to
    # go on -- neither a question nor a data source.
    has_data_source = bool(data_path or evidence_server or evidence_config)

    # --- --find-data is decided BEFORE the interactive wizard ----------------
    #
    # It used to be decided inside the no-data branch far below, which is
    # guarded by `not from_interactive` -- and `from_interactive` is true for
    # `--interactive` AND for any run with no question and no data source. So
    # `kosmos run --find-data --find-data-intent "single-cell"` (no question,
    # which is the ordinary way to ask this command to go and look) dropped
    # straight into the wizard, discarded both flags with no line of output
    # saying so, and then started a full research run with no data at all --
    # paying for the plan, the hypotheses and the generated code before dying
    # inside every experiment's data-loading preamble. That blind run is the
    # exact outcome this feature exists to prevent, and silently discarding an
    # egress-bound argument is what the refusal below already refuses to do for
    # `--find-data-intent`.
    #
    # So the whole decision moves up here, above the wizard, where none of it
    # depends on how `question` was obtained. Search-and-exit needs no question;
    # it needs words to search for.
    if find_data_intent and not find_data:
        # Refused rather than ignored. It is the argument whose whole purpose is
        # to leave this machine, and silently discarding it would mean an
        # operator who believed they had narrowed their search got a run that
        # never searched at all -- with no line of output saying otherwise.
        print_error(
            "--find-data-intent gives --find-data the words to search for, and "
            "this run is not searching. Add --find-data, or drop it."
        )
        raise typer.Exit(1)
    if find_data and has_data_source:
        print_error(
            "--find-data searches for data this run does not have; it makes no "
            "sense alongside one that does. Drop it, or drop the data source."
        )
        raise typer.Exit(1)
    if find_data and interactive:
        # Refused rather than ordered, because --find-data ALWAYS ends in a
        # search and an exit and never in a run: the wizard would collect a
        # question, a budget and an iteration count, and this command would then
        # print a config and stop without using any of them.
        print_error(
            "--interactive collects the settings for a research run; --find-data "
            "never starts one -- it searches, writes a config for you to read, "
            "and stops. Run `kosmos run --find-data --find-data-intent \"<a few "
            "words>\"` on its own, then --interactive once you have data."
        )
        raise typer.Exit(1)
    if find_data:
        from kosmos.cli.commands.find_data import search_and_report

        # The intent and the question are NOT the same text, and defaulting one
        # to the other is the honest default rather than a good one.
        # `find_datasets` sends the intent verbatim, and a repository matches it
        # against dataset names and topics -- so a research question, which may
        # be all this command has, reliably matches nothing. Verified against
        # the live Hub: "single-cell" returns candidates and "Which genes drive
        # single-cell transcriptional heterogeneity?" returns zero.
        #
        # What this must not do is derive a query from the question. That text
        # leaves the machine and is written verbatim into the gateway's ledger
        # as what this principal searched for; a phrase this code (or worse, a
        # model) invented would make that row attribute words to an operator who
        # never wrote them, and the ledger is the whole reason an outbound
        # free-text channel is defensible here. So the operator supplies them,
        # or the question is used unchanged and the empty result explains itself.
        intent = (find_data_intent or question or "").strip()
        if not intent:
            # Reachable now that this runs before the wizard: `kosmos run
            # --find-data` with neither a question nor an intent has nothing to
            # send. Refused rather than searched for the empty string, which
            # would report "no candidates" for a query nobody wrote.
            print_error(
                "--find-data has nothing to search for. Give the words directly "
                '(--find-data-intent "single-cell CRISPR screen"), or give the '
                "research question as the argument and they will be taken from "
                "it -- though a whole question usually matches nothing."
            )
            raise typer.Exit(1)
        # It searches, writes an evidence.yaml for the chosen candidate, prints
        # the two verification commands and the exact `kosmos run` line -- and
        # then stops. It does NOT continue into a run over what it found. The
        # emitted `server:` line is a command this machine will execute, naming
        # a source AutoEvidence will fetch and classify; reading it is the
        # entire admission control on found data, and a config that is run the
        # instant it is written is a config nobody reads. The emitted
        # `subject_key` placeholder is the same argument: an undecided value
        # must not be silently decided by proceeding.
        raise typer.Exit(
            search_and_report(
                intent,
                emit=find_data_out,
                as_json=False,
                research_question=question,
                retry_flag="--find-data-intent",
            )
        )

    # Whether the question below came from the interactive wizard rather than
    # the command line. The no-data refusal further down must not fire for it:
    # the wizard is its own long-standing path, it asks no data question, and
    # aborting a session the operator just finished filling in -- to tell them
    # to run a different command -- would be a worse experience than the blind
    # run it prevents. That path is left exactly as it was.
    from_interactive = False
    if interactive or (not question and not has_data_source):
        from_interactive = True
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

    # Data-driven mode: a data source with no question. Seed a generic goal so
    # the pipeline has something to carry, and flag the run as data-driven: once
    # the director materialises the data it derives a concrete research question
    # from the actual variables (see _autogenerate_research_question) and grounds
    # its hypotheses in the data, not merely the column names.
    data_driven = bool(not question and has_data_source)
    if data_driven:
        question = (
            "Data-driven exploration: from the variables in the provided dataset, "
            "generate and test hypotheses about the relationships among them."
        )
        print_info(
            "No question given — Kosmos will formulate its own research question "
            "and hypotheses from the dataset."
        )

    # Validate inputs
    if not question:
        print_error(
            "No research question provided. Give a question, use --interactive, "
            "or supply a data source (--data-path / --evidence-server / --evidence-config)."
        )
        raise typer.Exit(1)

    # Data source: exactly one of a local CSV or an evidence server.
    sources_given = [
        name for name, given in (
            ("--data-path", bool(data_path)),
            ("--evidence-server", bool(evidence_server)),
            ("--evidence-config", bool(evidence_config)),
        ) if given
    ]
    if len(sources_given) > 1:
        print_error(
            f"Use exactly one data source; got {' and '.join(sources_given)}. "
            f"--evidence-config is the multi-gateway form of --evidence-server."
        )
        raise typer.Exit(1)
    if evidence_config and not evidence_config.exists():
        # A config path that does not exist yet is a request to find data, not
        # a mistake to refuse. The alternative was two commands where the first
        # wrote the file the second read -- correct, and it meant a question
        # about data you do not have could not be asked in one line.
        #
        # What is given up is the read of the emitted `server:` line before it
        # runs. That review still exists as `kosmos find-data --emit`; this path
        # is the deliberate other choice, and it says loudly what it chose.
        from kosmos.cli.commands.find_data import auto_emit_config

        found = auto_emit_config(
            question=question or "",
            out_dir=evidence_config.parent,
            intent=find_data_intent,
        )
        if found is None:
            print_error(
                f"Evidence config not found and no fetchable dataset was found "
                f"for this question: {evidence_config}",
                title="No data",
            )
            raise typer.Exit(1)
        evidence_config = found
    if data_path and not data_path.exists():
        print_error(f"Data file not found: {data_path}")
        raise typer.Exit(1)
    # An https:// gateway is one a steward runs, so it authenticates its callers.
    # Refuse here rather than letting the client discover it: a 401 mid-run reads
    # as a transport failure, and the remedy (ask the steward for a key) is not
    # something the operator can guess from one.
    if evidence_key and not evidence_server:
        print_error("--evidence-key applies only with --evidence-server.")
        raise typer.Exit(1)
    if evidence_server and evidence_server.lower().startswith(("http://", "https://")):
        # The env var is consulted here (not via typer's envvar= fill) so the
        # early fail-closed check still sees an exported key while an exported
        # key alone can never trip the mutual-exclusion error above.
        if not evidence_key and not os.environ.get("AUTOEVIDENCE_API_KEY"):
            print_error(
                "An https:// evidence gateway needs an API key. Pass "
                "--evidence-key or set $AUTOEVIDENCE_API_KEY. The data steward "
                "mints one with `autoevidence mint-key`."
            )
            raise typer.Exit(1)
        if evidence_server.lower().startswith("http://"):
            print_warning(
                "The evidence gateway URL is plain http://. The API key is the "
                "entire authorization and will cross the network in cleartext."
            )
    # --evidence-dataset is optional. An evidence gateway is bound to a single
    # policy, so the connection already fixes which dataset is in play; a
    # discovery server is asked directly via list_datasets. Name one only to
    # disambiguate a server that offers several.

    # --- a question with no data -------------------------------------------
    #
    # This branch is a deliberate behaviour change on a path whose only two
    # outcomes were bad. A run with a question and no data source pays for the
    # plan, the hypotheses and the generated code, and then every experiment
    # dies inside its own data-loading preamble ("No dataset available at
    # data_path") -- or, worse, the executor's repair loop was handed that
    # RuntimeError with a prompt that said only "fix this code", and a
    # sufficiently obliging model fixes it by inventing the data. (That second
    # path is closed separately, in `execution/executor.py`.) Neither outcome is
    # a capability worth preserving compatibility with, so this refuses.
    #
    # Refuses, rather than warns: a warning on a run that is going to fail
    # forty minutes and several dollars later is a warning nobody acts on.
    if not has_data_source and not data_driven and not from_interactive:
        print_error(
            f"No data source, so there is nothing to run this question against.\n\n"
            f"A question with no data runs the whole loop and then fails inside "
            f"every generated experiment, after paying for the plan, the "
            f"hypotheses and the code.\n\n"
            f"Find candidate data first:\n"
            f'    kosmos run "{question}" --find-data\n'
            f"      (add --find-data-intent \"<a few words>\" -- repositories "
            f"match dataset\n"
            f"       names, so the question itself usually matches nothing)\n"
            f"or, equivalently, on its own:\n"
            f'    kosmos find-data "<a few words>" --emit ./found\n\n'
            f"then read the config it writes and run with it:\n"
            f'    kosmos run "{question}" --evidence-config ./found/evidence.yaml\n\n'
            f"Or supply data directly: --data-path / --evidence-server / "
            f"--evidence-config.",
            title="No data",
        )
        raise typer.Exit(1)

    # --- provenance: was this dataset FOUND, or supplied? -------------------
    #
    # `kosmos find-data --emit` writes `found_datasets.json` beside the config
    # it generates, holding the verbatim intent that left this machine and every
    # candidate that came back. When the config handed to this run is one of
    # those, the run gets to SAY so -- in the banner, and in `flat_config`, which
    # is what the agents and the run artifacts carry. Without this the two cases
    # are indistinguishable downstream, and a dataset a search suggested would
    # be reported exactly like one a steward supplied.
    #
    # Silent and non-fatal for every ordinary hand-written config, which is the
    # common case: `read_provenance` returns None for anything it cannot read,
    # cannot parse, or that names a different config file.
    found_provenance = None
    if evidence_config:
        try:
            from kosmos.datasearch.emit import provenance_summary, read_provenance

            record = read_provenance(evidence_config)
            if record is not None:
                found_provenance = provenance_summary(record)
        except Exception as e:  # noqa: BLE001 -- provenance is said, not relied on
            logger.debug("Could not read found-data provenance: %s", e)
    if found_provenance:
        # Every value below came from a repository's API by way of the search
        # capsule, which is to say from strangers, so each is `escape`d before it
        # reaches a Rich console -- exactly as `find_data._print_capsule` does,
        # and for the reason stated there: an accession containing
        # `[/muted][bold green]VERIFIED PUBLIC[/bold green]` would otherwise
        # render as this panel's own prose, inside the one panel whose job is to
        # say the dataset is unverified. An unmatched closing tag is worse than
        # cosmetic: `[/nope]` raises `MarkupError` out of `console.print`, and
        # this print sits outside the try/except above, so it would kill the run
        # with a traceback before it started. The literal `[bold]`/`[muted]`
        # markup this module wrote itself stays unescaped.
        def _p(field: str, fallback: str = "") -> str:
            value = found_provenance.get(field)
            return escape(str(value)) if value else fallback

        # The gateway's own sentence about its own connectors, when it has one.
        # `provenance_summary` carries `access_note` specifically so this panel
        # can show it, and the panel did not -- so an operator running a config
        # emitted for a gated candidate got a confident provenance panel, then a
        # `HF_TOKEN is empty` refusal from `sources/hf.py` at fetch time, with
        # nothing anywhere saying it had been knowable at search time.
        # `find-data` warns when the config is WRITTEN; a config is meant to be
        # read and run later, by a different person, which is this moment.
        barred = ""
        if found_provenance.get("access_note"):
            barred = (
                f"\n[warning]The gateway that found this said it cannot fetch "
                f"it:[/warning] {_p('access_note')}\n"
            )
        console.print()
        console.print(
            Panel(
                f"This dataset was [bold]found by a search[/bold], not supplied.\n\n"
                f"**Searched for:** {_p('intent')}\n"
                f"**Repository:** {_p('repository', '?')} "
                f"(via {_p('tool', '?')} at "
                f"{_p('endpoint', 'an unreported endpoint')})\n"
                f"**Accession:** {_p('accession', '?')}\n"
                f"**Reference:** {_p('reference', '--')}\n"
                f"**Landing page:** {_p('landing_url', '--')}\n"
                f"{barred}\n"
                f"[muted]Nothing in that search verified this dataset. Whether it "
                f"is public, and what it actually contains, AutoEvidence decides "
                f"for itself when it fetches.[/muted]",
                title="[yellow]Found data[/yellow]",
                border_style="yellow",
            )
        )

    # Show starting message
    console.print()
    console.print(
        Panel(
            f"[cyan]Starting autonomous research...[/cyan]\n\n"
            f"**Question:** {question}\n"
            f"**Domain:** {domain or 'auto-detect'}\n"
            f"**Max Iterations:** {max_iterations}\n"
            # Parenthesised so ONLY the budget line is conditional. Without the
            # parens the adjacent string literals concatenate first and the
            # ternary chose between the whole block and the bare "No limit"
            # string -- so a run without --budget (every data-driven run) showed
            # only the budget line and dropped Question/Domain/Max-Iterations.
            + (f"**Budget:** ${budget} USD" if budget else "**Budget:** No limit"),
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

        # The key goes into the ENVIRONMENT, not into flat_config. flat_config is
        # handed to every agent and is dumped into run artifacts and reports; a
        # credential in it would be written to disk in half a dozen places.
        # `kosmos.evidence.client` reads $AUTOEVIDENCE_API_KEY as its fallback,
        # so the director needs no change and no agent ever holds the secret.
        if evidence_key:
            os.environ["AUTOEVIDENCE_API_KEY"] = evidence_key

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

            # Dataset path
            "data_path": str(data_path.resolve()) if data_path else None,
            # Evidence-gateway source (LECP): materialised into data_path by the director
            "evidence_server": evidence_server,
            "evidence_dataset": evidence_dataset,
            "evidence_config": str(evidence_config) if evidence_config else None,
            # Where this dataset came from, when it came from a search rather
            # than from an operator: the intent, the repository, the tool, the
            # endpoint and the accession. None for every ordinary run, so
            # nothing downstream has to change to ignore it. It goes in
            # flat_config rather than the environment precisely BECAUSE
            # flat_config is dumped into run artifacts and reports -- the
            # opposite of the reasoning that keeps the API key out of it. A
            # credential must not be written down; a provenance must.
            "found_data_provenance": found_provenance,
            # No question was given: the director derives one from the data.
            "data_driven": data_driven,

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

        # Be honest about the outcome: a run whose every experiment failed has
        # not "completed successfully", even if the director converged. Reserve
        # the success banner for runs that actually produced a result.
        experiments = results.get("experiments", [])
        pending = results.get("pending_experiments", []) or []
        n_exp = len(experiments)
        n_ok = _n_successful_experiments(experiments)
        if n_exp == 0:
            # A run that executed nothing has not completed successfully,
            # whatever state the director converged to. Say which of the two
            # things happened, because the remedies differ: more iterations for
            # a truncated run, a look at the design step for an empty one.
            if pending:
                print_error(
                    f"Research finished without executing any experiment: "
                    f"{len(pending)} were designed and still queued when the "
                    f"run ended (iteration {results.get('current_iteration', '?')}"
                    f"/{max_iterations}). Re-run with a larger --max-iterations.",
                    title="Nothing executed",
                )
            else:
                print_error(
                    "Research finished without designing or executing any "
                    "experiment; no results were produced.",
                    title="Nothing executed",
                )
        elif n_exp > 0 and n_ok == 0:
            print_error(
                f"Research finished, but all {n_exp} experiment(s) failed — no "
                f"results were produced (see the Experiments table above).",
                title="Completed with failures",
            )
        else:
            if 0 < n_ok < n_exp:
                console.print(
                    f"[warning]Research completed: {n_ok}/{n_exp} experiments "
                    f"produced a result.[/warning]"
                )
            print_success("Research completed successfully!", title="Complete")

    except KeyboardInterrupt:
        console.print("\n[warning]Research interrupted by user[/warning]")
        raise typer.Exit(130)

    except Exception as e:
        # Some exceptions stringify to "" (e.g. a bare raise), which produced an
        # unhelpful "Research failed:" with no clue. Fall back to the type name,
        # and always log the full traceback so a run is diagnosable without
        # needing --debug.
        msg = str(e).strip() or f"{type(e).__name__} (no message)"
        logger.exception("Research failed")
        # The logger does not always reach the console; write the full traceback
        # to a fixed file so a failure is always diagnosable without --debug.
        import traceback
        tb_path = None
        try:
            tb_path = "/tmp/kosmos_last_traceback.txt"
            with open(tb_path, "w") as fh:
                fh.write(traceback.format_exc())
        except OSError:
            tb_path = None
        detail = f"\n(full traceback: {tb_path})" if tb_path else ""
        print_error(f"Research failed: {msg}{detail}", title="Error")
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
            # Consecutive failures of a single research step. Reset on any
            # success, so isolated failures never accumulate into a false stop.
            consecutive_step_failures = 0
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

                # Execute next research step (async).
                #
                # Isolated per step. An exception escaping ONE step used to
                # propagate out of this loop and past the results/report
                # assembly below, discarding every prior step's work -- even
                # though the hypotheses, experiments, p-values and effect sizes
                # were already committed to the database. Losing a step is
                # tolerable; losing the run's entire output because step N
                # failed is not.
                logger.debug("Executing next research step")
                try:
                    await director.execute({"action": "step"})
                    consecutive_step_failures = 0
                except Exception as step_error:
                    consecutive_step_failures += 1
                    logger.error(
                        "Research step failed (%d consecutive): %s: %s",
                        consecutive_step_failures,
                        type(step_error).__name__,
                        step_error or "(no message)",
                        exc_info=True,
                    )
                    if consecutive_step_failures >= 3:
                        logger.error(
                            "Three consecutive step failures; stopping the loop "
                            "and reporting whatever completed."
                        )
                        break
                    continue

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

            # Build results from actual research
            # Fetch actual hypothesis and experiment objects from database
            from kosmos.db import get_session
            from kosmos.db.operations import (
                get_experiment,
                get_hypothesis,
                get_results_for_experiment,
            )

            hypotheses_data = []
            experiments_data = []
            pending_data = []

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
                                    exp_dict = experiment.to_dict() if hasattr(experiment, 'to_dict') else {"id": str(experiment)}
                                    # Attach the experiment's RESULTS. They live
                                    # in a separate table keyed by experiment_id,
                                    # so an experiment dict on its own carries
                                    # only status and duration -- which is why a
                                    # finished run could report "completed, 2.0s"
                                    # and nothing about what was found, while the
                                    # p-values and effect sizes sat in the
                                    # database unread.
                                    try:
                                        # Serialise explicitly: the Result ORM
                                        # model has NO to_dict(), unlike
                                        # Experiment/Hypothesis. Falling back to
                                        # str(r) here produced "<Result object
                                        # at 0x...>", which the renderer then
                                        # skipped as a non-dict -- so findings
                                        # silently vanished from every report
                                        # even though the rows were in the DB.
                                        exp_dict["results"] = [
                                            _serialize_result(r)
                                            for r in get_results_for_experiment(session, e_id)
                                        ]
                                    except Exception as e:
                                        # A missing result must not cost us the
                                        # experiment record itself.
                                        logger.warning(f"Could not fetch results for experiment {e_id}: {e}")
                                        exp_dict["results"] = []
                                    experiments_data.append(exp_dict)

                        # Experiments that were DESIGNED but never started.
                        # Only `completed_experiments` was ever read, so a run
                        # that ran out of iterations mid-queue exported an empty
                        # Experiments section -- indistinguishable from a run
                        # that designed nothing, and silently dropping the work
                        # the run actually did.
                        # Experiments that RAN and failed. They are dropped
                        # from the queue on quarantine and never reach
                        # `completed_experiments`, so reading only those two
                        # lists reported a run whose experiment died in the
                        # sandbox as one that designed nothing at all.
                        for e_id in getattr(director.research_plan, "failed_experiments", []) or []:
                            experiment = get_experiment(session, e_id)
                            if experiment:
                                exp_dict = (
                                    experiment.to_dict()
                                    if hasattr(experiment, "to_dict")
                                    else {"id": str(experiment)}
                                )
                                exp_dict.setdefault("results", [])
                                experiments_data.append(exp_dict)

                        for e_id in getattr(director.research_plan, "experiment_queue", []) or []:
                            experiment = get_experiment(session, e_id)
                            if experiment:
                                pending_data.append(
                                    experiment.to_dict()
                                    if hasattr(experiment, "to_dict")
                                    else {"id": str(experiment)}
                                )
            except Exception as e:
                logger.warning(f"Could not fetch all objects from database: {e}")
                # Fallback: use IDs as strings
                hypotheses_data = list(director.research_plan.hypothesis_pool)
                experiments_data = list(director.research_plan.completed_experiments)

            results = {
                "id": f"research_{int(time.time())}",
                # Read the question back off the director, not the local `question`
                # variable. On a data-driven run the director REPLACES the seeded
                # placeholder with a concrete question derived from the data
                # (see ResearchDirectorAgent._autogenerate_research_question), so
                # the local here is stale by this point and the report would be
                # titled with boilerplate while the hypotheses underneath it
                # answer something else entirely. Falls back to the local for a
                # normal run, where the two are the same string anyway.
                "question": getattr(director, "research_question", None) or question,
                "domain": final_status.get("domain", "auto"),
                "state": final_status.get("workflow_state", "COMPLETED"),
                "current_iteration": final_status.get("iteration", 0),
                "max_iterations": max_iterations,
                "has_converged": final_status.get("has_converged", False),
                "convergence_reason": final_status.get("convergence_reason"),
                "hypotheses": hypotheses_data,
                "experiments": experiments_data,
                "pending_experiments": pending_data,
                "metrics": {
                    "api_calls": getattr(director.llm_client, 'total_requests', 0),
                    "cache_hits": getattr(director.llm_client, 'cache_hits', 0),
                    "cache_misses": getattr(director.llm_client, 'cache_misses', 0),
                    "hypotheses_generated": final_status.get("hypothesis_pool_size", 0),
                    "hypotheses_tested": final_status.get("hypotheses_tested", 0),
                    "hypotheses_supported": final_status.get("hypotheses_supported", 0),
                    "hypotheses_rejected": final_status.get("hypotheses_rejected", 0),
                    # Count from the experiments' actual terminal status, not
                    # min(executed, results_count): a FAILED experiment can still
                    # write a Result row, which the old formula miscounted as a
                    # success (so the table showed "Successful: 1, Failed: 0" for
                    # a run whose only experiment failed). `failed_experiments`
                    # was also never set, so the viewer always printed 0.
                    "experiments_executed": len(experiments_data),
                    "successful_experiments": _n_successful_experiments(experiments_data),
                    "failed_experiments": len(experiments_data)
                    - _n_successful_experiments(experiments_data),
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
            # Many mid-run failures (deepseek emitting a protocol the strict
            # schema rejects) raise exceptions that stringify to "", which left
            # only an unhelpful "Error during research:". Name the type and log
            # the full traceback so an intermittent crash is diagnosable.
            msg = str(e).strip() or f"{type(e).__name__} (no message)"
            logger.exception("Error during research")
            console.print(f"\n[error]Error during research: {msg}[/error]")
            raise


if __name__ == "__main__":
    # Allow standalone testing
    typer.run(run_research)
