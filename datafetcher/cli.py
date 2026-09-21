"""`python -m datafetcher ...` -- search, fetch, list, ledger.

Machine-readable with `--json` on every subcommand, so a Kosmos-side script can
consume it without parsing prose. Exit codes: 0 success, 2 refused or failed.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import __version__
from .config import DataFetcherConfig
from .errors import DataFetcherError
from .fetch import fetch as run_fetch
from .plan import (
    PlanTask,
    build_plan,
    data_files_under,
    load_plan,
    write_plan,
)
from .profile import TaskShape, classify_path, plan_roles, profile_table
from .store import list_records, read_ledger


def _config(args: argparse.Namespace) -> DataFetcherConfig:
    overrides: dict = {}
    if getattr(args, "root", None):
        overrides["root"] = Path(args.root)
    if getattr(args, "max_bytes", None):
        overrides["max_bytes"] = int(args.max_bytes)
    if getattr(args, "timeout", None):
        overrides["timeout_s"] = float(args.timeout)
    if getattr(args, "offline", False):
        overrides["offline"] = True
    if getattr(args, "archive_members", None):
        overrides["archive_max_members"] = int(args.archive_members)
    return DataFetcherConfig.from_env(**overrides)


def _emit(payload, as_json: bool, human: str) -> None:
    print(json.dumps(payload, indent=2, default=str) if as_json else human)


def _cmd_list_files(args: argparse.Namespace) -> int:
    """What a reference contains, without downloading it."""
    from .fetch import list_reference_files

    payload = list_reference_files(args.reference, config=_config(args))
    lines = [f"{payload['reference']} ({payload['scheme']})"]
    if payload.get("about"):
        lines.append(f"  about: {payload['about']}")
    for record in payload["files"]:
        size = f"{_human(record['bytes'])}" if record.get("bytes") else ""
        lines.append(f"  {record['path']}  {size}")
    if payload.get("bytes"):
        lines.append(f"  total: {_human(payload['bytes'])}")
    if not payload["files"]:
        lines.append(f"  no listing: {payload.get('note', 'nothing to list')}")
    _emit(payload, args.json, "\n".join(lines))
    return 0


def _human(size: int) -> str:
    from .sources.geo import format_size

    return format_size(size)


def _cmd_fetch(args: argparse.Namespace) -> int:
    result = run_fetch(
        args.reference,
        config=_config(args),
        query=args.query,
        to_csv=args.to_csv,
        max_features=args.max_features,
        profile=not args.no_profile,
        reuse=not args.refresh,
    )
    lines = [f"fetched {result.reference}", f"  directory: {result.directory}"]
    for record in result.files:
        origin = f" (derived from {record.derived_from})" if record.derived_from else ""
        lines.append(f"  {record.path}  {record.bytes:,} bytes  sha256={record.sha256[:16]}...{origin}")
    if result.revision:
        lines.append(f"  revision: {result.revision}")
    for note in result.notes:
        lines.append(f"  note: {note}")
    handoff = result.handoff
    if handoff:
        lines.append(
            f"\nhand this to a Kosmos run with:  --data-path "
            f"{Path(result.directory) / handoff.path}"
        )
    else:
        data_files = result.data_files
        if len(data_files) > 1:
            lines.append(
                f"\n{len(data_files)} data files were fetched; pick one for "
                f"--data-path / --external-data-path:"
            )
            for record in data_files:
                lines.append(f"  {Path(result.directory) / record.path}")
    _emit(result.model_dump(mode="json"), args.json, "\n".join(lines))
    return 0


def _cmd_list(args: argparse.Namespace) -> int:
    config = _config(args)
    records = list_records(config, scheme=args.scheme)
    lines = []
    for record in records:
        primary = record.primary
        if primary:
            size = f"{primary.bytes:,} bytes"
        else:
            files = record.data_files or record.files
            total = sum(f.bytes for f in files)
            size = f"{len(files)} file(s), {total:,} bytes"
        lines.append(f"  {record.reference}  {record.retrieved_at}  {size}")
    _emit(
        [r.model_dump(mode="json") for r in records],
        args.json,
        "\n".join([f"{len(records)} fetch record(s) under {config.root}", *lines]),
    )
    return 0


def _cmd_ledger(args: argparse.Namespace) -> int:
    config = _config(args)
    rows = read_ledger(config, limit=args.limit)
    lines = [json.dumps(row, sort_keys=True) for row in rows]
    _emit(rows, args.json, "\n".join(lines) or f"(no ledger at {config.ledger_path})")
    return 0


def _task_shape(args: argparse.Namespace) -> TaskShape:
    return TaskShape(
        target_column=args.target_column,
        feature_columns=tuple(args.feature_column) if args.feature_column else None,
        feature_prefixes=tuple(args.feature_prefix or ()),
        exclude_columns=tuple(args.exclude_column or ()),
        sample_id_column=args.sample_id_column,
        description=args.description or "",
    )


def _cmd_profile(args: argparse.Namespace) -> int:
    profile = profile_table(
        args.path, sample_rows=args.sample_rows, count_rows=not args.no_count
    )
    lines = [
        f"{profile.path}",
        f"  delimiter      : {profile.delimiter!r}",
        f"  size / rows    : {profile.file_bytes:,} bytes, "
        f"{profile.n_rows if profile.n_rows is not None else f'sampled {profile.sampled_rows}'} rows",
        f"  columns        : {len(profile.columns)}",
    ]
    for column in profile.columns[: args.show]:
        lines.append(
            f"    {column.name:<28} {column.kind:<9} null={column.null_fraction:.2f}"
            f" levels={column.n_unique if column.n_unique is not None else '?'}"
        )
    if len(profile.columns) > args.show:
        lines.append(f"    ... {len(profile.columns) - args.show} more")
    for note in profile.notes:
        lines.append(f"  note: {note}")
    _emit(profile.model_dump(mode="json"), args.json, "\n".join(lines))
    return 0


def _cmd_roles(args: argparse.Namespace) -> int:
    task = _task_shape(args)
    lines: list[str] = []
    if len(args.paths) == 1:
        profile, decision = classify_path(args.paths[0], task)
        lines.append(f"{profile.path}")
        lines.append(f"  role    : {decision.role}")
        lines.append(f"  reason  : {decision.reason}")
        lines.append(f"  features: {len(decision.features)}")
        payload = {
            "task": task.as_dict(),
            "profile": profile.model_dump(mode="json"),
            **decision.to_dict(),
        }
    else:
        plan = plan_roles(args.paths, task)
        for role in ("gold", "supplementary", "unusable"):
            entries = plan[role]
            lines.append(f"{role}: {len(entries)}")
            for entry in entries:
                lines.append(f"  {entry['path']}")
                lines.append(f"      {entry['reason']}")
        payload = {"task": task.as_dict(), "plan": plan}
    _emit(payload, args.json, "\n".join(lines))
    return 0


def _plan_evidence(args: argparse.Namespace) -> dict:
    """What the plan says about how these tables were found.

    `--search-trace` takes a file the caller wrote (intents, hits, which hits it
    chose and why). The fetcher does not interpret it: it is a record, and
    keeping it opaque means the fetcher needs no opinion about how a search was
    planned.
    """
    evidence: dict = {"repositories": args.repository or []}
    if getattr(args, "search_trace", None):
        try:
            evidence["search"] = json.loads(
                Path(args.search_trace).read_text(encoding="utf-8")
            )
        except Exception as e:  # noqa: BLE001 - an unreadable trace is not fatal
            evidence["search"] = {"error": f"could not read {args.search_trace}: {e}"}
    return evidence


def _load_reviews(path: str | None) -> dict:
    """The LLM's per-table review, as a proposal for the plan to check."""
    if not path:
        return {}
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except Exception as e:  # noqa: BLE001 - an unreadable review file is not fatal
        print(f"warning: could not read reviews from {path}: {e}", file=sys.stderr)
        return {}
    return payload.get("reviews", payload) if isinstance(payload, dict) else {}


def _cmd_plan(args: argparse.Namespace) -> int:
    """Write the gold/supplementary decision as a file Kosmos can read."""
    paths = list(args.paths or [])
    if args.from_root or not paths:
        default_root = Path(args.root) if args.root else Path("data/fetched")
        discovered = data_files_under(default_root)
        if not paths:
            paths = [str(p) for p in discovered]
            if discovered:
                print(f"# {len(discovered)} fetched table(s) under {default_root}", file=sys.stderr)
    if not paths:
        print(
            f"error: no tables to plan (nothing under {args.root or 'data/fetched'}); "
            f"pass paths, or fetch something first",
            file=sys.stderr,
        )
        return 2

    task = PlanTask(
        objective=args.objective or "",
        target_column=args.target_column,
        target_source=args.target_source,
        target_confidence=args.target_confidence,
        feature_columns=tuple(args.feature_column) if args.feature_column else None,
        feature_prefixes=list(args.feature_prefix or []),
        exclude_columns=list(args.exclude_column or []),
        sample_id_column=args.sample_id_column,
        task_type=args.task_type,
        evaluation_metric=args.evaluation_metric,
        test_fraction=args.test_fraction,
        validation_fraction=args.validation_fraction,
        seed=args.seed,
        multi_gold_policy=args.multi_gold,
    )
    plan = build_plan(
        paths,
        task,
        root=args.root,
        evidence=_plan_evidence(args),
        count_rows=not args.no_count,
        primary_labeled=args.labeled_path,
        reviews=_load_reviews(args.reviews),
    )
    if args.output:
        written = write_plan(plan, args.output)
        if args.json:
            print(json.dumps({"plan": str(written), **plan.model_dump(mode="json")}, indent=2, default=str))
        else:
            print(plan.summary())
            print(f"\n# written: {written}")
    else:
        _emit(plan.model_dump(mode="json"), True, plan.summary())
    return 0


def _cmd_plan_show(args: argparse.Namespace) -> int:
    plan = load_plan(args.path)
    payload = plan.model_dump(mode="json")
    _emit(payload, args.json, plan.summary())
    return 0


def _cmd_list_tools(args: argparse.Namespace) -> int:
    """The download tools the model may choose from, from the configured interpreter."""
    from .tooluniverse import (
        ToolUniverseDownloader,
        ToolUniverseUnavailable,
        format_catalog,
    )

    config = _config(args)
    try:
        downloader = ToolUniverseDownloader(
            python=config.tooluniverse_python, timeout_s=config.tooluniverse_timeout_s
        )
    except ToolUniverseUnavailable as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    catalog = downloader.catalog()
    _emit(catalog, args.json, f"# {downloader.detail}\n" + format_catalog(catalog))
    return 0


def _cmd_sample(args: argparse.Namespace) -> int:
    """Print the review packet: raw head lines plus how we parsed them."""
    from .sample import compact, sample_packet

    packet = sample_packet(args.path, rows=args.rows, max_bytes=args.max_bytes)
    _emit(packet, args.json, compact(packet, max_columns=args.columns))
    return 0


def _cmd_search_mechanical(args: argparse.Namespace) -> int:
    """Round one: ask the repositories, by rule, with no model involved."""
    from .mechanical import mechanical_search

    config = _config(args)
    result = mechanical_search(
        args.question,
        config=config,
        repositories=args.repository,
        limit=args.limit,
        query=args.query,
        emit=lambda message: print("  " + message, file=sys.stderr),
    )
    _emit(
        result.to_dict(),
        args.json,
        "\n".join(
            [f"query: {result.query!r}", *[f"  {hit.accession} -> {hit.reference}  {hit.title[:70]}"
                                            for hit in result.hits]]
        ),
    )
    return 0 if result.hits else 2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="datafetcher",
        description=(
            "Fetch public datasets for Kosmos experiments. Search returns "
            "pointers; fetch downloads them and records provenance. No gating, "
            "no policies, no access control -- this is a fetcher."
        ),
    )
    parser.add_argument("--version", action="version", version=f"datafetcher {__version__}")
    # Global options are defined twice on purpose: on this parser so they may
    # precede the subcommand, and on a parent parser the subcommands inherit so
    # they may follow it. `SUPPRESS` defaults keep the second copy from
    # overwriting a value the first one already read.
    parent = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--root", help="staging root (default: data/fetched)")
    parser.add_argument("--offline", action="store_true", help="never touch the network")
    parser.add_argument("--json", action="store_true", help="machine-readable output")
    parent.add_argument("--root", default=argparse.SUPPRESS)
    parent.add_argument("--offline", action="store_true", default=argparse.SUPPRESS)
    parent.add_argument("--json", action="store_true", default=argparse.SUPPRESS)
    sub = parser.add_subparsers(dest="command", required=True)

    def common(p: argparse.ArgumentParser) -> None:
        """Kept for call sites; global flags now come from `parent`."""

    p = sub.add_parser(
        "list-files",
        parents=[parent],
        help="what a reference contains, without downloading it",
    )
    p.add_argument("reference", help="hf://owner/name or geo://GSE123")
    p.set_defaults(func=_cmd_list_files)

    p = sub.add_parser(
        "fetch", parents=[parent], help="download a reference and record provenance"
    )
    p.add_argument("reference", help="geo://GSE123, hf://owner/name, https://..., file:///...")
    p.add_argument("--query", help="the search intent this reference came from")
    p.add_argument("--to-csv", action="store_true", help="also write samples x features CSV")
    p.add_argument("--max-features", type=int, help="cap columns when transposing")
    p.add_argument(
        "--no-profile",
        action="store_true",
        help="skip recording the table's shape in the manifest",
    )
    p.add_argument(
        "--refresh",
        action="store_true",
        help="download again even when matching bytes are already staged",
    )
    p.add_argument(
        "--archive-members",
        type=int,
        help="how many files to unpack from an archive (default 12)",
    )
    p.add_argument("--max-bytes", type=int, help="size cap for the download")
    p.add_argument("--timeout", type=float, help="per-request timeout in seconds")
    common(p)
    p.set_defaults(func=_cmd_fetch)

    p = sub.add_parser("list", parents=[parent], help="list everything already fetched")
    p.add_argument("--scheme", help="only this scheme (geo, hf, http, file)")
    common(p)
    p.set_defaults(func=_cmd_list)

    p = sub.add_parser("ledger", parents=[parent], help="show the search/fetch ledger")
    p.add_argument("-n", "--limit", type=int, default=50)
    common(p)
    p.set_defaults(func=_cmd_ledger)

    def task_options(p: argparse.ArgumentParser) -> None:
        p.add_argument("--target-column", help="the label column this task predicts")
        p.add_argument(
            "--feature-prefix",
            action="append",
            help="keep columns starting with this (repeatable)",
        )
        p.add_argument(
            "--feature-column",
            action="append",
            help="exact feature column (repeatable; overrides prefixes)",
        )
        p.add_argument(
            "--exclude-column", action="append", help="never a feature (repeatable)"
        )
        p.add_argument("--sample-id-column", help="column holding per-row identifiers")
        p.add_argument("--description", help="free-form task description")

    p = sub.add_parser(
        "profile", parents=[parent], help="describe the shape of a fetched table"
    )
    p.add_argument("path")
    p.add_argument("--sample-rows", type=int, default=500)
    p.add_argument("--no-count", action="store_true", help="do not count rows exactly")
    p.add_argument("--show", type=int, default=25, help="columns to print")
    common(p)
    p.set_defaults(func=_cmd_profile)

    p = sub.add_parser(
        "roles",
        parents=[parent],
        help="decide gold / supplementary / unusable for one or more tables",
    )
    p.add_argument("paths", nargs="+")
    task_options(p)
    common(p)
    p.set_defaults(func=_cmd_roles)

    p = sub.add_parser(
        "plan",
        parents=[parent],
        help="write the gold/supplementary decision as a plan file for Kosmos",
    )
    p.add_argument("paths", nargs="*", help="tables to plan (default: everything fetched)")
    p.add_argument(
        "--from-root",
        action="store_true",
        help="include every table staged under --root (default when no paths given)",
    )
    p.add_argument("--output", "-o", help="where to write the plan JSON")
    p.add_argument("--objective", help="the research question this plan serves")
    p.add_argument(
        "--target-source",
        default="cli",
        choices=("cli", "protocol", "objective", "llm", "human"),
        help="who decided the target column (recorded for audit)",
    )
    p.add_argument(
        "--target-confidence",
        default="declared",
        choices=("declared", "exact_match", "inferred"),
    )
    p.add_argument("--task-type", default="classification")
    p.add_argument(
        "--multi-gold",
        default="external",
        choices=("external", "pool", "error"),
        help="with several labeled tables: one gold and the rest as evidence "
        "(external, default), pool them all, or refuse (error)",
    )
    p.add_argument(
        "--labeled-path",
        help="which labeled table is the primary gold (default: the largest)",
    )
    p.add_argument(
        "--reviews",
        help="JSON file of per-table LLM reviews (role proposals); the plan checks "
        "each one mechanically before applying it",
    )
    p.add_argument("--evaluation-metric", default="balanced_accuracy")
    p.add_argument("--test-fraction", type=float, default=0.2)
    p.add_argument("--validation-fraction", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--repository", action="append", help="repository label(s) searched")
    p.add_argument(
        "--search-trace",
        help="JSON file describing how these tables were found; stored in the plan",
    )
    p.add_argument("--no-count", action="store_true", help="do not count rows exactly")
    task_options(p)
    common(p)
    p.set_defaults(func=_cmd_plan)

    p = sub.add_parser("show-plan", parents=[parent], help="print a plan file")
    p.add_argument("path")
    common(p)
    p.set_defaults(func=_cmd_plan_show)

    p = sub.add_parser(
        "list-tools",
        parents=[parent],
        help="list the ToolUniverse download tools the retrieval step may call",
    )
    common(p)
    p.set_defaults(func=_cmd_list_tools)

    p = sub.add_parser(
        "sample",
        parents=[parent],
        help="header + first rows of a table, for an LLM or a human to review",
    )
    p.add_argument("path")
    p.add_argument("--rows", type=int, default=5, help="raw lines to show")
    p.add_argument("--max-bytes", type=int, default=4096, help="cap on the raw head")
    p.add_argument("--columns", type=int, default=60, help="columns to render")
    common(p)
    p.set_defaults(func=_cmd_sample)

    p = sub.add_parser(
        "search-mechanical",
        parents=[parent],
        help="round one retrieval: repository search by rule, no model involved",
    )
    p.add_argument("question", help="the research question; the query is built from it")
    p.add_argument("--query", help="override the rule-built query")
    p.add_argument("-r", "--repository", action="append", help="limit to a repository label")
    p.add_argument("-n", "--limit", type=int, default=5)
    common(p)
    p.set_defaults(func=_cmd_search_mechanical)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    # A flag given nowhere still has to exist for `getattr` lookups downstream.
    for name, default in (("root", None), ("offline", False), ("json", False)):
        if not hasattr(args, name):
            setattr(args, name, default)
    try:
        return args.func(args)
    except DataFetcherError as e:
        print(f"error: {type(e).__name__}: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
