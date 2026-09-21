#!/usr/bin/env python
"""Run the whole thing from one research question.

    .venv/bin/python run.py "Can wine quality be predicted from physicochemical
    measurements, and does unlabeled wine from a different variety improve it?" \
        --domain chemistry

That single command does:

    1. the model reads the question and names datasets to download
    2. the fetcher downloads them (ToolUniverse's tools or its own connectors),
       each with a sha256, a manifest and a ledger line
    3. the label column is inferred, a primary labeled table is chosen by the
       model (more labels is not automatically better here), and the remaining
       tables become unlabeled evidence
    4. Kosmos trains two arms -- gold-only, and gold plus unlabeled rows through
       the pseudo-label objective -- and evaluates both on held-out labels
    5. a report is printed and written

Everything lands under `artifacts/runs/<slug>-<timestamp>/`:

    pipeline_log.jsonl        every decision, printed and recorded
    search_trace.json         the retrieval record (proposals, gold choice)
    plan.json                 what was fetched and the role of each table
    run/summary.md            the report
    run/training_log.jsonl    per-epoch losses and validation metrics
    run/ppi_summary.json      the machine-readable result

Changing the question is changing the first argument; every knob has a default.
"""

from __future__ import annotations

import argparse
import os
import re
import shlex
import subprocess
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PYTHON = sys.executable
DEFAULT_TOOLUNIVERSE = Path.home() / "venvs" / "tooluniverse" / "bin" / "python"


def slug(text: str, words: int = 6) -> str:
    cleaned = re.sub(r"[^a-z0-9\s-]", "", text.lower()).split()
    return "-".join(cleaned[:words]) or "run"


def _budget_text(args: argparse.Namespace) -> str:
    """What the banner says the run may spend: the driver's default when unset."""
    return str(args.max_download_mb if args.max_download_mb is not None else 6144)


def run_dir_for(out_dir: Path) -> Path:
    """Where this attempt's training artifacts go.

    A repeat of the same question re-uses `--out`, and the engine refuses to
    write over a finished run; so the second attempt lands in `run-2`, the third
    in `run-3`. The banner and `show_run` both read this, so what is printed is
    where the files actually are.
    """
    from kosmos.ppi.flow import next_output_dir

    return next_output_dir(out_dir / "run")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("question", help="the research question, in plain words")
    parser.add_argument("--domain", default="biology", help="biology, chemistry, ...")
    parser.add_argument(
        "--out", help="output directory (default: artifacts/runs/<slug>-<timestamp>)"
    )

    data = parser.add_argument_group("data")
    data.add_argument(
        "--candidate",
        action="append",
        default=[],
        help="a table to use directly; repeatable. Omit to let the model find data",
    )
    data.add_argument("--no-retrieval", action="store_true", help="use only --candidate tables")
    data.add_argument("--fetch-limit", type=int, default=3, help="datasets to download at most")
    data.add_argument("--max-bytes", type=int, help="size cap per downloaded dataset")
    data.add_argument("--intent", help="a term the retrieval step should consider")
    data.add_argument("--hint", action="append", help="a name the label might have")
    data.add_argument("--exclude-col", action="append", help="never the label; repeatable")
    data.add_argument("--feature-prefix", action="append", help="feature columns; repeatable")
    data.add_argument("--sample-id-column", help="column holding per-row identifiers")
    data.add_argument(
        "--task-type",
        default="auto",
        choices=("auto", "classification", "regression"),
        help="what kind of label to look for (default: let the model decide)",
    )
    data.add_argument(
        "--mechanical-search",
        action="store_true",
        help=(
            "also run the rule-built keyword search first (off by default: it "
            "matches single words and downloads every hit)"
        ),
    )
    data.add_argument(
        "--search-hops",
        type=int,
        default=3,
        help="how many look-ups the model may make before naming datasets",
    )
    data.add_argument(
        "--supp-limit",
        type=int,
        default=3,
        help="how many supplementary tables the evidence round may download",
    )
    data.add_argument(
        "--fetch-retries",
        type=int,
        default=1,
        help="how many times the model may correct an identifier a repository refused",
    )
    data.add_argument(
        "--max-files-per-fetch",
        type=int,
        default=8,
        help="how many files one fetch may contribute as candidates",
    )
    data.add_argument(
        "--max-download-mb",
        type=float,
        default=None,
        help=(
            "how many megabytes the whole run may download, across all rounds "
            "(default: 6144). 0 means no run-level limit"
        ),
    )
    data.add_argument(
        "--archive-members",
        type=int,
        default=None,
        help=(
            "how many files to unpack from one archive (default: 12; a GEO "
            "RAW.tar holds one triplet per sample). 0 unpacks everything"
        ),
    )
    data.add_argument(
        "--refresh",
        action="store_true",
        help="download again even when matching bytes are already staged",
    )
    data.add_argument(
        "--min-shared-features",
        type=int,
        default=None,
        help=(
            "how many of the gold's measured columns an evidence table must share "
            "to be used (default: 2)"
        ),
    )
    data.add_argument(
        "--min-shared-fraction",
        type=float,
        default=None,
        help="and the share of the gold's columns that has to be shared (default: 0.5)",
    )
    data.add_argument("--test-path", help="fixed labeled test table (default: random split)")

    loss = parser.add_argument_group("loss")
    loss.add_argument(
        "--loss",
        default="plain",
        choices=("plain", "signed", "gold-plus-synthetic", "gradient-gated"),
        help=(
            "plain/signed are the PPI objectives; gold-plus-synthetic is the "
            "naive L_G + lambda*L_S baseline; gradient-gated admits the synthetic "
            "gradient only where it agrees with the gold one"
        ),
    )
    loss.add_argument("--lambda", dest="ppi_lambda", type=float, default=0.5)
    loss.add_argument("--ramp-epochs", type=int, default=3)
    loss.add_argument(
        "--gate-kappa",
        type=float,
        default=1.0,
        help="gradient-gated: most the accepted synthetic gradient may weigh "
        "relative to gold (1.0 = never louder)",
    )
    loss.add_argument(
        "--gate-scope",
        default="batch",
        choices=("batch", "sample"),
        help="gradient-gated: gate one synthetic gradient per step, or each row",
    )
    loss.add_argument("--gate-gamma", type=float, default=1.0)
    loss.add_argument("--gate-lambda", type=float, default=1.0)
    loss.add_argument(
        "--pseudo-mode", default="cross_fit", choices=("cross_fit", "in_sample", "pretrained")
    )
    loss.add_argument("--cross-fit-folds", type=int, default=3)
    loss.add_argument("--stage1-epochs", type=int, default=4, help="signed loss only")
    loss.add_argument("--stage2-epochs", type=int, default=1, help="signed loss only")

    training = parser.add_argument_group("training")
    training.add_argument("--max-epochs", type=int, default=20)
    training.add_argument("--patience", type=int, default=5)
    training.add_argument("--external-per-source", type=int, default=5000)
    training.add_argument("--max-external-samples", type=int, default=20000)
    training.add_argument("--seed", type=int, default=42)
    training.add_argument(
        "--model-design",
        default="deepseek",
        choices=("deepseek", "linear"),
        help="let the model design the architecture, or use a linear head",
    )
    training.add_argument("--budget", type=float, default=3.0, help="LLM spend cap in USD")
    training.add_argument("--max-iterations", type=int, default=1)

    parser.add_argument("--plan-only", action="store_true", help="stop after planning")
    parser.add_argument(
        "--tooluniverse-python",
        default=os.environ.get("KOSMOS_TOOLUNIVERSE_PYTHON")
        or (str(DEFAULT_TOOLUNIVERSE) if DEFAULT_TOOLUNIVERSE.exists() else None),
        help="interpreter with tooluniverse installed (the download layer)",
    )
    parser.add_argument("--dry-run", action="store_true", help="print the command, do nothing")
    return parser.parse_args(argv)


def environment(args: argparse.Namespace, out_dir: Path) -> dict[str, str]:
    """The environment the run needs, assembled from the flags."""
    env = dict(os.environ)
    # The training run needs a directory of its own: `run_experiment` refuses to
    # write over a finished run, and re-running the same `--out` is how a
    # question gets repeated. `run-2`, `run-3`, ... instead of failing inside the
    # research loop, three experiment-design calls later.
    env["PPI_OUTPUT_DIR"] = str(run_dir_for(out_dir))
    env.update(
        {
            # What the reference runs used: no world model, no redis, no sandbox,
            # one hypothesis, no literature round trips.
            "WORLD_MODEL_ENABLED": "false",
            "REDIS_ENABLED": "false",
            "ENABLE_SANDBOXING": "false",
            "USE_LITERATURE_CONTEXT": "false",
            "REQUIRE_NOVELTY_CHECK": "false",
            "NUM_HYPOTHESES": "1",
            "MPLCONFIGDIR": "/tmp/kosmos-mpl",
            "OMP_NUM_THREADS": "4",
            "MKL_NUM_THREADS": "4",
            "PPI_LOSS_MODE": args.loss.replace("-", "_"),
            "PPI_PSEUDO_TARGETS": "soft" if args.loss == "plain" else "hard",
            "PPI_LOSS_RAMP_EPOCHS": str(args.ramp_epochs),
            "PPI_LAMBDA": str(args.ppi_lambda),
            "PPI_GATE_KAPPA": str(args.gate_kappa),
            "PPI_GATE_SCOPE": args.gate_scope,
            "PPI_GATE_GAMMA": str(args.gate_gamma),
            "PPI_GATE_LAMBDA": str(args.gate_lambda),
            "PPI_PSEUDO_MODE": args.pseudo_mode,
            "PPI_CROSS_FIT_FOLDS": str(args.cross_fit_folds),
            "PPI_STAGE1_EPOCHS": str(args.stage1_epochs),
            "PPI_STAGE2_EPOCHS": str(args.stage2_epochs),
            "PPI_MAX_EPOCHS": str(args.max_epochs),
            "PPI_PATIENCE": str(args.patience),
            "PPI_EXTERNAL_PER_DONOR": str(args.external_per_source),
            "PPI_MAX_EXTERNAL_SAMPLES": str(args.max_external_samples),
            "PPI_SEED": str(args.seed),
            "PPI_MODEL_DESIGN": args.model_design,
        }
    )
    if args.tooluniverse_python:
        env["KOSMOS_TOOLUNIVERSE_PYTHON"] = args.tooluniverse_python
    return env


def build_command(args: argparse.Namespace, plan_path: Path) -> list[str]:
    """The `auto_task_run` command line this run is: one flag, one place."""
    command = [
        PYTHON,
        str(ROOT / "scripts" / "auto_task_run.py"),
        "--objective",
        args.question,
        "--domain",
        args.domain,
        "--fetch-limit",
        str(args.fetch_limit),
        "--plan-out",
        str(plan_path),
    ]
    for value in args.candidate:
        command += ["--candidate", value]
    for flag, values in (
        ("--hint", args.hint),
        ("--exclude-col", args.exclude_col),
        ("--feature-prefix", args.feature_prefix),
    ):
        for value in values or []:
            command += [flag, value]
    for flag, value in (
        ("--sample-id-column", args.sample_id_column),
        ("--task-type", args.task_type),
        ("--test-path", args.test_path),
        ("--intent", args.intent),
        ("--max-bytes", args.max_bytes),
        ("--budget", args.budget),
        ("--max-iterations", args.max_iterations),
    ):
        if value is not None:
            command += [flag, str(value)]
    if not args.plan_only:
        command += ["--yes", "--run"]
    if args.mechanical_search:
        command += ["--mechanical-search"]
    if args.search_hops is not None:
        command += ["--search-hops", str(args.search_hops)]
    if args.supp_limit is not None:
        command += ["--supp-limit", str(args.supp_limit)]
    if args.fetch_retries is not None:
        command += ["--fetch-retries", str(args.fetch_retries)]
    if args.max_files_per_fetch is not None:
        command += ["--max-files-per-fetch", str(args.max_files_per_fetch)]
    if args.max_download_mb is not None:
        command += ["--max-download-mb", str(args.max_download_mb)]
    if args.archive_members is not None:
        command += ["--archive-members", str(args.archive_members)]
    if args.refresh:
        # The fetcher reads this from the environment, so one flag here reaches
        # every download the run's rounds make.
        command += ["--refresh"]
    if args.min_shared_features is not None:
        command += ["--min-shared-features", str(args.min_shared_features)]
    if args.min_shared_fraction is not None:
        command += ["--min-shared-fraction", str(args.min_shared_fraction)]
    return command


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.no_retrieval and not args.candidate:
        print("error: --no-retrieval needs at least one --candidate table", file=sys.stderr)
        return 2

    stamp = datetime.now().strftime("%Y%m%d-%H%M")
    out_dir = (
        Path(args.out)
        if args.out
        else ROOT / "artifacts" / "runs" / f"{slug(args.question)}-{stamp}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    plan_path = out_dir / "plan.json"

    print("=" * 78)
    print(f"question : {args.question}")
    print(f"domain   : {args.domain}")
    print(
        f"loss     : {args.loss} (lambda={args.ppi_lambda}, ramp={args.ramp_epochs})"
        + (
            f"  gate: kappa={args.gate_kappa}, scope={args.gate_scope}, "
            f"gamma={args.gate_gamma}, lambda={args.gate_lambda}"
            if args.loss == "gradient-gated"
            else ""
        )
    )
    print(
        f"model    : design={args.model_design}, epochs<={args.max_epochs}, "
        f"patience={args.patience}"
    )
    print(
        f"data     : retrieval={'off' if args.no_retrieval else 'on'}, "
        f"fetch_limit={args.fetch_limit}, external<={args.max_external_samples}"
    )
    print(
        f"download : budget<={_budget_text(args)} MB for the run, "
        f"reuse={'off' if args.refresh else 'on'}, "
        f"archive_members={args.archive_members if args.archive_members is not None else 12}"
    )
    print(f"output   : {out_dir}")
    print("=" * 78)

    command = build_command(args, plan_path)
    if args.dry_run:
        print("# would run:\n#   " + " ".join(shlex.quote(part) for part in command))
        return 0

    # Built once: the run's directory is decided here, before anything can create
    # it, and printed at the end from the same value.
    env = environment(args, out_dir)
    run_dir = Path(env["PPI_OUTPUT_DIR"])
    if run_dir.name != "run":
        print(f"# training output: {run_dir} (that directory was already taken)")
    completed = subprocess.call(command, cwd=ROOT, env=env)
    if completed != 0:
        print(f"\n# pipeline stopped with exit code {completed}", file=sys.stderr)
        print(f"# records: {out_dir / 'pipeline_log.jsonl'}", file=sys.stderr)
        return completed

    if args.plan_only:
        print(f"\n# plan ready: {plan_path}")
        print(f"# records    : {out_dir / 'pipeline_log.jsonl'}")
        return 0

    print()
    subprocess.call([PYTHON, str(ROOT / "scripts" / "show_run.py"), str(run_dir)])
    print("=" * 78)
    print("records:")
    for path in (
        out_dir / "pipeline_log.jsonl",
        out_dir / "search_trace.json",
        plan_path,
        run_dir / "summary.md",
        run_dir / "training_log.jsonl",
        run_dir / "ppi_summary.json",
        ROOT / "data" / "fetched" / "ledger.jsonl",
    ):
        print(f"  {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
