#!/usr/bin/env python
"""Deprecated shim. The pipeline has **two** interfaces now, both on `kosmos run`:

    # 1. WITHOUT a plan -- fetch, gate, plan and train in one call
    .venv/bin/python -m kosmos.cli.main run "<question>" \
        --domain single_cell --task per_cell --hint <label column> \
        --fetch-limit 2 --supp-limit 3 --max-epochs 3

    # 2. WITH a plan -- train the plan a previous fetch wrote
    .venv/bin/python -m kosmos.cli.main run "<question>" \
        --data-plan artifacts/runs/<name>/plan.json --yes

`run.py` used to be a third, separate entry point that drove `auto_task_run`
itself and then launched the CLI. That duplication is gone: this file now only
translates its old flags into one of the two calls above, so an existing command
keeps working while there is a single implementation (the CLI ->
`kosmos.ppi.discovery_bridge`).

The previous implementation is kept verbatim as `run.py.bak-20260928`, and the
canonical flags live in `kosmos/cli/commands/run.py`. Flags that have no
equivalent on the new interface are rejected with a message pointing at
`--data-plan` rather than silently ignored.
"""

from __future__ import annotations

import argparse
import os
import re
import shlex
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PYTHON = sys.executable

#: Flags whose behaviour now lives on `kosmos run`/the fetcher rather than here.
_UNSUPPORTED = (
    "--candidate",
    "--no-retrieval",
    "--mechanical-search",
    "--search-hops",
    "--archive-members",
    "--min-shared-features",
    "--min-shared-fraction",
    "--exclude-col",
    "--feature-prefix",
    "--sample-id-column",
    "--task-type",
)

#: Only a flag the caller actually set counts as unsupported; argparse cannot
#: tell "--x 3" from the default 3, so compare against the default instead.
_UNSUPPORTED_DEFAULTS = {
    "candidate": [],
    "no_retrieval": False,
    "mechanical_search": False,
    "search_hops": 3,
    "archive_members": None,
    "min_shared_features": None,
    "min_shared_fraction": None,
    "exclude_col": None,     # argparse append without an explicit default -> None
    "feature_prefix": None,
    "sample_id_column": None,
    "task_type": "auto",
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("question", help="the research question, in plain words")
    parser.add_argument("--domain", default="biology")
    parser.add_argument("--out", help="output directory (default: artifacts/runs/<slug>-<stamp>)")

    data = parser.add_argument_group("data")
    data.add_argument("--candidate", action="append", default=[])
    data.add_argument("--no-retrieval", action="store_true")
    data.add_argument("--fetch-limit", type=int, default=3)
    data.add_argument("--max-bytes", type=int)
    data.add_argument("--intent")
    data.add_argument("--hint", action="append")
    data.add_argument("--exclude-col", action="append")
    data.add_argument("--feature-prefix", action="append")
    data.add_argument("--sample-id-column")
    data.add_argument("--task-type", default="auto", choices=("auto", "classification", "regression"))
    data.add_argument("--mechanical-search", action="store_true")
    data.add_argument("--search-hops", type=int, default=3)
    data.add_argument("--supp-limit", type=int, default=3)
    data.add_argument("--fetch-retries", type=int, default=1)
    data.add_argument("--max-files-per-fetch", type=int, default=8)
    data.add_argument("--max-download-mb", type=float, default=None)
    data.add_argument("--archive-members", type=int, default=None)
    data.add_argument("--refresh", action="store_true")
    data.add_argument("--min-shared-features", type=int, default=None)
    data.add_argument("--min-shared-fraction", type=float, default=None)
    data.add_argument("--test-path")

    loss = parser.add_argument_group("loss")
    loss.add_argument(
        "--loss", default="plain",
        choices=("plain", "signed", "gold-plus-synthetic", "gradient-gated"),
    )
    loss.add_argument("--lambda", dest="ppi_lambda", type=float, default=0.5)
    loss.add_argument("--ramp-epochs", type=int, default=3)
    loss.add_argument("--gate-kappa", type=float, default=1.0)
    loss.add_argument("--gate-scope", default="batch", choices=("batch", "sample"))
    loss.add_argument("--gate-gamma", type=float, default=1.0)
    loss.add_argument("--gate-lambda", type=float, default=1.0)
    loss.add_argument("--single-cell", dest="single_cell_preprocess", default="auto",
                      choices=("auto", "on", "off"))
    loss.add_argument("--single-cell-top-genes", type=int, default=2000)
    loss.add_argument("--single-cell-target-sum", type=float, default=10000.0)
    loss.add_argument("--single-cell-min-cells", type=int, default=3)
    loss.add_argument("--no-single-cell-figures", dest="single_cell_figures",
                      action="store_false", default=True)
    loss.add_argument("--pseudo-mode", default="cross_fit",
                      choices=("cross_fit", "in_sample", "pretrained"))
    loss.add_argument("--cross-fit-folds", type=int, default=3)
    loss.add_argument("--stage1-epochs", type=int, default=4)
    loss.add_argument("--stage2-epochs", type=int, default=1)

    training = parser.add_argument_group("training")
    training.add_argument("--max-epochs", type=int, default=20)
    training.add_argument("--patience", type=int, default=5)
    training.add_argument("--external-per-source", type=int, default=5000)
    training.add_argument("--max-external-samples", type=int, default=20000)
    training.add_argument("--seed", type=int, default=42)
    training.add_argument("--model-design", default="deepseek", choices=("deepseek", "linear"))
    training.add_argument("--budget", type=float, default=3.0)
    training.add_argument("--max-iterations", type=int, default=1)

    parser.add_argument("--task", default="simple", choices=("simple", "perturbation"))
    perturbation = parser.add_argument_group("perturbation response")
    perturbation.add_argument("--go-graph", default=None)
    perturbation.add_argument("--split-mode", default="mixed",
                              choices=("mixed", "unseen_single", "unseen_combination"))
    perturbation.add_argument("--test-fraction", type=float, default=0.2)
    perturbation.add_argument("--validation-fraction", type=float, default=0.1)
    perturbation.add_argument("--min-cells-per-perturbation", type=int, default=2)
    perturbation.add_argument("--go-k", type=int, default=20)
    perturbation.add_argument("--coexpress-threshold", type=float, default=0.4)
    perturbation.add_argument("--coexpress-k", type=int, default=20)
    perturbation.add_argument("--eta", type=float, default=1.0)
    perturbation.add_argument("--gold-table", default=None)
    perturbation.add_argument("--supp-table", action="append", default=[])

    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--tooluniverse-python", default=os.environ.get("KOSMOS_TOOLUNIVERSE_PYTHON"))
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def environment(args: argparse.Namespace, out_dir: Path) -> dict[str, str]:
    """The environment the delegated CLI run needs, from the old flags.

    These are exactly the knobs `run.py` used to set; the CLI reads them from the
    environment, so forwarding them keeps an existing command's behaviour.
    """
    env = dict(os.environ)
    env["PPI_OUTPUT_DIR"] = str(out_dir)
    env.update(
        {
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
            "PPI_SINGLE_CELL_PREPROCESS": args.single_cell_preprocess,
            "PPI_SINGLE_CELL_TOP_GENES": str(args.single_cell_top_genes),
            "PPI_SINGLE_CELL_TARGET_SUM": str(args.single_cell_target_sum),
            "PPI_SINGLE_CELL_MIN_CELLS": str(args.single_cell_min_cells),
            "PPI_SINGLE_CELL_FIGURES": "true" if args.single_cell_figures else "false",
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
            # perturbation-only knobs the CLI reads from the environment
            "PPI_SPLIT_MODE": args.split_mode,
            "PPI_TEST_FRACTION": str(args.test_fraction),
            "PPI_VALIDATION_FRACTION": str(args.validation_fraction),
            "PPI_MIN_CELLS_PER_PERTURBATION": str(args.min_cells_per_perturbation),
            "PPI_GO_K": str(args.go_k),
            "PPI_COEXPRESS_THRESHOLD": str(args.coexpress_threshold),
            "PPI_COEXPRESS_K": str(args.coexpress_k),
        }
    )
    if args.tooluniverse_python:
        env["KOSMOS_TOOLUNIVERSE_PYTHON"] = args.tooluniverse_python
    return env


def cli_command(args: argparse.Namespace) -> list[str]:
    command = [
        PYTHON, "-m", "kosmos.cli.main", "run", args.question,
        "--domain", args.domain,
        "--max-iterations", str(args.max_iterations),
        "--budget", str(args.budget),
        "--max-epochs", str(args.max_epochs),
        "--patience", str(args.patience),
        "--seed", str(args.seed),
    ]
    if args.test_path:
        command += ["--task-test-path", args.test_path]
    if args.task == "perturbation":
        command += ["--task", "perturbation"]
        if args.go_graph:
            command += ["--go-graph", args.go_graph]
        command += ["--split-mode", args.split_mode, "--eta", str(args.eta)]
        if args.gold_table:
            command += ["--gold-table", args.gold_table]
        for table in args.supp_table or []:
            command += ["--supp-table", table]
    else:
        command += ["--task", "per_cell"]
        if args.intent:
            command += ["--intent", args.intent]
        for hint in args.hint or []:
            command += ["--hint", hint]
        if args.fetch_limit is not None:
            command += ["--fetch-limit", str(args.fetch_limit)]
        if args.supp_limit is not None:
            command += ["--supp-limit", str(args.supp_limit)]
        if args.max_bytes is not None:
            command += ["--max-bytes", str(args.max_bytes)]
    if args.loss:
        command += ["--loss", args.loss]
    return command


def slug(text: str, words: int = 6) -> str:
    cleaned = re.sub(r"[^a-z0-9\s-]", "", text.lower()).split()
    return "-".join(cleaned[:words]) or "run"


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    unsupported = [
        flag for flag in _UNSUPPORTED
        if getattr(args, flag.lstrip("-").replace("-", "_"), None)
        != _UNSUPPORTED_DEFAULTS[flag.lstrip("-").replace("-", "_")]
    ]
    if unsupported:
        print(
            "error: run.py no longer implements "
            + ", ".join(unsupported)
            + ". Build a plan first and train it with `kosmos run --data-plan`, "
              "or use run.py.bak-20260928.",
            file=sys.stderr,
        )
        return 2

    if args.out:
        out_dir = Path(args.out)
    else:
        from datetime import datetime

        stamp = datetime.now().strftime("%Y%m%d-%H%M")
        out_dir = ROOT / "artifacts" / "runs" / f"{slug(args.question)}-{stamp}"
    out_dir.mkdir(parents=True, exist_ok=True)

    command = cli_command(args)
    if args.dry_run:
        print("# would run:\n#   " + " ".join(shlex.quote(part) for part in command))
        print(f"# PPI_OUTPUT_DIR={out_dir}")
        return 0

    completed = subprocess.call(command, cwd=ROOT, env=environment(args, out_dir))
    if completed != 0:
        print(f"\n# pipeline stopped with exit code {completed}", file=sys.stderr)
    return completed


if __name__ == "__main__":
    raise SystemExit(main())
