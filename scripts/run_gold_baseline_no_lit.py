"""Gold-only cross-donor baseline through ResearchDirector, without literature search.

Rationale: the CLI's HypothesisGeneratorAgent enables literature retrieval by
default, which can stall GENERATING_HYPOTHESES for many minutes (arXiv/PubMed
retries).  This runner disables that step, prints each state transition, and
aborts early if hypothesis generation keeps failing without producing output.

Usage:
    .venv/bin/python scripts/run_gold_baseline_no_lit.py
"""

import argparse
import asyncio
import json
import logging
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

load_dotenv(ROOT / ".env")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)

from kosmos.agents.research_director import ResearchDirectorAgent


DEFAULT_QUESTION = (
    "Gold-only cross-donor baseline: predict cell_type (36 classes) from the "
    "1500 ENSG expression columns in "
    "data/csv_donor_splits/gold_13272_19593_labeled.csv. Train only on rows "
    "with DonorID==13272 (split 80/20 stratified for validation) and report "
    "held-out metrics on DonorID==19593. Use sklearn StandardScaler + "
    "LogisticRegression with multinomial loss. Do not search literature, do "
    "not use external data or extra downloads, do not propose wet-lab "
    "experiments. Deliverable: balanced accuracy, macro F1 and accuracy on "
    "donor 19593, per-class recall for the 5 best and 5 worst classes, the "
    "top 10 discriminating genes, and a 2-3 sentence conclusion."
)


async def run(director, max_iterations, max_steps_without_hypotheses=2):
    """Run the director loop with visible state and an early-failure guard."""
    await director.execute({"action": "start_research"})
    print("\n=== research started ===", flush=True)

    iteration = 0
    stuck_steps = 0
    last_state = None
    last_hypothesis_count = -1

    while iteration < max_iterations:
        status = director.get_research_status()
        state = status["workflow_state"]
        n_hyps = status["hypothesis_pool_size"]
        print(
            f"[state] iteration={status['iteration']}/{status['max_iterations']} "
            f"state={state} hypotheses={n_hyps} "
            f"tested={status['hypotheses_tested']}",
            flush=True,
        )

        if status.get("has_converged"):
            print(f"[done] converged: {status.get('convergence_reason')}", flush=True)
            break

        # Early abort: stuck in hypothesis generation with no new hypotheses.
        if state == last_state and state == "GENERATING_HYPOTHESES":
            if n_hyps == last_hypothesis_count:
                stuck_steps += 1
                print(
                    f"[warn] no new hypotheses after {stuck_steps} consecutive "
                    f"GENERATING_HYPOTHESES steps",
                    flush=True,
                )
                if stuck_steps >= max_steps_without_hypotheses:
                    print(
                        "[abort] hypothesis generation is not producing output. "
                        "Check the ERROR logs above; likely the LLM JSON call "
                        "keeps failing.",
                        flush=True,
                    )
                    return "aborted_hypothesis_failure"
            else:
                stuck_steps = 0
        else:
            stuck_steps = 0
        last_state = state
        last_hypothesis_count = n_hyps

        await director.execute({"action": "step"})
        new_iteration = director.research_plan.iteration_count
        if new_iteration != iteration:
            iteration = new_iteration

    status = director.get_research_status()
    return status.get("has_converged", False) or iteration >= max_iterations


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--question",
        default=DEFAULT_QUESTION,
        help="Research objective (default: gold-only 13272 -> 19593 baseline).",
    )
    parser.add_argument(
        "--data-path",
        default=str(
            Path(__file__).resolve().parents[1]
            / "data/csv_donor_splits/gold_13272_19593_labeled.csv"
        ),
    )
    parser.add_argument("--max-iterations", type=int, default=1)
    parser.add_argument("--budget", type=float, default=3.0)
    parser.add_argument("--timeout-minutes", type=float, default=8.0)
    parser.add_argument("--output", default="artifacts/gold_baseline_no_lit.json")
    parser.add_argument(
        "--code-path",
        default=str(ROOT / "artifacts/gold_only_baseline_code.py"),
        help=(
            "Optional reviewed experiment code used for EXECUTING. If set, "
            "the step that would otherwise ask DeepSeek to write fresh code is "
            "replaced with this verified implementation (hypothesis, design and "
            "analysis still use DeepSeek)."
        ),
    )
    parser.add_argument(
        "--log-file",
        default="logs/run_gold_baseline_no_lit.log",
        help="Also write full INFO logs to this file.",
    )
    args = parser.parse_args()

    log_file = Path(args.log_file)
    log_file.parent.mkdir(parents=True, exist_ok=True)
    handler = logging.FileHandler(log_file, encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    logging.getLogger().addHandler(handler)

    config = {
        "max_iterations": args.max_iterations,
        "budget_usd": args.budget,
        "enable_concurrent_operations": False,
        "max_parallel_hypotheses": 1,
        "max_concurrent_experiments": 1,
        "use_literature_context": False,   # key: skip arXiv/PubMed retrieval
        "num_hypotheses": 1,               # keep the LLM step small
        "require_novelty_check": False,     # key: skip novelty DB/vector filtering
        "min_novelty_score": 0.0,
        "enable_sandboxing": False,         # repo's docker/ dir shadows docker SDK
        "use_experiment_templates": False,  # let DeepSeek write task-specific code
        "data_path": args.data_path,
        "enabled_domains": ["biology"],
        "enabled_experiment_types": ["data_analysis"],
    }

    director = ResearchDirectorAgent(
        research_question=args.question,
        domain="biology",
        config=config,
    )

    code_path = Path(args.code_path)
    if code_path.exists():
        reviewed_code = code_path.read_text(encoding="utf-8")

        class ReviewedCodeGenerator:
            """Minimal generator returning the reviewed experiment code."""

            def __init__(self, code: str):
                self.code = code
                self.llm_client = None

            def generate(self, protocol) -> str:
                return self.code

        director._code_generator = ReviewedCodeGenerator(reviewed_code)
        print(f"[info] EXECUTING will use reviewed code: {code_path}", flush=True)

    start = time.time()
    try:
        outcome = asyncio.run(
            asyncio.wait_for(
                run(director, max_iterations=args.max_iterations),
                timeout=args.timeout_minutes * 60,
            )
        )
    except asyncio.TimeoutError:
        outcome = "timed_out"
        print(
            f"[abort] exceeded {args.timeout_minutes} minutes; last state: "
            f"{director.workflow.current_state.value}",
            flush=True,
        )

    elapsed = round(time.time() - start, 1)
    status = director.get_research_status()
    payload = {
        "outcome": outcome,
        "elapsed_seconds": elapsed,
        "question": args.question,
        "status": status,
        "strategy_stats": getattr(director, "strategy_stats", None),
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    print(f"\n[result] outcome={outcome} elapsed={elapsed}s")
    print(f"[result] summary written to {out}", flush=True)
    print(f"[result] full log written to {log_file}", flush=True)


if __name__ == "__main__":
    main()
