"""
Research Director Agent - Master orchestrator for autonomous research (Phase 7).

This agent coordinates all other agents to execute the full research cycle:
Research Question → Hypotheses → Experiments → Results → Analysis → Refinement → Iteration

Uses message-based async coordination with all specialized agents.

Async Architecture (Issue #66 fix):
- execute(), _execute_next_action(), _do_execute_action() are now async
- All _send_to_* methods are now async
- Threading locks replaced with asyncio.Lock for async-safe operation
"""

from typing import Dict, Any, Optional, List
from datetime import datetime, timezone
import logging
import asyncio
import concurrent.futures
import threading
import time
from contextlib import contextmanager

from kosmos.agents.base import BaseAgent, AgentMessage, MessageType, AgentStatus
from kosmos.utils.compat import model_to_dict
from kosmos.core.rollout_tracker import RolloutTracker
from kosmos.core.workflow import (
    ResearchWorkflow,
    ResearchPlan,
    WorkflowState,
    NextAction
)
from kosmos.core.convergence import ConvergenceDetector, StoppingDecision, StoppingReason
from kosmos.core.llm import get_client
from kosmos.core.stage_tracker import get_stage_tracker
from kosmos.models.hypothesis import Hypothesis, HypothesisStatus
from kosmos.world_model import get_world_model, Entity, Relationship
from kosmos.db import get_session
from kosmos.db.operations import get_hypothesis, get_experiment, get_result
from kosmos.agents.skill_loader import SkillLoader

logger = logging.getLogger(__name__)

# Error recovery configuration
MAX_CONSECUTIVE_ERRORS = 3  # Halt after this many failures in a row
ERROR_BACKOFF_SECONDS = [2, 4, 8]  # Exponential backoff delays
ERROR_RECOVERY_LOG_PREFIX = "[ERROR-RECOVERY]"

# Infinite loop prevention (Issue #51)
MAX_ACTIONS_PER_ITERATION = 50  # Force convergence if exceeded


class ResearchDirectorAgent(BaseAgent):
    """
    Master orchestrator for autonomous research.

    Coordinates:
    - HypothesisGeneratorAgent: Generate and refine hypotheses
    - ExperimentDesignerAgent: Design experiment protocols
    - Executor: Run experiments
    - DataAnalystAgent: Interpret results
    - HypothesisRefiner: Refine hypotheses based on results
    - ConvergenceDetector: Detect when research is complete

    Uses message-based coordination for async agent communication.
    """

    def __init__(
        self,
        research_question: str,
        domain: Optional[str] = None,
        agent_id: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize Research Director.

        Args:
            research_question: The research question to investigate
            domain: Optional domain (biology, physics, etc.)
            agent_id: Optional agent ID
            config: Optional configuration (max_iterations, stopping_criteria, etc.)
        """
        super().__init__(
            agent_id=agent_id,
            agent_type="ResearchDirector",
            config=config or {}
        )

        self.research_question = research_question
        self.domain = domain

        # Domain validation and logging (Issue #51)
        self._validate_domain()

        # Load domain-specific skills (Issue #51 - skills integration)
        self.skills: Optional[str] = None
        self._load_skills()

        # Dataset path for experiments. When an evidence gateway is configured
        # instead of a local CSV, materialise one Evidence Capsule from it into a
        # data_path the rest of the pipeline reads unchanged (LECP). This is the
        # single seam: DataProvider, execute_with_data and the sandbox are all
        # downstream and untouched.
        self.data_path = self.config.get("data_path")
        self.evidence_server = self.config.get("evidence_server")
        self.evidence_dataset = self.config.get("evidence_dataset")
        self.evidence_config = self.config.get("evidence_config")
        # Every materialised source, in declaration order. Empty for a plain
        # --data-path run, one entry for a single --evidence-server run, N for a
        # federated one. `data_path` stays bound to exactly ONE file -- the
        # declared primary -- because everything downstream consumes it as a
        # file: the context reader opens it, the executor keys its data_files
        # dict on its basename, DataProvider treats it as a directory root. This
        # list is what GROUNDING reads; nothing else looks at it, which is what
        # keeps "hypotheses see N datasets" separate from "an experiment can
        # open N datasets".
        self.datasets: list = []
        if self.evidence_config and not self.data_path:
            self.data_path = self._materialize_federation()
        elif self.evidence_server and not self.data_path:
            self.data_path = self._materialize_evidence_capsule()

        # A question-less, data-driven run is seeded with a generic goal so the
        # pipeline has something to carry. Now that the data is materialised,
        # replace that boilerplate with a concrete research question the model
        # derives from the actual variables -- so the run investigates its own
        # question, not a placeholder. Fully guarded; a failure keeps the seed.
        # Skipped for a schema-only (no-rows) source, where there is no data to
        # ground a specific question in.
        if (
            self.config.get("data_driven")
            and self.data_path
            and not getattr(self, "_hypothesis_only", False)
        ):
            self._autogenerate_research_question()

        # Configuration
        self.max_iterations = self.config.get("max_iterations", 10)
        self.max_runtime_hours = self.config.get("max_runtime_hours", 12.0)  # Issue #56
        self.mandatory_stopping_criteria = self.config.get(
            "mandatory_stopping_criteria",
            ["iteration_limit", "no_testable_hypotheses"]
        )
        self.optional_stopping_criteria = self.config.get(
            "optional_stopping_criteria",
            ["novelty_decline", "diminishing_returns"]
        )

        # Initialize research plan and workflow. Use self.research_question, not
        # the constructor argument: a data-driven run may have just replaced the
        # seeded goal with a concrete, data-derived question above.
        self.research_plan = ResearchPlan(
            research_question=self.research_question,
            domain=domain,
            max_iterations=self.max_iterations
        )

        self.workflow = ResearchWorkflow(
            initial_state=WorkflowState.INITIALIZING,
            research_plan=self.research_plan
        )

        # Claude client for research planning and decision-making
        self.llm_client = get_client()

        # Initialize database if not already initialized
        from kosmos.db import init_from_config
        try:
            init_from_config()
        except RuntimeError as e:
            # Database already initialized - only log if it's a different error
            if "already initialized" not in str(e).lower():
                logger.warning("Database init RuntimeError: %s", e)
        except Exception as e:
            logger.warning(f"Database initialization failed: {e}")

        # Agent registry (will be populated during coordination)
        self.agent_registry: Dict[str, str] = {}  # agent_type -> agent_id

        # Lazy-init agent slots for direct-call pattern (Issue #76 extension)
        self._hypothesis_agent = None
        self._experiment_designer = None
        self._code_generator = None
        self._code_executor = None
        self._data_provider = None
        self._data_analyst = None
        self._hypothesis_refiner = None

        # Message correlation tracking
        self.pending_requests: Dict[str, Dict[str, Any]] = {}  # correlation_id -> request_info

        # Strategy effectiveness tracking
        self.strategy_stats: Dict[str, Dict[str, Any]] = {
            "hypothesis_generation": {"attempts": 0, "successes": 0, "cost": 0.0},
            "experiment_design": {"attempts": 0, "successes": 0, "cost": 0.0},
            "hypothesis_refinement": {"attempts": 0, "successes": 0, "cost": 0.0},
            "literature_review": {"attempts": 0, "successes": 0, "cost": 0.0}
        }

        # Agent rollout tracking (Issue #58)
        self.rollout_tracker = RolloutTracker()

        # Convergence detector - direct call, not message-based (Issue #76 fix)
        self.convergence_detector = ConvergenceDetector(
            mandatory_criteria=self.mandatory_stopping_criteria,
            optional_criteria=self.optional_stopping_criteria,
            config={
                "novelty_decline_threshold": self.config.get("novelty_decline_threshold", 0.3),
                "novelty_decline_window": self.config.get("novelty_decline_window", 5),
                "cost_per_discovery_threshold": self.config.get("cost_per_discovery_threshold", 1000.0),
                "min_experiments_before_convergence": self.config.get("min_experiments_before_convergence", 2),
                "max_iterations": self.max_iterations,
            }
        )

        # Research history
        self.iteration_history: List[Dict[str, Any]] = []

        # Error recovery tracking
        self._consecutive_errors: int = 0
        self._actions_this_iteration: int = 0
        self._error_history: List[Dict[str, Any]] = []
        self._last_error_time: Optional[datetime] = None

        # Runtime tracking (Issue #56)
        self._start_time: Optional[float] = None

        # Async-safe locks for concurrent operations (Issue #66)
        # Note: asyncio.Lock is not reentrant, refactored to avoid nested acquisitions
        self._research_plan_lock = asyncio.Lock()
        self._strategy_stats_lock = asyncio.Lock()
        self._workflow_lock = asyncio.Lock()
        self._agent_registry_lock = asyncio.Lock()
        # Keep threading locks for backwards compatibility in sync contexts
        self._research_plan_lock_sync = threading.RLock()
        self._strategy_stats_lock_sync = threading.Lock()
        self._workflow_lock_sync = threading.Lock()

        # Concurrent operations support
        self.enable_concurrent = self.config.get("enable_concurrent_operations", False)
        self.max_parallel_hypotheses = self.config.get("max_parallel_hypotheses", 3)
        self.max_concurrent_experiments = self.config.get("max_concurrent_experiments", 4)

        # Initialize ParallelExperimentExecutor if concurrent operations enabled
        self.parallel_executor = None
        if self.enable_concurrent:
            try:
                from kosmos.execution.parallel import ParallelExperimentExecutor
                self.parallel_executor = ParallelExperimentExecutor(
                    max_workers=self.max_concurrent_experiments
                )
                logger.info(
                    f"Parallel execution enabled with {self.max_concurrent_experiments} workers"
                )
            except ImportError:
                logger.warning("ParallelExperimentExecutor not available, using sequential execution")
                self.enable_concurrent = False

        # Initialize AsyncClaudeClient for concurrent LLM calls
        self.async_llm_client = None
        if self.enable_concurrent:
            try:
                from kosmos.core.async_llm import AsyncClaudeClient
                import os
                api_key = os.getenv("ANTHROPIC_API_KEY")
                if api_key:
                    self.async_llm_client = AsyncClaudeClient(
                        api_key=api_key,
                        max_concurrent=self.config.get("max_concurrent_llm_calls", 5),
                        max_requests_per_minute=self.config.get("llm_rate_limit_per_minute", 50)
                    )
                    logger.info("Async LLM client initialized for concurrent operations")
                else:
                    logger.warning("ANTHROPIC_API_KEY not set, async LLM disabled")
            except ImportError:
                logger.warning("AsyncClaudeClient not available, using sequential LLM calls")

        # Initialize world model for persistent knowledge graph
        try:
            self.wm = get_world_model()
            # Create ResearchQuestion entity
            question_entity = Entity.from_research_question(
                question_text=research_question,
                domain=domain,
                created_by=f"ResearchDirectorAgent:{self.agent_id}"
            )
            self.question_entity_id = self.wm.add_entity(question_entity)
            logger.info(f"Research question persisted to knowledge graph: {self.question_entity_id}")
        except Exception as e:
            logger.warning(f"Failed to initialize world model: {e}. Continuing without graph persistence.")
            self.wm = None
            self.question_entity_id = None

        logger.info(
            f"ResearchDirector initialized for question: '{research_question}' "
            f"(max_iterations={self.max_iterations}, concurrent={self.enable_concurrent})"
        )

    def _materialize_evidence_capsule(self) -> Optional[str]:
        """Fetch one Evidence Capsule from the gateway and stage it as a CSV.

        The gateway returns approved, banded statistics -- never raw records --
        which we render to one row per released group/estimate/cohort-cell. The
        rest of the pipeline reads that file exactly as it would a `--data-path`
        CSV, so this is the only place the evidence source touches Kosmos.

        The user asked for this evidence source explicitly (`--evidence-server`),
        so a failure to obtain a capsule is a hard, visible error -- NOT a silent
        fall-through to a dataless run that then reports cosmetic success. On
        failure it raises with the gateway's reason (each template it tried);
        run.py surfaces that as "Research failed: ...".
        """
        try:
            from kosmos.evidence.client import fetch_source, materialize
        except ImportError as e:
            raise RuntimeError(
                f"--evidence-server was requested but the evidence client is "
                f"unavailable: {e}"
            ) from e
        try:
            result = fetch_source(self.evidence_server, self.evidence_dataset)
        except Exception as e:
            raise RuntimeError(
                f"AutoEvidence returned nothing usable for "
                f"'{self.evidence_dataset}': {e}"
            ) from e

        import os
        import tempfile

        # Discovery path: a policy-less, private dataset returns a SCHEMA capsule
        # (column names + types, no rows). There is no data to run experiments on,
        # but the schema seeds data-driven hypothesis generation. Store it as the
        # data context and return no data_path.
        if result.get("kind") == "schema":
            cap = result["signed"]["capsule"]
            cols = cap.get("columns", [])
            if not self.evidence_dataset:
                self.evidence_dataset = cap.get("dataset") or "evidence"
            lines = [
                f"Dataset '{self.evidence_dataset}' ({cap.get('source_kind', 'private')}), "
                f"schema only -- {len(cols)} columns, no rows released:",
                "Columns (name: type):",
            ]
            lines += [f"  - {c.get('name')}: {c.get('dtype')}" for c in cols]
            self._data_context_override = "\n".join(lines)
            # No rows were released, so there is nothing to run experiments on.
            # The run is hypothesis-only: it generates hypotheses grounded in the
            # schema, then converges instead of designing dataless experiments.
            # The single-source form of "nothing released rows". `_hypothesis_only`
            # is now derived (see the property): with a source list it asks
            # whether ANY source has rows, and this flag answers for the
            # one-source path, which builds no list.
            self._schema_only_single = True
            print(
                f"  [evidence] policy-less dataset '{self.evidence_dataset}' -> "
                f"SCHEMA ONLY ({cap.get('source_kind', 'private')}); {len(cols)} "
                f"columns, no rows released"
            )
            for note in cap.get("limitations") or []:
                print(f"  [evidence] limitation  : {note}")
            print(f"  [evidence] columns     : {', '.join(c.get('name') for c in cols)}")
            return None  # no data_path: experiments have no data (private)

        # Evidence path: a released capsule -> materialise it. Two kinds arrive
        # here. A gated capsule of banded statistics, and an OPEN-DATA capsule
        # naming a public dataset staged in full; both end as a CSV at `out`,
        # which is all the experiment layer needs to know.
        capsule = result["signed"]
        body = capsule.get("capsule", capsule)
        # The dataset id is optional on the command line now -- an evidence
        # gateway is bound to one policy and a discovery server is asked -- so
        # take the name the capsule reports when none was supplied. Without
        # this the run writes `evidence_None.csv` and labels every line 'None'.
        if not self.evidence_dataset:
            self.evidence_dataset = body.get("dataset") or "evidence"
        out = os.path.join(tempfile.gettempdir(), f"evidence_{self.evidence_dataset}.csv")
        materialize(capsule, out)
        is_open_data = "staged_path" in body
        classification = (
            "OPEN DATA (public, released in full)"
            if is_open_data
            else str(body.get("classification", "?")).upper()
        )
        logger.info(
            f"Evidence capsule fetched for '{self.evidence_dataset}' "
            f"(classification={classification}); materialised to {out}"
        )

        # Observability / testing block. The whole point of --evidence-server is
        # that the run is sourced from the gateway, so surface what it released:
        # the flag (green/yellow/red), the gateway's own answer + what it redacted
        # or limited, and a header+rows preview of the banded table Kosmos will use
        # as data. Printed before the Live progress display starts, so it stays
        # visible at the top of the run.
        print(f"  [evidence] gateway '{self.evidence_dataset}' -> capsule flag: {classification}")
        if body.get("answer"):
            print(f"  [evidence] answer      : {body['answer']}")
        for note in body.get("redactions") or []:
            print(f"  [evidence] redaction   : {note}")
        for note in body.get("limitations") or []:
            print(f"  [evidence] limitation  : {note}")
        try:
            with open(out) as fh:
                preview = [ln.rstrip("\n") for ln in fh.readlines()[:6]]  # header + up to 5 rows
        except OSError:
            preview = []
        if is_open_data:
            print(
                f"  [evidence] released    : {body.get('n_rows', '?'):,} rows x "
                f"{body.get('n_columns', '?')} columns, exact values"
                if isinstance(body.get("n_rows"), int)
                else "  [evidence] released    : full public dataset"
            )
            print(f"  [evidence] reference   : {body.get('source_reference', '?')}")
        print(f"  [evidence] data preview (header + first {max(0, len(preview) - 1)} row(s)):")
        for line in preview:
            print(f"             {line}")
        print(f"  [evidence] materialised to {out}")

        # Ground hypotheses in the REAL variables (from the menu), not the
        # capsule's rendered columns (group/n_band/mean_band). Otherwise the
        # model hypothesises about disclosure artifacts -- e.g. "groups with
        # n_band '5000+' vs '1000-4999'" -- instead of the actual admission
        # variables. Experiments still use the materialised capsule (data_path).
        menu = result.get("menu")
        if menu:
            # Pass the regime explicitly: `is_open_data` above already knows a
            # bulk release arrived, and describing that as "banded aggregates,
            # never raw rows" is false about data the agent holds in full.
            self._data_context_override = self._menu_to_context(
                menu,
                body,
                dataset_id=self.evidence_dataset,
                release_regime=(
                    "released IN FULL: exact values, one row per record, no "
                    "banding or suppression applied to the released columns"
                    if is_open_data
                    else None
                ),
            )
        return out

    def _materialize_federation(self) -> Optional[str]:
        """Fetch EVERY configured source, stage each, return the primary's path.

        The multi-dataset entry point, and the whole of what "run over N datasets"
        means at this stage: N gateways are asked, N capsules are staged into
        their own directories, and all N are described to hypothesis generation
        at once. No experiment gains access to more than one of them -- the
        executor still mounts only the file `data_path` names.

        Fails the run when nothing released rows, for the same reason the
        single-source path does: a dataless run that reports cosmetic success is
        worse than a visible failure. A run where SOME sources released is
        allowed to proceed, because that is a real and useful state -- but the
        sources that failed are carried in `self.datasets` and named in the
        banner, never silently dropped.
        """
        from kosmos.evidence.federation import (
            EvidenceConfigError,
            load_sources,
            materialize_sources,
        )

        try:
            sources = load_sources(self.evidence_config)
        except EvidenceConfigError as e:
            raise RuntimeError(f"--evidence-config could not be used: {e}") from e

        run_dir = self._evidence_run_dir()
        print(f"  [evidence] federating {len(sources)} source(s) -> {run_dir}")
        self.datasets = materialize_sources(sources, run_dir, log=print)

        failed = [m for m in self.datasets if m.kind == "failed"]
        for m in failed:
            print(f"  [evidence] WARNING: source {m.source.name!r} contributed nothing")
        if all(m.kind == "failed" for m in self.datasets):
            reasons = "; ".join(f"{m.source.name}: {m.error}" for m in self.datasets)
            raise RuntimeError(f"no evidence source released anything -- {reasons}")

        with_rows = [m for m in self.datasets if m.has_rows]
        if not with_rows:
            # Every source that answered gave a schema. Hypotheses can still be
            # grounded in the declared variables; experiments cannot run, and
            # `_hypothesis_only` (below) reports that from the same list.
            print("  [evidence] no source released rows -- hypothesis-only run")
            return None

        primary = next(
            (m for m in with_rows if m.source.primary),
            with_rows[0],
        )
        if not self.evidence_dataset:
            self.evidence_dataset = primary.dataset_id
        print(
            f"  [evidence] primary dataset for experiments: "
            f"{primary.source.name} ({primary.dataset_id})"
        )
        if len(with_rows) > 1:
            others = ", ".join(m.source.name for m in with_rows if m is not primary)
            # Deliberately does NOT say whether these are mounted into
            # experiments: that is decided later, per the join-key rule, and
            # printed by `_mountable_datasets`. Asserting it here was stale the
            # moment multi-dataset experiments became possible, and a run log
            # that contradicts itself two lines apart is worse than one that
            # says less.
            print(f"  [evidence] also grounding hypotheses in: {others}")
        return str(primary.path)

    def _evidence_run_dir(self) -> str:
        """A per-run staging directory, reaped when the process exits.

        The single-source path wrote `$TMPDIR/evidence_<dataset>.csv` and left it
        there. One file in a shared location is untidy; N files per run, named
        for their datasets and readable by the executor's in-process fallback,
        is a channel between runs that nobody opened deliberately. A private
        directory per run closes it and costs one mkdtemp.
        """
        import atexit
        import shutil
        import tempfile

        existing = getattr(self, "_evidence_dir", None)
        if existing:
            return existing
        path = tempfile.mkdtemp(prefix="kosmos_evidence_")
        self._evidence_dir = path
        atexit.register(shutil.rmtree, path, True)
        return path

    def _combination_instructions(self, usable: list) -> list:
        """What the model may propose ACROSS the datasets, and what it must.

        This is the difference between "here are N datasets" and "here are N
        datasets that belong to one question". Left to itself the model writes
        one hypothesis per dataset -- each perfectly good, each answerable
        without the others, so the run learns nothing it could not have learned
        from N separate runs. Being handed several datasets at once is only
        worth anything if at least one claim NEEDS more than one of them.

        So the instruction does two things the previous wording did not. It
        REQUIRES a combined hypothesis rather than permitting one, and it says
        which columns the datasets actually share, because a joint claim that
        cannot be computed from the released columns is a claim no experiment
        can test.

        The permission itself is not assumed: it comes from the same join-key
        rule the executor enforces. When mounting is refused, asking for a
        joined hypothesis would invite one that no experiment is allowed to run.
        """
        allowed = bool(self._mountable_datasets())
        shared = self._shared_columns(usable)

        # The measured merge verdicts, which until now reached only CODE
        # generation. Hypothesis generation was choosing what to test with no
        # idea which datasets can actually be related, and proposed a
        # pQTL-versus-heart-eQTL comparison that needs a gene-symbol-to-Ensembl
        # mapping none of these datasets carries. The experiment then "completed"
        # having produced one line: "Gene symbol-to-Ensembl mapping required
        # (mygene) not available." A hypothesis the data cannot address is a
        # wasted iteration, and the run only gets ten.
        mount = self._mountable_datasets()
        compat = self._merge_compatibility(mount) if len(mount) > 1 else None

        if not allowed:
            return [
                "Relating them: you may propose a hypothesis that COMPARES "
                "findings across these datasets -- for example whether an "
                "effect measured in one is reproduced in another. Each "
                "experiment in this run reads exactly one dataset, so such a "
                "claim is tested by analysing each separately and comparing the "
                "results; do not propose anything that requires linking records "
                "between datasets.",
            ]

        lines = []
        if compat:
            lines.extend([
                "WHICH DATASETS CAN ACTUALLY BE RELATED (measured from the "
                "released data, not assumed):",
                compat,
                "Do NOT propose a hypothesis whose test requires a pair listed "
                "as CANNOT MERGE, or an identifier mapping that none of these "
                "datasets provides. Such a claim cannot be tested here however "
                "it is worded.",
                "",
            ])
        lines += [
            "COMBINING THEM -- REQUIRED. A single experiment in this run CAN "
            "open more than one of these datasets at once. At least one of your "
            "hypotheses MUST be one that cannot be tested from any single "
            "dataset alone: it must relate a quantity from one dataset to a "
            "quantity from another. A set of hypotheses each answerable from one "
            "dataset wastes the fact that you were given several.",
            "",
            "Two forms of combined claim are available, in order of preference:",
            "  (a) a per-feature JOIN -- relate the two datasets row-by-row on a "
            "shared feature identifier (a gene, a variant, a transcript), then "
            "test the relationship across features;",
            "  (b) an ESTIMATE-LEVEL relation -- compute a statistic in each "
            "dataset separately and test whether the two are related, which "
            "works even when no shared identifier is released.",
        ]
        if shared:
            lines.append(
                "Columns present in more than one dataset (candidate join keys): "
                + ", ".join(sorted(shared))
                + "."
            )
        else:
            lines.append(
                "NOTE: these datasets share NO column names, so form (a) is not "
                "available from the released columns -- a record-level join has "
                "nothing to join on. Prefer form (b), and do not assume an "
                "identifier column that is not listed below."
            )
        lines.append(
            "Never link rows that represent individual SUBJECTS between "
            "datasets, whatever identifiers appear."
        )
        return lines

    @property
    def _CAPSULE_RENDER_COLUMNS(self):
        """The release-renderer's own column vocabulary, from where it is emitted.

        Imported rather than restated. This list used to be a hand-copy of what
        `evidence/client.py::capsule_to_rows` emits, which would have gone stale
        silently the first time a column was added there -- and stale in the
        dangerous direction, since an unrecognised renderer column then looks
        like a feature two datasets share and gets offered as a join key.
        """
        from kosmos.evidence.client import RENDERED_COLUMNS

        return RENDERED_COLUMNS

    def _shared_columns(self, usable: list) -> set:
        """Feature columns appearing in more than one staged dataset.

        Read from the STAGED FILES, not the menus, for the same reason code
        generation is: the menu names variables the policy governs, while the
        file holds whatever the released capsule rendered. A join can only use
        what is actually in the file.

        Capsule-rendering columns are stripped first -- see
        `_CAPSULE_RENDER_COLUMNS`. What survives is a column two datasets share
        because they describe the same features, which is the only kind of
        shared column a join should be built on.
        """
        import collections

        counts = collections.Counter()
        for mat in usable:
            if not mat.has_rows:
                continue
            try:
                with open(mat.path) as fh:
                    header = fh.readline().strip()
            except OSError:
                continue
            counts.update({
                c.strip() for c in header.split(",")
                if c.strip() and c.strip() not in self._CAPSULE_RENDER_COLUMNS
            })
        return {col for col, n in counts.items() if n > 1}

    @staticmethod
    def _id_shape(value: str) -> Optional[tuple]:
        """(separator, field shapes) for an identifier, or None if not one.

        `17:40602553:TA:T:imp:v1` -> (':', ('#', '#', 'A', 'A', 'A', 'A#'))
        `chr11_47691640_G_A_b38` -> ('_', ('A#', '#', 'A', 'A', 'A#'))

        Digit runs become `#` and letter runs `A`, so the shape survives the
        specific variant while keeping the structure that decides whether two
        columns can meet.
        """
        import re

        text = str(value)
        for sep in (":", "_", "-", "|"):
            if text.count(sep) >= 2:
                fields = text.split(sep)
                shapes = tuple(
                    re.sub(r"[A-Za-z]+", "A", re.sub(r"[0-9]+", "#", f))
                    for f in fields
                )
                return sep, shapes
        return None

    @classmethod
    def _merge_compatibility(cls, mount: dict, sample_rows: int = 200) -> Optional[str]:
        """Which identifier columns across these datasets can actually meet.

        The staged-file summary already showed the model example values from
        every column, and it merged `17:40602553:TA:T:imp:v1` against
        `chr11_47691640_G_A_b38` regardless: raw examples are evidence, and what
        the model needs is the conclusion drawn from them.

        Compared by SHAPE rather than by sampled values. Sampling values would
        report a false "cannot join" whenever two sorted files happen to open on
        different chromosomes, and wrongly telling a run that its datasets
        cannot be related is worse than saying nothing. Shape is decided by a
        handful of rows and does not depend on which rows.

        Three verdicts, and the middle one is the useful one: identical shapes
        merge directly; one shape that is a PREFIX of another merges after
        truncating the longer to that many fields (the real relationship
        between the pQTL `ID` and the T1 `varId`); anything else cannot merge on
        those columns at all.
        """
        import pandas as pd

        columns: list[tuple[str, str, tuple, str]] = []
        for name, path in mount.items():
            try:
                df = pd.read_csv(path, nrows=sample_rows)
            except Exception:
                continue
            for col in df.columns:
                values = df[col].dropna().astype(str)
                if values.empty or values.nunique() < max(2, len(values) // 2):
                    continue
                shapes = [cls._id_shape(v) for v in values.head(20)]
                shapes = [s for s in shapes if s]
                if len(shapes) < len(values.head(20)) // 2 or not shapes:
                    continue
                modal = max(set(shapes), key=shapes.count)
                columns.append((name, col, modal, values.iloc[0]))

        if len(columns) < 2:
            return None

        direct, truncated, blocked = [], [], []
        for i, (dsa, ca, (sepa, sha), exa) in enumerate(columns):
            for dsb, cb, (sepb, shb), exb in columns[i + 1:]:
                if dsa == dsb:
                    continue
                if sepa == sepb and sha == shb:
                    direct.append(f"{dsa}.{ca} == {dsb}.{cb}")
                elif sepa == sepb and (
                    sha[: len(shb)] == shb or shb[: len(sha)] == sha
                ):
                    n = min(len(sha), len(shb))
                    longer = f"{dsa}.{ca}" if len(sha) > len(shb) else f"{dsb}.{cb}"
                    shorter = f"{dsb}.{cb}" if len(sha) > len(shb) else f"{dsa}.{ca}"
                    truncated.append(
                        f"{longer} joins {shorter} after truncating "
                        f"{longer} to its first {n} '{sepa}'-separated fields"
                    )
                else:
                    blocked.append(f"{dsa}.{ca} ({exa}) vs {dsb}.{cb} ({exb})")

        if not (direct or truncated or blocked):
            return None

        lines = [
            "MERGE COMPATIBILITY OF IDENTIFIER COLUMNS (computed from the staged "
            "files, not guessed). Choose join keys from this list:",
        ]
        for entry in direct[:8]:
            lines.append(f"  MERGES DIRECTLY: {entry}")
        for entry in truncated[:8]:
            lines.append(f"  MERGES AFTER TRUNCATION: {entry}")
        for entry in blocked[:8]:
            lines.append(f"  CANNOT MERGE: {entry}")
        if blocked:
            lines.append(
                "  A pair listed as CANNOT MERGE has incompatible identifier "
                "structure -- often a different genome build or namespace -- so "
                "merging on it returns ZERO rows however the columns are cleaned. "
                "Relate those datasets some other way, or state in `results` that "
                "the link cannot be made."
            )
        return "\n".join(lines)

    def _column_descriptions(self) -> dict:
        """{dataset name: {column: description}} from each source's menu.

        The menu is the only place a column's MEANING is written down -- the
        steward's own sentence, e.g. "Variant id, chr:pos:ref:alt on GRCh37 --
        the join key to the T1 GWAS" and, on the neighbouring column, "Position,
        GRCh38. NOT the position inside ID, which is GRCh37."

        Hypothesis generation has always seen those sentences; code generation
        never did. It received names, dtypes and ranges only, which is enough to
        write a merge and not enough to write a CORRECT one. Observed exactly
        that way: the model joined the two GWAS on coordinates that are in
        different genome builds and the container raised "No overlapping
        variants after merging cis-pQTL and T1 GWAS" -- a silent-looking failure
        that reads like the datasets do not overlap when in fact the join key
        was wrong.
        """
        out: dict = {}
        for mat in getattr(self, "datasets", []) or []:
            menu = getattr(mat, "menu", None)
            if not isinstance(menu, dict):
                continue
            described = {
                v.get("name"): (v.get("description") or "").strip()
                for v in menu.get("variables", [])
                if v.get("name") and (v.get("description") or "").strip()
            }
            if described:
                name = getattr(getattr(mat, "source", None), "name", None) or mat.dataset_id
                out[name] = described
        return out

    def _staged_file_context(
        self, mount: dict, *, single: bool = False
    ) -> Optional[str]:
        """Describe the STAGED FILES the generated code will open -- not the menu.

        These are two different column spaces and confusing them produces code
        that cannot run. The menu names a dataset's real variables (`BETA`,
        `CHISQ`, `LOG10P`) and is the right grounding for HYPOTHESES, which are
        claims about the science. But what lands on disk is a *rendering of the
        released capsule*: a banded group release stages
        `group, n_band, mean, mean_band, sd, median…`, and a passthrough release
        stages the granted columns plus `_row`. Code written against the menu
        therefore indexes columns that are not in the file.

        Observed exactly that way on the first two-dataset SOD2 run: the model
        was handed menu variables, wrote code expecting a `band` column, and the
        container raised `Missing required columns in sod2_mr_ready: ['band']`.
        So code generation is grounded in the file, hypothesis generation stays
        grounded in the menu, and each gets the column space it actually needs.
        """
        described = self._column_descriptions()
        compat_text = self._merge_compatibility(mount)
        compat = f"\n\n{compat_text}" if compat_text else ""
        blocks: list[str] = []
        for name, path in mount.items():
            summary = self._summarise_file(path, descriptions=described.get(name))
            if summary:
                # Name the variable the generated code will really use. On the
                # single-dataset path there is no `datasets` dict, so telling
                # the model to open `datasets['GSE2240']` would hand it a name
                # that does not exist.
                opener = "data_path" if single else f"datasets['{name}']"
                blocks.append(f"Dataset '{name}' (open with {opener}):\n{summary}")
        if not blocks:
            return None
        return (
            "These are the ACTUAL columns of the files this experiment will "
            "open. Use only these column names; they are the released capsule's "
            "columns, which may differ from the variable names a hypothesis "
            "refers to.\n\n"
            # Stated because recognising the dataset is what caused the failure.
            # The model knew the Wisconsin breast-cancer data from scikit-learn
            # and wrote its column names from memory -- `mean radius`, `worst
            # radius` -- while this copy names them `radius_mean`,
            # `radius_worst`. Not one of the six matched, and the run died on
            # its own column check. A recognised dataset is more dangerous than
            # an unfamiliar one: it invites recall instead of reading.
            "IF YOU RECOGNISE THIS DATASET from a library, a paper or a "
            "tutorial, do NOT write the column names you remember. This copy "
            "may name them differently -- `radius_mean` where you recall `mean "
            "radius`, `diagnosis` where you recall `target`. Every column you "
            "reference must appear verbatim in the list above.\n\n"
            # The single most repeated runtime failure across these runs:
            # `ValueError: could not convert string to float: 'B'`, from feeding
            # a label column into a scaler, a regression or a correlation. The
            # dtypes are listed above and were still ignored twice in one run,
            # so the consequence is spelled out rather than left to inference.
            "COLUMN TYPES ARE LISTED ABOVE AND THEY BIND. A column shown as "
            "`object` holds text: passing it to a scaler, a regression, a "
            "correlation or any numeric routine raises `could not convert "
            "string to float`. Build the feature matrix with "
            "`df.select_dtypes('number')`, drop identifier columns, and drop "
            "the outcome column from the features -- then encode the outcome "
            "separately if the analysis needs it numeric.\n\n"
            "Where a column carries a description, it is the data steward's own "
            "and it is authoritative -- especially about which column joins to "
            "which. Two columns that look like the same quantity may not be "
            "(different genome builds, units, or identifier conventions), so "
            "choose join keys from the descriptions rather than from the "
            "column names.\n\n"
            # Stated because it was assumed wrongly: GWAS summary statistics are
            # usually distributed as TSV, so the model wrote
            # `pd.read_csv(path, usecols=[...], sep='\t')`. Against a
            # comma-separated file that parses every row into ONE column, and
            # pandas reports it as "Usecols do not match columns" -- naming all
            # eleven real columns as missing, which reads like the wrong file
            # rather than the wrong delimiter.
            "FILE FORMAT: every staged file is COMMA-separated with a header "
            "row, whatever the upstream source's convention was. Read them with "
            "pd.read_csv(path) and do NOT pass sep= or delimiter=.\n\n"
            # A GPL96 series stages Affymetrix probe IDs and no symbols. The
            # generated code recognised that, concluded it could not map them,
            # and raised `Probe-to-gene annotation mapping is required ...
            # Cannot proceed` -- correctly refusing to invent a mapping, but
            # also discarding an analysis it could have completed. Refusing to
            # fabricate and refusing to analyse are different acts, and the
            # fabrication rule was being read as licence for the second.
            "IDENTIFIER LEVEL: analyse the identifiers the file actually holds. "
            "If its rows are keyed by platform probe or feature IDs "
            "(`1007_s_at`, `ILMN_1651209`, `ENSG00000141510`) rather than gene "
            "symbols, do the statistics AT THAT LEVEL and report those IDs. The "
            "sandbox has no network and no annotation package, so a probe-to-"
            "symbol mapping can be neither fetched nor invented -- but its "
            "absence is not a reason to stop. A ranked list of probe IDs with "
            "their effect sizes and adjusted p-values is a COMPLETE result; "
            "raising because symbols are unavailable is not. Say in `results` "
            "which identifier space the findings are in.\n\n"
            + "\n\n".join(blocks) + compat
        )

    def _mountable_datasets(self) -> dict:
        """The datasets one experiment may open together: {name: host path}.

        Empty unless a federated run has more than one source WITH ROWS and the
        join-key rule permits mounting them together (see
        `evidence.federation.mountable_together`). Empty means the caller takes
        the single-dataset path -- the primary file only -- which is Phase 1a
        behaviour and is what every non-federated run gets.

        The decision is cached per run, not per experiment: it depends only on
        the sources, and recomputing it per experiment would let a run drift
        between mounting and not mounting the same files.
        """
        cached = getattr(self, "_mount_decision", None)
        if cached is None:
            from kosmos.evidence.federation import mountable_together

            cached = mountable_together(getattr(self, "datasets", []) or [])
            self._mount_decision = cached
            if len(getattr(self, "datasets", []) or []) > 1:
                verdict = "ALLOWED" if cached.allowed else "REFUSED"
                print(f"  [evidence] multi-dataset experiments {verdict}: {cached.reason}")
        if not cached.allowed:
            return {}
        return {
            m.source.name: str(m.path)
            for m in self.datasets
            if m.has_rows and m.source.name in cached.names
        }

    @property
    def _hypothesis_only(self) -> bool:
        """True only when NOT ONE source released rows.

        Previously a plain attribute set by the schema-only branch, which was
        correct for one source and wrong for several: one schema-only source
        among four would converge the run with zero experiments
        (`decide_next_action` returns CONVERGE on it) even though the other three
        had real data staged. Derived from the source list so it can only be true
        when it is actually true.
        """
        if not getattr(self, "datasets", None):
            return bool(getattr(self, "_schema_only_single", False))
        return not any(m.has_rows for m in self.datasets)

    def _menu_to_context(
        self,
        menu: dict,
        capsule_body: Optional[dict] = None,
        dataset_id: Optional[str] = None,
        release_regime: Optional[str] = None,
    ) -> str:
        """Render the evidence menu into a hypothesis data context.

        Names the real variables (with types, descriptions, roles, levels) and
        states what this source ACTUALLY released -- which is not always bands.
        A policy-granted bulk release carries the same menu as a gated one, so a
        hardcoded "only banded aggregates" told the model the opposite of the
        truth about data it was holding in full: it would hedge conclusions it
        was entitled to draw, and read exact values as band labels. The regime
        sentence comes from the materialised source, which knows which kind of
        capsule arrived.
        """
        regime = release_regime or (
            "released ONLY as banded aggregates (summary statistics -- banded "
            "counts, banded means), never raw rows. Do NOT treat band labels "
            "(e.g. 'n_band') as measurements"
        )
        lines = [
            f"Dataset '{dataset_id or self.evidence_dataset}', accessed through an "
            f"AutoEvidence gateway; {regime}. Frame hypotheses as relationships "
            f"AMONG the real variables below that this data can address.",
        ]

        # The steward's own account of the dataset, and any notes attached to
        # it. `EvidenceMenu` carries both (autoevidence/schema/menu.py:62,68) and
        # this renderer dropped them, so the sentence saying COLOCALISATION
        # CANNOT BE COMPUTED FROM THIS RELEASE -- written on the policy after
        # coloc returned PP.H4 = 1e-08 across 85 loci -- never reached the model
        # that chooses what to test. It proposed colocalisation again, tested 15
        # loci, and got zero. Variable descriptions were reaching it; the
        # dataset-level ones were not.
        described = (menu.get("description") or "").strip()
        if described:
            lines.append(f"About this dataset, from the data steward: {described}")
        for note in menu.get("notes") or []:
            note = str(note).strip()
            if note:
                lines.append(f"  Note: {note}")

        lines.append("Variables:")
        for v in menu.get("variables", []):
            part = f"  - {v.get('name')} ({v.get('dtype')}): {(v.get('description') or '').strip()}"
            roles = ", ".join(v.get("allowed_roles", []))
            if roles:
                part += f" | usable as: {roles}"
            if v.get("levels"):
                part += f" | values: {', '.join(str(x) for x in v['levels'])}"
            if v.get("unit"):
                part += f" | unit: {v['unit']}"
            lines.append(part)
        tmpls = ", ".join(t.get("template_id") for t in menu.get("templates", []))
        if tmpls:
            lines.append(f"Available analyses: {tmpls}.")
        if capsule_body and capsule_body.get("answer"):
            lines.append(f"Example released evidence: {capsule_body['answer']}")
        return "\n".join(lines)

    def _autogenerate_research_question(self) -> None:
        """Derive a concrete research question from the materialised data.

        A question-less run is seeded with a generic exploration goal. Once the
        data exists we can do better: read the same data context the hypothesis
        generator uses and ask the model for one focused, testable question over
        the real variables -- so the overview, the report and the hypotheses all
        hang off the run's OWN question rather than boilerplate.

        Fully guarded. Any failure -- no data context, an LLM error, an empty or
        junk answer -- leaves the seeded goal in place, so this can never break a
        run. Uses the same `get_client()` the rest of the pipeline uses, so it
        honours the configured provider (e.g. OpenRouter/DeepSeek).
        """
        context = self._build_data_context()
        if not context:
            return
        try:
            from kosmos.core.llm import get_client

            client = get_client()
            prompt = (
                "Below is a summary of a dataset: its columns, their types, and "
                "basic statistics. Propose ONE focused, testable research "
                "question that can be investigated using ONLY these variables. "
                "Name real column names from the summary. Return the question as "
                "a single sentence, with no preamble or numbering.\n\n"
                f"DATASET SUMMARY:\n{context}"
            )
            # 400 rather than 80: with reasoning enabled the trace is drawn from
            # this same budget, so a tight cap yields a truncated or
            # reasoning-only response.
            response = client.generate(prompt=prompt, max_tokens=400, temperature=0.3)
            # get_client() returns a PROVIDER, whose generate() yields an
            # LLMResponse -- not the bare str that kosmos.core.llm's own
            # generate() returns. Treating it as a string raised
            # AttributeError on every call, which the broad except below
            # swallowed into "keeping the seeded goal" -- so this feature
            # silently never ran. Accept either shape rather than depending on
            # which client get_client() happens to hand back.
            answer = getattr(response, "content", response)

            # Take the line that is actually a QUESTION, not simply the first
            # line. Models often open with a restatement of the task ("We need
            # to propose one focused, testable research question...") and taking
            # splitlines()[0] captured that preamble and titled the whole report
            # with it -- an instruction masquerading as a finding. Scanning for
            # '?' is a much better filter than a length check, which the
            # preamble passed easily.
            candidates = [
                line.strip().strip('"').strip()
                for line in (answer or "").splitlines()
                if line.strip()
            ]
            question = next(
                (line for line in candidates if line.endswith("?") and len(line) >= 12),
                "",
            )

            if question:
                self.research_question = question
                logger.info(
                    f"Auto-generated research question from data: {question}"
                )
            else:
                logger.warning(
                    "Auto-generated response contained no usable question "
                    "(expected a line ending in '?'); keeping the seeded "
                    "data-driven goal. Response began: %r",
                    (answer or "")[:120],
                )
        except Exception as e:
            logger.warning(
                f"Could not auto-generate a research question from the data ({e}); "
                f"keeping the seeded data-driven goal."
            )

    # Rows read for the schema/stats sample, regardless of the file's actual
    # size. `nrows=` stops pandas reading after this many rows rather than
    # scanning the file first, so this bounds memory for a multi-GB CSV the
    # same way it does for a small one. 50,000 rows is comfortably enough for
    # a stable mean/std/min/max and a representative categorical level sample;
    # it is not enough to explain WHY a bigger number would matter here more
    # than for any other bounded sample.
    _CONTEXT_SAMPLE_ROWS = 50_000

    def _build_data_context(self, max_cols: int = 40) -> Optional[str]:
        """Summarise the dataset (schema + basic stats) for hypothesis grounding.

        Reads only the materialised data path -- for an AutoEvidence run that is
        the released capsule, so this exposes nothing the gateway did not already
        release. Returns column names/types, shape, and a compact numeric summary
        (never raw rows), or None when there is no readable dataset.

        Bounded to a fixed-size sample regardless of file size (see
        `_CONTEXT_SAMPLE_ROWS`). This used to be an unbounded `pd.read_csv` on
        the whole file: harmless for a small CSV, but a multi-GB dataset would
        pull many times its own size into this process's memory just to build a
        prompt -- before a single experiment runs, and regardless of what any
        downstream memory budget allows. The row COUNT reported is still exact
        (a cheap line count, not sampled), because an agent reasoning about
        statistical power from a wrong N is worse off than one told nothing.
        """
        import os

        # Federated run: describe EVERY source, in one string, at once. This is
        # what lets hypothesis generation propose a relationship that spans
        # datasets -- the model cannot relate two datasets it was shown one at a
        # time, or one of which it was never told about.
        #
        # Note what is NOT consulted here: `_data_context_override`. That flag
        # is set only when a menu arrives, and it short-circuits the whole
        # function, so with several sources one menu-bearing source would have
        # silently erased every menu-less one from the prompt. The list is built
        # from the sources themselves, so a source with no menu falls back to a
        # summary of its staged file rather than disappearing.
        if getattr(self, "datasets", None):
            return self._federated_data_context(max_cols=max_cols)

        # A discovery (schema-only) source sets this directly -- there is no data
        # file to summarise, only the released schema.
        override = getattr(self, "_data_context_override", None)
        if override:
            return override
        if not self.data_path or not os.path.exists(self.data_path):
            return None
        try:
            import pandas as pd

            with open(self.data_path, "rb") as fh:
                n_rows = sum(1 for _ in fh) - 1  # minus header
            df = pd.read_csv(self.data_path, nrows=self._CONTEXT_SAMPLE_ROWS)
        except Exception as e:
            logger.warning(f"Could not read dataset for hypothesis context: {e}")
            return None
        if df.shape[1] == 0:
            return None
        sampled = n_rows > len(df)

        cols = list(df.columns)[:max_cols]
        shape_line = f"Shape: {n_rows} rows x {df.shape[1]} columns."
        if sampled:
            shape_line += (
                f" Statistics below are computed from the first {len(df):,} "
                f"rows only (the file is larger than that sample)."
            )
        lines = [
            shape_line,
            "Columns (name: dtype):",
        ]
        lines += [f"  - {c}: {df[c].dtype}" for c in cols]
        if len(df.columns) > max_cols:
            lines.append(
                f"(Every column is named above. The statistics below cover "
                f"the first {max_cols} of {len(df.columns)}.)"
            )

        numeric = df[cols].select_dtypes(include="number")
        if not numeric.empty:
            lines.append("Numeric summary (mean / std / min / max):")
            for c in numeric.columns:
                s = numeric[c]
                try:
                    lines.append(
                        f"  - {c}: mean={s.mean():.4g}, std={s.std():.4g}, "
                        f"min={s.min():.4g}, max={s.max():.4g}"
                    )
                except (TypeError, ValueError):
                    continue
        categorical = df[cols].select_dtypes(exclude="number")
        for c in categorical.columns[:10]:
            vals = [str(v) for v in df[c].dropna().unique()[:8]]
            if vals:
                lines.append(f"  - {c} levels (sample): {', '.join(vals)}")
        return "\n".join(lines)

    def _federated_data_context(self, max_cols: int = 40) -> Optional[str]:
        """One string describing every source, for grounding hypotheses in all of them.

        Structure matters here as much as content. The model is given N labelled
        blocks and, when there is more than one, an explicit instruction about
        what it may and may not propose across them -- because the honest answer
        to "can these be related?" depends on something no analysis can derive
        from the data: whether the datasets describe the same subjects. Each
        block therefore carries the steward's declaration, including `unknown`,
        which is the default and means exactly what it says.

        A single source produces the block WITHOUT a header, so a one-source
        federated run reads identically to today's single-source run.
        """
        blocks: list[str] = []
        usable = [m for m in self.datasets if m.kind != "failed"]
        for mat in usable:
            body = self._describe_source(mat, max_cols=max_cols)
            if not body:
                continue
            if len(usable) == 1:
                blocks.append(body)
            else:
                # The header labels the block; it does not restate the release
                # regime, which `_describe_source` already puts in the body for
                # every shape of source. Saying it twice trains a reader -- human
                # or model -- to skip the line that matters most.
                blocks.append(f"### Dataset '{mat.dataset_id}' "
                              f"[source: {mat.source.name}]\n{body}")
        if not blocks:
            return None
        if len(blocks) == 1:
            return blocks[0]

        header = [
            f"You have been given {len(blocks)} datasets AT ONCE. Each is "
            f"described below. Base your hypotheses on these actual variables.",
            "",
        ]
        header.extend(self._combination_instructions(usable))
        overlaps = {m.source.subject_overlap for m in usable}
        if overlaps - {"disjoint"}:
            header.append(
                "Subject overlap between these datasets is "
                + ", ".join(sorted(overlaps))
                + ". Where it is 'shared' or 'unknown', findings across them are "
                "NOT statistically independent; say so in any claim that spans "
                "them."
            )
        failed = [m for m in self.datasets if m.kind == "failed"]
        if failed:
            header.append(
                "Unavailable in this run (do not reason about them): "
                + ", ".join(m.source.name for m in failed)
                + "."
            )
        return "\n".join(header) + "\n\n" + "\n\n".join(blocks)

    def _describe_source(self, mat, max_cols: int = 40) -> Optional[str]:
        """One source's block: its menu variables if it has a menu, else its file.

        The menu is preferred because it names the REAL variables with their
        descriptions, roles and levels, where a materialised capsule's columns
        are the rendered artifacts of a release (`group`, `n_band`, `mean_band`).
        A source with no menu -- the discovery path returns none -- still gets a
        block, derived from its schema or its staged file, which is the case the
        override mechanism used to lose entirely.
        """
        if mat.menu:
            return self._menu_to_context(
                mat.menu,
                mat.body,
                dataset_id=mat.dataset_id,
                release_regime=mat.describe_release(),
            )
        if mat.kind == "schema":
            cols = mat.body.get("columns") or []
            lines = [f"Columns (name: type), {len(cols)} total, no rows released:"]
            lines += [f"  - {c.get('name')}: {c.get('dtype')}" for c in cols[:max_cols]]
            return "\n".join(lines)
        if mat.path is not None:
            return self._summarise_file(str(mat.path), max_cols=max_cols)
        return None

    def _summarise_file(
        self,
        path: str,
        max_cols: int = 40,
        descriptions: Optional[Dict[str, str]] = None,
    ) -> Optional[str]:
        """Shape, columns and a compact numeric summary of one staged file.

        The same summary `_build_data_context` produces for a `--data-path` run,
        reached with an explicit path so it can be applied per source. Bounded to
        `_CONTEXT_SAMPLE_ROWS` for the same reason it is there: building a prompt
        must not pull a multi-GB file into memory, and with N sources that cost
        is multiplied.
        """
        import os

        if not os.path.exists(path):
            return None
        try:
            import pandas as pd

            with open(path, "rb") as fh:
                n_rows = sum(1 for _ in fh) - 1
            df = pd.read_csv(path, nrows=self._CONTEXT_SAMPLE_ROWS)
        except Exception as e:
            logger.warning(f"Could not read {path} for hypothesis context: {e}")
            return None
        if df.shape[1] == 0:
            return None

        # Every column NAME, not the first `max_cols`. Names are cheap --
        # 151 of them is a couple of kilobytes -- and the cost is the
        # per-column summary below, which stays capped. Truncating the names
        # hid all 111 columns past the 40th, including every `age:[0-10)`
        # bucket, so the model wrote `df['age']` from its memory of the UCI
        # dataset and the container raised `KeyError: 'age'`. A column the
        # model cannot see is one it will invent.
        all_cols = list(df.columns)
        cols = all_cols[:max_cols]
        shape_line = f"Shape: {n_rows} rows x {df.shape[1]} columns."
        if n_rows > len(df):
            shape_line += (
                f" Statistics below are computed from the first {len(df):,} rows only."
            )
        try:
            size_mb = os.path.getsize(path) / 1e6
        except OSError:
            size_mb = 0.0
        if size_mb >= 100:
            # A 548 MB CSV takes several GB to read whole, and the sandbox is
            # memory-capped: the container is killed with exit 137 and no
            # traceback, so the analysis fails before it computes anything.
            shape_line += (
                f" FILE IS {size_mb:,.0f} MB ON DISK -- read only the columns "
                f"you need with pd.read_csv(..., usecols=[...]); reading it "
                f"whole can exhaust the sandbox's memory."
            )

        lines = [shape_line, "Columns (name: dtype):"]
        descriptions = descriptions or {}
        for c in all_cols:
            line = f"  - {c}: {df[c].dtype}"
            # Missingness, stated per column. A name and a dtype cannot tell
            # anyone that `Unnamed_32` is 569/569 null -- the trailing-comma
            # artifact every hand-exported CSV carries -- and a run died on its
            # own "Dataset contains missing values" check because the column
            # looked like a 33rd feature. Counted on the sample, so it is
            # labelled as such rather than presented as the whole truth.
            n_missing = int(df[c].isna().sum())
            if n_missing == len(df) and len(df):
                line += "  [EMPTY: every sampled value is missing -- drop this column]"
            elif n_missing:
                line += f"  [{n_missing}/{len(df)} sampled values missing]"
            meaning = descriptions.get(c)
            if meaning:
                line += f" -- {meaning}"
            lines.append(line)
        if len(df.columns) > max_cols:
            lines.append(
                f"(Every column is named above. The statistics below cover "
                f"the first {max_cols} of {len(df.columns)}.)"
            )
        numeric = df[cols].select_dtypes(include="number")
        if not numeric.empty:
            lines.append("Numeric summary (mean / std / min / max):")
            for c in numeric.columns:
                s = numeric[c]
                try:
                    lines.append(
                        f"  - {c}: mean={s.mean():.4g}, std={s.std():.4g}, "
                        f"min={s.min():.4g}, max={s.max():.4g}"
                    )
                except (TypeError, ValueError):
                    continue
        # Low-cardinality columns, wherever they sit in the table.
        #
        # The statistics above cover the first `max_cols` columns, and an
        # outcome column is usually the LAST one. On the diabetes table
        # `readmitted` is column 150 of 151, so its values were never shown --
        # and the model encoded it as the original UCI strings, writing
        # `1 if x == '<30' else 0` against a column holding integers 0/1. Every
        # row became 0, its own filter emptied the frame, and train_test_split
        # got n_samples=0. A handful of distinct values costs a few bytes to
        # print and is exactly what a label or a flag needs to disclose.
        # Grouped by the value set, so completeness costs a line rather than a
        # hundred. Listing them one per column filled a 30-line quota with
        # one-hot dummies and still never reached `readmitted` at column 150 --
        # which is the single column whose values decide whether the analysis
        # works at all.
        _by_values: dict = {}
        for _c in all_cols:
            try:
                _uniq = df[_c].dropna().unique()
            except Exception:
                continue
            if 0 < len(_uniq) <= 10:
                _key = tuple(sorted(str(v) for v in _uniq))
                _by_values.setdefault(_key, []).append(_c)
        if _by_values:
            lines.append("Columns with few distinct values, by value set:")
            for _key, _members in list(_by_values.items())[:12]:
                # First few AND last few. A truncated list that always drops
                # the tail hides the outcome column, which is where an outcome
                # column usually is -- `readmitted` sat at position 150 of 151
                # and would vanish into "+113 more", which is the one column
                # whose values decide whether the analysis runs at all.
                if len(_members) <= 8:
                    _shown, _more = ", ".join(_members), ""
                else:
                    _shown = ", ".join(_members[:6] + ["..."] + _members[-2:])
                    _more = f"  ({len(_members)} columns)"
                lines.append(
                    f"  - takes values [{', '.join(_key)}]: {_shown}{_more}"
                )

        categorical = df[cols].select_dtypes(exclude="number")
        for c in categorical.columns[:10]:
            vals = [str(v) for v in df[c].dropna().unique()[:8]]
            if vals:
                lines.append(f"  - {c} levels (sample): {', '.join(vals)}")
        return "\n".join(lines)

    def _validate_domain(self):
        """Validate domain against enabled domains (Issue #51)."""
        # Default enabled domains if not configured
        default_domains = ["biology", "physics", "chemistry", "neuroscience"]
        enabled_domains = self.config.get("enabled_domains", default_domains)

        if self.domain:
            if self.domain.lower() not in [d.lower() for d in enabled_domains]:
                logger.warning(
                    f"[DOMAIN] Domain '{self.domain}' not in enabled domains: {enabled_domains}. "
                    "Research will proceed but domain-specific features may be limited."
                )
            else:
                logger.info(f"[DOMAIN] Research domain: {self.domain}")
        else:
            logger.info("[DOMAIN] No domain specified - using general research mode")

    def _load_skills(self):
        """Load domain-specific skills for enhanced prompts (Issue #51)."""
        try:
            skill_loader = SkillLoader()

            # Load skills based on domain or default research skills
            if self.domain:
                self.skills = skill_loader.load_skills_for_task(
                    task_type="research",
                    domain=self.domain,
                    include_examples=False,
                    include_common=True
                )
                if self.skills:
                    logger.info(f"Loaded skills for domain '{self.domain}'")
                else:
                    logger.debug(f"No specific skills found for domain '{self.domain}'")
            else:
                # Load common research skills
                self.skills = skill_loader.load_skills_for_task(
                    task_type="research",
                    include_examples=False,
                    include_common=True
                )
                if self.skills:
                    logger.info("Loaded common research skills")
        except Exception as e:
            logger.warning(f"Failed to load skills: {e}. Continuing without skill injection.")
            self.skills = None

    def get_skills_context(self) -> str:
        """Get skills context for prompt injection."""
        if self.skills:
            return f"\n{self.skills}\n"
        return ""

    # ========================================================================
    # LIFECYCLE HOOKS
    # ========================================================================

    def _on_start(self):
        """Initialize director when started."""
        logger.info(f"ResearchDirector {self.agent_id} starting research cycle")

        # Start runtime tracking (Issue #56)
        if self._start_time is None:
            self._start_time = time.time()
            logger.info(f"Research started at {datetime.now().isoformat()}, max runtime: {self.max_runtime_hours}h")

        with self._workflow_lock_sync:
            self.workflow.transition_to(
                WorkflowState.GENERATING_HYPOTHESES,
                action="Start research cycle"
            )

    def _check_runtime_exceeded(self) -> bool:
        """Check if research has exceeded maximum runtime (Issue #56)."""
        if self._start_time is None:
            return False
        elapsed_hours = (time.time() - self._start_time) / 3600
        return elapsed_hours >= self.max_runtime_hours

    def get_elapsed_time_hours(self) -> float:
        """Get elapsed research time in hours (Issue #56)."""
        if self._start_time is None:
            return 0.0
        return (time.time() - self._start_time) / 3600

    def _on_stop(self):
        """Cleanup when stopped."""
        logger.info(f"ResearchDirector {self.agent_id} stopped")

        # Cleanup async resources
        if self.async_llm_client:
            try:
                asyncio.run(self.async_llm_client.close())
            except Exception as e:
                logger.warning(f"Error closing async LLM client: {e}")

    # ========================================================================
    # THREAD-SAFE CONTEXT MANAGERS
    # ========================================================================

    @contextmanager
    def _research_plan_context(self):
        """Context manager for thread-safe research plan access (sync version)."""
        with self._research_plan_lock_sync:
            yield self.research_plan

    @contextmanager
    def _strategy_stats_context(self):
        """Context manager for thread-safe strategy stats access (sync version)."""
        with self._strategy_stats_lock_sync:
            yield self.strategy_stats

    @contextmanager
    def _workflow_context(self):
        """Context manager for thread-safe workflow access (sync version)."""
        with self._workflow_lock_sync:
            yield self.workflow

    # Async context manager helpers - use asyncio.Lock directly in async code
    # Example: async with self._research_plan_lock: ...

    # ========================================================================
    # GRAPH PERSISTENCE HELPERS
    # ========================================================================

    def _persist_hypothesis_to_graph(self, hypothesis_id: str, agent_name: str = "HypothesisGeneratorAgent"):
        """
        Persist hypothesis to knowledge graph with SPAWNED_BY relationship.

        Args:
            hypothesis_id: ID of hypothesis to persist
            agent_name: Name of agent that created the hypothesis
        """
        if not self.wm or not self.question_entity_id:
            return  # Graph persistence disabled

        try:
            with get_session() as session:
                # Fetch hypothesis from database
                hypothesis = get_hypothesis(session, hypothesis_id)
                if not hypothesis:
                    logger.warning(f"Hypothesis {hypothesis_id} not found in database")
                    return

                # Convert to Entity and persist
                entity = Entity.from_hypothesis(hypothesis, created_by=agent_name)
                entity_id = self.wm.add_entity(entity)

                # Create SPAWNED_BY relationship to research question
                rel = Relationship.with_provenance(
                    source_id=entity_id,
                    target_id=self.question_entity_id,
                    rel_type="SPAWNED_BY",
                    agent=agent_name,
                    generation=hypothesis.generation,
                    iteration=self.research_plan.iteration_count
                )
                self.wm.add_relationship(rel)

                # If refined from parent, add REFINED_FROM relationship
                if hypothesis.parent_hypothesis_id:
                    parent_rel = Relationship.with_provenance(
                        source_id=entity_id,
                        target_id=hypothesis.parent_hypothesis_id,
                        rel_type="REFINED_FROM",
                        agent=agent_name,
                        refinement_count=hypothesis.refinement_count
                    )
                    self.wm.add_relationship(parent_rel)

                logger.debug(f"Persisted hypothesis {hypothesis_id} to graph")

        except Exception as e:
            logger.warning(f"Failed to persist hypothesis {hypothesis_id} to graph: {e}")

    def _persist_protocol_to_graph(self, protocol_id: str, hypothesis_id: str, agent_name: str = "ExperimentDesignerAgent"):
        """
        Persist experiment protocol to knowledge graph with TESTS relationship.

        Args:
            protocol_id: ID of protocol to persist
            hypothesis_id: ID of hypothesis being tested
            agent_name: Name of agent that created the protocol
        """
        if not self.wm:
            return

        try:
            with get_session() as session:
                # Fetch protocol from database
                protocol = get_experiment(session, protocol_id)
                if not protocol:
                    logger.warning(f"Protocol {protocol_id} not found in database")
                    return

                # Convert to Entity and persist
                entity = Entity.from_protocol(protocol, created_by=agent_name)
                entity_id = self.wm.add_entity(entity)

                # Create TESTS relationship to hypothesis
                rel = Relationship.with_provenance(
                    source_id=entity_id,
                    target_id=hypothesis_id,
                    rel_type="TESTS",
                    agent=agent_name,
                    iteration=self.research_plan.iteration_count
                )
                self.wm.add_relationship(rel)

                logger.debug(f"Persisted protocol {protocol_id} to graph")

        except Exception as e:
            logger.warning(f"Failed to persist protocol {protocol_id} to graph: {e}")

    def _persist_result_to_graph(self, result_id: str, protocol_id: str, hypothesis_id: str, agent_name: str = "Executor"):
        """
        Persist experiment result to knowledge graph with PRODUCED_BY relationship.

        Args:
            result_id: ID of result to persist
            protocol_id: ID of protocol that produced this result
            hypothesis_id: ID of hypothesis being tested
            agent_name: Name of agent that created the result
        """
        if not self.wm:
            return

        try:
            with get_session() as session:
                # Fetch result from database
                result = get_result(session, result_id)
                if not result:
                    logger.warning(f"Result {result_id} not found in database")
                    return

                # Convert to Entity and persist
                entity = Entity.from_result(result, created_by=agent_name)
                entity_id = self.wm.add_entity(entity)

                # Create PRODUCED_BY relationship to protocol
                rel = Relationship.with_provenance(
                    source_id=entity_id,
                    target_id=protocol_id,
                    rel_type="PRODUCED_BY",
                    agent=agent_name,
                    iteration=self.research_plan.iteration_count
                )
                self.wm.add_relationship(rel)

                # Create TESTS relationship to hypothesis
                tests_rel = Relationship.with_provenance(
                    source_id=entity_id,
                    target_id=hypothesis_id,
                    rel_type="TESTS",
                    agent=agent_name
                )
                self.wm.add_relationship(tests_rel)

                logger.debug(f"Persisted result {result_id} to graph")

        except Exception as e:
            logger.warning(f"Failed to persist result {result_id} to graph: {e}")

    def _add_support_relationship(self, result_id: str, hypothesis_id: str, supports: bool, confidence: float, p_value: float = None, effect_size: float = None):
        """
        Add SUPPORTS or REFUTES relationship based on result analysis.

        Args:
            result_id: ID of result entity
            hypothesis_id: ID of hypothesis entity
            supports: True if result supports hypothesis, False if refutes
            confidence: Confidence score from analyst
            p_value: Statistical p-value if available
            effect_size: Effect size if available
        """
        if not self.wm:
            return

        try:
            rel_type = "SUPPORTS" if supports else "REFUTES"
            metadata = {"iteration": self.research_plan.iteration_count}
            if p_value is not None:
                metadata["p_value"] = p_value
            if effect_size is not None:
                metadata["effect_size"] = effect_size

            rel = Relationship.with_provenance(
                source_id=result_id,
                target_id=hypothesis_id,
                rel_type=rel_type,
                agent="DataAnalystAgent",
                confidence=confidence,
                **metadata
            )
            self.wm.add_relationship(rel)

            logger.debug(f"Added {rel_type} relationship: result {result_id} -> hypothesis {hypothesis_id}")

        except Exception as e:
            logger.warning(f"Failed to add {rel_type} relationship: {e}")

    # ========================================================================
    # MESSAGE HANDLING
    # ========================================================================

    async def process_message(self, message: AgentMessage):
        """
        Process incoming message from other agents.

        Routes messages to appropriate handlers based on source agent.
        """
        # Extract sender agent type from message metadata or from_agent
        sender_type = message.metadata.get("agent_type", "unknown")

        logger.debug(f"Processing message from {sender_type} ({message.from_agent})")

        # Route to appropriate handler
        if sender_type == "HypothesisGeneratorAgent":
            self._handle_hypothesis_generator_response(message)
        elif sender_type == "ExperimentDesignerAgent":
            self._handle_experiment_designer_response(message)
        elif sender_type == "Executor":
            self._handle_executor_response(message)
        elif sender_type == "DataAnalystAgent":
            self._handle_data_analyst_response(message)
        elif sender_type == "HypothesisRefiner":
            self._handle_hypothesis_refiner_response(message)
        elif sender_type == "ConvergenceDetector":
            self._handle_convergence_detector_response(message)
        else:
            logger.warning(f"No handler for agent type: {sender_type}")

    # ========================================================================
    # ERROR RECOVERY
    # ========================================================================

    def _handle_error_with_recovery(
        self,
        error_source: str,
        error_message: str,
        recoverable: bool = True,
        error_details: Optional[Dict[str, Any]] = None
    ) -> Optional[NextAction]:
        """
        Handle an error with recovery strategy.

        Implements:
        - Consecutive error counting
        - Exponential backoff
        - Circuit breaker (halt after MAX_CONSECUTIVE_ERRORS)
        - Error history tracking

        Args:
            error_source: Name of the agent/component that failed
            error_message: Human-readable error description
            recoverable: Whether this error type can be retried
            error_details: Additional error context

        Returns:
            NextAction if recovery possible, None if should abort current handler
        """
        import time

        # Update error tracking
        self.errors_encountered += 1
        self._consecutive_errors += 1
        self._last_error_time = datetime.now(timezone.utc)

        # Record in error history
        error_record = {
            'source': error_source,
            'message': error_message,
            'timestamp': self._last_error_time.isoformat(),
            'consecutive_count': self._consecutive_errors,
            'recoverable': recoverable,
            'details': error_details or {}
        }
        self._error_history.append(error_record)

        # Log the error
        logger.error(
            f"{ERROR_RECOVERY_LOG_PREFIX} {error_source}: {error_message} "
            f"(attempt {self._consecutive_errors}/{MAX_CONSECUTIVE_ERRORS})"
        )

        # Check if we've hit the circuit breaker threshold
        if self._consecutive_errors >= MAX_CONSECUTIVE_ERRORS:
            logger.error(
                f"{ERROR_RECOVERY_LOG_PREFIX} Max consecutive errors reached. "
                f"Transitioning to ERROR state."
            )

            with self._workflow_context():
                self.workflow.transition_to(
                    WorkflowState.ERROR,
                    action=f"Max errors exceeded: {error_message}",
                    metadata={'error_history': self._error_history[-MAX_CONSECUTIVE_ERRORS:]}
                )

            return NextAction.ERROR_RECOVERY

        # For recoverable errors, apply backoff and retry
        if recoverable:
            backoff_index = min(self._consecutive_errors - 1, len(ERROR_BACKOFF_SECONDS) - 1)
            backoff_seconds = ERROR_BACKOFF_SECONDS[backoff_index]

            logger.info(
                f"{ERROR_RECOVERY_LOG_PREFIX} Waiting {backoff_seconds}s before retry "
                f"(attempt {self._consecutive_errors + 1})"
            )

            # Plain blocking sleep. This is called synchronously from inside the
            # `except` block of async handlers, i.e. ON the event loop's own
            # thread.
            #
            # It previously did:
            #     loop = asyncio.get_running_loop()
            #     future = asyncio.run_coroutine_threadsafe(asyncio.sleep(b), loop)
            #     future.result(timeout=b + 5)
            # which SELF-DEADLOCKS. `run_coroutine_threadsafe` is for submitting
            # work from a *different* thread; scheduling onto the running loop
            # and then blocking that same thread on `.result()` means the loop
            # can never execute the sleep it was just handed. After b+5s the
            # future times out with `builtins.TimeoutError`, whose str() is ''.
            # And because TimeoutError subclasses OSError (not RuntimeError),
            # the fallback below never caught it -- so the error *recovery* path
            # raised out of every handler and killed the whole run with the
            # infamous "TimeoutError (no message)".
            #
            # Consequence: MAX_CONSECUTIVE_ERRORS and the [2,4,8] backoff ladder
            # were dead code, because attempt 1 always aborted the process.
            # Blocking here is what the original intent was anyway, and in the
            # CLI the director is the only task on the loop, so a 2-8s pause is
            # harmless. (Proper follow-up: make this method `async def` and
            # `await asyncio.sleep(...)`, updating its call sites.)
            time.sleep(backoff_seconds)

            # Re-evaluate what action to take
            return self.decide_next_action()

        # Non-recoverable error - just return None to exit handler
        logger.warning(
            f"{ERROR_RECOVERY_LOG_PREFIX} Non-recoverable error from {error_source}. "
            f"Skipping to next action."
        )
        return None

    def _reset_error_streak(self) -> None:
        """
        Reset consecutive error counter after successful operation.

        Call this at the end of each successful handler to reset the
        circuit breaker counter.
        """
        if self._consecutive_errors > 0:
            logger.debug(
                f"{ERROR_RECOVERY_LOG_PREFIX} Error streak reset "
                f"(was {self._consecutive_errors} consecutive errors)"
            )
        self._consecutive_errors = 0

    # ========================================================================
    # MESSAGE HANDLERS
    # ========================================================================

    def _handle_hypothesis_generator_response(self, message: AgentMessage):
        """
        Handle response from HypothesisGeneratorAgent.

        Expected content:
        - hypotheses: List of generated Hypothesis objects
        - count: Number of hypotheses generated
        """
        content = message.content

        if message.type == MessageType.ERROR:
            recovery_action = self._handle_error_with_recovery(
                error_source="HypothesisGeneratorAgent",
                error_message=content.get('error', 'Unknown error'),
                recoverable=True,
                error_details={'hypothesis_count_before': len(self.research_plan.hypothesis_pool)}
            )
            if recovery_action:
                self._execute_next_action(recovery_action)
            return

        # Success - reset error streak
        self._reset_error_streak()

        # Extract hypotheses
        hypothesis_ids = content.get("hypothesis_ids", [])
        count = content.get("count", 0)

        logger.info(f"Received {count} hypotheses from generator")

        # Update research plan (thread-safe)
        with self._research_plan_context():
            for hyp_id in hypothesis_ids:
                self.research_plan.add_hypothesis(hyp_id)

        # Persist hypotheses to knowledge graph
        for hyp_id in hypothesis_ids:
            self._persist_hypothesis_to_graph(hyp_id, agent_name="HypothesisGeneratorAgent")

        # Update strategy stats (thread-safe)
        with self._strategy_stats_context():
            self.strategy_stats["hypothesis_generation"]["attempts"] += 1
            if count > 0:
                self.strategy_stats["hypothesis_generation"]["successes"] += 1

        # Decide next action
        next_action = self.decide_next_action()
        self._execute_next_action(next_action)

    def _handle_experiment_designer_response(self, message: AgentMessage):
        """
        Handle response from ExperimentDesignerAgent.

        Expected content:
        - protocol_id: ID of designed experiment protocol
        - hypothesis_id: ID of hypothesis being tested
        """
        content = message.content

        if message.type == MessageType.ERROR:
            recovery_action = self._handle_error_with_recovery(
                error_source="ExperimentDesignerAgent",
                error_message=content.get('error', 'Unknown error'),
                recoverable=True,
                error_details={'untested_hypotheses': len(self.research_plan.get_untested_hypotheses())}
            )
            if recovery_action:
                self._execute_next_action(recovery_action)
            return

        # Success - reset error streak
        self._reset_error_streak()

        protocol_id = content.get("protocol_id")
        hypothesis_id = content.get("hypothesis_id")

        logger.info(f"Received experiment design: {protocol_id} for hypothesis {hypothesis_id}")

        # Update research plan (thread-safe)
        with self._research_plan_context():
            self.research_plan.add_experiment(protocol_id)

        # Persist protocol to knowledge graph
        if protocol_id and hypothesis_id:
            self._persist_protocol_to_graph(protocol_id, hypothesis_id, agent_name="ExperimentDesignerAgent")

        # Update strategy stats (thread-safe)
        with self._strategy_stats_context():
            self.strategy_stats["experiment_design"]["attempts"] += 1
            if protocol_id:
                self.strategy_stats["experiment_design"]["successes"] += 1

        # Decide next action
        next_action = self.decide_next_action()
        self._execute_next_action(next_action)

    def _handle_executor_response(self, message: AgentMessage):
        """
        Handle response from Executor.

        Expected content:
        - result_id: ID of experiment result
        - protocol_id: ID of protocol executed
        - status: SUCCESS/FAILURE/ERROR
        """
        content = message.content

        if message.type == MessageType.ERROR:
            recovery_action = self._handle_error_with_recovery(
                error_source="Executor",
                error_message=content.get('error', 'Unknown error'),
                recoverable=True,
                error_details={'experiments_queued': len(self.research_plan.experiment_queue)}
            )
            if recovery_action:
                self._execute_next_action(recovery_action)
            return

        # Success - reset error streak
        self._reset_error_streak()

        result_id = content.get("result_id")
        protocol_id = content.get("protocol_id")
        status = content.get("status")
        hypothesis_id = content.get("hypothesis_id")  # May not be present

        logger.info(f"Received experiment result: {result_id} (status: {status})")

        # Update research plan (thread-safe)
        with self._research_plan_context():
            self.research_plan.add_result(result_id)
            self.research_plan.mark_experiment_complete(protocol_id)

        # Persist result to knowledge graph (get hypothesis_id from protocol if needed)
        if result_id and protocol_id:
            if not hypothesis_id:
                # Fetch hypothesis_id from protocol
                try:
                    with get_session() as session:
                        protocol = get_experiment(session, protocol_id)
                        if protocol:
                            hypothesis_id = protocol.hypothesis_id
                except Exception as e:
                    logger.warning(f"Failed to fetch hypothesis_id from protocol: {e}")

            if hypothesis_id:
                self._persist_result_to_graph(result_id, protocol_id, hypothesis_id, agent_name="Executor")

        # Transition to analyzing state (thread-safe)
        with self._workflow_context():
            self.workflow.transition_to(
                WorkflowState.ANALYZING,
                action=f"Analyze result {result_id}"
            )

        # Send to DataAnalystAgent for interpretation
        next_action = NextAction.ANALYZE_RESULT
        self._execute_next_action(next_action)

    def _handle_data_analyst_response(self, message: AgentMessage):
        """
        Handle response from DataAnalystAgent.

        Expected content:
        - interpretation: ResultInterpretation object
        - result_id: ID of analyzed result
        - hypothesis_supported: bool
        """
        content = message.content

        if message.type == MessageType.ERROR:
            recovery_action = self._handle_error_with_recovery(
                error_source="DataAnalystAgent",
                error_message=content.get('error', 'Unknown error'),
                recoverable=True,
                error_details={'results_pending': len(self.research_plan.results)}
            )
            if recovery_action:
                self._execute_next_action(recovery_action)
            return

        # Success - reset error streak
        self._reset_error_streak()

        result_id = content.get("result_id")
        hypothesis_id = content.get("hypothesis_id")
        hypothesis_supported = content.get("hypothesis_supported")
        confidence = content.get("confidence", 0.8)  # Default confidence
        p_value = content.get("p_value")
        effect_size = content.get("effect_size")

        logger.info(
            f"Received result interpretation for {result_id}: "
            f"hypothesis {hypothesis_id} supported={hypothesis_supported}"
        )

        # Update hypothesis status in research plan (thread-safe)
        with self._research_plan_context():
            if hypothesis_supported is True:
                self.research_plan.mark_supported(hypothesis_id)
            elif hypothesis_supported is False:
                self.research_plan.mark_rejected(hypothesis_id)
            else:
                # Inconclusive
                self.research_plan.mark_tested(hypothesis_id)

        # Add SUPPORTS/REFUTES relationship to knowledge graph
        if result_id and hypothesis_id and hypothesis_supported is not None:
            self._add_support_relationship(
                result_id,
                hypothesis_id,
                supports=hypothesis_supported,
                confidence=confidence,
                p_value=p_value,
                effect_size=effect_size
            )

        # Transition to refining state (thread-safe)
        with self._workflow_context():
            self.workflow.transition_to(
                WorkflowState.REFINING,
                action=f"Refine based on result {result_id}"
            )

        # Decide next action (may refine hypothesis, generate new ones, or converge)
        next_action = self.decide_next_action()
        self._execute_next_action(next_action)

    def _handle_hypothesis_refiner_response(self, message: AgentMessage):
        """
        Handle response from HypothesisRefiner.

        Expected content:
        - refined_hypothesis_ids: List of refined/spawned hypothesis IDs
        - retired_hypothesis_ids: List of retired hypothesis IDs
        - action_taken: REFINED/RETIRED/SPAWNED
        """
        content = message.content

        if message.type == MessageType.ERROR:
            recovery_action = self._handle_error_with_recovery(
                error_source="HypothesisRefiner",
                error_message=content.get('error', 'Unknown error'),
                recoverable=True,
                error_details={
                    'tested_hypotheses': len(self.research_plan.tested_hypotheses),
                    'supported_hypotheses': len(self.research_plan.supported_hypotheses)
                }
            )
            if recovery_action:
                self._execute_next_action(recovery_action)
            return

        # Success - reset error streak
        self._reset_error_streak()

        refined_ids = content.get("refined_hypothesis_ids", [])
        retired_ids = content.get("retired_hypothesis_ids", [])

        logger.info(f"Hypothesis refinement: {len(refined_ids)} refined, {len(retired_ids)} retired")

        # Add refined hypotheses to pool (thread-safe)
        with self._research_plan_context():
            for hyp_id in refined_ids:
                self.research_plan.add_hypothesis(hyp_id)

        # Persist refined hypotheses to knowledge graph
        for hyp_id in refined_ids:
            self._persist_hypothesis_to_graph(hyp_id, agent_name="HypothesisRefiner")

        # Update strategy stats (thread-safe)
        with self._strategy_stats_context():
            self.strategy_stats["hypothesis_refinement"]["attempts"] += 1
            if refined_ids:
                self.strategy_stats["hypothesis_refinement"]["successes"] += 1

        # Decide next action
        next_action = self.decide_next_action()
        self._execute_next_action(next_action)

    def _handle_convergence_detector_response(self, message: AgentMessage):
        """
        Handle response from ConvergenceDetector.

        DEPRECATED (Issue #76): This method is no longer called.
        Convergence is now checked directly via _handle_convergence_action().
        Kept for backwards compatibility if message-based approach is reintroduced.

        Expected content:
        - should_converge: bool
        - reason: str (why convergence detected)
        - metrics: ConvergenceMetrics
        """
        content = message.content

        should_converge = content.get("should_converge", False)
        reason = content.get("reason", "")

        if should_converge:
            logger.info(f"Convergence detected: {reason}")

            # Update research plan (thread-safe)
            with self._research_plan_context():
                self.research_plan.has_converged = True
                self.research_plan.convergence_reason = reason

            # Add convergence annotation to research question in knowledge graph
            if self.wm and self.question_entity_id:
                try:
                    from kosmos.world_model.models import Annotation
                    convergence_annotation = Annotation(
                        text=f"Research converged: {reason}",
                        created_by="ConvergenceDetector"
                    )
                    self.wm.add_annotation(self.question_entity_id, convergence_annotation)
                    logger.debug("Added convergence annotation to research question")
                except Exception as e:
                    logger.warning(f"Failed to add convergence annotation: {e}")

            # Transition to converged state (thread-safe)
            with self._workflow_context():
                self.workflow.transition_to(
                    WorkflowState.CONVERGED,
                    action=f"Research converged: {reason}"
                )

            # Stop the director
            self.stop()
        else:
            logger.debug("Convergence check: not yet converged")

    # ========================================================================
    # MESSAGE SENDING (to other agents)
    # ========================================================================

    async def _send_to_hypothesis_generator(
        self,
        action: str,
        context: Optional[Dict[str, Any]] = None
    ) -> AgentMessage:
        """
        Send request to HypothesisGeneratorAgent asynchronously.

        Args:
            action: Action to request (generate, refine)
            context: Additional context (research_question, literature, etc.)

        Returns:
            AgentMessage: Sent message
        """
        content = {
            "action": action,
            "research_question": self.research_question,
            "domain": self.domain,
            "skills": self.get_skills_context(),  # Issue #51 - inject skills
            "context": context or {}
        }

        target_agent = self.agent_registry.get("HypothesisGeneratorAgent", "hypothesis_generator")

        message = await self.send_message(
            to_agent=target_agent,
            content=content,
            message_type=MessageType.REQUEST
        )

        self.pending_requests[message.id] = {
            "agent": "HypothesisGeneratorAgent",
            "action": action,
            "timestamp": datetime.now(timezone.utc)
        }

        # Track rollout (Issue #58)
        self.rollout_tracker.increment("hypothesis_generation")

        logger.debug(f"Sent {action} request to HypothesisGeneratorAgent")
        return message

    async def _send_to_experiment_designer(
        self,
        hypothesis_id: str,
        context: Optional[Dict[str, Any]] = None
    ) -> AgentMessage:
        """Send request to ExperimentDesignerAgent to design protocol asynchronously."""
        content = {
            "action": "design_experiment",
            "hypothesis_id": hypothesis_id,
            "domain": self.domain,
            "skills": self.get_skills_context(),  # Issue #51 - inject skills
            "context": context or {}
        }

        target_agent = self.agent_registry.get("ExperimentDesignerAgent", "experiment_designer")

        message = await self.send_message(
            to_agent=target_agent,
            content=content,
            message_type=MessageType.REQUEST
        )

        self.pending_requests[message.id] = {
            "agent": "ExperimentDesignerAgent",
            "hypothesis_id": hypothesis_id,
            "timestamp": datetime.now(timezone.utc)
        }

        # Track rollout (Issue #58)
        self.rollout_tracker.increment("experiment_design")

        logger.debug(f"Sent design request to ExperimentDesignerAgent for hypothesis {hypothesis_id}")
        return message

    async def _send_to_executor(
        self,
        protocol_id: str,
        context: Optional[Dict[str, Any]] = None
    ) -> AgentMessage:
        """Send request to Executor to run experiment asynchronously."""
        exec_context = context or {}
        if self.data_path:
            exec_context["data_path"] = self.data_path

        content = {
            "action": "execute_experiment",
            "protocol_id": protocol_id,
            "context": exec_context
        }

        target_agent = self.agent_registry.get("Executor", "executor")

        message = await self.send_message(
            to_agent=target_agent,
            content=content,
            message_type=MessageType.REQUEST
        )

        self.pending_requests[message.id] = {
            "agent": "Executor",
            "protocol_id": protocol_id,
            "timestamp": datetime.now(timezone.utc)
        }

        # Track rollout (Issue #58)
        self.rollout_tracker.increment("code_execution")

        logger.debug(f"Sent execution request to Executor for protocol {protocol_id}")
        return message

    async def _send_to_data_analyst(
        self,
        result_id: str,
        hypothesis_id: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> AgentMessage:
        """Send request to DataAnalystAgent to interpret results asynchronously."""
        content = {
            "action": "interpret_results",
            "result_id": result_id,
            "hypothesis_id": hypothesis_id,
            "context": context or {}
        }

        target_agent = self.agent_registry.get("DataAnalystAgent", "data_analyst")

        message = await self.send_message(
            to_agent=target_agent,
            content=content,
            message_type=MessageType.REQUEST
        )

        self.pending_requests[message.id] = {
            "agent": "DataAnalystAgent",
            "result_id": result_id,
            "timestamp": datetime.now(timezone.utc)
        }

        # Track rollout (Issue #58)
        self.rollout_tracker.increment("data_analysis")

        logger.debug(f"Sent interpretation request to DataAnalystAgent for result {result_id}")
        return message

    async def _send_to_hypothesis_refiner(
        self,
        hypothesis_id: str,
        result_id: Optional[str] = None,
        action: str = "evaluate",
        context: Optional[Dict[str, Any]] = None
    ) -> AgentMessage:
        """Send request to HypothesisRefiner asynchronously."""
        content = {
            "action": action,
            "hypothesis_id": hypothesis_id,
            "result_id": result_id,
            "context": context or {}
        }

        target_agent = self.agent_registry.get("HypothesisRefiner", "hypothesis_refiner")

        message = await self.send_message(
            to_agent=target_agent,
            content=content,
            message_type=MessageType.REQUEST
        )

        self.pending_requests[message.id] = {
            "agent": "HypothesisRefiner",
            "hypothesis_id": hypothesis_id,
            "timestamp": datetime.now(timezone.utc)
        }

        # Track rollout - refinement is part of hypothesis lifecycle (Issue #58)
        self.rollout_tracker.increment("hypothesis_generation")

        logger.debug(f"Sent {action} request to HypothesisRefiner for hypothesis {hypothesis_id}")
        return message

    def _check_convergence_direct(self) -> StoppingDecision:
        """
        Check convergence directly using ConvergenceDetector utility class.

        Issue #76 fix: ConvergenceDetector is not an agent that can receive messages.
        We call it directly instead of using message passing which silently failed.

        Returns:
            StoppingDecision: Decision on whether to stop research
        """
        # Get hypotheses and results from research plan
        # Note: For convergence checks, we primarily need counts, not full objects
        # The ConvergenceDetector's mandatory checks (iteration_limit, no_testable_hypotheses)
        # work with research_plan data, which already has what we need

        hypotheses = []
        results = []

        # Try to load actual hypothesis objects if available
        try:
            from kosmos.db import get_session
            from kosmos.db.models import HypothesisModel
            from kosmos.models.hypothesis import Hypothesis

            with get_session() as session:
                # Get hypotheses for this research (limited to avoid memory issues)
                hyp_ids = list(self.research_plan.hypothesis_pool)[:100]
                if hyp_ids:
                    db_hyps = session.query(HypothesisModel).filter(
                        HypothesisModel.id.in_(hyp_ids)
                    ).all()
                    hypotheses = [
                        Hypothesis(
                            id=h.id,
                            research_question=h.research_question or self.research_question,
                            statement=h.statement,
                            rationale=h.rationale or "",
                            domain=h.domain or self.domain or "general"
                        )
                        for h in db_hyps
                    ]
        except Exception as e:
            logger.debug(f"Could not load hypotheses for convergence check: {e}")

        # Perform convergence check (pass accumulated LLM cost if available)
        provider_cost = getattr(self.llm_client, 'total_cost_usd', None)
        decision = self.convergence_detector.check_convergence(
            research_plan=self.research_plan,
            hypotheses=hypotheses,
            results=results,
            total_cost=provider_cost
        )

        logger.info(f"[CONVERGENCE] Decision: should_stop={decision.should_stop}, reason={decision.reason.value}")

        return decision

    def _apply_multiple_comparison_correction(self):
        """
        Apply Benjamini-Hochberg FDR correction to p-values from the current iteration.

        When multiple hypotheses produce multiple p-values in the same iteration,
        each must be corrected for family-wise error to avoid inflated Type I error.
        """
        from kosmos.execution.statistics import StatisticalValidator
        from kosmos.db.operations import get_results_for_experiment

        current_iteration = self.research_plan.iteration_count

        # Collect p-values from recent results in DB
        results_with_pvalues = []
        try:
            with get_session() as session:
                for exp_id in self.research_plan.completed_experiments[-20:]:
                    try:
                        db_results = get_results_for_experiment(session, exp_id)
                        for db_r in db_results:
                            if db_r.p_value is not None:
                                results_with_pvalues.append({
                                    'result_id': db_r.id,
                                    'p_value': db_r.p_value,
                                    'supports_hypothesis': db_r.supports_hypothesis,
                                })
                    except Exception:
                        continue
        except Exception as e:
            logger.warning(f"Failed to collect p-values for correction: {e}")
            return

        if len(results_with_pvalues) < 2:
            return  # No correction needed for single test

        p_values = [r['p_value'] for r in results_with_pvalues]
        correction = StatisticalValidator.benjamini_hochberg_fdr(p_values)

        corrections_applied = 0
        for i, result_info in enumerate(results_with_pvalues):
            was_sig = result_info['p_value'] < 0.05
            now_sig = correction['significant'][i]
            if was_sig and not now_sig:
                corrections_applied += 1
                logger.info(
                    f"[FDR] Multiple comparison correction: result {result_info['result_id']} "
                    f"p={result_info['p_value']:.4f} no longer significant after BH-FDR "
                    f"(adjusted={correction['adjusted_p_values'][i]:.4f})"
                )

        if corrections_applied > 0:
            logger.info(
                f"[FDR] BH-FDR correction: {corrections_applied}/{len(p_values)} "
                f"results lost significance"
            )

    async def _handle_convergence_action(self):
        """
        Handle CONVERGE action by checking convergence directly.

        Issue #76 fix: Replaces message-based convergence check with direct call.
        The ConvergenceDetector is a utility class, not an agent that can receive messages.
        """
        # Apply multiple comparison correction before checking convergence
        try:
            self._apply_multiple_comparison_correction()
        except Exception as e:
            logger.warning(f"Multiple comparison correction failed (non-fatal): {e}")

        decision = self._check_convergence_direct()

        # Track rollout - convergence often involves literature review (Issue #58)
        self.rollout_tracker.increment("literature")

        if decision.should_stop:
            logger.info(f"Convergence detected: {decision.reason.value}")

            # Update research plan (thread-safe)
            with self._research_plan_context():
                self.research_plan.has_converged = True
                self.research_plan.convergence_reason = decision.reason.value

            # Add convergence annotation to research question in knowledge graph
            if self.wm and self.question_entity_id:
                try:
                    from kosmos.world_model.models import Annotation
                    convergence_annotation = Annotation(
                        text=f"Research converged: {decision.reason.value}",
                        created_by="ConvergenceDetector"
                    )
                    self.wm.add_annotation(self.question_entity_id, convergence_annotation)
                    logger.debug("Added convergence annotation to research question")
                except Exception as e:
                    logger.warning(f"Failed to add convergence annotation: {e}")

            # Transition to converged state (thread-safe)
            with self._workflow_context():
                self.workflow.transition_to(
                    WorkflowState.CONVERGED,
                    action=f"Research converged: {decision.reason.value}"
                )

            # Stop the director
            self.stop()
        else:
            logger.debug(f"Convergence check: not yet converged ({decision.details})")
            # Continue research - increment iteration if we've completed a full cycle
            if self._actions_this_iteration > 0:
                with self._research_plan_context():
                    self.research_plan.increment_iteration()
                self._actions_this_iteration = 0
                logger.info(f"[ITERATION] Continuing to iteration {self.research_plan.iteration_count}")

    async def _handle_generate_hypothesis_action(self):
        """
        Handle GENERATE_HYPOTHESIS action by calling HypothesisGeneratorAgent directly.

        Issue #76 extension: Replaces message-based hypothesis generation with direct call.
        Same pattern as _handle_convergence_action() — agents are not registered in the
        message router, so send_message() silently fails.
        """
        from kosmos.agents.hypothesis_generator import HypothesisGeneratorAgent

        try:
            # Lazy-init the agent
            if self._hypothesis_agent is None:
                self._hypothesis_agent = HypothesisGeneratorAgent(config=self.config)

            logger.info("Generating hypotheses via direct call (bypassing message router)")

            # Ground hypotheses in the dataset (schema + basic summary) when one
            # is available -- whether from --data-path or an AutoEvidence capsule.
            # This is what makes a data-driven (question-light) run meaningful.
            data_context = self._build_data_context()
            if data_context:
                logger.info("Hypothesis generation is data-driven (dataset schema supplied)")

            response = self._hypothesis_agent.generate_hypotheses(
                research_question=self.research_question,
                num_hypotheses=self.config.get("num_hypotheses", 3),
                domain=self.domain,
                store_in_db=True,
                data_context=data_context,
            )

            # Track rollout (Issue #58)
            self.rollout_tracker.increment("hypothesis_generation")

            hypothesis_ids = [h.id for h in response.hypotheses]
            count = len(hypothesis_ids)

            logger.info(f"Generated {count} hypotheses via direct call")

            # Reset error streak on success
            self._reset_error_streak()

            # Update research plan (thread-safe)
            with self._research_plan_context():
                for hyp_id in hypothesis_ids:
                    self.research_plan.add_hypothesis(hyp_id)

            # Persist hypotheses to knowledge graph
            for hyp_id in hypothesis_ids:
                self._persist_hypothesis_to_graph(hyp_id, agent_name="HypothesisGeneratorAgent")

            # Update strategy stats (thread-safe)
            with self._strategy_stats_context():
                self.strategy_stats["hypothesis_generation"]["attempts"] += 1
                if count > 0:
                    self.strategy_stats["hypothesis_generation"]["successes"] += 1

            # Transition workflow to DESIGNING_EXPERIMENTS on success
            if count > 0:
                with self._workflow_context():
                    self.workflow.transition_to(
                        WorkflowState.DESIGNING_EXPERIMENTS,
                        action=f"Generated {count} hypotheses"
                    )

        except Exception as e:
            logger.error(f"Direct hypothesis generation failed: {e}", exc_info=True)
            self._handle_error_with_recovery(
                error_source="HypothesisGeneratorAgent",
                error_message=str(e),
                recoverable=True,
                error_details={"hypothesis_count_before": len(self.research_plan.hypothesis_pool)}
            )

    async def _handle_design_experiment_action(self, hypothesis_id: str):
        """
        Handle DESIGN_EXPERIMENT action by calling ExperimentDesignerAgent directly.

        Issue #76 extension: Replaces message-based experiment design with direct call.
        Same pattern as _handle_convergence_action().
        """
        from kosmos.agents.experiment_designer import ExperimentDesignerAgent

        try:
            # Lazy-init the agent
            if self._experiment_designer is None:
                self._experiment_designer = ExperimentDesignerAgent(config=self.config)

            logger.info(f"Designing experiment for hypothesis {hypothesis_id} via direct call")

            response = self._experiment_designer.design_experiment(
                hypothesis_id=hypothesis_id,
                store_in_db=True
            )

            # Track rollout (Issue #58)
            self.rollout_tracker.increment("experiment_design")

            protocol_id = response.protocol.id if response.protocol else None

            logger.info(f"Designed experiment protocol {protocol_id} for hypothesis {hypothesis_id}")

            # Reset error streak on success
            self._reset_error_streak()

            # Update research plan (thread-safe)
            if protocol_id:
                with self._research_plan_context():
                    self.research_plan.add_experiment(protocol_id)

            # Persist protocol to knowledge graph
            if protocol_id and hypothesis_id:
                self._persist_protocol_to_graph(protocol_id, hypothesis_id, agent_name="ExperimentDesignerAgent")

            # Update strategy stats (thread-safe)
            with self._strategy_stats_context():
                self.strategy_stats["experiment_design"]["attempts"] += 1
                if protocol_id:
                    self.strategy_stats["experiment_design"]["successes"] += 1

            # Transition workflow to EXECUTING on success
            if protocol_id:
                with self._workflow_context():
                    self.workflow.transition_to(
                        WorkflowState.EXECUTING,
                        action=f"Designed experiment {protocol_id} for hypothesis {hypothesis_id}"
                    )

        except Exception as e:
            logger.error(f"Direct experiment design failed: {e}", exc_info=True)
            self._handle_error_with_recovery(
                error_source="ExperimentDesignerAgent",
                error_message=str(e),
                recoverable=True,
                error_details={"untested_hypotheses": len(self.research_plan.get_untested_hypotheses())}
            )

    async def _handle_execute_experiment_action(self, protocol_id: str):
        """
        Handle EXECUTE_EXPERIMENT action by running code generation + execution directly.

        Issue #76 extension: Replaces message-based _send_to_executor() with direct call.
        Replicates the logic from _handle_executor_response().
        """
        from kosmos.execution.code_generator import ExperimentCodeGenerator
        from kosmos.execution.executor import CodeExecutor
        from kosmos.execution.data_provider import DataProvider
        from kosmos.models.experiment import ExperimentProtocol
        from kosmos.db.operations import create_result
        from uuid import uuid4

        try:
            # Lazy-init components
            if self._code_generator is None:
                self._code_generator = ExperimentCodeGenerator(use_templates=True, use_llm=True)
            if self._code_executor is None:
                self._code_executor = CodeExecutor(max_retries=3)
            if self._data_provider is None:
                self._data_provider = DataProvider(
                    default_data_dir=self.data_path
                )

            logger.info(f"Executing experiment {protocol_id} via direct call")

            # Load protocol from DB
            hypothesis_id = None
            protocol = None
            with get_session() as session:
                db_experiment = get_experiment(session, protocol_id)
                if not db_experiment:
                    raise ValueError(f"Experiment {protocol_id} not found in database")
                hypothesis_id = db_experiment.hypothesis_id
                # Reconstruct Pydantic model from stored protocol JSON
                protocol_data = db_experiment.protocol
                if isinstance(protocol_data, dict):
                    protocol = ExperimentProtocol.model_validate(protocol_data)
                else:
                    raise ValueError(f"Experiment {protocol_id} has no valid protocol data")

            # Which datasets may this experiment open? Empty for a single-source
            # run, and empty whenever the join-key rule refuses -- in both cases
            # everything below is the single-dataset path unchanged.
            mount = self._mountable_datasets()
            self._code_generator.datasets = mount
            # The single-dataset path needs this just as much as the federated
            # one, and used to get None -- so the model wrote code against
            # columns it GUESSED. On GSE2240 it assumed a tidy long table
            # (`gene`, `sample`, `expression`, `fibrosis_status`, `study`)
            # against a wide 22,283 x 36 expression matrix and died on its own
            # column check, having never read a byte. One dataset is not a
            # reason to describe nothing; it is only a reason to describe it as
            # `data_path` rather than as an entry in `datasets`.
            if len(mount) > 1:
                self._code_generator.dataset_context = self._staged_file_context(mount)
            else:
                import os as _os

                lone = dict(mount) if mount else (
                    {
                        str(getattr(self, "evidence_dataset", None) or "")
                        or _os.path.splitext(_os.path.basename(self.data_path))[0]:
                        self.data_path
                    }
                    if getattr(self, "data_path", None) else {}
                )
                self._code_generator.dataset_context = (
                    self._staged_file_context(lone, single=True) if lone else None
                )

            # Generate code from protocol
            code = self._code_generator.generate(protocol)

            def _run(source: str):
                if self.data_path:
                    return self._code_executor.execute_with_data(
                        source, self.data_path, retry_on_error=True,
                        data_files=mount or None,
                    )
                return self._code_executor.execute(source, retry_on_error=True)

            _ERROR_ONLY_KEYS = {
                "error", "errors", "message", "traceback", "analysis_note",
            }

            def _payload(res) -> dict:
                """The experiment's actual output, or {} if it produced none.

                A dict whose only content is an error message is not a result.
                One run returned exactly `{"error": "Gene symbol-to-Ensembl
                mapping required (mygene) not available."}` and was recorded as
                a SUCCESSFUL experiment with no p-value, no effect size and
                nothing to report -- the failure dressed as an answer.
                """
                rv = getattr(res, "return_value", None)
                if not isinstance(rv, dict) or not rv:
                    return {}
                if not (set(rv) - _ERROR_ONLY_KEYS):
                    return {}
                return rv

            exec_result = _run(code)

            # Regenerate ONCE with the failure quoted back.
            #
            # The executor's own `retry_on_error` re-runs the SAME source, which
            # cannot fix anything the code got wrong: three attempts at
            # `KeyError: 'beta_t1'` produce three identical KeyErrors, and three
            # at `from scipy.stats import multipletests` three identical
            # ImportErrors. Every runtime failure this pipeline has hit --
            # NameError on a misspelled helper, AttributeError on an invented
            # method, KeyError on a column the merge never created, an import
            # from the wrong module -- is a mistake the model corrects readily
            # once shown the traceback. It cannot see the traceback unless we
            # hand it over.
            # Two failures, one remedy. An exception is the obvious one. The
            # other is a script that runs to completion, raises nothing, and
            # assigns no non-empty `results` at module level -- 16 seconds of
            # real work, `data: {}`, and nothing to report. Triggering only on
            # exceptions left that case with no second draft even though it is
            # exactly as correctable: the model simply has to be told that its
            # output was never collected.
            _ran_ok = bool(getattr(exec_result, "success", True))
            if not _ran_ok or not _payload(exec_result):
                error_text = str(getattr(exec_result, "error", "") or "")[:1200]
                if _ran_ok:
                    error_text = (
                        "the script ran to completion and raised nothing, but it "
                        "assigned no non-empty `results` dict at MODULE level, so "
                        "nothing could be collected from it. `results` must be a "
                        "module-level dict holding the numbers the protocol asks "
                        "for. If the analysis legitimately found nothing -- no "
                        "instrument passed the threshold, no variant survived the "
                        "join -- that is a FINDING: put the counts and a short "
                        "explanation in `results` rather than leaving it empty."
                    )
                if error_text:
                    logger.warning(
                        "Experiment %s failed at runtime (%s); regenerating once "
                        "with the error quoted back",
                        protocol_id, error_text.splitlines()[0],
                    )
                    try:
                        retry_code = self._code_generator.generate(
                            protocol, runtime_error=error_text
                        )
                        if retry_code and retry_code != code:
                            code = retry_code
                            exec_result = _run(code)
                    except Exception as regen_error:
                        # A failed regeneration must not replace the real
                        # failure with its own.
                        logger.warning(
                            "Regeneration after runtime failure failed: %s",
                            regen_error,
                        )

            # Track rollout (Issue #58)
            self.rollout_tracker.increment("experiment_execution")
            logger.info(
                "[EXEC-DONE] exp=%s success=%s rv_type=%s",
                protocol_id, getattr(exec_result, "success", None),
                type(getattr(exec_result, "return_value", None)).__name__,
            )

            # Extract metrics from execution result. Use getattr + isinstance to
            # avoid ambiguous-truthiness errors on numpy/DataFrame return values.
            _rv = getattr(exec_result, "return_value", None)
            return_value = _rv if isinstance(_rv, dict) else {}
            if isinstance(return_value, dict):
                p_value = return_value.get("p_value")
                effect_size = return_value.get("effect_size")
                statistical_tests = {
                    k: v for k, v in return_value.items()
                    if k in ("t_statistic", "p_value", "effect_size",
                             "mean_difference", "significance_label",
                             "correlation", "r_squared")
                }
            else:
                p_value = None
                effect_size = None
                statistical_tests = {}

            # Sanitize return_value for JSON serialization — filter out non-serializable
            # objects (e.g. sklearn Pipeline, numpy arrays) that would crash DB insert
            def _json_safe(obj):
                if obj is None or isinstance(obj, (str, int, float, bool)):
                    return obj
                if isinstance(obj, (list, tuple)):
                    return [_json_safe(v) for v in obj]
                if isinstance(obj, dict):
                    return {k: _json_safe(v) for k, v in obj.items()}
                try:
                    import numpy as np
                    if isinstance(obj, (np.integer,)):
                        return int(obj)
                    if isinstance(obj, (np.floating,)):
                        return float(obj)
                    if isinstance(obj, np.ndarray):
                        return obj.tolist()
                except ImportError:
                    pass
                # Fallback: convert to string representation
                return str(obj)

            def _as_float(value):
                """A scalar, or None. Never raises."""
                try:
                    return float(value) if value is not None else None
                except (TypeError, ValueError):
                    logger.warning(
                        "Non-scalar statistic %r kept out of the Result columns; "
                        "it stays in the results payload.", value,
                    )
                    return None

            safe_data = _json_safe(return_value) if isinstance(return_value, dict) else {}
            safe_stats = _json_safe(statistical_tests) if statistical_tests else {}

            # If code generation fell back, the protocol above no longer
            # describes what ran. Record that with the result, where the report
            # renders it beside the numbers -- otherwise a template's output is
            # narrated under the multi-dataset design it replaced.
            _note = self._code_generator.generation_note()
            if _note:
                logger.warning("Experiment %s: %s", protocol_id, _note)
                safe_data["analysis_note"] = _note

            # Store result in DB
            result_id = str(uuid4())
            try:
                with get_session() as session:
                    create_result(
                        session,
                        id=result_id,
                        experiment_id=protocol_id,
                        data=safe_data,
                        # Coerced BEFORE the insert, not inside it: a p_value
                        # that is a list, a numpy array or a dict raises in
                        # float(), and inside the try that discarded the ENTIRE
                        # Result row -- data, statistical tests and all -- while
                        # the experiment was still recorded as COMPLETED.
                        p_value=_as_float(p_value),
                        effect_size=_as_float(effect_size),
                        statistical_tests=safe_stats,
                    )
            except Exception as db_err:
                logger.error(f"Failed to store result in DB: {db_err}")

            # Persist experiment status to the DB. Previously only the in-memory
            # research plan was updated (mark_experiment_complete below), so the DB
            # row stayed CREATED and runs reported 0 successful experiments.
            try:
                from kosmos.db.operations import update_experiment_status
                from kosmos.db.models import ExperimentStatus as _ExpStatus
                # Two conditions, not one. The executor reports success when
                # the code RAN, which is not the same as the code having
                # produced anything: a run that returned an empty payload was
                # recorded COMPLETED, counted as a successful experiment, and
                # printed "Research completed successfully!" over `data: {}`.
                # An experiment that yields no output, no p-value and no test
                # statistic has not succeeded -- and this is deliberately NOT a
                # test of significance. A null result carries a p-value and
                # stays a success; what fails here is the absence of any output
                # at all.
                _ran = bool(getattr(exec_result, "success", True))
                _produced = bool(
                    {k: v for k, v in safe_data.items() if k != "analysis_note"}
                    or safe_stats
                    or p_value is not None
                    or effect_size is not None
                )
                _ok = _ran and _produced
                if _ran and not _produced:
                    _err = (
                        "experiment executed but produced no output: no results "
                        "payload, no p-value, no effect size, no test statistic"
                    )
                    logger.error("Experiment %s: %s", protocol_id, _err)
                else:
                    _err = None if _ok else getattr(exec_result, "error", None)
                with get_session() as session:
                    update_experiment_status(
                        session,
                        protocol_id,
                        _ExpStatus.COMPLETED if _ok else _ExpStatus.FAILED,
                        error_message=_err,
                        execution_time_seconds=getattr(exec_result, "execution_time", None),
                    )
            except Exception as st_err:
                logger.error(f"Failed to update experiment status: {st_err}")

            logger.info(f"Experiment {protocol_id} executed, result {result_id}")

            # Reset error streak on success
            self._reset_error_streak()

            # Update research plan (thread-safe)
            with self._research_plan_context():
                self.research_plan.add_result(result_id)
                self.research_plan.mark_experiment_complete(protocol_id)

            # Persist result to knowledge graph
            if hypothesis_id:
                self._persist_result_to_graph(
                    result_id, protocol_id, hypothesis_id, agent_name="CodeExecutor"
                )

            # Transition to analyzing state (thread-safe)
            with self._workflow_context():
                self.workflow.transition_to(
                    WorkflowState.ANALYZING,
                    action=f"Analyze result {result_id}"
                )

        except Exception as e:
            logger.error(f"Direct experiment execution failed: {e}", exc_info=True)

            # Quarantine the failed protocol.
            #
            # `mark_experiment_complete` is the ONLY thing that pops
            # experiment_queue, and every one of its callers sits on a success
            # path. The sequential executor reads `experiment_queue[0]`, so a
            # protocol that raises stayed at the head of the queue and was
            # retried identically forever -- same DB row, same template, same
            # broken code, with no LLM resampling to rescue it. It also blocked
            # the graceful exit, since convergence defers while the queue is
            # non-empty. Dropping it lets the next protocol run and lets the
            # session end; the DB row is already marked FAILED, so the record
            # keeps the failure.
            # Mark the DB row FAILED and remember the id. `update_experiment_status`
            # runs only on the success path, so a failure raised inside execution
            # leaves the row CREATED with no error message; and the protocol is
            # dropped from experiment_queue without ever reaching
            # completed_experiments, so it lands in NO list and vanishes from the
            # report -- which is how a Docker outage read as "no experiment was
            # designed". `failed_experiments` is the list run.py reads for exactly
            # this case, and nothing was writing it.
            try:
                from kosmos.db.operations import update_experiment_status
                from kosmos.db.models import ExperimentStatus as _ExpStatus
                with get_session() as session:
                    update_experiment_status(
                        session, protocol_id, _ExpStatus.FAILED,
                        error_message=str(e) or type(e).__name__,
                    )
            except Exception as status_error:
                logger.error(
                    "Could not mark protocol %s FAILED: %s", protocol_id, status_error,
                )

            try:
                with self._research_plan_context():
                    if protocol_id not in self.research_plan.failed_experiments:
                        self.research_plan.failed_experiments.append(protocol_id)
                    if protocol_id in self.research_plan.experiment_queue:
                        self.research_plan.experiment_queue.remove(protocol_id)
                        logger.warning(
                            "Quarantined failed protocol %s: removed from the "
                            "experiment queue so it cannot block the head.",
                            protocol_id,
                        )
            except Exception as dequeue_error:
                # Cleanup must never mask the original failure.
                logger.error(
                    "Could not dequeue failed protocol %s: %s",
                    protocol_id, dequeue_error,
                )

            self._handle_error_with_recovery(
                error_source="CodeExecutor",
                error_message=str(e),
                recoverable=True,
                error_details={"protocol_id": protocol_id}
            )

    async def _handle_analyze_result_action(self, result_id: str):
        """
        Handle ANALYZE_RESULT action by calling DataAnalystAgent directly.

        Issue #76 extension: Replaces message-based _send_to_data_analyst() with direct call.
        Replicates the logic from _handle_data_analyst_response().
        """
        from kosmos.agents.data_analyst import DataAnalystAgent
        from kosmos.db.operations import get_result as db_get_result
        from kosmos.models.result import ExperimentResult, ResultStatus, ExecutionMetadata
        from kosmos.models.hypothesis import Hypothesis as PydanticHypothesis

        try:
            # Lazy-init the agent
            if self._data_analyst is None:
                self._data_analyst = DataAnalystAgent(config=self.config)

            logger.info(f"Analyzing result {result_id} via direct call")

            # Load result from DB
            hypothesis_id = None
            pydantic_result = None
            pydantic_hyp = None

            with get_session() as session:
                db_result = db_get_result(session, result_id, with_experiment=True)
                if not db_result:
                    raise ValueError(f"Result {result_id} not found in database")

                # Get hypothesis_id through experiment
                experiment = db_result.experiment
                if experiment:
                    hypothesis_id = experiment.hypothesis_id

                # Build minimal ExperimentResult from DB fields
                import sys as _sys
                import platform as _platform
                _now = datetime.now(timezone.utc)
                pydantic_result = ExperimentResult(
                    id=db_result.id,
                    experiment_id=db_result.experiment_id,
                    protocol_id=db_result.experiment_id,
                    status=ResultStatus.SUCCESS,
                    raw_data=db_result.data or {},
                    primary_p_value=db_result.p_value,
                    primary_effect_size=db_result.effect_size,
                    supports_hypothesis=db_result.supports_hypothesis,
                    metadata=ExecutionMetadata(
                        start_time=_now,
                        end_time=_now,
                        duration_seconds=0.0,
                        python_version=_sys.version,
                        platform=_platform.platform(),
                        experiment_id=db_result.experiment_id,
                        protocol_id=db_result.experiment_id,
                    ),
                )

                # Load hypothesis if available
                if hypothesis_id:
                    db_hyp = get_hypothesis(session, hypothesis_id)
                    if db_hyp:
                        pydantic_hyp = PydanticHypothesis(
                            id=db_hyp.id,
                            research_question=db_hyp.research_question or self.research_question,
                            statement=db_hyp.statement,
                            rationale=db_hyp.rationale or "",
                            domain=db_hyp.domain or self.domain or "general",
                        )

            # Call data analyst
            interpretation = self._data_analyst.interpret_results(
                result=pydantic_result,
                hypothesis=pydantic_hyp,
            )

            # Track rollout (Issue #58)
            self.rollout_tracker.increment("data_analysis")

            # Extract interpretation fields
            hypothesis_supported = interpretation.hypothesis_supported
            confidence = interpretation.confidence if interpretation.confidence else 0.8
            p_value = pydantic_result.primary_p_value
            effect_size = pydantic_result.primary_effect_size

            logger.info(
                f"Result {result_id} interpretation: "
                f"hypothesis {hypothesis_id} supported={hypothesis_supported}"
            )

            # Reset error streak on success
            self._reset_error_streak()

            # Update hypothesis status in research plan (thread-safe)
            if hypothesis_id:
                with self._research_plan_context():
                    if hypothesis_supported is True:
                        self.research_plan.mark_supported(hypothesis_id)
                    elif hypothesis_supported is False:
                        self.research_plan.mark_rejected(hypothesis_id)
                    else:
                        self.research_plan.mark_tested(hypothesis_id)

            # Add SUPPORTS/REFUTES relationship to knowledge graph
            if result_id and hypothesis_id and hypothesis_supported is not None:
                self._add_support_relationship(
                    result_id,
                    hypothesis_id,
                    supports=hypothesis_supported,
                    confidence=confidence,
                    p_value=p_value,
                    effect_size=effect_size,
                )

            # Transition to refining state (thread-safe)
            with self._workflow_context():
                self.workflow.transition_to(
                    WorkflowState.REFINING,
                    action=f"Refine based on result {result_id}"
                )

        except Exception as e:
            logger.error(f"Direct result analysis failed: {e}", exc_info=True)
            self._handle_error_with_recovery(
                error_source="DataAnalystAgent",
                error_message=str(e),
                recoverable=True,
                error_details={"result_id": result_id}
            )

    async def _handle_refine_hypothesis_action(self, hypothesis_id: str):
        """
        Handle REFINE_HYPOTHESIS action by calling HypothesisRefiner directly.

        Issue #76 extension: Replaces message-based _send_to_hypothesis_refiner()
        with direct call. Replicates _handle_hypothesis_refiner_response().

        HypothesisRefiner is a utility class (not a BaseAgent), same as ConvergenceDetector.
        """
        from kosmos.hypothesis.refiner import HypothesisRefiner, RetirementDecision
        from kosmos.db.operations import (
            get_hypothesis as db_get_hypothesis,
            get_results_for_experiment,
            create_hypothesis as db_create_hypothesis,
        )
        from kosmos.models.hypothesis import Hypothesis as PydanticHypothesis
        from kosmos.models.result import ExperimentResult, ResultStatus, ExecutionMetadata

        try:
            # Lazy-init refiner
            if self._hypothesis_refiner is None:
                self._hypothesis_refiner = HypothesisRefiner(config=self.config)

            logger.info(f"Refining hypothesis {hypothesis_id} via direct call")

            # Load hypothesis from DB
            pydantic_hyp = None
            results_history = []

            with get_session() as session:
                db_hyp = db_get_hypothesis(session, hypothesis_id, with_experiments=True)
                if not db_hyp:
                    logger.warning(f"Hypothesis {hypothesis_id} not found, skipping refinement")
                    return

                pydantic_hyp = PydanticHypothesis(
                    id=db_hyp.id,
                    research_question=db_hyp.research_question or self.research_question,
                    statement=db_hyp.statement,
                    rationale=db_hyp.rationale or "",
                    domain=db_hyp.domain or self.domain or "general",
                    novelty_score=db_hyp.novelty_score,
                    testability_score=db_hyp.testability_score,
                    confidence_score=db_hyp.confidence_score,
                )

                # Get results for this hypothesis's experiments
                if hasattr(db_hyp, 'experiments') and db_hyp.experiments:
                    for exp in db_hyp.experiments:
                        db_results = get_results_for_experiment(session, exp.id)
                        for db_r in db_results:
                            try:
                                import sys as _sys
                                import platform as _platform
                                _now = datetime.now(timezone.utc)
                                pydantic_r = ExperimentResult(
                                    id=db_r.id,
                                    experiment_id=db_r.experiment_id,
                                    protocol_id=db_r.experiment_id,
                                    status=ResultStatus.SUCCESS,
                                    raw_data=db_r.data or {},
                                    primary_p_value=db_r.p_value,
                                    primary_effect_size=db_r.effect_size,
                                    supports_hypothesis=db_r.supports_hypothesis,
                                    metadata=ExecutionMetadata(
                                        start_time=_now,
                                        end_time=_now,
                                        duration_seconds=0.0,
                                        python_version=_sys.version,
                                        platform=_platform.platform(),
                                        experiment_id=db_r.experiment_id,
                                        protocol_id=db_r.experiment_id,
                                    ),
                                )
                                results_history.append(pydantic_r)
                            except Exception as conv_err:
                                logger.warning(f"Failed to convert result {db_r.id}: {conv_err}")

            # Get latest result (for evaluate_hypothesis_status)
            latest_result = results_history[-1] if results_history else None

            if not latest_result:
                logger.warning(f"No results found for hypothesis {hypothesis_id}, skipping refinement")
                # Still update stats
                with self._strategy_stats_context():
                    self.strategy_stats["hypothesis_refinement"]["attempts"] += 1
                self.rollout_tracker.increment("hypothesis_refinement")
                return

            # Evaluate hypothesis status
            decision = self._hypothesis_refiner.evaluate_hypothesis_status(
                pydantic_hyp, latest_result, results_history
            )

            refined_ids = []
            retired_ids = []

            if decision == RetirementDecision.RETIRE:
                self._hypothesis_refiner.retire_hypothesis(
                    pydantic_hyp, rationale="Retirement based on result evaluation"
                )
                retired_ids.append(hypothesis_id)
                logger.info(f"Retired hypothesis {hypothesis_id}")

            elif decision == RetirementDecision.REFINE:
                refined = self._hypothesis_refiner.refine_hypothesis(pydantic_hyp, latest_result)
                if refined and refined.id:
                    # Store refined hypothesis in DB
                    try:
                        with get_session() as session:
                            db_create_hypothesis(
                                session,
                                id=refined.id,
                                research_question=refined.research_question,
                                statement=refined.statement,
                                rationale=refined.rationale,
                                domain=refined.domain,
                                novelty_score=refined.novelty_score,
                                testability_score=refined.testability_score,
                            )
                    except Exception as store_err:
                        logger.warning(f"Failed to store refined hypothesis: {store_err}")
                    refined_ids.append(refined.id)
                    logger.info(f"Refined hypothesis {hypothesis_id} -> {refined.id}")

            elif decision == RetirementDecision.SPAWN_VARIANT:
                variants = self._hypothesis_refiner.spawn_variant(
                    pydantic_hyp, latest_result, num_variants=2
                )
                for variant in variants:
                    if variant and variant.id:
                        try:
                            with get_session() as session:
                                db_create_hypothesis(
                                    session,
                                    id=variant.id,
                                    research_question=variant.research_question,
                                    statement=variant.statement,
                                    rationale=variant.rationale,
                                    domain=variant.domain,
                                    novelty_score=variant.novelty_score,
                                    testability_score=variant.testability_score,
                                )
                        except Exception as store_err:
                            logger.warning(f"Failed to store variant hypothesis: {store_err}")
                        refined_ids.append(variant.id)
                logger.info(f"Spawned {len(refined_ids)} variants from {hypothesis_id}")

            # decision == CONTINUE_TESTING: no action needed

            # Track rollout
            self.rollout_tracker.increment("hypothesis_refinement")

            # Reset error streak on success
            self._reset_error_streak()

            # Add refined hypotheses to pool (thread-safe)
            with self._research_plan_context():
                for hyp_id in refined_ids:
                    self.research_plan.add_hypothesis(hyp_id)

            # Persist refined hypotheses to knowledge graph
            for hyp_id in refined_ids:
                self._persist_hypothesis_to_graph(hyp_id, agent_name="HypothesisRefiner")

            # Update strategy stats (thread-safe)
            with self._strategy_stats_context():
                self.strategy_stats["hypothesis_refinement"]["attempts"] += 1
                if refined_ids:
                    self.strategy_stats["hypothesis_refinement"]["successes"] += 1

            logger.info(f"Hypothesis refinement: {len(refined_ids)} refined, {len(retired_ids)} retired")

        except Exception as e:
            logger.error(f"Direct hypothesis refinement failed: {e}", exc_info=True)
            self._handle_error_with_recovery(
                error_source="HypothesisRefiner",
                error_message=str(e),
                recoverable=True,
                error_details={
                    "hypothesis_id": hypothesis_id,
                    "tested_hypotheses": len(self.research_plan.tested_hypotheses),
                }
            )

    async def _send_to_convergence_detector(
        self,
        context: Optional[Dict[str, Any]] = None
    ) -> AgentMessage:
        """
        DEPRECATED (Issue #76): Use _handle_convergence_action() instead.

        This method sent messages to a non-existent agent, causing infinite loops.
        Kept for backwards compatibility but now calls the direct method.
        """
        logger.warning(
            "[DEPRECATED] _send_to_convergence_detector is deprecated. "
            "Using direct convergence check instead (Issue #76)."
        )
        await self._handle_convergence_action()

        # Return a dummy message for compatibility
        return AgentMessage(
            type=MessageType.RESPONSE,
            from_agent="convergence_detector",
            to_agent=self.agent_id,
            content={"deprecated": True, "handled_directly": True}
        )

    # ========================================================================
    # PROMPT BUILDING
    # ========================================================================

    def _build_hypothesis_evaluation_prompt(self, hyp_id: str) -> str:
        """
        Build evaluation prompt with actual hypothesis data from database.

        Args:
            hyp_id: Hypothesis ID to load

        Returns:
            Formatted prompt string with hypothesis details, or fallback if unavailable
        """
        try:
            with get_session() as session:
                hypothesis = get_hypothesis(session, hyp_id, with_experiments=True)

                if hypothesis:
                    # Build rich prompt with actual data
                    related_papers = hypothesis.related_papers or []
                    related_str = ', '.join(related_papers[:5]) if related_papers else 'None identified'

                    testability = hypothesis.testability_score or 0.0
                    novelty = hypothesis.novelty_score or 0.0

                    return f"""Evaluate this hypothesis for testability and scientific merit:

## Hypothesis Details
- **ID**: {hyp_id}
- **Statement**: {hypothesis.statement}
- **Rationale**: {hypothesis.rationale or 'Not provided'}
- **Current Scores**: Testability={testability:.2f}, Novelty={novelty:.2f}

## Research Context
- **Research Question**: {self.research_question}
- **Domain**: {self.domain or 'General'}
- **Related Papers**: {related_str}

## Evaluation Criteria
Rate on scale 1-10:
1. Testability: Can this be experimentally tested?
2. Novelty: Is this approach novel?
3. Impact: Would confirmation significantly advance the field?

Provide brief JSON response:
{{"testability": X, "novelty": X, "impact": X, "recommendation": "proceed/refine/reject", "reasoning": "brief explanation"}}
"""
        except Exception as e:
            logger.warning(f"Failed to load hypothesis {hyp_id}: {e}")

        # Fallback to basic prompt
        return f"""Evaluate this hypothesis for testability and scientific merit:

Hypothesis ID: {hyp_id}
Research Question: {self.research_question}
Domain: {self.domain or "General"}

Rate on scale 1-10:
1. Testability: Can this be experimentally tested?
2. Novelty: Is this approach novel?
3. Impact: Would confirmation significantly advance the field?

Provide brief JSON response:
{{"testability": X, "novelty": X, "impact": X, "recommendation": "proceed/refine/reject", "reasoning": "brief explanation"}}
"""

    def _build_result_analysis_prompt(self, result_id: str) -> str:
        """
        Build analysis prompt with actual result data from database.

        Args:
            result_id: Result ID to load

        Returns:
            Formatted prompt string with result details, or fallback if unavailable
        """
        import json as json_module

        try:
            with get_session() as session:
                result = get_result(session, result_id)

                if result:
                    # Get related experiment and hypothesis
                    experiment = result.experiment
                    hypothesis = experiment.hypothesis if experiment else None

                    # Format data for prompt
                    result_data = result.data or {}
                    stats = result.statistical_tests or {}

                    return f"""Analyze this experiment result:

## Result Details
- **Result ID**: {result_id}
- **Experiment**: {experiment.description if experiment else 'Unknown'}
- **Hypothesis Tested**: {hypothesis.statement if hypothesis else 'Unknown'}

## Research Context
- **Research Question**: {self.research_question}
- **Domain**: {self.domain or 'General'}

## Result Data
```json
{json_module.dumps(result_data, indent=2, default=str)}
```

## Statistical Tests
{json_module.dumps(stats, indent=2) if stats else 'No statistical tests performed'}

## Previous Interpretation
{result.interpretation or 'None available'}

## Analysis Required
Provide analysis including:
1. Key findings
2. Statistical significance
3. Relationship to hypothesis (supported/refuted/inconclusive)
4. Next steps

Provide brief JSON response:
{{"significance": "high/medium/low", "hypothesis_supported": true/false/inconclusive, "key_finding": "summary", "next_steps": "recommendation"}}
"""
        except Exception as e:
            logger.warning(f"Failed to load result {result_id}: {e}")

        # Fallback to basic prompt
        return f"""Analyze this experiment result:

Result ID: {result_id}
Research Question: {self.research_question}

Provide analysis including:
1. Key findings
2. Statistical significance
3. Relationship to hypothesis
4. Next steps

Provide brief JSON response:
{{"significance": "high/medium/low", "hypothesis_supported": true/false/inconclusive, "key_finding": "summary", "next_steps": "recommendation"}}
"""

    # ========================================================================
    # CONCURRENT OPERATIONS
    # ========================================================================

    def execute_experiments_batch(self, protocol_ids: List[str]) -> List[Dict[str, Any]]:
        """
        Execute multiple experiments in parallel using ParallelExperimentExecutor.

        Args:
            protocol_ids: List of protocol IDs to execute

        Returns:
            List of execution results

        Example:
            results = director.execute_experiments_batch(["proto1", "proto2", "proto3"])
        """
        if not self.enable_concurrent or not self.parallel_executor:
            logger.warning("Concurrent execution not enabled, falling back to sequential")
            results = []
            for protocol_id in protocol_ids:
                # Sequential fallback - use direct call (Issue #76)
                try:
                    loop = asyncio.get_running_loop()
                    future = asyncio.run_coroutine_threadsafe(
                        self._handle_execute_experiment_action(protocol_id=protocol_id),
                        loop
                    )
                    future.result(timeout=600)
                except RuntimeError:
                    asyncio.run(
                        self._handle_execute_experiment_action(protocol_id=protocol_id)
                    )
                results.append({"protocol_id": protocol_id, "status": "executed"})
            return results

        logger.info(f"Executing {len(protocol_ids)} experiments in parallel")

        try:
            # Execute batch using parallel executor
            batch_results = self.parallel_executor.execute_batch(protocol_ids)

            # Process results and update research plan
            for result in batch_results:
                if result.get("success"):
                    result_id = result.get("result_id")
                    protocol_id = result.get("protocol_id")

                    # Thread-safe update
                    with self._research_plan_context():
                        if result_id:
                            self.research_plan.add_result(result_id)
                        if protocol_id:
                            self.research_plan.mark_experiment_complete(protocol_id)

                    logger.info(f"Experiment {protocol_id} completed successfully")
                else:
                    logger.error(f"Experiment {result.get('protocol_id')} failed: {result.get('error')}")

            return batch_results

        except Exception as e:
            logger.error(f"Batch experiment execution failed: {e}")
            return [{"protocol_id": pid, "success": False, "error": str(e)} for pid in protocol_ids]

    async def evaluate_hypotheses_concurrently(self, hypothesis_ids: List[str]) -> List[Dict[str, Any]]:
        """
        Evaluate multiple hypotheses concurrently using AsyncClaudeClient.

        Uses async LLM calls to evaluate testability and potential impact of hypotheses in parallel.

        Args:
            hypothesis_ids: List of hypothesis IDs to evaluate

        Returns:
            List of evaluation results with scores and recommendations

        Example:
            evaluations = await director.evaluate_hypotheses_concurrently(["hyp1", "hyp2", "hyp3"])
        """
        if not self.async_llm_client:
            logger.warning("Async LLM client not available, using sequential evaluation")
            return []

        logger.info(f"Evaluating {len(hypothesis_ids)} hypotheses concurrently")

        try:
            from kosmos.core.async_llm import BatchRequest

            # Create batch requests for hypothesis evaluation
            requests = []
            for i, hyp_id in enumerate(hypothesis_ids):
                # Build prompt with actual hypothesis data from database
                prompt = self._build_hypothesis_evaluation_prompt(hyp_id)

                requests.append(BatchRequest(
                    id=hyp_id,
                    prompt=prompt,
                    system="You are a research evaluator. Provide concise, objective assessments.",
                    temperature=0.3  # Lower temperature for more consistent evaluations
                ))

            # Execute concurrent evaluations
            responses = await self.async_llm_client.batch_generate(requests)

            # Process responses
            evaluations = []
            for resp in responses:
                if resp.success:
                    try:
                        import json
                        # Parse JSON response
                        eval_data = json.loads(resp.response)
                        eval_data["hypothesis_id"] = resp.id
                        evaluations.append(eval_data)
                    except json.JSONDecodeError:
                        logger.warning(f"Failed to parse evaluation for {resp.id}")
                        evaluations.append({
                            "hypothesis_id": resp.id,
                            "error": "Parse error",
                            "recommendation": "refine"
                        })
                else:
                    evaluations.append({
                        "hypothesis_id": resp.id,
                        "error": resp.error,
                        "recommendation": "retry"
                    })

            logger.info(f"Completed {len(evaluations)} hypothesis evaluations")
            return evaluations

        except Exception as e:
            logger.error(f"Concurrent hypothesis evaluation failed: {e}")
            return []

    async def analyze_results_concurrently(self, result_ids: List[str]) -> List[Dict[str, Any]]:
        """
        Analyze multiple experiment results concurrently using AsyncClaudeClient.

        Performs parallel interpretation of results to identify patterns and insights.

        Args:
            result_ids: List of result IDs to analyze

        Returns:
            List of analysis results

        Example:
            analyses = await director.analyze_results_concurrently(["res1", "res2", "res3"])
        """
        if not self.async_llm_client:
            logger.warning("Async LLM client not available, using sequential analysis")
            return []

        logger.info(f"Analyzing {len(result_ids)} results concurrently")

        try:
            from kosmos.core.async_llm import BatchRequest

            # Create batch requests for result analysis
            requests = []
            for result_id in result_ids:
                # Build prompt with actual result data from database
                prompt = self._build_result_analysis_prompt(result_id)

                requests.append(BatchRequest(
                    id=result_id,
                    prompt=prompt,
                    system="You are a data analyst. Provide objective, evidence-based interpretations.",
                    temperature=0.3
                ))

            # Execute concurrent analyses
            responses = await self.async_llm_client.batch_generate(requests)

            # Process responses
            analyses = []
            for resp in responses:
                if resp.success:
                    try:
                        import json
                        analysis_data = json.loads(resp.response)
                        analysis_data["result_id"] = resp.id
                        analyses.append(analysis_data)
                    except json.JSONDecodeError:
                        logger.warning(f"Failed to parse analysis for {resp.id}")
                        analyses.append({
                            "result_id": resp.id,
                            "error": "Parse error"
                        })
                else:
                    analyses.append({
                        "result_id": resp.id,
                        "error": resp.error
                    })

            logger.info(f"Completed {len(analyses)} result analyses")
            return analyses

        except Exception as e:
            logger.error(f"Concurrent result analysis failed: {e}")
            return []

    # ========================================================================
    # RESEARCH PLANNING (using Claude)
    # ========================================================================

    def generate_research_plan(self) -> str:
        """
        Generate initial research plan using Claude.

        Returns:
            str: Research plan description
        """
        prompt = f"""You are a research director planning an autonomous scientific investigation.

Research Question: {self.research_question}
Domain: {self.domain or "General"}

Please generate a research plan that includes:

1. **Initial Hypothesis Directions** (3-5 high-level directions to explore)
2. **Experiment Strategy** (what types of experiments would be most informative)
3. **Success Criteria** (how will we know when we've answered the question)
4. **Resource Considerations** (estimated experiments needed, complexity)

Provide a structured, actionable plan in 2-3 paragraphs.
"""

        try:
            response = self.llm_client.generate(prompt, max_tokens=1000)

            # Store in research plan
            self.research_plan.initial_strategy = response

            logger.info("Generated initial research plan using Claude")
            return response

        except Exception as e:
            logger.error(f"Failed to generate research plan: {e}")
            return f"Error generating plan: {str(e)}"

    # ========================================================================
    # DECISION MAKING
    # ========================================================================

    def decide_next_action(self) -> NextAction:
        """
        Decide what to do next based on current workflow state and research plan.

        Decision tree:
        - If no hypotheses: GENERATE_HYPOTHESIS
        - If untested hypotheses: DESIGN_EXPERIMENT
        - If experiments in queue: EXECUTE_EXPERIMENT
        - If results need analysis: ANALYZE_RESULT
        - If hypotheses need refinement: REFINE_HYPOTHESIS
        - If convergence criteria met: CONVERGE
        - Otherwise: Check convergence, then GENERATE_HYPOTHESIS

        Returns:
            NextAction: Next action to take
        """
        # Budget enforcement check - halt if budget exceeded
        try:
            from kosmos.core.metrics import get_metrics, BudgetExceededError
            metrics = get_metrics()
            if metrics.budget_enabled:
                metrics.enforce_budget()
        except BudgetExceededError as e:
            logger.error(f"[BUDGET] Research halted: {e}")

            # Transition to CONVERGED state gracefully
            with self._workflow_context():
                self.workflow.transition_to(
                    WorkflowState.CONVERGED,
                    action="Budget limit reached - research halted",
                    metadata={"reason": "budget_exceeded", "cost": e.current_cost, "limit": e.limit}
                )

            # Return CONVERGE to signal workflow completion
            return NextAction.CONVERGE
        except ImportError:
            # Metrics module not available - continue without enforcement
            logger.debug("Metrics module not available for budget check")

        # Runtime limit check (Issue #56)
        if self._check_runtime_exceeded():
            elapsed = self.get_elapsed_time_hours()
            logger.warning(
                f"[RUNTIME] Research halted: {elapsed:.2f}h elapsed, limit {self.max_runtime_hours}h"
            )

            # Transition to CONVERGED state gracefully
            with self._workflow_context():
                self.workflow.transition_to(
                    WorkflowState.CONVERGED,
                    action="Runtime limit reached - research halted",
                    metadata={
                        "reason": "runtime_exceeded",
                        "elapsed_hours": elapsed,
                        "limit_hours": self.max_runtime_hours
                    }
                )

            return NextAction.CONVERGE

        current_state = self.workflow.current_state

        # Action counter for infinite loop prevention (Issue #51)
        self._actions_this_iteration += 1

        if self._actions_this_iteration > MAX_ACTIONS_PER_ITERATION:
            logger.error(
                "[LOOP-GUARD] Exceeded %d actions in iteration %d - forcing convergence",
                MAX_ACTIONS_PER_ITERATION,
                self.research_plan.iteration_count
            )
            return NextAction.CONVERGE

        # Enhanced debug logging with comprehensive state info
        logger.debug(
            "[STATE] Iteration=%d/%d, State=%s, Hypotheses=%d (untested=%d), "
            "Queue=%d, Results=%d, Actions=%d/%d",
            self.research_plan.iteration_count,
            self.research_plan.max_iterations,
            current_state.value,
            len(self.research_plan.hypothesis_pool),
            len(self.research_plan.get_untested_hypotheses()),
            len(self.research_plan.experiment_queue),
            len(self.research_plan.results),
            self._actions_this_iteration,
            MAX_ACTIONS_PER_ITERATION
        )

        # Check convergence first
        if self._should_check_convergence():
            return NextAction.CONVERGE

        # State-based decision making
        if current_state == WorkflowState.GENERATING_HYPOTHESES:
            # Generate hypotheses
            return NextAction.GENERATE_HYPOTHESIS

        elif current_state == WorkflowState.DESIGNING_EXPERIMENTS:
            # Hypothesis-only mode: a schema-only evidence source released no rows,
            # so there is no data to experiment on. Converge with the hypotheses
            # rather than designing (and failing) dataless experiments.
            if getattr(self, "_hypothesis_only", False):
                logger.info(
                    "[DECISION] Hypothesis-only run (schema-only source, no data): "
                    "converging after hypothesis generation"
                )
                return NextAction.CONVERGE
            # Design experiments for untested hypotheses
            untested = self.research_plan.get_untested_hypotheses()
            if untested:
                return NextAction.DESIGN_EXPERIMENT
            elif self.research_plan.experiment_queue:
                # Bug A fix (Issue #51): Experiments designed, move to execution
                logger.debug("[DECISION] DESIGNING: No untested hypotheses but queue has %d experiments - executing",
                            len(self.research_plan.experiment_queue))
                return NextAction.EXECUTE_EXPERIMENT
            elif self.research_plan.results:
                # Have results to analyze
                logger.debug("[DECISION] DESIGNING: No untested, no queue, but have %d results - analyzing",
                            len(self.research_plan.results))
                return NextAction.ANALYZE_RESULT
            else:
                # No hypotheses AND no experiments AND no results - converge
                logger.debug("[DECISION] DESIGNING: No hypotheses, queue, or results - converging")
                return NextAction.CONVERGE

        elif current_state == WorkflowState.EXECUTING:
            # Execute queued experiments
            if self.research_plan.experiment_queue:
                return NextAction.EXECUTE_EXPERIMENT
            elif self.research_plan.results:
                # Bug D fix (Issue #51): Have results to analyze
                logger.debug("[DECISION] EXECUTING: Queue empty but have %d results - analyzing",
                            len(self.research_plan.results))
                return NextAction.ANALYZE_RESULT
            else:
                # Bug D fix (Issue #51): No queue AND no results - something went wrong
                logger.warning("[DECISION] EXECUTING: No queue and no results - refining to recover")
                return NextAction.REFINE_HYPOTHESIS

        elif current_state == WorkflowState.ANALYZING:
            # Bug B fix (Issue #51): Guard against empty results
            if not self.research_plan.results:
                logger.warning("[DECISION] ANALYZING: No results to analyze")
                if self.research_plan.experiment_queue:
                    logger.debug("[DECISION] ANALYZING: Falling back to execute queued experiments")
                    return NextAction.EXECUTE_EXPERIMENT
                else:
                    logger.debug("[DECISION] ANALYZING: No results or queue - refining")
                    return NextAction.REFINE_HYPOTHESIS
            return NextAction.ANALYZE_RESULT

        elif current_state == WorkflowState.REFINING:
            # Refine hypotheses based on results
            if self.research_plan.tested_hypotheses:
                return NextAction.REFINE_HYPOTHESIS
            else:
                return NextAction.GENERATE_HYPOTHESIS

        elif current_state == WorkflowState.CONVERGED:
            return NextAction.CONVERGE

        elif current_state == WorkflowState.ERROR:
            return NextAction.ERROR_RECOVERY

        else:
            # Default: generate hypotheses
            return NextAction.GENERATE_HYPOTHESIS

    async def _execute_next_action(self, action: NextAction):
        """
        Execute the decided next action asynchronously.

        Uses concurrent execution when enabled and multiple items available.

        Args:
            action: Action to execute
        """
        # Enhanced execution logging (Issue #51)
        action_count = getattr(self, '_actions_this_iteration', 0)
        logger.info(
            "[EXECUTE] Action=%s, Iteration=%d, ActionCount=%d/%d",
            action.value,
            self.research_plan.iteration_count,
            action_count,
            MAX_ACTIONS_PER_ITERATION
        )

        tracker = get_stage_tracker()
        with tracker.track(f"ACTION_{action.value}", action=action.value):
            await self._do_execute_action(action)

    async def _do_execute_action(self, action: NextAction):
        """Internal async method to execute action (wrapped by stage tracking)."""
        if action == NextAction.GENERATE_HYPOTHESIS:
            await self._handle_generate_hypothesis_action()

        elif action == NextAction.DESIGN_EXPERIMENT:
            # Get untested hypotheses
            with self._research_plan_context():
                untested = self.research_plan.get_untested_hypotheses()

            if untested:
                # Use concurrent evaluation if enabled and multiple hypotheses
                if self.enable_concurrent and self.async_llm_client and len(untested) > 1:
                    # Evaluate multiple hypotheses concurrently (up to max_parallel_hypotheses)
                    batch_size = min(len(untested), self.max_parallel_hypotheses)
                    hypothesis_batch = untested[:batch_size]

                    try:
                        # Run async evaluation - now we can await directly!
                        logger.info(f"Starting concurrent evaluation of {len(hypothesis_batch)} hypotheses")
                        evaluations = await asyncio.wait_for(
                            self.evaluate_hypotheses_concurrently(hypothesis_batch),
                            timeout=300
                        )
                        logger.info(f"Concurrent hypothesis evaluation completed")

                        if evaluations:
                            # Process best candidate(s)
                            for eval_result in evaluations:
                                if eval_result.get("recommendation") == "proceed":
                                    await self._handle_design_experiment_action(
                                        hypothesis_id=eval_result["hypothesis_id"]
                                    )
                                    break  # Design experiment for first promising hypothesis
                            else:
                                # No promising hypotheses, design for first untested
                                await self._handle_design_experiment_action(hypothesis_id=untested[0])
                        else:
                            # Fallback to sequential
                            await self._handle_design_experiment_action(hypothesis_id=untested[0])

                    except asyncio.TimeoutError:
                        logger.warning("Hypothesis evaluation timed out, falling back to sequential")
                        await self._handle_design_experiment_action(hypothesis_id=untested[0])
                    except Exception as e:
                        logger.error(f"Concurrent hypothesis evaluation failed: {e}")
                        await self._handle_design_experiment_action(hypothesis_id=untested[0])
                else:
                    # Sequential: design experiment for first untested hypothesis
                    await self._handle_design_experiment_action(hypothesis_id=untested[0])

        elif action == NextAction.EXECUTE_EXPERIMENT:
            # Get queued experiments
            with self._research_plan_context():
                experiment_queue = list(self.research_plan.experiment_queue)

            if experiment_queue:
                # Use batch execution if enabled and multiple experiments queued
                if self.enable_concurrent and self.parallel_executor and len(experiment_queue) > 1:
                    # Execute multiple experiments in parallel
                    batch_size = min(len(experiment_queue), self.max_concurrent_experiments)
                    experiment_batch = experiment_queue[:batch_size]

                    logger.info(f"Executing {batch_size} experiments in parallel")
                    self.execute_experiments_batch(experiment_batch)
                else:
                    # Sequential: execute first queued experiment (Issue #76 direct call)
                    protocol_id = experiment_queue[0]
                    await self._handle_execute_experiment_action(protocol_id=protocol_id)

        elif action == NextAction.ANALYZE_RESULT:
            # Get recent results
            with self._research_plan_context():
                results = list(self.research_plan.results)

            if results:
                # Use concurrent analysis if enabled and multiple results
                if self.enable_concurrent and self.async_llm_client and len(results) > 1:
                    # Analyze multiple recent results concurrently
                    batch_size = min(len(results), 5)  # Analyze up to 5 recent results
                    result_batch = results[-batch_size:]  # Most recent results

                    try:
                        # Run async analysis - now we can await directly!
                        logger.info(f"Starting concurrent analysis of {len(result_batch)} results")
                        analyses = await asyncio.wait_for(
                            self.analyze_results_concurrently(result_batch),
                            timeout=300
                        )
                        logger.info(f"Concurrent result analysis completed")

                        if analyses:
                            # Process analyses and update hypotheses
                            for analysis in analyses:
                                result_id = analysis.get("result_id")
                                # Send to data analyst for full processing
                                if result_id:
                                    await self._handle_analyze_result_action(result_id=result_id)
                                    break  # Process one at a time in workflow
                        else:
                            # Fallback to sequential
                            result_id = results[-1]
                            await self._handle_analyze_result_action(result_id=result_id)

                    except asyncio.TimeoutError:
                        logger.warning("Result analysis timed out, falling back to sequential")
                        result_id = results[-1]
                        await self._handle_analyze_result_action(result_id=result_id)
                    except Exception as e:
                        logger.error(f"Concurrent result analysis failed: {e}")
                        result_id = results[-1]
                        await self._handle_analyze_result_action(result_id=result_id)
                else:
                    # Sequential: analyze most recent result
                    result_id = results[-1]
                    await self._handle_analyze_result_action(result_id=result_id)

        elif action == NextAction.REFINE_HYPOTHESIS:
            # Refine most recently tested hypothesis
            with self._research_plan_context():
                tested = list(self.research_plan.tested_hypotheses)

            if tested:
                hypothesis_id = tested[-1]
                await self._handle_refine_hypothesis_action(
                    hypothesis_id=hypothesis_id
                )

            # Increment iteration after completing refinement phase
            # This marks the completion of one full research cycle
            with self._research_plan_context():
                self.research_plan.increment_iteration()
            # Reset action counter for new iteration (Issue #51)
            self._actions_this_iteration = 0
            logger.info(f"[ITERATION] Completed iteration {self.research_plan.iteration_count}")

        elif action == NextAction.CONVERGE:
            # Bug C fix (Issue #51): Don't increment iteration for convergence check
            # Convergence is a check, not a new research cycle
            # Issue #76 fix: Call convergence detector directly instead of message passing
            logger.info(f"[CONVERGE] Checking convergence at iteration {self.research_plan.iteration_count}")
            await self._handle_convergence_action()

        elif action == NextAction.ERROR_RECOVERY:
            logger.info("[ERROR-RECOVERY] Attempting recovery from error state")
            self._consecutive_errors = 0

            # ERROR state can only transition to: INITIALIZING, GENERATING_HYPOTHESES, PAUSED
            # So we always recover via GENERATING_HYPOTHESES — from there the normal
            # decide_next_action() logic will route to the correct phase.
            with self._workflow_context():
                self.workflow.transition_to(
                    WorkflowState.GENERATING_HYPOTHESES,
                    action="Error recovery: resume research from hypothesis generation"
                )
                logger.info("[ERROR-RECOVERY] Transitioning to GENERATING_HYPOTHESES")

        elif action == NextAction.PAUSE:
            self.pause()

        else:
            logger.warning(f"Unknown action: {action}")

    def _should_check_convergence(self) -> bool:
        """
        Check if convergence should be evaluated.

        Returns:
            bool: True if convergence check is needed
        """
        # Don't converge during active research phases where new work is being generated
        if self.workflow.current_state in {
            WorkflowState.DESIGNING_EXPERIMENTS,
            WorkflowState.EXECUTING,
            WorkflowState.ANALYZING,
        }:
            return False

        # Check iteration limit (mandatory) — but defer if too few experiments
        # completed and testable work remains (mirrors ConvergenceDetector logic)
        if self.research_plan.iteration_count >= self.research_plan.max_iterations:
            completed = len(self.research_plan.completed_experiments)
            min_exp = self.config.get("min_experiments_before_convergence", 2)
            untested = self.research_plan.get_untested_hypotheses()
            has_testable_work = len(untested) > 0 or len(self.research_plan.experiment_queue) > 0
            if completed < min_exp and has_testable_work:
                logger.info(
                    f"Iteration limit reached but deferring: "
                    f"{completed}/{min_exp} experiments, testable work remains"
                )
                return False
            logger.info("Iteration limit reached")
            return True

        # Don't converge if we haven't generated any hypotheses yet
        # Let the state machine handle generating hypotheses first
        if not self.research_plan.hypothesis_pool:
            # Only converge on empty pool if we've already tried generating
            # and moved past the early pipeline stages
            convergence_ok_states = {
                WorkflowState.REFINING,
                WorkflowState.ERROR,
            }
            if self.workflow.current_state in convergence_ok_states:
                logger.info("No hypotheses in pool after generation attempted")
                return True
            return False

        untested = self.research_plan.get_untested_hypotheses()
        if not untested and not self.research_plan.experiment_queue:
            logger.info("No untested hypotheses and no queued experiments")
            return True

        return False

    # ========================================================================
    # STRATEGY ADAPTATION
    # ========================================================================

    def select_next_strategy(self) -> str:
        """
        Select next strategy based on effectiveness tracking.

        Strategies with higher success rates are favored.

        Returns:
            str: Selected strategy name
        """
        # Calculate effectiveness scores
        scores = {}
        for strategy, stats in self.strategy_stats.items():
            attempts = stats["attempts"]
            if attempts == 0:
                # Favor unexplored strategies
                scores[strategy] = 1.0
            else:
                success_rate = stats["successes"] / attempts
                scores[strategy] = success_rate

        # Select strategy with highest score
        best_strategy = max(scores.items(), key=lambda x: x[1])[0]

        logger.debug(f"Selected strategy: {best_strategy} (scores: {scores})")
        return best_strategy

    def update_strategy_effectiveness(self, strategy: str, success: bool, cost: float = 0.0):
        """
        Update strategy effectiveness tracking.

        Args:
            strategy: Strategy name
            success: Whether strategy was successful
            cost: Cost incurred (API tokens, compute time, etc.)
        """
        # Thread-safe strategy stats update
        with self._strategy_stats_context():
            if strategy in self.strategy_stats:
                self.strategy_stats[strategy]["attempts"] += 1
                if success:
                    self.strategy_stats[strategy]["successes"] += 1
                self.strategy_stats[strategy]["cost"] += cost

                logger.debug(f"Updated strategy {strategy}: success={success}, cost={cost}")

    # ========================================================================
    # AGENT REGISTRY
    # ========================================================================

    def register_agent(self, agent_type: str, agent_id: str):
        """
        Register an agent for coordination.

        Args:
            agent_type: Type of agent (HypothesisGeneratorAgent, etc.)
            agent_id: Unique agent ID
        """
        self.agent_registry[agent_type] = agent_id
        logger.info(f"Registered {agent_type} with ID {agent_id}")

    def get_agent_id(self, agent_type: str) -> Optional[str]:
        """
        Get agent ID for a given type.

        Args:
            agent_type: Agent type

        Returns:
            Optional[str]: Agent ID if registered
        """
        return self.agent_registry.get(agent_type)

    # ========================================================================
    # EXECUTE (BaseAgent interface)
    # ========================================================================

    async def execute(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute research task asynchronously.

        Args:
            task: Task specification (usually {"action": "start_research"})

        Returns:
            dict: Task result
        """
        action = task.get("action", "start_research")

        if action == "start_research":
            # Generate initial research plan
            plan = self.generate_research_plan()

            # Start the workflow
            self.start()

            # Execute first action
            next_action = self.decide_next_action()
            await self._execute_next_action(next_action)

            return {
                "status": "research_started",
                "research_plan": plan,
                "next_action": next_action.value
            }

        elif action == "step":
            # Execute one step of research
            next_action = self.decide_next_action()
            await self._execute_next_action(next_action)

            return {
                "status": "step_executed",
                "next_action": next_action.value,
                "workflow_state": self.workflow.current_state.value
            }

        else:
            raise ValueError(f"Unknown action: {action}")

    def execute_sync(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """
        Synchronous wrapper for execute (backwards compatibility).
        """
        try:
            loop = asyncio.get_running_loop()
            future = asyncio.run_coroutine_threadsafe(self.execute(task), loop)
            return future.result(timeout=600)
        except RuntimeError:
            return asyncio.run(self.execute(task))

    # ========================================================================
    # STATUS & REPORTING
    # ========================================================================

    def get_research_status(self) -> Dict[str, Any]:
        """
        Get comprehensive research status.

        Returns:
            dict: Full research status including plan, workflow, statistics
        """
        return {
            "research_question": self.research_question,
            "domain": self.domain,
            "workflow_state": self.workflow.current_state.value,
            "iteration": self.research_plan.iteration_count,
            "max_iterations": self.research_plan.max_iterations,
            "elapsed_time_hours": self.get_elapsed_time_hours(),  # Issue #56
            "max_runtime_hours": self.max_runtime_hours,  # Issue #56
            "has_converged": self.research_plan.has_converged,
            "convergence_reason": self.research_plan.convergence_reason,
            "hypothesis_pool_size": len(self.research_plan.hypothesis_pool),
            "hypotheses_tested": len(self.research_plan.tested_hypotheses),
            "hypotheses_supported": len(self.research_plan.supported_hypotheses),
            "hypotheses_rejected": len(self.research_plan.rejected_hypotheses),
            "experiments_completed": len(self.research_plan.completed_experiments),
            "results_count": len(self.research_plan.results),
            "strategy_stats": self.strategy_stats,
            "rollouts": self.rollout_tracker.to_dict(),  # Issue #58
            "agent_status": self.get_status()
        }
