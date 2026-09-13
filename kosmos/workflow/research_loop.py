"""
Research Workflow Integration for Kosmos.

Orchestrates the complete autonomous research cycle integrating all 6 gaps.

This is the main entry point for running the Kosmos AI scientist system.
Emits streaming events for real-time visibility via EventBus.
"""

import logging
import uuid
import asyncio
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence
from datetime import datetime

# Gap imports
from kosmos.compression import ContextCompressor
from kosmos.world_model.artifacts import ArtifactStateManager
from kosmos.orchestration import (
    PlanCreatorAgent,
    PlanReviewerAgent,
    DelegationManager,
    NoveltyDetector,
)
from kosmos.validation import ScholarEvalValidator
from kosmos.agents import SkillLoader

logger = logging.getLogger(__name__)


class ResearchWorkflow:
    """
    Complete autonomous research workflow.

    Integrates all 6 gap implementations:
    - Gap 0: Context compression
    - Gap 1: State management
    - Gap 2: Task generation & orchestration
    - Gap 3: Agent integration with skills
    - Gap 4: Python-first tooling
    - Gap 5: Discovery validation

    Example:
        workflow = ResearchWorkflow(
            research_objective="Investigate KRAS mutations in cancer",
            anthropic_api_key="your-key"
        )

        # Run 5 cycles
        results = await workflow.run(num_cycles=5)

        # Generate report
        report = await workflow.generate_report()
    """

    def __init__(
        self,
        research_objective: str,
        anthropic_client=None,
        artifacts_dir: str = "artifacts",
        world_model=None,
        max_cycles: int = 20,
        seed: Optional[int] = None,
        temperature: Optional[float] = None,
        evidence_pipeline=None,
        evidence_candidates: Optional[Sequence[Any]] = None,
        evidence_discoverer: Optional[Callable[..., Sequence[Any]]] = None,
        evidence_predictor=None,
        evidence_trainer=None,
        evidence_config=None,
        evidence_output_dir: Optional[str] = None,
        evidence_every_cycle: bool = False,
    ):
        """
        Initialize Research Workflow.

        Args:
            research_objective: Main research goal
            anthropic_client: Anthropic client for LLM calls
            artifacts_dir: Directory for artifact storage
            world_model: Optional knowledge graph
            max_cycles: Maximum research cycles
            seed: Random seed for reproducibility (Issue #64)
            temperature: LLM temperature override (Issue #64)
            evidence_pipeline: Optional kosmos.evidence.EvidencePipeline.
                When supplied, enables the explicit non-gold evidence stage.
            evidence_candidates: CandidateDataset manifests already retrieved by
                Kosmos or an upstream scientific data discovery adapter.
            evidence_discoverer: Callable receiving research objective, cycle and
                context, returning CandidateDataset manifests.
            evidence_predictor: Frozen gold-derived or reviewed pretrained
                pseudo-label predictor.
            evidence_trainer: PPI trainer adapter, for example
                kosmos.ppi.PPIEvidenceTrainer.
            evidence_config: Routing/PPI configuration passed to the trainer.
            evidence_output_dir: Artifact root for evidence runs.
            evidence_every_cycle: Run the evidence stage after every cycle;
                otherwise run once after the final research cycle.
        """
        self.research_objective = research_objective
        self.max_cycles = max_cycles
        self._seed = seed
        self._temperature = temperature
        self.evidence_pipeline = evidence_pipeline
        self.evidence_candidates = list(evidence_candidates or [])
        self.evidence_discoverer = evidence_discoverer
        self.evidence_predictor = evidence_predictor
        self.evidence_trainer = evidence_trainer
        self.evidence_config = evidence_config
        self.evidence_output_dir = Path(evidence_output_dir or (Path(artifacts_dir) / "evidence"))
        self.evidence_every_cycle = evidence_every_cycle
        supplied_evidence = [
            evidence_pipeline,
            evidence_predictor,
            evidence_trainer,
            evidence_config,
        ]
        if any(value is not None for value in supplied_evidence) and not all(
            value is not None for value in supplied_evidence
        ):
            raise ValueError(
                "evidence_pipeline, evidence_predictor, evidence_trainer and "
                "evidence_config must be supplied together"
            )
        self.evidence_enabled = all(value is not None for value in supplied_evidence)
        if self.evidence_enabled and not (self.evidence_candidates or self.evidence_discoverer):
            raise ValueError("Evidence workflow requires candidates or an evidence_discoverer")

        # Apply seed if provided (Issue #64: Multi-Run Convergence)
        if seed is not None:
            from kosmos.safety.reproducibility import ReproducibilityManager

            self._reproducibility_manager = ReproducibilityManager(default_seed=seed)
            self._reproducibility_manager.set_seed(seed)
            logger.info(f"Set random seed: {seed}")
        else:
            self._reproducibility_manager = None

        # Initialize components
        logger.info("Initializing Kosmos Research Workflow...")

        # Gap 0: Context Compression
        self.context_compressor = ContextCompressor(anthropic_client)
        logger.info("✓ Gap 0: Context Compression initialized")

        # Gap 1: State Manager
        self.state_manager = ArtifactStateManager(
            artifacts_dir=artifacts_dir, world_model=world_model
        )
        logger.info("✓ Gap 1: State Manager initialized")

        # Gap 3: Skill Loader
        self.skill_loader = SkillLoader()
        logger.info("✓ Gap 3: Skill Loader initialized")

        # Gap 5: Discovery Validation
        self.scholar_eval = ScholarEvalValidator(anthropic_client)
        logger.info("✓ Gap 5: ScholarEval Validator initialized")

        # Gap 2: Orchestration Components
        self.plan_creator = PlanCreatorAgent(anthropic_client)
        self.plan_reviewer = PlanReviewerAgent(anthropic_client)

        # Wire real agents into DelegationManager when a client is available
        agents = {}
        if anthropic_client:
            from kosmos.agents import (
                DataAnalystAgent,
                HypothesisGeneratorAgent,
                ExperimentDesignerAgent,
                LiteratureAnalyzerAgent,
            )

            agents = {
                "data_analyst": DataAnalystAgent(),
                "hypothesis_generator": HypothesisGeneratorAgent(),
                "experiment_designer": ExperimentDesignerAgent(),
                "literature_analyzer": LiteratureAnalyzerAgent(),
            }

        self.delegation_manager = DelegationManager(agents=agents)
        self.novelty_detector = NoveltyDetector()
        logger.info("✓ Gap 2: Orchestration components initialized")

        # Tracking
        self.past_tasks = []
        self.cycle_results = []
        self.start_time = None
        self.evidence_results = []

        # Streaming event support
        self.process_id = f"research_{uuid.uuid4().hex[:8]}"
        self._event_bus = None
        self._emit_events = True

        # Try to get event bus
        try:
            from kosmos.core.event_bus import get_event_bus

            self._event_bus = get_event_bus()
        except ImportError:
            self._emit_events = False
            logger.debug("EventBus not available, streaming disabled")

    async def run(self, num_cycles: int = 5, tasks_per_cycle: int = 10) -> Dict:
        """
        Run autonomous research workflow.

        Args:
            num_cycles: Number of cycles to execute
            tasks_per_cycle: Tasks to generate per cycle

        Returns:
            Dictionary with:
            - cycles_completed: Number of cycles
            - total_findings: Total findings generated
            - validated_findings: Validated findings count
            - validation_rate: Percentage validated
            - total_time: Execution time in seconds
        """
        self.start_time = datetime.now()
        self._run_num_cycles = num_cycles

        logger.info(
            f"\n{'='*70}\n"
            f"Starting Kosmos Research Workflow\n"
            f"Objective: {self.research_objective}\n"
            f"Cycles: {num_cycles}\n"
            f"Tasks per cycle: {tasks_per_cycle}\n"
            f"{'='*70}\n"
        )

        # Emit workflow started event
        await self._emit_workflow_event("started", cycle=0, max_cycles=num_cycles)

        for cycle in range(1, num_cycles + 1):
            logger.info(f"\n--- Cycle {cycle}/{num_cycles} ---")

            # Emit cycle started event
            await self._emit_cycle_event(
                "started", cycle=cycle, max_cycles=num_cycles, tasks_count=tasks_per_cycle
            )

            try:
                cycle_result = await self._execute_cycle(cycle, tasks_per_cycle)
                if self.evidence_enabled and self.evidence_every_cycle:
                    cycle_result["evidence"] = await self._execute_evidence_stage(cycle)
                self.cycle_results.append(cycle_result)

                logger.info(
                    f"Cycle {cycle} complete: "
                    f"{cycle_result.get('tasks_completed', 0)}/{tasks_per_cycle} tasks, "
                    f"{cycle_result.get('validated_findings', 0)} validated findings"
                )

                # Emit cycle completed event
                await self._emit_cycle_event(
                    "completed",
                    cycle=cycle,
                    max_cycles=num_cycles,
                    tasks_count=tasks_per_cycle,
                    completed_tasks=cycle_result.get("tasks_completed", 0),
                    findings_count=cycle_result.get("validated_findings", 0),
                )

                # Emit workflow progress event
                progress_percent = (cycle / num_cycles) * 100
                total_findings = sum(r.get("validated_findings", 0) for r in self.cycle_results)
                await self._emit_workflow_event(
                    "progress",
                    cycle=cycle,
                    max_cycles=num_cycles,
                    progress_percent=progress_percent,
                    findings_count=total_findings,
                )

            except Exception as e:
                logger.error(f"Cycle {cycle} failed: {e}")

                # Emit cycle failed event
                await self._emit_cycle_event(
                    "failed", cycle=cycle, max_cycles=num_cycles, tasks_count=tasks_per_cycle
                )
                continue

        # Compute final statistics
        results = self._compute_final_statistics()

        # Emit workflow completed event
        await self._emit_workflow_event(
            "completed",
            cycle=num_cycles,
            max_cycles=num_cycles,
            progress_percent=100.0,
            findings_count=results.get("validated_findings", 0),
            validated_count=results.get("validated_findings", 0),
        )

        return results

    async def _execute_evidence_stage(
        self, cycle: int, context: Optional[dict] = None
    ) -> Dict[str, Any]:
        """Run routing, pseudo-labeling and PPI without changing the task loop.

        Dataset retrieval is intentionally upstream: candidates must contain
        persistent IDs, provenance and a scientific relevance assessment. This
        prevents the trainer from silently treating a search hit as evidence.
        The synchronous scientific/ML adapter runs in a worker thread so event
        streaming remains responsive.
        """
        output_dir = self.evidence_output_dir / f"cycle-{cycle:03d}"
        candidates = self.evidence_candidates
        if self.evidence_discoverer is not None:
            candidates = list(
                await asyncio.to_thread(
                    self.evidence_discoverer,
                    self.research_objective,
                    cycle,
                    context or {},
                )
            )
        if not candidates:
            raise ValueError("Evidence discovery returned no CandidateDataset manifests")
        logger.info("  Evidence stage: routing %d candidate datasets", len(candidates))
        result = await asyncio.to_thread(
            self.evidence_pipeline.run,
            candidates=candidates,
            predictor=self.evidence_predictor,
            trainer=self.evidence_trainer,
            config=self.evidence_config,
            output_dir=str(output_dir),
        )
        summary = {
            "cycle": cycle,
            "datasets": len(result.get("records", [])),
            "accepted": sum(r.decision == "accept" for r in result.get("records", [])),
            "deferred": sum(r.decision == "defer" for r in result.get("records", [])),
            "rejected": sum(r.decision == "reject" for r in result.get("records", [])),
            "evaluations": result.get("evaluations", []),
            "artifact_dir": str(output_dir),
        }
        self.evidence_results.append(summary)
        logger.info(
            "  Evidence stage complete: %d accepted, %d deferred, %d rejected",
            summary["accepted"],
            summary["deferred"],
            summary["rejected"],
        )
        return summary

    async def _execute_cycle(self, cycle: int, num_tasks: int) -> Dict:
        """Execute one research cycle."""

        # Step 1: Get context from State Manager
        context = self.state_manager.get_cycle_context(cycle, lookback=3)
        context["research_objective"] = self.research_objective

        logger.info(f"  Context: {context.get('findings_count', 0)} recent findings")

        # Step 2: Plan Creator generates tasks
        plan = self.plan_creator.create_plan(
            research_objective=self.research_objective, context=context, num_tasks=num_tasks
        )

        logger.info(f"  Generated plan with {len(plan.tasks)} tasks")

        # Step 3: Novelty Detector checks redundancy
        if self.past_tasks:
            self.novelty_detector.index_past_tasks(self.past_tasks)
            plan_novelty = self.novelty_detector.check_plan_novelty(plan.to_dict())
            logger.info(
                f"  Novelty: {plan_novelty['novel_task_count']}/{len(plan.tasks)} "
                f"novel tasks ({plan_novelty['plan_novelty_score']:.2f})"
            )

        # Step 4: Plan Reviewer validates quality
        review = self.plan_reviewer.review_plan(plan.to_dict(), context)

        logger.info(
            f"  Plan review: {'APPROVED' if review.approved else 'REJECTED'} "
            f"(score: {review.average_score:.1f}/10)"
        )

        # If rejected, attempt revision (once)
        if not review.approved:
            logger.info("  Revising plan based on feedback...")
            plan = self.plan_creator.revise_plan(plan, review.to_dict(), context)
            review = self.plan_reviewer.review_plan(plan.to_dict(), context)

            logger.info(f"  Revised plan: {'APPROVED' if review.approved else 'REJECTED'}")

        # Step 5: Delegation Manager executes approved tasks
        completed_tasks = []
        if review.approved:
            execution_result = await self.delegation_manager.execute_plan(
                plan.to_dict(), cycle, context
            )

            completed_tasks = execution_result.get("completed_tasks", [])
            logger.info(f"  Execution: {len(completed_tasks)}/{num_tasks} tasks completed")
        else:
            logger.warning("  Plan rejected after revision, skipping execution")

        # Step 6 & 7: Validate and save findings
        validated_count = 0
        for task_result in completed_tasks:
            finding = task_result.get("finding")
            if not finding:
                continue

            # ScholarEval validation
            eval_score = self.scholar_eval.evaluate_finding(finding)

            if eval_score.passes_threshold:
                # Save validated finding
                finding["scholar_eval"] = eval_score.to_dict()
                await self.state_manager.save_finding_artifact(
                    cycle, task_result.get("task_id", 0), finding
                )
                validated_count += 1
            else:
                logger.debug(f"    Finding rejected: score={eval_score.overall_score:.2f}")

        logger.info(f"  Validated: {validated_count}/{len(completed_tasks)} findings")

        # Step 8: Compress cycle results
        compressed_summary = None
        if completed_tasks:
            compressed_cycle = self.context_compressor.compress_cycle_results(
                cycle, completed_tasks
            )
            compressed_summary = compressed_cycle.summary
            logger.info(
                f"  Compressed: {len(completed_tasks)} tasks → " f"{len(compressed_summary)} chars"
            )

        # Track tasks for novelty detection
        for task in plan.tasks:
            self.past_tasks.append(task.to_dict())

        # Generate cycle summary
        await self.state_manager.generate_cycle_summary(cycle)

        evidence = None
        # By default evidence runs once after the final research cycle. Running
        # every cycle is opt-in because it may retrain PPI and create artifacts.
        if (
            self.evidence_enabled
            and not self.evidence_every_cycle
            and cycle == getattr(self, "_run_num_cycles", self.max_cycles)
        ):
            evidence = await self._execute_evidence_stage(cycle)

        return {
            "cycle": cycle,
            "tasks_generated": len(plan.tasks),
            "tasks_completed": len(completed_tasks),
            "validated_findings": validated_count,
            "plan_approved": review.approved,
            "plan_score": review.average_score,
            "compressed_summary": compressed_summary,
            "evidence": evidence,
        }

    def _compute_final_statistics(self) -> Dict:
        """Compute final statistics across all cycles."""
        total_time = (datetime.now() - self.start_time).total_seconds()

        all_findings = self.state_manager.get_all_findings()
        validated_findings = self.state_manager.get_validated_findings()

        total_tasks_completed = sum(r.get("tasks_completed", 0) for r in self.cycle_results)
        total_tasks_generated = sum(r.get("tasks_generated", 0) for r in self.cycle_results)

        results = {
            "cycles_completed": len(self.cycle_results),
            "total_findings": len(all_findings),
            "validated_findings": len(validated_findings),
            "validation_rate": (len(validated_findings) / len(all_findings) if all_findings else 0),
            "total_tasks_generated": total_tasks_generated,
            "total_tasks_completed": total_tasks_completed,
            "task_completion_rate": (
                total_tasks_completed / total_tasks_generated if total_tasks_generated else 0
            ),
            "total_time": total_time,
            "research_objective": self.research_objective,
        }
        results["evidence"] = {
            "enabled": self.evidence_enabled,
            "runs": self.evidence_results,
        }

        logger.info(
            f"\n{'='*70}\n"
            f"Research Workflow Complete!\n"
            f"Cycles: {results['cycles_completed']}\n"
            f"Findings: {results['total_findings']} "
            f"({results['validated_findings']} validated, "
            f"{results['validation_rate']*100:.1f}%)\n"
            f"Tasks: {results['total_tasks_completed']}/{results['total_tasks_generated']} "
            f"({results['task_completion_rate']*100:.1f}% completion)\n"
            f"Time: {results['total_time']:.1f}s\n"
            f"{'='*70}\n"
        )

        return results

    async def generate_report(self) -> str:
        """
        Generate research report from findings.

        Returns:
            Markdown-formatted research report
        """
        validated_findings = self.state_manager.get_validated_findings()

        report = f"# Research Report\n\n"
        report += f"**Objective**: {self.research_objective}\n"
        report += f"**Date**: {datetime.now().strftime('%Y-%m-%d')}\n"
        report += f"**Cycles Completed**: {len(self.cycle_results)}\n\n"

        report += f"## Summary\n\n"
        report += f"This autonomous research system completed {len(self.cycle_results)} "
        report += f"research cycles, generating {len(validated_findings)} validated findings.\n\n"

        if self.evidence_enabled:
            report += "## External Evidence\n\n"
            report += f"Evidence routing ran {len(self.evidence_results)} time(s).\n\n"
            for run in self.evidence_results:
                report += (
                    f"- Cycle {run['cycle']}: {run['accepted']} accepted, "
                    f"{run['deferred']} deferred, {run['rejected']} rejected; "
                    f"artifacts: `{run['artifact_dir']}`\n"
                )
            report += "\n"

        report += f"## Key Findings\n\n"
        for i, finding in enumerate(validated_findings[:10], 1):  # Top 10
            report += f"### Finding {i}\n\n"
            report += f"{finding.summary}\n\n"

            # Statistics
            if finding.statistics:
                report += "**Statistics**:\n"
                for key, value in finding.statistics.items():
                    if isinstance(value, float):
                        report += f"- {key}: {value:.4f}\n"
                    else:
                        report += f"- {key}: {value}\n"
                report += "\n"

            # Evidence with code provenance (Issue #62)
            if finding.code_provenance:
                prov = finding.code_provenance
                hyperlink = (
                    f"{prov['notebook_path']}#cell={prov['cell_index']}&line={prov['start_line']}"
                )
                filename = prov["notebook_path"].split("/")[-1]
                report += f"**Code Citation**: [{filename}]({hyperlink})"
                if prov.get("start_line") and prov.get("end_line"):
                    report += f" (lines {prov['start_line']}-{prov['end_line']})"
                report += "\n\n"
            elif finding.notebook_path:
                report += f"**Evidence**: `{finding.notebook_path}`\n\n"

            # Quality score
            if finding.scholar_eval:
                overall = finding.scholar_eval.get("overall_score", 0)
                report += f"**Quality Score**: {overall:.2f}/1.0\n\n"

        return report

    def get_statistics(self) -> Dict:
        """Get comprehensive statistics."""
        return {
            "workflow": {
                "research_objective": self.research_objective,
                "max_cycles": self.max_cycles,
                "cycles_completed": len(self.cycle_results),
            },
            "evidence": {
                "enabled": self.evidence_enabled,
                "runs": len(self.evidence_results),
            },
            "state_manager": self.state_manager.get_statistics(),
            "skill_loader": self.skill_loader.get_statistics(),
            "novelty_detector": self.novelty_detector.get_statistics(),
        }

    async def _emit_workflow_event(
        self,
        status: str,
        cycle: int = 0,
        max_cycles: int = 0,
        progress_percent: float = 0.0,
        findings_count: int = 0,
        validated_count: int = 0,
    ) -> None:
        """
        Emit a workflow lifecycle event.

        Args:
            status: Event status (started, progress, completed, failed)
            cycle: Current cycle number
            max_cycles: Total number of cycles
            progress_percent: Progress percentage
            findings_count: Total findings count
            validated_count: Validated findings count
        """
        if not self._emit_events or self._event_bus is None:
            return

        try:
            from kosmos.core.events import WorkflowEvent, EventType

            event_type = {
                "started": EventType.WORKFLOW_STARTED,
                "progress": EventType.WORKFLOW_PROGRESS,
                "completed": EventType.WORKFLOW_COMPLETED,
                "failed": EventType.WORKFLOW_FAILED,
            }.get(status, EventType.WORKFLOW_PROGRESS)

            event = WorkflowEvent(
                type=event_type,
                process_id=self.process_id,
                research_question=self.research_objective,
                state=status,
                cycle=cycle,
                max_cycles=max_cycles,
                progress_percent=progress_percent,
                findings_count=findings_count,
                validated_count=validated_count,
            )

            await self._event_bus.publish(event)

        except Exception as e:
            logger.debug(f"Failed to emit workflow event: {e}")

    async def _emit_cycle_event(
        self,
        status: str,
        cycle: int,
        max_cycles: int,
        tasks_count: int = 0,
        completed_tasks: int = 0,
        findings_count: int = 0,
        duration_ms: int = None,
    ) -> None:
        """
        Emit a research cycle event.

        Args:
            status: Event status (started, completed, failed)
            cycle: Current cycle number
            max_cycles: Total number of cycles
            tasks_count: Total tasks in cycle
            completed_tasks: Completed tasks count
            findings_count: Findings generated this cycle
            duration_ms: Cycle duration in milliseconds
        """
        if not self._emit_events or self._event_bus is None:
            return

        try:
            from kosmos.core.events import CycleEvent, EventType

            event_type = {
                "started": EventType.CYCLE_STARTED,
                "completed": EventType.CYCLE_COMPLETED,
                "failed": EventType.CYCLE_FAILED,
            }.get(status, EventType.CYCLE_STARTED)

            event = CycleEvent(
                type=event_type,
                process_id=self.process_id,
                cycle=cycle,
                max_cycles=max_cycles,
                tasks_count=tasks_count,
                completed_tasks=completed_tasks,
                findings_count=findings_count,
                duration_ms=duration_ms,
            )

            await self._event_bus.publish(event)

        except Exception as e:
            logger.debug(f"Failed to emit cycle event: {e}")
