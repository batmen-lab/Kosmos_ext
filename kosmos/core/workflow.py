"""
Research workflow state machine for autonomous iteration (Phase 7).

This module defines the state machine that orchestrates the research cycle:
INITIALIZING → GENERATING_HYPOTHESES → DESIGNING_EXPERIMENTS → EXECUTING
→ ANALYZING → REFINING → (loop or CONVERGED)
"""

from enum import Enum
from typing import List, Dict, Optional, Any
from datetime import datetime, timezone
from pydantic import BaseModel, Field, ConfigDict
import logging

logger = logging.getLogger(__name__)


class WorkflowState(str, Enum):
    """States in the autonomous research workflow."""

    INITIALIZING = "initializing"
    GENERATING_HYPOTHESES = "generating_hypotheses"
    DESIGNING_EXPERIMENTS = "designing_experiments"
    EXECUTING = "executing"
    ANALYZING = "analyzing"
    REFINING = "refining"
    CONVERGED = "converged"
    PAUSED = "paused"
    ERROR = "error"


class NextAction(str, Enum):
    """Possible next actions for the research director."""

    GENERATE_HYPOTHESIS = "generate_hypothesis"
    DESIGN_EXPERIMENT = "design_experiment"
    EXECUTE_EXPERIMENT = "execute_experiment"
    ANALYZE_RESULT = "analyze_result"
    REFINE_HYPOTHESIS = "refine_hypothesis"
    CONVERGE = "converge"
    PAUSE = "pause"
    ERROR_RECOVERY = "error_recovery"


def _state_value(state) -> str:
    """The string form of a workflow state, whether it arrives as enum or str.

    Needed because `WorkflowTransition` sets `use_enum_values=True` (so pydantic
    stores strings) while `ResearchWorkflow.current_state` is a plain attribute
    holding the enum member. Both spellings mean the same state and both must
    export identically.
    """
    return state.value if hasattr(state, "value") else str(state)


class WorkflowTransition(BaseModel):
    """A transition between workflow states."""

    model_config = ConfigDict(use_enum_values=True)

    from_state: WorkflowState
    to_state: WorkflowState
    action: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ResearchPlan(BaseModel):
    """The overall research plan for autonomous iteration."""

    model_config = ConfigDict(use_enum_values=True)

    research_question: str
    domain: Optional[str] = None
    initial_strategy: str = ""
    current_state: WorkflowState = WorkflowState.INITIALIZING

    # Hypothesis tracking
    hypothesis_pool: List[str] = Field(default_factory=list)  # Hypothesis IDs
    tested_hypotheses: List[str] = Field(default_factory=list)
    supported_hypotheses: List[str] = Field(default_factory=list)
    rejected_hypotheses: List[str] = Field(default_factory=list)

    # Experiment tracking
    experiment_queue: List[str] = Field(default_factory=list)  # Protocol IDs
    completed_experiments: List[str] = Field(default_factory=list)
    # Protocols that raised and were dropped from the queue. Neither queued nor
    # completed, so without this list a failed experiment is in no list at all
    # and vanishes from the report -- which is how a Docker outage read as
    # "no experiment was designed".
    failed_experiments: List[str] = Field(default_factory=list)

    # Results tracking
    results: List[str] = Field(default_factory=list)  # Result IDs

    # Iteration tracking
    iteration_count: int = 0
    max_iterations: int = 10

    # Convergence
    has_converged: bool = False
    convergence_reason: Optional[str] = None

    # Timestamps
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    # Additional metadata
    success_criteria: Dict[str, Any] = Field(default_factory=dict)
    resource_limits: Dict[str, Any] = Field(default_factory=dict)

    def update_timestamp(self):
        """Update the updated_at timestamp."""
        self.updated_at = datetime.now(timezone.utc)

    def add_hypothesis(self, hypothesis_id: str):
        """Add hypothesis to pool."""
        if hypothesis_id not in self.hypothesis_pool:
            self.hypothesis_pool.append(hypothesis_id)
            self.update_timestamp()

    def mark_tested(self, hypothesis_id: str):
        """Mark hypothesis as tested."""
        if hypothesis_id not in self.tested_hypotheses:
            self.tested_hypotheses.append(hypothesis_id)
            self.update_timestamp()

    def mark_supported(self, hypothesis_id: str):
        """Mark hypothesis as supported."""
        if hypothesis_id not in self.supported_hypotheses:
            self.supported_hypotheses.append(hypothesis_id)
            self.mark_tested(hypothesis_id)

    def mark_rejected(self, hypothesis_id: str):
        """Mark hypothesis as rejected."""
        if hypothesis_id not in self.rejected_hypotheses:
            self.rejected_hypotheses.append(hypothesis_id)
            self.mark_tested(hypothesis_id)

    def add_experiment(self, protocol_id: str):
        """Add experiment to queue."""
        if protocol_id not in self.experiment_queue:
            self.experiment_queue.append(protocol_id)
            self.update_timestamp()

    def mark_experiment_complete(self, protocol_id: str):
        """Mark experiment as completed."""
        if protocol_id in self.experiment_queue:
            self.experiment_queue.remove(protocol_id)
        if protocol_id not in self.completed_experiments:
            self.completed_experiments.append(protocol_id)
            self.update_timestamp()

    def add_result(self, result_id: str):
        """Add result."""
        if result_id not in self.results:
            self.results.append(result_id)
            self.update_timestamp()

    def increment_iteration(self):
        """Increment iteration counter."""
        self.iteration_count += 1
        self.update_timestamp()

    def get_untested_hypotheses(self) -> List[str]:
        """Get list of hypotheses that haven't been tested."""
        return [h for h in self.hypothesis_pool if h not in self.tested_hypotheses]

    def get_testability_rate(self) -> float:
        """Calculate ratio of tested to total hypotheses."""
        if not self.hypothesis_pool:
            return 0.0
        return len(self.tested_hypotheses) / len(self.hypothesis_pool)

    def get_support_rate(self) -> float:
        """Calculate ratio of supported to tested hypotheses."""
        if not self.tested_hypotheses:
            return 0.0
        return len(self.supported_hypotheses) / len(self.tested_hypotheses)


class ResearchWorkflow:
    """
    State machine for managing autonomous research workflow.

    Manages state transitions, validates allowed transitions,
    and maintains transition history for the research process.
    """

    # Define allowed transitions
    # NOTE on CONVERGED and ERROR as targets: both are *terminal outcomes*, and
    # arriving at either must never itself raise. Several states previously
    # omitted CONVERGED, so a run that decided it was finished while in
    # EXECUTING/ANALYZING/ERROR died with ValueError at the exact moment it
    # tried to succeed -- discarding the report after the science was already
    # done. That decision is reachable from those states in practice (the
    # action-limit guard returns CONVERGE regardless of state, and
    # `_should_check_convergence` explicitly whitelists ERROR), so the guard was
    # rejecting legitimate paths rather than preventing invalid ones. ERROR and
    # CONVERGED also permit self-transitions: re-entering a terminal state is a
    # no-op, not a violation.
    ALLOWED_TRANSITIONS = {
        # INITIALIZING deliberately CANNOT reach CONVERGED: a run that never
        # started cannot have converged, and no code path decides convergence
        # from here (unlike EXECUTING/ANALYZING/ERROR below, which are reachable
        # via the action-limit guard and `_should_check_convergence`). Two tests
        # pin this as the canonical invalid transition.
        WorkflowState.INITIALIZING: [
            WorkflowState.GENERATING_HYPOTHESES,
            WorkflowState.PAUSED,
            WorkflowState.ERROR
        ],
        WorkflowState.GENERATING_HYPOTHESES: [
            WorkflowState.DESIGNING_EXPERIMENTS,
            WorkflowState.CONVERGED,
            WorkflowState.PAUSED,
            WorkflowState.ERROR
        ],
        WorkflowState.DESIGNING_EXPERIMENTS: [
            WorkflowState.EXECUTING,
            WorkflowState.GENERATING_HYPOTHESES,  # If need more hypotheses
            WorkflowState.CONVERGED,  # Hypothesis-only run: no data to experiment on
            WorkflowState.PAUSED,
            WorkflowState.ERROR
        ],
        WorkflowState.EXECUTING: [
            WorkflowState.ANALYZING,
            WorkflowState.CONVERGED,
            WorkflowState.ERROR,
            WorkflowState.PAUSED
        ],
        WorkflowState.ANALYZING: [
            WorkflowState.REFINING,
            WorkflowState.DESIGNING_EXPERIMENTS,  # If need immediate retest
            WorkflowState.CONVERGED,
            WorkflowState.PAUSED,
            WorkflowState.ERROR
        ],
        WorkflowState.REFINING: [
            WorkflowState.GENERATING_HYPOTHESES,  # Generate new/refined hypotheses
            WorkflowState.DESIGNING_EXPERIMENTS,  # Design follow-up experiments
            WorkflowState.CONVERGED,  # Research complete
            WorkflowState.PAUSED,
            WorkflowState.ERROR
        ],
        WorkflowState.CONVERGED: [
            WorkflowState.GENERATING_HYPOTHESES,  # Restart if new question
            WorkflowState.CONVERGED,  # Already done; re-deciding is a no-op
        ],
        WorkflowState.PAUSED: [
            WorkflowState.GENERATING_HYPOTHESES,
            WorkflowState.DESIGNING_EXPERIMENTS,
            WorkflowState.EXECUTING,
            WorkflowState.ANALYZING,
            WorkflowState.REFINING,
            WorkflowState.ERROR
        ],
        WorkflowState.ERROR: [
            WorkflowState.INITIALIZING,  # Restart
            WorkflowState.GENERATING_HYPOTHESES,  # Resume from hypothesis gen
            WorkflowState.CONVERGED,  # Give up cleanly and report what we have
            WorkflowState.ERROR,  # Another error while already failed
            WorkflowState.PAUSED
        ]
    }

    def __init__(
        self,
        initial_state: WorkflowState = WorkflowState.INITIALIZING,
        research_plan: Optional[ResearchPlan] = None
    ):
        """
        Initialize workflow state machine.

        Args:
            initial_state: Starting state
            research_plan: Research plan to manage
        """
        self.current_state = initial_state
        self.research_plan = research_plan
        self.transition_history: List[WorkflowTransition] = []

        logger.info(f"ResearchWorkflow initialized in state: {self.current_state}")

    def can_transition_to(self, target_state: WorkflowState) -> bool:
        """
        Check if transition to target state is allowed.

        Args:
            target_state: Desired target state

        Returns:
            bool: True if transition is allowed
        """
        allowed = self.ALLOWED_TRANSITIONS.get(self.current_state, [])
        return target_state in allowed

    def transition_to(
        self,
        target_state: WorkflowState,
        action: str = "",
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Transition to a new state.

        Args:
            target_state: State to transition to
            action: Description of the action triggering transition
            metadata: Additional metadata about the transition

        Returns:
            bool: True if transition successful

        Raises:
            ValueError: If transition is not allowed
        """
        if not self.can_transition_to(target_state):
            raise ValueError(
                f"Invalid transition from {self.current_state} to {target_state}. "
                f"Allowed transitions: {self.ALLOWED_TRANSITIONS[self.current_state]}"
            )

        # Calculate time in previous state
        time_in_state = 0.0
        if self.transition_history:
            last_transition = self.transition_history[-1]
            time_in_state = (datetime.now(timezone.utc) - last_transition.timestamp).total_seconds()

        # Log workflow transition if enabled
        log_transitions = False
        try:
            from kosmos.config import get_config
            log_transitions = get_config().logging.log_workflow_transitions
        except Exception as e:
            logger.debug(f"Config loading for workflow transitions failed: {e}")

        if log_transitions:
            logger.debug(
                "[WORKFLOW] Transition: %s -> %s (was in %s for %.2fs) action='%s'",
                self.current_state.value,
                target_state.value,
                self.current_state.value,
                time_in_state,
                action
            )

        # Create transition record
        transition = WorkflowTransition(
            from_state=self.current_state,
            to_state=target_state,
            action=action or f"Transition to {target_state}",
            metadata=metadata or {}
        )

        # Update state
        self.current_state = target_state
        self.transition_history.append(transition)

        # Update research plan if exists
        if self.research_plan:
            self.research_plan.current_state = target_state
            self.research_plan.update_timestamp()

        logger.info(f"Transitioned to {target_state}: {action}")
        return True

    def get_allowed_next_states(self) -> List[WorkflowState]:
        """Get list of states that can be transitioned to from current state."""
        return self.ALLOWED_TRANSITIONS.get(self.current_state, [])

    def get_transition_history(self) -> List[WorkflowTransition]:
        """Get full transition history."""
        return self.transition_history.copy()

    def get_recent_transitions(self, n: int = 5) -> List[WorkflowTransition]:
        """Get N most recent transitions."""
        return self.transition_history[-n:]

    def reset(self):
        """Reset workflow to initial state."""
        self.current_state = WorkflowState.INITIALIZING
        self.transition_history = []
        if self.research_plan:
            self.research_plan.current_state = WorkflowState.INITIALIZING
            self.research_plan.update_timestamp()
        logger.info("ResearchWorkflow reset to INITIALIZING state")

    def to_dict(self) -> Dict[str, Any]:
        """Export workflow state to dictionary.

        `WorkflowTransition` declares `use_enum_values=True`, so pydantic stores
        its `from_state`/`to_state` as the enum's STRING value rather than the
        member -- while `self.current_state` is a plain attribute and stays an
        enum. Calling `.value` unconditionally therefore raised
        `AttributeError: 'str' object has no attribute 'value'` on any workflow
        that had actually transitioned, i.e. on every real run. `_state_value`
        normalises the two, so this stays correct whichever way a later change
        moves that model config.
        """
        return {
            "current_state": _state_value(self.current_state),
            "transition_count": len(self.transition_history),
            "recent_transitions": [
                {
                    "from": _state_value(t.from_state),
                    "to": _state_value(t.to_state),
                    "action": t.action,
                    "timestamp": t.timestamp.isoformat()
                }
                for t in self.get_recent_transitions(5)
            ]
        }

    def get_state_duration(self, state: WorkflowState) -> float:
        """
        Calculate total time spent in a given state.

        Args:
            state: State to calculate duration for

        Returns:
            float: Total seconds spent in state
        """
        total_seconds = 0.0

        for i, transition in enumerate(self.transition_history):
            if transition.to_state == state:
                # Find next transition out of this state
                end_time = None
                for j in range(i + 1, len(self.transition_history)):
                    if self.transition_history[j].from_state == state:
                        end_time = self.transition_history[j].timestamp
                        break

                # If still in this state, use current time
                if end_time is None and self.current_state == state:
                    end_time = datetime.now(timezone.utc)

                if end_time:
                    duration = (end_time - transition.timestamp).total_seconds()
                    total_seconds += duration

        return total_seconds

    def get_state_statistics(self) -> Dict[str, Any]:
        """Get statistics about state transitions."""
        state_counts = {}
        state_durations = {}

        for state in WorkflowState:
            count = sum(1 for t in self.transition_history if t.to_state == state)
            duration = self.get_state_duration(state)

            if count > 0:
                state_counts[state.value] = count
                state_durations[state.value] = duration

        return {
            "state_visit_counts": state_counts,
            "state_durations_seconds": state_durations,
            "total_transitions": len(self.transition_history),
            "current_state": self.current_state.value
        }
