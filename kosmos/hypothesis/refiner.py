"""
Hypothesis Refiner - Refines, retires, and spawns hypotheses based on experimental results (Phase 7).

Implements hybrid retirement logic:
- Rule-based: Consecutive failures
- Confidence-based: Bayesian updating
- Claude-powered: Ambiguous cases
"""

from typing import List, Dict, Optional, Any
from datetime import datetime, timezone
from enum import Enum
import logging
import json
import os
import uuid
from pydantic import BaseModel, Field

from kosmos.models.hypothesis import Hypothesis, HypothesisStatus
from kosmos.models.result import ExperimentResult, ResultStatus
from kosmos.core.llm import get_client
from kosmos.knowledge.vector_db import PaperVectorDB as VectorDB, HAS_CHROMADB

logger = logging.getLogger(__name__)

# Reasoning models spend this budget BEFORE emitting any content, so a
# prose-sized allowance buys nothing: the provider returns
# `finish_reason=length` with an empty message, the JSON never parses, and the
# refiner logs "Could not parse variants JSON" and moves on having refined
# nothing. That failure is silent -- the run continues, minutes per iteration
# go to calls that return no text -- which is why it survived 231 occurrences
# in one log before anyone traced it. Sized like `_CODEGEN_MAX_TOKENS`, for the
# same reason and with the same env override.
_REFINER_MAX_TOKENS = int(os.environ.get("KOSMOS_REFINER_MAX_TOKENS", "4096"))


class RetirementDecision(str, Enum):
    """Decision on hypothesis status."""

    CONTINUE_TESTING = "continue_testing"
    RETIRE = "retire"
    REFINE = "refine"
    SPAWN_VARIANT = "spawn_variant"


class RefinerAction(str, Enum):
    """Actions taken by the refiner."""

    NO_ACTION = "no_action"
    RETIRED = "retired"
    REFINED = "refined"
    SPAWNED = "spawned"
    MERGED = "merged"


class HypothesisLineage(BaseModel):
    """
    Hypothesis lineage tracking.

    Tracks parent-child relationships for hypothesis evolution.
    """

    hypothesis_id: str
    parent_id: Optional[str] = None
    children_ids: List[str] = Field(default_factory=list)
    generation: int = 1
    refinement_reason: Optional[str] = None
    evidence_basis: List[str] = Field(default_factory=list)  # Result IDs
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class HypothesisRefiner:
    """
    Refines hypotheses based on experimental results.

    Capabilities:
    - Evaluate hypothesis status (supported/rejected/inconclusive)
    - Retire hypotheses (rule-based, confidence-based, or Claude decision)
    - Refine hypotheses (improve based on results)
    - Spawn variants (generate new hypotheses from findings)
    - Merge similar supported hypotheses
    - Detect contradictions between hypotheses
    """

    def __init__(
        self,
        llm_client=None,
        vector_db: Optional[VectorDB] = None,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize hypothesis refiner.

        Args:
            llm_client: Claude LLM client
            vector_db: Vector database for semantic similarity
            config: Configuration dict
        """
        self.llm_client = llm_client or get_client()
        self.vector_db = vector_db

        # Configuration
        self.config = config or {}
        self.failure_threshold = self.config.get("failure_threshold", 3)  # Consecutive failures before retirement
        self.confidence_retirement_threshold = self.config.get("confidence_retirement_threshold", 0.1)  # Bayesian confidence
        self.similarity_threshold = self.config.get("similarity_threshold", 0.8)  # For contradiction/merging

        # Tracking
        self.lineage_tracking: Dict[str, HypothesisLineage] = {}  # hypothesis_id -> lineage

        logger.info("HypothesisRefiner initialized")

    # ========================================================================
    # HYPOTHESIS EVALUATION
    # ========================================================================

    def evaluate_hypothesis_status(
        self,
        hypothesis: Hypothesis,
        result: ExperimentResult,
        results_history: Optional[List[ExperimentResult]] = None
    ) -> RetirementDecision:
        """
        Evaluate hypothesis status and decide on next action.

        Uses hybrid approach:
        1. Rule-based: Check consecutive failures
        2. Confidence-based: Bayesian probability update
        3. Claude-powered: For ambiguous cases

        Args:
            hypothesis: Hypothesis to evaluate
            result: Latest experiment result
            results_history: Historical results for this hypothesis

        Returns:
            RetirementDecision: What to do with this hypothesis
        """
        logger.info(f"Evaluating hypothesis {hypothesis.id}")

        results_history = results_history or []
        all_results = results_history + [result]

        # 1. Rule-based check: Consecutive failures
        consecutive_failures = self._count_consecutive_failures(all_results)

        if consecutive_failures >= self.failure_threshold:
            logger.info(
                f"Hypothesis {hypothesis.id} has {consecutive_failures} consecutive failures "
                f"(threshold: {self.failure_threshold})"
            )
            return RetirementDecision.RETIRE

        # 2. Confidence-based check: Bayesian updating
        posterior_confidence = self._bayesian_confidence_update(hypothesis, all_results)

        if posterior_confidence < self.confidence_retirement_threshold:
            logger.info(
                f"Hypothesis {hypothesis.id} has low confidence {posterior_confidence:.3f} "
                f"(threshold: {self.confidence_retirement_threshold})"
            )
            return RetirementDecision.RETIRE

        # 3. Check if should refine vs continue
        if result.supports_hypothesis is False:
            # Rejected but not enough to retire - should refine
            return RetirementDecision.REFINE
        elif result.supports_hypothesis is None:
            # Inconclusive - spawn variant to explore
            return RetirementDecision.SPAWN_VARIANT
        else:
            # Supported - continue testing or spawn variants to explore related ideas
            if len(all_results) >= 2:
                # After 2+ successes, might want to explore variants
                return RetirementDecision.SPAWN_VARIANT
            else:
                return RetirementDecision.CONTINUE_TESTING

    def _count_consecutive_failures(self, results: List[ExperimentResult]) -> int:
        """Count consecutive failures (rejected or error) from most recent results."""
        count = 0
        for result in reversed(results):
            if result.supports_hypothesis is False or result.status == ResultStatus.FAILED:
                count += 1
            else:
                break
        return count

    def _bayesian_confidence_update(
        self,
        hypothesis: Hypothesis,
        results: List[ExperimentResult]
    ) -> float:
        """
        Update hypothesis confidence using Bayesian inference.

        Simple Bayesian model:
        - Prior: hypothesis.confidence_score (or 0.5 if None)
        - Likelihood: Based on p-value and effect size
        - Posterior: Updated confidence

        Args:
            hypothesis: Hypothesis
            results: All experimental results

        Returns:
            float: Posterior confidence (0.0 - 1.0)
        """
        # Start with prior
        prior = hypothesis.confidence_score if hypothesis.confidence_score is not None else 0.5

        # Update with each result
        confidence = prior

        for result in results:
            # Likelihood based on support and strength of evidence
            if result.supports_hypothesis is True:
                # Supported - increase confidence
                # Stronger evidence (lower p-value, larger effect) = larger update
                p_value = result.primary_p_value if result.primary_p_value is not None else 0.5
                effect_size = abs(result.primary_effect_size) if result.primary_effect_size is not None else 0.0

                # Evidence strength (0-1)
                evidence_strength = (1 - p_value) * min(effect_size, 1.0)

                # Bayesian update (simple version)
                confidence = confidence + (1 - confidence) * evidence_strength * 0.3

            elif result.supports_hypothesis is False:
                # Rejected - decrease confidence
                p_value = result.primary_p_value if result.primary_p_value is not None else 0.5
                effect_size = abs(result.primary_effect_size) if result.primary_effect_size is not None else 0.0

                # Evidence strength against hypothesis
                evidence_strength = (1 - p_value) * min(effect_size, 1.0)

                # Bayesian update
                confidence = confidence * (1 - evidence_strength * 0.3)

            # Inconclusive results don't change confidence much

        return max(0.0, min(1.0, confidence))

    # ========================================================================
    # HYPOTHESIS REFINEMENT
    # ========================================================================

    def refine_hypothesis(
        self,
        hypothesis: Hypothesis,
        result: ExperimentResult
    ) -> Hypothesis:
        """
        Refine hypothesis based on experimental result.

        Creates a new refined version of the hypothesis.

        Args:
            hypothesis: Original hypothesis
            result: Experiment result that suggests refinement

        Returns:
            Hypothesis: Refined hypothesis
        """
        logger.info(f"Refining hypothesis {hypothesis.id}")

        # Use Claude to generate refined hypothesis
        prompt = f"""You are refining a scientific hypothesis based on experimental results.

Original Hypothesis:
Statement: {hypothesis.statement}
Rationale: {hypothesis.rationale}
Domain: {hypothesis.domain}

Experimental Result:
- Supported: {result.supports_hypothesis}
- P-value: {result.primary_p_value}
- Effect size: {result.primary_effect_size}
- Primary test: {result.primary_test}

Task: Refine the hypothesis to better align with the evidence. The refined hypothesis should:
1. Keep the core idea but adjust the statement to match observed effects
2. Update the rationale to incorporate new evidence
3. Be more precise or nuanced based on what was learned

Respond with JSON:
{{
    "refined_statement": "Refined hypothesis statement",
    "refined_rationale": "Updated rationale incorporating evidence",
    "changes_made": "What was changed and why",
    "confidence": 0.0-1.0
}}
"""

        try:
            response = self.llm_client.generate(prompt, max_tokens=_REFINER_MAX_TOKENS)

            # Parse JSON
            json_start = response.find('{')
            json_end = response.rfind('}') + 1
            if json_start >= 0 and json_end > json_start:
                json_str = response[json_start:json_end]
                refinement = json.loads(json_str)

                # Create refined hypothesis with new ID
                refined = Hypothesis(
                    id=f"hyp_{uuid.uuid4().hex[:12]}",
                    research_question=hypothesis.research_question,
                    statement=refinement.get("refined_statement", hypothesis.statement),
                    rationale=refinement.get("refined_rationale", hypothesis.rationale),
                    domain=hypothesis.domain,
                    status=HypothesisStatus.GENERATED,
                    testability_score=hypothesis.testability_score,
                    novelty_score=hypothesis.novelty_score,  # May decrease slightly
                    confidence_score=refinement.get("confidence", 0.5),
                    priority_score=hypothesis.priority_score,
                    parent_hypothesis_id=hypothesis.id,
                    generation=hypothesis.generation + 1,
                    refinement_count=0,
                    evolution_history=[{
                        "action": "refined",
                        "based_on_result": result.id,
                        "changes": refinement.get("changes_made", ""),
                        "timestamp": datetime.now(timezone.utc).isoformat()
                    }]
                )

                # Track lineage
                self._track_lineage(refined, hypothesis, "refined", [result.id])

                logger.info(f"Created refined hypothesis (generation {refined.generation})")
                return refined

            else:
                logger.warning("Could not parse refinement JSON")
                return hypothesis

        except Exception as e:
            logger.error(f"Error refining hypothesis: {e}")
            return hypothesis

    # ========================================================================
    # HYPOTHESIS SPAWNING
    # ========================================================================

    def spawn_variant(
        self,
        hypothesis: Hypothesis,
        result: ExperimentResult,
        num_variants: int = 2
    ) -> List[Hypothesis]:
        """
        Spawn variant hypotheses based on findings.

        When a hypothesis is supported or shows interesting patterns,
        spawn related hypotheses to explore.

        Args:
            hypothesis: Original hypothesis
            result: Result that suggests variants
            num_variants: Number of variants to generate

        Returns:
            List[Hypothesis]: Spawned variant hypotheses
        """
        logger.info(f"Spawning {num_variants} variants from hypothesis {hypothesis.id}")

        prompt = f"""You are generating variant hypotheses based on experimental findings.

Original Hypothesis:
Statement: {hypothesis.statement}
Rationale: {hypothesis.rationale}

Experimental Result:
- Supported: {result.supports_hypothesis}
- P-value: {result.primary_p_value}
- Effect size: {result.primary_effect_size}

Task: Generate {num_variants} related but distinct hypotheses that:
1. Explore related aspects or mechanisms
2. Test boundary conditions or moderating factors
3. Investigate related phenomena

Respond with JSON array:
[
    {{
        "statement": "Variant hypothesis statement",
        "rationale": "Why this variant is worth testing",
        "relationship": "How it relates to original"
    }},
    ...
]
"""

        try:
            response = self.llm_client.generate(prompt, max_tokens=_REFINER_MAX_TOKENS)

            # Parse JSON
            json_start = response.find('[')
            json_end = response.rfind(']') + 1
            if json_start >= 0 and json_end > json_start:
                json_str = response[json_start:json_end]
                variants_data = json.loads(json_str)

                variants = []
                for i, variant_data in enumerate(variants_data[:num_variants]):
                    variant = Hypothesis(
                        research_question=hypothesis.research_question,
                        statement=variant_data.get("statement", ""),
                        rationale=variant_data.get("rationale", ""),
                        domain=hypothesis.domain,
                        status=HypothesisStatus.GENERATED,
                        parent_hypothesis_id=hypothesis.id,
                        generation=hypothesis.generation + 1,
                        evolution_history=[{
                            "action": "spawned",
                            "based_on_result": result.id,
                            "relationship": variant_data.get("relationship", ""),
                            "timestamp": datetime.now(timezone.utc).isoformat()
                        }]
                    )

                    if not variant.id:
                        variant.id = f"hyp_{uuid.uuid4().hex[:12]}"
                    self._track_lineage(variant, hypothesis, "spawned", [result.id])
                    variants.append(variant)

                logger.info(f"Spawned {len(variants)} variant hypotheses")
                return variants

            else:
                logger.warning("Could not parse variants JSON")
                return []

        except Exception as e:
            logger.error(f"Error spawning variants: {e}")
            return []

    # ========================================================================
    # HYPOTHESIS RETIREMENT
    # ========================================================================

    def retire_hypothesis(
        self,
        hypothesis: Hypothesis,
        rationale: str
    ) -> Hypothesis:
        """
        Retire a hypothesis.

        Args:
            hypothesis: Hypothesis to retire
            rationale: Reason for retirement

        Returns:
            Hypothesis: Updated hypothesis with REJECTED status
        """
        hypothesis.status = HypothesisStatus.REJECTED
        hypothesis.updated_at = datetime.now(timezone.utc)
        hypothesis.evolution_history.append({
            "action": "retired",
            "rationale": rationale,
            "timestamp": datetime.now(timezone.utc).isoformat()
        })

        logger.info(f"Retired hypothesis {hypothesis.id}: {rationale}")
        return hypothesis

    # ========================================================================
    # CONTRADICTION DETECTION
    # ========================================================================

    def _compute_semantic_similarity(self, text1: str, text2: str) -> float:
        """
        Compute semantic similarity between two texts.

        Uses vector DB if available, otherwise simple word overlap.

        Args:
            text1: First text
            text2: Second text

        Returns:
            float: Similarity score (0.0 - 1.0)
        """
        if self.vector_db and HAS_CHROMADB:
            try:
                # Use vector DB embeddings
                # (Simplified - actual implementation would query vector DB)
                # For now, use simple word overlap
                words1 = set(text1.lower().split())
                words2 = set(text2.lower().split())

                if not words1 or not words2:
                    return 0.0

                overlap = len(words1 & words2)
                union = len(words1 | words2)

                return overlap / union if union > 0 else 0.0

            except Exception as e:
                logger.warning(f"Error computing similarity with vector DB: {e}")
                return 0.0
        else:
            # Simple word overlap
            words1 = set(text1.lower().split())
            words2 = set(text2.lower().split())

            if not words1 or not words2:
                return 0.0

            overlap = len(words1 & words2)
            union = len(words1 | words2)

            return overlap / union if union > 0 else 0.0

    def _overall_support(self, results: List[ExperimentResult]) -> Optional[bool]:
        """Determine overall support from multiple results."""
        if not results:
            return None

        supports = [r.supports_hypothesis for r in results if r.supports_hypothesis is not None]

        if not supports:
            return None

        # Majority vote
        support_count = sum(1 for s in supports if s is True)
        reject_count = sum(1 for s in supports if s is False)

        if support_count > reject_count:
            return True
        elif reject_count > support_count:
            return False
        else:
            return None  # Tied

    # ========================================================================
    # HYPOTHESIS MERGING
    # ========================================================================

    # ========================================================================
    # LINEAGE TRACKING
    # ========================================================================

    def _track_lineage(
        self,
        hypothesis: Hypothesis,
        parent: Hypothesis,
        action: str,
        evidence: List[str]
    ):
        """Track hypothesis lineage."""
        lineage = HypothesisLineage(
            hypothesis_id=hypothesis.id,
            parent_id=parent.id,
            generation=hypothesis.generation,
            refinement_reason=action,
            evidence_basis=evidence
        )

        if hypothesis.id:
            self.lineage_tracking[hypothesis.id] = lineage

        # Update parent's children
        if parent.id and parent.id in self.lineage_tracking:
            parent_lineage = self.lineage_tracking[parent.id]
            if hypothesis.id and hypothesis.id not in parent_lineage.children_ids:
                parent_lineage.children_ids.append(hypothesis.id)

    def get_lineage(self, hypothesis_id: str) -> Optional[HypothesisLineage]:
        """Get lineage for a hypothesis."""
        return self.lineage_tracking.get(hypothesis_id)

    # ========================================================================
    # UTILITIES
    # ========================================================================

    def _format_results_for_claude(self, results: List[ExperimentResult]) -> str:
        """Format results for Claude prompt."""
        formatted = []

        for i, result in enumerate(results, 1):
            formatted.append(
                f"Experiment {i}:\n"
                f"  - Supported: {result.supports_hypothesis}\n"
                f"  - P-value: {result.primary_p_value}\n"
                f"  - Effect size: {result.primary_effect_size}\n"
                f"  - Test: {result.primary_test}"
            )

        return "\n\n".join(formatted)
