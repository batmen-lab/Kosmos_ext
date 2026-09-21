"""
Hypothesis Generator Agent.

Generates scientific hypotheses from research questions using Claude,
with literature context and novelty checking.
"""

import logging
import os
import time
import uuid
from typing import List, Dict, Any, Optional
from datetime import datetime

from kosmos.agents.base import BaseAgent, AgentMessage, MessageType, AgentStatus
from kosmos.core.llm import get_client
from kosmos.utils.compat import model_to_dict
from kosmos.core.prompts import HYPOTHESIS_GENERATOR
from kosmos.models.hypothesis import (
    Hypothesis,
    HypothesisGenerationRequest,
    HypothesisGenerationResponse,
    HypothesisStatus,
    ExperimentType
)
from kosmos.literature.unified_search import UnifiedLiteratureSearch
from kosmos.literature.base_client import PaperMetadata
from kosmos.db.models import Hypothesis as DBHypothesis, HypothesisStatus as DBHypothesisStatus
from kosmos.db import get_session

logger = logging.getLogger(__name__)


def _shorten(text: str, limit: int) -> str:
    """Cut to `limit` characters at a word boundary, and say it was cut.

    A model that writes 540 characters has answered the question; refusing the
    answer costs the whole hypothesis, and the run then converges with "no
    testable hypotheses" -- which is a parsing failure wearing a research
    failure's clothes.
    """
    text = " ".join(str(text).split())
    if len(text) <= limit:
        return text
    cut = text[: limit - 1].rsplit(" ", 1)[0].rstrip(",;:")
    return (cut or text[: limit - 1]) + "…"


def _build_hypothesis(
    hyp_data,
    *,
    hypothesis_cls,
    status,
    research_question,
    domain,
    exp_types,
    context_papers,
    generated_by,
):
    """One hypothesis from the model's JSON, repaired where it is only too long.

    `statement` has a 500-character limit in the model and a generative LLM
    overshoots it often; the length is not the hypothesis. Anything else that
    fails validation still raises, so a genuinely broken payload is reported.
    """
    payload = dict(hyp_data)
    statement = str(payload.get("statement") or "")
    if len(statement) > 500:
        payload["statement"] = _shorten(statement, 500)
        logger.warning(
            "Hypothesis statement was %d characters (limit 500); shortened for "
            "use rather than dropped",
            len(statement),
        )
    return hypothesis_cls(
        id=str(uuid.uuid4()),
        research_question=research_question,
        statement=payload["statement"],
        rationale=payload["rationale"],
        domain=domain,
        status=status,
        testability_score=payload.get("testability_score"),
        confidence_score=payload.get("confidence_score"),
        suggested_experiment_types=exp_types,
        related_papers=[
            p.arxiv_id or p.doi or p.title
            for p in context_papers
            if p is not None and (p.arxiv_id or p.doi or p.title)
        ],
        generated_by=generated_by,
    )


def _literature_enabled_by_default() -> bool:
    """`USE_LITERATURE_CONTEXT`, for an agent built without a config.

    Kosmos's own CLI turns literature off for a training run; an agent built by
    another entry point used to default it on regardless, which is how a run
    that asked for no literature still spent minutes on a rate-limited API.
    """
    raw = os.getenv("USE_LITERATURE_CONTEXT", "true").strip().lower()
    return raw not in ("0", "false", "no", "off")


def _novelty_enabled_by_default() -> bool:
    """`REQUIRE_NOVELTY_CHECK`, for an agent built without a config."""
    raw = os.getenv("REQUIRE_NOVELTY_CHECK", "true").strip().lower()
    return raw not in ("0", "false", "no", "off")


class HypothesisGeneratorAgent(BaseAgent):
    """
    Agent for generating scientific hypotheses.

    Capabilities:
    - Generate multiple hypotheses from research questions
    - Use literature context for informed hypothesis generation
    - Validate hypothesis quality and testability
    - Store hypotheses in database
    - Integration with novelty checking (optional)

    Example:
        ```python
        agent = HypothesisGeneratorAgent(config={
            "num_hypotheses": 3,
            "use_literature_context": True
        })
        agent.start()

        # Generate hypotheses
        response = agent.generate_hypotheses(
            "How does attention mechanism affect transformer performance?"
        )

        for hyp in response.hypotheses:
            print(f"{hyp.statement} (novelty: {hyp.novelty_score})")
        ```
    """

    def __init__(
        self,
        agent_id: Optional[str] = None,
        agent_type: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize Hypothesis Generator Agent.

        Args:
            agent_id: Unique agent identifier
            agent_type: Agent type name
            config: Configuration dictionary
        """
        super().__init__(agent_id, agent_type or "HypothesisGeneratorAgent", config)

        # Configuration
        self.num_hypotheses = self.config.get("num_hypotheses", 3)
        # The default follows the environment, not a hard-coded True: this agent
        # is also built by callers that pass no config, and a literature search
        # that the caller switched off must stay off -- it is the step that made
        # hypothesis generation look stuck (a rate-limited API, retried for
        # minutes on a thread nothing can cancel).
        self.use_literature_context = self.config.get(
            "use_literature_context", _literature_enabled_by_default()
        )
        self.max_papers_context = self.config.get("max_papers_context", 10)
        # Same rule as the literature switch: an agent built without a config
        # follows `REQUIRE_NOVELTY_CHECK`, because the novelty check embeds with
        # a transformer on the research loop's thread and a caller that turned it
        # off must not get it back by default.
        self.require_novelty_check = self.config.get(
            "require_novelty_check", _novelty_enabled_by_default()
        )
        self.min_novelty_score = self.config.get("min_novelty_score", 0.5)

        # Components
        self.llm_client = get_client()
        self.literature_search = UnifiedLiteratureSearch() if self.use_literature_context else None

        logger.info(f"Initialized HypothesisGeneratorAgent {self.agent_id}")

    def execute(self, message):
        """
        Execute agent task from message.

        Args:
            message: AgentMessage with task details

        Returns:
            AgentMessage: Response message with results
        """
        self.status = AgentStatus.WORKING

        try:
            task_type = message.content.get("task_type")

            if task_type == "generate_hypotheses":
                research_question = message.content.get("research_question")
                num_hypotheses = message.content.get("num_hypotheses", self.num_hypotheses)
                domain = message.content.get("domain")

                response = self.generate_hypotheses(
                    research_question=research_question,
                    num_hypotheses=num_hypotheses,
                    domain=domain
                )

                return AgentMessage(
                    type=MessageType.RESPONSE,
                    from_agent=self.agent_id,
                    to_agent=message.from_agent,
                    content={"response": model_to_dict(response)},
                    correlation_id=message.correlation_id
                )

            else:
                raise ValueError(f"Unknown task type: {task_type}")

        except Exception as e:
            logger.error(f"Error executing task: {e}", exc_info=True)
            self.status = AgentStatus.ERROR
            return AgentMessage(
                type=MessageType.ERROR,
                from_agent=self.agent_id,
                to_agent=message.from_agent,
                content={"error": str(e)},
                correlation_id=message.correlation_id
            )

        finally:
            self.status = AgentStatus.IDLE

    def generate_hypotheses(
        self,
        research_question: str,
        num_hypotheses: Optional[int] = None,
        domain: Optional[str] = None,
        store_in_db: bool = True
    ) -> HypothesisGenerationResponse:
        """
        Generate hypotheses from research question.

        Args:
            research_question: Research question to generate hypotheses for
            num_hypotheses: Number of hypotheses to generate (default: config value)
            domain: Scientific domain (auto-detected if None)
            store_in_db: Whether to store hypotheses in database

        Returns:
            HypothesisGenerationResponse: Generated hypotheses with metadata

        Example:
            ```python
            response = agent.generate_hypotheses(
                research_question="How does learning rate affect convergence?",
                num_hypotheses=3,
                domain="machine_learning"
            )
            ```
        """
        start_time = time.time()
        num_hypotheses = num_hypotheses or self.num_hypotheses

        logger.info(f"Generating {num_hypotheses} hypotheses for: '{research_question}'")

        # Step 1: Auto-detect domain if not provided
        if not domain:
            domain = self._detect_domain(research_question)
            logger.info(f"Auto-detected domain: {domain}")

        # Step 2: Gather literature context
        papers = []
        if self.use_literature_context and self.literature_search:
            papers = self._gather_literature_context(research_question, domain)
            logger.info(f"Gathered {len(papers)} papers for context")

        # Step 3: Generate hypotheses using Claude
        hypotheses = self._generate_with_claude(
            research_question=research_question,
            domain=domain,
            num_hypotheses=num_hypotheses,
            context_papers=papers
        )

        # Step 4: Validate hypotheses
        validated_hypotheses = []
        for hyp in hypotheses:
            try:
                if self._validate_hypothesis(hyp):
                    validated_hypotheses.append(hyp)
                else:
                    logger.warning(f"Hypothesis failed validation: {hyp.statement[:50]}...")
            except Exception as e:
                logger.error(f"Error validating hypothesis: {e}")

        logger.info(f"Generated {len(validated_hypotheses)} valid hypotheses")

        # Step 4b: Novelty scoring. It is annotated when the run wants it (and
        # only then: building the checker loads an embedding model and searching
        # prior art calls three APIs, none of which a training run needs), and it
        # is filtered only when `require_novelty_check` is on.
        if validated_hypotheses and self.require_novelty_check:
            try:
                from kosmos.hypothesis.novelty_checker import NoveltyChecker
                checker = NoveltyChecker(similarity_threshold=1.0 - self.min_novelty_score)
                novel = []
                for hyp in validated_hypotheses:
                    try:
                        report = checker.check_novelty(hyp)
                        hyp.novelty_score = report.novelty_score
                        if report.novelty_score < self.min_novelty_score:
                            logger.info("Filtered low-novelty hypothesis (%.2f): %s",
                                        report.novelty_score, hyp.statement[:60])
                        else:
                            novel.append(hyp)
                    except Exception as e:
                        logger.warning("Novelty check failed, keeping hypothesis: %s", e)
                        novel.append(hyp)  # Fail open
                validated_hypotheses = novel
                logger.info(f"After novelty scoring: {len(validated_hypotheses)} hypotheses")
            except ImportError:
                logger.warning("NoveltyChecker unavailable, skipping novelty scoring")
        elif validated_hypotheses:
            logger.info(
                "Novelty checking is off (REQUIRE_NOVELTY_CHECK=false); keeping "
                "all %d hypothesis(es) unscored",
                len(validated_hypotheses),
            )

        # Step 5: Store in database if requested
        if store_in_db:
            for hyp in validated_hypotheses:
                self._store_hypothesis(hyp)

        # Step 6: Calculate metrics
        generation_time = time.time() - start_time
        avg_novelty = None
        avg_testability = None

        if validated_hypotheses:
            novelty_scores = [h.novelty_score for h in validated_hypotheses if h.novelty_score is not None]
            if novelty_scores:
                avg_novelty = sum(novelty_scores) / len(novelty_scores)

            testability_scores = [h.testability_score for h in validated_hypotheses if h.testability_score is not None]
            if testability_scores:
                avg_testability = sum(testability_scores) / len(testability_scores)

        return HypothesisGenerationResponse(
            hypotheses=validated_hypotheses,
            research_question=research_question,
            domain=domain,
            generation_time_seconds=generation_time,
            num_papers_analyzed=len(papers),
            avg_novelty_score=avg_novelty,
            avg_testability_score=avg_testability
        )

    def _detect_domain(self, research_question: str) -> str:
        """
        Auto-detect scientific domain from research question.

        Args:
            research_question: Research question text

        Returns:
            str: Detected domain
        """
        prompt = f"""Analyze this research question and identify the primary scientific domain:

Research Question: "{research_question}"

Return ONLY the domain name (e.g., "machine_learning", "biology", "physics", "chemistry", "neuroscience", "astrophysics", "materials_science", "general").
No explanation needed."""

        try:
            response = self.llm_client.generate(
                prompt=prompt,
                max_tokens=50,
                temperature=0.0
            )
            domain = response.strip().lower().replace(" ", "_").replace("-", "_")
            return domain if domain else "general"

        except Exception as e:
            logger.error(f"Error detecting domain: {e}")
            return "general"

    def _gather_literature_context(
        self,
        research_question: str,
        domain: str
    ) -> List[PaperMetadata]:
        """
        Gather relevant literature for context.

        Args:
            research_question: Research question
            domain: Scientific domain

        Returns:
            List[PaperMetadata]: Relevant papers
        """
        if not self.literature_search:
            return []

        try:
            # Search for relevant papers
            query = research_question
            papers = self.literature_search.search(
                query=query,
                max_results=self.max_papers_context
            )

            logger.info(f"Found {len(papers)} papers for context")
            return papers

        except Exception as e:
            logger.error(f"Error gathering literature: {e}", exc_info=True)
            return []

    def _generate_with_claude(
        self,
        research_question: str,
        domain: str,
        num_hypotheses: int,
        context_papers: List[PaperMetadata]
    ) -> List[Hypothesis]:
        """
        Generate hypotheses using Claude with structured output.

        Args:
            research_question: Research question
            domain: Scientific domain
            num_hypotheses: Number of hypotheses to generate
            context_papers: Literature context

        Returns:
            List[Hypothesis]: Generated hypotheses
        """
        # Build literature context summary
        literature_context = ""
        if context_papers:
            literature_context = "Recent relevant literature:\n\n"
            # Filter out None papers and papers without titles
            valid_papers = [p for p in context_papers[:5] if p is not None and p.title]
            for i, paper in enumerate(valid_papers, 1):
                title = paper.title or "Untitled"
                year = paper.year or "N/A"
                literature_context += f"{i}. {title} ({year})\n"
                if paper.abstract:
                    literature_context += f"   Abstract: {paper.abstract[:200]}...\n"
                literature_context += "\n"

        # Create prompt
        prompt = HYPOTHESIS_GENERATOR.render(
            research_question=research_question,
            domain=domain,
            num_hypotheses=num_hypotheses,
            literature_context=literature_context or "No specific literature context provided."
        )

        # Define expected JSON schema
        schema = {
            "hypotheses": [
                {
                    "statement": "string (clear, testable hypothesis)",
                    "rationale": "string (scientific justification)",
                    "confidence_score": "float 0.0-1.0",
                    "testability_score": "float 0.0-1.0 (preliminary estimate)",
                    "suggested_experiment_types": ["computational | data_analysis | literature_synthesis"]
                }
            ]
        }

        try:
            # Call Claude with structured output
            response = self.llm_client.generate_structured(
                prompt=prompt,
                schema=schema,
                max_tokens=4000,
                temperature=0.7  # Slightly higher for creativity
            )

            # Parse response into Hypothesis objects
            hypotheses = []
            #: Why a hypothesis the model returned did not become one: dropping
            #: them silently is how a run reaches "no testable hypotheses" while
            #: the model did produce three.
            rejected: list[str] = []
            for i, hyp_data in enumerate(response.get("hypotheses", [])):
                try:
                    # Map experiment types
                    exp_types = []
                    for exp_type_str in hyp_data.get("suggested_experiment_types", []):
                        try:
                            exp_types.append(ExperimentType(exp_type_str))
                        except ValueError:
                            logger.warning(f"Unknown experiment type: {exp_type_str}")

                    hypothesis = _build_hypothesis(
                        hyp_data,
                        hypothesis_cls=Hypothesis,
                        research_question=research_question,
                        domain=domain,
                        status=HypothesisStatus.GENERATED,
                        exp_types=exp_types,
                        context_papers=context_papers,
                        generated_by=self.agent_id,
                    )
                    hypotheses.append(hypothesis)

                except Exception as e:
                    rejected.append(f"hypothesis {i}: {type(e).__name__}: {e}")
                    logger.error(f"Error parsing hypothesis {i}: {e}")
                    continue

            if rejected:
                # Said on the console, not only in the log file: a run whose
                # hypotheses were all rejected converges with "no testable
                # hypotheses", and that reads like a model failure when it is a
                # parsing one.
                print(
                    f"  ⚠ {len(rejected)} of {len(response.get('hypotheses', []))} "
                    f"hypothesis(es) from the model could not be used: "
                    + "; ".join(rejected[:2]),
                    flush=True,
                )
                logger.warning(
                    "Rejected %d hypothesis(es) from the model: %s",
                    len(rejected),
                    "; ".join(rejected),
                )

            return hypotheses

        except Exception as e:
            logger.error(f"Error generating hypotheses with Claude: {e}", exc_info=True)
            return []

    def _validate_hypothesis(self, hypothesis: Hypothesis) -> bool:
        """
        Validate hypothesis quality.

        Args:
            hypothesis: Hypothesis to validate

        Returns:
            bool: True if hypothesis passes validation
        """
        try:
            # Pydantic validation already happened during creation
            # Additional custom validation

            # Check statement is not too short
            if len(hypothesis.statement) < 15:
                logger.warning(f"Hypothesis statement too short: {hypothesis.statement}")
                return False

            # Check rationale is substantive
            if len(hypothesis.rationale) < 30:
                logger.warning(f"Hypothesis rationale too brief: {hypothesis.rationale[:50]}...")
                return False

            # Check for vague language
            vague_words = ["maybe", "might", "perhaps", "possibly", "potentially", "somewhat"]
            if any(word in hypothesis.statement.lower() for word in vague_words):
                logger.warning(f"Hypothesis contains vague language: {hypothesis.statement}")
                # Don't fail, but warn
                pass

            return True

        except Exception as e:
            logger.error(f"Validation error: {e}")
            return False

    def _store_hypothesis(self, hypothesis: Hypothesis) -> Optional[str]:
        """
        Store hypothesis in database.

        Args:
            hypothesis: Hypothesis to store

        Returns:
            Optional[str]: Hypothesis ID if successful
        """
        try:
            with get_session() as session:
                # Convert to DB model
                db_hypothesis = DBHypothesis(
                    id=hypothesis.id or str(uuid.uuid4()),
                    research_question=hypothesis.research_question,
                    statement=hypothesis.statement,
                    rationale=hypothesis.rationale,
                    domain=hypothesis.domain,
                    status=DBHypothesisStatus.GENERATED,
                    novelty_score=hypothesis.novelty_score,
                    testability_score=hypothesis.testability_score,
                    confidence_score=hypothesis.confidence_score,
                    related_papers=hypothesis.related_papers,
                    created_at=hypothesis.created_at,
                    updated_at=hypothesis.updated_at
                )

                session.add(db_hypothesis)
                session.commit()

                logger.info(f"Stored hypothesis {db_hypothesis.id} in database")
                hypothesis.id = db_hypothesis.id
                return db_hypothesis.id

        except Exception as e:
            logger.error(f"Error storing hypothesis: {e}", exc_info=True)
            return None

    def get_hypothesis_by_id(self, hypothesis_id: str) -> Optional[Hypothesis]:
        """
        Retrieve hypothesis from database by ID.

        Args:
            hypothesis_id: Hypothesis ID

        Returns:
            Optional[Hypothesis]: Hypothesis if found
        """
        try:
            with get_session() as session:
                db_hyp = session.query(DBHypothesis).filter(DBHypothesis.id == hypothesis_id).first()

                if not db_hyp:
                    return None

                # Convert DB model to Pydantic model
                hypothesis = Hypothesis(
                    id=db_hyp.id,
                    research_question=db_hyp.research_question,
                    statement=db_hyp.statement,
                    rationale=db_hyp.rationale,
                    domain=db_hyp.domain,
                    status=HypothesisStatus(db_hyp.status.value),
                    testability_score=db_hyp.testability_score,
                    novelty_score=db_hyp.novelty_score,
                    confidence_score=db_hyp.confidence_score,
                    related_papers=db_hyp.related_papers or [],
                    created_at=db_hyp.created_at,
                    updated_at=db_hyp.updated_at
                )

                return hypothesis

        except Exception as e:
            logger.error(f"Error retrieving hypothesis: {e}", exc_info=True)
            return None

    def list_hypotheses(
        self,
        domain: Optional[str] = None,
        status: Optional[HypothesisStatus] = None,
        limit: int = 100
    ) -> List[Hypothesis]:
        """
        List hypotheses from database with optional filtering.

        Args:
            domain: Filter by domain
            status: Filter by status
            limit: Maximum number to return

        Returns:
            List[Hypothesis]: Matching hypotheses
        """
        try:
            with get_session() as session:
                query = session.query(DBHypothesis)

                if domain:
                    query = query.filter(DBHypothesis.domain == domain)

                if status:
                    db_status = DBHypothesisStatus(status.value)
                    query = query.filter(DBHypothesis.status == db_status)

                query = query.order_by(DBHypothesis.created_at.desc()).limit(limit)

                hypotheses = []
                for db_hyp in query.all():
                    hypothesis = Hypothesis(
                        id=db_hyp.id,
                        research_question=db_hyp.research_question,
                        statement=db_hyp.statement,
                        rationale=db_hyp.rationale,
                        domain=db_hyp.domain,
                        status=HypothesisStatus(db_hyp.status.value),
                        testability_score=db_hyp.testability_score,
                        novelty_score=db_hyp.novelty_score,
                        confidence_score=db_hyp.confidence_score,
                        related_papers=db_hyp.related_papers or [],
                        created_at=db_hyp.created_at,
                        updated_at=db_hyp.updated_at
                    )
                    hypotheses.append(hypothesis)

                return hypotheses

        except Exception as e:
            logger.error(f"Error listing hypotheses: {e}", exc_info=True)
            return []
