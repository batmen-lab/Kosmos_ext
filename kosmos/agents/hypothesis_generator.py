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

# Budget for the hypothesis-generation call, overridable because 4000 was not
# enough and nothing said so until a run stalled on it.
#
# The failure is specific to reasoning models behind an OpenAI-compatible API:
# the reasoning trace and the answer SHARE max_tokens, so a long JSON schema
# can consume the whole budget and return finish_reason=length with no
# content at all. A live run logged
#
#   Model returned no content for a JSON-mode request (finish_reason=length).
#   With reasoning enabled the reasoning trace and the answer share max_tokens
#   (4000), so a long schema can leave nothing for the JSON itself.
#
# and then regenerated, and failed again -- 28 hypotheses recorded against 6
# experiments, because each failure left the director retrying instead of
# designing. The default is raised to 8192 on the same evidence; the override
# exists so a longer schema or a chattier model does not need a code change.
HYPOTHESIS_MAX_TOKENS = int(os.environ.get("KOSMOS_HYPOTHESIS_MAX_TOKENS", "8192"))


# `Hypothesis.statement` is capped at 500 characters by the model schema.
_STATEMENT_LIMIT = 500


def _fit_statement(statement: str, rationale: str) -> tuple:
    """Fit an over-long statement to the cap without losing what it says.

    Two of five hypotheses in one run were discarded outright for overrunning
    the 500-character cap -- the model writes the full test specification into
    the claim ("...at Benjamini-Hochberg FDR < 0.05"), and pydantic rejects
    the whole object. Losing a hypothesis the model did the work of forming,
    over a formatting rule, is disproportionate.

    So the overflow MOVES to the rationale rather than being cut away: a
    statement truncated mid-clause can assert something the model did not mean
    -- dropping "FDR < 0.05" turns a corrected claim into an uncorrected one.
    The split is taken at a sentence boundary where one exists, and at a word
    boundary otherwise.
    """
    text = (statement or "").strip()
    if len(text) <= _STATEMENT_LIMIT:
        return text, rationale

    head = text[:_STATEMENT_LIMIT]
    cut = max(head.rfind(". "), head.rfind("; "))
    if cut < _STATEMENT_LIMIT // 2:  # no usable sentence break near the end
        cut = head.rfind(" ")
    if cut <= 0:
        cut = _STATEMENT_LIMIT

    kept, overflow = text[:cut].rstrip(" ;,"), text[cut:].strip(" ;,.")
    if not kept.endswith("."):
        kept += "."
    logger.info(
        "Hypothesis statement was %d characters; moved %d to the rationale",
        len(text), len(overflow),
    )
    if overflow:
        rationale = (
            f"{rationale}\n\nFrom the hypothesis statement: {overflow}"
            if rationale else f"From the hypothesis statement: {overflow}"
        )
    return kept, rationale


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
        self.use_literature_context = self.config.get("use_literature_context", True)
        self.max_papers_context = self.config.get("max_papers_context", 10)
        self.require_novelty_check = self.config.get("require_novelty_check", True)
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
        store_in_db: bool = True,
        data_context: Optional[str] = None,
        data_driven: bool = False,
    ) -> HypothesisGenerationResponse:
        """
        Generate hypotheses from a research question and/or a dataset.

        Args:
            research_question: Research question to generate hypotheses for
            num_hypotheses: Number of hypotheses to generate (default: config value)
            domain: Scientific domain (auto-detected if None)
            store_in_db: Whether to store hypotheses in database
            data_context: Optional dataset schema/summary (column names, types,
                basic stats). When present, hypotheses are grounded in these
                actual variables -- this is what makes a data-driven run possible.

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

        # Step 3+4: Generate and validate, TOPPING UP to the requested count.
        #
        # A single generate call is not a reliable way to get N hypotheses: the
        # model returns a VALID array of whatever size it likes (observed 5, then
        # 3, then 1 across otherwise-identical runs), and nothing asked for the
        # rest -- so a run that wanted 5 proceeded on 1. The provider's retries
        # only cover empty/truncated/unparseable responses, not a valid-but-short
        # one, because only the generator knows the target count. So loop: keep
        # generating (deduping by statement) until we have `num_hypotheses` valid
        # ones or we run out of attempts. Bounded, and breaks early if a batch
        # yields nothing new (generation genuinely failing -- don't spin).
        validated_hypotheses: List[Hypothesis] = []
        seen_statements: set = set()
        max_attempts = max(2, int(os.environ.get("KOSMOS_HYPOTHESIS_MAX_ATTEMPTS", "5")))
        # OVERSHOOT the per-call request. Asking deepseek for exactly N under the
        # constrained D4 prompt returns a variable, usually-short count (observed
        # 1-6 for a request of 5); asking for N+buffer reliably returns ~N+buffer
        # (a request of 8 returned 8,8). We then dedupe and trim to N. This is the
        # single most effective lever -- the loop below is the backstop for the
        # rare short call.
        buffer = int(os.environ.get("KOSMOS_HYPOTHESIS_OVERSHOOT", "3"))
        request_n = num_hypotheses + buffer
        consecutive_empty = 0
        for attempt in range(max_attempts):
            if len(validated_hypotheses) >= num_hypotheses:
                break
            batch = self._generate_with_claude(
                research_question=research_question,
                domain=domain,
                num_hypotheses=request_n,
                context_papers=papers,
                data_context=data_context,
                data_driven=data_driven,
                # After the first attempt, tell the model what it already gave so
                # it adds DISTINCT hypotheses rather than repeating the same few.
                exclude_statements=[h.statement for h in validated_hypotheses] or None,
            )
            added = 0
            for hyp in batch:
                key = (hyp.statement or "").strip().lower()
                if not key or key in seen_statements:
                    continue  # dedupe across attempts
                try:
                    if self._validate_hypothesis(hyp):
                        seen_statements.add(key)
                        validated_hypotheses.append(hyp)
                        added += 1
                    else:
                        logger.warning(f"Hypothesis failed validation: {hyp.statement[:50]}...")
                except Exception as e:
                    logger.error(f"Error validating hypothesis: {e}")
            logger.info(
                "Hypothesis generation attempt %d/%d: +%d new (%d/%d)",
                attempt + 1, max_attempts, added, len(validated_hypotheses), num_hypotheses,
            )
            # A single empty batch is almost always a TRANSIENT flaky structured
            # response (deepseek occasionally returns no JSON), not a dead end --
            # the standalone generator returns the full N reliably on retry. So
            # retry it; only give up after TWO empties in a row, and never while
            # we still have nothing at all (that is exactly when a retry matters
            # most -- it is what previously capped a whole run at 1 hypothesis).
            if not batch:
                consecutive_empty += 1
                if consecutive_empty >= 2 and validated_hypotheses:
                    break
            else:
                consecutive_empty = 0
        # Do NOT trim to num_hypotheses here: keep the overshoot candidates so
        # novelty filtering (below) has spares to backfill from if it would
        # otherwise starve the run. The final trim happens AFTER novelty.
        if len(validated_hypotheses) < num_hypotheses:
            logger.warning(
                "Only %d/%d hypotheses after %d attempts (model returned short/empty "
                "batches); proceeding with what was generated.",
                len(validated_hypotheses), num_hypotheses, max_attempts,
            )

        logger.info(f"Generated {len(validated_hypotheses)} valid hypotheses")

        # Step 4b: Novelty scoring — always annotate, only filter if enabled
        if validated_hypotheses:
            try:
                from kosmos.hypothesis.novelty_checker import NoveltyChecker
                checker = NoveltyChecker(similarity_threshold=1.0 - self.min_novelty_score)
                novel, filtered = [], []
                for hyp in validated_hypotheses:
                    try:
                        report = checker.check_novelty(hyp)
                        hyp.novelty_score = report.novelty_score
                        if self.require_novelty_check and report.novelty_score < self.min_novelty_score:
                            logger.info("Filtered low-novelty hypothesis (%.2f): %s",
                                        report.novelty_score, hyp.statement[:60])
                            filtered.append(hyp)
                        else:
                            novel.append(hyp)
                    except Exception as e:
                        logger.warning("Novelty check failed, keeping hypothesis: %s", e)
                        novel.append(hyp)  # Fail open
                # Novelty FILTERS, but must not STARVE the run. An accumulated DB
                # of prior-run hypotheses makes a fresh batch look non-novel
                # (observed 5 -> 1-2 against the 29-row D4 history), which is how
                # a run that generated five ended up reporting one. So if filtering
                # left fewer than requested, backfill with the HIGHEST-novelty
                # filtered ones -- novelty stays a preference (the most-novel
                # survive and rank first), not a run-killer.
                if len(novel) < num_hypotheses and filtered:
                    filtered.sort(key=lambda h: getattr(h, "novelty_score", 0.0), reverse=True)
                    backfill = filtered[: num_hypotheses - len(novel)]
                    if backfill:
                        logger.info(
                            "Backfilling %d hypothesis(es) by novelty to reach the "
                            "requested %d (DB-novelty starvation).",
                            len(backfill), num_hypotheses,
                        )
                    novel = novel + backfill
                validated_hypotheses = novel
                logger.info(f"After novelty scoring: {len(validated_hypotheses)} hypotheses")
            except ImportError:
                logger.warning("NoveltyChecker unavailable, skipping novelty scoring")

        # FINAL trim: now that novelty has ranked/filtered (and backfilled), keep
        # exactly the requested count, most-novel first.
        validated_hypotheses.sort(key=lambda h: getattr(h, "novelty_score", 0.0), reverse=True)
        validated_hypotheses = validated_hypotheses[:num_hypotheses]

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
        context_papers: List[PaperMetadata],
        data_context: Optional[str] = None,
        data_driven: bool = False,
        exclude_statements: Optional[List[str]] = None,
    ) -> List[Hypothesis]:
        """
        Generate hypotheses using Claude with structured output.

        Args:
            research_question: Research question
            domain: Scientific domain
            num_hypotheses: Number of hypotheses to generate
            context_papers: Literature context
            data_context: Optional dataset schema/summary to ground hypotheses in.

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

        # Ground the hypotheses in the actual dataset when one is available --
        # but in ONE of two modes, because the two situations want opposite
        # things:
        #
        #   * data_driven (no question was asked, only data given): the run's
        #     whole purpose IS to explore the dataset, so ask the model to
        #     hypothesise about relationships AMONG the variables.
        #   * question-driven (a real research question was given): the columns
        #     are there to make the hypotheses TESTABLE, not to become the
        #     subject. Without this split, a run asked "which proteins causally
        #     affect fibrosis" produced one on-question hypothesis and four about
        #     incidental column relationships (|BETA| vs allele frequency, sQTL
        #     slope vs eQTL slope), because the data block said "base your
        #     hypotheses on these variables" and drowned out the question.
        #
        # The anti-fabrication guard (never substitute a column for an absent
        # variable; name the gap) is kept in BOTH modes.
        if data_context:
            if data_driven:
                framing = (
                    "DATASET UNDER STUDY — base your hypotheses on THESE actual "
                    "variables, and make each one testable with this dataset:\n"
                    f"{data_context}\n\n"
                    "If the variables above do not include what the question asks "
                    "about, or a block above reports that a file's header is not "
                    "on its first line, say so plainly in the rationale of every "
                    "hypothesis and frame the hypotheses over what IS measured. "
                    "Never treat a column as a stand-in for a variable that is "
                    "absent.\n\n"
                )
            else:
                framing = (
                    "DATASET AVAILABLE FOR TESTING — use these columns to make "
                    "each hypothesis concretely testable, but the hypotheses must "
                    "ANSWER THE RESEARCH QUESTION below; they are NOT an "
                    "invitation to explore the dataset:\n"
                    f"{data_context}\n\n"
                    "EVERY hypothesis must directly address the research question. "
                    "Do NOT propose hypotheses about incidental relationships "
                    "among columns (for example one variable's magnitude versus "
                    "another's frequency, or overlaps between identifier lists) "
                    "unless they bear on the question. Ground each hypothesis in "
                    "the columns above so it is testable; if a variable the "
                    "question needs is absent, say so plainly in the rationale "
                    "and frame the hypothesis over what IS measured — never "
                    "substitute a column for an absent variable.\n\n"
                    # Why: a discovery question asks WHICH items matter, so the
                    # finding is the existence of an effect, not its sign. A
                    # hypothesis that commits the claim to a direction is marked
                    # rejected the moment the real effect points the other way --
                    # turning a genuine discovery into a refutation on identical
                    # numbers. Stated generally; applies to any outcome/predictor.
                    "DIRECTION BELONGS IN THE RATIONALE, NOT THE CLAIM (for "
                    "discovery/existence questions — 'which X affect Y'). State "
                    "each such hypothesis as a NON-ZERO effect in EITHER "
                    "direction (e.g. 'X has a non-zero effect on Y at FDR<0.05'), "
                    "and put the expected or observed sign in the rationale. A "
                    "true effect in the unexpected direction is still a positive "
                    "finding here. Commit the claim to a specific sign or "
                    "threshold ONLY when that direction or threshold is itself "
                    "what the question asks.\n\n"
                )
            prompt = framing + prompt

        # One claim per hypothesis, stated where the model will act on it
        # rather than only in the schema's field description.
        #
        # A compound statement gets ONE verdict, and that verdict hides the
        # finding. A real run produced "across the joined variant set the
        # effects are concordant ... AND the per-protein MR estimate is
        # non-zero for a majority of proteins", which came back `rejected` on
        # the genome-wide half while the per-protein half contained the run's
        # top hit -- the result the run existed to find, filed under a
        # hypothesis marked rejected.
        prompt += (
            "\n\nONE CLAIM PER HYPOTHESIS. A hypothesis whose statement joins "
            "two testable claims with 'and' cannot be adjudicated: the report "
            "records a single status, so a genome-wide claim that fails and a "
            "per-item claim that succeeds are recorded together as a failure. "
            "If you find yourself writing 'X is true AND Y is true', emit two "
            "hypotheses. Prefer several sharp hypotheses to one broad one; "
            "`n` of them are requested precisely so they can differ.\n"
        )

        # Define expected JSON schema
        schema = {
            "hypotheses": [
                {
                    "statement": (
                        "string, UNDER 500 CHARACTERS: ONE claim, in one sentence. "
                        "If your statement joins two testable claims with 'and', "
                        "split it into two hypotheses instead -- a single verdict "
                        "cannot say which half was supported. Put the test, the "
                        "threshold and the correction method in `rationale`, not here"
                    ),
                    "rationale": (
                        "string (scientific justification, plus how the claim "
                        "should be tested: statistic, threshold, correction)"
                    ),
                    "confidence_score": "float 0.0-1.0",
                    "testability_score": "float 0.0-1.0 (preliminary estimate)",
                    "suggested_experiment_types": ["computational | data_analysis | literature_synthesis"]
                }
            ]
        }

        # On a top-up attempt, name what has already been proposed so the model
        # produces DISTINCT additions instead of re-returning the same few ideas.
        # Under a large, complex prompt deepseek tends to settle on a small set
        # of hypotheses and repeat it, so dedupe alone stalls below the target;
        # telling it what to avoid is what actually unlocks the remaining slots.
        if exclude_statements:
            _already = "\n".join(f"- {s}" for s in exclude_statements if s)
            prompt += (
                "\n\nALREADY PROPOSED — do NOT repeat or lightly reword any of "
                "these; propose genuinely DISTINCT, non-overlapping hypotheses "
                "that address different proteins, mechanisms, or tests:\n"
                f"{_already}\n"
            )

        try:
            # Call Claude with structured output
            response = self.llm_client.generate_structured(
                prompt=prompt,
                schema=schema,
                max_tokens=HYPOTHESIS_MAX_TOKENS,
                temperature=0.7  # Slightly higher for creativity
            )

            # Parse response into Hypothesis objects
            hypotheses = []
            for i, hyp_data in enumerate(response.get("hypotheses", [])):
                try:
                    # Map experiment types
                    exp_types = []
                    for exp_type_str in hyp_data.get("suggested_experiment_types", []):
                        try:
                            exp_types.append(ExperimentType(exp_type_str))
                        except ValueError:
                            logger.warning(f"Unknown experiment type: {exp_type_str}")

                    statement, rationale = _fit_statement(
                        hyp_data["statement"], hyp_data.get("rationale", "")
                    )
                    hypothesis = Hypothesis(
                        id=str(uuid.uuid4()),
                        research_question=research_question,
                        statement=statement,
                        rationale=rationale,
                        domain=domain,
                        status=HypothesisStatus.GENERATED,
                        testability_score=hyp_data.get("testability_score"),
                        confidence_score=hyp_data.get("confidence_score"),
                        suggested_experiment_types=exp_types,
                        related_papers=[
                            p.arxiv_id or p.doi or p.title
                            for p in context_papers
                            if p is not None and (p.arxiv_id or p.doi or p.title)
                        ],
                        generated_by=self.agent_id
                    )
                    hypotheses.append(hypothesis)

                except Exception as e:
                    logger.error(f"Error parsing hypothesis {i}: {e}")
                    continue

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
