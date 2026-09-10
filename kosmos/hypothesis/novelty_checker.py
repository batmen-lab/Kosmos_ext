"""
Novelty Checker for hypotheses.

Checks if generated hypotheses are novel by:
1. Searching existing literature for similar claims
2. Comparing semantic similarity with known hypotheses
3. Detecting prior art
4. Generating novelty scores and reports
"""

import logging
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
import numpy as np

from kosmos.models.hypothesis import Hypothesis, NoveltyReport
from kosmos.literature.unified_search import UnifiedLiteratureSearch
from kosmos.literature.base_client import PaperMetadata
from kosmos.knowledge.embeddings import get_embedder
from kosmos.knowledge.vector_db import get_vector_db
from kosmos.db.models import Hypothesis as DBHypothesis
from kosmos.db.models import HypothesisStatus as DBHypothesisStatus
from kosmos.db import get_session

logger = logging.getLogger(__name__)


# A hypothesis counts as prior art only once it has actually been tested.
# GENERATED and UNDER_REVIEW are proposals; TESTING is in flight and has no
# answer yet.
_TESTED_STATUSES = (
    DBHypothesisStatus.SUPPORTED,
    DBHypothesisStatus.REJECTED,
    DBHypothesisStatus.INCONCLUSIVE,
)


class NoveltyChecker:
    """
    Check novelty of hypotheses against existing work.

    Uses semantic similarity, literature search, and prior art detection
    to assess how novel a hypothesis is.

    Example:
        ```python
        checker = NoveltyChecker(similarity_threshold=0.75)

        report = checker.check_novelty(hypothesis)

        if report.is_novel:
            print(f"Novel hypothesis! Score: {report.novelty_score}")
        else:
            print(f"Similar to: {report.similar_papers[0]['title']}")
        ```
    """

    def __init__(
        self,
        similarity_threshold: float = 0.75,
        max_similar_papers: int = 10,
        use_vector_db: bool = True
    ):
        """
        Initialize novelty checker.

        Args:
            similarity_threshold: Similarity above this is considered "similar" (0.0-1.0)
            max_similar_papers: Maximum similar papers to return in report
            use_vector_db: Whether to use vector DB for similarity search
        """
        self.similarity_threshold = similarity_threshold
        self.max_similar_papers = max_similar_papers
        self.use_vector_db = use_vector_db

        # Components
        self.literature_search = UnifiedLiteratureSearch()
        self.embedder = get_embedder() if use_vector_db else None
        self.vector_db = get_vector_db() if use_vector_db else None

        logger.info(f"Initialized NoveltyChecker with threshold={similarity_threshold}")

    def check_novelty(self, hypothesis: Hypothesis) -> NoveltyReport:
        """
        Check novelty of a hypothesis.

        Args:
            hypothesis: Hypothesis to check

        Returns:
            NoveltyReport: Detailed novelty analysis

        Example:
            ```python
            report = checker.check_novelty(hypothesis)
            print(f"Novelty: {report.novelty_score:.2f}")
            print(f"Prior art: {report.prior_art_detected}")
            ```
        """
        logger.info(f"Checking novelty for hypothesis: {hypothesis.statement[:50]}...")

        # Step 1: Search literature for similar work
        similar_papers = self._search_similar_literature(hypothesis)

        # Step 2: Check against existing hypotheses in database
        similar_hypotheses = self._check_existing_hypotheses(hypothesis)

        # Step 3: Compute semantic similarity scores
        max_paper_similarity = 0.0
        if similar_papers:
            max_paper_similarity = max(
                self._compute_similarity(hypothesis, paper)
                for paper in similar_papers
            )

        max_hypothesis_similarity = 0.0
        if similar_hypotheses:
            max_hypothesis_similarity = max(
                self._compute_hypothesis_similarity(hypothesis, existing)
                for existing in similar_hypotheses
            )

        max_similarity = max(max_paper_similarity, max_hypothesis_similarity)

        # Step 4: Detect prior art (near-duplicates)
        prior_art_detected = max_similarity >= self.similarity_threshold

        # Step 5: Calculate novelty score
        # Score decreases as similarity increases
        # 1.0 = completely novel, 0.0 = exact duplicate
        if max_similarity >= 0.95:
            novelty_score = 0.0  # Essentially a duplicate
        elif max_similarity >= self.similarity_threshold:
            # Linear decay from threshold to 0.95
            novelty_score = 1.0 - ((max_similarity - self.similarity_threshold) / (0.95 - self.similarity_threshold))
            novelty_score = max(0.0, novelty_score * 0.5)  # Cap at 0.5 for similar work
        else:
            # Below threshold: high novelty score
            novelty_score = 1.0 - (max_similarity * 0.5)  # Scale so 0.0 similarity = 1.0 novelty

        novelty_score = max(0.0, min(1.0, novelty_score))

        # Step 6: Generate human-readable summary
        summary = self._generate_summary(
            novelty_score=novelty_score,
            prior_art_detected=prior_art_detected,
            max_similarity=max_similarity,
            similar_papers=similar_papers,
            similar_hypotheses=similar_hypotheses
        )

        # Step 7: Prepare similar work details (filter out None papers)
        similar_papers_info = [
            {
                "title": paper.title or "Untitled",
                "authors": (paper.authors or [])[:3],  # First 3 authors
                "year": paper.year,
                "source": paper.source,
                "similarity": self._compute_similarity(hypothesis, paper),
                "doi": paper.doi,
                "arxiv_id": paper.arxiv_id
            }
            for paper in similar_papers[:self.max_similar_papers]
            if paper is not None and paper.title
        ]

        similar_hypotheses_info = [
            {
                "statement": hyp.statement,
                "domain": hyp.domain,
                "created_at": hyp.created_at.isoformat() if hyp.created_at else None,
                "similarity": self._compute_hypothesis_similarity(hypothesis, hyp),
                "id": hyp.id
            }
            for hyp in similar_hypotheses[:5]  # Limit to 5
        ]

        # Update hypothesis with novelty score
        hypothesis.novelty_score = novelty_score

        return NoveltyReport(
            hypothesis_id=hypothesis.id or "unknown",
            novelty_score=novelty_score,
            similar_hypotheses=similar_hypotheses_info,
            similar_papers=similar_papers_info,
            max_similarity=max_similarity,
            prior_art_detected=prior_art_detected,
            is_novel=novelty_score >= (1.0 - self.similarity_threshold),  # Inverse of threshold
            novelty_threshold_used=self.similarity_threshold,
            summary=summary
        )

    def _search_similar_literature(self, hypothesis: Hypothesis) -> List[PaperMetadata]:
        """
        Search literature for papers related to the hypothesis.

        Args:
            hypothesis: Hypothesis to search for

        Returns:
            List[PaperMetadata]: Similar papers
        """
        try:
            # Use vector DB for semantic search if available
            if self.use_vector_db and self.vector_db and self.embedder:
                return self._vector_search_papers(hypothesis)

            # Fallback: keyword search
            query = f"{hypothesis.statement} {hypothesis.rationale[:100]}"
            papers = self.literature_search.search(
                query=query,
                max_results=20
            )

            logger.info(f"Found {len(papers)} similar papers via keyword search")

            # Index retrieved papers into vector DB for future semantic searches
            if papers and self.vector_db:
                try:
                    self.vector_db.add_papers(papers)
                    logger.info(f"Indexed {len(papers)} papers into vector DB")
                except Exception as e:
                    logger.warning(f"Failed to index papers into vector DB: {e}")

            return papers

        except Exception as e:
            logger.error(f"Error searching literature: {e}", exc_info=True)
            return []

    def _vector_search_papers(self, hypothesis: Hypothesis) -> List[PaperMetadata]:
        """
        Use vector database to find semantically similar papers.

        Args:
            hypothesis: Hypothesis to search for

        Returns:
            List[PaperMetadata]: Similar papers
        """
        try:
            # Create search query from hypothesis
            query = f"{hypothesis.statement}. {hypothesis.rationale}"

            # Search vector DB
            results = self.vector_db.search(query, top_k=20)

            # Convert results to PaperMetadata
            papers = []
            for result in results:
                # Reconstruct paper from metadata
                metadata = result.get("metadata", {})
                paper = PaperMetadata(
                    title=metadata.get("title", ""),
                    authors=metadata.get("authors", []),
                    abstract=metadata.get("abstract", ""),
                    year=metadata.get("year"),
                    doi=metadata.get("doi"),
                    arxiv_id=metadata.get("arxiv_id"),
                    source=metadata.get("source", "unknown")
                )
                papers.append(paper)

            logger.info(f"Found {len(papers)} similar papers via vector search")
            return papers

        except Exception as e:
            logger.error(f"Error in vector search: {e}", exc_info=True)
            return []

    def _check_existing_hypotheses(self, hypothesis: Hypothesis) -> List[Hypothesis]:
        """
        Check against existing hypotheses in database.

        Args:
            hypothesis: Hypothesis to check

        Returns:
            List[Hypothesis]: Similar existing hypotheses
        """
        try:
            with get_session() as session:
                # TESTED hypotheses only. An untested one is not prior art.
                #
                # This queried every hypothesis ever stored in the domain, and
                # the domain is "biology" for all of this work -- so each new
                # hypothesis was scored against every sibling from every past
                # run, tested or not. The cost is not abstract: a two-sample MR
                # hypothesis, the paper's own method and the only one of
                # nineteen that could reproduce the result, scored novelty 0.00
                # because it read like the untested correlation hypotheses
                # beside it. Priority is 30% novelty, so it was never executed.
                #
                # Novelty asks "has this question been answered". A hypothesis
                # nobody has tested answers nothing, so it cannot be the reason
                # to skip a better formulation of the same question. Literature
                # prior art is unaffected -- published work HAS been done, and
                # is still compared in full.
                db_hypotheses = session.query(DBHypothesis).filter(
                    DBHypothesis.domain == hypothesis.domain,
                    DBHypothesis.status.in_(_TESTED_STATUSES),
                ).all()

                # Convert to Pydantic models
                existing_hypotheses = []
                for db_hyp in db_hypotheses:
                    # Skip self if it's already in DB
                    if hypothesis.id and db_hyp.id == hypothesis.id:
                        continue

                    hyp = Hypothesis(
                        id=db_hyp.id,
                        research_question=db_hyp.research_question,
                        statement=db_hyp.statement,
                        rationale=db_hyp.rationale,
                        domain=db_hyp.domain,
                        created_at=db_hyp.created_at,
                        updated_at=db_hyp.updated_at
                    )
                    existing_hypotheses.append(hyp)

                # Filter by similarity
                similar = []
                for existing in existing_hypotheses:
                    similarity = self._compute_hypothesis_similarity(hypothesis, existing)
                    if similarity >= 0.5:  # Lower threshold for preliminary filtering
                        similar.append(existing)

                # Sort by similarity (highest first)
                similar.sort(
                    key=lambda h: self._compute_hypothesis_similarity(hypothesis, h),
                    reverse=True
                )

                logger.info(f"Found {len(similar)} similar existing hypotheses")
                return similar

        except Exception as e:
            logger.error(f"Error checking existing hypotheses: {e}", exc_info=True)
            return []

    def _compute_similarity(
        self,
        hypothesis: Hypothesis,
        paper: PaperMetadata
    ) -> float:
        """
        Compute semantic similarity between hypothesis and paper.

        Args:
            hypothesis: Hypothesis
            paper: Paper to compare

        Returns:
            float: Similarity score (0.0-1.0)
        """
        try:
            # Guard against None paper or missing title
            if paper is None or not paper.title:
                return 0.0

            paper_title = paper.title or ""
            paper_abstract = paper.abstract or ""

            if not self.embedder:
                # Fallback: simple keyword overlap
                return self._keyword_similarity(hypothesis.statement, paper_title + " " + paper_abstract)

            # Use embeddings for semantic similarity
            hyp_text = f"{hypothesis.statement}. {hypothesis.rationale}"
            paper_text = f"{paper_title}. {paper_abstract}"

            hyp_embedding = self.embedder.embed_query(hyp_text)
            paper_embedding = self.embedder.embed_query(paper_text)

            # Cosine similarity
            similarity = np.dot(hyp_embedding, paper_embedding) / (
                np.linalg.norm(hyp_embedding) * np.linalg.norm(paper_embedding)
            )

            return float(max(0.0, min(1.0, similarity)))

        except Exception as e:
            logger.error(f"Error computing similarity: {e}")
            return 0.0

    def _compute_hypothesis_similarity(
        self,
        hyp1: Hypothesis,
        hyp2: Hypothesis
    ) -> float:
        """
        Compute similarity between two hypotheses.

        Args:
            hyp1: First hypothesis
            hyp2: Second hypothesis

        Returns:
            float: Similarity score (0.0-1.0)
        """
        try:
            if not self.embedder:
                # Fallback: simple keyword similarity
                return self._keyword_similarity(hyp1.statement, hyp2.statement)

            # Use embeddings
            text1 = f"{hyp1.statement}. {hyp1.rationale}"
            text2 = f"{hyp2.statement}. {hyp2.rationale}"

            emb1 = self.embedder.embed_query(text1)
            emb2 = self.embedder.embed_query(text2)

            similarity = np.dot(emb1, emb2) / (
                np.linalg.norm(emb1) * np.linalg.norm(emb2)
            )

            return float(max(0.0, min(1.0, similarity)))

        except Exception as e:
            logger.error(f"Error computing hypothesis similarity: {e}")
            return 0.0

    def _keyword_similarity(self, text1: str, text2: str) -> float:
        """
        Simple keyword-based similarity (fallback).

        Args:
            text1: First text
            text2: Second text

        Returns:
            float: Similarity score (0.0-1.0)
        """
        # Convert to lowercase and split into words
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())

        # Remove common words
        stopwords = {"the", "a", "an", "is", "are", "was", "were", "in", "on", "at", "to", "for", "of", "by", "with"}
        words1 = words1 - stopwords
        words2 = words2 - stopwords

        if not words1 or not words2:
            return 0.0

        # Jaccard similarity
        intersection = len(words1 & words2)
        union = len(words1 | words2)

        return intersection / union if union > 0 else 0.0

    def _generate_summary(
        self,
        novelty_score: float,
        prior_art_detected: bool,
        max_similarity: float,
        similar_papers: List[PaperMetadata],
        similar_hypotheses: List[Hypothesis]
    ) -> str:
        """
        Generate human-readable novelty summary.

        Args:
            novelty_score: Calculated novelty score
            prior_art_detected: Whether prior art was detected
            max_similarity: Maximum similarity found
            similar_papers: Similar papers
            similar_hypotheses: Similar existing hypotheses

        Returns:
            str: Summary text
        """
        if prior_art_detected:
            if similar_hypotheses:
                return (
                    f"LOW NOVELTY (score: {novelty_score:.2f}). "
                    f"Very similar to existing hypothesis: '{similar_hypotheses[0].statement[:80]}...'. "
                    f"Maximum similarity: {max_similarity:.2f}. "
                    f"Consider revising or expanding this hypothesis."
                )
            elif similar_papers and similar_papers[0] is not None:
                # The list can contain None entries (a reconstruction that failed),
                # so guard the element itself, not just the list length.
                top_paper = similar_papers[0]
                paper_title = top_paper.title if top_paper.title else "Untitled"
                paper_year = top_paper.year if top_paper.year else "unknown"
                return (
                    f"LOW NOVELTY (score: {novelty_score:.2f}). "
                    f"Very similar to existing work: '{paper_title}' ({paper_year}). "
                    f"Maximum similarity: {max_similarity:.2f}. "
                    f"This hypothesis may already be addressed in the literature."
                )
            else:
                return (
                    f"LOW NOVELTY (score: {novelty_score:.2f}). "
                    f"High similarity detected (max: {max_similarity:.2f}) but source unclear."
                )

        elif novelty_score >= 0.8:
            return (
                f"HIGH NOVELTY (score: {novelty_score:.2f}). "
                f"This hypothesis appears highly novel with low similarity to existing work (max: {max_similarity:.2f}). "
                f"No significant prior art detected."
            )

        elif novelty_score >= 0.6:
            summary = (
                f"MODERATE NOVELTY (score: {novelty_score:.2f}). "
                f"Some similar work exists (max similarity: {max_similarity:.2f}), "
                f"but this hypothesis offers a distinct perspective. "
            )

            if similar_papers:
                summary += f"Related papers found: {len(similar_papers)}. "

            if similar_hypotheses:
                summary += f"Similar existing hypotheses: {len(similar_hypotheses)}. "

            return summary + "Consider emphasizing the novel aspects."

        else:
            return (
                f"MODERATE-LOW NOVELTY (score: {novelty_score:.2f}). "
                f"Considerable similarity to existing work (max: {max_similarity:.2f}). "
                f"Found {len(similar_papers)} related papers and {len(similar_hypotheses)} similar hypotheses. "
                f"Consider how this hypothesis extends or differs from prior work."
            )


def check_hypothesis_novelty(
    hypothesis: Hypothesis,
    similarity_threshold: float = 0.75
) -> NoveltyReport:
    """
    Convenience function to check hypothesis novelty.

    Args:
        hypothesis: Hypothesis to check
        similarity_threshold: Similarity threshold (default: 0.75)

    Returns:
        NoveltyReport: Novelty analysis report

    Example:
        ```python
        report = check_hypothesis_novelty(my_hypothesis)
        if report.is_novel:
            print("Novel hypothesis!")
        ```
    """
    checker = NoveltyChecker(similarity_threshold=similarity_threshold)
    return checker.check_novelty(hypothesis)
