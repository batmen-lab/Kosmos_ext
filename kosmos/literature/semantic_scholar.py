"""
Semantic Scholar API client for searching and retrieving scientific papers.

Uses the official semanticscholar Python package with enhanced caching and
citation support.
"""

from itertools import islice
from semanticscholar import SemanticScholar
from semanticscholar.Paper import Paper as S2Paper
from typing import List, Optional
from datetime import datetime

from kosmos.literature.base_client import (
    BaseLiteratureClient,
    PaperMetadata,
    PaperSource,
    Author
)
from kosmos.literature.cache import get_cache
from kosmos.config import get_config


class SemanticScholarClient(BaseLiteratureClient):
    """
    Client for interacting with the Semantic Scholar API.

    Semantic Scholar provides rich citation data, paper metadata, and
    AI-powered paper recommendations.

    API Docs: https://api.semanticscholar.org/
    """

    #: Once the API says "too many requests" (or refuses the connection), it is
    #: not going to answer a moment later: the library's own retry policy keeps
    #: asking for minutes, on a background thread that the caller's timeout
    #: cannot stop. A rate limit is a fact about the next few minutes, so the
    #: client goes quiet for the rest of the run instead of spending it.
    _UNAVAILABLE_MARKERS = (
        "429",
        "too many requests",
        "connectionrefused",
        "connection refused",
        "max retries",
        "retryerror",
    )

    def __init__(self, api_key: Optional[str] = None, cache_enabled: bool = True):
        """
        Initialize the Semantic Scholar client.

        Args:
            api_key: Optional API key (increases rate limits from 100 to 5000 requests/5min)
            cache_enabled: Whether to enable caching for API responses
        """
        super().__init__(api_key=api_key, cache_enabled=cache_enabled)

        # Get configuration
        config = get_config()
        self.max_results = config.literature.max_results_per_query

        # Initialize API client
        self.api_key = api_key or config.literature.semantic_scholar_api_key
        # `retry=False`: the requests the library would make again are the ones
        # that just came back 429, and the caller is a hypothesis step that is
        # waiting on this thread with a 90-second budget.
        self.client = SemanticScholar(api_key=self.api_key, timeout=30, retry=False)
        #: Set once the API has refused: the rest of the run skips this source.
        self.unavailable_reason: str = ""

        # Initialize cache if enabled
        self.cache = get_cache() if cache_enabled else None

        # Paper fields to request
        self.paper_fields = [
            'paperId', 'externalIds', 'title', 'abstract', 'authors',
            'year', 'publicationDate', 'venue', 'journal', 'url',
            'citationCount', 'referenceCount', 'influentialCitationCount',
            'fieldsOfStudy', 'openAccessPdf'
        ]

        self.logger.info(
            f"Initialized Semantic Scholar client (with{'out' if not self.api_key else ''} API key)"
        )

    def search(
        self,
        query: str,
        max_results: int = 10,
        fields: Optional[List[str]] = None,
        year_from: Optional[int] = None,
        year_to: Optional[int] = None,
        **kwargs
    ) -> List[PaperMetadata]:
        """
        Search for papers on Semantic Scholar.

        Args:
            query: Search query string
            max_results: Maximum number of results to return
            fields: Optional filter by fields of study (e.g., ["Computer Science", "Biology"])
            year_from: Optional start year filter
            year_to: Optional end year filter
            **kwargs: Additional options:
                - open_access_only: Only return papers with open access PDFs
                - min_citation_count: Minimum citation count filter

        Returns:
            List of PaperMetadata objects

        Example:
            ```python
            client = SemanticScholarClient(api_key="your-key")

            # Simple search
            papers = client.search("large language models", max_results=10)

            # Field-specific search with filters
            papers = client.search(
                "quantum computing",
                fields=["Physics"],
                year_from=2020,
                open_access_only=True,
                min_citation_count=10
            )
            ```
        """
        if not self._validate_query(query):
            return []

        if self.unavailable_reason:
            self.logger.debug(
                f"Skipping Semantic Scholar ({self.unavailable_reason})"
            )
            return []

        # Check cache
        cache_params = {
            "query": query,
            "max_results": max_results,
            "fields": fields,
            "year_from": year_from,
            "year_to": year_to,
            **kwargs
        }

        if self.cache:
            cached_result = self.cache.get("semantic_scholar", "search", cache_params)
            if cached_result is not None:
                return cached_result

        try:
            # Perform search
            results = self.client.search_paper(
                query=query,
                limit=min(max_results, self.max_results),
                fields=self.paper_fields,
                year=f"{year_from}-{year_to}" if year_from and year_to else None,
                fields_of_study=fields
            )

            # Convert to PaperMetadata
            # Use islice to limit iteration - PaginatedResults can fetch
            # infinitely, causing hangs if we iterate the full generator
            papers = []
            for result in islice(results, max_results):
                paper = self._s2_to_metadata(result)

                # Apply additional filters
                if kwargs.get("open_access_only") and not paper.pdf_url:
                    continue
                if kwargs.get("min_citation_count", 0) > paper.citation_count:
                    continue

                papers.append(paper)

            # Cache results
            if self.cache:
                self.cache.set("semantic_scholar", "search", cache_params, papers)

            self.logger.info(f"Found {len(papers)} papers on Semantic Scholar for query: {query}")
            return papers

        except Exception as e:
            # A refusal is remembered so the next caller does not pay for it
            # again; anything else is logged as before.
            if self._note_unavailable(e):
                self.logger.warning(f"Semantic Scholar search skipped: {e}")
            else:
                self._log_api_error(e, f"search query='{query}'")
            return []

    def _note_unavailable(self, error: Exception) -> str:
        """Remember a refusal, in one line, and stop asking this run."""
        text = str(error).lower()
        if any(marker in text for marker in self._UNAVAILABLE_MARKERS):
            reason = f"the API refused the request ({type(error).__name__})"
            if not self.unavailable_reason:
                self.unavailable_reason = reason
                self.logger.warning(
                    f"Semantic Scholar is unavailable ({reason}); skipping it for "
                    f"the rest of this run -- arXiv and PubMed still answer"
                )
        return self.unavailable_reason

    def _log_api_error(self, error: Exception, operation: str) -> None:
        """Record a failed call and carry on.

        The base client re-raises, which is right for a call the caller asked
        for by itself. This client is one of three sources behind a unified
        search: the hypothesis step is waiting on it with a 90-second budget, and
        a source that cannot answer must cost that budget, not the run.
        """
        if self._note_unavailable(error):
            self.logger.warning(f"Semantic Scholar {operation} skipped: {error}")
            return
        self.logger.error(
            f"Error in Semantic Scholar API during {operation}: {error}",
            exc_info=True,
        )

    def get_paper_by_id(self, paper_id: str) -> Optional[PaperMetadata]:
        """
        Retrieve a specific paper by Semantic Scholar ID or external ID.

        Args:
            paper_id: Paper ID (supports multiple formats):
                - Semantic Scholar ID: "649def34f8be52c8b66281af98ae884c09aef38b"
                - DOI: "10.1093/mind/lix.236.433"
                - arXiv: "arXiv:2106.15928"
                - PubMed: "PMID:19872477"

        Returns:
            PaperMetadata object or None if not found

        Example:
            ```python
            # By Semantic Scholar ID
            paper = client.get_paper_by_id("649def34f8be52c8b66281af98ae884c09aef38b")

            # By DOI
            paper = client.get_paper_by_id("10.1093/mind/lix.236.433")

            # By arXiv ID
            paper = client.get_paper_by_id("arXiv:2106.15928")
            ```
        """
        # Check cache
        cache_params = {"paper_id": paper_id}

        if self.cache:
            cached_result = self.cache.get("semantic_scholar", "get_paper", cache_params)
            if cached_result is not None:
                return cached_result

        try:
            result = self.client.get_paper(
                paper_id=paper_id,
                fields=self.paper_fields
            )

            if not result:
                self.logger.warning(f"Paper not found: {paper_id}")
                return None

            paper = self._s2_to_metadata(result)

            # Cache result
            if self.cache:
                self.cache.set("semantic_scholar", "get_paper", cache_params, paper)

            return paper

        except Exception as e:
            self._log_api_error(e, f"get_paper_by_id id={paper_id}")
            return None

    def get_paper_references(self, paper_id: str, max_refs: int = 50) -> List[PaperMetadata]:
        """
        Get papers cited by the given paper.

        Args:
            paper_id: Semantic Scholar paper ID or external ID
            max_refs: Maximum number of references to return

        Returns:
            List of PaperMetadata objects for referenced papers

        Example:
            ```python
            references = client.get_paper_references("649def34f8be52c8b66281af98ae884c09aef38b")
            ```
        """
        # Check cache
        cache_params = {"paper_id": paper_id, "max_refs": max_refs}

        if self.cache:
            cached_result = self.cache.get("semantic_scholar", "get_references", cache_params)
            if cached_result is not None:
                return cached_result

        try:
            result = self.client.get_paper_references(
                paper_id=paper_id,
                limit=max_refs,
                fields=self.paper_fields
            )

            # Extract and convert papers
            papers = [self._s2_to_metadata(ref.citedPaper) for ref in result if ref.citedPaper]

            # Cache results
            if self.cache:
                self.cache.set("semantic_scholar", "get_references", cache_params, papers)

            self.logger.info(f"Retrieved {len(papers)} references for paper {paper_id}")
            return papers

        except Exception as e:
            self._log_api_error(e, f"get_paper_references id={paper_id}")
            return []

    def get_paper_citations(self, paper_id: str, max_cites: int = 50) -> List[PaperMetadata]:
        """
        Get papers that cite the given paper.

        Args:
            paper_id: Semantic Scholar paper ID or external ID
            max_cites: Maximum number of citations to return

        Returns:
            List of PaperMetadata objects for citing papers

        Example:
            ```python
            citations = client.get_paper_citations("649def34f8be52c8b66281af98ae884c09aef38b")
            ```
        """
        # Check cache
        cache_params = {"paper_id": paper_id, "max_cites": max_cites}

        if self.cache:
            cached_result = self.cache.get("semantic_scholar", "get_citations", cache_params)
            if cached_result is not None:
                return cached_result

        try:
            result = self.client.get_paper_citations(
                paper_id=paper_id,
                limit=max_cites,
                fields=self.paper_fields
            )

            # Extract and convert papers
            papers = [self._s2_to_metadata(cite.citingPaper) for cite in result if cite.citingPaper]

            # Cache results
            if self.cache:
                self.cache.set("semantic_scholar", "get_citations", cache_params, papers)

            self.logger.info(f"Retrieved {len(papers)} citations for paper {paper_id}")
            return papers

        except Exception as e:
            self._log_api_error(e, f"get_paper_citations id={paper_id}")
            return []

    def _s2_to_metadata(self, result: S2Paper) -> PaperMetadata:
        """
        Convert Semantic Scholar Paper to PaperMetadata.

        Args:
            result: semanticscholar.Paper object

        Returns:
            PaperMetadata object
        """
        # Extract external IDs
        external_ids = result.externalIds or {}
        doi = external_ids.get("DOI")
        arxiv_id = external_ids.get("ArXiv")
        pubmed_id = external_ids.get("PubMed")

        # Convert authors
        authors = []
        if result.authors:
            for author in result.authors:
                authors.append(Author(
                    name=author.name,
                    author_id=author.authorId if hasattr(author, 'authorId') else None
                ))

        # Parse publication date
        pub_date = None
        if result.publicationDate:
            try:
                # S2 library may return datetime or string depending on version
                if isinstance(result.publicationDate, datetime):
                    pub_date = result.publicationDate
                else:
                    pub_date = datetime.strptime(result.publicationDate, "%Y-%m-%d")
            except (ValueError, TypeError):
                pass

        # Get PDF URL from open access
        pdf_url = None
        if result.openAccessPdf and hasattr(result.openAccessPdf, 'url'):
            pdf_url = result.openAccessPdf.url

        # Extract fields of study
        fields = result.fieldsOfStudy or []

        return PaperMetadata(
            id=result.paperId,
            source=PaperSource.SEMANTIC_SCHOLAR,
            doi=doi,
            arxiv_id=arxiv_id,
            pubmed_id=pubmed_id,
            title=result.title or "",
            abstract=result.abstract or "",
            authors=authors,
            publication_date=pub_date,
            journal=(result.journal.get("name") if isinstance(result.journal, dict)
                    else result.journal) if result.journal else None,
            venue=result.venue,
            year=result.year,
            url=result.url,
            pdf_url=pdf_url,
            citation_count=result.citationCount or 0,
            reference_count=result.referenceCount or 0,
            influential_citation_count=result.influentialCitationCount or 0,
            fields=[f.lower() for f in fields],
            raw_data={
                "paperId": result.paperId,
                "externalIds": external_ids
            }
        )
