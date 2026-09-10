"""
OpenAlex literature client.

Free, key-less scholarly search over ~250M works (https://openalex.org) — the open
successor to Microsoft Academic Graph and a drop-in replacement for Semantic Scholar.
Uses the "polite pool" (mailto=) for higher rate limits; no API key or credits required.

Emits the same PaperMetadata objects as the other Kosmos literature clients, so it
plugs straight into UnifiedLiteratureSearch.
"""

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

import requests

from kosmos.literature.base_client import (
    Author,
    BaseLiteratureClient,
    PaperMetadata,
    PaperSource,
)

logger = logging.getLogger(__name__)

OPENALEX_WORKS_URL = "https://api.openalex.org/works"
_SELECT = (
    "id,doi,ids,title,abstract_inverted_index,authorships,publication_year,"
    "publication_date,primary_location,cited_by_count,referenced_works_count,"
    "referenced_works,open_access,concepts"
)


class OpenAlexClient(BaseLiteratureClient):
    """Search OpenAlex — free, no API key. Replacement for Semantic Scholar."""

    def __init__(self, api_key: Optional[str] = None, cache_enabled: bool = True,
                 email: Optional[str] = None, timeout: int = 30):
        super().__init__(api_key=api_key, cache_enabled=cache_enabled)
        # Polite-pool contact + result cap from config (fall back to sane defaults).
        try:
            from kosmos.config import get_config
            cfg = get_config()
            self.email = email or getattr(cfg.literature, "pubmed_email", None) or "kosmos@example.com"
            self.max_results = getattr(cfg.literature, "max_results_per_query", 20)
        except Exception:
            self.email = email or "kosmos@example.com"
            self.max_results = 20
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": f"Kosmos/1.0 (mailto:{self.email})"})
        logger.info("Initialized OpenAlex client (email=%s)", self.email)

    # ------------------------------------------------------------------ search
    def search(self, query: str, max_results: int = 10,
               fields: Optional[List[str]] = None,
               year_from: Optional[int] = None,
               year_to: Optional[int] = None, **kwargs) -> List[PaperMetadata]:
        n = min(max(int(max_results or 10), 1), int(getattr(self, "max_results", 20) or 20), 200)
        params: Dict[str, Any] = {
            "search": query,
            "per-page": n,
            "mailto": self.email,
            "select": _SELECT,
        }
        filters = []
        if year_from and year_to:
            filters.append(f"publication_year:{int(year_from)}-{int(year_to)}")
        elif year_from:
            filters.append(f"from_publication_date:{int(year_from)}-01-01")
        elif year_to:
            filters.append(f"to_publication_date:{int(year_to)}-12-31")
        if filters:
            params["filter"] = ",".join(filters)

        try:
            resp = self.session.get(OPENALEX_WORKS_URL, params=params, timeout=self.timeout)
            resp.raise_for_status()
            results = resp.json().get("results", []) or []
        except Exception as e:
            logger.error("OpenAlex search failed for query=%r: %s", query, e)
            return []

        papers: List[PaperMetadata] = []
        for work in results:
            try:
                papers.append(self._to_metadata(work))
            except Exception as e:  # never let one malformed record kill the batch
                logger.debug("OpenAlex: skipped a record (%s)", e)
        logger.info("Retrieved %d papers from OpenAlex", len(papers))
        return papers

    # ------------------------------------------------------------- single work
    def get_paper_by_id(self, paper_id: str) -> Optional[PaperMetadata]:
        """Fetch one work. Accepts an OpenAlex ID/URL, a DOI, 'doi:...', or 'pmid:...'."""
        pid = paper_id.strip()
        if pid.startswith("http") and "openalex.org" in pid:
            path = pid
        elif pid.upper().startswith("W") and pid[1:].isdigit():
            path = f"{OPENALEX_WORKS_URL}/{pid}"
        elif pid.lower().startswith("doi:"):
            path = f"{OPENALEX_WORKS_URL}/https://doi.org/{pid[4:]}"
        elif pid.lower().startswith("pmid:"):
            path = f"{OPENALEX_WORKS_URL}/pmid:{pid[5:]}"
        elif "/" in pid and "." in pid:  # looks like a bare DOI
            path = f"{OPENALEX_WORKS_URL}/https://doi.org/{pid}"
        else:
            path = f"{OPENALEX_WORKS_URL}/{pid}"
        try:
            resp = self.session.get(path, params={"mailto": self.email, "select": _SELECT}, timeout=self.timeout)
            resp.raise_for_status()
            return self._to_metadata(resp.json())
        except Exception as e:
            logger.warning("OpenAlex get_paper_by_id failed for %r: %s", paper_id, e)
            return None

    def get_paper_references(self, paper_id: str, max_refs: int = 50) -> List[PaperMetadata]:
        """Works this paper cites (from its referenced_works list)."""
        try:
            work = self.get_paper_by_id(paper_id)
            refs = ((work.raw_data or {}).get("referenced_works") if work else None) or []
            if not refs:
                return []
            ids = "|".join(r.split("/")[-1] for r in refs[:max_refs])
            resp = self.session.get(
                OPENALEX_WORKS_URL,
                params={"filter": f"openalex_id:{ids}", "per-page": min(max_refs, 200),
                        "mailto": self.email, "select": _SELECT},
                timeout=self.timeout,
            )
            resp.raise_for_status()
            return [self._to_metadata(w) for w in resp.json().get("results", []) or []]
        except Exception as e:
            logger.debug("OpenAlex get_paper_references failed for %r: %s", paper_id, e)
            return []

    def get_paper_citations(self, paper_id: str, max_cites: int = 50) -> List[PaperMetadata]:
        """Works that cite this paper (filter=cites:<id>)."""
        try:
            work = self.get_paper_by_id(paper_id)
            oid = (work.id if work else "").split("/")[-1]
            if not oid:
                return []
            resp = self.session.get(
                OPENALEX_WORKS_URL,
                params={"filter": f"cites:{oid}", "per-page": min(max_cites, 200),
                        "mailto": self.email, "select": _SELECT},
                timeout=self.timeout,
            )
            resp.raise_for_status()
            return [self._to_metadata(w) for w in resp.json().get("results", []) or []]
        except Exception as e:
            logger.debug("OpenAlex get_paper_citations failed for %r: %s", paper_id, e)
            return []

    # ------------------------------------------------------------------ helpers
    @staticmethod
    def _reconstruct_abstract(inverted: Optional[Dict[str, List[int]]]) -> str:
        """OpenAlex stores abstracts as an inverted index {word: [positions]}."""
        if not inverted:
            return ""
        try:
            positions: Dict[int, str] = {}
            for word, idxs in inverted.items():
                for i in idxs:
                    positions[i] = word
            return " ".join(positions[i] for i in sorted(positions))
        except Exception:
            return ""

    def _to_metadata(self, work: Dict[str, Any]) -> PaperMetadata:
        ids = work.get("ids", {}) or {}
        doi = work.get("doi")
        if doi and doi.startswith("https://doi.org/"):
            doi = doi[len("https://doi.org/"):]

        pmid = ids.get("pmid")
        if pmid:
            pmid = pmid.rstrip("/").split("/")[-1]

        authors = []
        for a in (work.get("authorships") or []):
            au = (a.get("author") or {})
            if au.get("display_name"):
                authors.append(Author(name=au["display_name"], author_id=au.get("id")))

        pub_date = None
        if work.get("publication_date"):
            try:
                pub_date = datetime.strptime(work["publication_date"], "%Y-%m-%d")
            except (ValueError, TypeError):
                pass

        primary = work.get("primary_location") or {}
        venue = (primary.get("source") or {}).get("display_name")
        pdf_url = primary.get("pdf_url") or (work.get("open_access") or {}).get("oa_url")
        concepts = [c.get("display_name") for c in (work.get("concepts") or []) if c.get("display_name")]

        return PaperMetadata(
            id=work.get("id") or doi or "",
            source=PaperSource.OPENALEX,
            doi=doi,
            pubmed_id=pmid,
            title=work.get("title") or "",
            abstract=self._reconstruct_abstract(work.get("abstract_inverted_index")),
            authors=authors,
            publication_date=pub_date,
            journal=venue,
            venue=venue,
            year=work.get("publication_year"),
            url=work.get("id"),
            pdf_url=pdf_url,
            citation_count=work.get("cited_by_count") or 0,
            reference_count=work.get("referenced_works_count") or 0,
            fields=concepts[:8],
            raw_data=work,
        )
