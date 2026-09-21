"""Turning a research question into datasets to download.

Retrieval is the model's decision (it reads the question and names datasets);
downloading is mechanical and belongs to the fetcher.
"""

from .adjudicate import Disagreement, parse_answer, resolve
from .data_report import build_data_report, render_data_report, write_data_report
from .gold import GoldChoice, choose_primary_gold
from .planner import (
    Candidate,
    Intent,
    propose_intents,
    rank_candidates,
    score_by_keywords,
    usable_queries,
)
from .preflight import choose_downloads, plan_prompt
from .search import Proposal, propose_datasets, propose_evidence

__all__ = [
    "Candidate",
    "Intent",
    "propose_intents",
    "rank_candidates",
    "score_by_keywords",
    "usable_queries",
    "Proposal",
    "propose_datasets",
    "propose_evidence",
    "choose_downloads",
    "plan_prompt",
    "GoldChoice",
    "choose_primary_gold",
    "Disagreement",
    "parse_answer",
    "resolve",
    "build_data_report",
    "render_data_report",
    "write_data_report",
]
