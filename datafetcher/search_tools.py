"""Turning ToolUniverse search results into download references.

Round one of retrieval runs the repositories' own search tools and then has to
decide, mechanically, what is downloadable. Each repository returns its own
shape and its own idea of what an identifier is, so each gets a small adapter --
the same reason `finder/adapters.py` exists upstream.

An adapter returns `Hit(accession, reference, title, repository, tool)`. A `None`
reference is honest: the repository has the dataset but nothing here can fetch
it, so it stays a lead rather than becoming a broken download.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from .errors import SearchError

_GEO_SERIES = re.compile(r"^GSE\d+$", re.IGNORECASE)
_HF_REPO = re.compile(r"^[\w.-]+/[\w.-]+$")


@dataclass
class Hit:
    accession: str
    reference: str | None
    repository: str
    tool: str
    title: str = ""
    description: str = ""
    raw: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "accession": self.accession,
            "reference": self.reference,
            "repository": self.repository,
            "tool": self.tool,
            "title": self.title[:200],
            "description": self.description[:300],
        }


def _clip(value: Any, limit: int) -> str:
    return " ".join(str(value or "").split())[:limit]


def _records(payload: Any, *path: str) -> list:
    """Walk a tool's envelope down to its list of records, or refuse."""
    node = payload
    for key in path:
        if not isinstance(node, dict) or key not in node:
            raise SearchError(
                f"no {'.'.join(path)} in the result -- upstream may have changed shape"
            )
        node = node[key]
    if isinstance(node, list):
        return node
    raise SearchError(f"{'.'.join(path)} is a {type(node).__name__}, not a list")


# --- adapters ---------------------------------------------------------------


def omicsdi(raw: Any, tool: str) -> list[Hit]:
    """OmicsDI federates GEO, PRIDE, ArrayExpress, MetaboLights and MassIVE."""
    records = raw.get("datasets") if isinstance(raw, dict) else None
    if records is None:
        records = _records(raw, "data", "datasets")
    hits = []
    for entry in records:
        if not isinstance(entry, dict):
            continue
        accession = str(entry.get("id") or "").strip()
        if not accession:
            continue
        source = str(entry.get("source") or "")
        # `E-GEOD-21321` is ArrayExpress's import of GEO series 21321: the numeric
        # part is the GSE number, so the GEO connector can route it. The fetch
        # verifies the mapping -- a wrong number fails loudly rather than
        # downloading something else.
        equivalent = accession
        e_geod = re.match(r"^E-GEOD-(\d+)$", accession, re.IGNORECASE)
        if e_geod:
            equivalent = f"GSE{e_geod.group(1)}"
        hits.append(
            Hit(
                accession=accession,
                reference=(
                    f"geo://{equivalent}"
                    if _GEO_SERIES.match(equivalent)
                    else None
                ),
                repository=source or "omicsdi",
                tool=tool,
                title=_clip(entry.get("title"), 200),
                description=_clip(entry.get("description"), 300),
                raw=entry,
            )
        )
    return hits


def huggingface(raw: Any, tool: str) -> list[Hit]:
    # Probed shape through ToolUniverse: {"status": ..., "data": [ ... ]}.
    if isinstance(raw, list):
        records = raw
    elif isinstance(raw, dict):
        records = raw.get("data") or raw.get("datasets") or []
    else:
        records = []
    hits = []
    for entry in records:
        if isinstance(entry, str):
            entry = {"id": entry}
        if not isinstance(entry, dict):
            continue
        repo_id = str(entry.get("id") or "").strip()
        if not repo_id or not _HF_REPO.match(repo_id) or ".." in repo_id:
            continue
        hits.append(
            Hit(
                accession=repo_id,
                reference=f"hf://{repo_id}",
                repository="huggingface",
                tool=tool,
                title=_clip(entry.get("title") or repo_id, 200),
                description=_clip(
                    entry.get("description")
                    or (entry.get("cardData") or {}).get("pretty_name"),
                    300,
                ),
                raw=entry,
            )
        )
    return hits


def geo_datasets(raw: Any, tool: str) -> list[Hit]:
    """GEO's search tools return series, each with an accession like GSE194122."""
    if isinstance(raw, list):
        records = raw
    elif isinstance(raw, dict) and isinstance(raw.get("data"), dict):
        # Probed shape: {"status": ..., "data": {"total": n, "datasets": [...]}}
        records = raw["data"].get("datasets") or []
    elif isinstance(raw, dict):
        records = raw.get("datasets") or []
    else:
        records = []
    hits = []
    for entry in records:
        if isinstance(entry, str):
            entry = {"accession": entry}
        if not isinstance(entry, dict):
            continue
        accession = str(
            entry.get("accession") or entry.get("gse") or entry.get("id") or ""
        ).strip()
        if not _GEO_SERIES.match(accession):
            continue
        hits.append(
            Hit(
                accession=accession,
                reference=f"geo://{accession}",
                repository="geo",
                tool=tool,
                title=_clip(entry.get("title"), 200),
                description=_clip(entry.get("summary") or entry.get("description"), 300),
                raw=entry,
            )
        )
    return hits


def zenodo(raw: Any, tool: str) -> list[Hit]:
    records = _records(raw, "hits", "hits")
    hits = []
    for entry in records:
        if not isinstance(entry, dict):
            continue
        record_id = str(entry.get("id") or "").strip()
        if not record_id:
            continue
        hits.append(
            Hit(
                accession=record_id,
                reference=None,  # no connector routes a Zenodo record
                repository="zenodo",
                tool=tool,
                title=_clip(entry.get("metadata", {}).get("title") or entry.get("title"), 200),
                description=_clip(entry.get("metadata", {}).get("description"), 300),
                raw=entry,
            )
        )
    return hits


ADAPTERS: dict[str, Callable[[Any, str], list[Hit]]] = {
    "omicsdi_datasets": omicsdi,
    "huggingface_datasets": huggingface,
    "geo_datasets": geo_datasets,
    "zenodo_records": zenodo,
}

#: The repositories round one queries by default. Zenodo is available but not
#: default: its tool timed out at 30s when probed, and a slow repository costs
#: more than the leads it returns.
DEFAULT_REPOSITORIES = ("omicsdi", "huggingface", "geo")


#: The repositories round one searches, and how to call each tool. `query_param`
#: differs per tool and is not guessable from the name -- these were read off the
#: tool schemas, not assumed.
REPOSITORIES: dict[str, dict[str, Any]] = {
    "omicsdi": {
        "tool": "OmicsDI_search_datasets",
        "adapter": "omicsdi_datasets",
        "query_param": "query",
        "limit_param": "size",
    },
    "huggingface": {
        "tool": "HuggingFace_search_datasets",
        "adapter": "huggingface_datasets",
        "query_param": "search",
        "limit_param": "limit",
    },
    "geo": {
        "tool": "GEO_search_rnaseq_datasets",
        "adapter": "geo_datasets",
        # Read off the tool schema: GEO's parameter is `query`, not `term`.
        "query_param": "query",
        "limit_param": "limit",
    },
    "zenodo": {
        "tool": "Zenodo_search_records",
        "adapter": "zenodo_records",
        "query_param": "query",
        "limit_param": "limit",
    },
}


def adapt(adapter_name: str, raw: Any, tool: str) -> list[Hit]:
    adapter = ADAPTERS.get(adapter_name)
    if adapter is None:
        raise SearchError(f"no adapter named {adapter_name!r}")
    return adapter(raw, tool)


def hydrate(raw: Any, tool: str) -> list[Hit]:
    """Apply the adapter registered for a tool name."""
    for spec in REPOSITORIES.values():
        if spec["tool"] == tool:
            return adapt(spec["adapter"], raw, tool)
    raise SearchError(f"no repository configured for tool {tool!r}")
