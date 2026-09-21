"""Pure dataset fetching for Kosmos experiments. Search, resolve, download, record.

Scope, stated up front so it does not drift:

  * This is a **fetcher**. It finds public datasets, downloads them, and writes
    down where they came from. That is the whole product.
  * There is **no gating**: no policies, no roles, no disclosure thresholds, no
    signed capsules, no banded statistics, no review queue. None of that is
    implemented here, and nothing here refuses a request for privacy reasons.
  * It is **independent of `kosmos`**. This package never imports `kosmos`, and
    `kosmos` never imports this package. A fetched file is handed to Kosmos the
    same way any other file is: `--data-path` or `--external-data-path`.

The ideas worth keeping from AutoEvidence's `finder/`, minus the gateway:

  1. **Search returns pointers, never data.** A hit carries an accession, a
     landing URL and (only when a connector can actually route it) a reference
     such as ``geo://GSE55296``. Bytes arrive through a separate, explicit step.
  2. **One adapter per repository.** Upstream response envelopes differ in ways
     no naming convention predicts, so a payload an adapter does not recognise
     is an error, never a passthrough.
  3. **Provenance or it did not happen.** Every artifact gets a `manifest.json`
     with the source URL, size and sha256; every outbound search is written to
     `ledger.jsonl` *before* it leaves.
  4. **Refuse rather than guess.** An ambiguous GEO series (several platforms),
     an empty series matrix, an accession with the wrong shape: all refuse with
     the alternatives named, instead of landing something that reads downstream
     as a dataset with no rows.

What was deliberately dropped: ToolUniverse (search is plain HTTP here, so the
package has no interpreter and no dependency conflict), the MCP server, the
gateway, signing keys, policies and the three gates.
"""

from .config import DataFetcherConfig
from .errors import (
    DataFetcherError,
    FetchError,
    ReferenceError,
    SearchError,
    SourceError,
)
from .fetch import fetch
from .models import FetchResult, FileRecord
from .plan import (
    PLAN_VERSION,
    DataPlan,
    PlanEntry,
    PlanTask,
    build_plan,
    data_files_under,
    load_plan,
    write_plan,
)
from .profile import (
    ColumnProfile,
    RoleDecision,
    TableProfile,
    TaskShape,
    classify_path,
    classify_role,
    plan_roles,
    profile_table,
)
from .references import Reference, parse_reference
from .store import list_records, read_ledger

__all__ = [
    "DataFetcherConfig",
    "DataFetcherError",
    "FetchError",
    "ReferenceError",
    "SearchError",
    "SourceError",
    "FetchResult",
    "FileRecord",
    "ColumnProfile",
    "RoleDecision",
    "TableProfile",
    "TaskShape",
    "classify_path",
    "classify_role",
    "plan_roles",
    "profile_table",
    "PLAN_VERSION",
    "DataPlan",
    "PlanEntry",
    "PlanTask",
    "build_plan",
    "data_files_under",
    "load_plan",
    "write_plan",
    "Reference",
    "parse_reference",
    "fetch",
    "list_records",
    "read_ledger",
]

__version__ = "0.1.0"
