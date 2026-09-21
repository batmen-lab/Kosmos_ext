"""Where fetched data lands, and the limits that bound a download.

Endpoints live here rather than in the modules that call them so a test can
point a connector at a local server. That is the only reason they are fields:
there are no credentials in this package, because every source it knows about
is public. Nothing here decides what a caller *may* fetch -- see the module
docstring in `__init__.py`: this is a fetcher, not a gateway.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

#: Per artifact. A download that is going to be refused should be refused before
#: it starts: the ledger shows a 2.9 GB single-cell file pulled from a series
#: that also offers a 0.6 GB file answering the same question, and two unrelated
#: multi-gigabyte repositories pulled before anything had read them. Raising this
#: is a decision, so it is spelled out on the command line.
DEFAULT_MAX_BYTES = 2 * 1024**3
#: How many members of an archive are unpacked. A GEO `RAW.tar` holds one
#: triplet per sample -- 76 files for a 25-sample series -- and a bounded read
#: needs the first few samples, not the last twenty.
DEFAULT_ARCHIVE_MEMBERS = 12
USER_AGENT = "kosmos-datafetcher/0.1 (+https://github.com/batmen-lab)"


@dataclass(frozen=True)
class DataFetcherConfig:
    """Where to stage, how long to wait, how much is too much."""

    root: Path = field(default_factory=lambda: Path("data/fetched"))
    timeout_s: float = 60.0
    max_bytes: int = DEFAULT_MAX_BYTES
    #: `file://` is copied, not downloaded, so the download cap is the wrong
    #: bound for it: this one is here to stop a `file:///dev/disk0`-shaped
    #: mistake from filling the staging root.
    local_max_bytes: int = 64 * 1024**3
    user_agent: str = USER_AGENT
    offline: bool = False

    # Endpoint, overridable so tests never touch the network.
    geo_ftp_root: str = "https://ftp.ncbi.nlm.nih.gov/geo/series"
    #: NCBI's query endpoint, where the small `brief` series record lives.
    geo_query_root: str = "https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi"
    #: A reference already staged with matching bytes is reused, not re-fetched.
    reuse: bool = True
    #: Members unpacked from one archive. The rest stay inside it, and the note
    #: says how many were left.
    archive_max_members: int = DEFAULT_ARCHIVE_MEMBERS
    #: Interpreter that has tooluniverse installed; the mechanical download
    #: layer runs there (its numpy/mcp pins conflict with Kosmos's own).
    tooluniverse_python: str | None = None
    tooluniverse_timeout_s: float = 900.0
    #: Single-cell files are turned into a bounded table before anything else
    #: sees them: a 50,000 x 20,000 matrix is not a table, and reading all of it
    #: to find that out is how a fetch runs out of memory.
    single_cell_max_cells: int = 2000
    #: 0 means every gene the file has. The gene panel is part of the data for a
    #: single-cell question, not a knob to save memory: 2,000 cells x 14,087
    #: genes is a 132 MB table and 6 seconds to write, and dropping to 200 genes
    #: throws away 98% of the measurements to save nothing that matters.
    single_cell_max_genes: int = 0
    single_cell_scan_cells: int = 5000
    single_cell_label_column: str = "cell_type"
    single_cell_seed: int = 42

    def __post_init__(self) -> None:
        object.__setattr__(self, "root", Path(self.root).expanduser())

    @classmethod
    def from_env(cls, **overrides) -> DataFetcherConfig:
        """Defaults, with the few settings worth having in the environment."""
        values: dict = {}
        if "KOSMOS_DATAFETCHER_ROOT" in os.environ:
            values["root"] = Path(os.environ["KOSMOS_DATAFETCHER_ROOT"])
        if "KOSMOS_DATAFETCHER_MAX_BYTES" in os.environ:
            values["max_bytes"] = int(os.environ["KOSMOS_DATAFETCHER_MAX_BYTES"])
        if os.environ.get("KOSMOS_DATAFETCHER_OFFLINE", "").lower() in {"1", "true", "yes"}:
            values["offline"] = True
        if os.environ.get("KOSMOS_DATAFETCHER_NO_REUSE", "").lower() in {"1", "true", "yes"}:
            values["reuse"] = False
        if os.environ.get("KOSMOS_TOOLUNIVERSE_PYTHON"):
            values["tooluniverse_python"] = os.environ["KOSMOS_TOOLUNIVERSE_PYTHON"]
        if os.environ.get("KOSMOS_TOOLUNIVERSE_TIMEOUT"):
            values["tooluniverse_timeout_s"] = float(os.environ["KOSMOS_TOOLUNIVERSE_TIMEOUT"])
        if os.environ.get("KOSMOS_DATAFETCHER_ARCHIVE_MEMBERS"):
            values["archive_max_members"] = int(os.environ["KOSMOS_DATAFETCHER_ARCHIVE_MEMBERS"])
        for env, attribute, cast in (
            ("KOSMOS_SINGLE_CELL_MAX_CELLS", "single_cell_max_cells", int),
            ("KOSMOS_SINGLE_CELL_MAX_GENES", "single_cell_max_genes", int),
            ("KOSMOS_SINGLE_CELL_SCAN_CELLS", "single_cell_scan_cells", int),
            ("KOSMOS_SINGLE_CELL_LABEL", "single_cell_label_column", str),
            ("KOSMOS_SINGLE_CELL_SEED", "single_cell_seed", int),
        ):
            if os.environ.get(env):
                values[attribute] = cast(os.environ[env])
        # Endpoint override: a mirror, an offline fixture, or a test server.
        for env, attribute in (
            ("KOSMOS_DATAFETCHER_GEO_ROOT", "geo_ftp_root"),
        ):
            if os.environ.get(env):
                values[attribute] = os.environ[env]
        values.update(overrides)
        return cls(**values)

    @property
    def ledger_path(self) -> Path:
        return self.root / "ledger.jsonl"

    @property
    def tmp_dir(self) -> Path:
        """Partials live here so a killed download never looks like a dataset."""
        return self.root / ".partial"

    def ensure_dirs(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        self.tmp_dir.mkdir(parents=True, exist_ok=True)
