"""One command from a research question to a training run.

    question -> the model names datasets to download
             -> the fetcher downloads them (mechanical, with provenance)
             -> decide the label column -> DataPlan -> kosmos run

This script is the only place that touches both packages, and it does so
without importing either into the other: `datafetcher` is called as a
subprocess (`python -m datafetcher ...`), Kosmos's inference helper is imported
directly. So the packages stay independent and this file stays disposable.

Why the label column is inferred per candidate: a set of fetched tables does
not say which of them is labeled. Inference is what answers that -- a table
that yields a plausible label is the labeled one, and every other table becomes
supplementary evidence for it. The inference prints its reasoning and asks
before anything trains.

The retrieval step prints as it goes: the prompt it sent, each dataset the model
proposed with its reasoning, and the outcome of every download attempt.

Examples
--------
Two local tables, names only, no model call:

    python scripts/auto_task_run.py \
      --objective "Predict the cell type of each sample from its expression profile" \
      --candidate data/csv_donor_splits/gold_13272_19593_labeled.csv \
      --candidate data/csv_donor_splits/external_unlabeled.csv \
      --exclude-col cell_id --exclude-col DonorID --no-llm --yes

Let the model choose the datasets, download them, and run:

    python scripts/auto_task_run.py \
      --objective "Can wine quality be predicted from physicochemical measurements?" \
      --fetch-limit 3 --yes --run --budget 2
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import subprocess
import sys
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Defaults this driver applies to the run it launches: the research loop's
# optional retrieval steps are the wrong tools for a training run, and a single
# arXiv 429 once turned into a 90-second timeout before training even started.
#
# They are recorded *here*, before `.env` is loaded, because `load_dotenv()`
# writes into `os.environ` -- and the repository `.env` sets
# `ENABLE_SANDBOXING=true`. A plain `setdefault` in `main()` would therefore
# keep the file's value, the run would try to reach Docker, and every
# experiment would fail while the CLI still reported success. Snapshotting the
# exported environment instead means a value the caller typed on the command
# line wins, while a value that only came from `.env` does not.
RUN_ENV_DEFAULTS = {
    "USE_LITERATURE_CONTEXT": "false",
    "REQUIRE_NOVELTY_CHECK": "false",
    "WORLD_MODEL_ENABLED": "false",
    "REDIS_ENABLED": "false",
    "ENABLE_SANDBOXING": "false",
    "NUM_HYPOTHESES": "1",
}
EXPORTED_ENV = {name: os.environ[name] for name in RUN_ENV_DEFAULTS if name in os.environ}

# Load the repository `.env` before anything builds a provider: Kosmos's nested
# provider settings read `os.environ`, not the parent config's env_file, so a
# bare `get_client()` without this finds no key and the inference silently
# degrades to name matching. The CLI does the same at its entry point.
try:
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env")
except ImportError:  # pragma: no cover - python-dotenv is a Kosmos dependency
    pass

PYTHON = sys.executable


class PipelineLog:
    """Every decision and every download, printed and appended to one file.

    The console output is for watching a run; the file is for reading one
    afterwards. Both come from the same call so they cannot drift, and library
    steps that take a `log=` callback get `raw()` so their own messages land in
    the same record instead of only on screen.
    """

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def event(self, stage: str, message: str, **fields) -> None:
        print(message)
        record = {
            "at": datetime.now(UTC).isoformat(timespec="seconds"),
            "stage": stage,
            # The console version leads with a blank line for readability; the
            # file should not.
            "message": message.lstrip("\n"),
            **fields,
        }
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, default=str) + "\n")

    def raw(self, message: str) -> None:
        """A print-compatible sink for steps that log through a callback."""
        self.event("retrieval", message)

    def events(self) -> list[dict]:
        """Everything written so far, for a report that reads the run back."""
        if not self.path.exists():
            return []
        records: list[dict] = []
        for line in self.path.read_text(encoding="utf-8").splitlines():
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        return records


def run_command(args: list[str], *, check: bool = True) -> subprocess.CompletedProcess:
    """Run a subprocess, streaming nothing but keeping the output."""
    result = subprocess.run(args, cwd=ROOT, capture_output=True, text=True)
    if check and result.returncode != 0:
        sys.stderr.write(result.stdout)
        sys.stderr.write(result.stderr)
        raise SystemExit(f"command failed ({result.returncode}): {' '.join(args)}")
    return result


def datafetcher_cli(*args: str) -> list[str]:
    return [PYTHON, "-m", "datafetcher", *args]


#: Extensions worth screening as tables. An archive's extracted contents are
#: candidates in their own right; a `.zip` is not.
TABLE_SUFFIXES = (".csv", ".tsv", ".txt", ".parquet", ".arrow", ".json", ".jsonl", ".data", ".arff")


#: Parts of a single-cell dataset. They are pieces of one table rather than
#: separate tables, so the per-fetch cap does not apply to them: each sample in
#: a GEO `RAW.tar` is a donor, and truncating the list discards donors.
SINGLE_CELL_SUFFIXES = (".h5ad", ".h5ad.gz", ".mtx", ".mtx.gz", ".loom")
SINGLE_CELL_PARTS = ("_matrix_", "_barcodes_", "_genes_", "_features_")


def _single_cell_like(name: str) -> bool:
    lowered = str(name).lower()
    return lowered.endswith(SINGLE_CELL_SUFFIXES) or any(
        part in lowered for part in SINGLE_CELL_PARTS
    )


def _candidate_records(payload: dict, max_files: int | None = None) -> list[dict]:
    """Which staged files are worth screening as tables.

    A `--to-csv` derivative is the same rows in another shape and would double
    count, so a derived file counts only when the file it came from was not
    itself table-like -- which is exactly the archive case: the `.zip` is not a
    table, the CSV inside it is.

    `max_files` bounds how many one fetch may contribute. A GEO `RAW.tar` can
    hold a file per sample -- GSE120221 unpacked to 76 -- and every one of them
    was reviewed by the model before this cap existed. Single-cell artifacts are
    exempt: they are pieces of one dataset, and the samples are the donors.
    """
    records = payload.get("files") or []
    by_path = {str(record.get("path")): record for record in records}
    candidates = []
    for record in records:
        source = record.get("derived_from")
        if source:
            source_name = str((by_path.get(str(source)) or {}).get("path") or source)
            if source_name.lower().endswith(TABLE_SUFFIXES):
                continue
        candidates.append(record)
    if not max_files:
        return candidates
    single_cell = [record for record in candidates if _single_cell_like(record.get("path", ""))]
    rest = [record for record in candidates if not _single_cell_like(record.get("path", ""))]
    return [*single_cell, *rest[: max(0, max_files - len(single_cell))]]


class DownloadBudget:
    """How many bytes this run may still pull, across all of its rounds.

    `--max-bytes` says one artifact is too big; this says the run has spent
    enough. Without it nothing bounded a run: twelve artifacts under the per-file
    cap are twelve downloads, and the ledger's 13.7 GB is what that looks like.
    """

    def __init__(self, megabytes: float | None):
        self.limit = int(megabytes * 1024**2) if megabytes else None
        self.spent = 0

    @property
    def remaining(self) -> int | None:
        if self.limit is None:
            return None
        return max(0, self.limit - self.spent)

    @property
    def spent_out(self) -> bool:
        remaining = self.remaining
        return remaining is not None and remaining <= 0

    def spend(self, size: int) -> None:
        self.spent += max(0, int(size))

    def describe(self) -> str:
        if self.limit is None:
            return f"{self.spent / 1024**2:.0f} MB spent, no limit"
        return (
            f"{self.spent / 1024**2:.0f} of {self.limit / 1024**2:.0f} MB spent, "
            f"{self.remaining / 1024**2:.0f} MB left"
        )


class DownloadChoice:
    """One decided download: the reference to fetch, and why the run kept it."""

    #: A plain class, not a dataclass: this module is also loaded by path (the
    #: tests do it, to check the helper functions without running a pipeline) and
    #: `@dataclass` needs the module to be in `sys.modules` to resolve its own
    #: annotations.
    def __init__(self, reference: str, why: str = ""):
        self.reference = reference
        self.why = why

    @property
    def identifier(self) -> str:
        return self.reference


def describe_reference(reference: str, args, log: PipelineLog) -> dict | None:
    """What a reference contains and what it would cost, without downloading it."""
    base = ["--root", args.root] if args.root else []
    result = run_command(
        datafetcher_cli(*base, "list-files", reference, "--json"), check=False
    )
    if result.returncode != 0 or not result.stdout.strip():
        return None
    try:
        return json.loads(result.stdout)
    except json.JSONDecodeError:
        return None


def _smallest_bytes(listing: dict) -> int | None:
    sizes = [int(entry.get("bytes") or 0) for entry in listing.get("files") or []]
    sizes = [size for size in sizes if size]
    return min(sizes) if sizes else None


#: What a reference looks like: `scheme://locator`, checked before a subprocess
#: is spent on it. The connectors validate properly; this only catches a model
#: that answered with a file name, a title, or an empty string.
_REFERENCE_SHAPE = re.compile(r"^[a-z][a-z0-9+.-]*://\S+$")


def _reference_shape(reference: str) -> bool:
    return bool(_REFERENCE_SHAPE.match(str(reference).strip()))


def _missing_selector(listing: dict) -> str:
    """The file the caller named, when the listing does not contain it.

    This is the check that makes "verify before download" real: a model that
    writes `#train.csv` inside a repository that holds one `person.csv` is
    answered by the listing, not by a failed download two steps later.
    """
    selector = str(listing.get("selector") or "")
    if not selector:
        return ""
    paths = {str(entry.get("path")) for entry in listing.get("files") or []}
    return "" if selector in paths else selector


def _available(listing: dict, most: int = 6) -> str:
    """What the reference does contain, smallest first."""
    entries = sorted(
        (entry for entry in listing.get("files") or []),
        key=lambda entry: int(entry.get("bytes") or 0),
    )
    shown = ", ".join(
        f"{entry.get('path')} ({_human_size(int(entry.get('bytes') or 0))})"
        for entry in entries[:most]
    )
    if len(entries) > most:
        shown += f", ... ({len(entries) - most} more)"
    return shown


def _already_used(reference: str, exclude_refs: set[str], exclude_names: set[str]) -> str:
    """Why this candidate is already in the run, or "" when it is new.

    Two shapes of "already have it": the same reference string (the model copied
    the identifier it was told to avoid), and the same *file* (a series named
    without a selector, whose listing contains the file the run already took).
    The second is the one that bit: the evidence round proposed the whole GEO
    series and the model then re-chose the gold's own supplementary file from it.
    """
    if reference in exclude_refs:
        return "same reference"
    selector = reference.split("#", 1)[1] if "#" in reference else ""
    name = Path(selector or reference).name
    if name and name in exclude_names:
        return "same file"
    return ""


def _human_size(size: int) -> str:
    """`13.5 kB`, not `0.0 MB`: the small files are the ones being chosen."""
    for unit, scale in (("GB", 1024**3), ("MB", 1024**2), ("kB", 1024)):
        if size >= scale:
            return f"{size / scale:.1f} {unit}"
    return f"{size} bytes"


#: Names that say "this file points at the data" rather than "this file is the
#: data". A repository of 268 files and 243 GB lists its measurements, its
#: checksums and its readme at the same level; the model picked the manifest,
#: the run spent its round on three metadata tables, and the plan came out with
#: no gold at all.
_INDEX_SUFFIXES = (".md", ".txt", ".sha256", ".json", ".yaml", ".yml", ".lock", ".csv~")
_INDEX_PARTS = (
    "manifest", "readme", "index", "summary", "checksum", "sha256sums",
    "filelist", "file_list", "files.tsv", "files.csv", "inventory", "catalog",
    "provenance", "metadata.tsv", "metadata.csv", "units.tsv",
)
_DATA_SUFFIXES = (".h5ad", ".h5", ".parquet", ".arrow", ".mtx", ".csv", ".tsv", ".loom")


def _is_index_like(path: str) -> bool:
    lowered = str(path).lower()
    return any(part in lowered for part in _INDEX_PARTS) or lowered.endswith(
        _INDEX_SUFFIXES
    )


def _is_data_like(path: str) -> bool:
    lowered = str(path).lower()
    return lowered.endswith(_DATA_SUFFIXES) and not _is_index_like(lowered)


def _prefer_data_over_index(
    listing: dict, chosen: list[dict], log: PipelineLog
) -> list[dict]:
    """Swap a chosen index for the smallest real data file in the listing.

    The prompt says not to choose an index; this makes it true even when the
    model chooses one anyway -- which is what it did on a 243 GB repository of
    perturbation screens, picking `manifest/*.tsv` because they were the
    smallest files on offer.
    """
    files = list(listing.get("files") or [])
    data_files = [entry for entry in files if _is_data_like(str(entry.get("path") or ""))]
    if not data_files:
        return chosen
    index_choices = [
        entry
        for entry in chosen
        if _is_index_like(str(entry.get("path") or ""))
        or _is_index_like(str(entry.get("reference") or ""))
    ]
    if not index_choices:
        return chosen
    smallest = min(data_files, key=lambda entry: int(entry.get("bytes") or 0))
    reference = str(smallest.get("reference") or "")
    if not reference:
        return chosen
    log.event(
        "preflight",
        f"# preflight: {len(index_choices)} chosen file(s) are an index of the "
        f"data, not the data (e.g. "
        f"{Path(str(index_choices[0].get('path'))).name}); taking "
        f"{Path(str(smallest.get('path'))).name} "
        f"({_human_size(int(smallest.get('bytes') or 0))}) instead",
        replaced=[str(entry.get("reference")) for entry in index_choices],
        chosen=reference,
    )
    kept = [
        entry
        for entry in chosen
        if not _is_index_like(str(entry.get("path") or ""))
        and not _is_index_like(str(entry.get("reference") or ""))
    ]
    if not any(str(entry.get("reference")) == reference for entry in kept):
        kept.append({"reference": reference, "why": "the smallest measurement file"})
    return kept


def _cost_line(listing: dict) -> str:
    """`3 file(s), smallest 587.0 MB` -- what the choice is made on."""
    files = listing.get("files") or []
    smallest = _smallest_bytes(listing)
    parts = [f"{len(files)} file(s) on offer"]
    if smallest:
        parts.append(f"smallest {_human_size(smallest)}")
    if listing.get("bytes"):
        parts.append(f"total {_human_size(int(listing['bytes']))}")
    if not files:
        parts.append(str(listing.get("note") or "size unknown"))
    return ", ".join(parts)


def preflight(
    proposals,
    args,
    client,
    log: PipelineLog,
    *,
    budget: DownloadBudget | None = None,
    max_downloads: int | None = None,
    exclude_refs: set[str] | None = None,
    exclude_names: set[str] | None = None,
) -> tuple[list[DownloadChoice], list[dict]]:
    """Which proposals to download, decided from what they cost.

    Reading a listing costs one API call and no bytes; downloading costs
    whatever the listing says. The rules come first -- the smallest file over
    `--max-bytes`, or one that does not fit what is left of the run's budget, is
    dropped without asking anyone -- and the model then chooses among what is
    left, because relevance is a judgement about meaning and the size is not.

    Most of what a model names does not exist: a path inside a repository that
    has another file, a series that publishes no matrix, a dataset URL that is
    now the site's "not found" page. Those are facts the listing settles before
    anything is transferred, so they are answered here -- and returned as
    refusals, in the same shape a failed download produces, because they are
    just as correctable and the retrieval step is what corrects them.

    Returns `(kept, refusals)`. Every decision is printed with its reason.
    """
    exclude_refs = {str(item) for item in (exclude_refs or ())}
    exclude_names = {str(item) for item in (exclude_names or ())}
    listings: list[dict] = []
    refusals: list[dict] = []
    for proposal in proposals:
        reference = getattr(proposal, "identifier", None) or getattr(
            proposal, "reference", ""
        )
        if not reference:
            continue
        already = _already_used(str(reference), exclude_refs, exclude_names)
        if already:
            log.event(
                "preflight",
                f"# preflight: {reference} -> already in this run ({already}); "
                f"not fetched again",
                reference=str(reference),
                skipped="already used",
            )
            continue
        # An identifier the fetcher cannot even parse is refused now, rather
        # than by a subprocess two steps later.
        if not _reference_shape(str(reference)):
            refusals.append(
                {
                    "reference": str(reference),
                    "error": "this is not an identifier a downloader can act on",
                }
            )
            log.event(
                "preflight",
                f"# preflight: {reference} -> not a fetchable identifier; dropped",
                reference=str(reference),
                skipped="not an identifier",
            )
            continue
        listing = describe_reference(str(reference), args, log)
        if listing is None:
            # A listing the repository will not give says nothing about whether
            # the fetch would work; let the fetch answer that.
            listing = {
                "files": [],
                "note": "the repository did not answer a listing for this reference",
            }
        # The proposal is what will be fetched, so it is what the listing is
        # keyed by, whatever the repository echoed back.
        listing = {**listing, "reference": str(reference)}
        listing["why"] = str(getattr(proposal, "why", "") or "")
        listings.append(listing)

    kept: list[dict] = []
    for listing in listings:
        reference = str(listing.get("reference") or "")
        cost = _cost_line(listing)
        smallest = _smallest_bytes(listing)
        missing = _missing_selector(listing)
        if missing:
            detail = _available(listing)
            refusals.append(
                {
                    "reference": reference,
                    "error": (
                        f"no such file {missing!r} in this reference; it has "
                        f"{detail}"
                        if detail
                        else f"no such file {missing!r} in this reference"
                    ),
                }
            )
            log.event(
                "preflight",
                f"# preflight: {reference} -> there is no {missing!r} "
                f"{'-- it has ' + detail if detail else ''}; nothing was "
                f"downloaded. Name one of those, or another source.",
                reference=reference,
                skipped="no such file",
            )
            continue
        if listing.get("web_page"):
            refusals.append(
                {
                    "reference": reference,
                    "error": (
                        f"that URL is a web page ({listing['web_page']}), not the "
                        f"dataset; the path has probably moved"
                    ),
                }
            )
            log.event(
                "preflight",
                f"# preflight: {reference} -> a web page "
                f"({listing['web_page']}), not a data file; nothing was "
                f"downloaded. Find the current path.",
                reference=reference,
                skipped="web page, not data",
            )
            continue
        if listing.get("error"):
            # The probe could not read the first bytes: a 404, a refused
            # connection, a TLS failure. Better said here than after a retry.
            refusals.append(
                {"reference": reference, "error": str(listing["error"])}
            )
            log.event(
                "preflight",
                f"# preflight: {reference} -> {listing['error']}; nothing was "
                f"downloaded",
                reference=reference,
                skipped="unreachable",
            )
            continue
        if args.max_bytes and smallest and smallest > args.max_bytes:
            log.event(
                "preflight",
                f"# preflight: {reference} -> {cost}; the smallest file is over "
                f"--max-bytes ({args.max_bytes / 1024**2:.0f} MB), so nothing was "
                f"downloaded. Raise --max-bytes or name a smaller file.",
                reference=reference,
                skipped="over per-file cap",
            )
            continue
        if (
            budget is not None
            and budget.remaining is not None
            and smallest
            and smallest > budget.remaining
        ):
            log.event(
                "preflight",
                f"# preflight: {reference} -> {cost}; it does not fit what is "
                f"left of this run's download budget ({budget.describe()}), so "
                f"nothing was downloaded. Raise --max-download-mb to take it.",
                reference=reference,
                skipped="over run budget",
            )
            continue
        log.event(
            "preflight",
            f"# preflight: {reference} -> {cost}"
            + (f"\n#   about: {listing['about']}" if listing.get("about") else ""),
            reference=reference,
            files=listing.get("files"),
            about=listing.get("about"),
        )
        kept.append(listing)

    if (
        client is None
        or args.no_llm
        or getattr(args, "no_preflight", False)
        or not kept
    ):
        return (
            [
                DownloadChoice(
                    reference=str(listing.get("reference") or ""),
                    why=str(listing.get("why") or ""),
                )
                for listing in kept
                if listing.get("reference")
            ],
            refusals,
        )

    from kosmos.discovery import choose_downloads

    decisions = choose_downloads(
        args.objective,
        kept,
        client=client,
        log=log.raw,
        max_downloads=max_downloads,
        used=sorted(exclude_refs | exclude_names),
    )
    offered = {str(listing.get("reference") or "") for listing in kept}
    chosen: list[DownloadChoice] = []
    # The model's choices, before the index/data check below: the check needs to
    # know what it picked, and the log should show both.
    picked: list[tuple[DownloadChoice, dict]] = []
    for decision in decisions:
        reference = str(decision.get("reference"))
        if reference not in offered and not any(
            reference == str(entry.get("reference"))
            for listing in kept
            for entry in listing.get("files") or []
        ):
            continue
        already = _already_used(reference, exclude_refs, exclude_names)
        if already:
            # The model may pick a file out of a listing the run already took;
            # the listing does not know what this run has used, so it is checked
            # here rather than trusted.
            log.event(
                "preflight",
                f"# preflight: {reference} was chosen but is already in this run "
                f"({already}); not fetched again",
                reference=reference,
                skipped="already used",
            )
            continue
        picked.append(
            (
                DownloadChoice(
                    reference=reference, why=str(decision.get("why") or "")
                ),
                listing,
            )
        )
        log.event(
            "preflight",
            f"# preflight: keep {reference} -- {decision.get('why') or 'no reason given'}",
            reference=reference,
        )
    for choice, listing in picked:
        for replacement in _prefer_data_over_index(
            listing, [{"reference": choice.reference}], log
        ):
            chosen.append(
                DownloadChoice(
                    reference=str(replacement["reference"]), why=str(choice.why)
                )
            )
    for listing in kept:
        reference = str(listing.get("reference") or "")
        if any(choice.reference.startswith(reference) for choice in chosen):
            continue
        log.event(
            "preflight",
            f"# preflight: drop {reference} -- the model did not keep it",
            reference=reference,
            skipped="model dropped it",
        )
    if not chosen:
        log.event(
            "preflight",
            "# preflight: the model kept none of the candidates; nothing will be "
            "downloaded for this round",
        )
    return chosen, refusals


def fetch_selected(
    selection,
    root: str | None,
    max_bytes: int | None,
    query: str,
    log: PipelineLog,
    round_name: str = "llm",
    seen_hashes: set[str] | None = None,
    max_files: int | None = None,
    budget: DownloadBudget | None = None,
    archive_members: int | None = None,
) -> tuple[list[str], list[dict]]:
    """Download the chosen datasets, mechanically.

    This is the only part that touches data: give it an identifier, get bytes
    back (or a refusal, printed). A model-named dataset that does not exist fails
    here, which is the verification step -- not a silent omission.

    Returns the files that arrived and the refusals, because a refusal is an
    answer: a GEO series that publishes no series matrix names the supplementary
    files to use instead, and a repository that needs a file named inside it says
    so. Handing that back to the model is what turns a wasted proposal into a
    corrected one.
    """
    base = ["--root", root] if root else []
    fetched: list[str] = []
    failures: list[dict] = []
    # Shared across rounds when the caller passes one: the evidence round often
    # re-proposes the very file the gold round already downloaded, and paying for
    # it twice (download, profile, hash, log line) buys nothing.
    seen_hashes = seen_hashes if seen_hashes is not None else set()
    for candidate in selection:
        # A proposal from the model carries `identifier`; a search hit carries
        # `reference`. Both mean "the thing to download".
        reference = getattr(candidate, "identifier", None) or getattr(candidate, "reference", None)
        if not reference:
            continue
        if budget is not None and budget.spent_out:
            log.event(
                "download",
                f"#   not downloaded: this run's download budget is spent "
                f"({budget.describe()})",
                reference=reference,
                ok=False,
            )
            continue
        log.event(
            "download",
            f"# download [{round_name}]: {reference}",
            reference=reference,
            fetch_round=round_name,
        )
        args = datafetcher_cli(
            *base, "fetch", reference, "--query", query, "--json"
        )
        if max_bytes:
            args += ["--max-bytes", str(max_bytes)]
        if archive_members is not None:
            args += ["--archive-members", str(archive_members)]
        result = run_command(args, check=False)
        if result.returncode != 0:
            reason = (result.stderr or result.stdout).strip().splitlines()
            detail = reason[-1] if reason else "fetch failed"
            log.event(
                "download",
                f"#   FAILED: {detail}",
                reference=reference,
                fetch_round=round_name,
                ok=False,
                error=detail,
            )
            failures.append({"reference": reference, "error": detail})
            continue
        payload = json.loads(result.stdout)
        candidates = _candidate_records(payload, max_files=max_files)
        names = [record["path"] for record in candidates]
        if budget is not None:
            # What came over the wire: a derived table was written here, not
            # downloaded, so it does not count against the budget.
            budget.spend(
                sum(
                    int(record.get("bytes") or 0)
                    for record in (payload.get("files") or [])
                    if not record.get("derived_from")
                )
            )
        log.event(
            "download",
            f"#   ok: {len(names)} file(s) -> {payload['directory']}"
            + (f"  sha256={payload['files'][0]['sha256'][:12]}..." if names else ""),
            reference=reference,
            fetch_round=round_name,
            ok=True,
            directory=payload.get("directory"),
            files=[
                {"path": f["path"], "sha256": f["sha256"], "derived_from": f.get("derived_from")}
                for f in candidates
            ],
        )
        if max_files and len(payload.get("files") or []) > len(candidates):
            log.event(
                "download",
                f"#   {len(payload['files']) - len(candidates)} further file(s) "
                f"from this fetch were not screened (--max-files-per-fetch "
                f"{max_files}); they stay in the staging directory",
            )
        for note in payload.get("notes") or []:
            # A conversion that failed says why here; without this line the run
            # looks like it simply ignored the file.
            log.event("download", f"#   note: {note}"[:400])
        for record in candidates:
            digest = str(record.get("sha256") or "")
            if digest and digest in seen_hashes:
                # The same bytes under another name: round two often re-proposes
                # what round one already downloaded (`imodels/Diabetes-Readmission`
                # and `imodels/diabetes-readmission` are one repository). Fetching
                # it twice doubles the review cost and confuses the audit trail.
                log.event(
                    "download",
                    f"#   duplicate of an earlier download (sha256={digest[:12]}...); skipped",
                    reference=reference,
                    fetch_round=round_name,
                    duplicate=True,
                )
                continue
            if digest:
                seen_hashes.add(digest)
            fetched.append(str(Path(payload["directory"]) / record["path"]))
    return fetched, failures


def download_tools(root: str | None, log: PipelineLog) -> list[dict]:
    """The ToolUniverse tools available to the retrieval step.

    Asked of the fetcher (which owns the ToolUniverse interpreter), not of
    Kosmos: the tool list is part of the mechanical layer. An unconfigured
    interpreter is not fatal -- retrieval can still propose plain identifiers --
    so this returns an empty list and says why. The list carries both the
    read-only search tools (the model's own retrieval) and the download tools.
    """
    base = ["--root", root] if root else []
    result = run_command(datafetcher_cli(*base, "list-tools", "--json"), check=False)
    if result.returncode != 0:
        reason = (result.stderr or result.stdout).strip().splitlines()
        log.event(
            "catalog",
            f"# retrieval: ToolUniverse download tools unavailable "
            f"({reason[-1][:120] if reason else '?'})",
            available=False,
        )
        return []
    tools = json.loads(result.stdout)
    searches = sum(1 for tool in tools if tool.get("kind") == "search")
    log.event(
        "catalog",
        f"# retrieval: {len(tools)} ToolUniverse tool(s) available to the model "
        f"({searches} search, {len(tools) - searches} download)",
        available=True,
        tools=[tool["name"] for tool in tools],
    )
    return tools


def mechanical_round(
    question: str, args, log: PipelineLog, budget: DownloadBudget | None = None
) -> list[str]:
    """Round one: the repositories' own search tools, with a rule-built query.

    No model is involved, so nothing here can invent a dataset. What comes back
    exists; whether it is usable is decided later, by both a rule and a model.
    """
    base = ["--root", args.root] if args.root else []
    result = run_command(
        datafetcher_cli(*base, "search-mechanical", question, "--limit", str(args.fetch_limit), "--json"),
        check=False,
    )
    if result.returncode != 0:
        reason = (result.stderr or result.stdout).strip().splitlines()
        log.event("round1", f"# round 1: mechanical search found nothing "
                            f"({reason[-1][:120] if reason else '?'})", ok=False)
        return []
    payload = json.loads(result.stdout)
    log.event(
        "round1",
        f"# round 1 (mechanical, no model): query={payload.get('query')!r} -> "
        f"{len(payload.get('hits') or [])} hit(s), "
        f"{len(payload.get('references') or [])} fetchable",
        query=payload.get("query"),
        hits=payload.get("hits"),
        references=payload.get("references"),
        failures=payload.get("failures"),
    )
    # The per-repository hit limit is not a download budget: eight references
    # once came back for `--fetch-limit 4`, and every one of them was reviewed by
    # the model. Round one downloads what the caller asked for, best first.
    references = list(payload.get("references") or [])[: args.fetch_limit]
    if len(payload.get("references") or []) > len(references):
        log.event(
            "round1",
            f"# round 1: keeping {len(references)} of "
            f"{len(payload['references'])} fetchable reference(s) (--fetch-limit)",
        )
    fetched: list[str] = []
    for reference in references:
        choice = type("C", (), {"identifier": reference, "why": "round 1 hit"})()
        wanted, _ = preflight([choice], args, None, log, budget=budget, max_downloads=1)
        arrived, _ = fetch_selected(
            wanted,
            args.root,
            args.max_bytes,
            question,
            log,
            round_name="round1",
            max_files=args.max_files_per_fetch,
            archive_members=getattr(args, "archive_members", None),
            budget=budget,
        )
        fetched.extend(arrived)
    return fetched


def _print_rescue(plan: dict, decisions: dict, args, log: PipelineLog, plan_path: Path) -> None:
    """What a person can do to get a run out of "no gold".

    A run that ends with nothing to train on has usually *found* the data: a
    manifest that lists the experiment files, a matrix with genes in rows, a
    table whose labels live in the file beside it. Those are one manual step
    away from being training data, so the last thing this prints is that step --
    per table, with the command that picks the rescued table back up.
    """
    lines = ["", "# how to rescue this run:"]
    salvaged = 0
    for entry in plan.get("unusable") or []:
        path = str(entry.get("path"))
        review = (decisions.get(path) or {}).get("model") or {}
        step = str(review.get("salvage") or "")
        if not step:
            continue
        salvaged += 1
        lines.append(f"#   {Path(path).name}: {step}")
        lines.append(f"#     then: --candidate {path}")
    if not salvaged:
        lines.append(
            "#   no table came with a concrete manual step; the reasons are in "
            f"{Path(plan_path).parent / 'data_report.md'}"
        )
    lines += [
        "#   or re-run with the file a manifest names: "
        ".venv/bin/python -m datafetcher fetch <reference>",
        "#   or hand the run a table you fixed yourself: "
        ".venv/bin/python run.py \"<question>\" --candidate <fixed.csv> "
        "--no-retrieval --out artifacts/runs/<name>",
        "#   every decision is recorded in "
        f"{Path(plan_path).parent / 'data_report.md'}",
    ]
    print("\n".join(lines), file=sys.stderr)
    log.event("rescue", "\n".join(lines), tables=len(plan.get("unusable") or []))


def _write_data_report(
    plan: dict,
    decisions: dict,
    pending: list,
    args,
    log: PipelineLog,
    plan_path: Path,
) -> None:
    """Write the account of what was fetched and why the rest was refused.

    The model's own paragraph per table goes in beside the rules' one-line
    verdict, so a person can judge the gating without re-reading the packets.
    """
    from kosmos.discovery import write_data_report

    markdown, record = write_data_report(
        question=args.objective,
        plan=plan,
        out_dir=plan_path.parent,
        decisions=decisions,
        pending=pending,
        # The log is the reasoning chain: which keywords were searched, what the
        # model proposed, what each listing answered, what was downloaded.
        events=log.events(),
    )
    summary = json.loads(record.read_text())["summary"]
    log.event(
        "report",
        f"# data report: {summary['gold']} gold, {summary['supplementary']} "
        f"supplementary, {summary['unusable']} unusable, "
        f"{summary['awaiting_human']} awaiting a human -> {markdown}",
        path=str(markdown),
        record=str(record),
        **summary,
    )


def run_model_search(identifier: str, args, log: PipelineLog) -> str:
    """Run a search the model asked for, through the fetcher's own search tools.

    The model is offered ToolUniverse's tools, and sometimes it uses one as a
    search (`tu://HuggingFace_search_datasets#{"query": "..."}`) instead of
    naming a dataset. Running it and handing back the hits costs one call and
    keeps the lead; dropping it threw away the best suggestion of the run.
    """
    arguments: dict = {}
    if "#" in identifier:
        _, _, payload = identifier.partition("#")
        try:
            arguments = json.loads(payload)
        except json.JSONDecodeError:
            arguments = {}
    text = ""
    for key in ("query", "q", "search", "keyword", "term", "text"):
        if isinstance(arguments.get(key), str) and arguments[key].strip():
            text = arguments[key].strip()
            break
    if not text:
        log.event("retrieval", f"# search: {identifier!r} carried no query term", ok=False)
        return ""
    return _mechanical_query_search(text, args, log)


def _mechanical_query_search(text: str, args, log: PipelineLog) -> str:
    """Search the repositories with one query, and render the hits as text."""
    base = ["--root", args.root] if args.root else []
    result = run_command(
        datafetcher_cli(
            *base, "search-mechanical", text, "--query", text, "--limit", "5", "--json"
        ),
        check=False,
    )
    if result.returncode != 0:
        return ""
    hits = (json.loads(result.stdout).get("hits") or []) if result.stdout.strip() else []
    lines = [
        f"  {hit.get('accession')} -> {hit.get('reference')}  {(hit.get('title') or '')[:90]}"
        for hit in hits
    ]
    log.event("retrieval", f"# search {text!r}: {len(lines)} hit(s)", query=text, hits=lines)
    return "\n".join(lines)


def failure_report(failures: list[dict]) -> str:
    """What the repositories said, as text a model can correct against.

    A refusal is the most useful thing a fetch produces: "no series matrix,
    here are the supplementary files" and "this series has one matrix per
    platform, pick one" are instructions. Printed and then dropped, they leave
    the run with nothing; handed back, they are a next attempt.
    """
    return "\n".join(
        f"  {failure.get('reference')}\n    {failure.get('error')}"
        for failure in failures
    )


def retry_refused(
    failures: list[dict],
    *,
    args,
    client,
    log: PipelineLog,
    round_name: str,
    seen_hashes: set[str] | None = None,
    budget: DownloadBudget | None = None,
) -> list[str]:
    """Ask the model once more, with the refusals in front of it."""
    if not failures or client is None or args.fetch_retries <= 0:
        return []
    from kosmos.discovery import propose_datasets

    log.event(
        "retrieval",
        f"# retrieval: {len(failures)} identifier(s) were refused; asking the "
        f"model to correct them (retry {round_name})",
        failures=failures,
    )
    try:
        proposals = propose_datasets(
            args.objective,
            client=client,
            max_items=args.fetch_limit,
            hints=list(args.hint or []) + ([args.intent] if args.intent else []),
            tools=download_tools(args.root, log),
            log=log.raw,
            run_search=lambda identifier: run_model_search(identifier, args, log),
            max_hops=args.search_hops,
            retry=failure_report(failures),
        )
    except Exception as e:  # noqa: BLE001 - a failed retry is not a failed run
        log.event("retrieval", f"# retrieval: the correction call failed: {e}", ok=False)
        return []
    wanted, pre_refused = preflight(
        proposals,
        args,
        client,
        log,
        budget=budget,
        max_downloads=args.fetch_limit,
    )
    arrived, _ = fetch_selected(
        wanted,
        args.root,
        args.max_bytes,
        args.objective,
        log,
        round_name=f"{round_name}-retry",
        seen_hashes=seen_hashes,
        max_files=args.max_files_per_fetch,
        archive_members=getattr(args, "archive_members", None),
        budget=budget,
    )
    if not arrived and pre_refused:
        log.event(
            "retrieval",
            "# retrieval: the correction was answered by the listings too; "
            "nothing further was downloaded",
            failures=pre_refused,
        )
    return arrived


def _gold_reference(path: str, args, log: PipelineLog) -> str:
    """Which reference the labeled file came from, so its siblings can be listed."""
    base = ["--root", args.root] if args.root else []
    result = run_command(datafetcher_cli(*base, "list", "--json"), check=False)
    if result.returncode != 0 or not result.stdout.strip():
        return ""
    try:
        records = json.loads(result.stdout)
    except json.JSONDecodeError:
        return ""
    wanted = Path(path).resolve()
    for record in records:
        for entry in record.get("files") or []:
            if Path(entry.get("path", "")).resolve() == wanted:
                return str(record.get("reference") or "")
    return ""


def _distinctive_columns(columns: list[str], limit: int = 4) -> list[str]:
    """Column names worth searching with: measurements, not row numbers."""
    generic = {"id", "index", "row_id", "row_index"}
    picked = [
        name
        for name in columns
        if name.strip().lower() not in generic
        and not name.lower().startswith("unnamed")
        # `__index_level_0__` is how a parquet export spells a row number.
        and not name.strip().lower().startswith("__index_level")
    ]
    picked.sort(key=lambda name: (-len(name), name))
    return picked[:limit]


def find_supplementary(
    args,
    log: PipelineLog,
    client,
    *,
    gold: str,
    target: str,
    seen_hashes: set[str] | None = None,
    budget: DownloadBudget | None = None,
    used: Sequence[str] = (),
) -> list[str]:
    """Round two: other tables carrying the labeled table's columns.

    An external table has to supply the *same measured columns*, so this round
    can only start once the labeled one is chosen. It hands the model three
    cheap things it could not have had before: the column list, the labeled
    repository's other files (listed, not downloaded), and the hits of a
    mechanical search on the column names themselves.
    """
    from kosmos.discovery import propose_evidence

    roles = run_command(
        datafetcher_cli(
            *(["--root", args.root] if args.root else []),
            "roles",
            gold,
            "--target-column",
            target,
            "--json",
        ),
        check=False,
    )
    columns: list[str] = []
    if roles.returncode == 0 and roles.stdout.strip():
        columns = list(json.loads(roles.stdout).get("features") or [])
    if not columns:
        log.event(
            "supplementary",
            "# supplementary: the labeled table has no usable column list; "
            "skipping the evidence round",
            ok=False,
        )
        return []

    reference = _gold_reference(gold, args, log)
    # Everything this run has already taken: the labeled table, whatever came
    # with it, and every file fetched in an earlier round. Naming any of them
    # again is a wasted round trip -- and the evidence round did exactly that,
    # proposing the series and then re-choosing the gold's own file from it.
    used_names = {Path(path).name for path in used if path}
    used_names.add(Path(gold).name)
    used_references = {reference} if reference else set()
    for path in used:
        earlier = _gold_reference(path, args, log) if path else ""
        if earlier:
            used_references.add(earlier)
    log.event(
        "supplementary",
        f"# supplementary: looking for other tables with these {len(columns)} "
        f"column(s) of {Path(gold).name}"
        + (f" (from {reference})" if reference else ""),
        columns=columns,
        reference=reference,
    )

    siblings: list[dict] = []
    if reference:
        listed = run_command(
            datafetcher_cli(
                *(["--root", args.root] if args.root else []),
                "list-files",
                reference,
                "--json",
            ),
            check=False,
        )
        if listed.returncode == 0 and listed.stdout.strip():
            payload = json.loads(listed.stdout)
            siblings = [
                record
                for record in payload.get("files") or []
                if Path(record.get("path", "")).name != Path(gold).name
            ]
            log.event(
                "supplementary",
                f"# supplementary: {reference} holds {len(siblings) + 1} file(s); "
                f"offering the {len(siblings)} the labeled one is not",
                files=[record.get("path") for record in siblings],
            )

    terms = _distinctive_columns(columns)
    hits = _mechanical_query_search(" ".join(terms), args, log) if terms else ""

    proposals = propose_evidence(
        args.objective,
        columns=columns,
        gold_reference=reference or gold,
        sibling_files=siblings,
        search_hits=hits,
        exclude=tuple(sorted(used_names | used_references)),
        client=client,
        max_items=args.supp_limit,
        tools=download_tools(args.root, log),
        log=log.raw,
        run_search=lambda identifier: run_model_search(identifier, args, log),
        max_hops=args.search_hops,
    )
    if not proposals:
        log.event("supplementary", "# supplementary: the model named nothing to add")
        return []
    wanted, pre_refused = preflight(
        proposals,
        args,
        client,
        log,
        budget=budget,
        max_downloads=args.supp_limit,
        exclude_refs=used_references,
        exclude_names=used_names,
    )
    fetched, refused = fetch_selected(
        wanted,
        args.root,
        args.max_bytes,
        args.objective,
        log,
        round_name="supplementary",
        seen_hashes=seen_hashes,
        max_files=args.max_files_per_fetch,
        archive_members=getattr(args, "archive_members", None),
        budget=budget,
    )
    refused = [*pre_refused, *refused]
    if not fetched and refused:
        # The evidence round gets the same correction pass as the gold round:
        # a series that names its per-platform matrix is a next attempt, not a
        # dead end.
        fetched.extend(
            retry_refused(
                refused,
                args=args,
                client=client,
                log=log,
                round_name="supplementary",
                seen_hashes=seen_hashes,
                budget=budget,
            )
        )
    log.event(
        "supplementary",
        f"# supplementary: {len(fetched)} file(s) added as candidates",
        files=fetched,
    )
    return fetched


def screen_tables(candidates: list[str], log: PipelineLog) -> list[str]:
    """The candidates that are per-observation tables at all."""
    tabular, _ = screen_candidates(candidates, log)
    return tabular


def screen_candidates(
    candidates: list[str], log: PipelineLog
) -> tuple[list[str], list[dict]]:
    """Which candidates are per-observation tables at all.

    Target inference and the model review are the expensive steps, and they were
    running over `.gitattributes`, READMEs, archives and single-column series
    matrices. That is how a climate-claims text corpus became this task's label
    column, with a reason invented to fit ("likely representing a binary
    classification of high vs. low efficiency"). A file the profiler cannot see
    at least two columns in is not a table for this purpose; it is still
    reported in the plan, with the rules' own reason.

    The reasons are returned as well as logged: a set of downloads that are all
    "not a table" is a correctable answer about the *identifiers* (a moved URL
    that returns a 404 page looks exactly like this), and the retrieval step can
    only act on it if it is handed back.

    A container that was converted on the way in (`.h5ad` -> `-table.csv`) is
    not one of those: it is the *source* of a candidate, and reading it as
    delimited text is what produced a stray "not a table: 0 parsed column(s)"
    beside the table it had just been converted into. Those are returned with
    `converted: True` for the caller to leave out of its correction round.
    """
    tabular: list[str] = []
    skipped: list[dict] = []
    # The fetcher names a derived table `<source name>-table.csv`, so a source
    # is recognisable from the candidates alone, wherever the two were staged.
    derived = [Path(path).name for path in candidates if Path(path).name.endswith("-table.csv")]
    for path in candidates:
        source_of = next(
            (
                name
                for name in derived
                if name.startswith(f"{Path(path).name}-")
            ),
            None,
        )
        result = run_command(
            datafetcher_cli("profile", path, "--no-count", "--json"), check=False
        )
        profile: dict = {}
        if result.returncode == 0:
            try:
                profile = json.loads(result.stdout)
            except json.JSONDecodeError:
                profile = {}
        if len(profile.get("columns") or []) >= 2:
            tabular.append(path)
            continue
        if source_of is not None:
            skipped.append(
                {
                    "path": path,
                    "converted": True,
                    "derived": source_of,
                    "reason": (
                        f"converted to {source_of}; that table is the candidate "
                        f"being reviewed"
                    ),
                }
            )
            continue
        notes = profile.get("notes") or []
        skipped.append(
            {"path": path, "reason": (notes[0] if notes else "not a parsed table")[:160]}
        )
    converted = [entry for entry in skipped if entry.get("converted")]
    log.event(
        "screen",
        f"# screening: {len(tabular)} of {len(candidates)} candidate(s) are tables; "
        f"{len(skipped)} skipped before target inference and review"
        + (
            f" ({len(converted)} of them are the source a converted table came from)"
            if converted
            else ""
        )
        + "".join(
            f"\n#   {Path(entry['path']).name}: {entry['reason']}"
            for entry in skipped[:8]
        ),
        tabular=tabular,
        skipped=skipped,
        converted=converted,
    )
    return tabular, skipped


def infer_for_candidates(candidates: list[str], args, log: PipelineLog) -> tuple[str, str, dict]:
    """Try each candidate until one yields a label column, and say why."""
    from kosmos.ppi.task_inference import column_facts, infer_target_column

    client = None
    if not args.no_llm:
        try:
            from kosmos.core.llm import get_client

            client = get_client()
        except Exception as e:  # noqa: BLE001
            print(f"# no model available ({e}); name matching only")
    attempts = []
    for candidate in candidates:
        try:
            facts = column_facts(candidate, max_rows=args.scan_rows)
        except Exception as e:  # noqa: BLE001 - an unreadable table is a skip
            attempts.append((candidate, None, f"could not read: {e}"))
            continue
        result = infer_target_column(
            candidate,
            objective=args.objective,
            hints=list(args.hint or []),
            exclude=list(args.exclude_col or []),
            feature_prefixes=list(args.feature_prefix or []),
            client=client,
            facts=facts,
            task_type=str(getattr(args, "task_type", "classification")),
        )
        attempts.append((candidate, result, result.reason))
        if result.column:
            log.event(
                "target",
                f"\n# labeled candidate: {candidate}\n"
                f"#   target column : {result.column}\n"
                f"#   task type     : {result.task_type}\n"
                f"#   decided by    : {result.source} ({result.confidence})\n"
                f"#   reason        : {result.reason}",
                path=candidate,
                target_column=result.column,
                task_type=result.task_type,
                source=result.source,
                confidence=result.confidence,
                reason=result.reason,
                rejected_total=len(result.rejected),
            )
            if result.candidates:
                log.event(
                    "target",
                    "#   runners-up   : "
                    + ", ".join(f"{c.column} ({c.score:.2f})" for c in result.candidates[1:4]),
                    runners_up=[
                        {"column": c.column, "score": c.score}
                        for c in result.candidates[1:4]
                    ],
                )
            return candidate, result.column, {
                "source": result.source,
                "confidence": result.confidence,
                "reason": result.reason,
                "task_type": result.task_type,
            }
    log.event("target", "\n# no candidate yielded a label column:", ok=False)
    for candidate, _result, reason in attempts:
        log.event("target", f"#   {candidate}: {reason}", path=candidate, reason=reason)
    raise SystemExit(2)


def review_candidates(
    candidates: list[str],
    args,
    client,
    log: PipelineLog,
    out_dir: Path,
    target_hint: str = "",
) -> tuple[dict, dict]:
    """Sample every candidate's head, let the model review it, and record both.

    Returns `(reviews, normalized)`: the review file the plan will consume, and a
    map from original path to a header-corrected copy for tables the review found
    to be headerless.
    """
    from kosmos.discovery.review import review_table

    reviews: dict[str, dict] = {}
    normalized: dict[str, str] = {}
    for path in candidates:
        packet = sample_packet(path, args, log)
        if packet is None:
            continue
        review = review_table(
            args.objective,
            packet,
            client=client,
            known_paths=candidates,
            log=log.raw,
            target_hint=target_hint,
        )
        if review is None:
            continue
        entry = review.to_dict()
        # Carry the first lines into the decision record: when the model argues
        # with the rules, a person has to see the same evidence the model did.
        entry["raw_head"] = list(packet.get("raw_head") or [])
        reviews[path] = entry
        if not review.header_present and review.header_names:
            target = out_dir / "normalized" / Path(path).name
            if write_with_header(path, review.header_names, target):
                normalized[path] = str(target)
                log.event(
                    "review",
                    f"# review: {Path(path).name} has no header; wrote a normalized "
                    f"copy with the names the model supplied -> {target}",
                    path=path,
                    normalized=str(target),
                    header_names=review.header_names,
                    source=review.source,
                )
    return reviews, normalized


def sample_packet(path: str, args, log: PipelineLog) -> dict | None:
    """The review packet for one table, via the fetcher (which owns the readers)."""
    result = run_command(
        datafetcher_cli(
            *(["--root", args.root] if args.root else []),
            "sample",
            path,
            "--rows",
            "5",
            "--json",
        ),
        check=False,
    )
    if result.returncode != 0:
        log.event("review", f"# review: could not sample {Path(path).name}", path=path)
        return None
    packet = json.loads(result.stdout)
    log.event(
        "review",
        f"# review: sampled {Path(path).name} ({len(packet.get('columns') or [])} columns, "
        f"{len(packet.get('raw_head') or [])} lines)",
        path=path,
        columns=packet.get("columns"),
        raw_head=packet.get("raw_head"),
    )
    return packet


def write_with_header(path: str, names: list[str], target: Path) -> bool:
    """Copy a headerless table with `names` as its header, mechanically checked.

    The names come from the model, so the shape has to agree exactly: one name
    per column, all non-empty and distinct. A mismatch is a rejected review, not
    a rewritten file.
    """
    import csv

    source = Path(path)
    with source.open("r", encoding="utf-8", errors="replace", newline="") as handle:
        reader = csv.reader(handle)
        rows = list(reader)
    if not rows:
        return False
    width = len(rows[0])
    if len(names) != width or len(set(names)) != width or any(not n.strip() for n in names):
        return False
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(names)
        writer.writerows(rows)
    return True


def mechanical_role(
    path: str, target: str, args, log: PipelineLog
) -> dict[str, str]:
    """What the rules say about one table, for the target it claims.

    Per table, not once for all of them: the model may name a *different* target
    for a table than the one the global inference picked (a one-hot table's
    `diag_1:Diabetes` versus a clean table's `Diabetes_binary`), and checking a
    table against a target that lives in another file is how a perfectly good
    labeled table gets called unusable.
    """
    command = datafetcher_cli(
        *(["--root", args.root] if args.root else []),
        "roles",
        path,
        "--target-column",
        target,
    )
    for flag, values in (
        ("--exclude-column", args.exclude_col),
        ("--feature-prefix", args.feature_prefix),
    ):
        for value in values or []:
            command += [flag, value]
    command.append("--json")
    result = run_command(command, check=False)
    if result.returncode != 0:
        return {"role": "unusable", "reason": "the mechanical check could not run"}
    payload = json.loads(result.stdout)
    if "plan" in payload:  # multiple paths: take ours
        for role in ("gold", "supplementary", "unusable"):
            for entry in payload["plan"].get(role) or []:
                if str(entry.get("path")) == path:
                    return {**entry, "role": role}
        return {"role": "unusable", "reason": "not classified"}
    return payload


def dual_review(
    candidates: list[str],
    target: str,
    args,
    client,
    log: PipelineLog,
    out_dir: Path,
    review_paths: list[str] | None = None,
    known_reviews: dict[str, dict] | None = None,
    write: bool = True,
    target_decision: dict | None = None,
) -> tuple[dict, list[dict], dict[str, str]]:
    """Rules and model on every table; disagreements about *more* access go to a person.

    Returns the decisions the plan will apply, and the cases left for a human.
    Non-interactive runs do not block: the rules hold, and the case is listed in
    `adjudication.json` and in the report.

    `known_reviews` is how the supplementary round reuses the gold round: the
    tables already judged keep their verdicts, and only the new ones reach the
    model. `write=False` lets a caller judge one batch and record the merged
    result later.

    `target_decision` is how the label column was found (`infer_for_candidates`).
    It decides whether a refusal by the model can be overruled: the rules hold
    when the column came from an explicit hint, because then they are reading a
    protocol, and they do not when it came from the question's wording, because
    then they are reading prose.
    """
    from kosmos.discovery import Disagreement, parse_answer, resolve

    # Only tables reach the model. Everything else keeps its mechanical verdict,
    # which is what the plan will read: fewer calls, and no review of a README.
    # A table that was already judged keeps its verdict: the evidence round
    # re-proposes the labeled table's own source often enough that re-reviewing
    # it is a second call for the same answer.
    already = set(known_reviews or ())
    llm_reviews, normalized = review_candidates(
        [
            path
            for path in (review_paths if review_paths is not None else candidates)
            if path not in already
        ],
        args,
        client,
        log,
        out_dir,
        target_hint=target,
    )
    llm_reviews = {**(known_reviews or {}), **llm_reviews}
    by_name = {Path(p).name: p for p in candidates}
    llm_reviews = {by_name.get(Path(p).name, p): r for p, r in llm_reviews.items()}
    answers = load_adjudications(args.adjudicate)
    decisions: dict[str, dict] = {}
    pending: list[dict] = []
    interactive = sys.stdin.isatty() and not args.yes
    # The rules may only overrule a model's "this table has no label column"
    # when the label they found was named by a person (`--hint`). A column read
    # out of the question's prose is a guess, and a guess does not outvote the
    # model: that is how a gene (`WAS`, from "a cell *was* treated") turned a
    # gene-expression matrix into a labeled dataset.
    hinted_target = str((target_decision or {}).get("source") or "") in {
        "hint_exact",
        "hint_match",
    }

    for path in candidates:
        review = llm_reviews.get(path) or {}
        # The target this table claims: the model's own reading of the table when
        # it named one, otherwise the column the global inference found.
        table_target = str(review.get("target_column") or target)
        mechanical = mechanical_role(path, table_target, args, log)
        review["mechanical_target"] = table_target
        model_role = review.get("role")
        role = resolve(mechanical["role"], model_role)
        decided_by = "mechanical+model"

        # A refusal on a fact the rules can check is a disagreement, not a
        # verdict. The model called a 202-column single-cell table "headerless
        # with no cell-type column" while the rules had already found
        # `cell_type` in it -- and a model refusal silently overruling a rule
        # that is demonstrably right is how a run ends with no data at all.
        if (
            role == "unusable"
            and model_role == "unusable"
            and review.get("blocker") == "no_label_column"
            and mechanical.get("has_target")
            and hinted_target
        ):
            log.event(
                "review",
                f"# review: the model refused {Path(path).name} for a missing "
                f"label column, but the rules see {table_target!r} "
                f"in it; asking a person",
                path=path,
            )
            role = "pending"
        elif (
            role == "unusable"
            and model_role == "unusable"
            and review.get("blocker") == "no_label_column"
            and mechanical.get("has_target")
        ):
            # The rules found a label column, the model read the file and says
            # there is none, and the rules' column came from the question rather
            # than from a hint. The model's reading is the better evidence, so
            # the table is not labeled data; the model's own rescue step travels
            # with it (`adjudication.json`, `data_report.md`).
            log.event(
                "review",
                f"# review: the model refused {Path(path).name} for a missing "
                f"label column, and the rules' candidate {table_target!r} came "
                f"from the question's wording rather than from a --hint; the "
                f"refusal stands",
                path=path,
                target=table_target,
                salvage=str(review.get("salvage") or ""),
            )

        if role == "pending":
            case = Disagreement(
                path=path,
                mechanical_role=mechanical["role"],
                mechanical_reason=mechanical.get("reason", ""),
                model_role=str(model_role),
                model_reason=str(review.get("reason", "")),
                raw_head=list(review.get("raw_head") or []),
                model_salvage=str(review.get("salvage") or ""),
            )
            answer = answers.get(path) or answers.get(Path(path).name)
            if answer is None and interactive:
                print()
                print(case.question())
                try:
                    answer = input("> ")
                except EOFError:
                    answer = "s"
            if answer is not None:
                role = parse_answer(answer)
                decided_by = "human"
                log.event(
                    "adjudication",
                    f"# adjudication: human said {role} for {Path(path).name}",
                    path=path,
                    answer=answer,
                    role=role,
                )
                if role == "pending":  # "skip for now": the rules hold
                    role = mechanical["role"]
                    decided_by = "mechanical (human deferred)"
            else:
                pending.append(case.to_dict())
                log.event(
                    "adjudication",
                    f"# adjudication NEEDED: rules say {mechanical['role']!r}, model says "
                    f"{model_role!r} for {Path(path).name} — the rules hold until you decide",
                    path=path,
                    mechanical=mechanical,
                    model=review,
                )
                role = mechanical["role"]
                decided_by = "mechanical (pending human)"

        decisions[path] = {
            "role": role,
            "decided_by": decided_by,
            "mechanical": mechanical,
            "mechanical_entry": mechanical,
            "model": review,
            "reason": review.get("reason") or mechanical.get("reason", ""),
            "header_present": review.get("header_present", True),
            "header_names": review.get("header_names") or [],
            "target_column": review.get("target_column"),
            "id_columns": review.get("id_columns") or [],
            "label_problem": review.get("label_problem") or "",
            "raw_head": review.get("raw_head") or [],
        }

    if decisions and write:
        reviews_path = out_dir / "reviews.json"
        reviews_path.write_text(
            json.dumps(
                {
                    "objective": args.objective,
                    "originals": normalized,
                    "reviews": {
                        normalized.get(path, path): decision
                        for path, decision in decisions.items()
                    },
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        log.event(
            "review",
            f"# review: decisions written to {reviews_path} "
            f"({len(decisions)} table(s), {len(pending)} awaiting a human)",
            path=str(reviews_path),
            roles={Path(p).name: d["role"] for p, d in decisions.items()},
            decided_by={Path(p).name: d["decided_by"] for p, d in decisions.items()},
        )
    if pending and write:
        pending_path = out_dir / "adjudication.json"
        pending_path.write_text(json.dumps({"pending": pending}, indent=2), encoding="utf-8")
        log.event(
            "adjudication",
            f"# adjudication: {len(pending)} case(s) need you -> {pending_path}\n"
            f"#   answer them with --adjudicate {pending_path} (add a 'role' per case) "
            f"and re-run, or re-run interactively",
            path=str(pending_path),
        )
    return decisions, pending, normalized


def load_adjudications(path: str | None) -> dict[str, str]:
    """Human answers from a previous run, keyed by path or by file name."""
    if not path:
        return {}
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001 - an unreadable file means "nothing answered"
        return {}
    out: dict[str, str] = {}
    for case in (payload.get("pending") or payload.get("answers") or []):
        if isinstance(case, dict) and case.get("role"):
            out[str(case.get("path", ""))] = str(case["role"])
    return out


def plan_source(inference_source: str, hints: list[str]) -> str:
    """Map "how the column was chosen" onto the plan's provenance vocabulary."""
    if inference_source == "llm":
        return "llm"
    if inference_source in {"hint_exact", "hint_match"}:
        # A hint is either the protocol's dependent variable or a flag the
        # caller typed; both mean a person or a document named it.
        return "protocol" if not hints else "cli"
    return "objective"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--objective", required=True, help="the research question")
    parser.add_argument(
        "--candidate", action="append", default=[], help="a table to consider (repeatable)"
    )
    parser.add_argument(
        "--intent",
        help="a term the model should consider when choosing datasets (optional)",
    )
    parser.add_argument(
        "--fetch-limit",
        type=int,
        default=3,
        help="how many datasets the model may propose and the fetcher may download",
    )
    parser.add_argument(
        "--max-bytes",
        type=int,
        help="size cap for one fetched dataset (default: the fetcher's 8 GiB)",
    )
    parser.add_argument("--root", help="fetcher staging root (default: data/fetched)")
    parser.add_argument("--hint", action="append", help="a name the label might have")
    parser.add_argument("--exclude-col", action="append", help="never the label (repeatable)")
    parser.add_argument("--feature-prefix", action="append", help="feature columns (repeatable)")
    parser.add_argument("--sample-id-column", help="column holding per-row identifiers")
    parser.add_argument(
        "--task-type",
        default="auto",
        choices=("auto", "classification", "regression"),
        help=(
            "what kind of label to look for: a category, a measured value, or "
            "let the model decide from the question (default: auto)"
        ),
    )
    parser.add_argument(
        "--mechanical-search",
        action="store_true",
        help=(
            "also run the rule-built keyword search before the model chooses "
            "(off by default: it matches on single words and downloads what it "
            "finds)"
        ),
    )
    parser.add_argument(
        "--search-hops",
        type=int,
        default=3,
        help=(
            "how many times the model may look something up before it has to "
            "name the datasets (default: 3)"
        ),
    )
    parser.add_argument(
        "--supp-limit",
        type=int,
        default=3,
        help=(
            "how many supplementary tables the evidence round may download "
            "(default: 3; the gold round is capped by --fetch-limit)"
        ),
    )
    parser.add_argument(
        "--fetch-retries",
        type=int,
        default=1,
        help=(
            "how many times the model may correct an identifier a repository "
            "refused (default: 1; the refusal text is handed back to it)"
        ),
    )
    parser.add_argument(
        "--max-files-per-fetch",
        type=int,
        default=8,
        help=(
            "how many files one fetch may contribute as candidates (default: 8; "
            "a GEO RAW.tar can hold one file per sample)"
        ),
    )
    parser.add_argument(
        "--max-download-mb",
        type=float,
        default=6144.0,
        help=(
            "how many megabytes the whole run may download, across all rounds "
            "(default: 6144). A run stops taking artifacts when it is spent, and "
            "says so; 0 means no run-level limit"
        ),
    )
    parser.add_argument(
        "--no-preflight",
        action="store_true",
        help=(
            "skip the model call that reads each candidate's size and description "
            "before downloading (the size and budget rules still apply)"
        ),
    )
    parser.add_argument(
        "--archive-members",
        type=int,
        default=12,
        help=(
            "how many files to unpack from one archive (default: 12; a GEO "
            "RAW.tar holds one triplet per sample). 0 unpacks everything"
        ),
    )
    parser.add_argument(
        "--refresh",
        action="store_true",
        help=(
            "download again even when matching bytes are already staged (the "
            "default reuses them: the ledger shows the same file fetched nine "
            "times)"
        ),
    )
    parser.add_argument(
        "--min-shared-features",
        type=int,
        default=None,
        help=(
            "how many of the gold's measured columns an evidence table must share "
            "to be used (default: 2). The correction then trains on the "
            "intersection, for both arms"
        ),
    )
    parser.add_argument(
        "--min-shared-fraction",
        type=float,
        default=None,
        help=(
            "and the share of the gold's columns that has to be shared (default: "
            "0.5; a 12-gene overlap with a 14,000-gene gold is not evidence)"
        ),
    )
    parser.add_argument(
        "--test-path",
        help="fixed labeled test table; omit to split the labeled table at random",
    )
    parser.add_argument("--plan-out", default="artifacts/auto/plan.json")
    parser.add_argument("--no-llm", action="store_true", help="name matching only")
    parser.add_argument(
        "--scan-rows",
        type=int,
        default=20000,
        help="rows to read per candidate while choosing the label column",
    )
    parser.add_argument("--yes", action="store_true", help="do not ask before running")
    parser.add_argument("--run", action="store_true", help="launch kosmos run afterwards")
    parser.add_argument("--budget", type=float, default=2.0)
    parser.add_argument("--max-iterations", type=int, default=1)
    parser.add_argument(
        "--domain",
        default="biology",
        help="research domain passed to the Kosmos run (biology, chemistry, ...)",
    )
    parser.add_argument(
        "--adjudicate",
        help="JSON file of human answers to earlier disagreements "
        "(the adjudication.json a previous run wrote)",
    )
    args = parser.parse_args()

    # See RUN_ENV_DEFAULTS: exported values win, `.env` values do not. This is
    # the same overwrite `run.py`'s `environment()` does, so both entry points
    # hand the child the same switches.
    for name, value in RUN_ENV_DEFAULTS.items():
        os.environ[name] = EXPORTED_ENV.get(name, value)
    # The fetcher runs as a subprocess per download, so a run-level policy has to
    # reach it through the environment: `--refresh` is the one that turns reuse
    # off for every round at once.
    if args.refresh:
        os.environ["KOSMOS_DATAFETCHER_NO_REUSE"] = "1"
    # The plan is built in a subprocess, so these reach it the same way.
    if args.min_shared_features is not None:
        os.environ["KOSMOS_MIN_SHARED_FEATURES"] = str(args.min_shared_features)
    if args.min_shared_fraction is not None:
        os.environ["KOSMOS_MIN_SHARED_FRACTION"] = str(args.min_shared_fraction)

    candidates = [str(Path(c)) for c in args.candidate]
    trace: dict = {"objective": args.objective}
    trace_path = Path(args.plan_out).with_name("search_trace.json")
    plan_path = Path(args.plan_out)
    log = PipelineLog(trace_path.with_name("pipeline_log.jsonl"))
    log.event(
        "start",
        f"# pipeline start: {args.objective[:120]}",
        objective=args.objective,
        fetch_limit=args.fetch_limit,
        candidates=list(candidates),
    )
    # The model is used twice: to name datasets when the pool has to be found,
    # and always to review the tables that were found. It is built once here so
    # the `--candidate` path (which skips retrieval) still has one for the
    # review; with `--no-llm` it stays None and both steps fall back to rules.
    client = None
    if not args.no_llm:
        try:
            from kosmos.core.llm import get_client

            client = get_client()
        except Exception as e:  # noqa: BLE001
            log.event("retrieval", f"# retrieval: no model available ({e})", ok=False)

    retrieval_on = bool(args.intent) or not args.candidate
    # One hash set for the whole run: a file fetched in the gold round is not
    # fetched again by the evidence round.
    fetch_hashes: set[str] = set()
    # One budget for the whole run: the per-file cap bounds one artifact, and
    # this bounds what all the rounds together may pull.
    budget = DownloadBudget(args.max_download_mb)
    if retrieval_on:
        # Retrieval is the model's decision. A rule-built keyword query used to
        # run first -- the repositories' own search tools, no model involved --
        # and it cost more than it found: "chronic kidney disease" matched four
        # commit-chronicle repositories, and every one of them was downloaded,
        # reviewed and refused. `--mechanical-search` puts it back for a caller
        # who wants a provenance-free first pass.
        from kosmos.discovery import propose_datasets

        if client is None:
            print(
                "# retrieval needs the model to choose datasets; pass --candidate "
                "to name tables directly, or fix the provider configuration",
                file=sys.stderr,
            )
            return 2
        if args.mechanical_search:
            candidates = mechanical_round(args.objective, args, log, budget) + candidates
        else:
            log.event(
                "round1",
                "# round 1 (mechanical keyword search) is off: the model chooses "
                "the datasets; ToolUniverse is used to download them, and the "
                "model may still call a search tool itself",
            )
        proposals = propose_datasets(
            args.objective,
            client=client,
            max_items=args.fetch_limit,
            hints=list(args.hint or []) + ([args.intent] if args.intent else []),
            tools=download_tools(args.root, log),
            log=log.raw,
            run_search=lambda identifier: run_model_search(identifier, args, log),
            max_hops=args.search_hops,
        )
        trace["proposals"] = [p.__dict__ for p in proposals]
        log.event(
            "retrieval",
            f"# retrieval: the model named {len(proposals)} dataset(s) to download",
            proposals=[
                {"identifier": p.identifier, "confidence": p.confidence, "why": p.why}
                for p in proposals
            ],
        )
        wanted, pre_refused = preflight(
            proposals,
            args,
            client,
            log,
            budget=budget,
            max_downloads=args.fetch_limit,
        )
        chosen, refused = fetch_selected(
            wanted,
            args.root,
            args.max_bytes,
            args.objective,
            log,
            round_name="model",
            seen_hashes=fetch_hashes,
            max_files=args.max_files_per_fetch,
            archive_members=getattr(args, "archive_members", None),
            budget=budget,
        )
        # A candidate the listing already answered is as correctable as a failed
        # download, so the correction round sees both.
        refused = [*pre_refused, *refused]
        for attempt in range(max(0, args.fetch_retries)):
            if chosen or not refused:
                break
            corrected = retry_refused(
                refused,
                args=args,
                client=client,
                log=log,
                round_name=f"model-{attempt + 1}",
                seen_hashes=fetch_hashes,
            )
            chosen.extend(corrected)
            if corrected:
                break
        candidates.extend(chosen)
    if not candidates:
        print(
            f"\n# nothing usable to train on:\n"
            f"#   proposals   : {len(trace.get('proposals') or [])}\n"
            f"#   downloaded  : 0 files\n"
            f"#   trace       : {trace_path}\n"
            f"# every proposal the model named failed to download -- see the FAILED "
            f"lines above. Pass --candidate to name tables that already exist "
            f"locally, or try again: a different question may name datasets that do.",
            file=sys.stderr,
        )
        return 2

    print(f"# {len(candidates)} candidate table(s)")
    # Inference and review see tables, not whatever else came down the wire.
    tables, not_tables = screen_candidates(candidates, log)
    # A converted source is not a fetch that went wrong, so the model is not
    # asked to name it again; it is already in the list as its own table.
    not_tables = [entry for entry in not_tables if not entry.get("converted")]
    if not tables and retrieval_on and client is not None and args.fetch_retries > 0:
        # Downloads that are all "not a table" are as correctable as a refused
        # identifier: a moved dataset URL answers 200 with the site's 404 page,
        # and the model can name the file it meant if it is told that. One round,
        # then the run stops either way.
        corrected = retry_refused(
            [
                {
                    "reference": _gold_reference(entry["path"], args, log)
                    or entry["path"],
                    "error": f"downloaded, but {entry['reason']}",
                }
                for entry in not_tables
            ],
            args=args,
            client=client,
            log=log,
            round_name="screening",
            seen_hashes=fetch_hashes,
            budget=budget,
        )
        if corrected:
            candidates.extend(corrected)
            tables, _ = screen_candidates(corrected, log)
    if not tables:
        reasons = "; ".join(
            f"{Path(entry['path']).name}: {entry['reason']}" for entry in not_tables[:4]
        )
        print(
            "# none of the fetched files is a table (see the screening lines "
            "above): there is nothing to infer a label from, and nothing to "
            "train on"
            + (f"\n#   {reasons}" if reasons else ""),
            file=sys.stderr,
        )
        return 2
    _, target, decision = infer_for_candidates(tables, args, log)

    # Both judgements on every candidate: the rules, and the model reading the
    # head of the file. Where the model argues for *more* access than the rules
    # allow, the case is parked for a person instead of being decided here.
    out_dir = plan_path.parent
    decisions, pending, normalized = dual_review(
        candidates,
        target,
        args,
        client,
        log,
        out_dir,
        review_paths=tables,
        target_decision=decision,
    )
    if normalized:
        # A headerless table was re-written with the names the model recognized;
        # everything downstream must use that copy, or the plan reads the file
        # whose columns are values.
        candidates = [normalized.get(path, path) for path in candidates]
        trace["normalized"] = normalized
    trace["decisions"] = {
        path: {"role": d["role"], "decided_by": d["decided_by"]}
        for path, d in decisions.items()
    }
    trace["pending_adjudication"] = pending
    plan_reviews_args = (
        ["--reviews", str(out_dir / "reviews.json")] if decisions else []
    )

    trace_path.parent.mkdir(parents=True, exist_ok=True)

    # Which labeled table to train on is a judgement about relevance, and the
    # model makes it: more labels is not automatically better here. The
    # candidates are the tables the rules and the model agreed are gold -- each
    # with the target column *it* carries, not the one the global inference
    # happened to find in some other file.
    staging_root = args.root or "data/fetched"
    primary_labeled_args: list[str] = []
    gold_entries = [
        {**entry["mechanical_entry"], "path": path}
        for path, entry in decisions.items()
        if entry["role"] == "gold"
    ]
    if gold_entries:
        # The plan's target is the one the chosen gold actually carries. Without
        # this the run trains against a column name invented for another table.
        target = str(decisions[gold_entries[0]["path"]].get("target_column") or target)
        # Any other table the rules accepted still has to be judged against
        # *this* target: one model means one label column, and a table whose
        # target differs is evidence at best. Said out loud, because the decision
        # record would otherwise claim a gold the plan is about to refuse.
        for path, entry in decisions.items():
            if entry["role"] == "gold" and str(entry.get("target_column") or "") != target:
                log.event(
                    "review",
                    f"# review: {Path(path).name} is labeled for "
                    f"{entry.get('target_column')!r}, not {target!r}; it is evidence "
                    f"for this task at best",
                    path=path,
                    role="gold (different target)",
                    decided_by=entry["decided_by"],
                )
    if len(gold_entries) > 1:
        from kosmos.discovery import choose_primary_gold

        choice = choose_primary_gold(args.objective, gold_entries, client=client, log=log.raw)
        trace["primary_gold"] = choice.__dict__
        primary_labeled_args = ["--labeled-path", normalized.get(choice.path, choice.path)]
        target = str(decisions.get(choice.path, {}).get("target_column") or target)
        primary_gold = choice.path
        log.event(
            "primary_gold",
            f"# primary gold: {choice.path}  ({choice.source})",
            path=choice.path,
            source=choice.source,
            reason=choice.reason,
            target_column=target,
            candidates=[entry["path"] for entry in gold_entries],
        )
    elif gold_entries:
        primary_gold = str(gold_entries[0]["path"])
    else:
        primary_gold = ""

    # ---- round two: evidence for the table that was just chosen -------------
    # Until now there was nothing to aim at. An external table has to supply the
    # labeled table's own columns, so the search for one can only start once the
    # labeled table is known -- and it runs automatically, because "is there
    # external data at all" is part of the question being asked.
    if primary_gold:
        supp_candidates = (
            find_supplementary(
                args,
                log,
                client,
                gold=normalized.get(primary_gold, primary_gold),
                target=target,
                seen_hashes=fetch_hashes,
                budget=budget,
                used=candidates,
            )
            if client is not None
            else []
        )
        if client is None:
            log.event(
                "supplementary",
                "# supplementary: no model in this run (--no-llm), so the "
                "evidence round is skipped; named candidates are used as given",
            )
        if supp_candidates:
            candidates.extend(supp_candidates)
            new_tables = screen_tables(supp_candidates, log)
            if new_tables:
                # The tables already judged keep their verdicts; only the new
                # ones reach the model, and the merged record is written once.
                known = {
                    path: decision["model"]
                    for path, decision in decisions.items()
                    if decision.get("model")
                }
                decisions, pending, extra_normalized = dual_review(
                    candidates,
                    target,
                    args,
                    client,
                    log,
                    out_dir,
                    review_paths=new_tables,
                    known_reviews=known,
                    write=True,
                    target_decision=decision,
                )
                normalized.update(extra_normalized)
    # Written after the gold choice so the trace records it: the trace is the
    # record of every decision the retrieval step made, and "which table we
    # trained on" is one of them.
    trace_path.write_text(json.dumps(trace, indent=2, default=str), encoding="utf-8")
    print(f"\n# search trace: {trace_path}")
    # Always pass the staging root: it is what lets the plan attach each
    # table's provenance (reference, sha256, retrieved_at) from the manifests.
    plan_args = datafetcher_cli(
        "--root",
        staging_root,
        "plan",
        *candidates,
        *primary_labeled_args,
        *plan_reviews_args,
        "--target-column",
        target,
        "--task-type",
        str(decision.get("task_type") or "classification"),
        "--target-source",
        plan_source(decision["source"], list(args.hint or [])),
        "--target-confidence",
        decision["confidence"],
        "--objective",
        args.objective,
        "--search-trace",
        str(trace_path),
        "-o",
        str(plan_path),
    )
    for flag, values in (
        ("--feature-prefix", args.feature_prefix),
        ("--exclude-column", args.exclude_col),
    ):
        for value in values or []:
            plan_args += [flag, value]
    if args.sample_id_column:
        plan_args += ["--sample-id-column", args.sample_id_column]
    run_command(plan_args)

    plan = json.loads(plan_path.read_text())
    log.event(
        "plan",
        f"\n# plan: {len(plan['gold'])} gold, {len(plan['supplementary'])} supplementary, "
        f"{len(plan['unusable'])} unusable -> {plan_path}"
        + "".join(f"\n#   note: {note}" for note in (plan.get("notes") or [])[:6]),
        path=str(plan_path),
        gold=[entry["path"] for entry in plan["gold"]],
        supplementary=[entry["path"] for entry in plan["supplementary"]],
        unusable=[
            {"path": entry["path"], "reason": entry["reason"]} for entry in plan["unusable"]
        ],
        notes=plan.get("notes", []),
    )
    if not plan["gold"]:
        log.event("plan", "# no gold table in the plan; nothing to train on", ok=False)
        # The report is written before anything gives up: a run that found
        # nothing is the one whose data report actually gets read.
        _write_data_report(plan, decisions, pending, args, log, plan_path)
        _print_rescue(plan, decisions, args, log, plan_path)
        return 2

    plan_task = plan.get("task") or {}
    plan_type = str(plan_task.get("task_type") or "classification")
    _write_data_report(plan, decisions, pending, args, log, plan_path)
    log.event(
        "plan",
        f"# task type: {plan_type}"
        + (
            f" ({len(plan['supplementary'])} supplementary table(s): signed PPI "
            f"correction; none: plain supervised training)"
            if plan_type == "regression"
            else ""
        ),
        task_type=plan_type,
    )
    # Say up front that this task will be preprocessed as single-cell counts, so
    # "the gene values were transformed" is not something the run reveals only in
    # its artifacts. The check is the same one the training makes.
    try:
        from kosmos.ppi.singlecell import looks_like_counts
        from kosmos.ppi.tabular import read_table as _read

        gold_path = str(plan["gold"][0]["path"])
        head = _read(gold_path, nrows=200)
        features = [c for c in head.columns if c != plan_task.get("target_column")]
        if looks_like_counts(head, features):
            log.event(
                "plan",
                "# single-cell: the labeled table is a count matrix, so each "
                "source will be preprocessed on its own (HVG -> normalize -> "
                "log1p -> per-gene z-score -> clip at 0). The run writes "
                "preprocessing.md + preprocessing.json + figures/ beside its "
                "summary.",
                single_cell=True,
            )
    except Exception as e:  # noqa: BLE001 - a notice, never a failure
        log.event("plan", f"# single-cell probe skipped: {type(e).__name__}: {e}")

    command = [
        PYTHON,
        "-m",
        "kosmos.cli.main",
        "run",
        args.objective,
        "--domain",
        args.domain,
        "--max-iterations",
        str(args.max_iterations),
        "--budget",
        str(args.budget),
        "--data-plan",
        str(plan_path),
        "--yes",
    ]
    if args.test_path:
        command += ["--task-test-path", args.test_path]
    printable = " ".join(shlex.quote(part) for part in command)
    if not args.yes:
        log.event("train", "\n# would run:\n#   " + printable)
        return 0
    if not args.run:
        log.event("train", "\n# run this to train:\n#   " + printable)
        return 0

    # Everything a run produces belongs in the run's own directory. `run.py`
    # points PPI_OUTPUT_DIR at `<out_dir>/run`; this script drives the CLI
    # directly, so without this the training artifacts land in the shared
    # `artifacts/ppi/research-*` default and the run looks like it never trained.
    # An exported PPI_OUTPUT_DIR still wins, so a caller can split them on
    # purpose; otherwise the attempt gets a fresh directory (`run`, `run-2`, ...)
    # because the engine refuses to write over a finished run.
    environment = dict(os.environ)
    from kosmos.ppi.flow import next_output_dir

    run_dir = (
        Path(EXPORTED_ENV["PPI_OUTPUT_DIR"])
        if EXPORTED_ENV.get("PPI_OUTPUT_DIR")
        else next_output_dir(plan_path.parent / "run")
    )
    environment["PPI_OUTPUT_DIR"] = EXPORTED_ENV.get("PPI_OUTPUT_DIR", str(run_dir))
    if plan_type == "regression":
        # A regression run uses the signed PPI correction whenever there is
        # something to correct with. With no supplementary table the correction
        # is empty and the run is plain supervised training either way, so the
        # default does not need a second switch. An exported PPI_LOSS_MODE still
        # wins: the caller may have a reason to want distillation.
        environment["PPI_LOSS_MODE"] = EXPORTED_ENV.get("PPI_LOSS_MODE", "signed")
        environment.setdefault("PPI_PSEUDO_TARGETS", "hard")
    log.event(
        "train",
        f"\n# launching:\n#   {printable}\n# records:\n#   {run_dir}",
        command=command,
        output_dir=str(run_dir),
    )
    code = subprocess.call(command, cwd=ROOT, env=environment)
    if code != 0:
        log.event("train", f"\n# pipeline stopped with exit code {code}", ok=False)
        return code

    log.event("done", f"\n# records: {run_dir / 'summary.md'}")
    subprocess.call([PYTHON, str(ROOT / "scripts" / "show_run.py"), str(run_dir)], cwd=ROOT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
