"""``geo://`` -- an NCBI GEO series, fetched to a local file.

URL shape and the two traps are taken from AutoEvidence's `sources/geo.py`,
which was written for exactly this problem:

  * a series publishes the same data in two places -- a **series matrix** (the
    expression table between `!series_matrix_table_begin` and
    `!series_matrix_table_end`) and **supplementary files** under `suppl/`. An
    RNA-seq series commonly has an EMPTY matrix and its counts in `suppl/`;
    landing that empty table would read downstream as "a dataset with no rows",
    so it refuses and names the supplementary files instead;
  * a multi-platform series has no `<GSE>_series_matrix.txt.gz` at all, but one
    matrix per platform. Picking one silently would compare measurements on
    different probe sets, so it refuses and lists them.

GEO has no revisions: a series can be updated in place under the same
accession, so the identity of these bytes is the sha256 in the manifest.
"""

from __future__ import annotations

import csv
import gzip
import re
from pathlib import Path

from ..config import DataFetcherConfig
from ..errors import SourceError
from ..references import Reference
from ..store import stage_dir
from .base import Resolved, name_for, register
from .net import download, get_bytes, get_prefix

_ACCESSION = re.compile(r"^GSE\d+$", re.IGNORECASE)
_HREF = re.compile(r'href="([^"?/][^"]*)"')
_TABLE_BEGIN = "!series_matrix_table_begin"
_TABLE_END = "!series_matrix_table_end"
#: NCBI's directory index prints `Name  Last modified  Size` inside a <pre>.
_SIZE = re.compile(r"^\d+(?:\.\d+)?[KMGTP]$", re.IGNORECASE)
#: The same token anywhere after a link: NCBI prints `name  date  size`, and the
#: page's own markup (`<hr></pre>`) follows the last row.
_SIZE_IN_TEXT = re.compile(r"\b\d+(?:\.\d+)?[KMGTP]\b", re.IGNORECASE)
_UNITS = {"K": 1024, "M": 1024**2, "G": 1024**3, "T": 1024**4, "P": 1024**5}
_SERIES_FIELD = re.compile(r"^!Series_(title|summary|overall_design)\s*=\s*(.*)$")

_NO_REVISION = (
    "GEO has no revision id: a series is updated in place under the same "
    "accession, so pin the sha256 from this manifest instead"
)


def _series_dir(accession: str) -> str:
    """`GSE55296` -> `GSE55nnn`, NCBI's own grouping rule."""
    return accession[:-3] + "nnn" if len(accession) > 6 else accession


def _url(config: DataFetcherConfig, accession: str, *parts: str) -> str:
    tail = "/".join(parts)
    return f"{config.geo_ftp_root}/{_series_dir(accession)}/{accession}/{tail}"


def _list(config: DataFetcherConfig, accession: str, which: str) -> list[str]:
    return [entry["name"] for entry in _list_detailed(config, accession, which)]


def parse_size(text: str) -> int | None:
    """`587M` -> 615,514,112, the way NCBI's own index prints it."""
    text = (text or "").strip()
    if not _SIZE.match(text):
        return None
    return int(float(text[:-1]) * _UNITS[text[-1].upper()])


def format_size(size: int | None) -> str:
    if not size:
        return "size unknown"
    for unit, scale in (("GB", 1024**3), ("MB", 1024**2), ("kB", 1024)):
        if size >= scale:
            return f"{size / scale:.1f} {unit}"
    return f"{size} bytes"


def _list_detailed(
    config: DataFetcherConfig, accession: str, which: str
) -> list[dict[str, object]]:
    """The series' files of one kind, with the size the index already states.

    The size is what decides which artifact to download: a series in `suppl/`
    commonly offers the same data as a 0.6 GB and a 2.7 GB file, and the listing
    that names them is the only place that says so before the transfer starts.
    """
    try:
        html = get_bytes(_url(config, accession, which, ""), config).decode(
            "utf-8", "replace"
        )
    except SourceError:
        return []
    entries: list[dict[str, object]] = []
    matches = list(_HREF.finditer(html))
    for position, match in enumerate(matches):
        name = match.group(1)
        if name.startswith("http") or name.endswith("/"):
            continue
        if which == "matrix" and not name.endswith("_series_matrix.txt.gz"):
            continue
        if which == "suppl" and name in {".", ".."}:
            continue
        # The size is printed after the link, before the next one starts. An
        # index that puts several links on one line still parses.
        end = matches[position + 1].start() if position + 1 < len(matches) else len(html)
        tail = html[match.end() : end].replace("</a>", " ")
        found = _SIZE_IN_TEXT.findall(tail)
        size = parse_size(found[-1]) if found else None
        entries.append({"name": name, "bytes": size, "size": format_size(size)})
    return entries


def list_matrix_files(accession: str, config: DataFetcherConfig) -> list[str]:
    return _list(config, accession, "matrix")


def list_supplementary(accession: str, config: DataFetcherConfig) -> list[str]:
    return _list(config, accession, "suppl")


def list_files_detailed(
    accession: str, config: DataFetcherConfig
) -> list[dict[str, object]]:
    """Every `matrix/` and `suppl/` file of the series, smallest first.

    Smallest first is the order a caller wants: the question decides what is
    usable, and among the usable artifacts the smallest one is the cheapest.
    """
    entries = [
        {**entry, "path": f"{which}/{entry['name']}"}
        for which in ("matrix", "suppl")
        for entry in _list_detailed(config, accession, which)
    ]
    return sorted(entries, key=lambda entry: entry["bytes"] or 10**18)


def describe_files(
    accession: str, config: DataFetcherConfig, *, smallest: int = 6
) -> str:
    """The series' files with sizes, for a message that asks "which one?"."""
    entries = list_files_detailed(accession, config)
    if not entries:
        return ""
    head = entries[:smallest]
    text = ", ".join(f"{entry['path']} ({entry['size']})" for entry in head)
    if len(entries) > len(head):
        text += f", ... ({len(entries) - len(head)} more)"
    return text


def series_about(accession: str, config: DataFetcherConfig, *, limit: int = 65536) -> str:
    """Title and summary of the series, from NCBI's small `brief` record.

    This is the cheap half of "look before you download": a few kilobytes say
    what the series *is*, so an irrelevant one is dropped before its gigabytes
    move. Failure is not an error -- a run without the summary still runs.
    """
    url = f"{config.geo_query_root}?acc={accession}&targ=self&form=text&view=brief"
    try:
        # A prefix, not the whole record: a series with many samples has a long
        # brief, and the title and summary are at its start.
        text = get_prefix(url, config, limit=limit).decode("utf-8", "replace")
    except SourceError:
        return ""
    parts: list[str] = []
    for line in text.splitlines():
        match = _SERIES_FIELD.match(line.strip())
        if match:
            parts.append(f"{match.group(1)}: {match.group(2).strip()}")
    return " | ".join(parts)[:1200]


def _read_table(path: Path, limit: int = 1 << 22) -> str:
    """The series matrix text, decompressed if needed, bounded."""
    blob = path.read_bytes()
    if blob[:2] == b"\x1f\x8b":
        try:
            blob = gzip.decompress(blob)
        except OSError as e:
            raise SourceError(f"{path} is not readable as gzip: {e}") from e
    return blob[: limit + 1].decode("utf-8", "replace")


def matrix_has_rows(text: str) -> bool:
    """True when the series matrix actually carries a table."""
    if _TABLE_BEGIN not in text:
        return False
    body = text.split(_TABLE_BEGIN, 1)[1]
    body = body.split(_TABLE_END, 1)[0]
    rows = [line for line in body.splitlines() if line.strip()]
    return len(rows) > 1  # the first non-empty line is the header


def parse_series_matrix(text: str) -> tuple[list[str], list[str], list[list[str]]]:
    """Return `(sample_ids, feature_ids, values)` from a series matrix.

    Sample order comes from `!Sample_geo_accession`, which is the one place GEO
    states it; the table's own header repeats the same ids and is used only as a
    check. Values stay strings -- this is a fetcher, not a numerics layer.
    """
    samples: list[str] = []
    for line in text.splitlines():
        if line.startswith("!Sample_geo_accession"):
            samples = [cell.strip().strip('"') for cell in line.split("\t")[1:]]
            break
    if _TABLE_BEGIN not in text:
        raise SourceError(
            "this file has no !series_matrix_table_begin section, so there is "
            "no table to convert; use the supplementary files instead"
        )
    body = text.split(_TABLE_BEGIN, 1)[1].split(_TABLE_END, 1)[0]
    rows = [line for line in body.splitlines() if line.strip()]
    if len(rows) < 2:
        raise SourceError(
            "the series matrix table is empty; this series keeps its values in "
            "suppl/ -- name one with geo://<accession>#suppl/<filename>"
        )
    header = [cell.strip().strip('"') for cell in rows[0].split("\t")]
    features: list[str] = []
    values: list[list[str]] = []
    for row in rows[1:]:
        cells = [cell.strip().strip('"') for cell in row.split("\t")]
        features.append(cells[0])
        values.append(cells[1:])
    if samples and len(samples) != len(values[0]):
        # Header wins when the two disagree: it is the table's own ordering.
        samples = header[1:]
    if not samples:
        samples = header[1:]
    return samples, features, values


def write_samples_by_features_csv(
    text: str, out_path: Path, max_features: int | None = None
) -> tuple[int, int]:
    """Transpose a series matrix to `sample_id` x features, as CSV.

    GEO's matrix is features x samples; Kosmos reads tables the other way
    (one row per observation). `max_features` keeps a checkpointed choice in
    the caller's hands rather than truncating silently.
    """
    samples, features, values = parse_series_matrix(text)
    if max_features is not None and len(features) > max_features:
        keep = list(range(max_features))
    else:
        keep = list(range(len(features)))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["sample_id"] + [features[i] for i in keep])
        for column, sample in enumerate(samples):
            writer.writerow([sample] + [values[i][column] for i in keep])
    return len(samples), len(keep)


class GeoSource:
    scheme = "geo"

    def fetch(self, ref: Reference, config: DataFetcherConfig) -> Resolved:
        accession = ref.locator.strip().upper()
        if not _ACCESSION.match(accession):
            raise SourceError(
                f"geo:// expects a GEO SERIES accession such as geo://GSE55296; "
                f"got {ref.locator!r}. Sample (GSM), platform (GPL) and dataset "
                f"(GDS) accessions name different objects and are not handled."
            )
        if ref.revision:
            raise SourceError(f"geo://{accession}@{ref.revision}: {_NO_REVISION}")

        directory = stage_dir(config, self.scheme, name_for(ref))
        selector = (ref.selector or "").strip()
        if selector and not selector.startswith(("suppl/", "matrix/")):
            raise SourceError(
                f"geo:// selector must name a file as `#suppl/<filename>` or "
                f"`#matrix/<filename>`; got {selector!r}"
            )

        if selector:
            which, filename = selector.split("/", 1)
            if "/" in filename or ".." in filename:
                raise SourceError(f"geo:// selector {selector!r} is not a filename")
            available = (
                list_supplementary(accession, config)
                if which == "suppl"
                else list_matrix_files(accession, config)
            )
            if available and filename not in available:
                raise SourceError(
                    f"{accession} has no {which} file {filename!r}. Available: "
                    f"{describe_files(accession, config)}"
                )
        else:
            matrices = list_matrix_files(accession, config)
            if not matrices:
                listing = describe_files(accession, config)
                raise SourceError(
                    f"{accession} publishes no series matrix. "
                    + (
                        f"Supplementary files (smallest first): {listing} -- fetch "
                        f"one with geo://{accession}#suppl/<filename>; the smallest "
                        f"file that answers the question is the one to take"
                        if listing
                        else "It also has no supplementary listing; check the "
                        "accession at https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi"
                    )
                )
            if len(matrices) > 1:
                raise SourceError(
                    f"{accession} publishes {len(matrices)} series matrices, one "
                    f"per platform: {describe_files(accession, config)}. Pick one with "
                    f"geo://{accession}#matrix/<filename> -- merging them would "
                    f"compare measurements on different probe sets."
                )
            which, filename = "matrix", matrices[0]

        url = _url(config, accession, which, filename)
        target = directory / filename
        record = download(url, target, config)
        notes = [f"fetched {which} file {filename} for {accession}", _NO_REVISION]
        if which == "matrix" and not matrix_has_rows(_read_table(target)):
            listing = describe_files(accession, config)
            raise SourceError(
                f"{accession}'s series matrix has a header and no rows, which is "
                f"normal for RNA-seq series. "
                + (
                    f"Supplementary files (smallest first): {listing} -- fetch one "
                    f"with geo://{accession}#suppl/<filename>; the smallest file "
                    f"that answers the question is the one to take"
                    if listing
                    else "It has no supplementary listing either."
                )
            )
        return Resolved(
            scheme=self.scheme,
            locator=accession,
            directory=directory,
            files=[record],
            revision=None,
            notes=notes,
        )


register(GeoSource())
