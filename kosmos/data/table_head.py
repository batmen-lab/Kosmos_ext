"""Read the head of a delimited file and report what its header REALLY is.

Kosmos meets every dataset through `pd.read_csv(path)`, which takes line 1 as
the header and everything below it as data. That is right for most files and
silently wrong for the ones that cost this project a run:

  * a metabolomics export whose SECOND row is a per-sample condition row
    (`Group,control,control,hypothermia,...`). pandas reads it as data, so
    every numeric column becomes an object column, no statistic is computable,
    and the first real analyte silently becomes row 1 -- which is exactly the
    row that run went on to test, instead of the molecules its question was
    about;
  * a table with a banner or a blank line above the header;
  * a matrix with no header at all.

The failure mode matters as much as the failure: pandas does not raise, and
the column list it produces looks entirely plausible. Nothing downstream can
detect it. So this module looks at the raw bytes first, decides which line is
the header, and returns BOTH a human/model-readable rendering of what it saw
and the `pd.read_csv` keyword arguments that read the file correctly.

Two design rules, both learned the hard way:

  * `sniff_head` NEVER raises. A dataset that cannot be sniffed must still be
    summarisable -- losing the whole run because a preview failed would be a
    worse bug than the one this fixes. Every failure becomes a verdict.
  * The preview is the raw text, before pandas applies type inference, quote
    handling and NA coercion. Showing a model the parsed frame cannot reveal a
    mis-parse; showing it the bytes can.
"""

from __future__ import annotations

import csv
import io
import logging
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Ten lines is what the operator asked for and roughly what the problem needs:
# a second header row sits at line 2, a banner at lines 1-3, and a numeric
# contrast ("is there a number under this name?") needs a few data rows below
# the candidate header. More lines would cost prompt budget without deciding
# anything the first ten do not.
HEAD_PREVIEW_LINES = 10

# Bound the READ, not just the line count: a file with no newline in its first
# gigabyte would otherwise be pulled into memory whole by `splitlines()`.
_HEAD_PREVIEW_BYTES = 262_144

# A 30,000-column pseudobulk line must not be pasted into a prompt verbatim.
_PREVIEW_FIELDS = 12
_PREVIEW_FIELD_CHARS = 40

# Candidate delimiters, in the order csv.Sniffer should consider them.
_DELIMITERS = ",\t;|"

# A second header row is a real pattern; a stack of six is a misdetection.
_MAX_ANNOTATION_ROWS = 3

_NA_TOKENS = {"", "na", "n/a", "nan", "null", "none", "-", "."}


def _is_number(cell: str) -> bool:
    """Does this cell parse as a number? Blanks and NA tokens do not count."""
    text = (cell or "").strip()
    if text.lower() in _NA_TOKENS:
        return False
    try:
        float(text.replace(",", "") if text.count(",") and " " not in text else text)
        return True
    except ValueError:
        return False


def _is_na(cell: str) -> bool:
    """Is this cell a blank or a missing-value placeholder?"""
    return (cell or "").strip().lower() in _NA_TOKENS


def _looks_like_names(cells: List[str]) -> bool:
    """Is this row a row of NAMES rather than of values?

    Two conditions, and the numeric one carries the weight: a header cell is
    never a number, and a row that is mostly blank is padding rather than a
    header.
    """
    if not cells:
        return False
    if any(_is_number(c) for c in cells):
        return False
    filled = sum(1 for c in cells if (c or "").strip())
    return filled >= max(1, int(0.7 * len(cells)))


def _all_numeric(cells: List[str]) -> bool:
    """Every cell in this row is a number (so it cannot be a header)."""
    return bool(cells) and all(_is_number(c) for c in cells)


def _is_annotation_row(candidate: List[str], below: List[List[str]]) -> bool:
    """Is `candidate` a second header row rather than the first data row?

    The evidence required is COLUMN-WISE, and that is the whole point. An
    earlier version asked only "is there a number anywhere below?", which
    deleted real data: `S1,control,NA` is 100% filled and contains no
    parseable number (NA does not count), so it read as a row of names, and
    the `5` sitting in a different column two rows down supplied the
    "contrast". The sample vanished from the frame with no error.

    So: some column must hold a real word in the candidate and a real number
    in EVERY row below it. Requiring every row below -- not merely one --
    keeps an all-text table safe (`gene,note / BRCA1,hello / TP53,3 /
    EGFR,world` has one number under `note` and is data, not a header), and
    excluding NA tokens from the candidate keeps a placeholder from
    masquerading as a name. Under-detecting only leaves pandas where it
    already was; over-detecting silently drops a row.
    """
    if not below:
        return False
    for j, cell in enumerate(candidate):
        text = (cell or "").strip()
        if not text or _is_na(text) or _is_number(text):
            continue
        column = [row[j] for row in below if j < len(row)]
        if column and all(_is_number(c) for c in column):
            return True
    return False


@dataclass
class HeadReport:
    """What the first few lines of one file actually contain.

    `skip_lines` holds RAW FILE LINE INDICES (0-based), because that is what
    `pd.read_csv(skiprows=...)` counts -- blank lines included. The parsed rows
    this module reasons over have blanks removed, so the two indexings differ
    and conflating them drops the wrong row.
    """

    path: str
    verdict: str  # plain | offset | headerless | empty | binary | unreadable
    header_line: Optional[int] = None
    skip_lines: List[int] = field(default_factory=list)
    delimiter: Optional[str] = ","
    names: List[str] = field(default_factory=list)
    annotation_rows: List[int] = field(default_factory=list)
    preview_lines: List[str] = field(default_factory=list)
    widths: List[int] = field(default_factory=list)
    duplicate_names: List[str] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)

    @property
    def is_plain(self) -> bool:
        """True when a bare `pd.read_csv(path)` reads this file correctly."""
        return self.verdict == "plain"

    def pandas_kwargs(self) -> Dict[str, Any]:
        """Keyword arguments that read this file correctly.

        `skiprows` + `header=0` rather than `header=N`: `header=N` counts rows
        AFTER pandas has already skipped blank lines, so a blank line above the
        header shifts it and the frame comes back with a data row as its
        column names. `skiprows` counts raw lines, which is what was measured.
        """
        if self.verdict in ("plain", "empty", "binary", "unreadable"):
            return {}
        kwargs: Dict[str, Any] = {}
        if self.delimiter is None:
            kwargs["sep"] = r"\s+"
        elif self.delimiter != ",":
            kwargs["sep"] = self.delimiter
        if self.skip_lines:
            kwargs["skiprows"] = sorted(self.skip_lines)
        if self.verdict == "headerless":
            # header=None AND the skiprows: the preamble above a headerless
            # matrix is narrower than the matrix, so omitting skiprows made the
            # prescribed call raise ParserError ("Expected 1 fields in line 2,
            # saw 3") on the very file the report had just described. A wider
            # banner was worse: no error, and the banner arrived as data row 0.
            kwargs["header"] = None
        elif self.skip_lines:
            kwargs["header"] = 0
        return kwargs

    def read_call(self, var: str = "path") -> str:
        """The literal `pd.read_csv(...)` call for this file, for a prompt."""
        kwargs = self.pandas_kwargs()
        if not kwargs:
            return f"pd.read_csv({var})"
        parts = []
        for key in ("sep", "skiprows", "header"):  # stable, readable order
            if key in kwargs:
                parts.append(f"{key}={kwargs[key]!r}")
        return f"pd.read_csv({var}, " + ", ".join(parts) + ")"

    def advisory(self) -> str:
        """One paragraph on what is wrong with this file, or ''.

        Separated from `render()` because a gated capsule may state the verdict
        without showing the rows: the fact that a header is displaced is a
        property of the file's shape, not a disclosure of anybody's data.
        """
        if self.verdict == "plain":
            return ""
        if self.verdict == "headerless":
            return (
                "This file has NO header row: every line is data. Read it with "
                "header=None; its columns are then the INTEGERS 0, 1, 2 ... , so "
                "refer to them as df[0], df[1] and never by a name."
            )
        if self.verdict == "empty":
            return "This file is empty."
        if self.verdict == "binary":
            return (
                "This file is not text. It was staged under a text extension "
                "but its first bytes are binary, so no column list can be read "
                "from it."
            )
        if self.verdict == "unreadable":
            return "The first lines of this file could not be read: " + "; ".join(
                self.notes or ["unknown reason"]
            )
        bits = []
        if self.annotation_rows:
            rows = ", ".join(str(i + 1) for i in self.annotation_rows)
            bits.append(
                f"line {rows} sits between the header and the data and is NOT a "
                f"data row"
                if len(self.annotation_rows) == 1
                else f"lines {rows} sit between the header and the data and are "
                f"NOT data rows"
            )
        ambiguous = [n for n in self.notes if n.startswith("more than one line")]
        preamble = [i for i in self.skip_lines if i not in self.annotation_rows]
        if preamble:
            rows = ", ".join(str(i + 1) for i in preamble)
            bits.append(f"line {rows} is preamble above the header")
        if self.header_line is not None and self.header_line > 0 and not bits:
            bits.append(f"the header is on line {self.header_line + 1}, not line 1")
        if not bits:
            # Reaching here means nothing is displaced: the file is "offset"
            # only because of its separator, or because the rows in the head
            # differ in width. Saying "the header is not on line 1" of a
            # perfectly ordinary TSV was simply false, and it told the model to
            # go looking for a displacement that is not there.
            if self.delimiter is None:
                return (
                    "This file is whitespace-separated, not comma-separated. "
                    "A bare pd.read_csv reads each line as a single column."
                )
            if self.delimiter and self.delimiter != ",":
                name = {"\t": "tab", ";": "semicolon", "|": "pipe"}.get(
                    self.delimiter, repr(self.delimiter)
                )
                return (
                    f"This file is {name}-separated, not comma-separated. Its "
                    f"header is on line 1; only the separator differs from the "
                    f"pd.read_csv default."
                )
            return (
                "The rows in the head of this file do not all have the same "
                "number of fields, so check the shape after reading."
            )
        detail = "; ".join(bits)
        text = (
            f"This file does NOT parse correctly with a bare pd.read_csv: "
            f"{detail}. Reading it that way turns numeric columns into text and "
            f"consumes a real row as data."
        )
        if ambiguous:
            text += " Check the head above before trusting the column names: " + ambiguous[0] + "."
        return text

    def render(self) -> str:
        """The raw head plus the verdict, for a prompt. '' when unusable.

        Wording here is constrained by existing assertions elsewhere in the
        suite (no ' -- ', no '  - ' bullets, and 'fields' rather than
        'columns' when truncating) so that adding this block cannot silently
        change what those tests were written to guard.
        """
        if self.verdict in ("empty", "unreadable"):
            return ""
        if self.verdict == "binary":
            return self.advisory()
        if not self.preview_lines:
            return ""
        out = [
            f"HEAD OF FILE ({len(self.preview_lines)} raw lines, exactly as "
            f"stored, before pandas interprets anything):"
        ]
        out += [f"    | {line}" for line in self.preview_lines]
        if self.verdict == "plain":
            out.append(
                "Header: line 1. Column names and types below are read from "
                "that line."
            )
            return "\n".join(out)
        out.append(self.advisory())
        out.append(f"READ THIS FILE WITH: {self.read_call()}")
        out.append(
            "The column names and types listed below come from THAT call, not "
            "from a plain pd.read_csv."
        )
        if self.duplicate_names:
            out.append(
                "Repeated column names in the header: "
                + ", ".join(self.duplicate_names[:6])
                + ". pandas will suffix them on read."
            )
        return "\n".join(out)


def _truncate(line: str, delimiter: Optional[str] = ",") -> str:
    """One preview line, bounded in both field count and field width.

    Splits on the file's ACTUAL delimiter. Hard-coding "," cut TSV header lines
    at 40 characters and then labelled the result "(+-11 more fields)" -- a
    negative count, because a tab-separated line has one comma-field, not
    twelve.
    """
    sep = delimiter if delimiter else None
    cells = line.split(sep) if sep is None or sep in line else [line]
    joiner = (sep or " ")
    if len(cells) <= _PREVIEW_FIELDS and len(line) <= _PREVIEW_FIELDS * _PREVIEW_FIELD_CHARS:
        return line if len(line) <= 600 else line[:600] + " ... (line truncated)"
    kept = [c[:_PREVIEW_FIELD_CHARS] for c in cells[:_PREVIEW_FIELDS]]
    hidden = max(0, len(cells) - _PREVIEW_FIELDS)
    return joiner.join(kept) + f" ... (+{hidden} more fields)"


def _detect_delimiter(lines: List[str]) -> Optional[str]:
    """The field separator, or None meaning whitespace-separated.

    csv.Sniffer first because it understands quoting; a plain character count
    second because Sniffer refuses single-column files; None last, which the
    caller renders as `sep=r'\\s+'`. Never `sep=None`: that forces pandas onto
    the python engine, which emits a ParserWarning, which this repo's
    `filterwarnings = error` turns into a failed run.
    """
    sample = "\n".join(lines[:5])
    try:
        return csv.Sniffer().sniff(sample, delimiters=_DELIMITERS).delimiter
    except (csv.Error, UnicodeDecodeError):
        pass
    counts = {d: sum(line.count(d) for line in lines[:5]) for d in _DELIMITERS}
    best = max(counts, key=lambda d: counts[d])
    if counts[best] > 0:
        return best
    # Whitespace only when whitespace actually yields a TABLE: a consistent
    # width greater than one. Asking merely "does some line contain a space?"
    # shredded a single column of free text -- a column of descriptions became
    # a four-column frame whose header was the first sentence's first word.
    ws_widths = [len(line.split()) for line in lines[:5] if line.strip()]
    if ws_widths:
        modal = max(set(ws_widths), key=ws_widths.count)
        if modal > 1 and ws_widths.count(modal) >= max(2, len(ws_widths) - 1):
            return None
    return ","


# Compressions pandas reads straight through (`pd.read_csv` infers them from
# the extension). Their first bytes are binary, so without this a .csv.gz was
# reported as "not text ... no column list can be read from it" -- a file
# pandas opens perfectly and that this project stages routinely.
_COMPRESSED_OPENERS = {".gz": "gzip", ".bz2": "bz2", ".xz": "lzma", ".zst": None}


def _open_maybe_compressed(path: str):
    """Open `path` for binary reading, transparently decompressing if needed."""
    suffix = os.path.splitext(path)[1].lower()
    module = _COMPRESSED_OPENERS.get(suffix)
    if module:
        import importlib

        return importlib.import_module(module).open(path, "rb")
    return open(path, "rb")


def sniff_head(path: str, max_lines: int = HEAD_PREVIEW_LINES) -> HeadReport:
    """Read the first lines of `path` and report where its header really is.

    Never raises. Every failure -- missing file, permission, binary content,
    an encoding this cannot decode -- comes back as a verdict with the reason
    in `.notes`, because the caller is building a prompt and a missing preview
    must not cost it the whole summary.
    """
    report = HeadReport(path=path, verdict="unreadable")
    try:
        with _open_maybe_compressed(path) as fh:
            raw = fh.read(_HEAD_PREVIEW_BYTES)
    except OSError as e:
        report.notes.append(str(e))
        return report
    except Exception as e:
        # A corrupt or truncated archive. Still a verdict, never a raise.
        report.notes.append(f"could not decompress: {e}")
        return report

    if not raw.strip():
        report.verdict = "empty"
        return report
    # A NUL in the first bytes means this is not text, whatever the extension
    # claims -- an .h5ad or .xlsx staged under a .csv name lands here.
    if b"\x00" in raw[:8192]:
        report.verdict = "binary"
        return report

    try:
        text = raw.decode("utf-8-sig", errors="replace")
    except Exception as e:  # pragma: no cover - decode with errors= cannot raise
        report.notes.append(f"decode failed: {e}")
        return report

    all_lines = text.splitlines()
    # A 30,000-column file hits the byte cap mid-line, and that fragment is
    # narrower than the rows above it. Counted as a row, it made the widths
    # disagree, which demoted an ordinary plain CSV to "offset" and attached a
    # confident, false advisory to it. A partial line is not a row.
    if len(raw) >= _HEAD_PREVIEW_BYTES and not text.endswith(("\n", "\r")) and len(all_lines) > 1:
        all_lines = all_lines[:-1]
    raw_lines = all_lines[:max_lines]
    if not raw_lines:
        report.verdict = "empty"
        return report

    # Keep the RAW index of every non-blank line. skiprows counts blanks; the
    # classification below does not, and conflating the two skips a real row.
    indexed = [(i, line) for i, line in enumerate(raw_lines) if line.strip()]
    if not indexed:
        report.verdict = "empty"
        return report

    delimiter = _detect_delimiter([line for _, line in indexed])
    report.delimiter = delimiter
    report.preview_lines = [_truncate(line, delimiter) for line in raw_lines]

    def _split(line: str) -> List[str]:
        if delimiter is None:
            return line.split()
        try:
            return next(csv.reader(io.StringIO(line), delimiter=delimiter), [])
        except csv.Error:
            # csv.reader refuses a field over 128 KB (csv.field_size_limit).
            # One giant cell -- a pasted sequence, a JSON blob in a column --
            # used to make the whole file "unreadable" with no preview at all.
            # A naive split loses quoting, which is a far smaller loss.
            return line.split(delimiter)

    try:
        rows = [(i, _split(line)) for i, line in indexed]
    except Exception as e:
        report.notes.append(f"could not split lines: {e}")
        return report

    widths = [len(cells) for _, cells in rows]
    report.widths = widths
    modal = max(set(widths), key=widths.count)

    # Lines narrower than the table are preamble (a banner, a title, a
    # provenance line); they sit above the header and must be skipped.
    body = [(i, cells) for i, cells in rows if len(cells) == modal]
    if not body:
        report.notes.append("no consistent row width in the head")
        return report
    first_body_raw = body[0][0]

    header_idx = None
    for pos, (raw_i, cells) in enumerate(body):
        if _looks_like_names(cells):
            header_idx = pos
            break
    if header_idx is None:
        # No row of names. That does NOT mean the file has no header: a
        # time-course matrix header is `gene,0,6,24`, whose numeric cells make
        # it fail _looks_like_names while pandas reads it perfectly. Claiming
        # headerless there destroyed a file that already parsed correctly.
        # Only an entirely numeric first row is positive evidence of no
        # header; otherwise defer to the convention pandas already follows.
        if _all_numeric(body[0][1]):
            report.verdict = "headerless"
            report.skip_lines = sorted(range(first_body_raw))
            report.notes.append("every row in the head is numeric")
            return report
        header_idx = 0

    header_raw, header_cells = body[header_idx]
    # EVERY raw line above the header is preamble, blank lines included.
    # Computing this from the first BODY row instead left the rows between the
    # first body row and a later header unskipped, so the prescribed call read
    # a banner as the header while the advisory said the header was elsewhere.
    preamble = list(range(header_raw))
    report.header_line = header_raw
    report.names = [c.strip() for c in header_cells]
    seen, dupes = set(), []
    for name in report.names:
        if name and name in seen:
            dupes.append(name)
        seen.add(name)
    report.duplicate_names = dupes

    # A row of names directly under the header is a SECOND header only if some
    # column turns numeric below it. Without that contrast a genuinely
    # all-text table ("name,note / alice,hello") would have its first data row
    # deleted -- a silent corruption worse than the bug being fixed.
    annotation: List[int] = []
    pos = header_idx + 1
    while pos < len(body) and len(annotation) < _MAX_ANNOTATION_ROWS:
        raw_i, cells = body[pos]
        if not _looks_like_names(cells):
            break
        below = [c for _, c in body[pos + 1 : pos + 4]]
        if not _is_annotation_row(cells, below):
            break
        annotation.append(raw_i)
        pos += 1

    report.annotation_rows = annotation
    if annotation and header_idx > 0:
        # Two plausible header lines: one was chosen and one is being skipped
        # as an annotation row, and which is which is genuinely ambiguous when
        # a metadata block happens to be as wide as the table. Say so rather
        # than assert it: a wrong header asserted confidently is the exact
        # failure this module exists to prevent.
        alt = body[header_idx + len(annotation)][0] + 1
        report.notes.append(
            f"more than one line in the head could be the header; line "
            f"{header_raw + 1} was chosen over line {alt}"
        )
    skip = sorted(set(preamble) | set(annotation))
    report.skip_lines = skip
    if len(set(widths)) > 1:
        report.notes.append("rows in the head do not all have the same width")

    plain = (
        not skip
        and header_raw == 0
        and delimiter == ","
        and not report.notes
    )
    report.verdict = "plain" if plain else "offset"
    return report
