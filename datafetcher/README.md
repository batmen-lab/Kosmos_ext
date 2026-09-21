# datafetcher

Fetch public datasets for Kosmos experiments. **Search finds pointers; fetch
downloads them and records where they came from.** That is the whole board.

No gating: no policies, no roles, no disclosure thresholds, no signed capsules,
no review queue, no banded statistics. Nothing here refuses a request for
privacy reasons; the only refusals are technical (unreachable host, over the
size cap, an ambiguous accession, an unrecognised upstream payload).

No coupling with Kosmos either: this package never imports `kosmos`, `kosmos`
never imports it, and it is deliberately outside `[tool.setuptools.packages.find]`
(`include = ["kosmos*"]`) so it is not part of the installed distribution. A
fetched file reaches a run the same way any other file does:

```bash
.venv/bin/python -m datafetcher fetch "geo://GSE55296#suppl/GSE55296_count_data.txt.gz"
# ... prints the staged path ...
.venv/bin/python -m kosmos.cli.main run "<objective>" \
    --data-path   <staged path>            # gold data
    --external-data-path <staged path>     # or PPI external evidence
```

## Usage

```bash
# Pointers only. Everything the search sent is in the ledger before it leaves.
.venv/bin/python -m datafetcher search "CITE-seq bone marrow mononuclear cells" -r omicsdi -n 5
.venv/bin/python -m datafetcher search "bmmc" -r huggingface --json

# Bytes. `geo://` and `hf://` need no credentials; `https://` and `file://` are
# the escape hatches for anything else a human already has a URL for.
.venv/bin/python -m datafetcher fetch geo://GSE55296
.venv/bin/python -m datafetcher fetch geo://GSE55296#suppl/GSE55296_count_data.txt.gz
.venv/bin/python -m datafetcher fetch hf://owner/name#data/train.csv
.venv/bin/python -m datafetcher fetch https://example.org/cohort.csv
.venv/bin/python -m datafetcher fetch file:///abs/path/local.h5ad

# What a reference holds and what it would cost, without downloading it: the
# size of every supplementary file, the repository's layout, and a one-line
# description from the series record or the dataset card.
.venv/bin/python -m datafetcher list-files geo://GSE194122

# GEO publishes matrices features x samples; --to-csv writes a derived
# samples x features table beside the raw download.
.venv/bin/python -m datafetcher fetch geo://GSE55296 --to-csv

# A reference whose bytes are already staged is reused (the ledger shows the
# same 616 MB file fetched nine times); --refresh downloads it again.
.venv/bin/python -m datafetcher fetch geo://GSE55296 --refresh

# One archive holds one triplet per sample: --archive-members bounds the unpack.
.venv/bin/python -m datafetcher fetch geo://GSE120221 --archive-members 6

.venv/bin/python -m datafetcher list            # what is already on disk
.venv/bin/python -m datafetcher ledger -n 20    # every search and fetch
```

Python API:

```python
from datafetcher import DataFetcherConfig, fetch, search

config = DataFetcherConfig(root="data/fetched", max_bytes=2 * 1024**3)
for hit in search("CITE-seq BMMC", config=config, repositories=["omicsdi"], limit=5).hits:
    print(hit.describe())
    if hit.reference:
        result = fetch(hit.reference, config=config, query="CITE-seq BMMC")
        print(result.directory, result.primary.sha256)
```

## Layout on disk

```
data/fetched/
├── ledger.jsonl                every search and fetch, appended
├── geo/GSE55296/
│   ├── GSE55296_count_data.txt.gz
│   └── manifest.json           url, size, sha256, revision, retrieved_at
└── hf/owner__name/
    └── manifest.json
```

Re-fetching a reference **reuses** the staged bytes when they are still there and
unchanged (recorded in the ledger as `fetch_reuse`); `--refresh` downloads again
and appends a second record, because the history of what was retrieved and when
is worth keeping. A failed download leaves no file behind: it streams into
`data/fetched/.partial/` and only lands on success.

## Where the ideas came from

Borrowed from AutoEvidence's `finder/` and `sources/` (see
`/home/ydong233/2026_yanglu_omics_os/src/AutoEvidence`), minus the gateway:

| Idea | Why it is kept |
|---|---|
| Search returns pointers, never data | A hit is an accession and a landing URL. Bytes need a second, explicit call. |
| One adapter per repository | Upstream envelopes differ unpredictably; an unrecognised payload is an error, never an empty result. |
| A failed repository is reported as a failure | Never as "no such dataset" — the two are different facts. |
| Ledger written before egress | An outbound query that cannot be attributed afterwards cannot be checked at all. |
| Accessions are validated, not sanitised | A rewritten accession is one this package invented for a dataset that does not have it. |
| Refuse rather than guess | Multi-platform series, empty RNA-seq matrices, unknown scheme, unknown repository label: each names the alternatives. |

Dropped on purpose: ToolUniverse (search is plain HTTP here, so no second
interpreter and no numpy/mcp version conflict), the MCP server, policies,
signing keys, the three gates, capsule banding, and every source that needs a
credential.

## Known behaviours worth knowing

* **GEO has no revisions.** A series can be updated in place under the same
  accession, so identity is the `sha256` in the manifest. `geo://GSE1@r2` is
  refused with that explanation rather than silently ignoring the revision.
* **An empty series matrix is a refusal, not a result.** RNA-seq series often
  keep their counts in `suppl/`; the error lists the available supplementary
  files instead of landing a table that reads downstream as "no rows".
* **A multi-platform series must be chosen explicitly.** `GSE2240` publishes one
  matrix per platform; merging them would compare different probe sets.
* **The Hub's search matches repository names and tags.** A multi-word intent
  often returns nothing, and the result says so rather than looking like an
  absence of data.
* **`--max-bytes` is enforced before and during the download** (default 2 GiB per
  artifact), so a wrong accession fails fast instead of filling the disk. A
  repository named without a file no longer arrives whole: the listing is read
  first and one bounded table is taken.
* **Sizes are read before bytes.** GEO's directory index prints a size beside
  every file, the Hub returns sizes with a listing, and `https://` answers a HEAD
  request; `list-files` shows them so the cheap artifact can be chosen.
* **Nothing already staged is downloaded twice.** A manifest record whose files
  are still on disk and unchanged is reused, verified by `sha256` (or by size
  above 512 MiB, which is where re-hashing costs more than it says).
* **An archive is unpacked to a bounded number of members** (default 12, and a
  byte budget): a 25-sample `RAW.tar` holds 76 files, and the note says how many
  were left inside for a caller who wants more.
* **Evidence is matched to the gold by intersection, not by identity.** A
  supplementary table does not need every measured column the gold has: what it
  shares is what the correction is trained on, and both arms then train on that
  same intersection, so "gold only" and "gold plus evidence" are compared on the
  same inputs. Two columns is the floor (`KOSMOS_MIN_SHARED_FEATURES`) and half
  of the gold's columns is the usual requirement
  (`KOSMOS_MIN_SHARED_FRACTION`) -- a 12-gene overlap with a 14,000-gene gold is
  a different measurement, not evidence. The plan records the shared columns per
  table, and the run's own report says which columns both arms used.

## Tests

```bash
.venv/bin/python -m pytest tests/datafetcher -o addopts="" -q
```

The suite never touches the network: `sources/net._open` is the single call site
for a request, and the tests replace it with a fake web, which also lets them
assert the exact URL a connector builds.
