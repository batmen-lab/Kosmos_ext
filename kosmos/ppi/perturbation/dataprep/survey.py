"""Step 1: survey the repositories for a perturbation question; download nothing.

    spec      what the data must be (cell type, organism, modality, topic, assay)
    search    anchored queries (topic + assay, cell type + assay, ...) for the gold
              and the supplementary pool; hits ranked by topic match, then by
              how many searches returned them
    describe  per candidate series: title/summary, every sample's title and
              characteristics, and every sample's own files with sizes
    judge     one model call per series, a fixed schema per sample (single cell?
              a screen? which modality, cell type, topic? labels available?)
    gate      rules on top of the judgement: a gold sample is a single-cell
              screen of the right cell type and modality with per-cell labels;
              a supplementary sample is single-cell data of the right cell type

Writes `survey.json` and `survey.md`: gold and supplementary candidates per
sample, with download size and an estimate of the cell count.

    .venv/bin/python -m kosmos.ppi.perturbation.dataprep.survey "<question>" --out DIR
"""

from __future__ import annotations

import argparse
import json
import re
import time
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .spec import ScreenSpec, ascii_terms, extract_spec, modality_compatible

#: Candidates described and judged per pool (gold, supplementary).
MAX_CANDIDATES = 10
#: The supplementary pool is the one that has to be large: more of it is judged.
MAX_SUPPLEMENTARY_CANDIDATES = 15
#: GEO full-text hits taken per query (sorted by date; ranking happens later).
GEO_FULL_TEXT_HITS = 30
#: Samples listed per series (each listing is one HTTP request).
MAX_SAMPLES = 24
#: Above this many "cells" a barcode file lists every droplet (unfiltered), not
#: cells (GSE249595: 18.8 MB per channel read as ~5M cells).
MAX_PLAUSIBLE_CELLS = 250_000
#: gzip'd 10x barcodes take ~3.8 bytes per cell (GSM2406681: 244 kB, 65,337 cells).
GZ_BYTES_PER_BARCODE = 3.8
PLAIN_BYTES_PER_BARCODE = 19.0

Log = Callable[[str], None]


@dataclass
class Candidate:
    reference: str
    title: str
    #: the series summary, when the search returned it (GEO full-text search)
    summary: str = ""
    queries: list[str] = field(default_factory=list)
    pools: set[str] = field(default_factory=set)
    on_topic: bool = False
    about: str = ""
    samples: list[dict[str, Any]] = field(default_factory=list)
    judgement: dict[str, Any] = field(default_factory=dict)


# -- search ---------------------------------------------------------------------


#: A cell name that is one of these words alone names no study: `neurons` turned
#: the supplementary search into HuggingFace repos about LLM "neurons".
GENERIC_CELL_WORDS = frozenset(
    "cell cells neuron neurons monocyte monocytes macrophage macrophages lymphocyte "
    "lymphocytes fibroblast fibroblasts tissue tissues".split()
)


def _generic(name: str) -> bool:
    words = [w for w in re.split(r"[\s\-]+", name.lower()) if w and w not in ("human", "mouse", "primary")]
    return not words or all(w in GENERIC_CELL_WORDS for w in words)


def _names_cells(title: str, name: str) -> bool:
    """Every specific word of a cell name is in the title (`ipsc` of `iPSC
    neurons`); a generic word alone (`neurons`) does not count."""
    from .scbasecount import name_keys

    keys = name_keys([name])
    specific = [w for w in (keys[0] if keys else []) if w not in GENERIC_CELL_WORDS]
    text = re.sub(r"[^a-z0-9]", "", str(title).lower())
    return bool(specific) and all(w in text for w in specific)


def specific_cell_terms(spec: ScreenSpec) -> list[str]:
    """The cell-type names that are not generic words alone."""
    return [name for name in spec.cell_type_terms() if not _generic(name)]


def short_cell_type(spec: ScreenSpec) -> str:
    """The briefest name for the cells: repository search matches every word, so
    `primary CD4+ T cells Perturb-seq` and `bone-marrow-derived dendritic cells
    Perturb-seq` missed the screens a shorter name finds."""
    names = [re.sub(r"\b(primary|human|mouse|murine)\b|\+", " ", name) for name in spec.cell_type_terms()]
    names = [" ".join(name.split()) for name in names if name.strip()]
    # an acronym (`BMDCs`) is not how titles name cells; `K562` (digits) is
    words = [name for name in names if not re.fullmatch(r"[A-Z]{2,}s?", name) and not _generic(name)] or names
    return min(words, key=lambda name: (len(name.split()), len(name))) if words else ""


def _queries(spec: ScreenSpec) -> dict[str, list[str]]:
    # both names: the short one finds `CD4 T cells`, the full one keeps `THP-1`
    # (the alias `THP1` missed every `THP-1` title)
    cells = list(
        dict.fromkeys(
            n for n in (ascii_terms(spec.cell_type), short_cell_type(spec)) if n and not _generic(n)
        )
    ) or [ascii_terms(spec.cell_type)]
    assays = spec.assays or ["Perturb-seq"]
    gold = []
    for topic in spec.topic_terms():
        gold += [f"{topic} {assays[0]}", f"{topic} CRISPR screen", f"{topic} {cells[0]}" if cells else topic]
    for cell in cells:
        gold += [f"{cell} {assay}" for assay in assays[:2]] + [f"{cell} CRISPR screen single-cell"]
        # without `single-cell`: GSE124703 is "CRISPR-based ... genetic screens in
        # human iPSC-derived neurons", which every query naming the assay missed
        gold += [f"{cell} CRISPR screen"]
        if spec.modality != "any":
            gold += [f"{cell} {spec.modality}"]
    supplementary = []
    for cell in cells:
        supplementary += [f"{cell} single-cell RNA-seq", f"{cell} scRNA-seq", f"{cell} {assays[0]}"]
    unique = lambda items: list(dict.fromkeys(" ".join(q.split()) for q in items if q.strip()))
    return {"gold": unique(gold), "supplementary": unique(supplementary)}


def _words(text: str) -> list[str]:
    # a hyphenated token is one word, hyphens dropped: `PD-L1` -> `pdl1`
    # (split, its parts were too short to count and the topic matched nothing)
    tokens = re.findall(r"[a-z0-9+]+(?:-[a-z0-9+]+)*", str(text).lower())
    return [word for word in (t.replace("-", "") for t in tokens) if len(word) > 2]


def _matches(title: str, phrase: str) -> bool:
    words = _words(phrase)
    text = str(title).lower().replace("-", "")
    hit = sum(word in text for word in words)
    return bool(words) and hit >= max(1, len(words) - 1)


def search(spec: ScreenSpec, *, run_query: Callable[[str], list[dict]], log: Log) -> list[Candidate]:
    """Anchored searches for both pools; the best candidates of each, ranked."""
    found: dict[str, Candidate] = {}
    for pool, queries in _queries(spec).items():
        for query in queries:
            hits = run_query(query)
            log(f"# survey: search [{pool}] {query!r}: {len(hits)} hit(s)")
            for hit in hits:
                reference = str(hit.get("reference") or "")
                if not reference.startswith(("geo://GSE", "hf://")):
                    continue
                candidate = found.setdefault(
                    reference, Candidate(reference=reference, title=str(hit.get("title") or ""))
                )
                candidate.summary = candidate.summary or str(hit.get("summary") or "")
                candidate.pools.add(pool)
                if query not in candidate.queries:
                    candidate.queries.append(query)
    cell_anchors = [ascii_terms(name) for name in specific_cell_terms(spec)] or [spec.cell_type]
    anchors = {"gold": spec.topic_terms() or cell_anchors, "supplementary": cell_anchors}
    chosen: dict[str, Candidate] = {}
    for pool in ("gold", "supplementary"):
        pool_candidates = [c for c in found.values() if pool in c.pools]
        hit = lambda c: any(_matches(c.title, anchor) for anchor in anchors[pool])
        # the summary counts too, after the title: GSE153056 (Papalexi, THP-1 PD-L1)
        # names neither in its title, both in its summary
        in_text = lambda c: any(_matches(c.summary, anchor) for anchor in anchors[pool])
        for candidate in pool_candidates:
            candidate.on_topic = candidate.on_topic or hit(candidate) or in_text(candidate)
        # a HuggingFace repo has only its name to go on: it must name the cells
        # (`uproai/chat-90k` matched "UPR"; `neuralblog/llama3-8b-mlp-neurons`, "neurons")
        pool_candidates = [
            c for c in pool_candidates
            if not c.reference.startswith("hf://") or any(_names_cells(c.title, a) for a in cell_anchors)
        ]
        ranked = sorted(pool_candidates, key=lambda c: (not hit(c), not in_text(c), -len(c.queries)))
        limit = MAX_SUPPLEMENTARY_CANDIDATES if pool == "supplementary" else MAX_CANDIDATES
        for candidate in ranked[:limit]:
            chosen[candidate.reference] = candidate
    return list(chosen.values())


# -- describe -------------------------------------------------------------------


def _estimate_cells(files: Sequence[dict]) -> int | None:
    barcodes = [f for f in files if "barcode" in str(f.get("name", "")).lower() and f.get("bytes")]
    if not barcodes:
        return None
    best = max(barcodes, key=lambda f: int(f["bytes"]))
    per = GZ_BYTES_PER_BARCODE if str(best["name"]).endswith(".gz") else PLAIN_BYTES_PER_BARCODE
    estimate = int(int(best["bytes"]) / per)
    return estimate if estimate <= MAX_PLAUSIBLE_CELLS else None


def _unfiltered(files: Sequence[dict]) -> bool:
    barcodes = [f for f in files if "barcode" in str(f.get("name", "")).lower() and f.get("bytes")]
    if any("raw" in str(f["name"]).lower() for f in files):
        return True
    return any(int(f["bytes"]) / GZ_BYTES_PER_BARCODE > MAX_PLAUSIBLE_CELLS for f in barcodes)


def _listed(sample: str, config: Any, attempts: int = 3) -> list[dict]:
    """A sample's files, retried: NCBI throttles bursts, and a throttled listing
    came back empty -- Adamson's UPR sample then had "no files", so no labels,
    so it was refused as gold. A sample with no files of its own (a guide
    library's SRA-only sample) stays empty after the retries."""
    from datafetcher.sources.geo import sample_files

    for attempt in range(attempts):
        files = sample_files(sample, config)
        if files:
            return files
        time.sleep(0.5 * (attempt + 1))
    return []


def describe(candidate: Candidate, config: Any) -> Candidate:
    """Series text, every sample's title and its own files; HF: the file list."""
    if candidate.reference.startswith("geo://"):
        from datafetcher.sources.geo import sample_titles, series_about

        accession = candidate.reference.split("://", 1)[1].split("#", 1)[0]
        candidate.about = series_about(accession, config)
        titles = sample_titles(accession, config)
        samples = list(titles.items())[:MAX_SAMPLES]
        with ThreadPoolExecutor(max_workers=4) as pool:
            listings = list(pool.map(lambda item: _listed(item[0], config), samples))
        candidate.samples = [
            {
                "id": gsm,
                "title": title,
                "files": [{"name": f["name"], "bytes": f.get("bytes")} for f in files],
                "bytes": sum(int(f.get("bytes") or 0) for f in files),
                "cells_estimate": _estimate_cells(files),
                "raw_barcodes": _unfiltered(files),
            }
            for (gsm, title), files in zip(samples, listings)
        ]
        # 3. a series whose samples have no files of their own keeps its data at
        # series level (GSE92872: CROP-seq Jurkat, every sample "0 B")
        if candidate.samples and not any(sample["files"] for sample in candidate.samples):
            from datafetcher.sources.geo import list_files_detailed

            series_files = [
                {"name": f["name"], "bytes": f.get("bytes")}
                for f in list_files_detailed(accession, config)
                if not str(f["name"]).endswith("_series_matrix.txt.gz")
            ]
            for sample in candidate.samples:
                sample["files"] = series_files
                sample["series_level"] = True
                sample["bytes"] = sum(int(f.get("bytes") or 0) for f in series_files)
                sample["cells_estimate"] = None
        if len(titles) > MAX_SAMPLES:
            candidate.about += f" | ({len(titles) - MAX_SAMPLES} more samples not listed)"
    else:
        from datafetcher.fetch import list_reference_files

        try:
            listing = list_reference_files(candidate.reference, config)
        except Exception as error:  # noqa: BLE001 - an unlistable repo is a finding
            listing = {"files": [], "about": f"listing failed: {error}"}
        files = [
            {"name": f.get("path") or f.get("name"), "bytes": f.get("bytes")}
            for f in listing.get("files") or []
        ]
        candidate.about = str(listing.get("about") or "")
        candidate.samples = [
            {
                "id": candidate.reference,
                "title": candidate.title,
                "files": files[:40],
                "bytes": sum(int(f.get("bytes") or 0) for f in files),
                "cells_estimate": None,
                "raw_barcodes": False,
            }
        ]
    return candidate


# -- judge ----------------------------------------------------------------------

_JUDGE_SCHEMA = {
    "type": "object",
    "properties": {
        "samples": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "string"},
                    "is_single_cell": {"type": "boolean"},
                    "is_perturbation_screen": {"type": "boolean"},
                    "modality": {"type": "string"},
                    "cell_type": {"type": "string"},
                    "cell_type_matches": {"type": "boolean"},
                    "organism": {"type": "string"},
                    "topic_match": {"type": "number"},
                    "has_perturbation_labels": {"type": "boolean"},
                    "role": {"type": "string", "enum": ["gold", "supplementary", "companion", "reject"]},
                    "why": {"type": "string"},
                },
                "required": ["id", "role", "is_single_cell", "cell_type_matches"],
            },
        }
    },
    "required": ["samples"],
}


def _size(value: Any) -> str:
    value = int(value or 0)
    for unit, scale in (("GB", 1e9), ("MB", 1e6), ("kB", 1e3)):
        if value >= scale:
            return f"{value / scale:.1f} {unit}"
    return f"{value} B"


def judge(spec: ScreenSpec, candidate: Candidate, client: Any) -> dict[str, Any]:
    """One model call: every sample of this series against the spec."""
    lines = []
    for sample in candidate.samples:
        files = ", ".join(f"{f['name']} ({_size(f.get('bytes'))})" for f in sample["files"][:12])
        lines.append(f"- {sample['id']}: {sample['title']}\n    files: {files or 'none listed'}")
    prompt = (
        "You are choosing data for a perturbation-response study. Judge every sample "
        "below against what the study needs.\n\n"
        f"Study question:\n{spec.question}\n\n"
        f"Needed: cell type {spec.cell_type!r} ({spec.organism}); perturbation "
        f"modality {spec.modality}; topic {spec.topic or '(none)'!r}.\n"
        "- gold: a SINGLE-CELL perturbation screen (Perturb-seq, CROP-seq, ECCITE-seq) "
        "in that cell type, with the perturbation of each cell recoverable (a guide "
        "table, guide calls, or guides counted as features of the matrix), ideally on "
        "the topic;\n"
        "- supplementary: single-cell expression of the same cell type, perturbed or "
        "not (its labels will be ignored);\n"
        "- companion: a file set that only serves another sample (guide/HTO/ADT "
        "counts of a screen);\n"
        "- reject: bulk data, another cell type or organism, or not expression.\n\n"
        f"Series {candidate.reference}: {candidate.title}\n{candidate.about[:1500]}\n\n"
        "Samples:\n" + "\n".join(lines) + "\n\n"
        "For each sample give: id (copied), is_single_cell, is_perturbation_screen, "
        "modality (CRISPRko/CRISPRi/CRISPRa/none), cell_type, cell_type_matches, "
        "organism, topic_match (0-1), has_perturbation_labels, role, why (<= 20 words). "
        "Answer as JSON with key 'samples'."
    )
    try:
        response = client.generate_structured(
            prompt=prompt, schema=_JUDGE_SCHEMA, max_tokens=4000, temperature=0
        )
        if isinstance(response, str):
            response = json.loads(response)
    except Exception as error:  # noqa: BLE001 - a failed judgement rejects the series
        return {"error": str(error), "samples": []}
    return response or {"samples": []}


#: File names that carry per-cell perturbation labels.
_LABEL_FILE = re.compile(
    r"identit|guide|sgrna|grna|cbc_gbc|protospacer|crispr|perturb|annotation|metadata|cell_?info",
    re.IGNORECASE,
)


#: A matrix file's name says what library it is, not that labels exist:
#: `GSM4632022_CRISPRi_filtered_feature_bc_matrix_lib1.h5` holds expression only.
_MATRIX_PART = re.compile(
    r"^.*(matrix|\.h5ad?$|\.loom$|barcodes|features|genes\.tsv).*$", re.IGNORECASE
)


def label_evidence(sample: dict[str, Any], companions: Sequence[dict[str, Any]]) -> str:
    """Why this sample's per-cell labels are recoverable, or "" when nothing says so.

    The model called Adamson's UPR sample "labels in companion files" and
    `has_perturbation_labels: false`, although its own `cell_identities.csv.gz`
    holds the guide calls; the file names are the evidence, not the model's word.
    """
    own = [
        f["name"] for f in sample.get("files") or []
        if _LABEL_FILE.search(_MATRIX_PART.sub("", str(f["name"])))
    ]
    if own:
        return f"label file {own[0]}"
    guides = [c for c in companions if re.search(r"guide|sgrna|grna|crispr", str(c.get("title")), re.I)]
    # the companion of *this* lane: GSE124703's neuron lane 2 goes with
    # "sgRNA enrichment for lane2", not with lane 1's or an iPSC lane's
    lane = re.search(r"lane[\s_]*(\d+)", str(sample.get("title")), re.I)
    if lane:
        same = [c for c in guides if re.search(rf"lane[\s_]*{lane.group(1)}\b", str(c.get("title")), re.I)]
        kind = re.sub(r"lane.*", "", str(sample.get("title")), flags=re.I).split("_")
        same = [c for c in same if any(k and k.lower() in str(c.get("title")).lower() for k in kind[1:3])] or same
        guides = same or guides
    guides = [c["id"] for c in guides]
    if guides:
        return f"guide companion sample(s) {', '.join(guides[:3])}"
    return ""


_GUIDE_COLUMN = re.compile(r"guide|sgrna|grna|perturb|target|gene_id|crispr", re.IGNORECASE)
#: The largest h5ad whose `obs` is peeked over HTTP range requests.
MAX_PEEK_BYTES = 20 * 1024**3


def peek_h5ad_labels(url: str, timeout_s: float = 240.0) -> str:
    """The per-cell label column of a remote `.h5ad`, read without downloading it.

    GSE289235 keeps its guide calls in `obs.guide` of a 1 GB h5ad: no file name
    says so. Only `obs` is read (column names, then one column's categories).
    Returns "" when there is no label-like column or the read fails.
    """
    from concurrent.futures import ThreadPoolExecutor
    from concurrent.futures import TimeoutError as FutureTimeout

    def read() -> str:
        import fsspec
        import h5py

        with fsspec.filesystem("https", block_size=1 << 20).open(url, "rb") as handle, h5py.File(handle, "r") as h5:
            obs = h5["obs"]
            for name in obs.keys():
                if not _GUIDE_COLUMN.search(name):
                    continue
                column = obs[name]
                if isinstance(column, h5py.Group) and "categories" in column:
                    values = [v.decode() if isinstance(v, bytes) else str(v) for v in column["categories"][:]]
                    return f"obs column {name!r} ({len(values)} values, e.g. {', '.join(values[:3])})"
                return f"obs column {name!r}"
        return ""

    pool = ThreadPoolExecutor(max_workers=1)
    try:
        return pool.submit(read).result(timeout=timeout_s)
    except (FutureTimeout, Exception):  # noqa: BLE001 - an unreadable file is just unverified
        return ""
    finally:
        pool.shutdown(wait=False)


def _verify_h5ad_labels(rows: list[dict[str, Any]], log: Log) -> None:
    """Unverified gold rows with an `.h5ad`: look inside for a label column."""
    for row in rows:
        if row["role"] != "gold" or not str(row.get("labels")).startswith("unverified"):
            continue
        for f in row["files"]:
            name = str(f.get("name"))
            if not name.endswith(".h5ad") or int(f.get("bytes") or 0) > MAX_PEEK_BYTES:
                continue
            gsm = row["sample"]
            url = (
                f"https://ftp.ncbi.nlm.nih.gov/geo/samples/{gsm[:-3]}nnn/{gsm}/suppl/{name}"
                if gsm.startswith("GSM") else str(f.get("url") or "")
            )
            found = peek_h5ad_labels(url) if url else ""
            log(f"# survey: {gsm} {name}: {found or 'no label column found'}")
            if found:
                row["labels"] = found
                break


def gate(spec: ScreenSpec, verdict: dict[str, Any], labels: str = "") -> tuple[str, str]:
    """The role the rules allow, and why a proposed role was refused.

    `labels`: the file-name evidence that per-cell labels exist (`label_evidence`).
    """
    role = str(verdict.get("role") or "reject")
    single = bool(verdict.get("is_single_cell"))
    cell_ok = bool(verdict.get("cell_type_matches"))
    organism_ok = spec.organism in str(verdict.get("organism") or spec.organism).lower()
    if role == "gold":
        problems = [
            reason
            for ok, reason in (
                (single, "not single-cell"),
                (bool(verdict.get("is_perturbation_screen")), "not a perturbation screen"),
                (cell_ok, "other cell type"),
                (organism_ok, "other organism"),
                (
                    bool(verdict.get("has_perturbation_labels")) or bool(labels),
                    "no per-cell perturbation labels",
                ),
                (
                    modality_compatible(spec.modality, _normal_modality(verdict.get("modality"))),
                    f"modality {verdict.get('modality')} is not {spec.modality}",
                ),
            )
            if not ok
        ]
        if not problems:
            return "gold", ""
        # a screen of the right cells that fails as gold can still be evidence
        if single and cell_ok and organism_ok:
            return "supplementary", "; ".join(problems)
        return "reject", "; ".join(problems)
    if role == "supplementary":
        if single and cell_ok and organism_ok:
            return "supplementary", ""
        return "reject", "not single-cell data of the right cell type"
    return role, ""


def _normal_modality(value: Any) -> str:
    text = str(value or "").lower()
    if "ko" in text or "knockout" in text:
        return "CRISPRko"
    if text.endswith("i") or "interference" in text or "knockdown" in text:
        return "CRISPRi"
    if text.endswith("a") or "activation" in text:
        return "CRISPRa"
    return "any"


# -- run ------------------------------------------------------------------------


def run_survey(
    question: str,
    out_dir: str | Path,
    *,
    client: Any,
    run_query: Callable[[str], list[dict]] | None = None,
    config: Any = None,
    log: Log = print,
    scbasecount: bool = True,
    scbasecount_table: Any = None,
    sra_links: Callable[[str], dict[int, str]] | None = None,
) -> dict[str, Any]:
    """`scbasecount`: add scBaseCount samples of the cells to the supplementary
    pool (`scbasecount_table`: the sample table, for tests; `sra_links`: GEO
    series -> {SRA uid: GSM}, for tests)."""
    from datafetcher.config import DataFetcherConfig

    started = time.time()
    config = config or DataFetcherConfig.from_env()
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    spec = extract_spec(question, client)
    run_query = run_query or _mechanical_query(config, spec.organism)
    log(f"# survey: spec {json.dumps({k: v for k, v in spec.to_dict().items() if k != 'question'})}")
    candidates = search(spec, run_query=run_query, log=log)
    log(f"# survey: describing {len(candidates)} candidate(s)")
    # two series at a time, four samples each: NCBI throttles larger bursts
    with ThreadPoolExecutor(max_workers=2) as pool:
        candidates = list(pool.map(lambda c: describe(c, config), candidates))
    rows: list[dict[str, Any]] = []
    for candidate in candidates:
        if not candidate.samples:
            continue
        candidate.judgement = judge(spec, candidate, client)
        verdicts = {str(v.get("id")): v for v in candidate.judgement.get("samples") or []}
        companions = [
            s for s in candidate.samples if (verdicts.get(str(s["id"])) or {}).get("role") == "companion"
        ]
        for sample in candidate.samples:
            verdict = verdicts.get(str(sample["id"]), {"role": "reject", "why": "not judged"})
            labels = label_evidence(sample, companions)
            role, refused = gate(spec, verdict, labels)
            if role == "gold" and not labels:
                # only the model's word: GSE152988's CROP-seq lanes are expression-only
                # h5 files, their guide calls are not on GEO
                labels = "unverified (the model's judgement; no label file or guide sample seen)"
            rows.append(
                {
                    "series": candidate.reference,
                    "series_title": candidate.title,
                    "sample": sample["id"],
                    "sample_title": sample["title"],
                    "role": role,
                    "proposed_role": verdict.get("role"),
                    "refused": refused,
                    "modality": verdict.get("modality"),
                    "cell_type": verdict.get("cell_type"),
                    "topic_match": verdict.get("topic_match"),
                    "bytes": sample["bytes"],
                    "cells_estimate": sample["cells_estimate"],
                    "raw_barcodes": sample["raw_barcodes"],
                    "series_level": bool(sample.get("series_level")),
                    "labels": labels,
                    "files": sample["files"],
                    "why": verdict.get("why", ""),
                }
            )
        kept = sum(r["role"] in ("gold", "supplementary") for r in rows if r["series"] == candidate.reference)
        log(f"# survey: {candidate.reference} ({len(candidate.samples)} sample(s)) -> {kept} kept")
    rows = _collapse_series_level(rows)
    _verify_h5ad_labels(rows, log)
    if scbasecount:
        rows += _scbasecount_rows(
            spec, rows, client=client, config=config, table=scbasecount_table, sra_links=sra_links, log=log
        )
    gold = sorted(
        (r for r in rows if r["role"] == "gold"),
        key=lambda r: (str(r.get("labels")).startswith("unverified"), -(r.get("topic_match") or 0), r["bytes"] or 0),
    )
    # GEO rows by cells, then scBaseCount's rows in their own order
    supplementary = sorted(
        (r for r in rows if r["role"] == "supplementary" and r.get("source") != "scbasecount"),
        key=lambda r: -(r["cells_estimate"] or 0),
    ) + [r for r in rows if r["role"] == "supplementary" and r.get("source") == "scbasecount"]
    result = {
        "question": question,
        "spec": spec.to_dict(),
        "gold": gold,
        "supplementary": supplementary,
        "rejected": [r for r in rows if r["role"] not in ("gold", "supplementary")],
        "fit": fit(gold, supplementary),
        "candidates": [
            {"reference": c.reference, "title": c.title, "queries": c.queries, "pools": sorted(c.pools)}
            for c in candidates
        ],
        "seconds": round(time.time() - started, 1),
    }
    (out / "survey.json").write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
    (out / "survey.md").write_text(render(result), encoding="utf-8")
    log(
        f"# survey: {len(gold)} gold and {len(supplementary)} supplementary sample(s) in "
        f"{result['seconds']:.0f}s -> {out / 'survey.md'}"
    )
    return result


def _cells(row: dict[str, Any]) -> int | None:
    """A row's cells: scBaseCount's count when the run is there, else the estimate."""
    linked = row.get("scbasecount") or {}
    return linked.get("cells") or row.get("cells_estimate")


def _scbasecount_rows(
    spec: ScreenSpec,
    rows: list[dict[str, Any]],
    *,
    client: Any,
    config: Any,
    table: Any = None,
    sra_links: Callable[[str], dict[int, str]] | None = None,
    log: Log = print,
) -> list[dict[str, Any]]:
    """scBaseCount's samples of the cells, minus the surveyed series' own runs.

    The surveyed GEO rows get scBaseCount's cell count and chemistry where their
    runs are there (GSE124703's neuron lanes list unfiltered barcodes: "?" cells).
    """
    from . import scbasecount

    try:
        table = scbasecount.sample_table(spec.organism) if table is None else table
    except Exception as error:  # noqa: BLE001 - the GEO survey stands without it
        log(f"# survey: scBaseCount unavailable ({error})")
        return []
    if sra_links is None:
        from datafetcher.sources.geo import series_sra_ids, sra_samples

        def sra_links(series: str) -> dict[int, str]:
            uids = series_sra_ids(series, config)
            return sra_samples(uids, config) if uids else {}

    kept = [r for r in rows if r["role"] in ("gold", "supplementary") and r["series"].startswith("geo://")]
    exclude: list[int] = []
    gold_chemistry: list[str] = []
    for series in dict.fromkeys(r["series"] for r in kept):
        links = sra_links(series.split("://", 1)[1])
        exclude += list(links)
        facts = scbasecount.gold_facts(list(links), table, links).get("per_sample", {})
        for row in (r for r in kept if r["series"] == series):
            runs = [run for gsm in row.get("samples") or [row["sample"]] for run in facts.get(gsm, [])]
            if not runs:
                continue
            # a GSM is a GEX run and maybe a guide run; the GEX run has the cells
            per_gsm = {}
            for gsm in row.get("samples") or [row["sample"]]:
                if facts.get(gsm):
                    per_gsm[gsm] = max(facts[gsm], key=lambda run: run["cells"])
            row["scbasecount"] = {
                "srx": [run["srx"] for run in per_gsm.values()],
                "cells": sum(run["cells"] for run in per_gsm.values()),
                "chemistry": next(iter(per_gsm.values()))["chemistry"],
            }
            if row["role"] == "gold":
                gold_chemistry.append(row["scbasecount"]["chemistry"])
    chemistry = max(set(gold_chemistry), key=gold_chemistry.count) if gold_chemistry else ""
    try:
        return scbasecount.survey_rows(
            spec, client=client, exclude_uids=exclude, chemistry=chemistry, table=table, log=log
        )
    except Exception as error:  # noqa: BLE001
        log(f"# survey: scBaseCount search failed ({error})")
        return []


#: The demonstration needs the supplementary pool this many times the gold.
MIN_SUPP_RATIO = 2.0


def fit(gold: list[dict[str, Any]], supplementary: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Per gold series: its cells against the supplementary pool's.

    A screen's lanes are one gold (GSE124703 lanes 1+2), so cells add up per
    series. Supplementary cells exclude the gold's own series.
    """
    out = []
    for series in dict.fromkeys(r["series"] for r in gold):
        rows = [r for r in gold if r["series"] == series]
        cells = [_cells(r) for r in rows]
        gold_cells = sum(c for c in cells if c)
        supp = [r for r in supplementary if r["series"] != series]
        supp_cells = sum(_cells(r) or 0 for r in supp)
        unlabeled = sum(_cells(r) or 0 for r in supp if not r.get("screen") and r.get("modality") in (None, "none", "-"))
        chemistry = next((r["scbasecount"]["chemistry"] for r in rows if r.get("scbasecount")), "")
        same_chem = sum(_cells(r) or 0 for r in supp if chemistry and r.get("chemistry") == chemistry)
        ratio = supp_cells / gold_cells if gold_cells else None
        out.append(
            {
                "series": series,
                "samples": [r["sample"] for r in rows],
                "gold_cells": gold_cells or None,
                "gold_cells_unknown": sum(c is None for c in cells),
                "gold_chemistry": chemistry,
                "supp_samples": len(supp),
                "supp_cells": supp_cells,
                "supp_unlabeled_cells": unlabeled,
                "supp_same_chemistry_cells": same_chem if chemistry else None,
                "supp_cells_unknown": sum(_cells(r) is None for r in supp),
                "ratio": round(ratio, 1) if ratio else None,
                "enough": bool(ratio and ratio >= MIN_SUPP_RATIO),
            }
        )
    return out


def _collapse_series_level(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """One row per series and role when the samples have no files of their own.

    GSE92872's twelve samples each carried the whole series' files: twelve gold
    rows of the same 19 MB, and GSE327057's eight showed 61 GB apiece.
    """
    out: list[dict[str, Any]] = []
    merged: dict[tuple[str, str], dict[str, Any]] = {}
    for row in rows:
        if not row.get("series_level"):
            out.append(row)
            continue
        key = (row["series"], row["role"])
        if key not in merged:
            merged[key] = {
                **row,
                "sample": row["series"].split("://", 1)[1],
                "sample_title": f"{row['series_title']} (series-level files; samples: {row['sample']}",
                "samples": [row["sample"]],
            }
            out.append(merged[key])
        else:
            merged[key]["samples"].append(row["sample"])
    for row in merged.values():
        row["sample_title"] += f"{', ...' if len(row['samples']) > 1 else ''} {len(row['samples'])} total)"
    return out


def _mechanical_query(config: Any, organism: str = "") -> Callable[[str], list[dict]]:
    """Repository search (ToolUniverse: GEO RNA-seq series, OmicsDI, HF) plus a
    GEO full-text search (`geo.search_series`: every series type, summary text)."""
    from datafetcher.mechanical import mechanical_search
    from datafetcher.sources.geo import search_series

    def run(query: str) -> list[dict]:
        try:
            result = mechanical_search(query, config=config, query=query, limit=8)
            hits = [hit.to_dict() for hit in result.hits]
        except Exception:  # noqa: BLE001 - one failed search is not fatal
            hits = []
        seen = {str(h.get("reference")) for h in hits}
        try:
            full_text = search_series(query, config, organism=organism, retmax=GEO_FULL_TEXT_HITS)
        except Exception:  # noqa: BLE001
            full_text = []
        for hit in full_text:
            if hit["reference"] in seen:
                # keep the summary for ranking
                next(h for h in hits if str(h.get("reference")) == hit["reference"])["summary"] = hit["summary"]
            else:
                hits.append(hit)
        return hits

    return run


#: Supplementary rows listed in survey.md (all of them are in survey.json).
MAX_LISTED = 40


def render(result: dict[str, Any]) -> str:
    spec = result["spec"]

    def table(rows: list[dict], limit: int | None = None) -> list[str]:
        if not rows:
            return ["(none)", ""]
        out = [
            "| sample | series | title | modality | topic | chemistry | download | cells |",
            "|---|---|---|---|---|---|---|---|",
        ]
        for r in rows[:limit]:
            linked = r.get("scbasecount") or {}
            cells = _cells(r)
            if not cells:
                cells_text = "?"
            elif linked or r.get("cells_exact"):
                cells_text = f"{cells:,}" + (" (scBaseCount)" if linked else "")
            else:
                cells_text = f"~{cells:,}" + (" (raw barcodes)" if r["raw_barcodes"] else "")
            chemistry = r.get("chemistry") or linked.get("chemistry") or "-"
            out.append(
                f"| {r['sample']} | {r['series']} | {str(r['sample_title'])[:60]} | "
                f"{r.get('modality') or '-'} | {r.get('topic_match') if r.get('topic_match') is not None else '-'} | "
                f"{chemistry} | {_size(r['bytes']) if r['bytes'] else '?'} | {cells_text} |"
            )
        if limit is not None and len(rows) > limit:
            rest = rows[limit:]
            out.append(
                f"| ... {len(rest)} more | | | | | | | {sum(_cells(r) or 0 for r in rest):,} |"
            )
        return out + [""]

    def fit_lines() -> list[str]:
        rows = result.get("fit") or []
        if not rows:
            return ["(no gold)", ""]
        out = [
            "| gold series | gold samples | gold cells | chemistry | supp samples | supp cells "
            "| of which unlabeled | same chemistry | supp / gold | enough (>= "
            f"{MIN_SUPP_RATIO:g}x) |",
            "|---|---|---|---|---|---|---|---|---|---|",
        ]
        for f in rows:
            gold_cells = f"{f['gold_cells']:,}" if f["gold_cells"] else "?"
            if f["gold_cells_unknown"] and f["gold_cells"]:
                gold_cells += f" (+{f['gold_cells_unknown']} unknown)"
            same = f"{f['supp_same_chemistry_cells']:,}" if f["supp_same_chemistry_cells"] is not None else "-"
            out.append(
                f"| {f['series']} | {len(f['samples'])} | {gold_cells} | {f['gold_chemistry'] or '-'} | "
                f"{f['supp_samples']} | {f['supp_cells']:,} | {f['supp_unlabeled_cells']:,} | {same} | "
                f"{f['ratio'] if f['ratio'] is not None else '?'} | {'yes' if f['enough'] else 'no'} |"
            )
        return out + [""]

    lines = [
        "# Survey",
        "",
        f"**Question:** {result['question']}",
        "",
        f"- cell type: `{spec['cell_type']}` (aliases: {', '.join(spec['cell_type_aliases']) or '-'}), "
        f"organism `{spec['organism']}`, modality `{spec['modality']}`, topic `{spec['topic'] or '-'}`",
        f"- {len(result['candidates'])} series described, {result['seconds']:.0f}s, nothing downloaded",
        "",
        "## Fit: gold size against the supplementary pool",
        "",
        *fit_lines(),
        "## Gold candidates (single-cell screens of the right cells, with labels)",
        "",
        *table(result["gold"]),
        *[f"- {r['sample']}: labels — {r.get('labels') or '-'}" for r in result["gold"]],
        "",
        "## Supplementary candidates (single-cell, same cell type; GEO largest first, then scBaseCount)",
        "",
        *table(result["supplementary"], MAX_LISTED),
        f"## Rejected / companion samples: {len(result['rejected'])}",
        "",
    ]
    for r in result["rejected"][:30]:
        reason = r["refused"] or r["why"]
        lines.append(f"- {r['sample']} ({r['series']}) — {r['role']}: {str(reason)[:140]}")
    return "\n".join(lines) + "\n"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("question")
    parser.add_argument("--out", required=True)
    parser.add_argument(
        "--no-scbasecount", action="store_true", help="do not add scBaseCount samples to the supplementary pool"
    )
    args = parser.parse_args(argv)
    from dotenv import load_dotenv

    load_dotenv(Path(__file__).resolve().parents[4] / ".env")
    from kosmos.core.llm import get_client

    run_survey(args.question, args.out, client=get_client(), scbasecount=not args.no_scbasecount)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
