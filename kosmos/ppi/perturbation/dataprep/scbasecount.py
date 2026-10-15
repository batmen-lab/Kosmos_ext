"""scBaseCount as a supplementary source: uniformly processed single-cell samples.

scBaseCount (Youngblut et al., Cell 2026, doi 10.1016/j.cell.2026.08.025) is
Arc Institute's agent-curated re-processing of the SRA's 10x data: every SRX
sample counted by STARsolo against one reference, with sample metadata
(cell line, tissue, perturbation, 10x chemistry, exact cell count). For a
supplementary pool that is what GEO cannot give: cells of the same line from
many unrelated studies, one gene space, and the chemistry stated.

    sample table   one parquet per organism (~3 MB human), cached under
                   data/fetched/scbasecount/
    match          rows whose cell line / tissue names the cells (rules), then
                   one model call over the distinct names: which are the cells
                   the question needs
    gold link      a GEO series' SRX uids (`geo.series_sra_ids`): the gold's own
                   samples are dropped from the pool, and their cell counts and
                   chemistry are read from the table

The bucket (`gs://arc-institute-virtual-cell-atlas`, the 2026-01-12 release) is
read over HTTPS; one sample is one `.h5ad` (`adata.X` = unique-gene UMI counts).
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

BUCKET = "arc-institute-virtual-cell-atlas"
RELEASE = "2026-01-12"
FEATURE = "GeneFull_Ex50pAS"
SPECIES = {"human": "Homo_sapiens", "mouse": "Mus_musculus"}
HTTP_ROOT = f"https://storage.googleapis.com/{BUCKET}"
#: Distinct (cell line, tissue) names shown to the model at most.
MAX_NAMES = 200
#: Rows whose download size is looked up (one small request each).
MAX_SIZED = 60

_SCREEN = re.compile(r"crispr|perturb-?seq|sgrna|guide|knock|screen|crop-?seq|eccite", re.I)
#: Words of a cell name that do not pick out the cells.
_UNSPECIFIC = frozenset(
    "cell cells line lines derived human mouse primary cultured culture in vitro the of and".split()
)

Log = Callable[[str], None]


def object_url(gs_path: str) -> str:
    return gs_path.replace(f"gs://{BUCKET}", HTTP_ROOT, 1)


def sample_table(organism: str = "human", cache_dir: str | Path | None = None):
    """The release's sample metadata for one organism, downloaded once."""
    import urllib.request

    import pandas as pd

    species = SPECIES.get(organism, organism)
    cache_dir = cache_dir or Path(__file__).resolve().parents[4] / "data" / "fetched" / "scbasecount"
    path = Path(cache_dir) / f"{RELEASE}_{species}_sample_metadata.parquet"
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        url = f"{HTTP_ROOT}/scbasecount/{RELEASE}/metadata/{FEATURE}/{species}/sample_metadata.parquet"
        tmp = path.with_suffix(".part")
        with urllib.request.urlopen(url, timeout=120) as response, open(tmp, "wb") as out:
            out.write(response.read())
        tmp.rename(path)
    return pd.read_parquet(path)


def _compact(text: Any) -> str:
    return re.sub(r"[^a-z0-9]", "", str(text or "").lower())


def name_keys(names: Sequence[str]) -> list[list[str]]:
    """Per cell name, the words that must all appear: `THP-1` -> [thp1];
    `iPSC-derived neurons` -> [ipsc, neurons]."""
    keys = []
    for name in names:
        # a token with a digit or a hyphen is one word (`THP-1`, `HEK293T`)
        tokens = re.findall(r"[A-Za-z0-9]+(?:-[A-Za-z0-9]+)*", str(name))
        words = []
        for token in tokens:
            parts = [token] if re.search(r"\d", token) else token.split("-")
            words += [_compact(p) for p in parts if _compact(p) not in _UNSPECIFIC and len(_compact(p)) > 1]
        # plural-insensitive: `neurons` matches `neuron`
        words = [w[:-1] if len(w) > 4 and w.endswith("s") else w for w in words]
        if words:
            keys.append(words)
    return keys


def prefilter(table, names: Sequence[str]):
    """Rows whose cell line / tissue / collection text has every word of a name."""
    text = (
        table["cell_line"].map(_compact) + "|" + table["tissue"].map(_compact) + "|"
        + table["czi_collection_name"].map(_compact)
    )
    keep = None
    for words in name_keys(names):
        hit = text.map(lambda t, words=words: all(w in t for w in words))
        keep = hit if keep is None else keep | hit
    return table[keep] if keep is not None else table.iloc[0:0]


_CHOOSE_SCHEMA = {
    "type": "object",
    "properties": {"keep": {"type": "array", "items": {"type": "integer"}}},
    "required": ["keep"],
}


def choose(spec: Any, rows, client: Any) -> Any:
    """One model call over the distinct names: which describe the needed cells.

    Without a model every prefiltered row is kept.
    """
    if client is None or rows.empty:
        return rows
    groups = (
        rows.assign(_name=rows["cell_line"].astype(str) + " | " + rows["tissue"].astype(str))
        .groupby("_name")["obs_count"].agg(["size", "sum"]).sort_values("sum", ascending=False)
    )
    names = list(groups.index[:MAX_NAMES])
    listing = "\n".join(
        f"{i}: {name}  ({groups.loc[name, 'size']} samples, {groups.loc[name, 'sum']:,} cells)"
        for i, name in enumerate(names)
    )
    prompt = (
        "These are sample descriptions (cell line | tissue) from a single-cell RNA-seq "
        "repository. Keep the ones that ARE the cells this study needs, so that their "
        "expression profiles can serve as unlabeled evidence for it.\n\n"
        f"Study question:\n{spec.question}\n\nNeeded cells: {spec.cell_type!r} "
        f"(also called {', '.join(spec.cell_type_aliases) or '-'}), {spec.organism}.\n\n"
        "Keep a name when it is that cell type or line in a comparable state "
        "(a differentiated derivative of a line is NOT the line; an undifferentiated "
        "stem cell is not the neurons derived from it). Reject mixtures of several "
        "cell lines, other cell types, tissues and organoids that contain the cells "
        "among many others.\n\n"
        f"{listing}\n\nAnswer as JSON: {{\"keep\": [indices]}}."
    )
    try:
        response = client.generate_structured(
            prompt=prompt, schema=_CHOOSE_SCHEMA, max_tokens=3000, temperature=0
        )
        if isinstance(response, str):
            response = json.loads(response)
        kept = {names[i] for i in (response or {}).get("keep") or [] if 0 <= int(i) < len(names)}
    except Exception:  # noqa: BLE001 - an unanswered choice keeps nothing rather than everything
        return rows.iloc[0:0]
    name = rows["cell_line"].astype(str) + " | " + rows["tissue"].astype(str)
    return rows[name.isin(kept)]


def is_screen(perturbation: Any) -> bool:
    return bool(_SCREEN.search(str(perturbation or "")))


def object_sizes(paths: Sequence[str]) -> dict[str, int | None]:
    import urllib.parse
    import urllib.request

    def size(path: str) -> int | None:
        # a gs:// path or its https:// form
        name = path.split(f"gs://{BUCKET}/", 1)[-1].split(f"{HTTP_ROOT}/", 1)[-1]
        url = (
            f"https://storage.googleapis.com/storage/v1/b/{BUCKET}/o/"
            f"{urllib.parse.quote(name, safe='')}?fields=size"
        )
        try:
            with urllib.request.urlopen(url, timeout=30) as response:
                return int(json.loads(response.read())["size"])
        except Exception:  # noqa: BLE001 - an unknown size is shown as such
            return None

    with ThreadPoolExecutor(max_workers=8) as pool:
        return dict(zip(paths, pool.map(size, paths)))


def survey_rows(
    spec: Any,
    *,
    client: Any,
    exclude_uids: Sequence[int] = (),
    chemistry: str = "",
    table: Any = None,
    log: Log = print,
) -> list[dict[str, Any]]:
    """Supplementary rows (survey format) for the spec's cells, largest first.

    Order: the gold's chemistry first when known (Papalexi's 5' screen vs the
    3' THP-1 macrophages), unlabeled samples before other screens, then cells.
    """
    from .survey import specific_cell_terms

    if table is None:
        table = sample_table(spec.organism)
    names = specific_cell_terms(spec) or [spec.cell_type]
    rows = prefilter(table, names)
    log(f"# survey: scBaseCount: {len(rows)} sample(s) name the cells {names}")
    rows = choose(spec, rows, client)
    excluded = rows["entrez_id"].isin(set(int(u) for u in exclude_uids))
    if excluded.any():
        log(f"# survey: scBaseCount: {int(excluded.sum())} sample(s) are the gold's own runs; dropped")
    rows = rows[~excluded]
    out = []
    for _, r in rows.iterrows():
        out.append(
            {
                "series": "scbasecount",
                "series_title": "scBaseCount (Arc Institute), uniformly processed SRA 10x data",
                "sample": r["srx_accession"],
                "sample_title": f"{r['cell_line']} | {r['tissue']} | {r['perturbation']}",
                "role": "supplementary",
                "proposed_role": "supplementary",
                "refused": "",
                "modality": "none" if not is_screen(r["perturbation"]) else "screen (labels ignored)",
                "cell_type": r["cell_line"],
                "topic_match": None,
                "bytes": None,
                "cells_estimate": int(r["obs_count"]),
                "cells_exact": True,
                "chemistry": r["tech_10x"],
                "cell_prep": r["cell_prep"],
                "screen": is_screen(r["perturbation"]),
                "raw_barcodes": False,
                "series_level": False,
                "labels": "",
                "files": [{"name": object_url(r["file_path"]), "bytes": None}],
                "why": "",
                "source": "scbasecount",
            }
        )
    out.sort(
        key=lambda row: (
            bool(chemistry) and row["chemistry"] != chemistry,
            row["screen"],
            -(row["cells_estimate"] or 0),
        )
    )
    sizes = object_sizes([row["files"][0]["name"] for row in out[:MAX_SIZED]])
    for row in out[:MAX_SIZED]:
        row["bytes"] = row["files"][0]["bytes"] = sizes.get(row["files"][0]["name"])
    log(
        f"# survey: scBaseCount: {len(out)} supplementary sample(s), "
        f"{sum(r['cells_estimate'] for r in out):,} cells "
        f"({sum(not r['screen'] for r in out)} not from screens)"
    )
    return out


def gold_facts(uids: Sequence[int], table: Any, gsm_of: dict[int, str] | None = None) -> dict[str, Any]:
    """The gold series' runs as scBaseCount counted them: cells and chemistry,
    per GSM when `gsm_of` (SRA uid -> GSM, `geo.sra_samples`) is given."""
    rows = table[table["entrez_id"].isin(set(int(u) for u in uids))]
    if rows.empty:
        return {}
    gsm_of = gsm_of or {}
    per_gsm: dict[str, list[dict[str, Any]]] = {}
    for _, r in rows.iterrows():
        per_gsm.setdefault(gsm_of.get(int(r["entrez_id"]), "") or r["srx_accession"], []).append(
            {"srx": r["srx_accession"], "cells": int(r["obs_count"]), "chemistry": r["tech_10x"]}
        )
    return {
        "chemistry": str(rows["tech_10x"].value_counts().index[0]),
        "per_sample": per_gsm,
    }
