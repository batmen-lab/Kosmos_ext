"""Gold check for the iPSC-neuron question: read each candidate gold in full,
call cells and guides, and measure split-half reliability of the observed Δ.

Candidates (from artifacts/dataprep/ipsc-neuron-v3/survey.md):
    GSE124703  Tian 2019, CROP-seq day-7 neurons, lanes 1+2 (mtx, unfiltered) with
               per-lane sgRNA mapping companions (protospacer sequences; genes
               from the hCRISPRi-v2 library table, Horlbeck 2016)
    GSE289235  CaMPARI2 CRISPRi platform, CROP-seq day-21 neurons, one h5ad
               (unfiltered, `obs.guide`)
GSE152988 (Tian 2021) is not checked: its GEO files are expression-only.

    .venv/bin/python Perturb_middle/check_gold_ipsc_neuron.py \
        --out artifacts/dataprep/ipsc-neuron-v3/gold_check.md
"""

from __future__ import annotations

import argparse
import sys
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from kosmos.ppi.perturbation.dataprep.load import Sample, call_cells, load_mtx  # noqa: E402
from kosmos.ppi.perturbation.dataprep.reliability import render, split_half  # noqa: E402

SAMPLES = ROOT / "data" / "fetched" / "samples"
LIBRARY_URL = (
    "https://raw.githubusercontent.com/mhorlbeck/ScreenProcessing/master/"
    "library_tables/CRISPRi_v2_human_librarytable.txt"
)
LIBRARY = ROOT / "data" / "fetched" / "reference" / "CRISPRi_v2_human_librarytable.txt"
GEO_SAMPLE = "https://ftp.ncbi.nlm.nih.gov/geo/samples/{group}/{gsm}/suppl/{name}"


def fetch(gsm: str, name: str, series: str) -> Path:
    path = SAMPLES / series / gsm / name
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        url = GEO_SAMPLE.format(group=gsm[:-3] + "nnn", gsm=gsm, name=name)
        print(f"downloading {url}")
        urllib.request.urlretrieve(url, path)
    return path


def tian_2019(min_umis: int, min_reads: int, min_fraction: float) -> tuple[Sample, np.ndarray, list[str], str]:
    """Lanes 1+2. A cell's guide: the amplified guide dominates its guide reads
    (thousands vs ambient single digits)."""
    if not LIBRARY.exists():
        LIBRARY.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(LIBRARY_URL, LIBRARY)
    library = pd.read_csv(LIBRARY, sep="\t")
    gene_of = {seq[-19:]: gene for seq, gene in zip(library.sequence.str.upper(), library.gene)}
    parts, notes = [], []
    for lane, gex, guide in ((1, "GSM3543622", "GSM3543624"), (2, "GSM3543623", "GSM3543625")):
        files = [fetch(gex, f"{gex}_neuron_{lane}_{kind}", "GSE124703")
                 for kind in ("matrix.mtx.gz", "barcodes.tsv.gz", "genes.tsv.gz")]
        mapping = fetch(guide, f"{guide}_neuron_{lane}_sgRNA_mapping.txt.gz", "GSE124703")
        sample, before = call_cells(load_mtx(*files), min_umis)
        reads = pd.read_csv(mapping, sep="\t")
        reads = reads[(reads.read_count > 0) & ~reads.barcode.str.startswith("unprocessed")]
        reads = reads[reads.cell.isin(set(sample.barcodes))]
        total = reads.groupby("cell").read_count.sum()
        top = reads.sort_values("read_count", ascending=False).drop_duplicates("cell").set_index("cell")
        share = top.read_count / total.reindex(top.index)
        call = pd.Series("no_guide", index=sample.barcodes, dtype=object)
        good = top.index[(top.read_count >= min_reads) & (share >= min_fraction)]
        call.loc[top.index] = "ambiguous"
        call.loc[good] = top.loc[good, "barcode"]
        notes.append(f"lane {lane}: {before:,} barcodes -> {sample.n_cells:,} cells (>= {min_umis} UMIs); "
                     f"guide called {len(good):,}")
        parts.append((sample, call.values))
    labels = np.concatenate([p[1] for p in parts])
    # a protospacer not in the library: the two abundant ones behave as
    # non-targeting controls (no gene knocked down); the rare two stay unnamed
    genes = np.array([gene_of.get(g[-19:], g) if len(g) == 20 else g for g in labels], dtype=object)
    unnamed = pd.Series([g for g in genes if len(str(g)) == 20 and str(g).isupper() and set(str(g)) <= set("ACGT")])
    controls = unnamed.value_counts().index[:2].tolist()
    genes = np.array(["NT" if g in controls else g for g in genes], dtype=object)
    counts = sparse.vstack([p[0].counts for p in parts]).tocsr()
    sample = Sample(counts, [b for p in parts for b in p[0].barcodes], parts[0][0].genes, "GSE124703")
    notes.append(f"non-targeting = the two abundant protospacers absent from hCRISPRi-v2: {', '.join(controls)}")
    return sample, genes, ["NT"], "\n".join(f"- {n}" for n in notes)


def gse289235(min_umis: int) -> tuple[Sample, np.ndarray, list[str], str]:
    import anndata

    path = fetch("GSM8787485", "GSM8787485_EXP224.h5ad", "GSE289235")
    adata = anndata.read_h5ad(path)
    umis = np.asarray(adata.X.sum(axis=1)).ravel()
    keep = np.flatnonzero(umis >= min_umis)
    guides = adata.obs["guide"].astype(str).values[keep]
    genes = np.array(
        ["NT" if g.startswith("non-targeting") else "no_guide" if g == "nan" else g.rsplit("_", 1)[0] for g in guides],
        dtype=object,
    )
    sample = Sample(adata.X[keep].tocsr(), list(adata.obs_names[keep]), list(adata.var_names), "GSE289235")
    note = (f"- {adata.n_obs:,} barcodes -> {len(keep):,} cells (>= {min_umis} UMIs); "
            f"with a guide {int((genes != 'no_guide').sum()):,}; non-targeting {int((genes == 'NT').sum()):,}")
    return sample, genes, ["NT"], note


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", required=True)
    parser.add_argument("--min-umis", type=int, default=1000)
    parser.add_argument("--min-reads", type=int, default=50)
    parser.add_argument("--min-fraction", type=float, default=0.8)
    args = parser.parse_args()
    sections = ["# Gold check: iPSC-derived neurons, CRISPRi", ""]
    for title, load in (
        ("GSE124703 (Tian 2019, day-7 neurons, lanes 1+2)",
         lambda: tian_2019(args.min_umis, args.min_reads, args.min_fraction)),
        ("GSE289235 (CaMPARI2 CRISPRi platform, day-21 neurons)", lambda: gse289235(args.min_umis)),
    ):
        sample, labels, controls, notes = load()
        result = split_half(sample, labels, controls, min_cells=25)
        sections += [f"## {title}", "", notes, "", render(result, top=50), ""]
        print(f"{title}: {result['summary']}")
    Path(args.out).write_text("\n".join(sections), encoding="utf-8")
    print(f"-> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
