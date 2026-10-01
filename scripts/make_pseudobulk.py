"""Derive a perturbation-level pseudobulk from a scPerturb Replogle h5ad.

Reproduces the STRUCTURE of Replogle's own *_normalized_bulk_01.h5ad (one row
per perturbation, gemgroup Z-normalized) from the raw-count scPerturb file.
It is NOT byte-identical to the official file -- the official one is on
figshare, which is WAF-blocking this host. Method, stated plainly:

  1. per cell   : x / total_UMI * median_total_UMI      (depth normalisation)
  2. per batch  : z = (x - mean_batch) / std_batch      (gemgroup Z, 56 batches)
  3. per pert   : mean of z over that perturbation's cells

Two streaming passes so an 8.7 GB dense matrix never lands in RAM at once.
"""
import sys, time
import h5py, numpy as np, anndata as ad, pandas as pd

SRC, OUT, CHUNK = sys.argv[1], sys.argv[2], 10_000

def _strings(v):
    return np.array([x.decode() if isinstance(x, bytes) else str(x) for x in v])


def read_node(node):
    """h5 node -> string array, across anndata categorical/plain encodings."""
    if isinstance(node, h5py.Group):                   # categorical: codes+categories
        return _strings(node["categories"][:])[node["codes"][:]]
    return _strings(node[:])


def read_cat(h, col):
    return read_node(h["obs"][col])


def read_index(group):
    """The frame's index, named by the `_index` attr (e.g. var -> 'gene_name')."""
    key = dict(group.attrs).get("_index", "_index")
    return read_node(group[key])

t0 = time.time()
with h5py.File(SRC, "r") as h:
    X = h["X"]
    n, g = X.shape
    pert  = read_cat(h, "perturbation")
    batch = read_cat(h, "batch")
    genes = read_index(h["var"])                       # 'gene_name', not a placeholder
    ensembl = (read_node(h["var"]["ensembl_id"])
               if "ensembl_id" in h["var"] else None)
    assert len(genes) == g, f"var index is {len(genes)}, expected {g}"

    b_lev, b_idx = np.unique(batch, return_inverse=True)
    p_lev, p_idx = np.unique(pert,  return_inverse=True)
    print(f"{n:,} cells x {g:,} genes | {len(p_lev):,} perturbations | {len(b_lev)} batches",
          flush=True)

    # --- pass 1: depth-normalise, then per-batch mean/std -------------------
    bsum  = np.zeros((len(b_lev), g), dtype=np.float64)
    bsq   = np.zeros((len(b_lev), g), dtype=np.float64)
    bn    = np.zeros(len(b_lev), dtype=np.int64)
    totals = np.empty(n, dtype=np.float64)

    for i in range(0, n, CHUNK):                       # first read: also get totals
        raw = X[i:i+CHUNK].astype(np.float64)
        totals[i:i+raw.shape[0]] = raw.sum(1)
    med = np.median(totals[totals > 0])
    print(f"pass0 totals done ({time.time()-t0:.0f}s) median UMI={med:.0f}", flush=True)

    for i in range(0, n, CHUNK):
        raw = X[i:i+CHUNK].astype(np.float64)
        t = totals[i:i+raw.shape[0]].copy(); t[t == 0] = 1.0
        norm = raw / t[:, None] * med
        bi = b_idx[i:i+raw.shape[0]]
        np.add.at(bsum, bi, norm)
        np.add.at(bsq,  bi, norm * norm)
        np.add.at(bn,   bi, 1)
        if (i // CHUNK) % 5 == 0:
            print(f"  pass1 {i:,}/{n:,} ({time.time()-t0:.0f}s)", flush=True)

    bmean = bsum / bn[:, None]
    bstd  = np.sqrt(np.maximum(bsq / bn[:, None] - bmean**2, 0))
    bstd[bstd < 1e-8] = 1.0                            # constant gene in a batch -> z=0

    # --- pass 2: Z within batch, average within perturbation ----------------
    psum = np.zeros((len(p_lev), g), dtype=np.float64)
    pn   = np.zeros(len(p_lev), dtype=np.int64)
    for i in range(0, n, CHUNK):
        raw = X[i:i+CHUNK].astype(np.float64)
        t = totals[i:i+raw.shape[0]].copy(); t[t == 0] = 1.0
        norm = raw / t[:, None] * med
        bi = b_idx[i:i+raw.shape[0]]
        z = (norm - bmean[bi]) / bstd[bi]
        pi = p_idx[i:i+raw.shape[0]]
        np.add.at(psum, pi, z)
        np.add.at(pn,   pi, 1)
        if (i // CHUNK) % 5 == 0:
            print(f"  pass2 {i:,}/{n:,} ({time.time()-t0:.0f}s)", flush=True)

bulk = (psum / pn[:, None]).astype(np.float32)
obs = pd.DataFrame({"perturbation": p_lev, "n_cells": pn}).set_index("perturbation")
var = pd.DataFrame(index=pd.Index(genes, name=None))
if ensembl is not None:
    var["ensembl_id"] = ensembl
a = ad.AnnData(X=bulk, obs=obs, var=var)
a.uns["derivation"] = (
    "Derived locally from %s (scPerturb harmonised raw counts, Zenodo record "
    "13350497) by Kosmos/scripts/make_pseudobulk.py. Method: per-cell depth "
    "normalisation to median UMI, Z-score within `batch` (gemgroup), mean per "
    "`perturbation`. This is NOT any official figshare *_bulk_01.h5ad file -- "
    "those live only on plus.figshare.com record 20029387 and are not "
    "byte-comparable to this." % SRC.split("/")[-1])
a.write_h5ad(OUT)
print(f"WROTE {OUT}  shape={a.shape}  ({time.time()-t0:.0f}s)", flush=True)
