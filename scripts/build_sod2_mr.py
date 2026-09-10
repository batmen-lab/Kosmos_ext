#!/usr/bin/env python
"""
Build a Mendelian-randomization-ready dataset for SOD2 -> myocardial T1 (fibrosis),
reproducing Discovery 4 (Figure 5) of the Kosmos paper.

  Exposure : cis-pQTL for circulating SOD2 (genetic instruments for protein level).
  Outcome  : myocardial T1 relaxation GWAS (Nauffal et al.), SOD2 cis-window.
  Method   : harmonize exposure & outcome on chr:pos, align effect alleles, keep
             independent genome-wide-significant cis instruments, then inverse-
             variance-weighted (IVW) Mendelian randomization.

Writes an MR-ready CSV (one row per instrument) and prints a reference IVW estimate
so a downstream Kosmos run can be validated against it.

REAL DATA ONLY — this script never generates synthetic/placeholder values. If an
input is missing or instruments cannot be formed, it exits with an error.

Paper reference values (UKB-PPP pQTL + Nauffal T1): Kosmos beta = -0.231 (p=4.23e-13),
human beta = -0.258 (p=1.22e-22), protective direction (higher SOD2 -> lower T1).
"""
import argparse
import sys

import numpy as np
import pandas as pd
from scipy import stats

# SOD2 gene, GRCh37 (both the Nauffal T1 outcome and the UKB-PPP `ID` column use
# GRCh37 positions; UKB-PPP `GENPOS` is GRCh38, so we merge on the ID-derived b37
# position — see load_pqtl).
SOD2_CHR = "6"
SOD2_START = 160_090_588
SOD2_END = 160_183_517
CIS_PAD = 500_000  # +/- 500 kb around the gene defines the "cis" window


def _norm_allele(a):
    return str(a).upper().strip()


def load_pqtl(path):
    """Load a SOD2 cis-pQTL summary-stats file, auto-detecting common column names."""
    df = pd.read_csv(path, sep=None, engine="python")
    cols = {c.lower(): c for c in df.columns}

    def pick(*names):
        for n in names:
            if n in cols:
                return cols[n]
        return None

    m = dict(
        chrom=pick("chromosome", "chr", "chrom", "#chrom"),
        pos=pick("position", "pos", "pos_b38", "base_pair_location", "genpos", "bp"),
        ea=pick("effect_allele", "ea", "allele1", "a1", "alt", "tested_allele"),
        oa=pick("other_allele", "oa", "allele0", "allele2", "a2", "ref"),
        beta=pick("beta", "effect", "effect_size", "beta1"),
        se=pick("se", "standard_error", "standarderror", "se_beta", "sebeta"),
        p=pick("p", "pval", "p_value", "pvalue", "log10p", "neg_log10_p_value"),
        idc=pick("id", "variant_id", "snp", "rsid", "markername"),
    )
    need = [k for k in ("chrom", "pos", "ea", "oa", "beta", "se") if m[k] is None]
    if need:
        sys.exit(f"[pQTL] missing columns for {need}. Found: {list(df.columns)}")

    pos = pd.to_numeric(df[m["pos"]], errors="coerce")
    # UKB-PPP: GENPOS is GRCh38 but the ID (chr:pos_b37:a0:a1) carries the GRCh37
    # position that matches the Nauffal outcome. Prefer the ID-derived b37 position.
    if m["idc"]:
        id_pos = df[m["idc"]].astype(str).str.extract(r"^[^:]+:(\d+):")[0]
        id_pos = pd.to_numeric(id_pos, errors="coerce")
        if id_pos.notna().mean() > 0.9:  # ID column really is chr:pos:...
            pos = id_pos

    out = pd.DataFrame({
        "chrom": df[m["chrom"]].astype(str).str.replace("chr", "", case=False, regex=False),
        "pos": pos,
        "ea_e": df[m["ea"]].map(_norm_allele),
        "oa_e": df[m["oa"]].map(_norm_allele),
        "beta_exp": pd.to_numeric(df[m["beta"]], errors="coerce"),
        "se_exp": pd.to_numeric(df[m["se"]], errors="coerce"),
    })
    if m["p"]:
        p = pd.to_numeric(df[m["p"]], errors="coerce")
        if "log10" in m["p"].lower():
            p = 10 ** (-p)
        out["p_exp"] = p
    else:
        z = out["beta_exp"] / out["se_exp"]
        out["p_exp"] = 2 * stats.norm.sf(np.abs(z))
    return out.dropna(subset=["pos", "beta_exp", "se_exp"])


def load_outcome(path):
    """Load the Nauffal T1 SOD2-cis outcome window (columns from the Nauffal GWAS)."""
    df = pd.read_csv(path)
    out = pd.DataFrame({
        "chrom": df["Chromosome"].astype(str),
        "pos": pd.to_numeric(df["Position_b37"], errors="coerce"),
        "ea_o": df["Effect_allele"].map(_norm_allele),
        "oa_o": df["Other_allele"].map(_norm_allele),
        "beta_out": pd.to_numeric(df["Beta"], errors="coerce"),
        "se_out": pd.to_numeric(df["SE"], errors="coerce"),
        "p_out": pd.to_numeric(df["P_value"], errors="coerce"),
    })
    return out.dropna(subset=["pos", "beta_out", "se_out"])


def prune_by_distance(df, kb):
    """Distance-based independence proxy (no LD panel available): keep the strongest
    exposure SNP within each `kb` window. Approximates LD clumping at r2<0.01."""
    df = df.sort_values("p_exp")
    kept_pos = []
    keep_idx = []
    for idx, r in df.iterrows():
        if all(abs(r["pos"] - k) > kb * 1000 for k in kept_pos):
            kept_pos.append(r["pos"])
            keep_idx.append(idx)
    return df.loc[keep_idx]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pqtl", required=True, help="SOD2 cis-pQTL summary stats (exposure)")
    ap.add_argument("--outcome", required=True, help="Nauffal T1 SOD2-cis window CSV (outcome)")
    ap.add_argument("--out", required=True, help="output MR-ready CSV")
    ap.add_argument("--pthresh", type=float, default=5e-8, help="instrument p-value threshold")
    ap.add_argument("--clump-kb", type=int, default=250, help="distance-pruning window (kb)")
    a = ap.parse_args()

    exp = load_pqtl(a.pqtl)
    exp = exp[(exp["chrom"] == SOD2_CHR) &
              (exp["pos"] >= SOD2_START - CIS_PAD) &
              (exp["pos"] <= SOD2_END + CIS_PAD)]
    exp = exp[exp["p_exp"] <= a.pthresh]
    if exp.empty:
        sys.exit("No genome-wide-significant SOD2 cis-pQTL instruments. Is this the SOD2 protein file?")
    exp = prune_by_distance(exp, a.clump_kb)

    out = load_outcome(a.outcome)
    m = exp.merge(out, on=["chrom", "pos"])
    if m.empty:
        sys.exit("No chr:pos overlap between pQTL instruments and T1 outcome (build mismatch?).")

    # Align outcome beta to the exposure effect allele; drop allele mismatches.
    def harmonize(r):
        if r["ea_e"] == r["ea_o"] and r["oa_e"] == r["oa_o"]:
            return r["beta_out"]
        if r["ea_e"] == r["oa_o"] and r["oa_e"] == r["ea_o"]:
            return -r["beta_out"]
        return np.nan

    m["beta_out_h"] = m.apply(harmonize, axis=1)
    m = m.dropna(subset=["beta_out_h"])
    if m.empty:
        sys.exit("All instruments dropped during allele harmonization.")

    ready = pd.DataFrame({
        "SNP_chr": m["chrom"], "SNP_pos": m["pos"].astype(int),
        "effect_allele": m["ea_e"], "other_allele": m["oa_e"],
        "beta_exposure": m["beta_exp"], "se_exposure": m["se_exp"],
        "beta_outcome": m["beta_out_h"], "se_outcome": m["se_out"],
    }).sort_values("SNP_pos")
    ready.to_csv(a.out, index=False)

    # Reference IVW MR (fixed-effect): beta = sum(w*bx*by)/sum(w*bx^2), w = 1/se_out^2
    bx = ready["beta_exposure"].values
    by = ready["beta_outcome"].values
    sy = ready["se_outcome"].values
    w = 1.0 / sy ** 2
    denom = np.sum(w * bx ** 2)
    ivw_beta = np.sum(w * bx * by) / denom
    ivw_se = np.sqrt(1.0 / denom)
    z = ivw_beta / ivw_se
    p = 2 * stats.norm.sf(abs(z))

    print(f"Instruments (independent cis-pQTL): {len(ready)}")
    print(f"IVW MR  SOD2 -> myocardial T1 :  beta = {ivw_beta:.4f}   se = {ivw_se:.4f}   p = {p:.3e}")
    print(f"Direction: {'protective (higher SOD2 -> lower T1/fibrosis)' if ivw_beta < 0 else 'risk (higher SOD2 -> higher T1)'}")
    print(f"Paper: Kosmos beta=-0.231 (p=4.23e-13), human beta=-0.258 (p=1.22e-22)")
    print(f"MR-ready CSV written: {a.out}  ({len(ready)} rows)")


if __name__ == "__main__":
    main()
