# Prepare report — Adamson UPR (GSE90546)

**Question:** In K562 cells, can the transcriptional response to CRISPRi knockdown of unfolded protein response (UPR) genes not seen in training be predicted from the other knockdowns, using single-cell profiles of K562 cells from independent Perturb-seq experiments as supplementary evidence?

## Gold cell filter

| step | cells |
|---|---|
| cells in the matrix | 65,337 |
| with a guide call | 65,257 |
| with good coverage | 59,219 |
| with exactly one guide | 50,467 |
| with a named guide (not `*`) | 50,440 |

- 50,440 cells, 5,241 controls (`63(mod)`, `Gal4-4(mod)`)
- 82 perturbed genes; cells per perturbation: min 185, median 507
- perturbed genes not in the measured genes: none
- panel: 5,014 genes (top HVG + 14 perturbed genes); log1p(CPM 10000)

## Supplementary

- `Perturb_middle/output_adamson_upr/supp_dixit_k562_wt.parquet` ← `data/fetched/geo/GSE90063/GSE90063_k562_umi_wt.txt.gz`
- 5,409 unperturbed K562 cells (independent lab/run); 5,014 of 5,014 panel genes measured

## Study

- held-out unit: perturbed genes never seen in training (`unseen_single`), test 25%, validation 20% of the rest
- training batch size 128
