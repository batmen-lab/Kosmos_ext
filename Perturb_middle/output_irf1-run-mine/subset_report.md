# Subset report — irf1-run-mine

- objective: Does CRISPR knockout of the gene IRF1 change the transcriptome of human macrophages, and can the transcriptional response to the perturbation be predicted?
- stage-1 gold: `data/fetched/geo/GSE153056/GSE153056_ECCITE_metadata.tsv.gz`
- condition column: `gene` (barcodes in `Unnamed: 0`)

## Subset (gold)

- cells in the label table: 20,729
- cells kept: 20,729
  - `ATF2`: 1,102
  - `BRD4`: 341
  - `CAV1`: 882
  - `CD86`: 1,197
  - `CMTM6`: 817
  - `CUL3`: 274
  - `ETV7`: 841
  - `IFNGR1`: 1,341
  - `IFNGR2`: 1,213
  - `IRF1`: 1,073
  - `IRF7`: 614
  - `JAK2`: 1,024
  - `MARCH8`: 1,003
  - `MYC`: 109
  - `NFKBIA`: 787
  - `NT`: 2,386
  - `PDCD1LG2`: 617
  - `POU2F2`: 662
  - `SMAD4`: 659
  - `SPI1`: 48
  - `STAT1`: 444
  - `STAT2`: 661
  - `STAT3`: 493
  - `STAT5A`: 659
  - `TNFRSF14`: 964
  - `UBE2L6`: 518

## Pre-processing

1. normalise `log1p_cpm` (target_sum 10000)
2. HVG: top 2000 by variance -> 2009 (from `GSM4633614_ECCITE_cDNA_counts.tsv.gz`)
3. z-score each gene across all cells
4. clip negatives to zero

Top HVG: CXCL10, SPP1, S100A8, CCL2, APOC1, S100A9, IFI27, CD74, HLA-DRA, HSP90B1, CXCL9, HSPA5, IL8, FN1, RGS16

## Training inputs (stage 2)

- `Perturb_middle/output_irf1-run-mine/gold_subset.csv.gz` — labels
- `Perturb_middle/output_irf1-run-mine/supp_1_GSM4633614_ECCITE_cDNA_counts.csv.gz` — 20,729 cells x 2009 genes (from `GSM4633614_ECCITE_cDNA_counts.tsv.gz`)
- `Perturb_middle/output_irf1-run-mine/supp_2_GSM4633605_CITE_cDNA_counts.csv.gz` — 7,566 cells x 2009 genes (from `GSM4633605_CITE_cDNA_counts.tsv.gz`)
- `Perturb_middle/output_irf1-run-mine/supp_3_GSM4633608_ECCITE_Arrayed_cDNA_counts.csv.gz` — 16,695 cells x 2009 genes (from `GSM4633608_ECCITE_Arrayed_cDNA_counts.tsv.gz`)
- `Perturb_middle/output_irf1-run-mine/plan.json` — gold + supplementary, backend joins on barcode

## Warnings

- GSM4633612_ECCITE_Arrayed_GO_CITE03_counts.tsv.gz: shares <10 panel genes; skipped
- GSM4633613_ECCITE_Arrayed_GO_LENTI_counts.tsv.gz: shares <10 panel genes; skipped
