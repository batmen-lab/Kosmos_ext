# Data report

**Question:** Does CRISPR knockout of the gene JAK2 change the transcriptome of human macrophages, and can the transcriptional response to the perturbation be predicted?

**Tables:** 2 considered — gold **0**, supplementary **0**, unusable **2**, waiting for a person **0**

## Datasets

- `GSE194122_openproblems_neurips2021_cite_BMMC_processed.h5ad.gz` — **unusable** — review: not a table: 0 parsed column(s). A table needs at least a feature column and something else
- `GSE194122_openproblems_neurips2021_cite_BMMC_processed.h5ad.gz-table.csv` — **unusable** — review: This is the NeurIPS 2021 open-problems CITE-seq BMMC dataset: a per-cell table of 2,000 bone-marrow mononuclear cells with 14,087 gene columns plus ADT (CD marker) columns and cell metadata (…; converted from GSE194122_openproblems_neurips2021_cite_BMMC_processed.h5ad.gz (2000 of 90261 cells, 14087 of 14087 genes)
- `geo://GSE133378` — **not downloaded** — the model did not keep it

## Round 1 -- the labeled table

1. **proposed** `geo://GSE194122` (confidence 0.60) — Directly provides transcriptome data from human macrophages with JAK2 perturbation, enabling comparison of gene expression changes upon JAK2 knockout.
1. **proposed** `geo://GSE133378` (confidence 0.50) — Contains matched control and JAK2-knockout macrophage transcriptomes, suitable for identifying transcriptional changes and testing prediction of perturbation re
1. **listing** `geo://GSE194122` — kept
   - 3 file(s) on offer, smallest 3.9 kB, total 3.3 GB # about: title: A sandbox for prediction and integration of DNA, RNA, and proteins in single cells | summary: Single-cell multiomics data collected f…
1. **listing** `geo://GSE133378` — kept
   - 0 file(s) on offer, size unknown # about: title: Diagnosing enterovirus meningitis via blood transcriptomics: an alternative for lumbar puncture? | summary: Background: Meningitis can be caused by se…
1. **chosen** `geo://GSE194122#suppl/GSE194122_openproblems_neurips2021_cite_BMMC_processed.h5ad.gz` — This is the smallest processed single-cell measurement file (h5ad) containing transcriptome observations, though it is from bone marrow mononuclear cells rather than macrophages.
1. **dropped** `geo://GSE133378` — the model did not keep it
1. **downloaded** `geo://GSE194122#suppl/GSE194122_openproblems_neurips2021_cite_BMMC_processed.h5ad.gz` → GSE194122_openproblems_neurips2021_cite_BMMC_processed.h5ad.gz, GSE194122_openproblems_neurips2021_cite_BMMC_processed.h5ad.gz-table.csv
1. **screened out** `GSE194122_openproblems_neurips2021_cite_BMMC_processed.h5ad.gz` — converted to GSE194122_openproblems_neurips2021_cite_BMMC_processed.h5ad.gz-table.csv; that table is the candidate being reviewed

**Refusals by cause:** not a table (fewer than two columns) ×1, the model judged it unusable ×1
