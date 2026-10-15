# Data report

**Question:** In K562 cells, can the transcriptional response to CRISPR activation (CRISPRa) of two genes together be predicted from the responses to each gene activated alone?

**Tables:** 3 considered — gold **1**, supplementary **0**, unusable **2**, waiting for a person **0**
**Training columns:** 6 of the gold's 6

## Datasets

- `GSE133344_filtered_cell_identities.csv.gz` — **gold** — review: screen label half (condition column 'guide_identity'); its counts matrix is joined at training time; left out cell_barcode — **a person could salvage it:** join the labels in this file to the GSE133344 expression/counts matrix on cell_barcode
- `GSE133344_filtered_matrix.mtx.gz` — **unusable** — review: not a table: 1 parsed column(s). A table needs at least a feature column and something else
- `GSE133344_raw_cell_identities.csv.gz` — **unusable** — not a table: 0 parsed column(s). A table needs at least a feature column and something else
- `geo://GSE194122` — **not downloaded** — the model did not keep it
- `geo://GSE133344#suppl/GSE133344_RAW.tar` — **not downloaded** — error: SourceError: https://ftp.ncbi.nlm.nih.gov/geo/series/GSE133nnn/GSE133344/suppl/GSE133344_RAW.tar exceeds the 2,147,483,648-byte cap; raise --max-bytes or fetch a smaller file
- `hf://cellxgene-census/cellxgene-census` — **not downloaded** — the model did not keep it

## Round 1 -- the labeled table

1. **proposed** `geo://GSE133344` (confidence 0.95) — This is the gold-standard CRISPRa Perturb-seq screen in K562 cells that directly measures transcriptional responses to activating one gene alone and to activati
1. **proposed** `geo://GSE194122` (confidence 0.60) — Provides the control/untreated K562 cells needed as a baseline for a co-expression graph and for normalizing the CRISPRa perturbation responses; same tissue and
1. **listing** `geo://GSE133344` — kept
   - 11 file(s) on offer, smallest 3.6 kB, total 12.4 GB # about: title: Exploring genetic interaction manifolds constructed from rich phenotypes | summary: Rich phenotyping using single-cell RNA sequenci…
1. **listing** `geo://GSE194122` — kept
   - 3 file(s) on offer, smallest 3.9 kB, total 3.3 GB # about: title: A sandbox for prediction and integration of DNA, RNA, and proteins in single cells | summary: Single-cell multiomics data collected f…
1. **chosen** `geo://GSE133344#suppl/GSE133344_filtered_matrix.mtx.gz` — This is the filtered count matrix of single-cell CRISPRa Perturb-seq measurements in K562 cells, containing the transcriptional observations needed to compare single vs. combinato…
1. **chosen** `geo://GSE133344#suppl/GSE133344_filtered_cell_identities.csv.gz` — This file maps each cell barcode to its CRISPRa perturbation identity, which is required to group cells by single-gene or two-gene activation for the prediction analysis.
1. **listing** `geo://GSE133344#suppl/GSE133344_RAW.tar` — kept
   - keeping GSE133344_RAW.tar as well -- a screen's measurements live in the counts file, and the label table does not carry them
1. **listing** `geo://GSE133344#suppl/GSE133344_filtered_cell_identities.csv.gz` — kept
   - keeping GSE133344_filtered_cell_identities.csv.gz as well -- this task's labels (a perturbation screen's condition/guide) usually live in the annotated file, not the raw matrix
1. **dropped** `geo://GSE194122` — the model did not keep it
1. **downloaded** `geo://GSE133344#suppl/GSE133344_filtered_matrix.mtx.gz` → GSE133344_filtered_matrix.mtx.gz
1. **download failed** `geo://GSE133344#suppl/GSE133344_RAW.tar` — error: SourceError: https://ftp.ncbi.nlm.nih.gov/geo/series/GSE133nnn/GSE133344/suppl/GSE133344_RAW.tar exceeds the 2,147,483,648-byte cap; raise --max-bytes or fetch a smaller file
1. **downloaded** `geo://GSE133344#suppl/GSE133344_filtered_cell_identities.csv.gz` → GSE133344_filtered_cell_identities.csv.gz
1. **downloaded** `geo://GSE133344#suppl/GSE133344_filtered_cell_identities.csv.gz` → GSE133344_filtered_cell_identities.csv.gz
1. **download failed** `geo://GSE133344#suppl/GSE133344_RAW.tar` — error: SourceError: https://ftp.ncbi.nlm.nih.gov/geo/series/GSE133nnn/GSE133344/suppl/GSE133344_RAW.tar exceeds the 2,147,483,648-byte cap; raise --max-bytes or fetch a smaller file
1. **screened out** `GSE133344_filtered_matrix.mtx.gz` — parsed as a single column; this file is probably not a plain table (a matrix with a preamble, or a non-tabular artifact)

## Round 2 -- supplementary evidence for that table

1. **proposed** `geo://GSE133344` (confidence 0.55) — Same study and same K562 CRISPRa perturbation platform; other supplementary files in this series provide per-cell read_count, UMI_count, coverage, gemgroup, goo
1. **proposed** `geo://GSE194122` (confidence 0.35) — Provides per-cell read_count, UMI_count, coverage, and cell identity measurements for other cohorts/sites, matching the same measured column types
1. **proposed** `hf://cellxgene-census/cellxgene-census` (confidence 0.30) — Supplies per-cell read_count, UMI_count, coverage, and cell identity columns for many other studies/cohorts, aligning with the measured columns
1. **listing** `geo://GSE133344` — kept
   - 11 file(s) on offer, smallest 3.6 kB, total 12.4 GB # about: title: Exploring genetic interaction manifolds constructed from rich phenotypes | summary: Rich phenotyping using single-cell RNA sequenci…
1. **listing** `geo://GSE194122` — kept
   - 3 file(s) on offer, smallest 3.9 kB, total 3.3 GB # about: title: A sandbox for prediction and integration of DNA, RNA, and proteins in single cells | summary: Single-cell multiomics data collected f…
1. **listing** `hf://cellxgene-census/cellxgene-census` — kept
   - 0 file(s) on offer, the repository did not answer a listing for this reference
1. **chosen** `geo://GSE133344#suppl/GSE133344_raw_cell_identities.csv.gz` — Per-cell identities for the raw K562 CRISPRa single-cell experiment, giving the perturbation labels needed to compare single vs combined gene activation.
1. **listing** `geo://GSE133344#suppl/GSE133344_RAW.tar` — kept
   - keeping GSE133344_RAW.tar as well -- a screen's measurements live in the counts file, and the label table does not carry them
1. **listing** `geo://GSE133344#suppl/GSE133344_filtered_cell_identities.csv.gz` — kept
   - keeping GSE133344_filtered_cell_identities.csv.gz as well -- this task's labels (a perturbation screen's condition/guide) usually live in the annotated file, not the raw matrix
1. **dropped** `geo://GSE194122` — the model did not keep it
1. **dropped** `hf://cellxgene-census/cellxgene-census` — the model did not keep it
1. **downloaded** `geo://GSE133344#suppl/GSE133344_raw_cell_identities.csv.gz` → GSE133344_raw_cell_identities.csv.gz
1. **download failed** `geo://GSE133344#suppl/GSE133344_RAW.tar` — error: SourceError: https://ftp.ncbi.nlm.nih.gov/geo/series/GSE133nnn/GSE133344/suppl/GSE133344_RAW.tar exceeds the 2,147,483,648-byte cap; raise --max-bytes or fetch a smaller file
1. **downloaded** `geo://GSE133344#suppl/GSE133344_filtered_cell_identities.csv.gz` → GSE133344_filtered_cell_identities.csv.gz
1. **screened out** `GSE133344_raw_cell_identities.csv.gz` — not readable as a delimited text table: EmptyDataError: No columns to parse from file

**Refusals by cause:** not a table (fewer than two columns) ×2
