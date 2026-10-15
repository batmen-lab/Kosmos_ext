# Data report

**Question:** Does CRISPR knockout of the gene JAK2 change the transcriptome of human macrophages, and can the transcriptional response to the perturbation be predicted?

**Tables:** 13 considered — gold **1**, supplementary **0**, unusable **12**, waiting for a person **0**
**Training columns:** 22 of the gold's 22

## Datasets

- `GSE153056_ECCITE_metadata.tsv.gz` — **gold** — target column 'guide_ID' is present with 22 feature(s) (11 categorical); this table can train, select and test; 2 column(s) are not features (free text, a row index, or empty), e.g. ['Unnamed: 0', 'N… (the model had said unusable); left out Unnamed: 0, NT — **a person could salvage it:** join the labels in data/fetched/geo/GSE153056/GSE153056_RAW.tar-contents/GSM4633614_ECCITE_cDNA_counts.tsv.gz on barcode
- `GSE171406_RAW.tar` — **unusable** — review: not a table: 0 parsed column(s). A table needs at least a feature column and something else
- `GSM5224978_A1.tif.gz` — **unusable** — review: not a table: 0 parsed column(s). A table needs at least a feature column and something else
- `GSM5224979_B1.tif.gz` — **unusable** — review: not a table: 0 parsed column(s). A table needs at least a feature column and something else
- `GSM5224980_C1.tif.gz` — **unusable** — review: not a table: 0 parsed column(s). A table needs at least a feature column and something else
- `GSM5224981_D1.tif.gz` — **unusable** — review: not a table: 0 parsed column(s). A table needs at least a feature column and something else
- `GSE194122_openproblems_neurips2021_cite_BMMC_processed.h5ad.gz` — **unusable** — review: not a table: 0 parsed column(s). A table needs at least a feature column and something else
- `GSM4633614_ECCITE_cDNA_counts.tsv.gz` — **unusable** — review: This is a genes-by-cells count matrix (20,730 columns: first column holds gene names, remaining columns are cell barcodes like l1_AAACCTGAGCCAGAAC). It carries the transcriptome but no pertur…; left out Unnamed: 0 — **a person could salvage it:** join the labels in data/fetched/geo/GSE153056/GSE153056_ECCITE_metadata.tsv.gz on barcode
- `GSM5224978_counts_A1_S1.csv.gz` — **unusable** — review: This is a genes-by-cells count matrix (rows are genes, columns are cell barcodes) with no per-cell perturbation/condition column. The first line is data (barcodes as column names, no header r…; left out Unnamed: 0 — **a person could salvage it:** join the labels in the GSE171406 cell-metadata/guide-call table on barcode
- `GSM5224979_counts_B1_S2.csv.gz` — **unusable** — review: This is a genes-by-cells count matrix (rows are genes like Xkr4, Gm1992; columns are cell barcodes). It carries the transcriptome but no perturbation/condition column, so it cannot be a per-c…; left out Unnamed: 0 — **a person could salvage it:** join the labels in the GSE171406 metadata table on barcode
- `GSM5224980_counts_C1_S3.csv.gz` — **unusable** — review: This is a genes-by-cells count matrix (rows are genes like Xkr4, Gm1992; columns are cell barcodes) with no perturbation/condition column. It is one of the GSE171406 per-sample count matrices…; left out Unnamed: 0 — **a person could salvage it:** join the guide/condition labels from the GSE171406 metadata or guide-call table on barcode, then transpose so cells are rows
- `GSM5224981_counts_D1_S4.csv.gz` — **unusable** — review: This is a genes-by-cells count matrix (rows are genes like Xkr4, Gm1992; columns are cell barcodes) with no perturbation/condition column. It carries the transcriptome but not the perturbatio…; left out Unnamed: 0 — **a person could salvage it:** transpose it, genes are rows; then join the perturbation labels for this sample on barcode
- `GSE194122_openproblems_neurips2021_cite_BMMC_processed.h5ad.gz-table.csv` — **unusable** — review: This is the NeurIPS 2021 open-problems CITE-seq BMMC dataset (GSE194122): a per-cell table of 2,000 sampled bone-marrow mononuclear cells with 14,087 gene columns plus ADT protein columns and…; converted from GSE194122_openproblems_neurips2021_cite_BMMC_processed.h5ad.gz (2000 of 90261 cells, 14087 of 14087 genes)

## Round 1 -- the labeled table

1. **proposed** `geo://GSE194122` (confidence 0.60) — Provides an unperturbed human macrophage/monocyte atlas with control cells, serving as the co-expression/control reference for the same tissue and cell type nee
1. **proposed** `geo://GSE171406` (confidence 0.50) — A CRISPR knockout screen in the relevant cell type that can supply the labeled perturbation data for JAK2 knockout and its transcriptome effects.
1. **listing** `geo://GSE194122` — kept
   - 2 file(s) on offer, smallest 587.0 MB, total 3.3 GB # about: title: A sandbox for prediction and integration of DNA, RNA, and proteins in single cells | summary: Single-cell multiomics data collected…
1. **listing** `geo://GSE171406` — kept
   - 8 file(s) on offer, smallest 2.8 kB, total 721.0 MB # about: title: Integration of spatial and single cell transcriptomics localizes epithelial-immune cross-talk in kidney injury | summary: Single ce…
1. **chosen** `geo://GSE194122#suppl/GSE194122_openproblems_neurips2021_cite_BMMC_processed.h5ad.gz` — This processed single-cell h5ad provides the unperturbed human macrophage/monocyte control transcriptomes needed as the reference for predicting the JAK2 knockout response.
1. **listing** `geo://GSE171406#suppl/GSE171406_RAW.tar` — kept
   - keeping GSE171406_RAW.tar as well -- a screen's measurements live in the counts file, and the label table does not carry them
1. **downloaded** `geo://GSE194122#suppl/GSE194122_openproblems_neurips2021_cite_BMMC_processed.h5ad.gz` → GSE194122_openproblems_neurips2021_cite_BMMC_processed.h5ad.gz, GSE194122_openproblems_neurips2021_cite_BMMC_processed.h5ad.gz-table.csv
1. **downloaded** `geo://GSE171406#suppl/GSE171406_RAW.tar` → GSE171406_RAW.tar, GSE171406_RAW.tar-contents/GSM5224978_A1.tif.gz, GSE171406_RAW.tar-contents/GSM5224978_counts_A1_S1.csv.gz, GSE171406_RAW.tar-contents/GSM5224979_B1.tif.gz
1. **screened out** `GSE194122_openproblems_neurips2021_cite_BMMC_processed.h5ad.gz` — converted to GSE194122_openproblems_neurips2021_cite_BMMC_processed.h5ad.gz-table.csv; that table is the candidate being reviewed
1. **screened out** `GSE171406_RAW.tar` — not readable as a delimited text table: UnicodeDecodeError: 'utf-8' codec can't decode byte 0x8b in position 513: invalid start byte
1. **screened out** `GSM5224978_A1.tif.gz` — not readable as a delimited text table: UnicodeDecodeError: 'utf-8' codec can't decode byte 0xea in position 6: invalid continuation byte
1. **screened out** `GSM5224979_B1.tif.gz` — not readable as a delimited text table: UnicodeDecodeError: 'utf-8' codec can't decode byte 0x8c in position 6: invalid start byte
1. **screened out** `GSM5224980_C1.tif.gz` — not readable as a delimited text table: UnicodeDecodeError: 'utf-8' codec can't decode byte 0x9c in position 4: invalid start byte
1. **screened out** `GSM5224981_D1.tif.gz` — not readable as a delimited text table: UnicodeDecodeError: 'utf-8' codec can't decode byte 0xe8 in position 4: invalid continuation byte

**Refusals by cause:** the model judged it unusable ×6, not a table (fewer than two columns) ×6
