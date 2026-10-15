# Data report

**Question:** In K562 cells, can the transcriptional response to CRISPRi knockdown of unfolded protein response (UPR) genes not seen in training be predicted from the other knockdowns, using single-cell profiles of K562 cells from independent Perturb-seq experiments as supplementary evidence?

**Tables:** 19 considered — gold **1**, supplementary **6**, unusable **12**, waiting for a person **0**
**Training columns:** 5 of the gold's 5

## Datasets

- `GSM2406675_10X001_cell_identities.csv.gz` — **gold** — review: screen label half (condition column 'guide identity'); its counts matrix is joined at training time; left out cell BC — **a person could salvage it:** join the labels in data/fetched/geo/GSE90546/GSE90546_RAW.tar-contents/GSM2406675_10X001_genes.tsv.gz on barcode
- `GSM2406677_10X005_cell_identities.csv.gz` — **supplementary** — labeled table, demoted to supplementary evidence (the primary gold is data/fetched/geo/GSE90546/GSE90546_RAW.tar-contents/GSM2406675_10X001_cell_identities.csv.gz, named as the primary labeled table)… (the model had said gold); shares 5 column(s) with the gold: read count, UMI count, coverage, good coverage, number of cells — **a person could salvage it:** join the labels in GSM2406677_10X005_cell_identities.csv.gz to the counts in GSM2406677_10X005_genes.tsv.gz / GSM2406677_10X005_matrix.mtx.txt.gz-table.csv on the cell barcode
- `GSM2406681_10X010_cell_identities.csv.gz` — **supplementary** — labeled table, demoted to supplementary evidence (the primary gold is data/fetched/geo/GSE90546/GSE90546_RAW.tar-contents/GSM2406675_10X001_cell_identities.csv.gz, named as the primary labeled table)… (the model had said gold); shares 5 column(s) with the gold: read count, UMI count, coverage, good coverage, number of cells — **a person could salvage it:** join the labels in GSM2406681_10X010_cell_identities.csv.gz to the expression matrix GSM2406681_10X010_genes.tsv.gz / matrix.mtx on the cell barcode
- `GSM2406675_10X001_matrix.mtx.txt.gz-table.csv` — **supplementary** — labeled table, demoted to supplementary evidence (the primary gold is data/fetched/geo/GSE90546/GSE90546_RAW.tar-contents/GSM2406675_10X001_cell_identities.csv.gz, named as the primary labeled table)… (the model had said gold); shares 5 column(s) with the gold: read count, UMI count, coverage, good coverage, number of cells; converted from GSM2406675_10X001_matrix.mtx.txt.gz (2000 of 5768 cells, 35635 of 35635 genes)
- `GSM2406677_10X005_matrix.mtx.txt.gz-table.csv` — **supplementary** — labeled table, demoted to supplementary evidence (the primary gold is data/fetched/geo/GSE90546/GSE90546_RAW.tar-contents/GSM2406675_10X001_cell_identities.csv.gz, named as the primary labeled table)… (the model had said gold); shares 5 column(s) with the gold: read count, UMI count, coverage, good coverage, number of cells; converted from GSM2406677_10X005_matrix.mtx.txt.gz (2000 of 15006 cells, 32738 of 32738 genes)
- `GSM2406681_10X010_matrix.mtx.txt.gz-table.csv` — **supplementary** — labeled table, demoted to supplementary evidence (the primary gold is data/fetched/geo/GSE90546/GSE90546_RAW.tar-contents/GSM2406675_10X001_cell_identities.csv.gz, named as the primary labeled table)… (the model had said gold); shares 5 column(s) with the gold: read count, UMI count, coverage, good coverage, number of cells; converted from GSM2406681_10X010_matrix.mtx.txt.gz (2000 of 65337 cells, 32738 of 32738 genes)
- `GSE90063_k562_umi_wt.txt.gz` — **supplementary** — review: measured transcriptome, usable as evidence for the screen; left out Unnamed: 0 — **a person could salvage it:** transpose it, genes are rows; then join the guide-call labels on barcode
- `GSE153056_ECCITE_metadata.tsv.gz` — **unusable** — it cannot be evidence for the same model: it shares only 0 measured column(s) with the gold, and 3 are needed to correct the same model (the gold has 5 feature(s)) (the model had said gold); left out Unnamed: 0, guide_ID, NT — **a person could salvage it:** join the labels in this file on barcode with the expression matrix from the same GEO series
- `GSE90546_RAW.tar` — **unusable** — review: not a table: 0 parsed column(s). A table needs at least a feature column and something else
- `GSM2406675_10X001_barcodes.tsv.gz` — **unusable** — review: not a table: 1 parsed column(s). A table needs at least a feature column and something else
- `GSM2406675_10X001_matrix.mtx.txt.gz` — **unusable** — review: not a table: 1 parsed column(s). A table needs at least a feature column and something else
- `GSM2406677_10X005_barcodes.tsv.gz` — **unusable** — review: not a table: 1 parsed column(s). A table needs at least a feature column and something else
- `GSM2406677_10X005_matrix.mtx.txt.gz` — **unusable** — review: not a table: 1 parsed column(s). A table needs at least a feature column and something else
- `GSM2406681_10X010_barcodes.tsv.gz` — **unusable** — review: not a table: 1 parsed column(s). A table needs at least a feature column and something else
- `GSM2406681_10X010_matrix.mtx.txt.gz` — **unusable** — review: not a table: 1 parsed column(s). A table needs at least a feature column and something else
- `GSE153056_CITE_metadata.tsv.gz` — **unusable** — review: This is a per-cell CITE-seq metadata table (barcodes with RNA/ADT/HTO QC columns) from GSE153056. It has no gene-expression columns and no perturbation/guide identity column, so it is neither…; left out Unnamed: 0 — **a person could salvage it:** join the labels in the GSE153056 guide-calling table on barcode, and pair with the matching counts matrix
- `GSM2406675_10X001_genes.tsv.gz` — **unusable** — review: This is a genes-by-cells feature file: two columns mapping Ensembl gene IDs to gene symbols, one row per gene, no header, no cell barcodes and no perturbation/guide column. It is the gene ann…; left out ENSG00000243485, MIR1302-10 — **a person could salvage it:** use it as the row annotation for GSM2406675_10X001_matrix.mtx.txt.gz, joining labels from GSM2406675_10X001_cell_identities.csv.gz on barcode
- `GSM2406677_10X005_genes.tsv.gz` — **unusable** — review: This is a genes-by-cells feature file: two columns mapping Ensembl gene IDs to gene symbols, one row per gene, no header, no cell barcodes and no perturbation/guide column. It is the gene axi…; left out ENSG00000243485, MIR1302-10 — **a person could salvage it:** join the labels in data/fetched/geo/GSE90546/GSE90546_RAW.tar-contents/GSM2406677_10X005_cell_identities.csv.gz on barcode and use this file as the gene axis of the matching counts matrix
- `GSM2406681_10X010_genes.tsv.gz` — **unusable** — review: This is a genes-by-cells feature file (gene id and gene symbol per row) from a 10x Perturb-seq experiment; it has no per-cell rows and no perturbation column, so it cannot be a screen on its …; left out ENSG00000243485, MIR1302-10 — **a person could salvage it:** join the labels in data/fetched/geo/GSE90546/GSE90546_RAW.tar-contents/GSM2406681_10X010_cell_identities.csv.gz on barcode and use this file as the gene index for the matching matrix
- `geo://GSE194122` — **not downloaded** — the model did not keep it
- `geo://GSE132080` — **not downloaded** — the model did not keep it
- `geo://GSE153056#matrix/GSE153056-GPL24676_series_matrix.txt.gz` — **not downloaded** — error: SourceError: GSE153056's series matrix has a header and no rows, which is normal for RNA-seq series. Supplementary files (smallest first): suppl/filelist.txt (1.7 kB), matrix/GSE153056-GPL1857…

## Round 1 -- the labeled table

1. **proposed** `geo://GSE90546` (confidence 0.95) — Directly matches the research question: K562 cells, CRISPRi knockdown of UPR genes, single-cell Perturb-seq. Provides the labeled screen needed to predict trans
1. **proposed** `geo://GSE194122` (confidence 0.85) — Serves as supplementary evidence: same tissue and cell type (K562), contains control/untreated cells, and measures a comparable gene panel, enabling co-expressi
1. **listing** `geo://GSE90546` — kept
   - 4 file(s) on offer, smallest 2.6 kB, total 942.0 MB # about: title: A multiplexed single-cell CRISPR screening platform enables systematic dissection of the unfolded protein response | summary: Funct…
1. **listing** `geo://GSE194122` — kept
   - 3 file(s) on offer, smallest 3.9 kB, total 3.3 GB # about: title: A sandbox for prediction and integration of DNA, RNA, and proteins in single cells | summary: Single-cell multiomics data collected f…
1. **chosen** `geo://GSE90546#suppl/GSE90546_RAW.tar` — Contains the raw single-cell Perturb-seq measurements (cells x genes) for K562 CRISPRi UPR knockdowns, the labeled data needed to predict transcriptional responses.
1. **dropped** `geo://GSE194122` — the model did not keep it
1. **downloaded** `geo://GSE90546#suppl/GSE90546_RAW.tar` → GSE90546_RAW.tar, GSE90546_RAW.tar-contents/GSM2406675_10X001_barcodes.tsv.gz, GSE90546_RAW.tar-contents/GSM2406675_10X001_cell_identities.csv.gz, GSE90546_RAW.tar-contents/GSM2406675_10X001_genes.tsv.gz
1. **screened out** `GSE90546_RAW.tar` — not readable as a delimited text table: UnicodeDecodeError: 'utf-8' codec can't decode byte 0x8b in position 513: invalid start byte
1. **screened out** `GSM2406675_10X001_barcodes.tsv.gz` — parsed as a single column; this file is probably not a plain table (a matrix with a preamble, or a non-tabular artifact)
1. **screened out** `GSM2406675_10X001_matrix.mtx.txt.gz` — converted to GSM2406675_10X001_matrix.mtx.txt.gz-table.csv; that table is the candidate being reviewed
1. **screened out** `GSM2406677_10X005_barcodes.tsv.gz` — parsed as a single column; this file is probably not a plain table (a matrix with a preamble, or a non-tabular artifact)
1. **screened out** `GSM2406677_10X005_matrix.mtx.txt.gz` — converted to GSM2406677_10X005_matrix.mtx.txt.gz-table.csv; that table is the candidate being reviewed
1. **screened out** `GSM2406681_10X010_barcodes.tsv.gz` — parsed as a single column; this file is probably not a plain table (a matrix with a preamble, or a non-tabular artifact)
1. **screened out** `GSM2406681_10X010_matrix.mtx.txt.gz` — converted to GSM2406681_10X010_matrix.mtx.txt.gz-table.csv; that table is the candidate being reviewed

## Round 2 -- supplementary evidence for that table

1. **proposed** `geo://GSE132080` (confidence 0.85) — Same measurement type as the labeled table: single-cell gene expression counts across the same gene columns (e.g., A1BG, AARS, AATF, etc.) for K562 cells under
1. **proposed** `geo://GSE90063` (confidence 0.80) — Shares the same measurement columns: per-cell gene expression counts for the same gene set in K562 cells. Offers additional independent CRISPRi knockdown profil
1. **proposed** `geo://GSE153056` (confidence 0.75) — Same measurement type and cell line: single-cell expression counts across the same gene columns. Provides an independent cohort of K562 CRISPRi knockdowns that
1. **listing** `geo://GSE132080` — kept
   - 6 file(s) on offer, smallest 2.8 kB, total 336.7 MB # about: title: Titrating gene expression with allelic series of CRISPR guide RNAs | summary: Cellular phenotypes arise from the degrees to which d…
1. **listing** `geo://GSE90063` — kept
   - 7 file(s) on offer, smallest 2.0 kB, total 1.4 GB # about: title: Perturb-seq: Dissecting molecular circuits with scalable single cell RNA profiling of pooled genetic screens | summary: Methods: We c…
1. **listing** `geo://GSE153056` — kept
   - 7 file(s) on offer, smallest 1.7 kB, total 124.0 MB # about: title: Characterizing the molecular regulation of inhibitory immune checkpoints with multi-modal single-cell screens. | summary: The expre…
1. **chosen** `geo://GSE90063#suppl/GSE90063_k562_umi_wt.txt.gz` — This is the K562 single-cell UMI count matrix from an independent Perturb-seq experiment, giving per-cell gene expression observations needed as supplementary evidence for predict…
1. **chosen** `geo://GSE153056#suppl/GSE153056_ECCITE_metadata.tsv.gz` — Provides per-cell metadata linking K562 CRISPRi perturbations to cell identities, enabling use of the independent ECCITE-seq knockdown cohort as supplementary evidence.
1. **listing** `geo://GSE153056#matrix/GSE153056-GPL24676_series_matrix.txt.gz` — kept
   - keeping GSE153056-GPL24676_series_matrix.txt.gz as well -- a screen's measurements live in the counts file, and the label table does not carry them
1. **listing** `geo://GSE153056#suppl/GSE153056_CITE_metadata.tsv.gz` — kept
   - keeping GSE153056_CITE_metadata.tsv.gz as well -- this task's labels (a perturbation screen's condition/guide) usually live in the annotated file, not the raw matrix
1. **dropped** `geo://GSE132080` — the model did not keep it
1. **downloaded** `geo://GSE90063#suppl/GSE90063_k562_umi_wt.txt.gz` → GSE90063_k562_umi_wt.txt.gz
1. **downloaded** `geo://GSE153056#suppl/GSE153056_ECCITE_metadata.tsv.gz` → GSE153056_ECCITE_metadata.tsv.gz
1. **download failed** `geo://GSE153056#matrix/GSE153056-GPL24676_series_matrix.txt.gz` — error: SourceError: GSE153056's series matrix has a header and no rows, which is normal for RNA-seq series. Supplementary files (smallest first): suppl/filelist.txt (1.7 kB), matrix/GSE153056-GPL1857…
1. **downloaded** `geo://GSE153056#suppl/GSE153056_CITE_metadata.tsv.gz` → GSE153056_CITE_metadata.tsv.gz
1. **screening** kept all 3 table(s)

**Refusals by cause:** not a table (fewer than two columns) ×7, the model judged it unusable ×4, cannot supply the columns the gold has ×1
