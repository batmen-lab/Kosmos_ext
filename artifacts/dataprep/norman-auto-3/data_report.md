# Data report

**Question:** In K562 cells, can the transcriptional response to CRISPR activation (CRISPRa) of two genes together be predicted from the responses to each gene activated alone?

**Tables:** 9 considered — gold **1**, supplementary **0**, unusable **8**, waiting for a person **0**
**Training columns:** 33667 of the gold's 33667

## Datasets

- `GSE133344_filtered_matrix.mtx.gz-table.csv` — **gold** — target column 'guide_identity' is present with 33667 feature(s) (32669 categorical); this table can train, select and test; converted from GSE133344_filtered_matrix.mtx.gz (2000 of 111668 cells, 33694 of 33694 genes)
- `GSE133344_filtered_cell_identities.csv.gz` — **unusable** — it cannot be evidence for the same model: it shares only 6 measured column(s) with the gold, and 16834 are needed to correct the same model (the gold has 33667 feature(s)) (the model had said gold); left out cell_barcode — **a person could salvage it:** join the labels in GSE133344_filtered_cell_identities.csv.gz on cell_barcode to the counts in GSE133344_filtered_matrix.mtx.gz-table.csv
- `GSE133344_raw_cell_identities.csv.gz` — **unusable** — it cannot be evidence for the same model: it shares only 6 measured column(s) with the gold, and 16834 are needed to correct the same model (the gold has 33667 feature(s)) (the model had said gold); left out cell_barcode — **a person could salvage it:** join the labels in this file to the expression matrix (GSE133344_filtered_matrix.mtx.gz / GSE133344_raw_genes.tsv.gz) on cell_barcode
- `GSE133344_filtered_barcodes.tsv.gz` — **unusable** — review: not a table: 1 parsed column(s). A table needs at least a feature column and something else
- `GSE133344_filtered_matrix.mtx.gz` — **unusable** — review: not a table: 1 parsed column(s). A table needs at least a feature column and something else
- `GSE133344_raw_barcodes.tsv.gz` — **unusable** — review: not a table: 1 parsed column(s). A table needs at least a feature column and something else
- `GSE133344_raw_matrix.mtx.gz` — **unusable** — review: not a table: 1 parsed column(s). A table needs at least a feature column and something else
- `GSE133344_filtered_genes.tsv.gz` — **unusable** — review: Two-column tab-delimited file of Ensembl gene IDs and gene symbols; it is a gene annotation/feature list, not a per-cell table. No guide_identity column and no cells.; left out ENSG00000243485, RP11-34P13.3
- `GSE133344_raw_genes.tsv.gz` — **unusable** — review: Two-column gene annotation table (Ensembl gene id and gene symbol), one row per gene, no cells and no perturbation column. It is the gene list for the GSE133344 matrix, not a per-cell screen.; left out ENSG00000243485, RP11-34P13.3 — **a person could salvage it:** use it as the gene index for GSE133344_filtered_matrix.mtx.gz and join the labels in GSE133344_filtered_cell_identities.csv.gz on barcode
- `geo://GSE194122` — **not downloaded** — the model did not keep it
- `geo://GSE152988` — **not downloaded** — the model did not keep it
- `hf://genome-scale-tools/perturb-seq` — **not downloaded** — the model did not keep it
- `geo://GSE133344#matrix/GSE133344_series_matrix.txt.gz` — **not downloaded** — error: SourceError: GSE133344's series matrix has a header and no rows, which is normal for RNA-seq series. Supplementary files (smallest first): matrix/GSE133344_series_matrix.txt.gz (3.6 kB), suppl…

## Round 1 -- the labeled table

1. **proposed** `geo://GSE133344` (confidence 0.95) — This is the exact labeled CRISPR activation screen in K562 cells containing single-gene and paired-gene perturbations needed to test whether combinatorial CRISP
1. **proposed** `geo://GSE194122` (confidence 0.70) — Provides control or untreated K562 cells of the same tissue and cell type for co-expression graph supplementary evidence; it measures the same gene panel and qu
1. **proposed** `geo://GSE133344` (confidence 0.95) — This is the Norman 2019 CRISPR activation Perturb-seq dataset in K562, the exact assay (CRISPRa/over-expression) and cell type needed to test whether paired-gen
1. **proposed** `geo://GSE194122` (confidence 0.60) — Provides the same tissue/cell type control or untreated cells needed as supplementary evidence for a co-expression graph; it measures the same gene panel and qu
1. **listing** `geo://GSE133344` — kept
   - 11 file(s) on offer, smallest 3.6 kB, total 12.4 GB # about: title: Exploring genetic interaction manifolds constructed from rich phenotypes | summary: Rich phenotyping using single-cell RNA sequenci…
1. **listing** `geo://GSE194122` — kept
   - 3 file(s) on offer, smallest 3.9 kB, total 3.3 GB # about: title: A sandbox for prediction and integration of DNA, RNA, and proteins in single cells | summary: Single-cell multiomics data collected f…
1. **chosen** `geo://GSE133344#suppl/GSE133344_filtered_matrix.mtx.gz` — This is the smallest expression matrix file containing the single-cell transcriptional measurements for the K562 CRISPRa single and combinatorial perturbations needed to test the …
1. **listing** `geo://GSE133344#suppl/GSE133344_filtered_cell_identities.csv.gz` — kept
   - keeping GSE133344_filtered_cell_identities.csv.gz as well -- this task's labels (a perturbation screen's condition/guide) usually live in the annotated file, not the raw matrix
1. **listing** `geo://GSE133344#suppl/GSE133344_filtered_genes.tsv.gz` — kept
   - keeping GSE133344_filtered_genes.tsv.gz -- GSE133344_filtered_matrix.mtx.gz cannot be read without it
1. **listing** `geo://GSE133344#suppl/GSE133344_filtered_barcodes.tsv.gz` — kept
   - keeping GSE133344_filtered_barcodes.tsv.gz -- GSE133344_filtered_matrix.mtx.gz cannot be read without it
1. **dropped** `geo://GSE194122` — the model did not keep it
1. **chosen** `geo://GSE133344#suppl/GSE133344_filtered_cell_identities.csv.gz` — This small file maps each filtered cell barcode to its guide/perturbation identity, which is required to assign cells to single- and paired-gene activation conditions.
1. **listing** `geo://GSE133344#suppl/GSE133344_raw_matrix.mtx.gz` — kept
   - keeping GSE133344_raw_matrix.mtx.gz as well -- a screen's measurements live in the counts file, and the label table does not carry them
1. **listing** `geo://GSE133344#suppl/GSE133344_raw_genes.tsv.gz` — kept
   - keeping GSE133344_raw_genes.tsv.gz -- GSE133344_raw_matrix.mtx.gz cannot be read without it
1. **listing** `geo://GSE133344#suppl/GSE133344_raw_barcodes.tsv.gz` — kept
   - keeping GSE133344_raw_barcodes.tsv.gz -- GSE133344_raw_matrix.mtx.gz cannot be read without it
1. **listing** `geo://GSE133344#suppl/GSE133344_raw_cell_identities.csv.gz` — kept
   - keeping GSE133344_raw_cell_identities.csv.gz -- the cell labels of GSE133344_raw_matrix.mtx.gz
1. **downloaded** `geo://GSE133344#suppl/GSE133344_filtered_matrix.mtx.gz` → GSE133344_filtered_matrix.mtx.gz, GSE133344_filtered_matrix.mtx.gz-table.csv
1. **downloaded** `geo://GSE133344#suppl/GSE133344_filtered_cell_identities.csv.gz` → GSE133344_filtered_cell_identities.csv.gz
1. **downloaded** `geo://GSE133344#suppl/GSE133344_filtered_genes.tsv.gz` → GSE133344_filtered_genes.tsv.gz
1. **downloaded** `geo://GSE133344#suppl/GSE133344_filtered_barcodes.tsv.gz` → GSE133344_filtered_barcodes.tsv.gz
1. **downloaded** `geo://GSE133344#suppl/GSE133344_filtered_matrix.mtx.gz` → GSE133344_filtered_matrix.mtx.gz, GSE133344_filtered_matrix.mtx.gz-table.csv
1. **downloaded** `geo://GSE133344#suppl/GSE133344_filtered_cell_identities.csv.gz` → GSE133344_filtered_cell_identities.csv.gz
1. **downloaded** `geo://GSE133344#suppl/GSE133344_filtered_genes.tsv.gz` → GSE133344_filtered_genes.tsv.gz
1. **downloaded** `geo://GSE133344#suppl/GSE133344_filtered_barcodes.tsv.gz` → GSE133344_filtered_barcodes.tsv.gz
1. **downloaded** `geo://GSE133344#suppl/GSE133344_filtered_cell_identities.csv.gz` → GSE133344_filtered_cell_identities.csv.gz
1. **screened out** `GSE133344_filtered_matrix.mtx.gz` — converted to GSE133344_filtered_matrix.mtx.gz-table.csv; that table is the candidate being reviewed
1. **screened out** `GSE133344_filtered_barcodes.tsv.gz` — parsed as a single column; this file is probably not a plain table (a matrix with a preamble, or a non-tabular artifact)

## Round 2 -- supplementary evidence for that table

1. **proposed** `geo://GSE133344` (confidence 0.60) — Same study and same measurement modality: single-cell RNA-seq with per-cell read_count, UMI_count, coverage, gemgroup, good_coverage, and number_of_cells column
1. **proposed** `geo://GSE152988` (confidence 0.35) — Same cell type and assay type; cell-level tables typically include read_count, UMI_count, coverage, gemgroup, good_coverage, and number_of_cells, matching the l
1. **proposed** `geo://GSE90063` (confidence 0.30) — Same K562 Perturb-seq context; per-cell metadata tables share read_count, UMI_count, coverage, gemgroup, good_coverage, and number_of_cells columns, supplying t
1. **proposed** `geo://GSE133344` (confidence 0.60) — Same study and same measurement type (gene expression counts across the same gene columns); provides additional rows/conditions (other CRISPRa combinations or r
1. **proposed** `geo://GSE194122` (confidence 0.30) — Provides gene-by-cell count matrices with gene symbol columns that can overlap the labeled table's gene columns, offering another cohort/site of the same measur
1. **proposed** `hf://genome-scale-tools/perturb-seq` (confidence 0.20) — Perturb-seq datasets measure gene expression counts per perturbation, sharing gene symbol columns with the labeled table and providing other rows (different per
1. **proposed** `geo://GSE133344#suppl/GSE133344_filtered_cell_identities.csv.gz` (confidence 0.85) — This is the gold-standard CRISPR activation Perturb-seq dataset in K562; the cell identities file provides the perturbation labels (which genes were activated a
1. **proposed** `geo://GSE133344#suppl/GSE133344_filtered_genes.tsv.gz` (confidence 0.80) — Provides the same gene panel expression measurements across control and perturbed K562 cells, enabling comparison of single-gene versus paired-gene CRISPRa tran
1. **listing** `geo://GSE133344` — kept
   - 11 file(s) on offer, smallest 3.6 kB, total 12.4 GB # about: title: Exploring genetic interaction manifolds constructed from rich phenotypes | summary: Rich phenotyping using single-cell RNA sequenci…
1. **listing** `geo://GSE152988` — kept
   - 5 file(s) on offer, smallest 3.4 kB, total 455.8 MB # about: title: Genome-wide CRISPRi/a screens in human neurons link lipid imbalance to ferroptosis | summary: Neurons are unique among all human ce…
1. **listing** `geo://GSE90063` — kept
   - 7 file(s) on offer, smallest 2.0 kB, total 1.4 GB # about: title: Perturb-seq: Dissecting molecular circuits with scalable single cell RNA profiling of pooled genetic screens | summary: Methods: We c…
1. **chosen** `geo://GSE90063#suppl/GSE90063_k562_umi_wt.txt.gz` — K562 Perturb-seq UMI count matrix of cells by genes, the measurement needed to compare single vs combined CRISPRa transcriptional responses.
1. **chosen** `geo://GSE133344#suppl/GSE133344_raw_cell_identities.csv.gz` — Per-cell identities for the same study's pooled CRISPR single-cell experiment, giving perturbation labels to pair with expression for single and combined activations.
1. **listing** `geo://GSE133344#suppl/GSE133344_filtered_cell_identities.csv.gz` — kept
   - keeping GSE133344_filtered_cell_identities.csv.gz as well -- this task's labels (a perturbation screen's condition/guide) usually live in the annotated file, not the raw matrix
1. **dropped** `geo://GSE152988` — the model did not keep it
1. **listing** `geo://GSE194122` — kept
   - 3 file(s) on offer, smallest 3.9 kB, total 3.3 GB # about: title: A sandbox for prediction and integration of DNA, RNA, and proteins in single cells | summary: Single-cell multiomics data collected f…
1. **listing** `hf://genome-scale-tools/perturb-seq` — kept
   - 0 file(s) on offer, the repository did not answer a listing for this reference
1. **chosen** `geo://GSE133344#matrix/GSE133344_series_matrix.txt.gz` — Series matrix metadata for the same CRISPRa study, but it is an index/metadata file, not the measurement data.
1. **dropped** `geo://GSE194122` — the model did not keep it
1. **dropped** `hf://genome-scale-tools/perturb-seq` — the model did not keep it
1. **listing** `geo://GSE133344#suppl/GSE133344_filtered_genes.tsv.gz` — kept
   - 11 file(s) on offer, smallest 3.6 kB, total 12.4 GB # about: title: Exploring genetic interaction manifolds constructed from rich phenotypes | summary: Rich phenotyping using single-cell RNA sequenci…
1. **chosen** `geo://GSE133344#suppl/GSE133344_filtered_cell_identities.csv.gz` — Provides the per-cell perturbation labels (which genes were activated alone or in combination) needed to define single-gene and paired-gene CRISPRa conditions for the K562 Perturb…
1. **chosen** `geo://GSE133344#suppl/GSE133344_filtered_genes.tsv.gz` — Contains the gene expression measurements across control and perturbed K562 cells, enabling comparison of single-gene versus paired-gene CRISPRa transcriptional responses.
1. **downloaded** `geo://GSE90063#suppl/GSE90063_k562_umi_wt.txt.gz` → GSE90063_k562_umi_wt.txt.gz
1. **downloaded** `geo://GSE133344#suppl/GSE133344_raw_cell_identities.csv.gz` → GSE133344_raw_cell_identities.csv.gz
1. **downloaded** `geo://GSE133344#suppl/GSE133344_filtered_cell_identities.csv.gz` → GSE133344_filtered_cell_identities.csv.gz
1. **downloaded** `geo://GSE133344#suppl/GSE133344_raw_matrix.mtx.gz` → GSE133344_raw_matrix.mtx.gz
1. **downloaded** `geo://GSE133344#suppl/GSE133344_raw_genes.tsv.gz` → GSE133344_raw_genes.tsv.gz
1. **downloaded** `geo://GSE133344#suppl/GSE133344_raw_barcodes.tsv.gz` → GSE133344_raw_barcodes.tsv.gz
1. **downloaded** `geo://GSE133344#suppl/GSE133344_raw_cell_identities.csv.gz` → GSE133344_raw_cell_identities.csv.gz
1. **download failed** `geo://GSE133344#matrix/GSE133344_series_matrix.txt.gz` — error: SourceError: GSE133344's series matrix has a header and no rows, which is normal for RNA-seq series. Supplementary files (smallest first): matrix/GSE133344_series_matrix.txt.gz (3.6 kB), suppl…
1. **downloaded** `geo://GSE133344#suppl/GSE133344_filtered_cell_identities.csv.gz` → GSE133344_filtered_cell_identities.csv.gz
1. **downloaded** `geo://GSE133344#suppl/GSE133344_filtered_cell_identities.csv.gz` → GSE133344_filtered_cell_identities.csv.gz
1. **downloaded** `geo://GSE133344#suppl/GSE133344_filtered_genes.tsv.gz` → GSE133344_filtered_genes.tsv.gz
1. **screening** kept all 2 table(s)
1. **screened out** `GSE133344_filtered_matrix.mtx.gz` — converted to GSE133344_filtered_matrix.mtx.gz-table.csv; that table is the candidate being reviewed
1. **screened out** `GSE133344_filtered_barcodes.tsv.gz` — parsed as a single column; this file is probably not a plain table (a matrix with a preamble, or a non-tabular artifact)
1. **screened out** `GSE133344_raw_matrix.mtx.gz` — a matrix-market (sparse) file, not a table: it becomes one only together with its barcodes/features companion files (and a cell label table), which were not sta
1. **screened out** `GSE133344_raw_barcodes.tsv.gz` — parsed as a single column; this file is probably not a plain table (a matrix with a preamble, or a non-tabular artifact)

**Refusals by cause:** not a table (fewer than two columns) ×4, the model judged it unusable ×2, cannot supply the columns the gold has ×2
