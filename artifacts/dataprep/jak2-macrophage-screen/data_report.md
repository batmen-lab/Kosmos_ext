# Data report

**Question:** Does CRISPR knockout of the gene JAK2 change the transcriptome of human macrophages, and can the transcriptional response to the perturbation be predicted?

**Tables:** 4 considered — gold **1**, supplementary **0**, unusable **3**, waiting for a person **0**
**Training columns:** 21 of the gold's 21

## Datasets

- `GSE153056_ECCITE_metadata.tsv.gz` — **gold** — target column 'gene' is present with 21 feature(s) (10 categorical); this table can train, select and test; 3 column(s) are not features (free text, a row index, or empty), e.g. ['Unnamed: 0', 'guide… (the model had said unusable); left out Unnamed: 0, guide_ID, NT — **a person could salvage it:** join the labels in this metadata (guide_ID/gene/NT) onto the barcodes of GSM4633608_ECCITE_Arrayed_cDNA_counts.tsv.gz
- `GSE153056_CITE_metadata.tsv.gz` — **unusable** — review: This is a per-cell CITE-seq metadata table (barcodes with RNA/ADT/HTO QC columns and hashtag classification). It has no gene-expression columns and no CRISPR perturbation column, so it is not…; left out Unnamed: 0 — **a person could salvage it:** join this metadata to the matching counts matrix (GSM4633605_CITE_cDNA_counts.tsv.gz) on barcode, and take the perturbation labels from the ECCITE files
- `GSM4633605_CITE_cDNA_counts.tsv.gz` — **unusable** — review: This is a genes-by-cells count matrix (gene symbols as rows, cell barcodes as columns) with no perturbation/condition column; the label 'gene' is the row index (gene names), not a per-cell pe…; left out Unnamed: 0 — **a person could salvage it:** transpose it so barcodes are rows and genes are columns, then join the labels in data/fetched/geo/GSE153056/GSE153056_CITE_metadata.tsv.gz on barcode
- `GSM4633608_ECCITE_Arrayed_cDNA_counts.tsv.gz` — **unusable** — review: This is a genes-by-cells count matrix (genes as rows, cell barcodes as columns) from the ECCITE-seq dataset GSE153056. The first column holds gene names and the remaining 16,695 columns are i…; left out Unnamed: 0 — **a person could salvage it:** transpose it, genes are rows; then join the labels in data/fetched/geo/GSE153056/GSE153056_ECCITE_metadata.tsv.gz on barcode
- `geo://GSE194122` — **not downloaded** — the model did not keep it
- `geo://GSE153056#matrix/GSE153056_RAW.tar` — **not downloaded** — there is no 'matrix/GSE153056_RAW.tar' ; nothing was downloaded. Name one of those, or another source.
- `geo://GSE153056` — **not downloaded** — error: SourceError: GSE153056 publishes 2 series matrices, one per platform: . Pick one with geo://GSE153056#matrix/<filename> -- merging them would compare measurements on different probe sets.

## Round 1 -- the labeled table

1. **proposed** `geo://GSE194122` (confidence 0.60) — Provides an unperturbed human macrophage/monocyte atlas with the same cell type and gene panel, satisfying the requirement for a co-expression graph control dat
1. **proposed** `geo://GSE153056` (confidence 0.40) — A CRISPR knockout/gene deletion screen that can supply labeled perturbation data for predicting transcriptional responses, though cell type may differ from macr
1. **proposed** `geo://GSE153056#matrix/GSE153056_RAW.tar` (confidence 0.10) — This is the platform-specific file the repository asked for; however it is a .tar archive, which this pipeline cannot read, so it should not be proposed as a re
1. **proposed** `geo://GSE194122` (confidence 0.60) — A well-known human macrophage CRISPR knockout dataset that measures transcriptome-wide gene expression and includes control/unperturbed cells, matching the tiss
1. **listing** `geo://GSE194122` — kept
   - 0 file(s) on offer, size unknown # about: title: A sandbox for prediction and integration of DNA, RNA, and proteins in single cells | summary: Single-cell multiomics data collected from bone marrow m…
1. **listing** `geo://GSE153056` — kept
   - 0 file(s) on offer, size unknown # about: title: Characterizing the molecular regulation of inhibitory immune checkpoints with multi-modal single-cell screens. | summary: The expression of inhibitory…
1. **chosen** `geo://GSE153056` — This ECCITE-seq CRISPR knockout screen provides labeled single-cell perturbation transcriptomes needed to predict transcriptional responses to gene knockouts.
1. **dropped** `geo://GSE194122` — the model did not keep it
1. **refused** `geo://GSE153056#matrix/GSE153056_RAW.tar` — no such file: there is no 'matrix/GSE153056_RAW.tar' ; nothing was downloaded. Name one of those, or another source.
1. **chosen** `geo://GSE194122` — Single-cell multiome/RNA-seq measurements of human immune cells provide the transcriptome observations needed to assess JAK2 CRISPR knockout effects and predict perturbation respo…
1. **download failed** `geo://GSE153056` — error: SourceError: GSE153056 publishes 2 series matrices, one per platform: . Pick one with geo://GSE153056#matrix/<filename> -- merging them would compare measurements on different probe sets.
1. **download failed** `geo://GSE194122` — error: SourceError: GSE194122 publishes no series matrix. Supplementary files (smallest first): suppl/GSE194122_openproblems_neurips2021_cite_BMMC_processed.h5ad.gz (587.0 MB), suppl/GSE194122_openpr…
1. **screening** kept all 4 table(s)

**Refusals by cause:** the model judged it unusable ×3
