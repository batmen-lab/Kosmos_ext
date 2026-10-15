# Data report

**Question:** Does CRISPR knockout of the gene JAK2 change the transcriptome of human macrophages, and can the transcriptional response to the perturbation be predicted?

**Tables:** 3 considered — gold **1**, supplementary **0**, unusable **2**, waiting for a person **0**
**Training columns:** 21 of the gold's 21

## Datasets

- `GSE153056_ECCITE_metadata.tsv.gz` — **gold** — target column 'gene' is present with 21 feature(s) (10 categorical); this table can train, select and test; 3 column(s) are not features (free text, a row index, or empty), e.g. ['Unnamed: 0', 'guide… (the model had said unusable); left out Unnamed: 0, guide_ID, NT — **a person could salvage it:** join the labels in GSM4633614_ECCITE_cDNA_counts.tsv.gz on barcode (the row index) to attach the 'gene'/'NT' perturbation column to the expression matrix
- `GSE171964_countsmatrix_v2.mtx.gz` — **unusable** — not a table: 0 parsed column(s). A table needs at least a feature column and something else
- `GSM4633614_ECCITE_cDNA_counts.tsv.gz` — **unusable** — review: This is a genes-by-cells count matrix (genes as rows, cell barcodes as columns) with no perturbation/condition column; the perturbation identities live in the paired ECCITE metadata file.; left out Unnamed: 0 — **a person could salvage it:** transpose it, genes are rows, then join the labels in data/fetched/geo/GSE153056/GSE153056_ECCITE_metadata.tsv.gz on barcode
- `geo://GSE194122` — **not downloaded** — the model did not keep it
- `geo://GSE153056#matrix/GSE153056-GPL18573_series_matrix.txt.gz` — **not downloaded** — error: SourceError: GSE153056's series matrix has a header and no rows, which is normal for RNA-seq series. Supplementary files (smallest first): suppl/filelist.txt (1.7 kB), suppl/GSE153056_CITE_met…

## Round 1 -- the labeled table

1. **screening** kept all 2 table(s)

## Round 2 -- supplementary evidence for that table

1. **proposed** `geo://GSE153056` (confidence 0.85) — Same study series; other samples/GSMs within it provide the same measured columns (nCount_RNA, nFeature_RNA, nCount_HTO, nFeature_HTO, nCount_GDO, nFeature_GDO,
1. **proposed** `geo://GSE194122` (confidence 0.40) — Provides RNA and ADT count columns (nCount_RNA, nFeature_RNA, nCount_ADT, nFeature_ADT, percent.mito) and perturbation/guide labels (guide_ID, crispr) for other
1. **proposed** `geo://GSE171964` (confidence 0.35) — Shares RNA count columns (nCount_RNA, nFeature_RNA, percent.mito) and CRISPR guide/perturbation columns (guide_ID, crispr, con) for additional perturbed rows
1. **listing** `geo://GSE153056` — kept
   - 2 file(s) on offer, smallest 3.0 kB, total 6.2 kB # about: title: Characterizing the molecular regulation of inhibitory immune checkpoints with multi-modal single-cell screens. | summary: The express…
1. **listing** `geo://GSE194122` — kept
   - 2 file(s) on offer, smallest 587.0 MB, total 3.3 GB # about: title: A sandbox for prediction and integration of DNA, RNA, and proteins in single cells | summary: Single-cell multiomics data collected…
1. **listing** `geo://GSE171964` — kept
   - 4 file(s) on offer, smallest 118.0 kB, total 1.4 GB # about: title: CITE-seq of 6 patients treated with the Pfizer BioNTech BNT162b2 mRNA vaccine | summary: Purpose: To understand the single cell lan…
1. **chosen** `geo://GSE153056#matrix/GSE153056-GPL18573_series_matrix.txt.gz` — This series matrix file contains the sample-level metadata and expression measurements for the ECCITE-seq CRISPR screen study, which is the same study series as the JAK2 knockout …
1. **listing** `geo://GSE171964#suppl/GSE171964_countsmatrix_v2.mtx.gz` — kept
   - keeping GSE171964_countsmatrix_v2.mtx.gz as well -- a screen's measurements live in the counts file, and the label table does not carry them
1. **dropped** `geo://GSE194122` — the model did not keep it
1. **download failed** `geo://GSE153056#matrix/GSE153056-GPL18573_series_matrix.txt.gz` — error: SourceError: GSE153056's series matrix has a header and no rows, which is normal for RNA-seq series. Supplementary files (smallest first): suppl/filelist.txt (1.7 kB), suppl/GSE153056_CITE_met…
1. **downloaded** `geo://GSE171964#suppl/GSE171964_countsmatrix_v2.mtx.gz` → GSE171964_countsmatrix_v2.mtx.gz
1. **screened out** `GSE171964_countsmatrix_v2.mtx.gz` — not readable as a delimited text table: EmptyDataError: No columns to parse from file

**Refusals by cause:** the model judged it unusable ×1, not a table (fewer than two columns) ×1
