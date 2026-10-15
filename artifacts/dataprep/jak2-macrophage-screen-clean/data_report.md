# Data report

**Question:** Does CRISPR knockout of the gene JAK2 change the transcriptome of human macrophages, and can the transcriptional response to the perturbation be predicted?

**Tables:** 2 considered — gold **1**, supplementary **0**, unusable **1**, waiting for a person **0**
**Training columns:** 21 of the gold's 21

## Datasets

- `GSE153056_ECCITE_metadata.tsv.gz` — **gold** — target column 'gene' is present with 21 feature(s) (10 categorical); this table can train, select and test; 3 column(s) are not features (free text, a row index, or empty), e.g. ['Unnamed: 0', 'guide… (the model had said unusable); left out Unnamed: 0, guide_ID, NT — **a person could salvage it:** join the labels in GSM4633614_ECCITE_cDNA_counts.tsv.gz on barcode
- `GSM4633614_ECCITE_cDNA_counts.tsv.gz` — **unusable** — review: This is a genes-by-cells count matrix (20,730 columns: one gene-name column plus one column per cell barcode). Rows are genes, columns are cells, and there is no column recording which pertur…; left out Unnamed: 0 — **a person could salvage it:** transpose it, genes are rows, then join the labels in data/fetched/geo/GSE153056/GSE153056_ECCITE_metadata.tsv.gz on barcode

## Round 1 -- the labeled table

1. **screening** kept all 2 table(s)

**Refusals by cause:** the model judged it unusable ×1
