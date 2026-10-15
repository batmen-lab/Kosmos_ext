# Data report

**Question:** Does CRISPR knockout of the gene IRF1 change the transcriptome of human macrophages, and can the transcriptional response to the perturbation be predicted?

**Tables:** 2 considered — gold **1**, supplementary **1**, unusable **0**, waiting for a person **0**
**Training columns:** 18647 of the gold's 22

## Datasets

- `GSE153056_ECCITE_metadata.tsv.gz` — **gold** — review: screen label half (condition column 'guide_ID'); its counts matrix is joined at training time; left out Unnamed: 0, NT — **a person could salvage it:** join the labels in GSE153056_ECCITE_metadata.tsv.gz to the counts matrix GSM4633614_ECCITE_cDNA_counts.tsv.gz on barcode
- `GSM4633614_ECCITE_cDNA_counts.tsv.gz` — **supplementary** — review: measured transcriptome, usable as evidence for the screen; 2 column(s) were left out of the intersection because more than one of the table's columns normalises to the gold's name, e.g. ['C2O…; shares 18647 column(s) with the gold: 7SK, 7SK.1, 7SK.2, A1BG, A1BG-AS1, A2M — **a person could salvage it:** transpose it, genes are rows, then join the labels in data/fetched/geo/GSE153056/GSE153056_ECCITE_metadata.tsv.gz on barcode
- `geo://GSE153056#suppl/GSE153056_ECCITE_metadata.tsv.gz` — **not downloaded** — geo://GSE153056#suppl/GSE153056_ECCITE_metadata.tsv.gz was chosen but is already in this run (same file); not fetched again
- `geo://GSE153056` — **not downloaded** — the model did not keep it
- `geo://GSE194122` — **not downloaded** — the model did not keep it

## Round 1 -- the labeled table

1. **screening** kept all 2 table(s)

## Round 2 -- supplementary evidence for that table

1. **proposed** `geo://GSE153056` (confidence 0.60) — Same study and same ECCITE-seq cDNA count measurement modality; other samples in the series provide the same gene-expression columns (7SK, A1BG, A2M, etc.) for
1. **proposed** `geo://GSE194122` (confidence 0.40) — Independent macrophage CRISPR perturbation cohort measuring the same transcriptome gene columns (7SK, A1BG, A2M, ...) for other rows/samples
1. **listing** `geo://GSE153056` — kept
   - 7 file(s) on offer, smallest 1.7 kB, total 124.0 MB # about: title: Characterizing the molecular regulation of inhibitory immune checkpoints with multi-modal single-cell screens. | summary: The expre…
1. **listing** `geo://GSE194122` — kept
   - 3 file(s) on offer, smallest 3.9 kB, total 3.3 GB # about: title: A sandbox for prediction and integration of DNA, RNA, and proteins in single cells | summary: Single-cell multiomics data collected f…
1. **refused** `geo://GSE153056#suppl/GSE153056_ECCITE_metadata.tsv.gz` — already used: geo://GSE153056#suppl/GSE153056_ECCITE_metadata.tsv.gz was chosen but is already in this run (same file); not fetched again
1. **dropped** `geo://GSE153056` — the model did not keep it
1. **dropped** `geo://GSE194122` — the model did not keep it
