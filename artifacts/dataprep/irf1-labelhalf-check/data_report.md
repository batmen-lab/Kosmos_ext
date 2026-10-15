# Data report

**Question:** Does CRISPR knockout of the gene IRF1 change the transcriptome of human macrophages, and can the transcriptional response to the perturbation be predicted?

**Tables:** 2 considered — gold **1**, supplementary **0**, unusable **1**, waiting for a person **0**
**Training columns:** 2 of the gold's 2

## Datasets

- `GSM4633618_ECCITE_GDO_Barcodes.csv.gz` — **gold** — review: screen label half (condition column 'guide'); its counts matrix is joined at training time; left out CGGTGAACAGCTCCTCGCCC, eGFPg1 — **a person could salvage it:** join these guide labels onto the barcode column of GSM4633614_ECCITE_cDNA_counts.tsv.gz
- `GSM4633614_ECCITE_cDNA_counts.tsv.gz` — **unusable** — review: This is a genes-by-cells count matrix (rows are genes like AL627309.1, columns are cell barcodes like l1_AAACCTGAGCCAGAAC). It carries the transcriptome but no perturbation/condition column, …; left out Unnamed: 0 — **a person could salvage it:** join the labels in GSM4633618_ECCITE_GDO_Barcodes.csv.gz on barcode

## Round 1 -- the labeled table

1. **screening** kept all 2 table(s)
1. **screening** kept all 2 table(s)

## Round 2 -- supplementary evidence for that table

1. **screening** kept all 2 table(s)
1. **screening** kept all 2 table(s)
1. **screening** kept all 2 table(s)
1. **screening** kept all 2 table(s)

**Refusals by cause:** the model judged it unusable ×1
