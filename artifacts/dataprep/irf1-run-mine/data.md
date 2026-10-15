# Data package — Does CRISPR knockout of the gene IRF1 change the transcriptome of human macrophages, and can the transcriptional response to the perturbation be predicted?

Stage 1 of the **fetch → review → experiment** workflow. No model has been trained yet.

- plan to review: `plan.json`
- full reasoning: `data_report.md`

## The task the fetcher read

- objective           : Does CRISPR knockout of the gene IRF1 change the transcriptome of human macrophages, and can the transcriptional response to the perturbation be predicted?
- target / condition  : `guide_ID` (source: cli, confidence: exact_match)
- task type           : classification
- feature columns     : 22

## Gold — trainable (1)

- `data/fetched/geo/GSE153056/GSE153056_ECCITE_metadata.tsv.gz` — review: screen label half (condition column 'guide_ID'); its counts matrix is joined at training time

## Supplementary — evidence only, never labels (5)

- `data/fetched/geo/GSE153056/GSE153056_RAW.tar-contents/GSM4633605_CITE_cDNA_counts.tsv.gz` — review: measured transcriptome, usable as evidence for the screen
- `data/fetched/geo/GSE153056/GSE153056_RAW.tar-contents/GSM4633608_ECCITE_Arrayed_cDNA_counts.tsv.gz` — review: measured transcriptome, usable as evidence for the screen
- `data/fetched/geo/GSE153056/GSE153056_RAW.tar-contents/GSM4633612_ECCITE_Arrayed_GO_CITE03_counts.tsv.gz` — review: measured transcriptome, usable as evidence for the screen
- `data/fetched/geo/GSE153056/GSE153056_RAW.tar-contents/GSM4633613_ECCITE_Arrayed_GO_LENTI_counts.tsv.gz` — review: measured transcriptome, usable as evidence for the screen
- `data/fetched/geo/GSE153056/GSE153056_RAW.tar-contents/GSM4633614_ECCITE_cDNA_counts.tsv.gz` — review: measured transcriptome, usable as evidence for the screen

## Unusable — refused (20)

- `data/fetched/geo/GSE153056/GSE153056_RAW.tar` — review: not a table: 0 parsed column(s). A table needs at least a feature column and something else
- `data/fetched/geo/GSE153056/GSE153056_RAW.tar-contents/GSM4633606_CITE_ADT_Barcodes.csv.gz` — review: This is a barcode-to-protein lookup table for CITE-seq ADT (antibody-derived tag) features, not a per-cell perturbation screen. Each row maps a cell barcode to an antibody/protein name (PDL1, CD86, IDO1, TIM3, PDL2), with no gene expression and no CRISPR guide or target-gene column.
- `data/fetched/geo/GSE153056/GSE153056_RAW.tar-contents/GSM4633606_CITE_ADT_counts.tsv.gz` — review: This is a CITE-seq antibody-derived tag (ADT) protein count matrix: rows are surface-marker proteins (BTLA, CD47, CD66a, CD86, ...) and columns are cell barcodes. It carries no gene expression and no perturbation/condition column, so it cannot serve as a per-cell perturbation screen.
- `data/fetched/geo/GSE153056/GSE153056_RAW.tar-contents/GSM4633607_CITE_HTO_Barcodes.csv.gz` — review: This is a barcode-to-HTO (hashing antibody) assignment file: two columns, a cell barcode and an HTO tag, with no header and no gene/perturbation column. It is a cell-metadata/annotation table, not a per-cell perturbation screen, and it carries no transcriptome.
- `data/fetched/geo/GSE153056/GSE153056_RAW.tar-contents/GSM4633607_CITE_HTO_counts.tsv.gz` — review: This is a CITE-seq HTO (hashing) antibody-count matrix: rows are hashtag antibodies (HTO22, HTO23, HTO25, HTO26) and columns are cell barcodes, with no gene/transcriptome columns and no perturbation condition column. It is a genes-by-cells-style feature matrix, not a per-cell perturbation screen.
- `data/fetched/geo/GSE153056/GSE153056_RAW.tar-contents/GSM4633609_ECCITE_Arrayed_ADT_Barcodes.csv.gz` — review: This is an antibody-derived tag (ADT) barcode-to-antibody mapping file: each row pairs a cell barcode with a single surface-protein tag (CD86, PDL1, PDL2, CD366). It has no header, no gene-expression columns, and no perturbation/condition column, so it is not a per-cell perturbation screen.
- `data/fetched/geo/GSE153056/GSE153056_RAW.tar-contents/GSM4633609_ECCITE_Arrayed_ADT_counts.tsv.gz` — review: This is an antibody-derived tag (ADT) / cell-surface protein count matrix from a CITE-seq/ECCITE experiment: rows are surface markers (CD86, PDL1, PDL2, CD366) and columns are cell barcodes. It carries no gene expression and no perturbation identity column, so it cannot serve as a per-cell perturbation screen.
- `data/fetched/geo/GSE153056/GSE153056_RAW.tar-contents/GSM4633610_ECCITE_Arrayed_HTO1_Barcodes.csv.gz` — review: This is a barcode-to-hashtag (HTO) assignment file: two unnamed columns mapping a cell barcode to an HTO tag (e.g. HTO_ETV7g1, HTO_NT5). It carries no gene-expression measurements and no gene-target column, so it is not a per-cell perturbation screen.
- `data/fetched/geo/GSE153056/GSE153056_RAW.tar-contents/GSM4633610_ECCITE_Arrayed_HTO_counts.tsv.gz` — review: This is an antibody-hashtag (HTO) count matrix from a CITE/ECCITE experiment: rows are HTO tags (HTO-IRF7g1, HTO-MARCH8g1, ...) and columns are cell barcodes (r_... and g_...). It is a genes-by-cells-shaped table with no per-cell perturbation condition column and no transcriptome; the 'gene' label column is not present.
- `data/fetched/geo/GSE153056/GSE153056_RAW.tar-contents/GSM4633611_ECCITE_Arrayed_HTO2_Barcodes.csv.gz` — review: This is a barcode-to-HTO hashtag assignment file (two columns: cell barcode and HTO tag such as HTO_IRF7g1, HTO_eGFP), not a per-cell perturbation screen with a transcriptome. It has no gene-expression columns and no 'gene' label column.
- `data/fetched/geo/GSE153056/GSE153056_RAW.tar-contents/GSM4633615_ECCITE_ADT_Barcodes.csv.gz` — review: This is a barcode-to-ADT (antibody-derived tag) assignment file from the ECCITE-seq dataset: each row pairs a cell barcode with a surface-protein tag name (CD86, PDL1, PDL2, CD366). It has no header, no gene-expression columns, and no CRISPR perturbation/condition column, so it is not a per-cell perturbation screen.
- `data/fetched/geo/GSE153056/GSE153056_RAW.tar-contents/GSM4633615_ECCITE_ADT_counts.tsv.gz` — review: This is an antibody-derived tag (ADT/CITE-seq) protein count matrix: rows are surface-marker proteins (CD86, PDL1, PDL2, CD366) and columns are cell barcodes. It carries no gene expression and no perturbation identity, so it cannot serve as a per-cell perturbation screen.
- … and 8 more in `data_report.md`

## What to review

1. Is the gold the table this question needs, and are its rows the right observation unit?
2. Is `guide_ID` the right column to supervise on?
3. For a perturbation task: is that same column the perturbation identity, and are the control labels right?
4. Is any `unusable` table actually the data, one manual step away?

Fix the plan by editing `plan.json` in place (the fields above), or by overriding them on the stage-2 command line. Nothing else in this directory needs to change.

## Then run stage 2

```bash
.venv/bin/python -m kosmos.cli.main experiment "Does CRISPR knockout of the gene IRF1 change the transcriptome of human macrophages, and can the transcriptional response to the perturbation be predicted?" --data-plan artifacts/dataprep/irf1-run-mine/plan.json --yes
```

Stage 2 writes `run/summary.md` and `run/metrics.json` beside this file; re-running it never overwrites a finished run (it lands in `run-2`, ...).

