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

## Supplementary — evidence only, never labels (0)

- (none: the run will be plain supervised training)

## Unusable — refused (25)

- `data/fetched/geo/GSE153056/GSE153056_RAW.tar` — review: not a table: 0 parsed column(s). A table needs at least a feature column and something else
- `data/fetched/geo/GSE153056/GSE153056_RAW.tar-contents/GSM4633605_CITE_cDNA_counts.tsv.gz` — review: This is a genes-by-cells count matrix (gene symbols as rows, cell barcodes as columns) with no perturbation/condition column; the label 'gene' is not present as a column. It is the CITE cDNA (transcriptome) matrix for the ECCITE-seq experiment, so it carries expression but no perturbation identity.
- `data/fetched/geo/GSE153056/GSE153056_RAW.tar-contents/GSM4633606_CITE_ADT_Barcodes.csv.gz` — review: This is a barcode-to-antibody (ADT/CITE-seq) feature mapping file: each row pairs a cell barcode with an antibody-derived tag name (PDL1, CD86, IDO1, TIM3, PDL2). It has no header, no gene-expression columns, and no CRISPR perturbation column, so it is not a per-cell perturbation screen.
- `data/fetched/geo/GSE153056/GSE153056_RAW.tar-contents/GSM4633606_CITE_ADT_counts.tsv.gz` — review: This is a CITE-seq antibody-derived tag (ADT) protein count matrix: rows are antibody features (BTLA, CD47, CD66a, CD86, ...) and columns are cell barcodes. It is a features-by-cells matrix with no perturbation/condition column, so it is not a per-cell perturbation screen and carries no transcriptome.
- `data/fetched/geo/GSE153056/GSE153056_RAW.tar-contents/GSM4633607_CITE_HTO_Barcodes.csv.gz` — review: This is a barcode-to-HTO (hashing antibody) assignment file: two columns, a cell barcode and an HTO tag. It carries neither a gene perturbation identity nor any transcriptome measurements, so it cannot serve as a perturbation screen.
- `data/fetched/geo/GSE153056/GSE153056_RAW.tar-contents/GSM4633607_CITE_HTO_counts.tsv.gz` — review: This is a CITE-seq HTO (hashing) antibody-count matrix: rows are hashtag oligos (HTO22, HTO23, HTO25, HTO26) and columns are cell barcodes, with no gene/transcriptome columns and no perturbation condition column. It is a genes-by-cells-style feature matrix, not a per-cell perturbation screen.
- `data/fetched/geo/GSE153056/GSE153056_RAW.tar-contents/GSM4633608_ECCITE_Arrayed_cDNA_counts.tsv.gz` — review: This is a genes-by-cells counts matrix (16,696 columns: gene rows in the first column, cell barcodes as columns, with r_ and g_ prefixed barcodes). It carries the transcriptome but no per-cell perturbation identity column; the perturbation labels live in the ECCITE metadata table.
- `data/fetched/geo/GSE153056/GSE153056_RAW.tar-contents/GSM4633609_ECCITE_Arrayed_ADT_Barcodes.csv.gz` — review: This is a two-column barcode-to-ADT mapping file (cell barcode, antibody-derived tag name such as CD86/PDL1), not a per-cell perturbation screen. It has no gene/guide/condition column and no transcriptome columns, so it cannot serve as a perturbation-response table.
- `data/fetched/geo/GSE153056/GSE153056_RAW.tar-contents/GSM4633609_ECCITE_Arrayed_ADT_counts.tsv.gz` — review: This is an antibody-derived tag (ADT) / cell-surface protein count matrix from a CITE-seq/ECCITE experiment: rows are surface markers (CD86, PDL1, PDL2, CD366) and columns are cell barcodes. It carries no transcriptome (no gene-named columns) and no perturbation identity column, so it cannot serve as a per-cell perturbation screen.
- `data/fetched/geo/GSE153056/GSE153056_RAW.tar-contents/GSM4633610_ECCITE_Arrayed_HTO1_Barcodes.csv.gz` — review: This is a barcode-to-hashtag (HTO) assignment file: two columns mapping cell barcodes to hashtag labels (HTO_ETV7g1, HTO_NT5, etc.). It has no header, no gene/perturbation-target column, and no transcriptome measurements. It is a cell-identity/assignment table, not a per-cell perturbation screen.
- `data/fetched/geo/GSE153056/GSE153056_RAW.tar-contents/GSM4633610_ECCITE_Arrayed_HTO_counts.tsv.gz` — review: This is an antibody-hashtag (HTO) count matrix from a CITE/ECCITE experiment: rows are HTO tags (HTO-IRF7g1, HTO-MARCH8g1, ...) and columns are cell barcodes (r_/g_ prefixes). It is a genes-by-cells-shaped table of hashtag counts, not a per-cell perturbation screen with a condition column and transcriptome.
- `data/fetched/geo/GSE153056/GSE153056_RAW.tar-contents/GSM4633611_ECCITE_Arrayed_HTO2_Barcodes.csv.gz` — review: This is a two-column barcode-to-HTO assignment file (cell barcode, hashtag identity such as HTO_IRF7g1, HTO_eGFP). It has no header, no gene-expression columns, and no per-cell gene-target column; the HTO tags are sample/hashtag multiplexing labels, not the CRISPR perturbation identity.
- … and 13 more in `data_report.md`

## What to review

1. Is the gold the table this question needs, and are its rows the right observation unit?
2. Is `guide_ID` the right column to supervise on?
3. For a perturbation task: is that same column the perturbation identity, and are the control labels right?
4. Is any `unusable` table actually the data, one manual step away?

Fix the plan by editing `plan.json` in place (the fields above), or by overriding them on the stage-2 command line. Nothing else in this directory needs to change.

## Then run stage 2

```bash
.venv/bin/python -m kosmos.cli.main experiment "Does CRISPR knockout of the gene IRF1 change the transcriptome of human macrophages, and can the transcriptional response to the perturbation be predicted?" --data-plan artifacts/dataprep/irf1-macrophage-test3/plan.json --yes
```

Stage 2 writes `run/summary.md` and `run/metrics.json` beside this file; re-running it never overwrites a finished run (it lands in `run-2`, ...).

