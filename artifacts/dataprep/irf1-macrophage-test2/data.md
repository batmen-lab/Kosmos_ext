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
- `data/fetched/geo/GSE153056/GSE153056_RAW.tar-contents/GSM4633605_CITE_cDNA_counts.tsv.gz` — review: This is a genes-by-cells count matrix (gene symbols in the first column, cell barcodes as the remaining 7566 columns). It carries the transcriptome but no perturbation/condition column, so it is not a per-cell screen on its own.
- `data/fetched/geo/GSE153056/GSE153056_RAW.tar-contents/GSM4633606_CITE_ADT_Barcodes.csv.gz` — review: This is a barcode-to-protein lookup table for CITE-seq ADT (antibody-derived tag) features, not a per-cell perturbation screen. Each row maps a cell barcode to an antibody/protein name (PDL1, CD86, IDO1, TIM3, PDL2), with no gene expression and no CRISPR perturbation identity.
- `data/fetched/geo/GSE153056/GSE153056_RAW.tar-contents/GSM4633606_CITE_ADT_counts.tsv.gz` — review: This is a CITE-seq antibody-derived tag (ADT) protein count matrix: rows are antibody features (BTLA, CD47, CD66a, CD86, ...) and columns are cell barcodes. It is a features-by-cells matrix with no perturbation/condition column, so it is not a per-cell perturbation screen and carries no transcriptome.
- `data/fetched/geo/GSE153056/GSE153056_RAW.tar-contents/GSM4633607_CITE_HTO_Barcodes.csv.gz` — review: This is a barcode-to-hashtag (HTO) assignment file: two columns mapping a cell barcode to an HTO tag. It has no gene/perturbation column and no transcriptome, so it is not a per-cell perturbation screen.
- `data/fetched/geo/GSE153056/GSE153056_RAW.tar-contents/GSM4633607_CITE_HTO_counts.tsv.gz` — review: This is a CITE-seq HTO (hashing antibody) counts matrix: rows are hashtag oligos (HTO22, HTO23, HTO25, HTO26) and columns are cell barcodes, with no gene/transcriptome columns and no perturbation (gene target) column. It is a genes-by-cells style matrix of antibody counts, not a per-cell perturbation screen.
- `data/fetched/geo/GSE153056/GSE153056_RAW.tar-contents/GSM4633608_ECCITE_Arrayed_cDNA_counts.tsv.gz` — review: This is a genes-by-cells counts matrix (16,696 columns: gene rows in the first column, cell barcodes as columns). It carries the transcriptome but no per-cell perturbation/condition column, so it is not a per-cell screen on its own.
- `data/fetched/geo/GSE153056/GSE153056_RAW.tar-contents/GSM4633609_ECCITE_Arrayed_ADT_Barcodes.csv.gz` — review: This is an antibody-derived tag (ADT) barcode-to-protein mapping file: two columns, a cell barcode and a protein marker name (CD86, PDL1, PDL2, CD366). It has no header, no gene-expression columns, and no perturbation/condition column. It is a companion file to the ADT counts matrix, not a per-cell perturbation screen.
- `data/fetched/geo/GSE153056/GSE153056_RAW.tar-contents/GSM4633609_ECCITE_Arrayed_ADT_counts.tsv.gz` — review: This is an antibody-derived tag (ADT) / cell-surface protein counts matrix from a CITE-seq/ECCITE experiment: rows are surface markers (CD86, PDL1, PDL2, CD366, ...) and columns are cell barcodes. It carries no gene-expression transcriptome and no perturbation-identity column, so it cannot serve as a per-cell perturbation screen.
- `data/fetched/geo/GSE153056/GSE153056_RAW.tar-contents/GSM4633610_ECCITE_Arrayed_HTO1_Barcodes.csv.gz` — review: This is a barcode-to-hashtag (HTO) assignment file: two columns mapping cell barcodes to hashtag labels such as HTO_ETV7g1 and HTO_NT5. It carries no gene expression and no CRISPR guide/target-gene column, so it is not a per-cell perturbation screen.
- `data/fetched/geo/GSE153056/GSE153056_RAW.tar-contents/GSM4633610_ECCITE_Arrayed_HTO_counts.tsv.gz` — review: This is an antibody-hashtag (HTO) count matrix from a CITE/ECCITE experiment: rows are HTO tags (HTO-IRF7g1, HTO-MARCH8g1, ...) and columns are cell barcodes (r_... and g_...). It carries neither the transcriptome nor a per-cell perturbation label column; the perturbation identity is encoded in the row names of this HTO table, not in a column.
- `data/fetched/geo/GSE153056/GSE153056_RAW.tar-contents/GSM4633611_ECCITE_Arrayed_HTO2_Barcodes.csv.gz` — review: This is a barcode-to-HTO assignment file (two columns: cell barcode and hashtag oligo identity), not a per-cell perturbation screen. It carries no gene-level perturbation identity and no transcriptome measurements.
- … and 13 more in `data_report.md`

## What to review

1. Is the gold the table this question needs, and are its rows the right observation unit?
2. Is `guide_ID` the right column to supervise on?
3. For a perturbation task: is that same column the perturbation identity, and are the control labels right?
4. Is any `unusable` table actually the data, one manual step away?

Fix the plan by editing `plan.json` in place (the fields above), or by overriding them on the stage-2 command line. Nothing else in this directory needs to change.

## Then run stage 2

```bash
.venv/bin/python -m kosmos.cli.main experiment "Does CRISPR knockout of the gene IRF1 change the transcriptome of human macrophages, and can the transcriptional response to the perturbation be predicted?" --data-plan artifacts/dataprep/irf1-macrophage-test2/plan.json --yes
```

Stage 2 writes `run/summary.md` and `run/metrics.json` beside this file; re-running it never overwrites a finished run (it lands in `run-2`, ...).

