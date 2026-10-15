# Data package — Does CRISPR knockout of the gene IRF1 change the transcriptome of human macrophages, and can the transcriptional response to the perturbation be predicted?

Stage 1 of the **fetch → review → experiment** workflow. No model has been trained yet.

- plan to review: `plan.json`
- full reasoning: `data_report.md`

## The task the fetcher read

- objective           : Does CRISPR knockout of the gene IRF1 change the transcriptome of human macrophages, and can the transcriptional response to the perturbation be predicted?
- target / condition  : `gene` (source: cli, confidence: exact_match)
- task type           : classification
- feature columns     : 21

## Gold — trainable (1)

- `data/fetched/geo/GSE153056/GSE153056_ECCITE_metadata.tsv.gz` — review: screen label half (condition column 'gene'); its counts matrix is joined at training time

## Supplementary — evidence only, never labels (0)

- (none: the run will be plain supervised training)

## Unusable — refused (25)

- `data/fetched/geo/GSE153056/GSE153056_RAW.tar-contents/GSM4633614_ECCITE_cDNA_counts.tsv.gz` — review: This is a genes-by-cells count matrix (genes as rows, cell barcodes as columns) with no perturbation/condition column; the perturbation labels live in the companion metadata file.
- `data/fetched/geo/GSE153056/GSE153056_ECCITE_Arrayed_metadata.tsv.gz` — review: This is a per-cell metadata table (barcodes with HTO/ADT/GO-lenti/GO-cite QC columns and guide classifications) but it carries no transcriptome — no gene-named count columns. It is the metadata companion to the ECCITE-Arrayed counts matrices, not a trainable screen on its own.
- `data/fetched/geo/GSE153056/GSE153056_RAW.tar` — review: not a table: 0 parsed column(s). A table needs at least a feature column and something else
- `data/fetched/geo/GSE153056/GSE153056_RAW.tar-contents/GSM4633605_CITE_cDNA_counts.tsv.gz` — review: This is a genes-by-cells count matrix (gene symbols as rows, cell barcodes as columns) with no perturbation/condition column; the label 'gene' is not present as a column, only as row identifiers. It carries the transcriptome but not the perturbation identity, so it cannot be trained on alone.
- `data/fetched/geo/GSE153056/GSE153056_RAW.tar-contents/GSM4633606_CITE_ADT_Barcodes.csv.gz` — review: This is a two-column barcode-to-protein mapping file (barcode, ADT/protein name) for CITE-seq antibody-derived tags, not a per-cell perturbation screen. It has no gene-expression columns and no perturbation/condition column.
- `data/fetched/geo/GSE153056/GSE153056_RAW.tar-contents/GSM4633606_CITE_ADT_counts.tsv.gz` — review: This is a CITE-seq antibody-derived-tag (ADT) protein count matrix: rows are antibody features (BTLA, CD47, CD66a, CD86, ...) and columns are cell barcodes. It is a genes-by-cells-shaped matrix with no perturbation/condition column, so it is not a per-cell perturbation screen and carries no transcriptome (it measures surface protein, not mRNA).
- `data/fetched/geo/GSE153056/GSE153056_RAW.tar-contents/GSM4633607_CITE_HTO_Barcodes.csv.gz` — review: This is a barcode-to-HTO (hashing antibody) assignment file: two unnamed columns mapping cell barcodes to hashtag identities. It carries neither a gene perturbation identity nor any transcriptome measurements, so it is not a per-cell perturbation screen.
- `data/fetched/geo/GSE153056/GSE153056_RAW.tar-contents/GSM4633607_CITE_HTO_counts.tsv.gz` — review: This is a CITE-seq HTO (hashing) antibody-count matrix: rows are HTO barcodes (HTO22, HTO23, HTO25, HTO26...) and columns are cell barcodes, holding hashtag counts. It is a genes-by-cells-style feature matrix with no perturbation/condition column and no transcriptome (no gene columns), so it is not a per-cell perturbation screen.
- `data/fetched/geo/GSE153056/GSE153056_RAW.tar-contents/GSM4633608_ECCITE_Arrayed_cDNA_counts.tsv.gz` — review: This is a genes-by-cells counts matrix (16,696 columns: gene rows in the first column, cell barcodes as columns, with r_ and g_ prefixed barcodes). It carries the transcriptome but no per-cell perturbation/condition column, so it is not a per-cell screen on its own.
- `data/fetched/geo/GSE153056/GSE153056_RAW.tar-contents/GSM4633609_ECCITE_Arrayed_ADT_Barcodes.csv.gz` — review: This is a barcode-to-ADT (antibody-derived tag) mapping file for the ECCITE-seq arrayed experiment: two unnamed columns, a cell barcode and an antibody name (CD86, PDL1, PDL2, CD366). It is a feature-barcode lookup, not a per-cell perturbation screen and not a transcriptome table.
- `data/fetched/geo/GSE153056/GSE153056_RAW.tar-contents/GSM4633609_ECCITE_Arrayed_ADT_counts.tsv.gz` — review: This is an antibody-derived tag (ADT) / cell-surface protein count matrix from a CITE-seq/ECCITE experiment: rows are surface markers (CD86, PDL1, PDL2, CD366) and columns are cell barcodes. It carries no gene-expression measurements and no per-cell perturbation label column, so it cannot serve as a perturbation screen.
- `data/fetched/geo/GSE153056/GSE153056_RAW.tar-contents/GSM4633610_ECCITE_Arrayed_HTO1_Barcodes.csv.gz` — review: Two-column barcode-to-hashtag (HTO) assignment file with no header; it maps cell barcodes to hashtag labels (HTO_ETV7g1, HTO_NT5, etc.), not to gene perturbations, and carries no transcriptome. It is a cell-level metadata/assignment table, not a per-cell perturbation screen.
- … and 13 more in `data_report.md`

## What to review

1. Is the gold the table this question needs, and are its rows the right observation unit?
2. Is `gene` the right column to supervise on?
3. For a perturbation task: is that same column the perturbation identity, and are the control labels right?
4. Is any `unusable` table actually the data, one manual step away?

Fix the plan by editing `plan.json` in place (the fields above), or by overriding them on the stage-2 command line. Nothing else in this directory needs to change.

## Then run stage 2

```bash
.venv/bin/python -m kosmos.cli.main experiment "Does CRISPR knockout of the gene IRF1 change the transcriptome of human macrophages, and can the transcriptional response to the perturbation be predicted?" --data-plan artifacts/dataprep/irf1-supp-check2/plan.json --yes
```

Stage 2 writes `run/summary.md` and `run/metrics.json` beside this file; re-running it never overwrites a finished run (it lands in `run-2`, ...).

