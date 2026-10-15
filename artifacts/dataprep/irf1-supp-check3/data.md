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

- `data/fetched/geo/GSE153056/GSE153056_RAW.tar-contents/GSM4633614_ECCITE_cDNA_counts.tsv.gz` — review: This is a genes-by-cells count matrix (genes as rows, cell barcodes as columns) with no perturbation/condition column; the perturbation identities live in the paired ECCITE metadata file.
- `data/fetched/geo/GSE153056/GSE153056_ECCITE_Arrayed_metadata.tsv.gz` — review: This is a per-cell metadata table (barcodes with HTO/ADT/GO-lenti/GO-cite QC and classification columns plus nCount_RNA/nFeature_RNA/percent.mito). It carries perturbation identities (GO_lenti_maxID, GO_cite_maxID) but no gene expression columns, so it is not a trainable screen on its own.
- `data/fetched/geo/GSE153056/GSE153056_RAW.tar` — review: not a table: 0 parsed column(s). A table needs at least a feature column and something else
- `data/fetched/geo/GSE153056/GSE153056_RAW.tar-contents/GSM4633605_CITE_cDNA_counts.tsv.gz` — review: This is a genes-by-cells count matrix (gene symbols as rows, cell barcodes as columns) with no perturbation/condition column. It is the CITE cDNA counts for the GSE153056 ECCITE/CITE experiment; the perturbation labels live in the metadata tables, not here.
- `data/fetched/geo/GSE153056/GSE153056_RAW.tar-contents/GSM4633606_CITE_ADT_Barcodes.csv.gz` — review: This is a barcode-to-antibody (ADT/CITE-seq) feature file: each row maps a cell barcode to a single antibody-derived tag name (PDL1, CD86, IDO1, TIM3, PDL2). It has no header, no gene-expression columns, and no CRISPR perturbation column, so it is not a per-cell perturbation screen.
- `data/fetched/geo/GSE153056/GSE153056_RAW.tar-contents/GSM4633606_CITE_ADT_counts.tsv.gz` — review: This is a CITE-seq antibody-derived tag (ADT) protein count matrix: rows are surface-marker proteins (BTLA, CD47, CD66a, CD86, ...) and columns are cell barcodes. It carries no gene expression and no perturbation/condition column, so it is not a per-cell perturbation screen.
- `data/fetched/geo/GSE153056/GSE153056_RAW.tar-contents/GSM4633607_CITE_HTO_Barcodes.csv.gz` — review: This is a barcode-to-HTO (hashing antibody) assignment file: two columns, a cell barcode and an HTO tag. It carries no gene expression and no CRISPR perturbation identity, so it is not a per-cell perturbation screen.
- `data/fetched/geo/GSE153056/GSE153056_RAW.tar-contents/GSM4633607_CITE_HTO_counts.tsv.gz` — review: This is a CITE-seq HTO (hashing) antibody-count matrix: rows are hashtag oligos (HTO22, HTO23, HTO25, HTO26) and columns are cell barcodes, with no gene/transcriptome columns and no perturbation condition column. It is a genes-by-cells style feature matrix, not a per-cell perturbation screen.
- `data/fetched/geo/GSE153056/GSE153056_RAW.tar-contents/GSM4633608_ECCITE_Arrayed_cDNA_counts.tsv.gz` — review: This is a genes-by-cells counts matrix: the first column holds gene names (AL627309.1, AP006222.2, ...) and the remaining 16,695 columns are cell barcodes (r_* for RNA, g_* for gRNA). There is no per-cell condition column here; the perturbation identity lives in the paired metadata file.
- `data/fetched/geo/GSE153056/GSE153056_RAW.tar-contents/GSM4633609_ECCITE_Arrayed_ADT_Barcodes.csv.gz` — review: This is a two-column barcode-to-ADT (antibody-derived tag) mapping file for the ECCITE-seq arrayed experiment: each row pairs a cell barcode with an antibody name (CD86, PDL1, PDL2, CD366). It carries no gene expression and no CRISPR perturbation identity, so it is not a per-cell perturbation screen.
- `data/fetched/geo/GSE153056/GSE153056_RAW.tar-contents/GSM4633609_ECCITE_Arrayed_ADT_counts.tsv.gz` — review: This is an antibody-derived tag (ADT) / cell-surface protein count matrix from a CITE-seq/ECCITE experiment: rows are surface markers (CD86, PDL1, PDL2, CD366) and columns are cell barcodes. It carries no gene-expression transcriptome and no perturbation-identity column, so it cannot serve as a per-cell perturbation screen for a transcriptome-response question.
- `data/fetched/geo/GSE153056/GSE153056_RAW.tar-contents/GSM4633610_ECCITE_Arrayed_HTO1_Barcodes.csv.gz` — review: Two-column barcode-to-hashtag (HTO) assignment file: each row is a cell barcode and its hashtag tag (HTO_ETV7g1, HTO_NT5, ...), not a CRISPR perturbation and not a transcriptome. It carries no gene columns and no guide/sgRNA column, so it is not a per-cell perturbation screen.
- … and 13 more in `data_report.md`

## What to review

1. Is the gold the table this question needs, and are its rows the right observation unit?
2. Is `gene` the right column to supervise on?
3. For a perturbation task: is that same column the perturbation identity, and are the control labels right?
4. Is any `unusable` table actually the data, one manual step away?

Fix the plan by editing `plan.json` in place (the fields above), or by overriding them on the stage-2 command line. Nothing else in this directory needs to change.

## Then run stage 2

```bash
.venv/bin/python -m kosmos.cli.main experiment "Does CRISPR knockout of the gene IRF1 change the transcriptome of human macrophages, and can the transcriptional response to the perturbation be predicted?" --data-plan artifacts/dataprep/irf1-supp-check3/plan.json --yes
```

Stage 2 writes `run/summary.md` and `run/metrics.json` beside this file; re-running it never overwrites a finished run (it lands in `run-2`, ...).

