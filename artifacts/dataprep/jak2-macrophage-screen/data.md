# Data package — Does CRISPR knockout of the gene JAK2 change the transcriptome of human macrophages, and can the transcriptional response to the perturbation be predicted?

Stage 1 of the **fetch → review → experiment** workflow. No model has been trained yet.

- plan to review: `plan.json`
- full reasoning: `data_report.md`

## The task the fetcher read

- objective           : Does CRISPR knockout of the gene JAK2 change the transcriptome of human macrophages, and can the transcriptional response to the perturbation be predicted?
- target / condition  : `gene` (source: cli, confidence: exact_match)
- task type           : classification
- feature columns     : 21

## Gold — trainable (1)

- `data/fetched/geo/GSE153056/GSE153056_ECCITE_metadata.tsv.gz` — target column 'gene' is present with 21 feature(s) (10 categorical); this table can train, select and test; 3 column(s) are not features (free text, a row index, or empty), e.g. ['Unnamed: 0', 'guide_ID', 'NT']

## Supplementary — evidence only, never labels (0)

- (none: the run will be plain supervised training)

## Unusable — refused (3)

- `data/fetched/geo/GSE153056/GSE153056_RAW.tar-contents/GSM4633608_ECCITE_Arrayed_cDNA_counts.tsv.gz` — review: This is a genes-by-cells count matrix (genes as rows, cell barcodes as columns) from the ECCITE-seq dataset GSE153056. The first column holds gene names and the remaining 16,695 columns are individual cell barcodes (r_* for RNA, g_* for gRNA). There is no per-cell condition column here; the perturbation identities live in the companion metadata file.
- `data/fetched/geo/GSE153056/GSE153056_CITE_metadata.tsv.gz` — review: This is a per-cell CITE-seq metadata table (barcodes with RNA/ADT/HTO QC columns and hashtag classification). It has no gene-expression columns and no CRISPR perturbation column, so it is not a trainable screen.
- `data/fetched/geo/GSE153056/GSE153056_RAW.tar-contents/GSM4633605_CITE_cDNA_counts.tsv.gz` — review: This is a genes-by-cells count matrix (gene symbols as rows, cell barcodes as columns) with no perturbation/condition column; the label 'gene' is the row index (gene names), not a per-cell perturbation identity.

## What to review

1. Is the gold the table this question needs, and are its rows the right observation unit?
2. Is `gene` the right column to supervise on?
3. For a perturbation task: is that same column the perturbation identity, and are the control labels right?
4. Is any `unusable` table actually the data, one manual step away?

Fix the plan by editing `plan.json` in place (the fields above), or by overriding them on the stage-2 command line. Nothing else in this directory needs to change.

## Then run stage 2

```bash
.venv/bin/python -m kosmos.cli.main experiment "Does CRISPR knockout of the gene JAK2 change the transcriptome of human macrophages, and can the transcriptional response to the perturbation be predicted?" --data-plan artifacts/dataprep/jak2-macrophage-screen/plan.json --yes
```

Stage 2 writes `run/summary.md` and `run/metrics.json` beside this file; re-running it never overwrites a finished run (it lands in `run-2`, ...).

