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

## Unusable — refused (1)

- `data/fetched/geo/GSE153056/GSE153056_RAW.tar-contents/GSM4633614_ECCITE_cDNA_counts.tsv.gz` — review: This is a genes-by-cells count matrix (20,730 columns: one gene-name column plus one column per cell barcode). Rows are genes, columns are cells, and there is no column recording which perturbation each cell carries, so it is not a per-cell perturbation screen.

## What to review

1. Is the gold the table this question needs, and are its rows the right observation unit?
2. Is `gene` the right column to supervise on?
3. For a perturbation task: is that same column the perturbation identity, and are the control labels right?
4. Is any `unusable` table actually the data, one manual step away?

Fix the plan by editing `plan.json` in place (the fields above), or by overriding them on the stage-2 command line. Nothing else in this directory needs to change.

## Then run stage 2

```bash
.venv/bin/python -m kosmos.cli.main experiment "Does CRISPR knockout of the gene JAK2 change the transcriptome of human macrophages, and can the transcriptional response to the perturbation be predicted?" --data-plan artifacts/dataprep/jak2-macrophage-screen-clean/plan.json --yes
```

Stage 2 writes `run/summary.md` and `run/metrics.json` beside this file; re-running it never overwrites a finished run (it lands in `run-2`, ...).

