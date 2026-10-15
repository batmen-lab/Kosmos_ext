# Data package — Does CRISPR knockout of the gene IRF1 change the transcriptome of human macrophages, and can the transcriptional response to the perturbation be predicted?

Stage 1 of the **fetch → review → experiment** workflow. No model has been trained yet.

- plan to review: `plan.json`
- full reasoning: `data_report.md`

## The task the fetcher read

- objective           : Does CRISPR knockout of the gene IRF1 change the transcriptome of human macrophages, and can the transcriptional response to the perturbation be predicted?
- target / condition  : `guide` (source: objective, confidence: inferred)
- task type           : classification
- feature columns     : 2

## Gold — trainable (1)

- `data/fetched/geo/GSE153056/GSE153056_RAW.tar-contents/GSM4633618_ECCITE_GDO_Barcodes.csv.gz` — review: screen label half (condition column 'guide'); its counts matrix is joined at training time

## Supplementary — evidence only, never labels (0)

- (none: the run will be plain supervised training)

## Unusable — refused (1)

- `data/fetched/geo/GSE153056/GSE153056_RAW.tar-contents/GSM4633614_ECCITE_cDNA_counts.tsv.gz` — review: This is a genes-by-cells count matrix (rows are genes like AL627309.1, columns are cell barcodes like l1_AAACCTGAGCCAGAAC). It carries the transcriptome but no perturbation/condition column, so it cannot serve as a per-cell perturbation screen on its own.

## What to review

1. Is the gold the table this question needs, and are its rows the right observation unit?
2. Is `guide` the right column to supervise on?
3. For a perturbation task: is that same column the perturbation identity, and are the control labels right?
4. Is any `unusable` table actually the data, one manual step away?

Fix the plan by editing `plan.json` in place (the fields above), or by overriding them on the stage-2 command line. Nothing else in this directory needs to change.

## Then run stage 2

```bash
.venv/bin/python -m kosmos.cli.main experiment "Does CRISPR knockout of the gene IRF1 change the transcriptome of human macrophages, and can the transcriptional response to the perturbation be predicted?" --data-plan artifacts/dataprep/irf1-labelhalf-check/plan.json --yes
```

Stage 2 writes `run/summary.md` and `run/metrics.json` beside this file; re-running it never overwrites a finished run (it lands in `run-2`, ...).

