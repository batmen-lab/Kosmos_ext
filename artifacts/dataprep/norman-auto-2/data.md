# Data package — In K562 cells, can the transcriptional response to CRISPR activation (CRISPRa) of two genes together be predicted from the responses to each gene activated alone?

Stage 1 of the **fetch → review → experiment** workflow. No model has been trained yet.

- plan to review: `plan.json`
- full reasoning: `data_report.md`

## The task the fetcher read

- objective           : In K562 cells, can the transcriptional response to CRISPR activation (CRISPRa) of two genes together be predicted from the responses to each gene activated alone?
- target / condition  : `guide_identity` (source: cli, confidence: exact_match)
- task type           : classification
- feature columns     : 6

## Gold — trainable (1)

- `data/fetched/geo/GSE133344/GSE133344_filtered_cell_identities.csv.gz` — review: screen label half (condition column 'guide_identity'); its counts matrix is joined at training time

## Supplementary — evidence only, never labels (0)

- (none: the run will be plain supervised training)

## Unusable — refused (2)

- `data/fetched/geo/GSE133344/GSE133344_filtered_matrix.mtx.gz` — review: not a table: 1 parsed column(s). A table needs at least a feature column and something else
- `data/fetched/geo/GSE133344/GSE133344_raw_cell_identities.csv.gz` — not a table: 0 parsed column(s). A table needs at least a feature column and something else

## What to review

1. Is the gold the table this question needs, and are its rows the right observation unit?
2. Is `guide_identity` the right column to supervise on?
3. For a perturbation task: is that same column the perturbation identity, and are the control labels right?
4. Is any `unusable` table actually the data, one manual step away?

Fix the plan by editing `plan.json` in place (the fields above), or by overriding them on the stage-2 command line. Nothing else in this directory needs to change.

## Then run stage 2

```bash
.venv/bin/python -m kosmos.cli.main experiment "In K562 cells, can the transcriptional response to CRISPR activation (CRISPRa) of two genes together be predicted from the responses to each gene activated alone?" --data-plan artifacts/dataprep/norman-auto-2/plan.json --yes
```

Stage 2 writes `run/summary.md` and `run/metrics.json` beside this file; re-running it never overwrites a finished run (it lands in `run-2`, ...).

