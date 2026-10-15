# Data package — In primary human CD4+ T cells, can the transcriptional response to CRISPR knockout of transcriptional regulators not seen in training be predicted from the other knockouts, using large single-cell atlases of unperturbed human T cells as supplementary evidence?

Stage 1 of the **fetch → review → experiment** workflow. No model has been trained yet.

- plan to review: `plan.json`
- full reasoning: `data_report.md`

## The task the fetcher read

- objective           : In primary human CD4+ T cells, can the transcriptional response to CRISPR knockout of transcriptional regulators not seen in training be predicted from the other knockouts, using large single-cell atlases of unperturbed human T cells as supplementary evidence?
- target / condition  : `gRNA.ID` (source: objective, confidence: inferred)
- task type           : classification
- feature columns     : 2

## Gold — trainable (1)

- `data/fetched/geo/GSE119450/GSE119450_RAW.tar-contents/GSM3375487_D1N_CellBC_sgRNA.csv.gz` — review: screen label half (condition column 'gRNA.ID'); its counts matrix is joined at training time

## Supplementary — evidence only, never labels (7)

- `data/fetched/geo/GSE119450/GSE119450_RAW.tar-contents/GSM3375490_D2S_CellBC_sgRNA.csv.gz` — labeled table, demoted to supplementary evidence (the primary gold is data/fetched/geo/GSE119450/GSE119450_RAW.tar-contents/GSM3375487_D1N_CellBC_sgRNA.csv.gz, named as the primary labeled table); its labels are ignored and its rows enter the semi-supervised correction
- `data/fetched/geo/GSE119450/GSE119450_RAW.tar-contents/GSM3375488_D1S_CellBC_sgRNA.csv.gz` — labeled table, demoted to supplementary evidence (the primary gold is data/fetched/geo/GSE119450/GSE119450_RAW.tar-contents/GSM3375487_D1N_CellBC_sgRNA.csv.gz, named as the primary labeled table); its labels are ignored and its rows enter the semi-supervised correction
- `data/fetched/geo/GSE119450/GSE119450_RAW.tar-contents/GSM3375489_D2N_CellBC_sgRNA.csv.gz` — labeled table, demoted to supplementary evidence (the primary gold is data/fetched/geo/GSE119450/GSE119450_RAW.tar-contents/GSM3375487_D1N_CellBC_sgRNA.csv.gz, named as the primary labeled table); its labels are ignored and its rows enter the semi-supervised correction
- `data/fetched/geo/GSE119450/GSM3375483_D1N_matrix__matrix.mtx-table.csv` — review: measured transcriptome, usable as evidence for the screen
- `data/fetched/geo/GSE119450/GSM3375484_D1S_matrix__matrix.mtx-table.csv` — review: measured transcriptome, usable as evidence for the screen
- `data/fetched/geo/GSE119450/GSM3375485_D2N_matrix__matrix.mtx-table.csv` — review: measured transcriptome, usable as evidence for the screen
- `data/fetched/geo/GSE119450/GSM3375486_D2S_matrix__matrix.mtx-table.csv` — review: measured transcriptome, usable as evidence for the screen

## Unusable — refused (31)

- `data/fetched/geo/GSE119450/GSM3375483_D1N_matrix.tar.gz-contents/matrix.mtx` — review: not a table: 1 parsed column(s). A table needs at least a feature column and something else
- `data/fetched/geo/GSE119450/GSM3375484_D1S_matrix.tar.gz-contents/matrix.mtx` — review: not a table: 1 parsed column(s). A table needs at least a feature column and something else
- `data/fetched/geo/GSE119450/GSM3375485_D2N_matrix.tar.gz-contents/matrix.mtx` — review: not a table: 1 parsed column(s). A table needs at least a feature column and something else
- `data/fetched/geo/GSE119450/GSM3375486_D2S_matrix.tar.gz-contents/matrix.mtx` — review: not a table: 1 parsed column(s). A table needs at least a feature column and something else
- `data/fetched/geo/GSE119450/GSE119450_RAW.tar` — review: not a table: 0 parsed column(s). A table needs at least a feature column and something else
- `data/fetched/geo/GSE119450/GSE119450_RAW.tar-contents/GSM3375483_D1N_matrix.tar.gz` — review: not a table: 1 parsed column(s). A table needs at least a feature column and something else
- `data/fetched/geo/GSE119450/GSE119450_RAW.tar-contents/GSM3375484_D1S_matrix.tar.gz` — review: not a table: 1 parsed column(s). A table needs at least a feature column and something else
- `data/fetched/geo/GSE119450/GSE119450_RAW.tar-contents/GSM3375485_D2N_matrix.tar.gz` — review: not a table: 1 parsed column(s). A table needs at least a feature column and something else
- `data/fetched/geo/GSE119450/GSE119450_RAW.tar-contents/GSM3375486_D2S_matrix.tar.gz` — review: not a table: 1 parsed column(s). A table needs at least a feature column and something else
- `data/fetched/geo/GSE119450/GSM3375483_D1N_matrix.tar.gz-contents/barcodes.tsv` — review: not a table: 1 parsed column(s). A table needs at least a feature column and something else
- `data/fetched/geo/GSE119450/GSM3375483_D1N_matrix.tar.gz-contents/genes.tsv` — review: This is a two-column gene annotation table (Ensembl gene id, gene symbol) with no header and no cells, no perturbations and no expression values. It is a feature reference, not a per-cell screen.
- `data/fetched/geo/GSE119450/GSM3375484_D1S_matrix.tar.gz-contents/barcodes.tsv` — review: not a table: 1 parsed column(s). A table needs at least a feature column and something else
- … and 19 more in `data_report.md`

## What to review

1. Is the gold the table this question needs, and are its rows the right observation unit?
2. Is `gRNA.ID` the right column to supervise on?
3. For a perturbation task: is that same column the perturbation identity, and are the control labels right?
4. Is any `unusable` table actually the data, one manual step away?

Fix the plan by editing `plan.json` in place (the fields above), or by overriding them on the stage-2 command line. Nothing else in this directory needs to change.

## Then run stage 2

```bash
.venv/bin/python -m kosmos.cli.main experiment "In primary human CD4+ T cells, can the transcriptional response to CRISPR knockout of transcriptional regulators not seen in training be predicted from the other knockouts, using large single-cell atlases of unperturbed human T cells as supplementary evidence?" --data-plan artifacts/dataprep/tcell-small-gold/plan.json --yes
```

Stage 2 writes `run/summary.md` and `run/metrics.json` beside this file; re-running it never overwrites a finished run (it lands in `run-2`, ...).

