# Data package — In K562 cells, can the transcriptional response to CRISPRi knockdown of unfolded protein response (UPR) genes not seen in training be predicted from the other knockdowns, using single-cell profiles of K562 cells from independent Perturb-seq experiments as supplementary evidence?

Stage 1 of the **fetch → review → experiment** workflow. No model has been trained yet.

- plan to review: `plan.json`
- full reasoning: `data_report.md`

## The task the fetcher read

- objective           : In K562 cells, can the transcriptional response to CRISPRi knockdown of unfolded protein response (UPR) genes not seen in training be predicted from the other knockdowns, using single-cell profiles of K562 cells from independent Perturb-seq experiments as supplementary evidence?
- target / condition  : `guide identity` (source: cli, confidence: exact_match)
- task type           : classification
- feature columns     : 5

## Gold — trainable (1)

- `data/fetched/geo/GSE90546/GSE90546_RAW.tar-contents/GSM2406675_10X001_cell_identities.csv.gz` — review: screen label half (condition column 'guide identity'); its counts matrix is joined at training time

## Supplementary — evidence only, never labels (6)

- `data/fetched/geo/GSE90063/GSE90063_k562_umi_wt.txt.gz` — review: measured transcriptome, usable as evidence for the screen
- `data/fetched/geo/GSE90546/GSE90546_RAW.tar-contents/GSM2406677_10X005_cell_identities.csv.gz` — labeled table, demoted to supplementary evidence (the primary gold is data/fetched/geo/GSE90546/GSE90546_RAW.tar-contents/GSM2406675_10X001_cell_identities.csv.gz, named as the primary labeled table); its labels are ignored and its rows enter the semi-supervised correction
- `data/fetched/geo/GSE90546/GSE90546_RAW.tar-contents/GSM2406681_10X010_cell_identities.csv.gz` — labeled table, demoted to supplementary evidence (the primary gold is data/fetched/geo/GSE90546/GSE90546_RAW.tar-contents/GSM2406675_10X001_cell_identities.csv.gz, named as the primary labeled table); its labels are ignored and its rows enter the semi-supervised correction
- `data/fetched/geo/GSE90546/GSM2406675_10X001_matrix.mtx.txt.gz-table.csv` — labeled table, demoted to supplementary evidence (the primary gold is data/fetched/geo/GSE90546/GSE90546_RAW.tar-contents/GSM2406675_10X001_cell_identities.csv.gz, named as the primary labeled table); its labels are ignored and its rows enter the semi-supervised correction; 32939 of its column(s) are not part of this task and are ignored, e.g. ['MIR1302-10', 'FAM138A', 'OR4F5']
- `data/fetched/geo/GSE90546/GSM2406677_10X005_matrix.mtx.txt.gz-table.csv` — labeled table, demoted to supplementary evidence (the primary gold is data/fetched/geo/GSE90546/GSE90546_RAW.tar-contents/GSM2406675_10X001_cell_identities.csv.gz, named as the primary labeled table); its labels are ignored and its rows enter the semi-supervised correction; 32644 of its column(s) are not part of this task and are ignored, e.g. ['MIR1302-10', 'FAM138A', 'OR4F5']
- `data/fetched/geo/GSE90546/GSM2406681_10X010_matrix.mtx.txt.gz-table.csv` — labeled table, demoted to supplementary evidence (the primary gold is data/fetched/geo/GSE90546/GSE90546_RAW.tar-contents/GSM2406675_10X001_cell_identities.csv.gz, named as the primary labeled table); its labels are ignored and its rows enter the semi-supervised correction; 32644 of its column(s) are not part of this task and are ignored, e.g. ['MIR1302-10', 'FAM138A', 'OR4F5']

## Unusable — refused (12)

- `data/fetched/geo/GSE90546/GSE90546_RAW.tar` — review: not a table: 0 parsed column(s). A table needs at least a feature column and something else
- `data/fetched/geo/GSE90546/GSE90546_RAW.tar-contents/GSM2406675_10X001_barcodes.tsv.gz` — review: not a table: 1 parsed column(s). A table needs at least a feature column and something else
- `data/fetched/geo/GSE90546/GSE90546_RAW.tar-contents/GSM2406675_10X001_genes.tsv.gz` — review: This is a genes-by-cells feature file: two columns mapping Ensembl gene IDs to gene symbols, one row per gene, no header, no cell barcodes and no perturbation/guide column. It is the gene annotation companion to the 10X matrix, not a per-cell screen.
- `data/fetched/geo/GSE90546/GSE90546_RAW.tar-contents/GSM2406675_10X001_matrix.mtx.txt.gz` — review: not a table: 1 parsed column(s). A table needs at least a feature column and something else
- `data/fetched/geo/GSE90546/GSE90546_RAW.tar-contents/GSM2406677_10X005_barcodes.tsv.gz` — review: not a table: 1 parsed column(s). A table needs at least a feature column and something else
- `data/fetched/geo/GSE90546/GSE90546_RAW.tar-contents/GSM2406677_10X005_genes.tsv.gz` — review: This is a genes-by-cells feature file: two columns mapping Ensembl gene IDs to gene symbols, one row per gene, no header, no cell barcodes and no perturbation/guide column. It is the gene axis of the 10X matrix for sample 10X005, not a per-cell screen.
- `data/fetched/geo/GSE90546/GSE90546_RAW.tar-contents/GSM2406677_10X005_matrix.mtx.txt.gz` — review: not a table: 1 parsed column(s). A table needs at least a feature column and something else
- `data/fetched/geo/GSE90546/GSE90546_RAW.tar-contents/GSM2406681_10X010_barcodes.tsv.gz` — review: not a table: 1 parsed column(s). A table needs at least a feature column and something else
- `data/fetched/geo/GSE90546/GSE90546_RAW.tar-contents/GSM2406681_10X010_genes.tsv.gz` — review: This is a genes-by-cells feature file (gene id and gene symbol per row) from a 10x Perturb-seq experiment; it has no per-cell rows and no perturbation column, so it cannot be a screen on its own.
- `data/fetched/geo/GSE90546/GSE90546_RAW.tar-contents/GSM2406681_10X010_matrix.mtx.txt.gz` — review: not a table: 1 parsed column(s). A table needs at least a feature column and something else
- `data/fetched/geo/GSE153056/GSE153056_CITE_metadata.tsv.gz` — review: This is a per-cell CITE-seq metadata table (barcodes with RNA/ADT/HTO QC columns) from GSE153056. It has no gene-expression columns and no perturbation/guide identity column, so it is neither a screen nor trainable on its own.
- `data/fetched/geo/GSE153056/GSE153056_ECCITE_metadata.tsv.gz` — it cannot be evidence for the same model: it shares only 0 measured column(s) with the gold, and 3 are needed to correct the same model (the gold has 5 feature(s))

## What to review

1. Is the gold the table this question needs, and are its rows the right observation unit?
2. Is `guide identity` the right column to supervise on?
3. For a perturbation task: is that same column the perturbation identity, and are the control labels right?
4. Is any `unusable` table actually the data, one manual step away?

Fix the plan by editing `plan.json` in place (the fields above), or by overriding them on the stage-2 command line. Nothing else in this directory needs to change.

## Then run stage 2

```bash
.venv/bin/python -m kosmos.cli.main experiment "In K562 cells, can the transcriptional response to CRISPRi knockdown of unfolded protein response (UPR) genes not seen in training be predicted from the other knockdowns, using single-cell profiles of K562 cells from independent Perturb-seq experiments as supplementary evidence?" --data-plan artifacts/dataprep/adamson-auto/plan.json --yes
```

Stage 2 writes `run/summary.md` and `run/metrics.json` beside this file; re-running it never overwrites a finished run (it lands in `run-2`, ...).

