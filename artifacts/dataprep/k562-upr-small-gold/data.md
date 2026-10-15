# Data package — In K562 cells, can the transcriptional response to CRISPRi knockdown of unfolded protein response (UPR) genes not seen in training be predicted from a small labeled screen, using large collections of K562 single-cell profiles from other CRISPR screens, with their labels ignored, as supplementary evidence?

Stage 1 of the **fetch → review → experiment** workflow. No model has been trained yet.

- plan to review: `plan.json`
- full reasoning: `data_report.md`

## The task the fetcher read

- objective           : In K562 cells, can the transcriptional response to CRISPRi knockdown of unfolded protein response (UPR) genes not seen in training be predicted from a small labeled screen, using large collections of K562 single-cell profiles from other CRISPR screens, with their labels ignored, as supplementary evidence?
- target / condition  : `gene` (source: cli, confidence: exact_match)
- task type           : classification
- feature columns     : 4

## Gold — trainable (1)

- `data/fetched/geo/GSE132080/GSE132080_sgRNA_barcode_sequences_and_phenotypes.csv.gz` — review: screen label half (condition column 'gene'); its counts matrix is joined at training time

## Supplementary — evidence only, never labels (2)

- `data/fetched/geo/GSE132080/GSE132080_10X_matrix.mtx.gz-table.csv` — review: measured transcriptome, usable as evidence for the screen
- `data/fetched/geo/GSE308682/GSE308682_filtered_matrix.mtx.gz-table.csv` — review: measured transcriptome, usable as evidence for the screen

## Unusable — refused (28)

- `data/fetched/geo/GSE90546/GSE90546_RAW.tar` — review: not a table: 0 parsed column(s). A table needs at least a feature column and something else
- `data/fetched/geo/GSE90546/GSE90546_RAW.tar-contents/GSM2406675_10X001_barcodes.tsv.gz` — review: not a table: 1 parsed column(s). A table needs at least a feature column and something else
- `data/fetched/geo/GSE90546/GSE90546_RAW.tar-contents/GSM2406675_10X001_genes.tsv.gz` — review: This is a genes-by-cells feature file: each row is a gene (Ensembl id + symbol), not a single perturbed cell, and there is no guide/condition column. It carries the transcriptome measurements but no perturbation labels.
- `data/fetched/geo/GSE90546/GSE90546_RAW.tar-contents/GSM2406675_10X001_matrix.mtx.txt.gz` — review: not a table: 1 parsed column(s). A table needs at least a feature column and something else
- `data/fetched/geo/GSE90546/GSE90546_RAW.tar-contents/GSM2406677_10X005_barcodes.tsv.gz` — review: not a table: 1 parsed column(s). A table needs at least a feature column and something else
- `data/fetched/geo/GSE90546/GSE90546_RAW.tar-contents/GSM2406677_10X005_genes.tsv.gz` — review: Two-column gene annotation table (Ensembl gene id and gene symbol) with no header and no cells or perturbations; it is the gene list for the 10X005 matrix, not a per-cell screen.
- `data/fetched/geo/GSE90546/GSE90546_RAW.tar-contents/GSM2406677_10X005_matrix.mtx.txt.gz` — review: not a table: 1 parsed column(s). A table needs at least a feature column and something else
- `data/fetched/geo/GSE90546/GSE90546_RAW.tar-contents/GSM2406681_10X010_barcodes.tsv.gz` — review: not a table: 1 parsed column(s). A table needs at least a feature column and something else
- `data/fetched/geo/GSE90546/GSE90546_RAW.tar-contents/GSM2406681_10X010_genes.tsv.gz` — review: Two-column gene annotation table (Ensembl gene id and gene symbol) with no header and no per-cell rows; it is the gene feature list for the 10X matrix, not a perturbation screen.
- `data/fetched/geo/GSE90546/GSE90546_RAW.tar-contents/GSM2406681_10X010_matrix.mtx.txt.gz` — review: not a table: 1 parsed column(s). A table needs at least a feature column and something else
- `data/fetched/geo/GSE132080/GSE132080_10X_matrix.mtx.gz` — review: not a table: 1 parsed column(s). A table needs at least a feature column and something else
- `data/fetched/geo/GSE132080/GSE132080_10X_barcodes.tsv.gz` — review: not a table: 1 parsed column(s). A table needs at least a feature column and something else
- … and 16 more in `data_report.md`

## What to review

1. Is the gold the table this question needs, and are its rows the right observation unit?
2. Is `gene` the right column to supervise on?
3. For a perturbation task: is that same column the perturbation identity, and are the control labels right?
4. Is any `unusable` table actually the data, one manual step away?

Fix the plan by editing `plan.json` in place (the fields above), or by overriding them on the stage-2 command line. Nothing else in this directory needs to change.

## Then run stage 2

```bash
.venv/bin/python -m kosmos.cli.main experiment "In K562 cells, can the transcriptional response to CRISPRi knockdown of unfolded protein response (UPR) genes not seen in training be predicted from a small labeled screen, using large collections of K562 single-cell profiles from other CRISPR screens, with their labels ignored, as supplementary evidence?" --data-plan artifacts/dataprep/k562-upr-small-gold/plan.json --yes
```

Stage 2 writes `run/summary.md` and `run/metrics.json` beside this file; re-running it never overwrites a finished run (it lands in `run-2`, ...).

