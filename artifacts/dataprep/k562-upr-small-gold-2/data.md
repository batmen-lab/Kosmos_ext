# Data package — In K562 cells, can the transcriptional response to CRISPRi knockdown of unfolded protein response (UPR) genes not seen in training be predicted from a small labeled screen, using large collections of K562 single-cell profiles from other CRISPR screens, with their labels ignored, as supplementary evidence?

Stage 1 of the **fetch → review → experiment** workflow. No model has been trained yet.

- plan to review: `plan.json`
- full reasoning: `data_report.md`

## The task the fetcher read

- objective           : In K562 cells, can the transcriptional response to CRISPRi knockdown of unfolded protein response (UPR) genes not seen in training be predicted from a small labeled screen, using large collections of K562 single-cell profiles from other CRISPR screens, with their labels ignored, as supplementary evidence?
- target / condition  : `guide identity` (source: cli, confidence: exact_match)
- task type           : classification
- feature columns     : 30314

## Gold — trainable (1)

- `data/fetched/geo/GSE90546/GSM2406675_10X001_matrix.mtx.txt.gz-table.csv` — target column 'guide identity' is present with 32944 feature(s) (31946 categorical); this table can train, select and test

## Supplementary — evidence only, never labels (3)

- `data/fetched/geo/GSE308682/GSE308682_filtered_matrix.mtx.gz-table.csv` — review: measured transcriptome, usable as evidence for the screen; 2 column(s) were left out of the intersection because more than one of the table's columns normalises to the gold's name, e.g. ['C1ORF220', 'C2ORF15']; 3343 of its column(s) are not part of this task and are ignored, e.g. ['RP11-34P13.3', 'FO538757.3', 'FO538757.2']; 1 column(s) matched the gold after normalising the name, e.g. [('ZHX1-C8orf76', 'ZHX1-C8ORF76')]; it shares 30344 of the gold's 32944 feature(s) and the run trains on that intersection
- `data/fetched/geo/GSE90546/GSM2406677_10X005_matrix.mtx.txt.gz-table.csv` — labeled table, demoted to supplementary evidence (the primary gold is data/fetched/geo/GSE90546/GSM2406675_10X001_matrix.mtx.txt.gz-table.csv, named as the primary labeled table); its labels are ignored and its rows enter the semi-supervised correction; 8 column(s) were left out of the intersection because more than one of the table's columns normalises to the gold's name, e.g. ['C1orf220', 'C1ORF220', 'C2orf15']; 8 of its column(s) are not part of this task and are ignored, e.g. ['C1orf220', 'C1ORF220', 'C2orf15']; it shares 32641 of the gold's 32944 feature(s) and the run trains on that intersection
- `data/fetched/geo/GSE90546/GSM2406681_10X010_matrix.mtx.txt.gz-table.csv` — labeled table, demoted to supplementary evidence (the primary gold is data/fetched/geo/GSE90546/GSM2406675_10X001_matrix.mtx.txt.gz-table.csv, named as the primary labeled table); its labels are ignored and its rows enter the semi-supervised correction; 8 column(s) were left out of the intersection because more than one of the table's columns normalises to the gold's name, e.g. ['C1orf220', 'C1ORF220', 'C2orf15']; 8 of its column(s) are not part of this task and are ignored, e.g. ['C1orf220', 'C1ORF220', 'C2orf15']; it shares 32641 of the gold's 32944 feature(s) and the run trains on that intersection

## Unusable — refused (16)

- `data/fetched/geo/GSE90546/GSE90546_RAW.tar` — review: not a table: 0 parsed column(s). A table needs at least a feature column and something else
- `data/fetched/geo/GSE90546/GSE90546_RAW.tar-contents/GSM2406675_10X001_barcodes.tsv.gz` — review: not a table: 1 parsed column(s). A table needs at least a feature column and something else
- `data/fetched/geo/GSE90546/GSE90546_RAW.tar-contents/GSM2406675_10X001_genes.tsv.gz` — review: This is a genes-by-cells feature file (gene id and gene symbol per row), not a per-cell table. It carries no barcode rows and no guide identity column, so it cannot be a per-cell perturbation screen.
- `data/fetched/geo/GSE90546/GSE90546_RAW.tar-contents/GSM2406675_10X001_matrix.mtx.txt.gz` — review: not a table: 1 parsed column(s). A table needs at least a feature column and something else
- `data/fetched/geo/GSE90546/GSE90546_RAW.tar-contents/GSM2406677_10X005_barcodes.tsv.gz` — review: not a table: 1 parsed column(s). A table needs at least a feature column and something else
- `data/fetched/geo/GSE90546/GSE90546_RAW.tar-contents/GSM2406677_10X005_genes.tsv.gz` — review: This is a genes-by-cells feature file (gene id and gene symbol per row), not a per-cell table; it carries no barcodes and no guide identity column, so it cannot be a screen on its own.
- `data/fetched/geo/GSE90546/GSE90546_RAW.tar-contents/GSM2406677_10X005_matrix.mtx.txt.gz` — review: not a table: 1 parsed column(s). A table needs at least a feature column and something else
- `data/fetched/geo/GSE90546/GSE90546_RAW.tar-contents/GSM2406681_10X010_barcodes.tsv.gz` — review: not a table: 1 parsed column(s). A table needs at least a feature column and something else
- `data/fetched/geo/GSE90546/GSE90546_RAW.tar-contents/GSM2406681_10X010_genes.tsv.gz` — review: Two-column gene annotation table (Ensembl gene id, gene symbol) with no header and no per-cell rows; it is the gene list for the 10X010 sample, not a perturbation screen.
- `data/fetched/geo/GSE90546/GSE90546_RAW.tar-contents/GSM2406681_10X010_matrix.mtx.txt.gz` — review: not a table: 1 parsed column(s). A table needs at least a feature column and something else
- `data/fetched/geo/GSE308682/GSE308682_filtered_matrix.mtx.gz` — review: not a table: 1 parsed column(s). A table needs at least a feature column and something else
- `data/fetched/geo/GSE308682/GSE308682_filtered_barcodes.tsv.gz` — review: not a table: 1 parsed column(s). A table needs at least a feature column and something else
- … and 4 more in `data_report.md`

## What to review

1. Is the gold the table this question needs, and are its rows the right observation unit?
2. Is `guide identity` the right column to supervise on?
3. For a perturbation task: is that same column the perturbation identity, and are the control labels right?
4. Is any `unusable` table actually the data, one manual step away?

Fix the plan by editing `plan.json` in place (the fields above), or by overriding them on the stage-2 command line. Nothing else in this directory needs to change.

## Then run stage 2

```bash
.venv/bin/python -m kosmos.cli.main experiment "In K562 cells, can the transcriptional response to CRISPRi knockdown of unfolded protein response (UPR) genes not seen in training be predicted from a small labeled screen, using large collections of K562 single-cell profiles from other CRISPR screens, with their labels ignored, as supplementary evidence?" --data-plan artifacts/dataprep/k562-upr-small-gold-2/plan.json --yes
```

Stage 2 writes `run/summary.md` and `run/metrics.json` beside this file; re-running it never overwrites a finished run (it lands in `run-2`, ...).

