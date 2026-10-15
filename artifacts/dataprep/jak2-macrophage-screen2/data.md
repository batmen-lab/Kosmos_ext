# Data package — Does CRISPR knockout of the gene JAK2 change the transcriptome of human macrophages, and can the transcriptional response to the perturbation be predicted?

Stage 1 of the **fetch → review → experiment** workflow. No model has been trained yet.

- plan to review: `plan.json`
- full reasoning: `data_report.md`

## The task the fetcher read

- objective           : Does CRISPR knockout of the gene JAK2 change the transcriptome of human macrophages, and can the transcriptional response to the perturbation be predicted?
- target / condition  : `guide_ID` (source: cli, confidence: exact_match)
- task type           : classification
- feature columns     : 22

## Gold — trainable (1)

- `data/fetched/geo/GSE153056/GSE153056_ECCITE_metadata.tsv.gz` — target column 'guide_ID' is present with 22 feature(s) (11 categorical); this table can train, select and test; 2 column(s) are not features (free text, a row index, or empty), e.g. ['Unnamed: 0', 'NT']

## Supplementary — evidence only, never labels (0)

- (none: the run will be plain supervised training)

## Unusable — refused (12)

- `data/fetched/geo/GSE153056/GSE153056_RAW.tar-contents/GSM4633614_ECCITE_cDNA_counts.tsv.gz` — review: This is a genes-by-cells count matrix (20,730 columns: first column holds gene names, remaining columns are cell barcodes like l1_AAACCTGAGCCAGAAC). It carries the transcriptome but no perturbation/condition column, so it is not a per-cell screen on its own.
- `data/fetched/geo/GSE194122/GSE194122_openproblems_neurips2021_cite_BMMC_processed.h5ad.gz` — review: not a table: 0 parsed column(s). A table needs at least a feature column and something else
- `data/fetched/geo/GSE194122/GSE194122_openproblems_neurips2021_cite_BMMC_processed.h5ad.gz-table.csv` — review: This is the NeurIPS 2021 open-problems CITE-seq BMMC dataset (GSE194122): a per-cell table of 2,000 sampled bone-marrow mononuclear cells with 14,087 gene columns plus ADT protein columns and cell metadata. It carries the transcriptome and a cell_type label, but there is no perturbation/condition column — no guide, sgRNA or target gene — so it is not a CRISPR screen and cannot answer a JAK2 knockout question.
- `data/fetched/geo/GSE171406/GSE171406_RAW.tar` — review: not a table: 0 parsed column(s). A table needs at least a feature column and something else
- `data/fetched/geo/GSE171406/GSE171406_RAW.tar-contents/GSM5224978_A1.tif.gz` — review: not a table: 0 parsed column(s). A table needs at least a feature column and something else
- `data/fetched/geo/GSE171406/GSE171406_RAW.tar-contents/GSM5224978_counts_A1_S1.csv.gz` — review: This is a genes-by-cells count matrix (rows are genes, columns are cell barcodes) with no per-cell perturbation/condition column. The first line is data (barcodes as column names, no header row), and the first column holds gene names. There is no column recording which guide or target gene each cell carries, so it cannot serve as a per-cell perturbation screen on its own.
- `data/fetched/geo/GSE171406/GSE171406_RAW.tar-contents/GSM5224979_B1.tif.gz` — review: not a table: 0 parsed column(s). A table needs at least a feature column and something else
- `data/fetched/geo/GSE171406/GSE171406_RAW.tar-contents/GSM5224979_counts_B1_S2.csv.gz` — review: This is a genes-by-cells count matrix (rows are genes like Xkr4, Gm1992; columns are cell barcodes). It carries the transcriptome but no perturbation/condition column, so it cannot be a per-cell screen on its own.
- `data/fetched/geo/GSE171406/GSE171406_RAW.tar-contents/GSM5224980_C1.tif.gz` — review: not a table: 0 parsed column(s). A table needs at least a feature column and something else
- `data/fetched/geo/GSE171406/GSE171406_RAW.tar-contents/GSM5224980_counts_C1_S3.csv.gz` — review: This is a genes-by-cells count matrix (rows are genes like Xkr4, Gm1992; columns are cell barcodes) with no perturbation/condition column. It is one of the GSE171406 per-sample count matrices, so it carries the transcriptome but not the guide identity.
- `data/fetched/geo/GSE171406/GSE171406_RAW.tar-contents/GSM5224981_D1.tif.gz` — review: not a table: 0 parsed column(s). A table needs at least a feature column and something else
- `data/fetched/geo/GSE171406/GSE171406_RAW.tar-contents/GSM5224981_counts_D1_S4.csv.gz` — review: This is a genes-by-cells count matrix (rows are genes like Xkr4, Gm1992; columns are cell barcodes) with no perturbation/condition column. It carries the transcriptome but not the perturbation identity, so it cannot be a per-cell screen on its own.

## What to review

1. Is the gold the table this question needs, and are its rows the right observation unit?
2. Is `guide_ID` the right column to supervise on?
3. For a perturbation task: is that same column the perturbation identity, and are the control labels right?
4. Is any `unusable` table actually the data, one manual step away?

Fix the plan by editing `plan.json` in place (the fields above), or by overriding them on the stage-2 command line. Nothing else in this directory needs to change.

## Then run stage 2

```bash
.venv/bin/python -m kosmos.cli.main experiment "Does CRISPR knockout of the gene JAK2 change the transcriptome of human macrophages, and can the transcriptional response to the perturbation be predicted?" --data-plan artifacts/dataprep/jak2-macrophage-screen2/plan.json --yes
```

Stage 2 writes `run/summary.md` and `run/metrics.json` beside this file; re-running it never overwrites a finished run (it lands in `run-2`, ...).

