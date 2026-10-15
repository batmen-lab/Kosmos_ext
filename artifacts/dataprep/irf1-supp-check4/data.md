# Data package — Does CRISPR knockout of the gene IRF1 change the transcriptome of human macrophages, and can the transcriptional response to the perturbation be predicted?

Stage 1 of the **fetch → review → experiment** workflow. No model has been trained yet.

- plan to review: `plan.json`
- full reasoning: `data_report.md`

## The task the fetcher read

- objective           : Does CRISPR knockout of the gene IRF1 change the transcriptome of human macrophages, and can the transcriptional response to the perturbation be predicted?
- target / condition  : `guide_ID` (source: cli, confidence: exact_match)
- task type           : classification
- feature columns     : 18647

## Gold — trainable (1)

- `data/fetched/geo/GSE153056/GSE153056_ECCITE_metadata.tsv.gz` — review: screen label half (condition column 'guide_ID'); its counts matrix is joined at training time

## Supplementary — evidence only, never labels (1)

- `data/fetched/geo/GSE153056/GSE153056_RAW.tar-contents/GSM4633614_ECCITE_cDNA_counts.tsv.gz` — review: measured transcriptome, usable as evidence for the screen; 2 column(s) were left out of the intersection because more than one of the table's columns normalises to the gold's name, e.g. ['C2ORF15', 'C2orf15']; 2 of its column(s) are not part of this task and are ignored, e.g. ['C2ORF15', 'C2orf15']; it shares 18647 of the gold's 18649 feature(s) and the run trains on that intersection

## Unusable — refused (0)

- (none)

## What to review

1. Is the gold the table this question needs, and are its rows the right observation unit?
2. Is `guide_ID` the right column to supervise on?
3. For a perturbation task: is that same column the perturbation identity, and are the control labels right?
4. Is any `unusable` table actually the data, one manual step away?

Fix the plan by editing `plan.json` in place (the fields above), or by overriding them on the stage-2 command line. Nothing else in this directory needs to change.

## Then run stage 2

```bash
.venv/bin/python -m kosmos.cli.main experiment "Does CRISPR knockout of the gene IRF1 change the transcriptome of human macrophages, and can the transcriptional response to the perturbation be predicted?" --data-plan artifacts/dataprep/irf1-supp-check4/plan.json --yes
```

Stage 2 writes `run/summary.md` and `run/metrics.json` beside this file; re-running it never overwrites a finished run (it lands in `run-2`, ...).

