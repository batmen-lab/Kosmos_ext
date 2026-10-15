# Handoff — IRF1 macrophage perturbation: the three-stage design

_Last updated: 2026-10-06. Scope: the `Perturb_middle` workflow only. The repo
root `handoff.md` is the general project handoff; this file is about the
fetch → subset → train split for the IRF1 screen._

## 1. The task

> Does CRISPR knockout of the gene IRF1 change the transcriptome of human
> macrophages, and can the transcriptional response to the perturbation be
> predicted?

Data: **GSE153056** (ECCITE-seq, human macrophages/THP-1), 20,729 cells, 25
target genes + `NT` controls. The answer the design is after is a
**cross-perturbation** one: train on other perturbations, hold IRF1 out, and see
whether its response is predictable.

## 2. The three-stage design

The whole point is that a person reviews the data between fetching and training,
and the only handoff is `plan.json` (the same contract the existing `kosmos run`
paths use).

```
stage 1  fetch     kosmos dataprep "<question>" --task perturbation --out D
                   -> D/plan.json + data.md + data_report.md + reviews.json + search_trace.json
                      gold        = the screen's LABEL half (barcode -> condition)
                      supplementary = candidate MEASUREMENT halves (counts)
                   [human: read data.md, decide/repair the roles]

stage 2  subset    Perturb_middle/subset_process_irf1.py --stage1 D [--targets ...]
                   -> gold_subset.csv.gz        labels  (barcode, condition)
                      supp_*.csv.gz             expression (barcode + panel genes)
                      plan.json                 gold + supplementary, for stage 3
                      subset_report.md/.json    what was kept and why
                   [human: pick the subset, QC the report]

stage 3  train     KOSMOS_PERTURBATION_HOLDOUT=IRF1 KOSMOS_PERTURBATION_ARMS=base,signed \
                   kosmos experiment "<question>" --data-plan <stage-2 plan> --task perturbation
                   -> run/summary.md + run/metrics.json + figures
```

Each stage can be run and verified on its own; the intermediate artifacts
(`plan.json`, `subset_report.md`) are the verification points.

## 3. Current state — what works

### Stage 1 (already run, keep this one)

Run dir: `artifacts/dataprep/irf1-run-mine/`

```
plan.json: gold 1 / supplementary 5 / unusable 20
  gold          GSE153056_ECCITE_metadata.tsv.gz            (condition = gene or guide_ID)
  supplementary GSM4633605/3608/3612/3613/3614_*counts      (the screen's own measurement halves)
  unusable      ADT/HTO protein + hashtag matrices, .tar, guide dictionaries
```

`data.md` is the human review document; `data_report.md` is the full reasoning
chain. **Round 2 did propose independent cohorts** (GSE194122, GSE171524,
GSE149689, …) but none was kept, so the plan contains GSE153056 only.

### Stage 2 (this folder)

Script: `Perturb_middle/subset_process_irf1.py`

```bash
# whole screen (25 perturbations + NT) -- the split needs many perturbations
.venv/bin/python Perturb_middle/subset_process_irf1.py --targets all

# or a few perturbations (>= 2, or training cannot split)
.venv/bin/python Perturb_middle/subset_process_irf1.py --targets IRF1,IFNGR1,IFNGR2,CD86,ATF2
```

What it does: reads the stage-1 plan, subsets the gold to the target(s) +
controls, subsets every supplementary table to the **same gene panel**, then
**log1p-CPM → HVG → per-gene z-score across cells → clip negatives to 0** and
writes the stage-3 inputs.

Output dir: `Perturb_middle/output_irf1-run-mine/` (for `--targets all`):

```
gold_subset.csv.gz   20,729 cells, 25 perturbations + NT
supp_1_GSM4633614…   20,729 cells x 2009 genes  (the gold's own cells)
supp_2_GSM4633605…    7,566 cells x 2009 genes  (CITE arm, its own cells)
supp_3_GSM4633608…   16,695 cells x 2009 genes  (Arrayed arm, its own cells)
plan.json / subset_report.md / subset_report.json
```

Notes / decisions:

* The panel is `2000 HVG ∪ {perturbed genes}`. HVG alone dropped 9 perturbation
  genes (CD86, ATF2, MARCH8, CAV1, POU2F2, STAT5A, SMAD4, PDCD1LG2, CUL3); a
  perturbation whose gene is not a panel column is dropped by the backend and
  its cells vanish, so they are force-kept.
* `GSM4633612` / `GSM4633613` are skipped: they are **guide-count (GDO)
  matrices** (rows are `eGFPg1`, `CUL3g1`, …), 0 rows overlap the gene panel, so
  they are not expression evidence. The report says so under Warnings.
* Supplementary tables keep **all** their cells (`--supp-max-cells 0`). A cap
  was tried and removed: it discarded evidence for no reason.

### Stage 3 (verified to run)

```bash
KOSMOS_PERTURBATION_HOLDOUT=IRF1 KOSMOS_PERTURBATION_ARMS=base,signed \
.venv/bin/python -m kosmos.cli.main experiment \
  "Does CRISPR knockout of the gene IRF1 change the transcriptome of human macrophages, and can the transcriptional response to the perturbation be predicted?" \
  --data-plan Perturb_middle/output_irf1-run-mine/plan.json \
  --task perturbation --split-mode unseen_single --max-epochs 50 --yes
```

Latest verified run: `Perturb_middle/output_irf1-run-mine/run-9/`

```
splits: train 12,908 / validation 1,322 / test 4,113 cells
        = 17 / 2 / 6 perturbations, IRF1 pinned to test
arms:   base, signed ppi, mlp base, mlp signed ppi
        -- a bare variant selects it in BOTH families, so `base,signed` is
           2 GEARS arms + their 2 graph-free MLP counterparts
summary shows: test metrics table (4 arms x 6 metrics), 2 reference baselines,
               per-perturbation signal, graphs, figures
```

## 4. Data contract stage 3 expects

The backend joins the gold and the measurements on the **cell barcode**; the
plan's `gold[].path` and `supplementary[].path` are the only inputs.

* **gold / labels** — per cell: a barcode column (or the index) and a
  **condition** column. No genes needed. Controls are recognised as `NT`,
  `NTg5`, `ctrl`, `eGFP…` (`is_control`).
* **measurements / supplementary** — a `cells x genes` or `genes x cells`
  matrix; both orientations are tried and the better barcode overlap wins
  (`join_screen`). Values are counts (log1p-CPM applied) or already prepared.
* Splits are **by perturbation identity**, so a screen needs several
  perturbations; `--split-mode unseen_single` holds out single genes.

## 5. Knobs introduced for this workflow

| env | effect |
|---|---|
| `KOSMOS_PERTURBATION_ARMS` | which arms to train. A bare variant (`base`/`gold`, `signed`/`signed_ppi`/`ungated`) selects it in **both** families, GEARS and MLP; a qualified name (`gears_base`, `mlp_base`, …) selects exactly one. `base,signed` therefore trains 4 arms: `base`, `signed ppi`, `mlp base`, `mlp signed ppi`. |
| `KOSMOS_PERTURBATION_HOLDOUT` | perturbation(s) forced into **test** (`IRF1`), so the test metric is about that gene rather than a random draw. |
| `KOSMOS_PERTURBATION_MAX_GENES` | backend panel cap (default 2000); perturbed genes are always kept. |
| `PPI_LAMBDA`, `PPI_STAGE1_EPOCHS`, `PPI_STAGE2_EPOCHS`, `PPI_LAMBDA_LR_MULTIPLIER`, `PPI_STAGE2_LR_MULTIPLIER` | the two-stage signed-PPI schedule: 5 epochs of `L_true`, then 1 of `λ·(L_u − L_pg)` with a learnable λ (init 1, bounded to [0, 2]; λ at lr×10, the model at lr÷10). See the root `handoff.md` §3. |

## 6. Bugs — resolved (do not re-fix)

1. **preflight paired every model decision with the wrong listing** — so
   `_prefer_annotated_file` looked for a screen's label table in another
   series' listing and the label half was never kept. Fixed by resolving the
   owning listing per decision (`scripts/auto_task_run.py`).
2. **header normalisation recorded a `.gz` name but wrote a plain `.csv`** — the
   plan step then referenced a missing file and the whole run failed with
   `FileNotFoundError`. `write_with_header` now returns the path it wrote.
3. **`--target-confidence` received a float** — argv got a non-string and
   crashed the plan step. Mapped to `declared|exact_match|inferred`.
4. **a perturbation screen's label half was refused** (condition column, no
   genes) → `0 gold`. Now the label half is kept as gold and the measurement
   half as supplementary (`_perturbation_extract`, `dual_review`,
   `datafetcher/plan.py::_apply_review`).
5. **supplementary schema compared a transcriptome to the label table's QC
   columns** (`nCount_RNA`, `percent.mito`, …) → every real cohort rejected as
   "shares only 1 column". Now a table the review calls measured expression is
   evidence for a screen; the gene-panel re-reading was removed (slow and wrong).
6. **empty validation split** → `validation_pearson` NaN everywhere and the
   GEARS/MLP arms crashed with "no perturbation is in both the predictions and
   the observations". `perturbation_splits` now guarantees ≥1 validation
   perturbation when the train pool allows it; the arms also tolerate an empty
   validation set.
7. **a one-perturbation subset cannot be split.** The subset script now warns
   when a subset has <2 perturbations and supports `--targets all`.
8. **HVG dropped perturbation genes** → their cells were silently dropped.
   Perturbed genes are force-kept in the panel.
9. **supplementary cell cap** (`--supp-max-cells` 5000) discarded evidence.
   Default is now 0 = all cells.
10. **arm names in `summary.md`** were internal keys; they are now `base`,
    `signed ppi`, `finetune`, `mlp …` (`figures.py::_ARM_LABELS`).
11. **`KOSMOS_PERTURBATION_ARMS=base,signed` dropped the MLP arms** — a bare
    variant now selects it in *both* families, so the GEARS-vs-MLP comparison
    survives. `run_mlp_arms` also takes an `arms` subset, so the fine-tune MLP
    arm is not trained when only `base,signed` is asked for.

## 7. Bugs — OPEN (what the next agent should fix)

1. **The summary has no per-perturbation breakdown.** `run-9/summary.md` reports
   each arm's metrics **averaged over the 6 held-out perturbations** (SMAD4,
   POU2F2, STAT1, STAT2, IRF7, IRF1). The question is about **IRF1**, so its own
   mse_deg / pearson / direction cannot be read off the summary — a dedicated
   row (or table) per held-out perturbation is missing. The per-perturbation
   numbers exist inside `metrics.py::PerturbationMetrics.detail` but are not
   written out (`perturbation_results.json` → each arm's `test.detail` is empty).
   Fix: emit a per-held-out-perturbation table (mse_deg, pearson, direction,
   top_k) in `summary.md`, and/or a dedicated IRF1 row.
2. **[FIXED 2026-10-07: keys now follow summary names; finetune arms removed]**
   **`metrics.json`'s `finetune_*` keys are a lie when the finetune arm is off.**
   `discovery_bridge.run_perturbation_outcome` picks
   `headline_arm = FINETUNE_ARM if trained else gears_augmented_ungated`, then
   writes the arm's metrics under `finetune_*`. With `base,signed` that makes
   `finetune_* == ungated_*` and `delta = "finetune − base"` is really
   "signed − base". Fix: name the headline keys after the arm actually chosen
   (or omit `finetune_*` when it was not trained).
3. **Same-screen supplementary is not independent.** `supp_2`/`supp_3` are the
   screen's CITE / Arrayed arms — different cells of the *same* experiment.
   Stage 3 treats a table without a condition column as an unperturbed source,
   so `G_S` is built from perturbed cells of the same screen. Fine as a
   mechanical run, but it is not the "independent control cohort" `G_S` is meant
   for. Either fetch an independent macrophage cohort (stage 1 round 2 that
   actually keeps one) or restrict `supp` deliberately.
4. **Stage-1 retrieval is not deterministic.** Whether GSE153056 is proposed
   varies run to run (one run proposed GSE194122/GSE171555 instead). The working
   plan is the committed `artifacts/dataprep/irf1-run-mine/`; a fresh `dataprep`
   may not reproduce it.

## 8. Where things live

```
Perturb_middle/subset_process_irf1.py          stage 2 (subset + HVG + z-score + clip)
Perturb_middle/output_irf1-run-mine/           stage-2 output + stage-3 run/
Perturb_middle/handoff.md                      this file
artifacts/dataprep/irf1-run-mine/              stage 1 (gold + candidate supp)

kosmos/ppi/perturbation/contract.py            build_task / contraction, splits, holdout
kosmos/ppi/perturbation/run.py                 run_perturbation_task (join, panel, graphs, splits)
kosmos/ppi/perturbation/train.py               GEARS arms, two-stage signed PPI schedule
kosmos/ppi/perturbation/mlp.py                 MLP arms
kosmos/ppi/perturbation/figures.py             summary.md + figures (+ arm labels)
kosmos/ppi/discovery_bridge.py                 plan -> backend, arm selection, metrics.json
datafetcher/plan.py                            the plan contract + supplementary schema
scripts/auto_task_run.py                       the fetcher (preflight, review, plan)
```

## 9. Git state

Many uncommitted changes (this session plus the earlier arm work). Nothing has
been committed; `git status` is the source of truth. Tests touched:
`tests/unit/ppi/perturbation/`, `tests/unit/discovery/`, `tests/datafetcher/`,
`tests/unit/cli/test_auto_task_run_helpers.py`,
`tests/unit/cli/test_two_stage_workflow.py`.

```bash
.venv/bin/python -m pytest -o addopts="" \
  tests/unit/ppi tests/unit/discovery tests/datafetcher \
  tests/unit/cli/test_auto_task_run_helpers.py \
  tests/unit/cli/test_two_stage_workflow.py -q -p no:logging   # 399 passed
```

## 10. Next step

Fix the two OPEN bugs in §7 (per-perturbation metrics incl. IRF1; `finetune_*`
key mislabelling), then re-run stage 3 with `--max-epochs 50` on the `run-8`
plan and read IRF1's own row.
