# Handoff — Kosmos_ext perturbation backend (2026-10-14)

Written for the next context window. The full history (22 sections: GEARS port,
PPI design, open bugs B7–B11, PerturbNet, every stage-1 fix) is in
`handoff_archive_2026-10-14.md`; read it when a detail below is not enough.
Nothing is committed: `HEAD = aeff758`, everything is in the working tree.

---

## 0. Read first

**What the user is doing now.** Looking for a *demonstration example* for the
pipeline: a question where stage 1 finds the data by itself, the **gold is
small-ish but its observed Δ is reliable**, and a **large, similar supplementary
set** exists, so that "supp helps" has a chance to show. The last step was
reviewing the iPSC-neuron survey (§3); the user has **not chosen** a gold or supp
yet — they asked for a review first, then for the survey fixes now done.

**Working style (also in memory).**
- Reply in Chinese when the user writes Chinese (they asked for 中文 several times).
- The user decides design choices: present options with a recommendation and
  wait. "Paper setting first" for GEARS deviations.
- Pinning datasets by hand (e.g. `--candidate`) counts as cheating; stage 1 has to
  find the data itself.
- **Never download large data when testing; check `df -h /home` first.** Give
  commands the user can run themselves; they often run stage 1 / stage 3.
- Disk: /home ~660 GB free now (it was 96–99% full on 10-12/13 because of other
  users). `data/fetched` = 19 GB.

---

## 1. Pipeline shape (current)

```
stage 1  survey (NEW, no download)   question -> spec -> searches -> per-sample metadata
                                     -> judged gold / supp candidates -> survey.md/json
stage 2  per-task script (human)     download chosen samples, read in full, filter/pool,
                                     normalise, panel -> gold.parquet, supp_*.parquet,
                                     *_counts.parquet, plan.json(study), prepare_report.md
stage 3  experiment (unchanged)      kosmos experiment --data-plan <stage-2>/plan.json
```

The old stage 1 (`kosmos dataprep --task perturbation`, `scripts/auto_task_run.py`)
still exists and still works for per-cell tasks; for perturbation it failed on
almost every new question (see archive §14–19), which is why the survey was built.
User-approved plan: new stage 1 = Step 1 survey (done) → Step 2 per-sample fetch →
Step 3 load to a standard object + validation → plan/data.md. Steps 2–3 are
currently done **per task in stage-2 scripts**, using the shared loader below.
Old perturbation patches are to be deleted only after the new path is proven.

### Survey — `kosmos/ppi/perturbation/dataprep/`

```bash
.venv/bin/python -u -m kosmos.ppi.perturbation.dataprep.survey "<question>" --out DIR
```

- `spec.py`: one model call → cell type (+ aliases), organism, modality,
  topic (+ ≤3 `topic_aliases`), assays. Topic dropped if all words are generic
  (`specific_topic`); Greek letters → ASCII (`ascii_terms`, `γ`→`gamma`).
- `survey.py`:
  - queries (`_queries`): per topic term `topic assay`, `topic CRISPR screen`,
    `topic cell`; per cell name (full + shortest *specific* name) `cell assay`,
    `cell CRISPR screen single-cell`, `cell CRISPR screen`, `cell <modality>`;
    supp: `cell single-cell RNA-seq`, `cell scRNA-seq`, `cell assay`.
    Generic one-word cell names (`neurons`, `cells`, …) are never used
    (`GENERIC_CELL_WORDS`); acronyms (`BMDCs`) are not the short name.
  - search = `datafetcher.mechanical.mechanical_search` (GEO via ToolUniverse,
    OmicsDI, HF). Ranking: on-topic title first (any topic/cell anchor), then by
    how many queries hit. HF repos must contain the specific cell words
    (`_names_cells`). Caps: 8 gold, 15 supp candidates.
  - describe: `series_about`, `geo.sample_titles` (GSM titles), `geo.sample_files`
    (a GSM's own suppl files + sizes; retried, NCBI throttles bursts); a series
    with no per-sample files gets series-level files and collapses to one row.
    Cell estimate from the barcodes file (3.8 B/cell gz; >250k = unfiltered → `?`).
  - judge: one model call per series, per-sample schema (single-cell? screen?
    modality, cell type, topic match, labels?, role gold/supplementary/companion/
    reject).
  - gate (rules): gold = single-cell screen, right cell type/organism/modality,
    labels (model flag OR label-like file name OR a guide companion sample of the
    same lane — `label_evidence`); a failed gold of the right cells → supp.
- `load.py` (shared reader for stage 2): `load_sample(files)` → full cells×genes
  counts (10x mtx triplet / 10x h5 v2-v3 / h5ad), `call_cells(min_umis)`,
  `gene_means`, `similarity(reference, sample)` (Pearson/Spearman of mean
  log1p-CPM vs gold controls, median UMIs, ribosomal fraction).
- Tests: `tests/unit/ppi/perturbation/test_dataprep_survey.py` (13).

### Stage-2 pattern — `Perturb_middle/prepare_jurkat_tcr.py` (template)

Gold parsed and filtered; every survey supp candidate (+ unused gold candidates,
labels ignored) downloaded within `--supp-budget-gb` into
`data/fetched/samples/<series>/<sample>/`, cells called (`--min-umis`), scored by
`similarity`, pooled when Pearson ≥ `--min-similarity` (0.6); a series is judged on
its first sample, ≤ `--max-samples-per-series` pooled; pooled table has a
`supp_source` column (stage 3's condition guesser ignores it). Panel = top-5000
HVG among genes a pooled supp measures + perturbed genes; unmeasured → gold
control mean. `prepare_report.md` puts survey estimates next to actual sizes and
lists every candidate's similarity + decision.

---

## 2. Example questions tried (all surveyed; data facts measured)

Reliability = median split-half Pearson of a perturbation's observed Δ over all
panel genes (cells split in two halves). It bounds what any model can score.

| question | gold | gold reliability | supp | status |
|---|---|---|---|---|
| K562 UPR (Adamson) | GSM2406681, 50,440 cells, 82 genes, ~507 cells/gene | **0.79** | survey finds only same-series Adamson samples + 1 HF set (misses Norman GSE133344, Dixit GSE90063, Jost GSE132080) | user: gold too big; full 3-stage runs exist (`Perturb_middle/output_adamson_upr/run-finetune`, `run-perturbnet-v2`): supp (Dixit 5.4k) did not help |
| Jurkat TCR (Datlinger GSE92872) | 3,204 stimulated cells, 29 genes, 85 cells/gene, 705 CTRL | **0.10** (too noisy) | pooled 72,837 cells, 6 samples, Pearson 0.81–0.89 (GSE249595 probe-based, 0.24, dropped) | stage 2 + stage 3 done (`Perturb_middle/output_jurkat_tcr/run-v1`): all arms ≈ baselines; Pearson ≈ 0.05–0.1 = noise floor |
| THP-1 IFN-γ / PD-L1 (Papalexi GSE153056, cached) | 20,729 cells, 25 genes, 662 cells/gene, 2,386 NT | 0.34 (IFNGR1/2, JAK2, STAT1, IRF1 ≥ 0.93) | THP-1 supps few | **survey cannot find GSE153056**: title "regulation of inhibitory immune checkpoints with multi-modal single-cell screens" has no THP-1/IFN/PD-L1 words |
| **iPSC-derived neurons, CRISPRi** (current) | GSE124703 Tian 2019 neuron lanes 1+2 (GSM3543622 240 MB, GSM3543623 225 MB; unfiltered barcodes; guides in companion samples GSM3543624/25 `sgRNA_mapping.txt.gz`) | **unknown — measure next** | GSE277602 4 samples (~13–18k cells each, ~60k), GSE244370 NGN2-iN (small files), GSE277604 ADNP lines (14–31k each; found in an earlier run, not the last) | survey OK after the 10-14 fixes |
| CD4 T cells (KO) | none: GSE314342 is CD4 **CRISPRi** (2.6 GB/lane) | – | – | modality mismatch, correctly demoted |
| mouse BMDC LPS | GSE327057 (series-level, very large) | – | none | not pursued |

THP-1 survey rerun (`artifacts/dataprep/thp1-pdl1-v2`): finds Papalexi as gold, supp
~156k THP-1 cells (GSE301541 WT/FOLR2-KO, GSE268256, GSE176294 macrophages), 6.6x; gold
is 5′ chemistry, supp mostly unknown/3′; `gold_check.md` there.

Rejected for the neuron question: GSE207099 (perturbs AD/PSP enhancers, not genes;
partly microglia), GSE124703 iPSC lanes (undifferentiated), GSE69790 (C1, tiny).

---

## 3. Where the iPSC-neuron question stands (updated 2026-10-14, later session)

Stage 1 run end to end: `artifacts/dataprep/ipsc-neuron-v3/survey.md` (survey, see §7)
+ `gold_check.md` (golds downloaded, cells/guides called, split-half reliability;
`Perturb_middle/check_gold_ipsc_neuron.py`). Measured:

| gold | cells (called) | labeled perturbed | perts | median cells/pert | NT ctrl | rel. all genes | rel. top-20 DE |
|---|---|---|---|---|---|---|---|
| GSE124703 Tian 2019, day-7 neurons, lanes 1+2 | 22,313 | 8,508 (+10k ambiguous) | 29 | 110 | 1,824 | 0.16 | 0.65 |
| GSE289235 CaMPARI2 CRISPRi, day-21 neurons, 1 h5ad | 18,690 | 10,123 (+8k no guide) | 41 | 171 | 522 | 0.14 | 0.76 |
| (THP-1) GSE153056 Papalexi | 20,729 | 18,343 | 25 | 662 | 2,386 | 0.20 | 0.79 |
| GSE152988 Tian 2021 | 62k | – | – | – | – | – | expression-only on GEO: no labels |

Supp pool (survey): 93 samples, ~969k cells (GEO ~548k + scBaseCount 421k), ≫ 5x
either neuron gold; stage-2 similarity filtering will cut it.
Details: Tian 2019 guides are protospacer sequences → genes via the hCRISPRi-v2 library
table (`data/fetched/reference/`); the 2 abundant unmapped protospacers act as NT;
guide call = top guide ≥ 50 reads and ≥ 80% of the cell's guide reads. MAT2A has 3.4k
cells with a large shift (STMN2/GAP43 down). GSE289235: `obs.guide` (`GENE_i1/i2`,
`non-targeting_*`), unfiltered (1.5M barcodes) → ≥ 1000 UMIs.
**User decides next:** which gold (recommendation: GSE289235: labels in file, 41 genes,
but only 522 NT; or Tian 2019: 1.8k NT, 29 perts), then write
`Perturb_middle/prepare_ipsc_neuron.py` from the Jurkat template (scBaseCount h5ad
already readable by `load.load_sample`).

**Stage 2 done (gold = GSE289235, my recommendation; user asked to proceed):**
`Perturb_middle/prepare_ipsc_neuron.py` → `Perturb_middle/output_ipsc_neuron/`
(`gold.parquet` 10,693 cells = 522 ctrl + 44 genes, 3–1,509 cells each; supp pooled
74,355 cells = 7.0x: GSE124703 lanes 1+2 (Pearson 0.78/0.85), GSE152988 lanes 2+3
(0.92), GSE319096 iGLUT MG8 (0.76); stops at `--supp-ratio 5`; unused gold candidates
are tried first; panel 5,006 genes; `H2AFZ` guides → `H2AZ1`). Not yet run in stage 3.

## 4. Open issues / pending user decisions

- **Evaluation**: `summary.md` headlines Pearson over all panel genes, which is
  noise-dominated for small screens; GEARS reports top-20 DE metrics. Proposed:
  show `pearson_deg` / `mse_deg` as the headline (computed already, not shown).
  Pending user approval.
- **PerturbNet speed**: scVI 500 epochs on ~75k cells took ~50 min and printed
  nothing (tqdm uses `\r`; the runner only forwards `[perturbnet]` lines).
  Proposed: a heartbeat line per minute in `scripts/perturbnet_runner.py`, and
  epochs scaled to cell count (deviates from PerturbNet's 500 → user decides).
- **Survey retrieval** is keyword search over titles; still misses studies whose
  title lacks the question's words (Papalexi) and varies a little between runs.
  Ideas not done: query the GEO summary text, union of two runs, a rule-built
  query merged into the ranked pool (the old `--mechanical-search` round; the
  user asked why it was off — it downloaded unrelated hits blindly).
- **Survey should record assay chemistry** (3' GEX vs probe-based Flex vs
  Drop-seq): GSE249595 (probe-based) correlated 0.24 with a Drop-seq gold.
- Supp ranking when cells are unknown should fall back to download size.
- Old-stage-1 speed fixes proposed, not done: preview tables as small parquet,
  convert only when the reader version changes (`fetch._rebuild_derived` re-converts
  on every reuse), profile cache.
- PPI-specific open bugs (B7 stage-2 SGD too small, B8 BatchNorm stats, B9
  selection picks early epochs, B10 supp distribution shift, B11 λ bang-bang) —
  archive §5.4; D1/D2 decisions still pending from the user.

---

## 5. Changes made 10-12 → 10-14 (all uncommitted, tests green: 509 passed)

```bash
.venv/bin/python -m pytest -o addopts="" -q -p no:logging \
  tests/unit/ppi tests/unit/discovery tests/datafetcher \
  tests/unit/cli/test_auto_task_run_helpers.py tests/unit/cli/test_two_stage_workflow.py
```

- PerturbNet arms (`kosmos/ppi/perturbation/perturbnet_arm.py`,
  `scripts/perturbnet_runner.py`, conda env `perturbnet`; archive §12).
- `--supp-size large` switch, 200k supp cap, bundle cleanup (archive §13).
- Old stage 1 fixes: placeholder identifier example, search-first retrieval,
  screen gate (single-cell + perturbation words), nested archive unpack, per-sample
  table names, matrix converted when companions arrive later, per-cell gold filter,
  GEO sample titles for the primary-gold choice, 10x h5 + CROP-seq guide-feature
  reader, ToolUniverse default interpreter (archive §14–19).
- New survey + loader + Jurkat stage 2 (this file §1).
- 10-14: generic cell names excluded, HF repos must name the cells, supp cap 15,
  lane-matched label companions, gold queries `cell CRISPR screen` / `cell CRISPRi`.

## 7. Stage-1 survey changes (2026-10-14, later session; 519 tests pass)

- **GEO full-text search** `geo.search_series` (E-utilities, AND-ed words, `screen*`,
  all series types) merged into the survey's search; ranking also reads the summary.
  Cause found: ToolUniverse's GEO tool searches RNA-seq-typed series only, Papalexi is
  "Other"; Entrez reads `CRISPR screen` as a phrase. Survey now finds GSE153056.
- **scBaseCount source** `dataprep/scbasecount.py` (Arc, Cell 2026, doi
  10.1016/j.cell.2026.08.025): 2026-01-12 release on `gs://arc-institute-virtual-cell-atlas`,
  read over plain HTTPS (no auth needed as of 10-14), sample table cached at
  `data/fetched/scbasecount/` (human 35k SRX, 302M cells; cell_line/tissue/perturbation/
  10x chemistry/exact obs_count). Rows prefiltered by cell-name words, one model call
  keeps the right cells; the surveyed GEO series' own runs (elink gds→sra,
  `geo.series_sra_ids` + `sra_samples` GSM map) are dropped, and GEO rows get
  scBaseCount's exact cells + chemistry. Supp ordered: gold chemistry, unlabeled, cells.
  `--no-scbasecount` turns it off. One SRX can be a guide library (GDO) counted as cells.
- **Fit table** (`survey.fit`): per gold series, gold cells vs supp cells, ratio, ≥ 2x.
- Fixes: `_words` keeps hyphenated tokens (`PD-L1` matched nothing before); a matrix
  file named `..._CRISPRi_..._matrix.h5` is no longer label evidence; a gold with only
  the model's word on labels is marked unverified and ranked last; unverified `.h5ad`
  golds get their `obs` peeked over HTTP (`peek_h5ad_labels`, ~10 s).
- New `dataprep/reliability.py::split_half` (all genes + top-20 DE, halves share no cells).
- Downloads kept: GSE124703 lanes+guides (~470 MB), GSE289235 h5ad (1 GB).

## 8. Stage 3: 8 arms + gated loss (2026-10-14, user decisions; 523 tests pass)

- Arms (`KOSMOS_PERTURBATION_ARMS=all`): {GEARS, MLP} × {base, signed PPI
  (`*_augmented_ungated`), gated (`gears_gated`/`mlp_gated`)} + `perturbnet_base`,
  `perturbnet_supp` (PerturbNet gets no PPI/gated variant: user, structure too special).
  `gears_finetune_ppi` is opt-in, not in the default.
- Gated = `gatedlossdesign.md` batch level: per step g_G=∇L_true, g_S=∇L_u (supp cells vs
  frozen base teacher Δ, same pairing as stage 2), g = g_G + max(0,cos)·min(1,κ‖g_G‖/‖g_S‖)·g_S
  from detached grads, value clip, Adam; no stage cycle, no L_pg; same architecture
  (+G_S) and teacher as the signed arm. Per-epoch log: gold/synthetic loss, grad norms,
  cosine (mean, std), weight (mean, std), active fraction; summary.md "Gradient gate".
  Sample-level gating (§6 of the doc) not implemented yet.
- GEARS lr decay off by default (`lr_decay` 1.0; `KOSMOS_PERTURBATION_LR_DECAY=0.5`
  restores GEARS). New env: `KOSMOS_PERTURBATION_LR`, `PPI_GATED_KAPPA`.
- Batch 512 fits: peak 16.8 GB on an A6000 (gated GEARS, iPSC data).
- data report: gold "used" = examples + controls; perturbations outside the GO
  vocabulary are counted (`outside_vocabulary`) instead of silently skipped.

## 6. Where things live

```
kosmos/ppi/perturbation/dataprep/{spec,survey,load}.py   new stage 1 survey + shared reader
datafetcher/sources/geo.py      series_about, sample_titles, sample_files
Perturb_middle/prepare_jurkat_tcr.py    stage-2 template (pooled, similarity-filtered supp)
Perturb_middle/prepare_adamson_upr.py   stage 2, Adamson
Perturb_middle/output_{adamson_upr,jurkat_tcr}/   stage-2 outputs + stage-3 runs
artifacts/dataprep/{jurkat-tcr,thp1-ifng,k562-upr,ipsc-neuron}/survey.{md,json}
data/fetched/samples/<series>/<sample>/   per-sample downloads (stage 2)
kosmos/ppi/perturbation/{train,model,run,metrics,figures,perturbnet_arm}.py   stage 3
scripts/perturbnet_runner.py    PerturbNet (conda env `perturbnet`, Python 3.7)
```
