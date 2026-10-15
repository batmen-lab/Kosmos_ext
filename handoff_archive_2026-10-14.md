# Project handoff — single-cell perturbation backend (Kosmos_ext)

_Last updated: 2026-10-10. Written so a fresh context window can continue without
the previous conversation. Older versions: `git show HEAD:handoff.md`. IRF1-specific
notes: `Perturb_middle/handoff.md`._

**Read first:**
- §0 says what the user is doing right now and which decisions are still theirs.
- §5 lists every open problem with its evidence.
- Nothing is committed: `HEAD = aeff758`, everything below is in the working tree.

---

## 0. Where we are (state at handoff)

**What the user is doing now.** Writing a **report** on this project, not
debugging. Recent questions were explanatory: what the fetch module is, how big a
subset stage 2 extracted, why only one gold sample, what `specific pearson` is.
The report-ready material is collected in §9. Write report text in their style:
Chinese prose, English technical terms, tables, and no bug lists inside report
text.

**Language and working style.**
- The user writes in Chinese and English; reply in Chinese unless they write English.
- They decide design questions themselves. Present options with a
  recommendation and wait; do not implement a design choice they have not approved.
- Standing rule, also in memory: **when the implementation differs from the
  original paper (GEARS), default to the paper's setting first**, then flag the
  deviation.
- They prefer concrete commands they can run themselves, plus a short
  explanation.

**Decisions pending from the user.** Do not implement until they choose.

| # | question | options | my recommendation |
|---|---|---|---|
| D1 | B7: stage-2 SGD barely moves the weights | (a) raise the stage-2 SGD lr; (b) normalised step `d/‖d‖·η`; (c) back to Adam, projecting Adam's *preconditioned* update | (c) |
| D2 | B8: BatchNorm running stats polluted by the supplementary forward | freeze BN stats (eval-mode BN) for the supplementary forward in stage 2, then verify | do it (it is a bug) |
| D3 | speed-ups (§6) | 1. pre-stack the training tensors on GPU; 2. cache the teacher Δ for gold rows; 3. bf16 autocast | 1 + 2 change no results; 3 needs approval |

**Decisions already made by the user** (do not re-litigate):
- Keep only base and signed-PPI arms; a **finetune** branch was added later at their request.
- Stage 1 = `L_true`; stage 2 = `λ·(L_u − L_pg)`, λ learnable at lr×10, the
  model at lr×0.1; 5 stage-1 epochs then 1 stage-2 epoch.
- Stage 2 is constrained so that `L_u` and `L_true` do not increase
  (projection, D-2026-10-08); its model step uses SGD.
- The MLP gets **no** GEARS lr decay (constant lr).
- Epochs are selected by validation **`mse_deg`** (GEARS' rule).
- Gold inputs are sample-level real control cells. Supplementary cells are
  taken loosely (every cell; their expression is still useful in the correction
  term).
- Train on all genes (GEARS). `condition_on_control` is a switch for comparison.
- **Stage-2 data prep is written per task**; there is no generic stage-2 tool.
- Adamson: validation 20%, batch size 128.

---

## 1. What this fork is for

Kosmos' autoresearch loop asks a question, generates hypotheses, writes code and
runs it. This fork adds the **data layer for single-cell tasks**: given a
question, find real data, decide which table plays which role, train, and
report metrics.

| Backend | Question shape | Data contract |
|---|---|---|
| `per_cell` (column task) | "predict label Y from expression X" | one labeled table + optional unlabeled evidence |
| `perturbation` (this handoff) | "what does perturbing gene Z do; can we predict it" | a **screen**: per-cell condition column + control labels + measured transcriptome |

The research idea under test: **PPI-style semi-supervision** (prediction-powered
inference). A frozen teacher pseudo-labels cells from *independent* experiments
(supplementary). The student trains on gold, then on a rectified correction
`λ·(L_u − L_pg)`, and is compared with a gold-only base, for GEARS and a
graph-free MLP.

---

## 2. Three-stage workflow

The only handoff between stages is `plan.json`; a person reviews between stages.

```
stage 1  dataprep     question -> raw data            automatic     plan.json + data.md
stage 2  prepare      raw data -> training tables     per task      gold.parquet, supp_*.parquet,
                                                       (human)       plan.json with a `study` block
stage 3  experiment   tables -> trained arms + report  automatic     run/summary.md, metrics.json, ...
```

### Stage 1, two modes

```bash
# (a) search: the model proposes datasets; datafetcher downloads/unpacks/profiles
.venv/bin/python -m kosmos.cli.main dataprep "<question>" --task perturbation \
  --out artifacts/dataprep/<name>
# (b) pinned: a registered GEARS release (deterministic)
.venv/bin/python -m kosmos.cli.main dataprep "<question>" \
  --dataset replogle_k562_essential --supplementary-dataset adamson \
  --out artifacts/dataprep/replogle-k562
```

Search-mode pipeline. Code: `scripts/auto_task_run.py`, `datafetcher/`,
`kosmos/discovery/`.

1. **Retrieval intent.** From the question wording: modality (knockout /
   knockdown / overexpression → CRISPRko / i / a), the requirement on
   supplementary data (same tissue, has controls), and condition-column hints
   (gene, guide, sgRNA, perturbation, …).
2. **Candidates.** The model proposes datasets (`--fetch-limit 2`, up to 3
   hops). Without ToolUniverse it names GEO ids itself.
3. **Preflight.** List each series' files; the model chooses; rules correct the choice:
   - `_prefer_expression_file`: size cap, prefer non-archives, skip if a
     matrix is already chosen;
   - `_keep_matrix_companions`: barcodes / features / cellnames / genenames /
     label maps;
   - `_FAILED_REFERENCES` remembers failed downloads.
4. **Download, unpack, cache** (`datafetcher/fetch.py`, `sources/net.py`, `store.py`):
   - empty or short downloads raise; a 0-byte cache file is never reused;
     `manifest.json` drives reuse;
   - archive caps: 48 members, 24 files per fetch.
5. **Read** (`singlecell.py`):
   - 10x mtx triplets, including `.mtx.txt(.gz)`, read via scipy;
   - GEO prefix companions;
   - Dixit-style cellnames/genenames and the inverted `cbc_gbc_dict` label map;
   - h5ad;
   - metadata beside matrices (`fetch.py::_metadata_beside`).
6. **Profile** (`profile.py`). Every file is sniffed and sampled; non-tables are
   reported, not reviewed.
7. **Condition column** (`kosmos/ppi/task_inference.py`), then a **dual review**
   (rules + model; `kosmos/discovery/review.py`):
   - each table gets: is it a screen, the condition column, control labels,
     modality, whether it has expression;
   - disagreements go to `adjudication.json`;
   - a labels-only table is kept as the screen's **label half**.
8. **Output** (`datafetcher/plan.py`): `plan.json` (gold / supplementary /
   unusable), `data_report.md/.json`, `data.md` (for the human),
   `search_trace.json`, `reviews.json`.

Pinned mode: `kosmos/ppi/perturbation/registry.py` (`stage_registered`,
`write_registered_plan`) downloads GEARS' `perturb_processed.h5ad` into
`data/perturbation/gears/<name>`.

### Stage 2, per task

| script | task | output |
|---|---|---|
| `scripts/prepare_perturbation.py` | registered GEARS h5ad (Replogle + Adamson) | `artifacts/prepared/replogle-k562` |
| `Perturb_middle/prepare_adamson_upr.py` | Adamson UPR (full search pipeline) | `Perturb_middle/output_adamson_upr` |
| `Perturb_middle/subset_process_irf1.py` | IRF1 macrophage | `Perturb_middle/output_irf1-run-mine` |

The stage-2 contract:
- `gold.parquet`: `condition` (`GENE+ctrl` / `ctrl`), optional `cell_type`,
  then one column per gene.
- `supp_<name>.parquet`: the same gene columns; genes not measured there are
  filled with the gold control mean.
- `plan.json` with a `study` block: `split_mode`, `test_fraction`,
  `validation_fraction`, `max_panel_genes`, `condition_column`, and now
  `batch_size`.

Stage 3 reads the study block, and it overrides the CLI's `--split-mode` default.

### Stage 3

```bash
.venv/bin/python -m kosmos.cli.main experiment "<question>" \
  --data-plan <stage-2 dir>/plan.json --task perturbation \
  --max-epochs 20 --patience 0 --out <stage-2 dir>/<run name> --yes
```

- **Always pass `--out`.** The default is the plan's directory, so a later run
  lands in `run-2`, `run-3`, … and is easy to confuse. Output goes to `<out>/run/`.
- `--patience 0` means no early stopping, as in GEARS.
- Prefix `CUDA_VISIBLE_DEVICES=N` to pick a GPU. The machine has 8 GPUs (A6000-class).
- Env knobs are in §7. Useful ones: `KOSMOS_PERTURBATION_ARMS`, `PPI_LAMBDA`.

**Current full Adamson command** (stage 1 and 2 outputs are reused):

```bash
cd /home/ydong233/Kosmos_ext
CUDA_VISIBLE_DEVICES=0 .venv/bin/python -m kosmos.cli.main experiment \
  "In K562 cells, can the transcriptional response to CRISPRi knockdown of unfolded protein response (UPR) genes not seen in training be predicted from the other knockdowns, using single-cell profiles of K562 cells from independent Perturb-seq experiments as supplementary evidence?" \
  --data-plan Perturb_middle/output_adamson_upr/plan.json \
  --task perturbation --max-epochs 20 --patience 0 \
  --out Perturb_middle/output_adamson_upr/<run name> --yes
```

---

## 3. Worked examples (data)

### 3.1 Adamson UPR (the full search → human → experiment example)

**Question:** *In K562 cells, can the transcriptional response to CRISPRi
knockdown of unfolded protein response (UPR) genes not seen in training be
predicted from the other knockdowns, using single-cell profiles of K562 cells
from independent Perturb-seq experiments as supplementary evidence?*

**Stage 1** (`artifacts/dataprep/adamson-auto`) found three series:
- GSE90546 (Adamson 2016): the gold;
- GSE90063 (Dixit 2016, K562 wild-type): the supplementary;
- GSE153056 (ECCITE-seq): rejected, 0 shared genes.

The plan has gold 1, supplementary 6 and unusable 12 entries:
- gold = 10X001 `cell_identities` (label half);
- the other samples' tables and Dixit = supplementary;
- barcodes, genes, raw mtx and the tar = unusable.

The *sources* were right, but the *role split* was not trainable as is. The
human stage 2 fixed it, which is the reason for the human step.

**Stage 2** (`Perturb_middle/prepare_adamson_upr.py`):

| | gold (Adamson, GSE90546) | supp (Dixit, GSE90063) |
|---|---|---|
| files fetched | 3 samples (10X001, 10X005, 10X010) | 1 table (`k562_umi_wt`) |
| used | 1 sample: GSM2406681 (10X010), "UPR Perturb-seq experiment" | all |
| cells: raw → kept | 65,337 → **50,440** (77%) | 5,409 → **5,409** |
| cell filter | guide call → good coverage → exactly one guide → named guide (not `*`) | none |
| content | 5,241 controls (`63(mod)_pBA580`, `Gal4-4(mod)_pBA582` → `ctrl`) + 82 perturbed genes (≥185 cells each, median 507); a gene's guides are one perturbation (`GENE+ctrl`) | unperturbed K562, no condition column |
| genes: raw → kept | ~32,600 → **5,014** (top-5,000 HVG + 14 perturbed genes outside them) | aligned by Ensembl id, **5,014 / 5,014** measured |
| normalisation | log1p-CPM (1e4) | the same |
| role in training | labeled: `L_true`; split by perturbation | unlabeled: `G_S` graph and `L_u` |

Cell filter counts: 65,337 → 65,257 (guide call) → 59,219 (good coverage) →
50,467 (one guide) → 50,440 (named guide).

**Why only one gold sample.** The three GSE90546 samples are different
experiments: 10X001 is a pilot TF screen and 10X005 an epistasis screen (as
written in the script's docstring; not re-checked against the paper or GEO, so
verify before quoting). Only 10X010 targets UPR genes. Mixing the samples would
change the question and bring batch and control mismatches. 10X010 alone is
large enough.

**Split** (study block): `unseen_single`, test 25% of the perturbed genes,
validation **20%** of the rest (was 10% = 6 perturbations), `batch_size: 128`.
Cells: train 24,341 / validation 7,847 / test 10,307. Perturbations: about
46 / 13 / 19 (77 encodable genes).

**Batch effect** (gold controls vs Dixit), measured in-session: zeros 40% vs
79%; per-cell total expression 2,875 vs 1,819; correlation of gene means 0.81.
This matters for PPI, which assumes the labeled and unlabeled inputs share a
distribution, and for B8.

### 3.2 Replogle K562 essential (registered GEARS data)

- Gold: GEARS `replogle_k562_essential`: 162,751 cells, 1,087 encodable
  perturbations, 5,000 genes.
- Supplementary: GEARS `adamson`, 2,026 shared genes; the rest are filled with
  the gold control mean.
- Stage-1 output: `artifacts/dataprep/replogle-k562`. Stage-2 output:
  `artifacts/prepared/replogle-k562`.
- One GEARS epoch takes about 2.5 min.

---

## 4. Model and training design (`kosmos/ppi/perturbation/`)

### 4.1 Arms

| arm key (summary name) | architecture | init | objective |
|---|---|---|---|
| `gears_base` (`base`) | GEARS | random | `L_true`; the GEARS **teacher** |
| `gears_augmented_ungated` (`signed ppi`) | GEARS + `G_S` branch | random | two-stage PPI |
| `gears_finetune_ppi` (`finetune ppi`) | GEARS (no `G_S`) | **copy of the trained base** (weights + BN stats) | two-stage PPI, `finetune_epochs` 12, lr × 0.1 |
| `mlp_base` (`mlp base`) | graph-free MLP | random | `L_true`; the MLP teacher |
| `mlp_augmented_ungated` (`mlp signed ppi`) | MLP | random | two-stage PPI |

- The base arm of a family is always trained when a dependent arm is
  requested; it is the frozen teacher.
- One loop, `train.py::_train_arm`, trains every arm; `_Net` hides the architecture.
- **Finetune design** (user request 2026-10-08):
  - it isolates the PPI loss: same architecture as base, no `G_S`;
  - it starts with **stage 1**, because at student == teacher both `L_u` and
    `L_pg` sit at zero residual and the quartic loss has zero gradient there,
    so a first stage-2 step would do nothing.

### 4.2 GEARS port (`model.py`, `graphs.py`), following `/home/ydong233/GEARS`

- **Perturbation vocabulary** = the GO graph's genes (9,853; `go_vocabulary`),
  separate from the measured panel. A perturbed gene outside it is dropped.
- **SGConv(K=1)** = PyG `gcn_norm` (`add_remaining_self_loops`,
  source→target, in-degree) as a sparse matrix (`sparse_adjacency`).
- **Graphs are directed**: each target keeps its top-(k+1) sources (`_gears_top_k_edges`).
  - `G_C`: co-expression of training + control cells;
  - `G_GO`: GO similarity over the vocabulary;
  - `G_S`: co-expression of the supplementary cells; signed arm only,
    `pos = SG(G_C) + η·SG(G_S)`.
  - On Adamson `G_S` is nearly `G_C`: 99.8% of `G_S` edges are in `G_C`,
    Jaccard 0.90. So the `G_S` branch adds little structure there.
- **Layers as in GEARS**: gene/pos embeddings, `emb_trans_v2`, `pert_fuse`,
  `bn_pert_base`, `recovery_w`, `indv_w1/b1`, `cross_gene_state`,
  `indv_w2/b2`. Prediction = `x + Δ̂`. 2.6M parameters on Adamson. Contains
  **BatchNorm** (relevant to B8).
- **Extension `condition_on_control`** (default on; env `=0` gives original
  GEARS): adds `x_g·w_g` to each gene's state, so Δ̂ depends on the input cell.

### 4.3 MLP (`mlp.py`)

`Δ = W_o(ReLU(W_c x) + ReLU(W_p p) + context)`, where `p` is a one-hot / indicator
over the GO vocabulary. No BatchNorm. Structurally it cannot generalise to an
unseen perturbation, because a held-out gene's indicator column was never trained.

### 4.4 Inputs and batches (`train.py`)

- **Gold rows are single cells.** Each perturbed training cell `y` is paired
  with one real control cell `x_c` from its control group
  (`contract.control_cells`).
  - The pairing is drawn once with a fixed seed (GEARS' fixed pairing;
    `resample_controls=True` redraws per epoch).
  - Target `Δ = y − x_c`. `sample_controls=False` falls back to the group mean.
- Every gold training cell is used once per epoch. Batch size comes from the
  study block or env, default 32 (GEARS); Adamson uses 128.
- **Supplementary rows** (stage 2 only):
  - each gold batch of size B draws `supplementary_ratio × B` supplementary
    cells, without replacement until the pool is spent (`SupplementarySampler`);
  - supplementary row *i* is queried with gold row *i*'s perturbation, so
    `L_u` and `L_pg` cover the same perturbations;
  - teacher labels are computed per batch.

### 4.5 Losses (`losses.py::gears_loss`, as in the GEARS paper)

Per perturbation group, over the genes with non-zero mean expression in that
perturbation's training cells (`retained_genes`, GEARS' `non_zeros_gene_idx`;
on Adamson 5,010–5,013 of 5,014 genes):

    quartic error (Δ̂ − target)^4  +  0.1 × (sign(x + Δ̂ − μ_ctrl) − sign(x + target − μ_ctrl))^2

The result is averaged over groups.

| term | input | target |
|---|---|---|
| `L_true` | gold cells (paired `x_c`) | measured `y − x_c` |
| `L_pg` | the same gold cells | teacher Δ̂ |
| `L_u` | supplementary cells | teacher Δ̂ |

### 4.6 Optimisation and schedule

- **GEARS settings:** 20 epochs, Adam lr 1e-3, weight decay 5e-4,
  `clip_grad_value(1.0)`, hidden 64, no early stopping.
- **lr decay per family:**
  - GEARS arms: StepLR γ = 0.5 per epoch (`lr_decay`), applied to every
    optimiser of the arm, including stage 2 and λ;
  - MLP arms: `mlp_lr_decay = 1.0` (constant). The MLP is our baseline, not GEARS.
- **PPI arms** cycle 5 epochs of stage 1 and 1 of stage 2 (stage 2 = epochs 6,
  12, 18). Ported from scDesign3 `src/train_ppi.py`.

| stage | minimises | optimiser |
|---|---|---|
| 1 | `L_true` | Adam(model, lr) |
| 2 | `λ·(L_u − L_pg)` | model weights: **SGD, lr × 0.1**, no weight decay, on the **projected** gradient; λ: **Adam, lr × 10**, no weight decay |

- **λ** (`LearnableLambda`): `λ = λ_min + (λ_max − λ_min)·sigmoid(raw)`,
  bounded to [0, 2]. Its gradient is `L_u − L_pg`.
  - The init is `lambda_init`. The config default is 1, but **the bridge passes
    `PPI_LAMBDA` with default 0.5**, so runs start at 0.5.
  - λ·correction is linear in λ, so λ drifts to a bound ("bang-bang").
- **Projection** (`project_to_descent`, 2026-10-08). Alone,
  `λ·(L_u − L_pg)` has no lower bound: an MLP run with constant lr reached
  `L_pg ≈ 8.8e12` and `L_u ≈ 1.5e12`, with λ pinned at 2. So each stage-2 batch:
  - computes `g = ∇λ(L_u − L_pg)`, `∇L_u` and `∇L_true` (`L_true` from the same
    gold forward as `L_pg`), each clipped to ±1;
  - steps along the closest `d` to `g` with `⟨d, ∇L_u⟩ ≥ 0` and `⟨d, ∇L_true⟩ ≥ 0`
    (exact for 1–2 constraints). To first order neither `L_u` nor `L_true` rises,
    and the `−L_pg` correction acts only where it does not conflict.
  - SGD is used so that the step *is* this direction.
  - History records `projected` (fraction of batches that needed projecting) and
    `correction`.
- **Selection:** best epoch over both stages by validation **`mse_deg`**
  (`selection_metric`, env `KOSMOS_PERTURBATION_SELECTION`).
  `pearson_specific` was the default until 2026-10-08; it ignores scale and
  picked exploded stage-2 epochs.

### 4.7 Evaluation and report

- Per perturbation: the mean Δ̂ (controls drawn with a fixed seed) vs the
  observed `y − control mean`.
- **Metrics** (`metrics.py::perturbation_metrics`):
  - `mse`, `mse_deg`;
  - `pearson`: Pearson of Δ̂_p vs Δ_p over all genes, averaged over perturbations;
  - `pearson_specific`: the same, but on `Δ̂_p − mean_p Δ̂` vs `Δ_p − mean_p Δ`,
    with the mean taken over the evaluated perturbations. It credits only
    what distinguishes perturbations; a perturbation-constant prediction gives
    NaN. It depends on the evaluated set, so validation and test values are not
    comparable;
  - `pearson_deg` (top-K DE genes; computed but not shown in summary.md),
    `spearman`, `direction_accuracy`, `top_k_overlap`.
- **`run/summary.md`:**
  - test metrics per arm;
  - reference baselines: `predict 0`, and `mean response (leave-one-out)`,
    which predicts each held-out perturbation by the mean response of the
    others;
  - per-perturbation signal: the spread of the arm's per-perturbation profiles
    over the observed spread; below 10% is flagged as a per-gene constant;
  - graphs, warnings.
- **`run/metrics.json`** keys arms by summary name: `base_*`, `signed_ppi_*`,
  `finetune_ppi_*`, `mlp_base_*`, `mlp_signed_ppi_*`. `headline_arm` =
  signed ppi.
- **`perturbation_results.json`:** config, split sizes, and per arm the
  `history` (per epoch: losses, `correction`, `lambda`, `projected`, validation
  metrics), `best` and `signal`.
- Also written: `*_test_predictions.npz` and `<arm>.pt`.

---

## 5. Results and open problems

### 5.1 Latest full run: Adamson, all 5 arms

`Perturb_middle/output_adamson_upr/run-finetune/run` (2026-10-08). Settings:
batch 128, validation 20%, mse_deg selection, projected SGD stage 2, MLP
without decay, λ init 0.5.

| arm | test mse_deg | pearson | specific pearson | direction | signal | best epoch | λ end |
|---|---|---|---|---|---|---|---|
| base | **0.182** | 0.466 | 0.421 | 0.871 | 26% | **1** | — |
| signed ppi | 0.203 | 0.451 | 0.530 | 0.832 | 37% | 2 | 0.479 |
| finetune ppi | 0.207 | 0.474 | 0.461 | 0.866 | 31% | **1** (stage 1) | 0.498 |
| mlp base | 0.230 | 0.405 | 0.066 | 0.876 | 5.5% | 7 | — |
| mlp signed ppi | 0.228 | 0.404 | 0.069 | 0.876 | 5.6% | 7 | 1.896 |
| predict 0 | 0.399 | — | — | 0.000 | | | |
| leave-one-out mean | 0.236 | 0.488 | −1.000 | 0.863 | | | |

How to read it:
- **No arm's best epoch is a stage-2 epoch.** The finetune arm's chosen
  weights are the base plus one stage-1 epoch at lr 1e-4. The "PPI vs base"
  comparison is therefore not yet a test of PPI.
- **GEARS base selected epoch 1.** Validation mse_deg rises after epoch 1–2
  (0.141 → ~0.16) while training `L_true` keeps falling. Validation
  `pearson_specific` stays near 0 for base (test: 0.42).
- **Stage-2 epochs (GEARS):** validation mse_deg jumps and recovers in the next
  stage-1 epoch.
  - signed: 0.159 → 0.193 at epochs 6, 12, 18;
  - finetune: 0.162 → 0.239 at epoch 6, 0.233 at epoch 12.
  - This is B8.
- **Stage-2 epochs (MLP):** validation is identical to 4 decimals before and
  after (B7). `projected` = 1.0, so every batch conflicts. λ climbs 0.5 → 1.42 →
  1.80 → 1.90 because `L_u − L_pg` < 0.
- All GEARS arms beat leave-one-out on mse_deg. The MLP arms are at the
  leave-one-out level and carry no perturbation signal (specific pearson ≈ 0.07).
- Timing: base 321 s, signed 430 s, finetune 264 s, mlp base 203 s, mlp signed 261 s.

### 5.2 Earlier Adamson runs (history; all superseded)

All in `Perturb_middle/output_adamson_upr/`.

| dir | settings | headline |
|---|---|---|
| `run/` | first smoke (old trainer) | — |
| `run-lambda/run` | learnable λ, Adam stage 2, GEARS decay on MLP, `pearson_specific` selection, val 10%, batch 32 | signed picked stage-2 epoch 6: test mse_deg 0.311; mlp signed 4.18 |
| `run-2/` | projection + SGD + mse_deg selection, still val 10%, batch 32, no finetune | base 0.180 (epoch 10); signed, mlp base and mlp signed all picked **epoch 1** (0.259) |
| scratchpad `mlp-diag` | MLP with constant lr, *before* projection | stage 2 diverged: L_pg 8.8e12, val mse_deg 8.8e6 |
| scratchpad `proj-mlp` / `proj-gears` / `ft-smoke` | projection verification | no divergence; B7/B8 evidence |

### 5.3 Replogle (old trainer: GEARS decay on MLP, Adam two-optimiser stage 2, `pearson_specific` selection)

`artifacts/prepared/replogle-k562/run` (GEARS) and `run-2` (MLP):

| arm | mse_deg | pearson | specific pearson | direction |
|---|---|---|---|---|
| base | 0.178 | 0.414 | 0.312 | 0.849 |
| signed ppi | 0.166 | 0.394 | 0.310 | 0.851 |
| mlp base | 0.211 | 0.327 | 0.002 | 0.815 |
| mlp signed ppi | 0.211 | 0.327 | 0.002 | 0.815 |

- Computed in-session from the saved predictions: GEARS signed vs base gives
  pearson_de20 0.590 → 0.607 (CI +0.001..+0.032) and mse_de20 0.178 → 0.166
  (CI −0.015..−0.008). Both beat leave-one-out.
- The GEARS `run/` has **no `perturbation_results.json`**: the process
  core-dumped at the end. The suspect is the faulthandler stall dump, now
  opt-in, but this is not confirmed.
- MLP signed == MLP base because it selected a pre-stage-2 epoch.
- **Must be rerun under the current trainer before quoting.**

### 5.4 Open problems

**B7. Stage-2 SGD barely moves the weights** (open; decision D1). SGD's step is
lr × gradient, and the gradient of a loss averaged over ~5,000 genes is tiny.
- MLP (lr 1e-4): validation is identical before and after stage 2.
- GEARS: the StepLR decay takes stage-2 lr to ~3e-6 at epoch 6.

Options:
- (a) a larger stage-2 lr;
- (b) a normalised step;
- (c) Adam, with the projection applied to Adam's preconditioned update
  `m̂/(√v̂+ε)` (recommended).

**B8. GEARS' stage-2 jump is BatchNorm** (open; decision D2). With weights
barely moving, validation mse_deg still jumps in every GEARS stage-2 epoch and
recovers after; the MLP (no BN) never jumps. Strongest case: finetune epoch 6,
student ≈ teacher, SGD lr ~5e-6, validation 0.162 → 0.239 (smoke run: pearson
0.338 → 0.083).
- Cause: the student's train-mode forward on Dixit cells overwrites the BN
  running statistics with a different distribution (§3.1 batch effect).
- Fix: keep BN running stats frozen for the supplementary forward (and arguably
  for the whole stage 2), then verify with the finetune smoke.

**B9. mse_deg selection picks very early epochs.** Even with 13 validation
perturbations, the best epoch is 1–2 for GEARS base, signed and finetune.
Validation mse_deg is flat or rising while training loss falls.
- Partly a real generalisation gap: train and validation perturbations differ,
  and validation `pearson_specific` ≈ 0.
- Partly noise.
- Consequence: no selected model has seen stage 2 (§5.1).
- Options: combined selection, minimum epochs before selection, or selection
  restricted per stage. **Ask the user**; paper-first says GEARS uses val `mse_de`.

**B10. PPI distribution assumption violated on Adamson.** Gold vs Dixit inputs
are easily separable (§3.1), so a cell-conditioned student can treat the two
differently. This is root cause of the original divergence. The principled fix
is in stage 2: calibrate the supplementary cells to the gold-control
distribution (e.g. per-gene quantile / mean–variance matching). Not started.

**B11. λ is bang-bang.** λ·correction is linear in λ, so λ moves to a bound;
the sign of `L_u − L_pg` decides which. This is inherent to the scDesign3
formulation. It is not variance-optimal λ as in PPI++. Noted, not changed.

**B5. The MLP learns no perturbation signal** (structural: one-hot identity),
5–6% of the observed spread.

**Other open items:**
- Replogle needs a rerun under the current trainer, also to separate `G_S`
  from stage 2. The cause of the core dump is unconfirmed.
- Datafetcher:
  - the raw matrix is picked as the "largest file";
  - mtx → table conversion depends on download order;
  - companion files are labelled `unusable` instead of "part of gold";
  - the review preview is not stratified by label.
- Modality is dataset-level and mostly unread: a fetched screen is `unchecked`.
- Perturbation identity is gene-shaped: a drug, dose or shRNA cannot be encoded.

### 5.5 Fixed (do not re-fix)

- `parse_condition("UBL5+ctrl")` → `('UBL5','ctrl')` dropped every example.
- Only 412/1,092 Replogle perturbed genes were measured → GO vocabulary.
- Loss restricted to top-20 DEGs (not GEARS) → GEARS non-zero genes.
- Gold used the control *mean* as input → a real control cell per row.
- KeyError `base_mse_deg` in MLP-only runs.
- Datafetcher: 0-byte cache reuse; slow line-by-line mtx parsing (scipy now;
  Norman loads in 36 s); Dixit `.mtx.txt.gz` + cellnames/genenames + inverted
  `cbc_gbc_dict`; metadata beside matrices; the `*` guide treated as a gene.
- **B1** GEARS StepLR applied to the MLP → per-family decay.
- **B2** `pearson_specific` selection picked exploded epochs → mse_deg.
- **B3/B4** unbounded `λ·(L_u − L_pg)` → projected step (L_u, L_true non-increasing).
- **B6** Adamson validation of 6 perturbations → 20% (13).
- The stage-2 objective, first `L_u` (×10) and `−λ·L_pg` (÷10) as two Adams →
  `λ·(L_u − L_pg)`, learnable λ.

---

## 6. Speed (measured 2026-10-08, A6000, Adamson sizes, 2.6M params)

| per batch | B=32 | B=128 |
|---|---|---|
| stage 1 (student fwd + 1 bwd) | 14 ms | 35 ms |
| stage 2 | 60 ms | 135 ms |
| projection itself | ~1.5 ms | ~1 ms |
| teacher forward | 5 ms | 14 ms |
| peak memory | 1.0 GB | 3.8 GB |

- **Why stage 2 costs about 4×:** 2 teacher forwards, 2 student forwards, and 3
  backward passes (∇λ(L_u−L_pg), ∇L_u, ∇L_true).
- **Why a stage-1 epoch is ~17 s at B=128 against ~7 s of GPU work:** the
  rest is CPU batching (`_make_batch` stacks numpy rows) and per-epoch
  validation.
- **GEARS is heavy by design:** a (B·G, 64) state for every cell × gene.
- Profiling script: scratchpad `prof/prof.py`, since lost with the session. It
  rebuilt `GearsModel` with random graphs.

Proposed speed-ups (D3):
1. Pre-stack the train/val tensors on GPU and slice batches by index (about 2×
   for stage 1, same results).
2. Cache the teacher Δ for gold rows: the pairing is fixed and the teacher
   frozen (~10% of stage 2, same results).
3. bf16 autocast (1.5–2×; quartic loss precision must be checked; needs approval).

---

## 7. Knobs (env)

| env | default | effect |
|---|---|---|
| `KOSMOS_PERTURBATION_ARMS` | all five | `base`, `signed`, `finetune`, or arm keys (`gears_base`, `gears_augmented_ungated`, `gears_finetune_ppi`, `mlp_base`, `mlp_augmented_ungated`, `mlp_signed`) |
| `PPI_MLP_BASELINES` | on | `0` drops the MLP arms when ARMS is unset |
| `KOSMOS_PERTURBATION_BATCH_SIZE` | study `batch_size`, else 32 | batch size |
| `KOSMOS_PERTURBATION_SELECTION` | `mse_deg` | `pearson_specific`, `pearson`, `mse` |
| `KOSMOS_PERTURBATION_CONDITION_ON_CONTROL` | 1 | 0 = original GEARS |
| `PPI_LAMBDA` | **0.5** (bridge default; config default 1) | initial λ, must be in (0, 2) |
| `PPI_STAGE1_EPOCHS`, `PPI_STAGE2_EPOCHS` | 5, 1 | stage cycle |
| `PPI_LAMBDA_LR_MULTIPLIER`, `PPI_STAGE2_LR_MULTIPLIER` | 10, 0.1 | λ lr, model stage-2 lr |
| `PPI_FINETUNE_EPOCHS`, `PPI_FINETUNE_LR_MULTIPLIER` | 12, 0.1 | the finetune arm |
| `PPI_DIRECTION_LAMBDA` | 0.1 | direction-term weight |
| `KOSMOS_PERTURBATION_MAX_GENES` | 2000 | panel cap when the plan has no study (0 = all) |
| `KOSMOS_PERTURBATION_TARGET_SUM` | 1e4 | log1p normalisation target |
| `KOSMOS_PERTURBATION_MODALITY` | `warn` | `strict` raises on a mismatch |
| `KOSMOS_DATAFETCHER_MAX_BYTES` | 8 GiB | per-file fetch ceiling |
| `KOSMOS_STALL_REPORT_SECONDS` | 0 | stack-dump interval; 0 = off |
| `PPI_OUTPUT_DIR` | auto | output of `kosmos run` |

Config-only (no env, `PerturbationTrainingConfig` in `train.py`): `lr_decay` 0.5,
`mlp_lr_decay` 1.0, `lambda_min/max` 0/2, `supplementary_ratio` 1,
`sample_controls`, `resample_controls`, `grad_clip` 1.0, `weight_decay` 5e-4,
`hidden_dim` 64, `mlp_hidden_size` 128.

---

## 8. Tests and environment

```bash
.venv/bin/python -m pytest -o addopts="" -q -p no:logging \
  tests/unit/ppi tests/unit/discovery tests/unit/cli tests/datafetcher
```

- Last runs (2026-10-08): `tests/unit/ppi` + `test_two_stage_workflow.py` +
  `tests/unit/discovery` → **300 passed**. The broad run → 499 passed, 29 failed.
- The 29 failures are all in `cli/test_commands.py` / `cli/test_graph_commands.py`
  (they patch `kosmos.cli.main.get_session`). They are pre-existing, not ours.
- `-o addopts=""` is required (`pytest.ini` demands coverage).
- Key test files:
  - `tests/unit/ppi/perturbation/test_perturbation_training.py`: schedule,
    stage-2 optimisers, projection, per-family decay, finetune init, pairing;
  - `test_perturbation_gears_port.py`;
  - `test_perturbation_smoke.py`;
  - `tests/datafetcher/test_matrix_layouts.py`.
- Notes:
  - the CLI renders through `rich.Live`; redirecting stdout kills the panel, but
    the `# … epoch` and `… still running` lines still print;
  - `next_output_dir` never overwrites;
  - run untrusted downloaded data with `python -I`.

---

## 9. Report material (already explained to the user; reuse)

- **Fetch module.** `datafetcher/` is a standalone package that finds public
  single-cell data for a question and packages it for review; it does not train.
  - Layout: **fetch** (`fetch.py`, `sources/net.py`, `store.py`), **read
    format** (`singlecell.py`), **read content** (`profile.py`), **write
    result** (`plan.py` → gold / supplementary / unusable).
  - Around it: the CLI entry `kosmos/cli/commands/dataprep.py`; orchestration
    `scripts/auto_task_run.py`; judgement `kosmos/discovery/` +
    `kosmos/ppi/task_inference.py`; the deterministic alternative
    `kosmos/ppi/perturbation/registry.py`.
- **Adamson example:** §3.1 (fetch outcome, the stage-2 subset table, why one sample).
- **`specific pearson`:** §4.7. Both pearsons are on Δ vectors over all genes;
  the specific one first subtracts the mean Δ across the evaluated
  perturbations. Example on Adamson test: mlp base pearson 0.405 but specific
  0.066 (it predicts the shared response only); GEARS base 0.466 / 0.421.

---

## 10. Where things live

```
kosmos/ppi/perturbation/train.py      arms, _train_arm, LearnableLambda, project_to_descent,
                                      stage schedule, selection, finetune arm, config
kosmos/ppi/perturbation/model.py      GEARS port (+ G_S branch, condition_on_control)
kosmos/ppi/perturbation/mlp.py        graph-free MLP
kosmos/ppi/perturbation/losses.py     gears_loss, retained_genes
kosmos/ppi/perturbation/graphs.py     G_C / G_S / G_GO, go_vocabulary, directed top-k
kosmos/ppi/perturbation/contract.py   parse_condition, build_task, control_cells
kosmos/ppi/perturbation/run.py        table -> task -> splits -> graphs -> trainer
kosmos/ppi/perturbation/metrics.py    perturbation_metrics (pearson_specific), baselines, signal
kosmos/ppi/perturbation/figures.py    summary.md, figures, arm labels
kosmos/ppi/perturbation/registry.py   dataprep --dataset (registered GEARS releases)
kosmos/ppi/training.py                correction_stage (shared stage schedule)
kosmos/ppi/discovery_bridge.py        plan (+ study) -> backend, env overrides, arm aliases, metrics.json
kosmos/agents/research_director.py    metric whitelist (incl. finetune_ppi_*)
kosmos/cli/commands/{dataprep,experiment}.py
scripts/prepare_perturbation.py       stage 2 for registered datasets
scripts/auto_task_run.py              retrieval, file preference, review, plan
datafetcher/                          fetch / read / profile / plan
Perturb_middle/                       per-task stage-2 scripts and outputs
/home/ydong233/GEARS                  reference GEARS code
/home/ydong233/scDesign3/src/train_ppi.py   reference two-stage PPI (learnable λ)
```

## 11. Next steps (in order)

1. Get the user's decisions D1 (B7) and D2 (B8); implement; verify with a short
   finetune smoke. The stage-2 epoch should change validation through the
   weights, and BN should no longer jump.
2. Discuss B9 (selection picks epoch 1–2) with the user before the next full run.
3. Optionally D3 speed-ups 1 + 2.
4. Full Adamson rerun (command in §2) and compare the five arms.
5. Consider B10 (calibrate Dixit to the gold distribution in stage 2).
6. Rerun Replogle under the current trainer.
7. Commit when the user asks. Nothing is committed; `git status` shows the
   full change set.

---

## 12. PerturbNet arms (added 2026-10-12)

Why: a VAE-family model that predicts **unseen** genetic perturbations (scGen
cannot: its latent δ needs training cells of that perturbation). The question is
whether supplementary cells help when they train only the cell representation.

| arm | scVI (cell VAE) trained on | cINN / GenotypeVAE |
|---|---|---|
| `perturbnet_base` (`perturbnet base`) | gold train split (train perturbations + all controls) | identical |
| `perturbnet_supp` (`perturbnet supp`) | the same + Dixit, `batch_key = source` | identical |

User decisions (2026-10-12): separate env, Kosmos `.venv` untouched; predictions
normalised exactly as the gold; PerturbNet's fixed epochs (no selection;
validation only reported); supp enters scVI with a source batch covariate;
stage 2 writes raw counts.

- **Env:** conda `perturbnet` (Python 3.7, torch 1.13.1+cu117, scvi-tools 0.7.1,
  torchvision 0.14.1, rdkit; no TensorFlow). Clone `~/PerturbNet` is installed
  editable with `--no-deps`; the runner stubs `perturbnet.data_vae.vae` (TF VAE,
  imported by `perturbnet.util`, unused). Interpreter: `KOSMOS_PERTURBNET_PYTHON`.
- **Weights:** `data/perturbation/perturbnet/genotypeVAE/` (HuggingFace
  `cyclopeta/PerturbNet_reproduce`). All 82 Adamson genes are annotated once
  11 old symbols are mapped (`AARS→AARS1`, `SLMO2→PRELID3B`, `SRPR→SRPRA`, …;
  `SYMBOL_ALIASES` in the runner).
- **Stage 2:** `prepare_adamson_upr.py` now also writes `gold_counts.parquet` and
  `supp_dixit_k562_wt_counts.parquet` (panel counts + `total_counts`, same rows)
  and `study.counts` / `study.target_sum`. Rerun verified: `gold.parquet` and
  the supp table are unchanged; counts reproduce log1p-CPM to 3e-7.
- **Code:** `kosmos/ppi/perturbation/perturbnet_arm.py` (bundle → subprocess →
  score), `scripts/perturbnet_runner.py` (runs in the env). `EXTERNAL_ARMS` in
  `train.py`, never in the default arm set. Bridge `_study_counts`, aliases
  `perturbnet`, `perturbnet_base`, `perturbnet_supp`.
- **Settings** (tutorial / scPerturbBench): scVI `n_latent=10`, 500 epochs; cINN
  20 flows, hidden 1024, 45 epochs, batch 128, lr 4.5e-6. Env:
  `KOSMOS_PERTURBNET_SCVI_EPOCHS`, `KOSMOS_PERTURBNET_CINN_EPOCHS`.
- **Normalisation:** each generated cell is paired with a random gold training
  cell: scVI's inferred library of that cell sets the depth, and its panel
  fraction (panel counts / total) turns generated panel counts into
  whole-transcriptome CPM. Δ̂ = mean log1p-CPM − gold control mean.
  `generated_control_vs_real` in the results is a calibration check (a ctrl
  profile generated with the all-zero GO vector vs the real control mean).
- **Two PerturbNet quirks handled:** cINN reads scVI latents by AnnData row
  index, so gold rows come first (asserted); the stock `sample_data` passes the
  DataLoader counter as the batch index, so the supp arm uses
  `sample_data_with_batch` with the gold batch code.
- **Smoke** (scVI 5, cINN 2 epochs; not results): both arms ran end to end,
  ~130 s each. Timing per epoch: scVI ~3.5 s, cINN ~32 s, so a full arm ≈ 28 +
  24 min. Tests: `test_perturbation_perturbnet.py` (7) + previous 300 pass.

Full run:

```bash
CUDA_VISIBLE_DEVICES=0 KOSMOS_PERTURBATION_ARMS=perturbnet \
.venv/bin/python -m kosmos.cli.main experiment "<question>" \
  --data-plan Perturb_middle/output_adamson_upr/plan.json \
  --task perturbation --max-epochs 20 --patience 0 \
  --out Perturb_middle/output_adamson_upr/run-perturbnet --yes
```

## 13. "Small gold, large supp" switch (added 2026-10-12)

User decisions: the gold is NOT chosen by size (relevance first); the supp may
prefer size. Opt-in, default behaviour unchanged.

- `dataprep --supp-size large` (→ `auto_task_run.py --supp-size large`):
  - first round (the labeled screen): `supplementary_requirement_text(prefer_large=True)`
    tells the model to name labeled screens only (asking it for "the largest
    source" there made it spend the round on an atlas -- T-cell run 2026-10-12);
  - evidence round: `propose_evidence(prefer_large=True)` (rule
    `PREFER_LARGE_EVIDENCE`) and preflight `choose_downloads(prefer_large=True)`,
    which replaces "choose the smallest file" with "the one with the most cells";
  - plan: `datafetcher plan --supp-order rows` (`PlanTask.supplementary_order`)
    orders accepted supplementary tables by rows, largest first. Stage 3 and
    PerturbNet use only the **first** supplementary source (decision: no pooling yet).
- Stage 3 cap: `KOSMOS_PERTURBATION_MAX_SUPP_CELLS` (default 200,000; 0 = all),
  seeded + sorted (`contract.supplementary_rows`); the PerturbNet bundle takes
  the same rows. The cap is applied after `read_table`, so the read itself still
  peaks at the full table.
- PerturbNet bundles are deleted after a successful arm
  (`KOSMOS_PERTURBNET_KEEP_BUNDLE=1` keeps them).
- Old runs cleaned 2026-10-12: failed `run-perturbnet`, bundles of
  `run-perturbnet-v2`, scratchpad. Older Adamson/IRF1/Replogle runs kept.

## 14. Stage-1 retrieval fixes (2026-10-13), from the T-cell question

Question: *In primary human CD4+ T cells, can the transcriptional response to CRISPR
knockout of transcriptional regulators not seen in training be predicted ...,
using large single-cell atlases of unperturbed human T cells as supplementary
evidence?* Three runs found no gold; causes and fixes:

| problem | fix |
|---|---|
| the prompt's identifier example was a real accession (`geo://GSE194122`), and the model copied it as "a T-cell CRISPR screen" | placeholder `geo://GSE<number>` (`search.py::_identifiers_block`) |
| search tools offered but never used; accessions named from memory (COVID atlases as "screens") | `propose_datasets(search_first=True)` when search tools exist: the model writes 2-4 short queries (3-5 words, one = method + cell type), they run, it proposes from the hits (gold round + correction round) |
| ToolUniverse unset unless `KOSMOS_TOOLUNIVERSE_PYTHON` | default `~/venvs/tooluniverse/bin/python` when it exists (`tooluniverse.interpreter`) |
| gold round fetched atlases / pooled bulk screens | `auto_task_run._not_a_screen`: perturbation gold round refuses, before download, a candidate whose repository description has no screen term or no single-cell term; the reason goes to the correction round |
| per-sample `GSM*_matrix.tar.gz` inside `RAW.tar` never unpacked → label tables with no transcriptome | `fetch._unpack_archives` is a queue, `MAX_ARCHIVE_DEPTH = 2`; `.xlsx/.docx` no longer unzipped into XML parts |
| 4 samples' `matrix.mtx` converted to one `matrix.mtx-table.csv` (overwrote) | `_table_label`: archive prefix when names collide (`GSM3375483_D1N_matrix__matrix.mtx-table.csv`) |
| evidence round searched on the label table's columns (`sample.name UMI.count`) | perturbation: no gene panel → skip the round with a message, never fall back to label columns |

Verified on the real question (one run, stopped during the supp download): search →
GSE119450 (Shifrut 2018 CROP-seq, 4 samples D1N/D1S/D2N/D2S) → gate passed → nested
matrices converted → evidence round matched on 33,661 genes and proposed PBMC atlases
(GSE158055 1.4M cells, over the 2 GiB cap; GSE164378 being downloaded when stopped).
No plan was produced after the fixes. Tests: 484 pass.

Open (user decisions pending): with `--supp-size large`, a per-file supp cap
(~2 GB) and deleting archives after unpack+convert (now archive + extracted +
converted tables are all kept; GSE119450 = 3.0 GB on disk for a 344 MB download).
Disk: /home 96% full -- tests must set `--max-download-mb` (memory
`feedback-test-download-size`). Also: the 4 GSE119450 label tables become 1 gold +
3 "supplementary" (multi-gold external); they are samples of one screen (donor ×
stimulation), to be resolved in stage 2.

## 15. K562 "small gold, large supp" attempt (2026-10-13)

Question tried: *In K562 cells, can the transcriptional response to CRISPRi knockdown
of genes not seen in training be predicted from a small labeled Perturb-seq screen,
using large collections of K562 single-cell profiles from other CRISPR screens, with
their perturbation labels ignored, as supplementary evidence?* (`--max-download-mb 300`,
out `artifacts/dataprep/k562-small-gold`). Search-first found GSE220974 (CRISPRai,
pairwise combinations) and GSE308682 (CROP-seq, 7 TFs) instead of the cached Adamson /
Norman -- both unsuitable (combinatorial / too few perturbations). 466 MB downloaded.
Both matrices were unreadable because each GEO file is its own fetch and the matrix
landed before its barcodes/features. Fixed: `_convert_single_cell` now also converts
an unconverted matrix in the series directory once its companions exist (test
`test_a_matrix_fetched_before_its_companions_is_converted_when_they_arrive`).

## 16. Retrieval ranking (2026-10-13) -- no pinning (user: pinning the gold is cheating)

Second K562 run again landed on GSE220974/GSE308682 (deleted, 911 MB): the model's
"short" queries dropped the topic (`unfolded protein response`). Now
`search.py::search_repositories` (was `_search_first`):
- the model returns `topic`, `assay`, `cell_type` + queries; the code adds anchors it
  may have dropped (gold: `topic assay`, `topic cell_type`; evidence: `cell_type
  assay`, `cell_type single-cell`);
- hits are aggregated: titles containing most of the topic words first, then by how
  many searches returned them (HuggingFace returns the same K562 repos for every
  query, which tied them with the UPR screen on count alone);
- search titles kept to 200 chars (`run`'s 90-char cut removed "unfolded protein
  response" from GSE90546's title).
- perturbation evidence round: hits come from `search_repositories(purpose="evidence")`
  (cell type + assay), not a keyword search on gene names.
Live check (no downloads): gold ranking puts GSE90546 first ([5 searches, on topic]);
evidence ranking surfaces large K562 sources (e.g. `hf://pvd232/k562-gwps-raw`,
`hf://huggingbahl21/causalbench-k562`). 487 tests pass. /home 98% (88 GB free).

## 17. K562 UPR run review + fixes (2026-10-13)

Run `artifacts/dataprep/k562-upr-small-gold`: retrieval proposed GSE90546 first (0.95,
0.90) -- the ranking fix works -- but (1) the gold-round preflight downloaded the
"large collection" (GSE146194, over the 2 GiB cap) and dropped Adamson until the
retry; (2) the primary gold became GSE132080's per-guide phenotype table (128 rows,
not cells), which then made Adamson's cell identities "unusable"; (3) the large
K562 sources were all over the 2 GiB cap, so the evidence round kept small ones
(GSE220974 24,661 cells, GSE308682 10,847). Fixes:
- `choose_downloads(screen_round=True)` in the perturbation gold round (+ its retry):
  this round takes the labeled screen, not the unlabeled collection;
- `_per_cell_labels`: before `choose_primary_gold`, a screen's label tables must be
  per cell (>= 1000 rows or a cell/barcode column); `choose_primary_gold(screen=True)`
  asks for per-cell labels on the question's topic;
- evidence round (and its retry) per-file cap `SUPP_LARGE_MAX_BYTES` = 10 GiB under
  `--supp-size large` unless `--max-bytes` is given; the evidence retry no longer
  applies the screen gate. `--max-bytes` help fixed (default is 2 GiB, not 8).
Mechanical round: still `--mechanical-search` (off since c0a8107: rule-built
single-word queries downloaded unrelated repos). Proposed (not done, user to decide):
feed its rule-built query's hits into `search_repositories`' ranked pool instead of
downloading them directly. 491 tests pass.

## 18. Run 1 vs run 2 (`k562-upr-small-gold` vs `-2`) and the sample-title fix

| step | run 1 | run 2 (after §17) |
|---|---|---|
| gold-round proposals | GSE90546 0.95, GSE146194 | GSE90546 0.95, GSE146194 |
| gold-round download | GSE146194 (failed, >2 GiB); Adamson only in the retry | Adamson, first try (cache); GSE146194 dropped |
| primary gold | GSE132080 per-guide table (128 rows) | GSM2406675 **10X001 = pilot TF screen** (wrong sample) |
| supp proposals | GSE220974, GSE308682, hf pvd232 | hf causalbench-k562, hf pvd232/k562-gwps-raw, GSE308682 |
| supp kept | GSE308682 (10,847), GSE220974 (24,661) | GSE308682 (10,847); both HF dropped: their listing failed in-run (transient) |

Fix: `geo.sample_titles` (one `targ=gsm` brief record) + `auto_task_run._attach_sample_titles`
→ `sample=` in `choose_primary_gold`'s facts. Live check: the model picks
GSM2406681 (10X010, "UPR Perturb-seq experiment") twice. 493 tests pass.
Still open: HF listings can time out in-run, and `hf://pvd232/k562-gwps-raw` is
867 GB (a whole-repo pull would be refused by size only if listed); evidence round
still ends with a small supp.

## 19. Jurkat question + 10x h5 / CROP-seq reader (2026-10-13)

User wanted a smaller gold; a no-download survey (scratchpad `survey/survey.py`:
`search_repositories` + `describe_reference` + `geo.sample_titles`) over 5
questions found one pairing: gold **GSE137554** (Jurkat CROP-seq, TCR stimulation,
100 MB h5, 12,758 cells: naive 7,181 / stim 5,577; 87 guides → 29 targets, no
non-targeting guides; ~5,559 single-guide cells) and supp **GSE249595** (genome-scale
Jurkat CRISPRi Perturb-seq, 16 channels, RAW.tar 3.76 GB) or `hf://VirtualCell2025/jurkat.h5`.
Question: *In Jurkat T cells, can the transcriptional response to CRISPR knockout of
T cell receptor signaling genes not seen in training be predicted from a small labeled
screen, using large single-cell profiles of Jurkat cells from other CRISPR screens,
with their labels ignored, as supplementary evidence?* First stage-1 run
(`artifacts/dataprep/jurkat-tcr-small-gold`): 0 gold -- the reader did not know
CellRanger `.h5`, and CROP-seq guides are matrix features (`Tcrlibrary_LCK_1_gene`).
Fixed in `datafetcher/singlecell.py`:
- `read_10x_h5` (v2 genome group / v3 `matrix/features`), `is_10x_h5`; mtx and h5
  share `_matrix_table`;
- `guide_features` / `guide_labels`: v3 `CRISPR Guide Capture`, or a library prefix
  shared by >= 6 `Lib_GENE_n(_gene)` features → per-cell `guide` (`no_guide`,
  `multiple`) + `guide_umis`; guide features dropped from the genes;
- unfiltered matrix: when the metadata names < 50% of the matrix barcodes, keep only
  those (GSE137554: 12,758 of 1,474,560);
- `_metadata_frame`: the `-1/-2` suffix is stripped only when < 10% of the *metadata*
  barcodes match (stripping merged two samples: 12,758 → 25,408).
Decided (user): controls = cells with no detected guide, per stimulation condition
(stage 2). Pending (user): the speed fixes (§ below) -- reuse re-converts every
single-cell file (`_rebuild_derived`, 178 s for cached GSE90546), 2,000 × 33k-gene
preview CSVs (~260 MB) re-read by partner detection (≈3 min) and the plan profile
(≈2 min); proposed: small parquet preview + gene list/cell count in the manifest,
convert only when the reader version changes, profile cache by sha256; and a
`dataprep --survey` mode. 496 tests pass.

## 20. New stage 1 for perturbation, Step 1 = survey (2026-10-13, user-approved design)

Design (user approved): stage 1 for perturbation is rebuilt around datasets/samples:
Step 1 survey (no download) → Step 2 per-sample fetch → Step 3 load every format into
a standard `.h5ad` (counts, obs: perturbation/control/sample/condition, var:
symbol+ensembl) + validation → plan.json/data.md. Only for perturbation tasks;
the per-cell stage 1 is untouched; old perturbation patches are deleted only after
the new path passes its tests.

Step 1 is done: `kosmos/ppi/perturbation/dataprep/{spec,survey}.py`
(`python -m kosmos.ppi.perturbation.dataprep.survey "<q>" --out DIR`), plus
`datafetcher/sources/geo.py::sample_files` (a GSM's own suppl files with sizes).
- spec: cell type (+ aliases), organism, modality, topic (dropped when all its words are
  generic, `specific_topic`), assays;
- queries from the topic and the *shortest non-acronym* cell-type name;
- describe: series text, `sample_titles`, `sample_files` (retried: NCBI throttles
  bursts; 2 series × 4 samples at a time); a series whose samples have no files gets
  the series-level files, and its samples collapse into one row;
- cell estimate from the barcodes file (3.8 B/cell gz; > 250k → unfiltered → "?");
- judge: one model call per series, per-sample schema; `gate`: gold = single-cell
  screen, right cell type/organism/modality, labels (model flag OR a label-like file
  name OR guide companion samples, `label_evidence`); a failed gold of the right
  cells becomes supplementary.
Survey test, 6 questions (80-180 s each, nothing downloaded):
| question | gold | supplementary |
|---|---|---|
| K562 UPR | GSM2406681 Adamson UPR (741 MB, ~65.8k) | Adamson pilot/epistasis, HF K562 CRISPRi |
| Jurkat TCR | GSE92872 Datlinger CROP-seq (series-level, 19 MB) | GSE249595 genome-scale Jurkat CRISPRi (16 ch, ~230 MB each), J-Lat |
| CD4 T cells (KO) | none -- GSE314342 is a CD4 **CRISPRi** Perturb-seq (2.6 GB/lane): demoted for modality | GSE314342 lanes |
| THP-1 PD-L1 | GSE221321 KO screens (756 MB / 2.3 GB) | GSE268256 (~9.7k), GSE221321 KD |
| mouse BMDC LPS | GSE327057 BMDC E3 screen (series-level, very large) | none |
| iPSC neurons | GSE124703 Tian CROP-seq (225-240 MB/lane), GSE207099 | GSE277604/277602 (13-31k each) |
Next: Step 2 (per-sample fetch) + Step 3 (readers → h5ad, validation), then end-to-end
on 2 questions and the Adamson regression.

## 21. Jurkat TCR stage 2 (`Perturb_middle/prepare_jurkat_tcr.py`, 2026-10-13)

Survey (`artifacts/dataprep/jurkat-tcr/survey.md`) → gold GSE92872 (Datlinger CROP-seq,
dense genes x cells CSV with 5 header rows condition/replicate/cell/grna/gene; `CTRL` =
non-targeting) and supp GSE249595 channels (unfiltered mtx, 6.8M barcodes; cells called
at >= 1,000 UMIs). Output `Perturb_middle/output_jurkat_tcr/`: gold 3,204 stimulated
cells (705 CTRL, 29 targets, 28-156 cells each; DHODH/MVD/TUBB efficacy controls
dropped), supp 108,667 cells (ch1 54,697 + ch2 53,970), panel 5,011 genes (HVG among
supp-measured + 11 perturbed), counts tables + study block, `prepare_report.md` with the
survey-estimate vs actual table. Counts reproduce log1p-CPM (2e-7 / 5e-7).
**Problem found:** gold-ctrl vs supp gene-mean correlation 0.227 (Spearman 0.38): the
supp is a probe-based assay (top genes MT-ATP6 17.8%, LEF1, VIM; no RPS/RPL; 20,606
features; ~520 genes/cell), not 3' whole-transcriptome. Not an alignment bug. Pending
user decision: keep it, or switch supp to GSE263194 (Jurkat DMSO scRNA) /
`hf://VirtualCell2025/jurkat.h5`. Survey should also record the assay chemistry.

## 22. Stage 2 pools similar supplementary samples (2026-10-13, user direction)

User: stage 2 should pick several similar supp datasets and pool them, not fix one.
`Perturb_middle/prepare_jurkat_tcr.py` now: every survey supp candidate (+ unused gold
candidates, labels ignored) is downloaded within `--supp-budget-gb` (3) into
`data/fetched/samples/<series>/<sample>/`, read in full by the new
`kosmos/ppi/perturbation/dataprep/load.py` (`load_sample`: mtx triplet / 10x h5 /
h5ad; `call_cells` >= `--min-umis` 1000; `similarity`: Pearson/Spearman of mean
log1p-CPM vs the gold controls + median UMIs + ribosomal fraction), and pooled when
Pearson >= `--min-similarity` (0.6); a series is judged on its first sample, at most
`--max-samples-per-series` (2) pooled. Pooled table `supp_jurkat_pooled.parquet`
(+ `_counts`) with a `supp_source` column (stage 3's condition guesser ignores it).
Result: GSE249595 dropped (0.24, probe-based); pooled 72,837 cells from 6 samples:
GSE241207 J1.1 (0.87, 10,564) and J-Lat (0.88, 8,753) -- HIV-latency Jurkat models;
VirtualCell2025 jurkat.h5 (0.81, 21,412; no ribosomal genes in that file); GSE263194
DMSO x2 (0.89; 9,999 + 9,570); GSE148820 GSM4483270 Jurkat-Cas9 (0.87, 12,539).
Gold 3,204 stimulated cells (705 CTRL, 29 targets). Panel 5,013 genes. 2.5 GB downloaded.
