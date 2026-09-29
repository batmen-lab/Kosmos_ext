# Kosmos_ext — handoff

State as of 2026-09-27. Written for whoever picks this up next (human or agent).
The last work stream was **perturbation-response prediction**; the section that
matters most for continuing it is §4.2 and §6.

---

## 1. What this repository is

`/home/ydong233/Kosmos_ext` is a fork of the Kosmos research agent (upstream is
at `/home/ydong233/Kosmos`). Upstream is a general "AI scientist" loop:
hypothesis generation → experiment design → LLM-written code → execution →
analysis → report. This fork adds, on top of that loop:

* a **data layer** that actually fetches and gates real datasets (`datafetcher/`),
* a **task layer** that decides what a question is asking for (`kosmos/ppi/task_ontology.py`, `task_inference.py`),
* a **training layer** for supervised and semi-supervised learning on that data (`kosmos/ppi/`),
* a **perturbation-response backend** modelled on GEARS (`kosmos/ppi/perturbation/`),
* an explicit `single_cell` domain (`kosmos/domains/single_cell/`),
* run-time visibility (`kosmos/core/runcheck.py`).

Upstream's own defaults are still present and still matter:

* the code-generation path (`kosmos/execution/code_generator.py`) **fabricates synthetic data** when no `data_path` is given ("synthetic fallback", Issue #51), and `data_source` is recorded but never surfaced in the report. The data layer exists so that this path is not the default any more; it has *not* been removed.
* `CodeExecutor` sandboxes generated code with Docker. On this machine the daemon socket is not accessible to the user (`PermissionError` on `/var/run/docker.sock`), so execution currently falls back to in-process. `KOSMOS_REQUIRE_SANDBOX=1` makes that a refusal instead of a silent fallback, and every `ExecutionResult` now carries `sandbox_status`.

Sibling repositories:

| path | what it is |
|---|---|
| `/home/ydong233/Kosmos` | upstream (has the PPI loss + donor-split work this fork was seeded from) |
| `/home/ydong233/GEARS` | the GEARS reference implementation (Roohani/Huang/Leskovec, *Nature Biotechnology*). **PyTorch Geometric is not installed here**, so GEARS is not imported; the model in `kosmos/ppi/perturbation/model.py` is a plain-torch re-implementation of its architecture. |

---

## 2. Architecture map

```
question
   │
   ├── run.py  ─────────────────────────────► simple task: fetch+plan, then train
   │      └── scripts/auto_task_run.py ─────► retrieval, gating, plan.json
   │
   └── kosmos.cli.main run ─────────────────► research loop (hypotheses, design, …)
            └── research_director ──────────► picks a backend per experiment:
                  1. plan-driven (task + data_path)      → run_training
                  2. discovery data task (bridge)        → single-cell backend
                     └── perturbation question           → perturbation backend
                  3. otherwise                           → LLM-written code (upstream)
```

| layer | modules | role |
|---|---|---|
| data | `datafetcher/*` | list-files, fetch, unpack, convert h5ad/mtx → table, reuse + rebuild, ledger/manifest, search tools |
| gating | `kosmos/discovery/*` | mechanical + LLM review, adjudication, gold/supp/unusable roles, `data_report.md` |
| task | `kosmos/ppi/task_inference.py`, `task_ontology.py` | which column is the label (with guards), which *kind* of question this is, which backend answers it |
| simple training | `kosmos/ppi/{flow,trainer,losses,gating,features,split,singlecell}.py` | per-cell supervised training, PPI/pseudo-label/gated losses, single-cell preprocessing |
| perturbation training | `kosmos/ppi/perturbation/*` | contract, graphs, GEARS-shaped model, three arms, perturbation-level metrics |
| domain | `kosmos/domains/single_cell/` | modality judgement + the conventions the single-cell pipeline relies on |
| run visibility | `kosmos/core/runcheck.py`, `kosmos/core/diagnostics.py` | heartbeat + JSONL record + stage names during long runs |
| CLI | `run.py` | `--task simple` (default) and `--task perturbation` |

---

## 3. Environment and operations

* Python: `.venv` (built on miniforge `polarbear`), whose `site-packages` are **borrowed** from `/home/ydong233/Kosmos/.venv` through a `.pth`. torch 2.14, sklearn, numpy 1.26. **No PyG, no GPU use, CPU only.**
* Run tests with `-o addopts=""` (the repo's `pytest.ini` demands coverage otherwise):

  ```bash
  .venv/bin/python -m pytest -o addopts="" tests/unit/ppi tests/datafetcher -q -p no:logging
  ```
* Commands that touch the network (fetching, LLM calls, the CLI) need escalated execution in a sandboxed session; the workspace sandbox (`bubblewrap`) was broken for part of this work, so `apply_patch` sometimes failed and edits were made with anchor-asserted in-place writes. **Review `git diff` before committing.**
* Staged data (not in git): `data/fetched/**` (≈9 GB), `data/perturbation/{norman,replogle}` and `data/perturbation/reference/go_essential_all.csv` (355 MB, the GO similarity graph GEARS uses, Harvard Dataverse file 6934319). `.gitignore` already excludes `artifacts/runs/`, `data/fetched/`, `data/perturbation/reference/`, `ToolUniverse/`, `dockerfiles/` stays.
* **Gotcha:** the repo used to have a `docker/` directory at its root, which shadowed the PyPI `docker` package and silently disabled sandboxed execution. It is now `dockerfiles/`. Do not recreate `docker/`.
* `runcheck.jsonl` (one per run, beside the run's `run/` directory) is the first thing to read when a run looks stuck: `tail -f artifacts/runs/<name>/runcheck.jsonl`. `KOSMOS_RUNCHECK_SECONDS` (default 30) controls the heartbeat.

---

## 4. The two pipelines

### 4.1 Simple per-cell pipeline (kept working; do not regress)

Entry: `run.py "<question>" --domain biology --hint <label column> --out …`

1. **Retrieval + gating** (`scripts/auto_task_run.py`): the model proposes datasets (ToolUniverse tools do the mechanical search/download), preflight checks sizes and the run budget against `list-files`, files are downloaded (reuse of staged bytes by default), archives are unpacked (by name **or by sniffing the bytes**), single-cell containers are converted to one-row-per-cell tables with their non-numeric `obs` metadata.
2. **Screening**: a candidate must parse as a table (≥2 columns).
3. **Target inference** (`task_inference.py`): hint exact → hint partial → objective prose (strict: short names and function words cannot be promoted by prose) → constrained model choice; structural guards reject ids, continuous classification labels and feature-like columns.
4. **Dual review** (`kosmos/discovery/review.py`): rules and a model each call every table gold / supplementary / unusable; disagreements go to a human (`adjudication.json`), and the rules no longer overrule a model veto that was based on the question's own wording.
5. **Plan** (`plan.json` + `data_report.md`): gold = the table with the label; supplementary = tables with X and without the label; the plan names the target column, the feature list and per-source column renames.
6. **Training** (`kosmos/ppi/flow.py::run_training`): single-cell decision (`auto` = wide panel, non-negative, ≥50 % zeros, ≥99 % integers, **or** converted from `.h5ad/.h5mu/.mtx`), per-source preprocessing (drop rare genes → top-N HVG → library-size normalisation → log1p → per-gene z-score → clip negatives), panel = genes measured by every source, ranked by how many sources selected them. Two arms are always fitted: `baseline` (gold only) and `ppi` (with supplementary rows).
7. **Objectives** (`PPITrainingConfig.loss_mode`): `signed` (PPI correction, unbiased), `plain` (pseudo-label distillation), `gold_plus_synthetic` (naive baseline), `gradient_gated` (see §4.2 for the controller).
8. **Reports**: `run/summary.md` (data provenance, recipe + why, loss, validation/final-test tables, per-class recall), `run/ppi_summary.json`, `run/training_log.jsonl`, `run/figures/` (six single-cell panels), `run/preprocessing.md`.

Verified end-to-end most recently on the islet question (gold = GSE84133 `human1`, supp = `human2/3/4`), exit 0.

### 4.2 Perturbation-response pipeline (the current work stream)

**Task contract** (`kosmos/ppi/perturbation/contract.py`)

* One example = one perturbed cell: `control` (the mean control profile of its *context group*), `perturbation` (a tuple of gene symbols), `delta = y_pert − μ_ctrl`, `target = y_pert`, plus whatever context columns exist.
* `μ_ctrl` is context-matched (cell type / donor / batch when present) and the contract records which group was used and how often it fell back to the global mean.
* A **shared gene panel and a stable `gene_to_index`** are shared by every dataset and every graph.
* Splits are **by perturbation identity**: `mixed` (default), `unseen_single`, `unseen_combination`. Cells of one perturbation never straddle splits.
* Artifact: `perturbation_contract.json` (panel, mapping, splits, per-source detail).

**Graphs** (`graphs.py`)

| graph | built from | defaults |
|---|---|---|
| `G_C` | **control cells of the gold only** | \|Pearson\| ≥ 0.4, top-k = 20, symmetrised (GEARS' `coexpress_threshold`) |
| `G_S` | supplementary cells: their **control cells** if present; otherwise the whole source **residualised** on perturbation/dose/time/donor/batch; one graph per source, never pooled | same thresholds |
| `G_GO` | GO similarity edge list `source,target,importance` (dataset `go.csv` or the staged 6934319 file), top-k per target | k = 20 |

Every graph is saved (`*.npz` + `*.json`) with its statistics (threshold, k, edge count, density, degree quantiles, isolated nodes) and provenance in `graph_report.json`; the **edge overlap between `G_C` and `G_S`** is reported (it is the first diagnostic: two graphs sharing no edges cannot inform each other).

Why control cells only: the auxiliary graph measures covariation, and any dataset of the same tissue/cell line with unperturbed cells is usable regardless of what it perturbed. That is why the retrieval ask for a perturbation question is "a source with control/untreated cells", not "a source with the same perturbations" (`supplementary_requirement_text()`).

**Model** (`model.py`) — plain torch, CPU

* learnable gene embeddings `E`; a sparse GCN (shared parameters) gives `H_C = GNN(E, G_C)` and `H_S = GNN(E, G_S)`, combined as `H_A = H_C + η·H_S`;
* the **compositional module** encodes a perturbation as the sum of its perturbed genes' states (single and multi-gene perturbations share the mechanism);
* the **GO branch** pools each perturbed gene with its GO neighbours before encoding;
* a **cross-gene decoder** (`Linear(n_genes → hidden)` over the gene axis) and **gene-specific output layers** produce one number per gene; the prediction is `control + Δ`.

**Loss** (`losses.py`) — mirrors GEARS' `loss_fct`: per perturbation group, `Σ(pred − y)^{2+γ}` with γ = 2 (quartic) restricted to that perturbation's DEG set, plus `λ · Σ(sign(y−ctrl) − sign(pred−ctrl))²`, averaged over groups.

**Three arms** (`train.py`), identical perturbation-level splits:

| arm | objective |
|---|---|
| `gears_base` | `L_G` — gold rows, measured targets |
| `gears_augmented_ungated` | the column task's **signed PPI correction**: `L_G + λ·(L_S − L_pseudo_gold)` |
| `gears_augmented` | `g_G + λ_t·∇L_S` — the detached controller |

**The augmentation is teacher-generated data, not a graph trick.** `gears_base`
trains first, is frozen, and *labels the supplementary control cells* (queries
drawn from the gold **training** perturbations); the two augmented arms then
train on gold (measured) plus supp (synthetic) targets. `L_pseudo_gold` is the
same model scored against the teacher's labels on the *gold* rows — the control
variate that makes the signed correction unbiased. `kosmos/ppi/perturbation/synthetic.py`
builds those labels from a `predict(control, index)` callable, so the **MLP arms
use the identical mechanism** (`mlp_base` / `mlp_augmented_ungated` /
`mlp_augmented`): the two backends then differ only by architecture.
The supplementary co-expression graph `G_S` is still computed and fed to the
augmented arms (the panel is the shared gene intersection, so it is harmless),
but it is no longer the source of the second gradient. The old
`∇(L_A − L_G)` graph-increment objective is gone.

The gate is the detached controller in `kosmos/ppi/gating.py` (batch scope only; a graph built from an entire source has no per-row reading): `λ_t = max(0, cos(g_G, g_ΔG)) · min(1, κ‖g_G‖/‖g_ΔG‖) · λ`. If the increment conflicts, `λ_t = 0` and the update is exactly `g_G`. Per epoch the run logs cosine, weight, both gradient norms and the active-step fraction.

**Metrics** (`metrics.py`, `figures.py`) — perturbation-level: all-gene MSE, DEG MSE, Pearson/Spearman of Δ, direction accuracy, top-K DEG overlap; plus `figures/perturbation_overview.png` (arms compared, Δ predicted vs observed, gate curves, training loss), per-arm `*_test_predictions.npz` and `summary.md`.

**Data staging** (`sources.py`, `stage.py`) — for a perturbation question the registered screens are used:

| name | cell line | Dataverse file |
|---|---|---|
| `norman` (paired CRISPRa) | K562 | 6154020 |
| `adamson`, `dixit` | K562 | 6154417, 6154416 |
| `replogle_k562_essential` | K562 | 7458695 |
| `replogle_rpe1_essential` | RPE1 | 7458694 |

`stage_sources()` fetches the **supplementary source first**, reads the genes it measured, then converts the gold with `candidate_genes = supp genes ∪ perturbed genes` so the HVG selection happens **inside the intersection** (`intersect-then-select`). The fetcher reuses staged bytes, completes archives that were never unpacked (or were cut short by the member/size caps) and converts single-cell files that came out of an archive.

**CLI**

```bash
.venv/bin/python run.py "<question>" --task perturbation --domain single_cell \
  --go-graph <go.csv> --split-mode mixed|unseen_single|unseen_combination \
  --supp-limit 1 --test-fraction 0.2 --validation-fraction 0.1 \
  --max-epochs 8 --patience 4 --seed 0 --out artifacts/runs/<name>
```

Flags: `--task`, `--go-graph`, `--split-mode`, `--test-fraction`, `--validation-fraction`, `--min-cells-per-perturbation`, `--go-k`, `--coexpress-threshold`, `--coexpress-k`, `--eta`, `--gold-table`, `--supp-table` (skip staging), plus the existing `--gate-kappa/--gate-lambda/--max-epochs/--patience/--seed`.

Last end-to-end CLI run (auto-staged, panel 666, splits 296/44/58, ~2 min):

| arm | mse | mse_deg | pearson | spearman | direction | top-K |
|---|---|---|---|---|---|---|
| `gears_base` | 0.1859 | 0.6902 | 0.322 | 0.188 | 0.863 | 0.275 |
| `gears_augmented_ungated` | 0.1698 | 0.8146 | 0.291 | 0.176 | 0.838 | 0.250 |
| `gears_augmented` | 0.2151 | 0.7372 | 0.321 | 0.177 | 0.775 | 0.225 |

An earlier manual run with a 784-gene panel had the opposite ordering on direction/top-K (the auxiliary graph helped). **The panel is deciding the result — see §6.1.**

---

## 5. Test and verification status

* `tests/unit/ppi` (includes `perturbation/`), `tests/datafetcher`, `tests/unit/domains`, `tests/unit/core/test_runcheck.py`, `tests/unit/agents/test_discovery_data_tasks.py`: **652 passed, 1 deselected** (the deselected one is a pre-existing `openpyxl` failure in `tests/unit/domains/materials`).
* Not covered by tests: the real-data perturbation run (network + 3 GB downloads). It has been run manually several times; the artefacts are in `artifacts/runs/perturbation-*/`.
* `scripts/gradient_gate_stress.py` produced the corruption sweep for the gate (cosine/weight/active fraction fall as synthetic labels degrade) in `artifacts/runs/gate-stress/`.

---

## 6. Open problems, with pointers

### 6.1 The panel is narrow, and it decides the comparison (highest priority)

Norman's processed file has **5,045** genes and Replogle's **4,999**, but they share only **784 symbols / 819 Ensembl ids** — the two releases were filtered to different gene subsets. Consequences:

* the trained panel is 666–840 genes (17–21 % of the gold's 4,000), and the auxiliary graph can only reach that much of the screen;
* the arm ordering flipped between panels, so no conclusion about the value of the auxiliary graph can be drawn yet.

Options, in the order I would try them:

1. **Unfiltered screens** (Replogle's full release rather than the "essential" subset; or a screen published with a genome-wide panel). This is a data choice, not a code change.
2. **Harmonise identifiers** — intersect on Ensembl ids and map back to symbols (`_var_names` already prefers `feature_name → gene_name → _index`; a mapping layer would go in `stage.py`/`graphs.py`).
3. **Pick a different supplementary pair** with a compatible gene universe (the registry makes this a one-line change in `sources.py`, plus `default_pair()`).
4. If the panel stays narrow, report the comparison **within** the panel and say so; the warning already fires (`panel_share_of_gold`, `panel_warning`).

### 6.2 Model fidelity against GEARS

The implementation is GEARS-*shaped*, not GEARS: no PyG, no uncertainty head, no genetic-interaction prediction, no pretrained checkpoints, and the GO branch is a pooling step rather than GEARS' separate similarity head. If fidelity matters, either install PyG and wire `/home/ydong233/GEARS` (its `PertData`/`GEARS` API, `.h5ad` with `obs['condition']` + `var['gene_name']`), or port the missing heads.

### 6.3 Context features

The model accepts categorical + numeric context (cell type, dose, time, donor, batch), but the h5ad converter deliberately drops **numeric** `obs` columns, so dose/time never reach the table. A `KOSMOS_SINGLE_CELL_KEEP_NUMERIC_OBS`-style knob (or a per-task override in `stage.py`) is needed for dose-response work; residualisation in `graphs.residualize` also assumes those columns are present.

### 6.4 Routing and orchestration

* **Wired now.** Both entry points call the same `discovery_bridge.run_data_task`:
  `kosmos run --task perturbation ...` (explicit) or a perturbation-worded
  question with `--task auto` (the classifier decides), and `run.py --task
  perturbation` (which now delegates to the bridge instead of staging itself).
  The registry is the default source; a caller's own tables go through
  `--gold-table/--supp-table`, and a plan is honoured when one is passed.
* **Perturbation type is judged and recorded** (`kosmos/ppi/perturbation/modality.py`):
  the question's requested modality (crispra / crispri / crisprko / mixed / unknown)
  and the screen's declared one (from `PerturbationSource.perturbation`) are compared;
  the verdict lands in `perturbation_contract.json` (`detail.modality`,
  `sources.gold.modality`) and `summary.md`, and the keys
  `modality_requested/provided/verdict` in `metrics.json`. A mismatch warns by
  default; `KOSMOS_PERTURBATION_MODALITY=strict` refuses to train, `=off` disables.
  It does **not** change which screen is selected -- the registry default still
  decides that (handoff 6.4). Note this means the CBL question above was answered on
  Norman's CRISPRa (over-expression) screen, not a knockout one.
* **Both backends leave the same artifacts.** `backends` are `per_cell` (the
  column task) and `perturbation`; each writes `run/summary.md`,
  `run/metrics.json`, its figures, and `data_report.md`/`data_report.json`
  beside the run. `DataTaskOutcome` is the one return shape, so the director
  does not branch on the backend.
* **Two graph-free MLP baselines** (`mlp_base`, `mlp_augmented`) run alongside the
  three GEARS arms in the autoresearch flow (`kosmos/ppi/perturbation/mlp.py`): the
  same MLP architecture, gold-only vs teacher-labelled supplementary rows combined by
  the existing gate. **They optimise the same objective as the GEARS arms**
  (`mlp_objective="gears"`, the quartic DEG loss + direction on Δ, sharing the DEG
  sets), so a five-arm comparison is one of architecture/data, not of losses; the
  design doc's MSE form is `mlp_objective="mse"`. Off in
  `PerturbationTrainingConfig` by default (library callers unchanged); the bridge
  turns them on, `PPI_MLP_BASELINES=0` turns them off.
* `--task auto|per_cell|perturbation`, `--condition-column`, `--control-label`,
  `--split-mode`, `--go-graph`, `--eta`, `--max-epochs/--patience/--seed` are on
  `kosmos run` as well as `run.py`.
* Perturbation metrics are now recorded with the experiment (the whitelist in
  `research_director.py` carries `*_mse_deg`, `gated_pearson`, …). A real
  end-to-end loop run still has not been done (it needs the network).

### 6.5 Smaller items

* No CLI to inspect/resume a perturbation run (the artefacts are JSON; a `scripts/show_perturbation.py` would help).
* `go_graph()` filters the 355 MB reference per run (~30 s); a panel-keyed cache would help. Dataset `go.csv` files (19.7 MB) are the fast path.
* CPU only; `gnn_layers=1` in the runs so far. A GPU path would change wall-clock, not the API.
* The gate's sample scope is deliberately unavailable for the graph increment (documented in `gating.py`); keep it that way.

---

## 6.6 Research-loop robustness (why a re-run used to hang in hypothesis generation)

A second run of the same question never reached the experiment. Two independent
causes, both now fixed:

* **Novelty self-comparison (the loop).** `NoveltyChecker._check_existing_hypotheses`
  queried every stored hypothesis in the domain, including the ones an *earlier
  run of the same question* had written, and `min_novelty_score=0.5` filtered any
  match with similarity > 0.5. A regenerated hypothesis scored similarity ~0.8 to
  its own past output ⇒ novelty 0.16 ⇒ filtered ⇒ empty pool ⇒ regenerate
  forever. Fixes: skip stored hypotheses with the same `research_question`
  (`novelty_checker.py`); score novelty on the literature only and treat stored
  matches as advisory; and a safety net in `hypothesis_generator.py` keeps the
  best-scoring hypothesis when filtering would empty the set. The filter's own
  decisions now log at WARNING so they are visible at the CLI's default level.
* **Malformed/over-long JSON.** `generate_structured` asked for JSON only in
  prose and dropped the template's system prompt (which carried the JSON example
  and the length guidance). `openai.py` now passes
  `response_format={"type":"json_object"}` (provider-gated, `KOSMOS_STRUCTURED_JSON_MODE=0`
  to disable, retried without it if an endpoint rejects it), the hypothesis call
  sends `HYPOTHESIS_GENERATOR.system_prompt`, and parse failures log the full
  response + `finish_reason` instead of the first 500 characters.

Neither touches the training design: the task contract, splits, arms and metrics
live in `run_data_task`/`run_perturbation_task`, and `--task perturbation` pins
the backend so the hypothesis text never reaches it.

## 7. Command cookbook

```bash
cd /home/ydong233/Kosmos_ext
set -a; . ./.env; set +a
export MPLCONFIGDIR=/tmp/kosmos-mpl KOSMOS_TORCH_THREADS=4

# tests
.venv/bin/python -m pytest -o addopts="" tests/datafetcher tests/unit/ppi -q -p no:logging

# simple per-cell run (islet; gold+supp)
.venv/bin/python run.py "<single-cell question>" --domain biology \
  --intent "<what the data looks like>" --hint <label column> \
  --fetch-limit 2 --supp-limit 3 --max-epochs 3 --patience 2 \
  --out artifacts/runs/<name>

# perturbation run (stages the registered screens itself)
.venv/bin/python run.py "<perturbation question>" --task perturbation --domain single_cell \
  --split-mode mixed --supp-limit 1 --max-epochs 3 --patience 3 --seed 0 \
  --go-graph data/fetched/http/dataverse.harvard.edu__api__access__datafile__6154020/6154020-contents/norman/go.csv \
  --out artifacts/runs/<name>

# inspect
.venv/bin/python scripts/show_run.py artifacts/runs/<name>/run      # simple runs
tail -f artifacts/runs/<name>/runcheck.jsonl                       # any run
sed -n 1,30p artifacts/runs/<name>/run/summary.md
```

Useful env knobs: `KOSMOS_SINGLE_CELL_LABEL`, `KOSMOS_SINGLE_CELL_MAX_CELLS`, `KOSMOS_SINGLE_CELL_MAX_GENES`, `KOSMOS_SINGLE_CELL_SCAN_CELLS`, `KOSMOS_DATAFETCHER_MAX_BYTES`, `KOSMOS_DATAFETCHER_ARCHIVE_MEMBERS`, `KOSMOS_DISCOVERY_DATA_TASKS`, `KOSMOS_REQUIRE_SANDBOX`, `KOSMOS_RUNCHECK_SECONDS`, `KOSMOS_MAX_ACTIONS_PER_ITERATION` (the research loop's per-cycle action budget; raise it for a real run -- the 50 default can be spent by a slow hypothesis step and force convergence before any experiment runs), `KOSMOS_TASK_HINTS`, `KOSMOS_TASK_INTENT`, `KOSMOS_FETCH_LIMIT`, `KOSMOS_SUPP_LIMIT`, `KOSMOS_ALLOW_CROSS_SPECIES`.

---

## 8. Things that will bite

1. Recreating a top-level `docker/` directory breaks `import docker` and silently disables sandboxing.
2. `artifacts/` is gitignored; a run's numbers live only there and in `~/.kosmos/logs/kosmos.log`.
3. The fetcher's reuse path rebuilds derived tables and completes archives — if a staged table looks stale, re-fetch (reuse) rather than re-downloading; `--refresh` forces a fresh download.
4. `pytest.ini` enforces coverage: always pass `-o addopts=""`.
5. Long LLM/network steps are silent by design unless you watch `runcheck.jsonl` or the console heartbeat.
6. `data/perturbation/reference/go_essential_all.csv` is 355 MB and gitignored; a check that needs it must re-download Dataverse file 6934319 (the dataset zips also contain a per-dataset `go.csv`).
