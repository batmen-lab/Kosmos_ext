# PPI classification training

`kosmos.ppi` is self-contained. It reads no code or models from a sibling
repository at runtime. It supports binary/multiclass numeric tabular features,
a sklearn-compatible pseudo classifier or reviewed pretrained classifier, and
any CPU PyTorch downstream classifier factory returning class logits. Install
`python -m pip install -e '.[ppi,dev]'` in a stable Python 3.11+ environment.
Regression, automatic pretrained-model discovery and translator construction are
outside this first implementation.

## Data roles

- `GoldDataset(role="gold_train")`: genuinely observed target features and trusted
  labels. PPI materializes both `y_true` and `y_pred` (plus `y_prob` when available).
- `GoldDataset(role="gold_validation")`: held-out development labels, used only for
  downstream validation and early stopping. Neither pseudo-model fitting nor
  fitting its preprocessing pipeline receives this dataset.
- `ExternalEvidenceDataset`: target-space `X`, global `sample_ids`, source/route,
  optional weights and audit metadata. There is no `y_true` field. Source labels
  are stored separately and never consumed by the loss or weight construction.
- `GoldDataset(role="final_test")`: rejected by training/selection APIs. Only the
  separate, explicit `evaluate_final_test(result, final_test)` operation accepts it.

Direct-PPI observed/harmonized features and Translate-PPI translated features
enter **one** downstream training path. Route and translator information persist
as audit metadata; the loss neither reads source modality nor source labels.
Gold schemas reject translated representations and external route declarations.
They cannot detect a caller falsely relabeling translated data as observed.

## Implemented mathematical objective

The inspected local reference was `../scDesign3/src/train_ppi.py` (the requested
`../scDesign/src` path did not exist), with the same two-stage pattern in
`train_ppi_targetSyn.py` and `train_ppi_baseline.py`. The reference alternates
`L_true` and `lambda * (L_unlabeled - L_pseudo_gold)` and learns lambda.
The pseudo-generation script fits on gold; the trainer explicitly warns about
in-sample gold pseudo predictions. No definitive additional AutoInspire objective
was identified in the local notes searched.

The modular reference implementation in `losses.py` defines:

```text
L_true        = mean_i CE(h(x_i), y_i)
L_pseudo_gold = mean_i CE(h(x_i), g(x_i))
L_external    = sum_j w_j CE(h(x*_j), g(x*_j))
m             = sum_j w_j
L_correction  = lambda * (L_external - m * L_pseudo_gold)
L_total       = L_true + L_correction
```

With equal external weights `w_j = b/N`, this is
`L_true + lambda*b*(mean_external_CE - mean_gold_pseudo_CE)`.
The factor `m` applies the evidence budget to **both** sides of the correction;
zero mass disables correction instead of leaving a negative gold-pseudo term.
External minibatches sample uniformly with replacement and use
`N * mean_batch(w_j * CE_j)`, an unbiased estimate of the weighted external sum.
Gold and external minibatches remain separate. Epoch length depends on gold size,
not the number of external observations.

Default `schedule="two_stage"` uses four true-loss epochs followed by one
correction-loss epoch, with separate AdamW optimizers and a 0.1 correction learning
rate multiplier. `schedule="joint"` optimizes `L_total` at each step. Every epoch
logs true, gold-pseudo, external, correction and total losses plus the actual
optimized loss and phase. These are minibatch training diagnostics, not a
post-epoch evaluation of the full training population at one fixed parameter state.

**Explicit assumptions/differences from scDesign:** lambda is fixed (default 1,
restricted to [0,1]), the default external mass budget is 0.5, and the gold-trained
fallback uses cross-fitting. Learnable lambda and other PPI variants are not
implemented. This is a configurable reference objective inspired by the inspected
code, not a claim to be the definitive AutoInspire formulation or to provide
PPI inferential guarantees under biological/translation shift. Signed corrections
can destabilize training; non-finite optimized losses fail rather than being
silently clipped. Replace `loss_factory` to test another objective.

## Pseudo predictions and pretrained models

`prepare_pseudo_labeler(gold_train, estimator, config)` supports:

- `cross_fit` (default): clone and fit a sklearn estimator within each stratified
  fold; use group-aware folds when groups exist. Every gold observation receives
  exactly one held-out prediction. Each fold must contain all target classes in
  its training portion. Put preprocessing in a sklearn `Pipeline` so that it is
  fit within each fold. Fold IDs and estimator settings are persisted.
- `in_sample`: explicitly opt into the scDesign-style gold-fit/gold-predict
  behavior. This can overstate pseudo accuracy and weaken the correction; it is
  recorded, never silently substituted when cross-fitting fails.
- `pretrained`: use `PretrainedPseudoLabeler` around an already fitted task
  classifier, with model/code version, ordered features, applicability evidence,
  leakage review and known training sample IDs. It is never fit here. Probability
  columns are aligned to gold classes. An embedding model alone is not a target
  label predictor: a suitable classification head/ontology mapping is required.

Cross-fit models predict external data as a fold ensemble (mean probabilities,
otherwise majority votes). Gold predictions come from the held-out fold model.
This choice avoids in-sample gold predictions and reuses fitted fold models;
it is not an exact single fixed-g prediction regime. A fold-specific PPI estimator
would be a distinct future objective. For a reviewed independently pretrained
classifier, gold and external predictions use exactly the same frozen model.

Probabilities are retained even with default hard-target CE. `pseudo_targets="soft"`
uses them in both pseudo terms and requires `predict_proba`. Hard-only classifiers
are supported; calibration or probabilities are not fabricated from hard labels.
Source labels and confidence never automatically determine alpha.

A concrete biomedical pretrained model cannot be selected without the target
feature definition, label ontology and biological domain. Supply a reviewed one
when available; otherwise the default is the gold-trained cross-fit fallback.
No model is silently downloaded, selected on validation scores, or presented as
scientifically applicable based only on its modality name.

## Weights, budgets and leakage

`alpha` defaults effectively to one and is used only when
`use_evidence_weights=True`. It must be a supplied vector in [0,1].
`evidence_weight` is distinct: it is the upstream absolute influence allocation
and is always respected, including when alpha is ablated. Without upstream
weights, sources share the budget equally. After uniform row subsampling, a
source's retained base weights are scaled to its original allocated source mass;
then alpha is applied and total mass is capped downward to the configured budget.
This is an explicit finite-sample weighting policy, not learned evidence quality.

`max_external_samples` caps total rows; `max_rows_per_dataset` caps each source.
Sources receive balanced quotas so a huge source cannot consume the whole row
budget. Zero selected rows, zero total mass or lambda=0 returns the exact matched
baseline. With tiny evidence populations, external batches sample with replacement.

Sample IDs must use a shared namespace across gold, validation and external data.
Duplicate gold/validation/external IDs are rejected. If gold groups (donor,
patient, study, etc.) exist, both gold roles and external datasets must provide
groups; validation groups may not overlap training or external groups. Use
`split_gold()` for deterministic group-aware development splitting. Validation
classes must occur in gold training. Final-test evaluation checks development
IDs, known pretrained IDs, group overlap and feature/class schema separately.

These guards cannot discover hidden biological duplicates with different IDs,
unknown pretraining overlap or falsely declared roles. Full dataset/model lineage
review remains necessary for a real experiment. Do not tune anything against
final-test outputs or import final-test rows under another role.

## Python API

```python
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from kosmos.ppi import PPITrainingConfig, run_ppi_experiment

# Construct typed GoldDataset/ExternalEvidenceDataset objects from real files.
result = run_ppi_experiment(
    gold_train=gold_train,
    gold_validation=gold_validation,
    external_evidence=external_datasets,
    pseudo_labeler=make_pipeline(
        StandardScaler(), LogisticRegression(max_iter=1000)
    ),
    config=PPITrainingConfig(cross_fit_folds=5, schedule="two_stage"),
    output_dir="artifacts/ppi/experiment-001",  # must be empty/new
)
print(result.baseline_metrics, result.validation_metrics, result.delta_metrics)
```

`model_factory(n_features, n_classes)` can return your own `torch.nn.Module`.
The default is a linear classifier. One seeded initialization is deep-copied for
baseline and PPI, with the same architecture, validation protocol, batch settings,
learning rate, epoch cap and early-stopping rule. Optimizer schedules differ
intentionally for the two-stage method. Report your custom model's code/version
in `config.model_reference`; its factory name and model structure are logged.

Undefined classwise AUROC/AUPRC produce null macro metrics with explanations.
If an undefined ranking metric was requested for early stopping, selection
explicitly falls back to balanced accuracy and logs that choice. Early stopping
can select a pre-correction epoch; choose sufficient patience/epochs to exercise
the two-stage schedule rather than assuming PPI necessarily changed the model.

## Connect the evidence router

```python
from kosmos.ppi import prepare_pseudo_labeler, RoutingPredictor, PPIEvidenceTrainer

prepared = prepare_pseudo_labeler(gold_train, estimator, training_config)
predictor = RoutingPredictor(prepared)
trainer = PPIEvidenceTrainer(
    gold_train, gold_validation, prepared, training_config,
    output_dir="artifacts/ppi/routing-evaluations",
)
result = evidence_pipeline.run(
    candidates, predictor, trainer, routing_config,
    output_dir="artifacts/evidence/routing-001",
)
```

The router needs `candidate.metadata["sample_ids"]` for **all original rows** and
`metadata["groups"]` when applicable. The adapter tracks row selection and refuses
to invent global biological IDs from row numbers. It checks prediction agreement
with the routing model, preserves source audits, honors both routing and trainer
budgets, and returns validation utility through the existing `ValidationResult`
contract. The prepared pseudo-labeler is reused for gold and all source ablations.
No duplicated modality-specific training code is involved.

## CLI and artifacts

```bash
python -m kosmos.ppi.train --synthetic --output-dir artifacts/ppi/synthetic-001
python -m kosmos.ppi.train \
  --gold-train train.npz --gold-validation validation.npz \
  --external-evidence external-a.npz external-b.npz \
  --config ppi.json --output-dir artifacts/ppi/real-001
```

NPZ keys match dataset constructor fields (`X`, `y` for gold, `sample_ids`, `role`,
`dataset_id` for gold or `source_dataset_id`/`evidence_route` for external, optional
`feature_names`, `groups`, `alpha`, `evidence_weight`, `source_labels`). Store strings
as Unicode arrays; metadata/audit_metadata as JSON strings. The loader disables
pickle. CLI configuration is a JSON `PPITrainingConfig`; pretrained classifiers
and custom architectures use the Python API. No precomputed pseudo files are
required and the CLI has no final-test option.

Outputs: `result.json`, baseline/PPI state-dict checkpoints, `pseudo_gold.npz`,
`pseudo_external_N.npz`, and held-out validation predictions. Results include
configuration hash, library versions, fold/sample IDs, selected epochs, losses,
baseline/PPI/delta metrics, route counts, sample counts, source audit records,
confidence/weight summaries and the downstream model factory. Preserve the
referenced model factory and input data to reload checkpoints or rerun experiments.

## Verification and current environment

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m pytest \
  tests/unit/ppi tests/unit/evidence -q -o addopts=''
```

The synthetic run is a software smoke test, not a biomedical demonstration or
claim of improvement. The implementation uses CPU; it does not require a CUDA
installation compatible with the local driver.

Upstream development hit an environment constraint: the original `.venv` used
Python **3.11.0rc1**, which lacks `sys.get_int_max_str_digits`, so PyTorch 2.14
could not initialize AdamW. Validation therefore ran on a stable Python 3.11.15
interpreter with the same `.venv` packages (no PyTorch/runtime monkeypatches and
no dependency changes).

This checkout ships a dedicated `.venv` created from that stable 3.11.15
interpreter, so the commands below run without an explicit interpreter path:

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 MPLCONFIGDIR=/tmp/kosmos-mpl \
.venv/bin/python -m kosmos.ppi.train \
  --synthetic --output-dir /tmp/kosmos-ppi-new-run
```

`.venv` reuses the third-party packages of the original Kosmos venv through a
`.pth` file (see `_kosmos_borrowed_packages.pth`); reinstall normally with
`pip install -e .[ppi]` if you need a self-contained environment. Real
biomedical experiments additionally need curated gold/development splits, global
sample/group identifiers, a compatible pseudo classifier or gold training
estimator, reviewed translators where required, and a predeclared PPI
configuration/validation protocol. This module does not supply those scientific
inputs or guarantee that the signed correction helps under domain shift.

### Implementation verification record

- Relevant suite: **40 passed** (21 PPI tests plus 19 existing evidence tests),
  using stable Python 3.11.15 and the installed dependency set.
- Ruff passed for `kosmos/ppi`, `tests/unit/ppi`, `kosmos/evidence` and
  `tests/unit/evidence`; `git diff --check` passed.
- Default synthetic CLI run: 90 gold training, 90 gold validation and 300 external
  observations; seed 42. Baseline/PPI accuracy were both 0.344444; balanced
  accuracy both 0.328876; delta 0. The selected checkpoints did not demonstrate
  a benefit. Artifacts from this run: `/tmp/kosmos-ppi-synthetic-final/result.json`.
- The normal repository command `pytest tests/ -q` collected 3,754 tests in that
  run and was stopped around 25% after widespread failures; it did **not** pass
  or complete. `/tmp/kosmos-full-tests.log` retains progress. A targeted rerun of
  existing end-to-end tests under stable Python reproduced three setup errors:
  Pydantic Settings could not parse `enabled_domains`. Details are in
  `/tmp/kosmos-existing-failures.log`. These errors occur before PPI execution.

Files added for this extension:
`kosmos/ppi/__init__.py`, `schemas.py`, `pseudo_labeler.py`, `datasets.py`,
`losses.py`, `trainer.py`, `metrics.py`, `split.py`, `integration.py`, `train.py`;
`tests/unit/ppi/test_training.py`; and `docs/ppi_training.md`.
Files updated: `pyproject.toml` (optional torch dependency), `README.md`,
`docs/evidence_routing.md`, and `kosmos/evidence/pipeline.py` (prediction lineage
and JSON-safe numeric class labels). The existing `.venv` was not modified.
