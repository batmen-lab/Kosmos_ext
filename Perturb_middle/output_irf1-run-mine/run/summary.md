# Perturbation response run

- panel: 2000 gene(s), condition column `condition`
- splits (by perturbation): {'train': 3640, 'validation': 0, 'test': 2286}
- arms: gears_base, gears_augmented_ungated, gears_augmented_finetune

> **Partial run.** The arms above have finished; the run was still training (or was stopped) when this file was written. Read it again after the run ends for the complete table.

## Perturbation type

- requested by the question: `crisprko` (decided by words)
- provided by the data: `unknown`
- verdict: **unchecked** - perturbation modality was not compared: the data does not declare one

## Test metrics (held-out perturbations)

| arm | mse | mse_deg | pearson | spearman | direction | top-K overlap | epochs |
|---|---|---|---|---|---|---|---|
| `gears_base` | 0.0556 | 0.1218 | 0.013 | -0.006 | 0.800 | 0.025 | 1 |
| `gears_augmented_ungated` | 0.0463 | 0.1780 | 0.026 | 0.027 | 0.725 | 0.050 | 1 |
| `gears_augmented_finetune` | 0.0450 | 0.1289 | 0.012 | -0.009 | 0.750 | 0.025 | 1 |

## Reference baselines (same metrics, same held-out perturbations)

| baseline | mse | mse_deg | pearson | direction | top-K |
|---|---|---|---|---|---|
| predict 0 | 0.0139 | 0.1861 | - | 0.000 | 0.000 |
| mean response (leave-one-out) | 0.0208 | 0.1256 | 0.287 | 0.750 | 0.250 |

`predict 0` is the no-change reference; `mean response (leave-one-out)` predicts each held-out perturbation with the average response of the others. An arm that does not beat both is not predicting the perturbation.

## Per-perturbation signal

How much of the *observed* difference between perturbations the arm reproduces (spread of its per-perturbation profiles / spread of the observed ones). 0 means a per-gene constant.

| arm | across-perturbation std | of the data's |
|---|---|---|
| `gears_base` | 0.0002 | 0.5% |
| `gears_augmented_ungated` | 0.0021 | 3.8% |
| `gears_augmented_finetune` | 0.0004 | 0.8% |

> **No perturbation signal**: `gears_base`, `gears_augmented_ungated`, `gears_augmented_finetune` reproduce less than a tenth of the observed between-perturbation variation. Their prediction is a per-gene constant, so their `top_k_overlap` (and `pearson`) describe that constant rather than a ranking of the perturbation's genes -- read those rows as 'did not learn the perturbation', not as a ranking result.

## Graphs

- gold co-expression vs supplementary: 2868 shared edge(s), Jaccard 0.329 (first 4366, second 7232)

## Fine-tune

- starts from the gold-trained teacher and keeps supervising the measured gold rows *and* the cells that teacher labeled (`L_gold + L_pseudo`), with no correction term
- epochs: 1
- best held-out mse_deg: 1.5025
