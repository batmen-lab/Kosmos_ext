# Perturbation response run

- panel: 2000 gene(s), condition column `condition`
- splits (by perturbation): {'train': 2443, 'validation': 1197, 'test': 2286}
- arms: base, signed ppi, finetune, mlp base, mlp signed ppi, mlp finetune

## Perturbation type

- requested by the question: `crisprko` (decided by words)
- provided by the data: `unknown`
- verdict: **unchecked** - perturbation modality was not compared: the data does not declare one

## Test metrics (held-out perturbations)

| arm | mse | mse_deg | pearson | spearman | direction | top-K overlap | epochs |
|---|---|---|---|---|---|---|---|
| `base` | 0.0257 | 0.1202 | 0.022 | -0.020 | 0.850 | 0.025 | 9 |
| `signed ppi` | 0.0282 | 0.1312 | 0.050 | 0.029 | 0.825 | 0.025 | 11 |
| `finetune` | 0.0271 | 0.1187 | 0.021 | -0.021 | 0.800 | 0.025 | 6 |
| `mlp base` | 0.1339 | 0.1901 | 0.019 | 0.022 | 0.750 | 0.000 | 12 |
| `mlp signed ppi` | 0.1438 | 0.2259 | 0.014 | 0.011 | 0.550 | 0.000 | 9 |
| `mlp finetune` | 0.1679 | 0.2117 | 0.016 | 0.022 | 0.725 | 0.000 | 12 |

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
| `base` | 0.0015 | 2.8% |
| `signed ppi` | 0.0026 | 4.8% |
| `finetune` | 0.0016 | 2.9% |
| `mlp base` | 0.0025 | 4.6% |
| `mlp signed ppi` | 0.0025 | 4.5% |
| `mlp finetune` | 0.0024 | 4.4% |

> **No perturbation signal**: `base`, `signed ppi`, `finetune`, `mlp base`, `mlp signed ppi`, `mlp finetune` reproduce less than a tenth of the observed between-perturbation variation. Their prediction is a per-gene constant, so their `top_k_overlap` (and `pearson`) describe that constant rather than a ranking of the perturbation's genes -- read those rows as 'did not learn the perturbation', not as a ranking result.

## Graphs

- gold co-expression vs supplementary: 2868 shared edge(s), Jaccard 0.329 (first 4366, second 7232)

## Fine-tune

- starts from the gold-trained teacher and keeps supervising the measured gold rows *and* the cells that teacher labeled (`L_gold + L_pseudo`), with no correction term
- epochs: 6
- best held-out mse_deg: 0.0529

## Figures

- [`perturbation_overview.png`](figures/perturbation_overview.png)
