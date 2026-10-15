# Perturbation response run

- panel: 2000 gene(s), condition column `condition`
- splits (by perturbation): {'train': 12908, 'validation': 1322, 'test': 4113}
- arms: base, signed ppi, mlp base, mlp signed ppi

## Perturbation type

- requested by the question: `crisprko` (decided by words)
- provided by the data: `unknown`
- verdict: **unchecked** - perturbation modality was not compared: the data does not declare one

## Test metrics (held-out perturbations)

| arm | mse | mse_deg | pearson | spearman | direction | top-K overlap | epochs |
|---|---|---|---|---|---|---|---|
| `base` | 0.0169 | 0.1016 | 0.089 | 0.060 | 0.617 | 0.083 | 2 |
| `signed ppi` | 0.0172 | 0.1018 | 0.095 | 0.051 | 0.625 | 0.083 | 2 |
| `mlp base` | 0.2339 | 0.2739 | 0.022 | 0.016 | 0.650 | 0.025 | 2 |
| `mlp signed ppi` | 0.2339 | 0.2739 | 0.022 | 0.016 | 0.650 | 0.025 | 2 |

## Reference baselines (same metrics, same held-out perturbations)

| baseline | mse | mse_deg | pearson | direction | top-K |
|---|---|---|---|---|---|
| predict 0 | 0.0099 | 0.1154 | - | 0.000 | 0.017 |
| mean response (leave-one-out) | 0.0102 | 0.0994 | 0.286 | 0.742 | 0.017 |

`predict 0` is the no-change reference; `mean response (leave-one-out)` predicts each held-out perturbation with the average response of the others. An arm that does not beat both is not predicting the perturbation.

## Per-perturbation signal

How much of the *observed* difference between perturbations the arm reproduces (spread of its per-perturbation profiles / spread of the observed ones). 0 means a per-gene constant.

| arm | across-perturbation std | of the data's |
|---|---|---|
| `base` | 0.0183 | 24.7% |
| `signed ppi` | 0.0098 | 13.1% |
| `mlp base` | 0.0098 | 13.2% |
| `mlp signed ppi` | 0.0098 | 13.2% |

## Graphs

- gold co-expression vs supplementary: 2854 shared edge(s), Jaccard 0.334 (first 4293, second 7098)

## Figures

- [`perturbation_overview.png`](figures/perturbation_overview.png)
