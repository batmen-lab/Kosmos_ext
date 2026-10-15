# Perturbation response run

- panel: 2000 gene(s), condition column `condition`
- splits (by perturbation): {'train': 12908, 'validation': 1322, 'test': 4113}
- arms: base, signed ppi

## Perturbation type

- requested by the question: `crisprko` (decided by words)
- provided by the data: `unknown`
- verdict: **unchecked** - perturbation modality was not compared: the data does not declare one

## Test metrics (held-out perturbations)

| arm | mse | mse_deg | pearson | spearman | direction | top-K overlap | epochs |
|---|---|---|---|---|---|---|---|
| `base` | 0.0397 | 0.1204 | 0.046 | 0.013 | 0.692 | 0.067 | 12 |
| `signed ppi` | 0.0657 | 0.1532 | 0.024 | -0.004 | 0.658 | 0.075 | 8 |

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
| `base` | 0.0131 | 17.7% |
| `signed ppi` | 0.0157 | 21.1% |

## Graphs

- gold co-expression vs supplementary: 2854 shared edge(s), Jaccard 0.334 (first 4293, second 7098)

## Figures

- [`perturbation_overview.png`](figures/perturbation_overview.png)
