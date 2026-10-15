# Perturbation response run

- panel: 2000 gene(s), condition column `condition`
- splits (by perturbation): {'train': 12908, 'validation': 1322, 'test': 4113}
- arms: base, signed ppi, mlp base, mlp signed ppi

## Perturbation type

- requested by the question: `crisprko` (decided by words)
- provided by the data: `unknown`
- verdict: **unchecked** - perturbation modality was not compared: the data does not declare one

## Test metrics (held-out perturbations)

| arm | mse | mse_deg | pearson | specific pearson | spearman | direction | top-K overlap | epochs |
|---|---|---|---|---|---|---|---|---|
| `base` | 0.0112 | 0.1097 | 0.142 | 0.007 | 0.130 | 0.642 | 0.100 | 17 |
| `signed ppi` | 0.0109 | 0.1031 | 0.241 | 0.300 | 0.242 | 0.717 | 0.067 | 14 |
| `mlp base` | 0.0181 | 0.0978 | 0.149 | -0.059 | 0.144 | 0.750 | 0.108 | 7 |
| `mlp signed ppi` | 0.0156 | 0.0996 | 0.137 | 0.064 | 0.136 | 0.633 | 0.042 | 16 |

## Reference baselines (same metrics, same held-out perturbations)

| baseline | mse | mse_deg | pearson | specific pearson | direction | top-K |
|---|---|---|---|---|---|---|
| predict 0 | 0.0099 | 0.1154 | - | - | 0.000 | 0.017 |
| mean response (leave-one-out) | 0.0102 | 0.0994 | 0.286 | -1.000 | 0.742 | 0.017 |

`predict 0` is the no-change reference; `mean response (leave-one-out)` predicts each held-out perturbation with the average response of the others. An arm that does not beat both is not predicting the perturbation. `specific pearson` correlates each perturbation's Δ minus the mean Δ of the held-out perturbations, so it only credits what tells perturbations apart (`-` = the prediction is the same for all of them); it is also the epoch-selection metric.

## Per-perturbation signal

How much of the *observed* difference between perturbations the arm reproduces (spread of its per-perturbation profiles / spread of the observed ones). 0 means a per-gene constant.

| arm | across-perturbation std | of the data's |
|---|---|---|
| `base` | 0.0000 | 0.0% |
| `signed ppi` | 0.0264 | 35.5% |
| `mlp base` | 0.0084 | 11.3% |
| `mlp signed ppi` | 0.0072 | 9.6% |

> **No perturbation signal**: `base`, `mlp signed ppi` reproduce less than a tenth of the observed between-perturbation variation. Their prediction is a per-gene constant, so their `top_k_overlap` (and `pearson`) describe that constant rather than a ranking of the perturbation's genes -- read those rows as 'did not learn the perturbation', not as a ranking result.

## Graphs

- gold co-expression vs supplementary: 2854 shared edge(s), Jaccard 0.334 (first 4293, second 7098)

## Figures

- [`perturbation_overview.png`](figures/perturbation_overview.png)
