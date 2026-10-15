# Perturbation response run

- panel: 5014 gene(s), condition column `condition`
- splits (by perturbation): {'train': 28638, 'validation': 3550, 'test': 10307}
- arms: base, signed ppi, mlp base, mlp signed ppi

## Perturbation type

- requested by the question: `crispri` (decided by words)
- provided by the data: `unknown`
- verdict: **unchecked** - perturbation modality was not compared: the data does not declare one

## Test metrics (held-out perturbations)

| arm | mse | mse_deg | pearson | specific pearson | spearman | direction | top-K overlap | epochs |
|---|---|---|---|---|---|---|---|---|
| `base` | 0.0051 | 0.1619 | 0.466 | 0.682 | 0.460 | 0.876 | 0.271 | 20 |
| `signed ppi` | 0.0051 | 0.1802 | 0.466 | 0.685 | 0.462 | 0.871 | 0.297 | 20 |
| `mlp base` | 0.0064 | 0.2457 | 0.457 | 0.098 | 0.451 | 0.879 | 0.216 | 20 |
| `mlp signed ppi` | 0.0096 | 0.3200 | 0.091 | 0.017 | -0.002 | 0.742 | 0.142 | 20 |

## Reference baselines (same metrics, same held-out perturbations)

| baseline | mse | mse_deg | pearson | specific pearson | direction | top-K |
|---|---|---|---|---|---|---|
| predict 0 | 0.0090 | 0.3989 | - | - | 0.000 | 0.000 |
| mean response (leave-one-out) | 0.0063 | 0.2356 | 0.488 | -1.000 | 0.863 | 0.289 |

`predict 0` is the no-change reference; `mean response (leave-one-out)` predicts each held-out perturbation with the average response of the others. An arm that does not beat both is not predicting the perturbation. `specific pearson` correlates each perturbation's Δ minus the mean Δ of the held-out perturbations, so it only credits what tells perturbations apart (`-` = the prediction is the same for all of them); it is also the epoch-selection metric.

## Per-perturbation signal

How much of the *observed* difference between perturbations the arm reproduces (spread of its per-perturbation profiles / spread of the observed ones). 0 means a per-gene constant.

| arm | across-perturbation std | of the data's |
|---|---|---|
| `base` | 0.0137 | 23.4% |
| `signed ppi` | 0.0140 | 24.0% |
| `mlp base` | 0.0030 | 5.1% |
| `mlp signed ppi` | 0.0000 | 0.0% |

> **No perturbation signal**: `mlp base`, `mlp signed ppi` reproduce less than a tenth of the observed between-perturbation variation. Their prediction is a per-gene constant, so their `top_k_overlap` (and `pearson`) describe that constant rather than a ranking of the perturbation's genes -- read those rows as 'did not learn the perturbation', not as a ranking result.

## Graphs

- gold co-expression vs supplementary: 5027 shared edge(s), Jaccard 0.900 (first 5574, second 5038)

## Figures

- [`perturbation_overview.png`](figures/perturbation_overview.png)
