# Perturbation response run

- panel: 5014 gene(s), condition column `condition`
- splits (by perturbation): {'train': 28638, 'validation': 3550, 'test': 10307}
- arms: base, signed ppi, mlp base, mlp signed ppi

## Perturbation type

- requested by the question: `unknown` (decided by none)
- provided by the data: `unknown`
- verdict: **unchecked** - perturbation modality was not compared: the question does not name one

## Test metrics (held-out perturbations)

| arm | mse | mse_deg | pearson | specific pearson | spearman | direction | top-K overlap | epochs |
|---|---|---|---|---|---|---|---|---|
| `base` | 0.0051 | 0.1803 | 0.479 | 0.669 | 0.476 | 0.876 | 0.300 | 20 |
| `signed ppi` | 0.0072 | 0.2591 | 0.379 | 0.340 | 0.335 | 0.868 | 0.226 | 20 |
| `mlp base` | 0.0067 | 0.2594 | 0.428 | 0.071 | 0.396 | 0.887 | 0.208 | 20 |
| `mlp signed ppi` | 0.0067 | 0.2594 | 0.428 | 0.071 | 0.396 | 0.887 | 0.208 | 20 |

## Reference baselines (same metrics, same held-out perturbations)

| baseline | mse | mse_deg | pearson | specific pearson | direction | top-K |
|---|---|---|---|---|---|---|
| predict 0 | 0.0090 | 0.3989 | - | - | 0.000 | 0.000 |
| mean response (leave-one-out) | 0.0063 | 0.2356 | 0.488 | -1.000 | 0.863 | 0.289 |

`predict 0` is the no-change reference; `mean response (leave-one-out)` predicts each held-out perturbation with the average response of the others. An arm that does not beat both is not predicting the perturbation. `specific pearson` correlates each perturbation's Δ minus the mean Δ of the held-out perturbations, so it only credits what tells perturbations apart (`-` = the prediction is the same for all of them). The best epoch is chosen by validation `mse_deg` (GEARS' rule) unless `KOSMOS_PERTURBATION_SELECTION` says otherwise.

## Per-perturbation signal

How much of the *observed* difference between perturbations the arm reproduces (spread of its per-perturbation profiles / spread of the observed ones). 0 means a per-gene constant.

| arm | across-perturbation std | of the data's |
|---|---|---|
| `base` | 0.0135 | 23.2% |
| `signed ppi` | 0.0072 | 12.4% |
| `mlp base` | 0.0028 | 4.8% |
| `mlp signed ppi` | 0.0028 | 4.8% |

> **No perturbation signal**: `mlp base`, `mlp signed ppi` reproduce less than a tenth of the observed between-perturbation variation. Their prediction is a per-gene constant, so their `top_k_overlap` (and `pearson`) describe that constant rather than a ranking of the perturbation's genes -- read those rows as 'did not learn the perturbation', not as a ranking result.

## Graphs

- gold co-expression vs supplementary: 5027 shared edge(s), Jaccard 0.900 (first 5574, second 5038)

## Figures

- [`perturbation_overview.png`](figures/perturbation_overview.png)
