# Perturbation response run

- panel: 5014 gene(s), condition column `condition`
- splits (by perturbation): {'train': 24341, 'validation': 7847, 'test': 10307}
- arms: base, signed ppi, finetune ppi, mlp base, mlp signed ppi

## Perturbation type

- requested by the question: `crispri` (decided by words)
- provided by the data: `unknown`
- verdict: **unchecked** - perturbation modality was not compared: the data does not declare one

## Test metrics (held-out perturbations)

| arm | mse | mse_deg | pearson | specific pearson | spearman | direction | top-K overlap | epochs |
|---|---|---|---|---|---|---|---|---|
| `base` | 0.0056 | 0.1820 | 0.466 | 0.421 | 0.421 | 0.871 | 0.334 | 20 |
| `signed ppi` | 0.0052 | 0.2030 | 0.451 | 0.530 | 0.423 | 0.832 | 0.292 | 20 |
| `finetune ppi` | 0.0052 | 0.2074 | 0.474 | 0.461 | 0.450 | 0.866 | 0.324 | 12 |
| `mlp base` | 0.0074 | 0.2297 | 0.405 | 0.066 | 0.377 | 0.876 | 0.224 | 20 |
| `mlp signed ppi` | 0.0075 | 0.2283 | 0.404 | 0.069 | 0.379 | 0.876 | 0.224 | 20 |

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
| `base` | 0.0151 | 26.0% |
| `signed ppi` | 0.0216 | 37.2% |
| `finetune ppi` | 0.0182 | 31.2% |
| `mlp base` | 0.0032 | 5.5% |
| `mlp signed ppi` | 0.0032 | 5.6% |

> **No perturbation signal**: `mlp base`, `mlp signed ppi` reproduce less than a tenth of the observed between-perturbation variation. Their prediction is a per-gene constant, so their `top_k_overlap` (and `pearson`) describe that constant rather than a ranking of the perturbation's genes -- read those rows as 'did not learn the perturbation', not as a ranking result.

## Graphs

- gold co-expression vs supplementary: 5027 shared edge(s), Jaccard 0.896 (first 5598, second 5038)

## Figures

- [`perturbation_overview.png`](figures/perturbation_overview.png)
