# Perturbation response run

- panel: 2000 gene(s), condition column `condition`
- splits (by perturbation): {'train': 2443, 'validation': 1197, 'test': 2286}
- arms: base, signed ppi

## Perturbation type

- requested by the question: `crisprko` (decided by words)
- provided by the data: `unknown`
- verdict: **unchecked** - perturbation modality was not compared: the data does not declare one

## Test metrics (held-out perturbations)

| arm | mse | mse_deg | pearson | spearman | direction | top-K overlap | epochs |
|---|---|---|---|---|---|---|---|
| `base` | 0.0486 | 0.1089 | 0.016 | -0.021 | 0.800 | 0.025 | 1 |
| `signed ppi` | 0.0439 | 0.1308 | 0.027 | 0.010 | 0.700 | 0.025 | 1 |

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
| `base` | 0.0051 | 9.4% |
| `signed ppi` | 0.0006 | 1.1% |

> **No perturbation signal**: `base`, `signed ppi` reproduce less than a tenth of the observed between-perturbation variation. Their prediction is a per-gene constant, so their `top_k_overlap` (and `pearson`) describe that constant rather than a ranking of the perturbation's genes -- read those rows as 'did not learn the perturbation', not as a ranking result.

## Graphs

- gold co-expression vs supplementary: 2868 shared edge(s), Jaccard 0.329 (first 4366, second 7232)

## Figures

- [`perturbation_overview.png`](figures/perturbation_overview.png)
