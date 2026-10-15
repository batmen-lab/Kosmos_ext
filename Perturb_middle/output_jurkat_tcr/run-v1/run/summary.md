# Perturbation response run

- panel: 5013 gene(s), condition column `condition`
- splits (by perturbation): {'train': 1319, 'validation': 246, 'test': 641}
- arms: base, signed ppi, mlp signed ppi, perturbnet base, perturbnet supp

## Perturbation type

- requested by the question: `crisprko` (decided by words)
- provided by the data: `unknown`
- verdict: **unchecked** - perturbation modality was not compared: the data does not declare one

## Test metrics (held-out perturbations)

| arm | mse | mse_deg | pearson | specific pearson | spearman | direction | top-K overlap | epochs |
|---|---|---|---|---|---|---|---|---|
| `base` | 0.0067 | 0.0658 | 0.048 | -0.009 | 0.046 | 0.575 | 0.000 | 20 |
| `signed ppi` | 0.0060 | 0.0634 | 0.042 | 0.001 | 0.035 | 0.642 | 0.008 | 20 |
| `mlp signed ppi` | 0.0081 | 0.0698 | 0.109 | 0.062 | 0.117 | 0.575 | 0.042 | 20 |
| `perturbnet base` | 0.0103 | 0.0842 | 0.084 | 0.012 | 0.081 | 0.483 | 0.000 | 45 |
| `perturbnet supp` | 0.0082 | 0.0849 | 0.073 | -0.013 | 0.069 | 0.450 | 0.033 | 45 |

## Reference baselines (same metrics, same held-out perturbations)

| baseline | mse | mse_deg | pearson | specific pearson | direction | top-K |
|---|---|---|---|---|---|---|
| predict 0 | 0.0046 | 0.0688 | - | - | 0.000 | 0.000 |
| mean response (leave-one-out) | 0.0051 | 0.0634 | 0.196 | -1.000 | 0.675 | 0.008 |

`predict 0` is the no-change reference; `mean response (leave-one-out)` predicts each held-out perturbation with the average response of the others. An arm that does not beat both is not predicting the perturbation. `specific pearson` correlates each perturbation's Δ minus the mean Δ of the held-out perturbations, so it only credits what tells perturbations apart (`-` = the prediction is the same for all of them). The best epoch is chosen by validation `mse_deg` (GEARS' rule) unless `KOSMOS_PERTURBATION_SELECTION` says otherwise.

## Per-perturbation signal

How much of the *observed* difference between perturbations the arm reproduces (spread of its per-perturbation profiles / spread of the observed ones). 0 means a per-gene constant.

| arm | across-perturbation std | of the data's |
|---|---|---|
| `base` | 0.0352 | 65.1% |
| `signed ppi` | 0.0225 | 41.7% |
| `mlp signed ppi` | 0.0117 | 21.6% |
| `perturbnet base` | 0.0492 | 91.2% |
| `perturbnet supp` | 0.0485 | 89.9% |

## Graphs

- gold co-expression vs supplementary: 5193 shared edge(s), Jaccard 0.480 (first 5523, second 10480)

## Figures

- [`perturbation_overview.png`](figures/perturbation_overview.png)
