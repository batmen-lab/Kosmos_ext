# Perturbation response run

- panel: 5005 gene(s), condition column `condition`
- splits (by perturbation): {'train': 3728, 'validation': 1801, 'test': 2960}
- arms: base, signed ppi, gated, mlp base, mlp signed ppi, mlp gated, perturbnet base, perturbnet supp

## Perturbation type

- requested by the question: `crispri` (decided by words)
- provided by the data: `unknown`
- verdict: **unchecked** - perturbation modality was not compared: the data does not declare one

## Test metrics (held-out perturbations)

| arm | mse | mse_deg | pearson | specific pearson | spearman | direction | top-K overlap | epochs |
|---|---|---|---|---|---|---|---|---|
| `base` | 0.0157 | 0.1771 | 0.140 | -0.020 | 0.149 | 0.625 | 0.031 | 20 |
| `signed ppi` | 0.0172 | 0.1675 | 0.136 | 0.018 | 0.131 | 0.644 | 0.019 | 20 |
| `gated` | 0.0162 | 0.1785 | 0.108 | -0.025 | 0.121 | 0.613 | 0.031 | 20 |
| `mlp base` | 0.0156 | 0.1692 | 0.193 | 0.001 | 0.193 | 0.706 | 0.025 | 20 |
| `mlp signed ppi` | 0.0185 | 0.1784 | 0.109 | -0.040 | 0.103 | 0.594 | 0.019 | 20 |
| `mlp gated` | 0.0148 | 0.1740 | 0.211 | 0.022 | 0.214 | 0.656 | 0.038 | 20 |
| `perturbnet base` | 0.0333 | 0.2025 | 0.133 | -0.009 | 0.141 | 0.550 | 0.050 | 45 |
| `perturbnet supp` | 0.0289 | 0.1585 | 0.170 | 0.023 | 0.184 | 0.688 | 0.019 | 45 |

## Reference baselines (same metrics, same held-out perturbations)

| baseline | mse | mse_deg | pearson | specific pearson | direction | top-K |
|---|---|---|---|---|---|---|
| predict 0 | 0.0147 | 0.1802 | - | - | 0.000 | 0.006 |
| mean response (leave-one-out) | 0.0165 | 0.1817 | 0.127 | -1.000 | 0.569 | 0.038 |

`predict 0` is the no-change reference; `mean response (leave-one-out)` predicts each held-out perturbation with the average response of the others. An arm that does not beat both is not predicting the perturbation. `specific pearson` correlates each perturbation's Δ minus the mean Δ of the held-out perturbations, so it only credits what tells perturbations apart (`-` = the prediction is the same for all of them). The best epoch is chosen by validation `mse_deg` (GEARS' rule) unless `KOSMOS_PERTURBATION_SELECTION` says otherwise.

## Per-perturbation signal

How much of the *observed* difference between perturbations the arm reproduces (spread of its per-perturbation profiles / spread of the observed ones). 0 means a per-gene constant.

| arm | across-perturbation std | of the data's |
|---|---|---|
| `base` | 0.0034 | 3.4% |
| `signed ppi` | 0.0041 | 4.1% |
| `gated` | 0.0008 | 0.8% |
| `mlp base` | 0.0038 | 3.8% |
| `mlp signed ppi` | 0.0065 | 6.5% |
| `mlp gated` | 0.0020 | 2.0% |
| `perturbnet base` | 0.1094 | 109.4% |
| `perturbnet supp` | 0.0973 | 97.3% |

> **No perturbation signal**: `base`, `signed ppi`, `gated`, `mlp base`, `mlp signed ppi`, `mlp gated` reproduce less than a tenth of the observed between-perturbation variation. Their prediction is a per-gene constant, so their `top_k_overlap` (and `pearson`) describe that constant rather than a ranking of the perturbation's genes -- read those rows as 'did not learn the perturbation', not as a ranking result.

## Gradient gate (gated arms, mean over epochs)

Each step uses `g_G + max(0, cos)·min(1, κ‖g_G‖/‖g_S‖)·g_S`; weight 0 = the supplementary gradient conflicted with the gold one and the step was gold-only.

| arm | cos(g_G, g_S) | synthetic weight | steps with weight > 0 | ‖g_G‖ | ‖g_S‖ |
|---|---|---|---|---|---|
| `gated` | +0.204 | 0.247 | 79% | 0.0727 | 0.00576 |
| `mlp gated` | +0.109 | 0.162 | 61% | 0.0459 | 0.00174 |

## Graphs

- gold co-expression vs supplementary: 5450 shared edge(s), Jaccard 0.433 (first 6661, second 11366)

## Figures

- [`perturbation_overview.png`](figures/perturbation_overview.png)
