# Perturbation response run

- panel: 5014 gene(s), condition column `condition`
- splits (by perturbation): {'train': 24341, 'validation': 7847, 'test': 10307}
- arms: base, perturbnet base, perturbnet supp

## Perturbation type

- requested by the question: `crispri` (decided by words)
- provided by the data: `unknown`
- verdict: **unchecked** - perturbation modality was not compared: the data does not declare one

## Test metrics (held-out perturbations)

| arm | mse | mse_deg | pearson | specific pearson | spearman | direction | top-K overlap | epochs |
|---|---|---|---|---|---|---|---|---|
| `base` | 0.0053 | 0.1898 | 0.445 | 0.537 | 0.436 | 0.800 | 0.242 | 20 |
| `perturbnet base` | 0.0058 | 0.1941 | 0.466 | 0.441 | 0.428 | 0.847 | 0.371 | 45 |
| `perturbnet supp` | 0.0057 | 0.2025 | 0.462 | 0.419 | 0.429 | 0.837 | 0.395 | 45 |

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
| `base` | 0.0188 | 32.2% |
| `perturbnet base` | 0.0399 | 68.5% |
| `perturbnet supp` | 0.0377 | 64.7% |

## Graphs

- gold co-expression vs supplementary: 5027 shared edge(s), Jaccard 0.896 (first 5598, second 5038)

## Figures

- [`perturbation_overview.png`](figures/perturbation_overview.png)
