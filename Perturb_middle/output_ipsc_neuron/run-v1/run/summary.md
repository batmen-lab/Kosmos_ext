# Perturbation response run

- panel: 5006 gene(s), condition column `condition`
- splits (by perturbation): {'train': 6077, 'validation': 1221, 'test': 1044}
- arms: base, signed ppi, mlp signed ppi, perturbnet base, perturbnet supp

## Perturbation type

- requested by the question: `crispri` (decided by words)
- provided by the data: `unknown`
- verdict: **unchecked** - perturbation modality was not compared: the data does not declare one

## Test metrics (held-out perturbations)

| arm | mse | mse_deg | pearson | specific pearson | spearman | direction | top-K overlap | epochs |
|---|---|---|---|---|---|---|---|---|
| `base` | 0.0165 | 0.2007 | 0.225 | 0.016 | 0.223 | 0.694 | 0.031 | 20 |
| `signed ppi` | 0.0165 | 0.1993 | 0.231 | -0.034 | 0.227 | 0.656 | 0.050 | 20 |
| `mlp signed ppi` | 0.0174 | 0.1986 | 0.197 | -0.059 | 0.193 | 0.694 | 0.025 | 20 |
| `perturbnet base` | 0.0283 | 0.1977 | 0.154 | 0.029 | 0.149 | 0.644 | 0.044 | 45 |
| `perturbnet supp` | 0.0330 | 0.2150 | 0.149 | 0.010 | 0.157 | 0.625 | 0.062 | 45 |

## Reference baselines (same metrics, same held-out perturbations)

| baseline | mse | mse_deg | pearson | specific pearson | direction | top-K |
|---|---|---|---|---|---|---|
| predict 0 | 0.0167 | 0.2164 | - | - | 0.000 | 0.000 |
| mean response (leave-one-out) | 0.0185 | 0.2040 | 0.122 | -1.000 | 0.619 | 0.038 |

`predict 0` is the no-change reference; `mean response (leave-one-out)` predicts each held-out perturbation with the average response of the others. An arm that does not beat both is not predicting the perturbation. `specific pearson` correlates each perturbation's Δ minus the mean Δ of the held-out perturbations, so it only credits what tells perturbations apart (`-` = the prediction is the same for all of them). The best epoch is chosen by validation `mse_deg` (GEARS' rule) unless `KOSMOS_PERTURBATION_SELECTION` says otherwise.

## Per-perturbation signal

How much of the *observed* difference between perturbations the arm reproduces (spread of its per-perturbation profiles / spread of the observed ones). 0 means a per-gene constant.

| arm | across-perturbation std | of the data's |
|---|---|---|
| `base` | 0.0003 | 0.3% |
| `signed ppi` | 0.0004 | 0.4% |
| `mlp signed ppi` | 0.0114 | 10.6% |
| `perturbnet base` | 0.0945 | 87.8% |
| `perturbnet supp` | 0.1067 | 99.2% |

> **No perturbation signal**: `base`, `signed ppi` reproduce less than a tenth of the observed between-perturbation variation. Their prediction is a per-gene constant, so their `top_k_overlap` (and `pearson`) describe that constant rather than a ranking of the perturbation's genes -- read those rows as 'did not learn the perturbation', not as a ranking result.

## Graphs

- gold co-expression vs supplementary: 5525 shared edge(s), Jaccard 0.425 (first 7147, second 11367)

## Figures

- [`perturbation_overview.png`](figures/perturbation_overview.png)
