# data report: perturbation response

- backend: `perturbation_response`
- data source: `plan`
- gold: `Perturb_middle/output_adamson_upr/gold.parquet`
- requested perturbation type: `unknown` (from the question)
- provided perturbation type: `unknown`
- type verdict: **unchecked** - perturbation modality was not compared: the question does not name one
- supplementary: `Perturb_middle/output_adamson_upr/supp_dixit_k562_wt.parquet`

## contract

- measured genes (the shared panel): 5014
- condition column: `condition`
- control labels: ctrl, control, control_ctrl, vehicle, none, non-targeting, non_targeting, nt, wt, unperturbed
- perturbations per split: train 28638, validation 3550, test 10307

## sources

| source | path | cells | used | how | edges |
|---|---|---|---|---|---|
| gold | Perturb_middle/output_adamson_upr/gold.parquet | 50440 | 5241 |  |  |
| supp_0 | Perturb_middle/output_adamson_upr/supp_dixit_k562_wt.parquet | 5409 | 5409 | all cells (no condition column: an unperturbed source) | 5062 |

## auxiliary graph

- G_C/G_S edge overlap: 5027 shared (jaccard 0.900)
