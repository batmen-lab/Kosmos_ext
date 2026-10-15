# data report: In Jurkat T cells, can the transcriptional response to CRISPR knockout of T cell receptor signaling genes not seen in training be predicted from a small labeled screen, using large single-cell profiles of Jurkat cells from other CRISPR screens, with their labels ignored, as supplementary evidence?

- backend: `perturbation_response`
- data source: `plan`
- gold: `Perturb_middle/output_jurkat_tcr/gold.parquet`
- requested perturbation type: `crisprko` (from the question)
- provided perturbation type: `unknown`
- type verdict: **unchecked** - perturbation modality was not compared: the data does not declare one
- supplementary: `Perturb_middle/output_jurkat_tcr/supp_jurkat_pooled.parquet`

## contract

- measured genes (the shared panel): 5013
- condition column: `condition`
- control labels: ctrl, control, control_ctrl, vehicle, none, non-targeting, non_targeting, nt, wt, unperturbed
- perturbations per split: train 1319, validation 246, test 641

## sources

| source | path | cells | used | how | edges |
|---|---|---|---|---|---|
| gold | Perturb_middle/output_jurkat_tcr/gold.parquet | 3204 | 705 |  |  |
| supp_0 | Perturb_middle/output_jurkat_tcr/supp_jurkat_pooled.parquet | 72837 | 72837 | all cells (no condition column: an unperturbed source) | 11552 |

## auxiliary graph

- G_C/G_S edge overlap: 5193 shared (jaccard 0.480)
