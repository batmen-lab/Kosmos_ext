# data report: In human iPSC-derived neurons, can the transcriptional response to CRISPRi knockdown of genes not seen in training be predicted from a small labeled screen, using large single-cell profiles of iPSC-derived neurons from other experiments, with their labels ignored, as supplementary evidence?

- backend: `perturbation_response`
- data source: `plan`
- gold: `Perturb_middle/output_ipsc_neuron/gold.parquet`
- requested perturbation type: `crispri` (from the question)
- provided perturbation type: `unknown`
- type verdict: **unchecked** - perturbation modality was not compared: the data does not declare one
- supplementary: `Perturb_middle/output_ipsc_neuron/supp_ipsc_neuron_pooled.parquet`

## contract

- measured genes (the shared panel): 5005
- condition column: `condition`
- control labels: ctrl, control, control_ctrl, vehicle, none, non-targeting, non_targeting, nt, wt, unperturbed
- perturbations per split: train 3728, validation 1801, test 2960

## sources

| source | path | cells | used | how | edges |
|---|---|---|---|---|---|
| gold | Perturb_middle/output_ipsc_neuron/gold.parquet | 10693 | 9011 | 8,489 perturbed cells (34 perturbations) + 522 controls; 1,682 cells of 10 perturbation(s) outside the GO vocabulary dropped: ASXL3, CACNG2, CACNG7, CELF4, ELAVL3, GAP43, GRIA2, GRIN2A, GRIN2B, KCNT2 |  |
| supp_0 | Perturb_middle/output_ipsc_neuron/supp_ipsc_neuron_pooled.parquet | 74355 | 74355 | all cells (no condition column: an unperturbed source) | 12708 |

## auxiliary graph

- G_C/G_S edge overlap: 5450 shared (jaccard 0.433)
