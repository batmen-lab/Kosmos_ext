# data report: Does CRISPR knockout of the gene IRF1 change the transcriptome of human macrophages, and can the transcriptional response to the perturbation be predicted?

- backend: `perturbation_response`
- data source: `plan`
- gold: `Perturb_middle/output_irf1-run-mine/gold_subset.csv.gz`
- requested perturbation type: `crisprko` (from the question)
- provided perturbation type: `unknown`
- type verdict: **unchecked** - perturbation modality was not compared: the data does not declare one
- supplementary: `Perturb_middle/output_irf1-run-mine/supp_2_GSM4633605_CITE_cDNA_counts.csv.gz`
- supplementary: `Perturb_middle/output_irf1-run-mine/supp_3_GSM4633608_ECCITE_Arrayed_cDNA_counts.csv.gz`

## contract

- measured genes (the shared panel): 2000
- condition column: `condition`
- control labels: ctrl, control, control_ctrl, vehicle, none, non-targeting, non_targeting, nt, wt, unperturbed
- perturbations per split: train 12908, validation 1322, test 4113

## sources

| source | path | cells | used | how | edges |
|---|---|---|---|---|---|
| gold | Perturb_middle/output_irf1-run-mine/gold_subset.csv.gz | 20729 | 2386 |  |  |
| supp_0 | Perturb_middle/output_irf1-run-mine/supp_2_GSM4633605_CITE_cDNA_counts.csv.gz | 7566 | 7566 | all cells (no condition column: an unperturbed source) | 12196 |
| supp_1 | Perturb_middle/output_irf1-run-mine/supp_3_GSM4633608_ECCITE_Arrayed_cDNA_counts.csv.gz | 16695 | 16695 | all cells (no condition column: an unperturbed source) | 2654 |

## auxiliary graph

- G_C/G_S edge overlap: 2854 shared (jaccard 0.334)
