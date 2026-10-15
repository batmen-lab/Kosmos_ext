# Data package — replogle_k562_essential

**Question:** In K562 cells, can the transcriptional response to CRISPRi knockdown of genes never seen in training be predicted from the responses to other knockdowns?

## Gold

- `data/perturbation/gears/replogle_k562_essential/replogle_k562_essential/perturb_processed.h5ad`
- K562, CRISPRi (genome-wide, filtered) (filtered release; a good *control-cell* source for a K562 gold)
- 162,751 cells × 5,000 genes (GEARS-processed, log-normalised)
- 10,691 control cells; 1,092 single perturbations, 0 combinations
- cells per perturbation: median 115, min 15
- perturbed genes: 1,092; in GEARS' GO vocabulary 1,087 (the rest cannot be encoded and are dropped); measured in the panel 412

## Supplementary

- **adamson** `data/perturbation/gears/adamson/adamson/perturb_processed.h5ad`
  - K562, CRISPRi (sgRNA); 68,603 cells (24,263 controls) × 5,060 genes
  - shares 2,026 of the gold's 5,000 genes (by gene id); stage 2 aligns it to the gold panel and fills the rest with the gold control mean

## Review

- Is the gold the screen the question is about (cell line, perturbation type)?
- Is the supplementary source the same cell line? Its cells are the
  signed PPI arm's unlabeled inputs and its co-expression graph is `G_S`.

Next: `scripts/prepare_perturbation.py --stage1 <this dir> --out <dir>` (stage 2).
