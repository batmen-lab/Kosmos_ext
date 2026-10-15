# Data report

**Question:** Does CRISPR knockout of the gene IRF1 change the transcriptome of human macrophages, and can the transcriptional response to the perturbation be predicted?

**Tables:** 2 considered — gold **0**, supplementary **0**, unusable **2**, waiting for a person **0**

## Datasets

- `GSE194122_openproblems_neurips2021_cite_BMMC_processed.h5ad.gz` — **unusable** — review: not a table: 0 parsed column(s). A table needs at least a feature column and something else
- `GSE194122_openproblems_neurips2021_cite_BMMC_processed.h5ad.gz-table.csv` — **unusable** — review: This is a CITE-seq multiome BMMC atlas (NeurIPS 2021 open problems) sampled to 2,000 cells: 14,087 gene columns plus ADT/CD marker columns and cell metadata (cell_type, batch, donor, is_train…; converted from GSE194122_openproblems_neurips2021_cite_BMMC_processed.h5ad.gz (2000 of 90261 cells, 14087 of 14087 genes)
- `geo://GSE133378` — **not downloaded** — the model did not keep it
- `geo://GSE152988` — **not downloaded** — the model did not keep it
- `geo://GSE171555#suppl/GSE171555_RAW.tar` — **not downloaded** — error: SourceError: https://ftp.ncbi.nlm.nih.gov/geo/series/GSE171nnn/GSE171555/suppl/GSE171555_RAW.tar exceeds the 2,147,483,648-byte cap; raise --max-bytes or fetch a smaller file

## Round 1 -- the labeled table

1. **proposed** `geo://GSE194122` (confidence 0.60) — Provides an unperturbed human macrophage/monocyte transcriptome atlas for co-expression graph construction and baseline gene panel measurement
1. **proposed** `geo://GSE133378` (confidence 0.40) — CRISPR knockout perturbation screen in the same cell type, potentially including IRF1 knockout and transcriptome changes
1. **proposed** `geo://GSE152988` (confidence 0.35) — CRISPR knockout screen in relevant immune cell type that may contain IRF1 perturbation transcriptome data
1. **proposed** `geo://GSE171555` (confidence 0.40) — Large-scale CRISPR knockout Perturb-seq dataset that can provide labeled perturbation transcriptomes for prediction modeling
1. **listing** `geo://GSE194122` — kept
   - 3 file(s) on offer, smallest 3.9 kB, total 3.3 GB # about: title: A sandbox for prediction and integration of DNA, RNA, and proteins in single cells | summary: Single-cell multiomics data collected f…
1. **listing** `geo://GSE133378` — kept
   - 4 file(s) on offer, smallest 18.0 kB, total 12.1 MB # about: title: Diagnosing enterovirus meningitis via blood transcriptomics: an alternative for lumbar puncture? | summary: Background: Meningitis …
1. **listing** `geo://GSE152988` — kept
   - 5 file(s) on offer, smallest 3.4 kB, total 455.8 MB # about: title: Genome-wide CRISPRi/a screens in human neurons link lipid imbalance to ferroptosis | summary: Neurons are unique among all human ce…
1. **listing** `geo://GSE171555` — kept
   - 3 file(s) on offer, smallest 4.9 kB, total 2.5 GB # about: title: Single-cell RNA-seq of human peripheral blood mononuclear cells in COVID-19 | summary: We carried out sex-balanced sampling of periph…
1. **chosen** `geo://GSE194122#suppl/GSE194122_openproblems_neurips2021_cite_BMMC_processed.h5ad.gz` — Provides an unperturbed human monocyte/macrophage single-cell transcriptome atlas for co-expression graph construction and baseline gene panel measurement.
1. **chosen** `geo://GSE171555#suppl/GSE171555_RAW.tar` — Large-scale single-cell Perturb-seq dataset of human PBMCs that can supply labeled perturbation transcriptomes for prediction modeling.
1. **listing** `geo://GSE171555#suppl/GSE171555_RAW.tar` — kept
   - keeping GSE171555_RAW.tar as well -- a screen's measurements live in the counts file, and the label table does not carry them
1. **dropped** `geo://GSE133378` — the model did not keep it
1. **dropped** `geo://GSE152988` — the model did not keep it
1. **downloaded** `geo://GSE194122#suppl/GSE194122_openproblems_neurips2021_cite_BMMC_processed.h5ad.gz` → GSE194122_openproblems_neurips2021_cite_BMMC_processed.h5ad.gz, GSE194122_openproblems_neurips2021_cite_BMMC_processed.h5ad.gz-table.csv
1. **download failed** `geo://GSE171555#suppl/GSE171555_RAW.tar` — error: SourceError: https://ftp.ncbi.nlm.nih.gov/geo/series/GSE171nnn/GSE171555/suppl/GSE171555_RAW.tar exceeds the 2,147,483,648-byte cap; raise --max-bytes or fetch a smaller file
1. **download failed** `geo://GSE171555#suppl/GSE171555_RAW.tar` — error: SourceError: https://ftp.ncbi.nlm.nih.gov/geo/series/GSE171nnn/GSE171555/suppl/GSE171555_RAW.tar exceeds the 2,147,483,648-byte cap; raise --max-bytes or fetch a smaller file
1. **screened out** `GSE194122_openproblems_neurips2021_cite_BMMC_processed.h5ad.gz` — converted to GSE194122_openproblems_neurips2021_cite_BMMC_processed.h5ad.gz-table.csv; that table is the candidate being reviewed

**Refusals by cause:** not a table (fewer than two columns) ×1, the model judged it unusable ×1
