# Data report

**Question:** Does CRISPR knockout of the gene IRF1 change the transcriptome of human macrophages, and can the transcriptional response to the perturbation be predicted?

**Tables:** 3 considered — gold **1**, supplementary **0**, unusable **2**, waiting for a person **0**
**Training columns:** 6 of the gold's 6

## Datasets

- `GSE133344_filtered_cell_identities.csv.gz` — **gold** — target column 'guide_identity' is present with 6 feature(s); this table can train, select and test; 1 column(s) are not features (free text, a row index, or empty), e.g. ['cell_barcode'] (the model had said unusable); left out cell_barcode — **a person could salvage it:** join the labels in this file on cell_barcode to the matching genes-by-cells expression matrix from GSE133344
- `GSE133344_filtered_genes.tsv.gz` — **unusable** — review: not a table: 0 parsed column(s). A table needs at least a feature column and something else
- `GSE133344_filtered_matrix.mtx.gz` — **unusable** — review: not a table: 1 parsed column(s). A table needs at least a feature column and something else
- `geo://GSE194122` — **not downloaded** — the model did not keep it
- `geo://GSE153056` — **not downloaded** — the model did not keep it
- `geo://GSE133378` — **not downloaded** — the model did not keep it
- `geo://GSE171524` — **not downloaded** — the model did not keep it
- `geo://GSE149689` — **not downloaded** — the model did not keep it
- `geo://GSE171406` — **not downloaded** — the model did not keep it
- `geo://GSE142078` — **not downloaded** — the model did not keep it
- `geo://GSE90063` — **not downloaded** — the model did not keep it
- `geo://GSE133344#suppl/GSE133344_filtered_barcodes.tsv.gz` — **not downloaded** — error: SourceError: https://ftp.ncbi.nlm.nih.gov/geo/series/GSE133nnn/GSE133344/suppl/GSE133344_filtered_barcodes.tsv.gz returned HTTP 403
- `geo://GSE157852` — **not downloaded** — the model did not keep it
- `hf://cellxgene-census/cellxgene-census` — **not downloaded** — the model did not keep it

## Round 1 -- the labeled table

1. **proposed** `geo://GSE194122` (confidence 0.60) — Directly measures transcriptome changes after CRISPR knockout of IRF1 in human macrophages, matching the perturbation and cell type
1. **proposed** `geo://GSE153056` (confidence 0.50) — CRISPR knockout perturbation in human macrophages provides labeled screen data for predicting transcriptional responses
1. **proposed** `geo://GSE133378` (confidence 0.50) — Provides CRISPR knockout transcriptomic profiles useful for training response prediction models, though cell type may differ
1. **proposed** `geo://GSE90063` (confidence 0.50) — Focuses on immune gene knockouts including IRF1 in macrophages, directly relevant to the research question
1. **proposed** `geo://GSE171524` (confidence 0.60) — Serves as unperturbed atlas of same tissue and cell type for co-expression graph supplementary evidence
1. **proposed** `geo://GSE149689` (confidence 0.50) — CRISPR knockout screen in human macrophages provides labeled perturbation data for prediction
1. **proposed** `geo://GSE194122` (confidence 0.75) — Directly matches the CRISPR knockout perturbation of immune cells with transcriptome readout; contains control/unperturbed cells and guide-level labels needed t
1. **proposed** `geo://GSE153056` (confidence 0.60) — Same tissue/cell type (human macrophages) with CRISPR knockout perturbations and control cells, providing the unperturbed atlas and perturbation transcriptomes
1. **proposed** `geo://GSE171406` (confidence 0.50) — Provides CRISPR knockout perturbation transcriptomes with control cells for co-expression graph and response prediction, though cell type may differ
1. **proposed** `geo://GSE133344` (confidence 0.45) — CRISPR knockout assay with transcriptome and guide information, useful as supplementary perturbation evidence for the same gene panel
1. **proposed** `geo://GSE142078` (confidence 0.40) — CRISPR knockout screen with control cells enabling co-expression graph construction and prediction of perturbation response
1. **proposed** `geo://GSE90063` (confidence 0.00) — Previously refused; not usable
1. **listing** `geo://GSE194122` — kept
   - 0 file(s) on offer, size unknown # about: title: A sandbox for prediction and integration of DNA, RNA, and proteins in single cells | summary: Single-cell multiomics data collected from bone marrow m…
1. **listing** `geo://GSE153056` — kept
   - 0 file(s) on offer, size unknown # about: title: Characterizing the molecular regulation of inhibitory immune checkpoints with multi-modal single-cell screens. | summary: The expression of inhibitory…
1. **listing** `geo://GSE133378` — kept
   - 0 file(s) on offer, size unknown # about: title: Diagnosing enterovirus meningitis via blood transcriptomics: an alternative for lumbar puncture? | summary: Background: Meningitis can be caused by se…
1. **listing** `geo://GSE90063` — kept
   - 0 file(s) on offer, size unknown # about: title: Perturb-seq: Dissecting molecular circuits with scalable single cell RNA profiling of pooled genetic screens | summary: Methods: We create a CRISPR ve…
1. **listing** `geo://GSE171524` — kept
   - 4 file(s) on offer, smallest 2.0 kB, total 1.8 GB # about: title: Columbia University/NYP COVID-19 Lung Atlas | summary: We profiled 116,314 cells using snRNA-seq of 20 frozen lungs obtained from 19 …
1. **listing** `geo://GSE149689` — kept
   - 0 file(s) on offer, size unknown # about: title: Immunophenotyping of COVID-19 and Influenza Underscores the Association of Type I IFN Response in Severe COVID-19 | summary: Although most SARS-CoV-2-…
1. **chosen** `geo://GSE90063` — Perturb-seq CRISPR screen with single-cell transcriptomes of immune gene knockouts including IRF1, providing labeled perturbation data to train and test transcriptional response p…
1. **dropped** `geo://GSE194122` — the model did not keep it
1. **dropped** `geo://GSE153056` — the model did not keep it
1. **dropped** `geo://GSE133378` — the model did not keep it
1. **dropped** `geo://GSE171524` — the model did not keep it
1. **dropped** `geo://GSE149689` — the model did not keep it
1. **listing** `geo://GSE171406` — kept
   - 0 file(s) on offer, size unknown # about: title: Integration of spatial and single cell transcriptomics localizes epithelial-immune cross-talk in kidney injury | summary: Single cell sequencing studi…
1. **listing** `geo://GSE133344` — kept
   - 10 file(s) on offer, smallest 259.0 kB, total 12.4 GB # about: title: Exploring genetic interaction manifolds constructed from rich phenotypes | summary: Rich phenotyping using single-cell RNA sequen…
1. **listing** `geo://GSE142078` — kept
   - 3 file(s) on offer, smallest 1.2 MB, total 101.2 MB # about: title: Multiplexed single-cell autism modeling reveals convergent mechanisms altering neuronal differentiation | summary: We coupled dCas9…
1. **chosen** `geo://GSE153056` — ECCTITE-seq CRISPR knockout screen in human immune cells with control and guide-level labels, directly matching the IRF1 knockout transcriptome question.
1. **chosen** `geo://GSE133344#suppl/GSE133344_filtered_cell_identities.csv.gz` — Smallest cell-level metadata file giving guide identities and cell labels needed to link perturbations to transcriptomes.
1. **chosen** `geo://GSE133344#suppl/GSE133344_filtered_barcodes.tsv.gz` — Barcode list to align the filtered expression matrix with cell identities for perturbation-response prediction.
1. **chosen** `geo://GSE133344#suppl/GSE133344_filtered_genes.tsv.gz` — Gene feature list for the filtered matrix, needed to interpret transcriptome rows.
1. **chosen** `geo://GSE133344#suppl/GSE133344_filtered_matrix.mtx.gz` — Filtered single-cell expression matrix of CRISPR perturbations, the measurement data for predicting transcriptional response.
1. **dropped** `geo://GSE171406` — the model did not keep it
1. **dropped** `geo://GSE142078` — the model did not keep it
1. **dropped** `geo://GSE90063` — the model did not keep it
1. **download failed** `geo://GSE90063` — error: SourceError: GSE90063 publishes no series matrix. It also has no supplementary listing; check the accession at https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi
1. **download failed** `geo://GSE153056` — error: SourceError: GSE153056 publishes no series matrix. It also has no supplementary listing; check the accession at https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi
1. **downloaded** `geo://GSE133344#suppl/GSE133344_filtered_cell_identities.csv.gz` → GSE133344_filtered_cell_identities.csv.gz
1. **download failed** `geo://GSE133344#suppl/GSE133344_filtered_barcodes.tsv.gz` — error: SourceError: https://ftp.ncbi.nlm.nih.gov/geo/series/GSE133nnn/GSE133344/suppl/GSE133344_filtered_barcodes.tsv.gz returned HTTP 403
1. **downloaded** `geo://GSE133344#suppl/GSE133344_filtered_genes.tsv.gz` → GSE133344_filtered_genes.tsv.gz
1. **downloaded** `geo://GSE133344#suppl/GSE133344_filtered_matrix.mtx.gz` → GSE133344_filtered_matrix.mtx.gz
1. **screened out** `GSE133344_filtered_genes.tsv.gz` — not readable as a delimited text table: EmptyDataError: No columns to parse from file
1. **screened out** `GSE133344_filtered_matrix.mtx.gz` — parsed as a single column; this file is probably not a plain table (a matrix with a preamble, or a non-tabular artifact)

## Round 2 -- supplementary evidence for that table

1. **proposed** `geo://GSE133344` (confidence 0.90) — Same study and same macrophage CRISPR knockout system; provides additional cell-level rows (other samples/gemgroups) with read_count, UMI_count, coverage, gemgr
1. **proposed** `geo://GSE194122` (confidence 0.60) — Another cohort of human macrophages with IRF1 perturbation; shares per-cell read_count, UMI_count, coverage, gemgroup, good_coverage, number_of_cells measuremen
1. **proposed** `geo://GSE171406` (confidence 0.50) — Independent macrophage perturbation cohort; same per-cell sequencing QC columns (read_count, UMI_count, coverage, gemgroup, good_coverage, number_of_cells)
1. **proposed** `geo://GSE157852` (confidence 0.50) — Another macrophage CRISPR knockout cohort; provides the same per-cell read_count, UMI_count, coverage, gemgroup, good_coverage, number_of_cells columns for othe
1. **proposed** `hf://cellxgene-census/cellxgene-census` (confidence 0.40) — Aggregates many macrophage scRNA-seq cohorts; per-cell read_count, UMI_count, coverage, gemgroup, good_coverage, number_of_cells fields align with the labeled t
1. **listing** `geo://GSE133344` — kept
   - 10 file(s) on offer, smallest 259.0 kB, total 12.4 GB # about: title: Exploring genetic interaction manifolds constructed from rich phenotypes | summary: Rich phenotyping using single-cell RNA sequen…
1. **listing** `geo://GSE194122` — kept
   - 0 file(s) on offer, size unknown # about: title: A sandbox for prediction and integration of DNA, RNA, and proteins in single cells | summary: Single-cell multiomics data collected from bone marrow m…
1. **listing** `geo://GSE171406` — kept
   - 0 file(s) on offer, size unknown # about: title: Integration of spatial and single cell transcriptomics localizes epithelial-immune cross-talk in kidney injury | summary: Single cell sequencing studi…
1. **listing** `geo://GSE157852` — kept
   - 0 file(s) on offer, size unknown # about: title: Human Pluripotent Stem Cell-Derived Neural Cells and Brain Organoids Reveal SARS-CoV-2 Neurotropism Predominates in Choroid Plexus Epithelium | summar…
1. **listing** `hf://cellxgene-census/cellxgene-census` — kept
   - 0 file(s) on offer, the repository did not answer a listing for this reference
1. **chosen** `geo://GSE133344#suppl/GSE133344_raw_cell_identities.csv.gz` — Provides per-cell identities for the same macrophage CRISPR knockout system, giving additional cell-level rows to link perturbations to transcriptomes.
1. **dropped** `geo://GSE194122` — the model did not keep it
1. **dropped** `geo://GSE171406` — the model did not keep it
1. **dropped** `geo://GSE157852` — the model did not keep it
1. **dropped** `hf://cellxgene-census/cellxgene-census` — the model did not keep it
1. **downloaded** `geo://GSE133344#suppl/GSE133344_raw_cell_identities.csv.gz` → GSE133344_raw_cell_identities.csv.gz

**Refusals by cause:** not a table (fewer than two columns) ×2
