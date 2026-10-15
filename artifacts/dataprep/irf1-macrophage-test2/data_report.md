# Data report

**Question:** Does CRISPR knockout of the gene IRF1 change the transcriptome of human macrophages, and can the transcriptional response to the perturbation be predicted?

**Tables:** 26 considered — gold **1**, supplementary **0**, unusable **25**, waiting for a person **0**
**Training columns:** 22 of the gold's 22

## Datasets

- `GSE153056_ECCITE_metadata.tsv.gz` — **gold** — review: screen label half (condition column 'guide_ID'); its counts matrix is joined at training time; left out Unnamed: 0, NT — **a person could salvage it:** join the labels in this metadata to the cDNA counts matrix (GSM4633614_ECCITE_cDNA_counts.tsv.gz) on barcode
- `GSM4633612_ECCITE_Arrayed_GDO_Barcodes.csv.gz` — **unusable** — it cannot be evidence for the same model: it shares only 0 measured column(s) with the gold, and 11 are needed to correct the same model (the gold has 22 feature(s)) (the model had said gold) — **a person could salvage it:** join the guide labels in this file to the counts matrix GSM4633612_ECCITE_Arrayed_GO_CITE03_counts.tsv.gz on barcode
- `GSM4633616_ECCITE_HTO_1-4_Barcodes.csv.gz` — **unusable** — it cannot be evidence for the same model: it shares only 0 measured column(s) with the gold, and 11 are needed to correct the same model (the gold has 22 feature(s)) (the model had said gold) — **a person could salvage it:** join the labels in GSM4633616_ECCITE_HTO_1-4_Barcodes.csv.gz onto the barcodes of GSM4633616_ECCITE_HTO_counts.tsv.gz
- `GSM4633617_ECCITE_HTO_5-8_Barcodes.csv.gz` — **unusable** — it cannot be evidence for the same model: it shares only 0 measured column(s) with the gold, and 11 are needed to correct the same model (the gold has 22 feature(s)) (the model had said gold) — **a person could salvage it:** join these labels onto the matching cDNA counts matrix (GSM4633614_ECCITE_cDNA_counts.tsv.gz) on barcode
- `GSM4633618_ECCITE_GDO_Barcodes.csv.gz` — **unusable** — it cannot be evidence for the same model: it shares only 0 measured column(s) with the gold, and 11 are needed to correct the same model (the gold has 22 feature(s)) (the model had said gold); left out CGGTGAACAGCTCCTCGCCC, eGFPg1 — **a person could salvage it:** join these guide labels onto the barcodes of GSM4633618_ECCITE_GDO_counts.tsv.gz to build the per-cell screen
- `GSE153056_RAW.tar` — **unusable** — review: not a table: 0 parsed column(s). A table needs at least a feature column and something else
- `GSE153056_CITE_metadata.tsv.gz` — **unusable** — review: This is a per-cell CITE-seq metadata table (barcodes with RNA/ADT/HTO QC columns and hashtag classification). It has no gene-expression columns and no perturbation/condition column, so it is …; left out Unnamed: 0 — **a person could salvage it:** join this barcode table to the matching cDNA counts matrix (GSM4633605_CITE_cDNA_counts.tsv.gz) and to the ECCITE metadata that holds the perturbation labels
- `GSE153056_ECCITE_Arrayed_metadata.tsv.gz` — **unusable** — review: This is a per-cell ECCITE-seq metadata table (barcodes with HTO/lenti-guide/CITE/ADT/RNA QC columns). It carries perturbation identities (GO_lenti_maxID, GO_cite_maxID) but no transcriptome c…; left out Unnamed: 0, GO_cite_secondID — **a person could salvage it:** join the labels in the GSE153056 expression/counts matrix on the barcode column (Unnamed: 0), using GO_lenti_maxID as the guide identity
- `GSM4633605_CITE_cDNA_counts.tsv.gz` — **unusable** — review: This is a genes-by-cells count matrix (gene symbols in the first column, cell barcodes as the remaining 7566 columns). It carries the transcriptome but no perturbation/condition column, so it…; left out Unnamed: 0 — **a person could salvage it:** join the labels in data/fetched/geo/GSE153056/GSE153056_ECCITE_metadata.tsv.gz on barcode
- `GSM4633606_CITE_ADT_Barcodes.csv.gz` — **unusable** — review: This is a barcode-to-protein lookup table for CITE-seq ADT (antibody-derived tag) features, not a per-cell perturbation screen. Each row maps a cell barcode to an antibody/protein name (PDL1,… — **a person could salvage it:** join this barcode/protein annotation to GSM4633606_CITE_ADT_counts.tsv.gz as feature names; the perturbation labels and cDNA expression are in the ECCITE metadata and cDNA counts files
- `GSM4633606_CITE_ADT_counts.tsv.gz` — **unusable** — review: This is a CITE-seq antibody-derived tag (ADT) protein count matrix: rows are antibody features (BTLA, CD47, CD66a, CD86, ...) and columns are cell barcodes. It is a features-by-cells matrix w…; left out Unnamed: 0
- `GSM4633607_CITE_HTO_Barcodes.csv.gz` — **unusable** — review: This is a barcode-to-hashtag (HTO) assignment file: two columns mapping a cell barcode to an HTO tag. It has no gene/perturbation column and no transcriptome, so it is not a per-cell perturba… — **a person could salvage it:** join the labels in data/fetched/geo/GSE153056/GSE153056_CITE_metadata.tsv.gz on barcode
- `GSM4633607_CITE_HTO_counts.tsv.gz` — **unusable** — review: This is a CITE-seq HTO (hashing antibody) counts matrix: rows are hashtag oligos (HTO22, HTO23, HTO25, HTO26) and columns are cell barcodes, with no gene/transcriptome columns and no perturba…; left out Unnamed: 0 — **a person could salvage it:** join the labels in data/fetched/geo/GSE153056/GSE153056_ECCITE_metadata.tsv.gz on barcode, and use the cDNA counts matrix (GSM4633605/GSM4633614) for expression
- `GSM4633608_ECCITE_Arrayed_cDNA_counts.tsv.gz` — **unusable** — review: This is a genes-by-cells counts matrix (16,696 columns: gene rows in the first column, cell barcodes as columns). It carries the transcriptome but no per-cell perturbation/condition column, s…; left out Unnamed: 0 — **a person could salvage it:** join the labels in data/fetched/geo/GSE153056/GSE153056_ECCITE_metadata.tsv.gz on barcode (transpose so cells are rows, genes are columns)
- `GSM4633609_ECCITE_Arrayed_ADT_Barcodes.csv.gz` — **unusable** — review: This is an antibody-derived tag (ADT) barcode-to-protein mapping file: two columns, a cell barcode and a protein marker name (CD86, PDL1, PDL2, CD366). It has no header, no gene-expression co…
- `GSM4633609_ECCITE_Arrayed_ADT_counts.tsv.gz` — **unusable** — review: This is an antibody-derived tag (ADT) / cell-surface protein counts matrix from a CITE-seq/ECCITE experiment: rows are surface markers (CD86, PDL1, PDL2, CD366, ...) and columns are cell barc…; left out Unnamed: 0 — **a person could salvage it:** join the perturbation labels from GSE153056_ECCITE_metadata.tsv.gz on barcode and pair with the cDNA counts matrix GSM4633608_ECCITE_Arrayed_cDNA_counts.tsv.gz
- `GSM4633610_ECCITE_Arrayed_HTO1_Barcodes.csv.gz` — **unusable** — review: This is a barcode-to-hashtag (HTO) assignment file: two columns mapping cell barcodes to hashtag labels such as HTO_ETV7g1 and HTO_NT5. It carries no gene expression and no CRISPR guide/targe… — **a person could salvage it:** join the labels in data/fetched/geo/GSE153056/GSE153056_ECCITE_metadata.tsv.gz on barcode
- `GSM4633610_ECCITE_Arrayed_HTO_counts.tsv.gz` — **unusable** — review: This is an antibody-hashtag (HTO) count matrix from a CITE/ECCITE experiment: rows are HTO tags (HTO-IRF7g1, HTO-MARCH8g1, ...) and columns are cell barcodes (r_... and g_...). It carries nei…; left out Unnamed: 0 — **a person could salvage it:** join the labels in data/fetched/geo/GSE153056/GSE153056_ECCITE_metadata.tsv.gz on barcode and use the paired cDNA counts matrix GSM4633608_ECCITE_Arrayed_cDNA_counts.tsv.gz for expression
- `GSM4633611_ECCITE_Arrayed_HTO2_Barcodes.csv.gz` — **unusable** — review: This is a barcode-to-HTO assignment file (two columns: cell barcode and hashtag oligo identity), not a per-cell perturbation screen. It carries no gene-level perturbation identity and no tran… — **a person could salvage it:** join the HTO labels in this file onto the cDNA counts matrix (GSM4633608_ECCITE_Arrayed_cDNA_counts.tsv.gz) on barcode
- `GSM4633612_ECCITE_Arrayed_GO_CITE03_counts.tsv.gz` — **unusable** — review: This is a genes-by-cells guide/perturbation count matrix (rows are guides like eGFPg1, CUL3g1; columns are cell barcodes prefixed r_ for RNA and g_ for guides). It has no per-cell condition c…; left out Unnamed: 0 — **a person could salvage it:** join the labels in GSM4633612_ECCITE_Arrayed_GDO_Barcodes.csv.gz on barcode and pair with the cDNA counts matrix
- `GSM4633613_ECCITE_Arrayed_GO_LENTI_counts.tsv.gz` — **unusable** — review: This is a genes-by-cells count matrix (rows are guides/genes like eGFPg1, CUL3g1; columns are cell barcodes r_* and g_*). It has no per-cell condition column and no gene-expression columns in…; left out Unnamed: 0 — **a person could salvage it:** transpose it so cells are rows, then join the guide/condition labels from GSE153056_ECCITE_metadata.tsv.gz on barcode
- `GSM4633614_ECCITE_cDNA_counts.tsv.gz` — **unusable** — review: This is a genes-by-cells count matrix (20,730 columns: one gene-name column plus one column per cell barcode). Rows are genes, columns are cells, and there is no perturbation/condition column…; left out Unnamed: 0 — **a person could salvage it:** transpose it, genes are rows, then join the perturbation labels from GSE153056_ECCITE_metadata.tsv.gz on barcode
- `GSM4633615_ECCITE_ADT_Barcodes.csv.gz` — **unusable** — review: This is a barcode-to-ADT (antibody-derived tag) assignment file for the ECCITE-seq dataset: each row pairs a cell barcode with a surface-protein marker name (CD86, PDL1, PDL2, CD366). It has …
- `GSM4633615_ECCITE_ADT_counts.tsv.gz` — **unusable** — review: This is an antibody-derived tag (ADT/CITE-seq) protein-count matrix: rows are surface-marker proteins (CD86, PDL1, PDL2, CD366) and columns are cell barcodes. It carries no gene-expression me…; left out Unnamed: 0
- `GSM4633616_ECCITE_HTO_counts.tsv.gz` — **unusable** — review: This is a hashtag-oligo (HTO) antibody-derived tag count matrix: rows are HTO features (rep1-tx, rep1-ctrl, rep2-tx, rep2-ctrl, ...) and columns are cell barcodes. It carries neither gene exp…; left out Unnamed: 0 — **a person could salvage it:** use it only to demultiplex barcodes, then join the perturbation labels from GSE153056_ECCITE_metadata.tsv.gz and the expression from GSM4633614_ECCITE_cDNA_counts.tsv.gz on barcode
- `GSM4633618_ECCITE_GDO_counts.tsv.gz` — **unusable** — review: This is a genes-by-cells count matrix (GDO = guide-derived oligo counts) with genes/guides as rows and cell barcodes as columns. There is no per-cell condition column; the perturbation identi…; left out Unnamed: 0 — **a person could salvage it:** join the guide labels in GSM4633618_ECCITE_GDO_Barcodes.csv.gz onto the cDNA matrix GSM4633614_ECCITE_cDNA_counts.tsv.gz by barcode
- `geo://GSE194122` — **not downloaded** — the model did not keep it
- `hf://genome-scale/perturb-seq` — **not downloaded** — the model did not keep it
- `geo://GSE171964` — **not downloaded** — the model did not keep it
- `geo://GSE171406` — **not downloaded** — the model did not keep it
- `geo://GSE168620` — **not downloaded** — the model did not keep it
- `geo://GSE201333` — **not downloaded** — the model did not keep it
- `geo://GSE165080` — **not downloaded** — the model did not keep it
- `geo://GSE153697` — **not downloaded** — the model did not keep it
- `hf://katarinagresova/GSE153056` — **not downloaded** — the model did not keep it
- `hf://openproblems-bio/neurips2021_multimodal_viash` — **not downloaded** — the model did not keep it

## Round 1 -- the labeled table

1. **proposed** `geo://GSE194122` (confidence 0.60) — Contains human myeloid cells with CRISPR knockout perturbations and unperturbed controls, providing the same cell type and gene panel needed for co-expression a
1. **proposed** `geo://GSE133344` (confidence 0.50) — CRISPR knockout Perturb-seq dataset measuring transcriptional responses to gene deletions, matching the required assay type
1. **proposed** `geo://GSE90063` (confidence 0.40) — Provides CRISPR knockout perturbation transcriptomes that can support prediction of transcriptional responses
1. **proposed** `hf://genome-scale/perturb-seq` (confidence 0.30) — HuggingFace repository hosting perturbation transcriptomics that may include human macrophage or myeloid CRISPR knockout data
1. **proposed** `geo://GSE171964` (confidence 0.40) — Human macrophage scRNA-seq with perturbations and controls, suitable for same-cell-type co-expression evidence
1. **proposed** `geo://GSE157129` (confidence 0.40) — CRISPR knockout transcriptomic screen providing labeled perturbation data for response prediction
1. **proposed** `geo://GSE194122` (confidence 0.60) — Contains human myeloid cells with CRISPR knockout perturbations and unperturbed controls, providing the same cell type and gene panel needed for co-expression a
1. **proposed** `geo://GSE153056` (confidence 0.50) — Directly perturbs genes in human macrophages by CRISPR knockout, matching the assay and cell type required
1. **proposed** `geo://GSE171406` (confidence 0.50) — CRISPR knockout Perturb-seq in human immune cells provides labeled perturbation transcriptomes for prediction
1. **proposed** `geo://GSE168620` (confidence 0.45) — Knockout screen in human macrophages supplies the perturbation and control transcriptomes needed
1. **proposed** `geo://GSE201333` (confidence 0.50) — Unperturbed human macrophage atlas provides control cells and the same gene panel for co-expression graph evidence
1. **proposed** `geo://GSE165080` (confidence 0.45) — CRISPR knockout perturbation data in human myeloid lineage supports modeling transcriptional response
1. **listing** `geo://GSE194122` — kept
   - 3 file(s) on offer, smallest 3.9 kB, total 3.3 GB # about: title: A sandbox for prediction and integration of DNA, RNA, and proteins in single cells | summary: Single-cell multiomics data collected f…
1. **listing** `geo://GSE133344` — kept
   - 11 file(s) on offer, smallest 3.6 kB, total 12.4 GB # about: title: Exploring genetic interaction manifolds constructed from rich phenotypes | summary: Rich phenotyping using single-cell RNA sequenci…
1. **listing** `geo://GSE90063` — kept
   - 7 file(s) on offer, smallest 2.0 kB, total 1.4 GB # about: title: Perturb-seq: Dissecting molecular circuits with scalable single cell RNA profiling of pooled genetic screens | summary: Methods: We c…
1. **listing** `hf://genome-scale/perturb-seq` — kept
   - 0 file(s) on offer, the repository did not answer a listing for this reference
1. **listing** `geo://GSE171964` — kept
   - 5 file(s) on offer, smallest 3.7 kB, total 1.4 GB # about: title: CITE-seq of 6 patients treated with the Pfizer BioNTech BNT162b2 mRNA vaccine | summary: Purpose: To understand the single cell lands…
1. **listing** `geo://GSE157129` — kept
   - 2 file(s) on offer, smallest 3.0 kB, total 1.1 MB # about: title: RNA-seq profiling of RPE reprogramming | summary: The plasticity of the retinal pigment epithelium (RPE) has been observed during pro…
1. **chosen** `geo://GSE90063#suppl/GSE90063_k562_umi_wt.txt.gz` — This is the actual Perturb-seq UMI count matrix of cells with their genes from CRISPR knockout perturbations, the measurement table needed to predict transcriptional responses.
1. **chosen** `geo://GSE133344#suppl/GSE133344_filtered_matrix.mtx.gz` — This is the filtered single-cell expression matrix of CRISPR knockout Perturb-seq cells, providing labeled perturbation transcriptomes for response prediction.
1. **listing** `geo://GSE157129#suppl/GSE157129_normalizedCounts.csv.gz` — kept
   - keeping GSE157129_normalizedCounts.csv.gz as well -- a screen's measurements live in the counts file, and the label table does not carry them
1. **dropped** `geo://GSE194122` — the model did not keep it
1. **dropped** `hf://genome-scale/perturb-seq` — the model did not keep it
1. **dropped** `geo://GSE171964` — the model did not keep it
1. **listing** `geo://GSE153056` — kept
   - 5 file(s) on offer, smallest 1.7 kB, total 124.0 MB # about: title: Characterizing the molecular regulation of inhibitory immune checkpoints with multi-modal single-cell screens. | summary: The expre…
1. **listing** `geo://GSE171406` — kept
   - 2 file(s) on offer, smallest 2.8 kB, total 5.9 kB # about: title: Integration of spatial and single cell transcriptomics localizes epithelial-immune cross-talk in kidney injury | summary: Single cell…
1. **listing** `geo://GSE168620` — kept
   - 4 file(s) on offer, smallest 2.3 kB, total 3.4 GB # about: title: Ultra-high throughput single-cell RNA sequencing by combinatorial fluidic indexing | summary: Cell atlas projects and high-throughput…
1. **listing** `geo://GSE201333` — kept
   - 3 file(s) on offer, smallest 1.6 kB, total 15.0 GB # about: title: Tabula Sapiens | summary: Tabula Sapiens is a single cell transcriptomic atlas of 24 human tissues and organs. | overall_design: Var…
1. **listing** `geo://GSE165080` — kept
   - 5 file(s) on offer, smallest 6.9 kB, total 1.0 GB # about: title: Identification of distinct immune cell subsets associated with various clinical presentations, disease severity and viral persistence…
1. **chosen** `geo://GSE153056#suppl/GSE153056_ECCITE_metadata.tsv.gz` — Cell-level metadata for the ECCITE-seq CRISPR screen in human macrophages, giving the perturbation labels and cell identities needed to link IRF1 knockout to transcriptomes.
1. **chosen** `geo://GSE153056#suppl/GSE153056_RAW.tar` — Raw count matrices for the ECCITE-seq screen provide the actual transcriptome measurements per cell for IRF1 knockout and control macrophages.
1. **listing** `geo://GSE153056#suppl/GSE153056_RAW.tar` — kept
   - keeping GSE153056_RAW.tar as well -- a screen's measurements live in the counts file, and the label table does not carry them
1. **listing** `geo://GSE153056#suppl/GSE153056_CITE_metadata.tsv.gz` — kept
   - keeping GSE153056_CITE_metadata.tsv.gz as well -- this task's labels (a perturbation screen's condition/guide) usually live in the annotated file, not the raw matrix
1. **dropped** `geo://GSE171406` — the model did not keep it
1. **dropped** `geo://GSE168620` — the model did not keep it
1. **dropped** `geo://GSE201333` — the model did not keep it
1. **dropped** `geo://GSE165080` — the model did not keep it
1. **downloaded** `geo://GSE90063#suppl/GSE90063_k562_umi_wt.txt.gz` → GSE90063_k562_umi_wt.txt.gz
1. **downloaded** `geo://GSE157129#suppl/GSE157129_normalizedCounts.csv.gz` → GSE157129_normalizedCounts.csv.gz
1. **downloaded** `geo://GSE133344#suppl/GSE133344_filtered_matrix.mtx.gz` → GSE133344_filtered_matrix.mtx.gz
1. **downloaded** `geo://GSE157129#suppl/GSE157129_normalizedCounts.csv.gz` → GSE157129_normalizedCounts.csv.gz
1. **downloaded** `geo://GSE153056#suppl/GSE153056_ECCITE_metadata.tsv.gz` → GSE153056_ECCITE_metadata.tsv.gz
1. **downloaded** `geo://GSE153056#suppl/GSE153056_RAW.tar` → GSE153056_RAW.tar, GSE153056_RAW.tar-contents/GSM4633605_CITE_cDNA_counts.tsv.gz, GSE153056_RAW.tar-contents/GSM4633606_CITE_ADT_Barcodes.csv.gz, GSE153056_RAW.tar-contents/GSM4633606_CITE_ADT_counts.tsv.gz
1. **downloaded** `geo://GSE153056#suppl/GSE153056_CITE_metadata.tsv.gz` → GSE153056_CITE_metadata.tsv.gz
1. **downloaded** `geo://GSE153056#suppl/GSE153056_RAW.tar` → GSE153056_RAW.tar, GSE153056_RAW.tar-contents/GSM4633605_CITE_cDNA_counts.tsv.gz, GSE153056_RAW.tar-contents/GSM4633606_CITE_ADT_Barcodes.csv.gz, GSE153056_RAW.tar-contents/GSM4633606_CITE_ADT_counts.tsv.gz
1. **downloaded** `geo://GSE153056#suppl/GSE153056_CITE_metadata.tsv.gz` → GSE153056_CITE_metadata.tsv.gz
1. **screened out** `GSE133344_filtered_matrix.mtx.gz` — parsed as a single column; this file is probably not a plain table (a matrix with a preamble, or a non-tabular artifact)
1. **screened out** `GSE153056_RAW.tar` — not readable as a delimited text table: UnicodeDecodeError: 'utf-8' codec can't decode byte 0x8b in position 513: invalid start byte

## Round 2 -- supplementary evidence for that table

1. **proposed** `geo://GSE153056` (confidence 0.40) — Parent series of the labeled table; contains additional samples/cohorts with the same ECCITE-seq measurements (nCount_RNA, nFeature_RNA, guide_ID, crispr, HTO/A
1. **proposed** `geo://GSE153697` (confidence 0.20) — Related macrophage CRISPR perturbation study with matching RNA/ADT/HTO count columns and guide/crispr labels
1. **proposed** `hf://katarinagresova/GSE153056` (confidence 0.20) — HuggingFace mirror of the same study providing the same measured columns (nCount_RNA, nFeature_RNA, guide_ID, crispr, HTO/ADT/GDO counts) for additional rows
1. **proposed** `geo://GSE194122` (confidence 0.60) — Contains human myeloid cells with CRISPR knockout perturbations and unperturbed controls, providing the same cell type and gene panel needed for co-expression a
1. **proposed** `geo://GSE153056` (confidence 0.50) — Directly perturbs genes in human macrophages by CRISPR knockout, matching the assay and cell type required
1. **proposed** `geo://GSE171406` (confidence 0.50) — CRISPR knockout Perturb-seq in human immune cells provides labeled perturbation transcriptomes for prediction
1. **proposed** `geo://GSE168620` (confidence 0.45) — Knockout screen in human macrophages supplies the perturbation and control transcriptomes needed
1. **proposed** `geo://GSE201333` (confidence 0.50) — Unperturbed human macrophage atlas provides control cells and the same gene panel for co-expression graph evidence
1. **proposed** `geo://GSE165080` (confidence 0.45) — CRISPR knockout perturbation data in human myeloid lineage supports modeling transcriptional response
1. **proposed** `geo://GSE153056` (confidence 0.60) — Same study series; contains additional supplementary files (e.g., other sample-level count matrices) with the same gene-expression columns (7SK, A1BG, A2M, etc.
1. **proposed** `geo://GSE194122` (confidence 0.30) — Provides gene-expression count matrices with the same gene-symbol columns (7SK, A1BG, A2M, ...) for other samples/cohorts, usable as supplementary expression me
1. **proposed** `hf://openproblems-bio/neurips2021_multimodal_viash` (confidence 0.25) — Contains gene-expression count matrices with matching gene-symbol columns for other samples/splits, supplying the same measurement type for additional rows
1. **listing** `geo://GSE153056` — kept
   - 7 file(s) on offer, smallest 1.7 kB, total 124.0 MB # about: title: Characterizing the molecular regulation of inhibitory immune checkpoints with multi-modal single-cell screens. | summary: The expre…
1. **listing** `geo://GSE153697` — kept
   - 6 file(s) on offer, smallest 2.7 kB, total 110.8 MB # about: title: Single-cell profiling identifies pre-existing CD19-negative subclones in a B-ALL patient with CD19-negative relapse after CAR-T the…
1. **listing** `hf://katarinagresova/GSE153056` — kept
   - 0 file(s) on offer, the repository did not answer a listing for this reference
1. **chosen** `geo://GSE153056#suppl/GSE153056_ECCITE_Arrayed_metadata.tsv.gz` — Arrayed ECCITE-seq metadata with per-cell guide_ID/crispr labels and RNA counts for the IRF1 knockout macrophage perturbation, providing the observations needed to link perturbati…
1. **listing** `geo://GSE153056#suppl/GSE153056_RAW.tar` — kept
   - keeping GSE153056_RAW.tar as well -- a screen's measurements live in the counts file, and the label table does not carry them
1. **listing** `geo://GSE153056#suppl/GSE153056_CITE_metadata.tsv.gz` — kept
   - keeping GSE153056_CITE_metadata.tsv.gz as well -- this task's labels (a perturbation screen's condition/guide) usually live in the annotated file, not the raw matrix
1. **dropped** `geo://GSE153697` — the model did not keep it
1. **dropped** `hf://katarinagresova/GSE153056` — the model did not keep it
1. **listing** `geo://GSE194122` — kept
   - 3 file(s) on offer, smallest 3.9 kB, total 3.3 GB # about: title: A sandbox for prediction and integration of DNA, RNA, and proteins in single cells | summary: Single-cell multiomics data collected f…
1. **listing** `geo://GSE171406` — kept
   - 8 file(s) on offer, smallest 2.8 kB, total 721.0 MB # about: title: Integration of spatial and single cell transcriptomics localizes epithelial-immune cross-talk in kidney injury | summary: Single ce…
1. **listing** `geo://GSE168620` — kept
   - 4 file(s) on offer, smallest 2.3 kB, total 3.4 GB # about: title: Ultra-high throughput single-cell RNA sequencing by combinatorial fluidic indexing | summary: Cell atlas projects and high-throughput…
1. **listing** `geo://GSE201333` — kept
   - 3 file(s) on offer, smallest 1.6 kB, total 15.0 GB # about: title: Tabula Sapiens | summary: Tabula Sapiens is a single cell transcriptomic atlas of 24 human tissues and organs. | overall_design: Var…
1. **listing** `geo://GSE165080` — kept
   - 5 file(s) on offer, smallest 6.9 kB, total 1.0 GB # about: title: Identification of distinct immune cell subsets associated with various clinical presentations, disease severity and viral persistence…
1. **chosen** `geo://GSE153056#suppl/GSE153056_ECCITE_metadata.tsv.gz` — Metadata table of cells with their CRISPR perturbations and cell types from the ECCITE-seq screen, providing the labeled perturbation transcriptome observations needed to test IRF…
1. **chosen** `geo://GSE153056#suppl/GSE153056_RAW.tar` — Raw count matrices for the ECCITE-seq CRISPR screen in human immune cells, giving the actual gene expression measurements per perturbed cell for transcriptome analysis and predict…
1. **dropped** `geo://GSE194122` — the model did not keep it
1. **dropped** `geo://GSE171406` — the model did not keep it
1. **dropped** `geo://GSE168620` — the model did not keep it
1. **dropped** `geo://GSE201333` — the model did not keep it
1. **dropped** `geo://GSE165080` — the model did not keep it
1. **listing** `hf://openproblems-bio/neurips2021_multimodal_viash` — kept
   - 0 file(s) on offer, the repository did not answer a listing for this reference
1. **dropped** `hf://openproblems-bio/neurips2021_multimodal_viash` — the model did not keep it
1. **downloaded** `geo://GSE153056#suppl/GSE153056_ECCITE_Arrayed_metadata.tsv.gz` → GSE153056_ECCITE_Arrayed_metadata.tsv.gz
1. **downloaded** `geo://GSE153056#suppl/GSE153056_RAW.tar` → GSE153056_RAW.tar, GSE153056_RAW.tar-contents/GSM4633605_CITE_cDNA_counts.tsv.gz, GSE153056_RAW.tar-contents/GSM4633606_CITE_ADT_Barcodes.csv.gz, GSE153056_RAW.tar-contents/GSM4633606_CITE_ADT_counts.tsv.gz
1. **downloaded** `geo://GSE153056#suppl/GSE153056_CITE_metadata.tsv.gz` → GSE153056_CITE_metadata.tsv.gz
1. **downloaded** `geo://GSE153056#suppl/GSE153056_ECCITE_metadata.tsv.gz` → GSE153056_ECCITE_metadata.tsv.gz
1. **downloaded** `geo://GSE153056#suppl/GSE153056_RAW.tar` → GSE153056_RAW.tar, GSE153056_RAW.tar-contents/GSM4633605_CITE_cDNA_counts.tsv.gz, GSE153056_RAW.tar-contents/GSM4633606_CITE_ADT_Barcodes.csv.gz, GSE153056_RAW.tar-contents/GSM4633606_CITE_ADT_counts.tsv.gz
1. **downloaded** `geo://GSE153056#suppl/GSE153056_CITE_metadata.tsv.gz` → GSE153056_CITE_metadata.tsv.gz
1. **downloaded** `geo://GSE153056#suppl/GSE153056_RAW.tar` → GSE153056_RAW.tar, GSE153056_RAW.tar-contents/GSM4633605_CITE_cDNA_counts.tsv.gz, GSE153056_RAW.tar-contents/GSM4633606_CITE_ADT_Barcodes.csv.gz, GSE153056_RAW.tar-contents/GSM4633606_CITE_ADT_counts.tsv.gz
1. **downloaded** `geo://GSE153056#suppl/GSE153056_CITE_metadata.tsv.gz` → GSE153056_CITE_metadata.tsv.gz
1. **downloaded** `geo://GSE153056#suppl/GSE153056_ECCITE_Arrayed_metadata.tsv.gz` → GSE153056_ECCITE_Arrayed_metadata.tsv.gz
1. **downloaded** `geo://GSE153056#suppl/GSE153056_RAW.tar` → GSE153056_RAW.tar, GSE153056_RAW.tar-contents/GSM4633605_CITE_cDNA_counts.tsv.gz, GSE153056_RAW.tar-contents/GSM4633606_CITE_ADT_Barcodes.csv.gz, GSE153056_RAW.tar-contents/GSM4633606_CITE_ADT_counts.tsv.gz
1. **downloaded** `geo://GSE153056#suppl/GSE153056_CITE_metadata.tsv.gz` → GSE153056_CITE_metadata.tsv.gz
1. **screening** kept all 1 table(s)
1. **screened out** `GSE153056_RAW.tar` — not readable as a delimited text table: UnicodeDecodeError: 'utf-8' codec can't decode byte 0x8b in position 513: invalid start byte
1. **screening** kept all 1 table(s)

**Refusals by cause:** the model judged it unusable ×20, cannot supply the columns the gold has ×4, not a table (fewer than two columns) ×1
