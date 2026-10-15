# Data report

**Question:** Does CRISPR knockout of the gene IRF1 change the transcriptome of human macrophages, and can the transcriptional response to the perturbation be predicted?

**Tables:** 26 considered — gold **1**, supplementary **5**, unusable **20**, waiting for a person **0**
**Training columns:** 22 of the gold's 22

## Datasets

- `GSE153056_ECCITE_metadata.tsv.gz` — **gold** — review: screen label half (condition column 'guide_ID'); its counts matrix is joined at training time; left out Unnamed: 0, NT — **a person could salvage it:** join the labels in this metadata to the cDNA counts matrix (GSM4633614_ECCITE_cDNA_counts.tsv.gz) on barcode
- `GSM4633605_CITE_cDNA_counts.tsv.gz` — **supplementary** — review: measured transcriptome, usable as evidence for the screen; left out Unnamed: 0 — **a person could salvage it:** transpose it so barcodes are rows and genes are columns, then join the perturbation labels from GSE153056_ECCITE_metadata.tsv.gz on barcode
- `GSM4633608_ECCITE_Arrayed_cDNA_counts.tsv.gz` — **supplementary** — review: measured transcriptome, usable as evidence for the screen; left out Unnamed: 0 — **a person could salvage it:** join the labels in data/fetched/geo/GSE153056/GSE153056_ECCITE_metadata.tsv.gz on barcode
- `GSM4633612_ECCITE_Arrayed_GO_CITE03_counts.tsv.gz` — **supplementary** — review: measured transcriptome, usable as evidence for the screen; left out Unnamed: 0 — **a person could salvage it:** transpose it so cells are rows, then join the guide labels from GSM4633612_ECCITE_Arrayed_GDO_Barcodes.csv.gz (or GSE153056_ECCITE_metadata.tsv.gz) on barcode
- `GSM4633613_ECCITE_Arrayed_GO_LENTI_counts.tsv.gz` — **supplementary** — review: measured transcriptome, usable as evidence for the screen; left out Unnamed: 0 — **a person could salvage it:** transpose it so cells are rows, then join the guide/condition labels from GSE153056_ECCITE_metadata.tsv.gz on barcode
- `GSM4633614_ECCITE_cDNA_counts.tsv.gz` — **supplementary** — review: measured transcriptome, usable as evidence for the screen; left out Unnamed: 0 — **a person could salvage it:** join the labels in data/fetched/geo/GSE153056/GSE153056_ECCITE_metadata.tsv.gz on barcode
- `GSM4633612_ECCITE_Arrayed_GDO_Barcodes.csv.gz` — **unusable** — it cannot be evidence for the same model: it shares only 0 measured column(s) with the gold, and 11 are needed to correct the same model (the gold has 22 feature(s)) (the model had said gold) — **a person could salvage it:** join this barcode-to-guide table onto the counts matrix GSM4633612_ECCITE_Arrayed_GO_CITE03_counts.tsv.gz on barcode
- `GSM4633616_ECCITE_HTO_1-4_Barcodes.csv.gz` — **unusable** — it cannot be evidence for the same model: it shares only 0 measured column(s) with the gold, and 11 are needed to correct the same model (the gold has 22 feature(s)) (the model had said gold) — **a person could salvage it:** join these labels onto the barcode column of GSM4633616_ECCITE_HTO_counts.tsv.gz to obtain a per-cell labelled expression table
- `GSM4633617_ECCITE_HTO_5-8_Barcodes.csv.gz` — **unusable** — it cannot be evidence for the same model: it shares only 0 measured column(s) with the gold, and 11 are needed to correct the same model (the gold has 22 feature(s)) (the model had said gold) — **a person could salvage it:** join the labels in this file onto the cDNA counts matrix (GSM4633614_ECCITE_cDNA_counts.tsv.gz) on barcode
- `GSM4633618_ECCITE_GDO_Barcodes.csv.gz` — **unusable** — it cannot be evidence for the same model: it shares only 0 measured column(s) with the gold, and 11 are needed to correct the same model (the gold has 22 feature(s)) (the model had said gold); left out CGGTGAACAGCTCCTCGCCC, eGFPg1 — **a person could salvage it:** join these guide labels onto the barcodes of GSM4633618_ECCITE_GDO_counts.tsv.gz to build a labelled per-cell screen
- `GSE153056_RAW.tar` — **unusable** — review: not a table: 0 parsed column(s). A table needs at least a feature column and something else
- `GSE153056_CITE_metadata.tsv.gz` — **unusable** — review: This is a per-cell CITE-seq metadata table (barcodes with RNA/ADT/HTO QC columns and hashtag classification). It carries no gene-expression columns and no perturbation/condition column, so it…; left out Unnamed: 0 — **a person could salvage it:** join the perturbation labels from the ECCITE metadata / GDO barcode files onto this table by barcode, then attach the paired cDNA counts matrix
- `GSE153056_ECCITE_Arrayed_metadata.tsv.gz` — **unusable** — review: This is a per-cell metadata table (ECCITE-seq) with QC counts and guide/condition classifications but no gene expression columns; it is not a trainable screen on its own.; left out Unnamed: 0, GO_cite_secondID — **a person could salvage it:** join the labels in this metadata on barcode to the GSE153056 expression/counts matrix
- `GSM4633606_CITE_ADT_Barcodes.csv.gz` — **unusable** — review: This is a barcode-to-protein lookup table for CITE-seq ADT (antibody-derived tag) features, not a per-cell perturbation screen. Each row maps a cell barcode to an antibody/protein name (PDL1,… — **a person could salvage it:** join the labels in data/fetched/geo/GSE153056/GSE153056_CITE_metadata.tsv.gz on barcode
- `GSM4633606_CITE_ADT_counts.tsv.gz` — **unusable** — review: This is a CITE-seq antibody-derived tag (ADT) protein count matrix: rows are surface-marker proteins (BTLA, CD47, CD66a, CD86, ...) and columns are cell barcodes. It carries no gene expressio…; left out Unnamed: 0 — **a person could salvage it:** join the labels in data/fetched/geo/GSE153056/GSE153056_ECCITE_metadata.tsv.gz on barcode and use the cDNA counts in GSM4633605_CITE_cDNA_counts.tsv.gz for expression
- `GSM4633607_CITE_HTO_Barcodes.csv.gz` — **unusable** — review: This is a barcode-to-HTO (hashing antibody) assignment file: two columns, a cell barcode and an HTO tag, with no header and no gene/perturbation column. It is a cell-metadata/annotation table… — **a person could salvage it:** join these HTO labels onto the cDNA counts matrix (GSM4633605_CITE_cDNA_counts.tsv.gz) by barcode to demultiplex samples; the perturbation labels themselves must come from the ECCITE metadata file
- `GSM4633607_CITE_HTO_counts.tsv.gz` — **unusable** — review: This is a CITE-seq HTO (hashing) antibody-count matrix: rows are hashtag antibodies (HTO22, HTO23, HTO25, HTO26) and columns are cell barcodes, with no gene/transcriptome columns and no pertu…; left out Unnamed: 0 — **a person could salvage it:** join the labels in data/fetched/geo/GSE153056/GSE153056_ECCITE_metadata.tsv.gz on barcode, and pair with the cDNA counts matrix GSM4633605_CITE_cDNA_counts.tsv.gz for expression
- `GSM4633609_ECCITE_Arrayed_ADT_Barcodes.csv.gz` — **unusable** — review: This is an antibody-derived tag (ADT) barcode-to-antibody mapping file: each row pairs a cell barcode with a single surface-protein tag (CD86, PDL1, PDL2, CD366). It has no header, no gene-ex… — **a person could salvage it:** join the labels in data/fetched/geo/GSE153056/GSE153056_ECCITE_metadata.tsv.gz on barcode
- `GSM4633609_ECCITE_Arrayed_ADT_counts.tsv.gz` — **unusable** — review: This is an antibody-derived tag (ADT) / cell-surface protein count matrix from a CITE-seq/ECCITE experiment: rows are surface markers (CD86, PDL1, PDL2, CD366) and columns are cell barcodes. …; left out Unnamed: 0 — **a person could salvage it:** join the labels in GSE153056_ECCITE_metadata.tsv.gz on barcode and pair with the cDNA counts in GSM4633608_ECCITE_Arrayed_cDNA_counts.tsv.gz
- `GSM4633610_ECCITE_Arrayed_HTO1_Barcodes.csv.gz` — **unusable** — review: This is a barcode-to-hashtag (HTO) assignment file: two unnamed columns mapping a cell barcode to an HTO tag (e.g. HTO_ETV7g1, HTO_NT5). It carries no gene-expression measurements and no gene… — **a person could salvage it:** join these barcodes to the counts matrix GSM4633610_ECCITE_Arrayed_HTO_counts.tsv.gz and the cDNA counts to recover per-cell perturbation labels
- `GSM4633610_ECCITE_Arrayed_HTO_counts.tsv.gz` — **unusable** — review: This is an antibody-hashtag (HTO) count matrix from a CITE/ECCITE experiment: rows are HTO tags (HTO-IRF7g1, HTO-MARCH8g1, ...) and columns are cell barcodes (r_... and g_...). It is a genes-…; left out Unnamed: 0 — **a person could salvage it:** demultiplex the HTO counts to assign each barcode to its tagged gene, then join those labels onto the cDNA counts matrix GSM4633608_ECCITE_Arrayed_cDNA_counts.tsv.gz on barcode
- `GSM4633611_ECCITE_Arrayed_HTO2_Barcodes.csv.gz` — **unusable** — review: This is a barcode-to-HTO hashtag assignment file (two columns: cell barcode and HTO tag such as HTO_IRF7g1, HTO_eGFP), not a per-cell perturbation screen with a transcriptome. It has no gene-… — **a person could salvage it:** join the HTO tags onto the cDNA counts matrix GSM4633608_ECCITE_Arrayed_cDNA_counts.tsv.gz on barcode to obtain per-cell perturbation labels
- `GSM4633615_ECCITE_ADT_Barcodes.csv.gz` — **unusable** — review: This is a barcode-to-ADT (antibody-derived tag) assignment file from the ECCITE-seq dataset: each row pairs a cell barcode with a surface-protein tag name (CD86, PDL1, PDL2, CD366). It has no… — **a person could salvage it:** join the labels in data/fetched/geo/GSE153056/GSE153056_ECCITE_metadata.tsv.gz on barcode to the cDNA counts matrix
- `GSM4633615_ECCITE_ADT_counts.tsv.gz` — **unusable** — review: This is an antibody-derived tag (ADT/CITE-seq) protein count matrix: rows are surface-marker proteins (CD86, PDL1, PDL2, CD366) and columns are cell barcodes. It carries no gene expression an…; left out Unnamed: 0 — **a person could salvage it:** join the perturbation labels from GSE153056_ECCITE_metadata.tsv.gz on barcode and pair with the cDNA counts matrix
- `GSM4633616_ECCITE_HTO_counts.tsv.gz` — **unusable** — review: This is a hashtag-oligo (HTO) antibody-derived tag count matrix: rows are HTO features (rep1-tx, rep1-ctrl, rep2-tx, rep2-ctrl, ...) and columns are cell barcodes. It carries neither gene exp…; left out Unnamed: 0 — **a person could salvage it:** join the HTO labels to the cDNA counts matrix (GSM4633614_ECCITE_cDNA_counts.tsv.gz) and the guide table (GSM4633618_ECCITE_GDO_counts.tsv.gz) on barcode
- `GSM4633618_ECCITE_GDO_counts.tsv.gz` — **unusable** — review: This is a genes-by-cells count matrix (GDO = guide-derived oligo counts) with gene/guide names in rows and cell barcodes as columns; there is no per-cell condition column and no transcriptome…; left out Unnamed: 0 — **a person could salvage it:** transpose it so barcodes are rows, then join the cDNA counts (GSM4633614_ECCITE_cDNA_counts.tsv.gz) and metadata on barcode
- `geo://GSE194122` — **not downloaded** — the model did not keep it
- `geo://GSE185263` — **not downloaded** — the model did not keep it
- `geo://GSE154567` — **not downloaded** — the model did not keep it
- `geo://GSE171452#suppl/GSE171452_RAW.tar` — **not downloaded** — error: SourceError: https://ftp.ncbi.nlm.nih.gov/geo/series/GSE171nnn/GSE171452/suppl/GSE171452_RAW.tar exceeds the 2,147,483,648-byte cap; raise --max-bytes or fetch a smaller file
- `hf://genome-scale-tools/perturb-seq` — **not downloaded** — the model did not keep it
- `geo://GSE149689` — **not downloaded** — the model did not keep it
- `geo://GSE90063` — **not downloaded** — the model did not keep it
- `geo://GSE171406` — **not downloaded** — the model did not keep it
- `hf://genome-scale/perturb-seq-macrophage` — **not downloaded** — the model did not keep it
- `geo://GSE168620` — **not downloaded** — the model did not keep it
- `hf://cellxgene-census/cellxgene-census` — **not downloaded** — the model did not keep it

## Round 1 -- the labeled table

1. **proposed** `geo://GSE194122` (confidence 0.75) — Directly matches the question: CRISPR knockout perturbation of genes including IRF1 in human macrophages, with transcriptome readout suitable for predicting tra
1. **proposed** `geo://GSE153056` (confidence 0.60) — Provides a labeled CRISPR knockout screen with guide-level perturbation labels and transcriptome measurements, useful as a reference for predicting perturbation
1. **proposed** `geo://GSE171452` (confidence 0.50) — Same cell type (human macrophages) and CRISPR knockout assay, potentially containing IRF1 perturbation and control cells
1. **proposed** `geo://GSE185263` (confidence 0.55) — Serves as the unperturbed atlas of the same tissue and cell type, providing control cells and the same gene panel for co-expression graph evidence
1. **proposed** `geo://GSE154567` (confidence 0.50) — A labeled CRISPR knockout screen that can supply perturbation-transcriptome pairs for training a response predictor
1. **proposed** `geo://GSE90063` (confidence 0.45) — Provides control/untreated human macrophage cells measuring a broad gene panel, supporting the co-expression graph requirement
1. **listing** `geo://GSE194122` — kept
   - 3 file(s) on offer, smallest 3.9 kB, total 3.3 GB # about: title: A sandbox for prediction and integration of DNA, RNA, and proteins in single cells | summary: Single-cell multiomics data collected f…
1. **listing** `geo://GSE153056` — kept
   - 7 file(s) on offer, smallest 1.7 kB, total 124.0 MB # about: title: Characterizing the molecular regulation of inhibitory immune checkpoints with multi-modal single-cell screens. | summary: The expre…
1. **listing** `geo://GSE171452` — kept
   - 11 file(s) on offer, smallest 1.6 kB, total 4.7 GB # about: title: Discovery of target genes and pathways of blood trait loci using pooled CRISPR screens and single cell RNA sequencing | summary: The…
1. **listing** `geo://GSE185263` — kept
   - 2 file(s) on offer, smallest 15.0 kB, total 11.0 MB # about: title: Predicting sepsis severity at first clinical presentation: the role of endotypes and mechanistic signatures | summary: Identifying …
1. **listing** `geo://GSE154567` — kept
   - 6 file(s) on offer, smallest 1.4 kB, total 2.2 GB # about: title: Single cell RNAseq reveal covid19 patient immune response | summary: Eighteen patients admitted to CSMC and diagnosed with COVID19 by…
1. **listing** `geo://GSE90063` — kept
   - 7 file(s) on offer, smallest 2.0 kB, total 1.4 GB # about: title: Perturb-seq: Dissecting molecular circuits with scalable single cell RNA profiling of pooled genetic screens | summary: Methods: We c…
1. **chosen** `geo://GSE153056#suppl/GSE153056_ECCITE_metadata.tsv.gz` — Cell-level metadata with guide-level CRISPR perturbation labels for the ECCITE-seq screen, providing perturbation-transcriptome pairs to train a response predictor.
1. **chosen** `geo://GSE171452#suppl/GSE171452_RAW.tar` — Contains the macrophage CRISPR screen single-cell RNA-seq count matrices needed to test IRF1 knockout transcriptome changes.
1. **chosen** `geo://GSE90063#suppl/GSE90063_dc0hr_umi_wt.txt.gz` — Small control/untreated macrophage UMI count table supplying the unperturbed co-expression graph reference.
1. **listing** `geo://GSE153056#suppl/GSE153056_RAW.tar` — kept
   - keeping GSE153056_RAW.tar as well -- a screen's measurements live in the counts file, and the label table does not carry them
1. **listing** `geo://GSE153056#suppl/GSE153056_CITE_metadata.tsv.gz` — kept
   - keeping GSE153056_CITE_metadata.tsv.gz as well -- this task's labels (a perturbation screen's condition/guide) usually live in the annotated file, not the raw matrix
1. **listing** `geo://GSE90063#suppl/GSE90063_RAW.tar` — kept
   - keeping GSE90063_RAW.tar as well -- a screen's measurements live in the counts file, and the label table does not carry them
1. **dropped** `geo://GSE194122` — the model did not keep it
1. **dropped** `geo://GSE185263` — the model did not keep it
1. **dropped** `geo://GSE154567` — the model did not keep it
1. **downloaded** `geo://GSE153056#suppl/GSE153056_ECCITE_metadata.tsv.gz` → GSE153056_ECCITE_metadata.tsv.gz
1. **downloaded** `geo://GSE153056#suppl/GSE153056_RAW.tar` → GSE153056_RAW.tar, GSE153056_RAW.tar-contents/GSM4633605_CITE_cDNA_counts.tsv.gz, GSE153056_RAW.tar-contents/GSM4633606_CITE_ADT_Barcodes.csv.gz, GSE153056_RAW.tar-contents/GSM4633606_CITE_ADT_counts.tsv.gz
1. **downloaded** `geo://GSE153056#suppl/GSE153056_CITE_metadata.tsv.gz` → GSE153056_CITE_metadata.tsv.gz
1. **download failed** `geo://GSE171452#suppl/GSE171452_RAW.tar` — error: SourceError: https://ftp.ncbi.nlm.nih.gov/geo/series/GSE171nnn/GSE171452/suppl/GSE171452_RAW.tar exceeds the 2,147,483,648-byte cap; raise --max-bytes or fetch a smaller file
1. **downloaded** `geo://GSE90063#suppl/GSE90063_dc0hr_umi_wt.txt.gz` → GSE90063_dc0hr_umi_wt.txt.gz
1. **downloaded** `geo://GSE90063#suppl/GSE90063_RAW.tar` → GSE90063_RAW.tar, GSE90063_RAW.tar-contents/GSM2396856_dc_3hr.mtx.txt.gz, GSE90063_RAW.tar-contents/GSM2396856_dc_3hr_cbc_gbc_dict_lenient.csv.gz, GSE90063_RAW.tar-contents/GSM2396856_dc_3hr_cbc_gbc_dict_strict.csv.gz
1. **screened out** `GSE153056_RAW.tar` — not readable as a delimited text table: UnicodeDecodeError: 'utf-8' codec can't decode byte 0x8b in position 513: invalid start byte
1. **screened out** `GSE90063_dc0hr_umi_wt.txt.gz` — parsed as a single column; this file is probably not a plain table (a matrix with a preamble, or a non-tabular artifact)
1. **screened out** `GSE90063_RAW.tar` — not readable as a delimited text table: UnicodeDecodeError: 'utf-8' codec can't decode byte 0x8b in position 513: invalid start byte
1. **screened out** `GSM2396856_dc_3hr.mtx.txt.gz` — parsed as a single column; this file is probably not a plain table (a matrix with a preamble, or a non-tabular artifact)
1. **screened out** `GSM2396857_dc_0hr.mtx.txt.gz` — parsed as a single column; this file is probably not a plain table (a matrix with a preamble, or a non-tabular artifact)
1. **screened out** `GSM2396858_k562_tfs_7.mtx.txt.gz` — parsed as a single column; this file is probably not a plain table (a matrix with a preamble, or a non-tabular artifact)
1. **screened out** `GSM2396859_k562_tfs_13.mtx.txt.gz` — parsed as a single column; this file is probably not a plain table (a matrix with a preamble, or a non-tabular artifact)
1. **screened out** `GSM2396860_k562_tfs_highmoi.mtx.txt.gz` — parsed as a single column; this file is probably not a plain table (a matrix with a preamble, or a non-tabular artifact)
1. **screened out** `GSM2396861_k562_ccycle.mtx.txt.gz` — parsed as a single column; this file is probably not a plain table (a matrix with a preamble, or a non-tabular artifact)

## Round 2 -- supplementary evidence for that table

1. **proposed** `geo://GSE153056` (confidence 0.70) — Same study series; other samples/splits within GSE153056 provide the same gene expression measurement columns (7SK, A1BG, A2M, etc.) for additional perturbed ma
1. **proposed** `geo://GSE90063` (confidence 0.50) — Provides the same gene-level UMI count measurement columns for other CRISPR perturbation cohorts, matching the transcriptome measurement structure
1. **proposed** `hf://genome-scale-tools/perturb-seq` (confidence 0.30) — Holds gene expression count tables with the same gene symbol columns for additional perturbation rows
1. **proposed** `geo://GSE194122` (confidence 0.60) — Directly measures transcriptome changes after CRISPR knockout of IRF1 in human macrophages, matching the perturbation and cell type
1. **proposed** `geo://GSE153056` (confidence 0.50) — CRISPR knockout screen in a macrophage-like cell line that can provide labeled perturbation-transcriptome pairs for prediction
1. **proposed** `geo://GSE90063` (confidence 0.40) — Human macrophage CRISPR knockout data relevant to IRF1 perturbation effects
1. **proposed** `geo://GSE171524` (confidence 0.50) — Unperturbed macrophage atlas provides control cells for co-expression graph and baseline transcriptome
1. **proposed** `geo://GSE149689` (confidence 0.40) — Contains CRISPR-perturbed human macrophages with control cells for comparison
1. **proposed** `geo://GSE185263` (confidence 0.30) — CRISPR knockout transcriptomic data in relevant immune cell type for predicting perturbation response
1. **proposed** `geo://GSE153056` (confidence 0.85) — Same study as the labeled table; provides additional samples/rows (other sgRNAs, other donors, other timepoints) with the same gene-expression measurement colum
1. **proposed** `geo://GSE149689` (confidence 0.60) — Provides CRISPR knockout transcriptome measurements for other rows (other perturbations/donors) with the same gene-expression column space (gene symbols such as
1. **proposed** `geo://GSE90063` (confidence 0.50) — Supplies transcriptome measurements for other rows (different cell state/timepoint) with the same gene-level expression columns (gene symbols like A1BG, A2M, A4
1. **proposed** `geo://GSE194122` (confidence 0.60) — Directly matches the question: CRISPR knockout perturbation in human macrophages with transcriptome readout, enabling prediction of transcriptional response.
1. **proposed** `geo://GSE153056` (confidence 0.50) — Provides CRISPR knockout perturbation transcriptomes in human macrophages, suitable for modeling gene knockout effects including IRF1.
1. **proposed** `geo://GSE171406` (confidence 0.50) — Serves as the unperturbed control/atlas dataset of the same tissue and cell type for co-expression graph evidence.
1. **proposed** `geo://GSE149689` (confidence 0.40) — Contains macrophage perturbation transcriptomes that can complement the CRISPR knockout analysis.
1. **proposed** `hf://genome-scale/perturb-seq-macrophage` (confidence 0.30) — HuggingFace-hosted processed CRISPR knockout macrophage data for direct download and modeling.
1. **proposed** `geo://GSE168620` (confidence 0.30) — Another candidate CRISPR knockout macrophage dataset for cross-validation of IRF1 knockout effects.
1. **proposed** `geo://GSE153056` (confidence 0.60) — Parent series of the labeled ECCITE metadata; contains additional supplementary files (e.g., other GSM count matrices) with the same gene-expression columns (7S
1. **proposed** `geo://GSE194122` (confidence 0.30) — Shares the same gene-expression measurement columns (gene symbols like A1BG, A2M, AAK1, etc.) for other samples/cohorts, providing transcriptome measurements th
1. **proposed** `hf://cellxgene-census/cellxgene-census` (confidence 0.20) — Contains many scRNA-seq datasets measuring the same gene-expression columns (gene symbols) for other cohorts/samples, usable as supplementary transcriptome meas
1. **listing** `geo://GSE153056` — kept
   - 7 file(s) on offer, smallest 1.7 kB, total 124.0 MB # about: title: Characterizing the molecular regulation of inhibitory immune checkpoints with multi-modal single-cell screens. | summary: The expre…
1. **listing** `geo://GSE90063` — kept
   - 7 file(s) on offer, smallest 2.0 kB, total 1.4 GB # about: title: Perturb-seq: Dissecting molecular circuits with scalable single cell RNA profiling of pooled genetic screens | summary: Methods: We c…
1. **listing** `hf://genome-scale-tools/perturb-seq` — kept
   - 0 file(s) on offer, the repository did not answer a listing for this reference
1. **chosen** `geo://GSE153056#suppl/GSE153056_ECCITE_Arrayed_metadata.tsv.gz` — Arrayed ECCITE-seq metadata for the same study provides per-cell perturbation and expression-linked observations for additional perturbed macrophage rows.
1. **chosen** `geo://GSE90063#suppl/GSE90063_dc3hr_umi_wt.txt.gz` — Gene-level UMI count matrix from the Perturb-seq study gives transcriptome measurements for additional CRISPR perturbation cohorts.
1. **listing** `geo://GSE153056#suppl/GSE153056_RAW.tar` — kept
   - keeping GSE153056_RAW.tar as well -- a screen's measurements live in the counts file, and the label table does not carry them
1. **listing** `geo://GSE153056#suppl/GSE153056_CITE_metadata.tsv.gz` — kept
   - keeping GSE153056_CITE_metadata.tsv.gz as well -- this task's labels (a perturbation screen's condition/guide) usually live in the annotated file, not the raw matrix
1. **listing** `geo://GSE90063#suppl/GSE90063_RAW.tar` — kept
   - keeping GSE90063_RAW.tar as well -- a screen's measurements live in the counts file, and the label table does not carry them
1. **dropped** `hf://genome-scale-tools/perturb-seq` — the model did not keep it
1. **listing** `geo://GSE194122` — kept
   - 3 file(s) on offer, smallest 3.9 kB, total 3.3 GB # about: title: A sandbox for prediction and integration of DNA, RNA, and proteins in single cells | summary: Single-cell multiomics data collected f…
1. **listing** `geo://GSE171524` — kept
   - 5 file(s) on offer, smallest 2.0 kB, total 1.8 GB # about: title: Columbia University/NYP COVID-19 Lung Atlas | summary: We profiled 116,314 cells using snRNA-seq of 20 frozen lungs obtained from 19 …
1. **listing** `geo://GSE149689` — kept
   - 4 file(s) on offer, smallest 3.2 kB, total 467.7 MB # about: title: Immunophenotyping of COVID-19 and Influenza Underscores the Association of Type I IFN Response in Severe COVID-19 | summary: Althou…
1. **listing** `geo://GSE185263` — kept
   - 2 file(s) on offer, smallest 15.0 kB, total 11.0 MB # about: title: Predicting sepsis severity at first clinical presentation: the role of endotypes and mechanistic signatures | summary: Identifying …
1. **chosen** `geo://GSE153056#suppl/GSE153056_ECCITE_metadata.tsv.gz` — Provides labeled CRISPR perturbation-transcriptome metadata from a macrophage-like ECCITE-seq screen, enabling prediction of perturbation response.
1. **chosen** `geo://GSE149689#suppl/GSE149689_matrix.mtx.gz` — Expression matrix of CRISPR-perturbed human macrophages with controls, directly relevant to IRF1 knockout transcriptome changes.
1. **chosen** `geo://GSE171524#suppl/GSE171524_processed_data.csv.gz` — Unperturbed human macrophage transcriptomes from a lung atlas provide baseline control cells for co-expression and comparison.
1. **listing** `geo://GSE171524#suppl/GSE171524_RAW.tar` — kept
   - keeping GSE171524_RAW.tar as well -- a screen's measurements live in the counts file, and the label table does not carry them
1. **listing** `geo://GSE171524#suppl/GSE171524_lung_metaData.txt.gz` — kept
   - keeping GSE171524_lung_metaData.txt.gz as well -- this task's labels (a perturbation screen's condition/guide) usually live in the annotated file, not the raw matrix
1. **dropped** `geo://GSE194122` — the model did not keep it
1. **dropped** `geo://GSE185263` — the model did not keep it
1. **dropped** `geo://GSE149689` — the model did not keep it
1. **dropped** `geo://GSE90063` — the model did not keep it
1. **listing** `geo://GSE171406` — kept
   - 8 file(s) on offer, smallest 2.8 kB, total 721.0 MB # about: title: Integration of spatial and single cell transcriptomics localizes epithelial-immune cross-talk in kidney injury | summary: Single ce…
1. **listing** `hf://genome-scale/perturb-seq-macrophage` — kept
   - 0 file(s) on offer, the repository did not answer a listing for this reference
1. **listing** `geo://GSE168620` — kept
   - 4 file(s) on offer, smallest 2.3 kB, total 3.4 GB # about: title: Ultra-high throughput single-cell RNA sequencing by combinatorial fluidic indexing | summary: Cell atlas projects and high-throughput…
1. **chosen** `geo://GSE153056#suppl/GSE153056_RAW.tar` — Raw ECCITE-seq count matrices and guide assignments from the pooled CRISPR screen, the actual measurements needed to quantify transcriptional response to knockout.
1. **dropped** `geo://GSE171406` — the model did not keep it
1. **dropped** `hf://genome-scale/perturb-seq-macrophage` — the model did not keep it
1. **dropped** `geo://GSE168620` — the model did not keep it
1. **listing** `hf://cellxgene-census/cellxgene-census` — kept
   - 0 file(s) on offer, the repository did not answer a listing for this reference
1. **dropped** `hf://cellxgene-census/cellxgene-census` — the model did not keep it
1. **downloaded** `geo://GSE153056#suppl/GSE153056_ECCITE_Arrayed_metadata.tsv.gz` → GSE153056_ECCITE_Arrayed_metadata.tsv.gz
1. **downloaded** `geo://GSE153056#suppl/GSE153056_RAW.tar` → GSE153056_RAW.tar, GSE153056_RAW.tar-contents/GSM4633605_CITE_cDNA_counts.tsv.gz, GSE153056_RAW.tar-contents/GSM4633606_CITE_ADT_Barcodes.csv.gz, GSE153056_RAW.tar-contents/GSM4633606_CITE_ADT_counts.tsv.gz
1. **downloaded** `geo://GSE153056#suppl/GSE153056_CITE_metadata.tsv.gz` → GSE153056_CITE_metadata.tsv.gz
1. **downloaded** `geo://GSE90063#suppl/GSE90063_dc3hr_umi_wt.txt.gz` → GSE90063_dc3hr_umi_wt.txt.gz
1. **downloaded** `geo://GSE90063#suppl/GSE90063_RAW.tar` → GSE90063_RAW.tar, GSE90063_RAW.tar-contents/GSM2396856_dc_3hr.mtx.txt.gz, GSE90063_RAW.tar-contents/GSM2396856_dc_3hr_cbc_gbc_dict_lenient.csv.gz, GSE90063_RAW.tar-contents/GSM2396856_dc_3hr_cbc_gbc_dict_strict.csv.gz
1. **downloaded** `geo://GSE153056#suppl/GSE153056_ECCITE_metadata.tsv.gz` → GSE153056_ECCITE_metadata.tsv.gz
1. **downloaded** `geo://GSE153056#suppl/GSE153056_RAW.tar` → GSE153056_RAW.tar, GSE153056_RAW.tar-contents/GSM4633605_CITE_cDNA_counts.tsv.gz, GSE153056_RAW.tar-contents/GSM4633606_CITE_ADT_Barcodes.csv.gz, GSE153056_RAW.tar-contents/GSM4633606_CITE_ADT_counts.tsv.gz
1. **downloaded** `geo://GSE153056#suppl/GSE153056_CITE_metadata.tsv.gz` → GSE153056_CITE_metadata.tsv.gz
1. **downloaded** `geo://GSE90063#suppl/GSE90063_dc3hr_umi_wt.txt.gz` → GSE90063_dc3hr_umi_wt.txt.gz
1. **downloaded** `geo://GSE90063#suppl/GSE90063_RAW.tar` → GSE90063_RAW.tar, GSE90063_RAW.tar-contents/GSM2396856_dc_3hr.mtx.txt.gz, GSE90063_RAW.tar-contents/GSM2396856_dc_3hr_cbc_gbc_dict_lenient.csv.gz, GSE90063_RAW.tar-contents/GSM2396856_dc_3hr_cbc_gbc_dict_strict.csv.gz
1. **downloaded** `geo://GSE149689#suppl/GSE149689_matrix.mtx.gz` → GSE149689_matrix.mtx.gz
1. **downloaded** `geo://GSE171524#suppl/GSE171524_processed_data.csv.gz` → GSE171524_processed_data.csv.gz
1. **downloaded** `geo://GSE171524#suppl/GSE171524_RAW.tar` → GSE171524_RAW.tar, GSE171524_RAW.tar-contents/GSM5226574_C51ctr_raw_counts.csv.gz, GSE171524_RAW.tar-contents/GSM5226575_C52ctr_raw_counts.csv.gz, GSE171524_RAW.tar-contents/GSM5226576_C53ctr_raw_counts.csv.gz
1. **downloaded** `geo://GSE171524#suppl/GSE171524_lung_metaData.txt.gz` → GSE171524_lung_metaData.txt.gz
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
1. **screened out** `GSE90063_dc3hr_umi_wt.txt.gz` — parsed as a single column; this file is probably not a plain table (a matrix with a preamble, or a non-tabular artifact)
1. **screened out** `GSE153056_RAW.tar` — not readable as a delimited text table: UnicodeDecodeError: 'utf-8' codec can't decode byte 0x8b in position 513: invalid start byte
1. **screened out** `GSE90063_dc3hr_umi_wt.txt.gz` — parsed as a single column; this file is probably not a plain table (a matrix with a preamble, or a non-tabular artifact)
1. **screened out** `GSE90063_RAW.tar` — not readable as a delimited text table: UnicodeDecodeError: 'utf-8' codec can't decode byte 0x8b in position 513: invalid start byte
1. **screened out** `GSM2396856_dc_3hr.mtx.txt.gz` — parsed as a single column; this file is probably not a plain table (a matrix with a preamble, or a non-tabular artifact)
1. **screened out** `GSM2396857_dc_0hr.mtx.txt.gz` — parsed as a single column; this file is probably not a plain table (a matrix with a preamble, or a non-tabular artifact)
1. **screened out** `GSM2396858_k562_tfs_7.mtx.txt.gz` — parsed as a single column; this file is probably not a plain table (a matrix with a preamble, or a non-tabular artifact)
1. **screened out** `GSM2396859_k562_tfs_13.mtx.txt.gz` — parsed as a single column; this file is probably not a plain table (a matrix with a preamble, or a non-tabular artifact)
1. **screened out** `GSM2396860_k562_tfs_highmoi.mtx.txt.gz` — parsed as a single column; this file is probably not a plain table (a matrix with a preamble, or a non-tabular artifact)
1. **screened out** `GSM2396861_k562_ccycle.mtx.txt.gz` — parsed as a single column; this file is probably not a plain table (a matrix with a preamble, or a non-tabular artifact)
1. **screened out** `GSE149689_matrix.mtx.gz` — parsed as a single column; this file is probably not a plain table (a matrix with a preamble, or a non-tabular artifact)
1. **screened out** `GSE171524_RAW.tar` — not readable as a delimited text table: UnicodeDecodeError: 'utf-8' codec can't decode byte 0x8b in position 513: invalid start byte
1. **screening** kept all 1 table(s)
1. **screened out** `GSE153056_RAW.tar` — not readable as a delimited text table: UnicodeDecodeError: 'utf-8' codec can't decode byte 0x8b in position 513: invalid start byte
1. **screening** kept all 1 table(s)

**Refusals by cause:** the model judged it unusable ×15, cannot supply the columns the gold has ×4, not a table (fewer than two columns) ×1
