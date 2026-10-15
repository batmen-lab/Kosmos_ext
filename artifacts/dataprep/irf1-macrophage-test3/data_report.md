# Data report

**Question:** Does CRISPR knockout of the gene IRF1 change the transcriptome of human macrophages, and can the transcriptional response to the perturbation be predicted?

**Tables:** 26 considered — gold **1**, supplementary **0**, unusable **25**, waiting for a person **0**
**Training columns:** 22 of the gold's 22

## Datasets

- `GSE153056_ECCITE_metadata.tsv.gz` — **gold** — review: screen label half (condition column 'guide_ID'); its counts matrix is joined at training time; left out Unnamed: 0, NT — **a person could salvage it:** join the labels in this metadata (guide_ID/gene) to the cDNA counts matrix GSM4633614_ECCITE_cDNA_counts.tsv.gz on barcode
- `GSM4633612_ECCITE_Arrayed_GDO_Barcodes.csv.gz` — **unusable** — it cannot be evidence for the same model: it shares only 0 measured column(s) with the gold, and 11 are needed to correct the same model (the gold has 22 feature(s)) (the model had said gold) — **a person could salvage it:** join the labels in GSM4633612_ECCITE_Arrayed_GO_CITE03_counts.tsv.gz on barcode
- `GSM4633616_ECCITE_HTO_1-4_Barcodes.csv.gz` — **unusable** — it cannot be evidence for the same model: it shares only 0 measured column(s) with the gold, and 11 are needed to correct the same model (the gold has 22 feature(s)) (the model had said gold) — **a person could salvage it:** join these labels onto the barcode column of GSM4633616_ECCITE_HTO_counts.tsv.gz
- `GSM4633617_ECCITE_HTO_5-8_Barcodes.csv.gz` — **unusable** — it cannot be evidence for the same model: it shares only 0 measured column(s) with the gold, and 11 are needed to correct the same model (the gold has 22 feature(s)) (the model had said gold) — **a person could salvage it:** join the labels in this file onto the barcode column of GSM4633616_ECCITE_HTO_counts.tsv.gz
- `GSM4633618_ECCITE_GDO_Barcodes.csv.gz` — **unusable** — it cannot be evidence for the same model: it shares only 0 measured column(s) with the gold, and 11 are needed to correct the same model (the gold has 22 feature(s)) (the model had said gold); left out CGGTGAACAGCTCCTCGCCC, eGFPg1 — **a person could salvage it:** join the labels in GSM4633618_ECCITE_GDO_counts.tsv.gz on barcode
- `GSE153056_RAW.tar` — **unusable** — review: not a table: 0 parsed column(s). A table needs at least a feature column and something else
- `GSE153056_CITE_metadata.tsv.gz` — **unusable** — review: This is a per-cell CITE-seq metadata table (barcodes with RNA/ADT/HTO QC columns and hashtag classification). It has no gene-expression columns and no perturbation/condition column, so it is …; left out Unnamed: 0 — **a person could salvage it:** join the perturbation labels from GSE153056_ECCITE_metadata.tsv.gz onto these barcodes, and pair with the cDNA counts matrix (GSM4633605_CITE_cDNA_counts.tsv.gz) for expression
- `GSE153056_ECCITE_Arrayed_metadata.tsv.gz` — **unusable** — review: This is a per-cell metadata table (ECCITE-seq) with barcodes and QC/classification columns but no gene expression columns and no guide_ID column; the perturbation identity is encoded in GO_le…; left out Unnamed: 0, GO_cite_secondID — **a person could salvage it:** join the labels in this metadata (GO_lenti_maxID / GO_cite_maxID keyed on the barcode column) onto the matching GSE153056 expression matrix on barcode
- `GSM4633605_CITE_cDNA_counts.tsv.gz` — **unusable** — review: This is a genes-by-cells count matrix (gene symbols as rows, cell barcodes as columns) with no perturbation/condition column; the label 'gene' is not present as a column. It is the CITE cDNA …; left out Unnamed: 0 — **a person could salvage it:** join the labels in data/fetched/geo/GSE153056/GSE153056_CITE_metadata.tsv.gz on barcode
- `GSM4633606_CITE_ADT_Barcodes.csv.gz` — **unusable** — review: This is a barcode-to-antibody (ADT/CITE-seq) feature mapping file: each row pairs a cell barcode with an antibody-derived tag name (PDL1, CD86, IDO1, TIM3, PDL2). It has no header, no gene-ex… — **a person could salvage it:** join the labels in data/fetched/geo/GSE153056/GSE153056_ECCITE_metadata.tsv.gz on barcode
- `GSM4633606_CITE_ADT_counts.tsv.gz` — **unusable** — review: This is a CITE-seq antibody-derived tag (ADT) protein count matrix: rows are antibody features (BTLA, CD47, CD66a, CD86, ...) and columns are cell barcodes. It is a features-by-cells matrix w…; left out Unnamed: 0 — **a person could salvage it:** join the labels in data/fetched/geo/GSE153056/GSE153056_ECCITE_metadata.tsv.gz on barcode, and pair with the cDNA matrix GSM4633605_CITE_cDNA_counts.tsv.gz for the transcriptome
- `GSM4633607_CITE_HTO_Barcodes.csv.gz` — **unusable** — review: This is a barcode-to-HTO (hashing antibody) assignment file: two columns, a cell barcode and an HTO tag. It carries neither a gene perturbation identity nor any transcriptome measurements, so… — **a person could salvage it:** join these HTO labels onto the matching cDNA counts matrix (GSM4633607_CITE_HTO_counts.tsv.gz / the ECCITE cDNA counts) on barcode, then look for a perturbation column in the ECCITE metadata
- `GSM4633607_CITE_HTO_counts.tsv.gz` — **unusable** — review: This is a CITE-seq HTO (hashing) antibody-count matrix: rows are hashtag oligos (HTO22, HTO23, HTO25, HTO26) and columns are cell barcodes, with no gene/transcriptome columns and no perturbat…; left out Unnamed: 0 — **a person could salvage it:** join the labels in data/fetched/geo/GSE153056/GSE153056_RAW.tar-contents/GSM4633607_CITE_HTO_Barcodes.csv.gz on barcode to demultiplex, then pair with the cDNA counts matrix
- `GSM4633608_ECCITE_Arrayed_cDNA_counts.tsv.gz` — **unusable** — review: This is a genes-by-cells counts matrix (16,696 columns: gene rows in the first column, cell barcodes as columns, with r_ and g_ prefixed barcodes). It carries the transcriptome but no per-cel…; left out Unnamed: 0 — **a person could salvage it:** transpose it so cells are rows, then join the perturbation labels from GSE153056_ECCITE_metadata.tsv.gz on barcode
- `GSM4633609_ECCITE_Arrayed_ADT_Barcodes.csv.gz` — **unusable** — review: This is a two-column barcode-to-ADT mapping file (cell barcode, antibody-derived tag name such as CD86/PDL1), not a per-cell perturbation screen. It has no gene/guide/condition column and no … — **a person could salvage it:** join the labels in data/fetched/geo/GSE153056/GSE153056_ECCITE_metadata.tsv.gz on barcode, and pair with the cDNA counts matrix GSM4633608_ECCITE_Arrayed_cDNA_counts.tsv.gz
- `GSM4633609_ECCITE_Arrayed_ADT_counts.tsv.gz` — **unusable** — review: This is an antibody-derived tag (ADT) / cell-surface protein count matrix from a CITE-seq/ECCITE experiment: rows are surface markers (CD86, PDL1, PDL2, CD366) and columns are cell barcodes. …; left out Unnamed: 0 — **a person could salvage it:** join the perturbation labels from the ECCITE metadata / GDO barcode files on barcode, and pair with the cDNA counts matrix GSM4633608_ECCITE_Arrayed_cDNA_counts.tsv.gz for the transcriptome
- `GSM4633610_ECCITE_Arrayed_HTO1_Barcodes.csv.gz` — **unusable** — review: This is a barcode-to-hashtag (HTO) assignment file: two columns mapping cell barcodes to hashtag labels (HTO_ETV7g1, HTO_NT5, etc.). It has no header, no gene/perturbation-target column, and … — **a person could salvage it:** join these barcode-to-HTO labels onto the matching counts matrix (GSM4633610_ECCITE_Arrayed_HTO_counts.tsv.gz) and the ECCITE metadata on barcode
- `GSM4633610_ECCITE_Arrayed_HTO_counts.tsv.gz` — **unusable** — review: This is an antibody-hashtag (HTO) count matrix from a CITE/ECCITE experiment: rows are HTO tags (HTO-IRF7g1, HTO-MARCH8g1, ...) and columns are cell barcodes (r_/g_ prefixes). It is a genes-b…; left out Unnamed: 0 — **a person could salvage it:** join the labels in data/fetched/geo/GSE153056/GSE153056_ECCITE_metadata.tsv.gz on barcode to the cDNA counts matrix GSM4633608_ECCITE_Arrayed_cDNA_counts.tsv.gz
- `GSM4633611_ECCITE_Arrayed_HTO2_Barcodes.csv.gz` — **unusable** — review: This is a two-column barcode-to-HTO assignment file (cell barcode, hashtag identity such as HTO_IRF7g1, HTO_eGFP). It has no header, no gene-expression columns, and no per-cell gene-target co… — **a person could salvage it:** join this barcode-to-HTO map onto the cDNA counts matrix (GSM4633608_ECCITE_Arrayed_cDNA_counts.tsv.gz) and the ECCITE metadata to recover per-cell gene targets
- `GSM4633612_ECCITE_Arrayed_GO_CITE03_counts.tsv.gz` — **unusable** — review: This is a genes-by-cells count matrix (rows are genes/guides like eGFPg1, CUL3g1; columns are cell barcodes prefixed r_ and g_). It has no per-cell condition column: the perturbation identity…; left out Unnamed: 0 — **a person could salvage it:** join the labels in GSM4633612_ECCITE_Arrayed_GDO_Barcodes.csv.gz on barcode, and pair with GSM4633608_ECCITE_Arrayed_cDNA_counts.tsv.gz for expression
- `GSM4633613_ECCITE_Arrayed_GO_LENTI_counts.tsv.gz` — **unusable** — review: This is a genes-by-cells count matrix (rows are guides/genes like eGFPg1, CUL3g1; columns are cell barcodes r_* and g_*), not a per-cell table with a condition column. The label 'gene' is not…; left out Unnamed: 0 — **a person could salvage it:** transpose it so cells are rows, then join the guide/condition labels from GSE153056_ECCITE_metadata.tsv.gz on barcode
- `GSM4633614_ECCITE_cDNA_counts.tsv.gz` — **unusable** — review: This is a genes-by-cells count matrix (genes as rows, cell barcodes as columns) with no perturbation/condition column; the label 'gene' is the row index, not a per-cell perturbation identity.; left out Unnamed: 0 — **a person could salvage it:** transpose it, genes are rows, then join the labels in data/fetched/geo/GSE153056/GSE153056_ECCITE_metadata.tsv.gz on barcode
- `GSM4633615_ECCITE_ADT_Barcodes.csv.gz` — **unusable** — review: This is a barcode-to-ADT (antibody-derived tag) assignment file for the ECCITE-seq dataset: each row pairs a cell barcode with a surface-protein tag name (CD86, PDL1, PDL2, CD366). It has no …
- `GSM4633615_ECCITE_ADT_counts.tsv.gz` — **unusable** — review: This is an antibody-derived tag (ADT) / cell-surface protein count matrix from a CITE-seq/ECCITE experiment: rows are surface markers (CD86, PDL1, PDL2, CD366) and columns are cell barcodes. …; left out Unnamed: 0 — **a person could salvage it:** join the perturbation labels from GSE153056_ECCITE_metadata.tsv.gz (and the cDNA counts in GSM4633614_ECCITE_cDNA_counts.tsv.gz) on barcode; this ADT matrix alone is not trainable
- `GSM4633616_ECCITE_HTO_counts.tsv.gz` — **unusable** — review: This is a hashtag-oligo (HTO) antibody-derived-tag count matrix: rows are HTO features (rep1-tx, rep1-ctrl, rep2-tx, rep2-ctrl, ...) and columns are cell barcodes. It carries neither a per-ce…; left out Unnamed: 0 — **a person could salvage it:** use it only to demultiplex barcodes, then join the cDNA counts (GSM4633614_ECCITE_cDNA_counts.tsv.gz) with the guide/GDO labels (GSM4633618_ECCITE_GDO_counts.tsv.gz) on barcode
- `GSM4633618_ECCITE_GDO_counts.tsv.gz` — **unusable** — review: This is a genes-by-cells count matrix (GDO = guide-derived oligo counts) with genes/guides as rows and cell barcodes as columns; there is no per-cell condition column, so it is not a per-cell…; left out Unnamed: 0 — **a person could salvage it:** transpose it, genes are rows
- `geo://GSE194122` — **not downloaded** — the model did not keep it
- `geo://GSE171406` — **not downloaded** — the model did not keep it
- `geo://GSE168620` — **not downloaded** — the model did not keep it
- `geo://GSE201333` — **not downloaded** — the model did not keep it
- `geo://GSE165080` — **not downloaded** — the model did not keep it
- `geo://GSE153055` — **not downloaded** — the model did not keep it

## Round 1 -- the labeled table

1. **proposed** `geo://GSE194122` (confidence 0.60) — Contains human myeloid cells with CRISPR knockout perturbations and unperturbed controls, providing the same cell type and gene panel needed for co-expression a
1. **proposed** `geo://GSE153056` (confidence 0.50) — Directly perturbs genes in human macrophages by CRISPR knockout, matching the assay and cell type required
1. **proposed** `geo://GSE171406` (confidence 0.50) — CRISPR knockout Perturb-seq in human immune cells provides labeled perturbation transcriptomes for prediction
1. **proposed** `geo://GSE168620` (confidence 0.45) — Knockout screen in human macrophages supplies the perturbation and control transcriptomes needed
1. **proposed** `geo://GSE201333` (confidence 0.50) — Unperturbed human macrophage atlas provides control cells and the same gene panel for co-expression graph evidence
1. **proposed** `geo://GSE165080` (confidence 0.45) — CRISPR knockout perturbation data in human myeloid lineage supports modeling transcriptional response
1. **listing** `geo://GSE194122` — kept
   - 2 file(s) on offer, smallest 587.0 MB, total 3.3 GB # about: title: A sandbox for prediction and integration of DNA, RNA, and proteins in single cells | summary: Single-cell multiomics data collected…
1. **listing** `geo://GSE153056` — kept
   - 5 file(s) on offer, smallest 1.7 kB, total 124.0 MB # about: title: Characterizing the molecular regulation of inhibitory immune checkpoints with multi-modal single-cell screens. | summary: The expre…
1. **listing** `geo://GSE171406` — kept
   - 6 file(s) on offer, smallest 20.0 MB, total 721.0 MB # about: title: Integration of spatial and single cell transcriptomics localizes epithelial-immune cross-talk in kidney injury | summary: Single c…
1. **listing** `geo://GSE168620` — kept
   - 2 file(s) on offer, smallest 7.1 kB, total 16.3 kB # about: title: Ultra-high throughput single-cell RNA sequencing by combinatorial fluidic indexing | summary: Cell atlas projects and high-throughpu…
1. **listing** `geo://GSE201333` — kept
   - 3 file(s) on offer, smallest 1.6 kB, total 15.0 GB # about: title: Tabula Sapiens | summary: Tabula Sapiens is a single cell transcriptomic atlas of 24 human tissues and organs. | overall_design: Var…
1. **listing** `geo://GSE165080` — kept
   - 4 file(s) on offer, smallest 446.0 kB, total 1.0 GB # about: title: Identification of distinct immune cell subsets associated with various clinical presentations, disease severity and viral persisten…
1. **chosen** `geo://GSE153056#suppl/GSE153056_ECCITE_metadata.tsv.gz` — Metadata for the ECCITE-seq CRISPR screen in human macrophages, giving per-cell perturbation labels and cell identities needed to link IRF1 knockout to transcriptomes.
1. **chosen** `geo://GSE153056#suppl/GSE153056_RAW.tar` — Raw count matrices from the ECCITE-seq CRISPR screen in human macrophages, providing the actual transcriptome measurements for IRF1 knockout and control cells.
1. **listing** `geo://GSE153056#suppl/GSE153056_RAW.tar` — kept
   - keeping GSE153056_RAW.tar as well -- a screen's measurements live in the counts file, and the label table does not carry them
1. **listing** `geo://GSE153056#suppl/GSE153056_CITE_metadata.tsv.gz` — kept
   - keeping GSE153056_CITE_metadata.tsv.gz as well -- this task's labels (a perturbation screen's condition/guide) usually live in the annotated file, not the raw matrix
1. **dropped** `geo://GSE194122` — the model did not keep it
1. **dropped** `geo://GSE171406` — the model did not keep it
1. **dropped** `geo://GSE168620` — the model did not keep it
1. **dropped** `geo://GSE201333` — the model did not keep it
1. **dropped** `geo://GSE165080` — the model did not keep it
1. **downloaded** `geo://GSE153056#suppl/GSE153056_ECCITE_metadata.tsv.gz` → GSE153056_ECCITE_metadata.tsv.gz
1. **downloaded** `geo://GSE153056#suppl/GSE153056_RAW.tar` → GSE153056_RAW.tar, GSE153056_RAW.tar-contents/GSM4633605_CITE_cDNA_counts.tsv.gz, GSE153056_RAW.tar-contents/GSM4633606_CITE_ADT_Barcodes.csv.gz, GSE153056_RAW.tar-contents/GSM4633606_CITE_ADT_counts.tsv.gz
1. **downloaded** `geo://GSE153056#suppl/GSE153056_CITE_metadata.tsv.gz` → GSE153056_CITE_metadata.tsv.gz
1. **downloaded** `geo://GSE153056#suppl/GSE153056_RAW.tar` → GSE153056_RAW.tar, GSE153056_RAW.tar-contents/GSM4633605_CITE_cDNA_counts.tsv.gz, GSE153056_RAW.tar-contents/GSM4633606_CITE_ADT_Barcodes.csv.gz, GSE153056_RAW.tar-contents/GSM4633606_CITE_ADT_counts.tsv.gz
1. **downloaded** `geo://GSE153056#suppl/GSE153056_CITE_metadata.tsv.gz` → GSE153056_CITE_metadata.tsv.gz
1. **screened out** `GSE153056_RAW.tar` — not readable as a delimited text table: UnicodeDecodeError: 'utf-8' codec can't decode byte 0x8b in position 513: invalid start byte

## Round 2 -- supplementary evidence for that table

1. **proposed** `geo://GSE153056` (confidence 0.60) — Same study and same measurement modalities (nCount_RNA, nFeature_RNA, nCount_HTO, nFeature_HTO, nCount_GDO, nFeature_GDO, nCount_ADT, nFeature_ADT, percent.mito
1. **proposed** `geo://GSE153055` (confidence 0.40) — Likely shares the same measured columns (RNA, ADT, HTO, GDO counts and guide/crispr annotations) for other rows/guides from the same experimental design.
1. **proposed** `geo://GSE194122` (confidence 0.30) — Provides RNA and ADT count columns (nCount_RNA, nFeature_RNA, nCount_ADT, nFeature_ADT, percent.mito) that line up with the labeled table's measured columns, th
1. **listing** `geo://GSE153056` — kept
   - 5 file(s) on offer, smallest 1.7 kB, total 124.0 MB # about: title: Characterizing the molecular regulation of inhibitory immune checkpoints with multi-modal single-cell screens. | summary: The expre…
1. **listing** `geo://GSE153055` — kept
   - 0 file(s) on offer, size unknown # about: title: AlphaBeta: Computational inference of epimutation rates and spectra from high-throughput DNA methylation data in plants | summary: Heritable changes i…
1. **listing** `geo://GSE194122` — kept
   - 2 file(s) on offer, smallest 587.0 MB, total 3.3 GB # about: title: A sandbox for prediction and integration of DNA, RNA, and proteins in single cells | summary: Single-cell multiomics data collected…
1. **chosen** `geo://GSE153056#suppl/GSE153056_ECCITE_Arrayed_metadata.tsv.gz` — This is the arrayed ECCITE-seq metadata table with per-cell RNA/ADT/HTO/GDO counts and CRISPR guide annotations from the same IRF1 knockout macrophage screen, providing the observ…
1. **listing** `geo://GSE153056#suppl/GSE153056_RAW.tar` — kept
   - keeping GSE153056_RAW.tar as well -- a screen's measurements live in the counts file, and the label table does not carry them
1. **listing** `geo://GSE153056#suppl/GSE153056_CITE_metadata.tsv.gz` — kept
   - keeping GSE153056_CITE_metadata.tsv.gz as well -- this task's labels (a perturbation screen's condition/guide) usually live in the annotated file, not the raw matrix
1. **dropped** `geo://GSE153055` — the model did not keep it
1. **dropped** `geo://GSE194122` — the model did not keep it
1. **downloaded** `geo://GSE153056#suppl/GSE153056_ECCITE_Arrayed_metadata.tsv.gz` → GSE153056_ECCITE_Arrayed_metadata.tsv.gz
1. **downloaded** `geo://GSE153056#suppl/GSE153056_RAW.tar` → GSE153056_RAW.tar, GSE153056_RAW.tar-contents/GSM4633605_CITE_cDNA_counts.tsv.gz, GSE153056_RAW.tar-contents/GSM4633606_CITE_ADT_Barcodes.csv.gz, GSE153056_RAW.tar-contents/GSM4633606_CITE_ADT_counts.tsv.gz
1. **downloaded** `geo://GSE153056#suppl/GSE153056_CITE_metadata.tsv.gz` → GSE153056_CITE_metadata.tsv.gz
1. **screening** kept all 1 table(s)

**Refusals by cause:** the model judged it unusable ×20, cannot supply the columns the gold has ×4, not a table (fewer than two columns) ×1
