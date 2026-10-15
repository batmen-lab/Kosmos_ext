# Data report

**Question:** In primary human CD4+ T cells, can the transcriptional response to CRISPR knockout of transcriptional regulators not seen in training be predicted from the other knockouts, using large single-cell atlases of unperturbed human T cells as supplementary evidence?

**Tables:** 39 considered — gold **1**, supplementary **7**, unusable **31**, waiting for a person **0**
**Training columns:** 2 of the gold's 2

## Datasets

- `GSM3375487_D1N_CellBC_sgRNA.csv.gz` — **gold** — review: screen label half (condition column 'gRNA.ID'); its counts matrix is joined at training time; left out Cell.BC — **a person could salvage it:** join the labels in this file to the counts matrix GSM3375483_D1N_matrix on Cell.BC
- `GSM3375488_D1S_CellBC_sgRNA.csv.gz` — **supplementary** — labeled table, demoted to supplementary evidence (the primary gold is data/fetched/geo/GSE119450/GSE119450_RAW.tar-contents/GSM3375487_D1N_CellBC_sgRNA.csv.gz, named as the primary labeled table); it… (the model had said gold); shares 2 column(s) with the gold: sample.name, UMI.count — **a person could salvage it:** join the labels in data/fetched/geo/GSE119450/GSM3375484_D1S_matrix__matrix.mtx-table.csv on barcode (Cell.BC)
- `GSM3375489_D2N_CellBC_sgRNA.csv.gz` — **supplementary** — labeled table, demoted to supplementary evidence (the primary gold is data/fetched/geo/GSE119450/GSE119450_RAW.tar-contents/GSM3375487_D1N_CellBC_sgRNA.csv.gz, named as the primary labeled table); it… (the model had said gold); shares 2 column(s) with the gold: sample.name, UMI.count — **a person could salvage it:** join the labels in this file onto the matching GSM3375485_D2N_matrix barcodes to build a per-cell screen
- `GSM3375490_D2S_CellBC_sgRNA.csv.gz` — **supplementary** — labeled table, demoted to supplementary evidence (the primary gold is data/fetched/geo/GSE119450/GSE119450_RAW.tar-contents/GSM3375487_D1N_CellBC_sgRNA.csv.gz, named as the primary labeled table); it… (the model had said gold); shares 2 column(s) with the gold: sample.name, UMI.count — **a person could salvage it:** join the labels in data/fetched/geo/GSE119450/GSM3375486_D2S_matrix__matrix.mtx-table.csv on barcode (Cell.BC)
- `GSM3375483_D1N_matrix__matrix.mtx-table.csv` — **supplementary** — review: measured transcriptome, usable as evidence for the screen; converted from matrix.mtx (2000 of 11105 cells, 33694 of 33694 genes) — **a person could salvage it:** join the labels in data/fetched/geo/GSE119450/GSE119450_RAW.tar-contents/GSM3375487_D1N_CellBC_sgRNA.csv.gz on barcode
- `GSM3375484_D1S_matrix__matrix.mtx-table.csv` — **supplementary** — review: measured transcriptome, usable as evidence for the screen; converted from matrix.mtx (2000 of 15829 cells, 33694 of 33694 genes) — **a person could salvage it:** join the labels in data/fetched/geo/GSE119450/GSE119450_RAW.tar-contents/GSM3375488_D1S_CellBC_sgRNA.csv.gz on barcode
- `GSM3375485_D2N_matrix__matrix.mtx-table.csv` — **supplementary** — review: measured transcriptome, usable as evidence for the screen; converted from matrix.mtx (2000 of 11486 cells, 33694 of 33694 genes) — **a person could salvage it:** join the labels in data/fetched/geo/GSE119450/GSE119450_RAW.tar-contents/GSM3375489_D2N_CellBC_sgRNA.csv.gz on barcode
- `GSM3375486_D2S_matrix__matrix.mtx-table.csv` — **supplementary** — review: measured transcriptome, usable as evidence for the screen; converted from matrix.mtx (2000 of 13816 cells, 33694 of 33694 genes) — **a person could salvage it:** join the labels in data/fetched/geo/GSE119450/GSE119450_RAW.tar-contents/GSM3375490_D2S_CellBC_sgRNA.csv.gz on barcode
- `GSE153056_ECCITE_metadata.tsv.gz` — **unusable** — it cannot be evidence for the same model: it shares only 0 measured column(s) with the gold, and 2 are needed to correct the same model (the gold has 2 feature(s)) (the model had said gold); left out Unnamed: 0, guide_ID, NT — **a person could salvage it:** join this metadata to the GSE153056 gene-by-cell counts matrix on the cell barcode (first column) to attach the transcriptome
- `GSE119450_RAW.tar` — **unusable** — review: not a table: 0 parsed column(s). A table needs at least a feature column and something else
- `GSM3375483_D1N_matrix.tar.gz` — **unusable** — review: not a table: 1 parsed column(s). A table needs at least a feature column and something else
- `GSM3375484_D1S_matrix.tar.gz` — **unusable** — review: not a table: 1 parsed column(s). A table needs at least a feature column and something else
- `GSM3375485_D2N_matrix.tar.gz` — **unusable** — review: not a table: 1 parsed column(s). A table needs at least a feature column and something else
- `GSM3375486_D2S_matrix.tar.gz` — **unusable** — review: not a table: 1 parsed column(s). A table needs at least a feature column and something else
- `barcodes.tsv` — **unusable** — review: not a table: 1 parsed column(s). A table needs at least a feature column and something else
- `matrix.mtx` — **unusable** — review: not a table: 1 parsed column(s). A table needs at least a feature column and something else
- `barcodes.tsv` — **unusable** — review: not a table: 1 parsed column(s). A table needs at least a feature column and something else
- `matrix.mtx` — **unusable** — review: not a table: 1 parsed column(s). A table needs at least a feature column and something else
- `barcodes.tsv` — **unusable** — review: not a table: 1 parsed column(s). A table needs at least a feature column and something else
- `matrix.mtx` — **unusable** — review: not a table: 1 parsed column(s). A table needs at least a feature column and something else
- `barcodes.tsv` — **unusable** — review: not a table: 1 parsed column(s). A table needs at least a feature column and something else
- `matrix.mtx` — **unusable** — review: not a table: 1 parsed column(s). A table needs at least a feature column and something else
- `GSE158055_sample_metadata.xlsx` — **unusable** — review: not a table: 0 parsed column(s). A table needs at least a feature column and something else
- `[Content_Types].xml` — **unusable** — review: not a table: 1 parsed column(s). A table needs at least a feature column and something else
- `.rels` — **unusable** — review: not a table: 1 parsed column(s). A table needs at least a feature column and something else
- `app.xml` — **unusable** — review: not a table: 1 parsed column(s). A table needs at least a feature column and something else
- `core.xml` — **unusable** — review: not a table: 1 parsed column(s). A table needs at least a feature column and something else
- `custom.xml` — **unusable** — review: not a table: 1 parsed column(s). A table needs at least a feature column and something else
- `workbook.xml.rels` — **unusable** — review: not a table: 1 parsed column(s). A table needs at least a feature column and something else
- `comments1.xml` — **unusable** — review: not a table: 1 parsed column(s). A table needs at least a feature column and something else
- `sharedStrings.xml` — **unusable** — review: not a table: 1 parsed column(s). A table needs at least a feature column and something else
- `styles.xml` — **unusable** — review: not a table: 1 parsed column(s). A table needs at least a feature column and something else
- `workbook.xml` — **unusable** — review: not a table: 1 parsed column(s). A table needs at least a feature column and something else
- `sheet1.xml.rels` — **unusable** — review: not a table: 1 parsed column(s). A table needs at least a feature column and something else
- `sheet1.xml` — **unusable** — review: not a table: 1 parsed column(s). A table needs at least a feature column and something else
- `genes.tsv` — **unusable** — review: This is a two-column gene annotation table (Ensembl gene id, gene symbol) with no header and no cells, no perturbations and no expression values. It is a feature reference, not a per-cell scr…; left out ENSG00000243485, RP11-34P13.3
- `GSE153056_CITE_metadata.tsv.gz` — **unusable** — review: This is a per-cell CITE-seq metadata table (barcodes with RNA/ADT/HTO QC columns) from a 5' titration experiment. It has no gRNA.ID column and no gene-expression columns; the perturbation ide…; left out Unnamed: 0 — **a person could salvage it:** join the labels in data/fetched/geo/GSE153056/GSE153056_ECCITE_metadata.tsv.gz on barcode
- `GSE158055_cell_annotation.csv.gz` — **unusable** — review: This is a cell-annotation metadata table (cellName, sampleID, celltype, majorType) for GSE158055, a COVID-19 PBMC atlas. It has no gRNA.ID column and no gene-expression columns; rows are cell…; left out cellName — **a person could salvage it:** join this annotation to the GSE158055 counts matrix on cellName to obtain unperturbed reference cells
- `vmlDrawing1.vml` — **unusable** — review: The file is a VML drawing part (vmlDrawing1.vml) from an xlsx archive, not a data table. It contains Excel comment-shape XML, not rows of cells, so there is no gRNA.ID column and no measureme…; left out <xml xmlns:v="urn:schemas-microsoft-com:vml" xmlns:o="urn:schemas-microsoft-com:office:office" xmlns:x="urn:schemas-microsoft-com:office:excel" xmlns:w10="urn:schemas-microsoft-com:office:word"><v:shapetype id="_x0000_t202" coordsize="21600, 21600" o:spt="202" path="m, l, 21600l21600 — **a person could salvage it:** unzip the source .xlsx workbook and use the sheet/counts matrix it contains, ignoring the vmlDrawing*.vml parts
- `geo://GSE314342` — **not downloaded** — the model did not keep it
- `geo://GSE185655` — **not downloaded** — the model did not keep it
- `geo://GSE153056#matrix/GSE153056-GPL24676_series_matrix.txt.gz` — **not downloaded** — error: SourceError: GSE153056's series matrix has a header and no rows, which is normal for RNA-seq series. Supplementary files (smallest first): suppl/filelist.txt (1.7 kB), matrix/GSE153056-GPL1857…
- `geo://GSE158055#suppl/GSE158055_covid19_counts.mtx.gz` — **not downloaded** — error: SourceError: https://ftp.ncbi.nlm.nih.gov/geo/series/GSE158nnn/GSE158055/suppl/GSE158055_covid19_counts.mtx.gz exceeds the 2,147,483,648-byte cap; raise --max-bytes or fetch a smaller file

## Round 1 -- the labeled table

1. **keyword search** `CROP-seq primary human T cells` — 6 hit(s): None CROP-Seq in Primary Human T Cells; geo://GSE119450 CROP-Seq in Primary Human T Cells; None CROP-seq of 2 month old brain organoids; None Amplicon sequencing data from CROP-seq in 2 month old …; None Single-cell RNA sequencing and single-cell CRISPR scre…; geo://GSE266618 Systematic discovery of CRISPR-boosted CAR …
1. **keyword search** `Perturb-seq human CD4 T cells` — 11 hit(s): geo://GSE314342 Genome-scale perturb-seq in primary human C…; None; None Centralized control of dynamic gene regulatory circuit…; None Transcriptomic effects of rs4845604, an IBD and allerg…; None Disordered protein LAT encodes relative levels of sign…; hf://jamesjazz/perturb_seq jamesjazz/perturb_seq; ... (+5)
1. **keyword search** `CRISPR knockout T cells` — 15 hit(s): None Proteomic analysis of osteocyte line (MLO-Y4) with kno…; None scRNA-seq CRISPR screening of human posterior brain or…; geo://GSE71594 Transcriptome of iPSC-derived Neural Cells w…; None CRISPR screen exploring resistance to AKTi (Capivasert…; None Tumor cell-intrinsic Decr2 regulates ferroptosis and i…; hf://perturbai/wholebrain_crispr_atlas perturbai/wholebrain…; ... (+9)
1. **keyword search** `single-cell CRISPR screen T cells` — 15 hit(s): None Single-cell RNA sequencing and single-cell CRISPR scre…; None scRNA-seq CRISPR screening of human posterior brain or…; None CRISPR screen exploring resistance to AKTi (Capivasert…; None Mapping the functional impact of non-coding regulatory…; None Single cell RNA-seq of pancreatic mouse tumors upon tr…; hf://ratschlab/HEST_Visium_virtual_single_cell_transcriptom…; ... (+9)
1. **proposed** `geo://GSE314342` (confidence 0.90) — This is the labeled CRISPR knockout Perturb-seq screen in primary human CD4+ T cells, matching the required cell type and assay (CRISPRko), providing the pertur
1. **proposed** `geo://GSE119450` (confidence 0.75) — A well-known single-cell CRISPR knockout screen in primary human T cells; it provides an additional labeled CRISPRko perturbation dataset in the same cell type,
1. **listing** `geo://GSE314342` — kept
   - 55 file(s) on offer, smallest 13.0 kB, total 159.1 GB # about: title: Genome-scale perturb-seq in primary human CD4+ T cells reveals genes regulating T cell programs and human immune traits. | summar…
1. **listing** `geo://GSE119450` — kept
   - 3 file(s) on offer, smallest 2.3 kB, total 328.0 MB # about: title: CROP-Seq in Primary Human T Cells | summary: We adapted sgRNA lentiviral infection and Cas9 electroporation (SLICE) to allow for CR…
1. **chosen** `geo://GSE119450#suppl/GSE119450_RAW.tar` — This is the raw single-cell CRISPR knockout Perturb-seq data in primary human T cells, providing the labeled perturbation measurements needed as supplementary evidence.
1. **dropped** `geo://GSE314342` — the model did not keep it
1. **downloaded** `geo://GSE119450#suppl/GSE119450_RAW.tar` → GSM3375483_D1N_matrix.tar.gz-contents/matrix.mtx, GSM3375484_D1S_matrix.tar.gz-contents/matrix.mtx, GSM3375485_D2N_matrix.tar.gz-contents/matrix.mtx, GSM3375486_D2S_matrix.tar.gz-contents/matrix.mtx
1. **screened out** `matrix.mtx` — a matrix-market (sparse) file, not a table: it becomes one only together with its barcodes/features companion files (and a cell label table), which were not sta
1. **screened out** `matrix.mtx` — a matrix-market (sparse) file, not a table: it becomes one only together with its barcodes/features companion files (and a cell label table), which were not sta
1. **screened out** `matrix.mtx` — a matrix-market (sparse) file, not a table: it becomes one only together with its barcodes/features companion files (and a cell label table), which were not sta
1. **screened out** `matrix.mtx` — a matrix-market (sparse) file, not a table: it becomes one only together with its barcodes/features companion files (and a cell label table), which were not sta
1. **screened out** `GSE119450_RAW.tar` — not readable as a delimited text table: UnicodeDecodeError: 'utf-8' codec can't decode byte 0x8b in position 513: invalid start byte
1. **screened out** `GSM3375483_D1N_matrix.tar.gz` — parsed as a single column; this file is probably not a plain table (a matrix with a preamble, or a non-tabular artifact)
1. **screened out** `GSM3375484_D1S_matrix.tar.gz` — parsed as a single column; this file is probably not a plain table (a matrix with a preamble, or a non-tabular artifact)
1. **screened out** `GSM3375485_D2N_matrix.tar.gz` — parsed as a single column; this file is probably not a plain table (a matrix with a preamble, or a non-tabular artifact)
1. **screened out** `GSM3375486_D2S_matrix.tar.gz` — parsed as a single column; this file is probably not a plain table (a matrix with a preamble, or a non-tabular artifact)
1. **screened out** `barcodes.tsv` — parsed as a single column; this file is probably not a plain table (a matrix with a preamble, or a non-tabular artifact)
1. **screened out** `barcodes.tsv` — parsed as a single column; this file is probably not a plain table (a matrix with a preamble, or a non-tabular artifact)
1. **screened out** `barcodes.tsv` — parsed as a single column; this file is probably not a plain table (a matrix with a preamble, or a non-tabular artifact)
1. **screened out** `barcodes.tsv` — parsed as a single column; this file is probably not a plain table (a matrix with a preamble, or a non-tabular artifact)

## Round 2 -- supplementary evidence for that table

1. **proposed** `geo://GSE185655` (confidence 0.55) — Same measurement type as the labeled table: per-cell gene expression columns (the same gene symbols like A1BG, A2M, AAK1, etc.) plus sgRNA assignment, but for d
1. **proposed** `geo://GSE153056` (confidence 0.50) — Supplies the same gene-expression measurement columns (same gene symbol space) for many more unperturbed CD4+ T cell rows, serving as the large unperturbed refe
1. **proposed** `geo://GSE158055` (confidence 0.45) — Shares the same per-cell gene expression column space (gene symbols) and provides many additional CD4+ T cell rows from another cohort/site, usable as supplemen
1. **listing** `geo://GSE185655` — kept
   - 5 file(s) on offer, smallest 2.5 kB, total 1.8 MB # about: title: Regulation of hepatocellular carcinoma growth by UPF1 variants not through nonsense-mediated mRNA decay | summary: Multiple studies s…
1. **listing** `geo://GSE153056` — kept
   - 7 file(s) on offer, smallest 1.7 kB, total 124.0 MB # about: title: Characterizing the molecular regulation of inhibitory immune checkpoints with multi-modal single-cell screens. | summary: The expre…
1. **listing** `geo://GSE158055` — kept
   - 8 file(s) on offer, smallest 3.4 kB, total 7.7 GB # about: title: Large-scale single-cell analysis reveals critical immune characteristics of COVID-19 patients | summary: We profiled scRNA-seq of 284…
1. **chosen** `geo://GSE153056#suppl/GSE153056_ECCITE_metadata.tsv.gz` — Provides per-cell metadata (including sgRNA assignment and cell types) for the ECCITE-seq CRISPR screen in primary human T cells, matching the perturbation measurement space neede…
1. **chosen** `geo://GSE158055#suppl/GSE158055_cell_annotation.csv.gz` — Supplies cell-type annotations for a large unperturbed CD4+ T cell atlas, enabling selection of unperturbed reference cells as supplementary evidence.
1. **listing** `geo://GSE153056#matrix/GSE153056-GPL24676_series_matrix.txt.gz` — kept
   - keeping GSE153056-GPL24676_series_matrix.txt.gz as well -- a screen's measurements live in the counts file, and the label table does not carry them
1. **listing** `geo://GSE153056#suppl/GSE153056_CITE_metadata.tsv.gz` — kept
   - keeping GSE153056_CITE_metadata.tsv.gz as well -- this task's labels (a perturbation screen's condition/guide) usually live in the annotated file, not the raw matrix
1. **listing** `geo://GSE158055#suppl/GSE158055_covid19_counts.mtx.gz` — kept
   - keeping GSE158055_covid19_counts.mtx.gz as well -- a screen's measurements live in the counts file, and the label table does not carry them
1. **listing** `geo://GSE158055#suppl/GSE158055_sample_metadata.xlsx` — kept
   - keeping GSE158055_sample_metadata.xlsx as well -- this task's labels (a perturbation screen's condition/guide) usually live in the annotated file, not the raw matrix
1. **dropped** `geo://GSE185655` — the model did not keep it
1. **downloaded** `geo://GSE153056#suppl/GSE153056_ECCITE_metadata.tsv.gz` → GSE153056_ECCITE_metadata.tsv.gz
1. **download failed** `geo://GSE153056#matrix/GSE153056-GPL24676_series_matrix.txt.gz` — error: SourceError: GSE153056's series matrix has a header and no rows, which is normal for RNA-seq series. Supplementary files (smallest first): suppl/filelist.txt (1.7 kB), matrix/GSE153056-GPL1857…
1. **downloaded** `geo://GSE153056#suppl/GSE153056_CITE_metadata.tsv.gz` → GSE153056_CITE_metadata.tsv.gz
1. **downloaded** `geo://GSE158055#suppl/GSE158055_cell_annotation.csv.gz` → GSE158055_cell_annotation.csv.gz
1. **download failed** `geo://GSE158055#suppl/GSE158055_covid19_counts.mtx.gz` — error: SourceError: https://ftp.ncbi.nlm.nih.gov/geo/series/GSE158nnn/GSE158055/suppl/GSE158055_covid19_counts.mtx.gz exceeds the 2,147,483,648-byte cap; raise --max-bytes or fetch a smaller file
1. **downloaded** `geo://GSE158055#suppl/GSE158055_sample_metadata.xlsx` → GSE158055_sample_metadata.xlsx, GSE158055_sample_metadata.xlsx-contents/_rels/.rels, GSE158055_sample_metadata.xlsx-contents/xl/workbook.xml, GSE158055_sample_metadata.xlsx-contents/xl/styles.xml
1. **screened out** `GSE158055_sample_metadata.xlsx` — not readable as a delimited text table: UnicodeDecodeError: 'utf-8' codec can't decode byte 0xb1 in position 12: invalid start byte
1. **screened out** `.rels` — parsed as a single column; this file is probably not a plain table (a matrix with a preamble, or a non-tabular artifact)
1. **screened out** `workbook.xml` — parsed as a single column; this file is probably not a plain table (a matrix with a preamble, or a non-tabular artifact)
1. **screened out** `styles.xml` — parsed as a single column; this file is probably not a plain table (a matrix with a preamble, or a non-tabular artifact)
1. **screened out** `sheet1.xml.rels` — parsed as a single column; this file is probably not a plain table (a matrix with a preamble, or a non-tabular artifact)
1. **screened out** `workbook.xml.rels` — parsed as a single column; this file is probably not a plain table (a matrix with a preamble, or a non-tabular artifact)
1. **screened out** `sharedStrings.xml` — parsed as a single column; this file is probably not a plain table (a matrix with a preamble, or a non-tabular artifact)
1. **screened out** `comments1.xml` — parsed as a single column; this file is probably not a plain table (a matrix with a preamble, or a non-tabular artifact)
1. **screened out** `core.xml` — parsed as a single column; this file is probably not a plain table (a matrix with a preamble, or a non-tabular artifact)
1. **screened out** `app.xml` — parsed as a single column; this file is probably not a plain table (a matrix with a preamble, or a non-tabular artifact)
1. **screened out** `custom.xml` — parsed as a single column; this file is probably not a plain table (a matrix with a preamble, or a non-tabular artifact)
1. **screened out** `[Content_Types].xml` — parsed as a single column; this file is probably not a plain table (a matrix with a preamble, or a non-tabular artifact)
1. **screened out** `sheet1.xml` — parsed as a single column; this file is probably not a plain table (a matrix with a preamble, or a non-tabular artifact)

**Refusals by cause:** not a table (fewer than two columns) ×26, the model judged it unusable ×4, cannot supply the columns the gold has ×1
