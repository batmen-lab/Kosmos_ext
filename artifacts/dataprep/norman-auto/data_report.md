# Data report

**Question:** In K562 cells, can the transcriptional response to CRISPR activation (CRISPRa) of two genes together be predicted from the responses to each gene activated alone?

**Tables:** 0 considered — gold **0**, supplementary **0**, unusable **0**, waiting for a person **0**

## Datasets

- `geo://GSE194122` — **not downloaded** — the model did not keep it

## Round 1 -- the labeled table

1. **proposed** `geo://GSE194122` (confidence 0.60) — This is a K562 CRISPRa Perturb-seq dataset measuring transcriptional responses to activating individual genes; it provides the single-gene activation responses
1. **proposed** `geo://GSE90063` (confidence 0.50) — Same cell type (K562) and same gene panel context, providing an unperturbed/control expression atlas for co-expression graph construction, though its perturbati
1. **listing** `geo://GSE194122` — kept
   - 3 file(s) on offer, smallest 3.9 kB, total 3.3 GB # about: title: A sandbox for prediction and integration of DNA, RNA, and proteins in single cells | summary: Single-cell multiomics data collected f…
1. **listing** `geo://GSE90063` — kept
   - 7 file(s) on offer, smallest 2.0 kB, total 1.4 GB # about: title: Perturb-seq: Dissecting molecular circuits with scalable single cell RNA profiling of pooled genetic screens | summary: Methods: We c…
1. **chosen** `geo://GSE90063#suppl/GSE90063_k562_umi_wt.txt.gz` — This is the K562 single-cell UMI expression matrix (cells x genes) providing the unperturbed/control transcriptional measurements needed to build co-expression graphs and compare …
1. **listing** `geo://GSE90063#suppl/GSE90063_RAW.tar` — kept
   - keeping GSE90063_RAW.tar as well -- a screen's measurements live in the counts file, and the label table does not carry them
1. **dropped** `geo://GSE194122` — the model did not keep it
1. **downloaded** `geo://GSE90063#suppl/GSE90063_k562_umi_wt.txt.gz` → GSE90063_k562_umi_wt.txt.gz
1. **downloaded** `geo://GSE90063#suppl/GSE90063_RAW.tar` → GSE90063_RAW.tar, GSE90063_RAW.tar-contents/GSM2396856_dc_3hr.mtx.txt.gz, GSE90063_RAW.tar-contents/GSM2396856_dc_3hr_cbc_gbc_dict_lenient.csv.gz, GSE90063_RAW.tar-contents/GSM2396856_dc_3hr_cbc_gbc_dict_strict.csv.gz
1. **screened out** `GSE90063_RAW.tar` — not readable as a delimited text table: UnicodeDecodeError: 'utf-8' codec can't decode byte 0x8b in position 513: invalid start byte
1. **screened out** `GSM2396856_dc_3hr.mtx.txt.gz` — parsed as a single column; this file is probably not a plain table (a matrix with a preamble, or a non-tabular artifact)
1. **screened out** `GSM2396857_dc_0hr.mtx.txt.gz` — parsed as a single column; this file is probably not a plain table (a matrix with a preamble, or a non-tabular artifact)
1. **screened out** `GSM2396858_k562_tfs_7.mtx.txt.gz` — parsed as a single column; this file is probably not a plain table (a matrix with a preamble, or a non-tabular artifact)
1. **screened out** `GSM2396859_k562_tfs_13.mtx.txt.gz` — parsed as a single column; this file is probably not a plain table (a matrix with a preamble, or a non-tabular artifact)
1. **screened out** `GSM2396860_k562_tfs_highmoi.mtx.txt.gz` — parsed as a single column; this file is probably not a plain table (a matrix with a preamble, or a non-tabular artifact)
1. **screened out** `GSM2396861_k562_ccycle.mtx.txt.gz` — parsed as a single column; this file is probably not a plain table (a matrix with a preamble, or a non-tabular artifact)
