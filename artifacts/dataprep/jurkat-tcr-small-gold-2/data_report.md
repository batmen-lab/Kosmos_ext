# Data report

**Question:** In Jurkat T cells, can the transcriptional response to CRISPR knockout of T cell receptor signaling genes not seen in training be predicted from a small labeled screen, using large single-cell profiles of Jurkat cells from other CRISPR screens, with their labels ignored, as supplementary evidence?

**Tables:** 0 considered — gold **0**, supplementary **0**, unusable **0**, waiting for a person **0**

> **No table got a role.** Files were downloaded and screened (see the rounds below), but none could serve as the labeled table, so the plan assigned no roles. The plan's own notes:
> - no target column decided, so no table has a role yet; decide the target and rebuild the plan
> - profiled data/fetched/geo/GSE249595/GSE249595_guides_reference.fa.gz: 1 columns
> - profiled data/fetched/geo/GSE137554/GSE137554_raw_gene_bc_matrices_h5.h5: 0 columns
> - profiled data/fetched/geo/GSE137554/GSE137554_CellAnnotation.tsv.gz: 2 columns

## Datasets

- `geo://GSE249595#suppl/GSE249595_RAW.tar` — **not downloaded** — error: SourceError: https://ftp.ncbi.nlm.nih.gov/geo/series/GSE249nnn/GSE249595/suppl/GSE249595_RAW.tar exceeds the 2,147,483,648-byte cap; raise --max-bytes or fetch a smaller file

## Round 1 -- the labeled table

1. **keyword search** `Jurkat CRISPR knockout Perturb-seq` — 11 hit(s): None Microarray dataset of Jurkat cells following miR-93 ov…; geo://GSE37212 Identification of miR-21 targets in Jurkat c…; None Lipidomic characterization of Jurkat-derived T cell li…; None Metabolomics identifies the intersection of phosphoeth…; None Kinome analysis of CD151 overexpressing Jurkat T cells; hf://VirtualCell2025/jurkat.h5 VirtualCell2025/jurkat.h5; ... (+5)
1. **keyword search** `T cell receptor signaling CRISPR screen` — 12 hit(s): None Single-cell RNA sequencing and single-cell CRISPR scre…; None AMBRA1 controls the translation of immune-specific gen…; geo://GSE70038 Genome-wide CRISPR-Cas9 screens reveal loss …; None Disruption of the CRL5 ubiquitin ligase complex enhanc…; None Genome-wide CRISPR screen identifies FAM49B as a key r…; hf://datasocial/tiktok-5.6B-videos datasocial/tiktok-5.6B-v…; ... (+6)
1. **keyword search** `Jurkat single-cell CRISPR screen` — 10 hit(s): None Single-cell CRISPR Screen for HIV Host Dependency Fact…; None 10x 3’ scRNA-seq analysis on Jurkat cells post TCR aci…; None NeoPAIR-T: Functional Mapping of Neoantigen-TCR Pairs …; None Inhibition of a Chromatin and Transcription Modulator,…; None A Virus-Packageable CRISPR System Identifies Host Depe…; hf://VirtualCell2025/jurkat.h5 VirtualCell2025/jurkat.h5; ... (+4)
1. **keyword search** `Jurkat knockout transcriptional response` — 10 hit(s): None Transcriptional regulators of the amino acid response …; geo://GSE145783 Transcriptional regulators of the amino aci…; geo://GSE5346 Transcription profiling of human jurkat cell …; None MAML1-Dependent Notch-Responsive Genes Exhibit Differi…; None HuR controls apoptosis and activation response without…; hf://VirtualCell2025/jurkat.h5 VirtualCell2025/jurkat.h5; ... (+4)
1. **keyword search** `T cell receptor signaling Perturb-seq` — 15 hit(s): None Genome-scale CRISPRi Perturb-seq on both unstimulated …; None Disordered protein LAT encodes relative levels of sign…; None Disordered protein LAT encodes relative levels of sign…; geo://GSE249595 Genome-scale CRISPRi Perturb-seq on both un…; geo://GSE22928 Genome-Wide Analysis of Somatic Copy Number …; hf://datasocial/tiktok-5.6B-videos datasocial/tiktok-5.6B-v…; ... (+9)
1. **keyword search** `T cell receptor signaling Jurkat` — 13 hit(s): None Transcription profiling of Jurkat T cell line and mous…; geo://GSE46909 Expression data from human Jurkat T cells ex…; geo://GSE37212 Identification of miR-21 targets in Jurkat c…; None Dynamic Palmitoylation Events Following T-Cell Recepto…; geo://GSE24026 Comparison of gene expression profiles of Ju…; hf://datasocial/tiktok-5.6B-videos datasocial/tiktok-5.6B-v…; ... (+7)
1. **proposed** `geo://GSE249595` (confidence 0.85) — This is the labeled CRISPR perturbation screen in Jurkat T cells directly matching the research question: a CRISPR knockout/knockdown screen (CRISPRi Perturb-se
1. **proposed** `geo://GSE137554` (confidence 0.60) — A Jurkat single-cell CRISPR screen (CROP-seq) with TCR stimulation, providing additional labeled perturbation data in the same cell type and assay family (CRISP
1. **listing** `geo://GSE249595` — kept
   - 5 file(s) on offer, smallest 5.1 kB, total 3.5 GB # about: title: Genome-scale CRISPRi Perturb-seq on both unstimulated and stimulated Jurkat cells reveals regulators of human T-cell signaling | summ…
1. **listing** `geo://GSE137554` — kept
   - 3 file(s) on offer, smallest 2.8 kB, total 95.1 MB # about: title: CROP-seq analysis of TCR stimulation using the 10x Genomics platform | summary: T cell receptor (TCR) signaling in Jurkat cells was …
1. **chosen** `geo://GSE249595#suppl/GSE249595_RAW.tar` — This is the labeled CRISPRi Perturb-seq Jurkat screen with per-cell expression and guide labels needed to train the predictor on TCR signaling knockouts.
1. **chosen** `geo://GSE137554#suppl/GSE137554_raw_gene_bc_matrices_h5.h5` — This is the smallest labeled Jurkat CROP-seq single-cell expression matrix with perturbation labels, providing supplementary labeled training data in the same cell type and assay …
1. **listing** `geo://GSE249595#suppl/GSE249595_guides_reference.fa.gz` — kept
   - keeping GSE249595_guides_reference.fa.gz as well -- this task's labels (a perturbation screen's condition/guide) usually live in the annotated file, not the raw matrix
1. **listing** `geo://GSE137554#suppl/GSE137554_CellAnnotation.tsv.gz` — kept
   - keeping GSE137554_CellAnnotation.tsv.gz as well -- this task's labels (a perturbation screen's condition/guide) usually live in the annotated file, not the raw matrix
1. **download failed** `geo://GSE249595#suppl/GSE249595_RAW.tar` — error: SourceError: https://ftp.ncbi.nlm.nih.gov/geo/series/GSE249nnn/GSE249595/suppl/GSE249595_RAW.tar exceeds the 2,147,483,648-byte cap; raise --max-bytes or fetch a smaller file
1. **downloaded** `geo://GSE249595#suppl/GSE249595_guides_reference.fa.gz` → GSE249595_guides_reference.fa.gz
1. **downloaded** `geo://GSE137554#suppl/GSE137554_raw_gene_bc_matrices_h5.h5` → GSE137554_raw_gene_bc_matrices_h5.h5
1. **downloaded** `geo://GSE137554#suppl/GSE137554_CellAnnotation.tsv.gz` → GSE137554_CellAnnotation.tsv.gz
1. **screened out** `GSE249595_guides_reference.fa.gz` — parsed as a single column; this file is probably not a plain table (a matrix with a preamble, or a non-tabular artifact)
1. **screened out** `GSE137554_raw_gene_bc_matrices_h5.h5` — not readable as a delimited text table: UnicodeDecodeError: 'utf-8' codec can't decode byte 0x89 in position 0: invalid start byte
