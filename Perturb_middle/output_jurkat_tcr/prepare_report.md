# Prepare report — Jurkat TCR

**Question:** In Jurkat T cells, can the transcriptional response to CRISPR knockout of T cell receptor signaling genes not seen in training be predicted from a small labeled screen, using large single-cell profiles of Jurkat cells from other CRISPR screens, with their labels ignored, as supplementary evidence?

## Gold: GSE92872 (Datlinger 2017 CROP-seq)

| | survey (before download) | actual |
|---|---|---|
| download | 19.4 MB | 18.2 MB |
| cells | ? | 5,905 in the file, 3,204 used |

| step | cells |
|---|---|
| gold cells in the file | 5,905 |
| in condition(s) stimulated | 3,259 |
| without DHODH, MVD, TUBB | 3,204 |

- 705 non-targeting controls (`CTRL`), 29 perturbed genes, 28-156 cells each (median 85); 36,722 genes; median 6,323 UMIs, ribosomal 16.1%

## Supplementary candidates: survey vs downloaded, and similarity to the gold controls

Pooled when the Pearson correlation of mean log1p-CPM (shared expressed genes) with the gold controls is >= 0.6; a series is judged on its first sample and at most 2 of its samples are pooled. Download 2.50 GB of the 3 GB budget.

| sample | series | download (survey) | cells (survey) | barcodes | cells (>= 1000 UMIs) | Pearson | Spearman | median UMIs | ribosomal | decision |
|---|---|---|---|---|---|---|---|---|---|---|
| GSM7719390 | GSE241207 | 127.3 MB | ~14,282 | 10,773 | 10,564 | 0.87 | 0.76 | 9,606 | 21.0% | pooled |
| GSM7719391 | GSE241207 | 91.6 MB | ~11,856 | 8,816 | 8,753 | 0.88 | 0.78 | 7,402 | 22.3% | pooled |
| GSM7951413 | GSE249595 | 235.8 MB | ? | 6,794,880 | 54,697 | 0.24 | 0.40 | 2,837 | 1.3% | dropped: Pearson 0.24 < 0.6 |
| GSM7951414 | GSE249595 | 222.7 MB | ? | - | - | - | - | - | - | skipped: its series failed on an earlier sample |
| GSM7951415 | GSE249595 | 228.2 MB | ? | - | - | - | - | - | - | skipped: its series failed on an earlier sample |
| GSM7951416 | GSE249595 | 210.0 MB | ? | - | - | - | - | - | - | skipped: its series failed on an earlier sample |
| GSM7951417 | GSE249595 | 179.3 MB | ? | - | - | - | - | - | - | skipped: its series failed on an earlier sample |
| GSM7951418 | GSE249595 | 212.7 MB | ? | - | - | - | - | - | - | skipped: its series failed on an earlier sample |
| GSM7951419 | GSE249595 | 253.1 MB | ? | - | - | - | - | - | - | skipped: its series failed on an earlier sample |
| GSM7951420 | GSE249595 | 222.7 MB | ? | - | - | - | - | - | - | skipped: its series failed on an earlier sample |
| GSM7951421 | GSE249595 | 233.5 MB | ? | - | - | - | - | - | - | skipped: its series failed on an earlier sample |
| GSM7951422 | GSE249595 | 244.1 MB | ? | - | - | - | - | - | - | skipped: its series failed on an earlier sample |
| GSM7951423 | GSE249595 | 237.2 MB | ? | - | - | - | - | - | - | skipped: its series failed on an earlier sample |
| GSM7951424 | GSE249595 | 239.1 MB | ? | - | - | - | - | - | - | skipped: its series failed on an earlier sample |
| GSM7951425 | GSE249595 | 272.7 MB | ? | - | - | - | - | - | - | skipped: its series failed on an earlier sample |
| GSM7951426 | GSE249595 | 257.0 MB | ? | - | - | - | - | - | - | skipped: its series failed on an earlier sample |
| GSM7951427 | GSE249595 | 253.3 MB | ? | - | - | - | - | - | - | skipped: its series failed on an earlier sample |
| GSM7951428 | GSE249595 | 278.4 MB | ? | - | - | - | - | - | - | skipped: its series failed on an earlier sample |
| hf://VirtualCell2025/jurkat.h5 | VirtualCell2025/jurkat.h5 | 1.55 GB | ? | 21,412 | 21,412 | 0.81 | 0.78 | 4,548 | 0.0% | pooled |
| GSM8187448 | GSE263194 | 217.0 MB | ? | 1,786,407 | 9,999 | 0.89 | 0.79 | 9,730 | 19.1% | pooled |
| GSM8187449 | GSE263194 | 221.5 MB | ? | 1,895,064 | 9,570 | 0.89 | 0.79 | 10,738 | 19.4% | pooled |
| GSM8187450 | GSE263194 | 202.6 MB | ? | - | - | - | - | - | - | skipped: 2 sample(s) of this series already pooled |
| GSM8187451 | GSE263194 | 188.8 MB | ? | - | - | - | - | - | - | skipped: 2 sample(s) of this series already pooled |
| GSM4483270 | GSE148820 | 121.0 MB | ~17,515 | 13,435 | 12,539 | 0.87 | 0.74 | 7,325 | 17.0% | pooled |

**Pooled:** 72,837 cells from 6 sample(s): GSM7719390_J1_1_matrix.mtx.gz (4,999 panel genes measured), GSM7719391_J_Lat_matrix.mtx.gz (4,999 panel genes measured), jurkat.h5 (4,480 panel genes measured), GSM8187448_DMSO_WCE_1_matrix.mtx.gz (4,818 panel genes measured), GSM8187449_DMSO_WCE_2_matrix.mtx.gz (4,818 panel genes measured), GSM4483270_mRNA_library.matrix.mtx.gz (4,844 panel genes measured)

## Panel and study

- 5,013 genes (top HVG among genes a pooled supp measures + 13 perturbed genes); log1p(CPM 10000); a panel gene a source does not measure is the gold control mean
- split `unseen_single`: test 25%, validation 20% of the rest; batch 32
