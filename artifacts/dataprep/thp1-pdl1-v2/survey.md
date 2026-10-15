# Survey

**Question:** In THP-1 cells, can the transcriptional response to CRISPR knockout of regulators of PD-L1 expression not seen in training be predicted from a small labeled single-cell screen, using large single-cell profiles of THP-1 cells from other experiments, with their labels ignored, as supplementary evidence?

- cell type: `THP-1` (aliases: acute monocytic leukemia cells, monocytic leukemia cell line, human monocytic cells), organism `human`, modality `CRISPRko`, topic `PD-L1 regulation`
- 25 series described, 379s, nothing downloaded

## Fit: gold size against the supplementary pool

| gold series | gold samples | gold cells | chemistry | supp samples | supp cells | of which unlabeled | same chemistry | supp / gold | enough (>= 2x) |
|---|---|---|---|---|---|---|---|---|---|
| geo://GSE153056 | 2 | 23,706 (+1 unknown) | 5_prime_gex | 11 | 156,041 | 102,149 | 0 | 6.6 | yes |

## Gold candidates (single-cell screens of the right cells, with labels)

| sample | series | title | modality | topic | chemistry | download | cells |
|---|---|---|---|---|---|---|---|
| GSM4633608 | geo://GSE153056 | ECCITE-seq Arrayed cDNA: THP-1; Acute Myeloid Leukemia cell  | CRISPRko | 0.95 | 5_prime_gex | 39.8 MB | 23,706 (scBaseCount) |
| GSM4633614 | geo://GSE153056 | ECCITE-seq cDNA: THP-1; Acute Myeloid Leukemia cell line THP | CRISPRko | 0.95 | - | 64.0 MB | ? |

## Supplementary candidates (single-cell, same cell type; GEO largest first, then scBaseCount)

| sample | series | title | modality | topic | chemistry | download | cells |
|---|---|---|---|---|---|---|---|
| GSM9084977 | geo://GSE301541 | THP-1 WT cells replicate 4; THP-1; cell line: THP-1; genotyp | none | 0.3 | - | 211.0 MB | ~17,246 |
| GSM9084980 | geo://GSE301541 | THP-1 FOLR2 KO cells replicate 3; THP-1; cell line: THP-1; g | CRISPRko | 0.4 | - | 163.8 MB | ~15,090 |
| GSM9084981 | geo://GSE301541 | THP-1 FOLR2 KO cells replicate 4; THP-1; cell line: THP-1; g | CRISPRko | 0.4 | - | 170.1 MB | ~15,090 |
| GSM9084976 | geo://GSE301541 | THP-1 WT cells replicate 3; THP-1; cell line: THP-1; genotyp | none | 0.3 | - | 139.7 MB | ~13,473 |
| GSM9084979 | geo://GSE301541 | THP-1 FOLR2 KO cells replicate 2; THP-1; cell line: THP-1; g | CRISPRko | 0.4 | - | 99.8 MB | ~12,395 |
| GSM9084975 | geo://GSE301541 | THP-1 WT cells replicate 2; THP-1; cell line: THP-1; genotyp | none | 0.3 | - | 108.2 MB | ~11,856 |
| GSM9084978 | geo://GSE301541 | THP-1 FOLR2 KO cells replicate 1; THP-1; cell line: THP-1; g | CRISPRko | 0.4 | - | 88.3 MB | ~11,317 |
| GSM9084974 | geo://GSE301541 | THP-1 WT cells replicate 1; THP-1; cell line: THP-1; genotyp | none | 0.3 | - | 90.4 MB | ~10,240 |
| GSM8288773 | geo://GSE268256 | pDNA_12h_scRNA-seq; THP-1; cell line: THP-1; cell type: Mono | none | 0.3 | - | 116.6 MB | ~9,701 |
| GSM4633605 | geo://GSE153056 | CITE-seq cDNA: THP-1; Acute Myeloid Leukemia cell line THP-1 | none | 0.3 | - | 23.1 MB | ? |
| GSE176294 | geo://GSE176294 | Revealing immune responses in the Mycobacterium avium subsp. | none | 0.1 | 3_prime_gex | 73.7 MB | 39,633 (scBaseCount) |
| GSE249672 | geo://GSE249672 | A novel pancreatic ductal adenocarcinoma organoid recapitula | none | 0.2 | - | 2.3 MB | ? |

## Rejected / companion samples: 153

- GSM4633606 (geo://GSE153056) — companion: ADT counts only, serves CITE-seq sample, not expression.
- GSM4633607 (geo://GSE153056) — companion: HTO counts only, serves CITE-seq sample, not expression.
- GSM4633609 (geo://GSE153056) — companion: ADT counts only, serves arrayed ECCITE-seq screen.
- GSM4633610 (geo://GSE153056) — companion: HTO counts only, serves arrayed ECCITE-seq screen.
- GSM4633611 (geo://GSE153056) — companion: HTO barcodes only, serves arrayed ECCITE-seq screen.
- GSM4633612 (geo://GSE153056) — companion: GDO guide counts only, serves arrayed ECCITE-seq screen.
- GSM4633613 (geo://GSE153056) — companion: GDO guide counts only, serves arrayed ECCITE-seq screen.
- GSM4633615 (geo://GSE153056) — companion: ADT counts only, serves ECCITE-seq screen.
- GSM4633616 (geo://GSE153056) — companion: HTO counts only, serves ECCITE-seq screen.
- GSM4633617 (geo://GSE153056) — companion: HTO barcodes only, serves ECCITE-seq screen.
- GSM4633618 (geo://GSE153056) — companion: GDO guide counts only, serves ECCITE-seq screen.
- GSM9829757 (geo://GSE336192) — reject: Bulk RNA-seq in BxPC-3, not THP-1 single-cell; wrong cell type.
- GSM9829758 (geo://GSE336192) — reject: Bulk RNA-seq in BxPC-3, not THP-1 single-cell; wrong cell type.
- GSM9829759 (geo://GSE336192) — reject: Bulk RNA-seq in BxPC-3, not THP-1 single-cell; wrong cell type.
- GSM9829760 (geo://GSE336192) — reject: Bulk RNA-seq in BxPC-3, not THP-1 single-cell; wrong cell type.
- GSM9829761 (geo://GSE336192) — reject: Bulk CRISPRko in BxPC-3, not THP-1 single-cell; wrong cell type.
- GSM9829762 (geo://GSE336192) — reject: Bulk CRISPRko in BxPC-3, not THP-1 single-cell; wrong cell type.
- GSM9829763 (geo://GSE336192) — reject: Bulk CRISPRko in BxPC-3, not THP-1 single-cell; wrong cell type.
- GSM9829764 (geo://GSE336192) — reject: Bulk CRISPRko in BxPC-3, not THP-1 single-cell; wrong cell type.
- GSM8402311 (geo://GSE255700) — reject: Bulk HiChIP in BxPC-3, wrong cell type and not expression or single-cell.
- GSM8402312 (geo://GSE255700) — reject: Bulk HiChIP in BxPC-3, wrong cell type and not expression or single-cell.
- GSM8402313 (geo://GSE255700) — reject: Bulk HiChIP in BxPC-3, wrong cell type and not expression or single-cell.
- GSM8402314 (geo://GSE255700) — reject: Bulk HiChIP in BxPC-3, wrong cell type and not expression or single-cell.
- GSM8402315 (geo://GSE255700) — reject: Bulk HiChIP in BxPC-3, wrong cell type and not expression or single-cell.
- GSM8402316 (geo://GSE255700) — reject: Bulk HiChIP in BxPC-3, wrong cell type and not expression or single-cell.
- GSM8402317 (geo://GSE255700) — reject: Bulk HiChIP in BxPC-3, wrong cell type and not expression or single-cell.
- GSM8402318 (geo://GSE255700) — reject: Bulk HiChIP in BxPC-3, wrong cell type and not expression or single-cell.
- GSM8402319 (geo://GSE255700) — reject: Bulk HiChIP in BxPC-3, wrong cell type and not expression or single-cell.
- GSM8402320 (geo://GSE255700) — reject: Bulk HiChIP in BxPC-3, wrong cell type and not expression or single-cell.
- GSM8402321 (geo://GSE255700) — reject: Bulk HiChIP in BxPC-3, wrong cell type and not expression or single-cell.
