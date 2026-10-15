# Survey

**Question:** In human iPSC-derived neurons, can the transcriptional response to CRISPRi knockdown of genes not seen in training be predicted from a small labeled screen, using large single-cell profiles of iPSC-derived neurons from other experiments, with their labels ignored, as supplementary evidence?

- cell type: `iPSC-derived neurons` (aliases: induced pluripotent stem cell-derived neurons, human neurons, iPSC neurons, neural cells), organism `human`, modality `CRISPRi`, topic `-`
- 23 series described, 414s, nothing downloaded

## Fit: gold size against the supplementary pool

| gold series | gold samples | gold cells | chemistry | supp samples | supp cells | of which unlabeled | same chemistry | supp / gold | enough (>= 2x) |
|---|---|---|---|---|---|---|---|---|---|
| geo://GSE152988 | 4 | 62,383 | 3_prime_gex | 75 | 831,815 | 472,035 | 195,882 | 13.3 | yes |
| geo://GSE124703 | 2 | ? | - | 77 | 873,849 | 472,035 | - | ? | no |
| geo://GSE289420 | 1 | ? | - | 76 | 873,849 | 472,035 | - | ? | no |

## Gold candidates (single-cell screens of the right cells, with labels)

| sample | series | title | modality | topic | chemistry | download | cells |
|---|---|---|---|---|---|---|---|
| GSM4632025 | geo://GSE152988 | CROP-seq_CRISPRi_10x_lane_3; Human iPSC-derived neuron; cell | CRISPRi | 1 | 3_prime_gex | 69.2 MB | 12,387 (scBaseCount) |
| GSM4632023 | geo://GSE152988 | CROP-seq_CRISPRi_10x_lane_2; Human iPSC-derived neuron; cell | CRISPRi | 1 | 3_prime_gex | 77.6 MB | 16,658 (scBaseCount) |
| GSM4632022 | geo://GSE152988 | CROP-seq_CRISPRi_10x_lane_1; Human iPSC-derived neuron; cell | CRISPRi | 1 | 3_prime_gex | 79.7 MB | 15,617 (scBaseCount) |
| GSM4632026 | geo://GSE152988 | CROP-seq_CRISPRi_10x_lane_4; Human iPSC-derived neuron; cell | CRISPRi | 1 | 3_prime_gex | 88.1 MB | 17,721 (scBaseCount) |
| GSM3543623 | geo://GSE124703 | CROP-seq_Day 7_neuron_10x_lane_2; Human iPSC-derived neuron; | CRISPRi | 1 | - | 224.9 MB | ? |
| GSM3543622 | geo://GSE124703 | CROP-seq_Day 7_neuron_10x_lane_1; Human iPSC-derived neuron; | CRISPRi | 1 | - | 239.6 MB | ? |
| GSE289420 | geo://GSE289420 | Astrocyte-derived cholesterol drives synaptic gene expressio | CRISPRi | 0.6 | - | 19.9 MB | ? |

## Supplementary candidates (single-cell, same cell type; GEO largest first, then scBaseCount)

| sample | series | title | modality | topic | chemistry | download | cells |
|---|---|---|---|---|---|---|---|
| GSM9510613 | geo://GSE319096 | iGLUTs_D7_3182_mRNA_R4; iPSC derived cell; tissue: iPSC deri | CRISPRko | 0.6 | - | 345.4 MB | ~35,031 |
| GSM9510610 | geo://GSE319096 | iGLUTs_D7_3182_mRNA_R1; iPSC derived cell; tissue: iPSC deri | CRISPRko | 0.6 | - | 374.8 MB | ~32,336 |
| GSM9510620 | geo://GSE319096 | iGLUTs_D21_3182_mRNA_R3; iPSC derived cell; tissue: iPSC der | CRISPRko | 0.7 | - | 241.6 MB | ~31,797 |
| GSM9510611 | geo://GSE319096 | iGLUTs_D7_3182_mRNA_R2; iPSC derived cell; tissue: iPSC deri | CRISPRko | 0.6 | - | 366.4 MB | ~31,528 |
| GSM9510615 | geo://GSE319096 | iGABAs_D36_553_mRNA_R2; iPSC derived cell; tissue: iPSC deri | CRISPRko | 0.6 | - | 275.2 MB | ~31,258 |
| GSM8525895 | geo://GSE277604 | PGP1 ADNP Het; PGP1 (iPSC); cell line: PGP1 (iPSC); cell typ | none | 0.3 | - | 261.9 MB | ~30,720 |
| GSM9510612 | geo://GSE319096 | iGLUTs_D7_3182_mRNA_R3; iPSC derived cell; tissue: iPSC deri | CRISPRko | 0.6 | - | 348.6 MB | ~30,450 |
| GSM9510616 | geo://GSE319096 | iGABAs_D36_553_mRNA_R3; iPSC derived cell; tissue: iPSC deri | CRISPRko | 0.6 | - | 276.2 MB | ~28,294 |
| GSM9510614 | geo://GSE319096 | iGABAs_D36_553_mRNA_R1; iPSC derived cell; tissue: iPSC deri | CRISPRko | 0.6 | - | 242.6 MB | ~23,174 |
| GSM9510621 | geo://GSE319096 | iGLUTs_D21_3182_mRNA_R4; iPSC derived cell; tissue: iPSC der | CRISPRko | 0.7 | - | 159.8 MB | ~21,827 |
| GSM9234653 | geo://GSE307960 | AD3G2_GE01_IMR90_N8; Brain; tissue: Brain; cell line: iPSC-d | none | 0 | - | 531.5 MB | ~20,480 |
| GSM8525896 | geo://GSE277604 | PGP1 ADNP Homo; PGP1 (iPSC); cell line: PGP1 (iPSC); cell ty | none | 0.3 | - | 175.9 MB | ~19,132 |
| GSM8525898 | geo://GSE277604 | WTC11 ADNP Het; WTC11 (iPSC); cell line: WTC11 (iPSC); cell  | none | 0.3 | - | 169.6 MB | ~18,324 |
| GSM9510617 | geo://GSE319096 | iGABAs_D36_553_mRNA_R4; iPSC derived cell; tissue: iPSC deri | CRISPRko | 0.6 | - | 148.3 MB | ~18,054 |
| GSM8525857 | geo://GSE277602 | Dual-Smadi; PGP1 (iPSC); cell line: PGP1 (iPSC); cell type:  | none | 0.3 | - | 92.6 MB | ~17,785 |
| GSM8525897 | geo://GSE277604 | WTC11 WT; WTC11 (iPSC); cell line: WTC11 (iPSC); cell type:  | none | 0.3 | - | 149.6 MB | ~17,246 |
| GSM9510618 | geo://GSE319096 | iGLUTs_D21_3182_mRNA_R1; iPSC derived cell; tissue: iPSC der | CRISPRko | 0.7 | - | 200.7 MB | ~16,168 |
| GSM8525854 | geo://GSE277602 | PSC-iNs; PGP1 (iPSC); cell line: PGP1 (iPSC); cell type: ind | none | 0.3 | 3_prime_gex | 163.3 MB | 10,572 (scBaseCount) |
| GSM8525855 | geo://GSE277602 | 7DNPC-iNs; PGP1 (iPSC); cell line: PGP1 (iPSC); cell type: i | none | 0.3 | - | 108.1 MB | ~14,282 |
| GSM8525894 | geo://GSE277604 | PGP1 WT; PGP1 (iPSC); cell line: PGP1 (iPSC); cell type: ind | none | 0.3 | - | 108.7 MB | ~14,282 |
| GSM9510619 | geo://GSE319096 | iGLUTs_D21_3182_mRNA_R2; iPSC derived cell; tissue: iPSC der | CRISPRko | 0.7 | - | 189.1 MB | ~14,012 |
| GSM8525856 | geo://GSE277602 | 3DNPC-iNs; PGP1 (iPSC); cell line: PGP1 (iPSC); cell type: i | none | 0.3 | - | 42.3 MB | ~13,473 |
| GSM8082452 | geo://GSE255904 | Alice_1; iPSC-derived neurons; cell line: iPSC-derived neuro | none | 0.3 | 3_prime_gex | 57.0 MB | 4,380 (scBaseCount) |
| GSM8082453 | geo://GSE255904 | Alice_2; iPSC-derived neurons; cell line: iPSC-derived neuro | none | 0.3 | 3_prime_gex | 55.9 MB | 3,996 (scBaseCount) |
| GSM8082454 | geo://GSE255904 | Alice_3; iPSC-derived neurons; cell line: iPSC-derived neuro | none | 0.3 | 3_prime_gex | 43.3 MB | 3,394 (scBaseCount) |
| GSE335939 | geo://GSE335939 | Integration of cell-specific gene expression and chromatin a | none | 0.0 | - | 149.9 GB | ? |
| GSM4632027 | geo://GSE152988 | CROP-seq_CRISPRa_10x_lane_1; Human iPSC-derived neuron; cell | CRISPRa | 0.5 | 3_prime_gex | 89.1 MB | 25,941 (scBaseCount) |
| GSM4632028 | geo://GSE152988 | CROP-seq_CRISPRa_10x_lane_2; Human iPSC-derived neuron; cell | CRISPRa | 0.5 | 3_prime_gex | 72.4 MB | 16,093 (scBaseCount) |
| GSM9510626 | geo://GSE319096 | iGLUTs_D7_3182_gRNA_R1; iPSC derived cell; tissue: iPSC deri | CRISPRko | 0.6 | - | 32.5 MB | ? |
| GSM9510627 | geo://GSE319096 | iGLUTs_D7_3182_gRNA_R2; iPSC derived cell; tissue: iPSC deri | CRISPRko | 0.6 | - | 28.3 MB | ? |
| GSM9510628 | geo://GSE319096 | iGLUTs_D7_3182_gRNA_R3; iPSC derived cell; tissue: iPSC deri | CRISPRko | 0.6 | - | 31.5 MB | ? |
| GSM9510629 | geo://GSE319096 | iGLUTs_D7_3182_gRNA_R4; iPSC derived cell; tissue: iPSC deri | CRISPRko | 0.6 | - | 31.4 MB | ? |
| GSE289420 | geo://GSE289420 | Astrocyte-derived cholesterol drives synaptic gene expressio | CRISPRi | 0.5 | - | 19.9 MB | ? |
| GSM9134885 | geo://GSE303735 | 1046-GWI-Vehicle, 1046-GWI-Toxicant, 1039-Control-Vehicle, 1 | none | 0 | - | 82.8 MB | ? |
| GSM9134886 | geo://GSE303735 | 1024-GWI-Vehicle, 1024-GWI-Toxicant, 1083-Control-Vehicle, 1 | none | 0 | - | 82.8 MB | ? |
| GSE260498 | geo://GSE260498 | Insights for disease modeling from single cell transcriptomi | none | 0 | - | 58.7 MB | ? |
| GSM9717805 | geo://GSE330092 | Endothelial containing neural organoid (E) for originating i | none | 0.3 | - | 510.0 kB | ? |
| GSM9717806 | geo://GSE330092 | Traditional cerebral organoids (N) for originating iPSC samp | none | 0.3 | - | 506.9 kB | ? |
| GSM9717807 | geo://GSE330092 | Endothelial containing neural organoid (E) for originating i | none | 0.3 | - | 512.0 kB | ? |
| GSM9717808 | geo://GSE330092 | Traditional cerebral organoids (N) for originating iPSC samp | none | 0.3 | - | 512.0 kB | ? |
| ... 37 more | | | | | | | 329,820 |

## Rejected / companion samples: 42

- GSM3543612 (geo://GSE124703) — reject: Bulk Quant-seq, not single-cell expression.
- GSM3543613 (geo://GSE124703) — reject: Bulk Quant-seq, not single-cell expression.
- GSM3543614 (geo://GSE124703) — reject: Bulk Quant-seq, not single-cell expression.
- GSM3543615 (geo://GSE124703) — reject: Bulk Quant-seq, not single-cell expression.
- GSM3543616 (geo://GSE124703) — reject: Bulk Quant-seq, not single-cell expression.
- GSM3543617 (geo://GSE124703) — reject: Bulk Quant-seq, not single-cell expression.
- GSM3543618 (geo://GSE124703) — reject: CROP-seq in iPSCs, not iPSC-derived neurons.
- GSM3543619 (geo://GSE124703) — reject: CROP-seq in iPSCs, not iPSC-derived neurons.
- GSM3543620 (geo://GSE124703) — companion: sgRNA mapping file for iPSC CROP-seq lanes.
- GSM3543621 (geo://GSE124703) — companion: sgRNA mapping file for iPSC CROP-seq lanes.
- GSM3543624 (geo://GSE124703) — companion: sgRNA mapping file for neuron CROP-seq lane 1.
- GSM3543625 (geo://GSE124703) — companion: sgRNA mapping file for neuron CROP-seq lane 2.
- GSM4632030 (geo://GSE152988) — reject: Bulk 3'-tag RNA-seq, not single-cell; no files listed.
- GSM4632031 (geo://GSE152988) — reject: Bulk 3'-tag RNA-seq, not single-cell; no files listed.
- GSM9510606 (geo://GSE319096) — reject: NPC not iPSC-derived neurons; mRNA only, no guide labels in this sample
- GSM9510607 (geo://GSE319096) — reject: NPC not iPSC-derived neurons; mRNA only, no guide labels
- GSM9510608 (geo://GSE319096) — reject: NPC not iPSC-derived neurons; mRNA only, no guide labels
- GSM9510609 (geo://GSE319096) — reject: NPC not iPSC-derived neurons; mRNA only, no guide labels
- GSM9510622 (geo://GSE319096) — companion: gRNA guide counts for NPC screen; companion to mRNA samples
- GSM9510623 (geo://GSE319096) — companion: gRNA guide counts for NPC screen; companion to mRNA samples
- GSM9510624 (geo://GSE319096) — companion: gRNA guide counts for NPC screen; companion to mRNA samples
- GSM9510625 (geo://GSE319096) — companion: gRNA guide counts for NPC screen; companion to mRNA samples
- GSE182307 (geo://GSE182307) — reject: Bulk astrocyte RNA-Seq, wrong cell type and not single-cell.
- GSM9039677 (geo://GSE299483) — reject: Bulk RNA-seq in RPE1 retinal epithelial cells, not iPSC-derived neurons.
- GSM9039678 (geo://GSE299483) — reject: Bulk RNA-seq in RPE1 retinal epithelial cells, not iPSC-derived neurons.
- GSM9039679 (geo://GSE299483) — reject: Bulk RNA-seq in RPE1 retinal epithelial cells, not iPSC-derived neurons.
- GSM9039680 (geo://GSE299483) — reject: Bulk CRISPRko in RPE1 cells, wrong cell type and not single-cell.
- GSM9039681 (geo://GSE299483) — reject: Bulk CRISPRko in RPE1 cells, wrong cell type and not single-cell.
- GSM9039682 (geo://GSE299483) — reject: Bulk CRISPRko in RPE1 cells, wrong cell type and not single-cell.
- GSM9039683 (geo://GSE299483) — reject: Bulk CRISPRko in RPE1 cells, wrong cell type and not single-cell.
