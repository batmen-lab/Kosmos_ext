# Survey

**Question:** In human iPSC-derived neurons, can the transcriptional response to CRISPRi knockdown of genes not seen in training be predicted from a small labeled screen, using large single-cell profiles of iPSC-derived neurons from other experiments, with their labels ignored, as supplementary evidence?

- cell type: `iPSC-derived neurons` (aliases: induced pluripotent stem cell-derived neurons, human neurons, iPSC neurons, neuronal cells), organism `human`, modality `CRISPRi`, topic `-`
- 22 series described, 309s, nothing downloaded

## Fit: gold size against the supplementary pool

| gold series | gold samples | gold cells | chemistry | supp samples | supp cells | of which unlabeled | same chemistry | supp / gold | enough (>= 2x) |
|---|---|---|---|---|---|---|---|---|---|
| geo://GSE124703 | 2 | ? | - | 93 | 969,380 | 554,461 | - | ? | no |
| geo://GSE289235 | 1 | ? | - | 93 | 969,380 | 554,461 | - | ? | no |
| geo://GSE152988 | 4 | 62,383 | 3_prime_gex | 91 | 927,346 | 554,461 | 196,952 | 14.9 | yes |

## Gold candidates (single-cell screens of the right cells, with labels)

| sample | series | title | modality | topic | chemistry | download | cells |
|---|---|---|---|---|---|---|---|
| GSM3543623 | geo://GSE124703 | CROP-seq_Day 7_neuron_10x_lane_2; Human iPSC-derived neuron; | CRISPRi | 1 | - | 224.9 MB | ? |
| GSM3543622 | geo://GSE124703 | CROP-seq_Day 7_neuron_10x_lane_1; Human iPSC-derived neuron; | CRISPRi | 1 | - | 239.6 MB | ? |
| GSM8787485 | geo://GSE289235 | iPSC-Neuron, Day 21, CROP-seq, rep1; WTC11; cell line: WTC11 | CRISPRi | 0 | - | 1.0 GB | ? |
| GSM4632025 | geo://GSE152988 | CROP-seq_CRISPRi_10x_lane_3; Human iPSC-derived neuron; cell | CRISPRi | 1 | 3_prime_gex | 69.2 MB | 12,387 (scBaseCount) |
| GSM4632023 | geo://GSE152988 | CROP-seq_CRISPRi_10x_lane_2; Human iPSC-derived neuron; cell | CRISPRi | 1 | 3_prime_gex | 77.6 MB | 16,658 (scBaseCount) |
| GSM4632022 | geo://GSE152988 | CROP-seq_CRISPRi_10x_lane_1; Human iPSC-derived neuron; cell | CRISPRi | 1 | 3_prime_gex | 79.7 MB | 15,617 (scBaseCount) |
| GSM4632026 | geo://GSE152988 | CROP-seq_CRISPRi_10x_lane_4; Human iPSC-derived neuron; cell | CRISPRi | 1 | 3_prime_gex | 88.1 MB | 17,721 (scBaseCount) |

- GSM3543623: labels — guide companion sample(s) GSM3543625
- GSM3543622: labels — guide companion sample(s) GSM3543624
- GSM8787485: labels — obs column 'guide' (90 values, e.g. ADAM10_i1, ADAM10_i2, AP2S1_i1)
- GSM4632025: labels — unverified (the model's judgement; no label file or guide sample seen)
- GSM4632023: labels — unverified (the model's judgement; no label file or guide sample seen)
- GSM4632022: labels — unverified (the model's judgement; no label file or guide sample seen)
- GSM4632026: labels — unverified (the model's judgement; no label file or guide sample seen)

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
| GSM9134885 | geo://GSE303735 | 1046-GWI-Vehicle, 1046-GWI-Toxicant, 1039-Control-Vehicle, 1 | none | 0 | - | 82.8 MB | ? |
| GSM9134886 | geo://GSE303735 | 1024-GWI-Vehicle, 1024-GWI-Toxicant, 1083-Control-Vehicle, 1 | none | 0 | - | 82.8 MB | ? |
| GSM7814309 | geo://GSE244370 | iP11N-d28-DM-Rep1; NGN2-iN; cell type: NGN2-iN; day: iP11N;  | none | 0 | multiome | 2.2 MB | 3,919 (scBaseCount) |
| GSM7814310 | geo://GSE244370 | iP11N-d28-DM-Rep2; NGN2-iN; cell type: NGN2-iN; day: iP11N;  | none | 0 | - | 6.8 MB | ? |
| GSM7814311 | geo://GSE244370 | G12-d28-DM-Rep1; NGN2-iN; cell type: NGN2-iN; day: G12; medi | none | 0 | - | 1.2 MB | ? |
| GSM7814312 | geo://GSE244370 | G12-d28-DM-Rep2; NGN2-iN; cell type: NGN2-iN; day: G12; medi | none | 0 | - | 2.0 MB | ? |
| GSM7814313 | geo://GSE244370 | iP11N-d42-DM-Rep2; NGN2-iN; cell type: NGN2-iN; day: iP11N;  | none | 0 | - | 2.5 MB | ? |
| GSM7814314 | geo://GSE244370 | iP11N-d56-DM-Rep1; NGN2-iN; cell type: NGN2-iN; day: iP11N;  | none | 0 | - | 1.9 MB | ? |
| ... 53 more | | | | | | | 421,432 |

## Rejected / companion samples: 48

- GSM3543612 (geo://GSE124703) — reject: Bulk Quant-seq, not single-cell expression.
- GSM3543613 (geo://GSE124703) — reject: Bulk Quant-seq, not single-cell expression.
- GSM3543614 (geo://GSE124703) — reject: Bulk Quant-seq, not single-cell expression.
- GSM3543615 (geo://GSE124703) — reject: Bulk Quant-seq, not single-cell expression.
- GSM3543616 (geo://GSE124703) — reject: Bulk Quant-seq, not single-cell expression.
- GSM3543617 (geo://GSE124703) — reject: Bulk Quant-seq, not single-cell expression.
- GSM3543618 (geo://GSE124703) — reject: CROP-seq but iPSC, not iPSC-derived neurons.
- GSM3543619 (geo://GSE124703) — reject: CROP-seq but iPSC, not iPSC-derived neurons.
- GSM3543620 (geo://GSE124703) — companion: sgRNA mapping file for iPSC CROP-seq lanes.
- GSM3543621 (geo://GSE124703) — companion: sgRNA mapping file for iPSC CROP-seq lanes.
- GSM3543624 (geo://GSE124703) — companion: sgRNA mapping file for neuron CROP-seq lanes.
- GSM3543625 (geo://GSE124703) — companion: sgRNA mapping file for neuron CROP-seq lanes.
- GSM9038101 (geo://GSE299369) — reject: Bulk RNA-seq, not single-cell; no CRISPRi perturbation screen.
- GSM9038102 (geo://GSE299369) — reject: Bulk RNA-seq, not single-cell; no CRISPRi perturbation screen.
- GSM9038103 (geo://GSE299369) — reject: Bulk RNA-seq, not single-cell; no CRISPRi perturbation screen.
- GSM9038104 (geo://GSE299369) — reject: Bulk RNA-seq co-culture, mixed cell type, no CRISPRi screen.
- GSM9038105 (geo://GSE299369) — reject: Bulk RNA-seq co-culture, mixed cell type, no CRISPRi screen.
- GSM9038106 (geo://GSE299369) — reject: Bulk RNA-seq co-culture, mixed cell type, no CRISPRi screen.
- GSM4632030 (geo://GSE152988) — reject: Bulk RNA-seq, not single-cell; no files listed.
- GSM4632031 (geo://GSE152988) — reject: Bulk RNA-seq, not single-cell; no files listed.
- GSM9510606 (geo://GSE319096) — reject: NPC not iPSC-derived neurons; mRNA only, no guide labels in this file
- GSM9510607 (geo://GSE319096) — reject: NPC not iPSC-derived neurons; mRNA only, no guide labels
- GSM9510608 (geo://GSE319096) — reject: NPC not iPSC-derived neurons; mRNA only, no guide labels
- GSM9510609 (geo://GSE319096) — reject: NPC not iPSC-derived neurons; mRNA only, no guide labels
- GSM9510622 (geo://GSE319096) — companion: gRNA guide counts for NPC screen; companion to mRNA samples
- GSM9510623 (geo://GSE319096) — companion: gRNA guide counts for NPC screen; companion to mRNA samples
- GSM9510624 (geo://GSE319096) — companion: gRNA guide counts for NPC screen; companion to mRNA samples
- GSM9510625 (geo://GSE319096) — companion: gRNA guide counts for NPC screen; companion to mRNA samples
- GSE182307 (geo://GSE182307) — reject: Bulk astrocyte RNA-Seq, wrong cell type and not single-cell.
- GSE299976 (geo://GSE299976) — reject: Bulk RNA-seq, not single-cell; cannot serve as single-cell supplementary data.
