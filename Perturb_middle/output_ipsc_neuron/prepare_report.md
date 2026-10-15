# Prepare report — iPSC-derived neurons, CRISPRi

**Question:** In human iPSC-derived neurons, can the transcriptional response to CRISPRi knockdown of genes not seen in training be predicted from a small labeled screen, using large single-cell profiles of iPSC-derived neurons from other experiments, with their labels ignored, as supplementary evidence?

## Gold: GSE289235 / GSM8787485

| | survey (before download) | actual |
|---|---|---|
| download | 1.01 GB | 1.01 GB |
| cells | ? | 10,693 used |

| step | cells |
|---|---|
| barcodes in the file | 1,514,144 |
| cells (>= 1000 UMIs) | 18,690 |
| with a guide call | 10,693 |

- 522 non-targeting controls, 44 perturbed genes, 3-1509 cells each (median 159); 36,543 genes; median 10,068 UMIs, ribosomal 4.7%

| condition | cells |
|---|---|
| PTEN+ctrl | 1,509 |
| GSK3B+ctrl | 1,382 |
| TAOK1+ctrl | 581 |
| BAX+ctrl | 540 |
| ctrl | 522 |
| TSC1+ctrl | 438 |
| VEZF1+ctrl | 409 |
| TCF7L2+ctrl | 346 |
| MYT1L+ctrl | 324 |
| CALR+ctrl | 279 |
| KCNT2+ctrl | 277 |
| GAP43+ctrl | 258 |
| NPRL3+ctrl | 234 |
| UBR1+ctrl | 208 |
| ZMYND8+ctrl | 207 |
| TRIP12+ctrl | 197 |
| GRIN2B+ctrl | 197 |
| KMT2E+ctrl | 186 |
| CACNG2+ctrl | 177 |
| GRIN2A+ctrl | 176 |
| GRIA2+ctrl | 171 |
| PICALM+ctrl | 171 |
| UBE3A+ctrl | 160 |
| RMI1+ctrl | 158 |
| ADAM10+ctrl | 158 |
| H2AFZ+ctrl | 147 |
| CACNG7+ctrl | 141 |
| GIGYF2+ctrl | 125 |
| CELF4+ctrl | 124 |
| ASXL3+ctrl | 121 |
| PHF21A+ctrl | 100 |
| RFX3+ctrl | 98 |
| NSD1+ctrl | 91 |
| TLK2+ctrl | 79 |
| SETD5+ctrl | 70 |
| ATF4+ctrl | 61 |
| TARDBP+ctrl | 45 |
| POGZ+ctrl | 42 |
| ELAVL3+ctrl | 40 |
| AP2S1+ctrl | 38 |
| BLOC1S1+ctrl | 33 |
| KDM1A+ctrl | 25 |
| HNRNPA2B1+ctrl | 24 |
| APEX1+ctrl | 21 |
| CPNE1+ctrl | 3 |

## Supplementary candidates: survey vs downloaded, and similarity to the gold controls

Pooled when the Pearson correlation of mean log1p-CPM (shared expressed genes) with the gold controls is >= 0.6; a GEO series is judged on its first sample and at most 2 of its samples are pooled; pooling stops at 5x the gold cells. Downloaded 0.00 GB of the 8 GB budget.

| sample | series | title | download (survey) | cells (survey) | barcodes | cells (>= 1000 UMIs) | Pearson | Spearman | median UMIs | ribosomal | decision |
|---|---|---|---|---|---|---|---|---|---|---|---|
| GSM3543623 | GSE124703 | CROP-seq_Day 7_neuron_10x_lane_2; Human  | 224.9 MB | ? | 737,280 | 11,296 | 0.78 | 0.66 | 14,808 | 11.7% | pooled |
| GSM3543622 | GSE124703 | CROP-seq_Day 7_neuron_10x_lane_1; Human  | 239.6 MB | ? | 737,280 | 11,017 | 0.85 | 0.77 | 15,485 | 10.4% | pooled |
| GSM4632025 | GSE152988 | CROP-seq_CRISPRi_10x_lane_3; Human iPSC- | 69.2 MB | ? | 11,974 | 11,195 | 0.92 | 0.88 | 16,168 | 6.6% | pooled |
| GSM4632023 | GSE152988 | CROP-seq_CRISPRi_10x_lane_2; Human iPSC- | 77.6 MB | ? | 14,408 | 13,364 | 0.92 | 0.88 | 13,322 | 6.5% | pooled |
| GSM4632022 | GSE152988 | CROP-seq_CRISPRi_10x_lane_1; Human iPSC- | 79.7 MB | ? | - | - | - | - | - | - | skipped: 2 sample(s) of this series already pooled |
| GSM4632026 | GSE152988 | CROP-seq_CRISPRi_10x_lane_4; Human iPSC- | 88.1 MB | ? | - | - | - | - | - | - | skipped: 2 sample(s) of this series already pooled |
| GSM9510613 | GSE319096 | iGLUTs_D7_3182_mRNA_R4; iPSC derived cel | 345.4 MB | ~35,031 | 29,595 | 27,483 | 0.76 | 0.66 | 7,331 | 9.3% | pooled |
| GSM9510610 | GSE319096 | iGLUTs_D7_3182_mRNA_R1; iPSC derived cel | 374.8 MB | ~32,336 | - | - | - | - | - | - | not needed: 74,355 cells pooled (>= 5x gold) |
| GSM9510620 | GSE319096 | iGLUTs_D21_3182_mRNA_R3; iPSC derived ce | 241.6 MB | ~31,797 | - | - | - | - | - | - | not needed: 74,355 cells pooled (>= 5x gold) |
| GSM9510611 | GSE319096 | iGLUTs_D7_3182_mRNA_R2; iPSC derived cel | 366.4 MB | ~31,528 | - | - | - | - | - | - | not needed: 74,355 cells pooled (>= 5x gold) |
| GSM9510615 | GSE319096 | iGABAs_D36_553_mRNA_R2; iPSC derived cel | 275.2 MB | ~31,258 | - | - | - | - | - | - | not needed: 74,355 cells pooled (>= 5x gold) |
| GSM8525895 | GSE277604 | PGP1 ADNP Het; PGP1 (iPSC); cell line: P | 261.9 MB | ~30,720 | - | - | - | - | - | - | not needed: 74,355 cells pooled (>= 5x gold) |
| GSM9510612 | GSE319096 | iGLUTs_D7_3182_mRNA_R3; iPSC derived cel | 348.6 MB | ~30,450 | - | - | - | - | - | - | not needed: 74,355 cells pooled (>= 5x gold) |
| GSM9510616 | GSE319096 | iGABAs_D36_553_mRNA_R3; iPSC derived cel | 276.2 MB | ~28,294 | - | - | - | - | - | - | not needed: 74,355 cells pooled (>= 5x gold) |
| GSM9510614 | GSE319096 | iGABAs_D36_553_mRNA_R1; iPSC derived cel | 242.6 MB | ~23,174 | - | - | - | - | - | - | not needed: 74,355 cells pooled (>= 5x gold) |
| GSM9510621 | GSE319096 | iGLUTs_D21_3182_mRNA_R4; iPSC derived ce | 159.8 MB | ~21,827 | - | - | - | - | - | - | not needed: 74,355 cells pooled (>= 5x gold) |
| GSM9234653 | GSE307960 | AD3G2_GE01_IMR90_N8; Brain; tissue: Brai | 531.5 MB | ~20,480 | - | - | - | - | - | - | not needed: 74,355 cells pooled (>= 5x gold) |
| GSM8525896 | GSE277604 | PGP1 ADNP Homo; PGP1 (iPSC); cell line:  | 175.9 MB | ~19,132 | - | - | - | - | - | - | not needed: 74,355 cells pooled (>= 5x gold) |
| GSM8525898 | GSE277604 | WTC11 ADNP Het; WTC11 (iPSC); cell line: | 169.6 MB | ~18,324 | - | - | - | - | - | - | not needed: 74,355 cells pooled (>= 5x gold) |
| GSM9510617 | GSE319096 | iGABAs_D36_553_mRNA_R4; iPSC derived cel | 148.3 MB | ~18,054 | - | - | - | - | - | - | not needed: 74,355 cells pooled (>= 5x gold) |
| GSM8525857 | GSE277602 | Dual-Smadi; PGP1 (iPSC); cell line: PGP1 | 92.6 MB | ~17,785 | - | - | - | - | - | - | not needed: 74,355 cells pooled (>= 5x gold) |
| GSM8525897 | GSE277604 | WTC11 WT; WTC11 (iPSC); cell line: WTC11 | 149.6 MB | ~17,246 | - | - | - | - | - | - | not needed: 74,355 cells pooled (>= 5x gold) |
| GSM9510618 | GSE319096 | iGLUTs_D21_3182_mRNA_R1; iPSC derived ce | 200.7 MB | ~16,168 | - | - | - | - | - | - | not needed: 74,355 cells pooled (>= 5x gold) |
| GSM8525854 | GSE277602 | PSC-iNs; PGP1 (iPSC); cell line: PGP1 (i | 163.3 MB | ~15,090 | - | - | - | - | - | - | not needed: 74,355 cells pooled (>= 5x gold) |
| GSM8525855 | GSE277602 | 7DNPC-iNs; PGP1 (iPSC); cell line: PGP1  | 108.1 MB | ~14,282 | - | - | - | - | - | - | not needed: 74,355 cells pooled (>= 5x gold) |
| GSM8525894 | GSE277604 | PGP1 WT; PGP1 (iPSC); cell line: PGP1 (i | 108.7 MB | ~14,282 | - | - | - | - | - | - | not needed: 74,355 cells pooled (>= 5x gold) |
| GSM9510619 | GSE319096 | iGLUTs_D21_3182_mRNA_R2; iPSC derived ce | 189.1 MB | ~14,012 | - | - | - | - | - | - | not needed: 74,355 cells pooled (>= 5x gold) |
| GSM8525856 | GSE277602 | 3DNPC-iNs; PGP1 (iPSC); cell line: PGP1  | 42.3 MB | ~13,473 | - | - | - | - | - | - | not needed: 74,355 cells pooled (>= 5x gold) |
| GSM8082452 | GSE255904 | Alice_1; iPSC-derived neurons; cell line | 57.0 MB | ~5,928 | - | - | - | - | - | - | not needed: 74,355 cells pooled (>= 5x gold) |
| GSM8082453 | GSE255904 | Alice_2; iPSC-derived neurons; cell line | 55.9 MB | ~5,389 | - | - | - | - | - | - | not needed: 74,355 cells pooled (>= 5x gold) |
| GSM8082454 | GSE255904 | Alice_3; iPSC-derived neurons; cell line | 43.3 MB | ~4,581 | - | - | - | - | - | - | not needed: 74,355 cells pooled (>= 5x gold) |
| GSE335939 | GSE335939 | Integration of cell-specific gene expres | 149.89 GB | ? | - | - | - | - | - | - | not needed: 74,355 cells pooled (>= 5x gold) |
| GSM4632027 | GSE152988 | CROP-seq_CRISPRa_10x_lane_1; Human iPSC- | 89.1 MB | ? | - | - | - | - | - | - | not needed: 74,355 cells pooled (>= 5x gold) |
| GSM4632028 | GSE152988 | CROP-seq_CRISPRa_10x_lane_2; Human iPSC- | 72.4 MB | ? | - | - | - | - | - | - | not needed: 74,355 cells pooled (>= 5x gold) |
| GSM9510626 | GSE319096 | iGLUTs_D7_3182_gRNA_R1; iPSC derived cel | 32.5 MB | ? | - | - | - | - | - | - | not needed: 74,355 cells pooled (>= 5x gold) |
| GSM9510627 | GSE319096 | iGLUTs_D7_3182_gRNA_R2; iPSC derived cel | 28.3 MB | ? | - | - | - | - | - | - | not needed: 74,355 cells pooled (>= 5x gold) |
| GSM9510628 | GSE319096 | iGLUTs_D7_3182_gRNA_R3; iPSC derived cel | 31.5 MB | ? | - | - | - | - | - | - | not needed: 74,355 cells pooled (>= 5x gold) |
| GSM9510629 | GSE319096 | iGLUTs_D7_3182_gRNA_R4; iPSC derived cel | 31.4 MB | ? | - | - | - | - | - | - | not needed: 74,355 cells pooled (>= 5x gold) |
| GSM9134885 | GSE303735 | 1046-GWI-Vehicle, 1046-GWI-Toxicant, 103 | 82.8 MB | ? | - | - | - | - | - | - | not needed: 74,355 cells pooled (>= 5x gold) |
| GSM9134886 | GSE303735 | 1024-GWI-Vehicle, 1024-GWI-Toxicant, 108 | 82.8 MB | ? | - | - | - | - | - | - | not needed: 74,355 cells pooled (>= 5x gold) |
| GSM7814309 | GSE244370 | iP11N-d28-DM-Rep1; NGN2-iN; cell type: N | 2.2 MB | ? | - | - | - | - | - | - | not needed: 74,355 cells pooled (>= 5x gold) |
| GSM7814310 | GSE244370 | iP11N-d28-DM-Rep2; NGN2-iN; cell type: N | 6.8 MB | ? | - | - | - | - | - | - | not needed: 74,355 cells pooled (>= 5x gold) |
| GSM7814311 | GSE244370 | G12-d28-DM-Rep1; NGN2-iN; cell type: NGN | 1.2 MB | ? | - | - | - | - | - | - | not needed: 74,355 cells pooled (>= 5x gold) |
| GSM7814312 | GSE244370 | G12-d28-DM-Rep2; NGN2-iN; cell type: NGN | 2.0 MB | ? | - | - | - | - | - | - | not needed: 74,355 cells pooled (>= 5x gold) |
| GSM7814313 | GSE244370 | iP11N-d42-DM-Rep2; NGN2-iN; cell type: N | 2.5 MB | ? | - | - | - | - | - | - | not needed: 74,355 cells pooled (>= 5x gold) |
| GSM7814314 | GSE244370 | iP11N-d56-DM-Rep1; NGN2-iN; cell type: N | 1.9 MB | ? | - | - | - | - | - | - | not needed: 74,355 cells pooled (>= 5x gold) |
| GSM7814315 | GSE244370 | iP11N-d56-DM-Rep2; NGN2-iN; cell type: N | 3.7 MB | ? | - | - | - | - | - | - | not needed: 74,355 cells pooled (>= 5x gold) |
| GSE260498 | GSE260498 | Insights for disease modeling from singl | 58.7 MB | ? | - | - | - | - | - | - | not needed: 74,355 cells pooled (>= 5x gold) |
| SRX11541629 | scbasecount | iPSC-derived neurons (NLGN3 R451C genoty | ? | ~15,462 | - | - | - | - | - | - | not needed: 74,355 cells pooled (>= 5x gold) |
| SRX10657844 | scbasecount | SA005 | hiPSC-derived GABAergic neurons  | ? | ~13,707 | - | - | - | - | - | - | not needed: 74,355 cells pooled (>= 5x gold) |
| SRX26057557 | scbasecount | SF2019-227 | hiPSC-derived neuron | Papa | ? | ~13,543 | - | - | - | - | - | - | not needed: 74,355 cells pooled (>= 5x gold) |
| SRX10657842 | scbasecount | SA005 | hiPSC-derived GABAergic neurons  | ? | ~13,167 | - | - | - | - | - | - | not needed: 74,355 cells pooled (>= 5x gold) |
| SRX10657841 | scbasecount | SA005 | hiPSC-derived GABAergic neurons  | ? | ~11,721 | - | - | - | - | - | - | not needed: 74,355 cells pooled (>= 5x gold) |
| SRX15406697 | scbasecount | iPSC-derived neuron culture | excitatory | ? | ~11,347 | - | - | - | - | - | - | not needed: 74,355 cells pooled (>= 5x gold) |
| SRX17782703 | scbasecount | HPSI0114ikolf2-clone1 | neurons differen | ? | ~11,228 | - | - | - | - | - | - | not needed: 74,355 cells pooled (>= 5x gold) |
| SRX26240592 | scbasecount | iPSC lines | iPSC-derived neuronal cell  | ? | ~11,155 | - | - | - | - | - | - | not needed: 74,355 cells pooled (>= 5x gold) |
| SRX12478851 | scbasecount | human iPSC line 2131 | iPSC-derived neur | ? | ~10,485 | - | - | - | - | - | - | not needed: 74,355 cells pooled (>= 5x gold) |
| SRX12478850 | scbasecount | 2131 | iPSC-derived neural cells (dopami | ? | ~9,428 | - | - | - | - | - | - | not needed: 74,355 cells pooled (>= 5x gold) |
| SRX12478853 | scbasecount | human iPSC line 2131 (TH-P2A-tdTomato) | | ? | ~7,214 | - | - | - | - | - | - | not needed: 74,355 cells pooled (>= 5x gold) |
| SRX10657831 | scbasecount | SA005 | hiPSC-derived GABAergic neurons  | ? | ~6,826 | - | - | - | - | - | - | not needed: 74,355 cells pooled (>= 5x gold) |
| SRX10657832 | scbasecount | SA005 | hiPSC-derived GABAergic neurons  | ? | ~6,648 | - | - | - | - | - | - | not needed: 74,355 cells pooled (>= 5x gold) |
| SRX10657846 | scbasecount | SA005 | hiPSC-derived GABAergic neurons  | ? | ~6,367 | - | - | - | - | - | - | not needed: 74,355 cells pooled (>= 5x gold) |
| SRX22066416 | scbasecount | Human iPSC neurons | GABAergic Cortical  | ? | ~6,182 | - | - | - | - | - | - | not needed: 74,355 cells pooled (>= 5x gold) |
| SRX26240591 | scbasecount | iPSC lines | iPSC-derived neuronal cell  | ? | ~5,818 | - | - | - | - | - | - | not needed: 74,355 cells pooled (>= 5x gold) |
| ERX4791270 | scbasecount | SC102A1 | induced neurons from pluripote | ? | ~5,176 | - | - | - | - | - | - | not needed: 74,355 cells pooled (>= 5x gold) |
| SRX10657829 | scbasecount | NC3-1 | hiPSC-derived GABAergic neurons  | ? | ~3,628 | - | - | - | - | - | - | not needed: 74,355 cells pooled (>= 5x gold) |
| SRX10657830 | scbasecount | NC3-1 | hiPSC-derived GABAergic neurons  | ? | ~3,573 | - | - | - | - | - | - | not needed: 74,355 cells pooled (>= 5x gold) |
| SRX10657840 | scbasecount | NC3-1 | hiPSC-derived GABAergic neurons  | ? | ~3,548 | - | - | - | - | - | - | not needed: 74,355 cells pooled (>= 5x gold) |
| SRX10657839 | scbasecount | NC3-1 (hiPSC-derived GABAergic neurons)  | ? | ~3,538 | - | - | - | - | - | - | not needed: 74,355 cells pooled (>= 5x gold) |
| SRX14780583 | scbasecount | 297-1 (human induced pluripotent stem ce | ? | ~3,099 | - | - | - | - | - | - | not needed: 74,355 cells pooled (>= 5x gold) |
| SRX21770098 | scbasecount | KOLF2.1J, iPSC-derived neurons, clone: F | ? | ~2,592 | - | - | - | - | - | - | not needed: 74,355 cells pooled (>= 5x gold) |
| SRX13009786 | scbasecount | iPSC-derived neurons | iPSC-derived neur | ? | ~2,505 | - | - | - | - | - | - | not needed: 74,355 cells pooled (>= 5x gold) |
| SRX24595006 | scbasecount | hiPSC-derived motor neurons | motor neur | ? | ~1,276 | - | - | - | - | - | - | not needed: 74,355 cells pooled (>= 5x gold) |
| SRX6917529 | scbasecount | CS6ZLDiALS-n1 | iPSC-derived motor neuro | ? | ~1,127 | - | - | - | - | - | - | not needed: 74,355 cells pooled (>= 5x gold) |
| SRX6917525 | scbasecount | CS0179iCTR-n1 | iPSC-derived motor neuro | ? | ~834 | - | - | - | - | - | - | not needed: 74,355 cells pooled (>= 5x gold) |
| SRX6917531 | scbasecount | CS52iALS-C9n6.ISO | iPSC-derived motor n | ? | ~826 | - | - | - | - | - | - | not needed: 74,355 cells pooled (>= 5x gold) |
| SRX6917530 | scbasecount | CS83iCTR-33n1, CS0179iCTR-n1, CS2XWCiALS | ? | ~752 | - | - | - | - | - | - | not needed: 74,355 cells pooled (>= 5x gold) |
| SRX6917524 | scbasecount | CS29iALS-C9n1 | iPSC-derived motor neuro | ? | ~735 | - | - | - | - | - | - | not needed: 74,355 cells pooled (>= 5x gold) |
| SRX14847038 | scbasecount | hiPSCs | NGN2-induced glutamatergic neur | ? | ~3,205 | - | - | - | - | - | - | not needed: 74,355 cells pooled (>= 5x gold) |
| SRX14847043 | scbasecount | human-induced pluripotent stem cell | NP | ? | ~240 | - | - | - | - | - | - | not needed: 74,355 cells pooled (>= 5x gold) |
| SRX27325577 | scbasecount | iPS cell–derived neurons | Cultured glut | ? | ~20,171 | - | - | - | - | - | - | not needed: 74,355 cells pooled (>= 5x gold) |
| SRX27532794 | scbasecount | Neurons derived from hiPSCs | Cultured n | ? | ~18,084 | - | - | - | - | - | - | not needed: 74,355 cells pooled (>= 5x gold) |
| SRX27325578 | scbasecount | hiPSC-derived neurons | neuronal cells f | ? | ~16,864 | - | - | - | - | - | - | not needed: 74,355 cells pooled (>= 5x gold) |
| SRX27532313 | scbasecount | Derived from iPSC cell lines (e.g., CW20 | ? | ~15,365 | - | - | - | - | - | - | not needed: 74,355 cells pooled (>= 5x gold) |
| SRX27532803 | scbasecount | hiPSC-derived neurons | cultured glutama | ? | ~15,170 | - | - | - | - | - | - | not needed: 74,355 cells pooled (>= 5x gold) |
| SRX27325572 | scbasecount | cultured neurons from human iPS cells |  | ? | ~15,039 | - | - | - | - | - | - | not needed: 74,355 cells pooled (>= 5x gold) |
| SRX27325575 | scbasecount | Neurons cultured and differentiated from | ? | ~14,482 | - | - | - | - | - | - | not needed: 74,355 cells pooled (>= 5x gold) |
| SRX27325189 | scbasecount | iPSC-derived neurons | cultured neurons  | ? | ~13,480 | - | - | - | - | - | - | not needed: 74,355 cells pooled (>= 5x gold) |
| SRX27532322 | scbasecount | hiPSC-derived neurons | cultured neurons | ? | ~12,521 | - | - | - | - | - | - | not needed: 74,355 cells pooled (>= 5x gold) |
| SRX27325193 | scbasecount | hiPSC-derived neurons | cultured glutama | ? | ~11,968 | - | - | - | - | - | - | not needed: 74,355 cells pooled (>= 5x gold) |
| SRX27325183 | scbasecount | iPS cell-derived neuronal cultures | cul | ? | ~11,800 | - | - | - | - | - | - | not needed: 74,355 cells pooled (>= 5x gold) |
| ERX9138820 | scbasecount | HC1 (healthy control iPSC line) | iPSC-d | ? | ~4,025 | - | - | - | - | - | - | not needed: 74,355 cells pooled (>= 5x gold) |
| SRX14847034 | scbasecount | hiPSC-derived NGN2 neurons | hiPSC-deriv | ? | ~28,413 | - | - | - | - | - | - | not needed: 74,355 cells pooled (>= 5x gold) |
| SRX14847033 | scbasecount | hiPSC | NGN2-induced glutamatergic neuro | ? | ~9,977 | - | - | - | - | - | - | not needed: 74,355 cells pooled (>= 5x gold) |
| SRX14847035 | scbasecount | hiPSC-derived neurons (NGN2-induced) | h | ? | ~7,461 | - | - | - | - | - | - | not needed: 74,355 cells pooled (>= 5x gold) |
| SRX14847036 | scbasecount | human-induced pluripotent stem cell | hu | ? | ~3,432 | - | - | - | - | - | - | not needed: 74,355 cells pooled (>= 5x gold) |
| SRX14847037 | scbasecount | hiPSC-derived neurons | hiPSC-derived gl | ? | ~2,768 | - | - | - | - | - | - | not needed: 74,355 cells pooled (>= 5x gold) |
| SRX14847039 | scbasecount | human-induced pluripotent stem cells (iP | ? | ~2,323 | - | - | - | - | - | - | not needed: 74,355 cells pooled (>= 5x gold) |
| SRX14847042 | scbasecount | hiPSCs | Glutamatergic neurons derived f | ? | ~1,137 | - | - | - | - | - | - | not needed: 74,355 cells pooled (>= 5x gold) |

**Pooled:** 74,355 cells (6.95x the gold) from 5 sample(s): GSM3543623 (4,844 panel genes measured), GSM3543622 (4,844 panel genes measured), GSM4632025 (4,964 panel genes measured), GSM4632023 (4,964 panel genes measured), GSM9510613 (5,004 panel genes measured)

## Panel and study

- 5,005 genes (top HVG among genes a pooled supp measures + 5 perturbed genes); log1p(CPM 10000); a panel gene a source does not measure is the gold control mean; perturbed genes absent from the gold's genes: H2AFZ
- split `unseen_single`: test 25%, validation 20% of the rest; batch 32
