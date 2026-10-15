# Gold check: iPSC-derived neurons, CRISPRi

## GSE124703 (Tian 2019, day-7 neurons, lanes 1+2)

- lane 1: 737,280 barcodes -> 11,017 cells (>= 1000 UMIs); guide called 5,028
- lane 2: 737,280 barcodes -> 11,296 cells (>= 1000 UMIs); guide called 5,304
- non-targeting = the two abundant protospacers absent from hCRISPRi-v2: GACCCGAGACTGCTTCCCGG, GTACCACCCAAACGATAACG

- perturbations scored: 29 (cells 8,508, median 110 per perturbation), control cells 1,824
- split-half Pearson, median over perturbations: all panel genes **0.16**, top-20 DE **0.65**; share with top-20 DE > 0.5: 72%

| perturbation | cells | all genes | top-20 DE | max abs Δ |
|---|---|---|---|---|
| MAT2A | 3394 | 1.00 | 1.00 | 2.44 |
| UBA1 | 1037 | 0.93 | 0.99 | 0.65 |
| UROD | 176 | 0.84 | 0.99 | 0.86 |
| RAB7A | 625 | 0.58 | 0.98 | 0.99 |
| PPCDC | 240 | 0.43 | 0.98 | 0.72 |
| MAP3K12 | 595 | 0.58 | 0.97 | 0.46 |
| FECH | 116 | 0.82 | 0.97 | 0.86 |
| MMAB | 266 | 0.54 | 0.96 | 0.79 |
| UQCRQ | 192 | 0.50 | 0.96 | 1.08 |
| RFK | 324 | 0.41 | 0.96 | 0.52 |
| PTPLB | 201 | 0.24 | 0.91 | 0.77 |
| MVK | 157 | 0.16 | 0.89 | 0.50 |
| PMVK | 66 | 0.08 | 0.84 | 0.44 |
| PPP2R1A | 210 | 0.16 | 0.74 | 0.59 |
| NDUFA1 | 43 | 0.16 | 0.65 | 0.47 |
| KIF1A | 38 | -0.07 | 0.63 | 0.87 |
| ATP5A1 | 58 | 0.02 | 0.57 | 0.56 |
| SEPSECS | 33 | 0.03 | 0.56 | 0.40 |
| GAGTCTGGGGAGGACATTGT | 78 | 0.10 | 0.56 | 0.24 |
| PPP1R8 | 73 | 0.23 | 0.54 | 0.31 |
| TARS2 | 27 | 0.01 | 0.53 | 0.78 |
| HMGCS1 | 37 | 0.01 | 0.48 | 0.33 |
| HMGCR | 171 | 0.06 | 0.45 | 0.35 |
| PGGT1B | 76 | 0.03 | 0.45 | 0.27 |
| GSR | 41 | -0.01 | 0.25 | 0.32 |
| GACTCACGTAGCAGTGGAAA | 61 | -0.02 | 0.14 | 0.30 |
| SQLE | 110 | 0.00 | 0.11 | 0.33 |
| LSS | 34 | 0.00 | 0.04 | 0.30 |
| ATP5B | 29 | 0.00 | 0.01 | 0.54 |


## GSE289235 (CaMPARI2 CRISPRi platform, day-21 neurons)

- 1,514,144 barcodes -> 18,690 cells (>= 1000 UMIs); with a guide 10,693; non-targeting 522

- perturbations scored: 41 (cells 10,123, median 171 per perturbation), control cells 522
- split-half Pearson, median over perturbations: all panel genes **0.14**, top-20 DE **0.76**; share with top-20 DE > 0.5: 66%

| perturbation | cells | all genes | top-20 DE | max abs Δ |
|---|---|---|---|---|
| TRIP12 | 197 | 0.68 | 0.98 | 0.84 |
| GAP43 | 258 | 0.33 | 0.98 | 2.31 |
| PTEN | 1509 | 0.78 | 0.98 | 0.69 |
| VEZF1 | 409 | 0.46 | 0.97 | 0.49 |
| GSK3B | 1382 | 0.38 | 0.95 | 0.98 |
| NSD1 | 91 | 0.27 | 0.94 | 0.77 |
| CALR | 279 | 0.23 | 0.94 | 0.88 |
| TSC1 | 438 | 0.40 | 0.92 | 0.30 |
| NPRL3 | 234 | 0.31 | 0.90 | 0.27 |
| MYT1L | 324 | 0.23 | 0.88 | 0.86 |
| KMT2E | 186 | 0.21 | 0.87 | 0.89 |
| TAOK1 | 581 | 0.22 | 0.87 | 0.89 |
| SETD5 | 70 | 0.09 | 0.85 | 0.38 |
| BAX | 540 | 0.21 | 0.85 | 0.28 |
| ZMYND8 | 207 | 0.23 | 0.84 | 0.48 |
| GIGYF2 | 125 | 0.24 | 0.83 | 0.56 |
| UBE3A | 160 | 0.18 | 0.82 | 0.88 |
| ATF4 | 61 | 0.36 | 0.81 | 0.91 |
| RFX3 | 98 | 0.14 | 0.81 | 0.46 |
| CELF4 | 124 | 0.21 | 0.79 | 0.92 |
| UBR1 | 208 | 0.14 | 0.76 | 0.35 |
| PHF21A | 100 | 0.16 | 0.72 | 0.39 |
| ASXL3 | 121 | 0.12 | 0.71 | 0.77 |
| GRIN2A | 176 | 0.01 | 0.67 | 0.24 |
| TLK2 | 79 | 0.11 | 0.65 | 0.45 |
| GRIA2 | 171 | 0.07 | 0.57 | 0.36 |
| PICALM | 171 | 0.12 | 0.55 | 0.41 |
| ADAM10 | 158 | 0.11 | 0.47 | 0.30 |
| POGZ | 42 | 0.09 | 0.45 | 0.48 |
| ELAVL3 | 40 | 0.01 | 0.44 | 0.95 |
| AP2S1 | 38 | 0.01 | 0.38 | 0.39 |
| KDM1A | 25 | 0.04 | 0.35 | 0.70 |
| BLOC1S1 | 33 | 0.07 | 0.30 | 0.54 |
| TARDBP | 45 | 0.06 | 0.22 | 0.99 |
| TCF7L2 | 346 | 0.11 | 0.21 | 0.21 |
| CACNG2 | 177 | 0.03 | 0.21 | 0.26 |
| H2AFZ | 147 | -0.06 | 0.09 | 0.23 |
| RMI1 | 158 | -0.00 | 0.08 | 0.21 |
| KCNT2 | 277 | 0.02 | -0.00 | 0.17 |
| GRIN2B | 197 | -0.05 | -0.35 | 0.16 |
| CACNG7 | 141 | -0.02 | -0.38 | 0.25 |

