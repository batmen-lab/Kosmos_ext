# iPSC-derived neurons, CRISPRi：Stage 1–3 汇总（run-v2）

问题：*In human iPSC-derived neurons, can the transcriptional response to CRISPRi knockdown of genes not seen in training be predicted from a small labeled screen, using large single-cell profiles of iPSC-derived neurons from other experiments, with their labels ignored, as supplementary evidence?*

## Stage 1：data fetching

这次的 stage 1 是为 perturbation 重写的 **survey**（`kosmos/ppi/perturbation/dataprep/`）：先不下载任何数据，把候选数据集和每个样本的文件、大小、细胞数、标签情况都查清楚，再由 stage 2 脚本按样本下载。分工如下：

- **question → spec：** `spec.py` 一次模型调用，抽出细胞类型（及别名）、物种、扰动类型、主题、assay。
- **search：** `survey.py` 用 spec 拼出多组查询，分别搜 gold 和 supplementary；检索来源是 ToolUniverse（GEO、OmicsDI、HuggingFace）加上新增的 GEO 全文检索（`sources/geo.py::search_series`，查 title、summary、design，覆盖所有 series 类型），再加上 scBaseCount。
- **describe：** 对每个候选 series 取 title/summary、每个 GSM 的标题和它自己的补充文件（含大小）；细胞数由 barcodes 文件大小估计。
- **judge + gate：** 每个 series 一次模型调用，按样本判断是否单细胞、是否 screen、扰动类型、细胞类型、是否有 per-cell 标签；再由规则复核，标签必须有文件证据（标签文件名、guide 配套样本，或 `.h5ad` 里的 guide 列），否则标为 unverified。
- **scBaseCount：** `scbasecount.py` 把 Arc Institute 的 scBaseCount（人类约 3.5 万个 SRX、3 亿个细胞，统一流程处理，带 cell line / tissue / chemistry / 精确细胞数）作为 supplementary 来源，并把 gold 自己的样本从中剔除。
- **read format / content：** `load.py` 读 10x mtx、10x h5、h5ad，按 UMI 数调用细胞，并计算与 gold 对照的表达相似度；`reliability.py` 在 gold 下载后测每个扰动的 split-half 可靠性。
- **write result：** survey 输出 `survey.md` / `survey.json`（含 gold 与 supp 的 fit 表）；`plan.json` 与 data report 由 stage 2 脚本和 stage 3 写出。

以本问题为例：

1. 识别出扰动类型是 CRISPRi（knockdown）、细胞是人 iPSC-derived neurons，并要求补充数据同样是该细胞类型的单细胞数据。
2. **检索：** 描述了 22 个 GEO series（加 scBaseCount），共 7 个 gold 候选样本、93 个 supplementary 样本、48 个被拒或配套样本（bulk、其他细胞类型、guide 配套文件等）。gold 候选来自三个 series：
   - GSE289235（CaMPARI2 CRISPRi 平台，day-21 神经元，1 个 h5ad）；
   - GSE124703（Tian 2019，day-7 神经元，lane 1+2，guide 在配套样本里）；
   - GSE152988（Tian 2021，4 个 lane）：GEO 上只有表达矩阵，没有 guide 信息，标为 unverified 并排在最后，不能作为 gold。
3. **下载与读取：** gold 候选下载后调用细胞、识别 guide：GSE289235 的标签在 `obs.guide`（`GENE_i1/i2`、`non-targeting_*`），GSE124703 的 guide 是原型间隔序列，用 hCRISPRi-v2 文库表映射到基因。scBaseCount 的样本是单个 `.h5ad`，通过 HTTPS 直接取，不需要账号。
4. **review：** 对两个可用 gold 测 split-half 可靠性（见下表）；supplementary 在 stage 2 里与 gold 对照做表达相似度筛选。
5. **输出：** `survey.md` 的 fit 表给出每个 gold 与 supp 池的细胞数比值；supp 池共约 96.9 万个细胞（GEO 约 54.8 万 + scBaseCount 51 个样本 42.1 万），其中约 55.4 万来自未做扰动的样本。

| gold | 有标签细胞 | 扰动数 | 每个扰动细胞数（中位数） | NT 对照 | 可靠性（全部基因） | 可靠性（top-20 DE） |
|---|---|---|---|---|---|---|
| GSE289235（选用） | 10,123 | 41 | 171 | 522 | 0.14 | 0.76 |
| GSE124703 | 8,508 | 29 | 110 | 1,824 | 0.16 | 0.65 |

可靠性是同一扰动的细胞随机分成两半后，两半各自的 Δ 的 Pearson（取扰动间中位数），它是任何模型在该 gold 上能达到的相关性的上界。

## Stage 2：manually take question corresponding subset data

脚本 `Perturb_middle/prepare_ipsc_neuron.py`，输出在 `Perturb_middle/output_ipsc_neuron/`（`gold.parquet`、`supp_ipsc_neuron_pooled.parquet`、对应的 `*_counts.parquet`、`plan.json`、`prepare_report.md`）。

|  | gold（GSE289235 / GSM8787485） | supp（5 个样本合并） |
| --- | --- | --- |
| 取回的文件 | 1 个 h5ad（1.01 GB） | GSE124703 lane 1+2、GSE152988 lane 2+3、GSE319096 的 1 个 iGLUT 样本 |
| stage 2 使用的样本 | 1 个 | 5 个（survey 的 99 个候选按顺序试，累计达到 gold 的 5 倍以上即停止，其余标为 not needed） |
| 细胞：原始 → 保留 | 1,514,144 个 barcode → 18,690（≥1000 UMI）→ **10,693**（有 guide call） | 74,355 个（各样本 ≥1000 UMI） |
| 细胞过滤 | ≥1000 UMI；没有 guide call 的丢弃 | 每个样本与 gold 对照的 Pearson ≥ 0.6 才入池（实际 0.76–0.92）；同一 series 最多 2 个样本 |
| 细胞内容 | 522 个 non-targeting 对照 + 44 个被扰动基因（每个基因 3–1,509 个细胞，中位数 159） | 神经元，没有 condition 列（视为未扰动来源，labels 忽略） |
| 基因：原始 → 保留 | 36,543 → **5,005** | 与 gold panel 按基因符号对齐，各样本覆盖 4,844–5,004 个；缺失的基因用 gold 对照均值填 |
| 基因的选法 | 前 5,000 个高变基因（按 log1p-CPM 方差，限于 supp 能测到的基因）+ 不在其中的被扰动基因 | 取 gold 的 panel |
| 表达值 | log1p-CPM（目标总量 1e4） | 同样的 log1p-CPM |
| 在训练中的作用 | 有标签：L_true | 无标签：PPI 的 L_u 项 / gated 的合成梯度 / PerturbNet 的 scVI 表示 |

stage 3 读入后，gold 实际用于训练评估的是 9,011 个细胞（8,489 个被扰动细胞，34 个扰动，加 522 个对照）；另有 10 个扰动（1,682 个细胞）不在 GEARS 的 GO 扰动词表里，按 GEARS 的做法不参与训练。按扰动随机划分：

| split | 扰动数 | 细胞数 | 扰动 |
|---|---|---|---|
| train | 21 | 3,728 | MYT1L、CALR、NSD1、PICALM、KMT2E、GIGYF2、HNRNPA2B1、BAX、APEX1、VEZF1、UBE3A、ADAM10、SETD5、TRIP12、RMI1、UBR1、KDM1A、RFX3、AP2S1、PHF21A、TCF7L2 |
| validation | 5 | 1,801 | NPRL3、GSK3B、TARDBP、TLK2、ATF4 |
| test | 8 | 2,960 | PTEN（1,509）、TAOK1、TSC1、ZMYND8、H2AFZ、POGZ、BLOC1S1、CPNE1（3） |

## Stage 3：training（run-v2）

八个 arm，同一 split、同一 panel、同一指标：

| | gold only | signed PPI | gated |
|---|---|---|---|
| GEARS | `base` | `signed ppi` | `gated` |
| MLP | `mlp base` | `mlp signed ppi` | `mlp gated` |
| PerturbNet | `perturbnet base` | `perturbnet supp`（scVI 用 gold + supp） | — |

设置：GEARS/MLP 各 arm 20 个 epoch，batch 512，lr 4e-3，GEARS 不做 lr 衰减，weight decay 5e-4，梯度按值裁剪 1.0，按验证集 `mse_deg` 选最佳 epoch。每个 epoch 8 步，共 160 步。signed PPI 为 5 个 stage-1 epoch 加 1 个 stage-2 epoch 循环（stage 2：模型 SGD lr 4e-4，λ 的 Adam lr 4e-2，λ 初值 0.5，范围 [0, 2]）。gated 每步用 `g_G + max(0,cos)·min(1, κ‖g_G‖/‖g_S‖)·g_S`，κ = 1。PerturbNet：scVI 500 个 epoch，cINN 45 个 epoch。

测试集（8 个未见过的扰动）指标：

| Arm | MSE | MSE_DEG | Pearson | Specific Pearson | Spearman | Direction | Top-K Overlap | Epochs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| base gears | 0.0157 | 0.1771 | 0.140 | -0.020 | 0.149 | 0.625 | 0.031 | 20 |
| signed ppi gears | 0.0172 | 0.1675 | 0.136 | 0.018 | 0.131 | 0.644 | 0.019 | 20 |
| gated gears | 0.0162 | 0.1785 | 0.108 | -0.025 | 0.121 | 0.613 | 0.031 | 20 |
| mlp base | 0.0156 | 0.1692 | 0.193 | 0.001 | 0.193 | 0.706 | 0.025 | 20 |
| mlp signed ppi | 0.0185 | 0.1784 | 0.109 | -0.040 | 0.103 | 0.594 | 0.019 | 20 |
| mlp gated | 0.0148 | 0.1740 | 0.211 | 0.022 | 0.214 | 0.656 | 0.038 | 20 |
| perturbnet base | 0.0333 | 0.2025 | 0.133 | -0.009 | 0.141 | 0.550 | 0.050 | 45 |
| perturbnet supp | 0.0289 | 0.1585 | 0.170 | 0.023 | 0.184 | 0.688 | 0.019 | 45 |
| *predict 0* | 0.0147 | 0.1802 | - | - | - | 0.000 | 0.006 | - |
| *mean response (leave-one-out)* | 0.0165 | 0.1817 | 0.127 | -1.000 | - | 0.569 | 0.038 | - |

用 supp 与否的配对比较（同一架构，只差 supp 的用法）：

| 架构 | arm | mse_deg | pearson | specific pearson | direction |
| --- | --- | --- | --- | --- | --- |
| GEARS | base | 0.1771 | 0.140 | -0.020 | 0.625 |
| GEARS | signed ppi | **0.1675** | 0.136 | 0.018 | 0.644 |
| GEARS | gated | 0.1785 | 0.108 | -0.025 | 0.613 |
| MLP | base | **0.1692** | 0.193 | 0.001 | **0.706** |
| MLP | signed ppi | 0.1784 | 0.109 | -0.040 | 0.594 |
| MLP | gated | 0.1740 | **0.211** | **0.022** | 0.656 |
| PerturbNet | base | 0.2025 | 0.133 | -0.009 | 0.550 |
| PerturbNet | gold + supp | **0.1585** | **0.170** | **0.023** | **0.688** |

gated 的梯度诊断（各 epoch 平均）：

| arm | cos(g_G, g_S) | synthetic weight | weight > 0 的步数 | ‖g_G‖ | ‖g_S‖ |
|---|---|---|---|---|---|
| gated（GEARS） | +0.204 | 0.247 | 79% | 0.0727 | 0.00576 |
| mlp gated | +0.109 | 0.162 | 61% | 0.0459 | 0.00174 |

signed PPI 的 λ：GEARS 在三个 stage-2 epoch 依次为 0.397、0.312、0.248（最佳 epoch 为第 18 个，在 stage 2）；MLP 依次为 0.594、0.689、0.800（最佳 epoch 为第 13 个，在 stage 1）。

运行时间：GEARS base 28 s、signed ppi 37 s、gated 57 s；MLP 三个 arm 分别 9 s、10 s、15 s；perturbnet base 380 s，perturbnet supp 3,254 s（scVI 的细胞从 4,250 个 gold 加到 78,605 个）。

## 结果解读

1. **supp 对 PerturbNet 的帮助在所有指标上同向：** `perturbnet supp` 相对 `perturbnet base`，mse_deg 0.2025 → 0.1585，pearson 0.133 → 0.170，direction 0.550 → 0.688，是八个 arm 里 mse_deg 最低的，比 `predict 0`（0.1802）低 0.022。这里 supp 以 74,355 个神经元细胞进入 scVI 的细胞表示，gold 只有 4,250 个训练细胞，细胞表示主要由 supp 决定。
2. **GEARS 和 MLP 上，supp 的效果不一致：** GEARS 的 signed ppi 的 mse_deg 低于 base（0.1675 vs 0.1771），gated 没有；MLP 的 signed ppi 反而比 base 差，gated 的 pearson 和 specific pearson 最高但 mse_deg 比 base 高。六个 arm 的差别都在 0.01 以内，没有一个 arm 在所有指标上同时占优。
3. **GEARS 和 MLP 的六个 arm 几乎没有学到扰动之间的差异：** 它们预测的跨扰动变化只有数据真实变化的 0.8%–6.5%，预测接近每个基因的常数，所以它们的 pearson 和 top-K 描述的是这个常数，而不是对扰动的排序；在 mse_deg 上它们与 `predict 0`（0.1802）只相差 0.003–0.013。PerturbNet 两个 arm 的预测跨扰动的变化约为数据的 97%–109%，这只说明预测不是常数，不能说明预测准确。
4. **gated 的门控行为：** 合成梯度与 gold 梯度的 cosine 平均为 +0.20（GEARS）和 +0.11（MLP），有 79% 和 61% 的步数合成梯度参与更新。合成梯度范数只有 gold 的 8% 和 4%，幅度控制项等于 1，权重就是 cosine，所以平均权重只有 0.25 和 0.16，gated 与 gold-only 的差别因此很小。
5. **可靠性与样本量的限制：** gold 在全部基因上的 split-half 可靠性只有 0.14，对应 top-20 DE 基因是 0.76，所以全部基因上的 Pearson 本身被噪声压低，以 top-20 DE 的指标（`mse_deg`、`direction`）为主更合适。测试集只有 8 个扰动，其中 PTEN 占 1,509 个细胞，CPNE1 只有 3 个细胞；训练集 21 个扰动共 3,728 个细胞。这次只跑了一个 seed 和一次划分，上面各 arm 之间 0.01 量级的差别不应当作为排序结论。
