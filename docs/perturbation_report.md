# Perturbation-response 实验报告（GEARS-shaped backend）

日期：2026-09-27。数据来源：`artifacts/runs/kosmos-perturb-full/`（全量）、
`artifacts/runs/kosmos-perturb-smoke/`（smoke）。两者都走同一条实现：
`kosmos run --task perturbation` → `discovery_bridge.run_data_task` →
`perturbation.run_perturbation_task`。

---

## 0. 结论速览

* 两个 run 都完整跑通，产物齐（contract / graphs / results / metrics / summary / figures）。
* **全量 run（784 基因 panel）：gated 最好**——`mse_deg` 0.2096 vs base 0.2421
  （−13.4%），Pearson 0.397 vs 0.335，direction 0.900 vs 0.833。
* **smoke run（666 基因 panel，3 epoch）：gated 反而最差**。arm 顺序随 panel 翻转，
  与 handoff §6.1 的判断一致：**panel 决定结论**。
* 两个真正的限制被这次全量 run 量化了：
  1. **panel 只有 784 / 5045（16%）**——Norman 与 Replogle 的基因符号交集；
  2. **共表达图极稀疏**：G_C 只有 **62 条真实基因–基因边**（另加 784 个自环），
     G_S 121 条。也就是说辅助图 G_S 真正新增的边约 88 条，而 GO 图有 25,204 条。
     "gated 与 ungated 差距小"主要是这个原因，不是门控没用。

---

## 1. 数据结构

### 1.1 一行样本是什么（perturbation contract）

和"取列预测"分支不同，一行不是"一个细胞的标签"，而是一个**被扰动的细胞**：

```text
control        : (G,)  该细胞所在 context group 的对照均值 μ_ctrl
perturbation   : tuple 被扰动的基因符号，单个或多个（'CBL' 或 ('CBL','PTPN12')）
delta          : (G,)  Δ = y_pert − μ_ctrl      <- 训练目标
target         : (G,)  y_pert = control + Δ
context        : dict  cell_type / donor / batch / dose / time（本 run 为空）
control_group  : str   μ_ctrl 取自哪一组（本 run 全是 "global"）
```

* 基因 panel 与 `gene_to_index` 由 gold 与所有补充源的**交集**一次性确定，
  所有数据集、所有图共用同一套节点。
* split 按 **perturbation 身份**划分（`mixed` / `unseen_single` /
  `unseen_combination`），同一扰动的细胞绝不跨 split —— 否则是在测记忆。
* Δ 用 context 匹配的对照均值计算；缺 context 列时退化为全局均值
  （本 run `global_fallbacks=0`）。

### 1.2 本次全量 run 的实际数据

| 项 | gold = Norman (6154020) | supp = Replogle K562 essential (7458695) |
|---|---|---|
| 细胞数 | 91,205 | 162,751 |
| 对照细胞 | 7,353 | 10,691（实际使用） |
| 基因数 | 5,045 | 784（交集内） |
| 用途 | 训练/验证/测试 | **只用对照细胞**建辅助共表达图 G_S |

| 合同量 | 值 |
|---|---|
| panel（交集） | **784** 基因（gold 的 16.0%） |
| examples | 8,176 个扰动细胞 |
| perturbations | 32 |
| split（按扰动） | train 5,350 / validation 659 / test 2,167 细胞 |
| 测试扰动数 | 6 |
| min cells / perturbation | 2 |

note：这里 gold 基因 5045、supp 只有交集内的 784，所以 `panel_warning` 触发：
"the shared panel is 784 of the gold's 5045 gene(s) (16%)"。

### 1.3 产物清单（`run/` 下）

| 文件 | 内容 |
|---|---|
| `perturbation_contract.json` | panel、`gene_to_index`、splits、每个 split 的扰动、来源明细 |
| `graph_report.json` | G_C / G_S / G_GO 的阈值、k、边数、密度、度分位、孤立点、来源 sha |
| `perturbation_results.json` | 三臂的每 epoch 历史、验证/测试指标、逐扰动明细、config |
| `metrics.json` | 统一指标块（与取列分支同 schema：`value`/`baseline`/`delta`/`backend`…） |
| `summary.md` | 人读版结果表 + 图状态 + 门控状态 |
| `figures/perturbation_overview.png` | 三臂对比、Δ 预测 vs 观测、门控曲线、loss |
| `{arm}.pt`, `{arm}_test_predictions.npz` | 每臂权重与逐扰动预测均值 |
| `data_report.md/.json`（run 上一级） | 数据来源账本 |

### 1.4 与"取列预测"分支的对比

| | 取列预测（per_cell） | perturbation_response |
|---|---|---|
| 一行 | 一个细胞 | 一个**被扰动的**细胞 |
| 监督信号 | 标签列（cell_type 等） | Δ 向量（G 维） |
| 关键预处理 | 同 dataset 取列 | **同 dataset 内 control/perturbed 分类** + 求 μ_ctrl + 解析扰动基因 |
| split | 随机行（分层） | **按 perturbation 身份** |
| 无标注数据的作用 | PPI 校正 / 伪标签 | 建**辅助共表达图** G_S |
| 指标 | accuracy / macro-F1 / recall | mse / mse_deg / pearson / direction / top-K |

---

## 2. 三张图（本 run 实测）

| 图 | 来源 | 节点 | 存储边数 | 去重无向边 | 其中自环 | **真实基因–基因边** | 度均值 |
|---|---|---|---|---|---|---|---|
| G_C | gold 的 7,353 个对照细胞 | 784 | 908 | 846 | 784 | **62** | 1.16 |
| G_S | supp 的 10,691 个对照细胞 | 784 | 1,026 | 905 | 784 | **121** | 1.31 |
| G_GO | `go_essential_all.csv`（top-k=20） | 784 | 25,204 | — | — | 25,204 | 32.15 |

* G_C/G_S 用 |Pearson| ≥ 0.4、每节点 top-20，**对称化 + 自环**；度分位
  p50 = p90 = 1，说明绝大多数节点只有自环。
* G_C 与 G_S 的重叠：`shared = 817`，Jaccard 0.875 —— 但 **817 里有 784 是自环**，
  真实共享的基因–基因边只有 **33 条**（G_C 的 62 条里占 53%）。所以"Jaccard 0.875"
  不能读成"两张图几乎一样"，只能读成"两张图都很稀疏，且共享的那点边也算多"。
* G_GO 才是承载结构信息的那张图（25,204 条边，60 个孤立点）。

**含义**：辅助图 G_S 相对 G_C 只新增约 88 条边，所以 ungated 与 gated 的差别
天然就小；要放大未标注数据的作用，得先让共表达图不那么空（更多细胞 / 更低阈值 /
更大的 k / 换全基因组 panel），或把 G_S 接到更宽的结构上。

---

## 2.5 重要变更（2026-09-28）：增强方式改为"teacher 生成标签"，不是图增量

本文档 §3–§5 里那批 **5-arm 数字是在旧机制下跑的**：旧 `gears_augmented` 把 supp
当作一张图（`H_A = H_C + η·H_S`），第二梯度是 `∇(L_A − L_G)`，**没有任何合成标签**。

现在的机制（与模型无关，GEARS 与 MLP 共用 `kosmos/ppi/perturbation/synthetic.py`）：

1. 用 gold 训练一个 teacher（`gears_base` / `mlp_base`），冻结；
2. 取 supp 的对照细胞，配 gold-train 扰动 query，teacher 生成合成响应；
3. 重新训练一个同结构的 student，用
   - `*_augmented_ungated`：**signed PPI** `L_G + λ(L_S − L_pseudo_gold)`，
   - `*_augmented`：门控 `g_G + λ_t·∇L_S`。

`G_S` 共表达图仍保留（panel 已是交集，无副作用），但**不再是第二梯度的来源**。
下面的旧数字保留作为历史记录，**要重跑才有新机制下的结论**。

## 3. 原版 GEARS vs 本仓库修改版

### 3.1 逐项对照

| 维度 | 原版 GEARS (`/home/ydong233/GEARS`) | 本仓库 (`kosmos/ppi/perturbation/`) |
|---|---|---|
| 依赖 | PyTorch Geometric `SGConv`、CUDA | **纯 torch**，稀疏邻接 `torch.sparse.mm`，CPU |
| 数据 | `PertData`：`.h5ad` + `obs['condition']`、`var['gene_name']`；内置 split | 同一 `.h5ad` 经 datafetcher 转成 per-cell 表；**按扰动身份重新切分** |
| 扰动编码 | `pert_emb`：每个扰动一个可学习 **vocabulary embedding** | **compositional**：扰动 = 被扰动基因 embedding 之和（单/多基因同一机制） |
| 基因图 G_coexpress | 对照细胞的 |Pearson|≥0.4、top-k=20 → 位置 embedding 的 GNN | 同定义，G_C（gold 对照）/ 另加 G_S（补充源对照） |
| GO 图 | `G_go` 上再跑一个**独立 GNN** 得到 perturbation embedding（pert 词表上） | GO 用来**pool**扰动基因的表示（`states + G_GO·states`），不另设词表 |
| 第三张图 G_S | **没有** | **有**：补充源的共表达图，`H_A = H_C + η·H_S`（共享 GNN 参数） |
| 输出 | `control + Δ`；可选 uncertainty head（logvar） | `control + Δ`；**无** uncertainty head |
| loss | `Σ(pred−y)^(2+γ)`，γ=2，限制在 `de_idx`，+ direction loss | 同（`ERROR_EXPONENT=4`，`direction_lambda=0.1`，DEG=top-20 |Δ|） |
| 训练目标 | 单模型 | **三臂对照**：base / augmented-ungated / augmented-gated |
| 指标 | mse / pearson（含 *_de） | mse / mse_deg / pearson / spearman / direction_accuracy / top_k_overlap（逐扰动） |
| 遗传互作 / 预训练 | 有（GI 预测、pretrained checkpoint） | 无 |

### 3.2 三个实质性差异（会影响结果解读）

1. **compositional 扰动编码**取代词表 embedding。好处：没见过的**组合**
   （单基因没同时出现在训练里）也能编码，不必为组合新增一行参数；代价：失去了
   "每个扰动一个专属向量"的记忆能力，GEARS 用它拟合特定扰动的特异效应。
2. **GO 的用法不同**。GEARS 把 GO 图用在**扰动词表**上（perturbation-level
   GNN）；这里把 GO 图用在**基因表示**上做 pooling 后再做组合编码。两者共享
   "GO 帮助泛化到未见扰动"的动机，但参数化不同。
3. **G_S 是新增的**，也是"用无标注/其它 screen 数据"的落点。它只在
   augmented 两臂里生效，base 臂完全不看补充数据。

因此：本实现是 **GEARS-shaped，不是 GEARS-faithful**（handoff §6.2）。跨库比数字
要看这些差异，尤其是编码方式与 GO 分支。

---

## 3.3 新增：graph-free MLP baselines（Base MLP / Augmented MLP）

按设计文档加入两个**不使用任何图**的基线，用来和 GEARS 三臂对照。**关键前提：所有臂
共享同一个 objective**——比较架构时训练信号必须一致，否则结论是"损失 vs 架构"混在
一起。因此 MLP 臂默认直接使用 GEARS 的损失。

| arm | 输入 | 预测量 | 目标 | 损失 | 辅助数据 | 更新 |
|---|---|---|---|---|---|---|
| `mlp_base` | `(control, p)` | **Δ** | gold 实测 Δ | **GEARS `loss_fct`** | 不用 | `g = g_G` |
| `mlp_augmented` | 同上（架构完全相同） | Δ | gold 实测 Δ；补充行用老师生成的 Δ 标签 | 同上（gold + synthetic 两项都用它） | 补充 source 的对照细胞 | `g = g_G + λ_t·g_S`（复用 `gating.py`） |

* **统一 objective（默认）**：`PerturbationTrainingConfig.mlp_objective = "gears"`，
  MLP 臂调用 `kosmos/ppi/perturbation/losses.py::gears_loss`——每次扰动在其 **DEG 集**
  上 `Σ(ŷ−y)^(2+γ)`（γ=2，四次）+ `direction_lambda·Σ(sign(ŷ)−sign(y))²`，作用在 **Δ** 上。
  DEG 集合是**和 GEARS 三臂共用的同一份**（由 gold training split 的 top-20 |Δ| 计算）。
  因此五臂只差架构与"是否用补充数据"。
* 设计文档 §5 的 benchmark 形式（预测扰动后表达 + 全基因 MSE）保留为
  `mlp_objective="mse"`，但会破坏同 objective 比较；用了哪个会记进
  `results.config.mlp_objective` 与 `synthetic_label_metadata.json`。
* **架构**（固定，两臂一致）：两个独立的 ReLU 编码器 `control→128`、`p→128`，
  **相加**（非拼接），一个线性头输出 G 个基因。没有 GNN / GO / 共表达矩阵。
  输出被解释为 Δ（`y = control + Δ`），与 GEARS 同量。
* **`p`** 是基因空间指示向量：单基因置 1，多基因全部置 1；与 panel 共用 `gene_to_index`。
* **老师** 用 `mlp_base`（gold train 训练、gold validation 选择），对补充 source 的
  对照细胞生成合成标签；**补充 source 自己的扰动后测量值绝不作为目标**。
* **补充查询** 限定在 gold **训练**扰动量表内（每个补充对照细胞轮换一个），test 扰动不参与。
* **门控** 复用既有 detached 控制器（batch scope，κ=1，λ=1）：`cos≤0 ⇒ λ_t=0` 退化为
  gold 更新；**无有效补充行时 augmented 臂等价于 base**。
* **评估** 与 GEARS 三臂完全相同（perturbation-level、Δ 上的 mse/mse_deg/pearson/
  spearman/direction/top-K），所以五臂可直接比。
* **产物**：`mlp_base.pt`、`mlp_augmented.pt`、`teacher_checkpoint.pt`、
  `synthetic_label_metadata.json`（含 `objective` / `target` / `direction_lambda`），
  以及各臂 `*_test_predictions.npz`。

**开关**：`include_mlp_baselines` 默认 `False`（库调用/既有测试完全不变）；
autoresearch 入口默认打开，`PPI_MLP_BASELINES=0` 关闭。

## 4. base / ungated / gated 的区别

### 4.1 三个 arm

| arm | 目标 |
|---|---|
| `gears_base` | `L_G`（gold、实测标签） |
| `gears_augmented_ungated` | **signed PPI**：`L_G + λ(L_S − L_pseudo_gold)` |
| `gears_augmented` | 门控：`g_G + λ_t·∇L_S` |

（`L_S` = 模型在 supp 行上对 **teacher 合成标签** 的损失；`L_pseudo_gold` = 模型在 gold 行上对 teacher 标签的损失，控制变量。见 §2.5。）

### 4.2 门控的定义与性质（`kosmos/ppi/gating.py`）

```text
λ_t = max(0, cos(g_G, g_ΔG)) · min(1, κ·‖g_G‖ / ‖g_ΔG‖) · λ
g_final = g_G + λ_t · g_ΔG
```

* **gold 是兜底**：`cos ≤ 0` ⇒ `λ_t = 0` ⇒ 更新恰好等于 `g_G`。辅助数据只能
  "被忽略"，不能把模型推离监督方向。
* **辅助不会盖过 gold**：`min(1, κ‖g_G‖/‖g_ΔG‖)` 把辅助梯度幅度限制在 gold 的
  κ 倍以内（κ=1 即"至多和 gold 一样响"）。
* **控制器不可微**：cos 与缩放都取自 `detach` 后的梯度，用算术组合，而不是把两个
  loss 加权求和——否则模型会学会让辅助数据"看起来兼容"。
* scope=batch（默认）：每步一个辅助梯度；scope=sample 存在，但对"整源图增量"
  不适用（handoff 有说明）。

### 4.3 本 run 的门控轨迹（gated 臂）

| epoch | val mse_deg | cos(g_G,g_ΔG) | λ_t | ‖g_G‖ | ‖g_ΔG‖ | active |
|---|---|---|---|---|---|---|
| 1 | 0.290 | −0.017 | 0.046 | 6.12 | 43.20 | 47% |
| 5 | 0.114 | +0.112 | 0.089 | 3.83 | 10.11 | 57% |
| 6 | 0.103 | +0.190 | 0.206 | 3.58 | 3.18 | 80% |
| 10 | 0.061 | +0.130 | 0.157 | 3.24 | 2.01 | 80% |
| 14 | 0.072 | +0.105 | 0.130 | 2.89 | 1.72 | 82% |
| 18 | 0.102 | +0.093 | 0.122 | 2.77 | 1.67 | 76% |

读法：第 1 个 epoch 辅助梯度与 gold **略负相关**（cos=−0.017），门几乎关死
（λ_t=0.046）；随着训练推进，cos 转正并稳定在 +0.09~0.19，λ_t 升到 0.12~0.2，
同时 ‖g_ΔG‖ 从 43 降到 1.7（辅助增量越来越小、越来越对齐）。这正是门控设计的
预期行为：**先忽略、后按一致性接纳**。

---

## 5. 结果汇报

### 5.1 全量 run（panel 784，40 epoch 上限，早停）

| arm | mse | **mse_deg** | pearson | spearman | direction | top-K | epochs |
|---|---|---|---|---|---|---|---|
| `gears_base` | 0.0844 | 0.2421 | 0.335 | 0.120 | 0.833 | 0.250 | 11 |
| `gears_augmented_ungated` | 0.0846 | 0.2418 | 0.350 | 0.135 | 0.842 | 0.267 | 13 |
| `gears_augmented`（gated） | **0.0763** | **0.2096** | **0.397** | **0.166** | **0.900** | 0.217 | 18 |

* gated 相对 base：`mse_deg` **−13.4%**（−0.0325 绝对），Pearson +0.061，
  direction +6.7 个百分点，`mse` −9.6%。
* gated 的 **top-K overlap 反而略低**（0.217 vs 0.250）：它在"整块 DEG 的误差/相关/
  方向"上更好，但最靠前的 20 个基因命中率没有同步提高。两类指标不是同一件事。
* ungated ≈ base（差异在噪声内），说明**"加图"本身不等于有用**，起作用的是门控。

逐扰动（gated 臂，test 的 6 个扰动）：

| perturbation | mse_deg | pearson | direction | top-K |
|---|---|---|---|---|
| BPGM+ZBTB1 | 0.062 | 0.415 | 0.95 | 0.20 |
| CBL+PTPN12 | 0.398 | 0.348 | 0.75 | 0.35 |
| ETS2+CNN1 | 0.100 | 0.418 | 0.90 | 0.15 |
| ETS2+IKZF3 | 0.296 | 0.459 | 0.95 | 0.20 |
| LYL1+CEBPB | 0.165 | 0.394 | 0.95 | 0.25 |
| PTPN12+PTPN9 | 0.238 | 0.346 | 0.90 | 0.15 |

CBL+PTPN12 是最难的一个（它正是问题里点名的扰动）。

### 5.2 smoke run（panel 666，3 epoch，共 296/44/58 细胞）

| arm | mse | mse_deg | pearson | spearman | direction | top-K | epochs |
|---|---|---|---|---|---|---|---|
| `gears_base` | 0.1874 | 0.6881 | 0.321 | 0.187 | 0.863 | 0.275 | 3 |
| `gears_augmented_ungated` | 0.1693 | 0.8159 | 0.291 | 0.176 | 0.838 | 0.250 | 3 |
| `gears_augmented`（gated） | 0.2146 | 0.7464 | 0.320 | 0.174 | 0.788 | 0.262 | 3 |

### 5.3 两个 run 的对比与解读

* 全量 run 的所有指标都比 smoke 好一大截（如 gated `mse_deg` 0.746 → 0.210），
  因为 panel 更宽（666 → 784）、细胞更多、训练更久（3 → 18 epoch）。
* **arm 顺序翻转**：smoke 里 gated 最差，全量里 gated 最好。这正是 handoff §6.1
  说的"panel 决定结果"。在 panel 覆盖只有 16% 的情况下，两种排序都不该当作定论。
* 门控行为在两个 run 里一致（cos 从 ~-0.017/0.09 起步，λ_t 小），但全量 run 给
  辅助梯度更多机会变得对齐（active 40% → 76%）。

### 5.4 局限（读数字前必须知道）

1. **panel 16%**：784/5045，且只由 symbol 交集决定；换 Ensembl 主键映射或
   全基因组 panel 才可能突破（handoff §6.1）。
2. **共表达图近乎为空**：G_C 62 条、G_S 121 条真实边；辅助信息主要来自 G_GO。
3. **测试面很小**：只有 6 个 held-out 扰动（2,167 细胞），单 seed，置信区间会很宽。
4. **GEARS-shaped 非 faithful**：无 uncertainty head、无 GI、无预训练，编码方式与 GO
   用法都和原版不同（§3）。
5. 本 run 用了 `KOSMOS_SINGLE_CELL_MAX_CELLS=1000000 / MAX_GENES=0`，把每个 screen
   全量转成 CSV（supp 一个就 5.2 GB / 162,751 细胞），再读进 pandas；这是为了
   "全量"付出的代价，不是必须。

---

## 6. 工程接线状态（与训练无关，但影响可复现）

* `kosmos run --task perturbation` 与研究循环已接通；perturbation 走
  `discovery_bridge.run_data_task` → registry staging，不再被静默降级。
* 取列分支与 perturbation 分支现在共用同一套产物与 `DataTaskOutcome`。
* 两个 run 都是通过 `kosmos` CLI 跑的；smoke 用 `--max-epochs 3`，全量用默认 40。
* research loop 侧的两处坑已修：novelty 自比导致的重生成死循环、以及
  hypothesis JSON（JSON mode + 传 system prompt + 记录 finish_reason）。

---

## 7. 建议的下一步

1. **把 panel 做宽**（最高优先）：Ensembl 主键映射，或换全基因组 panel 的 screen；
   否则所有比较都在 16% 的基因上做。
2. **让共表达图有内容**：提高对照细胞数（已在全量里做了）、降阈值 / 升 k、
   或对 G_S 用残差化后的全细胞；现在 G_S 只多出 ~88 条边。
3. **多 seed + 多 split mode**：至少 `mixed` 与 `unseen_combination` 各 3 个 seed，
   6 个测试扰动的单次结果不足以排序 gated/ungated。
4. **若要和 GEARS 比数字**：要么装 PyG 直接跑 `/home/ydong233/GEARS`，要么明确
   声明是 GEARS-shaped（§3 的差异表）。
