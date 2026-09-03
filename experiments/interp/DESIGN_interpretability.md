# Interpretability 分析设计 — "Topology predictor 学到了什么" (2026-08-31)

**背景决策**：暂不 scale 到 1B+，以 ≤600M 结果投 ICLR。差异化不能只靠
"BPB 更低"（魔改架构太多），必须证明 **dynamic topology 是真的**：
predictor 输出的 per-token 路由是有意义、可解码、可操控的 pattern，
而不是等效于一组静态权重 + 噪声。

## 0. 对象

300M FourWay (`fourway_corrected`)，12 层 × 16 头。每 token 的路由向量
α ∈ R^3773：对每个目标层 l∈1..11，Q/K/V 每头对 (embedding+前l层输出)
共 l+1 个 source 的实值权重（16×(l+1) each），R 流共享（l+1）。
Predictor 是**外置** 2 层因果 encoder，只看 input_ids —— 所以
α 是纯 context 函数，抽取不需要跑 base model。correction MLP 是
per-token 的局部修正（看 hidden state），效果上 α_eff = α_pred + Δ_corr。

主 checkpoint：`checkpoints/fourway_300m_dagformer_mmap/checkpoint_step9000.pt`
（headline BPB 1.0224 所用），另有 1500..10500 共 7 个训练中 checkpoint
可画 emergence 曲线。

## 1. Claim 结构（paper 的证据链）

| Claim | 分析 | 判定标准 |
|---|---|---|
| **必要性**：per-token 动态本身带来收益 | eval-time swap ladder | NLL(dynamic) < NLL(static-mean) 且 gap 占 dynamic−identity 收益的可观比例 |
| **可解码性**：α 编码可解释的 context 特征 | linear probes + 对照 | probe(α) ≫ probe(token-id baseline)，且 α 残差化后 context 目标仍可解码 |
| **机制样例**：能指认一个具体行为 | induction（重复检测） | 重复位置 α 系统性偏移；static 化后 copy accuracy 显著掉 |
| **可操控性**：α 是因果杠杆 | steering（round 2） | 注入 domain-mean α 可预测地移动输出分布 |
| **涌现**：pattern 随训练出现 | 跨 checkpoint probe/统计 | probe 精度、identity 距离随 step 单调上升 |

## 2. Round 1 pilot（已提交：job 21666246，1×A40，~3h）

### 2.1 Swap ladder（`interp_swap_eval.py`）
同一份训好的权重，只在 eval 时替换 α，25 校准 / 25 测试 held-out 序列：

```
dynamic              α_pred(context) + corrections   ← 原模型
static_mean          α_pred → 校准集均值（跨 token/context 恒定），corr ON
shuffle_time         α 沿时间随机置换（保边缘分布，破坏对齐）
swap_context         用别的序列的 α（保时间平滑性，破坏内容对齐）
identity             α=[0..0,1]，corr ON（只剩局部修正）
*_no_corr            以上各挡再去掉 correction MLP
static_{q,k,v,r}     只静态化一路，其余动态（动态性住在哪条流）
dense_baseline       独立训练的 dense 参照
```

**解读表**：dynamic vs static_mean = 外置逐 token 动态的价值；
static_mean vs identity = 平均重连线的价值；dynamic vs no_corr = 局部
修正的价值；vs shuffle/swap = 内容对齐 vs 分布正则化。
**这同时是"组合复杂性 ladder"的零训练成本版**（见 §4）。

### 2.2 Probes（`interp_probe.py`）
特征 = 每 token α（3773 维）；对照 = ① 当前 token-id 条件多数类
（"只看当前词就够了"的零假设），② α 残差化（减去 per-token-id 均值 α，
剩下的可解码性必然来自 context）。目标：

- charclass（当前词性质，sanity）
- posbucket（位置 4 档）/ freqbucket（词频 4 档）
- **isrepeat64**（该 token 在前 64 内出现过 — context 属性）
- **nllquartile**（下一 token 难度 — "难的地方路由更用力吗"）
- **domain**（code / prose / latex 3 类，window 级 held-out）

附带输出 probe 权重按 (stream, layer) 聚合 → context 信息住在哪。

### 2.3 Induction（合成序列）
32 条"128 token 随机片段 ×8 重复" vs 32 条边缘分布匹配的纯随机对照。
- Δα = mean α(repeat, pos≥128) − mean α(random, pos≥128)，按 stream/layer
  聚合 + top-20 条目（位置混淆已被对照消除）
- copy accuracy / copy NLL @ dynamic vs static_mean vs identity vs dense
  → 若 dynamic 显著高于 static，得到最好讲的故事：
  **"predictor 检测到重复，当场改写接线"**

### 2.4 Emergence
7 个 checkpoint 上 isrepeat/charclass probe 精度 + identity 距离
+ per-entry std 曲线。

## 3. Round 2（视 round 1 结果，均为 eval-only 或极小训练）

1. **Steering**：prose 输入 + 注入 code-mean α → 输出分布向 code token
   偏移量；α-PCA 主方向推拉。
2. **Dynamic hotspot 图**：per-entry std(α) 热图（16×12×4 流），配 2.2
   的 probe-权重图对照——"少数连接承载大部分动态"是很干净的图。
3. **难度条件化**：per-token NLL(dynamic)−NLL(static) 对 token 类别 /
   位置 / 频率分层 → "动态拓扑在哪些 token 上赚到钱"。
4. **POS probe 升级**：用 spaCy 对 raw_text 标注 + 对齐到 OLMo token。
5. **FourWayStatic 训练 rung**（见 §4，需 1-2 天 A40）。

## 4. Framing："internal combinatorial complexity" 能不能 sell？

**结论：单独不够，但可以作为副线，用 2×2 mini-ladder + swap 分解撑住。**

- 只说"iso-param iso-data 但内部组合更复杂 → 更好"会被读成
  "又一个 wiring trick 在小尺度赢了"。要把"复杂性→收益"说成因果，
  需要一条**剂量-响应曲线**，而完整曲线（多复杂度 × 多尺度）训不起。
- 但复杂度轴可以拆成两个便宜的子轴，而且大部分 rung 已经存在：

| | 静态 routing | 逐 token 动态 |
|---|---|---|
| **层级粒度** | DenseFormer（已训 3 尺度） | MUDDFormer（已训） |
| **头级粒度** | FourWayStatic（`FourWayStaticPredictor` 已实现，需在 mmap 上补训 1-2 点） | **DAGFormer（已训）** |

  加上 dense 作原点，就是 granularity × conditionality 的 2×2 剂量表。
  若 DAGFormer > {行邻居, 列邻居} > dense，"细粒度×动态缺一不可"成立。
- **swap ladder 是它的免训练补充**：同一权重下 identity → static-mean
  → dynamic 就是 conditionality 单轴的三档，直接分解收益。
- 建议主叙事仍是 **dynamic topology**（必要性+可解码+可操控三条腿），
  "新 scaling 轴"降级为 discussion：说 evidence for，不说 established。

## 5. 风险与 kill 判据

- **static_mean ≈ dynamic**（gap < ~0.005 nats）：动态故事死，转
  granularity 故事（头级静态重连线 vs DenseFormer 层级），§4 表格仍成立。
- **probe(α) ≈ token-id baseline 且残差不可解码**：predictor 只是
  查表软化 —— 仍可讲"learned static+lexical rewiring"，但需要弱化标题。
- **两者都阳性但 induction 阴性**：故事仍立（去掉机制样例），找别的
  case study（domain 切换点、代码/自然语言边界）。
- 已知混杂：correction MLP 也逐 token（swap ladder 的 no_corr 挡专门
  隔离它）；probe 的 domain 语料来自 repo 内文本（风格差异大，够 pilot，
  paper 版应换 Pile/olmo-mix 的真实 domain 标签）。
