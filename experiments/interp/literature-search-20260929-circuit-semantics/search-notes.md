# 检索记录

日期：2026-09-29。目标是找到能检验“已定位的回路究竟承载什么变量、执行什么操作”的方法。筛选 21 项相关候选，纳入 14 项；时间覆盖 2022–2026，重点补查 2025–2026 的方法与反例。

## 检索范围

使用公开方法术语和论文标题组合查询，主要查询词包括：

- circuit semantics / causal variables / causal abstraction / interchange interventions
- distributed alignment search / RAVEL / CausalGym
- Circuit Probing intermediate variables
- Circuit Tracing feature interactions attention QK
- causal scrubbing / subspace activation patching interpretability illusion
- circuit consistency specificity 2026
- activation oracles / activation explanations 2025 2026
- inference time causal probing / gradient causal probing
- distribution matching distributed interchange interventions ConceptDAS
- Mechanistic Interpretability Benchmark MIB

外部查询没有发送项目内部结果、未公开假说、代码、模型路径或数据。模型消息结构和现有 mask 的判断来自本地只读检查。

## 来源与核验程度

- 官方来源：PMLR、ACL Anthology、JMLR、OpenReview、ICLR/ICML/COLM 正式论文记录。
- 原文来源：arXiv 论文记录与 HTML/PDF，Transformer Circuits 原作者报告，Redwood 原作者 causal scrubbing 文章。
- 读取了 DAS 的 PDF 方法与实验段；CausalGym/RAVEL 的方法、评估设计及主要结果；Circuit Probing 的方法和任务描述；Circuit Tracing/QK 的方法与局限；Illusion 的反例和干预解释；ConceptDAS/HDMI 的目标与方法；Oracles 的用途和局限；MIB 的两个 track；How Much Do Circuits Tell Us 的一致性/特异性设计。
- Causal Abstraction 本次以官方摘要和论文记录核验为主，未审计形式证明。其完整性评分为暂定值。
- 部分 web 页面直接读取失败或全文超过工具大小限制，改用公开 URL 下载 HTML/PDF 后提取文本；下载稿仅作本机临时阅读材料，不纳入仓库。
- 会议归属补核验：[DAS / CLeaR 2024](https://proceedings.mlr.press/v236/geiger24a.html)、[Circuit Probing / COLM 2024](https://openreview.net/pdf?id=gUNeyiLNxr)、[ConceptDAS / ICLR 2026](https://proceedings.iclr.cc/paper_files/paper/2026/hash/704357127afabbd5a6eb979a57810767-Abstract-Conference.html)、[MIB / ICML 2025](https://proceedings.mlr.press/v267/mueller25a.html)。ConceptDAS 的 arXiv v2 也注明 camera-ready 与会议归属。

## 候选账本

纳入的 14 项及逐项链接、类别、阅读判断在 [papers.md](papers.md) 与 [papers.csv](papers.csv)。下面 7 项经过相关性初筛，但未进入主表；未将摘要级阅读当作完整方法核验。

| 候选 | 初筛来源 | 暂不纳入主表的原因 |
|---|---|---|
| Interpretability at Scale: Identifying Causal Mechanisms in Alpaca（Boundless DAS） | [arXiv](https://arxiv.org/abs/2305.08809) | 与 DAS 同簇；当前先检验小的原生消息子空间，自动学习子空间大小可后续追加 |
| Causal Head Gating: A Framework for Interpreting Roles of Attention Heads in Transformers | [arXiv](https://arxiv.org/abs/2505.13737) | head 角色定位相关，但当前缺口更具体：变量内容及其下游使用 |
| Shared Semantics, Divergent Mechanisms: Unsupervised Feature Discovery by Aligning Semantics and Mechanisms | [arXiv](https://arxiv.org/abs/2606.08236) | 可用于未知语义候选发现；本次仅摘要初筛，先做有明确反事实标签的任务 |
| Verbalizable Representations Form a Global Workspace in Language Models | [原作者文章](https://transformer-circuits.pub/2026/workspace/index.html) | 与既有 verbalization/J-lens 路线相关；不作为这轮主要机制验证方案 |
| Building Better Activation Oracles | [arXiv](https://arxiv.org/abs/2606.02609) | oracle 能力与使用改进，先保留主论文解释其用途和局限 |
| Confidence and Calibration of Activation Oracles for Reliable Interpretation of Language Model Internals | [arXiv](https://arxiv.org/abs/2605.26045) | 解释器可信度相关，本轮不部署 oracle，尚不需要专门校准实验 |
| When Activation Oracles Learn Not to Read: Concept-Specific Blind Spots in Fine-Tuned Oracles | [arXiv](https://arxiv.org/abs/2607.23379) | 新的局限研究，摘要级初筛；不据此概括所有 oracle 或当前模型 |

## 排除与未知项

- 搜索中的聚合站、讨论转述与低相关同名论文只用于导航，不支持最终技术结论。MDPI 和无法追溯的稿件不纳入。
- How Much Do Circuits Tell Us、Inference Time Causal Probing、Activation Oracles 没有在本次核验中确认正式会议归属，按预印本记录。
- 未执行这些论文的代码，也未系统核验全部超参数或重算其数字；初步质量评分只反映已读证据。
- 没有确认适配本项目 300M checkpoint 的现成 Activation Oracle、SAE 或 CLT。
- 尚不知道当前 75 个坐标里哪些携带主语数、哪些只改变信息使用方式；不能从分布计数补出语义答案。

## 本地证据与后续接口

- 现有数值均引用先前已归档实验。唯一新增分析是从三组 `results.json.gz` 读取 `all_masks.sva_75`，按 `mask_unit_labels` 计数；结果在 [mask_inventory.json](mask_inventory.json)。未重新运行模型。
- 阅读了 FourWay project-then-mix、local correction、Q/K norm 及消息互换实现，据此提出 hook 适配要求。
- 下一阶段优先实验见 [README.md](README.md)：主语数与干扰名词数交叉，比较原模型、微调模型、修复模型的消息内容和接收端使用方式。
- 写作时可引用这些方法解释证据标准；在实际实验完成前，不将计划表述成已识别的语义回路。
