# Circuit semantics：文献筛选与方法选择

日期：2026-09-29。用途：解释本项目已定位路径各自承载的变量与执行的操作。不是投稿评审或完整的新颖性论证。

筛选 21 项相关候选，纳入以下 14 项。采用论文原文、官方会议记录与原作者方法文章；排除不明来源、仅有搜索摘要的证据和 MDPI。详细检索记录见 [search-notes.md](search-notes.md)。

## 文献表

I/C/N 分别为洞见、完整性、数值证据的 1–5 初步阅读评分；A 为优先阅读，B 为有用支持。分数是本次阅读判断，非接收概率或复现实验结论。理论工作与纯 benchmark 的数值项不评分；理论证明未逐条审计。结构化版本含分类和说明，见 [papers.csv](papers.csv)。

| # | 论文与来源 | 类型 | I/C/N | 优先级 | 对本项目的用途与适用边界 |
|---|---|---|---|---|---|
| 1 | [Causal Abstraction: A Theoretical Foundation for Mechanistic Interpretability](https://www.jmlr.org/papers/v26/23-0058.html), JMLR 2025 | theory/proof | 5/3/NA | A | 用高层变量与神经干预的对应关系定义解释；本次核验官方摘要与论文记录，完整性评分暂定，未审计证明 |
| 2 | [Finding Alignments Between Interpretable Causal Variables and Distributed Neural Representations](https://proceedings.mlr.press/v236/geiger24a.html), CLeaR 2024 | pure method | 5/4/4 | A | DAS 学习可交换的表征子空间；不用假定一个神经元对应一个概念；随机网络控制提醒拟合能力本身也能制造成功 |
| 3 | [CausalGym: Benchmarking causal interpretability methods on linguistic tasks](https://aclanthology.org/2024.acl-long.785/), ACL 2024 | pure benchmark | 4/4/NA | A | 29 个语言任务、Pythia 尺寸系列；最小对与任意输入—输出控制任务尤其有用；主要测受控语言现象，不能代表全部自然语义 |
| 4 | [RAVEL: Evaluating Interpretability Methods on Disentangling Language Model Representations](https://aclanthology.org/2024.acl-long.470/), ACL 2024 | method + benchmark | 4/4/4 | A | 用 Cause/Isolate 同时测属性改变与其他属性保留，提出 MDAS；任务侧重结构化实体属性，不能直接当作计算步骤追踪 |
| 5 | [Uncovering Intermediate Variables in Transformers using Circuit Probing](https://arxiv.org/abs/2311.04354), COLM 2024 | pure method | 4/4/3 | A | 冻结模型，学习稀疏 mask 找计算假设中间变量的组件；包括数一致等任务；仍依赖提出有意义的变量与后续因果验证 |
| 6 | [Circuit Tracing: Revealing Computational Graphs in Language Models](https://transformer-circuits.pub/2025/attribution-graphs/methods.html), 2025-03-27 原作者技术报告 | pure method | 5/4/4 | B | 特征间归因图连接变量与下游作用，再做干预验证；需训练替代模型，有近似误差，原方法局部图固定 attention pattern |
| 7 | [Tracing Attention Computation Through Feature Interactions](https://transformer-circuits.pub/2025/attention-qk/index.html), 2025-07-31 原作者研究更新 | pure method | 4/3/3 | A | QK 特征交互解释注意力选取，head loadings 定位边由哪些 head 实现；是方法与案例，不能等同普遍语义识别保证 |
| 8 | [Causal Scrubbing: a method for rigorously testing interpretability hypotheses](https://www.lesswrong.com/posts/JvZhhzycHu2Yd57RN/causal-scrubbing-redwood-research), Redwood 2022 原作者方法文章 | pure method | 5/3/2 | B | 保留假说变量、重采样无关细节来验证子图；需要手工指定假说及等价类，不自动发现解释；非同行评审论文 |
| 9 | [Is This the Subspace You Are Looking for? An Interpretability Illusion for Subspace Activation Patching](https://openreview.net/forum?id=Ebt7JgMHv1), ICLR 2024 | pure method | 5/4/4 | A | 展示输出干预成功却不反映原生机制的构造与实例；要求检查方向怎样被下游使用，不能据此断言所有 DAS 都失效 |
| 10 | [How Much Do Circuits Tell Us? Measuring the Consistency and Specificity of Language Model Circuits](https://arxiv.org/abs/2605.08348), 2026 预印本，v2 09-02 | pure method | 4/3/3 | B | 系统区分跨输入稳定性与功能特异性；组件和神经元粒度存在不同权衡，反对只凭结构集合命名功能；尚无已核验正式会议归属 |
| 11 | [Activation Oracles: Training and Evaluating LLMs as General-Purpose Activation Explainers](https://arxiv.org/abs/2512.15674), 2025 预印本，v2 2026-01-06 | system/tool | 4/4/3 | B | 从内部激活回答自然语言问题，可产生候选标签；作者明确将它与机制理解区分，存在错误与置信度问题；没有核验可直接接本项目的 oracle |
| 12 | [Faithful Bi-Directional Model Steering via Distribution Matching and Distributed Interchange Interventions](https://arxiv.org/abs/2602.05234), ICLR 2026 | pure method | 4/4/4 | A | ConceptDAS 用反事实完整分布匹配约束双向干预；可改进仅最大化答案词的目标，仍需原生机制验证；并非所有设置都胜过对照 |
| 13 | [Inference Time Causal Probing in LLMs](https://arxiv.org/abs/2605.07631), 2026 预印本 | pure method | 3/3/3 | B | HDMI 从模型输出 margin 的梯度取得干预方向，与梯度式方向搜索直接相关；作为 steering/probing 对照，输出改变不等于发现计算语义 |
| 14 | [MIB: A Mechanistic Interpretability Benchmark](https://proceedings.mlr.press/v267/mueller25a.html), ICML 2025 | pure benchmark | 4/4/NA | A | 分开 circuit localization 与 causal variable localization，四任务五模型；其变量定位设置中 DAS 优于所比方法、SAE 未优于神经元，不外推为普遍结论 |

## 哪些方法应组合使用

**变量识别：DAS、CausalGym、RAVEL。** 它们最直接补上“这个位置可编辑”与“它表示什么”的距离。可读出变量、可因果替换变量、能隔离其他变量是不同证据。先用低维原生消息，必要时再学习子空间，避免将大规模特征训练当作起步条件。

**计算过程：Circuit Probing、Circuit Tracing、QK feature interactions。** 第一个寻找计算中间变量的组件；后两个帮助追踪变量怎样产生下游效应、为什么选中某个 token。DAG 原生边可以减少接线定位工作，但边内部仍是混合向量；这并未自动解决变量命名或算法解释。

**解释检验：Causal abstraction、Causal Scrubbing、Interpretability Illusion。** 高层解释应产生可否证的干预预测，并经得住无关细节变化和下游机制检查。通过有限测试支持任务范围内的解释，不能唯一确定所有等价实现。

**发现候选：Activation Oracles、梯度方向、输入逆优化。** 优先用于提出候选变量或寻找解释反例。自然语言标签与优化出的输入本身都不是计算解释。

## 机会与已有方法的边界

| 问题 | 现状 | 对本项目可做的增量 | 需要的证据 |
|---|---|---|---|
| 学一个能交换语义的方向 | covered central claim：DAS、MDAS、ConceptDAS 已直接覆盖 | 研究语义如何通过原生来源边传输，以及微调改变其中哪个环节 | 消息交换、接收端追踪、跨词族迁移 |
| 为找到的路径贴标签 | mechanism gap：标签容易混淆相关性与因果角色 | 将控制信号、消息内容与接收端操作分别解释 | 系数/内容/完整消息对照与竞争假说 |
| 解释为什么注意某个词 | crowded but open：已有 QK feature 方法 | 使用验证过的语法/角色变量解释本架构的 query-key 匹配 | 所有候选位置的竞争与正常 norm/attention 下的有限干预 |
| 回路稳定是否等于单一功能 | negative-result opportunity：近期工作显示二者可分离 | 给稳定修复位置建立跨任务因果作用谱，允许一边多功能 | 相同 mask 在多变量、未见输入上的预测与特异性 |
| DAG 是否更容易解释 | 目前缺架构比较证据 | 在相近任务能力下比较解释复杂度、迁移和选择性 | 相同搜索预算、共同组件粒度与完整上游依赖统计 |

第一项的直接方法新颖性较弱，仍可转向新的机制证据：找出某段原生信息流如何承载和变换一个变量，再解释 predictor 微调与回滚为什么改变行为。

## 数据与评估入口

| 候选 | 指标与对照 | 适用程度与限制 |
|---|---|---|
| [CausalGym](https://aclanthology.org/2024.acl-long.785/) | 干预后 log-odds；DAS、均值差/探针类方向；任意映射控制 | 可借用语言最小对设计，先确认 checkpoint 能完成所选任务 |
| [RAVEL](https://github.com/explanare/ravel) | Cause 与 Isolate；MDAS 等因果特征方法 | 借用属性隔离原则；300M 未必具有原基准所需实体知识 |
| [MIB](https://proceedings.mlr.press/v267/mueller25a.html) | 分离路径定位与变量定位；多种定位/特征基线 | 可借用评估边界和基线，不把全部任务直接迁移成一次总分 |
| 本项目新的语法数交叉数据 | 反事实吻合、log-odds、其他属性保留、新词族/模板迁移 | 与现有回滚结果直接衔接；数据与实验尚未实施 |

CausalGym 的 benchmark 质量依据是清晰的语言变量、模型尺寸覆盖、控制任务和公开协议；主要局限是受控任务与干预选择对结果的影响。MIB 的依据是将两种研究目标分开并提供统一基线；主要局限是任务/模型范围有限、不能为所有解释给出语义真值。二者的数值项按纯 benchmark 记为 N/A，而非按榜单提升幅度评分。

## 引用与结论边界

- 任何“方向搜索”“因果变量定位”方法介绍应引用 DAS、RAVEL 或对应方法，不能把既有方法作为本项目的新方法。
- 原始 Circuit Tracing 对 attention 的处理与后续 QK 研究应区分，不能用前者单独支持“解释 attention 的形成”。
- 2026 预印本结果只在已读设置中引用，不写成公认结论。正式会议归属以已核验记录为准。
- 不从这份调研推导模型已存在主语数回路、75 个位置都具有单一语义，或 DAG 已优于 dense。这些都需要 [下一轮实验](README.md)。
