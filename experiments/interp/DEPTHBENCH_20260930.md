**DepthBench 对 DAGFormer 的适用性 — 2026-09-30**

结论：适合加入我们的架构分析。现有预训练 checkpoint 可以复用层级诊断；完整的深度扩展比较，需要在同一参数预算下重新训练不同深宽比例。它补充的是“模型有没有利用更多层做计算”的证据，不能单独给回路命名或解释具体语义。

后续进展：已完成 300M matched-step Dense / DAG 的首轮诊断，见 [真实结果与复现说明](../results/depthbench_20260930/README.md)。以下内容保留为评估前的可用性核查与接入方案。

本次完成论文、官方代码和 checkpoint 发布情况核查，没有运行 DepthBench，也没有提交新训练任务。检查的官方代码版本为 `edca05f8e5c62bd49f91b17461dadae822783eee`；本项目检查起点为 `c6f3095`。

**来源与可用性**

- [论文，arXiv:2609.32534v1](https://arxiv.org/abs/2609.32534v1)，2026-09-26 发布的预印本。
- [官方仓库](https://github.com/keyu-wang-2002/DepthBench)。已读取训练说明及分析代码。
- [官方示例 checkpoint](https://huggingface.co/aspect-ratio-scaling/preln-lr2e-3-llama-400M-L24-pretrain)：通过 Hugging Face API 确认公开可读，并存在 `step7600/config.json` 与 `step7600/model_and_optim/*.distcp`。只核对文件元数据，未下载或验证权重加载。
- [诊断定义与架构适配](https://github.com/keyu-wang-2002/DepthBench/blob/edca05f8e5c62bd49f91b17461dadae822783eee/analysis/depth_probes.py)，[causal score](https://github.com/keyu-wang-2002/DepthBench/blob/edca05f8e5c62bd49f91b17461dadae822783eee/analysis/compute_causal_score.py)，[checkpoint loader](https://github.com/keyu-wang-2002/DepthBench/blob/edca05f8e5c62bd49f91b17461dadae822783eee/analysis/analysis_utils.py)。

**它测什么，我们能复用什么**

论文主实验固定近似参数量和训练配方，改变深宽比例；另以层级诊断解释性能变化。比较对象包含 Pre-LN、HC、mHC、Full/Block AttnRes、MoDA 等。主套件约 400M、16–70 层，以 FineWeb-Edu 从头训练；HC 和 Full AttnRes 的深度收益最稳定。它提供一组证据，而不是把任意模型换算成一个“实际等于 N 层”的标量。[论文全文](https://arxiv.org/html/2609.32534v1)

| 部分 | 人话含义 | 我们的接入方式 |
| --- | --- | --- |
| 固定参数预算的深宽扫描 | 同样的模型容量，分给更多层是否更有用 | 新训练；原有 75M–1B 尺寸扫描不能替代 |
| Angular distance | 后面的层还在改变表示吗 | 现有 checkpoint，捕获逐层状态 |
| Causal score | 去掉前面一层，会怎样改变后面各层的更新 | 现有 checkpoint，适配跳层定义 |
| Permutation score | 换层顺序后，语言模型 loss 变化多大 | 需先定义 DAG 的换层操作 |
| Logit lens / 单层 pruning | 答案沿深度怎样形成，去掉各层损失多少 | 现有 checkpoint，适配读出及跳层 |

几何变化、干预敏感度都不能独立证明计算有用。尤其我们已有 pruning robustness：删层影响小，可能是替代路径承接功能，也可能是层未被使用，需要结合 loss 和深宽曲线区分。论文也没有把每层都不可删除作为有效深度的必要条件。

**对接时需要解决的实际差异**

1. **训练配方。** 官方使用 LLaMA 类主干、GPT-NeoX tokenizer、FineWeb-Edu、长度 2048；我们当前 [300M 配置](../../configs/fourway_300m_chinchilla.yaml) 是 OLMo2 类主干、OLMo tokenizer、Dolma、长度 1024，embedding 绑定方式也不同。现有结果适合在我们自己的配对模型间比较，不能把 NLL 数字直接加入官方表格。官方领域评估的 NLL 也不是我们 GSM8K few-shot 生成准确率。
2. **模型接口。** 官方 loader 读取其修改版 OLMo-core 分布式 checkpoint。我们的 [FourWay forward](../../src/model/olmo_graph.py) 显式读取各层 Q/K/V/O 权重，绕过普通 decoder block 的 `forward`；给原始 block 挂 hook 不足以获得正确状态。应在我们的执行路径加入状态捕获和干预接口，复用指标公式。
3. **跳层与跨层存储。** 官方已对 AttnRes 专门处理：跳过的层不新增可读源，并调整下一边界的混合。DAG 的 source 槽位与 predictor 输出一一对应，不能删除列表元素后让后续索引整体移动。把前一层输出复制进被跳过层的槽位，又会多出一个可读的重复源。接入前需分别明确“绕过层计算”和“移除该层可读源”的操作；报告采用的定义，不把两者混称为同一次删层。
4. **换层。** DAG 各层 predictor 输出头、local correction 输出维度随可用源数量改变，不能把整套层参数直接互换。先做“固定路由位置，只交换 backbone 变换”是可实现的扩展诊断，但不等同于连同路由一起交换完整层。该项放在适配第二阶段。
5. **读出。** 以每层实际写出的状态作为 DAG 的主诊断对象，额外记录 R 混合。Q/K/V 各头读取不同混合，不能声称存在一个与普通残差模型相同的唯一层输入。Logit lens 使用本模型最终 norm 和 LM head，并说明它读取哪种状态。
6. **基线身份。** 我们 [75M HC 配置](../../configs/pretrain_75m_hyperconnection.yaml) 注明 `hc_dynamic=False`；它不能代表官方动态 HC。最相关的外部比较对象是 Full AttnRes、MoDA，以及官方 HC/mHC；模型名称相近不等于实现相同。

以上接入判断来自官方分析代码与本项目代码的对照，尚未通过数值运行验证。

**参数预算有一个与我们尤其相关的问题**

当前 [FourWayPredictor](../../src/model/predictor.py) 对每个目标层分别预测所有更早源的 Q/K/V/R 系数。若层数为 L、头数为 H，路由输出坐标数为：

`(3H + 1) * (L(L+1)/2 - 1)`。

当前 predictor hidden size 为 512，输出矩阵和独立 bias 合计为坐标数乘以 513。固定 H=16，16 层时仅这部分约 3.39M 参数，70 层时约 62.44M；尚未计入 predictor encoder、embedding 和 local correction。这是由实现计算的参数量，不是实验结果。

因此做深宽扫描时必须把完整 predictor 和 correction 计入总参数量，重新求 backbone 宽度，并记录 FLOPs/实测吞吐。直接套官方的层数与宽度会破坏等参数比较。路由系数还允许负值，不能把它们当概率直接计算“期望路径长度”并称作有效计算深度。

**建议继续时的最小实验**

1. 选择现有、训练 token 数对齐的 Dense 和完整 FourWay DAG 预训练 checkpoint，先从 300M 做起，使用相同的 heldout 文本。保留各自原生架构及 predictor/local correction，不使用语义微调后的 adapter 作为架构主结果。
2. 实现状态捕获、原生读出和定义清楚的删层，先出逐层表示变化、下游更新变化、logit-lens NLL、删层 NLL 四组图。无干预 forward 与现有实现的 logits 对齐检查是关键接口验证。
3. 主结果保持模型正常动态执行；补充回放未干预时的有效路由，区分固定连接承载的依赖和 local correction 响应。外部 predictor 只看原始输入，不能把它描述成会自动观察到内部删层。Causal score 同时保存分母及未归一化变化，避免原始更新很小时比值误导解释。
4. 在接入正确后，用同一完整方法、同一训练配方做一个预算下的浅/中/深三档新训练，比较 Dense 与 DAG 的 NLL 随深宽比例变化。需要与官方数字直接比较时，再对齐官方 backbone、数据、tokenizer 和训练预算，加入 Full AttnRes 等基线。

前一阶段能够回答现有 DAG 是否表现出不同的跨层依赖；后一阶段才能支持“我们的连接设计让额外深度更有用”。如果收益主要来自更好的信息复用、没有更强的深度收益，也应按这个结果解释，不能用高干预敏感度替代性能证据。
