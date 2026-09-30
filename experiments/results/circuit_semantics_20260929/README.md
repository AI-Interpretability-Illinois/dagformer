# 主语数信息的传递与读取

2026-09-29 开始，09-30 完成。复用 300M FourWay + local correction、step9000，以及上一轮 seed 20260930 的 predictor adapter。模型参数均冻结；本轮只拟合干预方向和诊断探针。修复状态使用已有 75 坐标的精确功能回滚。

**这轮识别出一个可因果交换的单复数方向，并找到使用这份信息的主要 attention head。** 在标注的主语位置，只交换一条 R 来源消息中的一维内容，就能让模型按交换后的数选择多个动词的词形。微调后这份信息仍然可读，但原有读取通道的作用大幅下降；75 坐标回滚会恢复该通道。只恢复一个 head 的路由则不足以修好主要测试句式。

这里解释的是一个局部计算过程，尚未解释全部 75 个修复位置或完整的语法算法。该数方向也不自动编码“谁是语法主语”：干预位置来自数据的角色标注。

## 1. 一维信息交换能改变什么

位置是 native coordinate **440：layer 3，R stream，source 3**，即第 4 层接收前三层计算后的隐藏状态。R 消息是 1024 维。在主语 token 处，将 donor 与 recipient 内容差的一个方向移入 recipient：

`z_patched = z_recipient + u * dot(u, z_donor - z_recipient)`

`u` 是单位向量；其余 1023 个正交方向、接收者系数和输入 token 保持不变，后续所有 norm、attention、local correction 正常计算。

例如，输入仍为 `The young author near the singer`，donor 为 `The young authors near the singer`。原模型在 is/are 两候选内给 are 的概率从 **0.005 变成 0.982**。反向交换也有效：`The young authors near the singer` 中的 are 概率从 **0.984 变成 0.004**。这不是改写输入句子或最终 logits。

| 原模型测试集 | 未干预语法准确率 | 均值差方向：按移植后的数作答 | 学习方向：按移植后的数作答 |
|---|---:|---:|---:|
| 新句子、同词汇和模板范围，384 例 | 99.74% | 99.48% | 99.74% |
| 未见名词与新长句式，384 例 | 100.00% | 86.20% | 83.33% |
| 未见名词、主语移到后面，384 例 | 100.00% | 96.88% | 91.67% |

学习方向第一行的 block-bootstrap 95% 区间为 **[99.22%, 100.00%]**，后两行为 **[79.94%, 86.72%]、[88.54%, 94.53%]**。这里比较的是 is/are 两候选，不是开放生成。

简单的单复数均值差已经有效，学习方向没有稳定超过它。学习版只是用完整词表的反事实 KL 及属性保留目标微调这个方向；它在 validation 选择第 25 步，不能把结果归功于额外训练模型。

对原模型，同数换词的平均绝对 logit 差变化在三个测试集分别为 **0.057、0.067、0.089**，相应的改数效应为 **6.897、5.768、8.192**。三个随机方向与打乱标签的均值方向效果很小。主语在前时，改变后面的干扰名词不会改变主语位置的因果隐藏状态，所以这项零效应是模型因果结构预期的，不能独立作为语义解耦证据；主语在后的测试中，改变前面的干扰名词带来的平均绝对效应仍只有 **0.020**。

同一方向也影响 has/have、does/do、was/were、seems/seem、works/work。heldout 上，学习方向的反事实吻合率分别为 **99.22%、93.49%、99.74%、99.48%、88.54%**。likes/like 的原始表现接近随机，不能用它声称解释了模型本来就具备的稳定能力。七组动词共享相同前缀，不是七套独立数据。

## 2. 下游由谁读取这份信息

在 discovery 上逐一恢复后续 head 的原始输出，检查前述信息互换效应消失多少，定位出 **L6H5、L5H14、L11H5**。编号均从 0 开始，最强的 L6H5 是第 7 层的第 6 个 head。

| 测试集、模型状态 | L6H5 消去的移植效应 | 三个 head 联合消去 | 按层和数量匹配的随机三 head，三次均值 |
|---|---:|---:|---:|
| heldout，原模型 | 79.01% | 95.40% | 0.89% |
| transfer，原模型 | 85.43% | 94.95% | 3.99% |
| role，原模型 | 53.45% | 57.39% | 0.51% |
| heldout，微调后 | -0.17% | 13.86% | -0.07% |
| heldout，75 坐标修复后 | 76.97% | 97.89% | 1.14% |

这些比例衡量的是**指定的一维消息移植造成的答案变化，有多少经由这些 head 传递**。实验将 head 输出恢复为接收者原值，没有删除 head；比例不是语法准确率，也不能把各 head 的数值直接相加。换成主语在后的句式，前三个 head 只覆盖约 57% 的效应，说明读取路径有句式依赖。

在原模型 heldout 上，仅恢复这三个 head 的最终预测位置，就消去了 **91.42%** 的效应。这支持它们参与在预测动词时读取数信息。它们的完整上游依赖仍在运行，不能把这三个 head 当作可独立执行的完整小模型。

## 3. 微调改变了信息的存在，还是后续使用

原模型训练出的固定线性探针，在三种模型状态、三套测试集的源端主语位置，都能以 **100%** 准确率读出主语数。该结果与一维因果交换一起说明，至少在这个位置，微调没有抹掉这份数信息。

| heldout 状态 | 模型原始语法准确率 | L6H5 分给主语的 attention | 三个原读取 head 的中介比例 |
|---|---:|---:|---:|
| 原模型 | 99.74% | 62.52% | 95.40% |
| 微调后 | 3.12% | 0.25% | 13.86% |
| 75 坐标修复后 | 97.92% | 56.14% | 97.89% |

普通位置和长句式上，同一个原模型方向在微调模型中会产生反向的答案效应；但主语移到最后时，微调模型本来就有 **91.67%** 的正确语法，方向效应也变回正。这排除了“模型无条件把所有单复数规则反转”这个过强解释。后续使用方式随句式变化，还存在未定位的计算路径。

目前可支持的局部描述是：**主语位置保留数信息；原模型的一组 head 在后面使用它选择动词形式；微调削弱了这组读取，完整局部回滚恢复了它。** 为什么微调后的其他路径产生相反行为，尚未完整解释。

## 4. 更直接的恢复实验：一个 head 不够

在看到以上现象后，另生成三套各 **256 例** 的新句子，排除先前所有定位和评估 prompt。仅恢复 L6H5 的外置 predictor 输出，其他 predictor 输出继续用微调值，所有 local correction 仍正常计算。

| 新 heldout 句子上的干预 | 恢复坐标数 | 语法准确率 | L6H5 主语 attention |
|---|---:|---:|---:|
| 原模型 | 3773 | 99.61% | 62.14% |
| 微调后 | 0 | 1.95% | 0.21% |
| 只恢复 L6H5 的 K | 7 | 1.56% | 15.30% |
| 只恢复 L6H5 的 QK | 14 | 1.17% | 16.68% |
| 恢复 L6H5 的 QKV | 21 | 1.95% | 16.68% |
| 原 75 坐标修复 | 75 | 98.05% | 55.84% |

新长句式上的结论相同：QKV 恢复从 **4.69% 到 8.20%**，原 75 坐标修复达到 **99.61%**。主语在最后时 QKV 恢复则从 **90.62% 到 98.44%**；全部条件和随机同层 head 对照均在 [TABLES.md](TABLES.md)。

因此，恢复部分主语注意力不足以修复原句式的整体行为。前面的中介实验验证了读取通道，后面的失败则说明，完整修复还依赖这个 head 以外的改变。本轮没有评估这些新缩小的修复对 IOI 的影响。

## 5. 实验设计与执行范围

- 初始 discovery 192 例；heldout、transfer、role 各 384 例。每个 block 含两组不同名词搭配，每组交叉主语数与干扰名词数，共 8 例。上下文互换保留 token 长度和语义位置对齐。
- 先扫此前的 75 坐标及 V 来源边，再因 all-stream 效果明显强于 V-only，追加 Q/K/R 的 discovery 分解。最后选中 R 坐标 440；这一步没有使用测试集选择。
- 每个干预方向只在前 16 个 discovery block 拟合，后 8 个选 checkpoint。学习版是 rank-one 内容投影，Adam lr 0.005、100 步；subject/last 两个位置均选第 25 步。损失权重为改主语数 0.5、干扰数及同数换词各 0.25，目标是完整词表 KL。
- 两个均值方向与三个随机方向是简单对照；打乱标签的均值方向没有采用与学习方向相同的优化预算。非学习均值方向本身已能跨词和句式转移，因此主要数信息结果不依赖学习优化才出现。
- 所有状态使用同一条在原模型中得到的方向。线性探针分别在各状态拟合，并额外保持原模型探针固定做跨状态评估。
- head 定位在原模型 discovery 上比较 128 个后续 head，选出前 3 个，随后在全部测试状态中固定使用。恢复实验是自适应追加，使用另一批 768 个不重复的新 prompt。
- 参数和完整前向与先前实验一致。主语数方向约束只改变所指定消息；输入 token、最终 logits、其他消息通道不直接替换。QK 概率由正常 norm/RoPE 后的 Q/K 做只读 FP32 计算用于诊断，实际模型输出始终使用原生 SDPA。
- 三阶段在相同输入上的原始 margin 最大差异为 **0**；系数、内容、完整消息的同输入恒等互换 logits 最大误差均为 **0**。检查记录见 [verification.json](verification.json)。

这是一个预训练 checkpoint、一个 adapter seed 的受控语法实验。没有 dense 架构或 LoRA 的语义机制对照，也没有证明 DAG 独有优势。与上一轮不同的准确率来自新句子集合，没有重新训练一个更强的模型。

## 6. 文件与复现

- [TABLES.md](TABLES.md)：所有状态、七组动词、区间与单 head 恢复。
- [overview.json](overview.json)、[variable_metrics.csv](variable_metrics.csv)：紧凑机器可读结果。
- `*.json.gz`：逐例 margin、完整词表候选 log probability、数据和选择过程；三个子目录分别是 `variable/`、`trace/`、`read_repair/`。
- [调研与原始方案](../../interp/literature-search-20260929-circuit-semantics/README.md)：DAS、变量隔离、路径追踪的来源和方法边界。本实验借用了信息互换与分布匹配思路，不是完整复现任一外部 benchmark。

从仓库根目录依次运行；`ADAPTER` 指向上一轮完整 predictor 的 `predictor_best.pt`，并保留原 `checkpoints/pr_sync_20260917/300m-dagformer` 检查点和 tokenizer：

```bash
export CUDA_VISIBLE_DEVICES=3
PY=/scratch/yurenh2/venvs/dagformer-eval-20260917/bin/python
ADAPTER=/scratch/yurenh2/interp_followup_20260929/runs/full_s20260930/weights/predictor_best.pt
"$PY" scripts/circuit_semantics.py --phase prepare
"$PY" scripts/circuit_semantics.py --phase discovery --adapter "$ADAPTER"
"$PY" scripts/circuit_semantics.py --phase decompose --adapter "$ADAPTER"
"$PY" scripts/circuit_semantics.py --phase confirmation --adapter "$ADAPTER"
"$PY" scripts/circuit_semantics_variable.py --adapter "$ADAPTER"
"$PY" scripts/circuit_semantics_trace.py --adapter "$ADAPTER"
"$PY" scripts/circuit_semantics_read_repair.py --adapter "$ADAPTER"
"$PY" scripts/report_circuit_semantics.py
```

已有结果会被前几个阶段复用；重新执行模型实验时，通过 `--out` 指定新的结果目录，报告脚本的默认目录与本归档一致。读取归档无需 GPU。代码和结果副本存放在 Delta home：

`/u/yurenh2/dagformer-20260917/experiments/results/circuit_semantics_20260929/`
