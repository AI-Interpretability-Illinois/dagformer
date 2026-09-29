# 2026-09-29：损伤机制、规则切换与组合回路

本轮检验三个研究假说，模型权重冻结。以下是实验设计，不是实验结果。
结果与实际执行范围记录在 `experiments/results/circuit_followup_20260929/`。

## 1. 损伤之后，谁接手读取功能

从上一轮确定的读取 head 出发，比较原始模型、删除主 head A、删除候选
备用 B、同时删除 A/B。候选 B 只在 discovery 输入上选择，固定后测试
新的复制内容和更远距离。随机对照保留干预层与组件数量。

DAG 分别正常计算 local correction，以及重放每个输入受损前所有位置的
有效 routing tensor（global predictor + local correction）。冻结参数本身
不会固定 routing 输出；global predictor 只读取 input IDs，无法直接感知
同一输入下的 backbone 损伤。

共同 head 粒度用于 dense/DAG 对比；DAG 原生来源边另用于检验早层信息
读取。来源边与 head 不直接等价计数。除准确率外，报告答案 margin、
条件二次损伤效应及配对不确定性。有限搜索窗口中的阴性不能排除其他备份。

## 2. 内容不变时，routing 能否切换操作

先评估原值读取、一次转换、两次转换的任务能力。使用示例和多种表达形式，
避免把不会指令跟随误当作没有相关内部计算。模板选择使用 discovery；
最终测试保留新的内容、表和输入样本。

移植的 donor 与 recipient 使用不同内容，目标答案由 recipient 的内容和
donor 的操作共同决定，不能直接复制 donor 答案。分别移植 predictor、
correction 与有效系数，保留接收者内容计算，并记录恒等干预检查。

若模型未掌握转换规则，进一步测试已掌握复制任务中的查询选择；这项结果
只解释查询选择，不称为算法或转换规则的切换。

## 3. 中间查找结果能否接入另一个例子的后续计算

例如 recipient 为 `oak→B; B→blue; C→green`，donor 为
`oak→C; C→red`。若只移植中间键 C，recipient 应输出 green，既不是其
原答案 blue，也不是 donor 答案 red。

首先测 one-hop、隐式 two-hop，以及显式中间步骤的能力。显式输入中间键
的成功不代表模型原本在隐式任务中完成了第一步。实际干预只作用于内部
中间消息，不直接编辑最终输出。用 discovery 搜索共享路径，在新表和
实体上验证；保持 donor 中间键、改变 donor 最终答案，以及改变 recipient
后半张表，检验效果是否符合中间变量的解释。

报告全部样本及原本能完成任务的配对子集，并给出子集覆盖率。若任务能力
不足，将其记录为该 checkpoint 上尚无法检验的机制问题。

初筛后另做构造性实验：让 donor 单独完成第一步查找，recipient 明确接收
原中间键，再移植 donor 最后查询位置的中间层 head 输出，测试 recipient
能否按新键继续查自己的第二张表。这检验单步计算能否拼接，不能当作原本
已有隐式两步能力的解释。除完整样本外，报告 donor 第一跳、recipient 原
查询及目标键的输入反事实查询的成功子集与覆盖率。

第一轮用第三颜色相对原颜色的 margin 排序后，发现大集合会诱导输出
donor 的中间键、同时降低颜色概率。因此保留该轮负结果，另用第三颜色的
全词表 log probability 在新 discovery 上选路径，使用新 heldout 验证；
不能将该自适应后续视为对初始方案的独立复制。

## 模型与执行

- 首先使用已有 300M dense 和 FourWay + local correction、step9000。
- 语义任务另评估已完成的 1B checkpoint。1B 有 V norm，加载与干预位置
  按实际配置处理；它与 300M 的训练语料不同，不据此推断纯尺寸效应。
- 本轮不更新 backbone、predictor 或 correction 的参数。
- 同输入恒等干预与实际 hook 命中用于确认实验实现；科学结论来自独立
  heldout 干预与对照，不来自这些实现检查。

相关方法：[Hydra effect](https://arxiv.org/abs/2307.15771)、
[conditional co-ablation](https://arxiv.org/abs/2607.01940)、
[causal abstraction](https://arxiv.org/abs/2106.02997)。本轮重点是上述机制
在本项目实际架构中的因果证据，未预设 DAG 相比 dense 有优势。
