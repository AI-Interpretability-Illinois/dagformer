**DepthBench 层级诊断：300M Dense / DAG，2026-09-30**

这轮最清楚的现象是：DAG 的中后层持续改变表示，最后几层对预测的贡献也更大；但整个后 3/4 网络的平均层间依赖分数低于 Dense，暂时不能据此声称 DAG 具有更深的有效计算链。绕过层计算与进一步切断输出槽位，得到的损失差异很大，需要分别报告。

已完成两种完整预训练模型、两个语料缓存，以及 DAG 的四种干预。结果来自真实 checkpoint 推理。没有使用此前语义实验的 adapter，也没有进行新的深宽扫描训练。

**比较设置**

- 两个 checkpoint 都是 step 9000：12 层、宽度 1024、16 个 attention heads。原训练审计确认各完成 9001 次更新，每次 524288 tokens，共 4,719,116,288 tokens。
- Dense 为 304,137,216 参数；完整 DAG 为 336,447,805 参数，包含 external predictor 与 local correction。这里控制了 backbone 形状和训练 token 数，未控制总参数量。每种架构只有一个预训练 seed。
- 主分析使用完整 WikiText test 缓存：128 × 1024 tokens。补充分析使用完整 Dolma 训练期评估缓存：50 × 1024 tokens。Dolma 缓存与原 mmap 训练数据的严格不重叠尚未建立，因此称其为训练领域内的评估缓存。
- NLL 使用缓存中全部有效的 next-token labels，包括每个窗口最后一个 token 的下一 token。与官方直接在输入内移位、只评 T−1 个位置的实现略有区别。
- 区间来自 2000 次配对窗口 bootstrap；窗口不等于独立原始文档，区间也不包含预训练 seed 的变异。

**实际结果**

| 指标 | WikiText Dense | WikiText DAG | Dolma Dense | Dolma DAG |
| --- | ---: | ---: | ---: | ---: |
| 原始 NLL，nats/token | 3.8258 | 3.6178 | 3.4559 | 3.3488 |
| 对应 PPL | 45.87 | 37.26 | 31.69 | 28.47 |
| 绕过最后一层的 NLL 增量 | 0.1035 | 0.5077 | 0.1100 | 0.4139 |
| 分别绕过最后三层、取平均的 NLL 增量 | 0.1295 | 0.3807 | 0.1582 | 0.3452 |
| 分别绕过第 4–12 层、取平均的 NLL 增量 | 0.4229 | 0.4577 | 0.4125 | 0.3411 |
| 后 3/4 网络的平均 causal score | 0.5867 | 0.5502 | 0.6092 | 0.5279 |
| 后 3/4 网络中 causal score > 0.45 的层对比例 | 58.33% | 63.89% | 58.33% | 55.56% |

表中的删层均为单层 identity bypass，DAG 保持原生动态执行。最后三层的均值是三次独立干预的平均，不是同时删除三层。后 3/4 网络指零基索引 s≥3、l>s 的 36 个层对；阈值先作用于每个层对跨 token 的平均分数，再计算比例。

原始 NLL 的 DAG−Dense 差值：WikiText 为 −0.2080，95% 区间 [−0.2164, −0.1999]；Dolma 为 −0.1071，[−0.1155, −0.0991]。平均 causal score 的对应差值分别为 −0.0365，[−0.0394, −0.0337]，以及 −0.0813，[−0.0856, −0.0764]。阈值比例的方向在两种语料上不一致。

WikiText 绕过最后一层的增量：Dense 为 0.1035 [0.0996, 0.1072]，DAG 为 0.5077 [0.4895, 0.5262]。因此“DAG 的末层仍然重要”有直接性能证据；“DAG 所有层都更抗删”不受本轮结果支持，WikiText 第 4–12 层平均 bypass 损失反而略大。

**跨层读取与路由响应**

| DAG 干预；第 4–12 层分别干预后平均 | WikiText NLL 增量 | Dolma NLL 增量 |
| --- | ---: | ---: |
| 绕过计算，正常动态路由 | 0.4577 | 0.3411 |
| 绕过计算，回放原始有效路由 | 0.4428 | 0.3507 |
| 绕过计算并屏蔽输出槽位，正常动态路由 | 2.0053 | 1.8059 |
| 绕过计算并屏蔽输出槽位，回放原始有效路由 | 1.9030 | 1.8932 |

Identity bypass 把上一层状态留在当前层的可读槽位，因此这些数值不能被解释成“移除了该层的一切影响”。第二类操作进一步把后续 Q/K/V/R 混合中指向该槽位的系数设为零；保持原索引，不对其余有正有负的系数重新归一化。后一操作同时改变混合内容和幅度，其较大损失说明可读通道的处理会显著影响删层结论，不能单独证明某段语义不可替代。

固定路由时回放的是无干预 forward 的 predictor + correction 有效系数；不是换成平均路由。外部 predictor 始终看到相同输入，内部删层改变的是 local correction 的输入。动态与回放的损失差异有正有负，不能概括成 local correction 总会自动补偿损伤。

几何变化也不等于新计算：DAG 的输出包含跨层 R 混合。补充保留了输出减去 R 后的 attention/MLP 更新及 causal score；在相同后 3/4 层对上，DAG 的这一分数仍低于 Dense（WikiText 0.495 vs 0.587；Dolma 0.476 vs 0.609）。原始更新范数和未归一化变化量都保存在数组中。

**图与完整数据**

- [WikiText 四组曲线](overview_wikitext_test.png)，[可编辑 SVG](overview_wikitext_test.svg)，[PDF](overview_wikitext_test.pdf)。左上：表示变化；右上：逐层 readout NLL；左下：单层 bypass；右下：DAG 干预方式对比。
- [WikiText 层间依赖热图](causal_wikitext_test.png)，[SVG](causal_wikitext_test.svg)，[PDF](causal_wikitext_test.pdf)。四个面板共享线性色标和三角遮罩。
- [Dolma 四组曲线](overview_dolma_eval.png) 与 [依赖热图](causal_dolma_eval.png)，另有同名 SVG/PDF。
- [summary.json](summary.json) 包含所有汇总、区间、参数量、配置和代码版本；[逐层 WikiText 表](layers_wikitext_test.csv) 与 [Dolma 表](layers_dolma_eval.csv) 便于直接取数。
- 四个 `<model>_<corpus>.npz` 保留逐窗口结果：`nll`、`angular`、`lens`、`update_norm`、`branch_norm` 及每个干预的 `skip_nll`、`causal`、`change_norm`、`branch_causal`、`branch_change_norm`。`lens` 最后一维依次为 CE、KL(final || layer)、top-5 overlap；无定义的矩阵下三角为 NaN。

Readout 用模型自己的最终 norm 和 LM head 解码每层写出的状态。DAG 的各头 Q/K/V 有不同输入混合，因此这里不是一个统一的“下一层实际输入”，也不是针对中间层另行训练的分类器。

**验证与复现**

原始 forward 与开启观察接口后的 logits 完全相等，最终状态经原生 readout 也完全相等；DAG 无干预有效路由回放同样完全相等，三个最大误差均为 0。干预检查确认上游状态不变、跳层为 identity、指定下游源系数确为零。完整评估中最终层 readout CE 与原始 NLL 的最大差异小于 5e−7，最终 KL=0、top-5 overlap=1。

数值评估代码版本为 `86ce024547d2cc4ae1b05b80cb6d00140353987e`。之后将 metadata 中本项目的绝对路径改为仓库相对路径，并补充汇总与绘图，未更改评估数组。环境为 PyTorch 2.10.0、Transformers 4.57.1、NumPy 2.3.5；原生 BF16 backbone、FP32 predictor/correction，未启用 Triton routing。已有 checkpoint loader 严格核对 DAG 的 backbone、predictor 与全部 correction 参数。

在仓库根目录和相同 checkpoint/cache 布局下运行；用环境变量选择空闲 GPU：

```bash
python scripts/eval_depthbench.py --kind dense --out experiments/results/depthbench_20260930
python scripts/eval_depthbench.py --kind dag --out experiments/results/depthbench_20260930
python scripts/report_depthbench.py --directory experiments/results/depthbench_20260930
```

已有结果目录需使用 `--resume` 继续同一配置，或指定新的输出目录。指标定义参考 [DepthBench 论文](https://arxiv.org/abs/2609.32534v1) 和 [固定版本官方代码](https://github.com/keyu-wang-2002/DepthBench/tree/edca05f8e5c62bd49f91b17461dadae822783eee/analysis)；DAG-specific bypass、source removal 和 written-state readout 的区别在 [protocol.json](protocol.json) 中明确记录。

要回答“增加层数后，DAG 是否比 Dense 获益更多”，下一阶段仍需固定包含 predictor/correction 在内的总参数预算，训练浅／中／深三个形状。本轮只给出现有 12 层模型的诊断，不能把较大的末层作用直接换算成一个有效层数。
