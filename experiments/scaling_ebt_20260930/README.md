# EBT 六条 scaling 轴与 Xiaocong PR 结果

2026-09-30。审阅 [PR #3](https://github.com/AI-Interpretability-Illinois/dagformer/pull/3)
的 `c9757183c3695ff2d688922d3a38fdc0dc70f7dd`。PR 现已合入 `main`（`1919dbd`），
六轴执行脚本与设置见 [EXECUTION.md](EXECUTION.md)。下面保留审阅时的已有结果。
这里把“六个尺度”解释为 EBT 正文中的六条 scaling 轴。

最新进度（2026-10-05）：第一 seed 已完成 23/28，数据量轴已跑齐。
[完整表格与训练状态](PROGRESS_20261005.md)记录了新结果，以及 Dense 对 batch size 的明显敏感性。

## PR 中已经有什么

结果在 [experiments/scaling/](https://github.com/AI-Interpretability-Illinois/dagformer/tree/c9757183c3695ff2d688922d3a38fdc0dc70f7dd/experiments/scaling)，
核心文件是 `README.md`、`results.md`、`runs_table.md`、`scaling_fits.json`，
以及参数量、FLOPs、训练 tokens、跨尺寸收益四张图。新 300M 是完整 12000-step
配对终点，PR 记录训练量 6.291456B tokens；与此前 step9000 那对结果不同。

以下是 PR 同一个 WikiText-2 cache 上的 token NLL；每行匹配语料、主干和训练量。
75M/150M 采用 timan12b，300M 采用 delta21b。它们不是同一预训练数据流。

| 标称主干 | Dense NLL | FourWay NLL | NLL 降低 | Dense / FourWay 实际总参数 |
| --- | ---: | ---: | ---: | ---: |
| 75M | 4.9967 | 4.7109 | 0.2858 | 76.56M / 105.66M |
| 150M | 4.2684 | 4.0846 | 0.1839 | 152.59M / 182.56M |
| 300M | 3.8078 | 3.5439 | 0.2639 | 304.14M / 336.46M |

300M 的 token-PPL 相对下降约 23.2%。这里的 NLL 不能直接和以前 harness 的
word-NLL、BPB，或采用其他 cache 的 DepthBench NLL 横向拼接。

PR 的 dense 曲线把新 300M FourWay 拟合到约 522.55M dense 的位置：
按主干分母是 **1.718×**，把 predictor、correction 和该 checkpoint 的 V norm
算进去则是 **1.553×**。75M/150M 的对应总参数倍率约 0.922×/1.081×。
这些量来自对已有 ladder 的插值：ladder 中参数和训练 tokens 同时变化，
并未直接训练一组 522.55M dense 与它进行同数据预算比较。

原始配对数字与推导保存在 [pr_pairs.csv](pr_pairs.csv) 和
[pr_review.json](pr_review.json)。主线方法为 `fourway_corrected`，完整保留
外部 causal predictor 和 local correction；`fourway_modular` 是另一种架构，
其数据单列，不能混入同一 FourWay scaling 曲线。

## EBT 实际怎么做

参照 [ICLR 2026 会议版本](https://proceedings.iclr.cc/paper_files/paper/2026/file/e19a65fd53b6f9a88b354da98813465d-Paper-Conference.pdf)
的 Figure 4、Figure 5、Appendix D.1.1 和 Tables D.1–D.2：

- 六条轴是 **数据量、batch size、深度、参数量、FLOPs、宽度**。
- 参数/FLOPs 图共用五个模型尺寸，非 embedding 参数约 6.18、12.4、48.8、176、395.5M；
  因此六条轴不等于六个模型尺寸。该组用 FineWeb、256 context、64 sequences/batch，
  tokens 随参数量按约 20 倍增加；前三个尺寸各三 seeds，后两个各一 seed。
- depth/width 实验分别改变层数或 hidden dimension；它们不是固定总参数的深宽互换。
- 参数/FLOPs 图使用 loss；另几张图展示 PPL。我们保留两种读数，拟合主要使用 token NLL。

## 我们的六轴方案

先用现有完整 FourWay，中心设置为 L=6、d=512、8 heads、FFN=2048：
Dense 总参数 76.56M，FourWay 总参数 105.66M。完整 predictor 的 encoder 为
256 维、2 层、4 heads，trunk=512，correction hidden=128。

新增四组实验使用同一份 Dolma-v1.7 21B corpus manifest、同 tokenizer、1024 context。
中心训练 1.572864B tokens；主干 LR=5e-4、predictor LR=3e-4，沿用 AdamW recipe。
所有新实验使用相同 norm 设置。主指标为八个分词 worker 等量采样的 held-out Dolma NLL，
共用 WikiText-2 作第二条曲线。GSM8K/MathInstruct 使用 gold-text NLL 作补充指标。

| 轴 | 设置 | 保持什么不变 | 新训练需求 |
| --- | --- | --- | --- |
| 参数量 N | 接入 PR 已有 ladder，再统一评测现有完整 600M/1B | 按训练语料和预算规则分组 | 先补数据和评测 |
| FLOPs C | 为同一批 checkpoint 记录完整训练计算量 | eval cache 与方法版本 | 首轮复用 N/D 两组模型 |
| 数据量 D | 0.786432 / 1.572864 / 3.145728 / 6.291456B tokens | L=6、d=512、global batch=512 sequences | 4 点×2 方法 |
| 深度 L | 4 / 6 / 8 / 12 / 16 层 | d=512、H=8、FFN=2048、tokens、batch | 5 点×2 方法 |
| 宽度 d | 384 / 512 / 640 / 768 | L=6、H=8、FFN/d=4、tokens、batch | 4 点×2 方法 |
| Batch | 64 / 128 / 256 / 512 sequences，即 65,536–524,288 tokens/update | 同模型、同总 tokens、同数据流 | 4 点×2 方法 |

四组共享中心点，去重后每种方法 14 个配置，**第一 seed 共 28 次训练**。
这是完整四轴方案的训练数，不包括已有 parameter ladder、后续多 seed 和补充等计算量终点。
因为统一采用 21B corpus，旧 12B 语料的中心 checkpoint 没有直接算作可复用的新实验。
新增模型总参数范围为 Dense 52.70–133.71M、FourWay 81.72–162.98M，
所以四条新轴不需要都在 300M 上跑。

先执行 depth 组，再接 data、width、batch；首轮得到形状后，为中心点和支撑关键斜率
结论的端点各补两个 seeds。具体配置、参数计数和 optimizer update 数在
[six_axis_plan.json](six_axis_plan.json)，已生成 28 份可执行配置。

数据预算不同的 run 各自完成其 LR schedule；长 run 的中途 checkpoint 另外标为学习曲线，
不替代短预算的训练终点。Batch 组固定总 tokens，调整 optimizer update 数；warmup 和
decay 按 consumed tokens 对齐，避免大 batch 因看到更多数据而表面获益。

## 怎样把结果讲准确

PR 已支持“同主干、同训练量时，FourWay 在这些尺寸具有更低 NLL”。当前 corrected
拟合虽有六个 run，只有三个不同参数尺寸；dense 为七个 run、四个不同尺寸。
单独拟合各自的 loss floor 和 exponent，不能仅靠 exponent 的排序证明更好的 scaling。
新分析要同时展示观测点、实际总参数、非 embedding 总参数、训练 tokens 和模型版本；
拟合报告共享 floor 的敏感性与 seed 变异。

FLOPs 包含重复的 source QKV projection、predictor 和 correction。PR 已计入这些主要项，
但采用解析 MAC 估计；它拟合的 1.81× compute multiplier 不等于已完成直接 iso-compute
训练对照。需要配合实际 GPU-hours；涉及等计算量优势的结论再用对应终点直接验证。

PR 的 Dolma-21B eval tail 主要是 flan/wiki，不能拿它作为跨语料的中性评测。
现有 1B 的 5B/10B tokens 终点也不与小模型的约 20 tokens/主干参数预算混成同一规则。
本次查看的 PR 快照里 1B routed common-eval 尚未提交；没有把预计完成时间当作结果。

深度组训练完成后可直接接已有 [DepthBench diagnostics](../results/depthbench_20260930/README.md)：
检查增添的层是否改变预测、层间计算依赖是否增加。EBT 式 fixed-width depth scaling
回答“加层是否有效”；此前讨论的 **100M 固定总参数深宽扫描**进一步回答
“同参数预算下分给深度是否更划算”，适合作为独立补充实验。

## 复现本次整理

`python experiments/scaling_ebt_20260930/build_review.py`

脚本只读取固定 PR revision，生成 CSV、审阅 JSON 与实验计划；不合并分支或提交训练。
参数量公式已与当前完整 Dense、FourWay 和 predictor 在 meta device 上实例化的
全部 8 种不同结构逐一核对，计数完全一致；各预算也核对了 tokens/update × updates。
这些检查没有运行训练。训练时间需要对新形状实测吞吐后估计。
