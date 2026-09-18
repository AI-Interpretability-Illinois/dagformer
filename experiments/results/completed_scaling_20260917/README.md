# 完整预算模型的评估

这是 biro 续训流水线的结果目录快照。当前已完成 **1B dense / 20.00683B
training tokens** 的 14 项普通任务，实际参数量为 1,279,395,840：

- WikiText BPB：0.91638。
- LAMBADA accuracy：41.9755%。
- 完整 GSM8K 1,319 题：strict match 1.0614%，flexible extraction 1.3647%。

完整记录见 [1B dense 普通评估](standard_1b/1b-baseline__custom.json) 和
[完整 GSM8K](gsm8k_full_1b/1b-baseline__custom.json)。作业 22160905 于
2026-09-17 18:09 CDT 完成，总耗时 1:02:24；其中普通任务计时 616 秒，
GSM8K 生成 2,023 秒，其余包含环境准备、加载和结果同步。
这是 baseline 单侧结果；1B DAGFormer 完成同预算续训后才生成配对结论。
DAGFormer routing 干预以及 600M 完整预算评估均已设置后续作业。

实际作业输出持续写入 Delta
`/work/hdd/biro/yurenh2/dagformer-20260917/results/completed_scaling`。
每个阶段会将紧凑结果同步到 bfqt 镜像的同名结果目录，完整逐样本数据留在
biro。Git 中的快照不会自动假定未完成作业已有结果。

作业与数据恢复记录见 [启动说明](../../SCALING_COMPLETION_2026-09-17.md)。
