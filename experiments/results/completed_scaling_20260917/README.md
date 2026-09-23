# 完整预算模型评估：2026-09-23 更新

1B dense 与 DAGFormer 均已完成约 20.00683B training tokens 的评测。
DAGFormer 作业 22283422 于 2026-09-22 04:18:44 CDT 完成，耗时 02:25:28，
包括 14 项普通任务、routing 干预、固定 predictor 复测及完整 GSM8K。
600M 两个续训作业 22160902/22160903 在 9 月 23 日仍因 Priority 排队。

| 指标 | Dense | DAGFormer |
|---|---:|---:|
| WikiText BPB ↓ | 0.91638 | 0.88057 |
| WikiText word PPL ↓ | 29.8636 | 26.1508 |
| LAMBADA accuracy | 41.98% | 46.98% |
| HellaSwag acc_norm | 37.89% | 41.67% |
| BoolQ accuracy | 58.65% | 51.16% |
| GSM8K flexible extraction | 1.36% | 1.82% |
| GSM8K strict match | 1.06% | 0.99% |

[普通任务配对统计](standard_1b/paired_summary.md)中，WikiText、LAMBADA、
HellaSwag 等提升的文档配对 95% CI 不跨零；BoolQ 下降也不跨零。
[GSM8K 配对统计](gsm8k_full_1b/paired_summary.md)的 flexible 提升 CI 跨零。
这些区间衡量文档不确定性，不代表多训练 seed 的稳定性。

固定为按位置平均的 predictor 后，WikiText BPB 为 0.88062，几乎不变。
[干预记录](routing_dependence/1b.json)的 128 个自然文本窗口：完整模型 NLL
2.94441，predictor 固定位置表 2.94427，correction 置零 4.95114。
这支持本次模型的外部 predictor 对输入内容依赖很弱、correction 很重要；
不等价于整个路由机制没有输入依赖。详见 frozen_predictor_1b 的配对表。

原始逐样本结果在 Delta home：
`/u/yurenh2/dagformer-20260917/results/completed_scaling`。
汇总作业 22283596 因短 commit ID fetch 失败；9 月 23 日补跑时又修复了
汇总脚本仅识别 m、不识别 b 尺寸的问题。上述三份配对汇总已成功生成。
