# 600M 配对评估与 biro 续训

2026-09-17：600M 两侧使用旧 streaming run 的 step 14000 checkpoint，
各自 Adam 已完成 14,001 updates / 7,340,556,288 training tokens。
参数量分别为 dense 682,710,528，DAGFormer 717,003,240。
模型均严格加载，无缺失或意外参数。架构属于 per-head FourWay + correction，
含 V-norm。完整 12B 续训已在 biro 提交；这里的结果属于 7.34B 中途模型。

## 普通任务

14 个任务使用相同 tokenizer、1024 context、FP32 softmax、few-shot seed
1234，保存逐样本输出。下列区间是文档配对 bootstrap，未做多任务校正，
也不代表跨训练随机种子的方差。

| 指标 | dense | DAGFormer | DAGFormer 改善量与 95% CI |
|---|---:|---:|---|
| WikiText BPB ↓ | 0.9762 | 0.9431 | 0.0330 [0.0310, 0.0352] |
| LAMBADA accuracy | 33.77% | 38.02% | +4.25 pp [3.10, 5.45] |
| HellaSwag normalized accuracy | 33.57% | 35.50% | +1.93 pp [1.34, 2.54] |
| ARC-Easy normalized accuracy | 44.07% | 46.21% | +2.15 pp [0.46, 3.83] |
| SciQ accuracy | 79.10% | 82.60% | +3.50 pp [1.50, 5.50] |
| BoolQ accuracy | 56.61% | 60.24% | +3.64 pp [2.08, 5.23] |
| OpenBookQA normalized accuracy | 29.60% | 26.00% | −3.60 pp [−6.20, −1.00] |

PIQA、ARC-Challenge、WinoGrande、MathQA、CommonsenseQA 和 SocialIQA 的
区间跨零。完整数据见 [配对表](standard_600m/paired_summary.md) 和
[机器可读结果](standard_600m/paired_summary.json)。GSM8K BPB 也改善，
但这是 teacher-forced likelihood；完整 1,319 题生成正在独立运行，尚无结果。

## Predictor 与 correction 依赖

位置均值在 WikiText train 的 64 个窗口上校准，干预在不相交的 128 个
test 窗口和独立 synthetic copy sequences 上测试。

| 固定权重的推理条件 | 平均 WikiText NLL |
|---|---:|
| dense reference | 3.23735 |
| 完整 DAGFormer | 3.08885 |
| 外部 predictor 替换为位置均值 | 3.08863 |
| 移除局部 correction | 5.82049 |

位置均值替换的 NLL 差为 −0.000216，配对 normal 95% CI
[−0.000565, 0.000133]；correction 置零的 NLL 增量为 +2.73164。
period 256 teacher-forced copy accuracy：dense 96.02%，完整 DAGFormer 97.98%，
predictor 位置均值 98.37%，correction 置零 6.15%。

在完整的 14 任务评估上，predictor 位置均值让 WikiText BPB 从 0.9431
变到 0.9434，LAMBADA 从 38.02% 变到 37.88%；多数 accuracy 的差异区间
包含零。GSM8K teacher-forced BPB 变差 0.00250。600M 也表现出对外部
predictor 输入内容依赖很弱、对局部 correction 依赖明显的现象；这是
推理干预结果，不等同于重新训练后的架构消融。

原始干预统计见 [routing dependence](routing_dependence/600m.json)、
[dense reference](routing_dependence/dense_600m.json)；普通任务中的冻结
predictor 对照见 [配对结果](frozen_predictor_600m/paired_vs_dagformer.md)。

## 运行位置与复现

本地模型在 `checkpoints/scaling_20260917`，tokenizer 与校准数据在
`checkpoints/pr_sync_20260917`。使用已保存的 eval 环境：
`/scratch/yurenh2/venvs/dagformer-eval-20260917/bin/python`。
普通 eval 入口为 `experiments/results/lmeval/run_eval.py`；routing 入口为
`scripts/eval_routing_dependence.py`。全部结果 JSON 记录参数、代码版本和
软件版本；`samples/` 保存逐样本数据，不提交 Git。

普通与 routing 评估代码为 `9275b87`。冻结 predictor 任务启动于同一代码，
运行期间仅新增了数据准备脚本与训练恢复选项；模型和 eval 实现没有改动。
完整训练与后续 eval 的依赖作业和数据恢复限制见
[续训记录](../../SCALING_COMPLETION_2026-09-17.md)。
