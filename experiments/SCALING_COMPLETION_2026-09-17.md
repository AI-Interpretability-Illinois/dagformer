# 600M / 1B 补齐方案

2026-09-17 在 Delta 重新核对了 checkpoint、Adam update count、训练 CSV 和
Slurm 日志。建议优先续完已有 1B run，同时评估 600M 的同预算中途 checkpoint。
本次是训练资产与预算核查，尚未提交新的训练作业或生成新的大尺寸评估结果。

## 架构核对

**1B 是当前 per-head FourWay + correction 架构的大尺寸版本，可以纳入。**
本次同时读取了 OLMo-mix step 31000 和较早 Dolma step 10000 的实际权重：
两份都含 60 个 predictor tensors、30 个 correction tensors 和 15 个 V-norm
tensors。全部 predictor / routing 参数名及形状与当前代码的 1B 实例完全匹配，
见 [checkpoint 结构核对](results/scaling_audit_20260917/architecture_shapes.json)。

- 同样的独立 encoder predictor：256 维、2 层、4 个 encoder heads，trunk 512。
- Q/K/V 按 attention head 路由，R 跨 head 共享；来源是此前各层输出。
- 每层局部 correction MLP 的 hidden 为 128。
- Backbone 从 300M 的 12 层、hidden 1024 扩为 16 层、hidden 2048；仍为
  16 个 attention heads。每 token 的 routing 坐标由 3,773 增至 6,615。

原 job `20794467` 的日志明确写了 `fourway_corrected (4-way per-head per-token)`
和 `Predictor [encoder]`。日志更前面的固定标题含 `300M` / `static`，不能用
那个旧标题判断实际执行分支。按用户确认的口径，V-norm 差异不影响架构归类。

## 实际保留的模型

以下路径均相对于 Delta `/work/hdd/bfqt/dagformer_checkpoints`。
每次 update 为 524,288 tokens，已由原作业日志确认是 8 个 ranks。

| 模型 | checkpoint | Adam updates | 已处理 tokens | 状态 |
|---|---|---:|---:|---|
| 1B dense，OLMo-mix | `pretrain_1b_baseline_olmomix_20b/checkpoint_step38160.pt` | 38,160 | 20.006830080B | 完成原定预算 |
| 1B DAGFormer，OLMo-mix | `fourway_1b_dagformer_olmomix_20b/checkpoint_step31000.pt` | 31,001 | 16.253452288B | 主文件、backbone side file、optimizer 均保留 |
| 600M dense，旧 streaming run | `pretrain_600m_baseline/checkpoint_step14000.pt` | 14,001 | 7.340556288B | 可以补同预算评估 |
| 600M DAGFormer，旧 streaming run | `fourway_600m_chinchilla/checkpoint_step14000.pt` | 14,001 | 7.340556288B | 主文件、backbone side file、optimizer 均保留 |
| 600M dense，新 mmap run | `pretrain_600m_baseline_mmap/checkpoint_step22900.pt` | 22,900 | 12.006195200B | 已在上一轮评估，尚无完成预算的对应 DAGFormer |

前四项的 update count 来自本次 CPU 读取 optimizer 状态；最后一项沿用
[上一轮训练预算审计](results/eval_20260917/provenance/training_budgets.json)。
定期保存的 checkpoint 文件名与训练结束后保存的文件名有不同的边界语义，
因此预算使用实际 update count。

1B 日志到 step 31910 / 16.730554368B，但最新可恢复权重只到 step 31000。
原 job `20794467` 的 stderr 明确记录 2026-08-09 因 48 小时时限取消。
未保存的 910 步不能计入续训起点。

600M 旧作业 `19272481` / `19272482` 都从 step 0 开始，有相同的全局
token batch；两者分别有 26 / 82 条 streaming error 记录。它们可以组成
同训练 token 预算比较，尚未证明每一步实际训练样本完全相同。
另一个 `fourway_600m_chinchilla_mmap` 目录仅有失败日志备份，未找到可用权重。

## 推荐顺序

1. **先补 600M 的现有配对 eval。** 两边都用 step 14000，复用本轮普通任务、
   predictor 位置均值、correction 干预和 copy protocol。这个结果对应 7.34B
   tokens，可以先提供 600M 的实测点。
2. **训练优先续完 1B DAGFormer。** 保持既有 architecture、optimizer、LR
   schedule 和数据顺序，完成剩余 7,159 updates / 3.753377792B tokens，再与
   已完成的 1B dense 在相同 20B 预算上评估。优先检查普通能力与 predictor /
   correction 依赖，随后在 1B 自身上定位解释性方向；300M 选出的头和 SAE
   编号不能直接移植成 1B 的同一机制。
3. **再补 600M 完整预算。** 可从旧 streaming 配对各自的 step 14000 继续到
   22,900 updates，每边还差 8,899 updates / 4.665638912B tokens。需要先恢复
   原来的数据流程；现有 12B mmap dense 属于另一条训练轨迹。若改用新的 mmap
   配方，应另立 run，并核对它与 baseline 的对应关系。

按各原 run 最后一条累计 throughput 粗算，且仍使用原 8×H200 节点：

| 续训部分 | 原日志 tokens/s | 纯训练时间估算 |
|---|---:|---:|
| 1B DAGFormer 16.25B → 20.01B | 96,853 | 10.8 小时 |
| 600M DAGFormer 7.34B → 12.01B | 106,950 | 12.1 小时 |
| 600M dense 7.34B → 12.01B | 294,068 | 4.4 小时 |

这是历史吞吐外推，不含排队、数据恢复、streaming 前缀跳过和新的 I/O 影响。
旧 YAML 注释中的运行时估计不作为本次预算依据。

## 当前需要解决的实际问题

Delta `/work/hdd/bfqt/data` 当前为空。原来三个索引
`pretok/olmo_mix_21b_full`、`pretok/olmo_mix_21b`、`pretok/dolma_v1_7_12b`
都不存在于配置指向的位置；已检查的 HF cache 也没有 OLMo-mix 镜像。
所以 checkpoint 可以接续，训练输入还需要找回或重建。重建时需要核对原
语料版本、抽样、tokenizer、packing 和样本偏移，不能只让路径存在就开跑。

1B 20B 用 OLMo-mix，75M–300M 主比较与 600M 用 Dolma。这些结果能补同一
架构的大尺寸配对验证；统一训练语料和配方的规模曲线仍是另一个实验问题。
续训保留已有设置。

当前 H200 队列中的 `fw550m` / `fw550mbp` 来自 `ept_archive`，不是本项目
的 600M 训练，本次未修改这些作业。

参考配置：[1B dense](../configs/pretrain_1b_baseline_olmomix_20b.yaml)、
[1B DAGFormer](../configs/fourway_1b_dagformer_olmomix_20b.yaml)、
[600M dense](../configs/pretrain_600m_baseline.yaml)、
[600M DAGFormer](../configs/fourway_600m_chinchilla.yaml)。

旧 600M checkpoint 的训练早于当前配置切换到 mmap，恢复时需要使用旧 run
对应的数据设置；上述当前 YAML 提供架构和预算参考。
