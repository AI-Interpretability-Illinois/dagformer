# 600M / 1B 补齐方案

2026-09-17 在 Delta 重新核对了 checkpoint、Adam update count、训练 CSV 和
Slurm 日志。建议优先续完已有 1B run，同时评估 600M 的同预算中途 checkpoint。
用户授权后，已在 `biro-delta-gpu` 提交数据恢复、1B / 600M 续训与后续
评估链。600M step 14000 的普通和 routing 干预评估已完成，见
[600M 当前结果](results/scaling_eval_20260917/README.md)。

## 本次启动记录

模型及数据存储为 `/work/hdd/biro/yurenh2/dagformer-20260917`；代码、环境、
缓存和评估输出迁至 `/u/yurenh2/dagformer-20260917`，见
[存储布局](DELTA_STORAGE.md)。项目配额已实际核对：
soft 500 GiB / hard 550 GiB。旧 SSH ControlMaster 没有新组权限，但 Slurm
计算进程已获得 `delta_biro` 组，恢复和后续作业均通过它访问新目录。
日志留在 bfqt 镜像的 `logs/scaling_20260917`，方便现有 SSH 查看。

| 阶段 | 作业 | 依赖 / 目标 |
|---|---|---|
| 恢复语料及 checkpoint | 22160141，已完成 | 固定版本 263 个源文件；保留 optimizer |
| 1B 数据重建 | 22160736，已完成 | 按旧 reader 的实际顺序与文档选择恢复 21B cache |
| 1B DAGFormer 续训 | 22160737，已完成 | 数据完成后，8×H200，31,001 → 38,160 updates |
| 600M streaming 后缀冻结 | 22174301，已完成 | 按原 8 ranks，从每 rank 896,064 sequences 后取固定后缀 |
| 600M DAGFormer / dense 续训 | 22160902 / 22160903 | 同一份后缀；1B 启动后放行，8×H200，各 14,001 → 22,900 updates |
| 1B dense / DAGFormer eval | 22160905 / 22283422（重试） | 普通任务、完整 GSM8K；DAGFormer 加 routing 干预 |
| 600M dense / DAGFormer eval | 22160907 / 22160908 | 完成训练后按相同 protocol 评估 |
| 配对汇总 | 22160909 / 22160910 | 等待各尺寸两侧 eval 成功，再生成 paired CI |

这些是已提交的依赖链，不代表完整训练已结束。1B 数据恢复于 9 月 17 日
21:44 CDT 完成，写出 21,000,000,125 tokens；1B dense 的普通任务和完整
GSM8K 均已完成。完整状态与代码版本保存在
[启动记录](results/scaling_audit_20260917/biro_launch.json)。

1B 重建及训练代码固定在 `cb24428`，600M 训练与后续 eval 固定在 `6cfdc7b`。
600M 数据准备使用 `2586e43`。上一版本 `5c0b12d` 修复了不支持 HTTP range
requests 的问题，但作业 22161290 在 9 月 17 日 18:54 CDT 失败：aiohttp
默认 300 秒的总请求时限会中止仍在正常读取、分词的 shard，8 个 rank 均
反复触发，最终耗尽重试。新版本移除总时限，保留连接和空闲读取超时；
真实本地 HTTP 慢响应回归测试及冻结后缀顺序测试均通过。新的准备作业
为 22174301，两个 600M 训练的依赖已重新连接到它。
每个作业在 home 中使用对应 commit 的独立 worktree。训练保留 architecture、
原 optimizer 和原 LR schedule；DAGFormer 若 optimizer 恢复失败会报错退出。
完成检查使用 optimizer 的实际 update count，再导出供 eval 使用的模型。
在已分配的 A40 上，两份 DAGFormer checkpoint 已实际恢复 optimizer，
并通过一个 1024-token 窗口的反向传播检查：backbone、predictor / correction
和 routing biases 梯度均有限且非零，没有执行 optimizer step。
见 [GPU 恢复检查](results/scaling_audit_20260917/backward_audit.json)。

2026-09-21 更新：1B 续训于 9 月 19 日 21:52 CDT 完成，最终 checkpoint
核对为 38,160 次更新 / 20,006,830,080 tokens；715 条训练 NLL 记录均有限。
1B DAGFormer 首次 eval 作业 22160906 在写逐题结果时触及磁盘配额，已提交
22283422 重试，并更新 1B 汇总依赖。600M 两侧续训仍在排队。

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
   schedule 和重建后的历史数据选择，完成剩余 7,159 updates / 3.753377792B tokens，再与
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

9 月 18 日排队与完成时间的具体估算、条件和查询时间见
[时间估算](results/scaling_audit_20260917/eta_20260918.md)。

## 数据恢复的边界

Delta `/work/hdd/bfqt/data` 当前为空。原来三个索引
`pretok/olmo_mix_21b_full`、`pretok/olmo_mix_21b`、`pretok/dolma_v1_7_12b`
都不存在于配置指向的位置；已检查的 HF cache 也没有 OLMo-mix 镜像。
本次从固定 HF revision `99ee6aaace88779d1ef099d36251b91101c1679b` 恢复
原先 seed 0 / 相同文件上限的选择，使用已保存 tokenizer。

一致性检查发现，旧 `packed_token_stream` 的重试计数逻辑在首次迭代时也会
跳过交替文档；同时，本地 reader 按子集顺序遍历，没有按概率 interleave。
本次复现这两个实际行为。并行分词测试覆盖多 frame zstd、gzip、空白/坏 JSON、
跨文件文档奇偶性和 packing；实际 OLMo tokenizer 的 524,800-token 前缀与旧
reader 逐 token 相同。完成后 index 会记录真实的各子集 token 贡献。

旧 job 20794424 日志有三个解压失败的文件。恢复时完整排除这三个文件；
原日志没有记录失败前是否已输出部分文档，旧 cache 又已删除，因此不声称
新旧 cache 完全相同。具体文件与限制见
[重建记录](results/scaling_audit_20260917/olmo_reconstruction.json)。

600M 保留旧 streaming iterator 的名义后缀，将各 rank 数据离线冻结后按
rank 顺序交织。mmap 读取关闭额外 shuffle，续训 offset 相对 step 14001，
这样两个模型使用相同的后续样本。该序列无法复现原训练期间各自的 HTTP 错误
路径，因此仍只主张同预算与共享后续数据，不主张完整历史逐样本一致。

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
