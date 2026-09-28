# Xiaocong 新 75M / 150M 实测审计（2026-09-28）

只读核对 PR #3 最新 head、timan1 上真实 checkpoint / CSV / eval JSON / pruning summary；没有修改同事目录，没有运行训练或合并 PR。

**结论：新结果不是 modular 全面胜出。75M 的 WikiText 与重剪枝有优势；150M 普通 eval 和 head/neuron pruning 仍以 fourway_corrected 更好。150M routing-column whole-block pruning 有增益，但这是单次训练/剪枝结果。**

## 普通 eval：补齐未推送的 150M

新 150M 的三个原始 JSON 已在 `/srv/local/xy51/lmeval/` 找到；当前 PR 的 `experiments/results/lmeval/timan1_dolma12b/` 只包含 75M JSON。旧结果来自本仓库 `experiments/results/eval_20260917/standard_matched/`。所有新 JSON 的 limit 均为空；9个重叠任务的样本数和task version与旧结果一致，batch size 8，max length 1024，0-shot。新原始JSON不含逐样本输出，所以这里不构造未经计算的paired置信区间。

SciQ 同时列 `acc` 和 `acc_norm`：Xiaocong README 报 acc_norm，我们此前主表报 acc，不能直接交叉比较。

### 75m

| 指标 | 旧 dense | 新 dense | 旧 corrected | 新 corrected | 新 modular |
|---|---:|---:|---:|---:|---:|
| WikiText word PPL | 174.7959 | 174.5932 | 147.4688 | 141.3030 | 138.2497 |
| WikiText BPB | 1.3931 | 1.3928 | 1.3472 | 1.3357 | 1.3298 |
| LAMBADA acc | 3.47% | 3.20% | 7.04% | 8.66% | 8.38% |
| SciQ acc | 41.30% | 42.60% | 46.40% | 50.00% | 46.70% |
| SciQ acc_norm | 40.30% | 41.50% | 45.80% | 46.70% | 44.90% |
| BoolQ acc | 37.95% | 38.07% | 42.48% | 42.63% | 48.84% |
| HellaSwag acc_norm | 25.85% | 25.94% | 26.25% | 26.23% | 26.25% |
| ARC Easy acc_norm | 30.13% | 29.67% | 30.18% | 29.63% | 30.68% |
| PIQA acc_norm | 56.04% | 56.15% | 55.77% | 56.09% | 56.91% |
| OpenBookQA acc_norm | 24.40% | 24.80% | 25.00% | 23.20% | 23.80% |
| WinoGrande acc | 50.28% | 50.59% | 51.38% | 49.80% | 50.43% |

### 150m

| 指标 | 旧 dense | 新 dense | 旧 corrected | 新 corrected | 新 modular |
|---|---:|---:|---:|---:|---:|
| WikiText word PPL | 84.1560 | 81.3458 | 71.5478 | 69.4108 | 72.3947 |
| WikiText BPB | 1.1959 | 1.1867 | 1.1521 | 1.1439 | 1.1553 |
| LAMBADA acc | 15.62% | 16.11% | 21.37% | 21.87% | 20.20% |
| SciQ acc | 55.00% | 54.40% | 60.00% | 62.50% | 57.80% |
| SciQ acc_norm | 48.80% | 50.00% | 50.40% | 55.60% | 51.00% |
| BoolQ acc | 40.92% | 61.22% | 50.67% | 59.94% | 61.87% |
| HellaSwag acc_norm | 26.27% | 26.54% | 27.31% | 27.26% | 27.12% |
| ARC Easy acc_norm | 33.54% | 33.29% | 34.85% | 35.06% | 35.02% |
| PIQA acc_norm | 59.41% | 58.81% | 60.17% | 59.96% | 60.45% |
| OpenBookQA acc_norm | 27.40% | 25.80% | 25.60% | 27.60% | 27.00% |
| WinoGrande acc | 50.28% | 47.83% | 52.33% | 52.57% | 50.91% |

## 预算、架构、数据对齐

| 模型规模 | 旧实际tokens | 新实际tokens | 新最终optimizer updates | 新global batch |
|---|---:|---:|---:|---:|
| 75M | 1,572,864,000 | 1,585,152,000 | 3,000 | 528,384 tokens |
| 150M | 3,145,728,000 | 3,170,304,000 | 6,000 | 528,384 tokens |

六个新最终 checkpoint 的 Adam step 均通过 CPU mmap 实际读取；并非将配置 total_steps 当作完成证据。新配置是 3 GPU × micro4 × accum43 × seq1024；比旧训练每步524288多0.78125%。同架构的新旧配置没有LR、模型维度、seed、optimizer或warmup变化；主要是机器、batch、重建语料路径、保存间隔不同。

| 规模 | dense实际参数 | corrected实际参数 | modular实际参数 |
|---|---:|---:|---:|
| 75m | 76,558,848 | 105,657,332 | 105,486,242 |
| 150m | 152,593,152 | 182,561,679 | 182,284,916 |

旧 corrected 的 global encoder predictor 不是冻结的：`fourway_corrected` + predictor LR 3e-4。所谓 standard connections fixed 指固定模块内连接，而不是所有路由都不可学习。新 modular 使用 `fourway_modular`，checkpoint routing_state_dict为空、没有local correction；它把来源改成attention/MLP模块输出并增加MLP/final readout路由。因此 modular 与 corrected 不是只切换一个预测开关的消融。

新训练cache `/srv/local/xy51/pretok/dolma_v1_7_12b/index.json` 实际有12,000,000,950 tokens，由独立分文件tokenization合并（books-0000..0002及c4-0000..0022）。老训练cache已丢失，不能验证新旧逐token相同；并行分文件packing和3-rank取样也不能假设与旧执行完全一致。

预训练评估cache则已逐token核验：新六个cache的50×1024 inputs彼此相同，而且与旧 `checkpoints/pr_sync_20260917/eval_cache.pt` 完全相同。但PR自己披露这些文档部分与训练span重叠；因此训练时的“held-out NLL”不应当成独立无污染验证，也不能断言重叠不影响架构排序。公开WikiText/lm-eval仍另列。

| 规模 | dense训练末eval NLL | corrected | modular |
|---|---:|---:|---:|
| 75m | 4.3389 | 4.2031 | 4.1935 |
| 150m | 3.7752 | 3.6707 | 3.6863 |

## 新 pruning：全部40个已完成setting

以下从 `/srv/local/xy51/prune/checkpoints/*/summary.json` 实际读取。每个run均2000步、65,536,000 fine-tuning tokens、seed42；域内MathInstruct held-out 2000 documents；域外WikiText-2。数值为 domain NLL / general NLL，越小越好。普通预训练eval与这些域适配后结果不能混为同一评价。

### 75m

| setting | dense | corrected | modular |
|---|---:|---:|---:|
| s0 | 2.3871 / 5.3061 | 2.1851 / 4.9947 | 2.1876 / 5.1090 |
| s30 | 2.4747 / 5.4775 | 2.2585 / 5.1426 | 2.2605 / 5.2463 |
| s50 | 2.6023 / 5.7300 | 2.3501 / 5.2939 | 2.3501 / 5.4375 |
| s70 | 2.8153 / 6.0026 | 2.6694 / 5.7545 | 2.5512 / 5.6821 |
| mod_s33 | 2.6236 / 5.7574 | 2.4966 / 5.6313 | 2.4787 / 5.6080 |
| mod_s50 | 2.8293 / 6.0713 | 2.7755 / 5.9760 | 2.6198 / 5.9278 |
| mod_s33_rc | — | — | 2.4120 / 5.4533 |
| mod_s50_rc | — | — | 2.6547 / 5.7494 |

### 150m

| setting | dense | corrected | modular |
|---|---:|---:|---:|
| s0 | 1.8540 / 4.5341 | 1.7239 / 4.2955 | 1.7546 / 4.4003 |
| s30 | 1.9299 / 4.7568 | 1.7883 / 4.4923 | 1.8227 / 4.5838 |
| s50 | 2.0379 / 5.0270 | 1.8824 / 4.7764 | 1.9226 / 4.8325 |
| s70 | 2.2477 / 5.4573 | 2.0729 / 5.1395 | 2.1294 / 5.2892 |
| mod_s33 | 2.1883 / 5.3375 | 2.0438 / 4.9315 | 2.0692 / 5.0224 |
| mod_s50 | 2.2984 / 5.4936 | 2.2291 / 5.4022 | 2.2560 / 5.4479 |
| mod_s33_rc | — | — | 2.0145 / 4.8730 |
| mod_s50_rc | — | — | 2.1301 / 5.1657 |

s0/s30/s50/s70是head+neuron的目标稀疏度；mod是整个attention和MLP模块；rc是routing-column score，其余用Taylor。75M mod_s33实际移除2/6 attention +2/6 MLP；150M因取整是3/8+3/8，即37.5%模块而非实际33%。

75M modular在70%细粒度剪枝比corrected好0.1182 NLL；150M同设定反而比corrected差0.0565。150M全部0/30/50/70%细粒度设定均是corrected更好。Whole-block rc在150M对同一modular模型优于Taylor（33%:2.0692→2.0145；50%:2.2560→2.1301），75M则33%更好、50%更差。

旧shared模型的75M half-block corrected NLL为3.4520，新同架构为2.7755；新dense为2.8293，旧dense为2.8323。这说明旧“75M DAG整块剪枝会崩”的观察对重新训练和剪枝轨迹敏感，不能直接归因于modular架构，因为新corrected本身已不再出现同样崩塌。

这些结果是mask/zero-weight实验，不是物理压缩或速度测试；未修改原矩阵维度。参数剩余数使用解析计数，之前已实测确认Q/K全head归一化让被mask head的Q/K仍影响输出，所以细粒度head的可删除参数计数还需修正。

**旧 global predictor 冻结对照不能忽略。** 旧shared模型50% head/neuron剪枝，75M可训练predictor NLL 2.3907、冻结后2.3884；150M为1.8989、冻结后1.8994，几乎不变。这说明在这些设定中更新global predictor参数并非恢复收益的必要条件；冻结参数的predictor仍随输入计算路由，local correction也仍可训练，因此该对照不能排除输入条件路由或局部路由变化的贡献。新modular 40个run中没有对应freeze-predictor消融，尚不能判断其恢复是否需要更新predictor参数。来源：PR `experiments/pruning/results/prune_summary.md`；参数冻结与逐输入计算分别见 `scripts/prune_finetune.py:318–324`、`:231`。

## 可以解释差异的证据与仍未知事项

- 最明显的排序变化来自实际架构变更、重新训练/剪枝轨迹以及指标口径。Modular在75M的少量优势没有在150M普通eval和细粒度剪枝中一致复现；不是与旧“corrected胜dense”结论全面矛盾。
- 新旧same-architecture普通能力多数接近且略有提高；新预算高0.78%，新语料重建不能验证与旧cache逐token一致。无法把提升单独归因于这0.78%或语料变化。
- BoolQ旧150M dense40.92%、corrected50.67%，新61.22%/59.94%/61.87%；这项变化很大。旧逐题审计显示dense有87.80%选择no。新JSON没有预测标签分布，无法确认具体是yes/no偏置改变还是能力变化；全部新分数仍低于常数yes的62.17%，不能用其单独证明理解改善。
- 新普通JSON提供task versions和完整样本数，与旧一致；但没有逐题scores、完整runtime revision，不能在本次只读审计中给出新旧paired CI，或排除所有backend/tokenizer实现差别。
- 六个新模型只有一个training seed、40个剪枝run也只一个seed；README所谓低于0.02 nats算噪声没有多seed依据。
- 对新modular尚未看到与我们旧checkpoint完全对应的predictor-position-mean、cross-sequence、query-routing等干预。普通和剪枝优势不等于已证明输入语义由predictor控制。
- 同事 `/home/xy51` 不可读；本审计使用可共享 `/srv/local/xy51` 真正输出与PR代码，没有更改权限、模型或日志。

原始已读结果、准确路径、config差异、40个summary，以及尚未单独推送的75M sweep全部20份原始trajectory内容，保存在同目录 `new_small_models.json`。

## 补充：MUDDFormer 结果属于另一组 150M checkpoint

本次新 75M/150M 三族是 dense、fourway_corrected、fourway_modular，modular 不能当作 MUDDFormer。PR 另有真实 MUDDFormer 剪枝结果：`experiments/coherence/results/exp3/prune_summary.md:3` 起的旧 streamed-Dolma 150M 三方实验；其 README `experiments/coherence/README.md:97` 明确要求只在这个 trio 内比较，不与新的 mmap 12B checkpoint 混合。

该旧 trio 的 MathInstruct 最终 NLL（dense / MUDDFormer / DAGFormer）为：0% 剪枝 2.0752 / 1.9348 / 1.9237；30% 为 2.1569 / 2.0369 / 1.9902；50% 为 2.2729 / 2.1531 / 2.0866；70% 为 2.5066 / 2.3665 / 2.2676。50% 冻结router参数后MUDDFormer为2.1500、DAGFormer为2.0782，说明这些设定中的恢复收益不要求更新相应router参数。DAG 的 global predictor 参数冻结时local correction仍可训练；MUDDFormer冻结全部router参数（该 README:118–125）。两者被冻结的路由网络仍可根据当前输入或激活计算不同路由，不能据此认定恢复仅依赖固定路由图，也不能排除推理时的输入条件路由。

## 补充：Xiaocong 的曲线具体画什么

- `/srv/local/xy51/compare_runs.py:18` 选择 `train/nll` 或 dense 的 `train/loss`；`:19` 选择 `eval/nll_soft` 或 dense 的 `eval/nll`。`:60–71` 明确左图是训练 NLL 的 10 个日志点滑动平均（100 steps），右图是每 500 steps 的 50 条 web-text eval。已查看 `compare_75m_12b_three_way.png`，标题及图形与脚本一致。`:45` 所谓 `Identity-wiring eval NLL` 则是同一 DAG 模型路由归零消融，不是独立训练的 dense baseline。
- `/srv/local/xy51/compare_history.py:10–13` 将 April 旧 corrected/dense（脚本标记 streaming books-only）、September modular mmap books-only、September modular mmap 12B books+web 放在同图；`:28` 与实际 `compare_75m_history.png` 标题明确是 **train NLL**。图上末尾约 3.84 与 4.26 的上移不能归因于改变 eval 集；训练输入域变化是候选解释，具体采样分布须由实际索引验证，而非仅凭图例判断。
- 我们新分享的 `experiments/pretraining_handoff_20260928/README.md:60` 和 `plot_curves.py:19`、`:71` 同样是 **training minibatch NLL**。若拿它与同事两栏图的右侧比较，就混用了 train/eval；目前未获知对方指的具体图，不能认定对方确实比错。
- `/srv/local/xy51/run_build_eval_cache.sh:26–30` 的调用为 `allenai/dolma`、`v1_7`、skip 1,000,000、50 条长度 1024、batch size 4，未传训练混合比例或独立 source 参数。此前已逐 tensor 确认六个新模型及我们旧 cache 的 `[50,1024]` input 完全相同，未发现“这次换了另一个 eval cache”的证据；`compare_runs.py:39` 将该 cache 标注为 web-text。
- 早期 books-only modular 的 final eval NLL 为 **5.5817**（`/srv/local/xy51/logs/pretrain_75m_modular.log:60`）；新 12B 组 dense / corrected / modular 为 **4.3389 / 4.2031 / 4.1935**（同目录 `pretrain_75m_baseline_dolma12b.log:40`、`pretrain_75m_fourway_dolma12b.log:76`、`pretrain_75m_modular_dolma12b.log:60`）。不能混淆两次 modular run。
- 新 150M 三个 final eval NLL 是 **3.7752 / 3.6707 / 3.6863**（`pretrain_150m_dense.log:48`、`pretrain_150m_fourway.log:90`、`pretrain_150m_modular.log:74`）；`compare_150m_12b_progress.png` 尚只画到约 3000 steps，实际训练完成 6000 steps，旧 progress PNG 不是完成结果。
- 普通 lm-eval 是另一套指标：`/srv/local/xy51/lmeval/run_lmeval_75m.sh:13–17` 与 `run_lmeval_150m.sh:12–20` 调用 `--tasks default --batch_size 8`，对应 final 3000 / 6000 checkpoints。这些任务结果不应与上述 Dolma 缓存 NLL 直接比较。
- `/srv/local/xy51/scaling/dagformer` 为 0700，目前不可读；外部只有 `scaling/data/{dolma_v1_7_21b,wikitext2,mathinstruct,gsm8k}/eval_cache.pt` 与空 `scaling/experiments/scaling/`。目录名只能说明存在四种缓存，无法证明已运行了某个 21B setting 或新 OOD eval。
