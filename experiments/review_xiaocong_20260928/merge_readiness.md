# PR #3 merge readiness — 88d472b against main a6703d6

2026-09-28。只做评审和隔离合并预览；未合并 main，未改同事分支，未 push 或发送 GitHub 评论。

建议先做一轮小修再整体合并。已有质量评测并未因这些问题全部失效，但不能把当前版本直接作为经过验证的稀疏路由/参数压缩实现合入。最新 `88d472b` 相比已评审的 `ec72d8e` 只修改 `scripts/slurm/reserved_train.slurm`：后继作业提交失败后尝试 `biro-delta-gpu`；此前三个问题的相关代码完全未变。本次没有重复运行三个已证明的问题复现。

## 合并接口：需要解决，已做可审阅预览

PR当前目标分支为 `interp/routing-circuit-power`，不是 `main`。GitHub API显示
`mergeable: true`针对该目标分支，不能据此判断与当前main兼容；真正合入main时
需要以main为基线处理以下集成差异。

`git merge-tree --write-tree origin/main origin/pr-3` 返回冲突，仅一个冲突文件 `scripts/eval_lm_harness.py`，含 docstring 和 predictor 构建两处冲突。生成的临时 tree 为 `20a46628c025c3fbf45add5172df0661af609bec`。

另有一个自动合并不会标成冲突的运行错误：main 的严格加载代码被移入 PR 新建的 `_load_fourway_state` 后，日志仍引用 `cfg`，该函数却没有接收它。仅删除冲突标记会导致 FourWay 加载时 `NameError`。

已在独立 detached 工作区 `/tmp/dagformer-pr3-integration-88d472b` 解决这些接口问题：

- 保留 main 的 `encoder` / `static` / `pos_table` predictor 工厂，加入 PR 的 `per_layer`。
- 保留 PR 的 modular / modular_corrected 构建分支，并显式向公共加载函数传入 `cfg`。
- 保留 main 对 predictor、backbone、correction/v_norm 权重的严格完整性检查，避免直接采用 PR 的宽松 `strict=False` 路径而静默丢失权重。
- 保留 PR 的 `CausalLMOutputWithPast` 返回值改进。

`integration_preview.patch` 展示以上最终 loader 相对 main 的改动，附新增 per_layer checkpoint roundtrip 覆盖。它是合并解决方案的预览，不是整个 PR 的补丁，也没有修复下列三个科学/工具问题；应结合 PR 的其余新增模块使用。

## 尚未修复的问题及影响范围

1. **R stream 不参与输出，但计入重要性和稀疏指标。** `src/model/modular_routing.py:195` 读取 R 系数，`:216` 计算 `R`，后续 `:224–240` 的 MLP、running state 和最终读出均不使用它；`source_column_mass` 在 `:267` 仍累计 R，`_iter_edges` 在 `:309` 仍遍历 R。因此 R 的值/稀疏程度可改变 reported column importance、稀疏正则和统计，却不代表相应的功能路径。既有 checkpoint 的实测 logits/NLL 是该实际前向实现的结果，可以保留；“每条六路边都参与计算”和 column score 的因果解释不成立。最小兼容修法是保留 checkpoint 的旧 R 参数布局，但将无作用的 R 排除出功能性 column score、惩罚和统计，并准确标注现有实验使用旧分数。不要直接把 R 接入前向，否则会改变已训练 checkpoint 的函数。新分数的剪枝优劣需要重跑相关 routing-column arms，普通 Taylor/语言模型结果不需因此重跑。

   对 `fourway_modular_corrected` 还应限制“零 predictor 列就可删除模块”的表述：correction 本身会新增读入系数，且 `running` 在 `:234` 无条件包含模块输出，影响后续 correction。当前 zero-column 单测使用未启用 correction 的模型，不能验证 corrected 版本的这一性质。

2. **partial-head 的“可删除参数”计数不成立。** `src/pruning/masks.py:110` / `:186` 将每个 masked head 计为 `4*D*head_dim + 2*head_dim` 个参数；实际 mask 在 `:281–291` 仅将 O-projection 的 head 输入置零，导出时 `scripts/prune_finetune.py:412` 起也只烘焙 O 列等零值。OLMo 的 Q/K norm 跨整个拼接 head 维，被 mask 的 head 的 Q/K 投影仍影响其他 head 的归一化分母，不能按上述数量无损删除。未启用 V norm 时可证明的删除项包括对应 V/O；启用全维 V norm 的配置连 V 也会耦合其他 head，应更保守地计数。这里影响 remaining-parameter 横轴、压缩率和部署收益，**不抹掉当前 output-mask 实验的 NLL**。整体 attention/MLP block 删除与 neuron 删除应独立处理。当前模块未感知 wrapper 的 V norm，修计数时必须把实际结构传入或明确限定计数含义。

3. **提交的画图脚本跳过 timan1 三族结果。** `scripts/plot_prune_pareto.py:33–37` 的 family 常量和正则仅接收 `baseline/dagformer/muddformer/baselinest/dagformerst`，未接收新结果目录的 `dense/fourway/modular`；`:44–46` 会在 `--families` 过滤前直接跳过。README 的三族重新生成命令不能用已提交脚本重现。需同步 family 识别、配色、marker 和配对表逻辑。这是可复现性问题，不是原始 summary/trajectory 数值失效。

## 已完成的合并预览验证

CPU、无模型下载，`OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 CUDA_VISIBLE_DEVICES=''`：

```text
/scratch/yurenh2/venvs/dagformer-eval-20260917/bin/python -m pytest -q \
  tests/test_fourway_eval_loading.py tests/test_modular_routing.py \
  tests/test_pruning.py tests/test_pruning_muddformer.py \
  tests/test_prune_finetune_smoke.py tests/test_frozen_continuation.py \
  tests/test_ordered_pretokenize.py
44 passed, 11 warnings in 54.45s

# 对补入工厂的新分支额外覆盖实际 checkpoint 权重与 logits roundtrip：
.../bin/python -m pytest -q \
  'tests/test_fourway_eval_loading.py::test_checkpoint_roundtrip_preserves_variant_routing_and_logits[per_layer]'
1 passed, 1 warning in 8.88s
```

另通过 `bash -n scripts/slurm/reserved_train.slurm` 和临时合并预览的 `git diff --check`。这些测试覆盖旧 predictor variants、modular 权重加载、剪枝训练导出/加载、MUDDFormer hook、冻结续训数据和有序分词；没有执行真实多卡训练、Slurm 提交或长训练恢复，故不把通过 CPU 测试当作这些运行路径已验证。

当前 modular 测试会把 R 计作一个 reader，partial-head accounting 测试也沿用同一个公式，所以“现有测试通过”不能反证上述两个问题。修复时应加入相应的功能依赖/归一化保持性验证，并更新旧断言。

最小合并路径：采用预览中的 loader 集成，修正三个问题并同步受影响的描述/参数图；将旧 routing-column 结果标明原实现分数，后续用修正分数重跑两档关键对比。无需为了合并先重做所有预训练和普通 eval。若现在只想收代码与原始结果，也应至少隔离或明确标记未修正的 column-importance/参数压缩输出，避免将它们当成已验证指标继续使用。
