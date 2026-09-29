# 两轮构造性拼接的实验顺序与复现

两轮都使用冻结 DAG 1B step38160，探索把 first-hop 操作算出的键移植给已显式提供旧键的 recipient，使它继续使用自己的第二张表。它不是解释已存在的隐式两步能力；[能力筛查](../composition/README.md)已表明该能力在测试模板上不足。

1. 第一轮选择目标是第三颜色与原颜色的 logit margin。在 discovery32 上选层、head，heldout64 与全部控制上固定集合。结果揭示目标函数漏洞：压低原色并直接输出 donor 键，也能提高这个 margin。保留全部原始结果在本目录。
2. 依据第一轮的失败模式，将目标改成完整词表归一化的第三颜色 log-probability。重新抽取 discovery32 和 heldout64；没有拿第一轮 heldout 选择第二轮的 head。其余搜索预算和控制相同。结果在 `logp_objective/`。
3. 第二轮有第三颜色边际改善，随后对固定的全部 head 集合补采三种 recipient 颜色的 logp，进行 donor-key 双差、recipient-table 跟随和颜色概率分解。没有重新搜索，所有原有逐例指标重放完全一致。结果在 `logp_objective/semantic_diagnostics/`。

第二轮的改善没有通过 donor-key 特异性检验。它不能区分 donor 算出了 C 还是 D；目标色和无关备选色都会被抬高。第三答案机制目前没有成功建立。

```bash
CUDA_VISIBLE_DEVICES=2 /scratch/yurenh2/venvs/dagformer-eval-20260917/bin/python scripts/circuit_composition_constructive.py
CUDA_VISIBLE_DEVICES=2 /scratch/yurenh2/venvs/dagformer-eval-20260917/bin/python scripts/circuit_composition_constructive.py --objective third_logp --discovery-seed 2026092920 --eval-seed 2026092921 --out experiments/results/circuit_followup_20260929/composition_constructive/logp_objective
CUDA_VISIBLE_DEVICES=2 /scratch/yurenh2/venvs/dagformer-eval-20260917/bin/python scripts/circuit_composition_constructive.py --objective third_logp --discovery-seed 2026092920 --eval-seed 2026092921 --out experiments/results/circuit_followup_20260929/composition_constructive/logp_objective --diagnostics-only
/scratch/yurenh2/venvs/dagformer-eval-20260917/bin/python experiments/results/circuit_followup_20260929/composition_constructive/summarize.py
/scratch/yurenh2/venvs/dagformer-eval-20260917/bin/python experiments/results/circuit_followup_20260929/composition_constructive/logp_objective/semantic_diagnostics/summarize.py
```

最后两条是 CPU 汇总：补全真实所需的组件成功子集、三条件子集、配对控制同时成功率，以及键/颜色诊断。输入 token、目标映射、全部 head 预算和随机对照均已保存。
