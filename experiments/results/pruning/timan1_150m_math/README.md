# 150M gradual pruning on MathInstruct: dense vs fourway_corrected vs fourway_modular (timan1, 2026-09-27)

The 150M counterpart of `../timan1_75m_math/`, run by the Delta session on timan1 GPUs 1-3
(three `prune_worker.sh` workers, `runlist.txt`, 4.7 h wall). The three 150M models are the
timan1 12B-corpus triple (`experiments/results/lmeval/timan1_dolma12b/`, 6000 steps x 528K
tokens = 3.17B; pretraining eval NLL fourway_corrected 3.671 / modular 3.686 / dense 3.775).
Protocol identical to the 75M sweep: 2000 finetuning steps x 32K tokens (micro 4 x accum 8),
lr 1e-4, cubic sparsity schedule from step 200 to 1400 with a prune event every 100 steps, 600
recovery steps, Taylor importance unless `_rc` (routing column mass, modular only), eval on
held-out MathInstruct ("domain") and wikitext-2 ("general"). One seed; differences below
~0.02 nats are noise. Configs: `configs/prune/150m_{dense,fourway,modular}_math_timan1.yaml`.

## Heads + neurons, Taylor importance (domain NLL / general NLL after recovery)

| block sparsity | dense | fourway_corrected | modular |
|---|---|---|---|
| 0% (finetune only) | 1.854 / 4.534 | **1.724** / **4.296** | 1.755 / 4.400 |
| 30% | 1.930 / 4.757 | **1.788** / **4.492** | 1.823 / 4.584 |
| 50% | 2.038 / 5.027 | **1.882** / **4.776** | 1.923 / 4.833 |
| 70% | 2.248 / 5.457 | **2.073** / **5.140** | 2.129 / 5.289 |

Gap to dense (domain): fourway 0.130 / 0.142 / 0.156 / 0.175 and modular 0.099 / 0.107 / 0.115 /
0.118 at 0 / 30 / 50 / 70%. Degradation relative to each model's own unpruned finetune:
70%: +0.394 (dense) / +0.349 (fourway) / +0.375 (modular).

## Whole attention / MLP blocks (8 layers -> 16 blocks; 33% = 6 blocks, 50% = 8)

| blocks removed | dense | fourway_corrected | modular (Taylor) | modular (routing_column) |
|---|---|---|---|---|
| 6 of 16 (33%) | 2.188 / 5.338 | 2.044 / 4.932 | 2.069 / 5.022 | **2.015** / **4.873** |
| 8 of 16 (50%) | 2.298 / 5.494 | 2.229 / 5.402 | 2.256 / 5.448 | **2.130** / **5.166** |

Blocks removed, in event order (layer index 0..7):

| run | events | final |
|---|---|---|
| dense 33% | L6.attn+L4.mlp, L7.attn+L2.mlp, L5.attn+L3.mlp | 2.188 |
| fourway 33% | L5.attn+L3.mlp, L7.attn+L5.mlp, L3.attn+L7.mlp | 2.044 |
| modular 33% | L6.attn+L5.mlp, L7.attn+L6.mlp, L4.attn+L2.mlp | 2.069 |
| modular 33% rc | L7.attn+L7.mlp, L0.attn+L6.mlp, L6.attn+L4.mlp | **2.015** |
| dense 50% | ... + L3.attn+L5.mlp | 2.298 |
| fourway 50% | ... + L6.attn+L1.mlp | 2.229 |
| modular 50% | ... + L1.attn+L3.mlp | 2.256 |
| modular 50% rc | ... + L1.attn+L2.mlp | **2.130** |

## Reading, together with the 75M sweep

1. **Fine-grained pruning: both routed models beat dense at every sparsity and the gap grows
   with sparsity** (fourway 0.13 -> 0.18, modular 0.10 -> 0.12 nats), as on Delta's
   shared-model pairs. The 75M result that modular degrades least at 70% does **not**
   replicate at 150M: fourway_corrected is ahead of modular by 0.03 to 0.06 nats at every
   sparsity, which is the same ordering (and about the same margin) as their unpruned
   finetunes. At these sizes the modular routing is not more prunable per se.
2. **Whole-block pruning is where the modular routing pays off, and only through its own
   importance score.** Routing-column mass gives the best result at both budgets (2.015 and
   2.130 vs 2.044 / 2.229 for fourway_corrected with Taylor), and beats Taylor on the same
   modular model by 0.05 (33%) and 0.13 (50%) nats. It removes the last layer's two blocks
   first, then the embedding-side attention (L0.attn), a pattern Taylor never picks; the
   immediate damage after those events is larger (max post-prune NLL 4.24 vs 3.69) but the
   recovery is far better. At 75M the same score won at 4 blocks and lost at 6, so at 150M
   the picture is cleaner.
3. General-domain (wikitext) NLL moves with the domain NLL in every cell; no family trades
   general for domain ability differently.

Regenerate: `python scripts/plot_prune_pareto.py --ckpt-root /srv/local/xy51/prune/checkpoints
--families dense,fourway,modular --out experiments/results/pruning/timan1_150m_math` on timan1
(uses that checkout's family-aware plot script); `runs/<name>/{summary,trajectory}.json` are
copies of the run outputs. Checkpoints stay on timan1 under `/srv/local/xy51/prune/checkpoints`.
