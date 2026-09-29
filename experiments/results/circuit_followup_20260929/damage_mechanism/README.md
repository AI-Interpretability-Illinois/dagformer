# Damage, conditional backups, and routing response

Frozen 300M checkpoints; no finetuning. Head lesions zero attention-head outputs before o_proj at all positions. Native DAG lesions remove selected V source-read contributions only at the three known historical value-token locations.

A is the first non-layer0 V head from yesterday's discovery-only common-message ranking. Candidate backup heads are all 16 heads in the next layer. DAG source-read candidates are the next-layer heads reading sources 0/1/2 (embedding / layer0 output / layer1 output), which precede A. Selection maximizes conditional NLL interaction on discovery only, under clean effective-route clamps in DAG.

Clean clamps preserve the complete per-input effective alpha (global predictor plus local correction), at every layer/stream/position. They preserve input dependence and prevent effective routing from responding to the damage. The external global predictor reads only input IDs, so its coefficients cannot change from an internal lesion with the same input and frozen weights. Attention probabilities can still change.

Interaction = NLL(A+B) - NLL(A) - NLL(B) + NLL(clean). Positive interaction is additional joint damage; it is not sufficient by itself to prove a unique backup computation. Recent-source controls use the same destination head and three most recent sources; random controls preserve destination layer and number of heads/source reads. Controls are not matched for intervention norm.

The DAG native source-read experiments are not compared numerically with dense head counts. Head-pair interventions use the same head-output operator in both models, but model parameters and selected head locations differ. All positions are supplied by task construction. This is synthetic copying, not general semantic binding or pretrained pruning recovery.

Evaluation provenance: Fresh evaluation seed 2026092901; discovery and all selection rules unchanged from the prior inputs. Reused-input exploratory results are separate.

## 中文关键结论

**已有早层读取与即时 local correction 响应同时贡献损伤耐受。此次没有训练，因此不能据此解释 finetune 后的恢复。**

1. 删除 L6/H11 后，正常 correction 比钳住该输入原始有效路由的 NLL 更低。固定减正常的差：heldout +0.199 [+0.156, +0.247]，距离迁移 +0.268 [+0.198, +0.340]。global predictor 的输入和参数均未变，不能称其感知损伤后改道。

2. L7/H1 从 embedding、layer0、layer1 读取的三条 V 来源在损伤前后投影内容完全一致，实测 max abs diff=0。在有效路由固定时，再切断这些已有读取的条件交互为 heldout +0.419 [+0.349, +0.493]，迁移 +0.519 [+0.402, +0.649]。这支持未受损早层信息经已有跨层读取继续影响损伤后的输出；不能称产生了新连接。

3. 改问另一未变位置后，early-read 条件交互为 -0.000 [-0.002, +0.001]，接近零；完整 head 损伤则仍影响这个复制目标。定位到历史值读取的干预比笼统删 head 更有行为选择性。

4. 同 head 的 recent-source 对照作用很小，但编辑幅度不匹配：early-source 删除 L2 约为 recent 的 27 倍，不能据此声称早来源在等强扰动下具有特殊优势。多个随机目的 head 的早层读取也有正交互，因此不是唯一备用通路。原始 L2 和 paired contrasts 均已保存。

5. Dense 同样存在正的 head 双损伤交互。两模型的初始能力、主要损伤位置、参数量，以及 DAG 的 clamp 条件不同，此处不把交互大小或原始 NLL 解释成 DAG 总体更鲁棒。搜索只覆盖 A 的下一层 16 个 head。

6. 只替换 L7/H1 三条 early-read 的 donor 内容会降低正确答案 margin，但 donor 成为全词表首选的比例仅约 0.8%；这不是完整答案移植。

## 主图建议数值

NLL / 正确答案 margin / 全词表正确率。所有数值来自 fresh confirmation；误差条及逐样本数值见 main_figure_data.csv 与 results.json。headAB 是两个完整 head 损伤；earlyAB 是 A 加三条历史 V 来源读取切断，二者不是等规模干预。

| DAG arm | heldout NLL / margin / acc | transfer NLL / margin / acc |
|---|---:|---:|
| clean | 0.1756 / 14.562 / 98.44% | 0.5044 / 13.984 / 93.75% |
| A_dynamic | 0.6174 / 12.177 / 93.75% | 1.2528 / 11.062 / 82.81% |
| A_fixed | 0.8166 / 11.412 / 89.06% | 1.5210 / 10.367 / 77.34% |
| head_AB_dynamic | 0.8999 / 11.267 / 90.62% | 1.6556 / 10.046 / 80.47% |
| head_AB_fixed | 1.2387 / 10.495 / 83.59% | 2.0818 / 9.315 / 65.62% |
| early_AB_dynamic | 0.9696 / 11.195 / 89.06% | 1.7411 / 9.935 / 78.12% |
| early_AB_fixed | 1.3477 / 10.359 / 79.69% | 2.1817 / 9.187 / 64.06% |

| Read intervention | heldout L2 | transfer L2 |
|---|---:|---:|
| early_B_fixed | 47.773 | 51.088 |
| early_AB_fixed | 47.773 | 51.088 |
| recent_B | 1.785 | 1.841 |
| recent_AB | 1.772 | 1.812 |
| early_donor_AB | 66.308 | 71.608 |

## dag

Selection: {'A': [6, 11], 'backup_head': [7, 1], 'head_conditional_NLL_scores': [-0.010897813755946117, 0.29431894691060734, 0.00015151104435062734, 0.022935276616408373, -0.002240026189610944, 0.017137495705355832, 0.0021522498318518046, 0.08004012449782749, -9.873840099317022e-06, 0.21069844327666942, -0.0005645678102155216, 0.19763166619486583, -0.008009730835510709, 0.0013626994941660087, 0.03642128663886979, -0.021835827942595643], 'discovery_pairs': 32, 'selection_under_fixed_effective_alpha': True, 'control_heads': [3, 5, 8], 'early_read_head': [7, 1], 'early_sources': [0, 1, 2], 'early_conditional_NLL_scores': [-0.0006160844677651767, 0.3613989252326064, 0.001043873576236365, 0.025704839959871606, -0.0015607662817274104, 0.001363274661343894, 0.0009435658976144623, 0.12669930332776858, 0.004801961855264381, 0.22676109811527567, 0.0009407345787622035, 0.2514602980818381, 0.0018150211444663, -0.00024237730303866556, 0.046791598610980145, 5.78266099182656e-06], 'early_control_heads': [11, 7, 2]}

### heldout: 64 independent pairs

| Arm | Correct NLL | Correct full-vocabulary accuracy | Donor full-vocabulary accuracy |
|---|---:|---:|---:|
| A_dynamic | 0.6174 | 93.75% | 0.00% |
| head_B_dynamic | 0.2527 | 98.44% | 0.00% |
| head_AB_dynamic | 0.8999 | 90.62% | 0.00% |
| head_random0_B | 0.1827 | 98.44% | 0.00% |
| head_random0_AB | 0.6449 | 93.75% | 0.00% |
| head_random1_B | 0.1866 | 98.44% | 0.00% |
| head_random1_AB | 0.6604 | 94.53% | 0.00% |
| head_random2_B | 0.1748 | 98.44% | 0.00% |
| head_random2_AB | 0.6148 | 93.75% | 0.00% |
| clean_fixed | 0.1756 | 98.44% | 0.00% |
| A_fixed | 0.8166 | 89.06% | 0.00% |
| head_B_fixed | 0.2587 | 98.44% | 0.00% |
| head_AB_fixed | 1.2387 | 83.59% | 0.00% |
| early_B_dynamic | 0.2793 | 98.44% | 0.00% |
| early_AB_dynamic | 0.9696 | 89.06% | 0.00% |
| early_B_fixed | 0.2875 | 98.44% | 0.00% |
| early_AB_fixed | 1.3477 | 79.69% | 0.00% |
| recent_B | 0.1752 | 98.44% | 0.00% |
| recent_AB | 0.8167 | 89.06% | 0.00% |
| early_random0_B | 0.2427 | 98.44% | 0.00% |
| early_random0_AB | 1.1890 | 85.16% | 0.00% |
| early_random1_B | 0.2043 | 98.44% | 0.00% |
| early_random1_AB | 1.0065 | 86.72% | 0.00% |
| early_random2_B | 0.1760 | 98.44% | 0.00% |
| early_random2_AB | 0.8215 | 89.06% | 0.00% |
| early_donor_B | 0.3950 | 97.66% | 0.78% |
| early_donor_AB | 1.8125 | 66.41% | 0.78% |
| clean | 0.1756 | 98.44% | 0.00% |

| Contrast | NLL effect [paired 95% CI] |
|---|---:|
| head_interaction_dynamic | +0.2054 [+0.1520, +0.2606] |
| head_random0_interaction | +0.0204 [+0.0139, +0.0272] |
| head_random1_interaction | +0.0319 [+0.0225, +0.0435] |
| head_random2_interaction | -0.0018 [-0.0065, +0.0022] |
| head_interaction_fixed | +0.3390 [+0.2778, +0.4015] |
| early_interaction_dynamic | +0.2485 [+0.1896, +0.3109] |
| early_interaction_fixed | +0.4192 [+0.3489, +0.4933] |
| recent_interaction_fixed | +0.0005 [-0.0015, +0.0026] |
| early_random0_interaction_fixed | +0.3053 [+0.2492, +0.3644] |
| early_random1_interaction_fixed | +0.1612 [+0.1154, +0.2129] |
| early_random2_interaction_fixed | +0.0044 [+0.0022, +0.0067] |
| A_fixed_minus_dynamic_NLL | +0.1992 [+0.1561, +0.2466] |
| head_B_fixed_minus_dynamic_NLL | +0.0060 [+0.0006, +0.0120] |
| head_AB_fixed_minus_dynamic_NLL | +0.3389 [+0.2729, +0.4083] |
| early_B_fixed_minus_dynamic_NLL | +0.0082 [+0.0030, +0.0137] |
| early_AB_fixed_minus_dynamic_NLL | +0.3782 [+0.3026, +0.4543] |
| head_interaction_minus_random0 | +0.1849 [+0.1348, +0.2381] |
| head_interaction_minus_random1 | +0.1734 [+0.1250, +0.2236] |
| head_interaction_minus_random2 | +0.2071 [+0.1532, +0.2631] |
| early_interaction_minus_recent | +0.4187 [+0.3484, +0.4925] |
| early_interaction_minus_early_random0 | +0.1139 [+0.0428, +0.1868] |
| early_interaction_minus_early_random1 | +0.2580 [+0.2054, +0.3132] |
| early_interaction_minus_early_random2 | +0.4148 [+0.3443, +0.4893] |

Effective-alpha relative change after A: {'mean': 0.10043437569402158, 'paired_95ci': [0.0984184915869264, 0.10259598157572328]}

### transfer: 64 independent pairs

| Arm | Correct NLL | Correct full-vocabulary accuracy | Donor full-vocabulary accuracy |
|---|---:|---:|---:|
| A_dynamic | 1.2528 | 82.81% | 0.00% |
| head_B_dynamic | 0.6103 | 92.97% | 0.00% |
| head_AB_dynamic | 1.6556 | 80.47% | 0.00% |
| head_random0_B | 0.5111 | 93.75% | 0.00% |
| head_random0_AB | 1.2848 | 82.81% | 0.00% |
| head_random1_B | 0.5103 | 93.75% | 0.00% |
| head_random1_AB | 1.2932 | 82.81% | 0.00% |
| head_random2_B | 0.5064 | 93.75% | 0.00% |
| head_random2_AB | 1.2550 | 82.81% | 0.00% |
| clean_fixed | 0.5044 | 93.75% | 0.00% |
| A_fixed | 1.5210 | 77.34% | 0.00% |
| head_B_fixed | 0.6145 | 92.97% | 0.00% |
| head_AB_fixed | 2.0818 | 65.62% | 0.00% |
| early_B_dynamic | 0.6408 | 92.97% | 0.00% |
| early_AB_dynamic | 1.7411 | 78.12% | 0.00% |
| early_B_fixed | 0.6461 | 92.97% | 0.00% |
| early_AB_fixed | 2.1817 | 64.06% | 0.00% |
| recent_B | 0.5040 | 93.75% | 0.00% |
| recent_AB | 1.5208 | 77.34% | 0.00% |
| early_random0_B | 0.5845 | 92.97% | 0.00% |
| early_random0_AB | 1.9822 | 66.41% | 0.00% |
| early_random1_B | 0.5426 | 92.97% | 0.00% |
| early_random1_AB | 1.7460 | 73.44% | 0.00% |
| early_random2_B | 0.5051 | 93.75% | 0.00% |
| early_random2_AB | 1.5252 | 77.34% | 0.00% |
| early_donor_B | 0.8024 | 92.19% | 0.78% |
| early_donor_AB | 2.8323 | 50.00% | 0.78% |
| clean | 0.5044 | 93.75% | 0.00% |

| Contrast | NLL effect [paired 95% CI] |
|---|---:|
| head_interaction_dynamic | +0.2969 [+0.2081, +0.3926] |
| head_random0_interaction | +0.0253 [+0.0147, +0.0396] |
| head_random1_interaction | +0.0345 [+0.0255, +0.0441] |
| head_random2_interaction | +0.0002 [-0.0038, +0.0041] |
| head_interaction_fixed | +0.4506 [+0.3340, +0.5781] |
| early_interaction_dynamic | +0.3518 [+0.2614, +0.4494] |
| early_interaction_fixed | +0.5190 [+0.4023, +0.6494] |
| recent_interaction_fixed | +0.0002 [-0.0017, +0.0022] |
| early_random0_interaction_fixed | +0.3811 [+0.3071, +0.4651] |
| early_random1_interaction_fixed | +0.1868 [+0.1353, +0.2457] |
| early_random2_interaction_fixed | +0.0035 [+0.0015, +0.0058] |
| A_fixed_minus_dynamic_NLL | +0.2682 [+0.1985, +0.3395] |
| head_B_fixed_minus_dynamic_NLL | +0.0043 [-0.0038, +0.0125] |
| head_AB_fixed_minus_dynamic_NLL | +0.4262 [+0.3136, +0.5421] |
| early_B_fixed_minus_dynamic_NLL | +0.0052 [-0.0017, +0.0121] |
| early_AB_fixed_minus_dynamic_NLL | +0.4406 [+0.3269, +0.5625] |
| head_interaction_minus_random0 | +0.2716 [+0.1811, +0.3661] |
| head_interaction_minus_random1 | +0.2624 [+0.1758, +0.3550] |
| head_interaction_minus_random2 | +0.2967 [+0.2093, +0.3919] |
| early_interaction_minus_recent | +0.5188 [+0.4026, +0.6484] |
| early_interaction_minus_early_random0 | +0.1379 [+0.0317, +0.2408] |
| early_interaction_minus_early_random1 | +0.3322 [+0.2353, +0.4410] |
| early_interaction_minus_early_random2 | +0.5155 [+0.3995, +0.6457] |

Effective-alpha relative change after A: {'mean': 0.11035785963758826, 'paired_95ci': [0.10718069617869333, 0.11332614962448133]}

### unchanged_query: 64 independent pairs

| Arm | Correct NLL | Correct full-vocabulary accuracy | Donor full-vocabulary accuracy |
|---|---:|---:|---:|
| A_dynamic | 0.3146 | 98.44% | 0.00% |
| head_B_dynamic | 0.1113 | 100.00% | 0.00% |
| head_AB_dynamic | 0.5113 | 97.66% | 0.00% |
| head_random0_B | 0.0697 | 100.00% | 0.00% |
| head_random0_AB | 0.3367 | 98.44% | 0.00% |
| head_random1_B | 0.0733 | 100.00% | 0.00% |
| head_random1_AB | 0.3396 | 98.44% | 0.00% |
| head_random2_B | 0.0613 | 100.00% | 0.00% |
| head_random2_AB | 0.3116 | 97.66% | 0.00% |
| clean_fixed | 0.0651 | 100.00% | 0.00% |
| A_fixed | 0.4374 | 98.44% | 0.00% |
| head_B_fixed | 0.1122 | 100.00% | 0.00% |
| head_AB_fixed | 0.7284 | 94.53% | 0.00% |
| early_B_dynamic | 0.0643 | 100.00% | 0.00% |
| early_AB_dynamic | 0.3149 | 98.44% | 0.00% |
| early_B_fixed | 0.0643 | 100.00% | 0.00% |
| early_AB_fixed | 0.4362 | 98.44% | 0.00% |
| recent_B | 0.0648 | 100.00% | 0.00% |
| recent_AB | 0.4373 | 98.44% | 0.00% |
| early_random0_B | 0.0651 | 100.00% | 0.00% |
| early_random0_AB | 0.4378 | 98.44% | 0.00% |
| early_random1_B | 0.0652 | 100.00% | 0.00% |
| early_random1_AB | 0.4375 | 98.44% | 0.00% |
| early_random2_B | 0.0652 | 100.00% | 0.00% |
| early_random2_AB | 0.4376 | 98.44% | 0.00% |
| early_donor_B | 0.0643 | 100.00% | 0.00% |
| early_donor_AB | 0.4354 | 98.44% | 0.00% |
| clean | 0.0651 | 100.00% | 0.00% |

| Contrast | NLL effect [paired 95% CI] |
|---|---:|
| head_interaction_dynamic | +0.1505 [+0.0826, +0.2278] |
| head_random0_interaction | +0.0174 [+0.0089, +0.0274] |
| head_random1_interaction | +0.0168 [+0.0111, +0.0230] |
| head_random2_interaction | +0.0008 [-0.0018, +0.0036] |
| head_interaction_fixed | +0.2438 [+0.1580, +0.3378] |
| early_interaction_dynamic | +0.0010 [-0.0002, +0.0023] |
| early_interaction_fixed | -0.0005 [-0.0019, +0.0011] |
| recent_interaction_fixed | +0.0001 [-0.0013, +0.0017] |
| early_random0_interaction_fixed | +0.0003 [-0.0009, +0.0017] |
| early_random1_interaction_fixed | -0.0001 [-0.0011, +0.0010] |
| early_random2_interaction_fixed | +0.0000 [-0.0014, +0.0017] |
| A_fixed_minus_dynamic_NLL | +0.1228 [+0.0778, +0.1765] |
| head_B_fixed_minus_dynamic_NLL | +0.0010 [-0.0064, +0.0082] |
| head_AB_fixed_minus_dynamic_NLL | +0.2171 [+0.1446, +0.2972] |
| early_B_fixed_minus_dynamic_NLL | -0.0001 [-0.0003, +0.0002] |
| early_AB_fixed_minus_dynamic_NLL | +0.1213 [+0.0765, +0.1744] |
| head_interaction_minus_random0 | +0.1331 [+0.0693, +0.2055] |
| head_interaction_minus_random1 | +0.1337 [+0.0669, +0.2083] |
| head_interaction_minus_random2 | +0.1498 [+0.0810, +0.2278] |
| early_interaction_minus_recent | -0.0006 [-0.0021, +0.0008] |
| early_interaction_minus_early_random0 | -0.0008 [-0.0022, +0.0005] |
| early_interaction_minus_early_random1 | -0.0004 [-0.0017, +0.0009] |
| early_interaction_minus_early_random2 | -0.0005 [-0.0015, +0.0005] |

Effective-alpha relative change after A: {'mean': 0.09863793686963618, 'paired_95ci': [0.0966955775176757, 0.1007415648404276]}

## dense

Selection: {'A': [4, 1], 'backup_head': [5, 1], 'head_conditional_NLL_scores': [-0.04176338098113774, 0.1304580178893957, -0.015362660205937573, -0.005021477578338818, 0.01404349163385632, 0.0665632545005792, 0.036113975942498655, 0.08483847960269486, 0.002442676779537578, -0.06823579414958658, -0.0440159451263753, 0.0027919235344597837, -0.00017293930250161793, -0.0060989483190496685, 0.012263760465430096, -0.011696359024426783], 'discovery_pairs': 32, 'selection_under_fixed_effective_alpha': False, 'control_heads': [3, 5, 8]}

### heldout: 64 independent pairs

| Arm | Correct NLL | Correct full-vocabulary accuracy | Donor full-vocabulary accuracy |
|---|---:|---:|---:|
| A_dynamic | 0.8177 | 92.97% | 0.00% |
| head_B_dynamic | 0.3929 | 97.66% | 0.00% |
| head_AB_dynamic | 1.0448 | 89.84% | 0.00% |
| head_random0_B | 0.3555 | 98.44% | 0.00% |
| head_random0_AB | 0.8090 | 92.97% | 0.00% |
| head_random1_B | 0.4700 | 97.66% | 0.00% |
| head_random1_AB | 1.0579 | 92.19% | 0.00% |
| head_random2_B | 0.3591 | 97.66% | 0.00% |
| head_random2_AB | 0.8317 | 92.19% | 0.00% |
| clean | 0.3515 | 98.44% | 0.00% |

| Contrast | NLL effect [paired 95% CI] |
|---|---:|
| head_interaction_dynamic | +0.1857 [+0.1449, +0.2281] |
| head_random0_interaction | -0.0127 [-0.0217, -0.0039] |
| head_random1_interaction | +0.1217 [+0.0914, +0.1532] |
| head_random2_interaction | +0.0063 [-0.0035, +0.0170] |
| head_interaction_minus_random0 | +0.1984 [+0.1538, +0.2450] |
| head_interaction_minus_random1 | +0.0640 [+0.0177, +0.1136] |
| head_interaction_minus_random2 | +0.1794 [+0.1354, +0.2251] |

Effective-alpha relative change after A: None

### transfer: 64 independent pairs

| Arm | Correct NLL | Correct full-vocabulary accuracy | Donor full-vocabulary accuracy |
|---|---:|---:|---:|
| A_dynamic | 1.0096 | 85.16% | 0.00% |
| head_B_dynamic | 0.4358 | 96.88% | 0.00% |
| head_AB_dynamic | 1.2209 | 83.59% | 0.00% |
| head_random0_B | 0.4034 | 96.88% | 0.00% |
| head_random0_AB | 0.9994 | 87.50% | 0.00% |
| head_random1_B | 0.4816 | 96.09% | 0.00% |
| head_random1_AB | 1.2490 | 82.03% | 0.00% |
| head_random2_B | 0.4171 | 96.09% | 0.00% |
| head_random2_AB | 1.0422 | 85.94% | 0.00% |
| clean | 0.4017 | 96.88% | 0.00% |

| Contrast | NLL effect [paired 95% CI] |
|---|---:|
| head_interaction_dynamic | +0.1773 [+0.1307, +0.2322] |
| head_random0_interaction | -0.0118 [-0.0242, -0.0002] |
| head_random1_interaction | +0.1595 [+0.1229, +0.2000] |
| head_random2_interaction | +0.0173 [+0.0038, +0.0325] |
| head_interaction_minus_random0 | +0.1891 [+0.1372, +0.2512] |
| head_interaction_minus_random1 | +0.0178 [-0.0220, +0.0593] |
| head_interaction_minus_random2 | +0.1600 [+0.1143, +0.2144] |

Effective-alpha relative change after A: None

### unchanged_query: 64 independent pairs

| Arm | Correct NLL | Correct full-vocabulary accuracy | Donor full-vocabulary accuracy |
|---|---:|---:|---:|
| A_dynamic | 0.5417 | 96.09% | 0.00% |
| head_B_dynamic | 0.1643 | 98.44% | 0.00% |
| head_AB_dynamic | 0.7463 | 92.19% | 0.00% |
| head_random0_B | 0.1306 | 98.44% | 0.00% |
| head_random0_AB | 0.5493 | 95.31% | 0.00% |
| head_random1_B | 0.2087 | 98.44% | 0.00% |
| head_random1_AB | 0.6964 | 95.31% | 0.00% |
| head_random2_B | 0.1341 | 98.44% | 0.00% |
| head_random2_AB | 0.5494 | 95.31% | 0.00% |
| clean | 0.1325 | 98.44% | 0.00% |

| Contrast | NLL effect [paired 95% CI] |
|---|---:|
| head_interaction_dynamic | +0.1727 [+0.1239, +0.2373] |
| head_random0_interaction | +0.0094 [-0.0069, +0.0297] |
| head_random1_interaction | +0.0784 [+0.0501, +0.1085] |
| head_random2_interaction | +0.0061 [-0.0030, +0.0167] |
| head_interaction_minus_random0 | +0.1633 [+0.1179, +0.2181] |
| head_interaction_minus_random1 | +0.0943 [+0.0497, +0.1497] |
| head_interaction_minus_random2 | +0.1666 [+0.1188, +0.2279] |

Effective-alpha relative change after A: None

All directed-case values and discovery scans are in results.json. Pair bootstrap uses 4,000 resamples, grouping exchange directions; intervals are unadjusted and describe examples rather than checkpoint seeds. No weights were updated.
