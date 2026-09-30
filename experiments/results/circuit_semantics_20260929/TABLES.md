# 数值表

2026-09-30。由已完成结果自动生成；候选词比较准确率，不是开放生成准确率。区间按八例反事实 block bootstrap。

## 一维信息交换：原模型

| 测试集 | 原始语法 | 均值差方向：反事实吻合 | 学习方向：反事实吻合 | 学习方向：数信息效应 | 同数换词：绝对效应 |
|---|---:|---:|---:|---:|---:|
| heldout | 99.74% [99.22, 100.00] | 99.48% [98.44, 100.00] | 99.74% [99.22, 100.00] | 6.90 [6.61, 7.18] | 0.06 [0.05, 0.07] |
| transfer | 100.00% [100.00, 100.00] | 86.20% [83.07, 89.32] | 83.33% [79.94, 86.72] | 5.77 [5.33, 6.25] | 0.07 [0.05, 0.09] |
| role | 100.00% [100.00, 100.00] | 96.88% [95.05, 98.44] | 91.67% [88.54, 94.53] | 8.19 [7.87, 8.54] | 0.09 [0.07, 0.11] |

## 一维信息交换：所有状态与动词

| 测试集 | 状态 | 动词对 | 原始语法 | 学习方向：反事实语法吻合 | 有符号效应 |
|---|---|---|---:|---:|---:|
| heldout | original | is/are | 99.74% | 99.74% | 6.897 |
| heldout | original | has/have | 99.22% | 99.22% | 6.370 |
| heldout | original | does/do | 95.83% | 93.49% | 4.750 |
| heldout | original | was/were | 99.74% | 99.74% | 6.372 |
| heldout | original | seems/seem | 99.48% | 99.48% | 4.955 |
| heldout | original | works/work | 88.80% | 88.54% | 3.772 |
| heldout | original | likes/like | 50.78% | 50.52% | 2.936 |
| heldout | adapted | is/are | 3.12% | 28.91% | -2.542 |
| heldout | adapted | has/have | 4.17% | 37.24% | -2.442 |
| heldout | adapted | does/do | 5.21% | 32.29% | -2.233 |
| heldout | adapted | was/were | 4.95% | 30.99% | -1.859 |
| heldout | adapted | seems/seem | 9.11% | 37.76% | -1.965 |
| heldout | adapted | works/work | 19.53% | 38.54% | -1.693 |
| heldout | adapted | likes/like | 50.00% | 50.00% | -0.950 |
| heldout | repaired | is/are | 97.92% | 98.18% | 5.858 |
| heldout | repaired | has/have | 96.61% | 97.40% | 5.531 |
| heldout | repaired | does/do | 90.36% | 94.53% | 4.174 |
| heldout | repaired | was/were | 98.70% | 98.18% | 5.090 |
| heldout | repaired | seems/seem | 95.05% | 95.05% | 4.339 |
| heldout | repaired | works/work | 88.54% | 93.23% | 3.362 |
| heldout | repaired | likes/like | 56.51% | 56.77% | 2.580 |
| transfer | original | is/are | 100.00% | 83.33% | 5.768 |
| transfer | original | has/have | 100.00% | 80.47% | 5.293 |
| transfer | original | does/do | 95.57% | 67.45% | 4.097 |
| transfer | original | was/were | 100.00% | 83.59% | 5.406 |
| transfer | original | seems/seem | 99.22% | 80.21% | 4.318 |
| transfer | original | works/work | 84.90% | 64.06% | 3.100 |
| transfer | original | likes/like | 59.90% | 51.82% | 2.579 |
| transfer | adapted | is/are | 1.56% | 43.23% | -2.442 |
| transfer | adapted | has/have | 7.29% | 44.53% | -2.516 |
| transfer | adapted | does/do | 0.78% | 40.10% | -2.187 |
| transfer | adapted | was/were | 0.78% | 39.06% | -1.953 |
| transfer | adapted | seems/seem | 10.94% | 51.30% | -1.873 |
| transfer | adapted | works/work | 18.49% | 47.40% | -1.360 |
| transfer | adapted | likes/like | 50.00% | 50.00% | -1.021 |
| transfer | repaired | is/are | 100.00% | 95.31% | 6.296 |
| transfer | repaired | has/have | 99.22% | 94.53% | 5.858 |
| transfer | repaired | does/do | 95.31% | 89.84% | 4.441 |
| transfer | repaired | was/were | 99.74% | 96.09% | 5.622 |
| transfer | repaired | seems/seem | 99.22% | 94.79% | 4.848 |
| transfer | repaired | works/work | 88.54% | 87.24% | 3.649 |
| transfer | repaired | likes/like | 63.80% | 60.42% | 2.800 |
| role | original | is/are | 100.00% | 91.67% | 8.192 |
| role | original | has/have | 100.00% | 90.36% | 7.735 |
| role | original | does/do | 100.00% | 83.59% | 6.200 |
| role | original | was/were | 100.00% | 91.67% | 7.281 |
| role | original | seems/seem | 100.00% | 89.84% | 6.164 |
| role | original | works/work | 100.00% | 84.11% | 5.238 |
| role | original | likes/like | 89.58% | 64.58% | 4.298 |
| role | adapted | is/are | 91.67% | 62.24% | 1.981 |
| role | adapted | has/have | 94.01% | 56.77% | 2.009 |
| role | adapted | does/do | 94.53% | 51.30% | 1.590 |
| role | adapted | was/were | 90.10% | 59.11% | 1.687 |
| role | adapted | seems/seem | 89.32% | 53.39% | 1.602 |
| role | adapted | works/work | 85.42% | 56.51% | 1.260 |
| role | adapted | likes/like | 56.25% | 50.00% | 1.458 |
| role | repaired | is/are | 100.00% | 82.81% | 5.789 |
| role | repaired | has/have | 100.00% | 82.29% | 5.749 |
| role | repaired | does/do | 99.48% | 74.48% | 4.505 |
| role | repaired | was/were | 100.00% | 83.85% | 5.023 |
| role | repaired | seems/seem | 100.00% | 81.25% | 4.494 |
| role | repaired | works/work | 99.74% | 78.65% | 3.931 |
| role | repaired | likes/like | 71.09% | 56.25% | 3.161 |

adapted 的训练目标是错误语法，表中的“反事实语法吻合”仍按正确语法记分，不能当作对它原生反向规则的忠实度。其负效应表示与原模型相反的使用方向；role 上这个符号又变为正。

## 读取通道：信息互换效应的中介比例

| 测试集 | 状态 | L6H5 | 三个 head 联合 | 三个 head 仅最终位置 | 随机三 head，三次均值 |
|---|---|---:|---:|---:|---:|
| heldout | original | 79.01% [77.91, 79.94] | 95.40% [94.96, 95.83] | 91.42% [90.97, 91.85] | 0.89% |
| heldout | adapted | -0.17% [-0.35, 0.01] | 13.86% [13.26, 14.42] | 13.85% [13.27, 14.43] | -0.07% |
| heldout | repaired | 76.97% [75.62, 78.25] | 97.89% [97.04, 98.72] | 86.37% [85.42, 87.25] | 1.14% |
| transfer | original | 85.43% [84.50, 86.26] | 94.95% [93.95, 95.91] | 90.19% [89.28, 91.07] | 3.99% |
| transfer | adapted | 0.14% [-0.02, 0.27] | 16.29% [15.59, 17.02] | 16.39% [15.66, 17.16] | -0.29% |
| transfer | repaired | 79.83% [78.91, 80.77] | 94.23% [93.55, 94.88] | 83.45% [82.64, 84.24] | 2.59% |
| role | original | 53.45% [52.30, 54.53] | 57.39% [56.24, 58.45] | 57.39% [56.24, 58.45] | 0.51% |
| role | adapted | 29.25% [26.09, 32.70] | 36.47% [33.20, 40.01] | 36.47% [33.20, 40.01] | -0.32% |
| role | repaired | 54.53% [52.44, 56.70] | 58.09% [56.07, 60.13] | 58.09% [56.07, 60.13] | 0.42% |

比例衡量恢复指定 head 输出后，移植数信息造成的 logit 差变化被消去多少，不是删除 head 后的语法准确率；各 head 比例不可直接相加。

## L6H5 的主语注意力与源端可读信息

| 测试集 | 状态 | 主语 attention | 源端主语数：自身探针 | 源端主语数：固定原模型探针 |
|---|---|---:|---:|---:|
| heldout | original | 62.52% [61.03, 63.94] | 100.00% | 100.00% |
| heldout | adapted | 0.25% [0.17, 0.34] | 100.00% | 100.00% |
| heldout | repaired | 56.14% [53.97, 58.48] | 100.00% | 100.00% |
| transfer | original | 61.18% [57.31, 65.30] | 100.00% | 100.00% |
| transfer | adapted | 0.12% [0.10, 0.13] | 100.00% | 100.00% |
| transfer | repaired | 58.15% [54.21, 62.15] | 100.00% | 100.00% |
| role | original | 57.27% [54.49, 60.39] | 100.00% | 100.00% |
| role | adapted | 29.62% [26.39, 32.93] | 100.00% | 100.00% |
| role | repaired | 60.37% [56.76, 63.96] | 100.00% | 100.00% |

## 新句子上的单 head 路由恢复

| 测试集 | 恢复范围 | predictor 坐标数 | is/are 语法 | 主语 attention |
|---|---|---:|---:|---:|
| heldout | original | 3773 | 99.61% [98.83, 100.00] | 62.14% [60.39, 63.85] |
| heldout | adapted | 0 | 1.95% [0.39, 3.52] | 0.21% [0.14, 0.30] |
| heldout | repair75 | 75 | 98.05% [96.48, 99.61] | 55.84% [52.66, 59.02] |
| heldout | head_q | 7 | 2.34% [0.78, 4.30] | 1.34% [1.03, 1.71] |
| heldout | head_k | 7 | 1.56% [0.39, 3.12] | 15.30% [13.41, 17.41] |
| heldout | head_v | 7 | 2.34% [0.78, 4.30] | 0.21% [0.14, 0.30] |
| heldout | head_qk | 14 | 1.17% [0.00, 2.73] | 16.68% [15.41, 18.10] |
| heldout | head_qkv | 21 | 1.95% [0.39, 3.52] | 16.68% [15.41, 18.10] |
| heldout | mask_head_k | 6 | 1.17% [0.00, 2.73] | 15.44% [13.54, 17.58] |
| heldout | control_H14_qk | 14 | 1.56% [0.39, 3.12] | 0.22% [0.14, 0.31] |
| heldout | control_H0_qk | 14 | 2.34% [0.78, 4.30] | 0.22% [0.15, 0.32] |
| heldout | control_H15_qk | 14 | 1.56% [0.39, 3.12] | 0.23% [0.15, 0.32] |
| transfer | original | 3773 | 100.00% [100.00, 100.00] | 61.88% [57.23, 66.96] |
| transfer | adapted | 0 | 4.69% [1.56, 7.81] | 0.13% [0.10, 0.15] |
| transfer | repair75 | 75 | 99.61% [98.83, 100.00] | 59.71% [55.65, 64.29] |
| transfer | head_q | 7 | 4.69% [1.56, 7.81] | 0.72% [0.65, 0.79] |
| transfer | head_k | 7 | 4.69% [1.95, 7.81] | 19.83% [16.96, 22.80] |
| transfer | head_v | 7 | 5.08% [1.95, 8.59] | 0.13% [0.10, 0.15] |
| transfer | head_qk | 14 | 4.30% [1.95, 7.42] | 18.56% [16.61, 20.58] |
| transfer | head_qkv | 21 | 8.20% [3.52, 13.67] | 18.56% [16.61, 20.58] |
| transfer | mask_head_k | 6 | 4.69% [1.95, 7.81] | 20.00% [17.10, 22.99] |
| transfer | control_H14_qk | 14 | 4.30% [1.55, 7.42] | 0.13% [0.11, 0.15] |
| transfer | control_H0_qk | 14 | 4.30% [1.17, 7.42] | 0.13% [0.11, 0.15] |
| transfer | control_H15_qk | 14 | 5.47% [2.33, 8.98] | 0.13% [0.11, 0.16] |
| role | original | 3773 | 100.00% [100.00, 100.00] | 57.71% [54.34, 61.19] |
| role | adapted | 0 | 90.62% [87.50, 93.75] | 29.09% [25.74, 32.72] |
| role | repair75 | 75 | 100.00% [100.00, 100.00] | 61.76% [57.62, 65.59] |
| role | head_q | 7 | 91.80% [88.67, 94.53] | 34.62% [31.75, 37.62] |
| role | head_k | 7 | 94.92% [92.58, 97.27] | 66.56% [63.33, 69.80] |
| role | head_v | 7 | 91.02% [87.89, 94.14] | 29.09% [25.74, 32.72] |
| role | head_qk | 14 | 94.14% [91.41, 96.88] | 57.39% [54.34, 60.63] |
| role | head_qkv | 21 | 98.44% [96.88, 99.61] | 57.39% [54.34, 60.63] |
| role | mask_head_k | 6 | 94.92% [92.58, 97.27] | 67.41% [64.23, 70.61] |
| role | control_H14_qk | 14 | 90.23% [87.11, 93.36] | 28.95% [25.63, 32.55] |
| role | control_H0_qk | 14 | 90.23% [86.72, 93.75] | 26.98% [23.75, 30.49] |
| role | control_H15_qk | 14 | 90.23% [87.10, 93.75] | 28.76% [25.45, 32.32] |

这批每个 split 256 例，与前面定位及评估的全部 prompt 不重合。Q/K/V 恢复只涉及外置 predictor 输出，local correction 保持正常动态计算。
