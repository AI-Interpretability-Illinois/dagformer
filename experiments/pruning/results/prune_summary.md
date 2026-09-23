| run | units | importance | target | block sparsity | block params remaining | total params remaining | NLL before | NLL final | general NLL final |
|---|---|---|---|---|---|---|---|---|---|
| 75m_baseline_math_mod_s33 | attn,mlp | taylor | 0.33 | 0.333 | 16.8M | 68.2M | 4.2663 | 2.6097 | 5.7555 |
| 75m_dagformer_math_mod_s33 | attn,mlp | taylor | 0.33 | 0.333 | 16.8M | 97.3M | 4.0518 | 2.5469 | 5.6916 |
| 75m_baseline_math_mod_s50 | attn,mlp | taylor | 0.5 | 0.500 | 12.6M | 64.0M | 4.2663 | 2.8323 | 6.1208 |
| 75m_dagformer_math_mod_s50 | attn,mlp | taylor | 0.5 | 0.500 | 12.6M | 93.1M | 4.0518 | 3.4520 | 6.7356 |
| 75m_baseline_math_s0 | head,neuron | taylor | 0.0 | 0.000 | 25.2M | 76.6M | 4.2663 | 2.3882 | 5.3220 |
| 75m_dagformer_math_s0 | head,neuron | taylor | 0.0 | 0.000 | 25.2M | 105.7M | 4.0518 | 2.2231 | 5.1195 |
| 75m_baseline_math_s30 | head,neuron | taylor | 0.3 | 0.298 | 17.7M | 69.1M | 4.2663 | 2.4777 | 5.4934 |
| 75m_dagformer_math_s30 | head,neuron | taylor | 0.3 | 0.298 | 17.7M | 98.2M | 4.0518 | 2.2978 | 5.2649 |
| 75m_baseline_math_s50 | head,neuron | taylor | 0.5 | 0.500 | 12.6M | 64.0M | 4.2663 | 2.5899 | 5.6767 |
| 75m_dagformer_math_s50 | head,neuron | taylor | 0.5 | 0.500 | 12.6M | 93.1M | 4.0518 | 2.3907 | 5.4530 |
| 75m_baseline_math_s70 | head,neuron | taylor | 0.7 | 0.702 | 7.5M | 58.9M | 4.2663 | 2.8294 | 6.0420 |
| 75m_dagformer_math_s70 | head,neuron | taylor | 0.7 | 0.702 | 7.5M | 88.0M | 4.0518 | 2.6140 | 5.7418 |

| size | tag | baseline final | dagformer final | delta (baseline - dagformer) |
|---|---|---|---|---|
| 75m | mod_s33 | 2.6097 | 2.5469 | +0.0628 |
| 75m | mod_s50 | 2.8323 | 3.4520 | -0.6197 |
| 75m | s0 | 2.3882 | 2.2231 | +0.1651 |
| 75m | s30 | 2.4777 | 2.2978 | +0.1799 |
| 75m | s50 | 2.5899 | 2.3907 | +0.1993 |
| 75m | s70 | 2.8294 | 2.6140 | +0.2154 |
