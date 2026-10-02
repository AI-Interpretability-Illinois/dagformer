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
| 75m_dagformer_math_s50_frozenpred | head,neuron | taylor | 0.5 | 0.500 | 12.6M | 93.1M | 4.0518 | 2.3884 | 5.4020 |
| 75m_baseline_math_s50_random | head,neuron | random | 0.5 | 0.500 | 12.6M | 64.0M | 4.2663 | 3.0286 | 6.3866 |
| 75m_dagformer_math_s50_random | head,neuron | random | 0.5 | 0.500 | 12.6M | 93.1M | 4.0518 | 2.9351 | 6.2927 |
| 75m_baseline_math_s70 | head,neuron | taylor | 0.7 | 0.702 | 7.5M | 58.9M | 4.2663 | 2.8294 | 6.0420 |
| 75m_dagformer_math_s70 | head,neuron | taylor | 0.7 | 0.702 | 7.5M | 88.0M | 4.0518 | 2.6140 | 5.7418 |
| 150m_baseline_math_mod_s33 | attn,mlp | taylor | 0.33 | 0.375 | 47.2M | 124.3M | 3.6289 | 2.2002 | 5.2455 |
| 150m_dagformer_math_mod_s33 | attn,mlp | taylor | 0.33 | 0.375 | 47.2M | 154.2M | 3.4225 | 2.1702 | 5.2984 |
| 150m_baseline_math_mod_s50 | attn,mlp | taylor | 0.5 | 0.500 | 37.8M | 114.8M | 3.6289 | 2.3801 | 5.6651 |
| 150m_dagformer_math_mod_s50 | attn,mlp | taylor | 0.5 | 0.500 | 37.8M | 144.8M | 3.4225 | 2.2448 | 5.4760 |
| 150m_baseline_math_s0 | head,neuron | taylor | 0.0 | 0.000 | 75.5M | 152.6M | 3.6289 | 1.8825 | 4.5872 |
| 150m_dagformer_math_s0 | head,neuron | taylor | 0.0 | 0.000 | 75.5M | 182.6M | 3.4225 | 1.7432 | 4.3390 |
| 150m_baseline_math_s30 | head,neuron | taylor | 0.3 | 0.300 | 52.8M | 129.9M | 3.6289 | 1.9567 | 4.7957 |
| 150m_dagformer_math_s30 | head,neuron | taylor | 0.3 | 0.300 | 52.8M | 159.9M | 3.4225 | 1.8068 | 4.5346 |
| 150m_baseline_math_s50 | head,neuron | taylor | 0.5 | 0.500 | 37.8M | 114.8M | 3.6289 | 2.0709 | 5.0795 |
| 150m_dagformer_math_s50 | head,neuron | taylor | 0.5 | 0.500 | 37.8M | 144.8M | 3.4225 | 1.8989 | 4.7484 |
| 150m_dagformer_math_s50_frozenpred | head,neuron | taylor | 0.5 | 0.500 | 37.8M | 144.8M | 3.4225 | 1.8994 | 4.7412 |
| 150m_baseline_math_s50_random | head,neuron | random | 0.5 | 0.500 | 37.8M | 114.8M | 3.6289 | 2.2798 | 5.5577 |
| 150m_dagformer_math_s50_random | head,neuron | random | 0.5 | 0.500 | 37.8M | 144.8M | 3.4225 | 2.1285 | 5.3649 |
| 150m_baseline_math_s70 | head,neuron | taylor | 0.7 | 0.699 | 22.7M | 99.8M | 3.6289 | 2.2783 | 5.5274 |
| 150m_dagformer_math_s70 | head,neuron | taylor | 0.7 | 0.699 | 22.7M | 129.7M | 3.4225 | 2.0911 | 5.2018 |
| 300m_baseline_math_s0 | head,neuron | taylor | 0.0 | 0.000 | 201.4M | 304.1M | 3.2014 | 1.6078 | 3.9898 |
| 300m_dagformer_math_s0 | head,neuron | taylor | 0.0 | 0.000 | 201.4M | 336.4M | 3.0317 | 1.4881 | 3.7930 |
| 300m_baseline_math_s30 | head,neuron | taylor | 0.3 | 0.300 | 140.9M | 243.6M | 3.2014 | 1.6650 | 4.2198 |
| 300m_dagformer_math_s30 | head,neuron | taylor | 0.3 | 0.300 | 140.9M | 275.9M | 3.0317 | 1.5406 | 4.0112 |
| 300m_baseline_math_s50 | head,neuron | taylor | 0.5 | 0.500 | 100.7M | 203.5M | 3.2014 | 1.7624 | 4.5455 |
| 300m_dagformer_math_s50 | head,neuron | taylor | 0.5 | 0.500 | 100.7M | 235.8M | 3.0317 | 1.6202 | 4.2942 |
| 300m_baseline_math_s70 | head,neuron | taylor | 0.7 | 0.699 | 60.5M | 163.3M | 3.2014 | 1.9666 | 5.0828 |
| 300m_dagformer_math_s70 | head,neuron | taylor | 0.7 | 0.699 | 60.5M | 195.6M | 3.0317 | 1.7896 | 4.6909 |

| size | tag | baseline final | dagformer final | delta (baseline - dagformer) |
|---|---|---|---|---|
| 75m | mod_s33 | 2.6097 | 2.5469 | +0.0628 |
| 75m | mod_s50 | 2.8323 | 3.4520 | -0.6197 |
| 75m | s0 | 2.3882 | 2.2231 | +0.1651 |
| 75m | s30 | 2.4777 | 2.2978 | +0.1799 |
| 75m | s50 | 2.5899 | 2.3907 | +0.1993 |
| 75m | s50_random | 3.0286 | 2.9351 | +0.0935 |
| 75m | s70 | 2.8294 | 2.6140 | +0.2154 |
| 150m | mod_s33 | 2.2002 | 2.1702 | +0.0300 |
| 150m | mod_s50 | 2.3801 | 2.2448 | +0.1353 |
| 150m | s0 | 1.8825 | 1.7432 | +0.1393 |
| 150m | s30 | 1.9567 | 1.8068 | +0.1498 |
| 150m | s50 | 2.0709 | 1.8989 | +0.1720 |
| 150m | s50_random | 2.2798 | 2.1285 | +0.1513 |
| 150m | s70 | 2.2783 | 2.0911 | +0.1872 |
| 300m | s0 | 1.6078 | 1.4881 | +0.1197 |
| 300m | s30 | 1.6650 | 1.5406 | +0.1244 |
| 300m | s50 | 1.7624 | 1.6202 | +0.1422 |
| 300m | s70 | 1.9666 | 1.7896 | +0.1770 |
