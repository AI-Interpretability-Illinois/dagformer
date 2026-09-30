| run | units | importance | target | block sparsity | block params remaining | total params remaining | NLL before | NLL final | general NLL final |
|---|---|---|---|---|---|---|---|---|---|
| 150m_baselinest_math_s0 | head,neuron | taylor | 0.0 | 0.000 | 75.5M | 152.6M | 4.0620 | 2.0752 | 5.0591 |
| 150m_dagformerst_math_s0 | head,neuron | taylor | 0.0 | 0.000 | 75.5M | 182.6M | 3.8596 | 1.9237 | 4.8744 |
| 150m_muddformer_math_s0 | head,neuron | taylor | 0.0 | 0.000 | 75.5M | 152.7M | 3.8543 | 1.9348 | 4.8151 |
| 150m_baselinest_math_s30 | head,neuron | taylor | 0.3 | 0.300 | 52.8M | 129.9M | 4.0620 | 2.1569 | 5.2299 |
| 150m_dagformerst_math_s30 | head,neuron | taylor | 0.3 | 0.300 | 52.8M | 159.9M | 3.8596 | 1.9902 | 5.0241 |
| 150m_muddformer_math_s30 | head,neuron | taylor | 0.3 | 0.300 | 52.8M | 130.0M | 3.8543 | 2.0369 | 5.0604 |
| 150m_baselinest_math_s50 | head,neuron | taylor | 0.5 | 0.500 | 37.8M | 114.8M | 4.0620 | 2.2729 | 5.4629 |
| 150m_dagformerst_math_s50 | head,neuron | taylor | 0.5 | 0.500 | 37.8M | 144.8M | 3.8596 | 2.0866 | 5.2312 |
| 150m_muddformer_math_s50 | head,neuron | taylor | 0.5 | 0.500 | 37.8M | 115.0M | 3.8543 | 2.1531 | 5.2561 |
| 150m_dagformerst_math_s50_frozenrouter | head,neuron | taylor | 0.5 | 0.500 | 37.8M | 144.8M | 3.8596 | 2.0782 | 5.2031 |
| 150m_muddformer_math_s50_frozenrouter | head,neuron | taylor | 0.5 | 0.500 | 37.8M | 115.0M | 3.8543 | 2.1500 | 5.2937 |
| 150m_baselinest_math_s70 | head,neuron | taylor | 0.7 | 0.699 | 22.7M | 99.8M | 4.0620 | 2.5066 | 5.9168 |
| 150m_dagformerst_math_s70 | head,neuron | taylor | 0.7 | 0.699 | 22.7M | 129.7M | 3.8596 | 2.2676 | 5.5720 |
| 150m_muddformer_math_s70 | head,neuron | taylor | 0.7 | 0.699 | 22.7M | 99.9M | 3.8543 | 2.3665 | 5.6243 |

| size | tag | baseline final | dagformer final | delta (baseline - dagformer) |
|---|---|---|---|---|
