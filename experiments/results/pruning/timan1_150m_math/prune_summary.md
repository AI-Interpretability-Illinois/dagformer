| run | units | importance | target | block sparsity | block params remaining | total params remaining | NLL before | NLL final | general NLL final |
|---|---|---|---|---|---|---|---|---|---|
| 75m_dense_math_mod_s33 | attn,mlp | taylor | 0.33 | 0.333 | 16.8M | 68.2M | 4.2707 | 2.6236 | 5.7574 |
| 75m_fourway_math_mod_s33 | attn,mlp | taylor | 0.33 | 0.333 | 16.8M | 97.3M | 3.9837 | 2.4966 | 5.6313 |
| 75m_modular_math_mod_s33 | attn,mlp | taylor | 0.33 | 0.333 | 16.8M | 97.1M | 3.9724 | 2.4787 | 5.6080 |
| 75m_modular_math_mod_s33_rc | attn,mlp | routing_column | 0.33 | 0.333 | 16.8M | 97.1M | 3.9724 | 2.4120 | 5.4533 |
| 75m_dense_math_mod_s50 | attn,mlp | taylor | 0.5 | 0.500 | 12.6M | 64.0M | 4.2707 | 2.8293 | 6.0713 |
| 75m_fourway_math_mod_s50 | attn,mlp | taylor | 0.5 | 0.500 | 12.6M | 93.1M | 3.9837 | 2.7755 | 5.9760 |
| 75m_modular_math_mod_s50 | attn,mlp | taylor | 0.5 | 0.500 | 12.6M | 92.9M | 3.9724 | 2.6198 | 5.9278 |
| 75m_modular_math_mod_s50_rc | attn,mlp | routing_column | 0.5 | 0.500 | 12.6M | 92.9M | 3.9724 | 2.6547 | 5.7494 |
| 75m_dense_math_s0 | head,neuron | taylor | 0.0 | 0.000 | 25.2M | 76.6M | 4.2707 | 2.3871 | 5.3061 |
| 75m_fourway_math_s0 | head,neuron | taylor | 0.0 | 0.000 | 25.2M | 105.7M | 3.9837 | 2.1851 | 4.9947 |
| 75m_modular_math_s0 | head,neuron | taylor | 0.0 | 0.000 | 25.2M | 105.5M | 3.9724 | 2.1876 | 5.1090 |
| 75m_dense_math_s30 | head,neuron | taylor | 0.3 | 0.298 | 17.7M | 69.1M | 4.2707 | 2.4747 | 5.4775 |
| 75m_fourway_math_s30 | head,neuron | taylor | 0.3 | 0.298 | 17.7M | 98.2M | 3.9837 | 2.2585 | 5.1426 |
| 75m_modular_math_s30 | head,neuron | taylor | 0.3 | 0.298 | 17.7M | 98.0M | 3.9724 | 2.2605 | 5.2463 |
| 75m_dense_math_s50 | head,neuron | taylor | 0.5 | 0.500 | 12.6M | 64.0M | 4.2707 | 2.6023 | 5.7300 |
| 75m_fourway_math_s50 | head,neuron | taylor | 0.5 | 0.500 | 12.6M | 93.1M | 3.9837 | 2.3501 | 5.2939 |
| 75m_modular_math_s50 | head,neuron | taylor | 0.5 | 0.500 | 12.6M | 92.9M | 3.9724 | 2.3501 | 5.4375 |
| 75m_dense_math_s70 | head,neuron | taylor | 0.7 | 0.702 | 7.5M | 58.9M | 4.2707 | 2.8153 | 6.0026 |
| 75m_fourway_math_s70 | head,neuron | taylor | 0.7 | 0.702 | 7.5M | 88.0M | 3.9837 | 2.6694 | 5.7545 |
| 75m_modular_math_s70 | head,neuron | taylor | 0.7 | 0.702 | 7.5M | 87.8M | 3.9724 | 2.5512 | 5.6821 |
| 150m_dense_math_mod_s33 | attn,mlp | taylor | 0.33 | 0.375 | 47.2M | 124.3M | 3.5985 | 2.1883 | 5.3375 |
| 150m_fourway_math_mod_s33 | attn,mlp | taylor | 0.33 | 0.375 | 47.2M | 154.2M | 3.3831 | 2.0438 | 4.9315 |
| 150m_modular_math_mod_s33 | attn,mlp | taylor | 0.33 | 0.375 | 47.2M | 154.0M | 3.4383 | 2.0692 | 5.0224 |
| 150m_modular_math_mod_s33_rc | attn,mlp | routing_column | 0.33 | 0.375 | 47.2M | 154.0M | 3.4383 | 2.0145 | 4.8730 |
| 150m_dense_math_mod_s50 | attn,mlp | taylor | 0.5 | 0.500 | 37.8M | 114.8M | 3.5985 | 2.2984 | 5.4936 |
| 150m_fourway_math_mod_s50 | attn,mlp | taylor | 0.5 | 0.500 | 37.8M | 144.8M | 3.3831 | 2.2291 | 5.4022 |
| 150m_modular_math_mod_s50 | attn,mlp | taylor | 0.5 | 0.500 | 37.8M | 144.5M | 3.4383 | 2.2560 | 5.4479 |
| 150m_modular_math_mod_s50_rc | attn,mlp | routing_column | 0.5 | 0.500 | 37.8M | 144.5M | 3.4383 | 2.1301 | 5.1657 |
| 150m_dense_math_s0 | head,neuron | taylor | 0.0 | 0.000 | 75.5M | 152.6M | 3.5985 | 1.8540 | 4.5341 |
| 150m_fourway_math_s0 | head,neuron | taylor | 0.0 | 0.000 | 75.5M | 182.6M | 3.3831 | 1.7239 | 4.2955 |
| 150m_modular_math_s0 | head,neuron | taylor | 0.0 | 0.000 | 75.5M | 182.3M | 3.4383 | 1.7546 | 4.4003 |
| 150m_dense_math_s30 | head,neuron | taylor | 0.3 | 0.300 | 52.8M | 129.9M | 3.5985 | 1.9299 | 4.7568 |
| 150m_fourway_math_s30 | head,neuron | taylor | 0.3 | 0.300 | 52.8M | 159.9M | 3.3831 | 1.7883 | 4.4923 |
| 150m_modular_math_s30 | head,neuron | taylor | 0.3 | 0.300 | 52.8M | 159.6M | 3.4383 | 1.8227 | 4.5838 |
| 150m_dense_math_s50 | head,neuron | taylor | 0.5 | 0.500 | 37.8M | 114.8M | 3.5985 | 2.0379 | 5.0270 |
| 150m_fourway_math_s50 | head,neuron | taylor | 0.5 | 0.500 | 37.8M | 144.8M | 3.3831 | 1.8824 | 4.7764 |
| 150m_modular_math_s50 | head,neuron | taylor | 0.5 | 0.500 | 37.8M | 144.5M | 3.4383 | 1.9226 | 4.8325 |
| 150m_dense_math_s70 | head,neuron | taylor | 0.7 | 0.699 | 22.7M | 99.8M | 3.5985 | 2.2477 | 5.4573 |
| 150m_fourway_math_s70 | head,neuron | taylor | 0.7 | 0.699 | 22.7M | 129.7M | 3.3831 | 2.0729 | 5.1395 |
| 150m_modular_math_s70 | head,neuron | taylor | 0.7 | 0.699 | 22.7M | 129.5M | 3.4383 | 2.1294 | 5.2892 |

| size | tag | dense final | fourway final | modular final | dense - fourway | dense - modular |
|---|---|---|---|---|---|---|
| 75m | mod_s33 | 2.6236 | 2.4966 | 2.4787 | +0.1270 | +0.1449 |
| 75m | mod_s33_rc | - | - | 2.4120 | - | - |
| 75m | mod_s50 | 2.8293 | 2.7755 | 2.6198 | +0.0538 | +0.2095 |
| 75m | mod_s50_rc | - | - | 2.6547 | - | - |
| 75m | s0 | 2.3871 | 2.1851 | 2.1876 | +0.2020 | +0.1996 |
| 75m | s30 | 2.4747 | 2.2585 | 2.2605 | +0.2162 | +0.2142 |
| 75m | s50 | 2.6023 | 2.3501 | 2.3501 | +0.2522 | +0.2522 |
| 75m | s70 | 2.8153 | 2.6694 | 2.5512 | +0.1459 | +0.2640 |
| 150m | mod_s33 | 2.1883 | 2.0438 | 2.0692 | +0.1446 | +0.1191 |
| 150m | mod_s33_rc | - | - | 2.0145 | - | - |
| 150m | mod_s50 | 2.2984 | 2.2291 | 2.2560 | +0.0693 | +0.0423 |
| 150m | mod_s50_rc | - | - | 2.1301 | - | - |
| 150m | s0 | 1.8540 | 1.7239 | 1.7546 | +0.1301 | +0.0994 |
| 150m | s30 | 1.9299 | 1.7883 | 1.8227 | +0.1417 | +0.1073 |
| 150m | s50 | 2.0379 | 1.8824 | 1.9226 | +0.1555 | +0.1152 |
| 150m | s70 | 2.2477 | 2.0729 | 2.1294 | +0.1748 | +0.1182 |
