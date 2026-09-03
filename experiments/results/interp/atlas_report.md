
#### ablation = wire  (176 heads)

label counts: dispensable: 169, specialized:nll_domain_latex: 2, general: 1, specialized:nll_domain_code: 1, specialized:nll_class_digit: 1, specialized:nll_class_cap: 1, specialized:nll_class_ws: 1

| head | Δ overall NLL | label | top excess (metric: value) |
|---|---|---|---|
| L4/h1 | +0.1329 | specialized:nll_domain_latex | copy_acc_p128: +74.986, copy_acc_p32: +62.689 |
| L3/h11 | +0.0524 | specialized:nll_domain_code | copy_acc_p512: +15.015, copy_acc_p128: +12.040 |
| L1/h7 | +0.0435 | general | copy_acc_p512: +1.636, copy_acc_p128: +0.349 |
| L11/h1 | +0.0268 | dispensable | copy_acc_p512: +0.439, copy_acc_p128: +0.126 |
| L3/h3 | +0.0254 | dispensable | nll_hard_q1: +0.023, nll_noncopyable: +0.009 |
| L6/h0 | +0.0246 | dispensable | nll_hard_q1: +0.018, nll_noncopyable: +0.016 |
| L3/h13 | +0.0231 | dispensable | copy_acc_p512: +8.276, copy_acc_p128: +6.292 |
| L2/h12 | +0.0200 | dispensable | copy_acc_p512: +2.368, copy_acc_p128: +1.395 |
| L4/h10 | +0.0199 | dispensable | nll_hard_q1: +0.010, nll_noncopyable: +0.008 |
| L9/h4 | +0.0167 | dispensable | nll_freq_q0: +0.033, nll_noncopyable: +0.012 |
| L4/h12 | +0.0160 | dispensable | nll_hard_q2: +0.013, nll_class_punct: +0.013 |
| L5/h3 | +0.0145 | dispensable | copy_acc_p128: +0.223, copy_acc_p512: +0.049 |
| L6/h11 | +0.0130 | specialized:nll_class_digit | copy_acc_p512: +14.307, copy_acc_p128: +13.295 |
| L7/h7 | +0.0129 | dispensable | nll_domain_code: +0.021, nll_domain_latex: +0.015 |
| L4/h3 | +0.0128 | dispensable | nll_hard_q1: +0.010, nll_noncopyable: +0.008 |
| L7/h1 | +0.0118 | specialized:nll_domain_latex | copy_acc_p512: +1.514, copy_acc_p128: +0.516 |
| L9/h6 | +0.0117 | dispensable | copy_acc_p128: +0.028, nll_freq_q1: +0.010 |
| L5/h7 | +0.0111 | dispensable | copy_acc_p512: +1.050, copy_acc_p128: +0.739 |
| L6/h9 | +0.0107 | dispensable | copy_acc_p32: +0.025, nll_hard_q2: +0.009 |
| L7/h8 | +0.0098 | dispensable | nll_freq_q0: +0.015, nll_noncopyable: +0.010 |
| L2/h11 | +0.0093 | dispensable | copy_acc_p512: +0.659, nll_domain_code: +0.012 |
| L9/h2 | +0.0087 | dispensable | nll_class_cap: +0.027, copy_acc_p512: +0.024 |
| L6/h5 | +0.0084 | dispensable | nll_noncopyable: +0.007, nll_pos_q3: +0.004 |
| L11/h5 | +0.0082 | dispensable | nll_class_punct: +0.031, nll_freq_q2: +0.030 |
| L4/h15 | +0.0081 | dispensable | nll_domain_prose: +0.023, nll_domain_code: +0.008 |
| L5/h4 | +0.0076 | dispensable | copy_acc_p128: +2.009, copy_acc_p512: +1.001 |
| L5/h14 | +0.0071 | dispensable | copy_acc_p128: +0.265, copy_acc_p32: +0.013 |
| L4/h2 | +0.0064 | dispensable | copy_acc_p32: +0.025, nll_domain_code: +0.009 |
| L11/h4 | +0.0061 | dispensable | copy_acc_p128: +0.265, copy_acc_p512: +0.171 |
| L7/h11 | +0.0059 | specialized:nll_class_cap | copy_acc_p128: +0.293, copy_acc_p32: +0.113 |
| L8/h5 | +0.0056 | dispensable | copy_acc_p128: +0.014, nll_class_ws: +0.010 |
| L2/h4 | +0.0050 | dispensable | copy_acc_p512: +1.123, copy_acc_p128: +0.572 |
| L5/h8 | +0.0050 | dispensable | copy_acc_p128: +0.516, copy_acc_p32: +0.101 |
| L11/h7 | +0.0048 | dispensable | copy_acc_p128: +0.126, nll_hard_q3: +0.025 |
| L9/h13 | +0.0041 | dispensable | copy_acc_p128: +0.223, copy_acc_p512: +0.220 |
| L1/h2 | +0.0040 | dispensable | copy_acc_p128: +0.126, nll_freq_q0: +0.004 |
| L8/h10 | +0.0039 | dispensable | copy_acc_p128: +0.167, copy_acc_p512: +0.098 |
| L2/h14 | +0.0038 | dispensable | copy_acc_p128: +0.460, copy_acc_p512: +0.439 |
| L11/h9 | +0.0038 | dispensable | copy_acc_p32: +0.013, nll_hard_q0: +0.010 |
| L10/h15 | +0.0037 | dispensable | copy_acc_p512: +0.781, copy_acc_p128: +0.377 |

#### ablation = out  (192 heads)

label counts: dispensable: 174, specialized:nll_domain_latex: 5, general: 5, specialized:nll_domain_code: 3, specialized:nll_class_cap: 2, specialized:nll_freq_q0: 1, specialized:nll_class_digit: 1, specialized:nll_class_ws: 1

| head | Δ overall NLL | label | top excess (metric: value) |
|---|---|---|---|
| L4/h1 | +0.1427 | specialized:nll_domain_latex | copy_acc_p128: +75.209, copy_acc_p32: +73.374 |
| L0/h0 | +0.0851 | specialized:nll_class_cap | copy_acc_p512: +4.468, copy_acc_p128: +0.474 |
| L1/h7 | +0.0823 | specialized:nll_freq_q0 | copy_acc_p512: +1.831, copy_acc_p128: +0.725 |
| L3/h11 | +0.0602 | specialized:nll_domain_code | copy_acc_p512: +16.357, copy_acc_p128: +13.853 |
| L0/h14 | +0.0583 | general | copy_acc_p512: +3.491, copy_acc_p128: +2.358 |
| L2/h12 | +0.0488 | general | copy_acc_p512: +3.296, copy_acc_p128: +1.981 |
| L0/h5 | +0.0469 | specialized:nll_domain_latex | copy_acc_p512: +3.516, copy_acc_p128: +2.051 |
| L6/h0 | +0.0450 | general | nll_hard_q1: +0.031, nll_noncopyable: +0.025 |
| L3/h3 | +0.0318 | general | nll_hard_q1: +0.026, nll_noncopyable: +0.010 |
| L11/h1 | +0.0311 | general | copy_acc_p512: +0.269, nll_hard_q1: +0.042 |
| L3/h13 | +0.0282 | specialized:nll_domain_code | copy_acc_p512: +9.302, copy_acc_p128: +7.031 |
| L5/h3 | +0.0264 | dispensable | copy_acc_p128: +0.516, copy_acc_p512: +0.244 |
| L5/h14 | +0.0262 | dispensable | copy_acc_p128: +0.488, copy_acc_p32: +0.076 |
| L0/h12 | +0.0248 | dispensable | copy_acc_p32: +0.025, nll_hard_q1: +0.015 |
| L5/h7 | +0.0239 | dispensable | copy_acc_p512: +1.709, copy_acc_p128: +1.270 |
| L4/h10 | +0.0238 | dispensable | nll_hard_q1: +0.014, nll_domain_code: +0.014 |
| L4/h12 | +0.0210 | dispensable | nll_class_punct: +0.020, nll_hard_q2: +0.015 |
| L9/h4 | +0.0206 | dispensable | nll_freq_q0: +0.040, nll_noncopyable: +0.015 |
| L6/h9 | +0.0200 | dispensable | nll_hard_q2: +0.014, nll_noncopyable: +0.010 |
| L6/h5 | +0.0193 | dispensable | nll_noncopyable: +0.011, nll_freq_q0: +0.007 |
| L5/h4 | +0.0190 | specialized:nll_domain_latex | copy_acc_p128: +3.292, copy_acc_p512: +2.100 |
| L7/h7 | +0.0157 | dispensable | copy_acc_p128: +0.042, nll_domain_code: +0.018 |
| L2/h11 | +0.0157 | dispensable | copy_acc_p512: +0.488, copy_acc_p128: +0.084 |
| L4/h3 | +0.0150 | dispensable | nll_hard_q1: +0.013, nll_noncopyable: +0.007 |
| L9/h6 | +0.0149 | dispensable | copy_acc_p512: +0.073, nll_freq_q1: +0.011 |
| L6/h11 | +0.0146 | specialized:nll_class_digit | copy_acc_p512: +15.552, copy_acc_p128: +15.234 |
| L7/h1 | +0.0140 | specialized:nll_domain_latex | copy_acc_p512: +1.831, copy_acc_p128: +0.711 |
| L4/h15 | +0.0130 | dispensable | nll_domain_prose: +0.026, nll_hard_q1: +0.010 |
| L7/h8 | +0.0128 | dispensable | copy_acc_p128: +0.098, nll_freq_q0: +0.019 |
| L4/h2 | +0.0124 | dispensable | nll_domain_code: +0.020, copy_acc_p32: +0.013 |
| L5/h11 | +0.0111 | dispensable | copy_acc_p128: +0.377, copy_acc_p32: +0.050 |
| L5/h8 | +0.0111 | dispensable | copy_acc_p128: +0.753, copy_acc_p32: +0.113 |
| L9/h2 | +0.0105 | dispensable | copy_acc_p512: +0.122, nll_class_cap: +0.032 |
| L11/h7 | +0.0103 | dispensable | copy_acc_p128: +0.181, nll_hard_q3: +0.034 |
| L8/h5 | +0.0090 | dispensable | copy_acc_p128: +0.056, nll_class_cap: +0.013 |
| L0/h15 | +0.0089 | dispensable | nll_domain_code: +0.028, copy_acc_p32: +0.025 |
| L11/h5 | +0.0086 | dispensable | nll_freq_q2: +0.032, nll_class_punct: +0.030 |
| L2/h4 | +0.0082 | dispensable | copy_acc_p512: +1.270, copy_acc_p128: +0.656 |
| L8/h13 | +0.0077 | dispensable | nll_freq_q0: +0.006, nll_class_cap: +0.006 |
| L6/h14 | +0.0075 | dispensable | copy_acc_p32: +0.013, nll_class_punct: +0.010 |