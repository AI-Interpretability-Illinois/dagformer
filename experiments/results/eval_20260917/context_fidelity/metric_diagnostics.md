# Context metric diagnostics

Fixed edit: `both/circuit/gamma1.25`. Whole-vocabulary probability averages the
per-item probabilities, not the exponentiated mean log probability.

| Source | Cue | Metric | Reference | Edited | Difference | Paired 95% interval |
|---|---|---|---:|---:|---:|---|
| original | neutral | candidate_p_true | 0.65551 | 0.68974 | +0.0342294 | [+0.0328843, +0.0355745] |
| original | neutral | candidate_accuracy | 0.902344 | 0.919922 | +0.0175781 | [+0.0095252, +0.025631] |
| original | neutral | whole_vocab_p_true | 0.147306 | 0.164706 | +0.0174003 | [+0.0164028, +0.0183978] |
| original | honest | candidate_p_true | 0.66623 | 0.704788 | +0.0385579 | [+0.0370082, +0.0401075] |
| original | honest | candidate_accuracy | 0.913086 | 0.923828 | +0.0107422 | [+0.00442506, +0.0170593] |
| original | honest | whole_vocab_p_true | 0.144227 | 0.162298 | +0.0180711 | [+0.0171651, +0.0189771] |
| original | deceptive | candidate_p_true | 0.709554 | 0.747955 | +0.0384016 | [+0.0366797, +0.0401234] |
| original | deceptive | candidate_accuracy | 0.925781 | 0.9375 | +0.0117188 | [+0.00512399, +0.0183135] |
| original | deceptive | whole_vocab_p_true | 0.166653 | 0.188519 | +0.0218661 | [+0.0208901, +0.0228421] |
| qa | neutral | candidate_p_true | 0.449169 | 0.43445 | -0.0147186 | [-0.0161146, -0.0133226] |
| qa | neutral | candidate_accuracy | 0.672852 | 0.65332 | -0.0195312 | [-0.0312753, -0.00778724] |
| qa | neutral | whole_vocab_p_true | 2.94729e-05 | 1.89774e-05 | -1.04955e-05 | [-1.12758e-05, -9.71518e-06] |
| qa | honest | candidate_p_true | 0.386858 | 0.372566 | -0.0142924 | [-0.0157544, -0.0128304] |
| qa | honest | candidate_accuracy | 0.551758 | 0.548828 | -0.00292969 | [-0.013929, +0.00806967] |
| qa | honest | whole_vocab_p_true | 1.15902e-05 | 7.38103e-06 | -4.20922e-06 | [-4.54521e-06, -3.87322e-06] |
| qa | deceptive | candidate_p_true | 0.390152 | 0.37409 | -0.016062 | [-0.0175195, -0.0146045] |
| qa | deceptive | candidate_accuracy | 0.578125 | 0.572266 | -0.00585938 | [-0.0156174, +0.00389863] |
| qa | deceptive | whole_vocab_p_true | 1.48949e-05 | 9.19015e-06 | -5.70473e-06 | [-6.17823e-06, -5.23123e-06] |
| dialogue | neutral | candidate_p_true | 0.803041 | 0.842198 | +0.0391569 | [+0.0373136, +0.0410002] |
| dialogue | neutral | candidate_accuracy | 0.974609 | 0.984375 | +0.00976562 | [+0.00373951, +0.0157917] |
| dialogue | neutral | whole_vocab_p_true | 0.317428 | 0.382536 | +0.0651075 | [+0.0631928, +0.0670223] |
| dialogue | honest | candidate_p_true | 0.761848 | 0.80887 | +0.0470222 | [+0.0448875, +0.0491569] |
| dialogue | honest | candidate_accuracy | 0.953125 | 0.974609 | +0.0214844 | [+0.0125993, +0.0303695] |
| dialogue | honest | whole_vocab_p_true | 0.235931 | 0.288841 | +0.0529092 | [+0.0514397, +0.0543787] |
| dialogue | deceptive | candidate_p_true | 0.805492 | 0.842005 | +0.0365131 | [+0.0347594, +0.0382668] |
| dialogue | deceptive | candidate_accuracy | 0.974609 | 0.980469 | +0.00585938 | [+0.00118237, +0.0105364] |
| dialogue | deceptive | whole_vocab_p_true | 0.305506 | 0.360627 | +0.0551205 | [+0.0534675, +0.0567735] |

Intervals describe variation over these paired content items, not training seeds.
Candidate accuracy is not unrestricted next-token or generated-answer accuracy.
In a QA prompt, articles or other opening words may precede the factual value;
a low whole-vocabulary probability at the first position makes the candidate
ratio an incomplete measure of answer retrieval. See the separate generation evaluation.
