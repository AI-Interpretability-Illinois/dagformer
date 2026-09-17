# Dependence between natural-text windows

Adjacent nonoverlapping windows can share the same source article.
This sensitivity analysis resamples circular blocks of 1, 4, 8 or 16 consecutive
windows. Each window contains 1,024 evaluated tokens. The point estimate stays
fixed; the intervals reflect different assumptions about dependence length.
These are unadjusted paired intervals for fixed checkpoints and this packed corpus.

| Model | Intervention | ΔNLL | Block 1: 95% CI | Block 4 | Block 8 | Block 16 |
|---|---|---:|---|---|---|---|
| 75m | predictor → position table | +0.002870 | [+0.002377, +0.003361] | [+0.002301, +0.003418] | [+0.002245, +0.003454] | [+0.002212, +0.003484] |
| 75m | disable correction | +0.880609 | [+0.853036, +0.909563] | [+0.842429, +0.921727] | [+0.840346, +0.924445] | [+0.841381, +0.925696] |
| 75m | correction → position table | +0.412552 | [+0.401419, +0.423551] | [+0.398258, +0.426510] | [+0.396557, +0.427885] | [+0.395772, +0.428745] |
| 150m | predictor → position table | +0.001034 | [+0.000688, +0.001368] | [+0.000663, +0.001382] | [+0.000664, +0.001375] | [+0.000700, +0.001319] |
| 150m | disable correction | +1.380647 | [+1.345034, +1.417776] | [+1.328963, +1.436649] | [+1.322139, +1.444758] | [+1.319442, +1.451180] |
| 150m | correction → position table | +0.597612 | [+0.581464, +0.613774] | [+0.574595, +0.620106] | [+0.571805, +0.623661] | [+0.572292, +0.624290] |
| 300m | predictor → position table | +0.000614 | [+0.000325, +0.000893] | [+0.000300, +0.000881] | [+0.000296, +0.000881] | [+0.000274, +0.000893] |
| 300m | disable correction | +1.749293 | [+1.706379, +1.793503] | [+1.687196, +1.816791] | [+1.677748, +1.830328] | [+1.674611, +1.837130] |
| 300m | correction → position table | +0.749462 | [+0.727829, +0.770950] | [+0.716117, +0.782540] | [+0.710426, +0.789536] | [+0.706130, +0.795560] |
| 300m_step10500 | predictor → position table | +0.000850 | [+0.000567, +0.001126] | [+0.000505, +0.001181] | [+0.000454, +0.001187] | [+0.000416, +0.001178] |
| 300m_step10500 | disable correction | +1.779187 | [+1.736678, +1.823025] | [+1.719015, +1.845491] | [+1.710047, +1.857731] | [+1.707067, +1.863599] |
| 300m_step10500 | correction → position table | +0.768557 | [+0.746962, +0.790103] | [+0.735600, +0.802162] | [+0.730030, +0.809222] | [+0.726015, +0.814702] |
| 300m_fp32 | predictor → position table | +0.000550 | [+0.000315, +0.000774] | [+0.000289, +0.000777] | [+0.000263, +0.000778] | [+0.000241, +0.000765] |
| 300m_fp32 | disable correction | +1.749249 | [+1.706282, +1.793475] | [+1.687187, +1.816729] | [+1.677629, +1.830277] | [+1.674569, +1.837163] |
| 300m_fp32 | correction → position table | +0.749496 | [+0.727855, +0.770982] | [+0.716228, +0.782515] | [+0.710573, +0.789576] | [+0.706245, +0.795580] |
