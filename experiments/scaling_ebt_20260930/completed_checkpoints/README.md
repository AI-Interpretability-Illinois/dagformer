# Completed 600M and 1B common evaluation

Evaluated 2026-09-30 with revision `8f6338a`, PyTorch 2.10.0, BF16 backbone,
FP32 predictor and FP32 cross entropy. The caches are the same WikiText2,
MathInstruct and GSM8K caches used in PR #3. Values are nats per token.
MathInstruct and GSM8K measure text likelihood, not answer accuracy.

| Nominal size | Method | Actual total parameters | Training tokens | WikiText2 | MathInstruct | GSM8K |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 600M | Dense | 682,710,528 | 12,006,195,200 | 3.206835 | 2.340126 | 2.546689 |
| 600M | DAGFormer | 717,003,240 | 12,006,195,200 | 3.068660 | 2.221328 | 2.442499 |
| 1B | Dense | 1,279,395,840 | 20,006,830,080 | 3.135573 | 1.882516 | 2.251962 |
| 1B | DAGFormer | 1,316,049,239 | 20,006,830,080 | 2.979988 | 1.768624 | 2.091051 |

The complete encoder predictor, correction MLPs and V norms are included in
both DAGFormer parameter counts. The 600M runs use the Dolma continuation
recipe; the 1B runs use OLMo-mix and its reconstructed continuation. These
groups remain separate from the PR's 12B and 21B corpus series when plotted.

Every checkpoint's completed budget is supported by the optimizer update
counts recorded when the evaluation export was created: 22,900 and 38,160
updates respectively, with 524,288 tokens per update. The new split's Dolma
cache is not used to score these older models.

Each model directory includes `evaluation.json` and `evaluation.npz`.
The NPZ contains per-window loss sums and valid-token counts, permitting
paired comparisons on the identical windows. WikiText2 has 282 windows,
MathInstruct 436, and GSM8K 209, each of 1,024 predicted tokens.

Paired window-bootstrap 95% intervals for the WikiText2 NLL reduction are
[0.13380, 0.14254] for 600M and [0.15157, 0.15944] for 1B
(5,000 resamples, NumPy seed 730). These summarize variation across this cache's
windows; they do not estimate variation across training seeds or documents.
