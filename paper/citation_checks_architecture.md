# Citation verification report (architecture refs)

Generated 2026-10-01. Sources fetched live from arxiv.org / publisher pages; quotes verbatim.

## 1. key=zhu2024hyperconnections

- Fetched: YES (https://arxiv.org/abs/2409.19606 and https://arxiv.org/html/2409.19606v2)
- Bibliographic data (as shown on the arXiv page): Title "Hyper-Connections". Authors: Defa Zhu, Hongzhi Huang, Zihao Huang, Yutao Zeng, Yunyao Mao, Banggu Wu, Qiyang Min, Xun Zhou (8 authors; affiliation in the paper: "Seed-Foundation-Model Team, ByteDance"). Submitted 29 Sep 2024 (v1); current version v2, "arXiv:2409.19606v2 [cs.LG] 28 Nov 2024". Subjects: cs.LG; cs.CL; cs.CV; cs.NE. No "Comments" field, no journal ref, no DOI shown on the abstract page. (Year: 2024.)

- Claim (a) HC replaces residual connections with learnable connections between an expanded set of hidden-state streams (width/depth connections).
  - Quote (abstract): "We present hyper-connections, a simple yet effective method that can serve as an alternative to residual connections."
  - Quote (Sec. 2.1): "Initially, [h0] is replicated [n] times to form the initial hyper hidden matrix [H0]. Here, [n] is the expansion rate." and "The hyper-connections (HC) can be represented by a matrix [HC], where each element defines the connection weight."
  - Quote (Fig. 2 caption): "These connections enable lateral information exchange and vertical integration of features across depths. ... They can be decoupled into depth-connections and width-connections. (c) Depth-connections perform a weighted sum between the layer output and the hidden vector [h]. (d) Width-connections allow information exchange between the hidden vectors [h_i] and [h_j]."
  - Verdict: SUPPORTED. The paper defines an n-times expanded "hyper hidden matrix" and learnable matrices decomposed into depth- and width-connections, replacing residuals.

- Claim (b) weights can be static or dynamic; dynamic weights computed from the hidden states at each layer (input-dependent).
  - Quote (Sec. 2 headings): "2.1 Static Hyper-Connections" / "2.2 Dynamic Hyper-Connections".
  - Quote (Sec. 2.2): "The entries of [HC] can dynamically depend on the input [H], which the matrix representation of dynamic hyper-connections (DHC) is defined as follows:" and "In practice, we combine the dynamic and static matrices to achieve DHC. The dynamic parameters are obtained through a linear transformation. To stabilize the training process, we introduce normalization before the linear transformation and apply the tanh activation function after it, scaling it by a small initial learnable factor."
  - Quote (Fig. 2 caption): "[the connection weights] are learnable scalars or scalars predicted by the network, depending on the specific HC version."
  - Verdict: SUPPORTED. Dynamic HC entries are a function of the layer input H (normalized, linear map, tanh, learnable scale), combined with a static matrix.

- Claim (c) improved LLM pretraining vs residual connections; headline numbers.
  - Quote (abstract): "We conduct experiments focusing on the pre-training of large language models, including dense and sparse models, where hyper-connections show significant performance improvements over residual connections."
  - Quote (Sec. 1): "The model utilizing DHC converges 1.8 times faster and shows an improvement of 6 points on ARC-Challenge compared to the baseline trained with 500 B tokens." (this is for OLMoE-1B-7B, Fig. 1: "Our method converges 1.8 times faster compared to the baseline and maintains a significant advantage at the 500B tokens.")
  - Quote (Sec. 4.3, dense 7B): "In the V2 evaluation, OLMo-7B-DHC[x]4 shows improvements of 0.022 for loss and 0.293 for PPL. Furthermore, the average score of downstream benchmarks 0.710 surpasses the baseline 0.701". (Sec. 4.1, 1B: "OLMo-1B-DHC[x]8 W/O tanh excels on both V2 and V3 validation sets, with a reduction in V2 Eval Loss by 0.034 and V3 Eval Loss by 0.029 compared to the baseline.") Sec. 4.4 (MoE): "In many metrics, our method requires only half of the training tokens to achieve the same performance as the baseline."
  - Verdict: SUPPORTED. Headline numbers as stated above (1.8x convergence speedup and +6 ARC-C for OLMoE-1B-7B at 500B tokens; 1B/7B dense loss reductions of ~0.02-0.03).

- Training instability / constraints on mixing matrices: The paper does NOT report instability of HC itself; it claims the opposite: Sec. 4.1: "DHC demonstrates greater stability, with no spikes observed in any DHC experiments." Sec. 4.3: "the baseline model exhibits frequent spikes, while our model with DHCs shows no spikes throughout the training. This shows that our approach not only achieves better loss but also ensures more stable training." The only stabilization device is the tanh/normalization in the dynamic branch ("To stabilize the training process, we introduce normalization before the linear transformation and apply the tanh activation function after it, scaling it by a small initial learnable factor."). Section 4.1 notes expansion rate 1 is worse than baseline ("with an expansion rate of [1], the performance of DHC is inferior to the baseline") and Appendix F "How HC[x1] fails" explains it. No manifold/doubly-stochastic or other constraint on the mixing matrices is imposed; the static part is just initialized to recover Pre-Norm residuals (Sec. 2.3) and trained without weight decay (Sec. 4 Implementation: "The static component in Eqs. 1, 11, 12, 13 does not utilize weight decay, whereas the dynamic component does.").

## 2. key=mhc2026

- Fetched: YES (https://arxiv.org/abs/2512.24880 and https://arxiv.org/html/2512.24880v2). The id 2512.24880 is correct for this title.
- Bibliographic data (as shown on the arXiv page): Title "mHC: Manifold-Constrained Hyper-Connections". Authors (20): Zhenda Xie, Yixuan Wei, Huanqi Cao, Chenggang Zhao, Chengqi Deng, Jiashi Li, ... (20 authors in total; last author Wenfeng Liang); affiliation "DeepSeek-AI". Submitted 31 Dec 2025 (v1), "last revised 5 Jan 2026 (this version, v2)"; header "arXiv:2512.24880v2 [cs.CL] 05 Jan 2026". Subjects: cs.CL; cs.AI; cs.LG. No Comments field, no journal ref, no DOI shown. (Year on arXiv: 2025 submission; v2 Jan 2026.)

- Claim (a) unconstrained HC becomes unstable at scale; the problem they identify.
  - Quote (abstract): "While yielding substantial performance gains, this diversification fundamentally compromises the identity mapping property intrinsic to the residual connection, which causes severe training instability and restricted scalability, and additionally incurs notable memory access overhead."
  - Quote (Sec. 1): "In contrast to Eq. (2), the composite mapping [prod_i H^res_{L-i}] in HC fails to preserve the global mean of the features. This discrepancy leads to unbounded signal amplification or attenuation, resulting in instability during large-scale training."
  - Quote (Sec. 3.1): "Since the learnable mapping H^res_l is unconstrained, this composite mapping inevitably deviates from the identity mapping. Consequently, the signal magnitude is prone to explosion or vanishing during both the forward pass and backpropagation." / "We observe unstable loss behavior in large-scale experiments, as illustrated in Fig. 2. Taking mHC as the baseline, HC exhibits an unexpected loss surge around the 12k step, which is highly correlated with the instability in the gradient norm." / "As shown in Fig. 3 (b), the Amax Gain Magnitude yields extreme values with peaks of 3000, a stark divergence from 1 that confirms the presence of exploding residual streams." (Fig. 2 caption: "All results are based on 27B models.")
  - Verdict: SUPPORTED. The identified problem: the product of unconstrained per-layer residual-mixing matrices H^res across depth loses the identity-mapping / mean-conservation property, so the signal (forward) and gradient (backward) gains explode or vanish; observed as a loss surge and gradient-norm instability in a 27B run.

- Claim (b) mHC constrains the residual-mixing matrix to a manifold (doubly stochastic / Birkhoff polytope via Sinkhorn-Knopp).
  - Quote (Sec. 1): "Specifically, mHC utilizes the Sinkhorn-Knopp algorithm (Sinkhorn and Knopp, 1967) to entropically project H^res_l onto the Birkhoff polytope. This operation effectively constrains the residual connection matrices within the manifold that is constituted by doubly stochastic matrices."
  - Quote (Sec. 4.1): "To this end, we restrict H^res_l to be a doubly stochastic matrix, which has non-negative entries where both the rows and columns sum to 1. Formally, let M^res denote the manifold of doubly stochastic matrices (also known as the Birkhoff polytope)."
  - Quote (Sec. 4.1): "Additionally, we impose non-negativity constraints on the input mappings H^pre_l and output mappings H^post_l."
  - Verdict: SUPPORTED. (Note: only H^res is projected onto the Birkhoff polytope; H^pre and H^post are constrained to be non-negative via sigmoid, Eq. 8: H^pre = sigma(.), H^post = 2 sigma(.), H^res = Sinkhorn-Knopp(.), with t_max = 20 iterations.)

- Claim (c) mixing coefficients computed per layer from that layer's current hidden state (per-layer local router), not by a separate module predicting all layers jointly.
  - Quote (Sec. 3, describing HC, Eq. 5): "In the HC formulation, learnable mappings are composed of two parts of coefficients: the input-dependent one and the global one, referred to as dynamic mappings and static mappings, respectively." ... "The dynamic mappings are derived via linear projections parameterized by theta^pre_l, theta^post_l in R^{1xC} and theta^res_l in R^{nxC}, while the static mappings are represented by learnable biases b^pre_l, b^post_l in R^{1xn} and b^res_l in R^{nxn}."
  - Quote (Sec. 4.2, mHC): "Given the input hidden matrix x_l in R^{nxC} at the l-th layer, we first flatten it into a vector x_vec_l = vec(x_l) in R^{1 x nC} to preserve full context information. Then, we follow the original HC formulation to get the dynamic mappings and the static mappings as follows:" [Eq. 7: x'_l = RMSNorm(x_vec_l); H~^pre_l = alpha^pre_l (x'_l phi^pre_l) + b^pre_l; H~^post_l = alpha^post_l (x'_l phi^post_l) + b^post_l; H~^res_l = alpha^res_l mat(x'_l phi^res_l) + b^res_l] "where phi^pre_l, phi^post_l in R^{nC x n} and phi^res_l in R^{nC x n^2} are linear projections for dynamic mappings and mat(.) is a reshape function from R^{1 x n^2} to R^{n x n}."
  - Quote (Sec. 4.2, Eq. 8 text): "Then, the final constrained mappings are obtained via: [H^pre_l = sigma(H~^pre_l); H^post_l = 2 sigma(H~^post_l); H^res_l = Sinkhorn-Knopp(H~^res_l)] where sigma(.) denotes the Sigmoid function."
  - Verdict: SUPPORTED. The coefficients for layer l are a per-layer linear projection (phi_l) of the RMS-normalized, flattened n-stream hidden state x_l entering that layer, plus a per-layer static bias; there is no separate module predicting all layers' routing jointly.

- Headline results stated (Sec. 5.2, 27B): "mHC effectively mitigates the training instability observed in HC, achieving a final loss reduction of 0.021 compared to the baseline." Sec. 1: "In-house large-scale training indicates that mHC supports training at scale and introduces only a 6.7% additional time overhead when expansion rate n=4."

## 3. key=muddformer2025

- Fetched: YES (https://arxiv.org/abs/2502.12170 and https://arxiv.org/html/2502.12170 -> v2)
- Bibliographic data (as shown on the arXiv page): Title "MUDDFormer: Breaking Residual Bottlenecks in Transformers via Multiway Dynamic Dense Connections". Authors: Da Xiao, Qingye Meng, Shengping Li, Xingyuan Yuan (4). Submitted 13 Feb 2025 (v1), "last revised 28 May 2025 (this version, v2)"; header "arXiv:2502.12170v2 [cs.LG] 28 May 2025". Comments: "Accepted to the 42nd International Conference on Machine Learning (ICML'25)". Subjects: cs.LG; cs.AI; cs.CL. No journal ref, no DOI shown. Year 2025.

- Claim (a) each block's input is a weighted combination of all previous blocks' outputs.
  - Quote (Sec. 2): "With dense connections, the input to layer i+1 is an aggregation X-bar_i in R^{T x D} of outputs of all i+1 preceding layers, from the embedding till layer i: X_{:i} := {X_0, ..., X_i} (Figure 2 (a))."
  - Quote (Sec. 2.1, Eq. 4): "In the simplest case, the DA module aggregates previous layers' outputs by taking a weighted sum of them (Figure 2 (b), also equivalent to DenseFormer (Pagliardini et al., 2024)):" [X-bar_i = DA_i^static(X_{:i}; theta_i^s) = wsum(a_i, X_{:i}) := sum_{j=0}^{i} a_ij X_j]
  - Verdict: SUPPORTED. The Depth-wise Aggregate (DA) module after layer i forms the input of layer i+1 as a weighted sum over the embedding output X_0 and all block outputs X_1..X_i.

- Claim (b) weights are DYNAMIC, generated per sequence position from the hidden state at that depth by a module in each layer.
  - Quote (abstract): "MUDD generates connection weights dynamically depending on hidden states at each sequence position and for each decoupled input stream (the query, key, value or residual) of a Transformer block."
  - Quote (Sec. 2.2): "Dynamic dense connections expand the connection weight for X_j from a static scalar a_ij to a vector A_ij in R^T, allowing X_j to contribute differentially to each position t in [1,T] of X-bar_i based on the hidden state X_i[t] in R^D at that position. The i+1 weight vectors stack into a matrix A_i in R^{T x (i+1)} which is generated dynamically by a function A_i(.) depending on X_i (Figure 2 (c), cf. Eq. (4)):"
  - Quote (Sec. 2.2, Eq. 6): "We instantiate A_i : R^D -> R^{i+1} with an MLP parameterized by W_1 and W_2 which computes connection weights position-wise: A_i(X_i) = GELU(RMSNorm(X_i) W_1) W_2 + a_i" ... "We also add a static weight vector a_i acting as learnable prior for dense connectivity. The trainable parameters are theta_i^d = {W_1 in R^{D x (i+1)}, W_2 in R^{(i+1) x (i+1)}, a_i in R^{i+1}}."
  - Verdict: SUPPORTED. Weights for the DA module after layer i are produced position-wise by a per-layer MLP from X_i (the output of layer i, i.e. the hidden state at that depth), plus a static prior a_i. Note the weights are computed from X_i only (the current layer's output), "the attention weights are computed from the query side".

- Claim (c) "multiway" = separate weights for query, key, value and residual input streams.
  - Quote (Sec. 2.3): "In a Transformer block, a single input is reused simultaneously as the query, key, value and residual of the MHA module (Figure 2 (e) top). These input streams play divergent roles and we hypothesize that they benefit from differentiated dense connectivity. To enable this, we first turn a normal Transformer block B(X) into a multi-input one B'(X^Q, X^K, X^V, X^R) by decoupling its input into four streams for query, key, value and residual, respectively (Eq. (7), Figure 2 (e) bottom), and then instantiate four DA modules, each specializing in one stream's dense connectivity (Eq. (8), Figure 2 (d))"
  - Quote (Sec. 2.3): "Multiway dynamic dense connections can be seen as depth-wise multi(4)-head attention."
  - Verdict: SUPPORTED.

- Claim (d) headline results: ~1.8x-2.4x compute; MUDDPythia-2.8B matches Pythia-6.9B.
  - Quote (abstract): "Extensive experiments show that MUDDFormer significantly outperforms Transformers across various model architectures and scales in language modeling, achieving the performance of Transformers trained with 1.8X-2.4X compute. Notably, MUDDPythia-2.8B matches Pythia-6.9B in pretraining ppl and downstream tasks and even rivals Pythia-12B in five-shot settings, while adding only 0.23% parameters and 0.4% computation."
  - Quote (Sec. 1): "MUDDFormer significantly outperforms Transformer across various model architectures and scales (from 405M model on 7B tokens to 2.8B models on 300B tokens), achieving performance of Transformers trained with ~ 1.8x-2.4x compute."
  - Verdict: SUPPORTED.

- Claim (e) mixing is per layer (one scalar weight per source layer per position, shared across heads/channels), not per attention head.
  - Quote (Sec. 2.2, Eq. 5): "[X-bar_i] := sum_{j=0}^{i} A_ij (T x 1) ⊙ X_j (T x D) (with broadcasting)" and "where A_ij is the jth column of A_i (with a slight abuse of notation)."
  - Quote (Sec. 2.2): "Overall, dynamic dense connections can be viewed as a form of depth-wise single-headed self-attention (Vaswani et al., 2017) with query X_i and keys X_{:i}, and the attention weights are computed from the query side"
  - Verdict: SUPPORTED, with a caveat on wording. Per stream, the weight for source layer j at position t is a single scalar (A_i in R^{T x (i+1)}, broadcast over all D channels), so it is not per attention head. However, the paper itself describes the 4 Q/K/V/R streams as "depth-wise multi(4)-head attention", so if the citing text says "not per head" it should make clear that "head" means attention head; there are four separate weight sets (one per input stream).

## 4. key=pagliardini2024denseformer

- Fetched: YES (https://arxiv.org/abs/2402.02622 and https://arxiv.org/html/2402.02622 -> v2)
- Bibliographic data (as shown on the arXiv page): Title "DenseFormer: Enhancing Information Flow in Transformers via Depth Weighted Averaging". Authors: Matteo Pagliardini, Amirkeivan Mohtashami, Francois Fleuret, Martin Jaggi (4). Submitted 4 Feb 2024 (v1), "last revised 21 Mar 2024 (this version, v2)"; header "arXiv:2402.02622v2 [cs.CL] 21 Mar 2024". No Comments field, no journal ref. DOI cell: https://doi.org/10.48550/arXiv.2402.02622 (arXiv-issued DOI via DataCite). Subjects: cs.CL; cs.LG. Year 2024.

- Claim (a) after each block, a learned weighted average of the current and all past block outputs (DWA) with static scalar weights, not input-dependent.
  - Quote (abstract): "Our approach relies on an additional averaging step after each transformer block, which computes a weighted average of current and past representations -- we refer to this operation as Depth-Weighted-Average (DWA)."
  - Quote (Sec. 3): "The only change to the original architecture is the addition of a Depth Weighted Average module (DWA) after each transformer block." ... "The elements of the alpha matrix are the only additional parameters of our method." [Y_i := DWA_i({X_0,...,X_i}) = sum_{j=0}^{i} alpha_{i,j} . X_j]
  - Quote (Sec. 2, Related Work): "Our DenseFormer can be seen as a crude and much more efficient approximation of this mechanism in which we restrict each token to only attend to past representations of themselves, using static (as opposed to dynamic) attention weights." Also (Sec. 3.1): "At depth i the DWA module has i+1 weights. Therefore, for a DenseFormer of depth d, the total number of additional parameters is sum_{j=1}^{d}(j+1) = d(d+3)/2."
  - Verdict: SUPPORTED. The DWA weights alpha_{i,j} are learned scalars (one per (depth, source) pair), explicitly described as static, not input-dependent.

- Claim (b) improves perplexity at small extra cost.
  - Quote (abstract): "We propose DenseFormer, a simple modification to the standard architecture that improves the perplexity of the model without increasing its size -- adding a few thousand parameters for large-scale models in the 100B parameters range."
  - Quote (abstract): "Experiments demonstrate that DenseFormer is more data efficient, reaching the same perplexity of much deeper transformer models, and that for the same perplexity, these new models outperform transformer baselines in terms of memory efficiency and inference time."
  - Quote (Sec. 3.1): "For typical model depths (less than 100 blocks), this represents at most an order of 10^3 parameters, which is negligible when compared to the full size of the models." / "Therefore, the total memory overhead of DenseFormer is negligible." Note (Sec. 3.1): "Computing the output of the DWA modules increases the computational cost since it requires averaging over multiple large tensors ... In this work, we provide an efficient implementation of DWA to reduce the overhead" and dilation/periodicity variants are introduced to reduce it.
  - Verdict: SUPPORTED. Parameter/memory overhead is negligible; compute overhead is non-zero but reduced to "a negligible amount" via an efficient implementation and dilation/period hyperparameters.

## 5. key=elhage2021

- Fetched: YES (https://transformer-circuits.pub/2021/framework/index.html)
- Bibliographic data (as shown on the page): Title "A Mathematical Framework for Transformer Circuits". Authors (25): Nelson Elhage, Neel Nanda, Catherine Olsson, Tom Henighan, Nicholas Joseph, Ben Mann, ... Chris Olah (25 authors in total). Affiliation: Anthropic. "Published Dec 22, 2021". Page's own BibTeX: @article{elhage2021mathematical, journal={Transformer Circuits Thread}, year={2021}, note={https://transformer-circuits.pub/2021/framework/index.html}. No arXiv id, no DOI.

- Claim (a) the residual stream is a communication channel that attention heads and MLPs read from and write to.
  - Quote (Summary of Results): "All components of a transformer (the token embedding, attention heads, MLP layers, and unembedding) communicate with each other by reading and writing to different subspaces of the residual stream."
  - Quote (section "Virtual Weights and the Residual Stream as a Communication Channel"): "The residual stream is simply the sum of the output of all the previous layers and the original embedding. We generally think of the residual stream as a communication channel, since it doesn't do any processing itself and all layers communicate through it."
  - Quote (same section): "Every layer performs an arbitrary linear transformation to "read in" information from the residual stream at the start, ... and performs another arbitrary linear transformation before adding to "write" its output back into the residual stream."
  - Verdict: SUPPORTED.

- Claim (b) residual stream dimensions are in high demand / a bottleneck because all layers share it.
  - Quote (subsection "Subspaces and Residual Stream Bandwidth"): "Once added, information persists in a subspace unless another layer actively deletes it. From this perspective, dimensions of the residual stream become something like "memory" or "bandwidth"."
  - Quote (same subsection): "It seems like we should expect residual stream bandwidth to be in very high demand! There are generally far more "computational dimensions" (such as neurons and attention head result dimensions) than the residual stream has dimensions to move information. Just a single MLP layer typically has four times more neurons than the residual stream has dimensions. So, for example, at layer 25 of a 50 layer transformer, the residual stream has 100 times more neurons as it has dimensions before it, trying to communicate with 100 times as many neurons as it has dimensions after it, somehow communicating in superposition! We call tensors like this "bottleneck activations" and expect them to be unusually challenging to interpret."
  - Quote (Notation/Technical appendix, "Bottleneck Activations"): "For example, the residual stream is a bottleneck activation because it is the only way to pass information between MLP activations, which are typically four times larger than it. (Additionally, in addition to it being a bottleneck for the communication between adjacent MLP layers, it's also the only pathway by which arbitrary early MLP layers can communicate with arbitrary late MLP layers, so the stream may simultaneously be conveying different pieces of information between many different pairs of MLP layers, a much more extreme bottleneck than just 4x!)"
  - Also: "Perhaps because of this high demand on residual stream bandwidth, we've seen hints that some MLP neurons and attention heads may perform a kind of "memory management" role, clearing residual stream dimensions set by other layers by reading in information and writing out the negative version."
  - Verdict: SUPPORTED. Their exact wording is "residual stream bandwidth to be in very high demand" and "bottleneck activations".

## 6. key=vaswani2017

- Fetched: YES (https://arxiv.org/abs/1706.03762 and https://arxiv.org/pdf/1706.03762 -> v7 PDF)
- Bibliographic data (as shown on the arXiv page): Title "Attention Is All You Need". Authors: Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser, Illia Polosukhin (8). Submitted 12 Jun 2017 (v1), "last revised 2 Aug 2023 (this version, v7)". Comments: "15 pages, 5 figures". No journal ref; DOI cell: https://doi.org/10.48550/arXiv.1706.03762 (arXiv-issued, DataCite). Subjects: cs.CL; cs.LG. Year 2017.

- Claim: introduces the Transformer with residual connections around each sub-layer.
  - Quote (abstract): "We propose a new simple network architecture, the Transformer, based solely on attention mechanisms, dispensing with recurrence and convolutions entirely."
  - Quote (Sec. 3.1 Encoder and Decoder Stacks): "We employ a residual connection [11] around each of the two sub-layers, followed by layer normalization [1]. That is, the output of each sub-layer is LayerNorm(x + Sublayer(x)), where Sublayer(x) is the function implemented by the sub-layer itself."
  - Quote (Sec. 3.1, Decoder): "Similar to the encoder, we employ residual connections around each of the sub-layers, followed by layer normalization."
  - Verdict: SUPPORTED.

## 7. key=he2016resnet

- Fetched: YES (https://arxiv.org/abs/1512.03385 and https://arxiv.org/pdf/1512.03385 -> v1 PDF)
- Bibliographic data (as shown on the arXiv page): Title "Deep Residual Learning for Image Recognition". Authors: Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun (4). Submitted 10 Dec 2015 (v1 only). Comments: "Tech report". No journal ref on arXiv (the paper was published at CVPR 2016 but the arXiv page does not state that); DOI cell: https://doi.org/10.48550/arXiv.1512.03385 (arXiv-issued). Subjects: cs.CV. arXiv year 2015 (CVPR 2016 if cited as the conference version).

- Claim: identity shortcut connections ease optimization of deep networks.
  - Quote (abstract): "We present a residual learning framework to ease the training of networks that are substantially deeper than those used previously." / "We provide comprehensive empirical evidence showing that these residual networks are easier to optimize, and can gain accuracy from considerably increased depth."
  - Quote (Sec. 1): "In our case, the shortcut connections simply perform identity mapping, and their outputs are added to the outputs of the stacked layers (Fig. 2). Identity shortcut connections add neither extra parameter nor computational complexity."
  - Quote (Sec. 1): "We hypothesize that it is easier to optimize the residual mapping than to optimize the original, unreferenced mapping. To the extreme, if an identity mapping were optimal, it would be easier to push the residual to zero than to fit an identity mapping by a stack of nonlinear layers." and "We show that: 1) Our extremely deep residual nets are easy to optimize, but the counterpart "plain" nets (that simply stack layers) exhibit higher training error when the depth increases"
  - Verdict: SUPPORTED.

## 8. key=huang2017densenet

- Fetched: YES (https://arxiv.org/abs/1608.06993)
- Bibliographic data (as shown on the arXiv page): Title "Densely Connected Convolutional Networks". Authors: Gao Huang, Zhuang Liu, Laurens van der Maaten, Kilian Q. Weinberger (4). Submitted 25 Aug 2016 (v1), "last revised 28 Jan 2018 (this version, v5)". Comments: "CVPR 2017". No journal ref; DOI cell: https://doi.org/10.48550/arXiv.1608.06993 (arXiv-issued). Subjects: cs.CV; cs.LG. Year: 2017 (CVPR) / arXiv 2016.

- Claim: each layer receives the feature maps of all preceding layers as input.
  - Quote (abstract): "In this paper, we embrace this observation and introduce the Dense Convolutional Network (DenseNet), which connects each layer to every other layer in a feed-forward fashion. Whereas traditional convolutional networks with L layers have L connections - one between each layer and its subsequent layer - our network has L(L+1)/2 direct connections. For each layer, the feature-maps of all preceding layers are used as inputs, and its own feature-maps are used as inputs into all subsequent layers."
  - Verdict: SUPPORTED. (Note for a citing text: DenseNet concatenates the preceding feature maps rather than summing them; the abstract says "used as inputs".)

## 9. key=hoffmann2022chinchilla

- Fetched: YES (https://arxiv.org/abs/2203.15556 and https://arxiv.org/pdf/2203.15556 -> v1 PDF)
- Bibliographic data (as shown on the arXiv page): Title "Training Compute-Optimal Large Language Models". Authors (22): Jordan Hoffmann, Sebastian Borgeaud, Arthur Mensch, Elena Buchatskaya, Trevor Cai, Eliza Rutherford, ... Laurent Sifre (22 authors in total). Submitted 29 Mar 2022 (v1 only). No Comments field, no journal ref on arXiv (published at NeurIPS 2022, not stated on the arXiv page); DOI cell: https://doi.org/10.48550/arXiv.2203.15556 (arXiv-issued). Subjects: cs.CL; cs.LG. Year 2022.

- Claim (a) for compute-optimal training, parameters and tokens should be scaled equally.
  - Quote (abstract): "By training over 400 language models ranging from 70 million to over 16 billion parameters on 5 to 500 billion tokens, we find that for compute-optimal training, the model size and the number of training tokens should be scaled equally: for every doubling of model size the number of training tokens should also be doubled."
  - Quote (Sec. 3.4): "All three approaches suggest that as compute budget increases, model size and the amount of training data should be increased in approximately equal proportions."
  - Quote (Table 2 caption): "Our analysis suggests a near equal scaling in parameters and data with increasing compute which is in clear contrast to previous work on the scaling of large models." (fitted exponents a = 0.50/0.49/0.46 and b = 0.50/0.51/0.54 for the three approaches)
  - Verdict: SUPPORTED.

- Claim (b) roughly 20 tokens per parameter.
  - The paper does not state a "tokens per parameter" ratio in words (grep for "tokens per parameter" / "20 tokens" / "per parameter" returns nothing). What it states: Table 3 ("Estimated optimal training FLOPs and training tokens for various model sizes ... projections from Approach 1"): "400 Million ... 8.0 Billion", "1 Billion ... 20.2 Billion", "10 Billion ... 205.1 Billion", "67 Billion ... 1.5 Trillion", "175 Billion ... 3.7 Trillion", "280 Billion ... 5.9 Trillion", "1 Trillion ... 21.2 Trillion".
  - Quote (Sec. 1): "Based on our estimated compute-optimal frontier, we predict that for the compute budget used to train Gopher, an optimal model should be 4 times smaller, while being training on 4 times more tokens. We verify this by training a more compute-optimal 70B model, called Chinchilla, on 1.4 trillion tokens."
  - Quote (Sec. 3.4, text after Table 3): "For example, we find that a 175 billion parameter model should be trained with a compute budget of 4.41 x 10^24 FLOPs and on over 4.2 trillion tokens. A 280 billion Gopher-like model is the optimal model to train given a compute budget of approximately 10^25 FLOPs and should be trained on 6.8 trillion tokens."
  - Verdict: PARTIALLY SUPPORTED. The ~20 tokens/parameter figure is a reading of Table 3 (20.2B tokens for 1B params, 205.1B for 10B, 21.2T for 1T; Chinchilla itself 70B/1.4T = 20) and is widely quoted, but the paper never states the ratio "20 tokens per parameter"; cite it as implied by Table 3 / Approach 1, and note the paper's own text examples give somewhat higher ratios (175B -> 4.2T, 280B -> 6.8T, i.e. ~24 tokens/param).

- Claim (c) the parametric loss L(N,D) = E + A/N^alpha + B/D^beta.
  - Quote (Sec. 3.3 "Approach 3: Fitting a parametric loss function"): "Lastly, we model all final losses from experiments in Approach 1 & 2 as a parametric function of model parameter count and the number of seen tokens. Following a classical risk decomposition (see Section D.2), we propose the following functional form" [Eq. (2): L-hat(N, D) ≜ E + A/N^alpha + B/D^beta] "The first term captures the loss for an ideal generative process on the data distribution, and should correspond to the entropy of natural text. The second term captures the fact that a perfectly trained transformer with N parameters underperforms the ideal generative process. The final term captures the fact that the transformer is not trained to convergence, as we only make a finite number of optimisation steps, on a sample of the dataset distribution."
  - Also (Sec. 3.3, Efficient frontier): "We can approximate the functions N_opt and D_opt by minimizing the parametric loss L-hat under the constraint FLOPs(N, D) ≈ 6ND (Kaplan et al., 2020)."
  - Verdict: SUPPORTED (Eq. 2 of the paper; fitted via Huber loss, Eq. 3).

## 10. key=kaplan2020scaling

- Fetched: YES (https://arxiv.org/abs/2001.08361 and https://arxiv.org/pdf/2001.08361 -> v1 PDF)
- Bibliographic data (as shown on the arXiv page): Title "Scaling Laws for Neural Language Models". Authors: Jared Kaplan, Sam McCandlish, Tom Henighan, Tom B. Brown, Benjamin Chess, Rewon Child, Scott Gray, Alec Radford, Jeffrey Wu, Dario Amodei (10). Submitted 23 Jan 2020 (v1 only). Comments: "19 pages, 15 figures". No journal ref; DOI cell: https://doi.org/10.48550/arXiv.2001.08361 (arXiv-issued). Subjects: cs.LG; stat.ML. Year 2020.

- Claim: power laws in N, D, C.
  - Quote (abstract): "The loss scales as a power-law with model size, dataset size, and the amount of compute used for training, with some trends spanning more than seven orders of magnitude."
  - Quote (Sec. 1.1): "Smooth power laws: Performance has a power-law relationship with each of the three scale factors N, D, C when not bottlenecked by the other two, with trends spanning more than six orders of magnitude (see Figure 1)."
  - Quote (Sec. 1.2): "The test loss of a Transformer trained to autoregressively model language can be predicted using a power-law when performance is limited by only either the number of non-embedding parameters N, the dataset size D, or the optimally allocated compute budget C_min (see Figure 1):" [Eq. 1.1: L(N) = (N_c/N)^{alpha_N}, alpha_N ~ 0.076; Eq. 1.2: L(D) = (D_c/D)^{alpha_D}, alpha_D ~ 0.095; Eq. 1.3: L(C_min) = (C_c^min/C_min)^{alpha_C^min}, alpha_C^min ~ 0.050]
  - Verdict: SUPPORTED. (Note: N is non-embedding parameters and the compute law is stated for C_min, the optimally allocated compute.)

- Claim: the estimate C ≈ 6ND of training compute.
  - Quote (Sec. 1 notation): "C ≈ 6NBS - an estimate of the total non-embedding training compute, where B is the batch size, and S is the number of training steps (ie parameter updates)."
  - Quote (Sec. 2.1): "Accounting for the backwards pass (approximately twice the compute as the forwards pass), we then define the estimated non-embedding compute as C ≈ 6N floating point operators per training token."
  - Quote (Sec. 3.3 / p. 9): "The total amount of non-embedding compute used during training can be estimated as C = 6NBS, where B is the batch size, S is the number of parameter updates, and the factor of 6 accounts for the forward and backward passes."
  - Verdict: SUPPORTED. The paper writes the estimate as C ≈ 6N per training token, i.e. C ≈ 6NBS with BS the number of tokens processed; "6ND" is the same quantity with D = BS tokens (the Chinchilla paper cites it as "FLOPs(N, D) ≈ 6ND (Kaplan et al., 2020)"), though Kaplan et al. do not literally write "6ND".

## 11. key=olmo2

- Fetched: YES (https://arxiv.org/abs/2501.00656 and https://arxiv.org/html/2501.00656 -> v3)
- Bibliographic data (as shown on the arXiv page): Title "2 OLMo 2 Furious". Authors (43, first listed as "OLMo, Team"): Team OLMo, Pete Walsh, Luca Soldaini, Dirk Groeneveld, Kyle Lo, Shane Arora, ... Hannaneh Hajishirzi (43 names in total; affiliations Allen Institute for AI, University of Washington, New York University). Submitted 31 Dec 2024 (v1), "last revised 8 Oct 2025 (this version, v3)"; header "arXiv:2501.00656v3 [cs.CL] 08 Oct 2025". Comments: "Shorter version accepted to COLM 2025. Updated to include 32B results. Model demo available at this http URL". No journal ref; DOI cell: https://doi.org/10.48550/arXiv.2501.00656 (arXiv-issued). Subjects: cs.CL; cs.LG. Year 2024 (v1) / 2025 (COLM, v3).

- Claim (a) OLMo 2 applies RMSNorm to the outputs of the attention and feed-forward sub-layers (reordered/post-norm inside the residual) and uses QK-norm.
  - Quote (Sec. 2.1): "RMSNorm: We use the RMSNorm (Zhang and Sennrich, 2019) variant of LayerNorm (Ba et al., 2016) without a bias term to normalize activations, instead of nonparametric LayerNorm."
  - Quote (Sec. 2.1): "Reordered norm: We normalize the outputs to the attention and feedforward (MLP) layers within each transformer block, instead of the inputs. So the formula for each block becomes: h := x + RMSNorm(Attention(x)) (1)  h_out := h + RMSNorm(MLP(x)) (2) where x is the input to the layer, h is an intermediate hidden state, and h_out is the output. This strategy was first proposed by Liu et al. (2021) to stabilize training." [Note: Eq. (2) as printed in the HTML has MLP(x); the table in Sec. 3.3.2 writes the same block as "h_out := h + RMSNorm(MLP(h))".]
  - Quote (Sec. 2.1): "QK-norm: Following Dehghani et al. (2023b) we normalize the key and query projections with RMSNorm before calculating attention. This avoids attention logits being too large, which can lead to training loss divergence."
  - Quote (Sec. 3.3.2): "Figure 7 shows the effect of applying the layer normalization to the outputs of the MLP and attention blocks instead of the inputs. We further apply another normalization, also RMSNorm, to the queries and keys in the attention block. In isolation, neither of these changes yield good results, but together they improve both the growth and the spikiness of the L2 norm of the gradient."
  - Verdict: SUPPORTED. (Table 1 row: "Layer Norm Applied to: Inputs / Inputs / Outputs" for OLMo 1 / OLMo-0424 / OLMo 2; the residual stream itself remains un-normalized, i.e. the norm is applied to the sub-layer output before the residual add.)

- Claim (b) the pretraining data (name the mix) is Dolma-derived.
  - Quote (Sec. 2.4.1): "The mix used for this stage is shown in Table 4. It consists of approximately 3.9 trillion tokens, with over 95% derived from web data. We refer to this set as OLMo 2 Mix 1124. This is the same pretraining data used in OLMoE (Muennighoff et al., 2024): We combine data from DCLM (Li et al., 2024) and Dolma 1.7 (Soldaini et al., 2024). From DCLM, we use the "baseline 1.0" mix. From Dolma, we use the arXiv (Together AI, 2023), OpenWebMath (Paster et al., 2023), Algebraic Stack, peS2o (Soldaini and Lo, 2023), and Wikipedia subsets."
  - Quote (Table 4 caption): "Composition of the pretraining data for OLMo 2. The OLMo 2 1124 Mix is composed of StarCoder (Li et al., 2023b; Kocetkov et al., 2022), peS2o (Soldaini and Lo, 2023), web text from DCLM (Li et al., 2024) and Wiki come from Dolma 1.7 (Soldaini et al., 2024)."
  - Quote (abstract): "Our updated pretraining data mixture introduces a new, specialized data mix called Dolmino Mix 1124, which significantly improves model capabilities across many downstream task benchmarks when introduced via late-stage curriculum training (i.e. specialized data during the annealing phase of pretraining)."
  - Verdict: PARTIALLY SUPPORTED. The stage-1 mix is "OLMo 2 Mix 1124" (3.9T tokens); it is only partly Dolma-derived: the bulk (3.71T of 3.90T tokens, Table 4) is DCLM-Baseline web text, with peS2o, Wikipedia/Wikibooks (and FLAN in the mid-training "Dolmino Mix 1124") taken from Dolma 1.7, plus StarCoder, arXiv (RedPajama), OpenWebMath and Algebraic Stack (ProofPile II). "Dolma-derived" is therefore only accurate for the non-web subsets; the citing text should say the mix combines DCLM-Baseline with Dolma 1.7 subsets, and the mid-training mix is Dolmino Mix 1124.

- Claim (c) released sizes.
  - Quote (abstract): "OLMo 2 includes a family of dense autoregressive language models at 7B, 13B and 32B scales with fully released artifacts -- model weights, full training data, training code and recipes, training logs and thousands of intermediate checkpoints."
  - Quote (Sec. 1): "In this technical report, we introduce OLMo 2, a new family of 7B, 13B and 32B models trained on up to 6T tokens."
  - Quote (Appendix B): "We pretrain OLMo 2 1B to 4 trillion tokens on OLMo 2 Mix 1124 and perform a single 50B token anneal on Dolmino Mix 1124." (Table 7: "The results for OLMo 2 Instruct at 1B, 7B, 13B, and 32B relative to peer open weight models.")
  - Verdict: SUPPORTED for 7B/13B/32B as the main released sizes (v3 of the paper; v1 covered 7B and 13B only, per the Comments "Updated to include 32B results"); a 1B model is additionally described in Appendix B. Base checkpoints: "Pretrain checkpoints have been trained on 4 trillion (1B, 7B), 5 trillion (13B) and 7 trillion (32B) tokens respectively."

## 12. key=soldaini2024dolma

- Fetched: YES (https://arxiv.org/abs/2402.00159)
- Bibliographic data (as shown on the arXiv page): Title "Dolma: an Open Corpus of Three Trillion Tokens for Language Model Pretraining Research". Authors (36): Luca Soldaini, Rodney Kinney, Akshita Bhagia, Dustin Schwenk, David Atkinson, Russell Authur, ... Kyle Lo (36 authors in total). Submitted 31 Jan 2024 (v1), "last revised 6 Jun 2024 (this version, v2)". Comments: "Accepted at ACL 2024; Dataset: this https URL Code: this https URL". No journal ref; DOI cell: https://doi.org/10.48550/arXiv.2402.00159 (arXiv-issued). Subjects: cs.CL. Year 2024.

- Claim: an open 3T-token corpus from web, code, books, papers, Wikipedia etc.
  - Quote (abstract): "To facilitate scientific research on language model pretraining, we curate and release Dolma, a three-trillion-token English corpus, built from a diverse mixture of web content, scientific papers, code, public-domain books, social media, and encyclopedic materials."
  - Quote (abstract): "Finally, we open-source our data curation toolkit to enable reproduction of our work as well as support further research in large-scale data curation."
  - Verdict: SUPPORTED.

## 13. key=raposo2024mod

- Fetched: YES (https://arxiv.org/abs/2404.02258 and https://arxiv.org/html/2404.02258 -> v1)
- Bibliographic data (as shown on the arXiv page): Title "Mixture-of-Depths: Dynamically allocating compute in transformer-based language models". Authors: David Raposo, Sam Ritter, Blake Richards, Timothy Lillicrap, Peter Conway Humphreys, Adam Santoro (6). Submitted 2 Apr 2024 (v1 only); header "arXiv:2404.02258v1 [cs.LG] 02 Apr 2024". No Comments field, no journal ref; DOI cell: https://doi.org/10.48550/arXiv.2404.02258 (arXiv-issued). Subjects: cs.LG; cs.CL. Year 2024.

- Claim: per-block routers decide per token whether the block processes it, using the token's current representation.
  - Quote (abstract): "Our method enforces a total compute budget by capping the number of tokens (k) that can participate in the self-attention and MLP computations at a given layer. The tokens to be processed are determined by the network using a top-k routing mechanism."
  - Quote (Sec. 3.4 Routing implementation): "As a reminder of the high-level intuition, each token is processed by a router to produce a scalar weight, and the top-k weights are then used to choose the token identities that will route through a transformer's block, which comprises self-attention and the subsequent MLP."
  - Quote (Sec. 3.4): "Suppose we have the set of token embeddings in a sequence of length S for a given layer l; that is X^l = {x_i^l | i is an integer, 1 <= i <= S}. The router weight for a given token embedding is a scalar produced as a result of a linear projection, r_i^l = w_theta^T x_i^l." [Eq. 1: x_i^{l+1} = r_i^l f_i(X~^l) + x_i^l if r_i^l > P_beta(R^l); x_i^l otherwise]
  - Verdict: SUPPORTED. Each layer l has its own linear router w_theta applied to that layer's token embedding x_i^l; the top-k (expert-choice) tokens go through attention+MLP, the rest take the residual path. (Sec. 3.2: "we route tokens to one of two computational paths: (1) self-attention and MLP blocks, and (2) a residual connection.")

## 14. key=fedus2022switch and key=shazeer2017moe

- Fetched: YES (https://arxiv.org/abs/2101.03961 + https://arxiv.org/pdf/2101.03961 -> v3 PDF; https://arxiv.org/abs/1701.06538 + https://arxiv.org/pdf/1701.06538 -> v1 PDF)
- Bibliographic data, fedus2022switch (as shown on the arXiv page): Title "Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity". Authors: William Fedus, Barret Zoph, Noam Shazeer (3). Submitted 11 Jan 2021 (v1), "last revised 16 Jun 2022 (this version, v3)". Comments: "JMLR". No journal ref field; DOI cell: https://doi.org/10.48550/arXiv.2101.03961 (arXiv-issued). Subjects: cs.LG; cs.AI. Year 2021 (arXiv) / 2022 (JMLR).
- Bibliographic data, shazeer2017moe (as shown on the arXiv page): Title "Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer". Authors: Noam Shazeer, Azalia Mirhoseini, Krzysztof Maziarz, Andy Davis, Quoc Le, Geoffrey Hinton, Jeff Dean (7). Submitted 23 Jan 2017 (v1 only). No Comments field, no journal ref on arXiv (ICLR 2017 not stated on the page); DOI cell: https://doi.org/10.48550/arXiv.1701.06538 (arXiv-issued). Subjects: cs.LG; cs.CL; cs.NE; stat.ML. Year 2017.

- Claim: MoE uses a learned per-layer gating network routing each token based on its current representation.
  - Quote (Shazeer et al. 2017, abstract): "We introduce a Sparsely-Gated Mixture-of-Experts layer (MoE), consisting of up to thousands of feed-forward sub-networks. A trainable gating network determines a sparse combination of these experts to use for each example."
  - Quote (Shazeer et al. 2017, Sec. 2): "The Mixture-of-Experts (MoE) layer consists of a set of n "expert networks" E_1, ..., E_n, and a "gating network" G whose output is a sparse n-dimensional vector." / (Sec. 2.1 Gating Network): "Softmax Gating: A simple choice of non-sparse gating function (Jordan & Jacobs, 1994) is to multiply the input by a trainable weight matrix W_g and then apply the Softmax function. G_sigma(x) = Softmax(x · W_g) (2)" and "Training the Gating Network We train the gating network by simple back-propagation, along with the rest of the model."
  - Quote (Fedus et al., Sec. 2.1 "Mixture of Expert Routing"): "Shazeer et al. (2017) proposed a natural language Mixture-of-Experts (MoE) layer which takes as an input a token representation x and then routes this to the best determined top-k experts, selected from a set {E_i(x)}_{i=1}^N of N experts. The router variable W_r produces logits h(x) = W_r · x which are normalized via a softmax distribution over the available N experts at that layer."
  - Quote (Fedus et al., Sec. 2.1 "Switch Routing"): "Contrary to these ideas, we instead use a simplified strategy where we route to only a single expert. We show this simplification preserves model quality, reduces routing computation and performs better. This k = 1 routing strategy is later referred to as a Switch layer."
  - Verdict: SUPPORTED. In both papers the gate/router is a trainable linear map (W_g / W_r) of the layer's input representation x, learned by backpropagation, with its own parameters "at that layer"; Switch routes each token to a single expert (k = 1). Note Shazeer et al. 2017 describe routing "for each example" and apply MoE between LSTM layers (abstract: "a MoE with up to 137 billion parameters is applied convolutionally between stacked LSTM layers"), with the gating applied per timestep; the token-level Transformer phrasing comes from Fedus et al.

## 15. key=menghani2024laurel

- Fetched: YES (https://arxiv.org/abs/2411.07501 and https://arxiv.org/html/2411.07501 -> v4)
- Bibliographic data (as shown on the arXiv page): Title "LAuReL: Learned Augmented Residual Layer". Authors: Gaurav Menghani, Ravi Kumar, Sanjiv Kumar (3). Submitted 12 Nov 2024 (v1), "last revised 24 Jun 2025 (this version, v4)"; header "arXiv:2411.07501v4 [cs.LG] 24 Jun 2025". Comments: "Accepted at 42nd International Conference on Machine Learning (2025), Vancouver, Canada". No journal ref; DOI cell: https://doi.org/10.48550/arXiv.2411.07501 (arXiv-issued). Subjects: cs.LG; cs.AI; cs.CV. Year 2024 (arXiv) / 2025 (ICML).

- Claim: generalizes the residual connection with learned combinations of previous layer outputs at low parameter overhead.
  - Quote (abstract): "In this paper we introduce Learned Augmented Residual Layer (LAuReL) -- a novel generalization of the canonical residual connection -- with the goal to be an in-situ replacement of the latter while outperforming on both model quality and footprint metrics."
  - Quote (Sec. 2): "In its most general form, we reformulate the residual connection to be the following: x_{i+1} = alpha · f(x_i) + g(x_i, x_{i-1}, ..., x_0). (2) Here, alpha is a learned scalar parameter, and g(·) is a learned linear function with x_i, x_{i-1}, ..., x_0 as inputs, where x_j is the output of the j-th residual connection."
  - Quote (Sec. 2.3, LAuReL-PA): "This is similar to LAuReL-LR, except that we use k activations from the previous blocks. In particular, we set g(x_i, ..., x_0) = x_i + sum_{j=0}^{k-1} gamma_{i,j} · h_i(x_{i-j}), where gamma_{i,0}, ..., gamma_{i,k-1} are learned scalar parameters and h_i is another linear function." / "When using a rank r product for h_i, the number of new parameters per LAuReL layer is 2rD + k, where k is the number of previous activations used."
  - Quote (abstract, overhead): "For example, on the ResNet-50, ImageNet 1K task, it achieves 60% of the gains from adding an extra layer, while only adding 0.003% more parameters, and matches it while adding 2.6 times fewer parameters. Similarly, when pre-training 1B and 4B parameter LLMs, LAuReL improves performance on a variety of challenging downstream evaluation tasks by 2.54% to 20.05%, while adding only 0.012% and 0.1% additional parameters, respectively."
  - Verdict: SUPPORTED. Note that the combination weights (alpha, beta, gamma_{i,j}) and the low-rank maps are static learned parameters, not input-dependent; the "previous activations" variant (LAuReL-PA) is the one that mixes earlier block outputs, while LAuReL-RW and LAuReL-LR only reweight / linearly transform the current residual input x_i.

## 16. key=srivastava2015highway

- Fetched: YES (https://arxiv.org/abs/1505.00387 and https://arxiv.org/pdf/1505.00387 -> v2 PDF)
- Bibliographic data (as shown on the arXiv page): Title "Highway Networks". Authors: Rupesh Kumar Srivastava, Klaus Greff, Jürgen Schmidhuber (3). Submitted 3 May 2015 (v1), "last revised 3 Nov 2015 (this version, v2)". Comments: "6 pages, 2 figures. Presented at ICML 2015 Deep Learning workshop. Full paper is at arXiv:1507.06228". No journal ref; DOI cell: https://doi.org/10.48550/arXiv.1505.00387 (arXiv-issued). Subjects: cs.LG; cs.NE. Year 2015.

- Claim: input-dependent gated skip connections.
  - Quote (abstract): "The architecture is characterized by the use of gating units which learn to regulate the flow of information through a network."
  - Quote (Sec. 2): "For a highway network, we additionally define two non-linear transforms T(x, W_T) and C(x, W_C) such that y = H(x, W_H) · T(x, W_T) + x · C(x, W_C). (2) We refer to T as the transform gate and C as the carry gate, since they express how much of the output is produced by transforming the input and carrying it, respectively. For simplicity, in this paper we set C = 1 - T, giving y = H(x, W_H) · T(x, W_T) + x · (1 - T(x, W_T)). (3)"
  - Quote (Sec. 2): "Thus, depending on the output of the transform gates, a highway layer can smoothly vary its behavior between that of a plain layer and that of a layer which simply passes its inputs through."
  - Verdict: SUPPORTED. The carry/transform gates are functions T(x, W_T) of the layer input x, i.e. input-dependent gating of the skip path.

## 17. key=bapna2018transparent

- Fetched: YES (https://arxiv.org/abs/1808.07561 and https://arxiv.org/pdf/1808.07561 -> v2 PDF)
- Bibliographic data (as shown on the arXiv page): Title "Training Deeper Neural Machine Translation Models with Transparent Attention". Authors: Ankur Bapna, Mia Xu Chen, Orhan Firat, Yuan Cao, Yonghui Wu (5). Submitted 22 Aug 2018 (v1), "last revised 4 Sep 2018 (this version, v2)". Comments: "To appear in EMNLP 2018". No journal ref; DOI cell: https://doi.org/10.48550/arXiv.1808.07561 (arXiv-issued). Subjects: cs.CL; cs.AI; cs.LG. Year 2018.

- Claim: decoder attends over a learned static weighted combination of all encoder layers' outputs.
  - Quote (Sec. 2, Transparent Attention): "To improve gradient flow we let the decoder attend weighted combinations of all encoder layer outputs, instead of just the top encoder layer."
  - Quote (Sec. 2): "In transparent attention we evaluate M weighted combinations of the encoder outputs, one corresponding to each attention module. We define a (N + 1) x M weight vector W, which is learned during training. We apply dropout to W since we empirically found it helpful to stabilize training. We then compute softmax s to normalize the weights. s_{i,j} = e^{W_{i,j}} / sum_{k=0}^{N} e^{W_{k,j}}, j = 1 ... M (1) We now define z_t^j = sum_{i=1}^{N+1} s_{i,j} h_t^i, t = 1 ... T, j = 1 ... M (2) Now attention module j attends to {z_t^j | t = 1 ... T}."
  - Quote (Sec. 1): "Using trainable weights, this 'transparent' attention allows the model the flexibility to adjust the gradient flow to different layers in the encoder depending on its training phase."
  - Verdict: SUPPORTED. The weights W are learned parameters (one softmax-normalized vector over the N encoder layers plus the embedding layer, per decoder attention module), not functions of the input, i.e. static; one combination per attention module j.

## 18. key=peters2018elmo

- Fetched: YES (https://arxiv.org/abs/1802.05365 and https://arxiv.org/pdf/1802.05365 -> v2 PDF)
- Bibliographic data (as shown on the arXiv page): Title "Deep contextualized word representations". Authors: Matthew E. Peters, Mark Neumann, Mohit Iyyer, Matt Gardner, Christopher Clark, Kenton Lee, Luke Zettlemoyer (7). Submitted 15 Feb 2018 (v1), "last revised 22 Mar 2018 (this version, v2)". Comments: "NAACL 2018. Originally posted to openreview 27 Oct 2017. v2 updated for NAACL camera ready". No journal ref; DOI cell: https://doi.org/10.48550/arXiv.1802.05365 (arXiv-issued). Subjects: cs.CL. Year 2018.

- Claim: a task-specific learned scalar mixture over all biLM layers.
  - Quote (Sec. 3.2): "ELMo is a task specific combination of the intermediate layer representations in the biLM."
  - Quote (Sec. 3.2): "More generally, we compute a task specific weighting of all biLM layers: ELMo_k^task = E(R_k; Theta^task) = gamma^task sum_{j=0}^{L} s_j^task h_{k,j}^LM. (1) In (1), s^task are softmax-normalized weights and the scalar parameter gamma^task allows the task model to scale the entire ELMo vector."
  - Quote (Sec. 3.3): "Then, we let the end task model learn a linear combination of these representations, as described below."
  - Verdict: SUPPORTED. The mixture weights s^task_j (softmax-normalized scalars, one per biLM layer including the token layer j = 0) and the scale gamma^task are learned per task; they are static (not input-dependent).

---

## Summary table

| key | fetched? | overall verdict |
|---|---|---|
| zhu2024hyperconnections | yes (abs + HTML v2) | (a) SUPPORTED, (b) SUPPORTED, (c) SUPPORTED (1.8x convergence, +6 ARC-C on OLMoE-1B-7B; 1B/7B dense loss -0.02..-0.03). No instability reported for HC (claims fewer spikes); no constraint on mixing matrices, only tanh/norm on the dynamic branch and Pre-Norm-equivalent init |
| mhc2026 | yes (abs + HTML v2; id 2512.24880 correct) | (a) SUPPORTED, (b) SUPPORTED, (c) SUPPORTED (per-layer linear projection of the flattened RMS-normed hidden state + static bias; sigmoid for pre/post, Sinkhorn-Knopp for res) |
| muddformer2025 | yes (abs + HTML v2) | (a)-(d) SUPPORTED; (e) SUPPORTED with wording caveat (scalar per source layer per position per stream; paper calls the 4 Q/K/V/R streams "multi(4)-head") |
| pagliardini2024denseformer | yes (abs + HTML v2) | (a) SUPPORTED (static scalar alpha_{i,j}), (b) SUPPORTED |
| elhage2021 | yes (transformer-circuits.pub) | (a) SUPPORTED, (b) SUPPORTED ("residual stream bandwidth to be in very high demand", "bottleneck activations") |
| vaswani2017 | yes (abs + PDF v7) | SUPPORTED (Sec. 3.1 LayerNorm(x + Sublayer(x))) |
| he2016resnet | yes (abs + PDF v1) | SUPPORTED |
| huang2017densenet | yes (abs) | SUPPORTED |
| hoffmann2022chinchilla | yes (abs + PDF v1) | (a) SUPPORTED, (b) PARTIALLY SUPPORTED (ratio ~20 only implied by Table 3 / Chinchilla 70B-1.4T; never stated as a ratio), (c) SUPPORTED (Eq. 2) |
| kaplan2020scaling | yes (abs + PDF v1) | power laws SUPPORTED; C≈6ND SUPPORTED as C≈6N per token / C≈6NBS (not literally "6ND") |
| olmo2 | yes (abs + HTML v3) | (a) SUPPORTED, (b) PARTIALLY SUPPORTED (OLMo 2 Mix 1124 = DCLM-Baseline web 95% + Dolma 1.7 subsets; mid-training Dolmino Mix 1124), (c) SUPPORTED (7B/13B/32B; 1B in Appendix B) |
| soldaini2024dolma | yes (abs) | SUPPORTED |
| raposo2024mod | yes (abs + HTML v1) | SUPPORTED |
| fedus2022switch / shazeer2017moe | yes (abs + PDFs) | SUPPORTED |
| menghani2024laurel | yes (abs + HTML v4) | SUPPORTED (weights static; previous-layer mixing is the LAuReL-PA variant) |
| srivastava2015highway | yes (abs + PDF v2) | SUPPORTED |
| bapna2018transparent | yes (abs + PDF v2) | SUPPORTED |
| peters2018elmo | yes (abs + PDF v2) | SUPPORTED |
