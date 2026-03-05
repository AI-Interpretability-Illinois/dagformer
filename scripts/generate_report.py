"""Generate DAGFormer progress report PDF."""
from __future__ import annotations

import os

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    BaseDocTemplate, Frame, Image, NextPageTemplate, PageBreak, PageTemplate,
    Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle,
)

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG = os.path.join(BASE, "experiments", "figures")
OUT = os.path.join(BASE, "experiments", "dagformer_progress_report.pdf")

# ── Styles ──────────────────────────────────────────────────────────────
styles = getSampleStyleSheet()

title_style = ParagraphStyle(
    "Title2", parent=styles["Title"], fontSize=22, spaceAfter=4, leading=26,
)
subtitle_style = ParagraphStyle(
    "Subtitle", parent=styles["Normal"], fontSize=12, textColor=colors.grey,
    alignment=TA_CENTER, spaceAfter=20,
)
h1 = ParagraphStyle(
    "H1", parent=styles["Heading1"], fontSize=16, spaceAfter=8, spaceBefore=16,
    textColor=colors.HexColor("#1a3a6a"),
)
h2 = ParagraphStyle(
    "H2", parent=styles["Heading2"], fontSize=13, spaceAfter=6, spaceBefore=12,
    textColor=colors.HexColor("#2a5a3a"),
)
h3 = ParagraphStyle(
    "H3", parent=styles["Heading3"], fontSize=11, spaceAfter=4, spaceBefore=8,
    textColor=colors.HexColor("#4a4a6a"),
)
body = ParagraphStyle(
    "Body2", parent=styles["Normal"], fontSize=10, leading=14,
    alignment=TA_JUSTIFY, spaceAfter=6,
)
body_sm = ParagraphStyle(
    "BodySm", parent=body, fontSize=9, leading=12,
)
caption = ParagraphStyle(
    "Caption", parent=styles["Normal"], fontSize=9, textColor=colors.grey,
    alignment=TA_CENTER, spaceAfter=10, spaceBefore=2, italic=True,
)
bullet = ParagraphStyle(
    "Bullet", parent=body, leftIndent=20, bulletIndent=8,
    bulletFontName="Helvetica", bulletFontSize=10,
)

W, H = letter  # 612 × 792
CONTENT_W = W - 2 * 0.9 * inch


def fig(name: str, w: float = 6.5) -> list:
    """Return Image + caption as list of flowables."""
    path = os.path.join(FIG, name)
    if not os.path.exists(path):
        path = os.path.join(BASE, "experiments", name)
    img = Image(path, width=w * inch, height=w * 0.6 * inch)
    img.hAlign = "CENTER"
    return [img]


def table(data: list[list[str]], col_widths: list[float] | None = None,
          header: bool = True) -> Table:
    """Create a styled table."""
    t = Table(data, colWidths=col_widths, repeatRows=1 if header else 0)
    style_cmds = [
        ("FONTSIZE", (0, 0), (-1, -1), 8.5),
        ("LEADING", (0, 0), (-1, -1), 11),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#cccccc")),
        ("LEFTPADDING", (0, 0), (-1, -1), 4),
        ("RIGHTPADDING", (0, 0), (-1, -1), 4),
        ("TOPPADDING", (0, 0), (-1, -1), 2),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
    ]
    if header:
        style_cmds += [
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2a4a6a")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ]
        for row_idx in range(1, len(data)):
            if row_idx % 2 == 0:
                style_cmds.append(
                    ("BACKGROUND", (0, row_idx), (-1, row_idx), colors.HexColor("#f0f4f8"))
                )
    t.setStyle(TableStyle(style_cmds))
    t.hAlign = "CENTER"
    return t


# ── Build document ──────────────────────────────────────────────────────
doc = SimpleDocTemplate(OUT, pagesize=letter,
                        topMargin=0.7*inch, bottomMargin=0.6*inch,
                        leftMargin=0.9*inch, rightMargin=0.9*inch)

story: list = []

# ════════════════════════════════════════════════════════════
# TITLE PAGE
# ════════════════════════════════════════════════════════════
story.append(Spacer(1, 1.5 * inch))
story.append(Paragraph("DAGFormer: Learned Computation Graphs<br/>for Transformers", title_style))
story.append(Spacer(1, 0.1 * inch))
story.append(Paragraph("Progress Report &mdash; March 2026", subtitle_style))
story.append(Spacer(1, 0.5 * inch))
story.append(Paragraph(
    "This document summarizes our experiments on learning input-dependent "
    "computation graphs (DAGs) for transformer language models. We cover two main "
    "lines of work: <b>midtraining</b> on a pretrained OLMo-1B, and "
    "<b>pretraining from scratch</b> with a 300M-parameter model. "
    "We present results, analyze failure modes, and outline promising next steps.",
    body,
))
story.append(Spacer(1, 0.3 * inch))
story.append(Paragraph(
    "<b>Key finding:</b> Midtraining on pretrained models failed because A=1 (dense) "
    "is a flat optimum in the loss landscape. Pretraining from scratch shows that "
    "self-embed predictors <i>can</i> learn meaningful sparse topologies, but a 3.5x "
    "throughput overhead makes direct comparison with dense baselines unfair. "
    "We are currently testing staged, alternating, and improved predictor designs "
    "to resolve this tension.",
    body,
))
story.append(PageBreak())

# ════════════════════════════════════════════════════════════
# 1. INTRODUCTION
# ════════════════════════════════════════════════════════════
story.append(Paragraph("1. Introduction", h1))
story.append(Paragraph(
    "Standard transformers use a fixed sequential computation graph: "
    "layer 0 &rarr; layer 1 &rarr; ... &rarr; layer <i>L</i>. "
    "Every input sees the same wiring regardless of content. "
    "<b>DAGFormer</b> replaces this rigid structure with a learned, "
    "input-dependent directed acyclic graph (DAG) that controls which "
    "attention heads feed into which other heads across layers.",
    body,
))
story.append(Spacer(1, 0.05 * inch))
story.append(Paragraph(
    "The core modification: each attention head <i>j</i> in layer <i>l</i> "
    "receives a <b>gated combination</b> of prior heads' outputs, controlled by "
    "an adjacency matrix A &isin; [0,1]<super>N&times;N</super> "
    "(N = layers &times; heads). "
    "When A[i,j] = 1 for all valid cross-layer pairs, the behavior is "
    "identical to the standard residual stream (baseline reproduction). "
    "The matrix A is produced by a <b>structure predictor</b> that conditions "
    "on the input context.",
    body,
))

story.append(Paragraph("1.1 Oracle Search Motivation", h2))
story.append(Paragraph(
    "An oracle search (500 gradient steps per 1024-token window, directly optimizing A) "
    "showed that context-dependent topologies can reduce NLL from 2.58 to 0.12 on "
    "OLMo-1B (median across 50 windows, 100% improved). This demonstrates enormous "
    "headroom, but the per-window optimization is far too slow for practical use. "
    "The goal of DAGFormer is to <b>amortize</b> this search into a learned predictor.",
    body,
))

story.append(Paragraph("1.2 Architecture Overview", h2))
story.append(Paragraph(
    "The pipeline has two components: (1) a <b>structure predictor</b> that maps "
    "the input context to an adjacency matrix A via a low-rank parameterization "
    "(Z = UV<super>T</super> + bias &rarr; Gumbel-Sigmoid &rarr; cascading gate), and "
    "(2) a <b>modified transformer forward pass</b> that assembles per-head inputs "
    "according to A. Training is end-to-end: NLL gradients flow through the modified "
    "forward back to the predictor.",
    body,
))

# Architecture diagram
arch_path = os.path.join(FIG, "fig_architecture.png")
if not os.path.exists(arch_path):
    # Use the existing nll_comparison as placeholder, we'll note the HTML exists
    story.append(Paragraph(
        "<i>See <font face='Courier'>structure_predictor.html</font> for the full interactive architecture diagram.</i>",
        caption,
    ))
else:
    story += fig("fig_architecture.png", w=6.0)

story.append(PageBreak())

# ════════════════════════════════════════════════════════════
# 2. MIDTRAINING ON PRETRAINED OLMo-1B
# ════════════════════════════════════════════════════════════
story.append(Paragraph("2. Midtraining on Pretrained OLMo-1B", h1))
story.append(Paragraph(
    "Our first approach was to keep the pretrained OLMo-1B and learn topology on top "
    "of it. We explored two phases: <b>Phase 1</b> (frozen OLMo, only predictor trains) "
    "and <b>Phase 2</b> (unfrozen OLMo, joint training). Both ultimately failed.",
    body,
))

# 2.1 Phase 1
story.append(Paragraph("2.1 Phase 1: Frozen OLMo &mdash; A=1 Is a Flat Optimum", h2))
story.append(Paragraph(
    "With OLMo frozen, any deviation from dense connectivity (A=1) hurts performance. "
    "We tested multiple init_logit values that control how far from dense the "
    "predictor starts:",
    body,
))

story += fig("fig1_init_logit_ablation.png", w=5.5)
story.append(Paragraph(
    "<b>Figure 1.</b> Frozen OLMo-1B: lower init_logit &rarr; further from dense &rarr; worse NLL. "
    "S2 (init_logit=15) stays at baseline but sigmoid is saturated (no gradient). "
    "A12/A13/A14 have gradient flow but converge to fixed topologies worse than dense.",
    caption,
))

story.append(Paragraph(
    "To understand <i>why</i> A=1 is optimal, we computed &part;Loss/&part;A at A=1 "
    "across 50 eval windows:",
    body,
))

story += fig("fig2_gradient_at_A1.png", w=5.0)
story.append(Paragraph(
    "<b>Figure 2.</b> Gradient at A=1 is symmetric around zero (49.7% positive, 50.3% negative) "
    "with magnitude ~4&times;10<super>-5</super>. There is no consistent direction to move. "
    "The pretrained weights have made A=1 a flat plateau in the loss landscape.",
    caption,
))

story.append(PageBreak())

# 2.2 Phase 2
story.append(Paragraph("2.2 Phase 2: Unfrozen OLMo &mdash; The LR Dilemma", h2))
story.append(Paragraph(
    "Unfreezing OLMo allows the model to adapt to non-dense topologies. "
    "We ran 10 experiments varying OLMo LR, predictor LR, temperature schedule, "
    "initialization, and sparsity pressure.",
    body,
))

story += fig("fig3_midtrain_all_results.png", w=6.5)
story.append(Paragraph(
    "<b>Figure 3.</b> NLL relative to dense baseline for all midtraining experiments. "
    "Only P2b (-0.002) and D2 (-0.007) show any improvement, and both remain near-dense (mean_A > 0.97). "
    "Most experiments are substantially worse.",
    caption,
))
story.append(Spacer(1, 0.05 * inch))

story.append(Paragraph("The core dilemma:", h3))
story.append(Paragraph(
    "&bull; <b>High OLMo LR</b>: OLMo quickly adapts to dense via continual pretraining. "
    "The predictor has no incentive to deviate from A&asymp;1.<br/>"
    "&bull; <b>Low OLMo LR</b>: OLMo cannot adapt to non-dense topologies. "
    "Any deviation from A=1 destroys information the model expects.<br/>"
    "&bull; <b>High predictor LR</b>: Topology collapses to random (P2a: mean_A=0.5, NLL +0.23).<br/>"
    "&bull; <b>Starting sparse</b> (D1, D6): OLMo is damaged by missing connections it relies on.",
    body,
))

story.append(Paragraph("Best midtraining result: D2", h3))
story += fig("fig4_D2_trajectory.png", w=6.0)
story.append(Paragraph(
    "<b>Figure 4.</b> D2 (&#964;: 5&rarr;0.2 + sparsity ramp) achieved -0.007 NLL vs baseline, "
    "but topology stayed near-dense (mean_A &asymp; 0.97) and jaccard_var = 0 throughout "
    "(no context-dependent routing). The improvement is real (p &lt; 10<super>-6</super>, "
    "n=500 paired t-test) but tiny and context-independent.",
    caption,
))

story.append(PageBreak())

# 2.3 Summary table
story.append(Paragraph("2.3 Summary: All Midtraining Experiments", h2))

midtrain_data = [
    ["Exp", "OLMo LR", "Pred LR", "Strategy", "NLL-Base", "mean_A", "Jaccard"],
    ["P2-trial", "3e-5", "1e-4", "Default", "+0.001", "0.98", "0"],
    ["P2a", "3e-5", "1e-2", "High pred LR", "+0.228", "0.50", "0"],
    ["P2b", "5e-6", "1e-3", "Low OLMo LR", "-0.002", "0.99", "0"],
    ["P2c", "1e-5", "1e-3", "Med OLMo LR", "+0.003", "0.99", "0"],
    ["P2d", "3e-5", "1e-3", "Low tau+init", "+0.001", "0.99", "0"],
    ["P2e", "5e-6", "1e-3", "+Sparsity", "+0.008", "0.97", "0"],
    ["P2f", "5e-6", "1e-3", "init_logit=0", "+0.475", "0.56", "0"],
    ["D1", "5e-6", "1e-3", "A starts ~0.5", "+0.186", "0.71", "0"],
    ["D2", "5e-6", "1e-3", "tau anneal+lam", "-0.007", "0.97", "0"],
    ["D6", "5e-6", "1e-3", "A starts ~0", "+0.301", "0.56", "0"],
]
story.append(table(midtrain_data))
story.append(Paragraph(
    "<b>Table 1.</b> All midtraining experiments on OLMo-1B. jaccard_var=0 means the predictor "
    "produces identical topology for all inputs (no context-dependence). "
    "NLL-Base is eval/nll_soft minus the dense baseline at the same step.",
    caption,
))

story.append(Spacer(1, 0.15 * inch))
story.append(Paragraph("2.4 Conclusion: Why Midtraining Failed", h2))
story.append(Paragraph(
    "Pretrained transformer weights are optimized for dense (A=1) connectivity. "
    "The loss landscape at A=1 is a <b>flat plateau</b> with near-zero gradient, "
    "making gradient-based topology learning impossible without fundamentally "
    "restructuring the model. Joint training (Phase 2) creates a chicken-and-egg "
    "problem: the model adapts to dense faster than the predictor can propose "
    "useful alternatives. <b>We need to train from scratch</b> so that weights "
    "and topology co-evolve from random initialization.",
    body,
))

story.append(PageBreak())

# ════════════════════════════════════════════════════════════
# 3. PRETRAINING FROM SCRATCH — 300M
# ════════════════════════════════════════════════════════════
story.append(Paragraph("3. Pretraining from Scratch &mdash; 300M OLMo-2", h1))
story.append(Paragraph(
    "Given the failure of midtraining, we switched to pretraining a smaller model "
    "(300M parameters, OLMo-2 architecture: 12 layers &times; 16 heads) from random "
    "initialization with DAGFormer topology. The A matrix is 192&times;192. "
    "We first trained a <b>dense baseline</b> (no topology) for comparison, then "
    "trained DAGFormer variants with the same token budget (5.24B tokens).",
    body,
))

story.append(Paragraph("3.1 Dense Baseline", h2))
story.append(Paragraph(
    "Standard OLMo-2 300M trained for 10,000 steps (5.24B tokens) on Dolma v1.7. "
    "Throughput: 81K tokens/sec on 4&times;A40. "
    "Final eval NLL: <b>3.6094</b>. This is the target to beat.",
    body,
))

story.append(Paragraph("3.2 DAGFormer Variants (Batch 1: From Scratch)", h2))
story.append(Paragraph(
    "We tested three predictor types, all training from random init with the "
    "same 5.24B token budget:",
    body,
))

variant_data = [
    ["Variant", "Predictor", "tok/s", "Steps done", "Eval NLL\n(soft)", "Own\nBaseline", "mean_A"],
    ["Baseline", "None (dense)", "81K", "10000", "3.609", "-", "1.0"],
    ["Static", "Learnable UV^T\n(no input dep.)", "23K", "~4690", "~5.89", "~5.98", "0.98"],
    ["Self-embed", "Mean-pool\nembeddings", "23K", "~4670", "~5.75", "~5.96", "0.42"],
    ["Qwen", "Frozen Qwen-0.6B\n(external)", "14K", "~2920", "~6.15", "~6.28", "0.41"],
]
story.append(table(variant_data))
story.append(Paragraph(
    "<b>Table 2.</b> From-scratch pretraining results. All DAGFormer variants are far behind "
    "the baseline in absolute NLL because they see ~3.5x fewer tokens in the same wall time.",
    caption,
))
story.append(Spacer(1, 0.1 * inch))

story += fig("fig6_meanA_static_vs_selfembed.png", w=5.5)
story.append(Paragraph(
    "<b>Figure 5.</b> Topology density over training. "
    "Self-embed rapidly learns sparse routing (mean_A &rarr; 0.42), while Static stays dense. "
    "This is the <b>first successful topology learning</b> across all experiments.",
    caption,
))

story.append(PageBreak())

story += fig("nll_comparison_300m.png", w=6.0)
story.append(Paragraph(
    "<b>Figure 6.</b> Eval NLL comparison. Blue: dense baseline (standard training). "
    "Red: self-embed DAGFormer. Green: self-embed's own dense forward (A=1). "
    "Key insight: red &lt; green after step 3000, proving learned topology helps the "
    "same undertrained model. But both are far behind the blue baseline.",
    caption,
))

story += fig("fig7_throughput.png", w=5.0)
story.append(Paragraph(
    "<b>Figure 7.</b> Training throughput. DAGFormer's per-head input assembly (einsum over "
    "A matrix) is 3.5x slower than standard batched attention. "
    "Qwen adds another 1.6x from the external encoder. "
    "This is the root cause of the NLL gap: same wall time, 3.5x fewer tokens seen.",
    caption,
))

story.append(Spacer(1, 0.1 * inch))
story.append(Paragraph("Key Findings from Batch 1:", h3))
story.append(Paragraph(
    "&bull; <b>Self-embed learns sparse topology</b> (mean_A=0.42). This is the first experiment "
    "where the predictor meaningfully deviates from dense.<br/>"
    "&bull; <b>Learned topology helps</b>: soft NLL (5.75) &lt; own-baseline NLL (5.96), "
    "a 0.21 nat improvement on the same undertrained model.<br/>"
    "&bull; <b>Static predictor fails</b>: mean_A=0.98, predictor gradient ~0. Without "
    "input-dependent signal, the predictor cannot learn.<br/>"
    "&bull; <b>3.5x throughput overhead is the bottleneck</b>: the model sees far fewer tokens "
    "than the baseline, so the base model is severely undertrained.",
    body,
))

story.append(PageBreak())

# ════════════════════════════════════════════════════════════
# 4. CURRENT EXPERIMENTS
# ════════════════════════════════════════════════════════════
story.append(Paragraph("4. Current Experiments (In Progress)", h1))
story.append(Paragraph(
    "To address the throughput bottleneck, we are testing five new strategies. "
    "All are currently submitted to the cluster.",
    body,
))

story.append(Paragraph("4.1 Addressing Throughput: Staged &amp; Alternating Training", h2))

story.append(Paragraph("<b>Staged training</b>", h3))
story.append(Paragraph(
    "Load a fully-trained dense baseline checkpoint (10K steps), then switch to "
    "DAGFormer forward for an additional 2,000 steps. The base model starts with "
    "good language modeling ability; only needs to adapt to the new routing. "
    "Predictor starts fresh (from init_logit=15 &asymp; dense).",
    body,
))

story.append(Paragraph("<b>Alternating training</b>", h3))
story.append(Paragraph(
    "Train from scratch, but alternate between standard dense forward (fast, 9 steps) "
    "and DAGFormer forward (slow, 1 step) in a 9:1 ratio. "
    "Both step types update the base model; the predictor only updates on DAGFormer steps. "
    "Expected throughput: ~90% of baseline.",
    body,
))

story.append(Paragraph("<b>Self-embed without detach</b>", h3))
story.append(Paragraph(
    "Same as staged, but do not detach the embedding before feeding to the predictor. "
    "Gradients from the predictor's topology loss flow back into the embedding layer, "
    "encouraging the embedding to produce representations useful for topology prediction "
    "(not just language modeling).",
    body,
))

story.append(Paragraph("4.2 Improving Predictor Input Quality", h2))
story.append(Paragraph(
    "The original self-embed predictor applies <b>mean pooling</b> over raw token embeddings "
    "(before any transformer layers). This is essentially a bag-of-words &mdash; all "
    "positional and syntactic information is lost. We propose two improved designs:",
    body,
))

story.append(Paragraph("<b>Mini-encoder</b> (independent)", h3))
story.append(Paragraph(
    "A separate embedding table (vocab_size &times; 256) plus a 2-layer transformer encoder "
    "(dim=256, 4 heads). Learns its own contextual text representation optimized purely "
    "for topology prediction, completely independent of the LLM. "
    "~41M parameters (25.7M in embedding table).",
    body,
))

story.append(Paragraph("<b>Context-embed</b> (shared embedding)", h3))
story.append(Paragraph(
    "Reuses the LLM's embedding table (no extra vocab parameters), projects down "
    "(1024 &rarr; 256), then applies a 2-layer transformer encoder. Much fewer parameters "
    "(~16M) while still gaining contextual mixing. The LLM embedding is detached "
    "so the encoder's gradients don't interfere with LLM training.",
    body,
))
story.append(Spacer(1, 0.1 * inch))

current_data = [
    ["#", "Name", "Strategy", "Base Model", "Predictor Input", "Steps"],
    ["1", "Staged", "Load baseline ckpt\n+ DAGFormer", "Pretrained\n(10K steps)", "Self-embed\n(mean pool)", "2,000"],
    ["2", "Alternating", "9:1 standard:\nDAGFormer", "From scratch", "Self-embed\n(mean pool)", "10,000"],
    ["3", "No-detach", "Load baseline ckpt\n+ no .detach()", "Pretrained\n(10K steps)", "Self-embed\n(unfrozen)", "2,000"],
    ["4", "Mini-encoder", "Load baseline ckpt", "Pretrained\n(10K steps)", "Independent\n2L transformer", "2,000"],
    ["5", "Context-embed", "Load baseline ckpt", "Pretrained\n(10K steps)", "Shared embed\n+ 2L transformer", "2,000"],
]
story.append(table(current_data))
story.append(Paragraph(
    "<b>Table 3.</b> Five new experiments currently running. "
    "Key comparison axes: (1) staged vs alternating, (2) detach vs no-detach, "
    "(3) bag-of-words vs contextual predictor input, (4) independent vs shared embedding.",
    caption,
))

story.append(PageBreak())

# ════════════════════════════════════════════════════════════
# 5. DISCUSSION & NEXT STEPS
# ════════════════════════════════════════════════════════════
story.append(Paragraph("5. Discussion &amp; Next Steps", h1))

story.append(Paragraph("5.1 The Core Tension", h2))
story.append(Paragraph(
    "DAGFormer faces a fundamental speed-quality tradeoff: the per-head input assembly "
    "(computing <i>input<sub>j</sub> = base + &Sigma; A[i,j] &middot; head_output[i]</i>) "
    "requires iterating over layers sequentially and performing large einsum operations, "
    "making it ~3.5x slower than standard batched attention. "
    "More DAGFormer steps &rarr; better topology, but fewer total tokens &rarr; worse base model. "
    "The five current experiments explore different points on this tradeoff curve.",
    body,
))

story.append(Paragraph("5.2 Open Questions for Discussion", h2))
story.append(Paragraph(
    "&bull; <b>Does context-dependent topology matter?</b> The static predictor failed, "
    "but it may simply need a different optimization approach (e.g., population-based). "
    "If a single fixed topology suffices, the predictor is unnecessary at inference time.<br/><br/>"
    "&bull; <b>Can we amortize the overhead?</b> The 3.5x slowdown comes from per-head "
    "routing. Possible mitigations: sparse A (skip zero gates), fused kernels, or "
    "only applying DAGFormer to a subset of layers.<br/><br/>"
    "&bull; <b>What is the right predictor architecture?</b> Mean-pooled embeddings are "
    "too crude; a full external encoder (Qwen) is too expensive. The mini-encoder and "
    "context-embed designs are middle ground &mdash; but is 2 transformer layers enough?<br/><br/>"
    "&bull; <b>Optimal model scale for topology learning?</b> 300M may be too small for "
    "topology to matter (fewer redundant heads). Would 1B or 3B show clearer benefits?<br/><br/>"
    "&bull; <b>Distillation approach:</b> Train a dense teacher, then distill into a "
    "DAGFormer student with sparse topology. The student could match the teacher's "
    "performance with fewer active connections (efficiency gain at inference).",
    body,
))

story.append(Paragraph("5.3 Proposed Future Recipes", h2))
story.append(Paragraph(
    "Based on our findings, we propose several additional recipes for the team to discuss "
    "and prioritize:",
    body,
))

recipe_data = [
    ["Recipe", "Description", "Rationale"],
    ["Aggressive\nalternating", "50:1 or 100:1\nstandard:DAGFormer", "Near-baseline throughput;\npredictor gets minimal but\npossibly sufficient signal"],
    ["Layer-subset\nDAGFormer", "Only apply A to\nlast 4 layers", "Reduce overhead to ~1.5x;\nearly layers may not\nneed routing"],
    ["Sparse A\nforward", "Skip einsum for\nA[i,j] < 0.01", "Exploit learned sparsity\nfor speed; mean_A=0.42\nmeans 58% can be skipped"],
    ["Distillation", "Dense teacher &rarr;\nsparse student", "Decouple topology learning\nfrom language modeling;\nreduce inference cost"],
    ["Curriculum", "Short seq (256) first,\nthen increase", "Faster DAGFormer steps\nat small seq_len; gradually\nincrease complexity"],
    ["Scale up", "1B model,\nfrom scratch", "More heads = more\nredundancy = more room\nfor topology optimization"],
]
story.append(table(recipe_data, col_widths=[1.1*inch, 1.6*inch, 2.3*inch]))
story.append(Paragraph(
    "<b>Table 4.</b> Proposed future recipes for team discussion.",
    caption,
))

story.append(Spacer(1, 0.3 * inch))
story.append(Paragraph("5.4 Summary", h2))
story.append(Paragraph(
    "We have established that: (1) midtraining topology onto pretrained models does not work "
    "&mdash; the loss landscape is flat at A=1; (2) pretraining from scratch enables topology "
    "learning (self-embed achieves mean_A=0.42 with learned sparse routing that outperforms "
    "its own dense baseline); (3) the main bottleneck is now the 3.5x throughput overhead "
    "of DAGFormer's per-head forward pass. Our current experiments test five strategies to "
    "address this. The team can propose and prioritize additional recipes from the list above.",
    body,
))

# Build
doc.build(story)
print(f"PDF generated: {OUT}")
print(f"Size: {os.path.getsize(OUT) / 1024:.0f} KB")
