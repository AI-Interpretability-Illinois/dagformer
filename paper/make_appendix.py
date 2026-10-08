"""Generate paper/appendix_tables.tex from the repo's result files (no models or GPUs needed).

    python paper/make_appendix.py
"""
import json
import os
import re

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "appendix_tables.tex")


def md_rows(path):
    rows = []
    for line in open(path):
        if not line.startswith("|") or set(line.strip()) <= {"|", "-", " ", ":"}:
            continue
        rows.append([c.strip() for c in line.strip().strip("|").split("|")])
    return rows


def tex(s):
    return s.replace("_", "\\_").replace("%", "\\%").replace("&", "\\&")


L = []
def w(x=""):
    L.append(x)


# ------------------------------------------------------------- A. hyperparameters
w("\\subsection{Training hyperparameters}")
w("\\label{app:hparams}")
w("\\begin{table*}[h]\\centering\\footnotesize\\setlength{\\tabcolsep}{3pt}")
w("\\begin{tabular}{lcccc}\\toprule")
w("size & 75M & 150M & 300M & 1B \\\\\\midrule")
w("layers / width / heads & 6 / 512 / 8 & 8 / 768 / 12 & 12 / 1024 / 16 & 16 / 2048 / 16 \\\\")
w("MLP width & 2048 & 3072 & 4096 & 8192 \\\\")
w("backbone params (M) & 76.6 & 152.6 & 304.1 & 1279.4 \\\\")
w("routing params (M) & 29.1 & 30.0 & 32.3 & 36.7 \\\\")
w("steps & 3000 & 6000 & 12000 & 9540 (+9540) \\\\")
w("tokens (B) & 1.57 & 3.15 & 6.29 & 5.0 (10.0) \\\\")
w("peak LR backbone / predictor & $5\\mathrm{e}{-4}$ / $3\\mathrm{e}{-4}$ & $5\\mathrm{e}{-4}$ / $3\\mathrm{e}{-4}$ & $5\\mathrm{e}{-4}$ / $3\\mathrm{e}{-4}$ & $4\\mathrm{e}{-4}$ / $2\\mathrm{e}{-4}$ \\\\")
w("warmup steps & 300 & 300 & 300 & 500 \\\\")
w("routed FLOPs / dense & 1.18 & 1.33 & 1.65 & 2.13 \\\\")
w("\\bottomrule\\end{tabular}")
w("\\caption{Pretraining configurations. Common to all: 1024-token sequences, 512 sequences (524K tokens) per step, AdamW $(0.9, 0.95)$, weight decay 0.1, gradient clipping 1.0, linear decay to zero, bf16, tied embeddings, vocabulary 100,352 (the OLMo-2 tokenizer). Predictor: 256-d token and position embeddings, 2 causal encoder layers with 4 heads, trunk width 512, correction MLP width 128. The 1B runs were continued from 5B to 10B tokens with the schedule re-stretched and the learning rate re-warmed over 500 steps. Instruction tuning: global batch 64 sequences of 1024 tokens, cosine learning rate $2{\\times}10^{-4}$ ($10^{-4}$ at 1B), assistant-only loss, ChatML format.}")
w("\\label{tab:hparams}\\end{table*}")
w()

# ------------------------------------------------------------- B. matched pairs, all common sets
hdr = [c.strip() for c in open(os.path.join(REPO, "experiments/scaling/runs_table.md")).readline().strip().strip("|").split("|")]
runs = {r[0]: r for r in md_rows(os.path.join(REPO, "experiments/scaling/runs_table.md")) if r[0] in [None] or True}
runs = {r[0]: r for r in md_rows(os.path.join(REPO, "experiments/scaling/runs_table.md"))}
ci = {k: hdr.index(k) for k in ["common dolma21b", "common wikitext2", "common mathinstruct", "common gsm8k", "common dolma_balanced"]}
pairs = [("75M", "12B", "shared_75m_baseline", "shared_75m_dagformer"), ("75M", "12B'", "timan_75m_dense", "timan_75m_corrected"),
         ("150M", "12B", "shared_150m_baseline", "shared_150m_dagformer"), ("150M", "12B'", "timan_150m_dense", "timan_150m_corrected"),
         ("300M", "21B", "300m_dense", "300m_fourway_corrected"), ("300M, seed 2", "21B", "300m_dense_pseed2", "300m_fourway_corrected_pseed2"),
         ("1B, 5B tok", "21B", "1b_dense_5b", "1b_fourway_corrected_5b"),
         ("1B, 10B tok", "21B", "1b_dense_10b", "1b_fourway_corrected_10b")]
w("\\subsection{Matched pairs on all held-out sets}")
w("\\label{app:pairs}")
w("\\begin{table*}[h]\\centering\\footnotesize\\setlength{\\tabcolsep}{2.5pt}")
w("\\begin{tabular}{llcccccccc}\\toprule")
w("size & corpus & \\multicolumn{2}{c}{WikiText-2} & \\multicolumn{2}{c}{MathInstruct} & \\multicolumn{2}{c}{GSM8K} & \\multicolumn{2}{c}{balanced Dolma} \\\\")
w(" & & dense & \\dagf{} & dense & \\dagf{} & dense & \\dagf{} & dense & \\dagf{} \\\\\\midrule")
for size, corpus, d, r in pairs:
    def v(run, key):
        x = runs[run][ci[key]]
        return "\\todo{pending}" if x == "-" else f"{float(x):.3f}"
    w(f"{size} & {corpus} & {v(d,'common wikitext2')} & {v(r,'common wikitext2')} & {v(d,'common mathinstruct')} & {v(r,'common mathinstruct')} & {v(d,'common gsm8k')} & {v(r,'common gsm8k')} & {v(d,'common dolma_balanced')} & {v(r,'common dolma_balanced')} \\\\")
w("\\bottomrule\\end{tabular}")
w("\\caption{Held-out NLL (nats per token) of every matched pair on the four cross-corpus sets (final checkpoints): WikiText-2, MathInstruct, GSM8K and a source-balanced Dolma set (64 windows from each of 14 Dolma sources, drawn from files that no training corpus consumed; books, Wikipedia and open-web-math could not be represented). \\dagf{} is lower in every cell; ``seed 2'' is the 300M pair repeated with a second pretraining seed.}")
w("\\label{tab:pairs_all}\\end{table*}")
w()

# ------------------------------------------------------------- C. gap along training
w("\\subsection{Gap along training}")
w("\\label{app:gapcurve}")
w("The dense-minus-\\dagf{} gap on each run's own held-out cache at matched steps (nats): ")
items = []
for line in open(os.path.join(REPO, "experiments/scaling/results.md")):
    m = re.match(r"- (\w+) (\w+) dense - corrected: (.*)", line.strip())
    if m and m.group(1) in ("timan12b", "delta21b"):
        corpus = {"timan12b": "12B'", "delta21b": "21B"}[m.group(1)]
        items.append(f"{m.group(2).upper()} ({corpus}): {m.group(3)}")
w("; ".join(items) + ". The gap opens within the first quarter of each run, peaks, and narrows slowly; the 1B entry is the 5B-token run.")
w()

# ------------------------------------------------------------- D. locality arms
w("\\subsection{Router-locality ablation on all metrics}")
w("\\label{app:locality}")
loc = [("none (dense)", "loc75m_dense"), ("static learned table (no token input)", "loc75m_static"), ("per-layer predictors (unshared)", "loc75m_per_layer"), ("shared predictor only", "loc75m_global"),
       ("local routers only", "loc75m_local"), ("shared + local (\\dagf{})", "loc75m_both"), ("modular variant", "loc75m_modular"),
       ("modular + sparsity penalty", "loc75m_modular_sparse")]
sft = {}
for r in md_rows(os.path.join(REPO, "experiments/results/lmeval/sft/SUMMARY_locality.md")):
    if len(r) == 4 and "/" in r[1]:
        sft[r[0]] = r[2]
sftkey = {"loc75m_static": "static", "loc75m_dense": "dense", "loc75m_per_layer": "per_layer", "loc75m_global": "global", "loc75m_local": "local",
          "loc75m_both": "both", "loc75m_modular": "modular", "loc75m_modular_sparse": "modular_sparse"}
w("\\begin{table*}[h]\\centering\\footnotesize\\setlength{\\tabcolsep}{2.5pt}")
w("\\begin{tabular}{lccccc}\\toprule")
w("routing source & own cache & WikiText-2 & MathInstruct & GSM8K & MC w/o BoolQ \\\\")
w(" & & & & & base / Alpaca / SmolTalk \\\\\\midrule")
own = hdr.index("own-curve final")
for lab, run in loc:
    if run not in runs:
        continue
    r = runs[run]
    w(f"{lab} & {float(r[own]):.3f} & {float(r[ci['common wikitext2']]):.3f} & {float(r[ci['common mathinstruct']]):.3f} & {float(r[ci['common gsm8k']]):.3f} & {sft.get(sftkey[run], '-')} \\\\")
w("\\bottomrule\\end{tabular}")
w("\\caption{The 75M router-locality arms (1.57B tokens of a 1.7B-token Dolma slice, 4 GPUs each) on every held-out set, and the 11-task multiple-choice mean (without BoolQ, whose majority-class flips move the 12-task mean by up to 20 points at this size) before and after instruction tuning. The modular variant, which also routes the intra-layer edge, is best on loss at 75M; its routing cost is 1.33$\\times$ dense here and it is not better than FourWay routing at 150M or 300M (Appendix~\\ref{app:modular}).}")
w("\\label{tab:locality_all}\\end{table*}")
w()

# ------------------------------------------------------------- E. shared-slice downstream
w("\\subsection{Shared-slice 75M, 150M and 300M pairs downstream}")
w("\\label{app:shared_down}")
rows = [r for r in md_rows(os.path.join(REPO, "experiments/results/lmeval/sft_shared/SUMMARY_shared.md")) if len(r) >= 18 and r[0] in ("75M", "150M", "300M")]
w("\\begin{table*}[h]\\centering\\footnotesize\\setlength{\\tabcolsep}{2.5pt}")
w("\\begin{tabular}{llcccc}\\toprule")
w("size & model & stage & MC acc. (12) & GSM8K bpb & WikiText bpb \\\\\\midrule")
for r in rows:
    model = "dense" if "dense" in r[1] else "\\dagf{}"
    stage = r[2].replace("+Alpaca-Dolly", "+Alpaca")
    w(f"{r[0]} & {model} & {stage} & {r[3]} & {r[16]} & {r[17]} \\\\")
w("\\bottomrule\\end{tabular}")
w("\\caption{The original 12B-slice pairs before and after instruction tuning (the 300M \\dagf{} of this slice stopped at 9000 of 12000 steps, so its margin is a lower bound). Same metrics as Table~\\ref{tab:downstream}.}")
w("\\label{tab:shared_down}\\end{table*}")
w()

# ------------------------------------------------------------- F. pruning: absolute, controls, immediate damage
w("\\subsection{Pruning: absolute losses, controls and immediate damage}")
w("\\label{app:prune}")
ps = {}
for r in md_rows(os.path.join(REPO, "experiments/pruning/results/prune_summary.md")):
    if len(r) == 5 and r[0] in ("75m", "150m", "300m") and r[2].replace('.', '').isdigit():
        ps[(r[0], r[1])] = (float(r[2]), float(r[3]))
ckroot = "/work/hdd/bfqt/xiaocong/dagformer_pruning/checkpoints"


def peak_damage(run):
    p = os.path.join(ckroot, run, "trajectory.json")
    if not os.path.exists(p):
        return None
    t = json.load(open(p))
    post = [x["eval/domain_nll"] for x in t if x.get("tag") == "post_prune" and x["prune/events"] > 1]
    return max(post) if post else None


w("\\begin{table*}[h]\\centering\\footnotesize\\setlength{\\tabcolsep}{2.5pt}")
w("\\begin{tabular}{llcccc}\\toprule")
w("size & setting & dense & \\dagf{} & peak post-prune dense & peak post-prune \\dagf{} \\\\\\midrule")
labels = {"s0": "unpruned finetune", "s30": "30\\% heads+channels", "s50": "50\\% heads+channels", "s70": "70\\% heads+channels",
          "s50_random": "50\\%, random importance", "mod_s33": "33\\% whole blocks", "mod_s50": "50\\% whole blocks"}
for size in ("75m", "150m", "300m"):
    for tag in ("s0", "s30", "s50", "s70", "s50_random", "mod_s33", "mod_s50"):
        if (size, tag) not in ps:
            continue
        b, d = ps[(size, tag)]
        pb, pd = (None, None) if tag == "s0" else (peak_damage(f"{size}_baseline_math_{tag}"), peak_damage(f"{size}_dagformer_math_{tag}"))
        f = lambda x: "-" if x is None else f"{x:.3f}"
        w(f"{size.upper()} & {labels[tag]} & {b:.3f} & {d:.3f} & {f(pb)} & {f(pd)} \\\\")
    fr = [k for k in ps if k[0] == size and "frozenpred" in k[1]]
    for k in fr:
        w(f"{size.upper()} & 50\\%, predictor frozen & - & {ps[k][1]:.3f} & - & {peak_damage(f'{size}_dagformer_math_{k[1]}') or 0:.3f} \\\\")
w("\\bottomrule\\end{tabular}")
w("\\caption{Final held-out MathInstruct NLL after 2000 finetuning steps for every pruning run of the shared-slice pairs, and the highest NLL observed immediately after a pruning update (before recovery). Whole-block pruning removes entire attention or MLP blocks; the 75M \\dagf{} at half the blocks is the one setting where it loses (Section~\\ref{sec:pruning}). With random unit selection both families lose far more than with Taylor importance and the \\dagf{} edge shrinks at 75M.}")
w("\\label{tab:prune_all}\\end{table*}")
w()

# ------------------------------------------------------------- F2. static substitution on all sets
w("\\subsection{Static substitution on all held-out sets}")
w("\\label{app:static}")
w("\\begin{table*}[h]\\centering\\scriptsize\\setlength{\\tabcolsep}{3pt}")
w("\\begin{tabular}{llccccc}\\toprule")
w("model & set & intact & predictor $\\to$ position mean & predictor $\\to$ global mean & local $\\to$ mean & both $\\to$ mean \\\\\\midrule")
for m, lab in (("loc_both", "75M (locality arm, both)"), ("300m_corrected", "300M"), ("1b_corrected_5b", "1B, 5B tok"), ("1b_corrected_10b", "1B, 10B tok")):
    pth = os.path.join(REPO, f"experiments/topology/{m}_analysis_v1.json")
    if not os.path.exists(pth):
        continue
    st = json.load(open(pth))["static"]
    for ev, r in st.items():
        w(f"{lab} & {ev} & {r['intact']:.3f} & {r['pred_posmean']:.3f} & {r['pred_mean']:.3f} & {r.get('corr_mean', float('nan')):.3f} & {r.get('both_mean', float('nan')):.3f} \\\\")
w("\\bottomrule\\end{tabular*}" if False else "\\bottomrule\\end{tabular}")
w("\\caption{Held-out NLL when one routing channel is replaced by its mean over 128 held-out windows of the model's own corpus (`own' / `dolma21b' is that corpus's cache). The predictor's per-token output is replaceable at every size and on every set; the local correction channel's is not.}")
w("\\label{tab:static_all}\\end{table*}")
w()

# ------------------------------------------------------------- G. chat eval details
w("\\subsection{Judged chat evaluation: categories and scores}")
w("\\label{app:chat}")
res = json.load(open(os.path.join(REPO, "experiments/chat_eval/results.json")))
cats = ["writing", "roleplay", "reasoning", "math", "coding", "extraction", "stem", "humanities"]
w("\\begin{table*}[h]\\centering\\footnotesize\\setlength{\\tabcolsep}{2.2pt}")
w("\\begin{tabular}{ll" + "c" * len(cats) + "}\\toprule")
w("SFT & comparison & " + " & ".join(cats) + " \\\\\\midrule")
name = lambda n: n.rsplit("_", 2)[0].replace("_seed2", " s2").replace("300m_corrected", "300M").replace("300m_modular", "300M mod.").replace("1b_corrected_10b", "1B@10B").replace("1b_corrected_5b", "1B@5B").replace("300m_dense", "300M").replace("1b_dense_5b", "1B@5B").replace("1b_dense_10b", "1B@10B")
for c in res["comparisons"]:
    s = "Alpaca" if c["routed"].endswith("alpaca_dolly") else "SmolTalk"
    w(f"{s} & {name(c['routed'])} vs.\\ {name(c['dense'])} & " + " & ".join(f"{c['by_category'][k]:.2f}" for k in cats) + " \\\\")
w("\\bottomrule\\end{tabular}")
w("\\caption{Per-category win rate of \\dagf{} (10 prompts per category, so $\\pm 0.15$ is noise). ``300M mod.'' is the modular variant.}")
w("\\label{tab:chat_cats}\\end{table*}")
w("\\begin{table*}[h]\\centering\\footnotesize\\setlength{\\tabcolsep}{3pt}")
w("\\begin{tabular}{lccc}\\toprule")
w("model & mean score (1--10) & answer tokens & ended on end-of-turn \\\\\\midrule")
for n, m in res["models"].items():
    nn = n.replace("_seed2", " seed 2").replace("_alpaca_dolly", " / Alpaca").replace("_smol_smoltalk", " / SmolTalk").replace("1b_corrected_10b", "1B \\dagf{} 10B").replace("1b_corrected_5b", "1B \\dagf{} 5B").replace("1b_dense_5b", "1B dense 5B").replace("1b_dense_10b", "1B dense 10B").replace("300m_corrected", "300M \\dagf{}").replace("300m_modular", "300M modular").replace("300m_dense", "300M dense")
    w(f"{nn} & {m['mean_score']:.2f} & {m['answer_tokens']:.0f} & {m['stopped_frac']:.2f} \\\\")
w("\\bottomrule\\end{tabular}")
w("\\caption{Single-answer judge scores (MT-Bench single-answer prompt, same judge), mean answer length and the fraction of answers that stopped on the end-of-turn token rather than the 256-token cap. Scores sit near 2 of 10 for every model at this scale; the pairwise win rate is the informative number.}")
w("\\label{tab:chat_scores}\\end{table*}")

open(OUT, "w").write("\n".join(L) + "\n")
print("wrote", OUT, len(L), "lines")
