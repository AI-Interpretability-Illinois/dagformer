"""Collect the matched-corpus pruning runs (Table 4: every Table 1 model, owner 2026-10-07) into one file.

75M/150M runs live on timan1 (/srv/local/xy51/prune/checkpoints/<size>_<fam>_math_s<k>[_frozenall|_frozenpred]; fam in
dense, fourway (= DAGFormer), denseformer, hc_paper, mhc, muddformer), 300M runs on Delta
(/work/hdd/biro/xiaocong/dagformer_pruning2/checkpoints/300m_<fam>_21b_math_s<k>...; fam dagformer instead of fourway).
Writes experiments/pruning/results/matched/summary.json and prune_summary_matched.md.
    python3 scripts/collect_prune_matched.py
"""
import json
import os
import subprocess

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(REPO, "experiments/pruning/results/matched")
TIMAN = "/srv/local/xy51/prune/checkpoints"
DELTA = "/work/hdd/biro/xiaocong/dagformer_pruning2/checkpoints"
FAMS = ["dense", "denseformer", "hc_paper", "mhc", "muddformer", "dagformer"]
SUFFIXES = [f"_s{k}" for k in (0, 30, 50, 70)] + ["_s50_frozenall", "_s50_frozenpred"]
KEEP = ("final_domain_nll", "final_general_nll", "initial_domain_nll", "base_params_total", "base_params_remaining",
        "block_params_total", "block_params_remaining", "steps", "model")


def names(size):
    for fam in FAMS:
        f = "fourway" if (fam == "dagformer" and size != "300m") else fam
        mid = "_21b_math" if size == "300m" else "_math"
        for suf in SUFFIXES:
            yield fam, suf, f"{size}_{f}{mid}{suf}"


def timan_summaries(wanted):
    cmd = ("python3 -c \"import json,os,sys\nout={}\nfor n in sys.argv[1:]:\n p='%s/'+n+'/summary.json'\n"
           " if os.path.exists(p): out[n]=json.load(open(p))\nprint(json.dumps(out))\" " % TIMAN) + " ".join(wanted)
    r = subprocess.run(["timeout", "120", "ssh", "-o", "BatchMode=yes", "timan1", cmd], capture_output=True, text=True)
    line = [l for l in r.stdout.splitlines() if l.startswith("{")]
    return json.loads(line[-1]) if line else {}


def main():
    res = {}
    tw = [n for size in ("75m", "150m") for _, _, n in names(size)]
    found = timan_summaries(tw)
    for size in ("75m", "150m", "300m"):
        for fam, suf, n in names(size):
            if size == "300m":
                p = os.path.join(DELTA, n, "summary.json")
                s = json.load(open(p)) if os.path.exists(p) else None
            else:
                s = found.get(n)
            if s:
                res[f"{size}/{fam}{suf}"] = {"run": n, **{k: s.get(k) for k in KEEP}}
    os.makedirs(OUT, exist_ok=True)
    json.dump(res, open(os.path.join(OUT, "summary.json"), "w"), indent=1)
    lines = ["| size | family | run | NLL before | NLL final | general NLL final | block sparsity |", "|---|---|---|---|---|---|---|"]
    for key, s in res.items():
        size, rest = key.split("/")
        sp = 1 - s["block_params_remaining"] / s["block_params_total"] if s.get("block_params_total") else float("nan")
        lines.append(f"| {size} | {rest} | {s['run']} | {s['initial_domain_nll']:.4f} | {s['final_domain_nll']:.4f} | "
                     f"{(s['final_general_nll'] or float('nan')):.4f} | {sp:.3f} |")
    open(os.path.join(OUT, "prune_summary_matched.md"), "w").write(
        "# Matched-corpus pruning (Table 4), collected by scripts/collect_prune_matched.py\n\n" + "\n".join(lines) + "\n")
    total = sum(1 for size in ("75m", "150m", "300m") for _ in names(size))
    print(f"{len(res)} runs collected (of {total} possible names)")


if __name__ == "__main__":
    main()
