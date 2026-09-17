"""Bidirectional steering with routing-space SAE features (v3): semantic /
common-class features, wider alpha range, class-rate readout in generation
plus on-rule / off-rule NLL on held-out text.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from interp_common import load_elh, save_json
from interp_editing import layer_chunks
from interp_routing_sae import SAE
from interp_token_effects import build_corpus

# feature id -> (readable name, token-string set for the generation readout)
TARGETS = {
    3583: ("kinship nouns", {"brother", "son", "wife", "uncle", "father", "mother", "daughter", "sister", "husband", "nephew", "cousin", "aunt", "children"}),
    4271: ("body / spatial nouns", {"shoulders", "shoulder", "top", "left", "back", "side", "head", "hand", "hands", "feet", "arm", "arms", "face", "eyes"}),
    6409: ("speech-attribution verbs", {"added", "wrote", "said", "recalled", "says", "replied", "asked", "answered", "continued", "remarked", "explained"}),
    1407: ("time units", {"minute", "month", "year", "day", "week", "hour", "years", "days", "months", "hours", "minutes", "weeks"}),
    2983: ("superlatives / ordinals", {"only", "most", "first", "best", "chief", "last", "greatest", "largest", "highest"}),
    7585: ("verbs after 'to'", {"give", "know", "make", "see", "be", "stay", "take", "get", "go", "have", "do", "keep"}),
    4967: ("subject pronouns", {"you", "it", "i", "we", "he", "she", "they"}),
    2916: ("sentence-initial pronoun/determiner", {"it", "we", "the", "i", "in", "he", "she", "they", "this"}),
    4380: ("'century' after ordinal", {"century", "centuries"}),
    6367: ("negation after do/does", {"not", "n't", "never"}),
    3889: ("opening quote after 'said:'", {'"', "'", "--"}),
    1586: ("closing bracket of citations", {"]", ")", "],", ")."}),
    7245: ("pronoun after 'If'", {"you", "we", "there", "i", "it", "they", "he", "she"}),
    2801: ("two-digit numbers", {str(i) for i in range(10, 32)}),
}
PROMPTS = ["The", "Yesterday", "In the morning,", "It was a quiet day in the village.", "Here is a short story:",
           "The report begins as follows.", "According to the article,", "Once upon a time"]


def main():
    out = Path("experiments/results/interp/routing_sae")
    elh = load_elh()
    cfg = elh.load_config("configs/fourway_300m_dagformer_mmap.yaml")
    device = torch.device("cuda")
    L, H = cfg["num_hidden_layers"], cfg["num_attention_heads"]; T = cfg["seq_len"]
    chunks = layer_chunks(L, H); D = chunks[-1][1]
    fourway, predictor = elh.load_fourway("checkpoints/fourway_300m_dagformer_mmap/checkpoint_step9000.pt", cfg, device)
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(cfg["tokenizer_id"])
    ck = torch.load(out / "sae.pt", map_location="cpu")
    n_feat = ck["sae"]["enc.weight"].shape[0]
    sae = SAE(D, n_feat, 32).to(device); sae.load_state_dict(ck["sae"]); sae.eval()
    mu, sd = ck["mu"].to(device), ck["sd"].to(device)
    feats = {f["feature"]: f for f in json.load(open(out / "features.json"))["features"]}

    state = {"delta": None}
    hooks = []
    for i, mlp in enumerate(fourway.correction_mlps):
        def make(i):
            a, b = chunks[i]
            def hook(m, inp, o):
                dl = state["delta"]
                return o if dl is None else o + dl[a:b].to(o.device, o.dtype)
            return hook
        hooks.append(mlp.register_forward_hook(make(i)))

    ids_nll, lab_nll, _ = build_corpus("/work/hdd/bfqt/data/pretok/dolma_v1_7_12b", T, 100, 42, 1024, 5500000, offset_seq=0)
    @torch.no_grad()
    def per_token_nll():
        outs = []
        for i in range(0, ids_nll.shape[0], 16):
            rows = ids_nll[i:i + 16]
            lg = fourway(rows.to(device), predictor(rows.to(device)))
            outs.append(F.cross_entropy(lg.float().reshape(-1, lg.shape[-1]), lab_nll[i:i + 16].to(device).reshape(-1), reduction="none").view(rows.shape[0], T).cpu())
        return torch.cat(outs)

    @torch.no_grad()
    def sample(prompt_ids, n, new_tokens=60):
        rows = prompt_ids.unsqueeze(0).repeat(n, 1).to(device); gen = torch.zeros(n, 0, dtype=torch.long, device=device)
        for _ in range(new_tokens):
            full = torch.cat([rows, gen], 1)
            lg = fourway(full, predictor(full))[:, -1].float() / 0.9
            p = torch.softmax(lg, -1); sp, si = p.sort(-1, descending=True); cum = sp.cumsum(-1); sp[cum - sp > 0.95] = 0; sp /= sp.sum(-1, keepdim=True)
            gen = torch.cat([gen, si.gather(1, torch.multinomial(sp, 1))], 1)
        return gen.cpu()

    def norm(s):
        return s.replace("Ġ", "").replace("Ċ", "\n").strip().lower()
    vocab_strs = {}
    def tstr(t):
        if t not in vocab_strs: vocab_strs[t] = norm(tok.convert_ids_to_tokens([t])[0])
        return vocab_strs[t]
    state["delta"] = None
    base_nll = per_token_nll()
    lab_np = lab_nll.numpy()
    results = {}
    for f, (name, words) in TARGETS.items():
        if f not in feats:
            # feature not auto-named (rare): still usable if alive; estimate mean act from decoder scale = 1
            mean_act = 1.0
        else:
            mean_act = feats[f]["mean_act"]
        d_f = (sae.dec.weight[:, f] * sd).detach()
        rule_mask = torch.tensor(np.vectorize(lambda t: tstr(int(t)) in words)(lab_np))
        if rule_mask.sum() < 30:
            print(f"[v3] f{f} {name}: too few rule tokens ({int(rule_mask.sum())}), skipping NLL readout")
        res = {"name": name, "mean_act": mean_act, "n_rule_tokens_heldout": int(rule_mask.sum())}
        for alpha in (-8.0, -4.0, -2.0, 0.0, 2.0, 4.0, 8.0):
            state["delta"] = alpha * mean_act * d_f if alpha != 0 else None
            hits = tot = 0; ex = []
            for p in PROMPTS:
                pid = torch.tensor(tok(p, add_special_tokens=False)["input_ids"], dtype=torch.long)
                for g in sample(pid, 4):
                    toks = g.tolist(); hits += sum(tstr(t) in words for t in toks); tot += len(toks)
                    ex.append(p + tok.decode(toks))
            dn = per_token_nll() - base_nll
            on = float(dn[rule_mask].mean()) if rule_mask.sum() >= 30 else float("nan")
            off = float(dn[~rule_mask].mean())
            res[str(alpha)] = {"class_rate": hits / tot, "on_rule_dnll": on, "off_rule_dnll": off, "examples": ex[:3]}
            print(f"[v3] f{f} {name:34s} alpha={alpha:+.0f}: gen rate {100*hits/tot:.2f}% | on {on:+.3f} off {off:+.3f} | {ex[0][:100].replace(chr(10), '⏎')}", flush=True)
        state["delta"] = None
        results[f"f{f}"] = res
    save_json(results, out / "steering_v3.json")
    for hk in hooks: hk.remove()
    print("[v3] DONE")


if __name__ == "__main__":
    main()
