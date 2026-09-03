"""Final causal test: are the correction MLPs' content-reactive outputs
causally necessary, or epiphenomenal (like the external predictor's)?

Replaces each correction MLP's output (via forward hooks) with:
  live            the real hidden-state-conditioned output (sanity = dynamic)
  corr_pos_table  per-position calib mean (content-free positional schedule)
  corr_static     global calib mean (constant)
  corr_swap       another sequence's dumped delta-corr (content misaligned)
  corr_zero       zeros (sanity = dynamic_no_corr)

If live << corr_pos_table: content-reactivity is causally real, completing
the division-of-labor story (external predictor = positional schedule,
corrections = content controller). If live ~= corr_pos_table: the routing
system is causally positional end-to-end and content encoding is inert.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F

from interp_common import load_elh, save_json


def layer_chunks(num_layers: int, num_heads: int) -> list[tuple[int, int]]:
    offs, off = [], 0
    for l in range(1, num_layers):
        sz = (3 * num_heads + 1) * (l + 1)
        offs.append((off, off + sz))
        off += sz
    return offs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--dump-dir", default="experiments/results/interp/alpha_dumps")
    ap.add_argument("--out", default="experiments/results/interp")
    ap.add_argument("--n-calib", type=int, default=25)
    args = ap.parse_args()

    elh = load_elh()
    cfg = elh.load_config(args.config)
    device = torch.device("cuda")
    L, H = cfg["num_hidden_layers"], cfg["num_attention_heads"]
    chunks = layer_chunks(L, H)
    dump = Path(args.dump_dir)

    fourway, predictor = elh.load_fourway(args.ckpt, cfg, device)
    assert fourway.use_local_correction

    corp = torch.load(dump / "corpora.pt", map_location="cpu", weights_only=False)
    ids, labels = corp["eval_ids"], corp["eval_labels"]
    corr_dump = torch.load(dump / "corr_eval_step9000.pt", map_location="cpu").float()
    nc = args.n_calib
    n_test, T = ids.shape[0] - nc, ids.shape[1]

    pos_table = corr_dump[:nc].mean(0)                    # [T, D]
    static_vec = corr_dump[:nc].reshape(-1, corr_dump.shape[-1]).mean(0)  # [D]

    # Forward hooks replacing correction outputs. `replacement[i]` is set
    # before each forward: None = keep live output.
    replacement: list[torch.Tensor | None] = [None] * len(chunks)
    hooks = []
    for i, mlp in enumerate(fourway.correction_mlps):
        def make(i):
            def hook(m, inp, out):
                r = replacement[i]
                return out if r is None else r.to(device=out.device, dtype=out.dtype)
            return hook
        hooks.append(mlp.register_forward_hook(make(i)))

    def set_replacement(flat: torch.Tensor | None):
        for i, (a, b) in enumerate(chunks):
            replacement[i] = None if flat is None else flat[:, :, a:b]

    @torch.no_grad()
    def run(mode: str, i: int) -> float:
        if mode == "live":
            set_replacement(None)
        elif mode == "corr_pos_table":
            set_replacement(pos_table.unsqueeze(0))
        elif mode == "corr_static":
            set_replacement(static_vec.view(1, 1, -1).expand(1, T, -1))
        elif mode == "corr_swap":
            set_replacement(corr_dump[nc + (i + 1) % n_test].unsqueeze(0))
        elif mode == "corr_zero":
            set_replacement(torch.zeros(1, T, corr_dump.shape[-1]))
        row = ids[nc + i:nc + i + 1].to(device)
        routing = predictor(row)
        logits = fourway(row, routing)
        return F.cross_entropy(logits.float().view(-1, logits.shape[-1]),
                               labels[nc + i:nc + i + 1].to(device).view(-1)).item()

    results = {}
    try:
        for mode in ("live", "corr_pos_table", "corr_static", "corr_swap", "corr_zero"):
            vals = [run(mode, i) for i in range(n_test)]
            results[mode] = sum(vals) / len(vals)
            print(f"[corrcausal] {mode:16s} NLL = {results[mode]:.4f}")
    finally:
        for h in hooks:
            h.remove()

    save_json(results, Path(args.out) / "corr_causal_step9000.json")
    print("[corrcausal] DONE")


if __name__ == "__main__":
    main()
