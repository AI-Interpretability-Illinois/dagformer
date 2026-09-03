"""Does the copy-suppression intervention reduce degenerate repetition in
free (greedy) generation? Greedy decoding from held-out prefixes under
lambda in {0, 0.25, 0.5, 1.0}; measure repeated-4-gram rate and distinct-n
of the generated continuation (Holtzman et al. degeneration metrics), plus
the dense model's perplexity of the generated text as a fluency proxy.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F

from interp_common import load_elh, save_json
from interp_editing import layer_chunks


def rep_n(seq: list[int], n: int) -> float:
    grams = [tuple(seq[i:i + n]) for i in range(len(seq) - n + 1)]
    if not grams:
        return 0.0
    return 1.0 - len(set(grams)) / len(grams)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--dense-config", default="configs/pretrain_300m_baseline_mmap.yaml")
    ap.add_argument("--dense-ckpt", default="checkpoints/pretrain_300m_baseline_mmap/checkpoint_step9000.pt")
    ap.add_argument("--dump-dir", default="experiments/results/interp/alpha_dumps")
    ap.add_argument("--out", default="experiments/results/interp")
    ap.add_argument("--n-prefix", type=int, default=40)
    ap.add_argument("--prefix-len", type=int, default=128)
    ap.add_argument("--gen-len", type=int, default=200)
    args = ap.parse_args()

    elh = load_elh()
    cfg = elh.load_config(args.config)
    device = torch.device("cuda")
    L, H = cfg["num_hidden_layers"], cfg["num_attention_heads"]
    chunks = layer_chunks(L, H)
    D = chunks[-1][1]
    dump = Path(args.dump_dir)
    fourway, predictor = elh.load_fourway(args.ckpt, cfg, device)
    dense = elh.load_dense(args.dense_ckpt, elh.load_config(args.dense_config), device).eval()

    corp = torch.load(dump / "corpora.pt", map_location="cpu", weights_only=False)
    ids = corp["eval_ids"]
    prefixes = ids[-args.n_prefix:, : args.prefix_len]

    edit = {"vec": None}
    cap: dict[int, torch.Tensor] = {}
    for i, mlp in enumerate(fourway.correction_mlps):
        def make(i):
            a, b = chunks[i]
            def hook(m, inp, out):
                cap[i] = out.detach()
                v = edit["vec"]
                return out if v is None else out + v[a:b].to(out.device, out.dtype).view(1, 1, -1)
            return hook
        mlp.register_forward_hook(make(i))

    @torch.no_grad()
    def fwd(rows):
        rows = rows.to(device)
        return fourway(rows, predictor(rows))

    def corr_mean(rows, from_pos):
        edit["vec"] = None
        acc = torch.zeros(D); n = 0
        for i in range(rows.shape[0]):
            cap.clear(); fwd(rows[i:i + 1])
            acc += torch.cat([cap[k].float().cpu() for k in range(len(chunks))], -1)[0, from_pos:].mean(0); n += 1
        return acc / n
    fp = corr_mean(corp["repeat_ids"][:16], 128) - corr_mean(corp["random_ids"][:16], 128)

    @torch.no_grad()
    def generate(lam: float) -> torch.Tensor:
        edit["vec"] = None if lam == 0 else -lam * fp
        gen = prefixes.clone()
        for _ in range(args.gen_len):
            nxt = fwd(gen)[:, -1].argmax(-1).cpu()
            gen = torch.cat([gen, nxt.view(-1, 1)], dim=1)
        edit["vec"] = None
        return gen[:, args.prefix_len:]

    @torch.no_grad()
    def dense_ppl(cont: torch.Tensor) -> float:
        full = torch.cat([prefixes, cont], dim=1).to(device)
        lg = dense(full).logits[:, args.prefix_len - 1:-1].float()
        nll = F.cross_entropy(lg.reshape(-1, lg.shape[-1]), full[:, args.prefix_len:].reshape(-1))
        return nll.exp().item()

    results = {}
    for lam in (0.0, 0.25, 0.5, 1.0):
        cont = generate(lam)
        rows = cont.tolist()
        r = {
            "rep4": sum(rep_n(s, 4) for s in rows) / len(rows),
            "distinct2": sum(len(set(zip(s, s[1:]))) / max(1, len(s) - 1) for s in rows) / len(rows),
            "frac_seqs_looping": sum(rep_n(s, 4) > 0.5 for s in rows) / len(rows),
            "dense_ppl_of_generation": dense_ppl(cont),
        }
        results[f"lam{lam}"] = r
        print(f"[gen] lam={lam}: {r}")
    save_json(results, Path(args.out) / "generation_step9000.json")
    print("[gen] DONE")


if __name__ == "__main__":
    main()
