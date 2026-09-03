"""Audit & control of training-data regurgitation (parametric memorization).

Setup: sample training windows the step-9000 model has SEEN (first 4.6M
positions of the seed-42 block permutation) and windows it has NOT seen.
For each window: prefix P=64 tokens, continuation K=32 tokens.
  - teacher-forced continuation NLL (memorization gap = unseen - seen)
  - greedy extraction: fraction of the 32 continuation tokens reproduced
    exactly, and the rate of full 32-token extraction (Carlini-style)
Audit: does the per-token projection of correction outputs onto the copy
fingerprint separate seen from unseen windows (ROC-AUC)?
Control: repeat the measurements with the copy-suppression intervention
(subtract lambda * fingerprint). A selective effect closes the seen/unseen
gap and lowers extraction more than it raises unseen-window loss.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from interp_common import REPO, load_elh, save_json
from interp_editing import layer_chunks
from src.data.mmap_dataset import MmapPackedDataset, _block_permutation


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--dump-dir", default="experiments/results/interp/alpha_dumps")
    ap.add_argument("--out", default="experiments/results/interp")
    ap.add_argument("--seen-positions", type=int, default=4_608_000,
                    help="global permuted positions consumed by step 9000 (9000*4*32*4)")
    ap.add_argument("--n-windows", type=int, default=600)
    ap.add_argument("--prefix", type=int, default=64)
    ap.add_argument("--cont", type=int, default=32)
    ap.add_argument("--batch", type=int, default=50)
    args = ap.parse_args()

    elh = load_elh()
    cfg = elh.load_config(args.config)
    device = torch.device("cuda")
    L, H = cfg["num_hidden_layers"], cfg["num_attention_heads"]
    chunks = layer_chunks(L, H)
    D = chunks[-1][1]
    P, K = args.prefix, args.cont
    dump = Path(args.dump_dir)
    fourway, predictor = elh.load_fourway(args.ckpt, cfg, device)

    # ---- windows: seen vs unseen ----
    ds = MmapPackedDataset(cfg["mmap_index_path"], seq_len=cfg["seq_len"],
                           seed=cfg.get("seed", 42), block_size=1024)
    order = np.asarray(list(_block_permutation(ds.n_samples, cfg.get("seed", 42), 1024)))
    n_seen_blocks = args.seen_positions // 1024
    rng = np.random.default_rng(0)
    seen_blocks = order[: int(n_seen_blocks * 0.9)]            # margin from the boundary
    unseen_blocks = order[int(n_seen_blocks * 1.2):]
    def sample_windows(blocks, n):
        b = rng.choice(blocks, size=n, replace=False)
        idx = b * 1024 + rng.integers(0, 1024, size=n)
        idx = np.minimum(idx, ds.n_samples - 1)
        return torch.tensor(np.stack([ds._read_window(int(i))[: P + K].astype(np.int64)
                                      for i in idx]))
    seen = sample_windows(seen_blocks, args.n_windows)
    unseen = sample_windows(unseen_blocks, args.n_windows)
    print(f"[mem] n_samples={ds.n_samples:,} seen_blocks={len(seen_blocks):,} "
          f"unseen_blocks={len(unseen_blocks):,}; windows {tuple(seen.shape)}")

    # ---- hooks (vector broadcast over any B,T) ----
    edit = {"vec": None}
    captured: dict[int, torch.Tensor] = {}
    for i, mlp in enumerate(fourway.correction_mlps):
        def make(i):
            a, b = chunks[i]
            def hook(m, inp, out):
                captured[i] = out.detach()
                v = edit["vec"]
                return out if v is None else out + v[a:b].to(out.device, out.dtype).view(1, 1, -1)
            return hook
        mlp.register_forward_hook(make(i))

    @torch.no_grad()
    def fwd(rows):
        rows = rows.to(device)
        return fourway(rows, predictor(rows))

    # fingerprint from the synthetic repeat/random corpora
    corp = torch.load(dump / "corpora.pt", map_location="cpu", weights_only=False)
    def corr_mean(rows, from_pos):
        edit["vec"] = None
        acc = torch.zeros(D); n = 0
        for i in range(rows.shape[0]):
            captured.clear(); fwd(rows[i:i + 1])
            flat = torch.cat([captured[k].float().cpu() for k in range(len(chunks))], dim=-1)
            acc += flat[0, from_pos:].mean(0); n += 1
        return acc / n
    fp = corr_mean(corp["repeat_ids"][:16], 128) - corr_mean(corp["random_ids"][:16], 128)
    fp_unit = fp / fp.norm()

    @torch.no_grad()
    def measure(rows, lam: float) -> dict:
        edit["vec"] = None if lam == 0 else -lam * fp
        nlls, projs, exact_frac, full_rate, greedy_pref = [], [], [], [], []
        for s in range(0, rows.shape[0], args.batch):
            b = rows[s:s + args.batch]
            # teacher-forced continuation NLL + audit projection
            captured.clear()
            lg = fwd(b)[:, P - 1:P + K - 1].float()          # predicts tokens P..P+K-1
            tgt = b[:, P:P + K].to(device)
            nll = F.cross_entropy(lg.reshape(-1, lg.shape[-1]), tgt.reshape(-1),
                                  reduction="none").view(b.shape[0], K).mean(1)
            nlls.append(nll.cpu())
            flat = torch.cat([captured[k].float().cpu() for k in range(len(chunks))], dim=-1)
            projs.append((flat[:, P - 1:P + K - 1] @ fp_unit).mean(1))
            # greedy extraction from the prefix
            gen = b[:, :P].clone()
            for _ in range(K):
                nxt = fwd(gen)[:, -1].argmax(-1).cpu()
                gen = torch.cat([gen, nxt.view(-1, 1)], dim=1)
            match = (gen[:, P:P + K] == b[:, P:P + K])
            exact_frac.append(match.float().mean(1))
            full_rate.append(match.all(1).float())
            # length of the exactly-matching greedy prefix
            first_miss = (~match).float().argmax(1)   # index of first mismatch (0 if none)
            pref = torch.where(match.all(1), torch.full_like(first_miss, K), first_miss)
            greedy_pref.append(pref.float())
        edit["vec"] = None
        return {"cont_nll": torch.cat(nlls).mean().item(),
                "greedy_token_match": torch.cat(exact_frac).mean().item(),
                "full_extraction_rate": torch.cat(full_rate).mean().item(),
                "mean_matched_prefix": torch.cat(greedy_pref).mean().item(),
                "_proj": torch.cat(projs)}

    from sklearn.metrics import roc_auc_score
    results = {}
    for lam in (0.0, 0.5, 1.0):
        r_seen = measure(seen, lam)
        r_unseen = measure(unseen, lam)
        if lam == 0.0:
            y = np.r_[np.ones(len(r_seen["_proj"])), np.zeros(len(r_unseen["_proj"]))]
            sc = torch.cat([r_seen["_proj"], r_unseen["_proj"]]).numpy()
            results["audit_auc_seen_vs_unseen_projection"] = float(roc_auc_score(y, sc))
        results[f"lam{lam}"] = {
            "seen": {k: v for k, v in r_seen.items() if not k.startswith("_")},
            "unseen": {k: v for k, v in r_unseen.items() if not k.startswith("_")},
            "memorization_gap_nll": r_unseen["cont_nll"] - r_seen["cont_nll"],
        }
        print(f"[mem] lam={lam}: seen={results[f'lam{lam}']['seen']}")
        print(f"[mem] lam={lam}: unseen={results[f'lam{lam}']['unseen']}")
        print(f"[mem] lam={lam}: gap(unseen-seen) NLL = {results[f'lam{lam}']['memorization_gap_nll']:.4f}")
    print(f"[mem] audit AUC (projection, seen vs unseen) = "
          f"{results['audit_auc_seen_vs_unseen_projection']:.3f}")
    save_json(results, Path(args.out) / "memorization_step9000.json")
    print("[mem] DONE")


if __name__ == "__main__":
    main()
