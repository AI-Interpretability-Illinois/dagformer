"""DepthBench-style diagnostics on the complete Dense/FourWay checkpoints.

The metric definitions follow DepthBench (arXiv:2609.32534). DAG interventions
are explicit extensions: bypass preserves the source slot as an identity;
drop_source additionally masks that slot in every downstream Q/K/V/R mix.
This is checkpoint analysis, not the official fixed-budget training sweep.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import importlib.metadata
import json
import math
from pathlib import Path
import subprocess
import time

import numpy as np
import torch
import torch.nn.functional as F

from interp_common import load_elh, load_eval_ids


ROOT = Path(__file__).resolve().parents[1]
CKPTS = ROOT / "checkpoints/pr_sync_20260917"
UPSTREAM = "edca05f8e5c62bd49f91b17461dadae822783eee"


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temp.replace(path)


class DepthProbe:
    def __init__(self, skip=None, drop_source=False, replay=None):
        self.skip = skip
        self.drop_source = drop_source
        self.replay = replay
        self.states = []
        self.residuals = []
        self.alphas = {}

    def record_state(self, index, state, residual=None):
        assert index == len(self.states)
        self.states.append(state.detach())
        if index:
            self.residuals.append(residual.detach())

    def should_skip(self, layer):
        return layer == self.skip

    def mix(self, layer, *alphas):
        if self.replay is not None:
            alphas = self.replay[layer]
        if self.drop_source and self.skip is not None and layer > self.skip:
            # Source 0 is embedding; source s+1 is block s's output. Never
            # renumber sources or renormalize these signed coefficients.
            alphas = tuple(a.clone() for a in alphas)
            for a in alphas:
                a[..., self.skip + 1] = 0
        self.alphas[layer] = tuple(a.detach() for a in alphas)
        return alphas


class DepthModel:
    def __init__(self, kind, model_dir, device):
        self.kind = kind
        self.device = device
        self.model_dir = Path(model_dir)
        elh = load_elh()
        self.cfg = elh.load_config(str(self.model_dir / "config.yaml"))
        if kind == "dag":
            self.model, self.predictor = elh.load_fourway(
                str(self.model_dir / "checkpoint.pt"), self.cfg, device)
            # The established reference evaluation uses this native path.
            self.model.use_triton_kernel = False
            self.base = self.model.olmo
        else:
            self.model = elh.load_dense(str(self.model_dir / "checkpoint.pt"), self.cfg, device)
            self.model.config._attn_implementation = "sdpa"
            self.predictor = None
            self.base = self.model
        self.layers = self.cfg["num_hidden_layers"]

    @contextmanager
    def dense_probe(self, probe):
        handles = [self.base.model.embed_tokens.register_forward_hook(
            lambda module, args, out: probe.record_state(0, out))]
        skipped = None
        original = None
        for i, layer in enumerate(self.base.model.layers):
            def hook(module, args, out, i=i):
                probe.record_state(i + 1, out, args[0])
            handles.append(layer.register_forward_hook(hook))
            if probe.should_skip(i):
                skipped, original = layer, layer.forward
                layer.forward = lambda hidden_states, *args, **kwargs: hidden_states
        try:
            yield
        finally:
            for handle in handles:
                handle.remove()
            if skipped is not None:
                skipped.forward = original

    def forward(self, ids, probe=None, routing=None):
        if self.kind == "dag":
            if routing is None:
                routing = self.predictor(ids)
            return self.model(ids, routing, depth_probe=probe)
        if probe is None:
            return self.model(input_ids=ids, use_cache=False).logits
        with self.dense_probe(probe):
            return self.model(input_ids=ids, use_cache=False).logits

    def decode(self, state):
        return self.base.lm_head(self.base.model.norm(state))


def token_mean(values, valid):
    """values [..., B, T], valid [B, T]; retain each sequence as a unit."""
    return (values * valid).sum(-1) / valid.sum(-1).clamp_min(1)


def losses(logits, labels):
    values = F.cross_entropy(logits.float().flatten(0, 1), labels.flatten(),
                             reduction="none", ignore_index=-100).view_as(labels)
    return token_mean(values, labels != -100)


def updates(probe, branch=False):
    states = torch.stack(probe.states).float()
    left = torch.stack(probe.residuals).float() if branch else states[:-1]
    return states[1:] - left


def angular(probe, valid):
    units = F.normalize(torch.stack(probe.states).float(), dim=-1)
    result = torch.full((len(valid), len(units), len(units)), float("nan"), device=valid.device)
    for i in range(len(units) - 1):
        cosine = (units[i + 1:] * units[i]).sum(-1).clamp(-1, 1)
        result[:, i, i + 1:] = (token_mean(cosine.acos() / math.pi, valid)).T
    return result.cpu().numpy()


def logit_lens(model, probe, logits, labels, chunk=128):
    """Decode original written states; chunk only the vocab projection/readout."""
    B, T = labels.shape
    result = np.zeros((B, len(probe.states), 3), dtype=np.float64)
    valid_counts = (labels != -100).sum(-1).cpu().numpy()
    for start in range(0, T, chunk):
        end = min(T, start + chunk)
        target = labels[:, start:end]
        valid = target != -100
        final_logp = logits[:, start:end].float().log_softmax(-1)
        final_p = final_logp.exp()
        final_top = final_logp.topk(5, dim=-1).indices
        for i, state in enumerate(probe.states):
            # Final state uses the actual model distribution, avoiding numerical
            # differences from a differently shaped GEMM at the readout chunk.
            lp = final_logp if i == model.layers else model.decode(state[:, start:end]).float().log_softmax(-1)
            ce = F.nll_loss(lp.flatten(0, 1), target.flatten(), reduction="none",
                            ignore_index=-100).view_as(target)
            kl = (final_p * (final_logp - lp)).sum(-1)
            top = lp.topk(5, dim=-1).indices
            overlap = (top.unsqueeze(-1) == final_top.unsqueeze(-2)).any(-1).float().mean(-1)
            values = torch.stack([(x * valid).sum(-1) for x in (ce, kl, overlap)], -1)
            result[:, i] += values.cpu().double().numpy()
    return result / valid_counts[:, None, None]


def verify(model, ids, labels):
    """One check of each newly connected path, on the real checkpoint."""
    ids, labels = ids[:1, :32], labels[:1, :32]
    routing = model.predictor(ids) if model.predictor is not None else None
    native = model.forward(ids, routing=routing)
    probe = DepthProbe()
    observed = model.forward(ids, probe, routing)
    decoded = model.decode(probe.states[-1])
    result = dict(observer_max_logit_error=float((native - observed).abs().max()),
                  final_readout_max_logit_error=float((native - decoded).abs().max()))
    if not torch.equal(native, observed) or not torch.equal(native, decoded):
        raise AssertionError(f"Observer/readout changed native logits: {result}")
    skip = model.layers // 2
    edited = DepthProbe(skip=skip, drop_source=model.kind == "dag")
    changed = model.forward(ids, edited, routing)
    assert all(torch.equal(a, b) for a, b in zip(probe.states[:skip+1], edited.states[:skip+1]))
    assert torch.equal(edited.states[skip], edited.states[skip + 1])
    result["skip_logit_change"] = float((native - changed).abs().max())
    assert result["skip_logit_change"] > 0
    if model.kind == "dag":
        assert all(not a[..., skip+1].count_nonzero() for l, alphas in edited.alphas.items()
                   if l > skip for a in alphas)
        replay = model.forward(ids, DepthProbe(replay=probe.alphas), routing)
        result["replay_max_logit_error"] = float((native - replay).abs().max())
        assert torch.equal(native, replay)
    result["baseline_nll"] = float(losses(native, labels).mean())
    return result


def new_arrays(n, L, arms):
    arrays = {"nll": np.full(n, np.nan), "valid_tokens": np.zeros(n, dtype=np.int64),
              "angular": np.full((n, L+1, L+1), np.nan),
              "lens": np.full((n, L+1, 3), np.nan),
              "update_norm": np.full((n, L), np.nan),
              "branch_norm": np.full((n, L), np.nan), "completed": np.array(0)}
    for arm in arms:
        arrays[f"{arm}/skip_nll"] = np.full((n, L), np.nan)
        for metric in ("causal", "change_norm", "branch_causal", "branch_change_norm"):
            arrays[f"{arm}/{metric}"] = np.full((n, L, L), np.nan)
    return arrays


def save_arrays(path, arrays):
    temp = path.with_suffix(".tmp")
    with temp.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    temp.replace(path)


@torch.inference_mode()
def evaluate(model, ids, labels, args, name, out):
    arms = ["bypass_dynamic"] if model.kind == "dense" else [
        "bypass_dynamic", "bypass_replay", "drop_source_dynamic", "drop_source_replay"]
    path = out / f"{model.kind}_{name}.npz"
    if path.exists():
        if not args.resume:
            raise FileExistsError(f"{path} exists; use --resume to continue the same protocol")
        with np.load(path) as old:
            arrays = {key: old[key] for key in old.files}
        if arrays["lens"].shape != (len(ids), model.layers + 1, 3):
            raise ValueError("Resume data shape differs")
    else:
        arrays = new_arrays(len(ids), model.layers, arms)
    initial = int(arrays["completed"])
    started = time.monotonic()
    for start in range(initial, len(ids), args.batch_size):
        end = min(start + args.batch_size, len(ids))
        batch, target = ids[start:end].to(model.device), labels[start:end].to(model.device)
        valid = target != -100
        routing = model.predictor(batch) if model.predictor is not None else None
        baseline = DepthProbe()
        logits = model.forward(batch, baseline, routing)
        arrays["valid_tokens"][start:end] = valid.sum(-1).cpu().numpy()
        arrays["nll"][start:end] = losses(logits, target).cpu().numpy()
        arrays["angular"][start:end] = angular(baseline, valid)
        arrays["lens"][start:end] = logit_lens(model, baseline, logits, target)
        del logits
        reference = {key: updates(baseline, branch=(key == "branch")) for key in ("state", "branch")}
        norms = {key: value.norm(dim=-1) for key, value in reference.items()}
        arrays["update_norm"][start:end] = token_mean(norms["state"], valid).T.cpu().numpy()
        arrays["branch_norm"][start:end] = token_mean(norms["branch"], valid).T.cpu().numpy()
        for s in range(model.layers):
            for arm in arms:
                probe = DepthProbe(s, drop_source=arm.startswith("drop_source"),
                                   replay=baseline.alphas if arm.endswith("replay") else None)
                logits = model.forward(batch, probe, routing)
                arrays[f"{arm}/skip_nll"][start:end, s] = losses(logits, target).cpu().numpy()
                del logits
                for key, prefix in [("state", ""), ("branch", "branch_")]:
                    change = (updates(probe, branch=(key == "branch")) - reference[key]).norm(dim=-1)
                    ratio = change / norms[key].clamp_min(1e-8)
                    arrays[f"{arm}/{prefix}causal"][start:end, s, s+1:] = token_mean(ratio[s+1:], valid).T.cpu().numpy()
                    arrays[f"{arm}/{prefix}change_norm"][start:end, s, s+1:] = token_mean(change[s+1:], valid).T.cpu().numpy()
                del probe
        arrays["completed"] = np.array(end)
        elapsed = time.monotonic() - started
        rate = elapsed / (end - initial)
        progress = dict(model=model.kind, corpus=name, completed=end, total=len(ids),
                        elapsed_seconds=elapsed, estimated_seconds_remaining=rate*(len(ids)-end),
                        nll=float(np.mean(arrays["nll"][:end])))
        print(json.dumps(progress), flush=True)
        save_arrays(path, arrays)
        write_json(out / f"progress_{model.kind}.json", progress)
    return dict(corpus=name, windows=len(ids), tokens=int(arrays["valid_tokens"].sum()),
                nll=float(np.average(arrays["nll"], weights=arrays["valid_tokens"])),
                file=path.name, arms=arms)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--kind", choices=["dense", "dag"], required=True)
    ap.add_argument("--model-dir", type=Path)
    ap.add_argument("--cache", action="append", help="name=path; repeat for multiple corpora")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--memory-fraction", type=float, default=.65)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()
    torch.set_num_threads(4)
    torch.backends.mha.set_fastpath_enabled(False)
    memory_device = torch.device(args.device)
    if memory_device.index is None:
        memory_device = torch.cuda.current_device()
    torch.cuda.set_per_process_memory_fraction(args.memory_fraction, device=memory_device)
    torch.manual_seed(20260930)
    model_dir = args.model_dir or CKPTS / ("300m-baseline" if args.kind == "dense" else "300m-dagformer")
    caches = args.cache or [f"wikitext_test={CKPTS}/eval_corpora/wikitext_test.pt", f"dolma_eval={CKPTS}/eval_cache.pt"]
    args.out.mkdir(parents=True, exist_ok=True)
    model = DepthModel(args.kind, model_dir, args.device)
    meta_path = args.out / f"metadata_{args.kind}.json"
    protocol = dict(model_dir=str(model_dir), config=model.cfg, caches=caches,
                    upstream_commit=UPSTREAM, dtype="native BF16 backbone / FP32 predictor and correction",
                    state="embedding and layer-written states, before final norm",
                    scoring="all valid cached next-token labels, including the last target of each window",
                    causal="mean_tokens ||delta_after_skip - delta|| / max(||delta||, 1e-8)",
                    branch_causal="same formula on output minus R mix (dense: output minus incoming state)",
                    skip_bypass="identity layer, original indexed source slot remains readable",
                    skip_drop_source="bypass plus zero coefficients to skipped source in every downstream Q/K/V/R mix; no reindexing or renormalization",
                    replay="effective Q/K/V/R coefficients from unperturbed forward; external predictor always sees the unchanged input",
                    lens="final native norm/head on layer-written state; not a unique next-consumer Q/K/V input",
                    scope="native-checkpoint diagnostics adapted from DepthBench, not official matched-budget depth scaling")
    if args.resume and meta_path.exists():
        old = json.loads(meta_path.read_text())
        if old["protocol"] != protocol:
            raise ValueError("Cannot resume a different protocol")
    meta = dict(protocol=protocol, git_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                versions={name: importlib.metadata.version(name) for name in ("torch", "transformers", "numpy")},
                parameters=sum(p.numel() for p in model.model.parameters()) +
                    (sum(p.numel() for p in model.predictor.parameters()) if model.predictor is not None else 0),
                complete=False, results=[])
    checkpoint = torch.load(model_dir / "checkpoint.pt", map_location="cpu", weights_only=False, mmap=True)
    meta["checkpoint_step"] = checkpoint.get("step")
    del checkpoint
    with torch.inference_mode():
        for index, entry in enumerate(caches):
            name, source = entry.split("=", 1)
            ids, labels = load_eval_ids(source)
            if (labels != -100).sum(-1).min() == 0:
                raise ValueError("Empty evaluation window")
            check = labels[:, :-1] != -100
            assert torch.equal(labels[:, :-1][check], ids[:, 1:][check]), "Labels are not shifted next-token targets"
            if index == 0:
                meta["verification"] = verify(model, ids.to(args.device), labels.to(args.device))
                write_json(meta_path, meta)
                print(json.dumps(meta["verification"]), flush=True)
            meta["results"].append(evaluate(model, ids, labels, args, name, args.out))
            write_json(meta_path, meta)
    meta["complete"] = True
    write_json(meta_path, meta)


if __name__ == "__main__":
    main()
