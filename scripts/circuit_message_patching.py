"""Counterfactual routing-message interchange on frozen FourWay copy circuits.

Only this process temporarily wraps torch.einsum. Historical value-token reads
are changed, never the final output or query-position activation. Discovery
and held-out pairs are disjoint; all reported selection uses discovery only.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import importlib.metadata
import json
from pathlib import Path
import random
import subprocess
import time

import numpy as np
import torch
import torch.nn.functional as F

from interp_common import load_elh, load_eval_ids


def make_data(corpus, n_pairs, period, seed, query=31):
    """Three full blocks and an identical unfinished fourth prefix.

    Paired contexts differ only at the three historical value positions. The
    value to predict is absent from the last unfinished prefix. The alternate
    query=15 asks for a different, unchanged block location.
    """
    g = torch.Generator().manual_seed(seed)
    flat = corpus.flatten()
    draw = lambda n: flat[torch.randint(len(flat), (n,), generator=g)].tolist()
    rows, a, b, pair_ids, blocks = [], [], [], [], []
    value = 31
    assert query <= value < period
    for pair in range(n_pairs):
        block = draw(period)
        forbidden = set(block)
        vals = []
        while len(vals) != 2:
            token = draw(1)[0]
            if token not in forbidden and token not in vals and token < 100257:
                vals.append(token)
        original = []
        for token in vals:
            changed = block.copy()
            changed[value] = token
            original.append(changed)
            rows.append(changed * 3 + changed[:query])
        if query == value:
            a.extend(vals)
            b.extend(vals[::-1])
        else:
            a.extend([block[query]] * 2)
            b.extend(vals[::-1])  # wrong value introduced by the donor patch
        pair_ids.extend([pair] * 2)
        blocks.append(original)
    ids = torch.tensor(rows)
    positions = [value + i * period for i in range(3)]
    assert torch.all((ids[0::2] != ids[1::2]).sum(1) == 3)
    assert torch.equal(ids[0::2, 3 * period:], ids[1::2, 3 * period:])
    return dict(ids=ids, a=torch.tensor(a), b=torch.tensor(b), pairs=pair_ids,
                positions=positions, period=period, seed=seed, query=query,
                blocks=blocks, control=query != value)


def arm(name, mode="message", edges=None, streams="qkvr", layers=None,
        heads=None, fixed=False, identity=False):
    return dict(name=name, mode=mode, edges=edges, streams=streams,
                layers=layers, heads=heads, fixed=fixed, identity=identity)


class Interchange:
    def __init__(self, model, predictor, layers, heads):
        self.model, self.predictor = model, predictor
        self.layers, self.heads = layers, heads
        self.original = torch.einsum
        self.qkv_calls = self.r_calls = 0
        self.capture = None
        self.donor = self.recipient = self.spec = None
        self.positions = None

    @contextmanager
    def installed(self):
        torch.einsum = self.einsum
        # All upstream computation is unchanged; materialize only next-token
        # logits instead of T * vocabulary logits, which are unused here.
        handle = self.model.olmo.lm_head.register_forward_pre_hook(
            lambda module, inputs: (inputs[0][:, -1:],))
        try:
            yield
        finally:
            handle.remove()
            torch.einsum = self.original

    def mask(self, layer, stream, sources, device, dtype):
        spec = self.spec
        shape = (sources, self.heads) if stream != "r" else (sources,)
        mask = torch.zeros(shape, device=device, dtype=dtype)
        if stream not in spec["streams"]:
            return mask
        if spec["layers"] is not None and layer not in spec["layers"]:
            return mask
        if spec["edges"] is not None:
            for l, s, h, src in spec["edges"]:
                if l == layer and s == stream:
                    if stream == "r":
                        mask[src] = 1
                    else:
                        mask[src, h] = 1
        elif spec["heads"] is not None and stream != "r":
            for l, h in spec["heads"]:
                if layer == l:
                    mask[:, h] = 1
        elif spec["heads"] is None:
            mask.fill_(1)
        return mask

    def einsum(self, equation, *operands, **kwargs):
        eq = equation.replace(" ", "") if isinstance(equation, str) else ""
        if eq not in ("lbthd,bthl->bhtd", "lbtd,btl->btd"):
            return self.original(equation, *operands, **kwargs)
        z, alpha = operands
        layer = z.shape[0] - 1
        if eq == "lbthd,bthl->bhtd":
            stream = "qkv"[self.qkv_calls % 3]
            assert layer == self.qkv_calls // 3 + 1
            self.qkv_calls += 1
        else:
            stream = "r"
            assert layer == self.r_calls + 1
            self.r_calls += 1
        key = (layer, stream)
        pos = self.positions
        if self.capture is not None:
            # z has [source,batch,position,(head),dimension]. Copy only three
            # historical positions; alpha at all positions is small enough.
            self.capture[key] = (z[:, :, pos].detach().clone(), alpha.detach().clone())
        if self.spec is None:
            return self.original(equation, z, alpha)
        if self.spec["fixed"]:
            alpha = self.recipient[key][1]
        output = self.original(equation, z, alpha)
        mask = self.mask(layer, stream, z.shape[0], z.device, z.dtype)
        if not mask.any():
            return output
        donor_z, donor_alpha = self.donor[key]
        own_z = z[:, :, pos].float()
        own_alpha = alpha[:, pos].float()
        donor_alpha = donor_alpha[:, pos].float()
        donor_z = donor_z.float()
        if stream != "r":
            own_alpha = own_alpha.permute(3, 0, 1, 2)[..., None]
            donor_alpha = donor_alpha.permute(3, 0, 1, 2)[..., None]
            mask = mask[:, None, None, :, None]
        else:
            own_alpha = own_alpha.permute(2, 0, 1)[..., None]
            donor_alpha = donor_alpha.permute(2, 0, 1)[..., None]
            mask = mask[:, None, None, None]
        mode = self.spec["mode"]
        new_z = donor_z if mode in ("content", "message") else own_z
        new_alpha = donor_alpha if mode in ("alpha", "message") else own_alpha
        delta = ((new_z * new_alpha - own_z * own_alpha) * mask).sum(0)
        result = output.clone()
        if stream != "r":
            result[:, :, pos] = (output[:, :, pos].float() + delta.permute(0, 2, 1, 3)).to(output.dtype)
        else:
            result[:, pos] = (output[:, pos].float() + delta).to(output.dtype)
        return result

    @torch.inference_mode()
    def forward(self, ids, positions, capture=False, spec=None, donor=None, recipient=None):
        self.positions = positions
        self.capture = {} if capture else None
        self.spec, self.donor, self.recipient = spec, donor, recipient
        self.qkv_calls = self.r_calls = 0
        logits = self.model(ids, self.predictor(ids))[:, -1].float()
        assert self.qkv_calls == 3 * (self.layers - 1)
        assert self.r_calls == self.layers - 1
        cached = self.capture
        self.capture = self.spec = self.donor = self.recipient = None
        return logits, cached


def measures(logits, a, b, reference):
    lp = logits.log_softmax(-1)
    ref_lp = reference.log_softmax(-1)
    la, lb = lp.gather(1, a[:, None])[:, 0], lp.gather(1, b[:, None])[:, 0]
    best = logits.argmax(-1)
    values = dict(recipient_logp=la, donor_logp=lb, donor_margin=lb-la,
                  donor_two_choice_prob=torch.sigmoid(lb-la),
                  donor_two_choice_accuracy=(lb > la).float(),
                  recipient_vocab_accuracy=(best == a).float(),
                  donor_vocab_accuracy=(best == b).float(),
                  donor_probability=lb.exp(),
                  recipient_distribution_kl=(ref_lp.exp() * (ref_lp-lp)).sum(-1))
    return {k: v.cpu().tolist() for k, v in values.items()}


def summaries(values, reference, donor, paired_correct, control=False):
    n = len(values["donor_margin"])
    rng = np.random.default_rng(20260929)
    indices = rng.integers(n // 2, size=(4000, n // 2))
    result = {}
    for k, vals in values.items():
        x, ref = np.asarray(vals), np.asarray(reference[k])
        delta = (x - ref).reshape(-1, 2).mean(1)
        estimate = dict(mean=float(x.mean()), delta=float(delta.mean()),
                        paired_95ci=np.quantile(delta[indices].mean(1), [.025, .975]).tolist())
        if any(paired_correct):
            ok = np.repeat(np.asarray(paired_correct), 2)
            estimate["both_baselines_correct_mean"] = float(x[ok].mean())
            estimate["both_baselines_correct_delta"] = float((x-ref)[ok].mean())
        result[k] = estimate
    gap = (np.asarray(donor["donor_margin"]) - np.asarray(reference["donor_margin"]))
    if not control and gap.mean() > .01:
        effect = (np.asarray(values["donor_margin"]) - np.asarray(reference["donor_margin"]))
        effect, gap = effect.reshape(-1, 2).mean(1), gap.reshape(-1, 2).mean(1)
        ratios = effect[indices].mean(1) / gap[indices].mean(1)
        result["margin_recovery"] = dict(mean=float(effect.mean()/gap.mean()),
                                         paired_95ci=np.quantile(ratios, [.025, .975]).tolist())
    return result


class Experiment:
    def __init__(self, args):
        self.args = args
        self.device = torch.device("cuda")
        elh = load_elh()
        cfg = elh.load_config(str(args.model / "config.yaml"))
        self.model, self.predictor = elh.load_fourway(str(args.model / "checkpoint.pt"), cfg, self.device)
        self.model.use_triton_kernel = False
        self.L, self.H = cfg["num_hidden_layers"], cfg["num_attention_heads"]
        self.patch = Interchange(self.model, self.predictor, self.L, self.H)
        self.result = dict(complete=False, args={k:str(v) if isinstance(v, Path) else v for k,v in vars(args).items()},
                           git_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
                           versions={k:importlib.metadata.version(k) for k in ("torch", "transformers")},
                           stages={}, selection={}, checks={})
        self.started = time.time()
        args.out.mkdir(parents=True, exist_ok=True)

    def save(self):
        self.result["elapsed_seconds"] = time.time()-self.started
        (self.args.out / "results.json").write_text(json.dumps(self.result, indent=2)+"\n")

    def evaluate(self, stage, data, specs):
        print(f"START {stage}: {len(data['ids'])} directed cases, {len(specs)} intervention arms", flush=True)
        values = {s["name"]:{} for s in specs}
        values.update(reference={}, donor_baseline={})
        max_identity = 0.
        for start in range(0, len(data["ids"]), self.args.batch_size):
            ids = data["ids"][start:start+self.args.batch_size].to(self.device)
            a = data["a"][start:start+self.args.batch_size].to(self.device)
            b = data["b"][start:start+self.args.batch_size].to(self.device)
            baseline, cache = self.patch.forward(ids, data["positions"], capture=True)
            # Pairs are adjacent and every batch contains complete pairs.
            swap = torch.arange(len(ids), device=self.device) ^ 1
            donor = {k:(z[:, swap], al[swap]) for k,(z,al) in cache.items()}
            donor_logits = baseline[swap]
            def record(label, logits):
                for k,v in measures(logits, a, b, baseline).items():
                    values[label].setdefault(k, []).extend(v)
            record("reference", baseline)
            record("donor_baseline", donor_logits)
            for spec in specs:
                logits, _ = self.patch.forward(ids, data["positions"], spec=spec,
                                              donor=cache if spec["identity"] else donor,
                                              recipient=cache)
                if spec["identity"]:
                    max_identity = max(max_identity, (logits-baseline).abs().max().item())
                record(spec["name"], logits)
            if start == 0 or (start+self.args.batch_size) % 32 == 0:
                print(f"{stage}: {start+len(ids)}/{len(data['ids'])}, elapsed {time.time()-self.started:.1f}s", flush=True)
        ref, don = values["reference"], values["donor_baseline"]
        correct = np.asarray(ref["recipient_vocab_accuracy"]).reshape(-1,2).all(1).tolist()
        record = dict(n_pairs=len(data["ids"])//2, period=data["period"], query=data["query"],
                      patched_positions=data["positions"], seed=data["seed"],
                      both_baselines_correct_pairs=sum(correct), max_identity_logit_error=max_identity,
                      specs=specs, arms={})
        for name, vals in values.items():
            record["arms"][name] = dict(values=vals, summary=summaries(vals, ref, don, correct, data["control"]))
        self.result["stages"][stage] = record
        self.save()
        assert max_identity == 0., f"Identity intervention changed logits: {max_identity}"
        print("DONE", stage, "reference acc", np.mean(ref["recipient_vocab_accuracy"]), flush=True)
        return record


def edge_controls(edges, heads, seed):
    """Preserve layer, stream, source, shared head structure and edge count."""
    rng = random.Random(seed)
    mappings = {}
    for layer in sorted({e[0] for e in edges}):
        old = sorted({e[2] for e in edges if e[0] == layer})
        candidates = [h for h in range(heads) if h not in old]
        assert len(candidates) >= len(old)
        mappings[layer] = dict(zip(old, rng.sample(candidates, len(old))))
    return [(l,s,mappings[l][h],src) for l,s,h,src in edges]


def write_report(out):
    result = json.loads((out / 'results.json').read_text())
    controls = {}
    rng = np.random.default_rng(20260929)
    for split in ('heldout', 'transfer', 'unchanged_query'):
        stage = result['stages'][split]
        controls[split] = {}
        for k in result['selection']['topk_sizes']:
            selected = stage['arms'][f'top{k}/message']['values']
            for control in range(3):
                random_values = stage['arms'][f'top{k}/random{control}']['values']
                comparison = {}
                for metric in ('donor_margin', 'donor_two_choice_accuracy', 'donor_vocab_accuracy', 'recipient_logp'):
                    delta = (np.asarray(selected[metric])-np.asarray(random_values[metric])).reshape(-1,2).mean(1)
                    idx = rng.integers(len(delta),size=(4000,len(delta)))
                    comparison[metric] = dict(delta=float(delta.mean()),paired_95ci=np.quantile(delta[idx].mean(1),[.025,.975]).tolist())
                controls[split][f'top{k}_minus_random{control}'] = comparison
    (out/'control_comparisons.json').write_text(json.dumps(controls,indent=2)+'\n')
    def ci(est):
        lo,hi=est['paired_95ci']
        return f"{est['mean']:.3f} [{lo:.3f}, {hi:.3f}]"
    counts = {k:result['stages'][k]['n_pairs'] for k in ('discovery_broad','heldout','transfer')}
    selected_layers = sorted({e[0] for e in result['selection']['ordered_edges']})
    first_eight = result['selection']['ordered_edges'][:8]
    first_streams = sorted({e[1] for e in first_eight})
    first_sources = sorted({e[3] for e in first_eight})
    lines = ['# Counterfactual routing-message interchange — 300M FourWay', '',
             'This is a fixed-checkpoint, inference-only synthetic-copy experiment. It does not establish a dense-versus-DAG interpretability advantage or natural-language circuit semantics.', '',
             'The experiment finds a transferable information-read interface: exchanging projected source content changes the copied answer, while exchanging the coefficients alone has little effect. The masks were selected once on discovery data. All predeclared top-k sizes and three controls per size are reported below; top-8 is an illustrative sparse setting, not a held-out optimum.', '',
             'Three full token-marginal random blocks are followed by a shared unfinished fourth prefix. A counterfactual pair differs only at three historical occurrences of one value; the next-token query prefix is identical. Both exchange directions are evaluated. Interventions affect reads at those three past value positions, never query states or final outputs.', '',
             'These historical value-token locations are supplied by the task construction (oracle locations), not discovered by the mask search. This tests information transmission at known locations. The two candidate value tokens are sampled from the WikiText training-token marginal subject to being distinct and absent from the rest of their random block; hence they are also absent from the unfinished query prefix.', '',
             f"Discovery uses {counts['discovery_broad']} independent pairs at period 64. Held-out evaluation uses {counts['heldout']} new pairs at period 64; distance transfer uses {counts['transfer']} new pairs at period 128. The unchanged-query control reuses held-out contexts but asks for a different, unchanged block position. Exact token IDs are retained in datasets.json.", '',
             'Layer selection (best two KV layers), head selection (best four heads), and edge ranking all use only discovery donor-margin increases from complete-message interventions. Top-k masks are single-edge rankings, not proven minimal sufficient circuits. Random controls preserve layer, stream, source, head grouping, and edge count while moving the selected heads to other heads; neither edit norm nor language-model damage is matched.', '',
             'For projected source content z and effective coefficient a=pred+corr, alpha/content/message exchange uses donor-a times current-recipient-z, current-recipient-a times donor-z, or donor-a times donor-z respectively. Dynamic arms recompute downstream correction normally. Fixed arms hold recipient effective coefficients at all layers/streams/positions before the selected exchange, so they measure content flow conditional on the recipient routing.', '',
             'Confidence intervals are unadjusted 95% pair bootstrap intervals (4,000 resamples), grouping the two directions of each independent base block. Margin recovery is (patched-minus-recipient donor margin)/(donor-minus-recipient donor margin), where donor margin is log p(donor answer)-log p(recipient answer). Values can lie outside [0,1].', '',
             f"Identity intervention and last-token projection checks: {result['checks']}; maximum identity logit difference on all stages is {max(s['max_identity_logit_error'] for s in result['stages'].values())}.", '',
             '## Selected paths', '',
             f"Layers: {result['selection']['layers']}; heads (zero-based layer/head): {result['selection']['heads']}. Exact ordered edges and discovery scores are in results.json.", '',
             f"The broad discovery scan includes routed layers 1 through 11. Selected heads and top-k edges use layers {selected_layers}; final layer 11 is {'included' if 11 in selected_layers else 'absent'}. The first eight edges use streams {first_streams} and sources {first_sources}. Source 0 is the embedding, and source i>0 is layer-(i-1) output. The 300M checkpoint has no V norm. These are pathway interventions in a model whose upstream computation remains present, not eight self-contained computational operations.", '']
    lines += ['Each selected edge is instantiated at all three oracle positions: top-8 therefore replaces 24 spatiotemporal message instances, each carrying a 64-dimensional head projection.', '']
    for split in ('heldout','transfer'):
        stage=result['stages'][split]
        ref=stage['arms']['reference']['summary']
        lines += [f'## {split}', '',
                  f"Baseline full-vocabulary accuracy: {100*ref['recipient_vocab_accuracy']['mean']:.2f}%; both baseline directions correct: {stage['both_baselines_correct_pairs']}/{stage['n_pairs']} pairs. Period {stage['period']}.", '',
                  '| Arm | Margin recovery [95% CI] | Donor two-choice accuracy | Donor full-vocab accuracy | Donor logp |',
                  '|---|---:|---:|---:|---:|']
        for name, arm_result in stage['arms'].items():
            if name.startswith('identity'):
                continue
            s=arm_result['summary']
            lines.append(f"| {name} | {ci(s['margin_recovery'])} | {100*s['donor_two_choice_accuracy']['mean']:.2f}% | {100*s['donor_vocab_accuracy']['mean']:.2f}% | {s['donor_logp']['mean']:.3f} |")
        lines += ['']
    stage=result['stages']['unchanged_query']
    lines += ['## Unchanged-query control', '',
              'The correct answer is unchanged across donor and recipient. Here donor_probability means the probability of the wrong changed value from the donor; it is not a donor-correctness score. This control measures selectivity across two synthetic copy targets, not general language-model capability.', '',
              '| Arm | Correct-answer logp change | Correct full-vocab accuracy | Wrong changed-value probability |',
              '|---|---:|---:|---:|']
    for name, arm_result in stage['arms'].items():
        if name.startswith('identity'):
            continue
        s=arm_result['summary']
        e=s['recipient_logp'];lo,hi=e['paired_95ci']
        lines.append(f"| {name} | {e['delta']:+.4f} [{lo:+.4f}, {hi:+.4f}] | {100*s['recipient_vocab_accuracy']['mean']:.2f}% | {s['donor_probability']['mean']:.5f} |")
    lines += ['', '## Scope', '',
              'The native graph omits substantial upstream computation inside each source state. An edited edge is a projected source-state read, not a complete isolated circuit. Full-dimensional Q/K normalization couples heads, so an intervention can change other heads through normalization. A top-k result must be compared with direct paired random-control contrasts in control_comparisons.json; differences in intervention norm and capability cost remain unresolved.', '',
              'All metrics and per-directed-case values, including recipient and donor logp, donor-minus-recipient margin, two-choice probability, full-vocabulary correctness, and output KL relative to the recipient, are retained in results.json. No model weights were trained or saved.', '',
              'Run: `CUDA_VISIBLE_DEVICES=2 /scratch/yurenh2/venvs/dagformer-eval-20260917/bin/python scripts/circuit_message_patching.py --seed 20260930 --discovery-pairs 32`.', '']
    (out/'README.md').write_text('\n'.join(lines))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", type=Path, default=Path("checkpoints/pr_sync_20260917/300m-dagformer"))
    ap.add_argument("--token-cache", default="checkpoints/pr_sync_20260917/eval_corpora/wikitext_train.pt")
    ap.add_argument("--out", type=Path, default=Path("experiments/results/circuit_interpretability_20260928/message_patching"))
    ap.add_argument("--discovery-pairs", type=int, default=16)
    ap.add_argument("--test-pairs", type=int, default=64)
    ap.add_argument("--transfer-pairs", type=int, default=64)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--seed", type=int, default=20260929)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--report-only", action="store_true")
    args = ap.parse_args()
    if args.report_only:
        write_report(args.out)
        return
    assert args.batch_size % 2 == 0
    torch.manual_seed(args.seed)
    torch.set_num_threads(4)
    corpus, _ = load_eval_ids(args.token_cache)
    if args.smoke:
        args.discovery_pairs = args.test_pairs = args.transfer_pairs = 2
    data = dict(discovery=make_data(corpus,args.discovery_pairs,64,args.seed),
                heldout=make_data(corpus,args.test_pairs,64,args.seed+1),
                transfer=make_data(corpus,args.transfer_pairs,128,args.seed+2),
                unchanged_query=make_data(corpus,args.test_pairs,64,args.seed+1,query=15))
    experiment = Experiment(args)
    experiment.result["protocol"] = dict(
        task="Three repetitions of a token-marginal random block, then a shared unfinished prefix; paired historical values differ.",
        positions="Only the three historical value-token positions are patched; query states and final outputs are never directly patched.",
        modes={"alpha":"donor effective alpha times current recipient projected content",
               "content":"current recipient effective alpha times donor projected content",
               "message":"donor alpha times donor projected content"},
        downstream="Dynamic arms recompute downstream correction normally. Fixed arms replace effective alpha at every position/layer/stream by baseline recipient alpha before selected edits.",
        selection="Discovery-only single layer, then head, then edge interventions; final masks ordered by donor-margin increase, no heldout selection.",
        controls="Three head permutations preserving layer/stream/source and edge count, mapping all selected heads to different heads; not edit-norm or NLL matched.",
        uncertainty="4000 paired bootstrap resamples of independent base blocks, pooling both interchange directions; fixed checkpoint, unadjusted CIs.",
        limits="Synthetic copy and period transfer, not natural-language semantic binding or cross-template proof; projected source contents carry upstream computations; Q/K norms couple heads.")
    compact = {k:{**{x:v for x,v in d.items() if x not in ('ids','a','b')},
                  "ids":d['ids'].tolist(),"a":d['a'].tolist(),"b":d['b'].tolist()} for k,d in data.items()}
    (args.out/"datasets.json").write_text(json.dumps(compact)+"\n")
    # Check that projecting only the final hidden state preserves real logits.
    with torch.inference_mode():
        ids = data['discovery']['ids'][:2].cuda()
        full = experiment.model(ids, experiment.predictor(ids))[:,-1].float()
    with experiment.patch.installed():
        short, _ = experiment.patch.forward(ids,data['discovery']['positions'])
        error=(full-short).abs().max().item()
        experiment.result['checks']['last_logit_projection_max_error']=error
        assert error == 0., error
        broad = [arm('identity',identity=True), arm('identity_fixed',identity=True,fixed=True)]
        broad += [arm(f'all/{mode}/{"fixed" if fixed else "dynamic"}',mode=mode,fixed=fixed)
                  for fixed in (False,True) for mode in ('alpha','content','message')]
        broad += [arm(f'stream/{s}',streams=s) for s in ('q','k','v','r','kv','qkv')]
        broad += [arm(f'layer/{l}/kv',streams='kv',layers=[l]) for l in range(1,experiment.L)]
        stage=experiment.evaluate('discovery_broad',data['discovery'],broad)
        layers=sorted(range(1,experiment.L), key=lambda l: stage['arms'][f'layer/{l}/kv']['summary']['donor_margin']['delta'],reverse=True)[:2]
        head_specs=[arm(f'head/{l}/{h}',streams='kv',heads=[(l,h)]) for l in layers for h in range(experiment.H)]
        heads_stage=experiment.evaluate('discovery_heads',data['discovery'],head_specs)
        selected_heads=sorted([(l,h) for l in layers for h in range(experiment.H)],
                              key=lambda p:heads_stage['arms'][f'head/{p[0]}/{p[1]}']['summary']['donor_margin']['delta'],reverse=True)[:4]
        edges=[(l,s,h,src) for l,h in selected_heads for s in 'kv' for src in range(l+1)]
        def edge_name(edge): return 'edge/'+'/'.join(map(str,edge))
        edge_specs=[arm(edge_name(e),edges=[e],streams='kv') for e in edges]
        edge_stage=experiment.evaluate('discovery_edges',data['discovery'],edge_specs)
        ordered=sorted(edges,key=lambda e:edge_stage['arms'][edge_name(e)]['summary']['donor_margin']['delta'],reverse=True)
        sizes=[k for k in (4,8,16,32) if k<=len(ordered)]
        selection=dict(layers=layers,heads=selected_heads,ordered_edges=ordered,topk_sizes=sizes,
                       selection_metric='discovery donor-margin increase under complete-message interchange',
                       edge_scores=[edge_stage['arms'][edge_name(e)]['summary']['donor_margin']['delta'] for e in ordered])
        experiment.result['selection']=selection
        final=broad[:8]+[arm(f'stream/{s}',streams=s) for s in ('q','k','v','r','kv','qkv')]
        final += [arm(f'layer/{l}/kv',streams='kv',layers=[l]) for l in layers]
        final += [arm('selected_four_heads',streams='kv',heads=selected_heads)]
        for k in sizes:
            selected=ordered[:k]
            for mode in ('alpha','content','message'):
                final.append(arm(f'top{k}/{mode}',mode=mode,edges=selected,streams='kv'))
            final.append(arm(f'top{k}/message_fixed',edges=selected,streams='kv',fixed=True))
            for seed in range(3):
                final.append(arm(f'top{k}/random{seed}',edges=edge_controls(selected,experiment.H,args.seed+100+seed),streams='kv'))
        for split in ('heldout','transfer','unchanged_query'):
            experiment.evaluate(split,data[split],final)
    experiment.result['complete']=True
    experiment.save()
    write_report(args.out)
    print('COMPLETE',args.out, 'seconds',time.time()-experiment.started,flush=True)


if __name__ == '__main__':
    main()
