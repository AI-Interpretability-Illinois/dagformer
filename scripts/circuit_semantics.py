"""Counterfactual semantic-variable interchange in frozen FourWay messages.

Run prepare, discovery, then confirmation. All source selection uses discovery.
Each block contains two lexical frames and all four subject/distractor numbers.
The three donor permutations flip subject number, flip distractor number, or
change lexical identities while preserving both numbers. No model is trained.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from contextlib import contextmanager
import gzip
import json
from pathlib import Path
import random
import time

import numpy as np
import torch
from transformers import AutoTokenizer

from interp_audit_common import AuditModel, NOUNS, ROOT, TOKENIZER, write_json
from interp_adapter_audit import load_adapter


OUT = ROOT / 'experiments/results/circuit_semantics_20260929'
PRIOR = ROOT / 'experiments/results/interp_followup_20260929'
VERBS = [('is', 'are'), ('has', 'have'), ('does', 'do'), ('was', 'were'),
         ('seems', 'seem'), ('works', 'work'), ('likes', 'like')]
STATES = ['original', 'adapted', 'repaired']
CONTRASTS = {'subject': 2, 'distractor': 1, 'lexical': 4}


def read_gz(path):
    with gzip.open(path, 'rt') as f:
        return json.load(f)


def save_gz(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix('.tmp')
    with gzip.open(temp, 'wt') as f:
        json.dump(obj, f, allow_nan=False)
    temp.replace(path)


def selected_edges():
    record = read_gz(PRIOR / 'runs/full_s20260930/results.json.gz')
    return [dict(index=i, **record['mask_unit_labels'][i])
            for i in record['all_masks']['sva_75']]


def make_data(out, seed_offset=0, counts=None, excluded_prompts=None):
    if (out / 'data.json.gz').exists():
        return read_gz(out / 'data.json.gz')
    tok = AutoTokenizer.from_pretrained(str(TOKENIZER), local_files_only=True)
    nouns = [x for x in NOUNS if all(len(tok.encode(' '+s, add_special_tokens=False)) == 1 for s in x)]
    seen = set(excluded_prompts or [])
    old = ROOT / 'experiments/results/interp_audit_20260929/datasets.json'
    if old.exists():
        seen.update(x['prompt'] for rows in json.loads(old.read_text()).values() for x in rows)
    for rows in read_gz(PRIOR / 'extra_tests.json.gz').values():
        seen.update(x['prompt'] for x in rows)
    base_templates = ['The young {s} near the {d}', 'The tired {s} beside the {d}',
                      'The quiet {s} behind the {d}', 'The friendly {s} close to the {d}']
    transfer_templates = ['The {s}, despite the noisy {d},',
                          'The {s} that the {d} could clearly see',
                          'The {s} from the village near the {d}',
                          'The {s} whom the {d} had invited yesterday']
    role_templates = ['Near the {d}, the {s}', 'Behind the {d}, the {s}',
                      'According to the {d}, the {s}', 'Beside the {d}, the {s}']
    specs = [('discovery', 24, nouns[:16], base_templates),
             ('heldout', 48, nouns[:16], base_templates),
             ('transfer', 48, nouns[16:], transfer_templates),
             ('role', 48, nouns[16:], role_templates)]
    data = {}
    for split_id, (split, n_blocks, pool, templates) in enumerate(specs):
        n_blocks = (counts or {}).get(split, n_blocks)
        rng = random.Random(202609290700 + split_id + seed_offset)
        rows = []
        for block in range(n_blocks):
            template = templates[block % len(templates)]
            while True:
                words = rng.sample(pool, 4)
                block_rows = []
                for lexical in range(2):
                    s, d = words[2*lexical:2*lexical+2]
                    for sn in range(2):
                        for dn in range(2):
                            prompt = template.format(s=s[sn], d=d[dn])
                            encoded = tok(prompt, add_special_tokens=False, return_offsets_mapping=True)
                            positions = {}
                            for role, word in [('subject', s[sn]), ('distractor', d[dn])]:
                                start = prompt.index(word)
                                end = start + len(word)
                                ps = [i for i, (a, b) in enumerate(encoded['offset_mapping']) if a < end and b > start]
                                assert len(ps) == 1, (prompt, role, ps)
                                positions[role] = ps[0]
                            positions['last'] = len(encoded['input_ids'])-1
                            block_rows.append(dict(prompt=prompt, ids=encoded['input_ids'],
                                positions=positions, subject_number=sn, distractor_number=dn,
                                subject=s[sn], distractor=d[dn], lexical=lexical,
                                template=template, block=block, split=split))
                if len({len(x['ids']) for x in block_rows}) != 1:
                    continue
                if any(x['prompt'] in seen for x in block_rows):
                    continue
                for i, row in enumerate(block_rows):
                    for kind, flip in CONTRASTS.items():
                        donor = block_rows[i ^ flip]
                        assert row['positions'] == donor['positions']
                        if kind == 'subject':
                            assert row['distractor_number'] == donor['distractor_number']
                            assert row['subject_number'] != donor['subject_number']
                        elif kind == 'distractor':
                            assert row['subject_number'] == donor['subject_number']
                            assert row['distractor_number'] != donor['distractor_number']
                        else:
                            assert (row['subject_number'], row['distractor_number']) == (donor['subject_number'], donor['distractor_number'])
                rows.extend(block_rows)
                seen.update(x['prompt'] for x in block_rows)
                break
        data[split] = rows
    verb_ids = [[tok.encode(' '+w, add_special_tokens=False)[0] for w in pair] for pair in VERBS]
    for pair in VERBS:
        assert all(len(tok.encode(' '+w, add_special_tokens=False)) == 1 for w in pair)
    result = dict(splits=data, verb_pairs=VERBS, verb_ids=verb_ids)
    save_gz(out / 'data.json.gz', result)
    write_json(out / 'protocol.json', dict(
        created_before_model_evaluation=True,
        checkpoint='Complete 300M FourWay + local correction, step9000; native Q/K norm, no V norm.',
        adapter_seed=20260930,
        states='Original predictor; full trained predictor; exact functional rollback of the previously selected 75 predictor outputs. Local correction remains dynamic in all states.',
        source_mask='Prior full_s20260930 all_masks.sva_75; this is a repair mask, not a set of independently labeled semantic circuits.',
        donor_design='Eight-example blocks: two lexical frames, each with subject number x distractor number. Subject, distractor and same-number lexical interchanges are separate conditions.',
        positions='Single-token noun forms; semantic-role positions recorded. Donors in each block have identical token lengths and positions. Padding is excluded from patches.',
        selection='Discovery only; original-model is/are signed subject effect selects V groups and individual V edges. All verbs and states reported; no heldout selection.',
        metrics='Plural-minus-singular logits, full-vocabulary candidate log probabilities, grammatical and counterfactual candidate accuracy, baseline donor recovery; block-bootstrap intervals.',
        limits='Controlled sentences, one pretrained checkpoint and initially one adapter seed. Verb pairs are scored on the same prefix and are not seven independent input datasets.',
        counts={s:dict(blocks=len(rows)//8,prompts=len(rows)) for s,rows in data.items()}))
    return result


def arm(name, edges, mode='content', site='all', contrast='subject', **extra):
    return dict(name=name, edges=edges, mode=mode, site=site, contrast=contrast, **extra)


def basic_arms(edges):
    result = []
    for group, chosen in [('mask75', edges), ('maskV', [e for e in edges if e['stream']=='v'])]:
        for mode in ['content', 'alpha', 'message']:
            for site in ['subject', 'distractor', 'last', 'all']:
                result.append(arm(f'{group}/{mode}/{site}/subject', chosen, mode, site))
    for contrast in ['distractor', 'lexical']:
        for group, chosen in [('mask75', edges), ('maskV', [e for e in edges if e['stream']=='v'])]:
            result.append(arm(f'{group}/content/all/{contrast}', chosen, contrast=contrast))
    return result


def discovery_arms(edges):
    result = basic_arms(edges)
    groups = defaultdict(list)
    for edge in edges:
        if edge['stream'] == 'v':
            groups[(edge['layer'], edge['head'])].append(edge)
            result.append(arm(f'edge/{edge["index"]}', [edge]))
    for (layer, head), es in sorted(groups.items()):
        result.append(arm(f'group/L{layer}H{head}', es))
    return result


def decompose_arms(edges):
    """Additional discovery after the initial V-focused scan; no test access."""
    result = []
    for stream in ['q','k','r']:
        chosen = [e for e in edges if e['stream']==stream]
        for mode in ['content','alpha','message']:
            for site in ['subject','distractor','last','all']:
                result.append(arm(f'mask{stream.upper()}/{mode}/{site}/subject', chosen, mode, site))
        for contrast in ['distractor','lexical']:
            result.append(arm(f'mask{stream.upper()}/content/all/{contrast}', chosen, contrast=contrast))
        groups = defaultdict(list)
        for edge in chosen:
            result.append(arm(f'edge/{edge["index"]}', [edge]))
            groups[(edge['layer'],edge['head'])].append(edge)
        for (layer,head), es in sorted(groups.items()):
            result.append(arm(f'group/{stream}/L{layer}H{head}', es))
    return result


class MessageExperiment:
    def __init__(self, adapter, edges):
        self.runner = AuditModel(memory_fraction=.65)
        self.runner.copy_baseline_predictor()
        self.adapter_metadata = load_adapter(self.runner, 'predictor', adapter)
        self.runner.model.eval().requires_grad_(False)
        self.runner.predictor.eval().requires_grad_(False)
        self.edges = edges
        self.gate = torch.ones(len(self.runner.layout), device='cuda')
        self.gate[[e['index'] for e in edges]] = 0
        self.native_einsum = torch.einsum
        self.spec = None
        self.cache = None
        self.capture = False
        self.qkv_calls = self.r_calls = 0

    @contextmanager
    def installed(self):
        torch.einsum = self.einsum
        handles = []
        for layer, module in enumerate(self.runner.base.model.layers):
            handles.append(module.self_attn.o_proj.register_forward_pre_hook(self.head_hook(layer)))
        try:
            yield self
        finally:
            torch.einsum = self.native_einsum
            for handle in handles:
                handle.remove()

    def head_hook(self, layer):
        def hook(module, args):
            x = args[0]
            if self.capture:
                self.cache[('attention', layer)] = x.detach().clone()
            reset = None
            if self.spec is not None:
                reset = next((r for r in self.spec.get('resets', []) if r['layer']==layer), None)
                if self.spec.get('reset_layer') == layer:
                    reset = dict(heads=self.spec['reset_heads'], site=self.spec.get('reset_site','all'))
            if reset is not None:
                h = x.reshape(*x.shape[:2], self.runner.cfg['num_attention_heads'], -1).clone()
                original = self.cache[('attention', layer)].reshape_as(h)
                heads = reset['heads']
                if reset.get('site', 'all') == 'last':
                    b = torch.arange(len(h), device=h.device)
                    pos = self.runner.last_positions
                    for head in heads:
                        h[b, pos, head] = original[b, pos, head]
                else:
                    h[:, :, heads] = original[:, :, heads]
                return (h.reshape_as(x),)
            return None
        return hook

    def einsum(self, equation, *operands, **kwargs):
        eq = equation.replace(' ', '') if isinstance(equation, str) else ''
        if eq not in ('lbthd,bthl->bhtd', 'lbtd,btl->btd'):
            return self.native_einsum(equation, *operands, **kwargs)
        z, alpha = operands
        layer = z.shape[0]-1
        if eq == 'lbthd,bthl->bhtd':
            stream = 'qkv'[self.qkv_calls % 3]
            assert layer == self.qkv_calls//3+1
            self.qkv_calls += 1
        else:
            stream = 'r'
            assert layer == self.r_calls+1
            self.r_calls += 1
        key = (layer, stream)
        if self.capture:
            self.cache[key] = (z.detach().clone(), alpha.detach().clone())
        output = self.native_einsum(equation, z, alpha)
        if self.spec is None:
            return output
        edges = self.active.get(key, [])
        if not edges:
            return output
        donor_z, donor_alpha = self.cache[key]
        donor_z = donor_z[:, self.permutation]
        donor_alpha = donor_alpha[self.permutation]
        mode = self.spec['mode']
        new_z = donor_z if mode in ('content', 'message') else z
        new_alpha = donor_alpha if mode in ('alpha', 'message') else alpha
        if stream != 'r':
            mask = torch.zeros(z.shape[0], z.shape[3], device=z.device, dtype=torch.float32)
            for edge in edges:
                mask[edge['src'], edge['head']] = 1
            if self.spec.get('direction') is not None:
                direction = self.spec['direction']
                direction = direction / direction.norm().clamp_min(1e-8)
                dz = new_z.float()-z.float()
                new_z = z.float() + (dz*direction).sum(-1, keepdim=True)*direction
            old_a = alpha.permute(3, 0, 1, 2)[..., None].float()
            new_a = new_alpha.permute(3, 0, 1, 2)[..., None].float()
            delta = ((new_z.float()*new_a-z.float()*old_a)*mask[:, None, None, :, None]).sum(0)
            delta = delta.permute(0, 2, 1, 3)*self.site_mask[:, None, :, None]
        else:
            mask = torch.zeros(z.shape[0], device=z.device)
            for edge in edges:
                mask[edge['src']] = 1
            if self.spec.get('direction') is not None:
                direction = self.spec['direction']
                direction = direction / direction.norm().clamp_min(1e-8)
                dz = new_z.float()-z.float()
                new_z = z.float() + (dz*direction).sum(-1, keepdim=True)*direction
            old_a = alpha.permute(2, 0, 1)[..., None].float()
            new_a = new_alpha.permute(2, 0, 1)[..., None].float()
            delta = ((new_z.float()*new_a-z.float()*old_a)*mask[:, None, None, None]).sum(0)
            delta = delta*self.site_mask[:, :, None]
        return (output.float()+delta).to(output.dtype)

    def forward(self, rows, state, spec=None, capture=False):
        self.qkv_calls = self.r_calls = 0
        self.spec, self.capture = spec, capture
        if capture:
            self.cache = {}
        if spec is not None:
            self.active = defaultdict(list)
            for edge in spec['edges']:
                self.active[(edge['layer'], edge['stream'])].append(edge)
            flip = 0 if spec.get('identity') else CONTRASTS[spec['contrast']]
            self.permutation = torch.arange(len(rows), device='cuda') ^ flip
            self.site_mask = torch.zeros(len(rows), max(len(x['ids']) for x in rows), device='cuda')
            for i, row in enumerate(rows):
                if spec['site'] == 'all':
                    self.site_mask[i, :len(row['ids'])] = 1
                else:
                    self.site_mask[i, row['positions'][spec['site']]] = 1
        gate = self.gate if state == 'repaired' else None
        logits = self.runner.forward(rows, mode='base' if state=='original' else 'adapted', alpha_gate=gate)
        assert self.qkv_calls == 3*(self.runner.cfg['num_hidden_layers']-1)
        assert self.r_calls == self.runner.cfg['num_hidden_layers']-1
        self.capture = False
        self.spec = None
        return logits


def values(logits, ids):
    selected = logits[:, ids]
    return dict(margins=(selected[:, :, 1]-selected[:, :, 0]).detach().cpu().tolist(),
                logprobs=(selected-logits.logsumexp(-1)[:, None, None]).detach().cpu().tolist())


def extend(destination, source):
    for key, values_ in source.items():
        destination.setdefault(key, []).extend(values_)


def run_set(experiment, rows, state, arms, verb_ids, batch_size=16, validate=False):
    result = dict(baseline={}, arms={a['name']:{} for a in arms})
    ids = torch.tensor(verb_ids, device='cuda')
    started = time.time()
    with torch.no_grad():
        for start in range(0, len(rows), batch_size):
            batch = rows[start:start+batch_size]
            baseline = experiment.forward(batch, state, capture=True)
            extend(result['baseline'], values(baseline, ids))
            if validate and start == 0:
                checks = {}
                for mode in ['alpha', 'content', 'message']:
                    spec = arm('identity', experiment.edges, mode, identity=True)
                    patched = experiment.forward(batch, state, spec=spec)
                    checks[mode] = float((baseline-patched).abs().max())
                assert max(checks.values()) == 0, checks
                result['identity_max_logit_error'] = checks
            for spec in arms:
                logits = experiment.forward(batch, state, spec=spec)
                extend(result['arms'][spec['name']], values(logits, ids))
            if start == 0 or (start//batch_size+1)%4 == 0 or start+batch_size >= len(rows):
                print(json.dumps(dict(state=state, split=rows[0]['split'], done=min(start+batch_size,len(rows)),
                    total=len(rows), arms=len(arms), seconds=round(time.time()-started,1))), flush=True)
    result['elapsed_seconds'] = time.time()-started
    return result


def block_stat(x):
    x = np.asarray(x, dtype=float)
    blocks = x.reshape(-1, 8).mean(1)
    rng = np.random.default_rng(20260929)
    bs = blocks[rng.integers(len(blocks), size=(2000, len(blocks)))].mean(1)
    return dict(mean=float(x.mean()), ci95=np.quantile(bs,[.025,.975]).tolist(), blocks=len(blocks))


def summarize(rows, result, arms):
    sn = np.array([r['subject_number'] for r in rows])
    dn = np.array([r['distractor_number'] for r in rows])
    sign = 2*sn-1
    baseline = np.asarray(result['baseline']['margins'])
    summary = dict(baseline={}, arms={})
    for j, pair in enumerate(VERBS):
        margin = baseline[:,j]
        summary['baseline']['/'.join(pair)] = dict(grammar=block_stat(margin*sign>0),
            distractor=block_stat(margin*(2*dn-1)>0), grammar_margin=block_stat(margin*sign))
    for spec in arms:
        name = spec['name']
        patched = np.asarray(result['arms'][name]['margins'])
        permutation = np.arange(len(rows)) ^ CONTRASTS[spec['contrast']]
        target_sn = sn[permutation] if spec['contrast']=='subject' else sn
        result_ = {}
        for j, pair in enumerate(VERBS):
            original, changed, donor = baseline[:,j], patched[:,j], baseline[permutation,j]
            delta = changed-original
            target_sign = 2*target_sn-1
            signed_delta = delta*target_sign
            gap = (donor-original)*target_sign
            both = (original*sign>0)&(donor*(2*sn[permutation]-1)>0)
            result_['/'.join(pair)] = dict(
                counterfactual_grammar=block_stat(changed*target_sign>0),
                original_grammar=block_stat(changed*sign>0),
                donor_native_choice=block_stat(np.sign(changed)==np.sign(donor)),
                signed_effect=block_stat(signed_delta),
                absolute_effect=block_stat(np.abs(delta)),
                donor_gap=float(gap.mean()),
                donor_recovery=float(signed_delta.mean()/gap.mean()) if abs(gap.mean())>.05 else None,
                correct_pair_coverage=float(both.mean()),
                correct_pair_counterfactual=float((changed[both]*target_sign[both]>0).mean()) if both.any() else None)
        summary['arms'][name] = result_
    return summary


def choose_arms(discovery, candidates, edges, decomposition=None):
    scores = dict(discovery['original']['summary']['arms'])
    if decomposition is not None:
        scores.update(decomposition['original']['summary']['arms'])
        candidates = candidates + decompose_arms(edges)
    selected = {}
    for kind in ['group', 'edge']:
        options = [a for a in candidates if a['name'].startswith(kind+'/')]
        winner = max(options, key=lambda a:scores[a['name']]['is/are']['signed_effect']['mean'])
        selected[kind] = winner
    arms = basic_arms(edges)
    for stream in ['q','k','r']:
        chosen = [e for e in edges if e['stream']==stream]
        for mode in ['content','alpha','message']:
            for site in ['subject','all']:
                arms.append(arm(f'mask{stream.upper()}/{mode}/{site}/subject', chosen, mode, site))
    for kind, winner in selected.items():
        for contrast in CONTRASTS:
            for site in ['subject','distractor','last','all']:
                arms.append(arm(f'best_{kind}/content/{site}/{contrast}', winner['edges'], site=site, contrast=contrast))
        for mode in ['alpha','message']:
            arms.append(arm(f'best_{kind}/{mode}/all/subject', winner['edges'], mode=mode))
    return selected, arms


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--phase', choices=['prepare','discovery','decompose','confirmation'], required=True)
    p.add_argument('--out', type=Path, default=OUT)
    p.add_argument('--adapter', type=Path, default=Path('/scratch/yurenh2/interp_followup_20260929/runs/full_s20260930/weights/predictor_best.pt'))
    p.add_argument('--batch-size', type=int, default=16)
    args = p.parse_args()
    assert args.batch_size%8 == 0
    args.out.mkdir(parents=True, exist_ok=True)
    data = make_data(args.out)
    if args.phase == 'prepare':
        print({s:len(rows) for s,rows in data['splits'].items()})
        return
    edges = selected_edges()
    candidates = discovery_arms(edges)
    if args.phase == 'confirmation':
        discovery = read_gz(args.out/'discovery.json.gz')
        decomposition = read_gz(args.out/'decompose.json.gz') if (args.out/'decompose.json.gz').exists() else None
        selected, arms = choose_arms(discovery, candidates, edges, decomposition)
        write_json(args.out/'selection.json', dict(selection='Original discovery is/are signed subject effect only.', selected=selected, arms=arms))
    elif args.phase == 'decompose':
        arms = decompose_arms(edges)
        write_json(args.out/'decomposition_protocol.json', dict(
            reason='Initial discovery showed all-stream content transfer much stronger than the V-only mask. Examine Q/K/R before selecting a semantic-variable experiment.',
            data='Original discovery split only; no confirmation scores inspected.', arms=arms))
    else:
        arms = candidates
    splits = ['discovery'] if args.phase in ('discovery','decompose') else ['heldout','transfer','role']
    experiment = MessageExperiment(args.adapter, edges)
    with experiment.installed():
        for split in splits:
            stem = 'decompose' if args.phase=='decompose' else split
            path = args.out/f'{stem}.json.gz'
            result = read_gz(path) if path.exists() else {}
            for state in STATES:
                if state in result:
                    continue
                rows = data['splits'][split]
                result[state] = run_set(experiment, rows, state, arms, data['verb_ids'], args.batch_size,
                    validate=args.phase=='discovery' and state=='original')
                result[state]['summary'] = summarize(rows, result[state], arms)
                save_gz(path, result)
                write_json(args.out/f'{stem}_summary.json', {s:r['summary'] for s,r in result.items()})
    print('COMPLETE', args.phase, flush=True)


if __name__ == '__main__':
    main()
