"""Frozen-model soft-prompt optimization followed by sparse route mediation.

Exploratory experiment, not a benchmark of general honesty or instruction following.
The learned prefix is shared across all examples. Donor and recipient token positions
are identical; only donor embedding vectors differ. Sparse masks choose existing
donor route values, never fit new backbone weights or free route increments.
"""
from __future__ import annotations

import argparse
import copy
import json
import random
import subprocess
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

from interp_common import alpha_layout, flatten_alpha, load_elh, load_eval_ids, unflatten_alpha
from interp_editing import layer_chunks


def make_items(tokenizer, prefix, seed, n_pairs, objects, style):
    rng = random.Random(seed)
    colors = ['red', 'blue', 'green', 'black', 'white', 'yellow', 'brown']
    candidates = [tokenizer.encode(' ' + c, add_special_tokens=False) for c in colors]
    assert all(len(c) == 1 for c in candidates)
    candidates = [c[0] for c in candidates]
    items, seen = [], set()
    while len(items) < 2 * n_pairs:
        a, b = rng.sample(objects, 2)
        c, d = rng.sample(colors, 2)
        queried = rng.choice([a, b])
        key = (a, b, tuple(sorted([c, d])), queried)
        if key in seen:
            continue
        seen.add(key)
        pair = len(items) // 2
        for first, second in [(c, d), (d, c)]:
            target = first if queried == a else second
            if style == 'completion':
                prompt = (f'The {a} was {first}. The {b} was {second}. '
                          f'Later, someone described the {queried}. Its color was')
            elif style == 'qa':
                prompt = (f'There was a {first} {a} and a {second} {b} in the room.\n'
                          f'Question: What color was the {queried}?\nAnswer:')
            else:
                prompt = (f'A {a} and a {b} were nearby. The {a} was painted {first}; '
                          f'the {b} was painted {second}. The color of the {queried} was')
            ids = prefix + tokenizer.encode(prompt, add_special_tokens=False)
            items.append(dict(prompt=prompt, ids=ids, target=candidates[colors.index(target)],
                              distractor=candidates[colors.index(second if queried == a else first)],
                              answer=target, candidates=candidates, pair=pair, style=style))
    return items


def summarize(rows, reference=None):
    keys = ['logp', 'margin', 'candidate_correct', 'vocab_correct']
    keys += [k for k in ['binding_logp', 'binding_margin', 'binding_correct', 'color_logmass'] if k in rows[0]]
    result = {k: float(np.mean([r[k] for r in rows])) for k in keys}
    if reference is not None:
        rng = np.random.default_rng(20260928)
        for k in keys:
            delta = np.array([a[k] - b[k] for a, b in zip(rows, reference)])
            # Counterfactual color-swap pairs are one bootstrap unit.
            pair_delta = delta.reshape(-1, 2).mean(1)
            draws = pair_delta[rng.integers(len(pair_delta), size=(4000, len(pair_delta)))].mean(1)
            result[k + '_delta'] = float(delta.mean())
            result[k + '_delta_95ci'] = np.quantile(draws, [.025, .975]).tolist()
    return result


class Pilot:
    def __init__(self, args):
        self.args = args
        self.device = torch.device('cuda')
        self.tok = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)
        self.elh = load_elh()
        self.cfg = self.elh.load_config(str(args.model / 'config.yaml'))
        self.model, self.predictor = self.elh.load_fourway(str(args.model / 'checkpoint.pt'),
                                                        self.cfg, self.device)
        self.L, self.H = self.cfg['num_hidden_layers'], self.cfg['num_attention_heads']
        self.chunks = layer_chunks(self.L, self.H)
        self.E = self.chunks[-1][1]
        self.labels = [dict(channel=c, **row) for c in ['pred', 'corr']
                       for row in alpha_layout(self.L, self.H)]
        initial = self.tok.encode('Here is a short passage to read.\n', add_special_tokens=False)
        self.prefix = (initial * args.prefix_tokens)[:args.prefix_tokens]
        pids = torch.tensor(self.prefix, device=self.device)
        self.initial = {
            'base': self.model.olmo.model.embed_tokens(pids).detach().float(),
            'pred': self.predictor.embed(pids).detach().float(),
        }
        self.soft = torch.nn.ParameterDict({k: torch.nn.Parameter(v.clone())
                                           for k, v in self.initial.items()})
        self.active = False
        self.mode = args.prefix_target
        self.capture = False
        self.captured = []
        self.donor = None
        self.mask = None
        self.scope = 'content'
        self.handles = []
        for name, embedding in [('base', self.model.olmo.model.embed_tokens),
                                ('pred', self.predictor.embed)]:
            def embedding_hook(module, inputs, output, name=name):
                enabled = self.mode == 'both' or self.mode == name
                if not (self.active and enabled):
                    return output
                prefix = self.soft[name].to(output.dtype).unsqueeze(0).expand(output.shape[0], -1, -1)
                return torch.cat([prefix, output[:, len(self.prefix):]], dim=1)
            self.handles.append(embedding.register_forward_hook(embedding_hook))
        for i, mlp in enumerate(self.model.correction_mlps):
            def corr_hook(module, inputs, output, i=i):
                if self.capture:
                    self.captured.append(output.detach())
                if self.donor is not None:
                    a, b = self.chunks[i]
                    mask = self.mask[self.E+a:self.E+b].view(1, 1, -1)
                    if self.scope == 'content':
                        position = torch.arange(output.shape[1], device=output.device) >= len(self.prefix)
                        mask = mask * position.view(1, -1, 1)
                    return output + mask * (self.donor['corr'][..., a:b] - output)
                return output
            self.handles.append(mlp.register_forward_hook(corr_hook))

    def batch(self, items):
        lengths = torch.tensor([len(it['ids']) for it in items], device=self.device)
        ids = torch.zeros(len(items), int(lengths.max()), dtype=torch.long, device=self.device)
        for i, item in enumerate(items):
            ids[i, :len(item['ids'])] = torch.tensor(item['ids'], device=self.device)
        return ids, lengths

    def forward(self, items, soft=False, capture=False, mask=None, scope='content'):
        ids, lengths = self.batch(items)
        self.active, self.capture, self.captured = soft, capture, []
        self.scope, self.mask, self.donor = scope, mask, None
        if mask is not None:
            self.donor = {}
            for channel in ['pred', 'corr']:
                donor = torch.zeros(*ids.shape, self.E, device=self.device)
                for i, it in enumerate(items):
                    donor[i, :len(it['ids'])] = it['donor_' + channel].to(self.device)
                self.donor[channel] = donor
        routing = self.predictor(ids)
        flat = flatten_alpha(routing)
        if self.donor is not None:
            gate = mask[:self.E].view(1, 1, -1)
            if scope == 'content':
                gate = gate * (torch.arange(ids.shape[1], device=self.device) >= len(self.prefix)).view(1, -1, 1)
            flat = flat + gate * (self.donor['pred'] - flat)
            routing = unflatten_alpha(flat, self.L, self.H)
        all_logits = self.model(ids, routing)
        logits = all_logits[torch.arange(len(items), device=self.device), lengths-1].float()
        if capture:
            corr = torch.cat(self.captured, dim=-1)
            for i, item in enumerate(items):
                item['donor_pred'] = flat[i, :lengths[i]].detach().cpu()
                item['donor_corr'] = corr[i, :lengths[i]].detach().cpu()
        return logits

    @torch.no_grad()
    def measure(self, items, soft=False, capture=False, mask=None, scope='content', save_teacher=False):
        rows = []
        for start in range(0, len(items), self.args.batch_size):
            group = items[start:start+self.args.batch_size]
            logits = self.forward(group, soft, capture, mask, scope)
            logps = logits.log_softmax(-1)
            for i, item in enumerate(group):
                target = item['target']
                alts = [c for c in item['candidates'] if c != target]
                rows.append(dict(logp=float(logps[i, target]),
                                 margin=float(logits[i, target] - logits[i, alts].logsumexp(0)),
                                 candidate_correct=float(logits[i, item['candidates']].argmax() == item['candidates'].index(target)),
                                 vocab_correct=float(logits[i].argmax() == target)))
                if 'distractor' in item:
                    margin = logits[i, target] - logits[i, item['distractor']]
                    rows[-1].update(binding_logp=float(F.logsigmoid(margin)),
                                    binding_margin=float(margin), binding_correct=float(margin > 0),
                                    color_logmass=float(logps[i, item['candidates']].logsumexp(0)))
                if save_teacher:
                    item['teacher'] = logps[i].cpu()
        return rows

    @torch.no_grad()
    def natural_nll(self, path, count, soft=False, mask=None, scope='content'):
        ids, labels = load_eval_ids(path)
        values = []
        for x, y in zip(ids[:count, :128], labels[:count, :128]):
            item = {'ids': self.prefix + x.tolist()}
            if mask is not None:
                self.forward([item], soft=True, capture=True)
            self.forward([item], soft=soft, mask=mask, scope=scope)
            # Re-use the same intervention state for a token-level likelihood forward.
            row = torch.tensor([item['ids']], device=self.device)
            rw = self.predictor(row)
            if mask is not None:
                flat = flatten_alpha(rw)
                gate = mask[:self.E].view(1, 1, -1)
                if scope == 'content':
                    gate = gate * (torch.arange(row.shape[1], device=self.device) >= len(self.prefix)).view(1, -1, 1)
                rw = unflatten_alpha(flat + gate * (self.donor['pred'] - flat), self.L, self.H)
            logits = self.model(row, rw)[0, len(self.prefix):].float()
            values.append(float(F.cross_entropy(logits, y.to(self.device))))
        return values

    def objective(self, items):
        logits = self.forward(items, soft=True)
        targets = torch.tensor([it['target'] for it in items], device=self.device)
        ce = F.cross_entropy(logits, targets)
        if self.args.objective == 'binding':
            distractors = torch.tensor([it['distractor'] for it in items], device=self.device)
            row = torch.arange(len(items), device=self.device)
            return F.softplus(logits[row, distractors] - logits[row, targets]).mean() + .1 * ce
        return ce

    def save(self, result):
        self.args.out.mkdir(parents=True, exist_ok=True)
        (self.args.out / 'results.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')

    def run(self):
        a = self.args
        splits = {
            'train': make_items(self.tok, self.prefix, a.seed, 32, ['ball','book','coat','cup'], 'completion'),
            'validation': make_items(self.tok, self.prefix, a.seed+1, 16, ['bag','box','hat','car'], 'completion'),
            'test': make_items(self.tok, self.prefix, a.seed+2, 48, ['stone','chair','shirt','bottle'], 'completion'),
            'transfer': make_items(self.tok, self.prefix, a.seed+3, 48, ['stone','chair','shirt','bottle'], 'qa'),
        }
        if a.data_version == 2:
            specs = {
                'train': (128, ['ball','book','coat','cup','bag','box','hat','car',
                                'bike','door','flag','boat','pen','sock','key','wall'], 'completion'),
                'validation': (48, ['desk','table','bowl','shoe','shelf','clock','lamp','fence'], 'completion'),
                'test': (96, ['scarf','bench','truck','blanket','plate','ribbon','suitcase','candle'], 'completion'),
                'transfer': (96, ['scarf','bench','truck','blanket','plate','ribbon','suitcase','candle'], 'painted'),
            }
            splits = {name: make_items(self.tok, self.prefix, a.seed+i, n, objects, style)
                      for i, (name, (n, objects, style)) in enumerate(specs.items())}
        assert len({tuple(x['ids']) for split in splits.values() for x in split}) == sum(map(len,splits.values()))
        clean_items = copy.deepcopy(splits)
        result = dict(protocol='Frozen FourWay checkpoint; shared dual-space soft prefix; '
                      'entity-disjoint splits and color-swap pairs; right padding only, score at last real token; '
                      'donor route interchange, unpatched corrections recomputed dynamically; discovery-only gate fitting.',
                      args={k:str(v) if isinstance(v,Path) else v for k,v in vars(a).items()},
                      code_revision=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
                      items=clean_items, edge_dimensions_per_channel=self.E, stages={}, started=time.time())
        selection_metric = 'binding_logp' if a.objective == 'binding' else 'logp'
        result['selection_metric'] = selection_metric
        baseline = {name:self.measure(items) for name,items in splits.items()}
        result['baseline'] = {name:dict(summary=summarize(rows),rows=rows) for name,rows in baseline.items()}
        base_nll=self.natural_nll(a.corpora/'wikitext_test.pt',8)
        result['baseline']['natural_nll']=base_nll
        # A single critical gradient/path check, including causal right-padding parity.
        with torch.no_grad():
            natural=self.forward(splits['train'][:1])
            initial=self.forward(splits['train'][:1],soft=True)
            pair=self.forward([splits['train'][0],max(splits['train'],key=lambda x:len(x['ids']))])[:1]
            assert torch.allclose(natural,initial,atol=.02,rtol=.01)
            assert torch.allclose(natural,pair,atol=.08,rtol=.01)
        loss=self.objective(splits['train'][:a.batch_size]);loss.backward()
        gradients={k:float(p.grad.norm()) if p.grad is not None else 0. for k,p in self.soft.items()}
        assert all(np.isfinite(list(gradients.values()))) and max(gradients.values())>0
        assert all(p.grad is None for p in list(self.model.parameters())+list(self.predictor.parameters()))
        result['gradient_check']=gradients
        optimizer=torch.optim.Adam(self.soft.parameters(),lr=a.prefix_lr)
        rng=random.Random(a.seed+10)
        best=(float('-inf'),copy.deepcopy(self.soft.state_dict()),0)
        history=[]
        for step in range(a.prompt_steps+1):
            if step%20==0 or step==a.prompt_steps:
                rows=self.measure(splits['validation'],soft=True)
                summary=summarize(rows,baseline['validation'])
                history.append(dict(step=step,**summary))
                if summary[selection_metric]>best[0]:best=(summary[selection_metric],copy.deepcopy(self.soft.state_dict()),step)
                print('PREFIX',step,json.dumps(summary),flush=True)
            if step==a.prompt_steps:break
            optimizer.zero_grad(set_to_none=True)
            group=rng.sample(splits['train'],a.batch_size)
            loss=self.objective(group)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.soft.parameters(),1.)
            optimizer.step()
            with torch.no_grad():
                for key,p in self.soft.items():
                    delta=p-self.initial[key]
                    limit=self.initial[key].norm(dim=-1,keepdim=True)*a.prefix_radius
                    p.copy_(self.initial[key]+delta*(limit/delta.norm(dim=-1,keepdim=True).clamp_min(1e-8)).clamp(max=1))
        self.soft.load_state_dict(best[1])
        result['prefix_selection']=dict(step=best[2],criterion='validation '+selection_metric,history=history)
        torch.save({k:v.detach().cpu() for k,v in self.soft.items()},a.out/'soft_prefix.pt')
        arms={}
        for name,items in splits.items():
            rows=self.measure(items,soft=True,capture=True)
            arms.setdefault('soft_prompt',{})[name]=dict(summary=summarize(rows,baseline[name]),rows=rows)
        masks={'pred':torch.cat([torch.ones(self.E),torch.zeros(self.E)]).to(self.device),
               'corr':torch.cat([torch.zeros(self.E),torch.ones(self.E)]).to(self.device),
               'both':torch.ones(2*self.E,device=self.device)}
        for channel,mask in masks.items():
            for scope in ['content','all']:
                arm=channel+'_'+scope
                arms[arm]={}
                for name,items in splits.items():
                    rows=self.measure(items,mask=mask,scope=scope)
                    arms[arm][name]=dict(summary=summarize(rows,baseline[name]),rows=rows)
                print('TRANSFER',arm,json.dumps(arms[arm]['validation']['summary']),flush=True)
        arms['soft_prompt']['natural_nll']=self.natural_nll(a.corpora/'wikitext_test.pt',8,soft=True)
        result['stages']=arms
        self.save(result)
        # Choose channel using validation only; require positive transfer before a sparse search.
        candidates=[c+'_content' for c in masks]
        chosen=max(candidates,key=lambda c:arms[c]['validation']['summary'][selection_metric+'_delta'])
        gain=arms[chosen]['validation']['summary'][selection_metric+'_delta']
        result['mediation_selection']=dict(arm=chosen,metric=selection_metric,validation_gain=gain,minimum_gain=.02)
        if gain <= .02:
            result['sparse_search']='Not run: no content-route channel passed the prespecified validation gain threshold.'
            result['finished']=time.time();self.save(result);return
        fullmask=masks[chosen.split('_')[0]]
        allowed=fullmask.nonzero().flatten()
        self.measure(splits['train'],mask=fullmask,save_teacher=True)
        scores=torch.nn.Parameter(torch.zeros(len(allowed),device=self.device))
        optimizer=torch.optim.Adam([scores],lr=.05)
        k=min(a.gate_budget,len(allowed))
        gate_history=[]
        for step in range(a.gate_steps):
            group=rng.sample(splits['train'],a.batch_size)
            optimizer.zero_grad(set_to_none=True)
            soft=scores.sigmoid()
            hard=torch.zeros_like(soft).scatter(0,scores.topk(k).indices,1)
            gate=hard+soft-soft.detach()  # exact-k forward; straight-through search gradient
            mask=torch.zeros(2*self.E,device=self.device).scatter(0,allowed,gate)
            logits=self.forward(group,mask=mask)
            teacher=torch.stack([it['teacher'] for it in group]).to(self.device)
            loss=F.kl_div(logits.log_softmax(-1),teacher,log_target=True,reduction='batchmean')
            if a.objective == 'binding':
                choices = torch.tensor([[it['target'],it['distractor']] for it in group],device=self.device)
                conditional_teacher = teacher.gather(1,choices).log_softmax(-1)
                conditional_student = logits.gather(1,choices).log_softmax(-1)
                loss = F.kl_div(conditional_student,conditional_teacher,log_target=True,reduction='batchmean') + .1*loss
            loss.backward();optimizer.step()
            if step%20==0:
                gate_history.append(dict(step=step,kl=float(loss.detach())))
                print('GATES',step,float(loss.detach()),flush=True)
        ranked=allowed[scores.detach().argsort(descending=True)]
        result['gate_search']=dict(method='fixed-k straight-through binary gates; teacher is full content-route transplant',
                                   objective='conditional pair KL + 0.1 full-vocabulary KL' if a.objective == 'binding' else 'full-vocabulary KL',
                                   budget=k,history=gate_history,ranking=[self.labels[i] for i in ranked.cpu().tolist()])
        result['sparse_arms']={}
        for budget in sorted(set([16,64,k,1024])):
            if budget>len(allowed):continue
            selected=ranked[:budget].cpu().tolist()
            groups=defaultdict(list)
            for idx in allowed.cpu().tolist():
                label=self.labels[idx];groups[(label['channel'],label['layer'],label['stream'])].append(idx)
            selections={'learned':selected}
            for seed in range(3):
                rr=random.Random(a.seed+100+seed);counts=defaultdict(int)
                for idx in selected:
                    label=self.labels[idx];counts[(label['channel'],label['layer'],label['stream'])]+=1
                selections[f'random{seed}']=[i for key,count in counts.items() for i in rr.sample(groups[key],count)]
            for kind,indices in selections.items():
                mask=torch.zeros(2*self.E,device=self.device);mask[indices]=1
                arm=f'{kind}_{budget}'
                record=dict(edges=[self.labels[i] for i in indices],splits={})
                for name in ['validation','test','transfer']:
                    rows=self.measure(splits[name],mask=mask)
                    record['splits'][name]=dict(summary=summarize(rows,baseline[name]),rows=rows)
                record['natural_nll']=self.natural_nll(a.corpora/'wikitext_test.pt',8,mask=mask)
                result['sparse_arms'][arm]=record
                print('SPARSE',arm,json.dumps(record['splits']['test']['summary']),flush=True)
                self.save(result)
        result['finished']=time.time();self.save(result)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model',type=Path,default=Path('checkpoints/pr_sync_20260917/300m-dagformer'))
    p.add_argument('--tokenizer',default='checkpoints/pr_sync_20260917/tokenizer')
    p.add_argument('--corpora',type=Path,default=Path('checkpoints/pr_sync_20260917/eval_corpora'))
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--seed',type=int,default=20260928)
    p.add_argument('--prefix-tokens',type=int,default=8)
    p.add_argument('--prefix-target',choices=['base','pred','both'],default='both')
    p.add_argument('--prefix-lr',type=float,default=.01)
    p.add_argument('--prefix-radius',type=float,default=2.)
    p.add_argument('--prompt-steps',type=int,default=100)
    p.add_argument('--gate-steps',type=int,default=100)
    p.add_argument('--gate-budget',type=int,default=256)
    p.add_argument('--batch-size',type=int,default=4)
    p.add_argument('--objective',choices=['ce','binding'],default='ce')
    p.add_argument('--data-version',type=int,choices=[1,2],default=1)
    args=p.parse_args();args.out.mkdir(parents=True,exist_ok=True)
    torch.manual_seed(args.seed);torch.set_num_threads(4)
    torch.backends.mha.set_fastpath_enabled(False)
    exp=Pilot(args)
    try:exp.run()
    finally:
        for h in exp.handles:h.remove()


if __name__=='__main__':main()
