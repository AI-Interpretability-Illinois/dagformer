"""Trace a learned native number-message intervention through downstream heads.

Head output restoration measures mediation of the specified intervention, not
unique necessity of a head for all grammar. Discovery selects heads; held-out
inputs compare them with layer/count-matched random head restorations.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from contextlib import contextmanager
import json
from pathlib import Path
import random

import numpy as np
import torch
import torch.nn.functional as F

from circuit_semantics import (OUT, STATES, VERBS, MessageExperiment, read_gz,
    save_gz, selected_edges, arm, run_set, summarize, block_stat, write_json)


class TraceExperiment(MessageExperiment):
    @contextmanager
    def installed(self):
        native = F.scaled_dot_product_attention
        def attention(q,k,v,*args,**kwargs):
            output = native(q,k,v,*args,**kwargs)
            if q.ndim==4 and q.shape[1]==self.runner.cfg['num_attention_heads']:
                layer = self.attn_layer
                self.attn_layer += 1
                if self.capture:
                    b = torch.arange(len(q),device=q.device)
                    last = self.runner.last_positions
                    query = q[b,:,last].float()
                    scale = kwargs.get('scale', q.shape[-1]**-.5)
                    scores = (query[:,:,None,:]*k.float()).sum(-1)*scale
                    positions = torch.arange(k.shape[2],device=q.device)
                    scores.masked_fill_(positions[None,None,:]>last[:,None,None],float('-inf'))
                    probabilities = scores.softmax(-1)
                    role_weights = []
                    for site in ['subject','distractor','last']:
                        pos = torch.tensor([r['positions'][site] for r in self.rows],device=q.device)
                        role_weights.append(probabilities[b,:,pos])
                    self.cache[('attention_roles',layer)] = torch.stack(role_weights,-1).detach()
            return output
        with super().installed():
            F.scaled_dot_product_attention = attention
            try:
                yield self
            finally:
                F.scaled_dot_product_attention = native

    def forward(self,rows,state,spec=None,capture=False):
        self.attn_layer = 0
        self.rows = rows
        out = super().forward(rows,state,spec,capture)
        assert self.attn_layer==self.runner.cfg['num_hidden_layers'], self.attn_layer
        return out


def reset_specs(heads, site='all'):
    layers = defaultdict(list)
    for layer,head in heads:
        layers[layer].append(head)
    return [dict(layer=layer,heads=sorted(hs),site=site) for layer,hs in sorted(layers.items())]


def mediation_summary(rows, record, arms):
    baseline = np.asarray(record['baseline']['margins'])
    source = np.asarray(record['arms']['number_patch']['margins'])
    sn = np.array([r['subject_number'] for r in rows])
    sign = 1-2*sn
    result = {}
    for spec in arms:
        if spec['name']=='number_patch':
            continue
        changed = np.asarray(record['arms'][spec['name']]['margins'])
        result[spec['name']] = {}
        for j,pair in enumerate(VERBS):
            source_effect = (source[:,j]-baseline[:,j])*sign
            removed_effect = (source[:,j]-changed[:,j])*sign
            bsource = source_effect.reshape(-1,8).mean(1)
            bremoved = removed_effect.reshape(-1,8).mean(1)
            mean = source_effect.mean()
            fraction = None
            if abs(mean)>.05:
                rng = np.random.default_rng(20260929)
                ix = rng.integers(len(bsource),size=(2000,len(bsource)))
                denominator = bsource[ix].mean(1)
                valid = np.abs(denominator)>.05
                ratios = bremoved[ix][valid].mean(1)/denominator[valid]
                fraction = dict(mean=float(removed_effect.mean()/mean),
                                ci95=np.quantile(ratios,[.025,.975]).tolist(), bootstrap_valid_fraction=float(valid.mean()))
            result[spec['name']]['/'.join(pair)] = dict(removed_effect=block_stat(removed_effect),
                source_effect=block_stat(source_effect), mediated_fraction=fraction)
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',type=Path,default=OUT)
    p.add_argument('--adapter',type=Path,default=Path('/scratch/yurenh2/interp_followup_20260929/runs/full_s20260930/weights/predictor_best.pt'))
    args=p.parse_args()
    data=read_gz(args.out/'data.json.gz')
    edge=json.loads((args.out/'selection.json').read_text())['selected']['edge']['edges'][0]
    direction=torch.tensor(json.loads((args.out/'variable/direction_subject.json').read_text())['directions']['learned'],device='cuda')
    out=args.out/'trace';out.mkdir(exist_ok=True)
    base=arm('number_patch',[edge],site='subject',direction=direction)
    candidates=[base]
    for layer in range(edge['layer']+1,12):
        for head in range(16):
            candidates.append(dict(base,name=f'reset/L{layer}H{head}',reset_layer=layer,reset_heads=[head]))
    write_json(out/'protocol.json',dict(
        source_edge=edge,source_site='subject',direction='Frozen selected one-dimensional original-model subject direction.',
        search='Restore each downstream head output before o_proj to its unpatched recipient value, at all positions. Select top three individually by removed signed is/are effect on original-model discovery.',
        confirmation='Same selected heads in all predictor states and all test splits. Joint and individual head restoration; all-position vs last-position restoration; three random controls matched by layer and head count.',
        attention='Read-only FP32 QK softmax at last query after native norm and RoPE; native SDPA still produces every model output. These probabilities are diagnostic, not a replacement attention forward.',
        limits='Mediation of this injected number message, not unique necessity, a complete circuit, or architecture superiority.'))
    exp=TraceExperiment(args.adapter,selected_edges())
    with exp.installed():
        path=out/'discovery.json.gz'
        if path.exists():
            discovery=read_gz(path)
        else:
            discovery=run_set(exp,data['splits']['discovery'],'original',candidates,data['verb_ids'])
            discovery['mediation']=mediation_summary(data['splits']['discovery'],discovery,candidates)
            save_gz(path,discovery)
            write_json(out/'discovery_summary.json',discovery['mediation'])
        ranked=sorted(candidates[1:],key=lambda a:discovery['mediation'][a['name']]['is/are']['removed_effect']['mean'],reverse=True)
        chosen=[(s['reset_layer'],s['reset_heads'][0]) for s in ranked[:3]]
        arms=[base]
        for spec in ranked[:3]:
            arms.append(spec)
        for site in ['all','last']:
            arms.append(dict(base,name=f'top3/{site}',resets=reset_specs(chosen,site)))
        rng=random.Random(202609290901)
        random_sets=[]
        for replicate in range(3):
            selected=[]
            for layer,head in chosen:
                options=[h for h in range(16) if (layer,h) not in chosen and (layer,h) not in selected]
                selected.append((layer,rng.choice(options)))
            random_sets.append(selected)
            for site in ['all','last']:
                arms.append(dict(base,name=f'random{replicate}/{site}',resets=reset_specs(selected,site)))
        write_json(out/'selection.json',dict(chosen=chosen,random_sets=random_sets,
            ranked=[dict(layer=s['reset_layer'],head=s['reset_heads'][0],removed_effect=discovery['mediation'][s['name']]['is/are']['removed_effect']) for s in ranked]))
        attention_report={}
        for split in ['heldout','transfer','role']:
            path=out/f'{split}.json.gz'
            result=read_gz(path) if path.exists() else {}
            attention_report[split]={}
            rows=data['splits'][split]
            for state in STATES:
                if state not in result:
                    result[state]=run_set(exp,rows,state,arms,data['verb_ids'])
                    result[state]['mediation']=mediation_summary(rows,result[state],arms)
                    save_gz(path,result)
                    write_json(out/f'{split}_summary.json',{s:r['mediation'] for s,r in result.items()})
                weights=defaultdict(list)
                with torch.no_grad():
                    for start in range(0,len(rows),16):
                        exp.forward(rows[start:start+16],state,capture=True)
                        for layer,head in chosen:
                            weights[f'L{layer}H{head}'].extend(exp.cache[('attention_roles',layer)][:,head].cpu().tolist())
                attention_report[split][state]={name:{site:block_stat(np.asarray(values)[:,i])
                    for i,site in enumerate(['subject','distractor','last'])} for name,values in weights.items()}
            write_json(out/'attention_summary.json',attention_report)
    print('COMPLETE trace',flush=True)


if __name__=='__main__':main()
