"""Fresh-sentence test of restoring a discovered grammar reader's routing.

This is an adaptive follow-up to native message and mediation results. It uses
new prompt blocks disjoint from all earlier semantic localization/test prompts.
Only selected external-predictor outputs return to P0; local correction remains
dynamic. No backbone, predictor, or correction weights are trained.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import random

import numpy as np
import torch

from circuit_semantics import (OUT, VERBS, selected_edges, make_data, read_gz,
    save_gz, write_json, values, extend, block_stat)
from circuit_semantics_trace import TraceExperiment


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',type=Path,default=OUT)
    p.add_argument('--adapter',type=Path,default=Path('/scratch/yurenh2/interp_followup_20260929/runs/full_s20260930/weights/predictor_best.pt'))
    args=p.parse_args()
    out=args.out/'read_repair';out.mkdir(exist_ok=True)
    old=read_gz(args.out/'data.json.gz')
    seen={r['prompt'] for rows in old['splits'].values() for r in rows}
    data=make_data(out,seed_offset=1000,counts=dict(discovery=0,heldout=32,transfer=32,role=32),excluded_prompts=seen)
    chosen=json.loads((args.out/'trace/selection.json').read_text())['chosen']
    layer,head=chosen[0]
    exp=TraceExperiment(args.adapter,selected_edges())
    mask=[e['index'] for e in exp.edges]
    indices=lambda h,streams:[i for i,e in enumerate(exp.runner.layout) if e['layer']==layer and e['head']==h and e['stream'] in streams]
    conditions=dict(original=list(range(len(exp.runner.layout))),adapted=[],repair75=mask)
    for streams in ['q','k','v','qk','qkv']:
        conditions[f'head_{streams}']=indices(head,streams)
    conditions['mask_head_k']=[i for i in indices(head,'k') if i in mask]
    rng=random.Random(202609291001)
    random_heads=rng.sample([h for h in range(16) if h!=head],3)
    for h in random_heads:
        conditions[f'control_H{h}_qk']=indices(h,'qk')
    write_json(out/'protocol.json',dict(
        adaptive_followup=True,
        motivation='Original-model number-message mediation located this head; initial test attention showed a large loss of subject reading in P1 and restoration in the 75-coordinate repair.',
        selected_head=dict(layer=layer,head=head,indexing='zero based'),
        selection='Highest original discovery mediation, without selecting a head by this new set.',
        intervention='Restore listed external predictor outputs from P0 at every token position; retain all other P1 outputs and recompute every local correction and attention normally.',
        data='New blocks with seed offset 1000; exclude every prior semantic discovery/test prompt and prior adapter experiment prompts.',
        sizes={s:len(rows) for s,rows in data['splits'].items()},
        restored_coordinates={k:len(v) for k,v in conditions.items()},conditions=conditions,
        controls='Three same-layer random heads, equal QK coordinate count; all-stream 75-coordinate repair; separate Q, K, V, QK, QKV repairs.',
        scope='Candidate grammar accuracy and this head attention; does not establish IOI retention for the newly narrowed repair.'))
    ids=torch.tensor(data['verb_ids'],device='cuda')
    summary={}
    with exp.installed(),torch.no_grad():
        for split in ['heldout','transfer','role']:
            rows=data['splits'][split]
            result={}
            sn=np.array([r['subject_number'] for r in rows])
            summary[split]={}
            for name,coordinates in conditions.items():
                exp.gate=torch.ones(len(exp.runner.layout),device='cuda')
                exp.gate[coordinates]=0
                result[name]=dict(values={},attention=[])
                for start in range(0,len(rows),16):
                    logits=exp.forward(rows[start:start+16],'repaired',capture=True)
                    extend(result[name]['values'],values(logits,ids))
                    result[name]['attention'].extend(exp.cache[('attention_roles',layer)][:,head].cpu().tolist())
                margins=np.asarray(result[name]['values']['margins'])
                summary[split][name]=dict(grammar={ '/'.join(pair):block_stat(margins[:,j]*(2*sn-1)>0)
                     for j,pair in enumerate(VERBS)},
                     grammar_margin={ '/'.join(pair):block_stat(margins[:,j]*(2*sn-1)) for j,pair in enumerate(VERBS)},
                     attention={site:block_stat(np.asarray(result[name]['attention'])[:,i])
                                for i,site in enumerate(['subject','distractor','last'])})
                print(split,name,'grammar',summary[split][name]['grammar']['is/are']['mean'],
                      'subject_attention',summary[split][name]['attention']['subject']['mean'],flush=True)
            save_gz(out/f'{split}.json.gz',result)
            write_json(out/'summary.json',summary)
    print('COMPLETE read repair',flush=True)


if __name__=='__main__':main()
