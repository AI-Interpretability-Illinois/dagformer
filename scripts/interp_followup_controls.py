"""Matched controls, cross-seed masks, and single-model adapter exports."""
from __future__ import annotations
import argparse
from collections import defaultdict
import copy
import gzip
import json
from pathlib import Path
import time
import numpy as np
import torch
from interp_audit_common import AuditModel, ROOT, prepare_data, paired_stats, write_json
from interp_adapter_audit import prepare, load_adapter, save_adapter, evaluate_gate, effects, get_size
from interp_common import flatten_alpha

OUT=ROOT/'experiments/results/interp_followup_20260929'
WORK=Path('/scratch/yurenh2/interp_followup_20260929')


def gzread(path):
    with gzip.open(path,'rt') as f:return json.load(f)


def gzwrite(path,record):
    path.parent.mkdir(parents=True,exist_ok=True)
    with gzip.open(path,'wt') as f:json.dump(record,f,separators=(',',':'),allow_nan=False)


def grammar(ev):
    rows=[x for x in ev['values'] if x['task']=='sva']
    return np.mean([x['margin']<0 for x in rows])


def snapshot(runner,method):
    if method=='predictor':
        return {n:p.detach().clone() for n,p in runner.predictor.named_parameters() if n.startswith(('layer_heads.','layer_biases.'))}
    return {name:dict(a=m.a.detach().clone(),b=m.b.detach().clone()) for name,m in runner.loras}


@torch.no_grad()
def restore(runner,method,saved):
    if method=='predictor':
        for n,p in runner.predictor.named_parameters():
            if n in saved:p.copy_(saved[n])
    else:
        for name,m in runner.loras:
            m.a.copy_(saved[name]['a']);m.b.copy_(saved[name]['b']);m.gate=torch.ones_like(m.gate);m.row_gate=torch.ones_like(m.row_gate)


@torch.no_grad()
def materialize(runner,method,indices):
    selected=set(indices);off=0
    if method=='predictor':
        for l,(head,base) in enumerate(zip(runner.predictor.layer_heads,runner.baseline_predictor.layer_heads)):
            rows=[i-off for i in indices if off<=i<off+head.out_features]
            head.weight[rows]=base.weight[rows]
            runner.predictor.layer_biases[l][rows]=runner.baseline_predictor.layer_biases[l][rows]
            off+=head.out_features
    else:
        for _,m in runner.loras:
            rows=[i-off for i in indices if off<=i<off+m.b.shape[0]]
            m.b[rows]=0;m.gate=torch.ones_like(m.gate);m.row_gate=torch.ones_like(m.row_gate)
            off+=m.b.shape[0]
    assert all(0<=i<off for i in selected)


@torch.no_grad()
def matching_strata(runner,method,items):
    size=get_size(runner,method);strata={}
    if method=='predictor':
        sums=torch.zeros(size,device='cuda');count=0
        for start in range(0,len(items),16):
            ids=runner.batch(items[start:start+16]);diff=flatten_alpha(runner.predictor(ids))-flatten_alpha(runner.baseline_predictor(ids))
            valid=torch.arange(ids.shape[1],device='cuda')[None,:]<=runner.last_positions[:,None]
            sums+=(diff.square()*valid[:,:,None]).sum((0,1));count+=valid.sum().item()
        magnitude=(sums/count).sqrt().cpu().numpy();parts=defaultdict(list)
        for i,x in enumerate(runner.layout):parts[(x['layer'],x['stream'])].append(i)
    else:
        magnitude=np.zeros(size);parts={};off=0
        for name,m in runner.loras:
            gram=m.a@m.a.T
            magnitude[off:off+m.b.shape[0]]=(torch.einsum('or,rs,os->o',m.b,gram,m.b).clamp_min(0).sqrt()*m.scale).cpu().numpy()
            parts[name]=list(range(off,off+m.b.shape[0]));off+=m.b.shape[0]
    for key,indices in parts.items():
        cuts=np.quantile(magnitude[indices],[.25,.5,.75])
        for i in indices:strata[i]=(str(key),int(np.searchsorted(cuts,magnitude[i])))
    return strata,magnitude


def latency(runner,batch,gate=None):
    with torch.no_grad():
        for _ in range(5):runner.forward(batch,alpha_gate=gate)
        elapsed=[]
        for _ in range(30):
            start=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True)
            start.record();runner.forward(batch,alpha_gate=gate);end.record();end.synchronize();elapsed.append(start.elapsed_time(end))
    return dict(median_ms=float(np.median(elapsed)),mean_ms=float(np.mean(elapsed)),n_batches=len(elapsed),batch_size=len(batch))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--kind',choices=['outputs','lora_rows','full'],required=True);p.add_argument('--seed',type=int,required=True);args=p.parse_args()
    label=f'{args.kind}_s{args.seed}';run=WORK/'runs'/label;method='lora' if args.kind=='lora_rows' else 'predictor'
    r=gzread(OUT/'runs'/label/'results.json.gz');assert r['complete']
    trainargs=r['audit_args'];data=copy.deepcopy(prepare_data());data.update(gzread(OUT/'extra_tests.json.gz'))
    for rows in data.values():
        for x in rows:
            if x['task']=='sva':x['target'],x['foil_id']=x['foil_id'],x['target']
    runner=AuditModel();runner.lora_unit='row' if method=='lora' else 'rank'
    params=prepare(runner,method,7 if method=='lora' else 8,'outputs' if args.kind=='outputs' else 'all')
    for par in params:par.requires_grad_(False)
    load_adapter(runner,method,run/'weights'/f'{method}_best.pt')
    size=get_size(runner,method);ones=torch.ones(size,device='cuda')
    budget_scale=r['trainable_parameters']/r['parameters_per_edit_unit'] if args.kind!='full' else size
    names=[f'sva_{round(budget_scale*f)}' for f in [.02,.05]]
    report=dict(complete=False,kind=args.kind,seed=args.seed,protocol='Primary 2% and secondary 5% rollback. Controls match layer/stream plus discovery coefficient-change RMS quartile (predictor), or projection module plus effective weight-delta row norm quartile (LoRA). Three random replicates. Also use seed-0 masks without re-localizing, and discovery-selected global shrinking.',masks={},stages={},exports={})
    strata,magnitude=matching_strata(runner,method,data['discovery']);report['matching_magnitude']=magnitude.tolist()
    pools=defaultdict(list)
    for i,key in strata.items():pools[key].append(i)
    rng=np.random.default_rng(args.seed+504)
    masks={}
    for name in names:
        selected=r['all_masks'][name];counts=defaultdict(int)
        for i in selected:counts[strata[i]]+=1
        for repeat in range(3):
            mask=[]
            for key,n in counts.items():mask.extend(rng.choice(pools[key],n,False).tolist())
            masks[f'matched_{name}_{repeat}']=mask
    reference=gzread(OUT/'runs'/f'{args.kind}_s20260930'/'results.json.gz')
    for name in names:
        masks[f'cross_seed_{name}']=reference['all_masks'][name]
    report['cross_seed_jaccard']={name:len(set(r['all_masks'][name])&set(reference['all_masks'][name]))/len(set(r['all_masks'][name])|set(reference['all_masks'][name])) for name in names}
    report['masks']=masks
    global_candidates=[];base=r['discovery_baseline'];adapt=r['discovery_adapted']
    for level in [0.,.05,.1,.2,.3,.4,.5,.6,.7,.8,.9,.95,1.]:
        ev=evaluate_gate(runner,method,data['discovery'],ones*level,16)
        ef=effects(base,adapt,ev)
        gain={t:max(ef[t]['original_gain']['mean'],.03) for t in ['sva','ioi']}
        objective=min(ef['sva']['rollback_nll']['mean']/gain['sva'],1)-abs(ef['ioi']['rollback_nll']['mean']/gain['ioi'])-max(0,ef['retain']['rollback_nll']['mean'])/.15
        global_candidates.append(dict(level=level,grammar=grammar(ev),ioi=ev['summary']['ioi']['candidate_correct']['mean'],objective=objective,effects=ef))
    primary=ones.clone();primary[r['all_masks'][names[0]]]=0
    selected_discovery=evaluate_gate(runner,method,data['discovery'],primary,16)
    target=grammar(selected_discovery)
    global_best=max(global_candidates,key=lambda x:x['objective'])['level']
    global_match=min(global_candidates,key=lambda x:(abs(x['grammar']-target),-x['ioi']))['level']
    report['global_selection']=dict(candidates=global_candidates,primary_mask_discovery_grammar=target,best_objective_level=global_best,matched_grammar_level=global_match)
    for split in ['heldout','transfer','confirmation','hard_transfer']:
        stage={};report['stages'][split]=stage;s=r['stages'][split]
        for name,indices in masks.items():
            gate=ones.clone();gate[indices]=0
            ev=evaluate_gate(runner,method,data[split],gate,16);stage[name]=dict(**ev,effects=effects(s['base'],s['adapted'],ev))
        for name,level in [('global_best',global_best),('global_match',global_match)]:
            ev=evaluate_gate(runner,method,data[split],ones*level,16);stage[name]=dict(**ev,effects=effects(s['base'],s['adapted'],ev))
        print(label,'CONTROLS',split,flush=True)
        gzwrite(OUT/'runs'/label/'controls.json.gz',report)
    if args.kind!='full':
        if method=='predictor':
            report['frozen_predictor_prefix_unchanged']=all(torch.equal(p,dict(runner.baseline_predictor.named_parameters())[n]) for n,p in runner.predictor.named_parameters() if not n.startswith(('layer_heads.','layer_biases.')))
            assert report['frozen_predictor_prefix_unchanged']
        saved=snapshot(runner,method)
        for name in names:
            restore(runner,method,saved);gate=ones.clone();gate[r['all_masks'][name]]=0
            batches={split:data[split][:16] for split in ['heldout','transfer','confirmation','hard_transfer']}
            batches['retain_long']=[x for x in data['hard_transfer'] if x['task']=='retain'][:16]
            with torch.no_grad():
                if method=='lora':runner.set_lora_gate(gate)
                expected={split:runner.forward(batch,alpha_gate=gate if method=='predictor' else None).clone() for split,batch in batches.items()}
            functional_latency=latency(runner,batches['retain_long'],gate) if method=='predictor' else None
            materialize(runner,method,r['all_masks'][name])
            baseline=runner.baseline_predictor;runner.baseline_predictor=None
            calls={'predictor':0}
            def count(module,inputs,output):calls['predictor']+=1
            handle=runner.predictor.register_forward_hook(count)
            errors={}
            with torch.no_grad():
                for split,batch in batches.items():errors[split]=(runner.forward(batch)-expected[split]).abs().max().item()
            handle.remove();assert calls['predictor']==len(batches)
            assert max(errors.values())==0,(label,name,errors)
            measured=latency(runner,batches['retain_long'])
            exported=run/'weights'/f'{method}_{name}_single.pt'
            save_adapter(runner,method,exported,dict(seed=args.seed,materialized_rollback=name,indices=r['all_masks'][name],checkpoint='300m-dagformer step9000',single_predictor=True))
            report['exports'][name]=dict(path=str(exported),bytes=exported.stat().st_size,logit_max_errors=errors,predictor_calls=calls['predictor'],forward_batches=len(batches),single_model_latency=measured,functional_latency=functional_latency)
            runner.baseline_predictor=baseline
            print(label,'EXPORTED',name,errors,flush=True)
        restore(runner,method,saved)
    report['complete']=True;gzwrite(OUT/'runs'/label/'controls.json.gz',report)
    print(label,'COMPLETE',flush=True)

if __name__=='__main__':main()
