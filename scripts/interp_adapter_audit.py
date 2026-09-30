"""Stage 1: train one adapter on SVA+IOI, then selectively undo its changes.

Both methods adapt the same complete DAG checkpoint. Predictor-only updates all
predictor parameters; the LoRA comparator updates attention and MLP projections.
Trainable parameter counts differ and are reported; this is a behavior audit,
not a parameter-efficiency comparison. Mask selection never sees test data.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import copy
import json
import math
from pathlib import Path
import random
import subprocess
import time

import numpy as np
import torch

from interp_audit_common import AuditModel, RESULT_ROOT, evaluate, metrics, paired_stats, prepare_data, write_json


def save_adapter(runner, method, path, metadata):
    if method == 'predictor':
        state = {'predictor': {k:v.detach().cpu() for k,v in runner.predictor.state_dict().items()}}
    else:
        state = {'lora': {name: {'a':m.a.detach().cpu(), 'b':m.b.detach().cpu()}
                          for name,m in runner.loras}}
    torch.save(dict(**state,metadata=metadata),path)


def load_adapter(runner, method, path):
    state = torch.load(path,map_location='cuda',weights_only=False)
    if method == 'predictor': runner.predictor.load_state_dict(state['predictor'])
    else:
        for name,m in runner.loras:
            m.a.data.copy_(state['lora'][name]['a'])
            m.b.data.copy_(state['lora'][name]['b'])
    return state['metadata']


def prepare(runner, method, rank):
    if method == 'predictor':
        runner.copy_baseline_predictor()
        params = list(runner.predictor.parameters())
        for p in params: p.requires_grad_(True)
        return params
    return runner.add_lora(rank)


def score(summary):
    return sum(summary[t]['nll']['mean'] for t in ['sva','ioi'])/2


def train(args,runner,data,result):
    method = args.method
    out = args.out/method
    out.mkdir(parents=True,exist_ok=True)
    args.weights.mkdir(parents=True,exist_ok=True)
    weight_path = args.weights/f'{method}_best.pt'
    result['baseline_validation'] = evaluate(runner,data['validation'],args.eval_batch)
    params = prepare(runner,method,args.rank)
    result['trainable_parameters'] = sum(p.numel() for p in params)
    result['backbone_original_parameters_frozen'] = all(not p.requires_grad for n,p in runner.model.named_parameters()
                                                        if '.parametrizations.' not in n)
    result['protocol'] = dict(
        tasks=f'Controlled English subject-verb {args.sva_target} and indirect-object identification. For disagreement the canonical grammatical and ungrammatical targets are swapped. This is not a published benchmark-score reproduction.',
        objective='Full-vocabulary next-token CE; SVA 0.4 + IOI 0.4 + unrelated WikiText train tokens 0.2.',
        model_selection='Lowest mean SVA/IOI validation NLL among checkpoints whose retain NLL rises no more than 0.15 nats.',
        validation='New prompt pairs, same template/vocabulary distribution. Retain validation uses even-index validation cache chunks.',
        localization='Separate discovery prompts; retain discovery uses odd-index validation cache chunks. Cache chunks are packed token windows, not original articles.',
        confirmation='Heldout new prompt pairs; transfer uses disjoint nouns/names and held-out templates.',
        rollback='Predictor coefficients interpolate between baseline and adapted predictor, before unchanged local corrections. LoRA rank terms interpolate to zero.',
        units='Predictor units are layer/head/source/stream scalar output coordinates at all positions; LoRA units are rank terms. Unit counts are not equivalent circuit sizes.',
        controls='Full rollback identity; opposite-task masks; matched-count random units. Dense/MLP neuron analysis is a later stage.',
        limits='One training seed; fixed 300M pretrained checkpoint; trainable parameter counts differ; no efficiency superiority claim.')
    result['training'] = []
    write_json(out/'results.json',result)
    pools = {task:[x for x in data['train'] if x['task']==task] for task in ['sva','ioi','retain']}
    rng = random.Random(args.seed)
    optimizer = torch.optim.AdamW(params,lr=args.lr,betas=(.9,.95),weight_decay=.01)
    baseline_score = score(result['baseline_validation']['summary'])
    retain0 = result['baseline_validation']['summary']['retain']['nll']['mean']
    best = baseline_score
    save_adapter(runner,method,weight_path,dict(step=0,score=best))
    started = time.time()
    for step in range(1,args.steps+1):
        batch = sum((rng.sample(pools[t],args.per_task_batch) for t in ['sva','ioi','retain']),[])
        optimizer.zero_grad(set_to_none=True)
        logits = runner.forward(batch)
        nll,_ = metrics(logits,batch)
        n = args.per_task_batch
        loss = .4*nll[:n].mean()+.4*nll[n:2*n].mean()+.2*nll[2*n:].mean()
        if not torch.isfinite(loss): raise RuntimeError(f'Nonfinite training loss at {step}')
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(params,1.)
        if not torch.isfinite(norm): raise RuntimeError(f'Nonfinite gradient at {step}')
        lr = args.lr*min(1.,step/30)*(.2+.8*.5*(1+math.cos(math.pi*step/args.steps)))
        for group in optimizer.param_groups: group['lr']=lr
        optimizer.step()
        if step==1 or step%25==0:
            row=dict(step=step,loss=loss.item(),grad_norm=float(norm),lr=lr,elapsed=time.time()-started,
                     sva_nll=nll[:n].mean().item(),ioi_nll=nll[n:2*n].mean().item(),retain_nll=nll[2*n:].mean().item())
            result['training'].append(row)
            print(method,'TRAIN',json.dumps(row),flush=True)
        if step%args.eval_every==0 or step==args.steps:
            ev=evaluate(runner,data['validation'],args.eval_batch)
            item=dict(step=step,**ev)
            result.setdefault('validation',[]).append(item)
            candidate = score(ev['summary'])
            eligible = ev['summary']['retain']['nll']['mean'] <= retain0+.15
            if candidate < best and eligible:
                best=candidate
                save_adapter(runner,method,weight_path,dict(step=step,score=best))
            result['best_validation_score']=best
            result['elapsed_seconds']=time.time()-started
            write_json(out/'results.json',result)
            print(method,'VALIDATION',step,{k:{m:round(v['mean'],4) for m,v in s.items()} for k,s in ev['summary'].items()},'eligible',eligible,flush=True)
    del optimizer
    for p in params: p.requires_grad_(False)
    result['selected_checkpoint']=load_adapter(runner,method,weight_path)
    result['training_complete']=True
    write_json(out/'results.json',result)


def gates(runner,method,gate):
    if method=='lora':
        runner.set_lora_gate(gate)
        return None
    return gate


def get_size(runner,method):
    return len(runner.layout) if method=='predictor' else sum(m.a.shape[0] for _,m in runner.loras)


def evaluate_gate(runner,method,items,gate,batch_size):
    return evaluate(runner,items,batch_size,alpha_gate=gates(runner,method,gate))


def effects(base,adapted,edited):
    result={}
    for task in ['sva','ioi','retain']:
        idx=[i for i,x in enumerate(base['values']) if x['task']==task]
        groups=[base['values'][i]['group'] for i in idx]
        gain=np.array([base['values'][i]['nll']-adapted['values'][i]['nll'] for i in idx])
        undo=np.array([edited['values'][i]['nll']-adapted['values'][i]['nll'] for i in idx])
        result[task]=dict(original_gain=paired_stats(gain,groups),rollback_nll=paired_stats(undo,groups),
                          fraction_of_gain_undone=float(undo.mean()/gain.mean()) if gain.mean()>.03 else None)
    return result


def audit(args,runner,data,result):
    out=args.out/args.method
    method=args.method
    size=get_size(runner,method)
    ones=torch.ones(size,device='cuda')
    zeros=torch.zeros_like(ones)
    # The complete rollback must recover the original model on exactly the same batch.
    batch=data['discovery'][:args.eval_batch]
    with torch.no_grad():
        reference=runner.original_check_logits.to('cuda')
        restored=runner.forward(batch,alpha_gate=gates(runner,method,zeros))
        result['full_rollback_max_logit_error']=(reference-restored).abs().max().item()
    assert result['full_rollback_max_logit_error']==0
    base=evaluate_gate(runner,method,data['discovery'],zeros,args.eval_batch)
    adapted=evaluate_gate(runner,method,data['discovery'],ones,args.eval_batch)
    result['discovery_baseline']=base
    result['discovery_adapted']=adapted
    gains={t:base['summary'][t]['nll']['mean']-adapted['summary'][t]['nll']['mean'] for t in ['sva','ioi']}
    result['audit_identifiable']=all(v>.03 for v in gains.values())
    # Gradient integrated along the complete baseline-to-adapter interpolation.
    gradients={}
    for task in ['sva','ioi','retain']:
        items=[x for x in data['discovery'] if x['task']==task]
        grad=torch.zeros(size,device='cuda')
        for level in [.25,.5,.75,1.]:
            for start in range(0,len(items),args.eval_batch):
                batch=items[start:start+args.eval_batch]
                gate=torch.full((size,),level,device='cuda',requires_grad=True)
                logits=runner.forward(batch,alpha_gate=gates(runner,method,gate))
                nll,_=metrics(logits,batch)
                g=torch.autograd.grad(nll.mean(),gate)[0]
                grad+=g.detach()*len(batch)/len(items)/4
        gradients[task]=(-grad).cpu().numpy()
        print(method,'DISCOVERY gradients',task,flush=True)
    result['integrated_rollback_scores']={t:g.tolist() for t,g in gradients.items()}
    budgets=sorted(set(max(1,round(size*f)) for f in [.005,.01,.02,.05,.1]))
    selected={}
    result['discovery_candidates']=[]
    for task,other in [('sva','ioi'),('ioi','sva')]:
        a=gradients[task]/max(gains[task],.03)
        b=gradients[other]/max(gains[other],.03)
        retain=np.abs(gradients['retain'])/.15
        for k in budgets:
            best=None
            for penalty in [0.,1.,4.]:
                ranking=a-penalty*np.abs(b)-.2*retain
                indices=np.argsort(-ranking)[:k].tolist()
                gate=ones.clone(); gate[indices]=0
                ev=evaluate_gate(runner,method,data['discovery'],gate,args.eval_batch)
                eff=effects(base,adapted,ev)
                undo=eff[task]['rollback_nll']['mean']/max(gains[task],.03)
                spill=eff[other]['rollback_nll']['mean']/max(gains[other],.03)
                # Prefer selective rollback; unrelated-LM damage is also penalized.
                objective=min(undo,1.)-abs(spill)-max(0,eff['retain']['rollback_nll']['mean'])/.15
                row=dict(task=task,k=k,penalty=penalty,indices=indices,objective=objective,effects=eff)
                result['discovery_candidates'].append(row)
                if best is None or objective>best['objective']: best=row
            selected[f'{task}_{k}']=best
            print(method,'MASK',task,k,'score',round(best['objective'],4),flush=True)
            write_json(out/'results.json',result)
    rng=np.random.default_rng(args.seed+123)
    choices={name:row['indices'] for name,row in selected.items()}
    for k in budgets:
        for repeat in range(3): choices[f'random_{k}_{repeat}']=rng.choice(size,k,replace=False).tolist()
    result['selected_masks']=selected
    result['all_masks']=choices
    result['mask_unit_labels']=runner.layout if method=='predictor' else [dict(module=name,rank=i) for name,m in runner.loras for i in range(m.a.shape[0])]
    result['stages']={}
    for split in ['heldout','transfer']:
        base=evaluate_gate(runner,method,data[split],zeros,args.eval_batch)
        adapted=evaluate_gate(runner,method,data[split],ones,args.eval_batch)
        stage=dict(base=base,adapted=adapted,arms={})
        result['stages'][split]=stage
        for name,indices in choices.items():
            gate=ones.clone();gate[indices]=0
            ev=evaluate_gate(runner,method,data[split],gate,args.eval_batch)
            stage['arms'][name]=dict(**ev,effects=effects(base,adapted,ev))
        print(method,'TEST COMPLETE',split,flush=True)
        write_json(out/'results.json',result)
    result['complete']=True
    write_json(out/'results.json',result)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--method',choices=['predictor','lora'],required=True)
    p.add_argument('--out',type=Path,default=RESULT_ROOT/'adapter_audit')
    p.add_argument('--weights',type=Path,default=Path('/scratch/yurenh2/interp_audit_20260929'))
    p.add_argument('--steps',type=int,default=600)
    p.add_argument('--lr',type=float,default=3e-5)
    p.add_argument('--rank',type=int,default=8)
    p.add_argument('--sva-target',choices=['agreement','disagreement'],default='disagreement')
    p.add_argument('--per-task-batch',type=int,default=4)
    p.add_argument('--eval-batch',type=int,default=16)
    p.add_argument('--eval-every',type=int,default=100)
    p.add_argument('--seed',type=int,default=20260930)
    p.add_argument('--audit-only',action='store_true')
    p.add_argument('--train-only',action='store_true')
    args=p.parse_args()
    torch.manual_seed(args.seed)
    data=copy.deepcopy(prepare_data(args.out.parent))
    if args.sva_target=='disagreement':
        for items in data.values():
            for item in items:
                if item['task']=='sva':
                    item['target'],item['foil_id']=item['foil_id'],item['target']
                    item['answer'],item['foil']=item['foil'],item['answer']
    runner=AuditModel()
    # Capture before either adaptation or LoRA parametrization is installed.
    with torch.no_grad():
        runner.original_check_logits=runner.forward(data['discovery'][:args.eval_batch]).cpu()
    result=dict(complete=False,method=args.method,git_commit_at_start=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
                args={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
                dataset_counts={s:{t:sum(x['task']==t for x in rows) for t in ['sva','ioi','retain']} for s,rows in data.items()})
    if args.audit_only:
        result=json.loads((args.out/args.method/'results.json').read_text())
        params=prepare(runner,args.method,args.rank)
        for par in params: par.requires_grad_(False)
        result['selected_checkpoint']=load_adapter(runner,args.method,args.weights/f'{args.method}_best.pt')
    else: train(args,runner,data,result)
    if not args.train_only: audit(args,runner,data,result)


if __name__=='__main__': main()
