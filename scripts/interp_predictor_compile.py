"""Compile functional predictor rollback into sparse output-row corrections.

The adapted predictor's encoder/trunk stays frozen. Fit only the selected output
rows to baseline-predictor coefficients on training prompts, then select ridge
strength by full-vocabulary KL to the functional teacher on validation prompts.
Confirmation/test prompts never participate in this fit or model selection.
"""
from __future__ import annotations
import argparse
import copy
import gzip
import json
import random
from pathlib import Path
import time
import numpy as np
import torch
from interp_audit_common import AuditModel, ROOT, prepare_data, evaluate, write_json
from interp_adapter_audit import load_adapter, evaluate_gate, effects
from interp_common import flatten_alpha
from interp_followup_controls import gzread,gzwrite,latency

OUT=ROOT/'experiments/results/interp_followup_20260929'
WORK=Path('/scratch/yurenh2/interp_followup_20260929')


def locate(runner,indices):
    labels={};off=0
    for l,head in enumerate(runner.predictor.layer_heads):
        for i in indices:
            if off<=i<off+head.out_features:labels[i]=(l,i-off)
        off+=head.out_features
    assert len(labels)==len(indices)
    return labels


@torch.no_grad()
def coefficients(runner,items,indices):
    ids=runner.batch(items);holder={}
    def capture(module,inputs,output):holder['z']=output
    handle=runner.predictor.trunk.register_forward_hook(capture)
    adapted=flatten_alpha(runner.predictor(ids))[:,:,indices]
    handle.remove()
    baseline=flatten_alpha(runner.baseline_predictor(ids))[:,:,indices]
    x=holder['z'].float();x=torch.cat([x,torch.ones_like(x[:,:,:1])],-1)
    return x,baseline-adapted


@torch.no_grad()
def fit_statistics(runner,data,indices):
    items=[x for x in data['train'] if x['task']!='retain']+[x for x in data['train'] if x['task']=='retain'][:2048]
    counts={t:sum(x['task']==t for x in items) for t in ['sva','ioi','retain']}
    weights={'sva':.4,'ioi':.4,'retain':.2}
    d=runner.predictor.layer_heads[0].in_features+1
    gram=torch.zeros(d,d,device='cuda',dtype=torch.float64);rhs=torch.zeros(d,len(indices),device='cuda',dtype=torch.float64)
    y2=torch.zeros(len(indices),device='cuda',dtype=torch.float64)
    for start in range(0,len(items),32):
        batch=items[start:start+32];x,y=coefficients(runner,batch,indices)
        valid=torch.arange(x.shape[1],device='cuda')[None,:]<=runner.last_positions[:,None]
        weight=torch.tensor([weights[a['task']]/counts[a['task']]/len(a['ids']) for a in batch],device='cuda',dtype=torch.float64)
        w=(weight[:,None]*valid).flatten();x=x.reshape(-1,d).double();y=y.reshape(-1,len(indices)).double()
        gram+=x.T@(x*w[:,None]);rhs+=x.T@(y*w[:,None]);y2+=(y.square()*w[:,None]).sum(0)
    return gram,rhs,y2,counts


@torch.no_grad()
def get_rows(runner,indices):
    locations=locate(runner,indices)
    return torch.stack([torch.cat([runner.predictor.layer_heads[l].weight[row],runner.predictor.layer_biases[l][row:row+1]]) for l,row in [locations[i] for i in indices]])


@torch.no_grad()
def set_rows(runner,indices,rows):
    locations=locate(runner,indices)
    for pos,i in enumerate(indices):
        l,row=locations[i]
        runner.predictor.layer_heads[l].weight[row].copy_(rows[pos,:-1])
        runner.predictor.layer_biases[l][row].copy_(rows[pos,-1])


@torch.no_grad()
def teacher_logits(runner,items,gate):
    values=[]
    for start in range(0,len(items),16):values.append(runner.forward(items[start:start+16],alpha_gate=gate).cpu())
    return values


@torch.no_grad()
def validation_kl(runner,items,teachers):
    losses={t:[] for t in ['sva','ioi','retain']}
    for start,target in zip(range(0,len(items),16),teachers):
        batch=items[start:start+16];lp=runner.forward(batch).log_softmax(-1)
        lq=target.to('cuda').log_softmax(-1);kl=(lq.exp()*(lq-lp)).sum(-1)
        for i,x in enumerate(batch):losses[x['task']].append(kl[i].item())
    means={t:float(np.mean(v)) for t,v in losses.items()}
    return means,.4*means['sva']+.4*means['ioi']+.2*means['retain']


def refine_rows(runner,data,indices,union,gate,teachers,seed):
    """KL distill only selected rows; the functional teacher is invariant.

    Its selected rows come from P0; all other rows stay unchanged in the student.
    Therefore the same runner can evaluate both without another adapted copy.
    """
    locations=locate(runner,indices);groups={}
    for l,row in locations.values():groups.setdefault(l,[]).append(row)
    params=[];parameter_keys=[]
    before={n:p.detach().clone() for n,p in runner.predictor.named_parameters() if n.startswith(('layer_heads.','layer_biases.'))}
    for l,rows in groups.items():
        for key,p in [('weight',runner.predictor.layer_heads[l].weight),('bias',runner.predictor.layer_biases[l])]:
            p.requires_grad_(True);params.append(p);parameter_keys.append((l,key))
    train_rows=torch.nn.Parameter(get_rows(runner,indices).detach())
    optimizer=torch.optim.AdamW([train_rows],lr=1e-4,weight_decay=0.,betas=(.9,.95))
    pools={t:[x for x in data['train'] if x['task']==t] for t in ['sva','ioi','retain']}
    rng=random.Random(seed+800);record=[]
    initial,score=validation_kl(runner,data['validation'],teachers)
    best=dict(step=0,score=score,validation_kl=initial);best_rows=get_rows(runner,union);record.append(best)
    for step in range(1,201):
        batch=sum((rng.sample(pools[t],4) for t in ['sva','ioi','retain']),[])
        with torch.no_grad():lq=runner.forward(batch,alpha_gate=gate).log_softmax(-1)
        optimizer.zero_grad(set_to_none=True)
        lp=runner.forward(batch).log_softmax(-1);kl=(lq.exp()*(lq-lp)).sum(-1)
        loss=.4*kl[:4].mean()+.4*kl[4:8].mean()+.2*kl[8:].mean()
        gradients=dict(zip(parameter_keys,torch.autograd.grad(loss,params)))
        # Gather the exact coordinate gradients; only these rows have optimizer state.
        train_rows.grad=torch.stack([torch.cat([gradients[(locations[i][0],'weight')][locations[i][1]],
            gradients[(locations[i][0],'bias')][locations[i][1]:locations[i][1]+1]]) for i in indices]).detach()
        norm=torch.nn.utils.clip_grad_norm_([train_rows],1.)
        if not torch.isfinite(loss) or not torch.isfinite(norm):raise RuntimeError('Nonfinite distillation step')
        optimizer.step();set_rows(runner,indices,train_rows)
        if step%50==0:
            losses,score=validation_kl(runner,data['validation'],teachers)
            row=dict(step=step,score=score,validation_kl=losses);record.append(row)
            if score<best['score']:best=row;best_rows=get_rows(runner,union)
    del optimizer
    for p in params:p.requires_grad_(False)
    set_rows(runner,union,best_rows)
    # Explicitly verify that no unselected row moved under masked optimization.
    for l,head in enumerate(runner.predictor.layer_heads):
        keep=torch.ones(head.out_features,device='cuda',dtype=torch.bool);keep[groups.get(l,[])]=False
        assert torch.equal(head.weight[keep],before[f'layer_heads.{l}.weight'][keep])
        assert torch.equal(runner.predictor.layer_biases[l][keep],before[f'layer_biases.{l}'][keep])
    with torch.no_grad():
        teacher_error=(runner.forward(data['validation'][:16],alpha_gate=gate).cpu()-teachers[0]).abs().max().item()
    assert teacher_error==0
    return dict(validation=record,selected=best,unselected_rows_unchanged=True,teacher_max_logit_error=teacher_error,
                changed_parameter_count=len(indices)*(runner.predictor.layer_heads[0].in_features+1),
                optimizer_allocated_parameter_count=train_rows.numel())


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--seed',type=int,required=True);args=p.parse_args()
    label=f'full_s{args.seed}';r=gzread(OUT/'runs'/label/'results.json.gz');assert r['complete']
    data=copy.deepcopy(prepare_data());data.update(gzread(OUT/'extra_tests.json.gz'))
    for items in data.values():
        for x in items:
            if x['task']=='sva':x['target'],x['foil_id']=x['foil_id'],x['target']
    runner=AuditModel();runner.copy_baseline_predictor()
    load_adapter(runner,'predictor',WORK/'runs'/label/'weights/predictor_best.pt')
    names=['sva_75','sva_189'];union=sorted(set(i for name in names for i in r['all_masks'][name]))
    original=get_rows(runner,union);index={x:i for i,x in enumerate(union)}
    started=time.time();gram,rhs,y2,counts=fit_statistics(runner,data,union)
    ridge_base=gram.diag()[:-1].mean();regularizer=torch.eye(len(gram),device='cuda',dtype=torch.float64);regularizer[-1,-1]=0
    candidates={}
    for strength in [1e-5,1e-3,.1,10.]:
        delta=torch.linalg.solve(gram+strength*ridge_base*regularizer,rhs).T.float()
        candidates[strength]=delta
    report=dict(complete=False,seed=args.seed,protocol='Secondary follow-up after output-only finetuning lost rollback quality. Fit sparse corrections to selected output rows of the full adapted predictor. Training inputs only, task weights 0.4/0.4/0.2 distributed uniformly over prompts within a task and valid positions within a prompt. Baseline predictor coefficients are regression targets. Four ridge strengths selected separately per mask by full-vocabulary KL to the functional rollback teacher on original validation, followed by 200 steps of row-masked KL distillation (lr1e-4, weight_decay0). Choose checkpoint including step0 by validation KL. Frozen adapted encoder/trunk and unselected output rows. No test-based selection. Optimizer state is allocated only for the chosen output rows; full head gradients are gathered into those coordinates.',calibration_counts=counts,arms={})
    for name in names:
        indices=r['all_masks'][name];positions=[index[i] for i in indices]
        gate=torch.ones(len(runner.layout),device='cuda');gate[indices]=0
        set_rows(runner,union,original)
        teachers=teacher_logits(runner,data['validation'],gate)
        model_selection=[];best=None
        for strength,delta in candidates.items():
            updated=original.clone();updated[positions]+=delta[positions]
            set_rows(runner,union,updated)
            losses,score=validation_kl(runner,data['validation'],teachers)
            row=dict(ridge=strength,validation_kl=losses,score=score)
            model_selection.append(row)
            if best is None or score<best['score']:best=row
        delta=candidates[best['ridge']];updated=original.clone();updated[positions]+=delta[positions]
        set_rows(runner,union,updated)
        refinement=refine_rows(runner,data,indices,union,gate,teachers,args.seed)
        updated=get_rows(runner,union)
        baseline=runner.baseline_predictor;runner.baseline_predictor=None
        arm=dict(selected=best,model_selection=model_selection,refinement=refinement,changed_parameters=len(indices)*original.shape[1],stages={})
        calls={'n':0}
        def count(module,inputs,output):calls['n']+=1
        handle=runner.predictor.register_forward_hook(count)
        with torch.no_grad():runner.forward(data['validation'][:16])
        handle.remove();assert calls['n']==1;arm['predictor_calls_per_forward']=calls['n']
        # Only intended output rows change; a tiny patch can be overlaid on the adapted predictor.
        patch=WORK/'runs'/label/'weights'/f'compiled_{name}_patch.pt'
        torch.save(dict(format='dagformer_predictor_row_patch_v1',seed=args.seed,indices=indices,rows=updated[positions].cpu(),base_adapter='predictor_best.pt',metadata=best),patch)
        arm['patch_path']=str(patch);arm['patch_bytes']=patch.stat().st_size
        for split in ['heldout','transfer','confirmation','hard_transfer']:
            ev=evaluate(runner,data[split],16)
            teacher=r['stages'][split]['arms'][name]
            arm['stages'][split]=dict(student=ev,relative_to_teacher=effects(r['stages'][split]['base'],teacher,ev))
        arm['single_latency']=latency(runner,[x for x in data['hard_transfer'] if x['task']=='retain'][:16])
        runner.baseline_predictor=baseline
        set_rows(runner,union,original)
        arm['teacher_latency']=latency(runner,[x for x in data['hard_transfer'] if x['task']=='retain'][:16],gate)
        report['arms'][name]=arm
        gzwrite(OUT/'runs'/label/'compiled.json.gz',report)
        print(label,'COMPILED',name,'ridge',best['ridge'],'KL',best['score'],flush=True)
    set_rows(runner,union,original)
    report.update(complete=True,elapsed_seconds=time.time()-started)
    gzwrite(OUT/'runs'/label/'compiled.json.gz',report)

if __name__=='__main__':main()
