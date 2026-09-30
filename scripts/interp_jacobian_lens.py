"""Stage 3: average downstream Jacobians on complete frozen DAG/dense models.

Full 1024 x 1024 Jacobians, averaged over calibration contexts and all causal
source/destination token pairs. No low-rank or identity approximation is fitted.
This small-corpus experiment tests a J-lens-style readout, not a reproduction of
the paper's 1,000-prompt calibration or its interpretability claims.
"""
from __future__ import annotations
import argparse
from contextlib import contextmanager
import copy
import json
from pathlib import Path
import time
import numpy as np
import torch
from transformers import AutoTokenizer
from interp_audit_common import AuditModel, RESULT_ROOT, TOKENIZER, metrics, paired_stats, prepare_data, write_json
from interp_neuron_flow import counterfactuals


@contextmanager
def residuals(runner, layers, capture, injections=None, differentiable=False):
    """Add perturbations to native layer writes, hence also every later DAG reuse.

    X_l is MLP input + post-FF-normalized output in OLMo2 and FourWayDAGFormer.
    Injecting at the latter changes the full layer state by the same amount.
    Capture full states for readout; differentiate with respect to added leaves.
    """
    handles=[];mlp_inputs={};capture['states']={};capture['leaves']={}
    for l in layers:
        layer=runner.base.model.layers[l]
        def pre(module, inputs, l=l):mlp_inputs[l]=inputs[0]
        def post(module, inputs, output, l=l):
            delta=None
            if differentiable:
                delta=torch.zeros_like(output,requires_grad=True)
                capture['leaves'][l]=delta
            elif injections is not None and l in injections:delta=injections[l].to(output.dtype)
            modified=output if delta is None else output+delta
            capture['states'][l]=mlp_inputs[l]+modified
            return modified
        handles.append(layer.mlp.register_forward_pre_hook(pre))
        handles.append(layer.post_feedforward_layernorm.register_forward_hook(post))
    def final(module,inputs):capture['final']=inputs[0]
    handles.append(runner.base.model.norm.register_forward_pre_hook(final))
    try:yield
    finally:
        for h in handles:h.remove()


def fit(runner,items,layers,batch_size):
    d=runner.cfg['hidden_size'];result={l:torch.zeros(d,d,device='cuda') for l in layers}
    started=time.time();total_pairs=0
    for start in range(0,len(items),batch_size):
        batch=items[start:start+batch_size];cap={}
        assert len({len(x['ids']) for x in batch})==1
        with residuals(runner,layers,cap,differentiable=True):runner.forward(batch)
        output=cap['final'].float().sum((0,1))
        leaves=[cap['leaves'][l] for l in layers]
        # The causal graph supplies exact zeros for destination before source.
        for j in range(d):
            gs=torch.autograd.grad(output[j],leaves,retain_graph=j<d-1)
            for l,g in zip(layers,gs):result[l][j]+=g.float().sum((0,1))
        t=len(batch[0]['ids']);total_pairs+=len(batch)*t*(t+1)//2
        print(runner.kind,'JACOBIAN FIT',start+len(batch),'/',len(items),'seconds',round(time.time()-started,1),flush=True)
    return {l:m.cpu()/total_pairs for l,m in result.items()},total_pairs


@torch.no_grad()
def readout(runner,items,layers,matrices,batch_size,tok):
    rows={name:[] for name in ['model']+[f'{m}_{l}' for l in layers for m in ['direct','jacobian']]}
    examples=[]
    # Bypass AuditModel's LM-head position selector for already selected vectors.
    weight=runner.base.lm_head.weight
    for start in range(0,len(items),batch_size):
        batch=items[start:start+batch_size];cap={}
        with residuals(runner,layers,cap):logits=runner.forward(batch)
        variants={'model':logits}
        for l in layers:
            h=cap['states'][l][torch.arange(len(batch),device='cuda'),runner.last_positions]
            for name,x in [('direct',h),('jacobian',h.float()@matrices[l].T)]:
                normalized=runner.base.model.norm(x.to(weight.dtype))
                variants[f'{name}_{l}']=torch.nn.functional.linear(normalized,weight).float()
        for name,logits in variants.items():
            nll,margin=metrics(logits,batch)
            for i,x in enumerate(batch):rows[name].append(dict(group=x['group'],task=x['task'],nll=nll[i].item(),margin=margin[i].item(),
                correct=float(margin[i]>0),vocab_correct=float(logits[i].argmax()==x['target'])))
        for i,x in enumerate(batch):
            if len(examples)<12 and x['task']=='retain':
                examples.append(dict(prompt=x['prompt'],target=x['answer'],top_tokens={name:tok.convert_ids_to_tokens(v[i].topk(5).indices.tolist()) for name,v in variants.items()}))
    result=dict(examples=examples,arms={})
    for name,values in rows.items():
        summaries={}
        for task in sorted({x['task'] for x in values}):
            subset=[x for x in values if x['task']==task]
            summaries[task]={k:paired_stats([x[k] for x in subset],[x['group'] for x in subset]) for k in ['nll','margin','correct','vocab_correct']}
        result['arms'][name]=dict(summary=summaries,values=values)
    return result


@torch.no_grad()
def perturbation_check(runner,items,layers,matrices):
    """Predict finite causal responses on unseen contexts using the global J.

    A heldout direction is applied identically at every source token. Mean final
    residual response is J delta * (T+1)/2 by the fit's pair-weighting convention.
    Compare to an optimally rescaled identity by cosine (scale independent).
    Same-input zero injection also checks the hook's fidelity.
    """
    rng=torch.Generator(device='cuda').manual_seed(2026093031)
    values={str(l):[] for l in layers};identity_error=0.
    for index,item in enumerate(items[:32]):
        cap={}
        with residuals(runner,layers,cap):original=runner.forward([item])
        t=len(item['ids']);d=runner.cfg['hidden_size']
        for l in layers:
            h=cap['states'][l];base=cap['final']
            zero=torch.zeros_like(h);check={}
            if index==0:
                with residuals(runner,[l],check,{l:zero}):replay=runner.forward([item])
                identity_error=max(identity_error,(replay-original).abs().max().item())
            direction=torch.randn(d,device='cuda',generator=rng)
            direction=direction/direction.square().mean().sqrt()*h.float().square().mean().sqrt()
            predicted=matrices[l]@direction*((t+1)/2)
            for scale in [.05,.2]:
                delta=direction.view(1,1,-1).expand_as(h)*scale
                positive={};negative={}
                with residuals(runner,[l],positive,{l:delta}):runner.forward([item])
                with residuals(runner,[l],negative,{l:-delta}):runner.forward([item])
                observed=(positive['final'].float()-negative['final'].float()).mean((0,1))/(2*scale)
                cos=lambda a,b:torch.nn.functional.cosine_similarity(a,b,dim=0).item()
                rel=(observed-predicted).norm()/observed.norm().clamp_min(1e-8)
                values[str(l)].append(dict(group=item['group'],scale=scale,jacobian_cosine=cos(observed,predicted),
                    identity_cosine=cos(observed,direction),jacobian_relative_error=rel.item()))
        if index%8==7:print(runner.kind,'PERTURBATIONS',index+1,flush=True)
    assert identity_error==0
    return dict(identity_max_logit_error=identity_error,values=values,summary={l:{str(scale):{k:paired_stats([x[k] for x in rows if x['scale']==scale],
        [x['group'] for x in rows if x['scale']==scale]) for k in ['jacobian_cosine','identity_cosine','jacobian_relative_error']} for scale in [.05,.2]} for l,rows in values.items()})


@torch.no_grad()
def number_readout(runner,items,layers,matrices,batch_size):
    """Read grammatical number in a paired state difference, then test prediction.

    Lens logit directions are compared to *actual* changes caused by replacing
    the last-position full state with its grammatical-number counterfactual.
    This preserves DAG reuse of the edited source by all subsequent layers.
    """
    donors=counterfactuals(items);values={str(l):[] for l in layers}
    weight=runner.base.lm_head.weight
    for start in range(0,len(items),batch_size):
        batch=items[start:start+batch_size];donor=donors[start:start+batch_size]
        clean={};corrupt={}
        with residuals(runner,layers,clean):logits=runner.forward(batch)
        with residuals(runner,layers,corrupt):runner.forward(donor)
        ar=torch.arange(len(batch),device='cuda');pos=runner.last_positions
        target=torch.tensor([x['target'] for x in batch],device='cuda');foil=torch.tensor([x['foil_id'] for x in batch],device='cuda')
        for l in layers:
            h=clean['states'][l][ar,pos];counter=corrupt['states'][l][ar,pos]
            delta=torch.zeros_like(clean['states'][l]);delta[ar,pos]=counter-h
            with residuals(runner,[l],{},injections={l:delta}):edited=runner.forward(batch)
            true_effect=(edited[ar,target]-edited[ar,foil])-(logits[ar,target]-logits[ar,foil])
            predictions={}
            for name,matrix in [('direct',None),('jacobian',matrices[l])]:
                def get_margin(x):
                    transformed=x if matrix is None else x.float()@matrix.T
                    z=runner.base.model.norm(transformed.to(weight.dtype))
                    return (z.float()*(weight[target].float()-weight[foil].float())).sum(-1)
                predictions[name]=(get_margin(counter)-get_margin(h))
            for i,x in enumerate(batch):values[str(l)].append(dict(group=x['group'],true_effect=true_effect[i].item(),
                direct=predictions['direct'][i].item(),jacobian=predictions['jacobian'][i].item()))
    summary={}
    for l,rows in values.items():
        a=np.array([x['true_effect'] for x in rows]);summary[l]={}
        for key in ['direct','jacobian']:
            b=np.array([x[key] for x in rows]);summary[l][key]=dict(pearson=float(np.corrcoef(a,b)[0,1]),
                sign_agreement=paired_stats((a*b)>0,[x['group'] for x in rows]),actual_effect=paired_stats(a,[x['group'] for x in rows]))
    return dict(values=values,summary=summary)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--kind',choices=['dag','dense'],required=True)
    p.add_argument('--n-calibration',type=int,default=64);p.add_argument('--length',type=int,default=32)
    p.add_argument('--batch-size',type=int,default=8);p.add_argument('--layers',type=int,nargs='+',default=[3,6,9])
    p.add_argument('--out',type=Path,default=RESULT_ROOT/'jacobian_lens')
    p.add_argument('--weights',type=Path,default=Path('/scratch/yurenh2/interp_audit_20260929'));p.add_argument('--reuse-matrices',action='store_true')
    args=p.parse_args();runner=AuditModel(args.kind);data=prepare_data();tok=AutoTokenizer.from_pretrained(str(TOKENIZER),local_files_only=True)
    calibration=copy.deepcopy([x for x in data['train'] if x['task']=='retain'][:args.n_calibration])
    for x in calibration:x['ids']=x['ids'][:args.length]
    started=time.time();path=args.weights/f'{args.kind}_jacobians.pt'
    if args.reuse_matrices:
        checkpoint=torch.load(path,weights_only=False);matrices=checkpoint['matrices'];pairs=checkpoint['causal_pairs']
    else:
        matrices,pairs=fit(runner,calibration,args.layers,args.batch_size)
        torch.save(dict(matrices=matrices,causal_pairs=pairs,calibration=calibration),path)
    result=dict(complete=False,kind=args.kind,n_calibration=len(calibration),context_length=args.length,causal_pairs=pairs,layers_zero_based=args.layers,
        matrix_shape=[runner.cfg['hidden_size']]*2,
        protocol='Full downstream residual Jacobian, averaged uniformly over all causal source/destination pairs of 64 WikiText train contexts (32 tokens). Native frozen architecture, including state-dependent local correction. No model or lens fitting on heldout texts. Both models use identical prompts. A small calibration experiment, not a paper reproduction. Residual-write injection affects all subsequent DAG consumers; it is not an isolated per-edge intervention.')
    matrices={l:m.to('cuda') for l,m in matrices.items()}
    write_json(args.out/f'{args.kind}.json',result)
    result['readout']={}
    for split in ['heldout','transfer']:
        result['readout'][split]=readout(runner,data[split],args.layers,matrices,args.batch_size,tok)
        write_json(args.out/f'{args.kind}.json',result)
    natural=copy.deepcopy([x for x in data['heldout'] if x['task']=='retain'])
    for x in natural:x['ids']=x['ids'][:args.length]
    result['perturbations']=perturbation_check(runner,natural,args.layers,matrices)
    result['number_readout']={}
    for split in ['heldout','transfer']:
        result['number_readout'][split]=number_readout(runner,[x for x in data[split] if x['task']=='sva'],args.layers,matrices,args.batch_size)
    result.update(complete=True,elapsed_seconds=time.time()-started)
    write_json(args.out/f'{args.kind}.json',result)
    print(args.kind,'COMPLETE',result['elapsed_seconds'],flush=True)

if __name__=='__main__':main()
