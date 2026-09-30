"""Stage 2: native pre-down-projection neuron patching and routing mediation.

Uses paired grammatical number counterfactuals. Four-point embedding-path gradient attribution
ranks neurons on discovery data; heldout and transfer data test necessity and
sufficiency. This is not a reproduction of RelP or a published benchmark.
"""
from __future__ import annotations
import argparse
from collections import defaultdict
from contextlib import contextmanager
import json
from pathlib import Path
import time
import numpy as np
import torch
from interp_audit_common import AuditModel, RESULT_ROOT, evaluate, metrics, paired_stats, prepare_data, write_json
from interp_common import flatten_alpha


def counterfactuals(items):
    groups=defaultdict(list)
    for x in items: groups[x['group']].append(x)
    assert all(len(v)==2 for v in groups.values())
    return [next(y for y in groups[x['group']] if y is not x) for x in items]


@contextmanager
def neurons(runner, capture=None, donor=None, indices=None):
    handles=[]
    width=runner.base.model.layers[0].mlp.down_proj.in_features
    for l,layer in enumerate(runner.base.model.layers):
        chosen=[] if indices is None else [i%width for i in indices if i//width==l]
        def hook(module, inputs, l=l, chosen=chosen):
            z=inputs[0]
            if donor is not None and chosen:
                z=z.clone(); z[:,:,chosen]=donor[l][:,:,chosen]
            if capture is not None: capture.append(z)
            return (z,)
        handles.append(layer.mlp.down_proj.register_forward_pre_hook(hook))
    try: yield
    finally:
        for h in handles: h.remove()


@contextmanager
def embedding_path(runner, donor_items, level):
    ids=runner.batch(donor_items)
    modules=[runner.base.model.embed_tokens]
    if runner.kind=='dag': modules.append(runner.predictor.embed)
    handles=[]
    for module in modules:
        with torch.no_grad(): donor=module(ids).detach()
        def hook(module, inputs, output, donor=donor):
            return (donor+level*(output-donor)).detach().requires_grad_(True)
        handles.append(module.register_forward_hook(hook))
    try: yield
    finally:
        for h in handles: h.remove()


@contextmanager
def effective_routes(runner, gate, position_mask=None):
    """Gate the *effective* predictor+correction coefficients, retaining dynamics.

    At gate 1, return the unmodified correction exactly. At hard gate 0, return
    negative BF16 predictor coefficients so the effective coefficient is zero.
    Intermediate values have the derivative of a multiplicative effective gate.
    """
    if gate is None:
        yield; return
    assert runner.kind=='dag' and runner.model.use_local_correction
    routing={}
    def save_pred(module, inputs, output): routing['p']=flatten_alpha(output)
    handles=[runner.predictor.register_forward_hook(save_pred)]
    off=0
    for l,module in enumerate(runner.model.correction_mlps,1):
        n=(3*runner.cfg['num_attention_heads']+1)*(l+1)
        def hook(module, inputs, output, start=off, end=off+n):
            p=routing['p'][:,:,start:end].to(torch.bfloat16).float()
            effective=(p.to(torch.bfloat16)+output.to(torch.bfloat16)).float()
            g=gate[start:end].view(1,1,-1)
            if position_mask is not None:
                g=1+position_mask[:,:,None]*(g-1)
            return torch.where(g==0,-p,output+(g-1)*effective)
        handles.append(module.register_forward_hook(hook)); off+=n
    assert off==len(gate)
    try: yield
    finally:
        for h in handles: h.remove()


@torch.no_grad()
def capture_neurons(runner, batch):
    zs=[]
    with neurons(runner,capture=zs): logits=runner.forward(batch)
    return logits,[z.detach() for z in zs]


def rank_neurons(runner, items, batch_size):
    donors=counterfactuals(items)
    width=runner.base.model.layers[0].mlp.down_proj.in_features
    scores=torch.zeros(len(runner.base.model.layers),width,device='cuda')
    for start in range(0,len(items),batch_size):
        batch=items[start:start+batch_size]; donor=donors[start:start+batch_size]
        _,clean=capture_neurons(runner,batch)
        _,corrupt=capture_neurons(runner,donor)
        for level in [.125,.375,.625,.875]:
            zs=[]
            with embedding_path(runner,donor,level), neurons(runner,capture=zs):
                logits=runner.forward(batch)
                _,margin=metrics(logits,batch)
                grads=torch.autograd.grad(margin.sum(),zs)
            for l,(a,b,g) in enumerate(zip(clean,corrupt,grads)):
                scores[l]+=((a.float()-b.float())*g.float()).sum((0,1))/len(items)/4
    return scores.flatten().cpu().numpy()


def summarize(values):
    groups=[x['group'] for x in values]
    return {k:paired_stats([x[k] for x in values],groups) for k in ['margin','nll','correct']}


@torch.no_grad()
def patch_eval(runner, items, choices, batch_size, route_gate=None):
    donors=counterfactuals(items)
    values=defaultdict(list)
    max_error=0.
    for start in range(0,len(items),batch_size):
        batch=items[start:start+batch_size]; donor=donors[start:start+batch_size]
        clean_logits,clean_z=capture_neurons(runner,batch)
        corrupt_logits,corrupt_z=capture_neurons(runner,donor)
        # Same-input replay covers every native neuron, including nonlinear norms.
        if start==0:
            all_indices=list(range(sum(z.shape[-1] for z in clean_z)))
            with neurons(runner,donor=clean_z,indices=all_indices): replay=runner.forward(batch)
            max_error=(replay-clean_logits).abs().max().item()
            assert max_error==0
        def record(name,logits):
            nll,margin=metrics(logits,batch)
            for i,x in enumerate(batch):
                values[name].append(dict(group=x['group'],nll=nll[i].item(),margin=margin[i].item(),correct=float(margin[i]>0)))
        record('clean',clean_logits);record('corrupt',corrupt_logits)
        for name,indices in choices.items():
            for kind,inputs,source in [('necessity',batch,corrupt_z),('sufficiency',donor,clean_z)]:
                with effective_routes(runner,route_gate),neurons(runner,donor=source,indices=indices):
                    logits=runner.forward(inputs)
                record(name+'/'+kind,logits)
    result=dict(identity_max_logit_error=max_error,arms={})
    clean=np.array([x['margin'] for x in values['clean']])
    corrupt=np.array([x['margin'] for x in values['corrupt']])
    denom=(clean-corrupt).mean()
    for name,rows in values.items():
        margin=np.array([x['margin'] for x in rows])
        effect=clean-margin if name.endswith('/necessity') else margin-corrupt
        result['arms'][name]=dict(summary=summarize(rows),values=rows,
            margin_effect=paired_stats(effect,[x['group'] for x in rows]),
            fraction_total_counterfactual_effect=float(effect.mean()/denom) if denom>0 else None)
    return result


def matched_neurons(indices,width,seed):
    rng=np.random.default_rng(seed); result=[]
    for l in sorted(set(i//width for i in indices)):
        k=sum(i//width==l for i in indices)
        result.extend((l*width+rng.choice(width,k,False)).tolist())
    return result


def route_mediation(runner, data, indices, batch_size):
    size=len(runner.layout)
    items=[x for x in data['discovery'] if x['task']=='sva']
    donor=counterfactuals(items)
    gradients=[]
    for patched in [False,True]:
        total=torch.zeros(size,device='cuda')
        for start in range(0,len(items),batch_size):
            batch=items[start:start+batch_size]
            _,z=capture_neurons(runner,donor[start:start+batch_size])
            gate=torch.ones(size,device='cuda',requires_grad=True)
            with effective_routes(runner,gate),neurons(runner,donor=z if patched else None,indices=indices):
                logits=runner.forward(batch)
                _,margin=metrics(logits,batch)
                grad=torch.autograd.grad(margin.sum(),gate)[0]
            total+=grad.detach()/len(items)
        gradients.append(total.cpu().numpy())
    interaction=gradients[0]-gradients[1]
    ranking=interaction-.1*np.abs(gradients[0])
    choices={};rng=np.random.default_rng(2026093024)
    for k in [4,16,64]:
        picked=np.argsort(-ranking)[:k].tolist();choices[f'top_{k}']=picked
        for repeat in range(3):
            random_indices=[]
            strata=defaultdict(list)
            for i in picked:
                x=runner.layout[i];strata[(x['layer'],x['stream'])].append(i)
            for (l,s),v in strata.items():
                pool=[i for i,x in enumerate(runner.layout) if x['layer']==l and x['stream']==s]
                random_indices.extend(rng.choice(pool,len(v),False).tolist())
            choices[f'random_{k}_{repeat}']=random_indices
    result=dict(neurons=indices,route_interaction_scores=interaction.tolist(),choices=choices,labels=runner.layout,stages={})
    for split in ['heldout','transfer']:
        items=[x for x in data[split] if x['task']=='sva'];donor=counterfactuals(items)
        vals=defaultdict(list)
        for start in range(0,len(items),batch_size):
            batch=items[start:start+batch_size];_,z=capture_neurons(runner,donor[start:start+batch_size])
            for name,route_indices in {'intact':[],**choices}.items():
                gate=torch.ones(size,device='cuda');gate[route_indices]=0
                rows=[]
                for patched in [False,True]:
                    with torch.no_grad(),effective_routes(runner,gate),neurons(runner,donor=z if patched else None,indices=indices):
                        logits=runner.forward(batch);_,margin=metrics(logits,batch)
                    rows.append(margin.cpu().numpy())
                for i,x in enumerate(batch):
                    vals[name].append(dict(group=x['group'],clean_margin=float(rows[0][i]),patched_margin=float(rows[1][i]),
                                          feature_effect=float(rows[0][i]-rows[1][i]),clean_correct=float(rows[0][i]>0)))
        base=np.array([x['feature_effect'] for x in vals['intact']])
        result['stages'][split]={}
        for name,rows in vals.items():
            groups=[x['group'] for x in rows]
            reduction=base-np.array([x['feature_effect'] for x in rows])
            result['stages'][split][name]=dict(values=rows,summary={k:paired_stats([x[k] for x in rows],groups)
                for k in ['clean_margin','patched_margin','feature_effect','clean_correct']},
                feature_effect_reduction=paired_stats(reduction,groups),
                fraction_feature_effect_removed=float(reduction.mean()/base.mean()))
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--kind',choices=['dag','dense'],required=True)
    p.add_argument('--batch-size',type=int,default=8);p.add_argument('--out',type=Path,default=RESULT_ROOT/'neuron_flow')
    args=p.parse_args();torch.manual_seed(2026093020)
    data=prepare_data();runner=AuditModel(args.kind);started=time.time()
    result=dict(complete=False,kind=args.kind,protocol='Original pretrained model. Grammatical SVA paired counterfactuals. Discovery-only four midpoint embedding-path gradients times endpoint neuron-activation differences, jointly interpolating predictor and backbone input embeddings for DAG. Patch pre-down-projection MLP neuron identities at all positions; matched-layer random controls. Native neuron counts are comparable between same-width DAG and dense. One checkpoint per architecture.')
    result['baseline_tasks']=evaluate(runner,[x for x in data['discovery'] if x['task']!='retain'],args.batch_size)
    items=[x for x in data['discovery'] if x['task']=='sva']
    scores=rank_neurons(runner,items,args.batch_size)
    result['scores']=scores.tolist();width=runner.base.model.layers[0].mlp.down_proj.in_features
    choices={}
    for k in [16,32,64,128,256]:
        idx=np.argsort(-scores)[:k].tolist();choices[f'top_{k}']=idx
        for repeat in range(3):choices[f'random_{k}_{repeat}']=matched_neurons(idx,width,2026093021+repeat)
    result['choices']=choices;result['width']=width;result['stages']={}
    write_json(args.out/f'{args.kind}.json',result)
    for split in ['heldout','transfer']:
        items=[x for x in data[split] if x['task']=='sva']
        result['stages'][split]=patch_eval(runner,items,choices,args.batch_size)
        write_json(args.out/f'{args.kind}.json',result)
        print(args.kind,split,{k:round(v['summary']['correct']['mean'],3) for k,v in result['stages'][split]['arms'].items() if not k.startswith('random')},flush=True)
    if args.kind=='dag':
        # Primary neuron budget is fixed in advance; no choice using heldout data.
        result['routing_mediation']=route_mediation(runner,data,choices['top_64'],args.batch_size)
    result.update(complete=True,elapsed_seconds=time.time()-started)
    write_json(args.out/f'{args.kind}.json',result)

if __name__=='__main__':main()
