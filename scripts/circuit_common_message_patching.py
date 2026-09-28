"""Matched per-head Q/K/V message interchange in dense and DAG 300M models.

The common site is (layer, stream, head) at three oracle historical value-token
positions. Q/K are intervened before full-dimensional normalization; V before
attention. This is a comparison of local intervention sets, not full circuits.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import gc
import importlib.metadata
import json
from pathlib import Path
import random
import subprocess
import time

import numpy as np
import torch

from circuit_message_patching import measures, summaries
from interp_common import load_elh


def specification(name, sites=None, layer_stream=None, identity=False):
    return dict(name=name, sites=sites, layer_stream=layer_stream, identity=identity)


def load_data(path):
    raw = json.loads(path.read_text())
    return {key:{**data, **{field:torch.tensor(data[field]) for field in ('ids','a','b')}}
            for key,data in raw.items()}


class CommonInterchange:
    def __init__(self, model, predictor, kind, layers, heads, dimension):
        self.model, self.predictor, self.kind = model, predictor, kind
        self.layers, self.heads, self.dimension = layers, heads, dimension
        self.head_dim = dimension // heads
        self.base = model.olmo if kind == 'dag' else model
        self.spec = self.donor = self.capture = None
        self.positions = None
        self.qkv_calls = 0
        self.original_einsum = torch.einsum
        self.seen = set()
        self.edit_sq = None
        self.original_sq = None

    def edit(self, x, layer, stream):
        # Canonical shape for both models: [batch, token, head, head_dim].
        assert x.ndim == 4 and x.shape[-2:] == (self.heads,self.head_dim), x.shape
        key = (layer,stream)
        assert key not in self.seen, key
        self.seen.add(key)
        if self.capture is not None:
            self.capture[key] = x[:,self.positions].detach().clone()
        if self.spec is None:
            return x
        if self.spec['sites'] is not None:
            heads = [h for l,s,h in self.spec['sites'] if (l,s)==key]
        elif self.spec['layer_stream'] is not None:
            heads = list(range(self.heads)) if tuple(self.spec['layer_stream'])==key else []
        else:
            heads = list(range(self.heads))
        if not heads:
            return x
        old = x[:,self.positions]
        new = old.clone()
        new[:,:,heads] = self.donor[key][:,:,heads]
        self.edit_sq += (new.float()-old.float()).square().sum((1,2,3))
        self.original_sq += old[:,:,heads].float().square().sum((1,2,3))
        result = x.clone()
        result[:,self.positions] = new
        return result

    def norm_input(self, layer, stream):
        def hook(module, inputs):
            x = inputs[0]
            assert x.ndim == 3 and x.shape[-1] == self.dimension, x.shape
            changed = self.edit(x.reshape(*x.shape[:2],self.heads,self.head_dim),layer,stream)
            return (changed.reshape_as(x),)
        return hook

    def value_output(self, layer):
        def hook(module, inputs, output):
            assert output.ndim == 3 and output.shape[-1] == self.dimension, output.shape
            changed = self.edit(output.reshape(*output.shape[:2],self.heads,self.head_dim),layer,'v')
            return changed.reshape_as(output)
        return hook

    def einsum(self, equation, *operands, **kwargs):
        out = self.original_einsum(equation,*operands,**kwargs)
        eq = equation.replace(' ','') if isinstance(equation,str) else ''
        if eq == 'lbthd,bthl->bhtd':
            layer = operands[0].shape[0]-1
            stream = 'qkv'[self.qkv_calls%3]
            assert layer == self.qkv_calls//3+1
            self.qkv_calls += 1
            if stream == 'v':
                out = self.edit(out.permute(0,2,1,3),layer,'v').permute(0,2,1,3)
        return out

    @contextmanager
    def installed(self):
        handles=[]
        for layer, block in enumerate(self.base.model.layers):
            attn=block.self_attn
            handles.append(attn.q_norm.register_forward_pre_hook(self.norm_input(layer,'q')))
            handles.append(attn.k_norm.register_forward_pre_hook(self.norm_input(layer,'k')))
            # FourWay layers >=1 project sources by F.linear, then mix by einsum.
            if self.kind=='dense' or layer==0:
                handles.append(attn.v_proj.register_forward_hook(self.value_output(layer)))
        handles.append(self.base.lm_head.register_forward_pre_hook(lambda module,inputs:(inputs[0][:,-1:],)))
        if self.kind=='dag':
            torch.einsum=self.einsum
        try:
            yield
        finally:
            torch.einsum=self.original_einsum
            for handle in handles:
                handle.remove()

    @torch.inference_mode()
    def forward(self, ids, positions, capture=False, spec=None, donor=None):
        self.positions=positions
        self.capture={} if capture else None
        self.spec,self.donor=spec,donor
        self.qkv_calls=0
        self.seen=set()
        self.edit_sq=torch.zeros(len(ids),device=ids.device)
        self.original_sq=torch.zeros(len(ids),device=ids.device)
        if self.kind=='dag':
            logits=self.model(ids,self.predictor(ids))[:,-1].float()
            assert self.qkv_calls==3*(self.layers-1)
        else:
            logits=self.model(ids,use_cache=False).logits[:,-1].float()
        assert len(self.seen)==3*self.layers, self.seen
        cache=self.capture
        edit_norm=dict(intervention_l2=self.edit_sq.sqrt().cpu().tolist(),
                       recipient_local_l2=self.original_sq.sqrt().cpu().tolist(),
                       relative_intervention_l2=(self.edit_sq/self.original_sq.clamp_min(1.e-20)).sqrt().cpu().tolist())
        self.capture=self.spec=self.donor=None
        return logits,cache,edit_norm


def control_sites(sites, heads, seed):
    """Shared per-layer head permutation; counts per layer/stream stay exact.

    A selected subset may overlap its permutation, especially when many heads
    are selected. Permutations have no fixed heads, but are not norm matched.
    """
    rng=random.Random(seed)
    maps={}
    for layer in sorted({x[0] for x in sites}):
        perm=list(range(heads))
        while any(i==h for i,h in enumerate(perm)):
            rng.shuffle(perm)
        maps[layer]=perm
    return [(l,s,maps[l][h]) for l,s,h in sites]


class Run:
    def __init__(self,args):
        self.args=args
        args.out.mkdir(parents=True,exist_ok=True)
        self.start=time.time()
        self.result=dict(complete=False,args={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
                         git_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
                         versions={k:importlib.metadata.version(k) for k in ('torch','transformers')},
                         models={},protocol=dict(
                             data='Exact same directed pairs as the native-message pilot; three oracle past value-token locations. Discovery32/heldout64/period128-transfer64/unchanged-query64 independent pairs, two directions each.',
                             sites='One (layer, Q/K/V stream, head), 64 dimensions, at three historical positions. Q/K norm inputs and V before attention; no source-edge decomposition.',
                             selection='Each model: all 12*3=36 layer-stream scans, top4 layer-streams then 4*16=64 single-head scans. Rank the 64 sites by discovery donor-margin increase. Fixed top1/2/4/8/16; no heldout selection.',
                             intervention='Replace recipient mixed head message with donor message; later activations/routing/corrections recompute naturally. All-QKV and same-input identity controls.',
                             controls='Three deranged per-layer head permutations for each selected set, shared across streams; exact layer-stream counts; may overlap selected sites. Not edit-norm or NLL matched.',
                             statistics='4000 bootstrap resamples of independent paired blocks, keeping both directions together. DAG-dense contrasts use identical pair resamples across models. Fixed checkpoints; unadjusted intervals.',
                             limits='Comparison of local intervention interfaces at known token locations, not a complete circuit, minimality proof, natural-language semantics, or multi-seed architecture evidence. Parameter counts differ; same backbone scale.'))

    def save(self):
        self.result['elapsed_seconds']=time.time()-self.start
        (self.args.out/'results.json').write_text(json.dumps(self.result,indent=2)+'\n')

    def evaluate(self,kind,patch,stage,data,specs):
        print(f'START {kind}/{stage}: {len(data["ids"])} cases, {len(specs)} arms',flush=True)
        values={s['name']:{} for s in specs}
        values.update(reference={},donor_baseline={})
        max_error=0.
        for start in range(0,len(data['ids']),self.args.batch_size):
            ids=data['ids'][start:start+self.args.batch_size].cuda()
            a=data['a'][start:start+self.args.batch_size].cuda()
            b=data['b'][start:start+self.args.batch_size].cuda()
            baseline,cache,_=patch.forward(ids,data['positions'],capture=True)
            swap=torch.arange(len(ids),device=ids.device)^1
            donor={k:v[swap] for k,v in cache.items()}
            def record(name,logits,norm):
                row=measures(logits,a,b,baseline)
                row.update(norm)
                for k,v in row.items():
                    values[name].setdefault(k,[]).extend(v)
            zero={k:[0.]*len(ids) for k in ('intervention_l2','recipient_local_l2','relative_intervention_l2')}
            record('reference',baseline,zero)
            record('donor_baseline',baseline[swap],zero)
            for spec in specs:
                logits,_,norm=patch.forward(ids,data['positions'],spec=spec,
                                           donor=cache if spec['identity'] else donor)
                if spec['identity']:
                    max_error=max(max_error,(logits-baseline).abs().max().item())
                record(spec['name'],logits,norm)
            if start==0 or (start+len(ids))%32==0:
                print(f'{kind}/{stage}: {start+len(ids)}/{len(data["ids"])}, elapsed {time.time()-self.start:.1f}s',flush=True)
        ref,don=values['reference'],values['donor_baseline']
        correct=np.asarray(ref['recipient_vocab_accuracy']).reshape(-1,2).all(1).tolist()
        record=dict(n_pairs=len(data['ids'])//2,period=data['period'],query=data['query'],
                    both_baselines_correct_pairs=sum(correct),max_identity_logit_error=max_error,
                    specs=specs,arms={})
        for name,vals in values.items():
            record['arms'][name]=dict(values=vals,summary=summaries(vals,ref,don,correct,data['control']))
        self.result['models'][kind]['stages'][stage]=record
        self.save()
        assert max_error==0.,max_error
        print('DONE',kind,stage,'baseline acc',np.mean(ref['recipient_vocab_accuracy']),flush=True)
        return record

    def run_model(self,kind,data):
        path=self.args.dag_model if kind=='dag' else self.args.dense_model
        elh=load_elh()
        cfg=elh.load_config(str(path/'config.yaml'))
        if kind=='dag':
            model,predictor=elh.load_fourway(str(path/'checkpoint.pt'),cfg,torch.device('cuda'))
            model.use_triton_kernel=False
            assert not model.use_v_norm, 'Common V pre-attention protocol requires this 300M no-V-norm checkpoint'
        else:
            model,predictor=elh.load_dense(str(path/'checkpoint.pt'),cfg,torch.device('cuda')),None
            model.config._attn_implementation='sdpa'
        layers,heads,dim=cfg['num_hidden_layers'],cfg['num_attention_heads'],cfg['hidden_size']
        assert (layers,heads,dim)==(12,16,1024)
        patch=CommonInterchange(model,predictor,kind,layers,heads,dim)
        self.result['models'][kind]=dict(path=str(path),head_dimension=dim//heads,
                                        parameters=sum(p.numel() for p in model.parameters())+
                                        (sum(p.numel() for p in predictor.parameters()) if predictor is not None else 0),
                                        stages={},selection={})
        ids=data['discovery']['ids'][:2].cuda()
        with torch.inference_mode():
            unwrapped=(model(ids,predictor(ids)) if kind=='dag' else model(ids,use_cache=False).logits)[:,-1].float()
        with patch.installed():
            wrapped,_,_=patch.forward(ids,data['discovery']['positions'])
            error=(unwrapped-wrapped).abs().max().item()
            self.result['models'][kind]['unwrapped_parity_max_logit_error']=error
            assert error==0.,error
            broad=[specification('identity',identity=True),specification('all_qkv')]
            broad += [specification(f'layer/{l}/{s}',layer_stream=(l,s)) for l in range(layers) for s in 'qkv']
            stage=self.evaluate(kind,patch,'discovery_layer_streams',data['discovery'],broad)
            chosen=sorted([(l,s) for l in range(layers) for s in 'qkv'],
                          key=lambda x:stage['arms'][f'layer/{x[0]}/{x[1]}']['summary']['donor_margin']['delta'],reverse=True)[:4]
            candidates=[(l,s,h) for l,s in chosen for h in range(heads)]
            name=lambda x:'site/'+'/'.join(map(str,x))
            specs=[specification(name(x),sites=[x]) for x in candidates]
            stage=self.evaluate(kind,patch,'discovery_heads',data['discovery'],specs)
            ordered=sorted(candidates,key=lambda x:stage['arms'][name(x)]['summary']['donor_margin']['delta'],reverse=True)
            selection=dict(layer_streams=chosen,ordered_sites=ordered,
                           scores=[stage['arms'][name(x)]['summary']['donor_margin']['delta'] for x in ordered],
                           topk_sizes=[1,2,4,8,16])
            self.result['models'][kind]['selection']=selection
            final=[specification('identity',identity=True),specification('all_qkv')]
            for k in selection['topk_sizes']:
                selected=ordered[:k]
                final.append(specification(f'top{k}',sites=selected))
                for seed in range(3):
                    final.append(specification(f'top{k}/random{seed}',sites=control_sites(selected,heads,self.args.seed+seed)))
            for split in ('heldout','transfer','unchanged_query'):
                self.evaluate(kind,patch,split,data[split],final)
        del patch,model,predictor
        gc.collect()
        torch.cuda.empty_cache()


def compare_and_report(out):
    result=json.loads((out/'results.json').read_text())
    for model in result['models'].values():
        overlaps={}
        ordered=model['selection']['ordered_sites']
        for spec in model['stages']['heldout']['specs']:
            if '/random' not in spec['name']:
                continue
            size=int(spec['name'].split('/')[0].removeprefix('top'))
            selected=set(map(tuple,ordered[:size]))
            control=set(map(tuple,spec['sites']))
            overlaps[spec['name']]=dict(selected_count=size,control_count=len(control),
                                       overlap_count=len(selected&control),identical=selected==control)
        model['selection']['random_control_overlap']=overlaps
    (out/'results.json').write_text(json.dumps(result,indent=2)+'\n')
    comparisons={}
    rng=np.random.default_rng(20260930)
    for split in ('heldout','transfer','unchanged_query'):
        dense=result['models']['dense']['stages'][split]
        dag=result['models']['dag']['stages'][split]
        n=dense['n_pairs']
        idx=rng.integers(n,size=(4000,n))
        common=np.asarray(dense['arms']['reference']['values']['recipient_vocab_accuracy']).reshape(n,2).all(1)
        common &= np.asarray(dag['arms']['reference']['values']['recipient_vocab_accuracy']).reshape(n,2).all(1)
        comparisons[split]={'common_both_directions_correct_pairs':int(common.sum()),'arms':{}}
        for name in ['all_qkv']+[f'top{k}' for k in (1,2,4,8,16)]:
            d=dense['arms'][name]['values'];g=dag['arms'][name]['values']
            arm={}
            for metric in ('donor_vocab_accuracy','donor_two_choice_accuracy','recipient_logp','intervention_l2','relative_intervention_l2'):
                if split=='unchanged_query' and metric=='recipient_logp':
                    difference=(np.asarray(g[metric])-np.asarray(dag['arms']['reference']['values'][metric]))-(np.asarray(d[metric])-np.asarray(dense['arms']['reference']['values'][metric]))
                else:
                    difference=np.asarray(g[metric])-np.asarray(d[metric])
                paired=difference.reshape(n,2).mean(1)
                arm[metric]=dict(dag_minus_dense=float(paired.mean()),paired_95ci=np.quantile(paired[idx].mean(1),[.025,.975]).tolist(),
                                 common_success_dag_minus_dense=float(paired[common].mean()) if common.any() else None)
            if split!='unchanged_query':
                fractions=[]
                for model in (dense,dag):
                    current=np.asarray(model['arms'][name]['values']['donor_margin'])
                    ref=np.asarray(model['arms']['reference']['values']['donor_margin'])
                    don=np.asarray(model['arms']['donor_baseline']['values']['donor_margin'])
                    effect,gap=(current-ref).reshape(n,2).mean(1),(don-ref).reshape(n,2).mean(1)
                    fractions.append((float(effect.mean()/gap.mean()),effect[idx].mean(1)/gap[idx].mean(1),
                                      float(effect[common].mean()/gap[common].mean()) if common.any() else None))
                arm['margin_recovery']=dict(dag_minus_dense=fractions[1][0]-fractions[0][0],
                                            paired_95ci=np.quantile(fractions[1][1]-fractions[0][1],[.025,.975]).tolist(),
                                            common_success_dag_minus_dense=fractions[1][2]-fractions[0][2] if common.any() else None)
            comparisons[split]['arms'][name]=arm
    (out/'comparison.json').write_text(json.dumps(comparisons,indent=2)+'\n')
    lines=['# Dense versus DAG: matched head-message interchange', '',
           'Both frozen 300M-scale models receive exactly the same counterfactual copy pairs and interventions in 64-dimensional head messages at three known historical value-token positions. Q/K are changed before the full-dimensional norm; V before attention. No query activation or final output is directly changed. These positions are supplied by construction (oracle locations), not discovered.', '',
           'The search budget is identical: 36 single-layer/stream scans, then 64 single-head scans in each model’s top four layer/stream groups. Top-1/2/4/8/16 sets are frozen before held-out evaluation. Selection is by discovery donor-margin increase. The sets are local intervention interfaces, not complete or minimal circuits. Native DAG source-edge counts are not compared here.', '',
           '32 discovery pairs, 64 held-out pairs at period 64, 64 distance-transfer pairs at period 128, and 64 unchanged-query controls; both interchange directions. The data are exactly those in the native-message pilot datasets.json. CI: 4,000 paired bootstrap resamples of base blocks, unadjusted, keeping the directions together. Architecture contrasts use the same resampled blocks.', '',
           'Three head-permutation controls per size preserve layer/stream counts and share the head permutation across streams within a layer. Selected/control sites can overlap. They are not norm or language-model-damage matched; actual intervention L2 is reported. The unchanged-query control measures another synthetic copy target, not general natural-text capability.', '']
    lines += ['Intervention L2 is the per-case Euclidean norm pooled across all edited coordinates; relative L2 divides by the norm of the current recipient coordinates being replaced, then averages across cases. Top-k counts head/stream identities, each replaced at three oracle positions (3k spatiotemporal edits). All-QKV is a broad positive control, not a theoretical upper bound on a selected subset.', '']
    for kind,model in result['models'].items():
        lines += [f"{kind}: {model['parameters']:,} total parameters. Selected layer/stream groups: {model['selection']['layer_streams']}. Top 8 sites: {model['selection']['ordered_sites'][:8]}. Unwrapped parity error {model['unwrapped_parity_max_logit_error']}; all identity errors {max(s['max_identity_logit_error'] for s in model['stages'].values())}.", '']
        overlap_text=', '.join(f"{name}: {entry['overlap_count']}/{entry['selected_count']}" for name,entry in model['selection']['random_control_overlap'].items())
        lines += [f'Actual selected/control overlap — {kind}: {overlap_text}.', '']
    for split in ('heldout','transfer'):
        lines += [f'## {split}', '',
                  f"Pairs for which both models are correct in both directions: {comparisons[split]['common_both_directions_correct_pairs']}/64. Full-sample metrics below; common-success contrasts are retained in comparison.json.", '',
                  '| Model | Arm | Margin recovery [95% CI] | Donor full-vocab accuracy | Donor two-choice accuracy | Intervention L2 | Relative L2 |',
                  '|---|---|---:|---:|---:|---:|---:|']
        for name in ['reference','donor_baseline','all_qkv']+[f'top{k}{suffix}' for k in (1,2,4,8,16) for suffix in ('','/random0','/random1','/random2')]:
            for kind in ('dense','dag'):
                s=result['models'][kind]['stages'][split]['arms'][name]['summary']
                e=s['margin_recovery'];lo,hi=e['paired_95ci']
                lines.append(f"| {kind} | {name} | {e['mean']:.3f} [{lo:.3f}, {hi:.3f}] | {100*s['donor_vocab_accuracy']['mean']:.2f}% | {100*s['donor_two_choice_accuracy']['mean']:.2f}% | {s['intervention_l2']['mean']:.2f} | {s['relative_intervention_l2']['mean']:.3f} |")
        lines += ['', '| Arm | DAG−dense recovery [95% CI] | DAG−dense donor full-vocab accuracy [95% CI] |', '|---|---:|---:|']
        for name,a in comparisons[split]['arms'].items():
            e=a['margin_recovery'];lo,hi=e['paired_95ci'];v=a['donor_vocab_accuracy'];vl,vh=v['paired_95ci']
            lines.append(f"| {name} | {e['dag_minus_dense']:+.3f} [{lo:+.3f}, {hi:+.3f}] | {100*v['dag_minus_dense']:+.2f} pp [{100*vl:+.2f}, {100*vh:+.2f}] |")
        lines+=['']
    lines+=['## Unchanged-query control', '', '| Model | Arm | Correct-answer logp change [95% CI] | Correct full-vocab accuracy |', '|---|---|---:|---:|']
    for name in ['reference','all_qkv']+[f'top{k}' for k in (1,2,4,8,16)]:
        for kind in ('dense','dag'):
            s=result['models'][kind]['stages']['unchanged_query']['arms'][name]['summary']
            e=s['recipient_logp'];lo,hi=e['paired_95ci']
            lines.append(f"| {kind} | {name} | {e['delta']:+.4f} [{lo:+.4f}, {hi:+.4f}] | {100*s['recipient_vocab_accuracy']['mean']:.2f}% |")
    lines+=['', 'All per-case full-vocabulary log probabilities, margins, two-choice metrics, output KL and intervention L2 are in results.json. The comparison uses one checkpoint per architecture, different total parameter counts, synthetic token-marginal repetition, and known intervention positions. No model was trained. Search considers only heads in the four discovery-selected layer/stream groups; it does not establish the globally smallest set.', '',
            'Run: `CUDA_VISIBLE_DEVICES=2 /scratch/yurenh2/venvs/dagformer-eval-20260917/bin/python scripts/circuit_common_message_patching.py`.', '']
    (out/'README.md').write_text('\n'.join(lines))


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--dag-model',type=Path,default=Path('checkpoints/pr_sync_20260917/300m-dagformer'))
    ap.add_argument('--dense-model',type=Path,default=Path('checkpoints/pr_sync_20260917/300m-baseline'))
    ap.add_argument('--data',type=Path,default=Path('experiments/results/circuit_interpretability_20260928/message_patching/datasets.json'))
    ap.add_argument('--out',type=Path,default=Path('experiments/results/circuit_interpretability_20260928/common_message_patching'))
    ap.add_argument('--batch-size',type=int,default=8)
    ap.add_argument('--seed',type=int,default=20260930)
    ap.add_argument('--smoke',action='store_true')
    ap.add_argument('--report-only',action='store_true')
    ap.add_argument('--kinds',nargs='+',choices=['dense','dag'],default=['dense','dag'])
    args=ap.parse_args()
    if args.report_only:
        compare_and_report(args.out)
        return
    assert args.batch_size%2==0
    torch.set_num_threads(4)
    torch.manual_seed(args.seed)
    data=load_data(args.data)
    if args.smoke:
        for d in data.values():
            for key in ('ids','a','b'):
                d[key]=d[key][:4]
    run=Run(args)
    for kind in args.kinds:
        run.run_model(kind,data)
    run.result['complete']=True
    run.save()
    if set(args.kinds)=={'dense','dag'}:
        compare_and_report(args.out)
    print('COMPLETE',args.out,'seconds',time.time()-run.start,flush=True)


if __name__=='__main__':
    main()
