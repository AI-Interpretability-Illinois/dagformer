"""Constructive composition by transplanting computed first-hop head outputs.

This is separate from explaining an implicit two-hop capability. The recipient
gets an explicit original middle key; the donor must compute a different key.
No logits, embeddings, model weights, or recipient table entries are changed.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from contextlib import contextmanager
from copy import deepcopy
import json
from pathlib import Path
import random
import subprocess
import time

import numpy as np
import torch
from transformers import AutoTokenizer

from circuit_composition import COLORS, TRAIN_ENTITIES, TRANSFER_ENTITIES, record, render, write
from interp_common import load_elh


def make_cases(tok,n,seed,entities,variant='base'):
    rng=random.Random(seed)
    demo_rng=random.Random(seed+900000)
    demos=[record(demo_rng,TRAIN_ENTITIES) for _ in range(4)]
    def prompt(rec,task):
        return '\n\n'.join([render(x,'equals',task,True) for x in demos]+[render(rec,'equals',task)])
    cases=[]
    for i in range(n):
        rec=record(rng,entities)
        own=rec['middle']
        target_key=rng.choice([k for _,k in rec['first'] if k!=own])
        spare=next(k for _,k in rec['first'] if k not in (own,target_key))
        original_name=rec['name']
        donor=deepcopy(rec)
        donor['first']=[(name,target_key if key==own else own if key==target_key else key)
                        for name,key in donor['first']]
        available=[c for c in COLORS if c not in dict(rec['second']).values()]
        donor_colors=rng.sample(available,3)
        donor['second']=[(key,color) for (key,_),color in zip(rec['second'],donor_colors)]
        donor['middle']=target_key
        donor['answer']=dict(donor['second'])[target_key]
        if variant=='donor_color':
            donor['second']=[(key,donor_colors[(j+1)%3]) for j,(key,_) in enumerate(donor['second'])]
        elif variant=='recipient_table':
            colors=[color for _,color in rec['second']]
            rec['second']=[(key,colors[(j+1)%3]) for j,(key,_) in enumerate(rec['second'])]
        elif variant=='donor_query':
            donor['name']=next(name for name,key in donor['first'] if key==spare)
            target_key=spare
            donor['middle']=target_key
        donor['answer']=dict(donor['second'])[target_key]
        rec['answer']=dict(rec['second'])[own]
        third=dict(rec['second'])[target_key]
        cf=deepcopy(rec)
        cf['first']=[(name,target_key if key==own else own if key==target_key else key) for name,key in cf['first']]
        cf['middle']=target_key
        cf['answer']=third
        recipient_task='two_hop' if variant=='implicit_query' else 'supplied_middle'
        prompts=dict(recipient=prompt(rec,recipient_task),donor=prompt(donor,'first_hop'),
                     input_counterfactual=prompt(cf,recipient_task))
        tokens={key:tok.encode(value,add_special_tokens=False) for key,value in prompts.items()}
        values=dict(original=rec['answer'],third=third,donor_color=donor['answer'],donor_key=target_key,
                    spare=dict(rec['second'])[next(k for _,k in rec['first'] if k not in (own,target_key))])
        value_ids={key:tok.encode(' '+value,add_special_tokens=False) for key,value in values.items()}
        assert all(len(x)==1 for x in value_ids.values())
        assert len({values[x] for x in ('original','third','donor_color')})==3
        assert dict(donor['first'])[donor['name']]==target_key
        assert dict(rec['second'])[target_key]==third
        cases.append(dict(index=i,seed=seed,variant=variant,prompts=prompts,ids=tokens,
                          recipient_record=rec,donor_record=donor,values=values,
                          targets={key:value[0] for key,value in value_ids.items()}))
    return cases


class Patch:
    def __init__(self,model,predictor,L,H,D):
        self.model,self.predictor,self.L,self.H,self.D=model,predictor,L,H,D
        self.spec=self.donor=self.capture=None
        self.hits=set()
    def hook(self,layer):
        def inner(module,inputs):
            x=inputs[0]
            assert x.ndim==3 and x.shape[-1]==self.D
            assert layer not in self.hits
            self.hits.add(layer)
            old=x[:,-1].reshape(len(x),self.H,self.D//self.H)
            if self.capture is not None:self.capture[layer]=old.detach().clone()
            if self.spec is None:return None
            heads=[h for l,h in self.spec if l==layer]
            if not heads:return None
            new=old.clone()
            new[:,heads]=self.donor[layer][:,heads]
            self.edit_sq+=(new.float()-old.float()).square().sum((1,2))
            self.old_sq+=old[:,heads].float().square().sum((1,2))
            out=x.clone();out[:,-1]=new.reshape(len(x),self.D)
            return (out,)
        return inner
    @contextmanager
    def installed(self):
        handles=[block.self_attn.o_proj.register_forward_pre_hook(self.hook(l))
                 for l,block in enumerate(self.model.olmo.model.layers)]
        handles.append(self.model.olmo.lm_head.register_forward_pre_hook(lambda m,x:(x[0][:,-1:],)))
        try:yield
        finally:
            for h in handles:h.remove()
    @torch.inference_mode()
    def forward(self,ids,spec=None,donor=None,capture=False):
        self.spec,self.donor=spec,donor
        self.capture={} if capture else None
        self.hits=set();self.edit_sq=torch.zeros(len(ids),device=ids.device);self.old_sq=torch.zeros_like(self.edit_sq)
        logits=self.model(ids,self.predictor(ids))[:,-1].float()
        assert len(self.hits)==self.L
        cache=self.capture
        norms=dict(edit_l2=self.edit_sq.sqrt().cpu().tolist(),
                   relative_edit_l2=(self.edit_sq/self.old_sq.clamp_min(1e-20)).sqrt().cpu().tolist())
        self.capture=self.spec=self.donor=None
        return logits,cache,norms


def metrics(logits,targets):
    lp=logits.log_softmax(-1)
    pred=logits.argmax(-1)
    rows={}
    for key,target in targets.items():
        rows[key+'_logp']=lp.gather(1,target[:,None])[:,0].cpu().tolist()
        rows[key+'_vocab_accuracy']=(pred==target).float().cpu().tolist()
    color_lp=lp.gather(1,torch.stack([targets[k] for k in ('original','third','spare')],1))
    mass=color_lp.logsumexp(1)
    rows['recipient_color_logmass']=mass.cpu().tolist()
    rows['third_conditional_color_logp']=(color_lp[:,1]-mass).cpu().tolist()
    rows['third_vs_spare_margin']=(color_lp[:,1]-color_lp[:,2]).cpu().tolist()
    rows['third_margin']=(logits.gather(1,targets['third'][:,None])[:,0]-logits.gather(1,targets['original'][:,None])[:,0]).cpu().tolist()
    return rows


def summary(values,reference,cf,first_correct,both_correct):
    n=len(values['third_margin']);rng=np.random.default_rng(2026092919)
    indices=rng.integers(n,size=(4000,n))
    output={}
    for key,vals in values.items():
        x=np.array(vals,dtype=float)
        entry=dict(mean=float(x.mean()),bootstrap95ci=np.quantile(x[indices].mean(1),[.025,.975]).tolist())
        if key in reference:
            delta=x-np.array(reference[key]);entry['delta']=float(delta.mean());entry['delta_paired95ci']=np.quantile(delta[indices].mean(1),[.025,.975]).tolist()
        for label,mask in [('first_hop_correct',first_correct),('donor_and_original_recipient_correct',both_correct)]:
            if sum(mask):entry[label+'_mean']=float(x[mask].mean())
        output[key]=entry
    gap=np.array(cf['third_margin'])-np.array(reference['third_margin'])
    effect=np.array(values['third_margin'])-np.array(reference['third_margin'])
    if gap.mean()>.05:
        output['margin_recovery_vs_input_counterfactual']=dict(mean=float(effect.mean()/gap.mean()),bootstrap95ci=np.quantile(effect[indices].mean(1)/gap[indices].mean(1),[.025,.975]).tolist())
    return output


class Run:
    def __init__(self,args):
        self.args=args;self.start=time.time()
        self.tok=AutoTokenizer.from_pretrained(args.tokenizer,local_files_only=True)
        torch.set_num_threads(4);torch.cuda.set_per_process_memory_fraction(.9)
        elh=load_elh();cfg=elh.load_config(str(args.model/'config.yaml'))
        self.model,self.predictor=elh.load_fourway(str(args.model/'checkpoint.pt'),cfg,torch.device('cuda'))
        self.model.use_triton_kernel=False
        self.L,self.H,self.D=cfg['num_hidden_layers'],cfg['num_attention_heads'],cfg['hidden_size']
        self.patch=Patch(self.model,self.predictor,self.L,self.H,self.D)
        self.result=dict(complete=False,args={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
                         git_commit_at_start=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
                         protocol=dict(kind='Constructive single-operation splicing; not recovery of an existing implicit two-hop circuit.',
                             intervention='Replace selected attention-head outputs before o_proj at the final query token. Donor first-hop prompt asks for a computed key; recipient supplied-middle prompt already states its original key.',
                             selection=f'Discovery32, scan all-head blocks in layers2..13, then all heads in top4 layers; rank by all-case {args.objective} increase. Fixed top1/2/4/8/16 evaluated on untouched splits.',
                             target='Recipient second table evaluated at donor-computed first-hop key; distinct from recipient original color and donor table color.',
                             controls='Same-input identity; actual input-key counterfactual; three layer-count-matched random top4 head sets; donor final color swap; recipient table permutation; donor query-key change; new entities; implicit-query diagnostic.',
                             limits='Local message sets, all other computation active, not minimal/full circuits; intervention norms not matched; fixed checkpoint and exploratory unadjusted intervals.'),stages={})
    def save(self):
        self.result['elapsed_seconds']=time.time()-self.start
        write(self.args.out/'results.json',self.result)
    def evaluate(self,name,cases,arms):
        print('START',name,len(cases),'arms',len(arms),flush=True)
        vals={name:{} for name in ['recipient','donor_first_hop','input_counterfactual',*arms]}
        first_correct=[False]*len(cases);both_correct=[False]*len(cases)
        buckets=defaultdict(list)
        for index,case in enumerate(cases):buckets[tuple(len(case['ids'][k]) for k in ('recipient','donor','input_counterfactual'))].append((index,case))
        identity_error=0.;count=0
        for bucket in buckets.values():
            for start in range(0,len(bucket),self.args.batch_size):
                batch=bucket[start:start+self.args.batch_size]
                tensors={key:torch.tensor([case['ids'][key] for _,case in batch],device='cuda') for key in ('recipient','donor','input_counterfactual')}
                targets={key:torch.tensor([case['targets'][key] for _,case in batch],device='cuda') for key in ('original','third','donor_color','donor_key','spare')}
                own,owncache,_=self.patch.forward(tensors['recipient'],capture=True)
                donor,cache,_=self.patch.forward(tensors['donor'],capture=True)
                cf,_,_=self.patch.forward(tensors['input_counterfactual'])
                for j,(idx,_) in enumerate(batch):
                    first_correct[idx]=bool(donor[j].argmax()==targets['donor_key'][j])
                    both_correct[idx]=first_correct[idx] and bool(own[j].argmax()==targets['original'][j])
                def add(arm,logits,norm=None):
                    row=metrics(logits,targets)
                    if norm is not None:row.update(norm)
                    for key,value in row.items():
                        vals[arm].setdefault(key,[None]*len(cases))
                        for (idx,_),v in zip(batch,value):vals[arm][key][idx]=v
                add('recipient',own);add('donor_first_hop',donor);add('input_counterfactual',cf)
                for arm,sites in arms.items():
                    changed,_,norm=self.patch.forward(tensors['recipient'],sites,owncache if arm=='identity' else cache)
                    if arm=='identity':identity_error=max(identity_error,(changed-own).abs().max().item())
                    add(arm,changed,norm)
                count+=len(batch)
        assert identity_error==0.,identity_error
        ref,cf=vals['recipient'],vals['input_counterfactual']
        stage=dict(n=len(cases),first_hop_correct_count=sum(first_correct),donor_and_original_recipient_correct_count=sum(both_correct),
                   first_hop_correct=first_correct,donor_and_original_recipient_correct=both_correct,max_identity_logit_error=identity_error,
                   sites=arms,arms={key:dict(values=value,summary=summary(value,ref,cf,first_correct,both_correct)) for key,value in vals.items()})
        self.result['stages'][name]=stage;self.save()
        print('DONE',name,'own',np.mean(ref['original_vocab_accuracy']),'firsthop',sum(first_correct)/len(cases),
              'inputCF',np.mean(cf['third_vocab_accuracy']),'elapsed',time.time()-self.start,flush=True)
        return stage
    def run(self):
        a=self.args
        if a.diagnostics_only:
            original=json.loads((a.out/'results.json').read_text())
            saved_data=json.loads((a.out/'datasets.json').read_text())
            a.out=a.out/'semantic_diagnostics';a.out.mkdir(parents=True,exist_ok=True)
            self.result['fixed_selection_from_parent']=original['selection']
            self.result['protocol']['selection']='No selection: all original heldout head sets reused unchanged.'
            self.result['original_numeric_replay_checks']={}
            cases={name:make_cases(self.tok,64,original['args'].get('eval_seed',2026092911),TRAIN_ENTITIES,
                                'base' if name=='heldout' else name)
                   for name in ['heldout','donor_query','recipient_table']}
            for name,rows in cases.items():
                assert [r['ids'] for r in rows]==[r['ids'] for r in saved_data[name]]
            write(a.out/'datasets.json',cases)
            with self.patch.installed():
                for name,rows in cases.items():
                    stage=self.evaluate(name,rows,original['stages'][name]['sites'])
                    error=0.
                    for arm,record in stage['arms'].items():
                        for key in ['third_logp','third_vocab_accuracy','donor_key_vocab_accuracy']:
                            old=original['stages'][name]['arms'][arm]['values'][key]
                            error=max(error,float(np.max(np.abs(np.asarray(old)-np.asarray(record['values'][key])))))
                    self.result['original_numeric_replay_checks'][name]=error
                    assert error==0.,(name,error)
            self.result['complete']=True;self.save();return
        a.out.mkdir(parents=True,exist_ok=True)
        discovery=make_cases(self.tok,32,a.discovery_seed,TRAIN_ENTITIES)
        stages={'heldout':make_cases(self.tok,64,a.eval_seed,TRAIN_ENTITIES),
                'entity_transfer':make_cases(self.tok,64,a.eval_seed+1,TRANSFER_ENTITIES)}
        for variant in ('donor_color','recipient_table','donor_query','implicit_query'):
            stages[variant]=make_cases(self.tok,64,a.eval_seed,TRAIN_ENTITIES,variant)
        write(a.out/'datasets.json',dict(discovery=discovery,**stages))
        ids=torch.tensor([discovery[0]['ids']['recipient']],device='cuda')
        with torch.inference_mode():unwrapped=self.model(ids,self.predictor(ids))[:,-1].float()
        # Slicing a BF16 vocabulary GEMM can choose a different kernel. Check
        # head-hook parity against the exact same last-token projection path,
        # and separately disclose the full-projection rounding difference.
        projection=self.model.olmo.lm_head.register_forward_pre_hook(lambda m,x:(x[0][:,-1:],))
        with torch.inference_mode():optimized=self.model(ids,self.predictor(ids))[:,-1].float()
        projection.remove()
        self.result['projection_diagnostic_one_case']=dict(max_logit_error=(unwrapped-optimized).abs().max().item(),
            argmax_agreement=bool(unwrapped.argmax(-1)==optimized.argmax(-1)),
            explanation='Full-token versus last-token BF16 lm_head GEMM; every experimental arm uses the same last-token path.')
        with self.patch.installed():
            wrapped,_,_=self.patch.forward(ids)
            self.result['unwrapped_max_logit_error']=(optimized-wrapped).abs().max().item()
            assert self.result['unwrapped_max_logit_error']==0.
            layers=list(range(2,self.L-2))
            coarse={f'layer{l}':[(l,h) for h in range(self.H)] for l in layers}
            coarse['identity']=[(l,h) for l in layers for h in range(self.H)]
            result=self.evaluate('layer_discovery',discovery,coarse)
            selected=sorted(layers,key=lambda l:result['arms'][f'layer{l}']['summary'][a.objective]['delta'],reverse=True)[:4]
            fine={f'head{l}_{h}':[(l,h)] for l in selected for h in range(self.H)}
            result=self.evaluate('head_discovery',discovery,fine)
            sites=sorted([(l,h) for l in selected for h in range(self.H)],key=lambda s:result['arms'][f'head{s[0]}_{s[1]}']['summary'][a.objective]['delta'],reverse=True)
            self.result['selection']=dict(layers=selected,ordered_sites=sites,
                discovery_gains=[result['arms'][f'head{l}_{h}']['summary'][a.objective]['delta'] for l,h in sites])
            final={f'top{k}':sites[:k] for k in [1,2,4,8,16]}
            final['identity']=sites[:16]
            rng=random.Random(2026092913)
            for i in range(3):
                sample=[]
                for layer in selected:
                    n=sum(l==layer for l,h in sites[:4])
                    sample.extend((layer,h) for h in rng.sample(range(self.H),n))
                final[f'random4_{i}']=sample
            self.result['random_control_overlaps']={name:len(set(sites[:4])&set(map(tuple,selected))) for name,selected in final.items() if name.startswith('random')}
            self.save()
            for name,cases in stages.items():self.evaluate(name,cases,final)
        self.result['complete']=True;self.save();print('COMPLETE',time.time()-self.start,flush=True)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--model',type=Path,default=Path('/scratch/yurenh2/circuit-followup-checkpoints/1b-dagformer'))
    p.add_argument('--tokenizer',default='checkpoints/pr_sync_20260917/tokenizer')
    p.add_argument('--out',type=Path,default=Path('experiments/results/circuit_followup_20260929/composition_constructive'))
    p.add_argument('--batch-size',type=int,default=4)
    p.add_argument('--diagnostics-only',action='store_true')
    p.add_argument('--objective',choices=['third_margin','third_logp'],default='third_margin')
    p.add_argument('--discovery-seed',type=int,default=2026092910)
    p.add_argument('--eval-seed',type=int,default=2026092911)
    Run(p.parse_args()).run()

if __name__=='__main__':main()
