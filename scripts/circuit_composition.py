"""Capability-gated two-hop composition and third-answer message interchange.

All learned-model weights are frozen. A third answer is the recipient's second
lookup applied to the donor's intermediate key, not either original answer.
Do not interpret failure of the capability gate as a failure of causal circuits.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import random
import subprocess
import time

import numpy as np
import torch
from transformers import AutoTokenizer

from interp_common import load_elh

COLORS = ['red', 'blue', 'green', 'black', 'white', 'yellow', 'brown']
TRAIN_ENTITIES = ['oak', 'elm', 'ash', 'fir', 'pine', 'birch', 'maple', 'cedar']
TRANSFER_ENTITIES = ['rose', 'lily', 'iris', 'mint', 'sage', 'fern', 'moss', 'reed']
STYLES = ['arrows', 'boxes', 'tables', 'equals']
TASKS = ['first_hop', 'second_hop', 'two_hop', 'supplied_middle']


def record(rng, entities):
    names = rng.sample(entities, 3)
    keys = rng.sample(['B', 'C', 'D'], 3)
    colors = rng.sample(COLORS, 3)
    first = list(zip(names, keys))
    second = list(zip(keys, colors))
    rng.shuffle(first)
    rng.shuffle(second)
    name = rng.choice(names)
    middle = dict(first)[name]
    return dict(first=first, second=second, name=name, middle=middle,
                answer=dict(second)[middle])


def render(rec, style, task, answered=False):
    first, second, name, mid = rec['first'], rec['second'], rec['name'], rec['middle']
    if style == 'arrows':
        facts = '\n'.join(f'{a} -> {b}' for a,b in first + second)
        question = {'first_hop':f'{name} ->', 'second_hop':f'{mid} ->',
                    'two_hop':f'{name} -> color:', 'supplied_middle':f'{name} -> {mid} ->'}[task]
    elif style == 'boxes':
        facts = ' '.join(f'The {a} box is {b}.' for a,b in first) + '\n'
        facts += ' '.join(f'Box {a} contains a {b} ball.' for a,b in second)
        question = {'first_hop':f'The {name} box is',
                    'second_hop':f'The ball in box {mid} is',
                    'two_hop':f'The ball in the {name} box is',
                    'supplied_middle':f'The {name} box is {mid}. The ball in box {mid} is'}[task]
    elif style == 'tables':
        facts = 'Name | Box\n' + '\n'.join(f'{a} | {b}' for a,b in first)
        facts += '\nBox | Color\n' + '\n'.join(f'{a} | {b}' for a,b in second)
        question = {'first_hop':f'Name: {name}\nBox:', 'second_hop':f'Box: {mid}\nColor:',
                    'two_hop':f'Name: {name}\nColor:',
                    'supplied_middle':f'Name: {name}\nBox: {mid}\nColor:'}[task]
    else:
        facts = ' '.join(f'{a} = {b};' for a,b in first + second)
        question = {'first_hop':f'{name} =', 'second_hop':f'{mid} =',
                    'two_hop':f'color({name}) =',
                    'supplied_middle':f'{name} = {mid} ='}[task]
    answer = rec['middle'] if task == 'first_hop' else rec['answer']
    return facts + '\n' + question + (' ' + answer if answered else '')


def make_items(tok, n, seed, entities, styles=STYLES, shots=(0,2,4)):
    items = []
    for style_i, style in enumerate(styles):
        for shot in shots:
            for task_i, task in enumerate(TASKS):
                # Same query records across formatting/fewshot variants. Demo
                # records are independent and are never reused as eval targets.
                rng = random.Random(seed)
                demo_rng = random.Random(seed + 900000 + style_i * 100 + task_i)
                demos = [record(demo_rng, TRAIN_ENTITIES) for _ in range(shot)]
                prefix = '\n\n'.join(render(x,style,task,True) for x in demos)
                for i in range(n):
                    rec = record(rng, entities)
                    prompt = (prefix+'\n\n' if prefix else '') + render(rec,style,task)
                    answer = rec['middle'] if task == 'first_hop' else rec['answer']
                    candidates = [x[1] for x in rec['first' if task == 'first_hop' else 'second']]
                    tokens = [tok.encode(' '+x,add_special_tokens=False) for x in candidates]
                    assert all(len(x)==1 for x in tokens), (candidates,tokens)
                    ids = tok.encode(prompt,add_special_tokens=False)
                    items.append(dict(index=i,seed=seed,style=style,shot=shot,task=task,
                                      prompt=prompt,ids=ids,record=rec,answer=answer,
                                      candidates=candidates,candidate_ids=[x[0] for x in tokens],
                                      target=tokens[candidates.index(answer)][0]))
    return items


def summarize(rows):
    groups=defaultdict(list)
    for row in rows:
        groups[f"{row['style']}/{row['shot']}/{row['task']}"].append(row)
    result={}
    for name,values in groups.items():
        n=len(values)
        p=np.mean([v['candidate_correct'] for v in values])
        z=1.96
        center=(p+z*z/(2*n))/(1+z*z/n)
        half=z*np.sqrt(p*(1-p)/n+z*z/(4*n*n))/(1+z*z/n)
        result[name]=dict(n=n,candidate_accuracy=float(p),candidate_accuracy_wilson95=[center-half,center+half],
                          vocab_accuracy=float(np.mean([v['vocab_correct'] for v in values])),
                          target_logp=float(np.mean([v['target_logp'] for v in values])),
                          margin=float(np.mean([v['margin'] for v in values])))
    return result


def write(path,obj):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(obj,indent=2)+'\n')


class Run:
    def __init__(self,args):
        self.args=args
        self.start=time.time()
        self.tok=AutoTokenizer.from_pretrained(args.tokenizer,local_files_only=True)
        self.out=args.out/args.name
        self.out.mkdir(parents=True,exist_ok=True)
        self.result=dict(complete=False,git_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
                         args={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
                         protocol=dict(discovery='24 records per format; 4 styles x 0/2/4 demonstrations; 4 query tasks.',
                            selection='Best discovery two-hop candidate accuracy; target log-probability breaks ties. This selects format only.',
                            gate='For selected format: heldout two-hop candidate accuracy >=0.70 and first-hop and second-hop >=0.80. Full-vocab accuracy also reported. Heldout not used to select patches.',
                            scaffold='supplied_middle is teacher-forced correct intermediate key and is diagnostic, never evidence of autonomous composition.',
                            transfer='Heldout uses new random tables; entity transfer uses an entirely disjoint name vocabulary.',
                            intervention='If capability permits, search intermediate layers at the final query token for a computed middle-key state. First-table explicit key replacement is only an input-content positive control, not evidence of an internally computed key.',
                            limitations='Fixed checkpoints; synthetic natural-language/table tasks; no pretraining seed replication.'), stages={})
        if not args.build_only:
            torch.set_num_threads(4)
            torch.cuda.set_per_process_memory_fraction(args.memory_fraction)
            device=torch.device('cuda')
            elh=load_elh()
            cfg=elh.load_config(str(args.config or args.model/'config.yaml'))
            checkpoint=args.checkpoint or args.model/'checkpoint.pt'
            if args.kind=='dag':
                self.model,self.predictor=elh.load_fourway(str(checkpoint),cfg,device)
                self.model.use_triton_kernel=False
                base=self.model.olmo
            else:
                self.model=elh.load_dense(str(checkpoint),cfg,device)
                self.model.config._attn_implementation='sdpa'
                self.predictor=None
                base=self.model
            self.cfg=cfg
            self.handle=base.lm_head.register_forward_pre_hook(lambda m,x:(x[0][:,-1:],))
            self.result['parameters']=sum(p.numel() for p in self.model.parameters())+(
                sum(p.numel() for p in self.predictor.parameters()) if self.predictor is not None else 0)

    def save(self):
        self.result['elapsed_seconds']=time.time()-self.start
        write(self.out/'results.json',self.result)

    @torch.inference_mode()
    def logits(self,ids):
        if self.args.kind=='dag':
            return self.model(ids,self.predictor(ids))[:,-1].float()
        return self.model(ids,use_cache=False).logits[:,-1].float()

    def evaluate(self,stage,items):
        print('START',self.args.name,stage,len(items),flush=True)
        buckets=defaultdict(list)
        for idx,item in enumerate(items):
            buckets[len(item['ids'])].append((idx,item))
        rows=[None]*len(items)
        count=0
        for length,bucket in sorted(buckets.items()):
            for start in range(0,len(bucket),self.args.batch_size):
                batch=bucket[start:start+self.args.batch_size]
                ids=torch.tensor([x['ids'] for _,x in batch],device='cuda')
                logits=self.logits(ids)
                lp=logits.log_softmax(-1)
                for j,(index,item) in enumerate(batch):
                    cand=item['candidate_ids']
                    scores=logits[j,cand]
                    chosen=min(token for token,score in zip(cand,scores.tolist()) if score==scores.max().item())
                    others=[x for x in cand if x!=item['target']]
                    rows[index]={k:v for k,v in item.items() if k!='ids'}
                    rows[index].update(candidate_correct=int(chosen==item['target']),
                        vocab_correct=int(logits[j].argmax()==item['target']),
                        target_logp=lp[j,item['target']].item(),
                        margin=(logits[j,item['target']]-logits[j,others].max()).item(),
                        predicted_candidate=self.tok.decode([chosen]),
                        predicted_vocab=self.tok.decode([logits[j].argmax().item()]),
                        candidate_logits=scores.cpu().tolist(),length=length)
                count+=len(batch)
                if count%128==0:
                    print(stage,count,len(items),'elapsed',time.time()-self.start,flush=True)
        summary=summarize(rows)
        self.result['stages'][stage]=dict(summary=summary,rows=rows)
        self.save()
        print('DONE',stage,json.dumps(summary),flush=True)
        return summary

    def natural_fourshot(self):
        self.result['protocol']['fixed_fourshot_diagnostic']='Additional fixed 4-shot natural boxes format, fresh independent seeds; specified after primary screening to ensure a failed zero-shot format does not obscure few-shot competence.'
        for stage,seed,entities in [('natural_fourshot',2026092906,TRAIN_ENTITIES),('natural_fourshot_transfer',2026092907,TRANSFER_ENTITIES)]:
            items=make_items(self.tok,self.args.heldout,seed,entities,['boxes'],[4])
            write(self.out/f'{stage}_inputs.json',items)
            self.evaluate(stage,items)

    def run(self):
        a=self.args
        if a.natural_fourshot_only:
            self.result=json.loads((self.out/'results.json').read_text())
            self.start-=self.result.get('elapsed_seconds',0)
            self.natural_fourshot()
            self.save()
            return
        discovery=make_items(self.tok,a.discovery,2026092901,TRAIN_ENTITIES)
        write(self.out/'discovery_inputs.json',discovery)
        if a.build_only:
            self.result['build_checks']=dict(n=len(discovery),max_tokens=max(len(x['ids']) for x in discovery),
                                             all_answers_single_token=True)
            self.save()
            print(self.result['build_checks'])
            return
        scores=self.evaluate('discovery',discovery)
        options=[(v['candidate_accuracy'],v['target_logp'],key) for key,v in scores.items() if key.endswith('/two_hop')]
        _,_,selected=max(options)
        style,shot,_=selected.split('/')
        shot=int(shot)
        self.result['selected_format']=dict(style=style,shots=shot,discovery_key=selected)
        heldout=make_items(self.tok,a.heldout,2026092902,TRAIN_ENTITIES,[style],[shot])
        transfer=make_items(self.tok,a.heldout,2026092903,TRANSFER_ENTITIES,[style],[shot])
        write(self.out/'heldout_inputs.json',heldout)
        write(self.out/'transfer_inputs.json',transfer)
        hs=self.evaluate('heldout',heldout)
        self.evaluate('entity_transfer',transfer)
        # A separately labelled natural-language validation was requested after
        # initial capability screening. Select shot count on discovery only;
        # use fresh tables and seeds, not the primary heldout set.
        natural_options=[(v['candidate_accuracy'],v['target_logp'],key) for key,v in scores.items() if key.startswith('boxes/') and key.endswith('/two_hop')]
        _,_,natural_selected=max(natural_options)
        natural_shot=int(natural_selected.split('/')[1])
        self.result['natural_validation_format']=dict(style='boxes',shots=natural_shot,discovery_key=natural_selected)
        self.result['protocol']['natural_validation']='Supplemental natural-language boxes check requested after initial screening; shot count selected only on discovery, fresh seed/table/name data; not a replacement for the primary heldout result.'
        for stage,seed,entities in [('natural_validation',2026092904,TRAIN_ENTITIES),('natural_entity_transfer',2026092905,TRANSFER_ENTITIES)]:
            natural=make_items(self.tok,a.heldout,seed,entities,['boxes'],[natural_shot])
            write(self.out/f'{stage}_inputs.json',natural)
            self.evaluate(stage,natural)
        self.natural_fourshot()
        prefix=f'{style}/{shot}/'
        passed=(hs[prefix+'two_hop']['candidate_accuracy']>=.70 and
                hs[prefix+'first_hop']['candidate_accuracy']>=.80 and
                hs[prefix+'second_hop']['candidate_accuracy']>=.80)
        self.result['capability_gate_passed']=passed
        self.result['causal_status']='pending' if passed else 'not_run_insufficient_whole_task_capability'
        self.result['complete']=not passed
        self.save()
        print('CAPABILITY_GATE',passed,flush=True)
        if passed:
            # Patching is a separate explicit stage after capability validation;
            # its ranking data must be newly drawn, never chosen heldout rows.
            print('READY_FOR_CAUSAL_STAGE',flush=True)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--model',type=Path,default=Path('checkpoints/pr_sync_20260917/300m-dagformer'))
    p.add_argument('--checkpoint',type=Path)
    p.add_argument('--config',type=Path)
    p.add_argument('--kind',choices=['dag','dense'],default='dag')
    p.add_argument('--name',default='dag300m')
    p.add_argument('--tokenizer',default='checkpoints/pr_sync_20260917/tokenizer')
    p.add_argument('--out',type=Path,default=Path('experiments/results/circuit_followup_20260929/composition'))
    p.add_argument('--discovery',type=int,default=24)
    p.add_argument('--heldout',type=int,default=96)
    p.add_argument('--batch-size',type=int,default=8)
    p.add_argument('--memory-fraction',type=float,default=.32)
    p.add_argument('--build-only',action='store_true')
    p.add_argument('--natural-fourshot-only',action='store_true')
    Run(p.parse_args()).run()


if __name__=='__main__':
    main()
