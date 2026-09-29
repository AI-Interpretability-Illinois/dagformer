"""Capability-gated transfer of rule-conditioned FourWay routing coefficients.

The backbone always receives recipient token IDs. Donors supply predictor and/or
local-correction coefficients only. Independent-content donors use a disjoint
symbol vocabulary, so applying the donor rule to the recipient table must give
a different token from the donor's own answer. No training is performed.
"""
from __future__ import annotations

import argparse
import importlib.metadata
import json
from pathlib import Path
import random
import subprocess
import time

import numpy as np
import torch
from transformers import AutoTokenizer

from interp_common import flatten_alpha, load_elh, load_eval_ids, unflatten_alpha
from interp_editing import layer_chunks


FAMILIES = ['arrows', 'function', 'chain', 'index', 'copy_query']
POOLS = [list('ABCDEFGHIJKLM'), list('NOPQRSTUVWXYZ')]


def content(seed, side, noise=2):
    rng = random.Random(seed)
    symbols = rng.sample(POOLS[side], 4 + 2 * noise)
    chain = symbols[:4]
    edges = list(zip(chain[:-1], chain[1:]))
    edges += [(symbols[i], symbols[i+1]) for i in range(4, len(symbols), 2)]
    rng.shuffle(edges)
    return dict(chain=chain, edges=edges, symbols=symbols)


def render(family, table, rule):
    chain, edges = table['chain'], table['edges']
    if family == 'arrows':
        body = 'Table:\n' + ''.join(f'{a} -> {b}\n' for a,b in edges) + f'Start: {chain[0]}\n'
        query = f'Steps: {rule}\nAnswer:'
    elif family == 'function':
        body = 'Function:\n' + ''.join(f'f( {a} ) = {b}\n' for a,b in edges) + f'Input: {chain[0]}\n'
        query = f'Applications: {rule}\nResult:'
    elif family == 'chain':
        body = 'Chain: ' + ' -> '.join(chain) + f'\nStart: {chain[0]}\n'
        query = f'Steps: {rule}\nAnswer:'
    elif family == 'index':
        body = 'List: ' + ' '.join(chain) + '\n'
        query = f'Index: {rule}\nValue:'
    else:
        raise ValueError(family)
    return body, query, chain[rule]


def demonstrations(tok, family, shots, seed):
    parts=[]
    for i in range(shots):
        body,query,answer=render(family,content(seed+i, i%2),i%3)
        parts.append(body+query+' '+answer+'\n\n')
    return tok.encode(''.join(parts),add_special_tokens=False)


def symbolic_group(tok, family, shots, seed, demo_seed, noise=2):
    prefix=demonstrations(tok,family,shots,demo_seed)
    group=[]
    for side in range(2):
        table=content(seed+side*100000,side,noise)
        items=[]
        targets=[tok.encode(' '+s,add_special_tokens=False) for s in table['chain'][:3]]
        assert all(len(t)==1 for t in targets)
        targets=[t[0] for t in targets]
        for rule in range(3):
            body,query,answer=render(family,table,rule)
            body_ids=tok.encode(body,add_special_tokens=False)
            query_ids=tok.encode(query,add_special_tokens=False)
            ids=prefix+body_ids+query_ids
            items.append(dict(ids=ids,target=targets[rule],targets=targets,rule=rule,
                              query_start=len(prefix)+len(body_ids),test_start=len(prefix),
                              answer=answer,side=side,body=body,query=query))
        assert len({len(x['ids']) for x in items})==1
        # Same-content rule conditions differ in exactly one selector token.
        for left,right in ((0,1),(0,2),(1,2)):
            assert sum(a!=b for a,b in zip(items[left]['ids'],items[right]['ids']))==1
        group.append(items)
    assert len(group[0][0]['ids'])==len(group[1][0]['ids'])
    assert not (set(group[0][0]['targets']) & set(group[1][0]['targets']))
    return group


def copy_group(corpus, seed, period=64):
    group=[]
    for side in range(2):
        flat=corpus.flatten()
        flat=flat[(flat%2)==side]
        g=torch.Generator().manual_seed(seed+side*100000)
        while True:
            block=flat[torch.randint(len(flat),(period,),generator=g)].tolist()
            targets=[block[i] for i in (15,31,47)]
            if len(set(targets))==3:
                break
        items=[]
        for rule,end in enumerate((15,31,47)):
            # An exact 15-token key from a different part of the repeated block.
            # Rule here means queried address, not a new algorithm.
            ids=block*3+block[end-15:end]
            items.append(dict(ids=ids,target=targets[rule],targets=targets,rule=rule,
                              query_start=3*period,test_start=0,side=side,
                              answer=str(targets[rule]),body=f'three repeated {period}-token blocks',
                              query=f'15-token key ending before block offset {end}'))
        group.append(items)
    assert not (set(group[0][0]['targets']) & set(group[1][0]['targets']))
    return group


def build_groups(tok, corpus, family, shots, n, seed, demo_seed, noise=2,period=64):
    return [copy_group(corpus,seed+i,period) if family=='copy_query' else
            symbolic_group(tok,family,shots,seed+i,demo_seed,noise) for i in range(n)]


def prepare(args):
    tok=AutoTokenizer.from_pretrained(str(args.tokenizer),local_files_only=True)
    corpus,_=load_eval_ids(args.token_cache)
    variants={}
    for family in FAMILIES:
        for shots in ([0] if family=='copy_query' else args.shots):
            key=f'{family}_{shots}shot'
            variants[key]=dict(family=family,shots=shots,
                               groups=build_groups(tok,corpus,family,shots,args.screen_groups,
                                                   args.seed,args.seed+50000))
    args.out.mkdir(parents=True,exist_ok=True)
    result=dict(seed=args.seed,tokenizer=str(args.tokenizer),variants=variants,
                protocol='Same table/chain with selector 0/1/2; two content sides use disjoint uppercase symbol pools. Copy fallback uses disjoint even/odd token vocabularies and changes the 15-token lookup key.')
    (args.out/'screen_data.json').write_text(json.dumps(result)+'\n')
    summary={k:dict(groups=len(v['groups']),length=len(v['groups'][0][0][0]['ids']),
                    sample=[x['body']+x['query'] for x in v['groups'][0][0]]) for k,v in variants.items()}
    (args.out/'data_preview.json').write_text(json.dumps(summary,indent=2)+'\n')
    print('Prepared',len(variants),'variants',summary.keys(),flush=True)
    return result


class Runner:
    def __init__(self,args):
        torch.cuda.set_per_process_memory_fraction(args.memory_fraction)
        self.args=args
        self.device=torch.device('cuda')
        elh=load_elh()
        self.cfg=elh.load_config(str(args.model/'config.yaml'))
        self.model,self.predictor=elh.load_fourway(str(args.model/'checkpoint.pt'),self.cfg,self.device)
        self.model.use_triton_kernel=False
        self.L,self.H=self.cfg['num_hidden_layers'],self.cfg['num_attention_heads']
        self.chunks=layer_chunks(self.L,self.H)
        self.lengths=None
        self.capture=None
        self.edit=None
        self.handles=[]
        # Preserve all transformer computation and project only each real final
        # token, avoiding padded or unused T*vocabulary logits.
        def lm_input(module,inputs):
            x=inputs[0]
            return (x[torch.arange(len(x),device=x.device),self.lengths-1,None],)
        self.handles.append(self.model.olmo.lm_head.register_forward_pre_hook(lm_input))
        for index,mlp in enumerate(self.model.correction_mlps):
            def hook(module,inputs,output,index=index):
                if self.capture is not None:
                    self.capture.append(output.detach().clone())
                if self.edit is not None and self.edit['channel'] in ('corr','both'):
                    lo,hi=self.chunks[index]
                    donor=self.edit['corr'][...,lo:hi].to(output.dtype)
                    if self.edit.get('mode')=='delta':
                        donor=output+donor
                    mask=self.edit['positions'][...,None]
                    return torch.where(mask,donor,output)
                return output
            self.handles.append(mlp.register_forward_hook(hook))

    def batch(self,items):
        lengths=torch.tensor([len(x['ids']) for x in items],device=self.device)
        ids=torch.zeros(len(items),int(lengths.max()),dtype=torch.long,device=self.device)
        for i,item in enumerate(items):
            ids[i,:len(item['ids'])]=torch.tensor(item['ids'],device=self.device)
        return ids,lengths

    @torch.inference_mode()
    def forward(self,items,capture=False,edit=None):
        ids,self.lengths=self.batch(items)
        self.capture=[] if capture else None
        self.edit=edit
        routing=self.predictor(ids)
        pred=flatten_alpha(routing)
        if edit is not None and edit['channel'] in ('pred','both'):
            donor=edit['pred'].to(pred.dtype)
            if edit.get('mode')=='delta':
                donor=pred+donor
            pred=torch.where(edit['positions'][...,None],donor,pred)
            routing=unflatten_alpha(pred,self.L,self.H)
        logits=self.model(ids,routing)[:,0].float()
        captured=dict(pred=pred.detach().clone(),corr=torch.cat(self.capture,-1)) if capture else None
        self.capture=self.edit=None
        return logits,captured

    def close(self):
        for handle in self.handles:
            handle.remove()


def baseline_scores(logits,items):
    lp=logits.log_softmax(-1)
    targets=torch.tensor([x['target'] for x in items],device=logits.device)
    candidates=torch.tensor([x['targets'] for x in items],device=logits.device)
    choices=lp.gather(1,candidates)
    target_lp=lp.gather(1,targets[:,None])[:,0]
    prediction=logits.argmax(-1)
    return [dict(logp=float(target_lp[i]),vocab_correct=bool(prediction[i]==targets[i]),
                 candidate_correct=int(choices[i].argmax())==item['rule'],
                 choices=choices[i].cpu().tolist(),predicted_token=int(prediction[i]),
                 rule=item['rule'],side=item['side']) for i,item in enumerate(items)]


def screen(args,runner,data):
    result=dict(complete=False,model=str(args.model),threshold=args.capability_threshold,variants={})
    for name,variant in data['variants'].items():
        items=[item for group in variant['groups'] for side in group for item in side]
        rows=[]
        for start in range(0,len(items),args.batch_size):
            batch=items[start:start+args.batch_size]
            logits,_=runner.forward(batch)
            rows.extend(baseline_scores(logits,batch))
        matrix=np.asarray([x['vocab_correct'] for x in rows]).reshape(-1,2,3)
        scores={str(rule):dict(vocab_accuracy=float(matrix[:,:,rule].mean()),
                               candidate_accuracy=float(np.mean([x['candidate_correct'] for x in rows if x['rule']==rule])),
                               logp=float(np.mean([x['logp'] for x in rows if x['rule']==rule]))) for rule in range(3)}
        pairs={f'{a}-{b}':dict(both_correct=float((matrix[:,:,a]&matrix[:,:,b]).mean()),
                              eligible=min(scores[str(a)]['vocab_accuracy'],scores[str(b)]['vocab_accuracy'])>=args.capability_threshold
                                       and float((matrix[:,:,a]&matrix[:,:,b]).mean())>=args.capability_threshold-.1)
               for a,b in ((0,1),(1,2),(0,2))}
        result['variants'][name]=dict(family=variant['family'],shots=variant['shots'],scores=scores,pairs=pairs,rows=rows)
        (args.out/'capability.json').write_text(json.dumps(result,indent=2)+'\n')
        print('SCREEN',name,'vocab',[round(scores[str(i)]['vocab_accuracy'],3) for i in range(3)],
              'eligible',[k for k,v in pairs.items() if v['eligible']],flush=True)
    choices=[]
    for name,variant in result['variants'].items():
        for pair,score in variant['pairs'].items():
            if score['eligible']:
                # Prefer genuine table conversion, then explicit chain/index,
                # then address selection in the existing copy mechanism.
                rank={'arrows':0,'function':0,'chain':1,'index':2,'copy_query':3}[variant['family']]
                choices.append((rank,-score['both_correct'],variant['shots'],name,pair))
    result['selection']=None
    if choices:
        _,_,_,name,pair=min(choices)
        result['selection']=dict(variant=name,rules=list(map(int,pair.split('-'))),
                                 family=result['variants'][name]['family'],shots=result['variants'][name]['shots'])
    result['complete']=True
    (args.out/'capability.json').write_text(json.dumps(result,indent=2)+'\n')
    print('CAPABILITY SELECTION',result['selection'],flush=True)
    return result


def intervention_cases(groups,rules,seed,random_groups):
    cases=[]
    rng=random.Random(seed)
    for group_index,group in enumerate(groups):
        for side in range(2):
            for rule in rules:
                desired=next(r for r in rules if r!=rule)
                own=group[side][rule]
                same=group[side][desired]
                cross=group[1-side][desired]
                wrong=group[1-side][rule]
                random_rule=rng.randrange(3)
                random_item=random_groups[group_index][1-side][random_rule]
                assert len({len(x['ids']) for x in (own,same,cross,wrong,random_item)})==1
                assert cross['target'] not in own['targets']
                cases.append(dict(group=group_index,side=side,rule=rule,desired_rule=desired,
                                  own=own,same=same,cross=cross,wrong=wrong,random=random_item,
                                  original=own['target'],desired=same['target'],donor_answer=cross['target'],
                                  random_rule=random_rule))
    return cases


def switch_measures(logits,cases,reference):
    lp=logits.log_softmax(-1)
    device=logits.device
    gather=lambda key:lp.gather(1,torch.tensor([c[key] for c in cases],device=device)[:,None])[:,0]
    original,desired,donor=gather('original'),gather('desired'),gather('donor_answer')
    argmax=logits.argmax(-1)
    target=lambda key:torch.tensor([c[key] for c in cases],device=device)
    candidates=torch.tensor([c['own']['targets'] for c in cases],device=device)
    desired_rule=torch.tensor([c['desired_rule'] for c in cases],device=device)
    ref_lp=reference.log_softmax(-1)
    values=dict(original_logp=original,desired_logp=desired,desired_margin=desired-original,
                desired_two_choice_probability=torch.sigmoid(desired-original),
                desired_two_choice_accuracy=(desired>original).float(),
                desired_candidate_accuracy=(lp.gather(1,candidates).argmax(-1)==desired_rule).float(),
                desired_vocab_accuracy=(argmax==target('desired')).float(),
                original_vocab_accuracy=(argmax==target('original')).float(),
                donor_answer_logp=donor,donor_vocab_accuracy=(argmax==target('donor_answer')).float(),
                recipient_distribution_kl=(ref_lp.exp()*(ref_lp-lp)).sum(-1))
    return {k:v.cpu().tolist() for k,v in values.items()}


def summarize_switch(values,reference,target_reference,eligible,n_groups):
    rng=np.random.default_rng(20260929)
    indices=rng.integers(n_groups,size=(4000,n_groups))
    result={}
    for metric,x in values.items():
        x=np.asarray(x);ref=np.asarray(reference[metric])
        pairs=(x-ref).reshape(n_groups,4).mean(1)
        item=dict(mean=float(x.mean()),delta=float(pairs.mean()),
                  paired_95ci=np.quantile(pairs[indices].mean(1),[.025,.975]).tolist())
        if eligible.any():
            item['both_rules_correct_mean']=float(x[eligible].mean())
            item['both_rules_correct_delta']=float((x-ref)[eligible].mean())
        result[metric]=item
    effect=(np.asarray(values['desired_margin'])-np.asarray(reference['desired_margin'])).reshape(n_groups,4).mean(1)
    gap=(np.asarray(target_reference['desired_margin'])-np.asarray(reference['desired_margin'])).reshape(n_groups,4).mean(1)
    if gap.mean()>.01:
        draws=effect[indices].mean(1)/gap[indices].mean(1)
        result['margin_recovery']=dict(mean=float(effect.mean()/gap.mean()),
                                       paired_95ci=np.quantile(draws,[.025,.975]).tolist())
    return result


def intervention_stage(args,runner,stage,cases,only_donors=None):
    specs=[]
    for channel in ('pred','corr','both'):
        specs.append(dict(name=f'identity/{channel}',channel=channel,scope='test',donor='own',mode='replace'))
        for scope in ('query','test'):
            for donor in ('same','cross','wrong','random','cross_delta','reverse_delta'):
                specs.append(dict(name=f'{scope}/{channel}/{donor}',channel=channel,scope=scope,donor=donor,
                                  mode='delta' if donor.endswith('_delta') else 'replace'))
    if only_donors is not None:
        specs=[s for s in specs if s['donor'] in set(only_donors)|{'own'}]
    values={s['name']:{} for s in specs}
    values.update(reference={},target_rule_reference={})
    identity_error=0.
    started=time.time()
    for start in range(0,len(cases),args.batch_size):
        batch=cases[start:start+args.batch_size]
        own=[c['own'] for c in batch]
        baseline,caches_own=runner.forward(own,capture=True)
        caches={'own':caches_own}
        target_logits=None
        needed={'same'}|{s['donor'] for s in specs}
        if any(x.endswith('_delta') for x in needed):
            needed|={'cross','wrong'}
        for donor in ('same','cross','wrong','random'):
            if donor not in needed:
                continue
            logits,cached=runner.forward([c[donor] for c in batch],capture=True)
            caches[donor]=cached
            if donor=='same':
                target_logits=logits
        if 'cross' in caches and 'wrong' in caches:
            caches['cross_delta']={k:caches['cross'][k]-caches['wrong'][k] for k in ('pred','corr')}
            caches['reverse_delta']={k:-v for k,v in caches['cross_delta'].items()}
        def record(name,logits):
            for key,row in switch_measures(logits,batch,baseline).items():
                values[name].setdefault(key,[]).extend(row)
        record('reference',baseline)
        record('target_rule_reference',target_logits)
        for spec in specs:
            donor=caches[spec['donor']]
            length=donor['pred'].shape[1]
            positions=torch.arange(length,device=runner.device)[None]
            beginnings=torch.tensor([c['own'][spec['scope']+'_start'] for c in batch],device=runner.device)[:,None]
            endings=torch.tensor([len(c['own']['ids']) for c in batch],device=runner.device)[:,None]
            mask=(positions>=beginnings)&(positions<endings)
            logits,_=runner.forward(own,edit=dict(channel=spec['channel'],mode=spec['mode'],positions=mask,**donor))
            if spec['donor']=='own':
                identity_error=max(identity_error,(logits-baseline).abs().max().item())
            record(spec['name'],logits)
        if start==0 or (start+len(batch))%32==0:
            print(f'INTERVENTION {stage}: {start+len(batch)}/{len(cases)}, seconds {time.time()-started:.1f}',flush=True)
    assert identity_error==0.,identity_error
    ref=values['reference'];target=values['target_rule_reference']
    eligible=np.asarray(ref['original_vocab_accuracy']).astype(bool)&np.asarray(target['desired_vocab_accuracy']).astype(bool)
    n_groups=len(cases)//4
    result=dict(complete=True,n_groups=n_groups,n_directed_cases=len(cases),identity_max_logit_error=identity_error,
                baseline_original_accuracy=float(np.mean(ref['original_vocab_accuracy'])),
                baseline_target_rule_accuracy=float(np.mean(target['desired_vocab_accuracy'])),
                both_rules_correct_fraction=float(eligible.mean()),specs=specs,arms={})
    for name,vals in values.items():
        result['arms'][name]=dict(values=vals,summary=summarize_switch(vals,ref,target,eligible,n_groups))
    # The key control holds donor content fixed and changes only its rule.
    contrasts={}
    rng=np.random.default_rng(20260929)
    idx=rng.integers(n_groups,size=(4000,n_groups))
    for scope in ('query','test'):
        for channel in ('pred','corr','both'):
            for left,right in (('cross','wrong'),('cross_delta','reverse_delta')):
                if f'{scope}/{channel}/{left}' not in values:
                    continue
                name=f'{scope}/{channel}/{left}_minus_{right}'
                contrast={}
                for metric in ('desired_margin','desired_vocab_accuracy','desired_two_choice_accuracy','donor_answer_logp'):
                    delta=(np.asarray(values[f'{scope}/{channel}/{left}'][metric])-
                           np.asarray(values[f'{scope}/{channel}/{right}'][metric])).reshape(n_groups,4).mean(1)
                    contrast[metric]=dict(delta=float(delta.mean()),paired_95ci=np.quantile(delta[idx].mean(1),[.025,.975]).tolist())
                contrasts[name]=contrast
    result['donor_rule_contrasts']=contrasts
    (args.out/f'{stage}.json').write_text(json.dumps(result,indent=2)+'\n')
    print('DONE',stage,'baseline',result['baseline_original_accuracy'],'both rules',result['both_rules_correct_fraction'],flush=True)
    return result


def run_interventions(args,runner,capability):
    selection=capability['selection']
    if selection is None:
        (args.out/'results.json').write_text(json.dumps(dict(complete=True,status='no_capable_task',capability='capability.json'),indent=2)+'\n')
        report(args.out)
        return
    tok=AutoTokenizer.from_pretrained(str(args.tokenizer),local_files_only=True)
    corpus,_=load_eval_ids(args.token_cache)
    family,shots,rules=selection['family'],selection['shots'],selection['rules']
    groups={split:build_groups(tok,corpus,family,shots,n,args.seed+offset,args.seed+50000,
                               noise=3 if split=='transfer' else 2,
                               period=128 if split=='transfer' else 64)
            for split,n,offset in [('discovery',args.discovery_groups,10000),('heldout',args.test_groups,20000),('transfer',args.test_groups,30000)]}
    # Chain/index transfer changes the prefix example tables, since extra edges
    # are not part of those display formats. Their new content is still held out.
    if family in ('chain','index'):
        groups['transfer']=build_groups(tok,corpus,family,shots,args.test_groups,args.seed+30000,args.seed+60000)
    cases={}
    for i,(split,g) in enumerate(groups.items()):
        random_groups=build_groups(tok,corpus,family,shots,len(g),args.seed+90000+i*10000,
                                   args.seed+60000 if split=='transfer' and family in ('chain','index') else args.seed+50000,
                                   noise=3 if split=='transfer' and family not in ('chain','index') else 2,
                                   period=128 if split=='transfer' else 64)
        cases[split]=intervention_cases(g,rules,args.seed+70000+i,random_groups)
    (args.out/'intervention_data.json').write_text(json.dumps(dict(selection=selection,cases=cases))+'\n')
    result=dict(complete=False,model=str(args.model),selection=selection,
                args={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
                git_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
                versions={k:importlib.metadata.version(k) for k in ('torch','transformers')},stages={},
                protocol=dict(
                    task='Map traversal / explicit chain / indexed list' if family!='copy_query' else 'Copy address selection: same repeated body, a different 15-token query key. This is not a change of algorithm.',
                    content='Backbone token IDs always belong to the recipient; no donor embeddings, source activations, head messages, or logits are injected. Downstream activations recompute normally after routing edits.',
                    coefficients='Predictor and/or raw local-correction coefficients, all layers/streams; direct replacement or recipient + cross-content donor rule difference.',
                    donors='same: recipient content, desired rule; cross: other symbol vocabulary/content, desired rule; wrong: exactly that donor content, original rule; random: separately generated table for each recipient group, with a random rule (no reuse of recipient-group contents). cross_delta subtracts same donor content/original-rule coefficients; reverse_delta negates it.',
                    random_donor_generation_version=2,
                    donor_score_label='donor_answer_logp / donor_vocab_accuracy always refer to the cross-content desired-rule donor token, not the actual source token for every other control arm.',
                    alignment='Donor/recipient lengths exactly match; selector-only pairs differ in one token for symbolic tasks. Copy-query keys differ in 15 tokens. Query scope begins at the rule selector or lookup-key prefix; test scope includes the final table/body but excludes demonstrations.',
                    selection='Task/template/rule pair chosen only by independent capability screen. All channels/scopes/donor controls are reported; discovery split does not tune a sparse mask.',
                    uncertainty='4000 unadjusted cluster bootstrap samples of independently generated paired tables (four directions per table: two symbol vocabularies times both rule directions).',
                    transfer='New content and extra distractor table edges for map/function, new demonstration tables for chain/index, or period128 for copy fallback.',
                    limitation='A negative result is interpretable only where both original and desired recipient rules are mastered; fallback copy changes an address rather than a general algorithm. No natural-text capability-cost matching or architecture superiority claim.'))
    for split in ('discovery','heldout','transfer'):
        stage=intervention_stage(args,runner,split,cases[split])
        result['stages'][split]=dict(file=f'{split}.json',baseline_original_accuracy=stage['baseline_original_accuracy'],
                                    baseline_target_rule_accuracy=stage['baseline_target_rule_accuracy'],
                                    both_rules_correct_fraction=stage['both_rules_correct_fraction'],
                                    identity_max_logit_error=stage['identity_max_logit_error'])
        (args.out/'results.json').write_text(json.dumps(result,indent=2)+'\n')
    result['complete']=True
    (args.out/'results.json').write_text(json.dumps(result,indent=2)+'\n')
    report(args.out)


def repair_random(args,runner):
    """Replace only the exploratory neighboring-group random-donor arms.

    The first 300M execution predated independent random groups. Keep its random
    arms in an audit file, rerun only those six arms plus identity/baselines,
    and leave every main same/cross/wrong/delta measurement unchanged.
    """
    data=json.loads((args.out/'intervention_data.json').read_text())
    result=json.loads((args.out/'results.json').read_text())
    selection=data['selection'];family,shots=selection['family'],selection['shots']
    tok=AutoTokenizer.from_pretrained(str(args.tokenizer),local_files_only=True)
    corpus,_=load_eval_ids(args.token_cache)
    audit=dict(old_random_protocol='Random donor reused the next recipient group, inducing adjacent-cluster dependence; old random-arm intervals are exploratory.',stages={})
    for i,split in enumerate(('discovery','heldout','transfer')):
        cases=data['cases'][split]
        n=len(cases)//4
        random_groups=build_groups(tok,corpus,family,shots,n,args.seed+90000+i*10000,
                                   args.seed+60000 if split=='transfer' and family in ('chain','index') else args.seed+50000,
                                   noise=3 if split=='transfer' and family not in ('chain','index') else 2,
                                   period=128 if split=='transfer' else 64)
        for case in cases:
            case['random']=random_groups[case['group']][1-case['side']][case['random_rule']]
            assert len(case['own']['ids'])==len(case['random']['ids'])
        old=json.loads((args.out/f'{split}.json').read_text())
        audit['stages'][split]={name:a for name,a in old['arms'].items() if name.endswith('/random')}
        fresh=intervention_stage(args,runner,split+'_random_repair',cases,only_donors=['random'])
        for metric,values in old['arms']['reference']['values'].items():
            assert np.max(np.abs(np.asarray(values)-np.asarray(fresh['arms']['reference']['values'][metric])))==0.
        for name,value in fresh['arms'].items():
            if name.endswith('/random'):
                old['arms'][name]=value
        old['random_donor_generation_version']=2
        (args.out/f'{split}.json').write_text(json.dumps(old,indent=2)+'\n')
    (args.out/'random_repair_audit.json').write_text(json.dumps(audit,indent=2)+'\n')
    (args.out/'intervention_data.json').write_text(json.dumps(data)+'\n')
    result['protocol']['random_donor_generation_version']=2
    result['protocol']['donors']='same/cross/wrong/delta unchanged; random donor uses a separately generated independent content group for each recipient group, never another recipient group.'
    result['protocol']['donor_score_label']='donor_answer_logp / donor_vocab_accuracy refer to the cross-content desired-rule donor token for every arm.'
    result['random_repair']='Only six random-donor arms rerun with independent groups; all original main-arm scores preserved; baseline equality exact. Original exploratory random arms retained in random_repair_audit.json.'
    (args.out/'results.json').write_text(json.dumps(result,indent=2)+'\n')
    report(args.out)


def report(out):
    capability=json.loads((out/'capability.json').read_text())
    lines=['# Capability-gated routing rule interchange', '',
           'Model: '+capability['model']+'. This is a frozen-model inference experiment; no adapter or backbone training.', '',
           '## Capability screen', '',
           'The selected task and rule pair must have at least 80% full-vocabulary accuracy under both rules and 70% joint success on an independent screen. Candidate-only accuracy does not bypass this gate. All screened templates are retained.', '',
           '| Template | Rule 0 accuracy | Rule 1 accuracy | Rule 2 accuracy | Eligible pairs |', '|---|---:|---:|---:|---|']
    for name,v in capability['variants'].items():
        acc=[100*v['scores'][str(r)]['vocab_accuracy'] for r in range(3)]
        eligible=[k for k,p in v['pairs'].items() if p['eligible']]
        lines.append(f'| {name} | {acc[0]:.2f}% | {acc[1]:.2f}% | {acc[2]:.2f}% | {eligible} |')
    lines+=['', 'Selected: '+str(capability['selection'])+'.', '']
    if capability['selection'] and capability['selection']['family']=='copy_query':
        lines+=['None of the screened transformation templates met the two-rule capability gate. The fallback changes which repeated key is queried. It tests address-conditioned routing transfer, not switching a learned multi-step algorithm; failure here does not establish that routing cannot implement algorithm control.', '']
    path=out/'results.json'
    if path.exists():
        result=json.loads(path.read_text())
        lines += ['## Desired versus wrong donor rule', '',
                  'The central contrast holds donor content fixed and changes only its rule. An increase in desired-versus-original margin by itself can reflect reduced confidence in the original answer; the wrong-rule and reversed-difference controls distinguish that from directed operation transfer.', '',
                  '| Split | Query/both: correct minus wrong donor margin [95% CI] | Query/both: forward minus reversed rule-difference margin [95% CI] | Desired-answer accuracy, correct / wrong |',
                  '|---|---:|---:|---:|']
        for split,record in result.get('stages',{}).items():
            stage=json.loads((out/record['file']).read_text())
            c=stage['donor_rule_contrasts']['query/both/cross_minus_wrong']['desired_margin']
            d=stage['donor_rule_contrasts']['query/both/cross_delta_minus_reverse_delta']['desired_margin']
            cl,ch=c['paired_95ci'];dl,dh=d['paired_95ci']
            ca=stage['arms']['query/both/cross']['summary']['desired_vocab_accuracy']['mean']
            wa=stage['arms']['query/both/wrong']['summary']['desired_vocab_accuracy']['mean']
            lines.append(f"| {split} | {c['delta']:+.3f} [{cl:+.3f}, {ch:+.3f}] | {d['delta']:+.3f} [{dl:+.3f}, {dh:+.3f}] | {100*ca:.2f}% / {100*wa:.2f}% |")
        lines += ['', 'The experiment uses direct donor values or a donor rule difference at scale 1; it does not search amplified gains or train a new routing controller.', '']
        for split,record in result.get('stages',{}).items():
            stage=json.loads((out/record['file']).read_text())
            lines += [f'## {split}', '',
                      f"Original-rule full-vocabulary accuracy {100*stage['baseline_original_accuracy']:.2f}%; desired-rule reference {100*stage['baseline_target_rule_accuracy']:.2f}%; both rules correct {100*stage['both_rules_correct_fraction']:.2f}%. Identity max logit difference {stage['identity_max_logit_error']}.", '',
                      '| Arm | Desired-answer full-vocab accuracy | Desired−original margin change [95% CI] | Margin recovery | Cross-content desired-rule donor token accuracy |',
                      '|---|---:|---:|---:|---:|']
            for name,a in stage['arms'].items():
                if name.startswith('identity/'):
                    continue
                s=a['summary'];e=s['desired_margin'];lo,hi=e['paired_95ci']
                rec=s.get('margin_recovery',{}).get('mean',float('nan'))
                lines.append(f"| {name} | {100*s['desired_vocab_accuracy']['mean']:.2f}% | {e['delta']:+.3f} [{lo:+.3f}, {hi:+.3f}] | {rec:.4f} | {100*s['donor_vocab_accuracy']['mean']:.2f}% |")
            lines+=['']
    lines+=['The backbone always receives recipient tokens. Only predictor/local-correction coefficients are replaced or offset; downstream recipient computation then proceeds normally. Independent-content donors have a disjoint answer vocabulary, and the desired recipient answer differs from the donor answer. Correct-rule versus wrong-rule donor contrasts hold donor content fixed. They are saved in each stage JSON, alongside all case-level scores.', '',
            'The donor-token column always tracks the cross-content desired-rule donor token, including in other control arms. It is not the actual donor answer for every wrong/random/same arm. Random donors use separately generated content groups; each bootstrap group includes its own independent random donor. For the repaired initial 300M execution, old neighboring-group random arms are retained only as exploratory audit records.', '',
            'Copy-query uses disjoint even/odd token-ID vocabularies on the two content sides. This is a special synthetic out-of-distribution control, not a natural semantic rule switch.', '',
            'The 300M and 1B checkpoints differ in training corpus and budget. Their comparison extends the capability check and is not a controlled estimate of a size-only effect.', '',
            'Query scope starts at the explicit selector or copy key; test scope also includes the final table/body but excludes demonstrations. No donor final logits or head/source activations are injected. Full route replacement is a broad intervention, not a sparse circuit. CIs resample complete independent table pairs, keeping their four directed cases together; fixed checkpoints and unadjusted intervals.', '']
    if path.exists() and 'args' in result:
        a=result['args']
        lines += ['Reproduce: `CUDA_VISIBLE_DEVICES=3 /scratch/yurenh2/venvs/dagformer-eval-20260917/bin/python scripts/circuit_rule_switch.py --phase run '
                  f"--model {a['model']} --out {out} --batch-size {a['batch_size']} --memory-fraction {a['memory_fraction']} --seed {a['seed']}`.", '']
    (out/'README.md').write_text('\n'.join(lines))


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--model',type=Path,default=Path('checkpoints/pr_sync_20260917/300m-dagformer'))
    ap.add_argument('--tokenizer',type=Path,default=Path('checkpoints/pr_sync_20260917/tokenizer'))
    ap.add_argument('--token-cache',default='checkpoints/pr_sync_20260917/eval_corpora/wikitext_train.pt')
    ap.add_argument('--out',type=Path,default=Path('experiments/results/circuit_followup_20260929/rule_switch'))
    ap.add_argument('--phase',choices=['prepare','screen','run','report','repair-random'],default='prepare')
    ap.add_argument('--shots',type=int,nargs='+',default=[0,3,6])
    ap.add_argument('--screen-groups',type=int,default=8)
    ap.add_argument('--discovery-groups',type=int,default=16)
    ap.add_argument('--test-groups',type=int,default=32)
    ap.add_argument('--batch-size',type=int,default=2)
    ap.add_argument('--memory-fraction',type=float,default=.32)
    ap.add_argument('--capability-threshold',type=float,default=.8)
    ap.add_argument('--seed',type=int,default=20260929)
    args=ap.parse_args()
    if args.phase=='report':
        report(args.out)
        return
    torch.set_num_threads(2)
    torch.manual_seed(args.seed)
    path=args.out/'screen_data.json'
    data=json.loads(path.read_text()) if path.exists() else prepare(args)
    if args.phase=='prepare':
        return
    runner=Runner(args)
    try:
        if args.phase=='repair-random':
            repair_random(args,runner)
        elif args.phase=='screen':
            screen(args,runner,data)
            report(args.out)
        else:
            capability_path=args.out/'capability.json'
            capability=json.loads(capability_path.read_text()) if capability_path.exists() else screen(args,runner,data)
            assert capability['complete'] and capability['model']==str(args.model)
            run_interventions(args,runner,capability)
    finally:
        runner.close()


if __name__=='__main__':
    main()
