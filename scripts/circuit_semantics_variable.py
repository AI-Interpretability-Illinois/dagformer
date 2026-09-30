"""Fit and test a one-dimensional number interchange on a discovered native edge.

Frozen model, full forward, dynamic correction. Fits use only the first 16
discovery blocks; direction checkpoint and readout ridge strength use the last
8 discovery blocks. All confirmation sets remain outside fitting/selection.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import random

import numpy as np
import torch
import torch.nn.functional as F

from circuit_semantics import (OUT, STATES, CONTRASTS, VERBS, MessageExperiment,
    read_gz, save_gz, selected_edges, arm, run_set, summarize, block_stat, write_json)


def feature(cache, rows, edge, site):
    z, alpha = cache[(edge['layer'], edge['stream'])]
    b = torch.arange(len(rows), device=z.device)
    pos = torch.tensor([r['positions'][site] for r in rows], device=z.device)
    if edge['stream'] == 'r':
        return z[edge['src'], b, pos].float(), alpha[b, pos, edge['src']].float()
    return z[edge['src'], b, pos, edge['head']].float(), alpha[b, pos, edge['head'], edge['src']].float()


def cached_blocks(exp, rows, state):
    batches = []
    with torch.no_grad():
        for i in range(0, len(rows), 8):
            batch = rows[i:i+8]
            logits = exp.forward(batch, state, capture=True)
            batches.append(dict(rows=batch, cache=exp.cache, logp=logits.log_softmax(-1)))
    return batches


def loss_on(exp, batch, direction, edge, site, kind):
    exp.cache = batch['cache']
    spec = arm('fit', [edge], site=site, contrast=kind, direction=direction)
    logits = exp.forward(batch['rows'], 'original', spec=spec)
    perm = torch.arange(8, device='cuda') ^ CONTRASTS[kind]
    target = batch['logp'][perm] if kind=='subject' else batch['logp']
    # Distribution matching uses the natural number counterfactual as teacher.
    return F.kl_div(logits.log_softmax(-1), target, log_target=True, reduction='batchmean')


def fit_direction(exp, batches, edge, site, out, steps=100):
    train, validation = batches[:16], batches[16:]
    xs = torch.cat([feature(b['cache'], b['rows'], edge, site)[0] for b in train])
    sn = torch.tensor([r['subject_number'] for b in train for r in b['rows']], device='cuda')
    difference = xs[sn==1].mean(0)-xs[sn==0].mean(0)
    mean_direction = difference/difference.norm().clamp_min(1e-8)
    direction = torch.nn.Parameter(mean_direction.clone())
    optimizer = torch.optim.Adam([direction], lr=.005)
    rng = random.Random(202609290801)
    result = dict(site=site, method='Rank-one orthogonal content interchange; weighted full-vocabulary KL to natural counterfactual and unchanged recipient distributions.',
                  weights=dict(subject=.5, distractor=.25, lexical=.25), steps=steps, lr=.005,
                  train_blocks=16, validation_blocks=8, history=[], validation=[])
    best_loss = float('inf')
    best = None
    for step in range(steps+1):
        if step:
            batch = rng.choice(train)
            optimizer.zero_grad(set_to_none=True)
            total = 0
            losses = {}
            for kind, weight in [('subject', .5), ('distractor', .25), ('lexical', .25)]:
                loss = loss_on(exp, batch, direction, edge, site, kind)
                assert torch.isfinite(loss)
                (weight*loss).backward()
                losses[kind] = float(loss.detach())
                total += weight*losses[kind]
            grad = float(direction.grad.norm())
            assert np.isfinite(grad) and grad>0
            optimizer.step()
            with torch.no_grad():
                direction.div_(direction.norm())
            if step==1 or step%25==0:
                result['history'].append(dict(step=step, loss=total, grad_norm=grad, **losses))
        if step in [0,25,50,100]:
            vals = defaultdict(list)
            with torch.no_grad():
                for batch in validation:
                    for kind in CONTRASTS:
                        vals[kind].append(float(loss_on(exp,batch,direction,edge,site,kind)))
            summary = {k:float(np.mean(v)) for k,v in vals.items()}
            objective = .5*summary['subject']+.25*summary['distractor']+.25*summary['lexical']
            result['validation'].append(dict(step=step, objective=objective, **summary))
            if objective < best_loss:
                best_loss = objective
                best = direction.detach().clone()
                result['selected_step'] = step
            print('DIRECTION',site,step,summary,flush=True)
    generator = torch.Generator(device='cuda').manual_seed(202609290802)
    directions = dict(mean=mean_direction.detach(), learned=best)
    shuffled = sn[torch.randperm(len(sn), generator=generator, device='cuda')]
    shuffled_direction = xs[shuffled==1].mean(0)-xs[shuffled==0].mean(0)
    directions['shuffled_mean'] = shuffled_direction/shuffled_direction.norm().clamp_min(1e-8)
    for i in range(3):
        d = torch.randn(len(best), generator=generator, device='cuda')
        directions[f'random{i}'] = d/d.norm()
    result['directions'] = {k:v.cpu().tolist() for k,v in directions.items()}
    write_json(out/f'direction_{site}.json', result)
    return directions


def get_features(exp, rows, state, edge):
    result = {site:dict(x=[], alpha=[]) for site in ['subject','distractor','last']}
    with torch.no_grad():
        for start in range(0, len(rows), 16):
            batch = rows[start:start+16]
            exp.forward(batch, state, capture=True)
            for site, record in result.items():
                x, alpha = feature(exp.cache, batch, edge, site)
                record['x'].append(x.cpu())
                record['alpha'].append(alpha.cpu())
    return {site:{k:torch.cat(v) for k,v in record.items()} for site,record in result.items()}


def fit_readout(x, labels):
    x, y = x.to('cuda').double(), torch.tensor(labels,device='cuda',dtype=torch.float64)*2-1
    train_x, validation_x = x[:128], x[128:]
    train_y, validation_y = y[:128], y[128:]
    center, ycenter = train_x.mean(0), train_y.mean()
    xc = train_x-center
    gram = xc@xc.T
    scale = float(gram.diag().mean())
    best = None
    for strength in [.001,.01,.1,1.]:
        weight = xc.T@torch.linalg.solve(gram+strength*scale*torch.eye(len(xc),device='cuda'), train_y-ycenter)
        pred = (validation_x-center)@weight+ycenter
        error = float(((pred-validation_y)**2).mean())
        if best is None or error < best['validation_mse']:
            best = dict(weight=weight.cpu(), center=center.cpu(), intercept=float(ycenter),
                        ridge=strength, validation_mse=error)
    return best


def readout_score(readout, features, labels):
    pred = (features.double()-readout['center'])@readout['weight']+readout['intercept']
    label = torch.tensor(labels)*2-1
    return block_stat(((pred>=0)==(label>0)).numpy())


def direction_arms(edge, directions):
    arms = []
    for site, ds in directions.items():
        for name, direction in ds.items():
            for contrast in CONTRASTS:
                arms.append(arm(f'{site}/{name}/{contrast}', [edge], site=site,
                                contrast=contrast, direction=direction))
    return arms


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', type=Path, default=OUT)
    p.add_argument('--adapter', type=Path, default=Path('/scratch/yurenh2/interp_followup_20260929/runs/full_s20260930/weights/predictor_best.pt'))
    args = p.parse_args()
    data = read_gz(args.out/'data.json.gz')
    edge = json.loads((args.out/'selection.json').read_text())['selected']['edge']['edges'][0]
    out = args.out/'variable'
    out.mkdir(exist_ok=True)
    write_json(out/'protocol.json',dict(edge=edge, fit_state='original',
        fit='First 16 discovery blocks; remaining 8 select checkpoint by counterfactual/isolation KL.',
        sites=['subject','last'],
        projection='Swap only the donor-recipient difference projected onto one unit vector in the native source content. Keep recipient effective coefficient and normal downstream computation.',
        labels='Subject number for fit; distractor and lexical interchanges are isolation conditions.',
        controls='Untrained mean-difference, shuffled-label mean-difference and 3 random rank-one directions. Shuffled mean is not a budget-matched learned-intervention control.',
        transfer='The exact same direction learned on P0 is used on all three predictor states and all test sets.',
        readouts='Separate ridge readouts of subject and distractor number at subject, distractor and last positions; fit each state plus frozen-original cross-state readout.',
        limitation='This is a task-specific one-dimensional distribution-matching experiment, not a reproduction of the full DAS or ConceptDAS benchmark.'))
    exp = MessageExperiment(args.adapter, selected_edges())
    with exp.installed():
        directions = {}
        caches = None
        for site in ['subject','last']:
            path = out/f'direction_{site}.json'
            if path.exists():
                record = json.loads(path.read_text())
                directions[site] = {k:torch.tensor(v,device='cuda') for k,v in record['directions'].items()}
            else:
                if caches is None:
                    caches = cached_blocks(exp,data['splits']['discovery'],'original')
                directions[site] = fit_direction(exp,caches,edge,site,out)
        del caches
        exp.cache = None
        torch.cuda.empty_cache()
        arms = direction_arms(edge,directions)
        readouts = {}
        feature_scores = {}
        # Fitting and all selections finish before test readout or interchange.
        for state in STATES:
            rows = data['splits']['discovery']
            features = get_features(exp,rows,state,edge)
            readouts[state] = {}
            for site, record in features.items():
                readouts[state][site] = {variable:fit_readout(record['x'],[r[variable+'_number'] for r in rows])
                                         for variable in ['subject','distractor']}
        for split in ['heldout','transfer','role']:
            path = out/f'{split}.json.gz'
            result = read_gz(path) if path.exists() else {}
            feature_scores[split] = {}
            rows = data['splits'][split]
            for state in STATES:
                if state not in result:
                    result[state] = run_set(exp,rows,state,arms,data['verb_ids'])
                    result[state]['summary'] = summarize(rows,result[state],arms)
                    save_gz(path,result)
                    write_json(out/f'{split}_summary.json',{s:r['summary'] for s,r in result.items()})
                features = get_features(exp,rows,state,edge)
                fs = {}
                for site, record in features.items():
                    fs[site] = dict(effective_alpha=block_stat(record['alpha'].numpy()),
                        own_readout={}, original_readout={}, learned_direction={})
                    for variable in ['subject','distractor']:
                        labels = [r[variable+'_number'] for r in rows]
                        fs[site]['own_readout'][variable] = readout_score(readouts[state][site][variable],record['x'],labels)
                        fs[site]['original_readout'][variable] = readout_score(readouts['original'][site][variable],record['x'],labels)
                    if site in directions:
                        projection = record['x']@directions[site]['learned'].detach().cpu()
                        sn = np.array([r['subject_number'] for r in rows])
                        fs[site]['learned_direction'] = dict(
                            plural_minus_singular=float(projection[sn==1].mean()-projection[sn==0].mean()),
                            projection_sd=float(projection.std()))
                feature_scores[split][state] = fs
            write_json(out/'readout_summary.json',feature_scores)
        readable = {state:{site:{variable:{k:(v.tolist() if isinstance(v,torch.Tensor) else v)
                    for k,v in record.items()} for variable,record in variables.items()}
                    for site,variables in sites.items()} for state,sites in readouts.items()}
        save_gz(out/'readouts.json.gz',readable)
    print('COMPLETE variable',flush=True)


if __name__=='__main__':
    main()
