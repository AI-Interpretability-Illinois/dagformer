"""Shared inputs and faithful model loading for the ordered interpretability audit.

These experiments use the original checkpoints and their complete forward pass.
The generated SVA/IOI sentences are controlled task instances, not a reproduction
of an external benchmark's published scores.
"""
from __future__ import annotations

import copy
import json
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn.utils import parametrize
from transformers import AutoTokenizer

from interp_common import alpha_layout, flatten_alpha, load_elh, load_eval_ids, unflatten_alpha


ROOT = Path(__file__).resolve().parent.parent
RESULT_ROOT = ROOT / 'experiments/results/interp_audit_20260929'
TOKENIZER = ROOT / 'checkpoints/pr_sync_20260917/tokenizer'
MODEL_ROOT = ROOT / 'checkpoints/pr_sync_20260917'
NOUNS = [('pilot','pilots'),('doctor','doctors'),('teacher','teachers'),('student','students'),
         ('author','authors'),('driver','drivers'),('farmer','farmers'),('worker','workers'),
         ('actor','actors'),('singer','singers'),('nurse','nurses'),('lawyer','lawyers'),
         ('artist','artists'),('player','players'),('officer','officers'),('manager','managers'),
         ('chef','chefs'),('dancer','dancers'),('painter','painters'),('poet','poets'),
         ('engineer','engineers'),('scientist','scientists'),('writer','writers'),('sailor','sailors')]
NAMES = ['Alice','Bob','Mary','John','David','Sarah','James','Anna','Peter','Laura',
         'Michael','Emma','Daniel','Julia','Robert','Linda','Thomas','Helen','George','Nancy',
         'Paul','Susan','Mark','Lisa','Henry','Grace','Simon','Rachel','Adam','Emily']
SVA_TRAIN = ['The {s} near the {d}', 'The {s} behind the {d}', 'The {s} beside the {d}',
             'The {s} that the {d} can see', 'The {s} that the {d} will visit',
             'The {s} in front of the {d}']
SVA_TRANSFER = ['The {s} next to the {d}', 'The {s} whom the {d} will meet',
                'The {s} that the {d} may recognize']
IOI_TRAIN = [
    'When {a} and {b} went to the {place}, {actor} gave a {obj} to',
    'After {a} and {b} arrived at the {place}, {actor} handed a {obj} to',
    '{a} and {b} were at the {place}. {actor} passed the {obj} to',
    '{a} and {b} met at the {place}. {actor} offered a {obj} to',
    'At the {place}, {a} and {b} talked. Then {actor} sent a {obj} to',
    'Yesterday {a} and {b} visited the {place}, and {actor} brought a {obj} to']
IOI_TRANSFER = [
    '{a} and {b} spent the afternoon at the {place}. Later, {actor} lent the {obj} to',
    'Before {a} and {b} left the {place}, {actor} delivered a {obj} to',
    'While {a} and {b} waited at the {place}, {actor} showed the {obj} to']


def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(obj, indent=2, allow_nan=False) + '\n')
    temp.replace(path)


def paired_stats(values, groups=None, seed=2026093001):
    arr = np.asarray(values, dtype=float)
    if groups is not None:
        grouped = defaultdict(list)
        for g, v in zip(groups, arr):
            grouped[g].append(v)
        arr = np.asarray([np.mean(v) for v in grouped.values()])
    rng = np.random.default_rng(seed)
    means = arr[rng.integers(len(arr), size=(2000, len(arr)))].mean(1)
    return dict(mean=float(arr.mean()), ci95=np.quantile(means, [.025, .975]).tolist(),
                n_groups=len(arr))


def make_tasks(tok, n_groups, seed, transfer=False):
    rng = random.Random(seed)
    nouns = NOUNS[16:] if transfer else NOUNS[:16]
    names = NAMES[20:] if transfer else NAMES[:20]
    names = [s for s in names if len(tok.encode(' '+s, add_special_tokens=False)) == 1]
    templates_sva = SVA_TRANSFER if transfer else SVA_TRAIN
    templates_ioi = IOI_TRANSFER if transfer else IOI_TRAIN
    out = []
    for task in ['sva', 'ioi']:
        seen = set()
        for group in range(n_groups):
            while True:
                if task == 'sva':
                    subject, distractor = rng.sample(nouns, 2)
                    dn = rng.randrange(2)
                    template = rng.choice(templates_sva)
                    answers = rng.choice([('is','are'), ('has','have'), ('does','do')])
                    desc = (tuple(subject), tuple(distractor), dn, template)
                    if desc in seen: continue
                    rows = []
                    for number in [0, 1]:
                        prompt = template.format(s=subject[number], d=distractor[dn])
                        rows.append(dict(prompt=prompt, answer=answers[number], foil=answers[1-number],
                                         subject=subject[number], distractor=distractor[dn], number=number))
                else:
                    a, b = rng.sample(names, 2)
                    place = rng.choice(['park','store','office','school','market','station','museum','library'])
                    obj = rng.choice(['book','letter','gift','bag','ticket','bottle','note','box'])
                    template = rng.choice(templates_ioi)
                    desc = (a,b,place,obj,template)
                    if desc in seen: continue
                    rows = []
                    for actor, recipient in [(a,b),(b,a)]:
                        rows.append(dict(prompt=template.format(a=a,b=b,actor=actor,place=place,obj=obj),
                                         answer=recipient, foil=actor, actor=actor, recipient=recipient))
                for row in rows:
                    row['ids'] = tok.encode(row['prompt'], add_special_tokens=False)
                    targets = [tok.encode(' '+row[k], add_special_tokens=False) for k in ['answer','foil']]
                    if not all(len(x)==1 for x in targets): break
                    row.update(target=targets[0][0], foil_id=targets[1][0], task=task,
                               group=f'{seed}/{task}/{group}', template=template)
                else:
                    # Equal lengths let paired interventions use identical positions.
                    if len(rows[0]['ids']) == len(rows[1]['ids']):
                        seen.add(desc)
                        out.extend(rows)
                        break
    return out


def natural_items(tok, split, n, seed, length=48, partition=None):
    # Legacy `doc` fields below index packed cache chunks, not original articles.
    ids, labels = load_eval_ids(str(MODEL_ROOT/f'eval_corpora/wikitext_{split}.pt'))
    rng = random.Random(seed)
    out = []
    for i in range(n):
        choices = list(range(len(ids))) if partition is None else list(range(partition,len(ids),2))
        doc = rng.choice(choices)
        start = rng.randrange(ids.shape[1]-length)
        window = ids[doc,start:start+length].tolist()
        target = int(labels[doc,start+length-1])
        if target < 0: continue
        out.append(dict(ids=window, target=target, foil_id=0 if target != 0 else 1,
                        prompt=tok.decode(window), answer=tok.decode([target]), foil='',
                        task='retain', group=f'{split}/doc{doc}', doc=doc, start=start))
    return out


def prepare_data(out=RESULT_ROOT):
    path = Path(out)/'datasets.json'
    if path.exists(): return json.loads(path.read_text())
    tok = AutoTokenizer.from_pretrained(str(TOKENIZER), local_files_only=True)
    data = {
        'train': make_tasks(tok, 384, 2026093001),
        'validation': make_tasks(tok, 48, 2026093002),
        'discovery': make_tasks(tok, 64, 2026093003),
        'heldout': make_tasks(tok, 128, 2026093004),
        'transfer': make_tasks(tok, 96, 2026093005, True),
    }
    # Disjoint prompt pairs across training, model selection, mask discovery, and test.
    previous = set()
    for split in ['train','validation','discovery','heldout','transfer']:
        bad = {x['group'] for x in data[split] if x['prompt'] in previous}
        data[split] = [x for x in data[split] if x['group'] not in bad]
        previous.update(x['prompt'] for x in data[split])
    data['train'] += natural_items(tok,'train',4096,2026093010)
    data['validation'] += natural_items(tok,'validation',128,2026093011,partition=0)
    data['discovery'] += natural_items(tok,'validation',128,2026093012,partition=1)
    data['heldout'] += natural_items(tok,'test',256,2026093013,partition=0)
    data['transfer'] += natural_items(tok,'test',128,2026093014,64,partition=1)
    write_json(path, data)
    return data


class LoRAWeight(nn.Module):
    """LoRA effective weight; supports the DAG forward's direct .weight reads."""
    def __init__(self, weight, rank=8):
        super().__init__()
        self.a = nn.Parameter(torch.randn(rank,weight.shape[1],device=weight.device)*.02)
        self.b = nn.Parameter(torch.zeros(weight.shape[0],rank,device=weight.device))
        self.register_buffer('gate', torch.ones(rank,device=weight.device))
        self.scale = 2.

    def forward(self, weight):
        delta = (self.b * self.gate[None,:]) @ self.a
        return (weight.float() + self.scale*delta).to(weight.dtype)


class AuditModel:
    def __init__(self, kind='dag', model_path=None, memory_fraction=.65):
        torch.set_num_threads(4)
        torch.backends.mha.set_fastpath_enabled(False)
        torch.cuda.set_per_process_memory_fraction(memory_fraction)
        self.kind = kind
        model_path = Path(model_path or MODEL_ROOT/f'300m-{ "dagformer" if kind=="dag" else "baseline"}')
        self.model_path = model_path
        elh = load_elh()
        self.cfg = elh.load_config(str(model_path/'config.yaml'))
        if kind == 'dag':
            self.model,self.predictor = elh.load_fourway(str(model_path/'checkpoint.pt'),self.cfg,'cuda')
            self.model.use_triton_kernel = False
            self.base = self.model.olmo
        else:
            self.model = elh.load_dense(str(model_path/'checkpoint.pt'),self.cfg,'cuda')
            self.model.config._attn_implementation = 'sdpa'
            self.predictor = None
            self.base = self.model
        self.layout = alpha_layout(self.cfg['num_hidden_layers'],self.cfg['num_attention_heads'])
        self.baseline_predictor = None
        self.loras = []
        self.last_positions = None
        self.output_hook = self.base.lm_head.register_forward_pre_hook(self._select_positions)

    def _select_positions(self, module, inputs):
        h = inputs[0]
        if self.last_positions is None: return None
        return (h[torch.arange(len(h),device=h.device),self.last_positions][:,None,:],)

    def batch(self, items):
        lengths = [len(x['ids']) for x in items]
        ids = torch.zeros(len(items),max(lengths),dtype=torch.long,device='cuda')
        for i,x in enumerate(items): ids[i,:len(x['ids'])] = torch.tensor(x['ids'],device='cuda')
        self.last_positions = torch.tensor(lengths,device='cuda')-1
        return ids

    def forward(self, items, alpha_gate=None, mode='adapted', raw_ids=None):
        ids = self.batch(items) if raw_ids is None else raw_ids
        with parametrize.cached():
            if self.kind == 'dag':
                if mode == 'base' and self.baseline_predictor is not None:
                    routing = self.baseline_predictor(ids)
                elif alpha_gate is not None:
                    with torch.no_grad():
                        p0 = flatten_alpha(self.baseline_predictor(ids))
                        p1 = flatten_alpha(self.predictor(ids))
                    flat = p0 + alpha_gate.view(1,1,-1)*(p1-p0)
                    routing = unflatten_alpha(flat,self.cfg['num_hidden_layers'],self.cfg['num_attention_heads'])
                else:
                    routing = self.predictor(ids)
                return self.model(ids,routing)[:,0].float()
            return self.model(ids,use_cache=False).logits[:,0].float()

    def add_lora(self, rank=8):
        for layer_i, layer in enumerate(self.base.model.layers):
            for part, names in [(layer.self_attn,['q_proj','k_proj','v_proj','o_proj']),
                                (layer.mlp,['gate_proj','up_proj','down_proj'])]:
                for name in names:
                    module = getattr(part,name)
                    lora = LoRAWeight(module.weight,rank)
                    parametrize.register_parametrization(module,'weight',lora)
                    self.loras.append((f'L{layer_i}/{name}',lora))
        return [p for _,lora in self.loras for p in lora.parameters()]

    def set_lora_gate(self, gate):
        offset = 0
        for _,lora in self.loras:
            size = lora.a.shape[0]
            lora.gate = gate[offset:offset+size]
            offset += size
        assert offset == len(gate)

    def copy_baseline_predictor(self):
        self.baseline_predictor = copy.deepcopy(self.predictor).eval()
        self.baseline_predictor.requires_grad_(False)


def metrics(logits, items):
    targets = torch.tensor([x['target'] for x in items],device=logits.device)
    foils = torch.tensor([x['foil_id'] for x in items],device=logits.device)
    rows = torch.arange(len(items),device=logits.device)
    nll = -logits.log_softmax(-1)[rows,targets]
    margin = logits[rows,targets]-logits[rows,foils]
    return nll, margin


@torch.no_grad()
def evaluate(runner, items, batch_size=16, alpha_gate=None, mode='adapted'):
    values = []
    for start in range(0,len(items),batch_size):
        batch = items[start:start+batch_size]
        logits = runner.forward(batch,alpha_gate,mode)
        nll,margin = metrics(logits,batch)
        for i,x in enumerate(batch):
            values.append(dict(task=x['task'],group=x['group'],nll=nll[i].item(),margin=margin[i].item(),
                               candidate_correct=float(margin[i]>0),
                               vocab_correct=float(logits[i].argmax()==x['target']),
                               predicted_token=int(logits[i].argmax())))
    summary = {}
    for task in sorted({x['task'] for x in items}):
        rows = [x for x in values if x['task']==task]
        summary[task] = {k:paired_stats([x[k] for x in rows],[x['group'] for x in rows])
                         for k in ['nll','margin','candidate_correct','vocab_correct']}
    return dict(summary=summary,values=values)
