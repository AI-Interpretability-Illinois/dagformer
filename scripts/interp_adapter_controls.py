"""Confirm adapter localization against controls matched by layer and stream.

For predictor masks, also match quartiles of the discovery coefficient-change
RMS. For LoRA, match projection modules (rank terms have no intrinsic semantics).
"""
import argparse
import copy
import json
from collections import defaultdict
import numpy as np
import torch
from interp_audit_common import AuditModel, RESULT_ROOT, prepare_data, write_json
from interp_adapter_audit import prepare, load_adapter, evaluate_gate, effects, get_size
from interp_common import flatten_alpha

p=argparse.ArgumentParser();p.add_argument('--method',choices=['predictor','lora'],required=True);args=p.parse_args()
path=RESULT_ROOT/'adapter_audit'/args.method/'results.json';r=json.loads(path.read_text())
assert r['complete'];data=copy.deepcopy(prepare_data())
for items in data.values():
 for x in items:
  if x['task']=='sva':x['target'],x['foil_id']=x['foil_id'],x['target']
runner=AuditModel();params=prepare(runner,args.method,8)
for par in params:par.requires_grad_(False)
load_adapter(runner,args.method,f'/scratch/yurenh2/interp_audit_20260929/{args.method}_best.pt')
size=get_size(runner,args.method);labels=r['mask_unit_labels'];strata={}
if args.method=='predictor':
 sums=torch.zeros(size,device='cuda');ntokens=0
 with torch.no_grad():
  for start in range(0,len(data['discovery']),16):
   batch=data['discovery'][start:start+16];ids=runner.batch(batch)
   diff=flatten_alpha(runner.predictor(ids))-flatten_alpha(runner.baseline_predictor(ids))
   mask=torch.arange(ids.shape[1],device='cuda')[None,:]<=runner.last_positions[:,None]
   sums+=(diff.square()*mask[:,:,None]).sum((0,1));ntokens+=mask.sum().item()
 magnitude=(sums/ntokens).sqrt().cpu().numpy();r['coefficient_change_rms']=magnitude.tolist()
 bins={}
 for i,x in enumerate(labels):bins.setdefault((x['layer'],x['stream']),[]).append(i)
 for key,indices in bins.items():
  cuts=np.quantile(magnitude[indices],[.25,.5,.75])
  for i in indices:strata[i]=(*key,int(np.searchsorted(cuts,magnitude[i])))
 selected=['sva_75','sva_189','ioi_75','ioi_189']
else:
 for i,x in enumerate(labels):strata[i]=x['module']
 selected=['sva_13','sva_34','sva_67','ioi_67']
pools=defaultdict(list)
for i,key in strata.items():pools[key].append(i)
rng=np.random.default_rng(2026093050);choices={}
for name in selected:
 counts=defaultdict(int)
 for i in r['all_masks'][name]:counts[strata[i]]+=1
 for repeat in range(3):
  choice=[]
  for key,n in counts.items():choice.extend(rng.choice(pools[key],n,False).tolist())
  choices[f'matched_{name}_{repeat}']=choice
r['matched_controls_protocol']='Predictor: layer x stream x discovery coefficient-change RMS quartile; LoRA: projection module. Sampling without replacement within strata; masks may overlap selected masks. Three random replicates.'
r['all_masks'].update(choices)
for split in ['heldout','transfer']:
 s=r['stages'][split]
 for name,indices in choices.items():
  gate=torch.ones(size,device='cuda');gate[indices]=0
  ev=evaluate_gate(runner,args.method,data[split],gate,16)
  s['arms'][name]=dict(**ev,effects=effects(s['base'],s['adapted'],ev))
 print(args.method,'MATCHED',split,flush=True);write_json(path,r)
