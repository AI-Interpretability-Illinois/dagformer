"""Validate J-lens predictions against individual R-connection interventions.

Zero one effective R coefficient at the query token. Measure the resulting
full target-layer state change (including the local MLP response), read that
change through the calibrated J-lens, and compare with actual output changes.
This tests 21 predefined connections, not a best-edge search on test data.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from transformers import AutoTokenizer
from interp_audit_common import AuditModel, RESULT_ROOT, TOKENIZER, prepare_data, paired_stats, write_json
from interp_neuron_flow import effective_routes
from interp_jacobian_lens import residuals


def correlation(a,b):
    if np.std(a)<1e-8 or np.std(b)<1e-8:return None
    return float(np.corrcoef(a,b)[0,1])


@torch.no_grad()
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--batch-size',type=int,default=8)
    p.add_argument('--out',type=Path,default=RESULT_ROOT/'jacobian_lens'/'dag_routes.json');args=p.parse_args()
    runner=AuditModel();data=prepare_data();tok=AutoTokenizer.from_pretrained(str(TOKENIZER),local_files_only=True)
    checkpoint=torch.load('/scratch/yurenh2/interp_audit_20260929/dag_jacobians.pt',weights_only=False)
    matrices={l:m.to('cuda') for l,m in checkpoint['matrices'].items()};layers=list(matrices)
    edges=[(i,x) for i,x in enumerate(runner.layout) if x['stream']=='r' and x['layer'] in layers]
    weight=runner.base.lm_head.weight
    result=dict(complete=False,protocol='All 21 R source-to-target connections at zero-based layers 3,6,9. Zero only the final-query-token coefficient, including predictor+local correction. Read out the measured change in the full target state using either J-lens or direct logit lens. This includes the actual target MLP response and does not predict that local nonlinear response from the raw message. First 64 SVA prompts (32 pairs) per previously reserved split; all 21 edges reported.',edges=[x for _,x in edges],stages={})
    for split in ['heldout','transfer']:
        items=[x for x in data[split] if x['task']=='sva'][:64]
        values={str(i):[] for i,_ in edges};examples=[]
        for start in range(0,len(items),args.batch_size):
            batch=items[start:start+args.batch_size];cap={}
            with residuals(runner,layers,cap):base_logits=runner.forward(batch)
            ar=torch.arange(len(batch),device='cuda');pos=runner.last_positions
            targets=torch.tensor([x['target'] for x in batch],device='cuda');foils=torch.tensor([x['foil_id'] for x in batch],device='cuda')
            mask=torch.zeros(cap['final'].shape[:2],device='cuda');mask[ar,pos]=1
            base_margin=base_logits[ar,targets]-base_logits[ar,foils]
            for index,label in edges:
                l=label['layer'];gate=torch.ones(len(runner.layout),device='cuda');gate[index]=0;edit={}
                with effective_routes(runner,gate,mask),residuals(runner,layers,edit):logits=runner.forward(batch)
                true_margin=logits[ar,targets]-logits[ar,foils]
                a=cap['states'][l][ar,pos];b=edit['states'][l][ar,pos]
                predictions={};zdelta={}
                for name,matrix in [('direct',None),('jacobian',matrices[l])]:
                    def norm(x):return runner.base.model.norm((x if matrix is None else x.float()@matrix.T).to(weight.dtype))
                    dz=norm(b).float()-norm(a).float();zdelta[name]=dz
                    predictions[name]=(dz*(weight[targets].float()-weight[foils].float())).sum(-1)
                for j,x in enumerate(batch):
                    values[str(index)].append(dict(group=x['group'],baseline_margin=base_margin[j].item(),edited_margin=true_margin[j].item(),
                        actual_effect=(true_margin-base_margin)[j].item(),direct=predictions['direct'][j].item(),jacobian=predictions['jacobian'][j].item(),
                        state_change_norm=(b[j].float()-a[j].float()).norm().item()))
                if start==0:
                    delta_logits=torch.nn.functional.linear(zdelta['jacobian'][0].to(weight.dtype),weight).float()
                    examples.append(dict(edge=label,prompt=batch[0]['prompt'],target=batch[0]['answer'],foil=batch[0]['foil'],
                        actual_margin_change=(true_margin-base_margin)[0].item(),predicted_margin_change=predictions['jacobian'][0].item(),
                        lens_tokens_increased=tok.convert_ids_to_tokens(delta_logits.topk(5).indices.tolist()),
                        lens_tokens_decreased=tok.convert_ids_to_tokens((-delta_logits).topk(5).indices.tolist())))
        summary={}
        for index,rows in values.items():
            actual=np.array([x['actual_effect'] for x in rows]);groups=[x['group'] for x in rows]
            summary[index]=dict(edited_accuracy=paired_stats([x['edited_margin']>0 for x in rows],groups),methods={})
            for method in ['direct','jacobian']:
                pred=np.array([x[method] for x in rows])
                summary[index]['methods'][method]=dict(pearson=correlation(actual,pred),sign_agreement=paired_stats(actual*pred>0,groups))
        result['stages'][split]=dict(values=values,summary=summary,examples=examples)
        write_json(args.out,result);print('R CONNECTIONS',split,'complete',flush=True)
    result['complete']=True;write_json(args.out,result)

if __name__=='__main__':main()
