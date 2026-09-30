"""Compare original, adapted and selectively restored predictor on a prompt."""
import argparse
import json
from pathlib import Path
import torch
from transformers import AutoTokenizer
from interp_audit_common import AuditModel, RESULT_ROOT, TOKENIZER
from interp_adapter_audit import load_adapter


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--prompt',required=True)
    p.add_argument('--choices',nargs='+',required=True,help='Single-token continuations, with an implicit leading space.')
    p.add_argument('--mask',default='sva_75')
    p.add_argument('--adapter',type=Path,default=Path('/scratch/yurenh2/interp_audit_20260929/predictor_best.pt'))
    p.add_argument('--result',type=Path,default=RESULT_ROOT/'adapter_audit/predictor/results.json')
    p.add_argument('--model-path',type=Path)
    p.add_argument('--tokenizer',type=Path,default=TOKENIZER)
    p.add_argument('--single',action='store_true',help='Run one materialized predictor, without loading or evaluating a rollback teacher.')
    p.add_argument('--row-patch',type=Path,help='Optional compiled sparse row patch over the adapted predictor; requires --single.')
    args=p.parse_args();tok=AutoTokenizer.from_pretrained(str(args.tokenizer),local_files_only=True)
    choices=[tok.encode(' '+x,add_special_tokens=False) for x in args.choices]
    if any(len(x)!=1 for x in choices):p.error('Every choice must tokenize to one continuation token.')
    ids=[x[0] for x in choices];items=[dict(ids=tok.encode(args.prompt,add_special_tokens=False))]
    if args.row_patch and not args.single:p.error('--row-patch requires --single')
    runner=AuditModel(model_path=args.model_path)
    if not args.single:runner.copy_baseline_predictor()
    metadata=load_adapter(runner,'predictor',args.adapter)
    if metadata.get('materialized_rollback') and not args.single:p.error('Materialized adapters must be used with --single')
    if args.row_patch:
        patch=torch.load(args.row_patch,map_location='cuda',weights_only=False)
        if patch.get('format')!='dagformer_predictor_row_patch_v1':p.error('Unrecognized row-patch format')
        rows=patch['rows'];indices=patch['indices'];offset=0
        assert rows.shape==(len(indices),runner.predictor.layer_heads[0].in_features+1)
        assert all(0<=i<len(runner.layout) for i in indices)
        with torch.no_grad():
            for l,head in enumerate(runner.predictor.layer_heads):
                for j,i in enumerate(indices):
                    if offset<=i<offset+head.out_features:
                        head.weight[i-offset].copy_(rows[j,:-1]);runner.predictor.layer_biases[l][i-offset].copy_(rows[j,-1])
                offset+=head.out_features
    if args.single:
        calls={'predictor':0}
        def count(module,inputs,output):calls['predictor']+=1
        handle=runner.predictor.register_forward_hook(count)
        with torch.no_grad():logits=runner.forward(items)[0];selected=logits[ids]
        handle.remove();assert calls['predictor']==1
        print(json.dumps(dict(prompt=args.prompt,predictor_calls=1,chosen=args.choices[int(selected.argmax())],
            probabilities_within_choices=dict(zip(args.choices,selected.softmax(-1).tolist())),
            full_vocab_logprobs={word:float(logits.log_softmax(-1)[token]) for word,token in zip(args.choices,ids)}),ensure_ascii=False,indent=2))
        return
    record=json.loads(args.result.read_text());mask=record['all_masks'][args.mask]
    gate=torch.ones(len(runner.layout),device='cuda');gate[mask]=0
    result=dict(prompt=args.prompt,mask=args.mask,restored_coordinates=len(mask),conditions={})
    with torch.no_grad():
        for name,g in [('original',torch.zeros_like(gate)),('adapted',torch.ones_like(gate)),('restored',gate)]:
            logits=runner.forward(items,alpha_gate=g)[0];choice_logits=logits[ids]
            probabilities=choice_logits.softmax(-1).tolist()
            result['conditions'][name]=dict(chosen=args.choices[int(choice_logits.argmax())],
                probabilities_within_choices=dict(zip(args.choices,probabilities)),
                full_vocab_logprobs={word:float(logits.log_softmax(-1)[token]) for word,token in zip(args.choices,ids)})
    print(json.dumps(result,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
