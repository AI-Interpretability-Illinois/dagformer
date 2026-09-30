"""Compare original, adapted and selectively restored predictor on a prompt."""
import argparse
import json
from pathlib import Path
import torch
from transformers import AutoTokenizer
from interp_audit_common import AuditModel, RESULT_ROOT, TOKENIZER


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--prompt',required=True)
    p.add_argument('--choices',nargs='+',required=True,help='Single-token continuations, with an implicit leading space.')
    p.add_argument('--mask',default='sva_75')
    p.add_argument('--adapter',type=Path,default=Path('/scratch/yurenh2/interp_audit_20260929/predictor_best.pt'))
    p.add_argument('--result',type=Path,default=RESULT_ROOT/'adapter_audit/predictor/results.json')
    p.add_argument('--model-path',type=Path)
    args=p.parse_args();tok=AutoTokenizer.from_pretrained(str(TOKENIZER),local_files_only=True)
    choices=[tok.encode(' '+x,add_special_tokens=False) for x in args.choices]
    if any(len(x)!=1 for x in choices):p.error('Every choice must tokenize to one continuation token.')
    ids=[x[0] for x in choices];items=[dict(ids=tok.encode(args.prompt,add_special_tokens=False))]
    runner=AuditModel(model_path=args.model_path);runner.copy_baseline_predictor()
    state=torch.load(args.adapter,map_location='cpu',weights_only=False)
    runner.predictor.load_state_dict(state['predictor'])
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
