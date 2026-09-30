"""Run the fixed three-seed adapter confirmation sequentially on one GPU."""
import gzip
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from interp_audit_common import ROOT, write_json

OUT=ROOT/'experiments/results/interp_followup_20260929'
WORK=Path('/scratch/yurenh2/interp_followup_20260929')
LOG=ROOT/'logs/interp_followup_20260929'
seeds=[20260930,20261001,20261002]
splits=['heldout','transfer','confirmation','hard_transfer']
protocol=json.loads((OUT/'protocol.json').read_text())
validation=[]
for directory,lr in [('outputs_lr3e4',3e-4),('outputs_lr1e3',1e-3)]:
    r=json.loads((WORK/'validation'/directory/'predictor/results.json').read_text())
    assert r['training_complete']
    validation.append(dict(directory=directory,lr=lr,score=r['selected_checkpoint']['score'],selected_step=r['selected_checkpoint']['step']))
selected=min(validation,key=lambda x:x['score'])
protocol['output_lr_selection']=dict(candidates=validation,selected=selected,criterion='Lowest eligible mean validation SVA/IOI NLL. No test scores used.')
protocol['hard_gate_implementation']='Exact p0/p1 selection at binary gates, with straight-through interpolation derivative; permits exact parameter-materialization parity.'
write_json(OUT/'protocol.json',protocol)
for seed in seeds:
    for kind in ['outputs','lora_rows','full']:
        method='lora' if kind=='lora_rows' else 'predictor'
        run=WORK/'runs'/f'{kind}_s{seed}';weights=run/'weights'
        result_path=run/method/'results.json'
        if result_path.exists():
            record=json.loads(result_path.read_text())
            if record.get('complete') and all(s in record.get('stages',{}) for s in splits):
                print('SKIP COMPLETE',kind,seed,flush=True);continue
        run.mkdir(parents=True,exist_ok=True);weights.mkdir(exist_ok=True);(run/method).mkdir(exist_ok=True)
        if seed==seeds[0] and not result_path.exists() and kind!='lora_rows':
            source=WORK/'validation'/selected['directory'] if kind=='outputs' else ROOT/'experiments/results/interp_audit_20260929/adapter_audit'
            shutil.copy2(source/'predictor/results.json',result_path)
            src_weights=source/'weights/predictor_best.pt' if kind=='outputs' else Path('/scratch/yurenh2/interp_audit_20260929/predictor_best.pt')
            shutil.copy2(src_weights,weights/'predictor_best.pt')
        command=[sys.executable,str(ROOT/'scripts/interp_adapter_audit.py'),'--method',method,'--seed',str(seed),
            '--out',str(run),'--weights',str(weights),'--extra-tests',str(OUT/'extra_tests.json.gz'),'--test-splits',*splits]
        if kind=='outputs':command+=['--predictor-scope','outputs','--lr',str(selected['lr']),'--budget-unit','trainable_fraction']
        elif kind=='lora_rows':command+=['--rank','7','--lr','0.0003','--lora-unit','row','--budget-unit','trainable_fraction']
        else:command+=['--lr','0.00003']
        if result_path.exists() and json.loads(result_path.read_text()).get('training_complete'):
            command+=['--audit-only']
        print('START',kind,seed,flush=True)
        with (LOG/f'{kind}_s{seed}.log').open('w') as log:
            subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,check=True,env={**os.environ,'CUDA_VISIBLE_DEVICES':'3'})
        record=json.loads(result_path.read_text());assert record['complete']
        dest=OUT/'runs'/f'{kind}_s{seed}';dest.mkdir(parents=True,exist_ok=True)
        with gzip.open(dest/'results.json.gz','wt') as f:json.dump(record,f,separators=(',',':'),allow_nan=False)
        print('COMPLETE',kind,seed,flush=True)
print('ALL PRIMARY RUNS COMPLETE',flush=True)
