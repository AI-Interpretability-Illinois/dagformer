"""Finish controls and compiled exports after the primary runs complete."""
import gzip
import json
import os
from pathlib import Path
import subprocess
import shutil
import sys
import time
from interp_audit_common import ROOT
from interp_followup_controls import gzwrite

out=ROOT/'experiments/results/interp_followup_20260929'
log=ROOT/'logs/interp_followup_20260929'
env={**os.environ,'CUDA_VISIBLE_DEVICES':'3'}
for seed in [20260930,20261001,20261002]:
    for kind in ['outputs','lora_rows','full']:
        label=f'{kind}_s{seed}'
        with gzip.open(out/'runs'/label/'results.json.gz','rt') as f:assert json.load(f)['complete']
# Isolate the editing-unit change on exactly the old rank-8 trained weights.
work=Path('/scratch/yurenh2/interp_followup_20260929')
aux=work/'runs/lora_rows_original_rank8';aux_result=out/'runs/lora_rows_original_rank8/results.json.gz'
if not aux_result.exists():
    (aux/'lora').mkdir(parents=True,exist_ok=True);(aux/'weights').mkdir(exist_ok=True)
    shutil.copy2(ROOT/'experiments/results/interp_audit_20260929/adapter_audit/lora/results.json',aux/'lora/results.json')
    shutil.copy2('/scratch/yurenh2/interp_audit_20260929/lora_best.pt',aux/'weights/lora_best.pt')
    print('SAME-WEIGHTS LORA ROW AUDIT START',flush=True)
    with (log/'lora_rows_original_rank8.log').open('w') as f:
        subprocess.run([sys.executable,str(ROOT/'scripts/interp_adapter_audit.py'),'--method','lora','--rank','8','--seed','20260930',
            '--out',str(aux),'--weights',str(aux/'weights'),'--lora-unit','row','--budget-unit','trainable_fraction','--audit-only'],
            stdout=f,stderr=subprocess.STDOUT,env=env,check=True)
    record=json.loads((aux/'lora/results.json').read_text());assert record['complete'];gzwrite(aux_result,record)
    print('SAME-WEIGHTS LORA ROW AUDIT COMPLETE',flush=True)
for seed in [20260930,20261001,20261002]:
    for kind in ['outputs','lora_rows','full']:
        label=f'{kind}_s{seed}';p=out/'runs'/label/'controls.json.gz'
        if p.exists():
            with gzip.open(p,'rt') as f:
                if json.load(f).get('complete'):continue
        print('CONTROLS START',label,flush=True)
        with (log/f'{label}_controls.log').open('w') as f:
            subprocess.run([sys.executable,str(ROOT/'scripts/interp_followup_controls.py'),'--kind',kind,'--seed',str(seed)],stdout=f,stderr=subprocess.STDOUT,env=env,check=True)
        print('CONTROLS COMPLETE',label,flush=True)
    p=out/'runs'/f'full_s{seed}'/'compiled.json.gz'
    if p.exists():
        with gzip.open(p,'rt') as f:
            if json.load(f).get('complete'):continue
    print('COMPILE START',seed,flush=True)
    with (log/f'full_s{seed}_compile.log').open('w') as f:
        subprocess.run([sys.executable,str(ROOT/'scripts/interp_predictor_compile.py'),'--seed',str(seed)],stdout=f,stderr=subprocess.STDOUT,env=env,check=True)
    print('COMPILE COMPLETE',seed,flush=True)
print('FOLLOWUP COMPLETE',flush=True)
