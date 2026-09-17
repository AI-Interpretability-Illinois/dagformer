"""Validate and list the completed September 17 harness campaign and saved samples."""
import argparse
from datetime import datetime,timezone
import json
from pathlib import Path
import subprocess

root=Path(__file__).resolve().parents[1]
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--root',type=Path,default=root/'experiments/results/eval_20260917')
base=parser.parse_args().root.resolve()
groups={
 'standard_matched':(7,14), 'standard_latest':(1,14), 'standard_ladder':(5,14),
 'standard_frozen_predictor':(3,14),
 'standard_context_pred':(1,14), 'standard_context_corr':(1,14), 'standard_context_both':(1,14),
 'standard_context_random1000':(1,14), 'standard_context_random1001':(1,14),
 'commonsense_content':(7,1), 'gsm8k_components':(7,2),
 'gsm8k_full':(7,1), 'gsm8k_latest':(1,1), 'gsm8k_no_cache':(1,1),
}
result={'generated_utc':datetime.now(timezone.utc).isoformat(),
 'git_commit_at_inventory':subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip(),
 'scope':'completed harness result files; a setting can reuse a checkpoint with a different intervention or task',
 'groups':{}}
lines=['# Completed harness evaluation inventory','','A setting can reuse the same trained checkpoint with a different intervention or task.',
       'The natural-language and interpretability reports explain which settings are directly comparable.',
       'Sample counts below come from each saved harness result; per-document files are retained separately.','',
       '| Group | Completed settings | Tasks per setting | Recorded effective samples across settings/tasks |',
       '|---|---:|---:|---:|']
for group,(expected,ntasks) in groups.items():
 paths=sorted((base/group).glob('*__*.json'))
 assert len(paths)==expected,(group,len(paths),expected)
 records=[]
 for path in paths:
  x=json.loads(path.read_text()); assert len(x['results'])==ntasks,(path,len(x['results']),ntasks)
  sample_files=[]
  for task in x['results']:
   sample=path.parent/'samples'/f"{x['model']['name']}__{task}.jsonl"
   assert sample.exists(),sample
   count=sum(1 for line in sample.open() if line.strip())
   n=x['n-samples'][task]['effective']
   # Generation harness files may have separate records for each extraction filter.
   filter_names=set()
   ids_by_filter={}
   for line in sample.open():
    if not line.strip():continue
    row=json.loads(line); filt=row.get('filter','none')
    filter_names.add(filt)
    ids_by_filter.setdefault(filt,set()).add(str(row['doc_id']))
   assert all(len(ids)==n for ids in ids_by_filter.values()),(sample,n,{k:len(v) for k,v in ids_by_filter.items()})
   sample_files.append({'task':task,'effective_samples':n,'sample_file':str(sample.relative_to(base)),
                        'rows':count,'filters':sorted(filter_names)})
  records.append({'source':str(path.relative_to(base)),'model':x['model'],'tasks':sample_files})
 result['groups'][group]=records
 total=sum(t['effective_samples'] for r in records for t in r['tasks'])
 lines.append(f'| [{group}](../{group}) | {len(records)} | {ntasks} | {total:,} |')
result['completed_harness_settings']=sum(len(v) for v in result['groups'].values())
result['completed_task_endpoints']=sum(len(r['tasks']) for v in result['groups'].values() for r in v)
lines+=['',f"Total: {result['completed_harness_settings']} harness runs, {result['completed_task_endpoints']} task endpoints.",
        'Multiple task endpoints and interventions share checkpoints and documents; they are not independent training replications.','']
(base/'provenance/eval_inventory.json').write_text(json.dumps(result,indent=2)+'\n')
(base/'provenance/eval_inventory.md').write_text('\n'.join(lines))
print('\n'.join(lines))
