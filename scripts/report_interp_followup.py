"""Summarize seed replication, stronger controls, and one-predictor exports."""
import csv
import gzip
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from interp_audit_common import ROOT,paired_stats,write_json

OUT=ROOT/'experiments/results/interp_followup_20260929'
SEEDS=[20260930,20261001,20261002]
COLORS={'full':'#157F80','outputs':'#4178B6','lora_rows':'#B96126','compiled':'#704B97'}


def read(path):
    with gzip.open(path,'rt') as f:return json.load(f)


def metric(arm):
    sva=[x for x in arm['values'] if x['task']=='sva']
    return dict(grammar=paired_stats([x['margin']<0 for x in sva],[x['group'] for x in sva]),
                ioi=arm['summary']['ioi']['candidate_correct'],retain=arm['summary']['retain']['nll'])


def mean_range(values):
    values=np.array(values)
    return dict(mean=float(values.mean()),sd=float(values.std(ddof=1)),min=float(values.min()),max=float(values.max()),seeds=len(values))


def save(fig,name):
    fig.tight_layout();fig.savefig(OUT/f'{name}.png',dpi=170);fig.savefig(OUT/f'{name}.pdf');plt.close(fig)


records={};controls={};compiled={};rows=[];summary=dict(seeds=SEEDS,scope='Three adapter-training seeds on one fixed pretrained checkpoint. Ranges describe seed spread; they are not pretraining-seed confidence intervals.',runs={},aggregate={},compilation={},materialized={},same_weights_lora={})
for kind in ['full','outputs','lora_rows']:
    for seed in SEEDS:
        label=f'{kind}_s{seed}';r=read(OUT/'runs'/label/'results.json.gz');c=read(OUT/'runs'/label/'controls.json.gz')
        assert r['complete'] and c['complete'];records[(kind,seed)]=r;controls[(kind,seed)]=c
        scale=len(r['mask_unit_labels']) if kind=='full' else r['trainable_parameters']/r['parameters_per_edit_unit']
        summary['runs'][label]=dict(trainable_parameters=r['trainable_parameters'],checkpoint=r['selected_checkpoint'],budget_unit=r['budget_definition'],cross_seed_jaccard=c['cross_seed_jaccard'])
        for split,stage in r['stages'].items():
            for name,arm in {'base':stage['base'],'adapted':stage['adapted'],**stage['arms'],**c['stages'][split]}.items():
                m=metric(arm)
                rows.append(dict(kind=kind,seed=seed,split=split,arm=name,grammar=m['grammar']['mean'],grammar_low=m['grammar']['ci95'][0],grammar_high=m['grammar']['ci95'][1],ioi=m['ioi']['mean'],ioi_low=m['ioi']['ci95'][0],ioi_high=m['ioi']['ci95'][1],retain_nll=m['retain']['mean']))
        for name,export in c['exports'].items():summary['materialized'][f'{label}/{name}']=export
        if kind=='full':
            co=read(OUT/'runs'/label/'compiled.json.gz');assert co['complete'];compiled[seed]=co
            summary['compilation'][str(seed)]={}
            for name,arm in co['arms'].items():
                summary['compilation'][str(seed)][name]={k:v for k,v in arm.items() if k!='stages'}
                for split,stage in arm['stages'].items():
                    m=metric(stage['student'])
                    rows.append(dict(kind='compiled',seed=seed,split=split,arm=name,grammar=m['grammar']['mean'],grammar_low=m['grammar']['ci95'][0],grammar_high=m['grammar']['ci95'][1],ioi=m['ioi']['mean'],ioi_low=m['ioi']['ci95'][0],ioi_high=m['ioi']['ci95'][1],retain_nll=m['retain']['mean']))
with (OUT/'metrics.csv').open('w') as f:
    writer=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');writer.writeheader();writer.writerows(rows)
for kind in ['full','outputs','lora_rows','compiled']:
    summary['aggregate'][kind]={}
    for split in ['heldout','transfer','confirmation','hard_transfer']:
        summary['aggregate'][kind][split]={}
        for fraction in [.02,.05]:
            selected=[];cross=[];random=[]
            for seed in SEEDS:
                r=records[('full' if kind=='compiled' else kind,seed)]
                scale=len(r['mask_unit_labels']) if kind in ['full','compiled'] else r['trainable_parameters']/r['parameters_per_edit_unit']
                name=f'sva_{round(scale*fraction)}'
                selected.append(next(x for x in rows if (x['kind'],x['seed'],x['split'],x['arm'])==(kind,seed,split,name)))
                if kind!='compiled':
                    cross.append(next(x for x in rows if (x['kind'],x['seed'],x['split'],x['arm'])==(kind,seed,split,'cross_seed_'+name)))
                    random.append({k:np.mean([next(x for x in rows if (x['kind'],x['seed'],x['split'],x['arm'])==(kind,seed,split,f'matched_{name}_{i}'))[k] for i in range(3)]) for k in ['grammar','ioi','retain_nll']})
            summary['aggregate'][kind][split][str(fraction)]=dict(selected={k:mean_range([x[k] for x in selected]) for k in ['grammar','ioi','retain_nll']})
            if kind!='compiled':
                summary['aggregate'][kind][split][str(fraction)].update(
                    cross_seed={k:mean_range([x[k] for x in cross]) for k in ['grammar','ioi','retain_nll']},
                    transferred_seeds_only={k:mean_range([x[k] for x in cross[1:]]) for k in ['grammar','ioi','retain_nll']},
                    matched_random={k:mean_range([x[k] for x in random]) for k in ['grammar','ioi','retain_nll']})
        summary['aggregate'][kind][split]['reference_arms']={}
        if kind!='compiled':
            for name in ['base','adapted','global_best','global_match']:
                selected=[next(x for x in rows if (x['kind'],x['seed'],x['split'],x['arm'])==(kind,seed,split,name)) for seed in SEEDS]
                summary['aggregate'][kind][split]['reference_arms'][name]={k:mean_range([x[k] for x in selected]) for k in ['grammar','ioi','retain_nll']}
aux=read(OUT/'runs/lora_rows_original_rank8/results.json.gz');assert aux['complete']
old=json.loads((ROOT/'experiments/results/interp_audit_20260929/adapter_audit/lora/results.json').read_text())
for split in ['heldout','transfer']:
    summary['same_weights_lora'][split]=dict(rank_13=metric(old['stages'][split]['arms']['sva_13']),rank_67=metric(old['stages'][split]['arms']['sva_67']),row_5652=metric(aux['stages'][split]['arms']['sva_5652']),row_14131=metric(aux['stages'][split]['arms']['sva_14131']))
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
fig,axes=plt.subplots(1,2,figsize=(10,4.1))
for ax,split in zip(axes,['confirmation','hard_transfer']):
    for kind in ['outputs','lora_rows','full']:
        points=[]
        for f in [.005,.01,.02,.05,.1]:
            values=[]
            for seed in SEEDS:
                r=records[(kind,seed)];scale=len(r['mask_unit_labels']) if kind=='full' else r['trainable_parameters']/r['parameters_per_edit_unit'];name=f'sva_{round(scale*f)}'
                values.append(next(x for x in rows if (x['kind'],x['seed'],x['split'],x['arm'])==(kind,seed,split,name)))
            x=np.mean([v['grammar'] for v in values])*100;y=np.mean([v['ioi'] for v in values])*100;points.append((x,y))
            ax.scatter([v['grammar']*100 for v in values],[v['ioi']*100 for v in values],color=COLORS[kind],alpha=.2,s=14)
            if f==.02:ax.scatter([x],[y],s=75,color=COLORS[kind],edgecolors='black',linewidths=.6)
        labels={'outputs':'Predictor outputs (1.94M)','lora_rows':'LoRA B rows (1.98M)','full':'Full predictor (30.39M)'}
        points=np.array(points);ax.plot(points[:,0],points[:,1],'-o',ms=4,color=COLORS[kind],label=labels[kind])
    ax.set(title=split,xlabel='Restored grammatical agreement (%)',ylabel='Preserved recipient identification (%)',xlim=(-3,103),ylim=(40,103));ax.grid(alpha=.15)
axes[0].legend(fontsize=8)
fig.suptitle('Three adapter seeds; highlighted budget: 2% of coordinates (full) or trained parameters (others)',fontsize=10)
save(fig,'rollback_frontier')
fig,axes=plt.subplots(1,2,figsize=(10,4.1))
for ax,split in zip(axes,['confirmation','hard_transfer']):
    for offset,kind,label in [(-.18,'full','Two-predictor teacher'),(.18,'compiled','One-predictor student')]:
        for i,metric_name in enumerate(['grammar','ioi']):
            vals=[]
            for seed in SEEDS:
                row=next(x for x in rows if (x['kind'],x['seed'],x['split'],x['arm'])==(kind,seed,split,'sva_75'));vals.append(row[metric_name]*100)
            ax.bar(i+offset,np.mean(vals),.34,color=COLORS[kind],alpha=.8,label=label if i==0 else None)
            ax.scatter(np.full(3,i+offset),vals,c='black',s=14,zorder=3)
    ax.set(title=split,ylabel='Accuracy (%)',ylim=(0,105));ax.set_xticks([0,1],labels=['Grammar repair','Recipient task']);ax.grid(axis='y',alpha=.15)
axes[0].legend(fontsize=8);fig.suptitle('75-coordinate repair; points show three adapter seeds',fontsize=11)
save(fig,'single_predictor_compilation')
write_json(OUT/'summary.json',summary)
for kind in ['full','outputs','lora_rows','compiled']:
    for split in ['confirmation','hard_transfer']:
        a=summary['aggregate'][kind][split]['0.02']['selected'];print(kind,split,'grammar',a['grammar'],'ioi',a['ioi'])
