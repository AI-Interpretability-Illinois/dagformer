"""Generate tables and figures from the three completed interpretability audits."""
import csv
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from interp_audit_common import RESULT_ROOT, paired_stats, write_json


def read(path):
    return json.loads(path.read_text())


def csv_write(path,rows):
    with path.open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');writer.writeheader();writer.writerows(rows)


def save(fig,name):
    fig.tight_layout();fig.savefig(RESULT_ROOT/f'{name}.png',dpi=170);fig.savefig(RESULT_ROOT/f'{name}.pdf');plt.close(fig)


def grammar(arm):
    vals=[x for x in arm['values'] if x['task']=='sva']
    return paired_stats([x['margin']<0 for x in vals],[x['group'] for x in vals])


plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
colors={'predictor':'#007F7F','lora':'#B65C18','dag':'#007F7F','dense':'#B65C18'}
summary={};rows=[];fig,axes=plt.subplots(1,2,figsize=(10,3.8))
for method in ['predictor','lora']:
    r=read(RESULT_ROOT/'adapter_audit'/method/'results.json');assert r['complete'];size=len(r['mask_unit_labels'])
    summary[method]=dict(trainable_parameters=r['trainable_parameters'],selected_checkpoint=r['selected_checkpoint'],rollback_error=r['full_rollback_max_logit_error'],stages={})
    for split,s in r['stages'].items():
        summary[method]['stages'][split]={}
        for name,a in {'base':s['base'],'adapted':s['adapted'],**s['arms']}.items():
            g=grammar(a);ioi=a['summary']['ioi']['candidate_correct'];retain=a['summary']['retain']['nll']
            k=size if name=='base' else 0 if name=='adapted' else len(r['all_masks'][name])
            rows.append(dict(method=method,split=split,arm=name,units_removed=k,total_units=size,unit_fraction=k/size,
                grammar_accuracy=g['mean'],grammar_low=g['ci95'][0],grammar_high=g['ci95'][1],ioi_accuracy=ioi['mean'],ioi_low=ioi['ci95'][0],ioi_high=ioi['ci95'][1],retain_nll=retain['mean']))
            if name in ['base','adapted','sva_75','sva_189','ioi_377','sva_13','sva_67'] or name.startswith('matched_sva_75'):
                summary[method]['stages'][split][name]=dict(grammar=g,ioi=ioi,retain=retain)
        selected=[x for x in rows if x['method']==method and x['split']==split and x['arm'].startswith('sva_')]
        for axis,key in zip(axes,['grammar_accuracy','ioi_accuracy']):
            axis.plot([100*x['unit_fraction'] for x in selected],[100*x[key] for x in selected],marker='o',linestyle='-' if split=='heldout' else '--',color=colors[method],label=f'{method}, {split}')
for ax,title in zip(axes,['Restore grammatical agreement','Preserve learned recipient identification']):
    ax.set(xlabel='Removed fraction of each adapter\'s own units (%)',ylabel='Accuracy (%)',title=title,ylim=(-3,103));ax.set_xscale('log');ax.grid(alpha=.15)
axes[0].legend(fontsize=8);save(fig,'adapter_rollback')
csv_write(RESULT_ROOT/'adapter_metrics.csv',rows)

rows=[];fig,axes=plt.subplots(1,2,figsize=(10,3.8));nr={}
for kind in ['dag','dense']:
    r=read(RESULT_ROOT/'neuron_flow'/f'{kind}.json');assert r['complete'];nr[kind]=r
    for axis,(split,s) in zip(axes,r['stages'].items()):
        ks=[16,32,64,128,256];means=[];cis=[]
        for name,a in s['arms'].items():
            ac=a['summary']['correct'];rows.append(dict(model=kind,split=split,arm=name,accuracy=ac['mean'],ci_low=ac['ci95'][0],ci_high=ac['ci95'][1],margin=a['summary']['margin']['mean'],counterfactual_fraction=a['fraction_total_counterfactual_effect']))
        for k in ks:
            a=s['arms'][f'top_{k}/sufficiency']['summary']['correct'];means.append(a['mean']);cis.append(a['ci95'])
        means=np.array(means)*100;cis=np.array(cis)*100
        axis.plot(ks,means,'o-',color=colors[kind],label=kind.upper());axis.fill_between(ks,cis[:,0],cis[:,1],color=colors[kind],alpha=.12)
        random=[np.mean([s['arms'][f'random_{k}_{j}/sufficiency']['summary']['correct']['mean'] for j in range(3)])*100 for k in ks]
        axis.plot(ks,random,':',color=colors[kind],alpha=.7)
        axis.set(title=split,xlabel='Patched neuron identities (all token positions)',ylabel='Correct recipient behavior (%)',ylim=(-3,103));axis.set_xscale('log',base=2);axis.set_xticks(ks,labels=ks);axis.grid(alpha=.15)
axes[0].set_ylabel('Correct grammatical number (%)');axes[1].set_ylabel('Correct grammatical number (%)');axes[0].legend();save(fig,'neuron_sufficiency')
csv_write(RESULT_ROOT/'neuron_metrics.csv',rows)
summary['neuron_dag_minus_dense']={}
for split in ['heldout','transfer']:
    summary['neuron_dag_minus_dense'][split]={}
    for k in [16,32,64,128,256]:
        a=nr['dag']['stages'][split]['arms'][f'top_{k}/sufficiency']['values'];b=nr['dense']['stages'][split]['arms'][f'top_{k}/sufficiency']['values']
        assert [x['group'] for x in a]==[x['group'] for x in b]
        summary['neuron_dag_minus_dense'][split][str(k)]=paired_stats([x['correct']-y['correct'] for x,y in zip(a,b)],[x['group'] for x in a])
summary['routing_mediation']={split:{n:dict(accuracy=a['summary']['clean_correct'],fraction_feature_effect_removed=a['fraction_feature_effect_removed']) for n,a in s.items() if n in ['intact','top_4','top_16','top_64']} for split,s in nr['dag']['routing_mediation']['stages'].items()}

rows=[];fig,axes=plt.subplots(1,2,figsize=(10,3.8));summary['jacobian']={}
for kind in ['dag','dense']:
    r=read(RESULT_ROOT/'jacobian_lens'/f'{kind}.json');assert r['complete'];layers=r['layers_zero_based']
    summary['jacobian'][kind]=dict(perturbations=r['perturbations']['summary'],number_readout={s:v['summary'] for s,v in r['number_readout'].items()},readout={})
    for split,s in r['readout'].items():
        summary['jacobian'][kind]['readout'][split]={}
        for name,a in s['arms'].items():
            for task,stats in a['summary'].items():
                rows.append(dict(model=kind,split=split,lens=name,task=task,nll=stats['nll']['mean'],nll_low=stats['nll']['ci95'][0],nll_high=stats['nll']['ci95'][1],choice_accuracy=stats['correct']['mean'],vocab_accuracy=stats['vocab_correct']['mean']))
            summary['jacobian'][kind]['readout'][split][name]=a['summary']['retain']['nll']
    for name,style in [('direct','--'),('jacobian','-')]:
        y=[r['readout']['heldout']['arms'][f'{name}_{l}']['summary']['retain']['nll']['mean'] for l in layers]
        axes[0].plot([l+1 for l in layers],y,marker='o',linestyle=style,color=colors[kind],label=f'{kind}, {name}')
    for name,style in [('identity_cosine','--'),('jacobian_cosine','-')]:
        y=[r['perturbations']['summary'][str(l)]['0.05'][name]['mean'] for l in layers]
        axes[1].plot([l+1 for l in layers],y,marker='o',linestyle=style,color=colors[kind],label=f'{kind}, {name.split("_")[0]}')
axes[0].set(title='Natural next-token readout',ylabel='NLL (nats; lower is better)',xlabel='Source layer (1-based)');axes[0].legend(fontsize=8)
axes[1].set(title='Predict finite residual intervention',ylabel='Response cosine (higher is better)',xlabel='Source layer (1-based)');axes[1].legend(fontsize=8)
for ax in axes:ax.grid(alpha=.15)
save(fig,'jacobian_validation');csv_write(RESULT_ROOT/'jacobian_metrics.csv',rows)
route_path=RESULT_ROOT/'jacobian_lens'/'dag_routes.json'
if route_path.exists():
    r=read(route_path);assert r['complete'];summary['jacobian_routes']={}
    for split,s in r['stages'].items():
        summary['jacobian_routes'][split]={}
        for method in ['direct','jacobian']:
            corr=[a['methods'][method]['pearson'] for a in s['summary'].values() if a['methods'][method]['pearson'] is not None]
            signs=[a['methods'][method]['sign_agreement']['mean'] for a in s['summary'].values()]
            summary['jacobian_routes'][split][method]=dict(mean_edge_pearson=float(np.mean(corr)),median_edge_pearson=float(np.median(corr)),mean_sign_agreement=float(np.mean(signs)))
write_json(RESULT_ROOT/'summary.json',summary)
print(json.dumps(summary,indent=2))
