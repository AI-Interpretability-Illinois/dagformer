"""Check target-key specificity, table following and color-mass decomposition."""
import json
from pathlib import Path
import numpy as np

root=Path(__file__).resolve().parent
r=json.loads((root/'results.json').read_text());data=json.loads((root/'datasets.json').read_text())
assert r['complete'] and all(x==0. for x in r['original_numeric_replay_checks'].values())

def stat(x,ref=None):
 x=np.asarray(x,dtype=float);rng=np.random.default_rng(2026092931)
 if not len(x):return None
 idx=rng.integers(len(x),size=(4000,len(x)))
 out=dict(n=len(x),mean=float(x.mean()),paired95ci=np.quantile(x[idx].mean(1),[.025,.975]).tolist())
 if ref is not None:
  d=x-ref;out['delta']=float(d.mean());out['delta_paired95ci']=np.quantile(d[idx].mean(1),[.025,.975]).tolist()
 return out

def vals(stage,arm,key):return np.asarray(r['stages'][stage]['arms'][arm]['values'][key])
def colorlp(stage,arm,i,color):
 case=data[stage][i]
 label=next(k for k in ['original','third','spare'] if case['values'][k]==color)
 return vals(stage,arm,label+'_logp')[i]

def key_effect(arm):return vals('heldout',arm,'third_vs_spare_margin')+vals('donor_query',arm,'third_vs_spare_margin')
def table_effect(arm):
 out=[]
 for i in range(64):
  old=data['heldout'][i]['values']['third'];new=data['recipient_table'][i]['values']['third']
  assert old!=new
  x=colorlp('heldout',arm,i,old)-colorlp('heldout',arm,i,new)
  y=colorlp('recipient_table',arm,i,old)-colorlp('recipient_table',arm,i,new)
  out.append(x-y)
 return np.array(out)

for a,b in zip(data['heldout'],data['donor_query']):
 assert a['ids']['recipient']==b['ids']['recipient']
 assert a['values']['third']==b['values']['spare'] and a['values']['spare']==b['values']['third']
baseline_key_effect=key_effect('recipient')
first_both=vals('heldout','donor_first_hop','donor_key_vocab_accuracy').astype(bool)&vals('donor_query','donor_first_hop','donor_key_vocab_accuracy').astype(bool)
cf_both=vals('heldout','input_counterfactual','third_vocab_accuracy').astype(bool)&vals('donor_query','input_counterfactual','third_vocab_accuracy').astype(bool)
masks=dict(all=np.ones(64,dtype=bool),both_donor_queries_correct=first_both,all_required_operations_correct=first_both&cf_both)
output={'definition':{
 'key':'[lp(C)-lp(D)] using donor key C minus the same color difference using donor key D, at identical recipient inputs. Equivalent to base.third_vs_spare + donor_query.third_vs_spare because targets swap.',
 'table':'[lp(old third color)-lp(new third color)] before recipient table permutation minus that difference afterwards. Also subtract the baseline difference-in-differences to measure the patch contribution.',
 'color':'Delta third logp = Delta recipient-color logmass + Delta conditional third logp among recipient three colors.'},
 'baseline_key_effect_numeric_diagnostic':dict(mean=float(baseline_key_effect.mean()),max_abs=float(np.max(np.abs(baseline_key_effect))),explanation='Identical recipient tokens can be grouped with different donor lengths, changing BF16 batch/GEMM shapes. Subtract the corresponding baseline double difference for key-specificity effects.'),'replay_checks':r['original_numeric_replay_checks'],'coverage':{k:int(v.sum()) for k,v in masks.items()},'arms':{}}
for arm in r['stages']['heldout']['arms']:
 if arm=='donor_first_hop':continue
 key,table=key_effect(arm),table_effect(arm)
 o={'key_specificity':{k:stat(key[m],key_effect('recipient')[m]) for k,m in masks.items()},
    'table_following':stat(table,table_effect('recipient')),'color_decomposition':{}}
 for field in ['third_logp','recipient_color_logmass','third_conditional_color_logp']:
  o['color_decomposition'][field]=stat(vals('heldout',arm,field),vals('heldout','recipient',field))
 d=o['color_decomposition'];error=d['third_logp']['delta']-d['recipient_color_logmass']['delta']-d['third_conditional_color_logp']['delta']
 assert abs(error)<1e-6,error
 o['color_decomposition_identity_abs_error']=abs(error)
 output['arms'][arm]=o
(root/'target_specificity.json').write_text(json.dumps(output,indent=2)+'\n')

def fmt(v,delta=False):
 if v is None:return '—'
 mean=v['delta' if delta else 'mean'];lo,hi=v['delta_paired95ci' if delta else 'paired95ci']
 return f'{mean:+.4f} [{lo:+.4f}, {hi:+.4f}]'
lines=['# 第三颜色改善是否真正跟随 donor 的中间键？','',
 '固定 logp 搜索得到的全部预算、head 集合及随机对照，不重新搜索。对原 heldout64、同 recipient 换 donor 查询键、同 donor 换 recipient 第二张表重评；新增 recipient 三种颜色的 log-probability。原有第三颜色 logp、准确率及 donor-key 准确率逐例完全重放，最大误差均为 0。','',
 '关键控制：recipient 输入完全相同，donor 先问到 C，再改问另一个实体而问到 D。如果真正移植了可用的中间键，颜色 C 相对颜色 D 的偏好应该随 donor 改变。双差 = `[lp(C色)-lp(D色)](donor C) − [lp(C色)-lp(D色)](donor D)`。理论上未干预模型不随 donor 改变；本实现按 donor 长度分桶，相同 recipient 可落在不同 BF16 batch/GEMM 形状，基线出现微小数值差。因此正式键特异性指标还扣除对应的未干预双差。','',
 '| Arm | 键特异性双差增量 [paired 95% CI] | 改 recipient 表的目标跟随双差 | 表跟随相对未干预的增量 |','|---|---:|---:|---:|']
for arm in ['recipient','top1','top2','top4','top8','top16','random4_0','random4_1','random4_2']:
 o=output['arms'][arm]
 lines.append(f"| {arm} | {fmt(o['key_specificity']['all'],True)} | {fmt(o['table_following'])} | {fmt(o['table_following'],True)} |")
lines+=['','颜色概率分解：目标第三颜色的完整词表 logp 改善，可以来自颜色整体概率上升，也可以来自在 recipient 三色之间更偏向第三颜色。后者仍需上述换键对照；同时降低旧颜色、抬高另外两色也可能提高这一项。','',
 '| Arm | Δ第三颜色 logp | Δ三色总 logmass | Δ三色内正确颜色 logp |','|---|---:|---:|---:|']
for arm in ['top1','top2','top4','top8','top16']:
 o=output['arms'][arm]['color_decomposition']
 lines.append('| '+arm+' | '+' | '.join(fmt(o[k],True) for k in ['third_logp','recipient_color_logmass','third_conditional_color_logp'])+' |')
lines+=['',f"两种 donor query 本来均答对键的样本有 {int(first_both.sum())}/64；再要求两种 recipient 输入新键对照本来都能答对颜色，剩 {int((first_both&cf_both).sum())}/64。子集检验保存在 `target_specificity.json`，没有按 patch 成功筛选。",'',
 '置信区间为 4,000 次按同一随机表配对重采样，未调整多重比较；条件均值只描述这一组固定 checkpoint、提示和输入表，不证明完整组合回路。']
(root/'README.md').write_text('\n'.join(lines)+'\n')
for arm in ['top1','top2','top4','top8','top16']:
 o=output['arms'][arm]
 print(arm,'key',fmt(o['key_specificity']['all'],True),'table_delta',fmt(o['table_following'],True))
print('coverage',output['coverage'])
