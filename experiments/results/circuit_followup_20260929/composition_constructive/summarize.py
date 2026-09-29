"""CPU-only scientific summaries of the two constructive interchange searches."""
import json
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parent
BUDGETS=['top1','top2','top4','top8','top16']
STAGES=['heldout','entity_transfer','donor_color','recipient_table','donor_query','implicit_query']

def rename(x):
    if isinstance(x,dict):
        return {k.replace('both_components_correct','donor_and_original_recipient_correct'):rename(v) for k,v in x.items()}
    if isinstance(x,list):return [rename(v) for v in x]
    return x

def estimates(x,ref=None):
    x=np.asarray(x,dtype=float)
    if not len(x):return None
    rng=np.random.default_rng(2026092930)
    ix=rng.integers(len(x),size=(4000,len(x)))
    out={'mean':float(x.mean()),'bootstrap95ci':np.quantile(x[ix].mean(1),[.025,.975]).tolist()}
    if ref is not None:
        delta=x-np.asarray(ref,dtype=float)
        out.update(delta=float(delta.mean()),delta_paired95ci=np.quantile(delta[ix].mean(1),[.025,.975]).tolist())
    return out


def derive(path):
    data=rename(json.loads(path.read_text()))
    data['postprocessing_notes']='Renamed the original both_components_correct field to donor_and_original_recipient_correct; its numerical mask is unchanged. Required-component and three-condition subsets are derived separately.'
    path.write_text(json.dumps(data,indent=2)+'\n')
    derived={'subsets':{},'paired_control_success':{}}
    for stage in STAGES:
        source=data['stages'][stage];arms=source['arms']
        values=lambda arm,key:np.asarray(arms[arm]['values'][key])
        donor=values('donor_first_hop','donor_key_vocab_accuracy').astype(bool)
        original=values('recipient','original_vocab_accuracy').astype(bool)
        cf=values('input_counterfactual','third_vocab_accuracy').astype(bool)
        masks={'all':np.ones(len(donor),dtype=bool),'donor_key_correct':donor,
               'required_components_correct':donor&cf,'all_three_correct':donor&cf&original}
        derived['subsets'][stage]={}
        for label,mask in masks.items():
            group={'n':int(mask.sum()),'total':len(mask),'coverage':float(mask.mean()),'arms':{}}
            for name in arms:
                group['arms'][name]={key:estimates(values(name,key)[mask],values('recipient',key)[mask])
                    for key in ['third_vocab_accuracy','third_logp','donor_key_vocab_accuracy']}
            derived['subsets'][stage][label]=group
    base=data['stages']['heldout']['arms']
    for control in ['donor_color','recipient_table','donor_query']:
        other=data['stages'][control]['arms'];derived['paired_control_success'][control]={}
        ref=np.asarray(base['recipient']['values']['third_vocab_accuracy'])*np.asarray(other['recipient']['values']['third_vocab_accuracy'])
        for arm in base:
            x=np.asarray(base[arm]['values']['third_vocab_accuracy'])*np.asarray(other[arm]['values']['third_vocab_accuracy'])
            derived['paired_control_success'][control][arm]=estimates(x,ref)
    derived['all_four_conditions_success']={}
    ref=np.prod([np.asarray(data['stages'][s]['arms']['recipient']['values']['third_vocab_accuracy']) for s in ['heldout','donor_color','recipient_table','donor_query']],axis=0)
    for arm in base:
        x=np.prod([np.asarray(data['stages'][s]['arms'][arm]['values']['third_vocab_accuracy']) for s in ['heldout','donor_color','recipient_table','donor_query']],axis=0)
        derived['all_four_conditions_success'][arm]=estimates(x,ref)
    (path.parent/'qualified_subset_diagnostics.json').write_text(json.dumps(derived,indent=2)+'\n')
    return data,derived


def p(x):return f'{100*x:.1f}%'
def delta(v):
    lo,hi=v['delta_paired95ci'];return f"{v['delta']:+.3f} [{lo:+.3f}, {hi:+.3f}]"

def report(data,derived,path,title):
    conclusion=('**结论：margin 搜索主要把 donor 的键推到输出，没有稳定产生第三颜色。** Heldout top4 的第三颜色准确率为 0%，直接输出 donor 键为 71.9%。'
                if data['args'].get('objective','third_margin')=='third_margin' else
                '**结论：第三颜色的边际概率和部分准确率提高，但没有 donor 键特异性，不能声称组合回路成功。** 后续固定集合的[换键与颜色概率诊断](semantic_diagnostics/README.md)显示：改变 donor 的中间键时，模型并未相应区分两种目标颜色；主要现象是旧答案被压低，另外两色都上升。')
    lines=[f'# {title}','',conclusion,'',
           '这是构造性实验：donor 问“oak 对应哪个键”，应算出 C；recipient 已显式给出自己的旧键 B，继续查第二张表。移植 donor 最终查询位置的中间层 attention-head 输出，目标是让 recipient 改用 C 查自己的表。目标颜色不同于 recipient 原颜色与 donor 表中的最终颜色。它不解释模型原本已经具备的隐式两步能力。','',
           '冻结 DAG 1B step38160，含 V norm 和 local correction。只替换 o_proj 之前、最终查询 token 的选定 head 输出；不改输入、embedding、模型参数或 logits。完整上游与后续计算仍然运行。只搜索零基层号 2–13，排除最前和最后两层。','',
           f"Discovery32 的选择目标：`{data['args'].get('objective','third_margin')}`。先扫描 12 个完整 attention 层输出，再扫描前 4 层的每个 head，固定 top1/2/4/8/16。新表 heldout64 与新实体64；另外在 heldout 相同表上改变 donor 最终颜色、recipient 第二张表或 donor 查询实体。另有 implicit-query 诊断。",'',
           f"选择顺序前 8 个 (layer, head)：{data['selection']['ordered_sites'][:8]}。随机对照只匹配 top4 每层 head 数，不匹配编辑范数。",'',
           '## 全样本第三颜色准确率','',
           '| Split | 原 recipient | 输入改成新键的对照 | top1 | top2 | top4 | top8 | top16 |','|---|---:|---:|---:|---:|---:|---:|---:|']
    for stage in STAGES:
        arms=data['stages'][stage]['arms']
        score=lambda arm:p(arms[arm]['summary']['third_vocab_accuracy']['mean'])
        lines.append('| '+stage+' | '+' | '.join(score(arm) for arm in ['recipient','input_counterfactual',*BUDGETS])+' |')
    lines+=['','## Heldout 概率与输出键诊断','','| Arm | 第三颜色准确率 | 第三颜色 Δlogp [paired 95% CI] | 直接输出 donor 键 |','|---|---:|---:|---:|']
    for arm in ['recipient',*BUDGETS,'random4_0','random4_1','random4_2']:
        s=derived['subsets']['heldout']['all']['arms'][arm]
        lines.append(f"| {arm} | {p(s['third_vocab_accuracy']['mean'])} | {delta(s['third_logp'])} | {p(s['donor_key_vocab_accuracy']['mean'])} |")
    lines+=['','## 组件本来能完成的样本覆盖率','','“所需两个组件成功”要求 donor 本来答对中间键，且 recipient 在实际输入新键的对照中本来答对第三颜色。“三个条件都成功”还要求 recipient 本来答对旧颜色。所有条件都来自干预前结果，未按 patch 成功筛样本。','',
           '| Split | donor 键成功 | 所需两组件成功 | 三个条件成功 |','|---|---:|---:|---:|']
    for stage in STAGES:
        groups=derived['subsets'][stage]
        lines.append('| '+stage+' | '+' | '.join(f"{groups[k]['n']}/{groups[k]['total']}" for k in ['donor_key_correct','required_components_correct','all_three_correct'])+' |')
    lines+=['','| Heldout subset | n | 原第三色 | top1 | top2 | top4 | top8 | top16 |','|---|---:|---:|---:|---:|---:|---:|---:|']
    for label in ['donor_key_correct','required_components_correct','all_three_correct']:
        group=derived['subsets']['heldout'][label]
        score=lambda arm:'—' if group['arms'][arm]['third_vocab_accuracy'] is None else p(group['arms'][arm]['third_vocab_accuracy']['mean'])
        lines.append('| '+label+f" | {group['n']} | "+' | '.join(score(arm) for arm in ['recipient',*BUDGETS])+' |')
    lines+=['','## 同一例子是否随表和键变化','','下表要求同一个例子在原 heldout 和修改后的控制条件中都输出各自的第三颜色；不仅比较两个边际准确率。换 recipient 表或 donor 查询键时，目标颜色必须改变。','',
           '| Arm | 换 donor 最终色前后都正确 | 换 recipient 表前后都正确 | 换 donor 查询键前后都正确 | 四条件全部正确 |','|---|---:|---:|---:|---:|']
    for arm in ['recipient',*BUDGETS,'random4_0','random4_1','random4_2']:
        scores=[derived['paired_control_success'][c][arm]['mean'] for c in ['donor_color','recipient_table','donor_query']]+[derived['all_four_conditions_success'][arm]['mean']]
        lines.append('| '+arm+' | '+' | '.join(p(x) for x in scores)+' |')
    lines+=['','## 校验与解释范围','',
           f"Head hooks 相对同一末 token 投影路径的 logit 最大误差为 {data['unwrapped_max_logit_error']}；每个 split 的 same-input interchange 误差均为 0。全序列与仅末 token 的 BF16 lm_head GEMM 在单例诊断中最大差异为 {data['projection_diagnostic_one_case']['max_logit_error']}，argmax 一致为 {data['projection_diagnostic_one_case']['argmax_agreement']}。所有比较 arm 都使用完全相同的末 token 投影路径。",'',
           '每个分数是全词表 argmax 的第三颜色准确率；不是颜色候选内分数。logp 对完整词表归一化。bootstrap 4,000 次，以随机表为配对单位；置信区间未做多重比较调整。少量组件成功子集的均值只作诊断，不能外推为整体任务能力。记录全部预算与随机对照，不只挑最高的点。',
           '','逐例值与模型参数在 `results.json`，输入/目标与控制变换在 `datasets.json`。`qualified_subset_diagnostics.json` 含所有子集及 paired 控制。所有操作属于局部消息拼接，不能称完整、最小或自然语言语义回路。']
    (path/'README.md').write_text('\n'.join(lines)+'\n')

for sub,title in [('', '构造性单步拼接：margin 目标的诊断'),('logp_objective','构造性单步拼接：完整第三颜色概率目标')]:
    path=ROOT/sub
    data,derived=derive(path/'results.json')
    report(data,derived,path,title)
print('reports and qualified subsets complete')
