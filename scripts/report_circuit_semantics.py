"""Build compact tables from completed semantic-variable experiments."""
from pathlib import Path
import argparse
import csv
import json

from circuit_semantics import OUT, STATES, VERBS


def load(path):
    return json.loads(path.read_text())


def pct(value):
    return f'{100*value:.2f}%'


def interval(record, percent=True):
    f = 100 if percent else 1
    suffix = '%' if percent else ''
    return f"{f*record['mean']:.2f}{suffix} [{f*record['ci95'][0]:.2f}, {f*record['ci95'][1]:.2f}]"


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,default=OUT)
    root=parser.parse_args().out
    variable={s:load(root/'variable'/f'{s}_summary.json') for s in ['heldout','transfer','role']}
    trace={s:load(root/'trace'/f'{s}_summary.json') for s in variable}
    attention=load(root/'trace/attention_summary.json')
    repair=load(root/'read_repair/summary.json')
    repair_protocol=load(root/'read_repair/protocol.json')
    readouts=load(root/'variable/readout_summary.json')
    rows=[]
    for split,states in variable.items():
        for state,record in states.items():
            for pair in VERBS:
                verb='/'.join(pair)
                for name,metrics in [('baseline',record['baseline'][verb])]+[(name,a[verb]) for name,a in record['arms'].items()]:
                    for metric,values in metrics.items():
                        if isinstance(values,dict) and 'mean' in values:
                            rows.append([split,state,name,verb,metric,values['mean'],*values['ci95']])
    with (root/'variable_metrics.csv').open('w',newline='') as f:
        w=csv.writer(f,lineterminator='\n')
        w.writerow(['split','state','intervention','verb_pair','metric','mean','ci95_low','ci95_high']);w.writerows(rows)
    text=['# 数值表','',
          '2026-09-30。由已完成结果自动生成；候选词比较准确率，不是开放生成准确率。区间按八例反事实 block bootstrap。',
          '', '## 一维信息交换：原模型','',
          '| 测试集 | 原始语法 | 均值差方向：反事实吻合 | 学习方向：反事实吻合 | 学习方向：数信息效应 | 同数换词：绝对效应 |',
          '|---|---:|---:|---:|---:|---:|']
    for split,states in variable.items():
        r=states['original'];v='is/are';a=r['arms']
        text.append('| '+split+' | '+' | '.join([
            interval(r['baseline'][v]['grammar']),interval(a['subject/mean/subject'][v]['counterfactual_grammar']),
            interval(a['subject/learned/subject'][v]['counterfactual_grammar']),
            interval(a['subject/learned/subject'][v]['signed_effect'],False),
            interval(a['subject/learned/lexical'][v]['absolute_effect'],False)])+' |')
    text += ['', '## 一维信息交换：所有状态与动词','',
             '| 测试集 | 状态 | 动词对 | 原始语法 | 学习方向：反事实语法吻合 | 有符号效应 |',
             '|---|---|---|---:|---:|---:|']
    for split,states in variable.items():
        for state,r in states.items():
            for pair in VERBS:
                v='/'.join(pair);a=r['arms']['subject/learned/subject'][v]
                text.append(f"| {split} | {state} | {v} | {pct(r['baseline'][v]['grammar']['mean'])} | {pct(a['counterfactual_grammar']['mean'])} | {a['signed_effect']['mean']:.3f} |")
    text += ['', 'adapted 的训练目标是错误语法，表中的“反事实语法吻合”仍按正确语法记分，不能当作对它原生反向规则的忠实度。其负效应表示与原模型相反的使用方向；role 上这个符号又变为正。',
             '', '## 读取通道：信息互换效应的中介比例','',
             '| 测试集 | 状态 | L6H5 | 三个 head 联合 | 三个 head 仅最终位置 | 随机三 head，三次均值 |',
             '|---|---|---:|---:|---:|---:|']
    for split,states in trace.items():
        for state,r in states.items():
            f=lambda name:r[name]['is/are']['mediated_fraction']
            random_mean=sum(f(f'random{i}/all')['mean'] for i in range(3))/3
            text.append(f"| {split} | {state} | {interval(f('reset/L6H5'))} | {interval(f('top3/all'))} | {interval(f('top3/last'))} | {pct(random_mean)} |")
    text += ['', '比例衡量恢复指定 head 输出后，移植数信息造成的 logit 差变化被消去多少，不是删除 head 后的语法准确率；各 head 比例不可直接相加。',
             '', '## L6H5 的主语注意力与源端可读信息','',
             '| 测试集 | 状态 | 主语 attention | 源端主语数：自身探针 | 源端主语数：固定原模型探针 |',
             '|---|---|---:|---:|---:|']
    for split,states in attention.items():
        for state,r in states.items():
            q=readouts[split][state]['subject']
            text.append(f"| {split} | {state} | {interval(r['L6H5']['subject'])} | {pct(q['own_readout']['subject']['mean'])} | {pct(q['original_readout']['subject']['mean'])} |")
    text += ['', '## 新句子上的单 head 路由恢复','',
             '| 测试集 | 恢复范围 | predictor 坐标数 | is/are 语法 | 主语 attention |',
             '|---|---|---:|---:|---:|']
    for split,arms in repair.items():
        for name,r in arms.items():
            count=repair_protocol['restored_coordinates'][name]
            text.append(f"| {split} | {name} | {count} | {interval(r['grammar']['is/are'])} | {interval(r['attention']['subject'])} |")
    text += ['', '这批每个 split 256 例，与前面定位及评估的全部 prompt 不重合。Q/K/V 恢复只涉及外置 predictor 输出，local correction 保持正常动态计算。', '']
    (root/'TABLES.md').write_text('\n'.join(text))
    compact=dict(source_edge=load(root/'selection.json')['selected']['edge']['edges'][0],
                 downstream_heads=load(root/'trace/selection.json')['chosen'],
                 variable={split:{state:dict(baseline=r['baseline']['is/are']['grammar'],
                    learned=r['arms']['subject/learned/subject']['is/are'],
                    mean=r['arms']['subject/mean/subject']['is/are'],
                    lexical=r['arms']['subject/learned/lexical']['is/are']) for state,r in states.items()}
                    for split,states in variable.items()},
                 mediation={split:{state:{name:arms[name]['is/are'] for name in ['reset/L6H5','top3/all','top3/last']}
                    for state,arms in states.items()} for split,states in trace.items()},
                 attention=attention,read_repair=repair)
    (root/'overview.json').write_text(json.dumps(compact,indent=2)+'\n')
    print('Wrote TABLES.md, variable_metrics.csv, overview.json')


if __name__=='__main__':main()
