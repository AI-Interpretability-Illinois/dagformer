import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

root=Path(__file__).resolve().parent
order=['dense300m','dag300m','dense1b','dag1b']
labels=['Dense 300M','DAG 300M','Dense 1B','DAG 1B']
data={name:json.loads((root/name/'results.json').read_text()) for name in order}

def task(model,stage,name):
 return next(value for key,value in data[model]['stages'][stage]['summary'].items() if key.endswith('/'+name))
def pct(x):return f'{100*x:.1f}%'
def ci(x):return f"{pct(x['candidate_accuracy'])} [{100*x['candidate_accuracy_wilson95'][0]:.1f}, {100*x['candidate_accuracy_wilson95'][1]:.1f}]"

contrasts={}
for model in order:
 contrasts[model]={}
 for stage in ['heldout','entity_transfer','natural_fourshot','natural_fourshot_transfer']:
  rows=data[model]['stages'][stage]['rows']
  groups={task:{r['index']:r for r in rows if r['task']==task} for task in ['two_hop','supplied_middle']}
  assert groups['two_hop'].keys()==groups['supplied_middle'].keys()
  diff=np.array([groups['supplied_middle'][i]['candidate_correct']-groups['two_hop'][i]['candidate_correct'] for i in sorted(groups['two_hop'])])
  rng=np.random.default_rng(2026092999)
  samples=diff[rng.integers(len(diff),size=(4000,len(diff)))].mean(1)
  contrasts[model][stage]={'supplied_middle_minus_two_hop_accuracy':float(diff.mean()),'paired95ci':np.quantile(samples,[.025,.975]).tolist(),'n':len(diff)}
(root/'paired_diagnostics.json').write_text(json.dumps(contrasts,indent=2)+'\n')

lines=['# 第三答案组合回路：先验证模型是否会做两步查表','',
'**结果：四个冻结模型在本轮模板下，都没有稳定展示自动完成两步查表的能力；因此没有进入隐式中间状态的第三答案干预。** 这不是“模型没有组合回路”的证明，而是这项实验目前缺少可解释的目标行为。自然语言四示例提示的独立补充验证没有改变结论。','',
'直观任务：第一张表告诉我们“oak 对应 B”，第二张表告诉我们“B 是蓝色、C 是绿色”；问 oak 的颜色，正确答案是蓝色。设想从另一个例子取出内部已经算出的中间键 C，接入原来的第二张表，应输出绿色，即不同于两个原始答案的第三个答案。实验要求模型先能自行做这个两步任务；不能只替换输入里明写的 B，就声称找到了隐藏的组合步骤。','',
'## 主筛查','',
'每个模型：4 种格式（箭头、自然语言盒子、两张表、等式）× 0/2/4 个示例。每个配置使用 24 个 discovery 随机表。按 discovery 的两步候选准确率选择格式，target log-probability 打破并列。之后在 96 个独立随机表上验证，并在另外 96 个随机表和未在 discovery 出现的实体词上验证迁移。','',
'每张表有 3 行且行序独立随机；实体、映射、颜色和查询目标均随机。所有候选答案都是单 token。不同格式使用相同查询记录，示例前缀独立。每种任务问四类问题：第一步查中间键、第二步查颜色、直接两步查颜色，以及由实验者明确提供正确中间键之后查颜色。最后一项是诊断，不是自主两步推理。','',
'下表为 **3 候选内准确率**；随机选择是 33.3%。两步准确率后是 Wilson 95% 区间，单位均为百分比。全词表准确率单独列出，不与候选内分数混用。','',
'| 模型 | Discovery 选出的格式 | 第一步 | 第二步 | 两步 [95% CI] | 两步全词表 | 已提供中间键 |','|---|---|---:|---:|---:|---:|---:|']
for model,label in zip(order,labels):
 d=data[model]; fmt=d['selected_format']; t=task(model,'heldout','two_hop')
 lines.append(f"| {label} | {fmt['style']}, {fmt['shots']}-shot | {pct(task(model,'heldout','first_hop')['candidate_accuracy'])} | {pct(task(model,'heldout','second_hop')['candidate_accuracy'])} | {ci(t)} | {pct(t['vocab_accuracy'])} | {pct(task(model,'heldout','supplied_middle')['candidate_accuracy'])} |")
lines+=['','| 模型 | 新实体两步 [95% CI] | 新实体两步全词表 | 新实体已提供中间键 |','|---|---:|---:|---:|']
for model,label in zip(order,labels):
 t=task(model,'entity_transfer','two_hop')
 lines.append(f"| {label} | {ci(t)} | {pct(t['vocab_accuracy'])} | {pct(task(model,'entity_transfer','supplied_middle')['candidate_accuracy'])} |")
c=contrasts['dag1b']['heldout']
lines+=['',f"最清楚的诊断出现在 DAG 1B 的等式格式：直接两步为 {pct(task('dag1b','heldout','two_hop')['candidate_accuracy'])}，提供正确中间键后为 {pct(task('dag1b','heldout','supplied_middle')['candidate_accuracy'])}；同一批表上的差值是 {100*c['supplied_middle_minus_two_hop_accuracy']:.1f} 个百分点，paired bootstrap 95% CI [{100*c['paired95ci'][0]:.1f}, {100*c['paired95ci'][1]:.1f}]。提供中间键后的成功说明该提示可以利用给出的键查颜色，不说明它已经自行算出了这个键。",'',
'## 自然语言补充验证','',
'初筛完成后补充两个诊断：自然语言 boxes 模板按 discovery 选 shot 数，以及固定 4-shot。两者使用新的 seed、随机表和示例前缀；另外验证新实体。它们不替换主 heldout。下表列固定 4-shot 的独立结果：','',
'| 模型 | 两步 [95% CI] | 两步全词表 | 新实体两步 [95% CI] | 提供中间键 |','|---|---:|---:|---:|---:|']
for model,label in zip(order,labels):
 t=task(model,'natural_fourshot','two_hop'); tr=task(model,'natural_fourshot_transfer','two_hop')
 lines.append(f"| {label} | {ci(t)} | {pct(t['vocab_accuracy'])} | {ci(tr)} | {pct(task(model,'natural_fourshot','supplied_middle')['candidate_accuracy'])} |")
lines+=['','所有主筛查和固定自然语言 4-shot 两步分数的 95% 区间均包含 33.3% 随机水平。当前没有足够强的整体任务行为支撑寻找第三答案回路；没有通过筛选偶然正确的样本来宣称机制成功。','',
'## 尚未执行的因果阶段','',
'主筛查预设门槛是候选内两步准确率 ≥70%、两个单步各 ≥80%。门槛是为了保证可重复的目标行为，并非关于回路存在性的理论阈值。本轮两步成绩在随机水平附近，问题不仅是差一点没到门槛。四模型 `causal_status` 均为 `not_run_insufficient_whole_task_capability`。','',
'如果以后获得稳定行为，主干预应搜索最终查询位置的中间层 computed state，再要求输出跟接收者第二张表走。必须改变 donor 的最终颜色、保持 donor 中间键不变，并改变接收者第二张表验证目标随表而变。第一张表显式键 token 的内容替换只能当输入内容阳性对照；最终 logits 或答案表示的直接替换不符合这个目标。','',
'## 文件与复现','',
'每个模型子目录包含 `results.json`（全部提示、逐例预测、logits、汇总和参数）以及每阶段的 tokenized `*_inputs.json`。`paired_diagnostics.json` 保存相同表上的 paired bootstrap 对比。Wilson 区间以随机表为单位；paired bootstrap 使用 4,000 次相同表重采样。均为未调整的探索性区间，描述输入不确定性，不代表训练 seed 变化。','',
'模型全程冻结，无训练、optimizer 或参数更新。300M checkpoint 为 `checkpoints/pr_sync_20260917/{300m-baseline,300m-dagformer}`；1B 使用 step38160 的完整 checkpoint，经 eval 导出保存在 `/scratch/yurenh2/circuit-followup-checkpoints/{1b-baseline,1b-dagformer}`。1B 的训练语料与 300M 不同，且使用 V norm；本实验把它用于扩展能力筛查，不能据此作纯规模效应推断。','',
'```bash','CUDA_VISIBLE_DEVICES=3 /scratch/yurenh2/venvs/dagformer-eval-20260917/bin/python scripts/circuit_composition.py --kind dag --name dag300m',
'CUDA_VISIBLE_DEVICES=3 /scratch/yurenh2/venvs/dagformer-eval-20260917/bin/python scripts/circuit_composition.py --model checkpoints/pr_sync_20260917/300m-baseline --kind dense --name dense300m',
'CUDA_VISIBLE_DEVICES=3 /scratch/yurenh2/venvs/dagformer-eval-20260917/bin/python scripts/circuit_composition.py --model /scratch/yurenh2/circuit-followup-checkpoints/1b-dagformer --kind dag --name dag1b --batch-size 2',
'CUDA_VISIBLE_DEVICES=3 /scratch/yurenh2/venvs/dagformer-eval-20260917/bin/python scripts/circuit_composition.py --model /scratch/yurenh2/circuit-followup-checkpoints/1b-baseline --kind dense --name dense1b --batch-size 2','```','',
'上述命令会运行全部阶段，包括固定自然语言 4-shot。实际追加补充实验可用 `--natural-fourshot-only`，并从已有主结果读取。候选 logits 若出现 BF16 平手，统一选择较小 token id，与全词表 argmax 一致。记录的 git commit 是启动时的代码基线；最终脚本和报告随本轮结果一同提交。']
(root/'README.md').write_text('\n'.join(lines)+'\n')

fig,ax=plt.subplots(1,2,figsize=(11,4),layout='constrained')
x=np.arange(4)
for shift,stage,color,label in [(-.12,'heldout','#2878B5','Held-out'),(.12,'entity_transfer','#E07A28','New entities')]:
 vals=[task(m,stage,'two_hop') for m in order]
 y=np.array([v['candidate_accuracy'] for v in vals])*100
 lim=np.array([v['candidate_accuracy_wilson95'] for v in vals])*100
 ax[0].errorbar(x+shift,y,yerr=np.stack([y-lim[:,0],lim[:,1]-y]),fmt='o',capsize=3,color=color,label=label)
ax[0].axhline(100/3,color='#777',ls='--',lw=1,label='Chance (3 choices)')
ax[0].set_xticks(x,labels=['Dense\n300M','DAG\n300M','Dense\n1B','DAG\n1B']);ax[0].set_ylim(0,100);ax[0].set_ylabel('Candidate accuracy (%)');ax[0].set_title('Implicit two-step lookup');ax[0].legend(fontsize=8,loc='upper right')
names=['first_hop','second_hop','two_hop','supplied_middle'];vals=[task('dag1b','heldout',name) for name in names]
y=np.array([v['candidate_accuracy'] for v in vals])*100;lim=np.array([v['candidate_accuracy_wilson95'] for v in vals])*100
ax[1].bar(x,y,color=['#8299B1','#8299B1','#C46E68','#569C8C'],width=.6)
ax[1].errorbar(x,y,yerr=np.stack([y-lim[:,0],lim[:,1]-y]),fmt='none',ecolor='#333',capsize=3)
ax[1].axhline(100/3,color='#777',ls='--',lw=1);ax[1].set_xticks(x,labels=['First\nlookup','Second\nlookup','Implicit\ntwo-step','Given correct\nmiddle key']);ax[1].set_ylim(0,100);ax[1].set_title('DAG 1B, selected equals / 4-shot')
for axis in ax:axis.spines[['top','right']].set_visible(False)
fig.savefig(root/'capability_summary.png',dpi=180);fig.savefig(root/'capability_summary.pdf');plt.close(fig)
print('report complete')
