"""Frozen-checkpoint damage, conditional backups, and effective-routing clamps.

Only this script's process installs hooks; model weights are never updated.
Native source-read tests are DAG-only. Cross-model tests use backbone head output
gates in both models. Historical read positions are supplied by the copy task.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import csv
import gc
import json
from pathlib import Path
import time

import numpy as np
import torch

from circuit_message_patching import Interchange, arm, measures, make_data
from circuit_common_message_patching import load_data
from interp_common import load_elh, load_eval_ids


def spec(name, heads=(), edges=(), fixed=False, donor_content=False):
    return dict(name=name, heads=list(heads), edges=list(edges), fixed=fixed,
                donor_content=donor_content)


def pair_stats(values):
    x = np.asarray(values, dtype=float).reshape(-1, 2).mean(1)
    rng = np.random.default_rng(20260929)
    index = rng.integers(len(x), size=(4000, len(x)))
    return dict(mean=float(x.mean()), paired_95ci=np.quantile(x[index].mean(1), [.025, .975]).tolist())


class AuditedInterchange(Interchange):
    def einsum(self, equation, *operands, **kwargs):
        eq = equation.replace(' ', '') if isinstance(equation, str) else ''
        before = None
        if (eq == 'lbthd,bthl->bhtd' and self.qkv_calls % 3 == 2
                and self.spec is not None and self.spec['edges']):
            z, alpha = operands
            layer = z.shape[0] - 1
            edited_heads = sorted({h for l, s, h, src in self.spec['edges'] if l == layer and s == 'v'})
            if edited_heads:
                alpha = self.recipient[(layer, 'v')][1] if self.spec['fixed'] else alpha
                before = self.original(equation, z, alpha)
        result = super().einsum(equation, *operands, **kwargs)
        if before is not None:
            old = before[:, edited_heads][:, :, self.positions].float()
            new = result[:, edited_heads][:, :, self.positions].float()
            self.message_edit_sq += (new - old).square().sum((1, 2, 3))
        return result


class Runner:
    def __init__(self, kind, path):
        elh = load_elh()
        cfg = elh.load_config(str(path / 'config.yaml'))
        self.kind, self.device = kind, torch.device('cuda')
        if kind == 'dag':
            self.model, self.predictor = elh.load_fourway(str(path / 'checkpoint.pt'), cfg, self.device)
            self.model.use_triton_kernel = False
            assert not self.model.use_v_norm
            self.base = self.model.olmo
        else:
            self.model = elh.load_dense(str(path / 'checkpoint.pt'), cfg, self.device)
            self.predictor = None
            self.base = self.model
        self.L, self.H = cfg['num_hidden_layers'], cfg['num_attention_heads']
        self.gate = torch.ones(self.L, self.H, device=self.device)
        self.head_hits = set()
        self.patch = AuditedInterchange(self.model, self.predictor, self.L, self.H) if kind == 'dag' else None
        self.parameters = sum(p.numel() for p in self.model.parameters())
        if self.predictor is not None:
            self.parameters += sum(p.numel() for p in self.predictor.parameters())
        assert not any(p.requires_grad for p in self.model.parameters())

    @contextmanager
    def installed(self):
        handles = []
        for layer, block in enumerate(self.base.model.layers):
            def gate_hook(module, inputs, layer=layer):
                assert layer not in self.head_hits
                self.head_hits.add(layer)
                x = inputs[0]
                y = x.reshape(*x.shape[:-1], self.H, -1)
                return ((y * self.gate[layer].to(y.dtype).view(1, 1, self.H, 1)).reshape_as(x),)
            handles.append(block.self_attn.o_proj.register_forward_pre_hook(gate_hook))
        try:
            if self.patch is not None:
                with self.patch.installed():
                    yield
            else:
                handles.append(self.base.lm_head.register_forward_pre_hook(lambda m, x: (x[0][:, -1:],)))
                yield
        finally:
            for h in handles:
                h.remove()

    @torch.inference_mode()
    def forward(self, ids, positions, setting=None, clean=None, donor=None, zero=None, capture=False):
        self.gate.fill_(1)
        self.head_hits = set()
        if setting is not None:
            for layer, head in setting['heads']:
                self.gate[layer, head] = 0
        if self.patch is None:
            logits, cache = self.model(ids, use_cache=False).logits[:, -1].float(), None
            self.last_native_edit_l2 = [0.] * len(ids)
        else:
            self.patch.message_edit_sq = torch.zeros(len(ids), device=ids.device)
            native = None if setting is None else arm(setting['name'], mode='content',
                                                       edges=setting['edges'], streams='v', fixed=setting['fixed'])
            logits, cache = self.patch.forward(ids, positions, capture=capture, spec=native,
                                               donor=donor if setting and setting['donor_content'] else zero,
                                               recipient=clean)
            self.last_native_edit_l2 = self.patch.message_edit_sq.sqrt().cpu().tolist()
        assert len(self.head_hits) == self.L
        return logits, cache


def evaluate(runner, data, specifications, batch, output, stage):
    values = {s['name']: {} for s in specifications}
    values['clean'] = {}
    max_identity = 0.
    correction_change = []
    early_source_max_change = 0.
    recent_source_change = []
    for start in range(0, len(data['ids']), batch):
        ids = data['ids'][start:start + batch].cuda()
        a, b = data['a'][start:start + batch].cuda(), data['b'][start:start + batch].cuda()
        baseline, clean = runner.forward(ids, data['positions'], capture=True)
        swap = torch.arange(len(ids), device=ids.device) ^ 1
        donor = {k: (z[:, swap], alpha[swap]) for k, (z, alpha) in clean.items()} if clean is not None else None
        zero = {k: (torch.zeros_like(z), alpha) for k, (z, alpha) in clean.items()} if clean is not None else None
        def record(name, logits):
            item = measures(logits, a, b, baseline)
            item['target_nll'] = [-v for v in item['recipient_logp']]
            item['correct_margin'] = [-v for v in item['donor_margin']]
            item['native_V_read_intervention_l2'] = runner.last_native_edit_l2
            for key, vals in item.items():
                values[name].setdefault(key, []).extend(vals)
        record('clean', baseline)
        for s in specifications:
            capture = runner.kind == 'dag' and s['name'] == 'A_dynamic'
            logits, changed = runner.forward(ids, data['positions'], s, clean, donor, zero, capture=capture)
            if s['name'] == 'clean_fixed':
                max_identity = max(max_identity, float((logits - baseline).abs().max()))
            if capture:
                numerator = torch.zeros(len(ids), device=ids.device)
                denominator = torch.zeros_like(numerator)
                for key in clean:
                    old, new = clean[key][1].float(), changed[key][1].float()
                    numerator += (old - new).square().flatten(1).sum(1)
                    denominator += old.square().flatten(1).sum(1)
                correction_change.extend((numerator / denominator.clamp_min(1.e-20)).sqrt().cpu().tolist())
                key = (s['heads'][0][0] + 1, 'v')
                old_z, new_z = clean[key][0].float(), changed[key][0].float()
                early_source_max_change = max(early_source_max_change, float((old_z[:3] - new_z[:3]).abs().max()))
                numerator = (old_z[-3:] - new_z[-3:]).square().sum((0, 2, 3, 4))
                denominator = old_z[-3:].square().sum((0, 2, 3, 4))
                recent_source_change.extend((numerator / denominator.clamp_min(1.e-20)).sqrt().cpu().tolist())
            record(s['name'], logits)
        if start == 0 or (start + len(ids)) % 32 == 0:
            print(stage, start + len(ids), '/', len(data['ids']), flush=True)
    assert max_identity == 0, max_identity
    result = dict(n_pairs=len(data['ids']) // 2, period=data['period'], query=data['query'],
                  seed=data['seed'], positions=data['positions'], specs=specifications,
                  all_head_hooks_hit=True, clean_fixed_identity_max_logit_error=max_identity,
                  effective_alpha_relative_change_after_A=pair_stats(correction_change) if correction_change else None,
                  early_source_projected_content_max_abs_change_after_A=early_source_max_change if correction_change else None,
                  recent_source_projected_content_relative_change_after_A=pair_stats(recent_source_change) if recent_source_change else None,
                  arms={name: dict(values=v, summary={k: pair_stats(x) for k, x in v.items()}) for name, v in values.items()})
    output[stage] = result
    return result


def mean_nll(stage, name):
    return stage['arms'][name]['summary']['target_nll']['mean']


def interaction(stage, a, b, ab, clean='clean'):
    arms = stage['arms']
    n = lambda name: np.asarray(arms[name]['values']['target_nll'])
    return pair_stats(n(ab) - n(a) - n(b) + n(clean))


def add_contrasts(stage, backup_names, dag):
    contrasts = {}
    for label, b, ab, a in backup_names:
        if all(x in stage['arms'] for x in (a, b, ab)):
            contrasts[label] = interaction(stage, a, b, ab)
    if dag:
        for base in ['A', 'head_B', 'head_AB', 'early_B', 'early_AB']:
            dynamic, fixed = base + '_dynamic', base + '_fixed'
            if dynamic in stage['arms'] and fixed in stage['arms']:
                x = np.asarray(stage['arms'][dynamic]['values']['target_nll'])
                y = np.asarray(stage['arms'][fixed]['values']['target_nll'])
                contrasts[base + '_fixed_minus_dynamic_NLL'] = pair_stats(y - x)
    stage['causal_contrasts'] = contrasts
    nll = lambda name: np.asarray(stage['arms'][name]['values']['target_nll'])
    h = nll('head_AB_dynamic') - nll('head_B_dynamic')
    for i in range(3):
        control = nll(f'head_random{i}_AB') - nll(f'head_random{i}_B')
        contrasts[f'head_interaction_minus_random{i}'] = pair_stats(h - control)
    if dag:
        h = nll('early_AB_fixed') - nll('early_B_fixed')
        for name in ['recent'] + [f'early_random{i}' for i in range(3)]:
            control = nll(name + '_AB') - nll(name + '_B')
            contrasts['early_interaction_minus_' + name] = pair_stats(h - control)


def write_report(result, path):
    lines = ['# Damage, conditional backups, and routing response', '',
             'Frozen 300M checkpoints; no finetuning. Head lesions zero attention-head outputs before o_proj at all positions. Native DAG lesions remove selected V source-read contributions only at the three known historical value-token locations.', '',
             'A is the first non-layer0 V head from yesterday\'s discovery-only common-message ranking. Candidate backup heads are all 16 heads in the next layer. DAG source-read candidates are the next-layer heads reading sources 0/1/2 (embedding / layer0 output / layer1 output), which precede A. Selection maximizes conditional NLL interaction on discovery only, under clean effective-route clamps in DAG.', '',
             'Clean clamps preserve the complete per-input effective alpha (global predictor plus local correction), at every layer/stream/position. They preserve input dependence and prevent effective routing from responding to the damage. The external global predictor reads only input IDs, so its coefficients cannot change from an internal lesion with the same input and frozen weights. Attention probabilities can still change.', '',
             'Interaction = NLL(A+B) - NLL(A) - NLL(B) + NLL(clean). Positive interaction is additional joint damage; it is not sufficient by itself to prove a unique backup computation. Recent-source controls use the same destination head and three most recent sources; random controls preserve destination layer and number of heads/source reads. Controls are not matched for intervention norm.', '',
             'The DAG native source-read experiments are not compared numerically with dense head counts. Head-pair interventions use the same head-output operator in both models, but model parameters and selected head locations differ. All positions are supplied by task construction. This is synthetic copying, not general semantic binding or pretrained pruning recovery.', '']
    lines += [f"Evaluation provenance: {result.get('evaluation_provenance', 'Reuses the previous message-patching inputs; exploratory, not a new independent confirmation sample.')}", '']
    dag = result['models'].get('dag', {})
    stages = dag.get('stages', {})
    if all(k in stages for k in ['heldout', 'transfer', 'unchanged_query']):
        held, transfer, control = (stages[k] for k in ['heldout', 'transfer', 'unchanged_query'])
        def effect_text(stage, name):
            e = stage['causal_contrasts'][name]
            return f"{e['mean']:+.3f} [{e['paired_95ci'][0]:+.3f}, {e['paired_95ci'][1]:+.3f}]"
        lines += ['## 中文关键结论', '',
                  '**已有早层读取与即时 local correction 响应同时贡献损伤耐受。此次没有训练，因此不能据此解释 finetune 后的恢复。**', '',
                  f"1. 删除 L6/H11 后，正常 correction 比钳住该输入原始有效路由的 NLL 更低。固定减正常的差：heldout {effect_text(held, 'A_fixed_minus_dynamic_NLL')}，距离迁移 {effect_text(transfer, 'A_fixed_minus_dynamic_NLL')}。global predictor 的输入和参数均未变，不能称其感知损伤后改道。", '',
                  f"2. L7/H1 从 embedding、layer0、layer1 读取的三条 V 来源在损伤前后投影内容完全一致，实测 max abs diff=0。在有效路由固定时，再切断这些已有读取的条件交互为 heldout {effect_text(held, 'early_interaction_fixed')}，迁移 {effect_text(transfer, 'early_interaction_fixed')}。这支持未受损早层信息经已有跨层读取继续影响损伤后的输出；不能称产生了新连接。", '',
                  f"3. 改问另一未变位置后，early-read 条件交互为 {effect_text(control, 'early_interaction_fixed')}，接近零；完整 head 损伤则仍影响这个复制目标。定位到历史值读取的干预比笼统删 head 更有行为选择性。", '',
                  '4. 同 head 的 recent-source 对照作用很小，但编辑幅度不匹配：early-source 删除 L2 约为 recent 的 27 倍，不能据此声称早来源在等强扰动下具有特殊优势。多个随机目的 head 的早层读取也有正交互，因此不是唯一备用通路。原始 L2 和 paired contrasts 均已保存。', '',
                  '5. Dense 同样存在正的 head 双损伤交互。两模型的初始能力、主要损伤位置、参数量，以及 DAG 的 clamp 条件不同，此处不把交互大小或原始 NLL 解释成 DAG 总体更鲁棒。搜索只覆盖 A 的下一层 16 个 head。', '',
                  '6. 只替换 L7/H1 三条 early-read 的 donor 内容会降低正确答案 margin，但 donor 成为全词表首选的比例仅约 0.8%；这不是完整答案移植。', '',
                  '## 主图建议数值', '',
                  'NLL / 正确答案 margin / 全词表正确率。所有数值来自 fresh confirmation；误差条及逐样本数值见 main_figure_data.csv 与 results.json。headAB 是两个完整 head 损伤；earlyAB 是 A 加三条历史 V 来源读取切断，二者不是等规模干预。', '',
                  '| DAG arm | heldout NLL / margin / acc | transfer NLL / margin / acc |', '|---|---:|---:|']
        main_arms = ['clean', 'A_dynamic', 'A_fixed', 'head_AB_dynamic', 'head_AB_fixed', 'early_AB_dynamic', 'early_AB_fixed']
        for name in main_arms:
            cells = []
            for stage in [held, transfer]:
                s = stage['arms'][name]['summary']
                cells.append(f"{s['target_nll']['mean']:.4f} / {s['correct_margin']['mean']:.3f} / {100*s['recipient_vocab_accuracy']['mean']:.2f}%")
            lines.append('| ' + name + ' | ' + ' | '.join(cells) + ' |')
        lines += ['', '| Read intervention | heldout L2 | transfer L2 |', '|---|---:|---:|']
        for name in ['early_B_fixed', 'early_AB_fixed', 'recent_B', 'recent_AB', 'early_donor_AB']:
            vals = [stage['arms'][name]['summary'].get('native_V_read_intervention_l2', {}).get('mean') for stage in [held, transfer]]
            if all(v is not None for v in vals):
                lines.append(f'| {name} | {vals[0]:.3f} | {vals[1]:.3f} |')
        lines += ['']
    for kind, model in result['models'].items():
        lines += [f'## {kind}', '', f"Selection: {model.get('selection')}", '']
        for split in ['heldout', 'transfer', 'unchanged_query']:
            if split not in model['stages']:
                continue
            stage = model['stages'][split]
            lines += [f'### {split}: {stage["n_pairs"]} independent pairs', '',
                      '| Arm | Correct NLL | Correct full-vocabulary accuracy | Donor full-vocabulary accuracy |',
                      '|---|---:|---:|---:|']
            for name, item in stage['arms'].items():
                s = item['summary']
                lines.append(f"| {name} | {s['target_nll']['mean']:.4f} | {100*s['recipient_vocab_accuracy']['mean']:.2f}% | {100*s['donor_vocab_accuracy']['mean']:.2f}% |")
            lines += ['', '| Contrast | NLL effect [paired 95% CI] |', '|---|---:|']
            for name, estimate in stage.get('causal_contrasts', {}).items():
                lo, hi = estimate['paired_95ci']
                lines.append(f"| {name} | {estimate['mean']:+.4f} [{lo:+.4f}, {hi:+.4f}] |")
            lines += ['', f"Effective-alpha relative change after A: {stage['effective_alpha_relative_change_after_A']}", '']
    lines += ['All directed-case values and discovery scans are in results.json. Pair bootstrap uses 4,000 resamples, grouping exchange directions; intervals are unadjusted and describe examples rather than checkpoint seeds. No weights were updated.']
    (path / 'README.md').write_text('\n'.join(lines) + '\n')
    figure_rows, l2_rows = [], []
    keep = {'clean', 'A_dynamic', 'A_fixed', 'head_B_dynamic', 'head_AB_dynamic', 'head_AB_fixed',
            'early_B_fixed', 'early_AB_dynamic', 'early_AB_fixed', 'recent_B', 'recent_AB'}
    for kind, model in result['models'].items():
        for split in ['heldout', 'transfer', 'unchanged_query']:
            if split not in model['stages']:
                continue
            for name, arm_result in model['stages'][split]['arms'].items():
                row = dict(model=kind, split=split, arm=name)
                for metric in ['target_nll', 'correct_margin', 'recipient_vocab_accuracy', 'native_V_read_intervention_l2']:
                    estimate = arm_result['summary'].get(metric)
                    if estimate is not None:
                        row[metric] = estimate['mean']
                        row[metric + '_lo'], row[metric + '_hi'] = estimate['paired_95ci']
                if name in keep:
                    figure_rows.append(row)
                if row.get('native_V_read_intervention_l2', 0) > 0:
                    l2_rows.append(row)
    for name, rows in [('main_figure_data.csv', figure_rows), ('read_edit_l2.csv', l2_rows)]:
        if rows:
            with (path / name).open('w', newline='') as stream:
                writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator='\n')
                writer.writeheader()
                writer.writerows(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out', type=Path, default=Path('experiments/results/circuit_followup_20260929/damage_mechanism'))
    ap.add_argument('--data', type=Path, default=Path('experiments/results/circuit_interpretability_20260928/message_patching/datasets.json'))
    ap.add_argument('--prior', type=Path, default=Path('experiments/results/circuit_interpretability_20260928/common_message_patching/results.json'))
    ap.add_argument('--root', type=Path, default=Path('checkpoints/pr_sync_20260917'))
    ap.add_argument('--kinds', nargs='+', default=['dag', 'dense'], choices=['dag', 'dense'])
    ap.add_argument('--batch-size', type=int, default=8)
    ap.add_argument('--fresh-eval-seed', type=int)
    args = ap.parse_args()
    assert args.batch_size % 2 == 0
    torch.set_num_threads(4)
    torch.manual_seed(20260929)
    data, prior = load_data(args.data), json.loads(args.prior.read_text())
    if args.fresh_eval_seed is not None:
        corpus, _ = load_eval_ids(str(args.root / 'eval_corpora/wikitext_train.pt'))
        data['heldout'] = make_data(corpus, 64, 64, args.fresh_eval_seed)
        data['transfer'] = make_data(corpus, 64, 128, args.fresh_eval_seed + 1)
        data['unchanged_query'] = make_data(corpus, 64, 64, args.fresh_eval_seed, query=15)
    args.out.mkdir(parents=True, exist_ok=True)
    result = dict(complete=False, started=time.time(), args={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                  models={}, protocol='No training; discovery-only conditional head/source-read scans; per-input effective-alpha clamps; causal interaction heldout tests.')
    result['evaluation_provenance'] = (f"Fresh evaluation seed {args.fresh_eval_seed}; discovery and all selection rules unchanged from the prior inputs. Reused-input exploratory results are separate."
                                       if args.fresh_eval_seed is not None else 'Reuses previous message-patching evaluation inputs; exploratory, not a fresh confirmation sample.')
    saved_data = {k: {field: (value.tolist() if isinstance(value, torch.Tensor) else value) for field, value in item.items()}
                  for k, item in data.items()}
    (args.out / 'datasets.json').write_text(json.dumps(saved_data) + '\n')
    def save():
        result['elapsed_seconds'] = time.time() - result['started']
        (args.out / 'results.json').write_text(json.dumps(result, indent=2) + '\n')
        write_report(result, args.out)
    for kind in args.kinds:
        tag = 'dagformer' if kind == 'dag' else 'baseline'
        runner = Runner(kind, args.root / ('300m-' + tag))
        A = next((l, h) for l, s, h in prior['models'][kind]['selection']['ordered_sites'] if s == 'v' and l > 0)
        layer = A[0] + 1
        assert 2 < layer < runner.L
        model = dict(parameters=runner.parameters, A=A, stages={}, selection={})
        result['models'][kind] = model
        fixed = kind == 'dag'
        with torch.inference_mode():
            check_ids = data['discovery']['ids'][:2].cuda()
            unwrapped = (runner.model(check_ids, runner.predictor(check_ids)) if fixed else
                         runner.model(check_ids, use_cache=False).logits)[:, -1].float()
        with runner.installed():
            wrapped, _ = runner.forward(check_ids, data['discovery']['positions'])
            model['unwrapped_parity_max_logit_error'] = float((unwrapped - wrapped).abs().max())
            assert model['unwrapped_parity_max_logit_error'] == 0
            del check_ids, unwrapped, wrapped
            discovery_specs = [spec('A_dynamic', [A])]
            if fixed:
                discovery_specs += [spec('clean_fixed', fixed=True), spec('A_fixed', [A], fixed=True)]
            for h in range(runner.H):
                discovery_specs += [spec(f'head_B/{h}', [(layer, h)], fixed=fixed),
                                    spec(f'head_AB/{h}', [A, (layer, h)], fixed=fixed)]
                if fixed:
                    edges = [(layer, 'v', h, src) for src in (0, 1, 2)]
                    discovery_specs += [spec(f'early_B/{h}', edges=edges, fixed=True),
                                        spec(f'early_AB/{h}', [A], edges, fixed=True)]
            stage = evaluate(runner, data['discovery'], discovery_specs, args.batch_size, model['stages'], 'discovery')
            aname = 'A_fixed' if fixed else 'A_dynamic'
            head_scores = [interaction(stage, aname, f'head_B/{h}', f'head_AB/{h}')['mean'] for h in range(runner.H)]
            bh = int(np.argmax(head_scores))
            model['selection'] = dict(A=A, backup_head=(layer, bh), head_conditional_NLL_scores=head_scores,
                                      discovery_pairs=stage['n_pairs'], selection_under_fixed_effective_alpha=fixed)
            final = [spec('A_dynamic', [A]), spec('head_B_dynamic', [(layer, bh)]), spec('head_AB_dynamic', [A, (layer, bh)])]
            contrasts = [('head_interaction_dynamic', 'head_B_dynamic', 'head_AB_dynamic', 'A_dynamic')]
            rng = np.random.default_rng(20260929)
            control_heads = rng.choice([h for h in range(runner.H) if h != bh], 3, replace=False).tolist()
            model['selection']['control_heads'] = control_heads
            for i, h in enumerate(control_heads):
                final += [spec(f'head_random{i}_B', [(layer, h)]), spec(f'head_random{i}_AB', [A, (layer, h)])]
                contrasts.append((f'head_random{i}_interaction', f'head_random{i}_B', f'head_random{i}_AB', 'A_dynamic'))
            if fixed:
                scores = [interaction(stage, 'A_fixed', f'early_B/{h}', f'early_AB/{h}')['mean'] for h in range(runner.H)]
                eh = int(np.argmax(scores))
                model['selection'].update(early_read_head=(layer, eh), early_sources=[0, 1, 2], early_conditional_NLL_scores=scores)
                early = [(layer, 'v', eh, src) for src in (0, 1, 2)]
                final += [spec('clean_fixed', fixed=True), spec('A_fixed', [A], fixed=True),
                          spec('head_B_fixed', [(layer, bh)], fixed=True), spec('head_AB_fixed', [A, (layer, bh)], fixed=True)]
                contrasts.append(('head_interaction_fixed', 'head_B_fixed', 'head_AB_fixed', 'A_fixed'))
                for fixed_mode in (False, True):
                    suffix = 'fixed' if fixed_mode else 'dynamic'
                    final += [spec('early_B_' + suffix, edges=early, fixed=fixed_mode),
                              spec('early_AB_' + suffix, [A], early, fixed=fixed_mode)]
                    contrasts.append(('early_interaction_' + suffix, 'early_B_' + suffix, 'early_AB_' + suffix, 'A_' + suffix))
                late = [(layer, 'v', eh, src) for src in (layer - 2, layer - 1, layer)]
                groups = [('recent', late)]
                controls = rng.choice([h for h in range(runner.H) if h != eh], 3, replace=False).tolist()
                model['selection']['early_control_heads'] = controls
                for i, h in enumerate(controls):
                    groups.append((f'early_random{i}', [(layer, 'v', h, src) for src in (0, 1, 2)]))
                for name, edges in groups:
                    final += [spec(name + '_B', edges=edges, fixed=True), spec(name + '_AB', [A], edges, fixed=True)]
                    contrasts.append((name + '_interaction_fixed', name + '_B', name + '_AB', 'A_fixed'))
                final += [spec('early_donor_B', edges=early, fixed=True, donor_content=True),
                          spec('early_donor_AB', [A], early, fixed=True, donor_content=True)]
            save()
            for split in ['heldout', 'transfer', 'unchanged_query']:
                stage = evaluate(runner, data[split], final, args.batch_size, model['stages'], split)
                add_contrasts(stage, contrasts, fixed)
                save()
                print('RESULT', kind, split, json.dumps(stage['causal_contrasts']), flush=True)
        del runner
        gc.collect()
        torch.cuda.empty_cache()
    result['complete'] = True
    save()
    print('COMPLETE', args.out, result['elapsed_seconds'], flush=True)


if __name__ == '__main__':
    main()
