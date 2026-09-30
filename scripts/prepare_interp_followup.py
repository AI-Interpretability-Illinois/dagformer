"""Create the locked additional confirmation sets for adapter follow-up."""
import gzip
import json
from pathlib import Path
from transformers import AutoTokenizer
import interp_audit_common as common

ROOT=common.ROOT/'experiments/results/interp_followup_20260929'
ROOT.mkdir(parents=True,exist_ok=True)
path=ROOT/'extra_tests.json.gz'
if path.exists():raise SystemExit('Additional tests already exist; refusing to replace them.')
tok=AutoTokenizer.from_pretrained(str(common.TOKENIZER),local_files_only=True)
data=common.prepare_data();seen={x['prompt'] for rows in data.values() for x in rows if x['task']!='retain'}
extra={}
for split,seed,transfer in [('confirmation',2026100501,False),('hard_transfer',2026100502,True)]:
    if transfer:
        common.SVA_TRANSFER=[
          'The {s}, according to the {d},',
          'The {s} that the {d} could quietly hear',
          'The {s} who spoke to the {d} earlier today',
          'The {s} from the village beside the {d}',
          'The {s} whom the {d} had decided to invite',
          'The {s}, despite the complaints from the {d},']
        common.IOI_TRANSFER=[
          '{a} and {b} visited the {place} together. After a long conversation, {actor} gave a {obj} to',
          'At the {place}, {a} chatted with {b} for a while. Before leaving, {actor} handed the {obj} to',
          'When {a} and {b} finished talking at the {place}, {actor} decided to offer a {obj} to',
          'Earlier today, {a} and {b} were waiting at the {place}. During the wait, {actor} passed a {obj} to',
          '{a} went to the {place} with {b}. When they finally sat down, {actor} lent the {obj} to',
          'At the busy {place}, {a} and {b} stood near the door. Then {actor} brought a {obj} to']
    rows=common.make_tasks(tok,320 if not transfer else 192,seed,transfer)
    excluded={x['group'] for x in rows if x['prompt'] in seen}
    rows=[x for x in rows if x['group'] not in excluded]
    selected=[]
    for task in ['sva','ioi']:
        subset=[x for x in rows if x['task']==task][:256]
        assert len(subset)==256,(split,task,len(subset))
        selected.extend(subset)
    seen.update(x['prompt'] for x in selected)
    selected+=common.natural_items(tok,'test',128,seed+50,96 if transfer else 48,partition=int(transfer))
    extra[split]=selected
with gzip.open(path,'wt') as f:json.dump(extra,f)
common.write_json(ROOT/'protocol.json',dict(
    created_before_followup_confirmation_evaluation=True,
    hypothesis='Selective removal of learned grammatical disagreement while retaining learned recipient identification.',
    seeds=[20260930,20261001,20261002],
    pretrained_models='Same complete 300m-dagformer step9000 checkpoint and full forward as first audit. Three adapter seeds, not three pretraining seeds.',
    methods=dict(full_predictor='Replicate original 30.39M-parameter adaptation; functional output-coordinate rollback.',
        output_predictor='Freeze predictor embedding/encoder/trunk; train all 3773 output rows and biases (1,935,549 parameters). Revert rows for an exact single-predictor export.',
        lora='Rank 7 on all attention/MLP projections (1,978,368 parameters). Audit output-row rollback by zeroing B rows; A stays shared.'),
    training='600 steps; same full-vocabulary CE and 0.4/0.4/0.2 task/retain weights. Full predictor lr3e-5; output predictor choose 3e-4 or1e-3 on original validation only at first seed, then lock; LoRA lr3e-4.',
    localization='Original discovery only. Same gradient and penalty selection as first audit; no masks selected on either test set.',
    primary_budget='2% of trained adapter parameters reset for output predictor and LoRA B-row methods. Also 0.5%,1%,5%,10%; full predictor retains its old native-coordinate budgets.',
    primary_test='New confirmation and hard_transfer prompts, 128 counterfactual pairs per task per split. Existing heldout/transfer also retained. Hard transfer has unseen names/nouns plus new, longer templates.',
    retain='128 test-cache windows per additional split; partition by cache-chunk parity, not original articles.',
    deployment='Compare materialized parameter rollback with functional gating on identical batches. Report single predictor inference and compact adapter size.',
    controls='Within-layer/stream or projection-module matched random rollback, plus unfocused global shrinking of adapter updates. Report all seeds and budgets, including failures.',
    counts={s:{t:sum(x['task']==t for x in rows) for t in ['sva','ioi','retain']} for s,rows in extra.items()}))
print(json.dumps({s:len(x) for s,x in extra.items()}))
