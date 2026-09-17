# 复跑主要比较

以下命令从仓库根目录运行。在 timan1，可激活本轮环境：

```bash
source /scratch/yurenh2/venvs/dagformer-eval-20260917/bin/activate
DAG_EVAL_MODELS=checkpoints/pr_sync_20260917
DAG_EVAL_OUT=experiments/results/eval_repeat
DAG_EVAL_TASKS=wikitext,lambada_openai,hellaswag,piqa,arc_easy,arc_challenge,winogrande,openbookqa,sciq,boolq,mathqa,commonsense_qa,social_iqa,gsm8k_bpb
```

将 `CUDA_VISIBLE_DEVICES` 设为分配给此次运行的 GPU。本轮使用 timan1 的 2、3。
环境为 PyTorch 2.10.0+cu128、Transformers 4.57.1、lm_eval 0.4.13。
Delta 工作副本包含同一组输入；在 Delta 运行时选择安装了这些依赖的 Python。
下面的输出写到 `eval_repeat`，原始验收结果保留在 `eval_20260917`。

`DAG_EVAL_MODELS` 指向本轮导出的 checkpoint，300M baseline 和 DAGFormer
均为 step 9000。旧共享 bundle 的默认 baseline 是 step 12000，因此这里显式
传入模型目录和 tokenizer。各条 `run_eval.py` 命令可加 `--dry-run` 查看任务与模型。

## 300M 普通评估和成对区间

```bash
python experiments/results/lmeval/run_eval.py \
  --models-root "$DAG_EVAL_MODELS" --model 300m \
  --tokenizer "$DAG_EVAL_MODELS/tokenizer" \
  --tasks "$DAG_EVAL_TASKS" --batch-size 8 --max-length 1024 \
  --fewshot-seed 1234 --softmax-dtype float32 \
  --save-samples --save-likelihood-samples \
  --out-dir "$DAG_EVAL_OUT/standard_matched"

python scripts/summarize_paired_eval.py \
  --dir "$DAG_EVAL_OUT/standard_matched" --draws 10000 --seed 20260917
```

`--model 300m` 展开为该档位的两个模型。75M/150M 可用相同入口；
各次已完成运行的 batch size、任务、checkpoint step 等设置保存在对应 JSON
的 `model` 和 `eval` 字段。成对区间需要保存逐题 likelihood 样本。

## GSM8K 全量生成

```bash
python experiments/results/lmeval/run_eval.py \
  --models-root "$DAG_EVAL_MODELS" --model 300m \
  --tokenizer "$DAG_EVAL_MODELS/tokenizer" --suite gen \
  --gen-limit 1319 --gen-batch-size 1 --gen-num-fewshot 3 \
  --max-length 1024 --max-gen-toks 256 --fewshot-seed 1234 \
  --save-samples --out-dir "$DAG_EVAL_OUT/gsm8k_full"

python scripts/summarize_paired_eval.py \
  --dir "$DAG_EVAL_OUT/gsm8k_full" --draws 10000 --seed 20260917
```

它会重新生成全部答案，耗时明显长于 likelihood 评估。Dense 默认用 KV cache，
DAGFormer 每步重算 prefix；本轮另做了 `--no-kv-cache` 的 200 题敏感性检查。
报告主要分数时，应同时检查 strict/flexible extraction 和实际生成文本。

## 固定十条边的普通任务干预

```bash
python experiments/results/lmeval/run_eval.py \
  --models-root "$DAG_EVAL_MODELS" --model 300m-dagformer \
  --tokenizer "$DAG_EVAL_MODELS/tokenizer" --tasks "$DAG_EVAL_TASKS" \
  --batch-size 8 --max-length 1024 --routing-intervention context_both \
  --routing-gamma 1.25 --save-samples --save-likelihood-samples \
  --out-dir "$DAG_EVAL_OUT/standard_context_both"

python scripts/compare_eval_variants.py \
  --reference-dir "$DAG_EVAL_OUT/standard_matched" \
  --variant-dir "$DAG_EVAL_OUT/standard_context_both" \
  --draws 10000 --seed 20260917
```

随机头对照加 `--routing-control-seed 1000` 或 `1001`，并使用各自的输出目录。
保持相同 batch size。本轮 predictor-only 为 `context_pred`、gamma 1.5；
correction-only 为 `context_corr`、gamma 1.25。

## Predictor / correction 依赖性

```bash
python scripts/eval_routing_dependence.py \
  --config "$DAG_EVAL_MODELS/300m-dagformer/config.yaml" \
  --ckpt "$DAG_EVAL_MODELS/300m-dagformer/checkpoint.pt" \
  --calibration-cache "$DAG_EVAL_MODELS/eval_corpora/wikitext_train.pt" \
  --eval-cache "$DAG_EVAL_MODELS/eval_corpora/wikitext_test.pt" \
  --n-calibration 64 --n-eval 128 --seed 20260917 \
  --out "$DAG_EVAL_OUT/routing_dependence/300m.json" \
  --table-out "$DAG_EVAL_OUT/routing_dependence/300m_tables.pt"
```

此入口同时测自然文本和合成重复序列。`--upcast-fp32` 可复查 FP32 前向，
该次运行应另设结果和 table 输出路径。具体干预和各项差值均保存于 JSON。

## SAE 在新增文章上的完整控制组

```bash
python scripts/eval_sae_transfer.py \
  --config "$DAG_EVAL_MODELS/300m-dagformer/config.yaml" \
  --ckpt "$DAG_EVAL_MODELS/300m-dagformer/checkpoint.pt" \
  --directions "$DAG_EVAL_MODELS/sae_selected.pt" \
  --eval-cache "$DAG_EVAL_MODELS/eval_corpora/sae_replication/wikitext_test.pt" \
  --n-sequences 64 --control-mode whole-heads --stream-decomposition \
  --seed 20260917 \
  --corpus-note '64 windows from 14 additional articles; all 31 earlier test documents excluded.' \
  --out "$DAG_EVAL_OUT/sae_article_replication/300m.json"

python scripts/summarize_sae_transfer.py \
  --input "$DAG_EVAL_OUT/sae_article_replication/300m.json" \
  --old-screen experiments/results/interp/routing_sae/screen.json \
  --tokenizer "$DAG_EVAL_MODELS/tokenizer"

python scripts/summarize_sae_blocks.py \
  --input "$DAG_EVAL_OUT/sae_article_replication/300m.json" --mechanism-contrasts
```

这个缓存的选取和排除规则见 [预先写定的设置](provenance/sae_replication_protocol.md)。
它包含全部八个固定方向、五个整头对照和 R/QKV 拆分，而不仅是首批的正结果。
其他专项运行的完整参数在各自结果 JSON 的 `args` 字段；结果入口见
[总报告](README.md)。
