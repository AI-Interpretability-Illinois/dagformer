# Xiaocong 实际训练与 eval setting

2026-09-28，根据实际脚本、cache tensor、数据索引和结果JSON核对。

**没有发现“他改用OOD eval、我们用Dolma eval”能解释整体平移的证据。
新模型的训练时eval与我们的旧eval cache完全相同；公开WikiText评估也使用
同一个benchmark。训练池则确实不等于完整Dolma多源混合。**

## 预训练数据

新75M/150M使用 `allenai/dolma` / `v1_7`，实际读取：

`/srv/local/xy51/pretok/dolma_v1_7_12b/index.json`

此索引记录12,000,000,950 tokens。依据 `provenance.merged_from` 的顺序和
最终截断长度计算，池内组成是：

| 来源 | token数 | 占比 |
|---|---:|---:|
| Books，books-0000..0002 | 2,538,926,025 | 21.16% |
| C4，c4-0000..0022，最后一个截断 | 9,461,074,925 | 78.84% |

没有其他Dolma来源进入这个12B池。`build12b_local.sh` 和 `build12b_extra.sh`
先逐文件分词，然后按文件顺序拼接、截断；训练的 mmap reader 使用block
shuffle。因此“没有完整按多源权重混Dolma”是成立的，但不能说只训练了单一来源，
也不能把磁盘文件顺序等同于每步训练顺序。

这些占比是可供取样的训练池占比，不是每个1.585B/3.170B run实际消耗样本的
精确来源比例。旧12B训练cache已不在，尚不能验证新旧训练token逐个一致。
本次计算保存在 [training_pool_composition.json](training_pool_composition.json)。

## 预训练时 eval

`/srv/local/xy51/run_build_eval_cache.sh:26` 的真实调用是：

```python
build_eval_dataloader(
    dataset_name="allenai/dolma", dataset_version="v1_7",
    seq_len=1024, batch_size=4, eval_skip=1_000_000, eval_size=50,
    cache_path=".../pretrain_75m_dagformer_modular/eval_cache.pt",
)
```

实现从Dolma原始 `train` stream 跳过一百万**文档**，然后打包50条1024-token
序列；不是随机抽取整个Dolma混合的独立test split，也不是Wikipedia-only
或另一个OOD数据集。`eval_size=50` 指packed序列数，不是50个原始文档。

本次实际读取新75M/150M六个目录及300M目录的cache，与我们
`checkpoints/pr_sync_20260917/eval_cache.pt` 比较：**七个cache的50×1024
inputs和labels都逐元素完全相等**。证据保存在
[eval_cache_identity.json](eval_cache_identity.json)。所以这些cache之间的NLL
变化不能解释为更换eval样本。

PR已披露该固定片段与训练池有部分重叠；“skip了一百万文档”本身不保证与
从前缀构建的训练池隔离。本次未重新扫描原始语料计算重叠比例。

## 训练后普通 eval

`/srv/local/xy51/lmeval/run_lmeval_150m.sh:19` 运行
`scripts/eval_lm_harness.py --tasks default --batch_size 8`；实际结果JSON记录
`max_length=1024`、`limit=null`。与我们旧结果重叠的9个任务，task version
和样本数一致：WikiText、LAMBADA、HellaSwag、PIQA、ARC Easy、WinoGrande、
OpenBookQA、SciQ、BoolQ。0-shot；tokenizer为OLMo-2。

WikiText是独立benchmark，两边都在同一WikiText口径上评估。其word-PPL
以及取自然对数得到的word-NLL，不能与训练日志里的token-NLL直接放在同一
纵轴。具体训练后数值见 [new_small_models.md](new_small_models.md)。

## Pruning eval

`configs/prune/150m_fourway_math_timan1.yaml:7` 起明确区分：

| 用途 | 实际数据 |
|---|---|
| finetune训练 | MathInstruct，260,039个训练文档 |
| `domain_nll` | 同源预留2,000个文档，打包436个1024-token窗口 |
| `general_nll` | WikiText-2 test，62个文档，打包282个1024-token窗口 |

这些数量来自 `/srv/local/xy51/prune/data/{mathinstruct,wikitext2}/summary.json`。
所有新pruning family共享这两个评估cache。这里general相对math finetuning
确实是跨域控制；domain不是Dolma。它们也不应与预训练Dolma eval直接比较
绝对NLL，更不能把finetune后的模型当成同一个预训练checkpoint。

## 已找到的曲线口径

`/srv/local/xy51/compare_history.py:10–13` 将旧 streaming books-only、
新 mmap books-only、新 mmap Books+C4 放在同一张历史图上；`:28` 明确画的是
**train NLL**，已查看生成的 `compare_75m_history.png` 确认。图中末尾约
3.84→4.26 的上移不可能由更换 eval 样本直接造成；训练输入域变化是候选解释。
其中旧数据的 books-only 标签来自同事脚本，未在本次恢复旧训练数据逐样本核验。

常规 `/srv/local/xy51/compare_runs.py:18–21`、`:60–71` 则是左侧
train NLL、右侧固定缓存 eval NLL。我们分享的 handoff loss curve 是训练
minibatch NLL；若拿它与右侧 eval 比较，会混用口径。目前尚不能确定对方
所说“整体平移”具体指哪幅图，因此不认定对方确实比错。

另一个容易混淆的版本是 `compare_150m_12b_progress.png`：它只画到约
3000 steps，而实际训练已完成6000 steps，最终 dense / corrected / modular
eval NLL 分别为 **3.7752 / 3.6707 / 3.6863**。

已定位到的数据差异主要在重新构建的训练池和训练轨迹，而非更换了同名
预训练eval cache。曲线字段、日志位置和另一批旧MUDDFormer剪枝结果见
[new_small_models.md](new_small_models.md) 的补充部分；WikiText转换为每词
NLL后的跨尺寸分析见 [nll_scaling.md](scaling/nll_scaling.md)。
