# Delta storage layout

Code, Python environments, caches, evaluation outputs, and run metadata live in
`/u/yurenh2/dagformer-20260917`. Model checkpoints and training datasets live in
`/work/hdd/biro/yurenh2/dagformer-20260917`.

| Location | Contents |
|---|---|
| Home: `code/<revision>`, `repository.git` | Pinned Git worktrees and their shared repository |
| Home: `envs`, `cache` | Python environments, Hugging Face and pip caches |
| Home: `results`, `provenance`, `tokenizer`, `wandb` | Evaluation artifacts, records, tokenizer, and logs |
| Home: `run_metadata/<model>-complete` | Resume YAML, launch record, and metrics CSV |
| Biro: `checkpoints`, `runs` | Source, continuation, and exported model weights; optimizer states |
| Biro: `data/olmo-mix-1124`, `data/pretok`, `data/eval_corpora` | Raw corpora, packed training shards, and evaluation corpus tensors |

Old small-file paths on biro are compatibility symlinks to home, so already
submitted job scripts and absolute checkpoint references remain usable. The
large files themselves stay on biro. `biro_job_common.sh` places new code and
caches in home; run metadata paths also point there. The finalizer archives
evaluation artifacts directly into home without allocating a GPU to copy them.

The September 21 migration holds this project's pending jobs while files move.
Each directory is copied and compared with rsync before removing the old copy.
The shared Git worktree links are repaired after relocation. Records are kept
in home as `storage_migration_20260921.json` and
`tokenization_cache_cleanup_20260921.json`.

The completed 1B corpus has 42 shards and 21,000,000,125 stored tokens. The 600M
continuation corpus has one shard and 4,670,195,200 stored tokens. Both indices
and every shard's expected byte length were checked before removing the
reproducible per-file tokenization cache (`data/tokenized_files/olmo_mix`,
190,164,809,430 bytes). Its JSON manifests and the final dataset indices were
preserved in the cleanup record. Raw source corpora, packed datasets, and model
checkpoints were retained.

Job 22160737 completed 1B continuation on September 19 at 21:52 CDT, with
38,160 optimizer updates and 20,006,830,080 processed training tokens. Its first
evaluation job, 22160906, failed when writing samples with `Disk quota exceeded`.
The project was above its 550 GiB block hard limit at inspection; small-file
relocation also reduces its inode consumption. Evaluation is retried by job
22283422 using home for outputs. The 600M continuations remain separate queued
jobs, 22160902 and 22160903.
