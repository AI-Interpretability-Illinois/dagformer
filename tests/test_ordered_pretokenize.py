"""Parallel tokenization must preserve serial packing across documents and files."""
import argparse
import gzip
import json
from pathlib import Path
import sys

import numpy as np
from tokenizers import Tokenizer, models, pre_tokenizers, trainers
from transformers import AutoTokenizer, PreTrainedTokenizerFast
import zstandard

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.pretokenize import _OlmoMixLocalSource, packed_token_stream, ShardWriter
from scripts.pretokenize_olmo_ordered import OrderedShardWriter, run


def test_parallel_output_matches_serial_packing(tmp_path):
    texts = ["alpha beta gamma", "αβγ 中文 café", "x " * 200, "tail next file"]
    backend = Tokenizer(models.BPE(unk_token="[UNK]"))
    backend.pre_tokenizer = pre_tokenizers.ByteLevel()
    backend.train_from_iterator(texts, trainers.BpeTrainer(
        vocab_size=128, special_tokens=["[UNK]", "[EOS]"]))
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, eos_token="[EOS]")
    tokenizer_path = tmp_path / "tokenizer"
    tokenizer.save_pretrained(tokenizer_path)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    source = tmp_path / "source"
    source.mkdir()
    first = (json.dumps({"text": texts[0]}) + '\n\ninvalid json\n'
             + json.dumps({"text": "  "}) + '\n')
    second = '\n'.join(json.dumps({"text": t}) for t in texts[1:]) + '\n'
    zstd_path = source / "first.jsonl.zstd"
    compressor = zstandard.ZstdCompressor()
    zstd_path.write_bytes(compressor.compress(first.encode())
                          + compressor.compress(second.encode()))
    gzip_path = source / "second.json.gz"
    with gzip.open(gzip_path, "wt") as file:
        file.write(first + second * 4)
    # Both file order and subset order must survive asynchronous completion.
    manifest = {"repository": "fixture", "selection_seed": 0,
                "download_complete": True, "subsets": [
                    {"name": "first", "files": [zstd_path.name]},
                    {"name": "second", "files": [gzip_path.name]}]}
    manifest_path = source / "download_manifest.json"
    manifest_path.write_text(json.dumps(manifest))
    serial = ShardWriter(str(tmp_path / "serial"), shard_tokens=43)
    dataset = _OlmoMixLocalSource([{"files": [str(zstd_path), str(gzip_path)]}])
    for window in packed_token_stream(tokenizer, dataset, 16, tokenizer.eos_token_id):
        serial.add(window)
        if serial.total_tokens >= 400:
            break
    serial.close()
    args = argparse.Namespace(manifest=manifest_path, out_dir=tmp_path / "parallel",
                              cache_dir=tmp_path / "cache", tokenizer_id=str(tokenizer_path),
                              seq_len=16, token_budget=400, shard_tokens=43,
                              workers=2, batch_size=3, reconstruction=None)
    provenance = run(args)
    index = json.loads((args.out_dir / "index.json").read_text())
    assert index["shards"] == serial.shards
    for shard in serial.shards:
        assert ((tmp_path / "serial" / shard["name"]).read_bytes()
                == (args.out_dir / shard["name"]).read_bytes())
    assert set(provenance["actual_subset_tokens"]) == {"first", "second"}
    # Completed per-file caches can be reused without altering output.
    original = {p.name: p.stat().st_mtime_ns for p in args.cache_dir.glob("*.bin")}
    run(args)
    assert original == {p.name: p.stat().st_mtime_ns for p in args.cache_dir.glob("*.bin")}


def test_uint32_ids_and_cross_file_shard_boundaries(tmp_path):
    expected = np.arange(69995, 70055, dtype=np.uint32)
    paths = []
    for i, part in enumerate(np.split(expected, [7, 31])):
        path = tmp_path / f"input_{i}.bin"
        part.tofile(path)
        paths.append(path)
    writer = OrderedShardWriter(tmp_path / "out", seq_len=4, shard_tokens=13, budget=47)
    for path in paths:
        writer.append(path)
    writer.close()
    result = np.concatenate([np.fromfile(tmp_path / "out" / s["name"], dtype=np.uint32)
                             for s in writer.shards])
    np.testing.assert_array_equal(result, expected[:50])
    assert [s["n_tokens"] for s in writer.shards] == [15, 15, 15, 5]
