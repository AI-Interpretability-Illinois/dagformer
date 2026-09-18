"""Frozen rank suffixes must replay the same samples at optimizer boundaries."""
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import sys
from threading import Thread
import time

import aiohttp
import fsspec
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.freeze_dolma_continuation import dolma_http_options, merge_ranks, write_suffix
from src.data.dolma import DolmaPackedDataset
from src.data.mmap_dataset import MmapPackedDataset


class Tokenizer:
    eos_token_id = 100000

    def __call__(self, text, **kwargs):
        return {"input_ids": [70000 + int(x) for x in text.split()]}

    def decode(self, ids, **kwargs):
        return " ".join(map(str, ids))


class Source(DolmaPackedDataset):
    def _load_stream(self):
        return iter({"text": f"{i} {i + 1} {i + 2}"} for i in range(300))


def test_sequential_http_read_outlives_default_total_timeout(monkeypatch):
    # Compress the production failure (a healthy stream open beyond 300 s)
    # into a local HTTP response that stays active beyond a 0.1 s deadline.
    monkeypatch.setattr(aiohttp.client, "DEFAULT_TIMEOUT", aiohttp.ClientTimeout(total=0.1))

    class Handler(BaseHTTPRequestHandler):
        def do_HEAD(self):
            self.send_response(200)
            self.send_header("Content-Length", "4")
            self.end_headers()

        def do_GET(self):
            self.do_HEAD()
            for _ in range(4):
                self.wfile.write(b"x")
                self.wfile.flush()
                time.sleep(0.1)

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        fs = fsspec.filesystem("http", **dolma_http_options()["http"], skip_instance_cache=True)
        with fs.open(f"http://127.0.0.1:{server.server_port}/shard", "rb") as stream:
            assert stream.read() == b"xxxx"
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def test_frozen_suffix_roundtrips_rank_order_and_resume(tmp_path):
    records, expected = [], []
    for rank in range(3):
        samples = list(Source(Tokenizer(), seq_len=7, skip_samples=5 + rank, max_samples=12))
        path = tmp_path / f"rank_{rank}.bin"
        write_suffix(samples, path, 12, 7, rank)
        records.append({"rank": rank, "samples": 12, "path": str(path)})
        expected.append(samples)
    total = merge_ranks(records[::-1], tmp_path / "continuation.bin", 7)
    index = {"format_version": 1, "dtype": "uint32", "seq_len": 7,
             "eos_id": 100000, "total_tokens": total,
             "shards": [{"name": "continuation.bin", "n_tokens": total}]}
    (tmp_path / "index.json").write_text(json.dumps(index))
    for rank in range(3):
        restored = MmapPackedDataset(str(tmp_path), seq_len=7, rank=rank, world_size=3,
                                     block_size=37, skip_samples=3 * 4, max_samples=8)
        for actual, original in zip(restored, expected[rank][4:]):
            np.testing.assert_array_equal(actual["olmo_ids"], original["olmo_ids"])
            np.testing.assert_array_equal(actual["olmo_labels"], original["olmo_labels"])
