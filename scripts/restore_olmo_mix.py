"""Restore the original deterministic OLMo-mix file selection at a pinned revision."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
from pathlib import Path
import random
import sys
import time

from huggingface_hub import HfApi, hf_hub_download
from huggingface_hub.utils import disable_progress_bars

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.download_olmo_mix_local import CAPS, SUFFIXES  # noqa: E402

REPOSITORY = "allenai/olmo-mix-1124"
HISTORICAL_REVISION = "99ee6aaace88779d1ef099d36251b91101c1679b"


def write_manifest(path: Path, manifest: dict) -> None:
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(manifest, indent=2) + "\n")
    os.replace(temporary, path)


def restore(root: Path, revision: str, seed: int, workers: int, dry_run: bool) -> dict:
    root.mkdir(parents=True, exist_ok=True)
    info = HfApi().dataset_info(REPOSITORY, revision=revision)
    all_files = [s.rfilename for s in info.siblings]
    subsets = []
    for name, suffix in SUFFIXES.items():
        files = sorted(f for f in all_files if f.startswith(f"data/{name}/")
                       and f.endswith(suffix))
        random.Random(seed).shuffle(files)
        files = files[:CAPS[name]]
        if not files:
            raise RuntimeError(f"No files found for {name} at {info.sha}")
        subsets.append({"name": name, "files": files})
    manifest = {
        "repository": REPOSITORY,
        "revision": info.sha,
        "repository_last_modified": str(info.last_modified),
        "selection_seed": seed,
        "subsets": subsets,
        "download_complete": False,
    }
    manifest_path = root / "download_manifest.json"
    write_manifest(manifest_path, manifest)
    files = [f for subset in subsets for f in subset["files"]]
    print(json.dumps({"revision": info.sha, "files": len(files),
                      "destination": str(root), "dry_run": dry_run}), flush=True)
    if dry_run:
        return manifest

    disable_progress_bars()

    def download(relative: str) -> dict:
        for attempt in range(5):
            try:
                path = Path(hf_hub_download(
                    repo_id=REPOSITORY, repo_type="dataset", filename=relative,
                    revision=info.sha, local_dir=str(root),
                ))
                return {"path": relative, "bytes": path.stat().st_size}
            except Exception as exc:
                if attempt == 4:
                    raise RuntimeError(f"Download failed: {relative} ({type(exc).__name__})") from exc
                time.sleep(min(30, 2 ** attempt))
        raise AssertionError("unreachable")

    downloaded = {}
    start = time.monotonic()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        pending = {pool.submit(download, name): name for name in files}
        for future in as_completed(pending):
            record = future.result()
            downloaded[record["path"]] = record["bytes"]
            if len(downloaded) % 10 == 0 or len(downloaded) == len(files):
                print(f"Restored {len(downloaded)}/{len(files)} files; "
                      f"{sum(downloaded.values()) / 1e9:.2f} GB; "
                      f"elapsed {(time.monotonic() - start) / 60:.1f} min", flush=True)
    manifest["file_bytes"] = {name: downloaded[name] for name in files}
    manifest["download_complete"] = True
    write_manifest(manifest_path, manifest)
    print(f"Download complete: {manifest_path}", flush=True)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--revision", default=HISTORICAL_REVISION)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be positive")
    restore(args.out_dir.resolve(), args.revision, args.seed, args.workers, args.dry_run)


if __name__ == "__main__":
    main()
