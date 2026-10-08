#!/usr/bin/env python
"""Tiny GPU heartbeat: keeps every visible GPU minimally busy with negligible RAM.

Runs one small matmul on each GPU every INTERVAL seconds and sleeps in between.
GPU memory: ~1 MB of tensors per device (plus the CUDA context). Host RAM: just
the torch import. Exits cleanly on SIGTERM/SIGINT (SLURM time limit / scancel).

    python keepalive_gpu.py [--interval 15] [--size 512] [--log-every 600]
"""
import argparse
import os
import signal
import socket
import sys
import time

stop = False


def _handle(signum, _frame):
    global stop
    stop = True
    print(f"[{socket.gethostname()}] got signal {signum}, exiting", flush=True)


signal.signal(signal.SIGTERM, _handle)
signal.signal(signal.SIGINT, _handle)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--interval", type=float, default=15.0, help="seconds between heartbeats")
    ap.add_argument("--size", type=int, default=512, help="matmul side length")
    ap.add_argument("--log-every", type=float, default=600.0, help="seconds between log lines")
    ap.add_argument("--duty", type=float, default=0.0,
                    help="fraction of each interval spent doing matmuls (0 = one matmul per beat)")
    args = ap.parse_args()

    host = socket.gethostname()
    import torch

    if not torch.cuda.is_available():
        # Still hold the allocation (that is what keeps the reservation alive);
        # just idle-loop without GPU work.
        print(f"[{host}] WARNING: CUDA not available, idling only", flush=True)
        while not stop:
            time.sleep(args.interval)
        return 0

    n = torch.cuda.device_count()
    print(f"[{host}] heartbeat on {n} GPU(s), CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}",
          flush=True)
    mats = []
    for i in range(n):
        with torch.cuda.device(i):
            mats.append(torch.randn(args.size, args.size, device=f"cuda:{i}", dtype=torch.float16))

    beats = 0
    last_log = time.time()
    while not stop:
        busy_until = time.time() + args.duty * args.interval
        while True:
            for i, a in enumerate(mats):
                with torch.cuda.device(i):
                    (a @ a).sum().item()  # kernel + sync; with --duty this repeats for part of the interval
            if time.time() >= busy_until:
                break
        beats += 1
        now = time.time()
        if now - last_log >= args.log_every:
            print(f"[{host}] {time.strftime('%F %T')} alive, {beats} beats on {n} GPU(s)", flush=True)
            last_log = now
        # sleep in small steps so SIGTERM is noticed quickly
        end = time.time() + args.interval
        while not stop and time.time() < end:
            time.sleep(min(1.0, end - time.time()))
    return 0


if __name__ == "__main__":
    sys.exit(main())
