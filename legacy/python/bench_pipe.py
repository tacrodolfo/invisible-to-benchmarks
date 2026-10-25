#!/usr/bin/env python3
"""Two-stage pipelined variant of bench.py.

Stage A (producer thread): JPEG decode + pre-processing of frame k+1
Stage B (main thread): H2D copy + TensorRT inference + D2H copy + NMS of frame k
The stages communicate through a queue of size 1, so A is at most one frame ahead of B.
OpenCV and the CUDA/TensorRT calls release the GIL, so A and B truly run in parallel.

Same output columns as bench.py, plus t_deq (when B takes the frame from the queue).
Periodic mode: frame k is released at base + kT. If A has not started frame k when frame k+1
has already been released, frame k is dropped (drop=1), like a camera that overwrites its buffer.

Usage:
  python3 bench_pipe.py --engine yolox_s_fp16.engine --frames frames --mode periodic --period-ms 83.3 --tag t
  MOCK=1 python3 bench_pipe.py ...   # test without a GPU (simulated times)
"""
import argparse
import csv
import gc
import glob
import os
import queue
import threading
import time

import numpy as np

MOCK = os.environ.get("MOCK") == "1"
if MOCK:
    def preprocess(img):
        time.sleep(0.003)
        return np.zeros((3, 4, 4), np.float32), 1.0

    def postprocess(out, r):
        time.sleep(0.004)
        return 3

    def decode(jpeg):
        time.sleep(0.015)
        return jpeg

    class TRTModel:
        def __init__(self, path):
            pass

        def h2d(self, x):
            time.sleep(0.001)

        def infer(self):
            time.sleep(0.010)

        def d2h(self):
            time.sleep(0.0003)
            return None

    def wait_until(t_ns):
        while True:
            rem = t_ns - time.perf_counter_ns()
            if rem <= 0:
                return
            if rem > 2_000_000:
                time.sleep((rem - 1_000_000) / 1e9)
else:
    import cv2
    from bench import TRTModel, postprocess, preprocess, wait_until

    def decode(jpeg):
        return cv2.imdecode(jpeg, cv2.IMREAD_COLOR)

COLS = ["pass", "frame", "drop", "t_rel", "t_start", "t_dec", "t_pre", "t_deq",
        "t_h2d", "t_inf", "t_d2h", "t_end", "n_det"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", required=True)
    ap.add_argument("--frames", default="frames")
    ap.add_argument("--mode", choices=["b2b", "periodic"], default="periodic")
    ap.add_argument("--period-ms", type=float, default=83.3)
    ap.add_argument("--passes", type=int, default=3)
    ap.add_argument("--warmup", type=int, default=100)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--outdir", default="results_pipe")
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.frames, "*.jpg")))
    if args.limit:
        files = files[: args.limit]
    frames = [np.fromfile(f, np.uint8) for f in files]
    print(f"{len(frames)} frames loaded (2-stage pipeline)")
    model = TRTModel(args.engine)

    # sequential warm-up
    for i in range(args.warmup):
        x, r = preprocess(decode(frames[i % len(frames)]))
        model.h2d(x); model.infer(); out = model.d2h(); postprocess(out, r)

    period = int(args.period_ms * 1e6)
    now = time.perf_counter_ns
    rows, lock = [], threading.Lock()
    q = queue.Queue(maxsize=1)
    DONE = object()

    def producer():
        for p in range(args.passes):
            base = now() + 20_000_000
            for k, jpeg in enumerate(frames):
                if args.mode == "periodic":
                    rel = base + k * period
                    t = now()
                    if t < rel:
                        wait_until(rel)
                    elif t >= rel + period:
                        with lock:
                            rows.append([p, k, 1, rel] + [0] * 8 + [-1])
                        continue
                t_start = now()
                if args.mode == "b2b":
                    rel = t_start
                img = decode(jpeg); t_dec = now()
                x, r = preprocess(img); t_pre = now()
                q.put((p, k, rel, t_start, t_dec, t_pre, x, r))
            q.put(("PASS_END", p))
        q.put(DONE)

    gc.collect()
    gc.disable()
    th = threading.Thread(target=producer, daemon=True)
    th.start()
    while True:
        item = q.get()
        if item is DONE:
            break
        if item[0] == "PASS_END":
            print(f"pass {item[1] + 1}/{args.passes} done")
            continue
        p, k, rel, t_start, t_dec, t_pre, x, r = item
        t_deq = now()
        model.h2d(x); t_h2d = now()
        model.infer(); t_inf = now()
        out = model.d2h(); t_d2h = now()
        n_det = postprocess(out, r); t_end = now()
        with lock:
            rows.append([p, k, 0, rel, t_start, t_dec, t_pre, t_deq, t_h2d, t_inf, t_d2h, t_end, n_det])
    th.join()
    gc.enable()

    rows.sort(key=lambda r: (r[0], r[1]))
    os.makedirs(args.outdir, exist_ok=True)
    path = os.path.join(args.outdir, f"{args.tag}.csv")
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(COLS)
        w.writerows(rows)
    print(f"saved to {path}")


if __name__ == "__main__":
    main()
