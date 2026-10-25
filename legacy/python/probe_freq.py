#!/usr/bin/env python3
"""Samples GPU frequency and load and CPU frequency directly from sysfs, at high resolution.

Writes a CSV with t_ns (time.perf_counter_ns, the same CLOCK_MONOTONIC clock as bench.py),
gpu_hz, gpu_load (0-1000, if available) and cpu0_khz.
Runs until SIGTERM/SIGINT; data are kept in memory and written at the end.

Usage: sudo python3 probe_freq.py --out file.csv [--interval-ms 1]
"""
import argparse
import glob
import os
import signal
import time


def find_gpu_devfreq():
    c = sorted(glob.glob("/sys/class/devfreq/*gpu*"))
    return c[0] if c else None


def find_gpu_load(devfreq):
    cands = []
    if devfreq:
        cands.append(os.path.join(devfreq, "device", "load"))
    cands += glob.glob("/sys/devices/platform/bus@0/*.gpu/load") + glob.glob("/sys/devices/platform/*gpu*/load") \
        + glob.glob("/sys/devices/gpu.0/load")
    for p in cands:
        if os.path.exists(p):
            return p
    return None


def opener(path):
    if path is None:
        return None
    try:
        return open(path, "rb", buffering=0)
    except OSError:
        return None


def read(f):
    if f is None:
        return -1
    try:
        f.seek(0)
        return int(f.read().split()[0])
    except (OSError, ValueError, IndexError):
        return -1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--interval-ms", type=float, default=1.0)
    args = ap.parse_args()

    dev = find_gpu_devfreq()
    fg = opener(os.path.join(dev, "cur_freq") if dev else None)
    fl = opener(find_gpu_load(dev))
    fc = opener("/sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq")
    print(f"probe: gpu={dev} load={'ok' if fl else 'n/a'} cpu0={'ok' if fc else 'n/a'} "
          f"interval={args.interval_ms} ms", flush=True)

    stop = [False]

    def handler(*_):
        stop[0] = True
    signal.signal(signal.SIGTERM, handler)
    signal.signal(signal.SIGINT, handler)

    rows = []
    step = int(args.interval_ms * 1e6)
    now = time.perf_counter_ns
    nxt = now()
    while not stop[0]:
        rows.append((now(), read(fg), read(fl), read(fc)))
        nxt += step
        rem = nxt - now()
        if rem > 0:
            time.sleep(rem / 1e9)
        else:
            nxt = now()

    with open(args.out, "w") as f:
        f.write("t_ns,gpu_hz,gpu_load,cpu0_khz\n")
        for r in rows:
            f.write(",".join(map(str, r)) + "\n")
    print(f"probe: {len(rows)} samples in {args.out}", flush=True)


if __name__ == "__main__":
    main()
