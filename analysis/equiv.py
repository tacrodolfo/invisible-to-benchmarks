#!/usr/bin/env python3
"""Equivalence between the C++ harness and bench.py (output of 01_build_equiv.sh).

Usage: python3 analysis/equiv.py --py data/py_results --cpp results/equiv
(--py: CSVs from the Python harness in legacy/python, with the names it produces)
"""
import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import add_times, load_csv, parse_tegrastats, read_meta  # noqa: E402

ST = ["dec", "pre", "h2d", "inf", "d2h", "post"]


def find_py(d, clocks, rel):
    c = sorted(glob.glob(os.path.join(d, f"yolox_s*_25W_clocks-{clocks}_stress-off_{rel}.csv")))
    return c[0] if c else None


def stats(path, sync="stage"):
    df = load_csv(path)
    ok = add_times(df, sync)
    ok0 = ok[ok["pass"] == 0]
    r = {s: ok0[s].mean() for s in ST}
    r.update(mean=ok0["resp"].mean(), p99=np.percentile(ok0["resp"], 99), drops=int((df[df["pass"] == 0]["drop"]).sum()))
    t = parse_tegrastats(path[:-4] + ".tegrastats")
    r.update(gpu_mhz=t.get("gpu_mhz_med", np.nan), gpu_util=t.get("gpu_util", np.nan), power=t.get("power_W", np.nan))
    return r, ok0


def ndet_match(a, b):
    m = a[["frame", "n_det"]].merge(b[["frame", "n_det"]], on="frame", suffixes=("_a", "_b"))
    if m.empty:
        return np.nan, np.nan, 0
    same = (m.n_det_a == m.n_det_b).mean() * 100
    return same, (m.n_det_a - m.n_det_b).abs().mean(), len(m)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--py", required=True)
    ap.add_argument("--cpp", required=True)
    a = ap.parse_args()

    pairs = [("b2b, factory", find_py(a.py, "dvfs", "b2b"), "cpp_b2b_default.csv"),
             ("12 Hz, factory (hybrid wait)", find_py(a.py, "dvfs", "periodic"), "cpp_periodic_default_hybrid.csv"),
             ("12 Hz, factory (clock_nanosleep)", find_py(a.py, "dvfs", "periodic"), "cpp_periodic_default_sleep.csv"),
             ("12 Hz, locked", find_py(a.py, "fixed", "periodic"), "cpp_periodic_lock.csv")]
    print("== 1. Python vs C++ (pass 0; ms)\n")
    rows = []
    for name, py, cpp in pairs:
        cpp = os.path.join(a.cpp, cpp)
        if not os.path.exists(cpp):
            continue
        rc, okc = stats(cpp)
        row = {"condition": name, "harness": "C++", **rc}
        if py:
            rp, okp = stats(py)
            same, mad, n = ndet_match(okp, okc)
            rows.append({"condition": name, "harness": "Python", **rp})
            row["same n_det (%)"] = same
            row["frames"] = n
        rows.append(row)
    t = pd.DataFrame(rows)
    if len(t):
        cols = ["condition", "harness", "mean", "p99", "drops"] + ST + ["gpu_mhz", "gpu_util", "power", "same n_det (%)"]
        print(t[[c for c in cols if c in t]].to_string(index=False, float_format=lambda v: f"{v:.2f}"))
    else:
        print("(no runs)")

    print("\n== 2. New paths (200 frames, b2b, factory) against jpeg_cpu\n")
    ref = os.path.join(a.cpp, "smoke_jpeg_cpu.csv")
    rows = []
    if os.path.exists(ref):
        _, okr = stats(ref)
        for f in sorted(glob.glob(os.path.join(a.cpp, "smoke_*.csv"))):
            sync = read_meta(f[:-4] + ".meta.json").get("args", {}).get("sync", "stage")
            r, ok = stats(f, sync)
            same, mad, n = ndet_match(okr, ok)
            gpu = (ok["g_gpu"] / ok["svc"]).mean() * 100
            rows.append({"variant": os.path.basename(f)[6:-4], **{k: r[k] for k in ["mean"] + ST},
                         "GPU share of frame (%)": gpu, "same n_det (%)": same, "|Δn_det|": mad})
        print(pd.DataFrame(rows).to_string(index=False, float_format=lambda v: f"{v:.2f}"))
        print("\n   tensor/network only: n_det does not apply (no post-processing). nvJPEG and GPU pre-processing may")
        print("   differ from OpenCV by ±1 gray level; small n_det differences are expected in those cases.")

    print("\n== 3. Go/no-go (C++, 12 Hz)\n")
    d = os.path.join(a.cpp, "cpp_periodic_default_hybrid.csv")
    l = os.path.join(a.cpp, "cpp_periodic_lock.csv")
    if os.path.exists(d) and os.path.exists(l):
        rd, _ = stats(d)
        rl, _ = stats(l)
        print(f"   mean response: factory {rd['mean']:.1f} ms vs locked {rl['mean']:.1f} ms  (ratio {rd['mean'] / rl['mean']:.2f})")
        print(f"   inference:     factory {rd['inf']:.1f} ms vs locked {rl['inf']:.1f} ms  (ratio {rd['inf'] / rl['inf']:.2f})")
        print(f"   GPU (tegrastats, median): factory {rd['gpu_mhz']:.0f} MHz vs locked {rl['gpu_mhz']:.0f} MHz")
        go = rd["inf"] / rl["inf"] > 1.5
        print("\n   " + ("GO: the GPU governor still holds the frequency down in the C++ harness."
                         if go else "WARNING: the inference effect dropped sharply in C++; review before the campaign."))


if __name__ == "__main__":
    main()
