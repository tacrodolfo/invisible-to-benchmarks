#!/usr/bin/env python3
"""Go/no-go analysis: tail statistics, DMR(D), rankings, Kendall's tau, energy and the
decomposition into jitter (within a frame) x load variation (between frames).

Usage: python3 analyze.py --results results
Outputs: results/summary.csv, results/rankings.csv, results/ccdf_<mode>.png, results/dmr_<mode>.png
"""
import argparse
import glob
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import kendalltau

D_GRID = np.arange(5, 301, 1)        # ms
D_REF = [33, 40, 50, 60, 83]          # ms: 30 fps, 25 fps, 20 fps, ~16 fps and the camera period (12 Hz)
RNG = np.random.default_rng(0)


def boot_ci(x, q, n=1000):
    idx = RNG.integers(0, len(x), (n, len(x)))
    qs = np.percentile(x[idx], q, axis=1)
    return np.percentile(qs, [2.5, 97.5])


def energy(tag_path, n_frames, dur_s):
    log = tag_path.replace(".csv", ".tegrastats")
    if not os.path.exists(log):
        return np.nan, np.nan
    mw = [int(m) for m in re.findall(r"VDD_IN (\d+)mW", open(log).read())]
    if not mw:
        return np.nan, np.nan
    p_w = np.mean(mw) / 1000
    return p_w, p_w * dur_s / n_frames * 1000  # W, mJ/frame


def analyze_one(path):
    tag = os.path.basename(path)[:-4]
    df = pd.read_csv(path)
    ok = df[df["drop"] == 0].copy()
    ok["resp"] = (ok.t_end - ok.t_rel) / 1e6      # response time (includes waiting)
    ok["svc"] = (ok.t_end - ok.t_start) / 1e6     # service time
    for a, b, name in [("t_start", "t_dec", "dec"), ("t_dec", "t_pre", "pre"), ("t_pre", "t_h2d", "h2d"),
                       ("t_h2d", "t_inf", "inf"), ("t_inf", "t_d2h", "d2h"), ("t_d2h", "t_end", "post")]:
        ok[name] = (ok[b] - ok[a]) / 1e6

    n_total = len(df)
    n_drop = int(df["drop"].sum())
    x = ok.resp.to_numpy()
    lo, hi = boot_ci(x, 99)

    # response with drops = infinity, for the DMR
    resp_all = np.concatenate([x, np.full(n_drop, np.inf)])
    dmr = {D: float(np.mean(resp_all > D)) for D in D_REF}

    g = ok.groupby("frame").svc
    within = g.var(ddof=1).mean()
    between = g.mean().var(ddof=1)

    dur_s = (df.t_end[df.t_end > 0].max() - df.t_rel.min()) / 1e9
    p_w, mj = energy(path, len(ok), dur_s)

    row = dict(tag=tag, n=n_total, drops=n_drop,
               mean=x.mean(), median=np.median(x), p95=np.percentile(x, 95),
               p99=np.percentile(x, 99), p99_ci_lo=lo, p99_ci_hi=hi, max=x.max(),
               tail_ratio=np.percentile(x, 99) / np.median(x), cov=x.std() / x.mean(),
               frac_jitter=within / (within + between),
               power_W=p_w, energy_mJ_frame=mj,
               mean_n_det=ok.n_det.mean())
    for s in ["dec", "pre", "h2d", "inf", "d2h", "post"]:
        row[f"{s}_mean"] = ok[s].mean()
        row[f"{s}_p99"] = np.percentile(ok[s], 99)
    for D, v in dmr.items():
        row[f"dmr_{D}"] = v
    return row, resp_all


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    args = ap.parse_args()

    rows, curves = [], {}
    for p in sorted(glob.glob(os.path.join(args.results, "*.csv"))):
        b = os.path.basename(p)
        if b in ("summary.csv", "rankings.csv") or b.startswith("ranks_"):
            continue
        r, resp = analyze_one(p)
        rows.append(r)
        curves[r["tag"]] = resp
    summ = pd.DataFrame(rows)
    summ.to_csv(os.path.join(args.results, "summary.csv"), index=False, float_format="%.4f")
    pd.set_option("display.width", 200)
    print(summ[["tag", "drops", "mean", "median", "p95", "p99", "p99_ci_lo", "p99_ci_hi",
                "tail_ratio", "frac_jitter", "energy_mJ_frame"] + [f"dmr_{D}" for D in D_REF]]
          .to_string(index=False, float_format=lambda v: f"{v:.3f}"))

    # rankings split by release model
    rank_rows = []
    for rel in ["b2b", "periodic"]:
        sub = summ[summ.tag.str.endswith("_" + rel)]
        if len(sub) < 2:
            continue
        metrics = ["mean", "median", "p95", "p99"] + [f"dmr_{D}" for D in D_REF]
        ranks = sub.set_index("tag")[metrics].rank(method="min")
        for m in metrics:
            vals = sub.set_index("tag")[m]
            if vals.nunique() == 1:
                rank_rows.append(dict(release=rel, metric=m, best="(tie: all equal)",
                                      kendall_tau_vs_mean=np.nan))
                continue
            tau, _ = kendalltau(ranks["mean"], ranks[m])
            tops = vals[vals == vals.min()].index.tolist()
            best = tops[0] if len(tops) == 1 else f"(tie among {len(tops)})"
            rank_rows.append(dict(release=rel, metric=m, best=best, kendall_tau_vs_mean=tau))
        ranks.to_csv(os.path.join(args.results, f"ranks_{rel}.csv"))

        # CCDF
        plt.figure(figsize=(6, 4))
        for tag in sub.tag:
            v = np.sort(curves[tag][np.isfinite(curves[tag])])
            plt.step(v, 1 - np.arange(1, len(v) + 1) / len(v), where="post", label=tag)
        plt.yscale("log"); plt.xlabel("response time (ms)"); plt.ylabel("P(T > t)")
        plt.legend(fontsize=6); plt.grid(alpha=.3); plt.tight_layout()
        plt.savefig(os.path.join(args.results, f"ccdf_{rel}.png"), dpi=200); plt.close()

        # DMR(D)
        plt.figure(figsize=(6, 4))
        for tag in sub.tag:
            v = curves[tag]
            plt.plot(D_GRID, [np.mean(v > D) for D in D_GRID], label=tag)
        plt.yscale("log"); plt.xlabel("deadline D (ms)"); plt.ylabel("deadline miss rate")
        plt.legend(fontsize=6); plt.grid(alpha=.3); plt.tight_layout()
        plt.savefig(os.path.join(args.results, f"dmr_{rel}.png"), dpi=200); plt.close()

    rk = pd.DataFrame(rank_rows)
    rk.to_csv(os.path.join(args.results, "rankings.csv"), index=False, float_format="%.3f")
    print("\nRankings (best configuration per metric and Kendall's tau against the ranking by mean):")
    print(rk.to_string(index=False))


if __name__ == "__main__":
    main()
