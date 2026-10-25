#!/usr/bin/env python3
"""Analysis of all campaigns run with the C++ harness.

Usage: python3 analysis/analyze_cpp.py --res results [--nboot 2000] [--no-figs]

Outputs in results/tables/:
  runs.csv           one row per run (metrics, condition, tegrastats, probe)
  factorial.csv      B0..B3 ladder and CPU, GPU and interaction effects (95% CI, block bootstrap)
  rates.csv          sustainable rate per condition, with the interval of the locked -> factory reduction
  trtexec.csv        trtexec (B0 and --idleTime sweep) with frequency and load sampled at 1 ms
  inputs.csv, sync.csv, floors.csv, maxn.csv, idle_states.csv, probe.csv, equiv.csv
and a few sanity-check figures (rates_*.png, trtexec_idle_*.png, inputs_*.png).
"""
import argparse
import glob
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import Blocks, joint_mean_effects, parse_probe, parse_tegrastats, read_cond, run_metrics  # noqa: E402

pd.set_option("display.width", 220)
pd.set_option("display.max_columns", 40)
pd.set_option("display.max_colwidth", 60)


def fmt(df, cols):
    cols = [c for c in cols if c in df.columns]
    return df[cols].to_string(index=False, float_format=lambda v: f"{v:.3g}" if abs(v) < 1e4 else f"{v:.0f}")


# ------------------------------------------------------------------ collection
def collect(res, nboot):
    rows, oks = [], {}
    for p in sorted(glob.glob(os.path.join(res, "**", "*.csv"), recursive=True)):
        b = os.path.basename(p)
        if b.endswith(".probe.csv") or os.sep + "tables" + os.sep in p:
            continue
        if not os.path.exists(p[:-4] + ".meta.json"):
            continue
        try:
            r, ok = run_metrics(p, nboot)
        except Exception as e:  # noqa: BLE001
            print(f"WARNING: {p}: {e}", file=sys.stderr)
            continue
        rows.append(r)
        oks[r["tag"]] = ok
    return pd.DataFrame(rows), oks


def parse_trtexec(res):
    rows = []
    for j in sorted(glob.glob(os.path.join(res, "**", "*.json"), recursive=True)):
        if j.endswith(".meta.json"):
            continue
        base = j[:-5]
        try:
            t = json.load(open(j))
        except ValueError:
            continue
        if not isinstance(t, list) or not t or "computeMs" not in t[0]:
            continue
        d = pd.DataFrame(t)
        r = dict(tag=os.path.basename(base), **read_cond(base + ".cond"))
        r.update(n=len(d), compute_mean=d["computeMs"].mean(), compute_p99=d["computeMs"].quantile(.99),
                 latency_mean=d["latencyMs"].mean() if "latencyMs" in d else np.nan)
        if "startComputeMs" in d and "endComputeMs" in d and len(d) > 1:
            wall = d["endComputeMs"].iloc[-1] - d["startComputeMs"].iloc[0]
            r["duty"] = d["computeMs"].sum() / wall if wall > 0 else np.nan
            r["period_ms"] = wall / (len(d) - 1)
        r.update(parse_tegrastats(base + ".tegrastats"))
        r.update(parse_probe(base + ".probe.csv", trim_s=(2.0, 0.5)))
        rows.append(r)
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ factorial and ladder
def factorial(runs, oks, trt):
    f = runs[(runs["exp"] == "factorial") & (runs["step"] == "full")]
    out = []
    if f.empty:
        return pd.DataFrame()
    for (model, co, rel), g in f.groupby(["model", "corun", "rel"]):
        cell = {(r.cpu, r.gpu): r for r in g.itertuples()}
        need = [("lock", "lock"), ("lock", "default"), ("default", "lock"), ("default", "default")]
        if not all(k in cell for k in need):
            continue
        blocks = {f"{c}/{gp}": Blocks(oks[cell[(c, gp)].tag]) for c, gp in need}
        eff = joint_mean_effects(blocks, {
            "GPU | CPU lock": {"lock/default": 1, "lock/lock": -1},
            "GPU | CPU default": {"default/default": 1, "default/lock": -1},
            "CPU | GPU lock": {"default/lock": 1, "lock/lock": -1},
            "CPU | GPU default": {"default/default": 1, "lock/default": -1},
            "interaction": {"default/default": 1, "default/lock": -1, "lock/default": -1, "lock/lock": 1},
            "total B3-B1": {"default/default": 1, "lock/lock": -1},
        })
        b0 = trt[(trt["model"] == model) & (trt["step"] == "B0") & (trt["cpu"] == "jc")] if len(trt) else []
        b0v = float(b0["latency_mean"].iloc[0]) if len(b0) else np.nan
        row = dict(model=model, corun=co, rel=rel, B0=b0v)
        for (c, gp), lbl in zip(need, ["B1", "B2", "B2alt(cpu default, gpu lock)", "B3"]):
            r = cell[(c, gp)]
            row[lbl] = r.mean
            row[lbl + "_p99"] = r.p99
            row[lbl + "_inf"] = r.g_inf
            row[lbl + "_dec"] = r.dec
        for k, (e, lo, hi) in eff.items():
            row[k] = e
            row[k + " CI"] = f"[{lo:.1f}, {hi:.1f}]"
        row["B3/B0"] = row["B3"] / b0v if b0v == b0v else np.nan
        row["B3/B1"] = row["B3"] / row["B1"]
        out.append(row)
    return pd.DataFrame(out)


# ------------------------------------------------------------------ sustainable rate
def rates(res, runs):
    rows = []
    for f in sorted(glob.glob(os.path.join(res, "**", "*.rate"), recursive=True)):
        base = os.path.basename(f)[:-5]
        kv = dict(t.split("=", 1) for t in open(f).read().split() if "=" in t)
        cond = {}
        cands = sorted(glob.glob(os.path.join(os.path.dirname(f), base + "_fps*.cond"))) + \
            sorted(glob.glob(os.path.join(os.path.dirname(f), "quick", base + "_q*.cond")))
        if cands:
            cond = {k: v for k, v in read_cond(cands[0]).items() if k not in ("fps", "quick", "probe")}
        r = dict(base=base, **cond, ok=kv.get("ok"), fail=kv.get("fail"),
                 mr_ok=kv.get("mr_ok"), mr_fail=kv.get("mr_fail"))
        okrun = runs[runs["tag"] == f"{base}_fps{kv.get('ok')}"] if len(runs) else []
        if len(okrun):
            o = okrun.iloc[0]
            for c in ["power_W", "good", "energy_J_valid", "mean", "p99", "gpu_mhz_mean", "gpu_util"]:
                r[c + "_at_ok"] = o.get(c, np.nan)
        rows.append(r)
    d = pd.DataFrame(rows)
    if d.empty:
        return d
    num = lambda s: pd.to_numeric(s.astype(str).str.replace("<", "").str.replace(">", ""), errors="coerce")
    d["ok_n"], d["fail_n"] = num(d["ok"]), num(d["fail"])
    # locked -> factory reduction within the same group: true rate lies in [ok, fail)
    keys = [k for k in ["exp", "model", "pipeline", "variant", "sync", "sched", "idle", "obs", "mode", "plat"]
            if k in d.columns]
    d["reduction"] = ""
    if {"cpu", "gpu"} <= set(d.columns):
        for _, g in d.groupby(keys, dropna=False):
            L = g[(g.cpu == "lock") & (g.gpu == "lock")]
            if L.empty:
                continue
            okL, failL = L.ok_n.iloc[0], L.fail_n.iloc[0]
            censL = str(L["fail"].iloc[0]).startswith(">")   # locked rate has only a lower bound
            for i, r in g.iterrows():
                if r.cpu == "lock" and r.gpu == "lock":
                    continue
                if str(r["fail"]).startswith(">"):
                    d.loc[i, "reduction"] = "n/a (rate above the cap)"
                    continue
                lo = 1 - r.fail_n / okL   # smallest possible reduction
                if censL:
                    d.loc[i, "reduction"] = f">= {max(lo, 0) * 100:.0f}% (locked >= {okL:.0f} fps)"
                    continue
                hi = 1 - r.ok_n / failL   # largest possible reduction
                mid = 1 - r.ok_n / okL
                d.loc[i, "reduction"] = f"{mid * 100:.0f}% [{max(lo, 0) * 100:.0f}, {hi * 100:.0f}]"
    return d


def observer(runs, oks, nboot):
    """Effect of each monitor on the mean response time, against the condition without a monitor
    (same frames, joint block bootstrap)."""
    e = runs[(runs.exp == "observer") & (runs["rel"] == "periodic")] if "obs" in runs else runs.iloc[0:0]
    rows = []
    for (model, cpu), g in e.groupby(["model", "cpu"]):
        ref = g[g.obs == "none"]
        if ref.empty:
            continue
        rt = ref.tag.iloc[0]
        for _, r in g.iterrows():
            row = dict(model=model, cpu=cpu, obs=r.obs, mean=r["mean"], p99=r["p99"], dec=r.get("dec"),
                       inf=r.get("inf"), gpu_mhz=r.get("pr_gpu_mhz_mean", r.get("gpu_mhz_mean")),
                       cpu0_mhz=r.get("pr_cpu0_mhz_med", np.nan), power_W=r.get("power_W"))
            if r.obs != "none" and r.tag in oks and rt in oks:
                eff = joint_mean_effects({"x": Blocks(oks[r.tag]), "ref": Blocks(oks[rt])},
                                         {"d": {"x": 1, "ref": -1}}, nb=max(nboot, 200))
                if "d" in eff:
                    v, lo, hi = eff["d"]
                    row.update(delta_mean=v, delta_ci=f"[{lo:.1f}, {hi:.1f}]")
            rows.append(row)
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ sanity-check figures
def figs(runs, trt, out):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    sw = runs[(runs.exp == "rates") & runs.tag.str.contains("_sweep|_fps")].copy()
    if len(sw):
        sw["fps"] = pd.to_numeric(sw["fps"], errors="coerce")
        for (model, pipe), g in sw.groupby(["model", "pipeline"]):
            fig, ax = plt.subplots(2, 1, figsize=(6, 6), sharex=True)
            for (c, gp), h in g.groupby(["cpu", "gpu"]):
                h = h.sort_values("fps")
                ax[0].plot(h.fps, h.p99, "o-", label=f"CPU {c} / GPU {gp}")
                ax[1].plot(h.fps, h.mr * 100, "o-")
            ax[0].plot(g.fps.sort_values(), 1000 / g.fps.sort_values(), "k--", lw=.8, label="period")
            ax[0].set_ylabel("P99 (ms)"); ax[1].set_ylabel("missed (%)"); ax[1].set_xlabel("fps")
            ax[1].axhline(1, color="gray", lw=.6); ax[0].legend(fontsize=7)
            fig.tight_layout(); fig.savefig(os.path.join(out, f"rates_{model}_{pipe}.png"), dpi=150); plt.close(fig)
    if len(trt):
        t = trt[trt["exp"] == "trtexec_idle"].copy()
        if len(t) and "pr_gpu_mhz_mean" in t:
            t["idle"] = pd.to_numeric(t["idle"], errors="coerce")
            for model, g in t.groupby("model"):
                fig, ax = plt.subplots(figsize=(6, 4))
                for (c, sp), h in g.groupby(["cpu", "spin"]):
                    h = h.sort_values("duty")
                    ax.plot(h.duty * 100, h.pr_gpu_mhz_mean, "o-", label=f"{c} spin={sp}")
                ax.set_xlabel("GPU load in trtexec (% of time computing)"); ax.set_ylabel("mean GPU frequency (MHz)")
                ax.legend(fontsize=7); ax.grid(alpha=.3); fig.tight_layout()
                fig.savefig(os.path.join(out, f"trtexec_idle_{model}.png"), dpi=150); plt.close(fig)
    inp = runs[(runs.exp == "inputs") & (runs["rel"] == "periodic")]
    if len(inp):
        for model, g in inp.groupby("model"):
            fig, ax = plt.subplots(figsize=(7, 4))
            g = g.sort_values(["variant", "cpu"])
            lbl = g.variant + "\n" + g.cpu
            bottom = np.zeros(len(g))
            for s in ["wait", "dec", "pre", "h2d", "inf", "d2h", "post"]:
                v = g[s].fillna(0).to_numpy()
                ax.bar(lbl, v, bottom=bottom, label=s); bottom += v
            ax.set_ylabel("ms per frame (12 Hz)"); ax.legend(fontsize=7, ncol=4); fig.tight_layout()
            fig.savefig(os.path.join(out, f"inputs_{model}.png"), dpi=150); plt.close(fig)


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--res", default="results")
    ap.add_argument("--nboot", type=int, default=2000)
    ap.add_argument("--no-figs", action="store_true")
    a = ap.parse_args()
    out = os.path.join(a.res, "tables")
    os.makedirs(out, exist_ok=True)

    runs, oks = collect(a.res, a.nboot)
    trt = parse_trtexec(a.res)
    keys = ["tag", "exp", "step", "model", "mode", "corun", "rel", "cpu", "gpu", "variant", "sync", "sched", "label",
            "idle", "pipeline", "fps", "spin", "obs", "plat", "tegra_ms", "bg", "idle_dis", "emc_halt", "emc_mhz", "gpu3d", "fan", "pwm",
            "state_changed", "monitor_fail", "boot"]
    for d in (runs, trt):
        for c in keys:
            if c not in d.columns:
                d[c] = np.nan
    if len(runs):
        runs.to_csv(os.path.join(out, "runs.csv"), index=False, float_format="%.4f")
    if len(trt):
        trt.to_csv(os.path.join(out, "trtexec.csv"), index=False, float_format="%.4f")
    print(f"{len(runs)} bench runs, {len(trt)} trtexec runs\n")
    # check of the platform state recorded in each run
    allr = pd.concat([runs, trt], ignore_index=True, sort=False)
    if len(allr) and "boot" in allr:
        chk = []
        num = lambda c: pd.to_numeric(allr[c], errors="coerce") if c in allr else pd.Series(np.nan, index=allr.index)
        jc = allr["cell"].astype(str).eq("jc/jc") if "cell" in allr else pd.Series(False, index=allr.index)
        idle_exp = allr["exp"].astype(str).eq("idle_states")
        for name, m in [("state changed during the run", num("state_changed").eq(1)),
                        ("CPU idle states disabled (outside 17_idle_states and jc)", num("idle_dis").gt(0) & ~jc & ~idle_exp),
                        ("EMC locked (bwmgr_halt/override.emc = 1) outside jc", num("emc_halt").eq(1) & ~jc),
                        ("GPU scaling off (enable_3d_scaling = 0)", num("gpu3d").eq(0)),
                        ("requested monitor (MONITOR=jtop) was not started: discard in 19_observer", num("monitor_fail").eq(1)),
                        ("tegrastats at 100 ms (outside 19_observer)", num("tegra_ms").eq(100) & allr["exp"].astype(str).ne("observer"))]:
            if m.any():
                chk.append(f"  {m.sum():4d} runs: {name} (e.g. {allr.loc[m, 'tag'].iloc[0]})")
        if "fan" in allr and allr["fan"].notna().any():
            chk.append("  fan: " + ", ".join(f"{k}={v}" for k, v in allr["fan"].value_counts().items()))
        # temperature: highest recorded in the campaign, and where
        if "temp_max" in allr and allr["temp_max"].notna().any():
            i = allr["temp_max"].idxmax()
            chk.append(f"  maximum temperature (tegrastats): {allr.loc[i, 'temp_max']:.1f} C in {allr.loc[i, 'tag']}")
        # throttling: with CPU and GPU locked, the frequency must stay at its maximum; below 97% of the
        # maximum observed in the locked cells of the same mode = heat or power limit
        if "cell" in allr:
            lk = allr[allr["cell"].astype(str).eq("lock/lock")]
            for col, name in [("gpu_mhz_med", "GPU"), ("cpu_mhz_med_tegra", "CPU")]:
                if col not in lk or lk[col].isna().all():
                    continue
                for mode, g in lk.groupby(lk["mode"].astype(str) if "mode" in lk else 0):
                    ref = pd.to_numeric(g[col], errors="coerce").max()
                    low = g[pd.to_numeric(g[col], errors="coerce") < 0.97 * ref]
                    if len(low):
                        chk.append(f"  {len(low):4d} locked runs ({mode}) with {name} below 97% of the maximum "
                                   f"({ref:.0f} MHz): possible throttling (e.g. {low.tag.iloc[0]})")
        boots = allr["boot"].dropna().nunique()
        chk.append(f"  distinct boots across runs: {boots}")
        print("== Platform state check")
        print("\n".join(chk), "\n")
    if runs.empty and trt.empty:
        return

    base = ["tag", "mean", "p99", "p99_lo", "p99_hi", "mr", "drop_rate", "good", "dec", "pre", "queue", "inf",
            "g_inf", "post", "gpu_share", "gpu_mhz_med", "gpu_util", "pr_gpu_mhz_mean", "pr_gpu_frac_min",
            "power_W", "energy_J_valid"]

    fac = factorial(runs, oks, trt)
    if len(fac):
        fac.to_csv(os.path.join(out, "factorial.csv"), index=False, float_format="%.3f")
        print("== Ladder and CPU x GPU factorial (mean response time in ms; effects with 95% CI)")
        print(fac.T.to_string(header=False, float_format=lambda v: f"{v:.2f}"))
        print()
    net = runs[(runs.exp == "factorial") & (runs["step"] == "NET")]
    if len(net):
        print("== Network only (pre-processed tensor), per cell")
        print(fmt(net, ["tag", "mean", "p99", "g_inf", "gpu_mhz_med", "gpu_util", "power_W"]), "\n")

    rt = rates(a.res, runs)
    if len(rt):
        rt.to_csv(os.path.join(out, "rates.csv"), index=False, float_format="%.3f")
        print("== Sustainable rate (<= 1% deadline misses, deadline = period)")
        print(fmt(rt, ["base", "ok", "fail", "reduction", "power_W_at_ok", "energy_J_valid_at_ok", "gpu_mhz_mean_at_ok"]), "\n")

    obs = observer(runs, oks, a.nboot)
    if len(obs):
        obs.to_csv(os.path.join(out, "observer_effect.csv"), index=False, float_format="%.3f")
        print("== Observer effect (mean response time in ms; delta against 'none', 95% CI)")
        print(obs.to_string(index=False, float_format=lambda v: f"{v:.2f}"), "\n")

    if len(trt):
        cols = ["tag", "idle", "spin", "duty", "compute_mean", "compute_p99", "latency_mean", "pr_gpu_mhz_mean",
                "pr_gpu_mhz_dist", "pr_gpu_load", "gpu_mhz_med", "power_W"]
        print("== trtexec (B0 e --idleTime)")
        print(fmt(trt.sort_values([c for c in ["model", "cpu", "spin"] if c in trt] + ["tag"]), cols), "\n")

    for exp, extra in [("inputs", ["variant", "cpu", "rel"]), ("sync", ["sync", "sched", "cpu", "rel"]),
                       ("floors", ["label"]), ("maxn", ["cpu", "rel"]), ("idle_states", ["idle", "cpu", "rel"]),
                       ("probe", ["pipeline", "cpu", "fps"]), ("observer", ["obs", "cpu", "rel"]), ("equiv", [])]:
        e = runs[runs.exp == exp]
        if e.empty:
            continue
        e = e.sort_values("tag")
        e.to_csv(os.path.join(out, f"{exp}.csv"), index=False, float_format="%.4f")
        print(f"== {exp}")
        print(fmt(e, base + ["pr_gpu_mhz_dist", "pr_cpu0_mhz_med"] if exp == "probe" else base), "\n")

    if not a.no_figs:
        figs(runs, trt, out)
    print(f"tables and figures in {out}")


if __name__ == "__main__":
    main()
