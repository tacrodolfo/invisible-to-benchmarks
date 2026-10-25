"""Parsing and metrics shared by the C++ harness analyses (also compatible with the bench.py CSV)."""
import datetime as dt
import json
import os
import re

import numpy as np
import pandas as pd

BLOCK = 50  # consecutive frames per bootstrap block (all passes of a block kept together)
RNG = np.random.default_rng(0)
STAGES = ["dec", "pre", "queue", "h2d", "inf", "d2h", "post"]


# ------------------------------------------------------------------ files of one run
def read_cond(path):
    d = {}
    if os.path.exists(path):
        for tok in open(path).read().split():
            if "=" in tok:
                k, v = tok.split("=", 1)
                d[k] = v
    return d


def read_meta(path):
    try:
        return json.load(open(path))
    except (OSError, ValueError):
        return {}


def load_csv(path):
    df = pd.read_csv(path)
    if "t_deq" not in df.columns:  # bench.py CSV
        df["t_deq"] = df["t_pre"]
    for c in ["g_dec", "g_pre", "g_h2d", "g_inf", "g_d2h"]:
        if c not in df.columns:
            df[c] = 0
    return df


def add_times(df, sync="stage"):
    """Columns in ms: resp, svc, wait and one time per stage. With sync=edge, the GPU stages use
    the CUDA events (g_*), because the host timestamps only mark enqueueing."""
    ok = df[df["drop"] == 0].copy()
    ms = lambda a, b: (ok[b] - ok[a]) / 1e6
    ok["resp"] = ms("t_rel", "t_end")
    ok["svc"] = ms("t_start", "t_end")
    ok["wait"] = ms("t_rel", "t_start")
    ok["dec"] = ms("t_start", "t_dec")
    ok["pre"] = ms("t_dec", "t_pre")
    ok["queue"] = ms("t_pre", "t_deq")
    if sync == "edge":
        ok["h2d"] = ok["g_h2d"] / 1e6
        ok["inf"] = ok["g_inf"] / 1e6
        ok["d2h"] = ms("t_deq", "t_d2h") - ok["h2d"] - ok["inf"]
    else:
        ok["h2d"] = ms("t_deq", "t_h2d")
        ok["inf"] = ms("t_h2d", "t_inf")
        ok["d2h"] = ms("t_inf", "t_d2h")
    ok["post"] = ms("t_d2h", "t_end")
    ok["g_gpu"] = (ok["g_dec"] + ok["g_pre"] + ok["g_h2d"] + ok["g_inf"] + ok["g_d2h"]) / 1e6
    ok["blk"] = ok["frame"] // BLOCK
    return ok


# ------------------------------------------------------------------ block bootstrap
class Blocks:
    """Response values grouped by frame block, for a fast bootstrap."""

    def __init__(self, ok, col="resp"):
        g = ok.groupby("blk")[col]
        self.ids = np.array(sorted(g.groups.keys()))
        self.vals = [g.get_group(b).to_numpy() for b in self.ids]
        self.sums = np.array([v.sum() for v in self.vals])
        self.cnts = np.array([len(v) for v in self.vals])

    def p_ci(self, q=99, nb=500):
        n = len(self.ids)
        if n < 2:
            return np.nan, np.nan
        pick = RNG.integers(0, n, (nb, n))
        qs = [np.percentile(np.concatenate([self.vals[i] for i in row]), q) for row in pick]
        return tuple(np.percentile(qs, [2.5, 97.5]))


def joint_mean_effects(blocks, combos, nb=1000):
    """blocks: dict name -> Blocks (same frames). combos: dict effect -> {name: weight}.
    Joint bootstrap (the same blocks are drawn for every run). Returns
    {effect: (estimate, ci_lo, ci_hi)}."""
    common = set.intersection(*[set(b.ids) for b in blocks.values()])
    common = np.array(sorted(common))
    if len(common) < 2:
        return {}
    idx = {k: np.searchsorted(b.ids, common) for k, b in blocks.items()}
    pick = RNG.integers(0, len(common), (nb, len(common)))
    means_b, means = {}, {}
    for k, b in blocks.items():
        s, c = b.sums[idx[k]], b.cnts[idx[k]]
        means[k] = s.sum() / c.sum()
        means_b[k] = s[pick].sum(1) / c[pick].sum(1)
    out = {}
    for e, w in combos.items():
        est = sum(wt * means[k] for k, wt in w.items())
        bs = sum(wt * means_b[k] for k, wt in w.items())
        lo, hi = np.percentile(bs, [2.5, 97.5])
        out[e] = (est, lo, hi)
    return out


# ------------------------------------------------------------------ tegrastats
_TS = re.compile(r"^(\d\d-\d\d-\d{4} \d\d:\d\d:\d\d)")
_GR3D = re.compile(r"GR3D_FREQ (\d+)%(?:@\[?(\d+))?")
_VIN = re.compile(r"(?:VDD_IN|POM_5V_IN) (\d+)(?:mW)?/")  # Orin (JetPack 6) | Nano (JetPack 4)
_CPU = re.compile(r"CPU \[([^\]]+)\]")
_EMC = re.compile(r"EMC_FREQ (\d+)%(?:@(\d+))?")
_TEMP = re.compile(r"\b([A-Za-z][A-Za-z0-9_]*)@(-?\d+(?:\.\d+)?)C\b")          # cpu@46.0C, tj@46C, GPU@31.5C
_RAIL = re.compile(r"\b(VDD_[A-Z0-9_]+|POM_5V_[A-Z0-9_]+) (\d+)(?:mW)?/")      # VDD_SOC 2290mW/..., POM_5V_GPU 0/0
# zones that are not silicon temperatures (the Nano PMIC reads a fixed 50 C)
_TEMP_SKIP = {"pmic", "ao", "pll"}


def parse_tegrastats(path, real_window=None, interval_s=None):
    """tegrastats averages. If real_window=(t0_ns, t1_ns) in real time (UTC), only lines inside the
    measurement window are used. Time of each line: tegrastats' own timestamp (JetPack 6; 1 s
    resolution, lines within the same second are offset by interval_s) or, without timestamps
    (JetPack 4), the start recorded in <file>.t0 + k * interval_s.
    interval_s: the one in the .cond (tegra_ms) or 0.1 s (older campaigns)."""
    if not path or not os.path.exists(path):
        return {}
    if interval_s is None:
        c = read_cond(path[: -len(".tegrastats")] + ".cond") if path.endswith(".tegrastats") else {}
        interval_s = float(c.get("tegra_ms", 100)) / 1000
    t_start = np.nan
    if os.path.exists(path + ".t0"):
        try:
            t_start = float(open(path + ".t0").read().split()[0])
        except (ValueError, IndexError):
            pass
    rows, seen, k_line = [], {}, 0
    for line in open(path, errors="ignore"):
        m = _GR3D.search(line)
        if not m:
            continue
        t = np.nan
        ts = _TS.match(line)
        if ts:
            k = ts.group(1)
            j = seen.get(k, 0)
            seen[k] = j + 1
            t = dt.datetime.strptime(k, "%m-%d-%Y %H:%M:%S").timestamp() + j * interval_s
        elif not np.isnan(t_start):
            t = t_start + k_line * interval_s
        k_line += 1
        cpu = _CPU.search(line)
        mhz = [int(x.split("@")[1]) for x in cpu.group(1).split(",") if "@" in x] if cpu else []
        vin = _VIN.search(line)
        emc = _EMC.search(line)
        row = dict(t=t, gpu_util=int(m.group(1)), gpu_mhz=float(m.group(2)) if m.group(2) else np.nan,
                   vin_mw=int(vin.group(1)) if vin else np.nan, cpu_mhz=np.mean(mhz) if mhz else np.nan,
                   emc_mhz=float(emc.group(2)) if emc and emc.group(2) else np.nan)
        for name, v in _TEMP.findall(line):
            if name.lower() not in _TEMP_SKIP:
                row["T_" + name.lower()] = float(v)
        for name, v in _RAIL.findall(line):
            row["P_" + name] = int(v)
        rows.append(row)
    if not rows:
        return {}
    d = pd.DataFrame(rows)
    win = "all"
    if real_window and d["t"].notna().any():
        t0, t1 = real_window[0] / 1e9, real_window[1] / 1e9
        sub = d[(d["t"] >= t0) & (d["t"] <= t1)]
        if len(sub) >= 3:
            d, win = sub, "meas"
    out = dict(power_W=d["vin_mw"].mean() / 1000, gpu_util=d["gpu_util"].mean(),
               gpu_mhz_med=d["gpu_mhz"].median(), gpu_mhz_mean=d["gpu_mhz"].mean(),
               cpu_mhz_med_tegra=d["cpu_mhz"].median(), emc_mhz=d["emc_mhz"].median(),
               tegra_n=len(d), tegra_window=win, tegra_ms=interval_s * 1000)
    # temperatures (C): mean and max per zone; temp_max = highest among the silicon zones
    tcols = [c for c in d.columns if c.startswith("T_")]
    for c in tcols:
        out[f"temp_{c[2:]}_mean"] = d[c].mean()
        out[f"temp_{c[2:]}_max"] = d[c].max()
    if tcols:
        out["temp_max"] = d[tcols].max().max()
    # power rails (W, mean over the window)
    for c in [c for c in d.columns if c.startswith("P_")]:
        out[f"rail_{c[2:]}_W"] = d[c].mean() / 1000
    return out


# ------------------------------------------------------------------ 1 ms probe
def parse_probe(path, mono_window=None, trim_s=(0.0, 0.0)):
    if not path or not os.path.exists(path):
        return {}
    p = pd.read_csv(path)
    if len(p) == 0:
        return {}
    if mono_window:
        p = p[(p["t_ns"] >= mono_window[0]) & (p["t_ns"] <= mono_window[1])]
    elif any(trim_s):
        t0, t1 = p["t_ns"].min() + trim_s[0] * 1e9, p["t_ns"].max() - trim_s[1] * 1e9
        p = p[(p["t_ns"] >= t0) & (p["t_ns"] <= t1)]
    if len(p) == 0:
        return {}
    g = p["gpu_hz"] / 1e6
    out = dict(pr_n=len(p), pr_gpu_mhz_mean=g.mean(), pr_gpu_mhz_med=g.median(), pr_gpu_mhz_max=g.max(),
               pr_gpu_mhz_min=g.min(), pr_gpu_load=p["gpu_load"].mean() / 10 if (p["gpu_load"] >= 0).any() else np.nan)
    vc = g.round().value_counts(normalize=True).sort_index()
    out["pr_gpu_mhz_dist"] = " ".join(f"{int(k)}:{v * 100:.1f}%" for k, v in vc.items() if v >= 0.005)
    out["pr_gpu_frac_min"] = float((g <= g.min() + 1).mean())
    for c in p.columns:
        if c.startswith("cpu") and c.endswith("_khz"):
            out[f"pr_{c[:-4]}_mhz_med"] = p[c].median() / 1000
    if "vin_mw" in p.columns and (p["vin_mw"] > 0).any():
        out["pr_power_W"] = p.loc[p["vin_mw"] > 0, "vin_mw"].mean() / 1000
    if "emc_hz" in p.columns and (p["emc_hz"] > 0).any():
        e = p.loc[p["emc_hz"] > 0, "emc_hz"] / 1e6
        out["pr_emc_mhz_mean"] = e.mean()
        out["pr_emc_mhz_min"] = e.min()
    return out


# ------------------------------------------------------------------ metrics of one run
def run_metrics(csv_path, nboot=500):
    base = csv_path[:-4]
    meta = read_meta(base + ".meta.json")
    cond = read_cond(base + ".cond")
    args = meta.get("args", {})
    sync = args.get("sync", cond.get("sync", "stage"))
    df = load_csv(csv_path)
    ok = add_times(df, sync)
    period = float(args.get("period-ms", 83.333))
    D = float(args.get("deadline-ms", period))
    n, drops = len(df), int(df["drop"].sum())
    x = ok["resp"].to_numpy()
    late = int((x > D).sum())
    t0 = df["t_rel"][df["t_rel"] > 0].min()
    t1 = df["t_end"][df["t_end"] > 0].max()
    dur = (t1 - t0) / 1e9
    r = dict(tag=os.path.basename(base), path=csv_path, **cond)
    r.update(n=n, drops=drops, late=late, deadline_ms=D, mr=(late + drops) / n, drop_rate=drops / n,
             thr=len(x) / dur, good=(len(x) - late) / dur, dur_s=dur)
    if len(x):
        b = Blocks(ok)
        lo, hi = b.p_ci(99, nboot) if nboot else (np.nan, np.nan)
        r.update(mean=x.mean(), p50=np.median(x), p95=np.percentile(x, 95), p99=np.percentile(x, 99),
                 p99_lo=lo, p99_hi=hi, max=x.max(), svc=ok["svc"].mean(), wait=ok["wait"].mean())
        for s in STAGES:
            r[s] = ok[s].mean()
        for g in ["g_dec", "g_pre", "g_h2d", "g_inf", "g_d2h"]:
            r[g] = ok[g].mean() / 1e6
        r["gpu_share"] = (ok["g_gpu"] / ok["svc"]).mean()
        r["n_det_mean"] = ok["n_det"].mean()
    for k in ["pipeline", "input", "decode", "pre", "post", "sync", "sched", "wait", "mode", "passes", "limit"]:
        if k in args:
            r.setdefault("a_" + k, args[k])
    r["period_ms"] = period
    win_real = meta.get("meas_real_ns")
    r.update(parse_tegrastats(base + ".tegrastats", tuple(win_real) if win_real else None))
    if len(x):
        r.update(parse_probe(base + ".probe.csv", (int(t0), int(t1))))
    if r.get("power_W") and r.get("good"):
        r["energy_J_valid"] = r["power_W"] / r["good"] if r["good"] > 0 else np.nan
    return r, ok
