#!/usr/bin/env bash
# Tests run_bench/find_rate and the analysis without a Jetson: fake bench (tests/mock), synthetic
# tegrastats and probe. Usage: tests/test_scripts.sh <bench_mock> <frames_folder> <output_folder>
set -euo pipefail
[ $# -eq 3 ] || { echo "usage: $0 <bench_mock> <frames_folder> <output_folder>"; exit 1; }
MOCKB=$1; FR=$2; OUTD=$3
export BENCH=$MOCKB FRAMES=$FR RES=$OUTD COOL=0 DRY=0
source "$(dirname "$0")/../scripts/lib.sh"
trap - EXIT
# replaces what depends on the Jetson
clk_show() { echo "mock"; }
tegra_start() {
  python3 - "$1" <<'PY'
import sys, time
t = time.time()
with open(sys.argv[1], "w") as f:
    for i in range(-20, 400):
        ts = time.strftime("%m-%d-%Y %H:%M:%S", time.localtime(t + i * 0.1))
        f.write(f"{ts} RAM 3000/7620MB CPU [20%@960,10%@1344,5%@729,0%@729,1%@1344,2%@1344] EMC_FREQ 0%@3199 "
                f"GR3D_FREQ {20 + i % 5}%@[305] VDD_IN {6600 + i % 7}mW/6650mW VDD_CPU_GPU_CV 900mW/900mW\n")
PY
}
tegra_stop() { :; }
probe_start() {
  python3 - "$1" <<'PY'
import sys, time
t0 = time.clock_gettime_ns(time.CLOCK_MONOTONIC)
with open(sys.argv[1], "w") as f:
    f.write("t_ns,gpu_hz,gpu_load,cpu0_khz,cpu4_khz,vin_mw\n")
    for i in range(60000):
        f.write(f"{t0 + i * 1000000},{305000000 if i % 10 else 714000000},{230 if i % 3 else 600},960000,1344000,{6600 if i % 10 == 0 else -1}\n")
PY
}
probe_stop() { :; }
E=$OUTD/fake.engine; echo x > "$E"
CUR_CELL="default/default"
run_bench "$OUTD/factorial" yolox_s_cpu-default_gpu-default_co-off_periodic "exp=factorial model=yolox_s mode=25W step=full cpu=default gpu=default rel=periodic" --engine "$E" --mode periodic --period-ms 50 --passes 2 --warmup 5
for c in "lock lock" "lock default" "default lock"; do set -- $c
  MOCK_INF_MS=$([ "$2" = lock ] && echo 4 || echo 8) run_bench "$OUTD/factorial" "yolox_s_cpu-$1_gpu-$2_co-off_periodic" "exp=factorial model=yolox_s mode=25W step=full cpu=$1 gpu=$2 rel=periodic" --engine "$E" --mode periodic --period-ms 50 --passes 2 --warmup 5
done
PROBE_RUN=1 run_bench "$OUTD/probe" yolox_s_seq_cpu-default_gpu-default_probe12 "exp=probe model=yolox_s mode=25W pipeline=seq cpu=default gpu=default fps=12 probe=1" --engine "$E" --mode periodic --period-ms 40 --passes 1 --warmup 5
for c in "lock lock" "default default"; do set -- $c
  MOCK_INF_MS=$([ "$1" = lock ] && echo 4 || echo 15) QUICK_LIMIT=60 FULL_LIMIT=120 FULL_PASSES=1 \
    find_rate "$OUTD/rates" "yolox_s_seq_cpu-$1_gpu-$2" "exp=rates model=yolox_s mode=25W pipeline=seq cpu=$1 gpu=$2" 8 60 --engine "$E" --warmup 5
done
# fake trtexec
mkdir -p "$OUTD/trtexec_idle"
for I in 0 10 76; do
  T="$OUTD/trtexec_idle/yolox_s_trt_cpu-default_gpu-default_idle$I"
  echo "exp=trtexec_idle model=yolox_s mode=25W cpu=default gpu=default idle=$I spin=0" > "$T.cond"
  python3 - "$T.json" "$I" <<'PY'
import json, sys
idle = float(sys.argv[2]); c = 7 if idle < 5 else 19.5; t = 0; rows = []
for i in range(200):
    rows.append(dict(startEnqMs=t, endEnqMs=t + .1, startComputeMs=t, endComputeMs=t + c, computeMs=c, latencyMs=c + .4, h2dMs=.3, d2hMs=.1))
    t += c + idle
json.dump(rows, open(sys.argv[1], "w"))
PY
  probe_start "$T.probe.csv"
done
echo "TEST OK"
