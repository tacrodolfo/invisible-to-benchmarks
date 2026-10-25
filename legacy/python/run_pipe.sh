#!/usr/bin/env bash
# Parallel pipeline (2 stages): clocks {dvfs, fixed} x {back-to-back, periodic at 5-30 fps}.
# Same rates, frames and passes as the sequential sweep (sweep/), for direct comparison.
#
# Prerequisite: 25 W mode and `sudo jetson_clocks --store` already taken in the default state.
# Usage: ./run_pipe.sh yolox_s_fp16.engine frames [passes] [n_frames]
set -euo pipefail

ENGINE=${1:?engine}
FRAMES=${2:-frames}
PASSES=${3:-3}
LIMIT=${4:-1000}
OUT="$(pwd)/results_pipe"
MODEL=$(basename "$ENGINE" .engine)
PERIODS="33.3 50 66.7 83.3 100 200"
mkdir -p "$OUT"

MODE=$(sudo nvpmodel -q 2>/dev/null | grep -i "power mode" | head -1 | awk -F': ' '{print $2}' | tr -d ' ')
[ "$MODE" != "25W" ] && { echo "ERROR: current mode is $MODE; run 'sudo nvpmodel -m 1 && sudo jetson_clocks --restore'" >&2; exit 1; }

trap 'sudo pkill -x tegrastats 2>/dev/null || true; sudo jetson_clocks --restore >/dev/null 2>&1 || true; echo ">> default state restored"' EXIT

run() {  # $1 tag, $2 mode, $3 period
  local TAG="${MODEL}_25W_pipe_$1"
  if [ -f "$OUT/$TAG.csv" ]; then echo "=== $TAG: already exists, skipping ==="; return; fi
  echo "=== $TAG ==="
  sudo pkill -x tegrastats 2>/dev/null || true
  sudo tegrastats --interval 100 --logfile "$OUT/$TAG.tegrastats" &
  sleep 1
  python3 bench_pipe.py --engine "$ENGINE" --frames "$FRAMES" --mode "$2" --period-ms "$3" \
    --passes "$PASSES" --limit "$LIMIT" --tag "$TAG" --outdir "$OUT"
  sudo pkill -x tegrastats 2>/dev/null || true
  sleep 1
  sudo chown "$(id -u):$(id -g)" "$OUT/$TAG.tegrastats" 2>/dev/null || true
  sleep 10
}

for CLOCKS in dvfs fixed; do
  if [ "$CLOCKS" = fixed ]; then sudo jetson_clocks; else sudo jetson_clocks --restore; fi
  echo ">> clocks: $CLOCKS"
  sudo jetson_clocks --show | grep -E "cpu0|GPU MinFreq"
  echo ">> cooling down 30 s"; sleep 30
  run "clocks-${CLOCKS}_b2b" b2b 83.3
  for P in $PERIODS; do
    run "clocks-${CLOCKS}_period-${P}" periodic "$P"
  done
done
echo ">> done"
