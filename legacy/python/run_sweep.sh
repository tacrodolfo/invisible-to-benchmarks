#!/usr/bin/env bash
# Camera rate sweep: periodic release with deadline = period,
# clocks {dvfs, fixed}, no concurrent load.
#
# Prerequisite: `sudo jetson_clocks --store` already taken in the default state.
# Usage: ./run_sweep.sh yolox_s_fp16.engine frames [passes] [n_frames]
set -euo pipefail

ENGINE=${1:?engine}
FRAMES=${2:-frames}
PASSES=${3:-3}
LIMIT=${4:-1000}
OUT="$(pwd)/sweep"
MODEL=$(basename "$ENGINE" .engine)
# periods (ms): 30, 20, 15, 12, 10 and 5 fps
PERIODS="33.3 50 66.7 83.3 100 200"

MODE=$(sudo nvpmodel -q 2>/dev/null | grep -i "power mode" | head -1 | awk -F': ' '{print $2}' | tr -d ' ')
[ -z "$MODE" ] && MODE=unknown
echo ">> mode: $MODE | passes: $PASSES | frames: $LIMIT | periods: $PERIODS"

if ! sudo test -f /root/.jetsonclocks_conf.txt; then
  echo "ERROR: run 'sudo jetson_clocks --store' in the default state first." >&2
  exit 1
fi

mkdir -p "$OUT"
trap 'sudo pkill -x tegrastats 2>/dev/null || true; sudo jetson_clocks --restore >/dev/null 2>&1 || true' EXIT

for CLOCKS in dvfs fixed; do
  if [ "$CLOCKS" = fixed ]; then sudo jetson_clocks; else sudo jetson_clocks --restore; fi
  echo ">> clocks: $CLOCKS - cooling down 30 s"; sleep 30

  for P in $PERIODS; do
    TAG="${MODEL}_${MODE}_clocks-${CLOCKS}_period-${P}"
    if [ -f "$OUT/$TAG.csv" ]; then echo "=== $TAG: already exists, skipping ==="; continue; fi
    echo "=== $TAG ==="

    sudo pkill -x tegrastats 2>/dev/null || true
    sudo tegrastats --interval 100 --logfile "$OUT/$TAG.tegrastats" &
    sleep 1

    python3 bench.py --engine "$ENGINE" --frames "$FRAMES" --mode periodic \
      --period-ms "$P" --passes "$PASSES" --limit "$LIMIT" --tag "$TAG" --outdir "$OUT"

    sudo pkill -x tegrastats 2>/dev/null || true
    sleep 1
    sudo chown "$(id -u):$(id -g)" "$OUT/$TAG.tegrastats" 2>/dev/null || true
    if [ ! -s "$OUT/$TAG.tegrastats" ]; then echo "WARNING: empty power log in $TAG" >&2; fi
    echo ">> pause 15 s"; sleep 15
  done
done
echo ">> sweep done (clocks restored to default)"
