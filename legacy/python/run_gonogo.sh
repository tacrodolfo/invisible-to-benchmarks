#!/usr/bin/env bash
# Go/no-go matrix in the CURRENT power mode of the Orin Nano:
#   clocks {dvfs, fixed} x memory stress {off, on} x release {b2b, periodic}
#
# Prerequisite: `sudo jetson_clocks --store` taken with the system in its default state
# (preferably right after a reboot), so that `--restore` returns to the real DVFS.
#
# Usage: ./run_gonogo.sh yolox_s_fp16.engine frames [passes]
set -euo pipefail

ENGINE=${1:?engine}
FRAMES=${2:-frames}
PASSES=${3:-3}
OUT="$(pwd)/results"   # absolute: tegrastats --start runs from /
PERIOD_MS=83   # 12 Hz of the nuScenes CAM_FRONT
MODEL=$(basename "$ENGINE" .engine)

MODE=$(sudo nvpmodel -q 2>/dev/null | grep -i "power mode" | head -1 | awk -F': ' '{print $2}' | tr -d ' ')
[ -z "$MODE" ] && MODE=unknown
echo ">> current power mode: $MODE | passes: $PASSES"

if ! sudo test -f /root/.jetsonclocks_conf.txt; then
  echo "ERROR: run 'sudo jetson_clocks --store' in the default state first." >&2
  exit 1
fi

mkdir -p "$OUT"
trap 'pkill -KILL -x stress-ng 2>/dev/null || true; sudo pkill -x tegrastats 2>/dev/null || true; sudo jetson_clocks --restore >/dev/null 2>&1 || true' EXIT

for CLOCKS in dvfs fixed; do
  if [ "$CLOCKS" = fixed ]; then
    sudo jetson_clocks
  else
    sudo jetson_clocks --restore
  fi
  sudo jetson_clocks --show > "$OUT/${MODEL}_${MODE}_clocks-${CLOCKS}.clocks"
  echo ">> clocks: $CLOCKS - cooling down 30 s"; sleep 30

  for STRESS in off on; do
    for REL in b2b periodic; do
      TAG="${MODEL}_${MODE}_clocks-${CLOCKS}_stress-${STRESS}_${REL}"
      if [ -f "$OUT/$TAG.csv" ]; then
        echo "=== $TAG: already exists, skipping ==="
        continue
      fi
      echo "=== $TAG ==="

      SPID=""
      if [ "$STRESS" = on ]; then
        stress-ng --vm 2 --vm-bytes 1G --vm-method all -t 0 >/dev/null 2>&1 &
        SPID=$!
        sleep 5
      fi

      sudo pkill -x tegrastats 2>/dev/null || true
      sudo tegrastats --interval 100 --logfile "$OUT/$TAG.tegrastats" &
      sleep 1
      cat /sys/devices/virtual/thermal/thermal_zone*/temp 2>/dev/null | head -3 > "$OUT/$TAG.temp_start" || true

      python3 bench.py --engine "$ENGINE" --frames "$FRAMES" --mode "$REL" \
        --period-ms "$PERIOD_MS" --passes "$PASSES" --tag "$TAG" --outdir "$OUT"

      cat /sys/devices/virtual/thermal/thermal_zone*/temp 2>/dev/null | head -3 > "$OUT/$TAG.temp_end" || true
      sudo pkill -x tegrastats 2>/dev/null || true
      sleep 1
      sudo chown "$(id -u):$(id -g)" "$OUT/$TAG.tegrastats" 2>/dev/null || true
      if [ ! -s "$OUT/$TAG.tegrastats" ]; then echo "WARNING: empty power log in $TAG" >&2; fi

      if [ -n "$SPID" ]; then
        # SIGTERM: background processes of a script ignore SIGINT
        kill -TERM "$SPID" 2>/dev/null || true
        sleep 2
        pkill -KILL -x stress-ng 2>/dev/null || true
        wait "$SPID" 2>/dev/null || true
      fi
      echo ">> pause 20 s"; sleep 20
    done
  done
done
echo ">> done for mode $MODE (clocks restored to default)"
