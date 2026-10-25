#!/usr/bin/env bash
# Intermediate alternatives to locking everything (frequency floors; Sections "Sustainable Camera Rate" and "Energy").
#   GPU floor: devfreq min_freq at each available frequency, with nvhost_podgov active
#              and the CPU at its factory default (the governor can still go higher).
#   CPU floor: scaling_min_freq at a few steps, with the GPU at its default.
#   Combination: GPU floor + locked CPU (the gain left with the CPU at maximum).
# At each point: periodic at 12 Hz (time, P99, misses, power) and sustainable rate.
# The frequency vs sustainable rate vs power curve comes out of the analysis.
#
# Usage: ./scripts/15_floors.sh s|m|l
#   GPU_FLOORS="..." (Hz; default: all available)   CPU_FLOORS="..." (kHz; default: 2 steps)
#   FIND_RATE=1   COMBO=1 (GPU floor with locked CPU; off by default)
source "$(dirname "$0")/lib.sh"
M=${1:?model s|m|l}
check_built; assert_mode "$BASE_MODE"; clk_save

ENG=$(engine_of "$M"); MODEL=$(model_of "$M")
OUT="$RES/floors"
FIND=${FIND_RATE:-1}
L=${FLOOR_LIMIT:-1000}; P=${FLOOR_PASSES:-2}
B="exp=floors model=$MODEL mode=$BASE_MODE"

if [ "$DRY" = 1 ]; then
  GPU_FLOORS=${GPU_FLOORS:-"408000000 510000000 612000000 714000000 816000000"}
  CPU_FLOORS=${CPU_FLOORS:-"960000 1190400"}
else
  GMAX=$(saved "$GPU_DEV/max_freq"); GMIN=$(saved "$GPU_DEV/min_freq")
  GPU_FLOORS=${GPU_FLOORS:-$(tr ' ' '\n' < "$GPU_DEV/available_frequencies" | grep -v '^$' | sort -n | \
    awk -v lo="$GMIN" -v hi="$GMAX" '$1>lo && $1<hi' | tr '\n' ' ')}
  P0=$(echo "$POLICIES" | head -1)
  CMAX=$(saved "$P0/scaling_max_freq"); CMIN=$(saved "$P0/scaling_min_freq")
  if [ -z "${CPU_FLOORS:-}" ]; then
    CF=$(tr ' ' '\n' < "$P0/scaling_available_frequencies" | grep -v '^$' | sort -n | \
      awk -v lo="$CMIN" -v hi="$CMAX" '$1>lo && $1<hi')
    N=$(echo "$CF" | wc -l)
    CPU_FLOORS=$(echo "$CF" | awk -v n="$N" 'NR==int(n/2)+1 || NR==int(3*n/4)+1' | tr '\n' ' ')
  fi
fi
echo ">> GPU floors (Hz): $GPU_FLOORS"
echo ">> CPU floors (kHz): $CPU_FLOORS"

point() {  # point <cpu> <gpu> <label> [rate search 0|1]
  local CPU=$1 GPU=$2 LBL=$3 F=${4:-$FIND}
  set_cell "$CPU" "$GPU"
  local T="${MODEL}_${LBL}" C="$B cpu=$CPU gpu=$GPU label=$LBL"
  run_bench "$OUT" "${T}_periodic" "$C rel=periodic fps=$P_FPS" --engine "$ENG" --mode periodic --period-ms "$P_MS" --limit "$L" --passes "$P"
  if [ "$F" = 1 ]; then find_rate "$OUT" "$T" "$C" "$RATE_LO" "$RATE_HI" --engine "$ENG"; fi
}

# extremes: only the 12 Hz point (their sustainable rate comes from 11_rates)
point default default "cpu-default_gpu-default" 0
point lock lock "cpu-lock_gpu-lock" 0
for G in $GPU_FLOORS; do point default "floor:$G" "cpu-default_gpu-floor$((G / 1000000))"; done
for Cf in $CPU_FLOORS; do point "floor:$Cf" default "cpu-floor$((Cf / 1000))_gpu-default"; done
if [ "${COMBO:-0}" = 1 ]; then
  for G in $GPU_FLOORS; do point lock "floor:$G" "cpu-lock_gpu-floor$((G / 1000000))"; done
fi
log "15_floors $MODEL done"
