#!/usr/bin/env bash
# CPU idle states enabled vs disabled (Section "Sustainable Camera Rate").
# An earlier jetson_clocks had left the idle states disabled. This script measures B1 and B3 in both
# situations, to show how much this affects the results, whatever the factory default is
# (confirm the default with ./scripts/00_report.sh boot right after a reboot).
#
# Usage: ./scripts/17_idle_states.sh s|m|l
source "$(dirname "$0")/lib.sh"
M=${1:?model s|m|l}
check_built; assert_mode "$BASE_MODE"; clk_save

ENG=$(engine_of "$M"); MODEL=$(model_of "$M")
OUT="$RES/idle_states"
L=${IDLE_LIMIT:-1000}; P=${IDLE_PASSES:-2}
B="exp=idle_states model=$MODEL mode=$BASE_MODE"

for IDLE in enable disable; do
  idle_set "$IDLE"
  for CELL in "lock lock" "default default"; do
    set -- $CELL; CPU=$1; GPU=$2
    CUR_CELL=""; set_cell "$CPU" "$GPU"
    idle_set "$IDLE"   # cpu_set/gpu_set do not touch the idle states, but make sure
    T="${MODEL}_idle-${IDLE}_cpu-${CPU}_gpu-${GPU}"
    C="$B idle=$IDLE cpu=$CPU gpu=$GPU"
    run_bench "$OUT" "${T}_periodic" "$C rel=periodic fps=$P_FPS" --engine "$ENG" --mode periodic --period-ms "$P_MS" --limit "$L" --passes "$P"
    run_bench "$OUT" "${T}_b2b" "$C rel=b2b" --engine "$ENG" --mode b2b --limit "$L" --passes "$P"
    if [ "$CPU" = default ]; then
      find_rate "$OUT" "$T" "$C" "$RATE_LO" "$RATE_HI" --engine "$ENG"
      PROBE_RUN=1 run_bench "$OUT/probe" "${T}_probe${P_FPS}" "$C rel=periodic fps=$P_FPS probe=1" \
        --engine "$ENG" --mode periodic --period-ms "$P_MS" --limit 300 --passes 1
    fi
  done
done
log "17_idle_states $MODEL done (idle states return to the saved state)"
