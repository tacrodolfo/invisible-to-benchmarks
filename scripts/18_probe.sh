#!/usr/bin/env bash
# 1 ms sampling for the frequency figures (e.g. Fig. 3).
# Short, separate runs: latency results come from runs without the probe.
#   seq and pipe, factory, at 12, 17/20 and 30 fps; back-to-back; and locked at 12 fps.
#   Also measures how much the probe perturbs (same condition with and without the probe).
#
# Usage: ./scripts/18_probe.sh s|m|l       PROBE_RATES="12 20 30"
# The probe wakes the CPU every 1 ms and schedutil reacts (on Orin, 48.3 -> 41.7 ms in the
# factory default): use these runs only for the GPU frequency, never for latency.
source "$(dirname "$0")/lib.sh"
M=${1:?model s|m|l}
check_built; assert_mode "$BASE_MODE"; clk_save

ENG=$(engine_of "$M"); MODEL=$(model_of "$M")
OUT="$RES/probe"
RATES=$PROBE_RATES   # Orin: 12 20 30; Nano: 5 8 10 (platform.sh)
B="exp=probe model=$MODEL mode=$BASE_MODE"

for CELL in "default default" "lock lock"; do
  set -- $CELL; CPU=$1; GPU=$2
  set_cell "$CPU" "$GPU"
  for PIPE in seq pipe; do
    T="${MODEL}_${PIPE}_cpu-${CPU}_gpu-${GPU}"
    C="$B pipeline=$PIPE cpu=$CPU gpu=$GPU"
    for F in $RATES; do
      [ "$CPU" = lock ] && [ "$F" != "$P_FPS" ] && continue
      PROBE_RUN=1 run_bench "$OUT" "${T}_probe${F}" "$C rel=periodic fps=$F probe=1" \
        --engine "$ENG" --pipeline "$PIPE" --mode periodic --period-ms "$(fps_ms "$F")" --limit 600 --passes 1
    done
    PROBE_RUN=1 run_bench "$OUT" "${T}_probe_b2b" "$C rel=b2b probe=1" \
      --engine "$ENG" --pipeline "$PIPE" --mode b2b --limit 600 --passes 1
  done
done
# probe perturbation: same condition without it
set_cell default default
run_bench "$OUT" "${MODEL}_seq_cpu-default_gpu-default_noprobe${P_FPS}" "$B pipeline=seq cpu=default gpu=default rel=periodic fps=$P_FPS probe=0" \
  --engine "$ENG" --mode periodic --period-ms "$P_MS" --limit 600 --passes 1
log "18_probe $MODEL done"
