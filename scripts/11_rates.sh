#!/usr/bin/env bash
# Sustainable camera rate and P99/miss vs rate curves (Fig. 2, Table III, Table IV).
#   Coarse sweep (SWEEP_RATES) with deadline = period, for the figures;
#   search for the highest rate with <= 1% deadline misses (1 fps resolution, confirmed
#   with full runs at the two adjacent rates), per clock cell.
#   In the sequential pipeline, the search also runs in the mixed cells (lock/default and default/lock),
#   which gives the decomposition of the effect on the sustainable rate.
#
# Usage: ./scripts/11_rates.sh s|m|l [seq|pipe]
#   SWEEP_RATES="5 10 12 15 20 25 30"   SWEEP_LIMIT=1000 SWEEP_PASSES=2   (per-platform defaults)
source "$(dirname "$0")/lib.sh"
M=${1:?model s|m|l}; PIPE=${2:-seq}
check_built; assert_mode "$BASE_MODE"; clk_save

ENG=$(engine_of "$M"); MODEL=$(model_of "$M")
OUT="$RES/rates"
SWEEP_RATES=${SWEEP_RATES:-$PLAT_SWEEP_RATES}   # Orin: 5..30 fps; Nano: 1..7 fps (platform.sh)
SL=${SWEEP_LIMIT:-$PLAT_SWEEP_LIMIT}; SP=${SWEEP_PASSES:-2}
B="exp=rates model=$MODEL mode=$BASE_MODE pipeline=$PIPE"

CELLS=("lock lock" "default default")
[ "$PIPE" = seq ] && CELLS+=("lock default" "default lock")

for CELL in "${CELLS[@]}"; do
  set -- $CELL; CPU=$1; GPU=$2
  set_cell "$CPU" "$GPU"
  T="${MODEL}_${PIPE}_cpu-${CPU}_gpu-${GPU}"
  if [ "$CPU" = "$GPU" ]; then
    for F in $SWEEP_RATES; do
      run_bench "$OUT" "${T}_sweep${F}" "$B cpu=$CPU gpu=$GPU fps=$F" \
        --engine "$ENG" --pipeline "$PIPE" --mode periodic --period-ms "$(fps_ms "$F")" --limit "$SL" --passes "$SP"
    done
    [ "$PIPE" = pipe ] && run_bench "$OUT" "${T}_b2b" "$B cpu=$CPU gpu=$GPU rel=b2b" \
      --engine "$ENG" --pipeline "$PIPE" --mode b2b --limit "$SL" --passes "$SP"
  fi
  find_rate "$OUT" "$T" "$B cpu=$CPU gpu=$GPU" "$RATE_LO" "$RATE_HI" --engine "$ENG" --pipeline "$PIPE"
done
log "11_rates $MODEL $PIPE done"
