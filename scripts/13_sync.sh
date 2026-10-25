#!/usr/bin/env bash
# Does the instrumentation cause the effect? (control in Section "Platforms, Application, and Workload")
#   stage/auto   synchronizes the stream after each GPU operation, CUDA's default wait
#                (as in bench.py and the paper)
#   edge/auto    a single synchronization per frame; GPU times from cudaEvent
#   stage/block  blocking wait (cudaDeviceScheduleBlockingSync): the CPU sleeps during GPU work
#   edge/block   typical production: one blocking synchronization per frame
# (auto = busy wait when there are fewer CUDA contexts than cores, which is the case here)
# If end-to-end time and GPU frequency do not change between stage and edge, per-stage
# synchronization does not create the effect. The auto/block pair shows how much the busy wait
# keeps the CPU busy (which interacts with schedutil).
#
# Usage: ./scripts/13_sync.sh s|m|l     SYNCS="stage:auto edge:auto ..."
source "$(dirname "$0")/lib.sh"
M=${1:?model s|m|l}
check_built; assert_mode "$BASE_MODE"; clk_save

ENG=$(engine_of "$M"); MODEL=$(model_of "$M")
OUT="$RES/sync"
SYNCS=${SYNCS:-"stage:auto edge:auto stage:block edge:block"}
L=${SYNC_LIMIT:-1000}; P=${SYNC_PASSES:-2}
FIND=${FIND_RATE:-1}
FIND_SYNCS=${FIND_SYNCS:-"stage:auto edge:block"}   # sustainable rate only at the extremes
B="exp=sync model=$MODEL mode=$BASE_MODE"

for CELL in "lock lock" "default default"; do
  set -- $CELL; CPU=$1; GPU=$2
  set_cell "$CPU" "$GPU"
  for SV in $SYNCS; do
    SY=${SV%%:*}; SC=${SV##*:}
    T="${MODEL}_sync-${SY}_sched-${SC}_cpu-${CPU}_gpu-${GPU}"
    C="$B sync=$SY sched=$SC cpu=$CPU gpu=$GPU"
    A="--engine $ENG --sync $SY --sched $SC"
    # shellcheck disable=SC2086
    run_bench "$OUT" "${T}_periodic" "$C rel=periodic fps=$P_FPS" $A --mode periodic --period-ms "$P_MS" --limit "$L" --passes "$P"
    # shellcheck disable=SC2086
    run_bench "$OUT" "${T}_b2b" "$C rel=b2b" $A --mode b2b --limit "$L" --passes "$P"
    if [ "$CPU" = default ]; then
      if [ "$FIND" = 1 ] && [[ " $FIND_SYNCS " == *" $SV "* ]]; then
        # shellcheck disable=SC2086
        find_rate "$OUT" "$T" "$C" "$RATE_LO" "$RATE_HI" $A
      fi
      for F in 12 30; do
        # shellcheck disable=SC2086
        PROBE_RUN=1 run_bench "$OUT/probe" "${T}_probe${F}" "$C rel=periodic fps=$F probe=1" \
          $A --mode periodic --period-ms "$(fps_ms "$F")" --limit 300 --passes 1
      done
    fi
  done
done
log "13_sync $MODEL done"
