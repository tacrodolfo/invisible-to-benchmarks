#!/usr/bin/env bash
# B0..B3 ladder + full CPU x GPU factorial (Table I, Table II, Table IV).
#
#   B0      trtexec (jetson_clocks and factory), the conventional benchmark
#   NET     network only (pre-processed tensor, no post-processing) with periodic release at 12 Hz and
#           back-to-back, in the 4 cells: separates the periodic-release effect from the CPU stages
#   2x2     CPU {lock, default} x GPU {lock, default} x co-runner x release, full pipeline
#             (lock, lock) = B1   (lock, default) = B2   (default, default) = B3
#             (default, lock) = the fourth combination
#   JC      B1 with jetson_clocks, to check that the sysfs "lock" is equivalent to the vendor's
#
# Usage: ./scripts/10_factorial.sh s|m|l
#   CORUNS="off on" (default s: off on; m and l: off)   RELS="b2b periodic"   PASSES=3   PERIOD_MS=83.333
source "$(dirname "$0")/lib.sh"
M=${1:?model s|m|l}
check_built; assert_mode "$BASE_MODE"; clk_save

ENG=$(engine_of "$M"); MODEL=$(model_of "$M")
OUT="$RES/factorial"
PASSES=${PASSES:-3}
if [ "$M" = s ]; then CORUNS=${CORUNS:-"off on"}; else CORUNS=${CORUNS:-off}; fi
RELS=${RELS:-"b2b periodic"}
P12=$P_MS   # platform period (Orin: 83.333 ms = 12 Hz); change with PERIOD_MS=...
B="exp=factorial model=$MODEL mode=$BASE_MODE"

# ---------------- B0: trtexec
for CELL in "jc jc" "lock lock" "default default"; do
  set -- $CELL; set_cell "$1" "$2"
  run_trtexec "$OUT" "${MODEL}_B0_cpu-$1_gpu-$2" "$B step=B0 cpu=$1 gpu=$2" \
    --loadEngine="$ENG" --iterations=5000 --warmUp=500 --duration=0
done

# ---------------- 4 cells
for CELL in "lock lock" "lock default" "default lock" "default default"; do
  set -- $CELL; CPU=$1; GPU=$2
  set_cell "$CPU" "$GPU"
  # network only (short)
  for REL in b2b periodic; do
    run_bench "$OUT" "${MODEL}_NET_cpu-${CPU}_gpu-${GPU}_${REL}" "$B step=NET cpu=$CPU gpu=$GPU rel=$REL" \
      --engine "$ENG" --mode "$REL" --period-ms "$P12" --input tensor --post none --limit 1000 --passes 1
  done
  # full pipeline
  for CO in $CORUNS; do
    for REL in $RELS; do
      CORUN=$CO run_bench "$OUT" "${MODEL}_cpu-${CPU}_gpu-${GPU}_co-${CO}_${REL}" \
        "$B step=full cpu=$CPU gpu=$GPU rel=$REL" \
        --engine "$ENG" --mode "$REL" --period-ms "$P12" --passes "$PASSES"
    done
  done
done

# ---------------- B1 with jetson_clocks (check)
set_cell jc jc
run_bench "$OUT" "${MODEL}_cpu-jc_gpu-jc_co-off_periodic" "$B step=full cpu=jc gpu=jc rel=periodic" \
  --engine "$ENG" --mode periodic --period-ms "$P12" --passes "$PASSES"
log "10_factorial $MODEL done"
