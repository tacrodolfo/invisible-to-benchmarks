#!/usr/bin/env bash
# Can the vendor tool see the gap? (Fig. 1)
# trtexec with --idleTime inserts pauses between inferences, which mimics periodic release
# without any CPU stage. Sweeps the pause and measures GPU frequency and load with 1 ms
# sampling. Outcomes:
#   - if, with pause = period - inference, the GPU also drops to ~305 MHz and inference time
#     rises, trtexec CAN reveal the gap, but the recommended procedure does not;
#   - the mean load (duty cycle) vs frequency curve shows above which load nvhost_podgov
#     raises the frequency.
#
# Usage: ./scripts/14_trtexec_idle.sh s|m|l     IDLES="0 1 2 5 10 20 40"  TRT_DUR=20
source "$(dirname "$0")/lib.sh"
M=${1:?model s|m|l}
check_built; assert_mode "$BASE_MODE"; clk_save

ENG=$(engine_of "$M"); MODEL=$(model_of "$M")
OUT="$RES/trtexec_idle"
IDLES=${IDLES:-"0 1 2 5 10 20 40"}
TRT_DUR=${TRT_DUR:-20}
B="exp=trtexec_idle model=$MODEL mode=$BASE_MODE"
TA=(--loadEngine="$ENG" --warmUp=1000 --duration="$TRT_DUR" --iterations=50)
# the 1 ms probe gives the GPU frequency distribution, which is the goal here; it wakes the CPU
# (see 19_observer.sh), but what is measured is GPU compute time (PROBE_RUN=0 disables it)
export PROBE_RUN=${PROBE_RUN:-1}

# 1) locked reference without pause: time per inference, to derive the equivalent pauses
set_cell lock lock
REF="${MODEL}_trt_cpu-lock_gpu-lock_idle0"
run_trtexec "$OUT" "$REF" "$B cpu=lock gpu=lock idle=0 spin=0" "${TA[@]}" --idleTime=0
LAT=7
if [ "$DRY" != 1 ]; then
  LAT=$(grep -m1 -E "\[I\] Latency: min" "$OUT/$REF.log" | sed -E 's/.*mean = ([0-9.]+) ms.*/\1/' || true)
  [ -n "$LAT" ] || LAT=$(grep -m1 "GPU Compute Time" "$OUT/$REF.log" | sed -E 's/.*mean = ([0-9.]+) ms.*/\1/' || true)
  [ -n "$LAT" ] || die "could not find the latency in $OUT/$REF.log"
fi
I12=$(awk -v l="$LAT" -v p="$P_MS" 'BEGIN{v=int(p-l); print (v<0?0:v)}')
I30=$(awk -v l="$LAT" -v p="$P_MS" 'BEGIN{v=int(33.333-l); print (v<0?0:v)}')
echo ">> locked latency ${LAT} ms -> equivalent pause at 12 Hz: ${I12} ms, at 30 fps: ${I30} ms"
ALL=$(echo "$IDLES $I12 $I30" | tr ' ' '\n' | grep -v '^$' | sort -n | uniq | tr '\n' ' ')

for I in "$I12" "$I30"; do
  run_trtexec "$OUT" "${MODEL}_trt_cpu-lock_gpu-lock_idle${I}" "$B cpu=lock gpu=lock idle=$I spin=0" "${TA[@]}" --idleTime="$I"
done

# 2) factory: full pause sweep; with and without busy wait at the equivalent pauses
set_cell default default
for I in $ALL; do
  run_trtexec "$OUT" "${MODEL}_trt_cpu-default_gpu-default_idle${I}" "$B cpu=default gpu=default idle=$I spin=0" \
    "${TA[@]}" --idleTime="$I"
done
for I in "$I12" "$I30"; do
  run_trtexec "$OUT" "${MODEL}_trt_cpu-default_gpu-default_idle${I}_spin" "$B cpu=default gpu=default idle=$I spin=1" \
    "${TA[@]}" --idleTime="$I" --useSpinWait
done
log "14_trtexec_idle $MODEL done"
