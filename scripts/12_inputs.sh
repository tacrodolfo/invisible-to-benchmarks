#!/usr/bin/env bash
# Input path (Section "Power Mode, Pipelining, and Input Path").
# Does the GPU governor effect persist when the CPU no longer dominates the frame?
#
# Variants (the same INPUT_LIMIT frames in all of them, for direct comparison):
#   jpeg_cpu     JPEG -> OpenCV (CPU) -> letterbox on the CPU          (the paper's reference)
#   raw_cpu      raw frame in memory (as from CSI/ISP), letterbox on the CPU   (no decode)
#   jpeg_gpupre  JPEG -> OpenCV (CPU) -> letterbox on the GPU
#   nvjpeg       JPEG -> nvJPEG (Huffman on the GPU) -> letterbox on the GPU   (accelerated decode)
#   nvjpeg_hybrid JPEG -> nvJPEG (Huffman on the CPU, rest on the GPU) -> letterbox on the GPU
#   raw_gpu      raw frame -> DMA copy -> letterbox on the GPU            (minimum CPU)
# For each variant and cell (lock/lock, default/default): periodic at 12 Hz, back-to-back,
# sustainable rate and, in the factory cell, 1 ms sampling (Orin: 12 and 30 fps; PROBE_RATES_IN).
#
# Usage: ./scripts/12_inputs.sh s|m|l      VARIANTS="..." INPUT_LIMIT=400 INPUT_PASSES=4
# Raw frames: 400 x 4.3 MB = 1.7 GB of RAM; do not raise INPUT_LIMIT beyond ~600.
source "$(dirname "$0")/lib.sh"
M=${1:?model s|m|l}
check_built; assert_mode "$BASE_MODE"; clk_save

ENG=$(engine_of "$M"); MODEL=$(model_of "$M")
OUT="$RES/inputs"; mkdir -p "$OUT"
VARIANTS=${VARIANTS:-"jpeg_cpu raw_cpu jpeg_gpupre nvjpeg nvjpeg_hybrid raw_gpu"}
L=${INPUT_LIMIT:-$PLAT_INPUT_LIMIT}; P=${INPUT_PASSES:-4}
FIND=${FIND_RATE:-1}
# without nvJPEG in the binary, the nvjpeg variants are dropped (otherwise they would become a false "rate < 8 fps")
NVJ_TEST=$([ -x "$BENCH" ] && "$BENCH" --engine x --tag x --decode gpu --pre gpu 2>&1 || true)
if [[ "$NVJ_TEST" == *"without nvJPEG"* ]]; then
  VARIANTS=$(echo "$VARIANTS" | tr ' ' '\n' | grep -v '^nvjpeg' | tr '\n' ' ')
  echo "################################################################"
  echo "WARNING: build/bench was built WITHOUT nvJPEG. The nvjpeg and nvjpeg_hybrid variants"
  echo "(GPU decoding) will NOT be measured."
  echo "To include them: ./scripts/setup.sh --install  (installs libnvjpeg and rebuilds)"
  echo "################################################################"
  echo "nvjpeg variants missing: binary without nvJPEG ($(date))" > "$OUT/NVJPEG_MISSING.txt" 2>/dev/null || true
  sleep 5
fi
B="exp=inputs model=$MODEL mode=$BASE_MODE"

vargs() {
  case "$1" in
    jpeg_cpu) echo "" ;;
    raw_cpu) echo "--input raw" ;;
    jpeg_gpupre) echo "--pre gpu" ;;
    nvjpeg) echo "--decode gpu --pre gpu" ;;
    nvjpeg_hybrid) echo "--decode gpu --pre gpu --nvjpeg-backend hybrid" ;;
    raw_gpu) echo "--input raw --pre gpu" ;;
    *) die "variant $1" ;;
  esac
}

for CELL in "lock lock" "default default"; do
  set -- $CELL; CPU=$1; GPU=$2
  set_cell "$CPU" "$GPU"
  for V in $VARIANTS; do
    A=$(vargs "$V")
    T="${MODEL}_${V}_cpu-${CPU}_gpu-${GPU}"
    C="$B variant=$V cpu=$CPU gpu=$GPU"
    # shellcheck disable=SC2086
    run_bench "$OUT" "${T}_periodic" "$C rel=periodic fps=$P_FPS" --engine "$ENG" $A --mode periodic --period-ms "$P_MS" --limit "$L" --passes "$P"
    # shellcheck disable=SC2086
    run_bench "$OUT" "${T}_b2b" "$C rel=b2b" --engine "$ENG" $A --mode b2b --limit "$L" --passes "$P"
    if [ "$FIND" = 1 ]; then
      # shellcheck disable=SC2086
      QUICK_LIMIT=$L FULL_LIMIT=$L FULL_PASSES=$P find_rate "$OUT" "$T" "$C" "$RATE_LO" "$((RATE_HI * 3 / 2))" --engine "$ENG" $A
    fi
    if [ "$CPU" = default ]; then
      for F in $PROBE_RATES_IN; do
        # shellcheck disable=SC2086
        PROBE_RUN=1 run_bench "$OUT/probe" "${T}_probe${F}" "$C rel=periodic fps=$F probe=1" \
          --engine "$ENG" $A --mode periodic --period-ms "$(fps_ms "$F")" --limit 300 --passes 1
        # control without the probe (at the platform period, the control is the periodic run above)
        if [ "$F" != "$P_FPS" ]; then
          # shellcheck disable=SC2086
          run_bench "$OUT/probe" "${T}_noprobe${F}" "$C rel=periodic fps=$F probe=0" \
            --engine "$ENG" $A --mode periodic --period-ms "$(fps_ms "$F")" --limit 300 --passes 1
        fi
      done
    fi
  done
done
log "12_inputs $MODEL done"
