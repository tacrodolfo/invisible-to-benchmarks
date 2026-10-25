#!/usr/bin/env bash
# Builds the harness and checks equivalence with the Python version (YOLOX-s).
#  1. same detections per frame (n_det) as bench.py, under the same conditions;
#  2. per-stage times side by side (decode should match; pre- and post-processing should drop);
#  3. quick test of the new paths (raw, nvJPEG, GPU pre-processing, network only, sync edge);
#  4. go/no-go: B1 vs B3 in C++ (does the GPU governor effect persist?).
# Duration: ~15 min. Usage: ./scripts/01_build_equiv.sh
source "$(dirname "$0")/lib.sh"

if [ "$DRY" != 1 ]; then
  log "building"
  cmake -S "$CPP_DIR" -B "$CPP_DIR/build" -DCMAKE_BUILD_TYPE=Release
  cmake --build "$CPP_DIR/build" -j4
fi
check_built
assert_mode "$BASE_MODE"
clk_save

OUT="$RES/equiv"
ENG=$(engine_of s)
C="exp=equiv model=yolox_s mode=$BASE_MODE"
PY_RES=${PY_RES:-$DATA/py_results}

# 83 ms period, as in the run_gonogo.sh used with bench.py (on Orin; on other boards, the
# platform period, and the comparison with Python only happens if its results exist)
EQ_P=83; [ "$PLATFORM" = orin ] || EQ_P=$P_MS
set_cell default default
run_bench "$OUT" cpp_b2b_default "$C cpu=default gpu=default rel=b2b" --engine "$ENG" --mode b2b --passes 1
run_bench "$OUT" cpp_periodic_default_hybrid "$C cpu=default gpu=default rel=periodic wait=hybrid" \
  --engine "$ENG" --mode periodic --period-ms "$EQ_P" --passes 1 --wait hybrid
run_bench "$OUT" cpp_periodic_default_sleep "$C cpu=default gpu=default rel=periodic wait=sleep" \
  --engine "$ENG" --mode periodic --period-ms "$EQ_P" --passes 1 --wait sleep

# new paths: 200 frames back-to-back
S="--engine $ENG --mode b2b --passes 1 --limit 200 --warmup 20"
run_bench "$OUT" smoke_jpeg_cpu "$C variant=jpeg_cpu" $S
run_bench "$OUT" smoke_raw_cpu "$C variant=raw_cpu" $S --input raw
run_bench "$OUT" smoke_jpeg_cpu_gpupre "$C variant=jpeg_cpu_gpupre" $S --pre gpu
run_bench "$OUT" smoke_raw_gpu "$C variant=raw_gpu" $S --input raw --pre gpu
NVJ_TEST=$([ -x "$BENCH" ] && "$BENCH" --engine x --tag x --decode gpu --pre gpu 2>&1 || true)
if [[ "$NVJ_TEST" != *"without nvJPEG"* ]]; then
  run_bench "$OUT" smoke_nvjpeg "$C variant=nvjpeg" $S --decode gpu --pre gpu
  run_bench "$OUT" smoke_nvjpeg_hybrid "$C variant=nvjpeg_hybrid" $S --decode gpu --pre gpu --nvjpeg-backend hybrid
fi
run_bench "$OUT" smoke_edge "$C variant=edge" $S --sync edge
run_bench "$OUT" smoke_block "$C variant=block" $S --sched block
run_bench "$OUT" smoke_pipe "$C variant=pipe" $S --pipeline pipe
[[ "$NVJ_TEST" == *"without nvJPEG"* ]] || run_bench "$OUT" smoke_pipe_nvjpeg "$C variant=pipe_nvjpeg" $S --pipeline pipe --decode gpu --pre gpu --sync edge
run_bench "$OUT" smoke_tensor "$C variant=tensor" $S --input tensor --post none

set_cell lock lock
run_bench "$OUT" cpp_periodic_lock "$C cpu=lock gpu=lock rel=periodic wait=hybrid" \
  --engine "$ENG" --mode periodic --period-ms "$EQ_P" --passes 1 --wait hybrid

if [ "$DRY" != 1 ] && [ -e "$PY_RES" ] && python3 -c "import pandas" 2>/dev/null; then
  python3 "$CPP_DIR/analysis/equiv.py" --py "$PY_RES" --cpp "$OUT" | tee "$OUT/equiv_report.txt"
fi
