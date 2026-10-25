# Earlier Python harness

First version of the measurement code (bench.py, bench_pipe.py, probe_freq.py, run_*.sh,
analyze.py). It is kept because `scripts/01_build_equiv.sh` checks that the C++ harness produces
the same detections and the same decoding times. It expects the engines and frames in the current
folder (`frames/`, `yolox_s_fp16.engine`). The results of the paper come from the C++ harness
(`src/`, `scripts/`).
