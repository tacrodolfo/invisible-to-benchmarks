# edgebench: response time of edge-AI perception under deployment conditions

Code, scripts and result tables for the paper

> T. A. Rodolfo and A. C. S. Beck, "Invisible to Benchmarks: How Deployment Conditions Limit the
> Real-Time Capacity of Edge AI Perception," *IEEE International Conference on Industrial
> Technology (ICIT)*, 2027.

The repository contains a C++ harness that runs the complete YOLOX perception pipeline (JPEG
decoding, letterbox, H2D copy, TensorRT inference, D2H copy, NMS) on NVIDIA Jetson boards under
back-to-back or periodic frame release, a 1 ms sysfs sampler, one script per experiment, and the
analysis that produces the tables reported in the paper.

- **Platforms:** Jetson Orin Nano Super (JetPack 6, TensorRT 10.3, CUDA 12.6, OpenCV 4.8), the
  paper's main platform, and Jetson Nano (JetPack 4.6, TensorRT 8.2, CUDA 10.2, OpenCV 4.1). The
  same scripts run on both; `scripts/platform.sh` detects the SoC and sets the power mode, camera
  period, models and sysfs paths.
- **Workload:** YOLOX-s/m/l (official ONNX, COCO weights, FP16, 640 x 640) on 1652 CAM_FRONT
  frames from nuScenes v1.0-mini.
- **Results of the paper:** `results_orin/` and `results_nano/` (tables and platform logs). The raw
  per-frame data are archived separately (see [Raw data](#raw-data)).

## Where each result of the paper comes from

| Paper | Script | Output (in `results_<platform>/tables/`) |
|---|---|---|
| Table I: B0-B3 ladder, the four CPU/GPU frequency combinations, `jetson_clocks` row | `10_factorial.sh` | `factorial.csv`, `trtexec.csv` |
| Fig. 1: trtexec, trtexec `--idleTime` and in-pipeline GPU compute time; GPU utilization threshold | `14_trtexec_idle.sh` | `trtexec.csv` |
| Table II: release models (back-to-back vs periodic) and co-runner | `10_factorial.sh` | `factorial.csv` |
| Fig. 2 and Table III: P99 and miss rate vs camera rate, sustainable rates | `11_rates.sh` | `rates.csv`, `runs.csv` |
| GPU-only/CPU-only locking, GPU and CPU frequency floors | `11_rates.sh`, `15_floors.sh` | `rates.csv`, `floors.csv` |
| CPU idle states | `17_idle_states.sh` | `idle_states.csv`, `rates.csv` |
| MAXN_SUPER power mode | `16_maxn.sh` | `maxn.csv`, `rates.csv` |
| Two-stage pipeline | `11_rates.sh <m> pipe` | `rates.csv`, `runs.csv` |
| Input path (nvJPEG, pre-decoded frames, GPU pre-processing) | `12_inputs.sh` | `inputs.csv`, `rates.csv` |
| Table IV and Fig. 3: detector size, GPU frequency vs camera rate | `10_factorial.sh m/l`, `11_rates.sh m/l` | `factorial.csv`, `rates.csv`, `runs.csv` |
| Energy per valid detection | `11_rates.sh`, `15_floors.sh` | `rates.csv` (`energy_J_valid_at_ok`), `floors.csv` |
| Table V: Jetson Nano | same scripts on the Nano | `results_nano/tables/` |
| Control: one synchronization per frame instead of per stage | `13_sync.sh` | `sync.csv`, `rates.csv` |
| Control: effect of the monitors (tegrastats, 1 ms probe, jtop) | `19_observer.sh` | `observer_effect.csv` |
| Platform state (power mode, governors, idle states, EMC, fan) | `00_report.sh`, `run_all.sh` | `results_<platform>/platform/` |
| Equivalence of the C++ harness with the earlier Python harness | `01_build_equiv.sh` | `results_nano/tables/equiv.csv` |

`runs.csv` has one row per run (all metrics and the recorded experimental condition); the other
tables are views of it per experiment. `analysis/analyze_cpp.py` also writes sanity-check figures
(`rates_*.png`, `trtexec_idle_*.png`, `inputs_*.png`) next to the tables.

The Orin results come from 874 bench runs and 57 trtexec runs; the Nano results from 425 bench runs
and 15 trtexec runs. Each platform was measured within a single boot.

## Raw data

The raw files of every run (per-frame CSV, `.cond`, `.clocks`, `.log`, tegrastats, 1 ms probe) are
too large for git and are attached to the GitHub release
[v1.0.0](https://github.com/tacrodolfo/invisible-to-benchmarks/releases/tag/v1.0.0) of this repository:

| Archive | Contents |
|---|---|
| `edgebench_raw_orin.tgz` | `results_orin/`: Orin Nano campaign (16-17 Oct 2026) with the observer experiment completed on 19 Oct 2026 (locked-clock runs and jtop sustainable-rate runs) |
| `edgebench_raw_nano.tgz` | `results_nano/`: Jetson Nano campaign (16-17 Oct 2026) |
| `edgebench_raw_fan_quiet.tgz` | `results_fan_quiet/`: control of the fan profile (YOLOX-s factorial with the factory `quiet` profile) |

To regenerate the tables from the raw data:
```bash
wget https://github.com/tacrodolfo/invisible-to-benchmarks/releases/download/v1.0.0/edgebench_raw_orin.tgz
tar xzf edgebench_raw_orin.tgz          # creates results_orin/ (raw files next to tables/)
python3 analysis/analyze_cpp.py --res results_orin | tee results_orin/tables/summary.txt
```
The bootstrap uses a fixed seed and 2000 resamples (the default), so the tables are reproduced exactly.

Notes on the raw data:
- The Orin campaign started with the factory fan profile (`quiet`), but the profile changed to `cool`
  during tier 1, so most Orin runs used `cool` (recorded as `fan=` in each `.cond` and in the
  campaign log). The control in `results_fan_quiet/` repeats the YOLOX-s factorial at 12 Hz (B0, the
  four CPU/GPU cells, network only, `jetson_clocks`) with `quiet`; the mean response times differ by
  at most 0.03 ms and the P99 by at most 0.09 ms.
- The campaigns were run with an earlier build of this repository whose console messages and
  platform logs were in Portuguese (e.g. `passada 1/3 concluída` = `pass 1/3 done`, `n/d` = `n/a`).
  Only the messages were translated afterwards; the measurement code, file formats and keys are
  unchanged, and the analysis reads both. The platform logs in `results_<platform>/platform/` are
  English renderings of those logs (labels translated, values unchanged); the originals are in the
  raw archives.

## Reproducing the measurements

**1. Clone the repository and download nuScenes v1.0-mini** (https://www.nuscenes.org/, free
registration, CC BY-NC-SA 4.0 license; the frames are not redistributed here):
```bash
git clone https://github.com/tacrodolfo/invisible-to-benchmarks.git ~/edgebench && cd ~/edgebench
mkdir -p ~/nuscenes && tar xzf v1.0-mini.tgz -C ~/nuscenes
```

**2. Install, build and generate the data** with a single command:
```bash
./scripts/setup.sh                                          # only checks and lists what is missing
NUSCENES=~/nuscenes ./scripts/setup.sh --install            # installs, builds and generates the data
```
`--install`:
- installs whatever is missing with apt: CUDA dev, TensorRT dev, OpenCV, nvJPEG, trtexec, stress-ng
  and Python libraries;
- builds `build/bench` and `build/probe`;
- builds the FP16 engines from the official YOLOX ONNX models (a few minutes per model);
- assembles the 1652 frames from nuScenes (`data_prep/select_nuscenes.py`) and checks that they are
  identical to the paper's, using the manifest `data/frames.sha256`.

The output ends with `ready: ... with nvJPEG`, and the data check reports
`frames identical to the paper's`. Running it again is safe: whatever already exists is kept.

**3. Reboot and record the factory state** before any other measurement. If jtop (jetson-stats) is
installed, disable it at boot first, because it may apply `jetson_clocks` when it starts:
```bash
sudo systemctl disable jtop 2>/dev/null; sudo systemctl disable jetson_stats 2>/dev/null
sudo reboot
# reconnect
cd ~/edgebench && ./scripts/00_report.sh boot
```
The report warns if the CPU idle states are disabled right after boot. Without a reboot, use
`./scripts/00_report.sh now`.

**4. Build and check equivalence** with the earlier Python harness (about 15 min):
```bash
sudo -v
tmux new -s edgebench
./scripts/01_build_equiv.sh 2>&1 | tee equiv_log.txt
```
The report shows the share of frames with the same number of detections as the Python harness, the
per-stage times side by side and a go/no-go check (factory/locked inference ratio above 1.5).

**5. Show the plan and the estimated time:**
```bash
DRY=1 ./scripts/run_all.sh 1 | tail -10
```

**6. Run tier 1** (YOLOX-s, every experiment of the paper; about 10 h on the Orin):
```bash
./scripts/run_all.sh 1 2>&1 | tee run1_log.txt
```
`run_all.sh` asks for the sudo password once and renews it every 60 s. Before and after each tier,
the platform state (mode, idle states, `bwmgr_halt`, fan, boot) is written to
`results/platform/campaign_<date>.txt`. To leave tmux without stopping: `Ctrl+B`, then `D`; to come
back: `tmux attach -t edgebench`. If the run stops, run the same command again: finished runs are
skipped.

**7. Tiers 2 and 3** (YOLOX-m/l, two-stage pipeline, MAXN_SUPER, extras; about 8 h each):
```bash
./scripts/run_all.sh 2 2>&1 | tee run2_log.txt
./scripts/run_all.sh 3 2>&1 | tee run3_log.txt
```
Or everything in one call (about 25 h on the Orin): `./scripts/run_all.sh all 2>&1 | tee run_log.txt`.

**8. Analysis:**
```bash
python3 analysis/analyze_cpp.py --res results | tee results/tables/summary.txt
```

## Details that matter

- **Monitoring and background services.** Every process that wakes the CPU periodically is load for
  schedutil, which raises the frequency and masks part of the measured effect. tegrastats therefore
  runs every 1 s (`TEGRA_MS`, recorded in the `.cond`), the 1 ms probe only runs in dedicated runs
  (never used for latency), and with `QUIET=1` (default) jtop and the desktop environment are
  stopped during the campaign and restarted at the end (`./scripts/bg.sh status|stop|restore`).
  `19_observer.sh` measures the size of this effect. Other local processes that wake the CPU can be
  frozen during the campaign with `QUIET_PROCS="name1 name2"` (SIGSTOP/SIGCONT).
- **Factory state.** On the first run in each power mode and boot, the scripts check the state
  against the mode's default (`schedutil` and `nvhost_podgov` governors and the frequency limits
  that `/etc/nvpmodel.conf` sets for the mode). If something is off, they reapply the mode with
  `nvpmodel -m <id>` and restore the governors, without a reboot. The state is saved in
  `state/clocks_<mode>.env` (with the boot id) and restored on exit, also on error or `Ctrl+C`.
- **Invisible residues.** A `jetson_clocks` undone by another path (`nvpmodel -m`, sysfs) returns the
  frequencies to default but leaves the CPU idle states disabled and the EMC locked at maximum
  (`bwmgr_halt=1`). The reference state of a boot is therefore only saved with the default mode, all
  idle states enabled and `bwmgr_halt=0`; `jetson_clocks --store` goes to a new file per boot; after
  each restore the idle states and `bwmgr_halt` are checked; and each run records them before and
  after (`state_changed=1` if something changed).
- **Clock cells.** The cells are applied directly in sysfs: CPU `lock` is the `performance` governor,
  GPU `lock` is `min_freq = max_freq`. Memory frequency and CPU idle states stay at their defaults.
  `jetson_clocks` is used only for B0 and for one B1 check (cell `jc`).
- **Temperature, power and throttling.** The analysis takes, from the same tegrastats log, the mean
  and maximum temperature of each zone and the mean power of each rail within the measurement
  window, and flags locked runs with the GPU or CPU below 97% of the maximum (heat or power limit).
- **EMC at idle.** With the lock off, the platform's default policy sets the memory frequency; on
  the Orin Nano Super it stays at 2133 MHz and on the Jetson Nano at 1600 MHz. Each run records it.
- **Fan.** The expected profile is `FAN_DEFAULT_PROFILE` of the `/etc/nvfancontrol.conf` in use (Orin
  Nano: `quiet`); `FAN_PROFILE_EXPECTED=...` overrides it. The scripts warn if the active profile is
  different and if `dpkg -V` reports locally modified fan or nvpmodel configuration files.
- **Period.** On the Orin, the campaigns use 83.333 ms (12 Hz); on the Jetson Nano 200 ms (5 Hz).
  `PERIOD_MS=...` overrides it.
- **Waiting for the release.** The default is an absolute `clock_nanosleep`; `--wait hybrid` busy-waits
  the last 1 ms, as the earlier Python harness did.
- **Synchronization.** The default (`--sync stage --sched auto`) synchronizes after each GPU operation,
  as in the paper; `13_sync.sh` measures what changes without it.
- **Raw frames.** 400 frames take 1.7 GB of pinned RAM (Nano: 150 frames).
- **GPU pre-processing.** The CUDA letterbox kernel differs from OpenCV by ±1 gray level in about 7%
  of the pixels (OpenCV fixed-point rounding); this is checked by the CPU tests.
- **Power.** Mean of VDD_IN (Nano: POM_5V_IN) from tegrastats, only within the measurement window.
  These module sensors do not cover the whole developer kit.
- **Sustainable rate.** The highest integer rate with at most 1% deadline misses (deadline = period).
  A binary search with short runs (400 frames) is confirmed with full runs (2 x 1000 frames) at the
  two adjacent rates. The `.rate` file holds `ok`, `fail` and the miss rates.
- **Another dataset.** `scripts/prep_frames.py` converts a sequence to JPEG. Run with
  `FRAMES=data/frames_<name> RES=$PWD/results_<name>` to keep it apart from the nuScenes results.

## Jetson Nano (JetPack 4.6)

Same repository and scripts. Differences handled automatically:
- TensorRT 8.2: the bench uses the bindings API (`enqueueV2`); CUDA 10.2: kernels in C++14, `sm_53`;
  gcc 7: `std::experimental::filesystem`.
- CMake >= 3.18 comes from pip (`~/.local/bin`), because apt's (3.10) is too old. The scripts add
  `~/.local/bin` to the PATH; to build by hand, add it to yours.
- The GPU architecture comes from the SoC in `/proc/device-tree/compatible` (tegra234 -> sm_87,
  tegra194 -> sm_72, tegra210 -> sm_53); for an unknown SoC, pass `-DCMAKE_CUDA_ARCHITECTURES=<sm>`.
- No nvJPEG: the GPU decoding variants are skipped.
- Boot state checked against the L4T 32.7 documentation: MAXN mode (10 W, default), 8 idle states
  enabled, CPU `schedutil` 102-1479 MHz, GPU `nvhost_podgov` 76.8-921.6 MHz with
  `enable_3d_scaling=1`, dynamic EMC from 204 to 1600 MHz, automatic fan.
- Power from the INA3221 iio (channel 0, POM_5V_IN).
- YOLOX-s only (4 GB of RAM); 200 ms period; raw frames limited to 150.
- `OPENBLAS_CORETYPE=ARMV8` is exported by the scripts (without it, numpy on JetPack 4 crashes).
- The analysis (pandas) runs on another machine: copy `results/` and run `analyze_cpp.py` there.

The frames can be copied from the Orin instead of being rebuilt from nuScenes:
`scp -r orin:~/edgebench/data/frames ~/edgebench/data/` (setup checks the manifest).

## Repository layout

```
CMakeLists.txt, src/   C++ harness (bench, probe, letterbox kernel, NMS)
scripts/               00_report, 01_build_equiv, 10..19 (one per experiment), run_all, lib.sh,
                       platform.sh (board detection), state.sh, bg.sh (background services)
analysis/              analyze_cpp.py (all tables), equiv.py (C++ vs Python), common.py
data_prep/             engines, nuScenes frame selection, SHA-256 manifests
data/                  local data (outside git, except README and manifests)
results_orin/          paper results on the Orin Nano Super (tables/, platform/)
results_nano/          paper results on the Jetson Nano (tables/, platform/)
legacy/python/         earlier Python harness, used by the equivalence check
tests/                 GPU-free tests (post-processing and letterbox against Python; fake CUDA/TensorRT)
```

## Tests without a Jetson

```bash
# post-processing and letterbox against the Python reference (needs numpy and opencv-python)
g++ -std=c++17 -O2 -Isrc tests/test_host.cpp -o /tmp/test_host && python3 tests/ref_check.py /tmp/test_host /tmp

# run plan of every tier, without touching any hardware
DRY=1 ./scripts/run_all.sh all

# run_bench/find_rate with a fake CUDA/TensorRT (needs the CUDA runtime headers and OpenCV dev)
g++ -std=c++17 -O1 -Isrc -Itests/mock/include -I/usr/local/cuda/include src/bench.cpp tests/mock/mock.cpp \
    -o /tmp/bench_mock $(pkg-config --cflags --libs opencv4) -lpthread
tests/test_scripts.sh /tmp/bench_mock <folder_with_jpgs> /tmp/edgebench_test
```

## Citation

See `CITATION.cff`.

## License

Code: MIT (see `LICENSE`). nuScenes frames are not included and remain under their own license
(CC BY-NC-SA 4.0). YOLOX models are downloaded from the official repository (Apache 2.0).

## Acknowledgment

This work was fully funded through the Rede Agrointeligente project, supported by the Center for
Embedded Devices and Research in Digital Agriculture (CEDRA) of SENAI-RS, with financial resources
from the PPI IoT/Manufatura 4.0 / PPI HardwareBR of MCTI, grant no. 056/2023, under an agreement with
EMBRAPII.
