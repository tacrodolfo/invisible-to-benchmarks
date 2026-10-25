#!/usr/bin/env bash
# Checks (and, with --install, installs) everything the campaigns use, and builds the harness.
#
#   ./scripts/setup.sh            only checks and lists what is missing
#   ./scripts/setup.sh --install  installs what is missing (apt), builds and generates the data:
#       NUSCENES=/path/to/nuscenes ./scripts/setup.sh --install
#   (the extracted root folder of v1.0-mini, containing samples/, sweeps/ and v1.0-mini/)
#   DATA_SRC=/folder              instead of generating, links existing data in that folder
#
# Platforms: Orin (JetPack 6: L4T r36, CUDA 12.x, TensorRT 10.x), the paper's reference, and
# Jetson Nano (JetPack 4.6: L4T r32, CUDA 10.2, TensorRT 8.2). System packages come from apt
# (Ubuntu and NVIDIA repositories, already configured in JetPack); on JetPack 4 the apt CMake is
# too old and a newer one is installed with pip for the user (~/.local/bin).
set -uo pipefail
D=$(cd "$(dirname "$0")/.." && pwd)
INSTALL=0; [ "${1:-}" = "--install" ] && INSTALL=1
_SETUP_ORIG_PATH=$PATH   # before platform.sh adds ~/.local/bin
# shellcheck source=platform.sh
source "$D/scripts/platform.sh"
# shellcheck source=bg.sh
source "$D/scripts/bg.sh"

ok=0; miss=0; MISSING_PKGS=(); NOTES=(); PIP_PKGS=()
ver_ge() { [ "$(printf '%s\n%s\n' "$2" "$1" | sort -V | head -1)" = "$2" ]; }  # ver_ge a b: a >= b
pass() { printf "  [ok]    %s\n" "$1"; ok=$((ok + 1)); }
fail() { printf "  [miss]  %s\n" "$1"; miss=$((miss + 1)); [ -n "${2:-}" ] && MISSING_PKGS+=("$2"); [ -n "${3:-}" ] && NOTES+=("$3"); }
has_pkg() { dpkg-query -W -f='${Status}' "$1" 2>/dev/null | grep -q "install ok installed"; }

echo "== platform"
if [ -f /etc/nv_tegra_release ]; then
  pass "Jetson: $(head -1 /etc/nv_tegra_release | sed 's/^# //; s/, GCID.*//')"
  pass "platform: $PLATFORM ($SOC), default mode $BASE_MODE, period $P_MS ms, models: $PLAT_MODELS"
  [ "$PLATFORM" != unknown ] || NOTES+=("SoC not recognized; using generic defaults (see scripts/platform.sh)")
else
  fail "does not look like a Jetson (/etc/nv_tegra_release missing)" "" "the campaigns require a Jetson (Orin with JetPack 6 or Nano with JetPack 4.6)"
fi
CUDA_VER=$(sed -n 's/.*"cuda" *: *{ *"name"[^}]*"version" *: *"\([0-9]*\)\.\([0-9]*\).*/\1-\2/p' /usr/local/cuda/version.json 2>/dev/null | head -1)
[ -n "$CUDA_VER" ] || CUDA_VER=$(ls -d /usr/local/cuda-* 2>/dev/null | sed -n 's#.*/cuda-\([0-9]*\)\.\([0-9]*\)$#\1-\2#p' | sort -V | tail -1)
[ -n "$CUDA_VER" ] && pass "CUDA ${CUDA_VER/-/.}" || fail "CUDA not found in /usr/local/cuda" "" "install JetPack (nvidia-jetpack)"
CV=${CUDA_VER:-12-6}

echo "== build tools"
for t in g++ make pkg-config; do
  command -v $t >/dev/null && pass "$t" || fail "$t" "$t"
done
CMV=$(cmake --version 2>/dev/null | head -1 | awk '{print $3}')
if [ -n "$CMV" ] && ver_ge "$CMV" 3.18; then
  pass "cmake $CMV ($(command -v cmake))"
  # the scripts put ~/.local/bin on the PATH; the user's terminal does not
  if [ "$(command -v cmake)" = "$HOME/.local/bin/cmake" ] && ! (echo ":${_SETUP_ORIG_PATH:-}:" | grep -q ":$HOME/.local/bin:"); then
    printf "  [note]  the new cmake is in ~/.local/bin, outside your terminal's PATH; to build by hand:\n"
    printf "          echo 'export PATH=\$HOME/.local/bin:\$PATH' >> ~/.bashrc && source ~/.bashrc\n"
  fi
else
  # apt's is 3.10 on Ubuntu 18.04; pip installs a newer one for the user (up to 3.27: Python 3.6)
  fail "cmake >= 3.18 (found: ${CMV:-none})"
  PIP_PKGS+=("cmake>=3.18,<3.28")
fi
[ -x /usr/local/cuda/bin/nvcc ] && pass "nvcc" || fail "nvcc (/usr/local/cuda/bin/nvcc)" "cuda-nvcc-$CV"
[ -f /usr/local/cuda/include/cuda_runtime.h ] && pass "CUDA runtime (headers)" || fail "CUDA runtime (headers)" "cuda-cudart-dev-$CV"
ls /usr/include/aarch64-linux-gnu/NvInfer.h >/dev/null 2>&1 && pass "TensorRT (headers)" || fail "TensorRT (headers)" "libnvinfer-dev"
pkg-config --exists opencv4 2>/dev/null && pass "OpenCV $(pkg-config --modversion opencv4)" || fail "OpenCV (dev)" "libopencv-dev"
if [ -f /usr/local/cuda/include/nvjpeg.h ] && ls /usr/local/cuda/lib64/libnvjpeg.so >/dev/null 2>&1; then
  pass "nvJPEG (GPU decoding variants of 12_inputs.sh)"
elif [ "$NVJPEG_EXPECTED" = 1 ]; then
  fail "nvJPEG (without it, the nvjpeg and nvjpeg_hybrid input variants are skipped)" "libnvjpeg-$CV libnvjpeg-dev-$CV"
else
  printf "  [note]  nvJPEG not available on this platform: nvjpeg variants are skipped\n"
fi

echo "== campaign tools"
TRTEXEC=${TRTEXEC:-/usr/src/tensorrt/bin/trtexec}
[ -x "$TRTEXEC" ] && pass "trtexec" || fail "trtexec ($TRTEXEC)" "libnvinfer-bin"
for t in tegrastats nvpmodel jetson_clocks; do
  command -v $t >/dev/null || [ -x /usr/bin/$t ] || [ -x /usr/sbin/$t ] && pass "$t" || fail "$t" "nvidia-l4t-tools"
done
command -v stress-ng >/dev/null && pass "stress-ng (co-runner)" || fail "stress-ng (co-runner)" "stress-ng"
command -v tmux >/dev/null && pass "tmux" || fail "tmux" "tmux"
command -v git >/dev/null && pass "git" || fail "git" "git"
command -v wget >/dev/null && pass "wget (build_engines.sh)" || fail "wget" "wget"

echo "== Python (analysis and data preparation)"
command -v python3 >/dev/null && pass "python3 $(python3 -V 2>&1 | cut -d' ' -f2)" || fail "python3" "python3"
# Orin: the analysis runs on the board. On the others (Python 3.6 on JetPack 4), only numpy and cv2
# are needed (frame preparation); the analysis runs on another machine with the results copied over.
for m in numpy:python3-numpy cv2:libopencv-python pandas:python3-pandas scipy:python3-scipy matplotlib:python3-matplotlib; do
  mod=${m%%:*}; pkg=${m##*:}
  if python3 -c "import $mod" 2>/dev/null; then
    pass "python: $mod $(python3 -c "import $mod; print(getattr($mod, '__version__', ''))" 2>/dev/null)"
  elif [ "$PLATFORM" != orin ] && [ "$mod" != numpy ] && [ "$mod" != cv2 ]; then
    printf "  [note]  python: %s missing (analysis only; run it on another machine)\n" "$mod"
  else
    # cv2 comes from JetPack's OpenCV (libopencv-python); do not use Ubuntu's python3-opencv
    fail "python: $mod" "$pkg"
  fi
done
[ "$PLATFORM" = nano ] && printf "  [note]  OPENBLAS_CORETYPE=%s (exported by the scripts; without it numpy crashes with 'Illegal instruction')\n" "$OPENBLAS_CORETYPE"

echo "== data (engines, frames, manifest)"
# By default, the data are generated from the public sources:
#   engines: official YOLOX ONNX models built here (data_prep/build_engines.sh)
#   frames:  nuScenes v1.0-mini sequence (NUSCENES=/extracted/root, with samples/ and sweeps/)
# With DATA_SRC=/folder, links existing data in that folder (yolox_*_fp16.engine, frames/,
# results/) instead of generating them; useful when the environment is already set up.
DATA=$D/data
SRC=${DATA_SRC:-}
NUSC=${NUSCENES:-}
# accepts either the extracted root or the v1.0-mini subfolder (images live in the root)
if [ -n "$NUSC" ] && [ ! -d "$NUSC/samples/CAM_FRONT" ] && [ -d "$NUSC/../samples/CAM_FRONT" ]; then
  NUSC=$(cd "$NUSC/.." && pwd)
fi
mkdir -p "$DATA/engines"
GOT_FRAMES=0
for m in $PLAT_MODELS; do
  e="$DATA/engines/yolox_${m}_fp16.engine"
  if [ -s "$e" ]; then pass "engine yolox_$m"
  elif [ -n "$SRC" ] && [ -f "$SRC/yolox_${m}_fp16.engine" ]; then
    if [ "$INSTALL" = 1 ]; then ln -sfn "$SRC/yolox_${m}_fp16.engine" "$e" && pass "engine yolox_$m (linked to $SRC)"
    else fail "engine yolox_$m (exists in $SRC; --install links it)"; fi
  elif [ "$INSTALL" = 1 ] && [ -x "${TRTEXEC:-/usr/src/tensorrt/bin/trtexec}" ]; then
    echo "  ..      building yolox_$m (official ONNX + trtexec, a few minutes)"
    if "$D/data_prep/build_engines.sh" "$m" >/dev/null 2>&1 && [ -s "$e" ]; then pass "engine yolox_$m (built)"
    else fail "engine yolox_$m: build failed (see ${e%.engine}.build.log)"; fi
  else
    fail "engine yolox_$m" "" "engines: ./scripts/setup.sh --install builds them from the official YOLOX ONNX models"
  fi
done
if [ -n "$(ls "$DATA/frames"/*.jpg 2>/dev/null | head -1)" ]; then
  GOT_FRAMES=1; pass "frames: $(ls "$DATA/frames"/*.jpg | wc -l) in data/frames"
elif [ -n "$SRC" ] && [ -d "$SRC/frames" ] && [ "$INSTALL" = 1 ]; then
  ln -sfn "$SRC/frames" "$DATA/frames" && GOT_FRAMES=1 && pass "frames (linked to $SRC/frames)"
elif [ -n "$NUSC" ] && [ "$INSTALL" = 1 ]; then
  if [ -d "$NUSC/sweeps/CAM_FRONT" ] && [ -d "$NUSC/samples/CAM_FRONT" ]; then
    python3 "$D/data_prep/select_nuscenes.py" "$DATA/frames" 1500 "$NUSC/sweeps/CAM_FRONT" "$NUSC/samples/CAM_FRONT" | tail -1
    GOT_FRAMES=1; pass "frames assembled from $NUSC"
  else
    fail "frames: $NUSC has no sweeps/CAM_FRONT and samples/CAM_FRONT" "" "NUSCENES must point to the extracted root of nuScenes v1.0-mini (the folder with samples/, sweeps/ and v1.0-mini/)"
  fi
else
  fail "frames" "" "frames: download nuScenes v1.0-mini (nuscenes.org, free registration), extract it and run NUSCENES=/extracted/root ./scripts/setup.sh --install (or copy data/frames from another board already set up)"
fi
if [ "$GOT_FRAMES" = 1 ]; then
  if [ -f "$DATA/frames.sha256" ]; then
    if python3 "$D/data_prep/manifest.py" verify "$DATA/frames" "$DATA/frames.sha256" >/tmp/edgebench_manifest 2>&1; then
      pass "frames identical to the paper's (data/frames.sha256)"
    else
      fail "frames DIFFERENT from the paper's: $(cat /tmp/edgebench_manifest)" "" "check the nuScenes version (v1.0-mini) and rebuild data/frames"
    fi
  else
    fail "manifest data/frames.sha256 missing: there is no way to prove the frames are the paper's" "" \
      "the manifest ships with the repository (data/frames.sha256); without it, check your copy. Authors: python3 data_prep/manifest.py create data/frames data/frames.sha256"
  fi
  # provenance: select_nuscenes.py writes frames/origin.csv; data/origin.csv is the reference
  if [ -f "$DATA/frames/origin.csv" ]; then
    if [ ! -f "$DATA/origin.csv" ]; then
      cp "$DATA/frames/origin.csv" "$DATA/origin.csv" && pass "origin.csv copied to data/"
    elif cmp -s "$DATA/frames/origin.csv" "$DATA/origin.csv"; then
      pass "origin.csv of the frames matches the reference (data/origin.csv)"
    else
      fail "origin.csv of the frames DIFFERS from the reference (data/origin.csv)" "" "the nuScenes selection did not reproduce the same sequence; check the dataset version"
    fi
  fi
fi
if [ -n "$SRC" ] && [ ! -e "$DATA/py_results" ] && [ -d "$SRC/results" ] && [ "$INSTALL" = 1 ]; then
  ln -sfn "$SRC/results" "$DATA/py_results" && pass "Python harness results linked (equivalence check)"
fi

echo "== background services (stopped during the campaigns with QUIET=1)"
bg_status | sed 's/^/  [info]  /'

echo "== permissions"
if sudo -n true 2>/dev/null; then pass "sudo (no password prompt right now)"
else printf "  [note]  sudo asks for a password: run_all.sh asks once and renews it during the campaign\n"; fi

echo
echo "== summary: $ok ok, $miss missing"
[ "$miss" -gt 0 ] && [ -z "$(printf "%s" "${MISSING_PKGS[@]:-}")" ] && [ "$INSTALL" = 0 ] && echo "   run: NUSCENES=/nuscenes/root ./scripts/setup.sh --install"
PKGS=$(printf "%s\n" "${MISSING_PKGS[@]:-}" | tr ' ' '\n' | grep -v '^$' | sort -u | tr '\n' ' ')
for n in "${NOTES[@]:-}"; do [ -n "$n" ] && echo "   note: $n"; done

if [ -n "$PKGS" ] || [ ${#PIP_PKGS[@]} -gt 0 ]; then
  [ -n "$PKGS" ] && echo "   packages to install (apt): $PKGS"
  [ ${#PIP_PKGS[@]} -gt 0 ] && echo "   packages to install (pip3 --user): ${PIP_PKGS[*]}"
  if [ "$INSTALL" = 1 ] && [ -z "${SETUP_REEXEC:-}" ]; then
    if [ -n "$PKGS" ]; then
      sudo apt-get update -q && sudo apt-get install -y $PKGS || { echo "ERROR: apt failed"; exit 1; }
    fi
    if [ ${#PIP_PKGS[@]} -gt 0 ]; then
      pip3 install --user "${PIP_PKGS[@]}" || { echo "ERROR: pip3 failed"; exit 1; }
      hash -r
    fi
    echo ">> installed; checking again"; SETUP_REEXEC=1 exec "$0" "$@"
  elif [ "$INSTALL" = 1 ]; then
    echo "ERROR: something is still missing after the installation (see above)"; exit 1
  else
    echo "   to install: ./scripts/setup.sh --install"
    exit 1
  fi
fi

if [ "$INSTALL" = 1 ] || [ ! -x "$D/build/bench" ]; then
  echo "== building"
  cmake -S "$D" -B "$D/build" -DCMAKE_BUILD_TYPE=Release | grep -E "nvJPEG|Error|error" || true
  cmake --build "$D/build" -j4 2>&1 | tail -2
fi
if [ -x "$D/build/bench" ]; then
  NVJ=$("$D/build/bench" --engine x --tag x --decode gpu --pre gpu 2>&1 || true)
  if [[ "$NVJ" == *"without nvJPEG"* ]]; then
    if [ "$NVJPEG_EXPECTED" = 1 ]; then
      echo "WARNING: build/bench was built without nvJPEG. Rebuild: rm -rf build && ./scripts/setup.sh --install"
      exit 1
    fi
    echo ">> ready: build/bench and build/probe built (no nvJPEG on this platform)"
  else
    echo ">> ready: build/bench and build/probe built, with nvJPEG"
  fi
else
  echo "ERROR: build/bench was not produced (see the cmake output above)"; exit 1
fi
