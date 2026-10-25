#!/usr/bin/env bash
# Downloads the official YOLOX ONNX models (COCO, release 0.1.1rc0) and builds FP16 640x640 engines on
# the device itself. Engines depend on the GPU and the TensorRT version, so they are not in git.
#
# Usage: ./data_prep/build_engines.sh [s m l]     (no arguments: the platform's models)
#   LOCK_CLOCKS=1 (default) locks the clocks with jetson_clocks during the build, so TensorRT's
#   kernel selection does not depend on DVFS; the clocks return to default at the end.
# trtexec options: --fp16, fixed input of the official ONNX (1x3x640x640); everything else is the 10.3 default.
# On TensorRT 8.x (JetPack 4/5) trtexec's default workspace is only 16 MB, which limits kernel
# selection; there we use --workspace=TRT_WS_MB (default 1024).
set -euo pipefail
R=$(cd "$(dirname "$0")/.." && pwd)
D=$R/data
# shellcheck source=../scripts/platform.sh
source "$R/scripts/platform.sh"
TRTEXEC=${TRTEXEC:-/usr/src/tensorrt/bin/trtexec}
URL=https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0
mkdir -p "$D/onnx" "$D/engines"

# jetson_clocks disables the CPU idle states and locks the EMC (bwmgr_halt=1).
# --store goes to a fresh file of its own (avoiding the "Can I overwrite it? Y/N" prompt and not
# relying on /root/.jetsonclocks_conf.txt, which may hold another state). The idle states and
# bwmgr_halt (EMC lock) are read before and checked/restored afterwards.
LOCKED=0
TMPD=$(mktemp -d)
JC="$TMPD/jetsonclocks_store.txt"
SAVE="$TMPD/residuos.txt"; : > "$SAVE"
# shellcheck source=../scripts/state.sh
source "$R/scripts/state.sh"   # EMC_HALT_F (Orin: bwmgr_halt; Nano: override.emc), GPU_3D_F
for f in /sys/devices/system/cpu/cpu*/cpuidle/state*/disable; do
  [ -e "$f" ] && echo "$f $(cat "$f")" >> "$SAVE"
done
h=$(sudo cat "$EMC_HALT_F" 2>/dev/null || true); [ -n "$h" ] && echo "$EMC_HALT_F $h" >> "$SAVE"
[ -n "$GPU_3D_F" ] && echo "$GPU_3D_F $(cat "$GPU_3D_F")" >> "$SAVE"
restore_all() {
  if [ "$LOCKED" = 1 ]; then
    sudo jetson_clocks --restore "$JC" </dev/null
    local bad=0 f v c
    while read -r f v; do
      c=$(sudo cat "$f" 2>/dev/null || echo "?")
      if [ "$c" != "$v" ]; then echo "$v" | sudo tee "$f" >/dev/null; bad=$((bad + 1)); fi
    done < "$SAVE"
    [ "$bad" = 0 ] || echo ">> WARNING: --restore left $bad value(s) different (idle states/EMC/GPU scaling); restored"
    echo ">> clocks, idle states and EMC restored"
  fi
  sudo rm -rf "$TMPD" 2>/dev/null || rm -rf "$TMPD"
}
if [ "${LOCK_CLOCKS:-1}" = 1 ] && command -v jetson_clocks >/dev/null; then
  sudo jetson_clocks --store "$JC" </dev/null >/dev/null
  sudo jetson_clocks && LOCKED=1
fi
trap restore_all EXIT

WS=()
TRT_HELP=$("$TRTEXEC" --help 2>&1 || true)
[[ "$TRT_HELP" == *--memPoolSize* ]] || WS=(--workspace="${TRT_WS_MB:-1024}")

for m in ${*:-$PLAT_MODELS}; do
  o="$D/onnx/yolox_$m.onnx"; e="$D/engines/yolox_${m}_fp16.engine"
  [ -s "$o" ] || wget -q --show-progress -O "$o" "$URL/yolox_$m.onnx"
  if [ -e "$e" ]; then echo "already exists: $e"; continue; fi
  t0=$(date +%s)
  "$TRTEXEC" --onnx="$o" --fp16 "${WS[@]}" --saveEngine="$e" > "${e%.engine}.build.log" 2>&1 \
    || { echo "ERROR building yolox_$m (see ${e%.engine}.build.log)"; rm -f "$e"; exit 1; }
  echo "engine: $e ($(( $(date +%s) - t0 )) s; log: ${e%.engine}.build.log)"
done
# record of what was used (kept with the results)
{ echo "date: $(date -Iseconds)"; echo "platform: $PLATFORM ($SOC)"; "$TRTEXEC" 2>&1 | grep -m1 -i "TensorRT v" || true
  [ ${#WS[@]} -gt 0 ] && echo "extra options: ${WS[*]}"
  sha256sum "$D"/onnx/*.onnx 2>/dev/null; } > "$D/engines/BUILD_INFO.txt"
