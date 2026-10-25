# shellcheck shell=bash
# Platform detection (sourced by lib.sh and setup.sh). Defines:
#   SOC            tegra234 | tegra194 | tegra210 | unknown
#   PLATFORM       orin | xavier | nano | unknown
#   BASE_MODE      factory default power mode (PM_CONFIG DEFAULT in /etc/nvpmodel.conf)
#   PLAT_MODELS    models that fit on the board (engines built by setup)
#   PLAT_PERIOD_MS camera period used in the campaigns
#   RATE_LO/HI     initial range of the sustainable-rate search (fps; the search extends itself)
#   PLAT_INPUT_LIMIT  frames of the input variants (raw frames take ~4.3 MB each)
#   NVJPEG_EXPECTED   1 if the platform has nvJPEG in apt (setup requires it)
#   FAN_PROFILE_EXPECTED  optional: expected fan profile (default: FAN_DEFAULT_PROFILE in
#                  /etc/nvfancontrol.conf; see state.sh)
#   QUIET_PROCS    optional, empty by default: names of programs to freeze (SIGSTOP/SIGCONT)
#                  during the campaign with QUIET=1, for local processes that cannot be
#                  stopped (e.g. QUIET_PROCS="jupyter-lab"). Nothing is frozen unless requested.
# Everything can be overridden in the environment (e.g. PERIOD_MS=33.333).

SOC=unknown
if [ -r /proc/device-tree/compatible ]; then
  case "$(tr '\0' ' ' < /proc/device-tree/compatible)" in
    *tegra234*) SOC=tegra234 ;;
    *tegra194*) SOC=tegra194 ;;
    *tegra210*) SOC=tegra210 ;;
  esac
fi
case "$SOC" in
  tegra234) PLATFORM=orin ;;
  tegra194) PLATFORM=xavier ;;
  tegra210) PLATFORM=nano ;;
  *) PLATFORM=unknown ;;
esac
PLATFORM=${PLATFORM_OVERRIDE:-$PLATFORM}

# factory default mode: name of the POWER_MODEL whose id is PM_CONFIG DEFAULT
_nvp_default_mode() {
  local f=${NVP_CONF:-/etc/nvpmodel.conf} id
  [ -r "$f" ] || return 0
  id=$(sed -n 's/^< *PM_CONFIG DEFAULT=\([0-9]*\) *>.*/\1/p' "$f" | head -1)
  [ -n "$id" ] || return 0
  sed -n "s/^< *POWER_MODEL ID=$id NAME=\([^ >]*\).*/\1/p" "$f" | head -1
}
BASE_MODE=${BASE_MODE:-$(_nvp_default_mode)}

case "$PLATFORM" in
  orin)
    BASE_MODE=${BASE_MODE:-25W}
    PLAT_MODELS=${PLAT_MODELS:-"s m l"}
    PLAT_PERIOD_MS=${PLAT_PERIOD_MS:-83.333}   # 12 Hz (nuScenes CAM_FRONT)
    RATE_LO=${RATE_LO:-8}; RATE_HI=${RATE_HI:-40}
    PLAT_INPUT_LIMIT=${PLAT_INPUT_LIMIT:-400}
    NVJPEG_EXPECTED=1 ;;
  xavier)
    PLAT_MODELS=${PLAT_MODELS:-"s m l"}
    PLAT_PERIOD_MS=${PLAT_PERIOD_MS:-83.333}
    RATE_LO=${RATE_LO:-6}; RATE_HI=${RATE_HI:-30}
    PLAT_INPUT_LIMIT=${PLAT_INPUT_LIMIT:-400}
    NVJPEG_EXPECTED=0 ;;
  nano)
    # Maxwell, 4 GB: YOLOX-s only; at 12 Hz the pipeline does not fit in the period even with locked
    # clocks, so the default is 5 Hz (confirm with B0 and the rate search; PERIOD_MS=... to change it)
    PLAT_MODELS=${PLAT_MODELS:-"s"}
    PLAT_PERIOD_MS=${PLAT_PERIOD_MS:-200}
    RATE_LO=${RATE_LO:-2}; RATE_HI=${RATE_HI:-10}
    PLAT_INPUT_LIMIT=${PLAT_INPUT_LIMIT:-150}
    NVJPEG_EXPECTED=0
    export OPENBLAS_CORETYPE=${OPENBLAS_CORETYPE:-ARMV8} ;;  # JetPack 4 numpy/cv2 without it: "Illegal instruction"
  *)
    PLAT_MODELS=${PLAT_MODELS:-"s"}
    PLAT_PERIOD_MS=${PLAT_PERIOD_MS:-83.333}
    RATE_LO=${RATE_LO:-4}; RATE_HI=${RATE_HI:-40}
    PLAT_INPUT_LIMIT=${PLAT_INPUT_LIMIT:-150}
    NVJPEG_EXPECTED=0 ;;
esac

# rates (fps) of the 1 ms probe runs (12_inputs.sh, 18_probe.sh) and of the rate sweep in
# 11_rates.sh (P99/miss vs rate curves), with the number of frames per rate
case "$PLATFORM" in
  nano)
    PROBE_RATES_IN=${PROBE_RATES_IN:-"5 8"}; PROBE_RATES=${PROBE_RATES:-"5 6 7"}
    PLAT_SWEEP_RATES=${PLAT_SWEEP_RATES:-"1 2 3 4 5 6 7"}; PLAT_SWEEP_LIMIT=${PLAT_SWEEP_LIMIT:-300} ;;
  *)
    PROBE_RATES_IN=${PROBE_RATES_IN:-"12 30"}; PROBE_RATES=${PROBE_RATES:-"12 20 30"}
    PLAT_SWEEP_RATES=${PLAT_SWEEP_RATES:-"5 10 12 15 20 25 30"}; PLAT_SWEEP_LIMIT=${PLAT_SWEEP_LIMIT:-1000} ;;
esac

# campaign period and rate (PERIOD_MS in the environment takes precedence)
P_MS=${PERIOD_MS:-$PLAT_PERIOD_MS}
P_FPS=$(awk -v p="$P_MS" 'BEGIN{v = 1000 / p; r = int(v + 0.5); if (v - r < 0.01 && r - v < 0.01) print r; else printf "%.1f\n", v}')

# cmake installed via pip (JetPack 4: apt's is too old)
case ":$PATH:" in *":$HOME/.local/bin:"*) ;; *) PATH="$HOME/.local/bin:$PATH" ;; esac
export PATH
