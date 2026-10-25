#!/usr/bin/env bash
# Observer effect: how much the monitoring itself changes what is measured.
# Every monitor that wakes the CPU periodically is load for schedutil, which raises the frequency;
# on Orin, the 1 ms probe lowered the factory latency from 48.3 to 41.7 ms (YOLOX-s, 12 Hz).
# Conditions (the same measurement, changing only what runs alongside):
#   none       nothing besides the bench (reference)
#   tegra5000  tegrastats every 5 s
#   tegra1000  tegrastats every 1 s (the campaign default: TEGRA_MS=1000)
#   tegra100   tegrastats every 100 ms (used by older campaigns)
#   probe1     1 ms probe (18_probe.sh, frequency figures)
#   jtop       jtop service (jetson-stats), if installed
# In each condition: periodic release at the platform period, factory default and locked;
# in the factory cell, also the sustainable rate.
#
# Usage: ./scripts/19_observer.sh s      OBS="none tegra5000 tegra1000 tegra100 probe1 jtop"  OBS_RATE=1
source "$(dirname "$0")/lib.sh"
M=${1:?model s|m|l}
check_built; assert_mode "$BASE_MODE"; clk_save

ENG=$(engine_of "$M"); MODEL=$(model_of "$M")
OUT="$RES/observer"
OBS=${OBS:-"none tegra5000 tegra1000 tegra100 probe1 jtop"}
L=${OBS_LIMIT:-1000}; P=${OBS_PASSES:-2}
B="exp=observer model=$MODEL mode=$BASE_MODE"

JTOP_SVC=$(bg_jtop_svc)   # kept once for all runs with jtop
if [[ " $OBS " == *" jtop "* ]] && [ -z "$JTOP_SVC" ]; then
  echo ">> jtop not installed: the jtop condition is skipped"
  OBS=$(echo "$OBS" | tr ' ' '\n' | grep -vx jtop | tr '\n' ' ')
fi

obs_env() {  # obs_env <condition> -> assignments for env
  case "$1" in
    none) echo "TEGRA_MS=0 PROBE_RUN=0 MONITOR=none" ;;
    tegra5000) echo "TEGRA_MS=5000 PROBE_RUN=0 MONITOR=none" ;;
    tegra1000) echo "TEGRA_MS=1000 PROBE_RUN=0 MONITOR=none" ;;
    tegra100) echo "TEGRA_MS=100 PROBE_RUN=0 MONITOR=none" ;;
    probe1) echo "TEGRA_MS=0 PROBE_RUN=1 MONITOR=none" ;;
    jtop) echo "TEGRA_MS=0 PROBE_RUN=0 MONITOR=jtop" ;;
    *) die "condition $1?" ;;
  esac
}

for CELL in "default default" "lock lock"; do
  set -- $CELL; CPU=$1; GPU=$2
  set_cell "$CPU" "$GPU"
  for O in $OBS; do
    T="${MODEL}_obs-${O}_cpu-${CPU}_gpu-${GPU}"
    C="$B obs=$O cpu=$CPU gpu=$GPU"
    # shellcheck disable=SC2046
    eval "$(obs_env "$O")" run_bench "$OUT" "${T}_periodic" "'$C rel=periodic fps=$P_FPS'" \
      --engine "$ENG" --mode periodic --period-ms "$P_MS" --limit "$L" --passes "$P"
    if [ "$CPU" = default ] && [ "${OBS_RATE:-1}" = 1 ]; then
      eval "$(obs_env "$O")" find_rate "$OUT" "$T" "'$C'" "$RATE_LO" "$RATE_HI" --engine "$ENG"
    fi
  done
done
log "19_observer $MODEL done"
