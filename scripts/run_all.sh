#!/usr/bin/env bash
# Runs the campaigns by priority tier. Runs already done are skipped, so the campaign can be
# interrupted (Ctrl+C) and resumed with the same command.
#
#   ./scripts/run_all.sh 1      tier 1: YOLOX-s, covers every experiment in the paper
#   ./scripts/run_all.sh 2      tier 2: the board's other models (Orin: YOLOX-m and -l)
#   ./scripts/run_all.sh 3      tier 3: two-stage pipeline, maximum power mode (if any), extras
#   ./scripts/run_all.sh 1 2 3  all three in sequence, in a single call (same as "all")
#   DRY=1 ./scripts/run_all.sh 1   only lists the runs and estimates the duration
#
# Asks for the sudo password once and renews it every 60 s while running (the renewal loop is
# stopped on exit, including Ctrl+C). Before and after each tier it records the platform state
# (mode, idle states, bwmgr_halt, fan, boot) in
# results/platform/campaign_<date>.txt.
#
# Prerequisites: ./scripts/00_report.sh boot (right after a reboot) and ./scripts/01_build_equiv.sh
# With QUIET=1 (default), jtop and the desktop environment stay stopped for the whole campaign and
# are restarted at the end (also if it is interrupted with Ctrl+C).
set -uo pipefail
D=$(cd "$(dirname "$0")" && pwd)
[ $# -gt 0 ] || { echo "usage: $0 1|2|3|all [more tiers...]"; exit 1; }
TIERS=()
for a in "$@"; do
  case "$a" in
    1|2|3) TIERS+=("$a") ;;
    all) TIERS+=(1 2 3) ;;
    *) echo "tier $a? (use 1, 2, 3 or all)"; exit 1 ;;
  esac
done
export DRY=${DRY:-0} EST_KEEP=1 EST_FILE=/tmp/edgebench_total_$$
rm -f "$EST_FILE"
# shellcheck source=platform.sh
source "$D/platform.sh"
# shellcheck source=bg.sh
source "$D/bg.sh"
# shellcheck source=state.sh
source "$D/state.sh"
RES=${RES:-$(cd "$D/.." && pwd)/results}
CLOG="$RES/platform/campaign_$(date +%Y%m%d_%H%M).txt"
OTHERS=$(echo "$PLAT_MODELS" | tr ' ' '\n' | grep -vx s | tr '\n' ' ')
echo ">> platform: $PLATFORM ($SOC), mode $BASE_MODE, period $P_MS ms ($P_FPS fps), models: $PLAT_MODELS"
SUDO_KEEP=""
finish() {
  local rc=$?
  trap - EXIT INT TERM
  if [ "$DRY" != 1 ]; then
    [ "${QUIET:-1}" = 1 ] && bg_restore_all
    if [ -n "$SUDO_KEEP" ]; then pkill -P "$SUDO_KEEP" 2>/dev/null; kill "$SUDO_KEEP" 2>/dev/null; wait "$SUDO_KEEP" 2>/dev/null; fi
    state_log "end (code $rc, after restarting the services)"
    echo ">> platform state log in $CLOG"
  fi
  rm -f /tmp/edgebench_plan_$$ "$EST_FILE"
  exit "$rc"
}
trap finish EXIT
trap 'echo; echo ">> interrupted"; exit 130' INT TERM

state_log() {  # state_log <label>: records and prints the platform state
  [ "$DRY" = 1 ] && return 0
  local l; l="$(date '+%F %T') $1: $(platform_state) bg=$(bg_now)"
  echo "$l" >> "$CLOG"
  echo ">> $l"
}

if [ "$DRY" != 1 ]; then
  sudo -v || exit 1
  # sudo renewal during the whole campaign (stopped in finish)
  ( while true; do sudo -n true 2>/dev/null; sleep 60; done ) &
  SUDO_KEEP=$!
  disown "$SUDO_KEEP" 2>/dev/null || true   # no "Terminated" message at the end
  mkdir -p "$(dirname "$CLOG")"
  echo "# campaign: tiers ${TIERS[*]} on $PLATFORM ($SOC), period $P_MS ms; $(date '+%F %T')" >> "$CLOG"
  state_log "start (before QUIET)"
  if [ "${QUIET:-1}" = 1 ]; then
    export EDGEBENCH_BG_OWNER=$$
    bg_restore_all   # leftovers from an interrupted campaign
    bg_stop
    sleep 5
  fi
fi

step() {  # step <script> [args]   (extra variables via env before the name)
  echo
  echo "################################################################"
  echo "## $(date +%H:%M:%S) $*"
  echo "################################################################"
  local before=0 after=0
  [ -f "$EST_FILE" ] && before=$(cat "$EST_FILE")
  "$@" || echo "!! $* ended with an error; continuing"
  if [ "$DRY" = 1 ]; then
    [ -f "$EST_FILE" ] && after=$(cat "$EST_FILE")
    awk -v a="$before" -v b="$after" -v n="${*//$D\//}" 'BEGIN{printf "== %-40s %6.1f h\n", n, (b-a)/3600}' >> /tmp/edgebench_plan_$$
  fi
}

tier1() {
  step "$D/19_observer.sh" s
  step "$D/10_factorial.sh" s
  step "$D/13_sync.sh" s
  step "$D/12_inputs.sh" s
  step "$D/14_trtexec_idle.sh" s
  step "$D/11_rates.sh" s seq
  step "$D/15_floors.sh" s
}
tier2() {
  for M in $OTHERS; do
    step "$D/10_factorial.sh" "$M"
    step "$D/11_rates.sh" "$M" seq
    step "$D/14_trtexec_idle.sh" "$M"
    step env FIND_RATE=0 VARIANTS="jpeg_cpu raw_cpu nvjpeg raw_gpu" "$D/12_inputs.sh" "$M"
  done
  step "$D/18_probe.sh" s
}
tier3() {
  for M in $PLAT_MODELS; do step "$D/11_rates.sh" "$M" pipe; done
  step "$D/16_maxn.sh" $PLAT_MODELS   # skips itself if the default mode is already the maximum
  step "$D/17_idle_states.sh" s
  for M in $OTHERS; do step "$D/18_probe.sh" "$M"; done
  for M in $OTHERS; do step env FIND_RATE=0 SYNCS="stage:auto edge:block" "$D/13_sync.sh" "$M"; done
}

rm -f /tmp/edgebench_plan_$$
for T in "${TIERS[@]}"; do
  state_log "before tier $T"
  case "$T" in
    1) tier1 ;;
    2) tier2 ;;
    3) tier3 ;;
  esac
  state_log "after tier $T"
  [ "$DRY" = 1 ] || echo ">> tier $T done $(date '+%F %T')"
done

if [ "$DRY" = 1 ]; then
  echo; echo "==================== plan (tiers ${TIERS[*]})"
  cat /tmp/edgebench_plan_$$
  awk '{s+=$(NF-1)} END{printf "== %-40s %6.1f h\n", "TOTAL", s}' /tmp/edgebench_plan_$$
else
  echo ">> run_all ${TIERS[*]} done $(date '+%F %T')"
fi
