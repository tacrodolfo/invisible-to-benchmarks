#!/usr/bin/env bash
# Functions shared by the experiment scripts. Usage: source "$(dirname "$0")/lib.sh"
#
# Variables that can be overridden in the environment:
#   DATA     data folder (default: the repository's data/) with engines/, frames/ and py_results/
#   FRAMES   folder with the JPEGs (default: $DATA/frames)
#   DRY=1    only lists the runs and estimates the time, without touching the Jetson
#   COOL     pause between runs in s (default 20)
#   TEGRA_MS tegrastats interval in ms (default 1000; 0 = no tegrastats). Each sample wakes the
#            CPU and schedutil reacts to it: short intervals mask part of the DVFS effect
#   QUIET    1 (default) stops services that wake the CPU/GPU periodically (jtop, desktop
#            environment) during the campaign and restarts them at the end; 0 = leave as is (recorded in the .cond)
#   PERIOD_MS camera period (platform default: Orin 83.333 ms = 12 Hz)
set -euo pipefail

CPP_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
DATA=${DATA:-$CPP_DIR/data}
FRAMES=${FRAMES:-$DATA/frames}
BENCH=${BENCH:-$CPP_DIR/build/bench}
PROBE=${PROBE:-$CPP_DIR/build/probe}
TRTEXEC=${TRTEXEC:-/usr/src/tensorrt/bin/trtexec}
RES=${RES:-$CPP_DIR/results}
STATE_DIR=$CPP_DIR/state
DRY=${DRY:-0}
COOL=${COOL:-20}
N_ALL=${N_ALL:-1652}  # frames in the folder (for the time estimate)
EST_FILE=${EST_FILE:-/tmp/edgebench_est_$$}
TEGRA_MS=${TEGRA_MS:-1000}
mkdir -p "$RES" "$STATE_DIR"
# shellcheck source=platform.sh
source "$CPP_DIR/scripts/platform.sh"
# shellcheck source=bg.sh
source "$CPP_DIR/scripts/bg.sh"
# shellcheck source=state.sh
source "$CPP_DIR/scripts/state.sh"

log() { echo "######## $(date +%H:%M:%S) | $*"; }
die() { echo "ERROR: $*" >&2; exit 1; }

engine_of() { echo "$DATA/engines/yolox_${1}_fp16.engine"; }
model_of() { echo "yolox_${1}"; }

# ms per frame in back-to-back, only for the DRY time estimate
b2b_ms() {
  if [ "$PLATFORM" = nano ]; then echo 160; return; fi
  case "$1" in s) echo 55 ;; m) echo 80 ;; l) echo 120 ;; *) echo 60 ;; esac
}

est_add() {  # adds seconds to the estimate
  [ "$DRY" = 1 ] || return 0
  local cur=0
  [ -f "$EST_FILE" ] && cur=$(cat "$EST_FILE")
  awk -v a="$cur" -v b="$1" 'BEGIN{printf "%.0f\n", a+b}' > "$EST_FILE"
}
est_report() {
  [ "$DRY" = 1 ] || return 0
  local s=0
  [ -f "$EST_FILE" ] && s=$(cat "$EST_FILE")
  awk -v s="$s" -v p="$PLATFORM" 'BEGIN{printf ">> estimate: %.1f h (%d min) on %s\n", s/3600, s/60, p}'
}

sx() { if [ "$DRY" = 1 ]; then echo "   [dry] $*"; else "$@"; fi; }

power_mode() {
  local m
  m=$(sudo nvpmodel -q 2>/dev/null | grep -i "power mode" | head -1 | awk -F': ' '{print $2}' | tr -d ' ' || true)
  echo "${m:-unknown}"
}
assert_mode() {
  [ "$DRY" = 1 ] && return 0
  local m; m=$(power_mode)
  [ "$m" = "$1" ] || die "current power mode is $m; expected $1 (sudo nvpmodel -m <id>)"
}
mode_id() {  # mode id from its name, read from /etc/nvpmodel.conf
  grep -E "POWER_MODEL ID=[0-9]+ NAME=$1( |>)" "${NVP_CONF:-/etc/nvpmodel.conf}" | head -1 | sed -E 's/.*ID=([0-9]+).*/\1/'
}

# ------------------------------------------------------------------ clock sysfs
GPU_DEV=$(ls -d /sys/class/devfreq/*gpu* 2>/dev/null | head -1 || true)
POLICIES=$(ls -d /sys/devices/system/cpu/cpufreq/policy* 2>/dev/null || true)

wr() {  # wr value file
  if [ "$DRY" = 1 ]; then return 0; fi
  echo "$1" | sudo tee "$2" >/dev/null || echo "WARNING: failed to write $1 to $2" >&2
}
rd() { cat "$1" 2>/dev/null || sudo cat "$1" 2>/dev/null || echo "n/a"; }

state_file() { echo "$STATE_DIR/clocks_$(power_mode).env"; }

# ------------------------------------------------------------------ invisible residues
# A jetson_clocks undone by another path (nvpmodel -m, sysfs writes) returns the frequencies to
# default but leaves the CPU idle states disabled and the EMC locked at maximum (bwmgr_halt=1).
# None of this shows up in the governors or the frequencies, so both are checked when saving the
# reference state, after each restore and before/after each run. The fan (nvfancontrol profile)
# is recorded as well.
# readers (emc_halt, idle_dis_n, idle_tot, fan_profile, fan_pwm): scripts/state.sh
jc_file() { echo "$STATE_DIR/jetsonclocks_$(boot_id).txt"; }   # this boot's own --store

# Checks the reference state: mode, all idle states enabled, EMC not locked.
# Prints the problems and returns their count. Usage: ref_check <expected mode>
ref_check() {
  local bad=0 m n h
  m=$(power_mode)
  [ "$m" = "$1" ] || { echo "   power mode $m (expected $1)"; bad=$((bad + 1)); }
  n=$(idle_dis_n)
  [ "$n" = 0 ] || { echo "   $n of $(idle_tot) CPU idle states disabled (expected: all enabled)"; bad=$((bad + 1)); }
  h=$(emc_halt)
  [ "$h" = n/a ] || [ "$h" = 0 ] || { echo "   EMC locked ($(basename "$EMC_HALT_F")=$h; expected 0)"; bad=$((bad + 1)); }
  local g; g=$(gpu_3d)
  [ "$g" = n/a ] || [ "$g" = 1 ] || { echo "   GPU frequency scaling off (enable_3d_scaling=$g; expected 1)"; bad=$((bad + 1)); }
  return "$bad"
}

# After a restore: do the idle states and bwmgr_halt match the saved ones? If not, retries and,
# if it persists, stops (die) or only warns (warn). Logs to state/restore_problems.log.
# Usage: verify_state die|warn <where>
verify_state() {
  [ "$DRY" = 1 ] && return 0
  local f; f=$(state_file)
  [ -f "$f" ] || return 0
  local diff try
  for try in 1 2; do
    diff=$(while read -r path val; do
      case "$path" in */cpuidle/*/disable|*bwmgr_halt|*override.emc/clk_state|*enable_3d_scaling|*pwm-fan/temp_control) ;; *) continue ;; esac
      cur=$(rd "$path")
      [ "$cur" = "$val" ] || echo "$path=$cur (saved $val)"
    done < "$f")
    [ -z "$diff" ] && return 0
    echo "WARNING: partial restore ($2): $(echo "$diff" | wc -l) value(s) differ from the saved state" >&2
    echo "$(date -Iseconds) $2: $diff" | tr '\n' ' ' >> "$STATE_DIR/restore_problems.log"
    echo >> "$STATE_DIR/restore_problems.log"
    [ "$try" = 1 ] && { echo ">> restoring the saved state again" >&2; clk_restore_raw; }
  done
  echo "$diff" | head -5 >&2
  [ "$1" = die ] && die "the saved state could not be restored ($2); see state/restore_problems.log"
  return 0
}

# ------------------------------------------------------------------ factory state
# Limits of the current mode read from /etc/nvpmodel.conf (e.g. nvp_get CPU_A78_0 MIN_FREQ).
NVP_CONF=${NVP_CONF:-/etc/nvpmodel.conf}
nvp_get() {
  local mode; mode=$(power_mode)
  awk -v m="$mode" -v k="$1" -v f="$2" '
    /^< *POWER_MODEL/ { inb = ($0 ~ ("NAME=" m "( |>)")) }
    inb && $1 == k && $2 == f { print $3; exit }' "$NVP_CONF" 2>/dev/null || true
}
# CPU limit of the mode for the policy whose first core is <cpu0>. The key name changes with the
# SoC: CPU_A78_0 (Orin), CPU_A57 (Nano), CPU_CARMEL... accepts the key with the core suffix or
# without a numeric suffix.  Usage: nvp_cpu <cpu0> <MIN_FREQ|MAX_FREQ>
nvp_cpu() {
  local mode; mode=$(power_mode)
  awk -v m="$mode" -v c="$1" -v f="$2" '
    /^< *POWER_MODEL/ { inb = ($0 ~ ("NAME=" m "( |>)")) }
    inb && $1 ~ /^CPU_/ && $1 !~ /ONLINE/ && $2 == f && ($1 ~ ("_" c "$") || $1 !~ /_[0-9]+$/) { print $3; exit }' \
    "$NVP_CONF" 2>/dev/null || true
}
# Effective frequency the kernel applies for a request: the lowest available >= v (min) or the
# highest available <= v (max). Usage: eff_freq <min|max> <value> <list of available>
eff_freq() {
  echo "$3" | tr ' ' '\n' | grep -v '^$' | sort -n | awk -v t="$1" -v v="$2" '
    t == "min" && $1 >= v && !done { print; done = 1 }
    t == "max" && $1 <= v { last = $1 }
    END { if (t == "max" && last != "") print last }'
}

# Compares the current state with the mode's default (schedutil and nvhost_podgov governors,
# nvpmodel frequency limits). Prints the differences and returns their count.
check_factory() {
  local bad=0 p cpu0 want got av
  for p in $POLICIES; do
    got=$(rd "$p/scaling_governor")
    [ "$got" = schedutil ] || { echo "   $(basename "$p"): governor $got (expected schedutil)"; bad=$((bad + 1)); }
    cpu0=$(rd "$p/affected_cpus" | awk '{print $1}')
    av=$(rd "$p/scaling_available_frequencies")
    for lim in MIN MAX; do
      want=$(nvp_cpu "$cpu0" "${lim}_FREQ")
      # -1 = no limit (hardware maximum); 0 = hardware minimum (eff_freq resolves the 0)
      [ -n "$want" ] && [ "$want" != -1 ] || continue
      want=$(eff_freq "$(echo $lim | tr A-Z a-z)" "$want" "$av")
      got=$(rd "$p/scaling_$(echo $lim | tr A-Z a-z)_freq")
      [ -z "$want" ] || [ "$got" = "$want" ] || { echo "   $(basename "$p"): ${lim,,} $got kHz (mode: $want)"; bad=$((bad + 1)); }
    done
  done
  got=$(rd "$GPU_DEV/governor")
  [ "$got" = nvhost_podgov ] || { echo "   GPU: governor $got (expected nvhost_podgov)"; bad=$((bad + 1)); }
  av=$(rd "$GPU_DEV/available_frequencies")
  for lim in MIN MAX; do
    want=$(nvp_get GPU "${lim}_FREQ")
    [ -n "$want" ] && [ "$want" != -1 ] || continue
    want=$(eff_freq "$(echo $lim | tr A-Z a-z)" "$want" "$av")
    got=$(rd "$GPU_DEV/$(echo $lim | tr A-Z a-z)_freq")
    [ -z "$want" ] || [ "$got" = "$want" ] || { echo "   GPU: ${lim,,} $got Hz (mode: $want)"; bad=$((bad + 1)); }
  done
  return "$bad"
}

# Returns to the mode's default without a reboot: reapplies the current mode with nvpmodel
# (frequency limits) and restores the governors. Does not touch the idle states (recorded separately).
factory_reset() {
  local id; id=$(mode_id "$(power_mode)" || true)
  [ -n "$id" ] || die "could not find the id of mode $(power_mode) in $NVP_CONF"
  echo ">> reapplying mode $(power_mode) (nvpmodel -m $id) and the default governors"
  sudo nvpmodel -m "$id" >/dev/null 2>&1 || echo "WARNING: nvpmodel -m $id returned an error" >&2
  for p in $POLICIES; do wr schedutil "$p/scaling_governor"; done
  wr nvhost_podgov "$GPU_DEV/governor"
  sleep 2
}

boot_id() { cat /proc/sys/kernel/random/boot_id 2>/dev/null || echo unknown; }

# Saves the factory state (governors, limits and idle states) of the current mode, after checking
# that it matches the mode's default; if it does not, fixes it with factory_reset.
# The saved state is valid for one boot: after a reboot it is saved again, because right after
# boot is when the state is truly the factory one (idle states included, which factory_reset
# cannot check). Also prepares the environment (background services, see bg_prepare).
#   FORCE_SAVE=1  saves the current state even if off-default (or overwrites the saved one)
#   Before writing a boot's reference state it requires: the expected mode (default BASE_MODE;
#   16_maxn.sh passes MAXN_SUPER), all idle states enabled and the EMC not locked
#   (bwmgr_halt=0). Otherwise it refuses instead of writing an altered state.
#   Usage: clk_save [expected mode]
clk_save() {
  [ "$DRY" = 1 ] && return 0
  local want=${1:-$BASE_MODE}
  bg_prepare
  fan_check
  local f; f=$(state_file)
  if [ -f "$f" ] && [ "${FORCE_SAVE:-0}" != 1 ]; then
    if grep -q "^# boot_id $(boot_id)\$" "$f"; then
      # restores the saved state (idle states and EMC included): something between campaigns may have
      # changed it, such as a jetson_clocks applied by jtop when restarted
      clk_restore_raw
      verify_state die "start of $(basename "$0")"
      return 0
    fi
    echo ">> state saved in $f is from another boot; saving this boot's"
    mv -f "$f" "$f.previous_boot"
  fi
  [ -n "$GPU_DEV" ] || die "GPU devfreq not found"
  if ! check_factory > /tmp/edgebench_factory_$$; then
    echo ">> the current state differs from the default of mode $(power_mode):"
    cat /tmp/edgebench_factory_$$
    if [ "${FORCE_SAVE:-0}" != 1 ]; then
      factory_reset
      if ! check_factory > /tmp/edgebench_factory_$$; then
        cat /tmp/edgebench_factory_$$
        rm -f /tmp/edgebench_factory_$$
        die "still off-default after reapplying the mode; reboot or use FORCE_SAVE=1"
      fi
      echo ">> mode default restored"
    fi
  fi
  rm -f /tmp/edgebench_factory_$$
  if ! ref_check "$want" > /tmp/edgebench_ref_$$; then
    echo ">> the current state is NOT a clean reference state:"
    cat /tmp/edgebench_ref_$$; rm -f /tmp/edgebench_ref_$$
    if [ "${FORCE_SAVE:-0}" != 1 ]; then
      echo "   Probably the residue of a jetson_clocks undone by another path. Reboot the board"
      echo "   (with jtop disabled at boot) and run ./scripts/00_report.sh boot before measuring."
      die "refusing to write an altered reference state (FORCE_SAVE=1 writes it anyway)"
    fi
    echo ">> FORCE_SAVE=1: writing anyway"
  fi
  rm -f /tmp/edgebench_ref_$$
  # our own copy of jetson_clocks --store for this boot (a new file: avoids the prompt
  # "Can I overwrite it? Y/N", which would hang the script inside tmux)
  local jf; jf=$(jc_file)
  if [ ! -f "$jf" ] && command -v jetson_clocks >/dev/null; then
    sudo jetson_clocks --store "$jf" </dev/null >/dev/null 2>&1 || echo "WARNING: jetson_clocks --store $jf failed" >&2
  fi
  {
    echo "# boot_id $(boot_id)"
    echo "# saved $(date -Iseconds) platform=$PLATFORM mode=$(power_mode)"
    for p in $POLICIES; do
      echo "$p/scaling_governor $(rd "$p/scaling_governor")"
      echo "$p/scaling_max_freq $(rd "$p/scaling_max_freq")"
      echo "$p/scaling_min_freq $(rd "$p/scaling_min_freq")"
    done
    echo "$GPU_DEV/governor $(rd "$GPU_DEV/governor")"
    echo "$GPU_DEV/max_freq $(rd "$GPU_DEV/max_freq")"
    echo "$GPU_DEV/min_freq $(rd "$GPU_DEV/min_freq")"
    for s in /sys/devices/system/cpu/cpu*/cpuidle/state*/disable; do
      if [ -e "$s" ]; then echo "$s $(rd "$s")"; fi
    done
    local h; h=$(emc_halt)
    [ "$h" = n/a ] || echo "$EMC_HALT_F $h"
    local g; g=$(gpu_3d)
    [ "$g" = n/a ] || echo "$GPU_3D_F $g"
    # Nano: automatic fan (jetson_clocks fixes it at maximum; --restore gives it back)
    if [ -e /sys/devices/pwm-fan/temp_control ]; then
      echo "/sys/devices/pwm-fan/temp_control $(cat /sys/devices/pwm-fan/temp_control)"
    fi
  } > "$f"
  local p0; p0=$(echo "$POLICIES" | head -1)
  echo ">> factory state saved in $f: CPU $(rd "$p0/scaling_governor") $(( $(rd "$p0/scaling_min_freq") / 1000 ))-$(( $(rd "$p0/scaling_max_freq") / 1000 )) MHz, GPU $(rd "$GPU_DEV/governor") $(( $(rd "$GPU_DEV/min_freq") / 1000000 ))-$(( $(rd "$GPU_DEV/max_freq") / 1000000 )) MHz, idle states disabled: $(idle_dis_n)/$(idle_tot), bwmgr_halt=$(emc_halt), fan=$(fan_profile)"
}

# Fan profile: warns (once per script) if it is not the platform's expected one
FAN_CHECKED=""
fan_check() {
  [ "$DRY" = 1 ] && return 0
  [ -n "$FAN_CHECKED" ] && return 0
  FAN_CHECKED=1
  local e p m
  m=$(conf_modified)
  [ -z "$m" ] || echo "WARNING: configuration files changed relative to the L4T packages: $(echo "$m" | tr '\n' ' ')" >&2
  e=$(fan_expected)
  [ -n "$e" ] || return 0
  p=$(fan_profile)
  if [ "$p" != n/a ] && [ "$p" != "$e" ]; then
    echo "WARNING: fan profile '$p' (default in /etc/nvfancontrol.conf: $e). Check with 'sudo nvfancontrol -q'" >&2
  fi
}

saved() {  # saved path -> saved value
  local f; f=$(state_file)
  awk -v k="$1" '$1==k{print $2}' "$f"
}

clk_restore_raw() {  # restores the whole saved state (governors, limits, idle states, EMC)
  [ "$DRY" = 1 ] && return 0
  local f; f=$(state_file)
  [ -f "$f" ] || return 0
  while read -r path val; do
    case "$path" in ''|'#'*) continue ;; esac
    wr "$val" "$path"
  done < "$f"
}
clk_restore() { clk_restore_raw; verify_state warn "${1:-clk_restore}"; }

cpu_set() {  # lock | default | floor:KHZ
  [ "$DRY" = 1 ] && return 0
  for p in $POLICIES; do
    case "$1" in
      lock) wr performance "$p/scaling_governor" ;;
      default)
        wr "$(saved "$p/scaling_governor")" "$p/scaling_governor"
        wr "$(saved "$p/scaling_max_freq")" "$p/scaling_max_freq"
        wr "$(saved "$p/scaling_min_freq")" "$p/scaling_min_freq" ;;
      floor:*)
        wr "$(saved "$p/scaling_governor")" "$p/scaling_governor"
        wr "$(saved "$p/scaling_max_freq")" "$p/scaling_max_freq"
        wr "${1#floor:}" "$p/scaling_min_freq" ;;
      *) die "cpu_set $1" ;;
    esac
  done
}

gpu_set() {  # lock | default | floor:HZ | fix:HZ
  [ "$DRY" = 1 ] && return 0
  local mn mx; mn=$(saved "$GPU_DEV/min_freq"); mx=$(saved "$GPU_DEV/max_freq")
  case "$1" in
    lock) wr "$mx" "$GPU_DEV/max_freq"; wr "$mx" "$GPU_DEV/min_freq" ;;
    default) wr "$mx" "$GPU_DEV/max_freq"; wr "$mn" "$GPU_DEV/min_freq" ;;
    floor:*) wr "$mx" "$GPU_DEV/max_freq"; wr "${1#floor:}" "$GPU_DEV/min_freq" ;;
    fix:*) wr "$mn" "$GPU_DEV/min_freq"; wr "${1#fix:}" "$GPU_DEV/max_freq"; wr "${1#fix:}" "$GPU_DEV/min_freq" ;;
    *) die "gpu_set $1" ;;
  esac
}

idle_set() {  # enable | disable
  [ "$DRY" = 1 ] && return 0
  local v=0; [ "$1" = disable ] && v=1
  for s in /sys/devices/system/cpu/cpu*/cpuidle/state*/disable; do
    if [ -e "$s" ]; then wr "$v" "$s"; fi
  done
}

# Undoes jetson_clocks with this boot's own --store (the global /root/.jetsonclocks_conf.txt
# may hold another state). Without our own, uses the global one.
jc_restore() {
  local jf; jf=$(jc_file)
  if [ -f "$jf" ]; then sx sudo jetson_clocks --restore "$jf" </dev/null
  else sx sudo jetson_clocks --restore </dev/null; fi
}

# Applies a CPU x GPU cell. "jc" = sudo jetson_clocks (vendor reference).
CUR_CELL=""
set_cell() {  # set_cell <cpu> <gpu>
  local want="$1/$2"
  [ "$CUR_CELL" = "$want" ] && return 0
  if [ "$1" = jc ]; then
    sx sudo jetson_clocks
  else
    if [ "$CUR_CELL" = "jc/jc" ]; then
      # --restore uses the jetson_clocks --store file, which may hold another state (it touches
      # the idle states); that is why our saved state is restored right afterwards
      jc_restore
      clk_restore_raw
      verify_state die "leaving the jc cell"
    fi
    cpu_set "$1"; gpu_set "$2"
  fi
  CUR_CELL=$want
  log "clocks: CPU=$1 GPU=$2 - cooling down 30 s"
  [ "$DRY" = 1 ] || sleep 30
  est_add 30
}

clk_show() {
  echo "mode=$(power_mode)"
  for p in $POLICIES; do
    echo "$(basename "$p") gov=$(rd "$p/scaling_governor") min=$(rd "$p/scaling_min_freq") max=$(rd "$p/scaling_max_freq") cur=$(rd "$p/scaling_cur_freq")"
  done
  echo "gpu gov=$(rd "$GPU_DEV/governor") min=$(rd "$GPU_DEV/min_freq") max=$(rd "$GPU_DEV/max_freq") cur=$(rd "$GPU_DEV/cur_freq")"
  echo "emc=$(sudo cat /sys/kernel/debug/bpmp/debug/clk/emc/rate 2>/dev/null || sudo cat /sys/kernel/debug/clk/emc/clk_rate 2>/dev/null || echo n/a)"
  local dis tot
  dis=$(cat /sys/devices/system/cpu/cpu*/cpuidle/state*/disable 2>/dev/null | grep -c 1 || true)
  tot=$(ls /sys/devices/system/cpu/cpu*/cpuidle/state*/disable 2>/dev/null | wc -l || true)
  echo "cpuidle_disabled=$dis/$tot"
}

# ------------------------------------------------------------------ measurement
TEGRA_PID=""
tegra_start() {  # TEGRA_MS=0 disables it
  [ "$DRY" = 1 ] && return 0
  [ "$TEGRA_MS" = 0 ] && return 0
  sudo pkill -x tegrastats 2>/dev/null || true
  date +%s.%N > "$1.t0"   # start in real time (JetPack 4 tegrastats does not timestamp its lines)
  sudo tegrastats --interval "$TEGRA_MS" --logfile "$1" &
  disown $! 2>/dev/null || true   # no "Terminated" message when pkill ends it
  sleep 1
}
tegra_stop() {
  [ "$DRY" = 1 ] && return 0
  [ "$TEGRA_MS" = 0 ] && return 0
  sudo pkill -x tegrastats 2>/dev/null || true
  sleep 1
  sudo chown "$(id -u):$(id -g)" "$1" 2>/dev/null || true
  [ -s "$1" ] || echo "WARNING: empty tegrastats log: $1" >&2
}

PROBE_PID=""
probe_start() {
  [ "$DRY" = 1 ] && return 0
  sudo "$PROBE" --out "$1" --interval-ms 1 > "${1%.csv}.log" 2>&1 &
  PROBE_PID=$!
  sleep 1
}
probe_stop() {
  [ "$DRY" = 1 ] && return 0
  [ -n "$PROBE_PID" ] || return 0
  sudo pkill -TERM -x probe 2>/dev/null || true
  wait "$PROBE_PID" 2>/dev/null || true
  PROBE_PID=""
  sudo chown "$(id -u):$(id -g)" "$1" 2>/dev/null || true
}

# Monitors started on purpose (19_observer.sh): MONITOR=jtop starts the jtop service during the
# run. tegrastats and the probe are controlled by TEGRA_MS and PROBE_RUN.
MON_ON=""
monitor_start() {
  [ "$DRY" = 1 ] && return 0
  case "${MONITOR:-none}" in
    none) ;;
    jtop)
      # service name kept from the first query; systemd may answer empty right after a stop,
      # so it retries before giving up
      local j=${JTOP_SVC:-} i
      for i in 1 2 3; do
        [ -n "$j" ] && break
        j=$(bg_jtop_svc); [ -n "$j" ] || sleep 2
      done
      if [ -z "$j" ]; then
        echo "WARNING: MONITOR=jtop, but the jtop service was not found; running without it" >&2
        return 1
      fi
      JTOP_SVC=$j
      fan_snapshot   # jtop reapplies the fan profile it stores (it edits the nvfancontrol .conf)
      if ! bg_active "$j"; then
        sudo systemctl start "$j" || { echo "WARNING: systemctl start $j failed; running without jtop" >&2; return 1; }
        MON_ON=$j; sleep 10
      fi ;;
    *) die "MONITOR=$MONITOR?" ;;
  esac
}
monitor_stop() {
  [ "$DRY" = 1 ] && return 0
  [ -n "$MON_ON" ] || return 0
  # start/stop only, never enable/disable: jtop must not start at boot again
  sudo systemctl stop "$MON_ON" 2>/dev/null || true
  MON_ON=""
  fan_restore
  # jtop may apply jetson_clocks when it starts: restore and check the saved state
  clk_restore_raw
  verify_state warn "after jtop"
}

# Fan (nvfancontrol): jtop changes the profile by editing /etc/nvfancontrol.conf (the file it
# points to) and restarting the service; the profile persists after jtop stops. Before starting
# jtop, the .conf and the profile are saved; afterwards the .conf is restored, nvfancontrol's saved
# state (/var/lib/nvfancontrol/status, which takes precedence over the .conf) is deleted and the
# service is restarted.
FAN_SNAP=""
fan_snapshot() {
  [ -e /etc/nvfancontrol.conf ] || return 0
  local c; c=$(readlink -f /etc/nvfancontrol.conf)
  sudo cp -p "$c" "$STATE_DIR/fanconf.bak" 2>/dev/null || return 0
  FAN_SNAP="$c $(fan_profile)"
}
fan_restore() {
  [ -n "$FAN_SNAP" ] || return 0
  local c want now; c=${FAN_SNAP%% *}; want=${FAN_SNAP##* }
  if ! sudo cmp -s "$STATE_DIR/fanconf.bak" "$c" || [ "$(fan_profile)" != "$want" ]; then
    echo ">> jtop changed the fan; restoring $c and profile $want" >&2
    sudo cp -p "$STATE_DIR/fanconf.bak" "$c"
    sudo systemctl stop nvfancontrol 2>/dev/null || true
    sudo rm -f /var/lib/nvfancontrol/status
    sudo systemctl start nvfancontrol 2>/dev/null || true
    sleep 3
    now=$(fan_profile)
    [ "$now" = "$want" ] || echo "WARNING: fan profile '$now' after restoring (expected $want)" >&2
  fi
  FAN_SNAP=""
}

# Environment conditions written to the .cond of each run
env_cond() {
  local dis
  dis=$(cat /sys/devices/system/cpu/cpu*/cpuidle/state*/disable 2>/dev/null | grep -c 1 || true)
  echo "plat=$PLATFORM tegra_ms=$TEGRA_MS monitor=${MONITOR:-none} bg=$(bg_now) idle_dis=$dis emc_halt=$(emc_halt) emc_mhz=$(emc_rate_mhz) gpu3d=$(gpu_3d) fan=$(fan_profile) pwm=$(fan_pwm) rpm=$(fan_rpm) t0max=$(temp_max_c) boot=$(boot_id | cut -c1-8)"
}

# Before each run: a locked EMC outside the jc cell is a residue. Returns "idle_dis/emc_halt"
# to compare with the state afterwards (did something change the state DURING the run?).
run_pre_check() {
  local h; h=$(emc_halt)
  if [ "$h" = 1 ] && [ "$CUR_CELL" != "jc/jc" ]; then
    echo "WARNING: EMC locked ($(basename "$EMC_HALT_F")=1) in cell $CUR_CELL, which is not the jetson_clocks cell" >&2
  fi
  echo "$(idle_dis_n)/$h/$(gpu_3d)/$(fan_profile)"
}
run_post_check() {  # run_post_check <pre> <.cond file>
  [ "$DRY" = 1 ] && return 0
  local post; post="$(idle_dis_n)/$(emc_halt)/$(gpu_3d)/$(fan_profile)"
  if [ "$post" != "$1" ]; then
    echo "WARNING: the state changed during the run (idle states disabled/EMC lock/GPU scaling/fan: before $1, after $post)" >&2
    echo "state_changed=1 post=$post" >> "$2"
  fi
}

CO_PID=""
corunner_start() {
  [ "$DRY" = 1 ] && return 0
  # 2 workers, each with 1/8 of the RAM (Orin 8 GB: 1 GB; Nano 4 GB: 512 MB)
  local kb; kb=$(awk '/MemTotal/{print int($2 / 8)}' /proc/meminfo)
  stress-ng --vm 2 --vm-bytes "${kb}K" --vm-method all -t 0 >/dev/null 2>&1 &
  CO_PID=$!
  sleep 5
}
corunner_stop() {
  [ "$DRY" = 1 ] && return 0
  [ -n "$CO_PID" ] || return 0
  kill -TERM "$CO_PID" 2>/dev/null || true
  sleep 2
  pkill -KILL -x stress-ng 2>/dev/null || true
  wait "$CO_PID" 2>/dev/null || true
  CO_PID=""
}

cleanup() {
  corunner_stop || true
  monitor_stop || true
  [ "$DRY" = 1 ] || { sudo pkill -x tegrastats 2>/dev/null || true; sudo pkill -TERM -x probe 2>/dev/null || true; }
  if [ "$CUR_CELL" = "jc/jc" ]; then jc_restore >/dev/null 2>&1 || true; fi
  clk_restore "end of $(basename "$0")" || true
  [ "$DRY" = 1 ] || echo ">> clocks restored to the saved factory state"
  bg_restore || true
  est_report
  [ "${EST_KEEP:-0}" = 1 ] || rm -f "$EST_FILE"
}
trap cleanup EXIT

# Estimated duration of a bench run from its arguments
est_bench() {
  local mode=periodic period=83.333 limit=0 passes=3 model=s
  while [ $# -gt 0 ]; do
    case "$1" in
      --mode) mode=$2; shift ;;
      --period-ms) period=$2; shift ;;
      --limit) limit=$2; shift ;;
      --passes) passes=$2; shift ;;
      --engine) model=$(basename "$2" | sed -E 's/yolox_([sml]).*/\1/'); shift ;;
    esac
    shift
  done
  [ "$limit" = 0 ] && limit=$N_ALL
  local per=$period; [ "$mode" = b2b ] && per=$(b2b_ms "$model")
  awk -v n="$limit" -v p="$passes" -v t="$per" 'BEGIN{printf "%.0f\n", n*p*t/1000 + 15}'
}

# run_bench <dir> <tag> "<k=v ...>" [bench args...]
#   Skips if the CSV already exists. Writes tag.csv, tag.meta.json, tag.log, tag.tegrastats,
#   tag.clocks (clock readout) and tag.cond (experimental condition, for the analysis).
#   PROBE_RUN=1: runs the 1 ms probe alongside (writes tag.probe.csv).
#   CORUN=on: runs stress-ng during the run.
run_bench() {
  local dir=$1 tag=$2 cond=$3; shift 3
  mkdir -p "$dir"
  if [ -f "$dir/$tag.csv" ]; then echo "=== $tag: already exists, skipping"; return 0; fi
  echo "=== $tag"
  if [ "$DRY" = 1 ]; then
    echo "   [dry] bench $* (${CORUN:-off}${PROBE_RUN:+, probe})"
    est_add "$(est_bench "$@")"; est_add "$COOL"
    return 0
  fi
  local a prev=""
  for a in "$@"; do [ "$prev" = --engine ] && need_engine "$a"; prev=$a; done
  echo "$cond corun=${CORUN:-off} probe=${PROBE_RUN:-0} cell=$CUR_CELL $(env_cond)" > "$dir/$tag.cond"
  local pre; pre=$(run_pre_check)
  clk_show > "$dir/$tag.clocks"
  [ "${CORUN:-off}" = on ] && corunner_start
  monitor_start || echo "monitor_fail=1" >> "$dir/$tag.cond"
  tegra_start "$dir/$tag.tegrastats"
  [ "${PROBE_RUN:-0}" = 1 ] && probe_start "$dir/$tag.probe.csv"
  "$BENCH" --frames "$FRAMES" --tag "$tag" --outdir "$dir" "$@" 2>&1 | tee "$dir/$tag.log" | grep -E "SUMMARY|ERROR|error" || true
  [ "${PROBE_RUN:-0}" = 1 ] && probe_stop "$dir/$tag.probe.csv"
  tegra_stop "$dir/$tag.tegrastats"
  run_post_check "$pre" "$dir/$tag.cond"
  monitor_stop
  [ "${CORUN:-off}" = on ] && corunner_stop
  [ -f "$dir/$tag.csv" ] || { echo "WARNING: $tag produced no CSV (see $dir/$tag.log)" >&2; return 0; }
  sleep "$COOL"
}

mr_of() { grep -o 'mr=[0-9.]*' "$1" 2>/dev/null | tail -1 | cut -d= -f2; }
mr_ok() { awk -v m="$1" -v t="${MR_MAX:-0.01}" 'BEGIN{exit !(m!="" && m<=t)}'; }
fps_ms() { awk -v f="$1" 'BEGIN{printf "%.3f", 1000/f}'; }

# find_rate <dir> <tagbase> "<cond>" <lo> <hi> [bench args, without mode/period/limit/passes]
#   Highest integer rate (fps) with deadline miss rate <= MR_MAX (1%), deadline = period.
#   Binary search with short runs (QUICK_LIMIT frames, 1 pass) and confirmation with full runs
#   (FULL_LIMIT x FULL_PASSES) at the two rates adjacent to the limit.
#   Result in <dir>/<tagbase>.rate:  ok=<fps> fail=<fps> mr_ok=... mr_fail=...
find_rate() {
  local dir=$1 base=$2 cond=$3 lo=$4 hi=$5; shift 5
  local QL=${QUICK_LIMIT:-400} FL=${FULL_LIMIT:-1000} FP=${FULL_PASSES:-2}
  mkdir -p "$dir/quick"
  if [ -f "$dir/$base.rate" ]; then echo "=== $base: $(cat "$dir/$base.rate")"; return 0; fi
  if [ "$DRY" = 1 ]; then
    echo "   [dry] find_rate $base in [$lo, $hi] fps"
    est_add $(( (5 * QL / 18) + (2 * FL * FP / 15) + 5 * 8 + 2 * COOL + 30 ))
    return 0
  fi
  quick() {  # quick <fps> [args] -> 0 if it passes
    local f=$1; shift
    local t="${base}_q$f"
    COOL=8 run_bench "$dir/quick" "$t" "$cond fps=$f quick=1" --mode periodic --period-ms "$(fps_ms "$f")" \
      --limit "$QL" --passes 1 "$@"
    mr_ok "$(mr_of "$dir/quick/$t.log")"
  }
  full() {  # full <fps> [args]
    local f=$1; shift
    local t="${base}_fps$f"
    run_bench "$dir" "$t" "$cond fps=$f" --mode periodic --period-ms "$(fps_ms "$f")" \
      --limit "$FL" --passes "$FP" "$@"
    mr_ok "$(mr_of "$dir/$t.log")"
  }
  local args=("$@")
  # if the lower bound already fails, halve it until it passes (floor of 1 fps)
  while ! quick "$lo" "${args[@]}"; do
    hi=$lo; lo=$((lo / 2))
    if [ "$lo" -lt 1 ]; then echo "ok=<1 fail=1" > "$dir/$base.rate"; cat "$dir/$base.rate"; return 0; fi
  done
  # if the upper bound still passes, double it until it fails (cap of RATE_CAP fps)
  while quick "$hi" "${args[@]}"; do
    lo=$hi; hi=$((hi * 2))
    if [ "$hi" -gt "${RATE_CAP:-240}" ]; then
      echo "ok=$lo fail=>$lo note=cap_${RATE_CAP:-240}fps" > "$dir/$base.rate"; cat "$dir/$base.rate"; return 0
    fi
  done
  while [ $((hi - lo)) -gt 1 ]; do
    local mid=$(((lo + hi) / 2))
    if quick "$mid" "${args[@]}"; then lo=$mid; else hi=$mid; fi
  done
  # confirmation with full runs: step down to a rate that passes, then up to one that
  # fails (at most 4 steps in each direction)
  local i note=""
  for i in 1 2 3 4; do
    full "$lo" "${args[@]}" && break
    hi=$lo; lo=$((lo - 1))
    [ "$i" = 4 ] && note=" note=ok_not_confirmed"
  done
  for i in 1 2 3 4; do
    full "$hi" "${args[@]}" || break
    lo=$hi; hi=$((hi + 1))
    [ "$i" = 4 ] && note="$note note=fail_not_confirmed"
  done
  echo "ok=$lo fail=$hi mr_ok=$(mr_of "$dir/${base}_fps$lo.log") mr_fail=$(mr_of "$dir/${base}_fps$hi.log")$note" > "$dir/$base.rate"
  echo ">> $base: $(cat "$dir/$base.rate")"
}

# trtexec with tegrastats (and the probe with PROBE_RUN=1). run_trtexec <dir> <tag> "<cond>" [args]
run_trtexec() {
  local dir=$1 tag=$2 cond=$3; shift 3
  mkdir -p "$dir"
  if [ -f "$dir/$tag.json" ]; then echo "=== $tag: already exists, skipping"; return 0; fi
  echo "=== $tag"
  if [ "$DRY" = 1 ]; then
    echo "   [dry] trtexec $*"
    est_add $(( ${TRT_DUR:-20} + 10 + COOL ))
    return 0
  fi
  local a
  for a in "$@"; do case "$a" in --loadEngine=*) need_engine "${a#--loadEngine=}" ;; esac; done
  echo "$cond probe=${PROBE_RUN:-0} cell=$CUR_CELL $(env_cond)" > "$dir/$tag.cond"
  local pre; pre=$(run_pre_check)
  clk_show > "$dir/$tag.clocks"
  tegra_start "$dir/$tag.tegrastats"
  [ "${PROBE_RUN:-0}" = 1 ] && probe_start "$dir/$tag.probe.csv"
  if ! "$TRTEXEC" "$@" --exportTimes="$dir/$tag.json" > "$dir/$tag.log" 2>&1; then
    probe_stop "$dir/$tag.probe.csv"; tegra_stop "$dir/$tag.tegrastats"
    grep -E "\[E\]" "$dir/$tag.log" | head -3 >&2
    die "trtexec failed in $tag (log: $dir/$tag.log)"
  fi
  probe_stop "$dir/$tag.probe.csv"
  tegra_stop "$dir/$tag.tegrastats"
  run_post_check "$pre" "$dir/$tag.cond"
  grep -E "GPU Compute Time" "$dir/$tag.log" | head -1 || true
  sleep "$COOL"
}

check_built() {
  [ "$DRY" = 1 ] && return 0
  [ -x "$BENCH" ] && [ -x "$PROBE" ] || die "binaries missing: run ./scripts/setup.sh --install"
  [ -s "$DATA/engines/yolox_s_fp16.engine" ] || die "engines missing in $DATA/engines: run ./scripts/setup.sh --install (see the README)"
  [ -n "$(ls "$FRAMES"/*.jpg 2>/dev/null | head -1)" ] || die "no frames in $FRAMES: run ./scripts/setup.sh --install"
}
need_engine() {  # stops if the engine does not exist (before spending board time)
  [ "$DRY" = 1 ] && return 0
  [ -s "$1" ] || die "engine not found: $1 (run ./scripts/setup.sh --install)"
}
