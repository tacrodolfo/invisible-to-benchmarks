# shellcheck shell=bash
# Readers of the platform state that does not show up in the governors or frequencies: CPU idle
# states, EMC lock (bwmgr_halt) and fan. Sourced by lib.sh and run_all.sh.
# platform_state prints one line with everything, for the campaign logs.
# EMC lock: on Orin (JetPack 6) the BPMP bwmgr_halt; on Nano (L4T 32) override.emc, which
# jetson_clocks turns on (clk_state=1) to fix the EMC at maximum. In both, 0 = EMC with DVFS.
# The name emc_halt (and the emc_halt= key in the .cond) covers both.
case "${PLATFORM:-}" in
  nano) EMC_HALT_F=/sys/kernel/debug/clk/override.emc/clk_state ;;
  *)    EMC_HALT_F=/sys/kernel/debug/bpmp/debug/bwmgr/bwmgr_halt ;;
esac
emc_halt() { [ "${DRY:-0}" = 1 ] && { echo n/a; return 0; }; sudo cat "$EMC_HALT_F" 2>/dev/null || echo n/a; }
emc_rate_mhz() {  # current EMC frequency in MHz
  [ "${DRY:-0}" = 1 ] && { echo n/a; return 0; }
  local f v
  for f in /sys/kernel/debug/bpmp/debug/clk/emc/rate /sys/kernel/debug/tegra_bwmgr/emc_rate /sys/kernel/debug/clk/emc/clk_rate; do
    v=$(sudo cat "$f" 2>/dev/null) && [ -n "$v" ] && { awk -v v="$v" 'BEGIN{printf "%.0f\n", v / 1e6}'; return 0; }
  done
  echo n/a
}
# GPU frequency scaling (Nano: /sys/devices/57000000.gpu/enable_3d_scaling; 1 = DVFS on, the
# default). On boards without the file: n/a.
GPU_3D_F=$(ls /sys/devices/*.gpu/enable_3d_scaling /sys/devices/platform/*gpu*/enable_3d_scaling 2>/dev/null | head -1 || true)
gpu_3d() { [ -n "$GPU_3D_F" ] && cat "$GPU_3D_F" 2>/dev/null || echo n/a; }
idle_dis_n() { cat /sys/devices/system/cpu/cpu*/cpuidle/state*/disable 2>/dev/null | grep -c 1 || true; }
idle_tot() { ls /sys/devices/system/cpu/cpu*/cpuidle/state*/disable 2>/dev/null | wc -l || true; }
fan_profile() {
  [ "${DRY:-0}" = 1 ] && { echo n/a; return 0; }
  local q p
  q=$(sudo nvfancontrol -q 2>/dev/null || true)
  p=$(echo "$q" | grep -o 'FAN_PROFILE:[A-Za-z0-9_]*' | head -1 | cut -d: -f2)
  echo "${p:-n/a}"
}
# Expected fan profile: FAN_DEFAULT_PROFILE of the /etc/nvfancontrol.conf in use (valid for any
# Jetson module with nvfancontrol); FAN_PROFILE_EXPECTED in the environment takes precedence.
fan_expected() {
  if [ -n "${FAN_PROFILE_EXPECTED:-}" ]; then echo "$FAN_PROFILE_EXPECTED"; return 0; fi
  [ -e /etc/nvfancontrol.conf ] || { echo ""; return 0; }
  awk '$1 == "FAN_DEFAULT_PROFILE" {print $2; exit}' "$(readlink -f /etc/nvfancontrol.conf)" 2>/dev/null || true
}
# Configuration files (fan, nvpmodel) changed relative to the L4T packages
conf_modified() {
  command -v dpkg >/dev/null || return 0
  dpkg -V nvidia-l4t-nvfancontrol nvidia-l4t-nvpmodel 2>/dev/null | awk 'substr($1, 3, 1) == "5" {print $NF}' || true
}

fan_pwm() {
  local f
  # Orin: hwmon pwm1 (nvfancontrol); Nano: pwm-fan/target_pwm (0 when idle; 255 with jetson_clocks)
  for f in /sys/class/hwmon/hwmon*/pwm1 /sys/devices/pwm-fan/target_pwm /sys/devices/pwm-fan/cur_pwm; do
    [ -r "$f" ] && { cat "$f"; return 0; }
  done
  echo n/a
}

temp_max_c() {  # highest temperature among the kernel thermal zones (C), for spot records
  local m
  m=$(cat /sys/devices/virtual/thermal/thermal_zone*/temp 2>/dev/null | awk '$1 > m {m = $1} END {if (m != "") printf "%.1f", m / 1000}')
  echo "${m:-n/a}"
}

fan_rpm() {  # measured speed (tachometer), when the board exposes it; otherwise n/a
  local f
  for f in /sys/class/hwmon/hwmon*/rpm /sys/class/hwmon/hwmon*/fan1_input /sys/devices/pwm-fan/rpm_measured; do
    [ -r "$f" ] && { cat "$f" 2>/dev/null; return 0; }
  done
  echo n/a
}

platform_state() {  # one line: mode, idle states, EMC, fan, boot
  local m
  m=$(sudo nvpmodel -q 2>/dev/null | grep -i "power mode" | head -1 | awk -F': ' '{print $2}' | tr -d ' ' || true)
  echo "mode=${m:-unknown} idle_dis=$(idle_dis_n)/$(idle_tot) emc_halt=$(emc_halt) emc_mhz=$(emc_rate_mhz) gpu3d=$(gpu_3d) fan=$(fan_profile) pwm=$(fan_pwm) rpm=$(fan_rpm) tmax=$(temp_max_c) boot=$(cat /proc/sys/kernel/random/boot_id 2>/dev/null | cut -c1-8)"
}
