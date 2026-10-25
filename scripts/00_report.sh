#!/usr/bin/env bash
# Platform report.
# Records the power mode and its factory default, governors and frequency tables,
# schedutil and nvhost_podgov parameters, CPU idle states, INA3221 rails,
# the EMC and what jetson_clocks has stored.
#
# RUN RIGHT AFTER A REBOOT, before any other script or jetson_clocks:
#   ./scripts/00_report.sh boot
# and again at any time to compare:
#   ./scripts/00_report.sh now
source "$(dirname "$0")/lib.sh"
trap - EXIT
set +e   # informational report: a missing command must not abort it
LABEL=${1:-now}
OUT="$RES/platform"; mkdir -p "$OUT"
F="$OUT/report_${LABEL}_$(date +%Y%m%d_%H%M).txt"

sec() { echo; echo "===== $*"; }
{
  sec "identification"
  date; uptime; uname -a
  echo "platform: $PLATFORM ($SOC), factory default mode: $BASE_MODE"
  echo "boot_id: $(boot_id)"
  cat /etc/nv_tegra_release 2>/dev/null || true
  dpkg -l 2>/dev/null | grep -E "nvidia-jetpack |nvidia-l4t-core|libnvinfer10 |cuda-cudart-12" | awk '{print $2, $3}' || true
  tr -d '\0' < /proc/device-tree/model 2>/dev/null; echo

  sec "power mode (nvpmodel)"
  sudo nvpmodel -q --verbose 2>&1 | head -40 || true
  echo "-- factory default in /etc/nvpmodel.conf (PM_CONFIG DEFAULT):"
  grep -E "PM_CONFIG DEFAULT" /etc/nvpmodel.conf || true
  readlink -f /etc/nvpmodel.conf || true
  grep -E "POWER_MODEL ID" /etc/nvpmodel.conf || true

  sec "CPU: cpufreq"
  for p in $POLICIES; do
    echo "$(basename "$p"): cpus=$(rd "$p/affected_cpus") gov=$(rd "$p/scaling_governor") min=$(rd "$p/scaling_min_freq") max=$(rd "$p/scaling_max_freq") cur=$(rd "$p/scaling_cur_freq") hw=[$(rd "$p/cpuinfo_min_freq"), $(rd "$p/cpuinfo_max_freq")]"
    echo "   available: $(rd "$p/scaling_available_frequencies")"
    echo "   governors: $(rd "$p/scaling_available_governors")"
    [ -d "$p/schedutil" ] && echo "   schedutil: rate_limit_us=$(rd "$p/schedutil/rate_limit_us")"
  done
  [ -f /sys/devices/system/cpu/cpufreq/schedutil/rate_limit_us ] && \
    echo "schedutil global rate_limit_us=$(rd /sys/devices/system/cpu/cpufreq/schedutil/rate_limit_us)"

  sec "CPU: idle states (cpuidle)"
  echo "driver=$(rd /sys/devices/system/cpu/cpuidle/current_driver) governor=$(rd /sys/devices/system/cpu/cpuidle/current_governor_ro)"
  for c in /sys/devices/system/cpu/cpu[0-9]*; do
    line="$(basename "$c"):"
    for s in "$c"/cpuidle/state*; do
      [ -d "$s" ] || continue
      line="$line $(rd "$s/name")(disable=$(rd "$s/disable"),lat=$(rd "$s/latency")us,usage=$(rd "$s/usage"))"
    done
    echo "$line"
  done
  echo "-- kernel cmdline (cpuidle.off / idle=?):"; cat /proc/cmdline
  echo "-- services that touch the clocks:"
  systemctl list-unit-files 2>/dev/null | grep -iE "nvpmodel|jetson_clocks|nvfan|clocks|jtop|jetson_stats" || true

  sec "background services (wake the CPU/GPU periodically)"
  bg_status
  echo "boot target: $(systemctl get-default 2>/dev/null)"
  echo "-- jtop: configuration (jetson_clocks at boot?)"
  for f in /etc/jtop/jtop.conf /usr/local/jtop/jtop.conf /usr/local/jtop/config.json /opt/jetson_stats/config.json; do
    sudo test -f "$f" && { echo "$f:"; sudo cat "$f"; echo; }
  done
  systemctl cat jtop jetson_stats 2>/dev/null | grep -E "^\[|ExecStart|Environment" || true
  pip3 show jetson-stats 2>/dev/null | grep -E "^Version" || true
  echo "-- processes of other users: $(bg_others)"

  sec "GPU: devfreq"
  echo "dev=$GPU_DEV"
  for k in governor available_governors available_frequencies min_freq max_freq cur_freq target_freq polling_interval; do
    echo "$k: $(rd "$GPU_DEV/$k")"
  done
  echo "load: $(rd "$GPU_DEV/device/load")"
  echo "railgate/power control: $(rd "$GPU_DEV/device/power/control") $(rd "$GPU_DEV/device/railgate_enable")"
  echo "-- nvhost_podgov parameters:"
  for d in "$GPU_DEV"/nvhost_podgov "$GPU_DEV"/device/nvhost_podgov /sys/kernel/debug/*podgov*; do
    [ -d "$d" ] || continue
    for f in "$d"/*; do [ -f "$f" ] && echo "   $(basename "$f") = $(rd "$f")"; done
  done
  sudo find /sys/kernel/debug -maxdepth 3 -iname "*podgov*" 2>/dev/null | head || true

  sec "invisible jetson_clocks residues (idle states, EMC lock)"
  echo "idle states disabled: $(idle_dis_n) of $(idle_tot)"
  echo "EMC lock ($EMC_HALT_F): $(emc_halt)   (1 = EMC fixed at maximum by jetson_clocks; default 0)"
  echo "EMC now: $(emc_rate_mhz) MHz"
  if [ "$PLATFORM" = nano ]; then
    echo "-- Nano (L4T 32): override.emc and tegra_bwmgr"
    for f in /sys/kernel/debug/clk/override.emc/clk_state /sys/kernel/debug/clk/override.emc/clk_rate \
             /sys/kernel/debug/tegra_bwmgr/emc_rate /sys/kernel/debug/tegra_bwmgr/emc_min_rate \
             /sys/kernel/debug/tegra_bwmgr/emc_max_rate; do
      echo "   $f = $(sudo cat "$f" 2>/dev/null || echo n/a)"
    done
  fi
  echo "GPU scaling (${GPU_3D_F:-no enable_3d_scaling}): $(gpu_3d)   (1 = GPU DVFS on; default)"
  echo "-- --store files of this repository:"; ls -l "$STATE_DIR"/jetsonclocks_*.txt 2>/dev/null || echo "none"
  [ -f "$STATE_DIR/restore_problems.log" ] && { echo "-- partial restores recorded:"; tail -20 "$STATE_DIR/restore_problems.log"; }

  sec "fan"
  echo "FAN_PROFILE: $(fan_profile)   (default in /etc/nvfancontrol.conf: $(fan_expected))"
  echo "PWM: $(fan_pwm)"
  for f in /sys/devices/pwm-fan/target_pwm /sys/devices/pwm-fan/cur_pwm /sys/devices/pwm-fan/temp_control; do
    [ -e "$f" ] && echo "   $f = $(cat "$f" 2>/dev/null)"   # Nano: automatic = temp_control 1, 0 when idle
  done
  sudo nvfancontrol -q 2>&1 | head -10 || true
  for f in /etc/nvfancontrol.conf; do
    [ -e "$f" ] && echo "$f -> $(readlink -f "$f") md5 $(md5sum < "$(readlink -f "$f")" | cut -c1-32)"
  done
  echo "-- configuration files changed relative to the packages (dpkg -V; empty = untouched):"
  dpkg -V nvidia-l4t-nvfancontrol nvidia-l4t-nvpmodel nvidia-l4t-tools 2>&1 || true

  sec "memory: EMC"
  echo "emc rate: $(sudo cat /sys/kernel/debug/bpmp/debug/clk/emc/rate 2>/dev/null || sudo cat /sys/kernel/debug/clk/emc/clk_rate 2>/dev/null || echo n/a)"
  echo "emc min/max: $(sudo cat /sys/kernel/debug/bpmp/debug/clk/emc/min_rate 2>/dev/null) $(sudo cat /sys/kernel/debug/bpmp/debug/clk/emc/max_rate 2>/dev/null)"
  ls /sys/class/devfreq/ 2>/dev/null

  sec "power: INA3221 rails"
  for hw in /sys/class/hwmon/hwmon*; do
    echo "$(basename "$hw") name=$(rd "$hw/name")"
    for l in "$hw"/in*_label; do [ -f "$l" ] && echo "   $(basename "$l") = $(rd "$l")"; done
  done
  for d in /sys/bus/i2c/drivers/ina3221x/*/iio:device*; do   # JetPack 4
    [ -d "$d" ] || continue
    echo "$d: $(for i in 0 1 2; do printf "ch%s=%s mW " "$i" "$(sudo cat "$d/in_power${i}_input" 2>/dev/null)"; done)"
  done
  echo "-- one tegrastats line:"
  sudo timeout 3 tegrastats --interval 1000 2>/dev/null | head -1 || true

  sec "jetson_clocks"
  sudo jetson_clocks --show 2>&1 | head -40 || true
  echo "-- --store file (/root/.jetsonclocks_conf.txt); date and idle states it restores:"
  sudo stat -c "%y" /root/.jetsonclocks_conf.txt 2>/dev/null || true
  sudo test -f /root/.jetsonclocks_conf.txt && sudo ls -l /root/.jetsonclocks_conf.txt && sudo cat /root/.jetsonclocks_conf.txt | grep -iE "idle|gpu|emc|governor" | head -40 || echo "does not exist"

  sec "temperatures"
  for z in /sys/devices/virtual/thermal/thermal_zone*; do echo "$(rd "$z/type"): $(rd "$z/temp")"; done

  sec "processes that may interfere (top 10 by memory)"
  ps -eo pid,comm,%cpu,rss --sort=-rss | head -11
} > "$F" 2>&1

echo ">> report in $F"
grep -aE "PM_CONFIG DEFAULT|^policy|\(disable=1,|^governor|^available_frequencies|rate_limit|VDD_IN|^platform|^jtop:|^desktop environment|^EMC lock|^EMC now|^GPU scaling|^idle states disab|^FAN_PROFILE|^PWM" "$F" | head -30 || true
if ! ref_check "$BASE_MODE" > /tmp/edgebench_ref_report_$$; then
  echo
  echo "WARNING: the current state is not a clean reference state:"
  cat /tmp/edgebench_ref_report_$$
fi
rm -f /tmp/edgebench_ref_report_$$
FP=$(fan_profile); FE=$(fan_expected)
[ -z "$FE" ] || [ "$FP" = n/a ] || [ "$FP" = "$FE" ] || echo "WARNING: fan profile '$FP' (default in /etc/nvfancontrol.conf: $FE)"
CM=$(conf_modified)
[ -z "$CM" ] || echo "WARNING: configuration files changed relative to the L4T packages: $(echo "$CM" | tr '\n' ' ')"
DIS=$(grep -ao "(disable=1," "$F" | wc -l || true)
if [ "$DIS" -gt 0 ]; then
  echo
  echo "WARNING: $DIS CPU idle states are disabled. In the factory default (reboot without jtop"
  echo "or jetson_clocks) all of them are enabled. This is usually the residue of a jetson_clocks"
  echo "undone by another path (nvpmodel -m, sysfs) or of a --restore with an old --store."
fi
