#!/usr/bin/env bash
# Background services that wake the CPU/GPU periodically and interfere with DVFS:
#   jtop (jetson-stats: service "jtop" or, in older versions, "jetson_stats")
#   desktop environment (display-manager: gdm3, lightdm...; gnome-shell uses the GPU)
#   optional: processes listed in QUIET_PROCS (empty by default), frozen with SIGSTOP and
#   resumed with SIGCONT, without losing their state
# Also warns about running containers and processes of other users.
#
# Sourced by lib.sh (functions) or run directly:
#   ./scripts/bg.sh status     shows what is active
#   ./scripts/bg.sh stop       stops jtop and the desktop environment (remembers what it stopped)
#   ./scripts/bg.sh restore    restarts what was stopped
# With QUIET=1 (default), each campaign script stops them at the start and restarts them at the end;
# run_all.sh does this only once for the whole campaign (EDGEBENCH_BG_OWNER).

_BG_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/state
_BG_FILE=$_BG_DIR/bg_stopped
if [ -z "${PLATFORM:-}" ]; then
  # shellcheck source=platform.sh
  source "$(dirname "${BASH_SOURCE[0]}")/platform.sh"
fi

# PIDs of the QUIET_PROCS processes (and descendants). Matches only the program itself: argv[0] with
# that name, or an interpreter (python*, bash, sh) whose argv[1] is that program. Never a process
# that merely mentions the name in its arguments (a grep, a shell with the name in the command),
# nor this script and its ancestors.
bg_procs_pids() {
  [ -n "${QUIET_PROCS:-}" ] || return 0
  local me=$$ anc="" p=$$ pid kid
  while [ -n "$p" ] && [ "$p" -gt 1 ] 2>/dev/null; do anc="$anc $p"; p=$(ps -o ppid= -p "$p" 2>/dev/null | tr -d ' '); done
  for pid in $(ps -eo pid=,args= 2>/dev/null | awk -v names=" ${QUIET_PROCS} " '
      function base(x) { sub(/.*\//, "", x); return x }
      { a0 = base($2); a1 = (NF >= 3) ? base($3) : "" }
      index(names, " " a0 " ") || (a0 ~ /^(python[0-9.]*|bash|sh)$/ && a1 != "" && index(names, " " a1 " ")) { print $1 }'); do
    case " $anc $me " in *" $pid "*) continue ;; esac
    echo "$pid"
    for kid in $(pgrep -P "$pid" 2>/dev/null); do echo "$kid"; done
  done | sort -un
}

bg_jtop_svc() {  # name of the installed jtop service (empty if none), enabled or not
  local s ls
  for s in jtop jetson_stats; do
    ls=$(systemctl show -p LoadState --value "$s.service" 2>/dev/null || true)
    if [ "$ls" = loaded ] || [ -e "/etc/systemd/system/$s.service" ] || [ -e "/lib/systemd/system/$s.service" ]; then
      echo "$s"; return 0
    fi
  done
}
bg_active() { systemctl is-active --quiet "$1" 2>/dev/null; }

bg_docker_n() {  # running containers (empty if it cannot be determined)
  local out
  out=$(sudo -n docker ps -q 2>/dev/null) || { echo ""; return 0; }
  echo "$out" | grep -c . || true
}

bg_now() {  # short list of what is active, for the .cond: "jtop,dm,docker2" or "none"
  local out=() j n
  j=$(bg_jtop_svc)
  [ -n "$j" ] && bg_active "$j" && out+=(jtop)
  bg_active display-manager && out+=(dm)
  local p st
  for p in $(bg_procs_pids); do
    st=$(ps -o stat= -p "$p" 2>/dev/null | cut -c1)
    [ "$st" = T ] || { out+=("proc:$(ps -o comm= -p "$p" 2>/dev/null)"); break; }
  done
  n=$(bg_docker_n)
  [ "${n:-0}" -gt 0 ] 2>/dev/null && out+=("docker$n")
  if [ ${#out[@]} -eq 0 ]; then echo none; else (IFS=,; echo "${out[*]}"); fi
}

bg_others() {  # processes of other human users (uid >= 1000 with a login shell), per user
  local nologin
  nologin=$(getent passwd 2>/dev/null | awk -F: '$7 ~ /(nologin|false)$/ {print $1}' | tr '\n' ' ')
  ps -eo uid=,user= 2>/dev/null | awk -v me="$(id -u)" -v sys=" $nologin " '$1 >= 1000 && $1 != 65534 && $1 != me && index(sys, " " $2 " ") == 0 {c[$2]++}
    END { for (u in c) printf "%s:%d ", u, c[u] }'
}

bg_status() {
  local j; j=$(bg_jtop_svc)
  if [ -n "$j" ]; then
    local en ac
    en=$(systemctl is-enabled "$j" 2>/dev/null || true); ac=$(systemctl is-active "$j" 2>/dev/null || true)
    echo "jtop: installed ($j.service), at boot: ${en:-?}, now: ${ac:-?}"
  else
    echo "jtop: not installed"
  fi
  local dm n
  dm=$(systemctl is-active display-manager 2>/dev/null || true)
  n=$(bg_docker_n)
  echo "desktop environment: ${dm:-n/a}"
  echo "containers: ${n:-n/a}"
  local pp; pp=$(bg_procs_pids | tr '\n' ' ')
  [ -z "${QUIET_PROCS:-}" ] || echo "processes to freeze (${QUIET_PROCS}): ${pp:-none running}"
  echo "other users with processes: $(bg_others)"
  [ -f "$_BG_FILE" ] && echo "stopped by us (to restart): $(tr '\n' ' ' < "$_BG_FILE")"
  return 0
}

bg_stop() {
  mkdir -p "$_BG_DIR"
  local j; j=$(bg_jtop_svc)
  if [ -n "$j" ] && bg_active "$j"; then
    sudo systemctl stop "$j" && echo "$j" >> "$_BG_FILE" && echo ">> jtop ($j) stopped during the campaign"
  fi
  if bg_active display-manager; then
    sudo systemctl stop display-manager && echo display-manager >> "$_BG_FILE" && \
      echo ">> desktop environment (display-manager) stopped during the campaign"
  fi
  local p n=0
  for p in $(bg_procs_pids); do
    if sudo kill -STOP "$p" 2>/dev/null; then echo "pid:$p" >> "$_BG_FILE"; n=$((n + 1)); fi
  done
  [ "$n" = 0 ] || echo ">> $n process(es) of ${QUIET_PROCS} frozen (SIGSTOP) during the campaign"
  return 0
}

bg_restore_all() {
  [ -f "$_BG_FILE" ] || return 0
  local s
  # first the frozen processes, then the services
  for s in $(grep '^pid:' "$_BG_FILE" | sort -u); do
    sudo kill -CONT "${s#pid:}" 2>/dev/null && echo ">> process ${s#pid:} resumed (SIGCONT)"
  done
  for s in $(grep -v '^pid:' "$_BG_FILE" | sort -u); do
    # start only: whatever was disabled at boot stays disabled (never enable)
    sudo systemctl start "$s" && echo ">> $s restarted"
  done
  rm -f "$_BG_FILE"
}

# Called by clk_save in each script. With QUIET=1 it stops whatever interferes (once per
# campaign); with QUIET=0 it only warns. It always warns about containers and other users.
bg_prepare() {
  [ "${DRY:-0}" = 1 ] && return 0
  [ -n "${_BG_PREPARED:-}" ] && return 0
  _BG_PREPARED=1
  # anything left stopped by an interrupted campaign? restart it before deciding
  if [ -f "$_BG_FILE" ] && [ -z "${EDGEBENCH_BG_OWNER:-}" ]; then bg_restore_all; fi
  if [ "${QUIET:-1}" = 1 ]; then
    if [ -z "${EDGEBENCH_BG_OWNER:-}" ]; then
      export EDGEBENCH_BG_OWNER=$$
      bg_stop
      sleep 5
    fi
  else
    local now; now=$(bg_now)
    [ "$now" = none ] || echo "WARNING: QUIET=0, active during the measurement: $now (recorded in the .cond)" >&2
  fi
  local o; o=$(bg_others)
  [ -z "$o" ] || echo "WARNING: other users with processes: $o (check that nothing heavy is running)" >&2
  local n; n=$(bg_docker_n)
  [ "${n:-0}" = 0 ] || echo "WARNING: $n container(s) running" >&2
  return 0
}
bg_restore() {  # only the one who stopped them restarts them
  [ "${DRY:-0}" = 1 ] && return 0
  [ "${EDGEBENCH_BG_OWNER:-}" = "$$" ] || return 0
  bg_restore_all
}

if [ "${BASH_SOURCE[0]}" = "$0" ]; then
  case "${1:-status}" in
    status) bg_status ;;
    stop) bg_stop ;;
    restore) bg_restore_all ;;
    *) echo "usage: $0 status|stop|restore"; exit 1 ;;
  esac
fi
