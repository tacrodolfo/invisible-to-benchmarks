#!/usr/bin/env bash
# Maximum power mode (Section "Power Mode, Pipelining, and Input Path").
#   B0 with jetson_clocks and factory; periodic pipeline locked and factory;
#   factory back-to-back; sustainable rate in both cells.
# The maximum mode is read from /etc/nvpmodel.conf: MAXN_SUPER if present (Orin Nano/NX Super),
# otherwise the first MAXN* (MAXN_MAX_MODE=... overrides). If the default mode is already the maximum
# (e.g. Jetson Nano), there is nothing to measure. Switches mode once, saves its factory state and
# returns to the default mode at the end.
#
# Usage: ./scripts/16_maxn.sh s|m|l      (or "s m l" for all three in the same mode switch)
source "$(dirname "$0")/lib.sh"
MODELS=${*:?model(s) s|m|l}
NAMES=$(sed -n 's/^< *POWER_MODEL ID=[0-9]* NAME=\([^ >]*\).*/\1/p' "${NVP_CONF:-/etc/nvpmodel.conf}" 2>/dev/null || true)
MAXM=${MAXN_MAX_MODE:-$(echo "$NAMES" | grep -x 'MAXN_SUPER' || echo "$NAMES" | grep -m1 '^MAXN' || true)}
if [ -z "$MAXM" ] || [ "$MAXM" = "$BASE_MODE" ]; then
  echo ">> 16_maxn: the default mode ($BASE_MODE) is already the maximum, or there is no MAXN* mode in nvpmodel.conf; nothing to do"
  trap - EXIT; exit 0
fi
check_built; assert_mode "$BASE_MODE"; clk_save

ID_MAX=$(mode_id "$MAXM" || true); ID_BASE=$(mode_id "$BASE_MODE" || true)
[ "$DRY" = 1 ] || [ -n "$ID_MAX" ] || die "$MAXM is not in /etc/nvpmodel.conf"
OUT="$RES/maxn"

back_to_base() {
  if [ "$DRY" != 1 ] && [ "$(power_mode)" != "$BASE_MODE" ]; then
    [ "$CUR_CELL" = "jc/jc" ] || clk_restore || true
    sudo nvpmodel -m "$ID_BASE"
    sleep 5
    echo ">> mode: $(power_mode)"
  fi
}
trap 'back_to_base; cleanup' EXIT

log "switching to $MAXM (id $ID_MAX)"
sx sudo nvpmodel -m "$ID_MAX"
[ "$DRY" = 1 ] || sleep 10
assert_mode "$MAXM"
CUR_CELL=""
clk_save "$MAXM"   # factory state of the maximum mode (its own file)

for M in $MODELS; do
  ENG=$(engine_of "$M"); MODEL=$(model_of "$M")
  B="exp=maxn model=$MODEL mode=$MAXM"
  set_cell default default
  run_trtexec "$OUT" "${MODEL}_MAXN_B0_cpu-default_gpu-default" "$B step=B0 cpu=default gpu=default" \
    --loadEngine="$ENG" --iterations=5000 --warmUp=500 --duration=0
  for CELL in "lock lock" "default default"; do
    set -- $CELL; CPU=$1; GPU=$2
    set_cell "$CPU" "$GPU"
    T="${MODEL}_MAXN_cpu-${CPU}_gpu-${GPU}"
    run_bench "$OUT" "${T}_periodic" "$B cpu=$CPU gpu=$GPU rel=periodic fps=$P_FPS" --engine "$ENG" --mode periodic --period-ms "$P_MS" --passes 3
    if [ "$CPU" = default ]; then
      run_bench "$OUT" "${T}_b2b" "$B cpu=$CPU gpu=$GPU rel=b2b" --engine "$ENG" --mode b2b --passes 3
    fi
    find_rate "$OUT" "$T" "$B cpu=$CPU gpu=$GPU" "$RATE_LO" "$RATE_HI" --engine "$ENG"
  done
done

# B0 with the vendor procedure (jetson_clocks) last: the stored --restore belongs to the
# default mode, so it is only used after returning to that mode (on script exit)
for M in $MODELS; do
  ENG=$(engine_of "$M"); MODEL=$(model_of "$M")
  set_cell jc jc
  run_trtexec "$OUT" "${MODEL}_MAXN_B0_cpu-jc_gpu-jc" "exp=maxn model=$MODEL mode=$MAXM step=B0 cpu=jc gpu=jc" \
    --loadEngine="$ENG" --iterations=5000 --warmUp=500 --duration=0
done
log "16_maxn done"
