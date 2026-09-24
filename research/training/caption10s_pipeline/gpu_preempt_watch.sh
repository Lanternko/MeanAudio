#!/bin/bash
# If P1 owner is gone and GPU stays idle, start P2 filler.
# If P1 owner appears while P2 is training, request cooperative pause and wait for ckpt ack.
set -uo pipefail
IDLE_SEC="${IDLE_SEC:-180}"
PAUSE_REQ=/home/kojiek/logs/p2_pause.request
P2_SH=/home/kojiek/research/meanaudio_training/caption10s_pipeline/run_p2_s2q_filler.sh
P2_LOG=/home/kojiek/logs/c2p0_p2_s2q_filler.log
STATE=/home/kojiek/logs/p2_filler.state
NOTIFY=/home/kojiek/MeanAudio/scripts/notify_experiment_webhook.py
IDLE_ACC=0

ts() { date -u +%FT%TZ; }
p1_owner() {
  pgrep -f "/run_p1_fake_true_worst.sh" >/dev/null 2>&1
}
p2_owner() {
  pgrep -f "/run_p2_s2q_filler.sh" >/dev/null 2>&1
}
notify() {
  /home/kojiek/venvs/dac/bin/python "$NOTIFY" --status "$1" --experiment "$2" --summary "$3" --exit-code 0 \
    || echo "NOTIFY_FAIL $2 $(ts)"
}

echo "WATCH_START $(ts) idle_sec=$IDLE_SEC"
while true; do
  if p1_owner && p2_owner; then
    echo "PREEMPT_P2 $(ts)"
    touch "$PAUSE_REQ"
    for i in $(seq 1 90); do
      if [ -f "${PAUSE_REQ}.ack.json" ]; then
        echo "P2_ACK $(ts) $(cat "${PAUSE_REQ}.ack.json")"
        break
      fi
      if ! p2_owner; then
        echo "P2_EXIT_NO_ACK $(ts)"
        break
      fi
      sleep 2
    done
    if p2_owner; then
      echo "P2_PAUSE_TIMEOUT_TERM $(ts)"
      pkill -TERM -f "/run_p2_s2q_filler.sh" || true
      pkill -TERM -f "exp_id=phase8_qwen_caption2p0_s2q_from_noq_full" || true
    fi
    IDLE_ACC=0
    sleep 5
    continue
  fi

  if p1_owner; then
    IDLE_ACC=0
    sleep 10
    continue
  fi

  if p2_owner; then
    IDLE_ACC=0
    sleep 15
    continue
  fi

  if [ -f "$STATE" ] && grep -qx done "$STATE"; then
    sleep 60
    continue
  fi

  util=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d " ")
  mem=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d " ")
  if [ "${util:-100}" -lt 8 ] && [ "${mem:-99999}" -lt 2500 ]; then
    IDLE_ACC=$((IDLE_ACC + 15))
  else
    IDLE_ACC=0
  fi
  if [ "$IDLE_ACC" -ge "$IDLE_SEC" ]; then
    echo "START_P2_FILLER $(ts) idle=${IDLE_ACC}s"
    notify held c2p0-p2-filler-launch "GPU idle ${IDLE_ACC}s with no P1 owner; starting P2 S2Q filler (K=3 then K=5). P1 can preempt after checkpoint."
    rm -f "$PAUSE_REQ" "${PAUSE_REQ}.ack.json"
    setsid "$P2_SH" </dev/null >"$P2_LOG" 2>&1 &
    IDLE_ACC=0
    sleep 20
  fi
  sleep 15
done
