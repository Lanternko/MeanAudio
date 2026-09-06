#!/usr/bin/env bash
# Outer supervisor: keep Phase-8 legacy repro alive until DONE + strict audit.
# Recoverable stops (isolated AMP overflow / process death with intact gates +
# resumable ckpt) auto-resume. Hard gate failures and thrashing stop closed.

set -euo pipefail

WORK_DIR=/home/kojiek/MeanAudio
STATE_DIR=/home/kojiek/logs/phase8_legacy_repro_guard
STATE_FILE="$STATE_DIR/state.json"
ALERT_FILE="$STATE_DIR/ALERT.json"
SUP_LOG="$STATE_DIR/supervisor.log"
SUP_LOCK="$STATE_DIR/supervisor.lock"
SUP_STATUS="$STATE_DIR/supervisor_status.json"
GATE=/mnt/HDD/kojiek/phase8_legacy_matched_npz/FULL_GATE_PASSED.json
VALIDATION=/mnt/HDD/kojiek/phase8_legacy_matched_npz/FULL_VALIDATION.json
S1_CKPT="$WORK_DIR/exps/phase8_legacy_repro_stage1_400000/phase8_legacy_repro_stage1_400000_ckpt_last.pth"
S2_CKPT="$WORK_DIR/exps/phase8_legacy_repro_stage2_200000/phase8_legacy_repro_stage2_200000_ckpt_last.pth"
S2_EMA="$WORK_DIR/exps/phase8_legacy_repro_stage2_200000/phase8_legacy_repro_stage2_200000_ema_final.pth"
FINAL_METRICS="$WORK_DIR/eval_output/metrics/phase8_legacy_repro_stage2_200000_musiccaps/metrics.txt"
MAX_AUTO_RESUMES=8
SLEEP_SEC=90

mkdir -p "$STATE_DIR"
exec 8>"$SUP_LOCK"
if ! flock -n 8; then
  echo "[STOP] supervisor already running" >&2
  exit 2
fi

exec >>"$SUP_LOG" 2>&1
cd "$WORK_DIR"
source /home/kojiek/venvs/dac/bin/activate

write_status() {
  python - "$SUP_STATUS" "$1" "$2" <<'PY'
import json, os, sys, time
from datetime import datetime, timezone
from pathlib import Path
path, phase, detail = Path(sys.argv[1]), sys.argv[2], sys.argv[3]
payload = {
  "phase": phase,
  "detail": detail,
  "updated_at": datetime.now(timezone.utc).isoformat(),
  "epoch": time.time(),
}
tmp = path.with_suffix(path.suffix + ".tmp")
tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
os.replace(tmp, path)
PY
}

read_phase() {
  python - "$STATE_FILE" <<'PY'
import json, sys
from pathlib import Path
try:
  print(json.loads(Path(sys.argv[1]).read_text()).get("phase", "UNKNOWN"))
except Exception:
  print("UNKNOWN")
PY
}

gates_ok() {
  python - "$VALIDATION" "$GATE" <<'PY'
import hashlib, json, sys
from pathlib import Path
vpath, gpath = map(Path, sys.argv[1:])
v = json.loads(vpath.read_text()); g = json.loads(gpath.read_text())
assert v.get("status") == "passed" and v.get("expected_rows") == 251599
assert g.get("status") == "passed" and g.get("decoded_samples") == 512
assert g.get("decoded_cache_clap", 0) >= g.get("minimum_clap", 1)
assert g.get("validation_report_sha256") == hashlib.sha256(vpath.read_bytes()).hexdigest()
print("ok")
PY
}

guard_running() {
  pgrep -f "scripts/training_pipelines/run_phase8_legacy_guarded.sh" >/dev/null 2>&1
}

train_running() {
  pgrep -f "exp_id=phase8_legacy_repro" >/dev/null 2>&1
}

audit_running() {
  pgrep -f "wait_and_audit_phase8_legacy_repro.sh" >/dev/null 2>&1
}

start_guard() {
  local resume="$1"
  if guard_running; then
    echo "[supervisor] guard already running"
    return 0
  fi
  # archive any alert into incidents before resume
  if [ -f "$ALERT_FILE" ]; then
    local stamp
    stamp=$(date +%Y%m%d_%H%M%S)
    mkdir -p "$STATE_DIR/incidents/${stamp}_auto_resume"
    mv -f "$ALERT_FILE" "$STATE_DIR/incidents/${stamp}_auto_resume/" 2>/dev/null || true
  fi
  if tmux has-session -t phase8_legacy_guarded 2>/dev/null; then
    tmux kill-session -t phase8_legacy_guarded 2>/dev/null || true
    sleep 2
  fi
  tmux new-session -d -s phase8_legacy_guarded \
    "cd $WORK_DIR && RESUME_EXISTING=$resume bash scripts/training_pipelines/run_phase8_legacy_guarded.sh"
  echo "[supervisor] launched guard RESUME_EXISTING=$resume"
}

start_audit_loop() {
  if audit_running; then
    return 0
  fi
  # wait_and_audit has its own lock; start detached
  nohup bash scripts/training_pipelines/wait_and_audit_phase8_legacy_repro.sh \
    >/dev/null 2>&1 &
  echo "[supervisor] started wait_and_audit pid=$!"
}

resume_count=0
echo "[$(date -Is)] supervisor started"

while true; do
  if [ -f "$FINAL_METRICS" ] && [ -f "$S2_EMA" ]; then
    phase=$(read_phase)
    if [ "$phase" = "DONE" ]; then
      write_status DONE "final metrics present; ensuring audit"
      start_audit_loop
      # wait for audit to finish if running
      for _ in $(seq 1 120); do
        if ! audit_running; then break; fi
        sleep 30
      done
      if python "$WORK_DIR/scripts/audit_phase8_legacy_repro.py"; then
        write_status COMPLETE "strict audit passed"
        echo "[$(date -Is)] COMPLETE"
        exit 0
      else
        write_status AUDIT_HOLD "metrics exist but audit did not pass"
        echo "[$(date -Is)] audit hold — not auto-retraining"
        exit 3
      fi
    fi
  fi

  if ! gates_ok; then
    write_status HARD_STOP "gates invalid; refuse to train"
    echo "[FAIL] gates not ok" >&2
    exit 2
  fi

  phase=$(read_phase)
  if guard_running || train_running; then
    write_status MONITORING "phase=$phase guard=$(guard_running && echo y || echo n) train=$(train_running && echo y || echo n)"
    start_audit_loop
    sleep "$SLEEP_SEC"
    continue
  fi

  # nothing training
  if [ "$phase" = "DONE" ] && [ -f "$FINAL_METRICS" ]; then
    start_audit_loop
    sleep "$SLEEP_SEC"
    continue
  fi

  # recoverable resume
  if [ $resume_count -ge $MAX_AUTO_RESUMES ]; then
    write_status HARD_STOP "exceeded MAX_AUTO_RESUMES=$MAX_AUTO_RESUMES"
    echo "[FAIL] resume thrash" >&2
    exit 2
  fi

  if [ -f "$S1_CKPT" ] || [ -f "$S2_CKPT" ]; then
    resume_count=$((resume_count + 1))
    write_status RESUME "auto-resume #$resume_count after stop phase=$phase"
    start_guard true
  else
    write_status START "fresh guarded launch (no ckpt)"
    start_guard false
  fi
  start_audit_loop
  sleep "$SLEEP_SEC"
done
