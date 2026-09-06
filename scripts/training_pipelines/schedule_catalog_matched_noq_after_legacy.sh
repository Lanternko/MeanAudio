#!/usr/bin/env bash
# Chain after phase8_legacy_repro:
#   1) wait for DONE + strict audit PASSED
#   2) small-scale NoQ medium gate (100+100+64) — fix/stop on failure
#   3) only then launch full phase8_catalog_matched_noq (400k+200k)
#
# Guardrails (GPU idle backlog policy):
#   defined purpose, resumable, P1 priority, launch record.

set -euo pipefail

WORK_DIR=/home/kojiek/MeanAudio
STATE_DIR=/home/kojiek/logs/phase8_legacy_repro_guard
STATE_FILE="$STATE_DIR/state.json"
ALERT_FILE="$STATE_DIR/ALERT.json"
AUDIT_STATUS="$STATE_DIR/final_audit_loop_status.json"
FINAL_AUDIT="$STATE_DIR/FINAL_AUDIT.json"
GATE_SCRIPT="$WORK_DIR/scripts/training_pipelines/train_pipeline_phase8_catalog_matched_noq_medium_gate.sh"
GATE_SENTINEL="$STATE_DIR/noq_medium_gate_PASSED.json"
GATE_LOG=/home/kojiek/logs/phase8_catalog_matched_noq_medium_gate.log
FULL_SCRIPT="$WORK_DIR/scripts/training_pipelines/train_pipeline_phase8_catalog_matched_noq.sh"
CONTRACT_AUDIT="$WORK_DIR/scripts/audit_phase8_clean_noq_contract.py"
FOCUSED_MONITOR="$WORK_DIR/scripts/monitor_phase8_clean_noq.py"
FULL_LOG=/home/kojiek/logs/phase8_catalog_matched_noq_pipeline.log
SCHEDULE_LOG=/home/kojiek/logs/schedule_catalog_matched_noq_after_legacy.log
LAUNCH_RECORD="$STATE_DIR/next_experiment_catalog_matched_noq.json"
LOCK_FILE="$STATE_DIR/schedule_catalog_noq.lock"
GATE_TMUX=p8_catalog_noq_gate
FULL_TMUX=p8_catalog_noq
MIN_FREE_GB="${MIN_FREE_GB:-50}"
SKIP_GATE="${SKIP_GATE:-false}"  # emergency only

mkdir -p "$STATE_DIR" /home/kojiek/logs
exec 9>"$LOCK_FILE"
if ! flock -n 9; then
    echo "[STOP] schedule already running (lock $LOCK_FILE)" >&2
    exit 2
fi

exec > >(tee -a "$SCHEDULE_LOG") 2>&1
cd "$WORK_DIR"
source /home/kojiek/venvs/dac/bin/activate

log() { echo "[$(date -Is)] $*"; }

read_json_field() {
    python - "$1" "$2" <<'PY'
import json, sys
from pathlib import Path
path, key = Path(sys.argv[1]), sys.argv[2]
try:
    print(json.loads(path.read_text()).get(key, "") or "")
except Exception:
    print("")
PY
}

free_gb_root() {
    python - <<'PY'
import shutil
print(int(shutil.disk_usage("/").free // (1024**3)))
PY
}

write_launch_record() {
    local status="$1"
    local detail="$2"
    python - "$LAUNCH_RECORD" "$status" "$detail" <<'PY'
import json, os, sys, time
from datetime import datetime, timezone
from pathlib import Path
path, status, detail = Path(sys.argv[1]), sys.argv[2], sys.argv[3]
payload = {
    "status": status,
    "detail": detail,
    "experiment": "phase8_catalog_matched_noq",
    "priority": "P1",
    "flow": "legacy_audit_PASSED → medium_gate(100+100+64) → S1_NoQ_400k → S2_NoQ_200k → eval_no_q",
    "why_now": (
        "After phase8_legacy_repro (catalog-matched + historical Q) completes, "
        "run a small NoQ wiring gate first, then S1+S2 true-NoQ on the same cache "
        "to isolate Q contribution for paper Table 1."
    ),
    "gate_script": "scripts/training_pipelines/train_pipeline_phase8_catalog_matched_noq_medium_gate.sh",
    "pipeline": "scripts/training_pipelines/train_pipeline_phase8_catalog_matched_noq.sh",
    "tmux_gate": "p8_catalog_noq_gate",
    "tmux_full": "p8_catalog_noq",
    "checkpoint_points": "gate: every 50; full: weights 10k / ckpt 20k / ema 10k",
    "updated_at": datetime.now(timezone.utc).isoformat(),
    "epoch": time.time(),
}
tmp = path.with_suffix(path.suffix + ".tmp")
tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
os.replace(tmp, path)
PY
}

wait_gpu_idle() {
    local label="$1"
    local tries="${2:-40}"  # 40 * 30s = 20 min
    for i in $(seq 1 "$tries"); do
        if ! pgrep -f "exp_id=phase8_legacy_repro" >/dev/null 2>&1 \
           && ! pgrep -f "exp_id=phase8_catalog_matched_noq" >/dev/null 2>&1; then
            # allow residual eval to finish
            if pgrep -af "eval.py|phase4_eval" 2>/dev/null | grep -v grep | grep -q .; then
                log "waiting for eval to finish ($label try $i/$tries)"
                sleep 30
                continue
            fi
            return 0
        fi
        log "waiting for GPU train idle ($label try $i/$tries)"
        sleep 30
    done
    return 1
}

prune_legacy_intermediates() {
    for d in \
        "$WORK_DIR/exps/phase8_legacy_repro_stage1_400000/ema_ckpts" \
        "$WORK_DIR/exps/phase8_legacy_repro_stage2_200000/ema_ckpts"
    do
        if [ -d "$d" ]; then
            # EMA intermediates are *.pt; full ckpts are *.pth — delete both.
            find "$d" -type f \( -name '*.pth' -o -name '*.pt' \) -delete || true
            log "pruned $d"
        fi
    done
}

log "schedule started: legacy DONE+audit → NoQ medium gate → full NoQ"
write_launch_record WAITING "waiting for phase8_legacy_repro DONE + audit PASSED"

# --- 1) wait for legacy audit ---
while true; do
    if [ -s "$ALERT_FILE" ]; then
        write_launch_record BLOCKED "guard ALERT present: $ALERT_FILE"
        log "[FAIL] guarded watcher alert present — not launching"
        exit 2
    fi

    guard_phase=$(read_json_field "$STATE_FILE" phase)
    audit_phase=$(read_json_field "$AUDIT_STATUS" phase)

    if [ "$guard_phase" = "FAILED" ]; then
        write_launch_record BLOCKED "guard phase=FAILED"
        log "[FAIL] guarded pipeline FAILED"
        exit 2
    fi
    if [ "$audit_phase" = "FAILED" ]; then
        write_launch_record BLOCKED "final audit FAILED"
        log "[FAIL] strict audit FAILED — investigate (may be clap_needs_review)"
        exit 2
    fi

    if [ "$guard_phase" = "DONE" ] && [ "$audit_phase" = "PASSED" ]; then
        if [ -f "$FINAL_AUDIT" ]; then
            final_status=$(read_json_field "$FINAL_AUDIT" status)
            if [ "$final_status" != "passed" ]; then
                write_launch_record BLOCKED "FINAL_AUDIT status=$final_status"
                log "[FAIL] FINAL_AUDIT status=$final_status"
                exit 2
            fi
        fi
        log "legacy DONE + audit PASSED"
        break
    fi

    log "waiting… guard=$guard_phase audit=$audit_phase free_root=$(free_gb_root)G"
    sleep 60
done

# --- disk ---
free_gb=$(free_gb_root)
if [ "$free_gb" -lt "$MIN_FREE_GB" ]; then
    log "root free ${free_gb}G < ${MIN_FREE_GB}G — pruning legacy intermediate ema_ckpts"
    prune_legacy_intermediates
    free_gb=$(free_gb_root)
fi
if [ "$free_gb" -lt "$MIN_FREE_GB" ]; then
    write_launch_record BLOCKED "insufficient disk free=${free_gb}G need>=${MIN_FREE_GB}G"
    log "[FAIL] disk free ${free_gb}G"
    exit 2
fi

if ! wait_gpu_idle "post-legacy"; then
    write_launch_record BLOCKED "GPU still busy after legacy"
    log "[FAIL] GPU still busy"
    exit 2
fi

# --- 2) medium gate ---
gate_ok=false
if [ -f "$GATE_SENTINEL" ] && [ "$(read_json_field "$GATE_SENTINEL" status)" = "passed" ]; then
    log "reusing existing passed gate sentinel: $GATE_SENTINEL"
    gate_ok=true
elif [ "$SKIP_GATE" = "true" ]; then
    log "[WARN] SKIP_GATE=true — skipping medium gate (not recommended)"
    gate_ok=true
else
    write_launch_record GATE_RUNNING "medium gate 100+100+64 in tmux $GATE_TMUX"
    log "launching NoQ medium gate in tmux $GATE_TMUX"
    if tmux has-session -t "$GATE_TMUX" 2>/dev/null; then
        log "[WARN] killing leftover $GATE_TMUX"
        tmux kill-session -t "$GATE_TMUX" || true
    fi
    # run gate in foreground of this scheduler (still logged); use separate tmux
    # so chain_watch can see it, but wait here for result.
    tmux new-session -d -s "$GATE_TMUX" \
        "cd $WORK_DIR && source /home/kojiek/venvs/dac/bin/activate && export CUDA_VISIBLE_DEVICES=0 && bash $GATE_SCRIPT; echo GATE_EXIT=\$? | tee -a $GATE_LOG; sleep 2"

    # wait for gate sentinel or tmux death
    while true; do
        if [ -f "$GATE_SENTINEL" ]; then
            gstatus=$(read_json_field "$GATE_SENTINEL" status)
            if [ "$gstatus" = "passed" ]; then
                gate_ok=true
                log "medium gate PASSED"
                break
            elif [ "$gstatus" = "failed" ]; then
                write_launch_record GATE_FAILED "medium gate failed — see $GATE_LOG and $GATE_SENTINEL"
                log "[FAIL] medium gate failed — will NOT launch full train"
                cat "$GATE_SENTINEL" || true
                exit 2
            fi
        fi
        if ! tmux has-session -t "$GATE_TMUX" 2>/dev/null; then
            # session ended; re-check sentinel
            sleep 3
            if [ -f "$GATE_SENTINEL" ] && [ "$(read_json_field "$GATE_SENTINEL" status)" = "passed" ]; then
                gate_ok=true
                log "medium gate PASSED (session ended)"
                break
            fi
            write_launch_record GATE_FAILED "gate tmux exited without passed sentinel"
            log "[FAIL] gate tmux exited without PASS sentinel — check $GATE_LOG"
            tail -80 "$GATE_LOG" || true
            exit 2
        fi
        # progress heartbeat
        if [ -f "$GATE_LOG" ]; then
            tail -1 "$GATE_LOG" | sed 's/^/[gate] /' || true
        fi
        sleep 30
    done
fi

if [ "$gate_ok" != "true" ]; then
    write_launch_record GATE_FAILED "gate not ok"
    exit 2
fi

# Re-audit static q=None->q10 routing and full cache sentinels immediately
# before spending ~20 GPU-hours.  Runtime Hydra/eval checks begin after launch.
python "$CONTRACT_AUDIT" --phase preflight

# ensure gate processes fully released GPU
if ! wait_gpu_idle "post-gate" 20; then
    write_launch_record BLOCKED "GPU busy after gate"
    exit 2
fi

# --- 3) full train ---
if tmux has-session -t "$FULL_TMUX" 2>/dev/null; then
    write_launch_record BLOCKED "tmux $FULL_TMUX already exists"
    log "[FAIL] full tmux already exists"
    exit 2
fi

write_launch_record LAUNCHING "starting full train tmux $FULL_TMUX after gate PASS"
log "launching FULL catalog-matched NoQ in tmux $FULL_TMUX"
tmux new-session -d -s "$FULL_TMUX" \
    "cd $WORK_DIR && source /home/kojiek/venvs/dac/bin/activate && export CUDA_VISIBLE_DEVICES=0 && bash $FULL_SCRIPT 2>&1 | tee -a $FULL_LOG; echo EXIT=\$? | tee -a $FULL_LOG"

sleep 90
if ! tmux has-session -t "$FULL_TMUX" 2>/dev/null; then
    write_launch_record FAILED "full tmux died within 90s"
    log "[FAIL] full tmux died early — $FULL_LOG"
    tail -60 "$FULL_LOG" || true
    exit 2
fi

if ! python "$FOCUSED_MONITOR" --once; then
    write_launch_record FAILED "focused clean-NoQ monitor failed after launch"
    log "[FAIL] clean-NoQ runtime contract/health check failed after launch"
    exit 2
fi

write_launch_record RUNNING "full train launched after medium gate PASS; log=$FULL_LOG"
log "[DONE] chain advanced: gate PASS → full NoQ running in $FULL_TMUX"
log "monitor: tmux attach -t $FULL_TMUX | tail -f $FULL_LOG"
log "status:  cat $LAUNCH_RECORD"
