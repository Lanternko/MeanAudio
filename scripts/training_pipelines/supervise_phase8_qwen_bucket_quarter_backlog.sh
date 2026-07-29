#!/usr/bin/env bash
# Outer, non-destructive supervisor for the Phase-8 Qwen quarter backlog.
#
# This process never deletes artifacts, changes experiment parameters, or kills
# training.  It keeps the sequence, read-only watcher, and durable repair
# controller alive.  Every sequence launch uses EXPERIMENT_RUN_MODE=resume so
# the sequence remains the sole authority for validating/reusing historical
# arms and skipping completed work.

set -Eeuo pipefail

ROOT=/home/kojiek/MeanAudio
LOG_ROOT=/home/kojiek/logs
STATE_DIR="$LOG_ROOT/phase8_qwen_bucket_quarter_backlog"
WATCHER_STATE_DIR="$LOG_ROOT/phase8_qwen_bucket_quarter_backlog_monitor"
SEQUENCE="$ROOT/scripts/training_pipelines/sequence_phase8_qwen_bucket_quarter_backlog.sh"
WATCHER="$ROOT/scripts/monitor_phase8_qwen_bucket_quarter_backlog.py"
REPAIR_CONTROLLER="$ROOT/scripts/repair_phase8_qwen_bucket_incident_with_agents.sh"
RESUME_MARKER_VALIDATOR="$ROOT/scripts/validate_phase8_qwen_resume_marker.py"
BACKLOG_CONTRACT="$ROOT/docs/experiments/phase8_qwen_bucket_quarter_backlog_2026_07_26.md"
# These are the sequence/watcher canonical contracts.  Do not infer completion
# from individual arm artifacts: the sequence owns all seven-arm validation.
FINAL_REPORT="$LOG_ROOT/phase8_qwen_bucket_quarter_backlog_FINAL_METRICS.json"
HARD_ALERT="$WATCHER_STATE_DIR/ALERT.json"

SUPERVISOR_STATUS="$STATE_DIR/supervisor_status.json"
SUPERVISOR_ALERT="$STATE_DIR/SUPERVISOR_ALERT.json"
SUPERVISOR_LOCK="$STATE_DIR/supervisor.lock"
SUPERVISOR_LOG="$STATE_DIR/supervisor.log"
SEQUENCE_LOG="$STATE_DIR/sequence.supervised.log"
WATCHER_LOG="$STATE_DIR/watcher.supervised.log"
CONTROLLER_LOG="$STATE_DIR/repair-controller.supervised.log"
SEQUENCE_PID_FILE="$STATE_DIR/sequence.pid"
WATCHER_PID_FILE="$STATE_DIR/watcher.pid"
CONTROLLER_PID_FILE="$STATE_DIR/repair-controller.pid"
CONTROLLER_STATE_DIR="$LOG_ROOT/phase8_qwen_bucket_quarter_backlog_repair"
CONTROLLER_PROCESS_NEEDLE="$CONTROLLER_STATE_DIR"
RESUME_MARKER="$CONTROLLER_STATE_DIR/RESUME_AUTHORIZED.json"
COMPAT_ENV="$ROOT/scripts/runtime/phase8_nvidia_compat_env.sh"
NVIDIA_SMI_BIN="${NVIDIA_SMI_BIN:-nvidia-smi}"

if [[ ! -r "$COMPAT_ENV" ]]; then
    echo "[FAIL] missing NVIDIA compatibility environment: $COMPAT_ENV" >&2
    exit 2
fi
# shellcheck source=/dev/null
source "$COMPAT_ENV"

PREFLIGHT_ONLY="${PREFLIGHT_ONLY:-false}"
DRY_RUN="${DRY_RUN:-false}"
ENABLE_REPAIR_CONTROLLER="${ENABLE_REPAIR_CONTROLLER:-false}"
POLL_SECONDS="${SUPERVISOR_POLL_SECONDS:-30}"
INITIAL_BACKOFF_SECONDS="${SUPERVISOR_INITIAL_BACKOFF_SECONDS:-30}"
MAX_BACKOFF_SECONDS="${SUPERVISOR_MAX_BACKOFF_SECONDS:-600}"
STABLE_SECONDS="${SUPERVISOR_STABLE_SECONDS:-1800}"
MAX_SEQUENCE_RESTARTS="${SUPERVISOR_MAX_SEQUENCE_RESTARTS:-6}"
MAX_WATCHER_RESTARTS="${SUPERVISOR_MAX_WATCHER_RESTARTS:-12}"
MAX_CONTROLLER_RESTARTS="${SUPERVISOR_MAX_CONTROLLER_RESTARTS:-12}"

normalize_bool() {
    case "${1,,}" in
        1|true|yes) printf 'true\n' ;;
        0|false|no) printf 'false\n' ;;
        *) return 1 ;;
    esac
}

if ! PREFLIGHT_ONLY=$(normalize_bool "$PREFLIGHT_ONLY"); then
    echo "[FAIL] PREFLIGHT_ONLY must be true/false (or 1/0)" >&2
    exit 2
fi
if ! DRY_RUN=$(normalize_bool "$DRY_RUN"); then
    echo "[FAIL] DRY_RUN must be true/false (or 1/0)" >&2
    exit 2
fi
if ! ENABLE_REPAIR_CONTROLLER=$(normalize_bool "$ENABLE_REPAIR_CONTROLLER"); then
    echo "[FAIL] ENABLE_REPAIR_CONTROLLER must be true/false (or 1/0)" >&2
    exit 2
fi
for value_name in POLL_SECONDS INITIAL_BACKOFF_SECONDS MAX_BACKOFF_SECONDS \
    STABLE_SECONDS MAX_SEQUENCE_RESTARTS MAX_WATCHER_RESTARTS \
    MAX_CONTROLLER_RESTARTS; do
    value="${!value_name}"
    if [[ ! "$value" =~ ^[0-9]+$ ]]; then
        echo "[FAIL] $value_name must be a non-negative integer" >&2
        exit 2
    fi
done
if (( POLL_SECONDS < 1 || INITIAL_BACKOFF_SECONDS < 1 ||
      MAX_BACKOFF_SECONDS < INITIAL_BACKOFF_SECONDS ||
      MAX_SEQUENCE_RESTARTS < 1 || MAX_WATCHER_RESTARTS < 1 ||
      MAX_CONTROLLER_RESTARTS < 1 )); then
    echo "[FAIL] invalid supervisor timing or attempt-cap configuration" >&2
    exit 2
fi

mkdir -p "$STATE_DIR"
exec 9>"$SUPERVISOR_LOCK"
if ! flock -n 9; then
    echo "[STOP] phase8 Qwen quarter backlog supervisor already running" >&2
    exit 3
fi

log() {
    local line
    line="[SUPERVISOR] $(date --iso-8601=seconds) $*"
    printf '%s\n' "$line" | tee -a "$SUPERVISOR_LOG"
}

GPU_OK=false
GPU_DETAIL="not_checked"
GPU_COMPAT_READY=false
SEQUENCE_PID=""
WATCHER_PID=""
CONTROLLER_PID=""
SEQUENCE_RESTARTS=0
WATCHER_RESTARTS=0
CONTROLLER_RESTARTS=0
SEQUENCE_NEXT_EPOCH=0
WATCHER_NEXT_EPOCH=0
CONTROLLER_NEXT_EPOCH=0
SEQUENCE_STARTED_EPOCH=0
WATCHER_STARTED_EPOCH=0
CONTROLLER_STARTED_EPOCH=0

write_status() {
    local phase="$1"
    local detail="$2"
    python - "$SUPERVISOR_STATUS" "$phase" "$detail" \
        "$SEQUENCE_PID" "$WATCHER_PID" \
        "$CONTROLLER_PID" \
        "$SEQUENCE_RESTARTS" "$WATCHER_RESTARTS" \
        "$CONTROLLER_RESTARTS" \
        "$SEQUENCE_NEXT_EPOCH" "$WATCHER_NEXT_EPOCH" \
        "$CONTROLLER_NEXT_EPOCH" \
        "$SEQUENCE_STARTED_EPOCH" "$WATCHER_STARTED_EPOCH" \
        "$CONTROLLER_STARTED_EPOCH" \
        "$GPU_OK" "$GPU_DETAIL" "$FINAL_REPORT" "$HARD_ALERT" <<'PY'
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

(
    raw_path,
    phase,
    detail,
    sequence_pid,
    watcher_pid,
    controller_pid,
    sequence_restarts,
    watcher_restarts,
    controller_restarts,
    sequence_next,
    watcher_next,
    controller_next,
    sequence_started,
    watcher_started,
    controller_started,
    gpu_ok,
    gpu_detail,
    final_report,
    hard_alert,
) = sys.argv[1:]
path = Path(raw_path)
payload = {
    "schema_version": 1,
    "updated_at": datetime.now(timezone.utc).isoformat(),
    "updated_epoch": time.time(),
    "phase": phase,
    "detail": detail,
    "sequence": {
        "pid": int(sequence_pid) if sequence_pid else None,
        "restart_attempts": int(sequence_restarts),
        "next_attempt_epoch": int(sequence_next),
        "started_epoch": int(sequence_started),
        "launch_mode": "resume",
    },
    "watcher": {
        "pid": int(watcher_pid) if watcher_pid else None,
        "restart_attempts": int(watcher_restarts),
        "next_attempt_epoch": int(watcher_next),
        "started_epoch": int(watcher_started),
        "read_only": True,
    },
    "repair_controller": {
        "pid": int(controller_pid) if controller_pid else None,
        "restart_attempts": int(controller_restarts),
        "next_attempt_epoch": int(controller_next),
        "started_epoch": int(controller_started),
        "durable": True,
        "llm_calls_on_healthy_observation": 0,
    },
    "gpu_probe": {
        "ok": gpu_ok == "true",
        "detail": gpu_detail,
        "advisory_only": True,
    },
    "contracts": {
        "canonical_final_report": final_report,
        "hard_alert": hard_alert,
    },
}
tmp = path.with_name(f".{path.name}.tmp.{os.getpid()}")
with tmp.open("w") as handle:
    json.dump(payload, handle, indent=2, sort_keys=True)
    handle.write("\n")
    handle.flush()
    os.fsync(handle.fileno())
os.replace(tmp, path)
dir_fd = os.open(path.parent, os.O_RDONLY)
try:
    os.fsync(dir_fd)
finally:
    os.close(dir_fd)
PY
}

write_supervisor_alert() {
    local component="$1"
    local detail="$2"
    python - "$SUPERVISOR_ALERT" "$component" "$detail" \
        "$SEQUENCE_RESTARTS" "$WATCHER_RESTARTS" \
        "$CONTROLLER_RESTARTS" \
        "$MAX_SEQUENCE_RESTARTS" "$MAX_WATCHER_RESTARTS" \
        "$MAX_CONTROLLER_RESTARTS" <<'PY'
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

path = Path(sys.argv[1])
payload = {
    "schema_version": 1,
    "created_at": datetime.now(timezone.utc).isoformat(),
    "status": "hard_stop",
    "component": sys.argv[2],
    "detail": sys.argv[3],
    "restart_attempts": {
        "sequence": int(sys.argv[4]),
        "watcher": int(sys.argv[5]),
        "repair_controller": int(sys.argv[6]),
    },
    "restart_caps": {
        "sequence": int(sys.argv[7]),
        "watcher": int(sys.argv[8]),
        "repair_controller": int(sys.argv[9]),
    },
    "stop_authorized": False,
    "training_was_not_killed": True,
}
tmp = path.with_name(f".{path.name}.tmp.{os.getpid()}")
with tmp.open("w") as handle:
    json.dump(payload, handle, indent=2, sort_keys=True)
    handle.write("\n")
    handle.flush()
    os.fsync(handle.fileno())
os.replace(tmp, path)
PY
}

load_restart_state() {
    [[ -f "$SUPERVISOR_STATUS" ]] || return 0
    local loaded
    loaded=$(python - "$SUPERVISOR_STATUS" <<'PY'
import json
import sys
from pathlib import Path

try:
    payload = json.loads(Path(sys.argv[1]).read_text())
    sequence = payload.get("sequence") or {}
    watcher = payload.get("watcher") or {}
    controller = payload.get("repair_controller") or {}
    values = (
        int(sequence.get("restart_attempts", 0)),
        int(watcher.get("restart_attempts", 0)),
        int(controller.get("restart_attempts", 0)),
        int(sequence.get("next_attempt_epoch", 0)),
        int(watcher.get("next_attempt_epoch", 0)),
        int(controller.get("next_attempt_epoch", 0)),
        int(sequence.get("started_epoch", 0)),
        int(watcher.get("started_epoch", 0)),
        int(controller.get("started_epoch", 0)),
    )
    if any(value < 0 for value in values):
        raise ValueError("negative restart state")
    print(*values, sep="\t")
except Exception:
    print("0\t0\t0\t0\t0\t0\t0\t0\t0")
PY
)
    IFS=$'\t' read -r SEQUENCE_RESTARTS WATCHER_RESTARTS CONTROLLER_RESTARTS \
        SEQUENCE_NEXT_EPOCH WATCHER_NEXT_EPOCH CONTROLLER_NEXT_EPOCH \
        SEQUENCE_STARTED_EPOCH WATCHER_STARTED_EPOCH CONTROLLER_STARTED_EPOCH <<<"$loaded"
}

probe_gpu() {
    local output rc
    # This must run in the supervisor's shell: phase8_nvidia_compat_apply
    # exports process-local library paths consumed by the following nvidia-smi.
    set +e
    phase8_nvidia_compat_apply >/dev/null 2>&1
    rc=$?
    set -e
    if (( rc != 0 )); then
        GPU_COMPAT_READY=false
        GPU_OK=false
        GPU_DETAIL="phase8 NVIDIA compatibility unavailable: ${PHASE8_NVIDIA_COMPAT_ERROR:-unknown error}"
        log "$GPU_DETAIL"
        return
    fi
    GPU_COMPAT_READY=true
    set +e
    output=$("$NVIDIA_SMI_BIN" \
        --query-gpu=index,utilization.gpu,memory.used,memory.total \
        --format=csv,noheader,nounits 2>&1)
    rc=$?
    set -e
    output="${output//$'\n'/; }"
    GPU_DETAIL="${output:0:1000}"
    if (( rc == 0 )); then
        GPU_OK=true
    else
        GPU_COMPAT_READY=false
        GPU_OK=false
        GPU_DETAIL="nvidia-smi rc=$rc: $GPU_DETAIL"
        log "GPU probe unavailable (advisory only): $GPU_DETAIL"
    fi
}

valid_final_report() {
    [[ -f "$FINAL_REPORT" ]] || return 1
    python - "$FINAL_REPORT" <<'PY' >/dev/null 2>&1
import json
import sys
from pathlib import Path

payload = json.loads(Path(sys.argv[1]).read_text())
expected_arms = [
    ("primary", "primary_control", "noq"),
    ("primary", "primary_resolution", "k2_balanced"),
    ("primary", "primary_resolution", "k5_balanced"),
    ("primary", "primary_resolution", "k10_balanced"),
    ("backup", "backup_resolution", "k3_balanced"),
    ("backup", "diagnostic_backup", "k5_fixed"),
    ("backup", "historical_reference", "k10_fixed"),
]
expected_primary = [item[2] for item in expected_arms[:4]]
expected_backup = [item[2] for item in expected_arms[4:]]
expected_claim_policy = {
    "primary_k_resolution_arms": ["k2_balanced", "k5_balanced", "k10_balanced"],
    "k5_fixed": (
        "diagnostic/backup strategy comparison against k5_balanced; "
        "excluded from the primary K-resolution claim"
    ),
}
if (
    not isinstance(payload, dict)
    or payload.get("status") != "passed"
    or payload.get("experiment") != "phase8_qwen_bucket_quarter_backlog"
    or payload.get("scale") != "quarter"
    or payload.get("stage1_updates") != 100000
    or payload.get("stage2_updates") != 50000
    or payload.get("train_rows") != 251599
    or payload.get("primary_order") != expected_primary
    or payload.get("backup_order") != expected_backup
    or payload.get("claim_policy") != expected_claim_policy
):
    raise SystemExit(1)
arms = payload.get("arms")
if (
    not isinstance(arms, list)
    or len(arms) != len(expected_arms)
):
    raise SystemExit(1)
for entry, (tier, role, arm) in zip(arms, expected_arms):
    if not isinstance(entry, dict):
        raise SystemExit(1)
    if entry.get("tier") != tier or entry.get("role") != role or entry.get("arm") != arm:
        raise SystemExit(1)
    report = Path(entry.get("report", ""))
    embedded = entry.get("payload")
    if (
        not report.is_file()
        or not isinstance(embedded, dict)
        or embedded.get("status") != "passed"
        or embedded.get("scale") != "quarter"
        or embedded.get("experiment") != f"phase8_qwen_bucket_quarter_{arm}"
    ):
        raise SystemExit(1)
    if arm == "noq":
        if (
            embedded.get("arm") != "noq"
            or embedded.get("matched_bucket_arm") != "k2_balanced"
            or embedded.get("q_conditioning") is not False
            or embedded.get("train_rows") != 251599
        ):
            raise SystemExit(1)
    else:
        if (
            embedded.get("k") != int(arm[1:].split("_", 1)[0])
            or embedded.get("strategy") != arm.split("_", 1)[1]
        ):
            raise SystemExit(1)
    audit_path = Path(embedded.get("training_audit", ""))
    if not audit_path.is_file():
        raise SystemExit(1)
    try:
        on_disk = json.loads(report.read_text())
        audit = json.loads(audit_path.read_text())
    except Exception:
        raise SystemExit(1)
    if (
        on_disk != embedded
        or audit.get("status") != "passed"
        or audit.get("stage1_iteration") != 100000
        or audit.get("stage2_iteration") != 150000
    ):
        raise SystemExit(1)
PY
}

hard_alert_present() {
    [[ -f "$HARD_ALERT" ]]
}

supervisor_alert_present() {
    [[ -f "$SUPERVISOR_ALERT" ]]
}

resume_marker_authorized() {
    [[ -f "$RESUME_MARKER" ]] || return 1
    python "$RESUME_MARKER_VALIDATOR" \
        --marker "$RESUME_MARKER" \
        --state "$CONTROLLER_STATE_DIR/state.json" \
        --status "$WATCHER_STATE_DIR/status.json" \
        --contract "$BACKLOG_CONTRACT" >/dev/null 2>&1
}

consume_resume_marker() {
    local destination
    destination="$CONTROLLER_STATE_DIR/RESUME_AUTHORIZED.consumed.$(date +%s).json"
    mv "$RESUME_MARKER" "$destination"
    log "consumed approved repair resume marker: $destination"
}

pid_matches() {
    local pid="$1"
    local needle="$2"
    [[ "$pid" =~ ^[0-9]+$ && -r "/proc/$pid/cmdline" ]] || return 1
    local command
    command=$(tr '\0' ' ' <"/proc/$pid/cmdline" 2>/dev/null || true)
    [[ " $command " == *" $needle "* ]]
}

find_component_pid() {
    local pid_file="$1"
    local needle="$2"
    local pid=""
    if [[ -f "$pid_file" ]]; then
        read -r pid <"$pid_file" || true
        if pid_matches "$pid" "$needle"; then
            printf '%s\n' "$pid"
            return 0
        fi
    fi
    local proc
    for proc in /proc/[0-9]*/cmdline; do
        pid="${proc#/proc/}"
        pid="${pid%/cmdline}"
        if pid_matches "$pid" "$needle"; then
            printf '%s\n' "$pid"
            return 0
        fi
    done
    return 1
}

write_pid_file() {
    local path="$1"
    local pid="$2"
    local tmp="${path}.tmp.$$"
    printf '%s\n' "$pid" >"$tmp"
    mv -f "$tmp" "$path"
}

backoff_for_attempt() {
    local attempt="$1"
    local delay="$INITIAL_BACKOFF_SECONDS"
    local count=1
    while (( count < attempt && delay < MAX_BACKOFF_SECONDS )); do
        delay=$((delay * 2))
        if (( delay > MAX_BACKOFF_SECONDS )); then
            delay="$MAX_BACKOFF_SECONDS"
        fi
        count=$((count + 1))
    done
    printf '%s\n' "$delay"
}

launch_sequence() {
    SEQUENCE_RESTARTS=$((SEQUENCE_RESTARTS + 1))
    log "launching sequence attempt=$SEQUENCE_RESTARTS mode=resume"
    nohup setsid env EXPERIMENT_RUN_MODE=resume \
        bash "$SEQUENCE" 9>&- >>"$SEQUENCE_LOG" 2>&1 </dev/null &
    SEQUENCE_PID=$!
    SEQUENCE_STARTED_EPOCH=$(date +%s)
    write_pid_file "$SEQUENCE_PID_FILE" "$SEQUENCE_PID"
    local delay
    delay=$(backoff_for_attempt "$SEQUENCE_RESTARTS")
    SEQUENCE_NEXT_EPOCH=$((SEQUENCE_STARTED_EPOCH + delay))
}

launch_watcher() {
    WATCHER_RESTARTS=$((WATCHER_RESTARTS + 1))
    log "launching read-only watcher attempt=$WATCHER_RESTARTS"
    nohup setsid python -u "$WATCHER" 9>&- >>"$WATCHER_LOG" 2>&1 </dev/null &
    WATCHER_PID=$!
    WATCHER_STARTED_EPOCH=$(date +%s)
    write_pid_file "$WATCHER_PID_FILE" "$WATCHER_PID"
    local delay
    delay=$(backoff_for_attempt "$WATCHER_RESTARTS")
    WATCHER_NEXT_EPOCH=$((WATCHER_STARTED_EPOCH + delay))
}

launch_controller() {
    CONTROLLER_RESTARTS=$((CONTROLLER_RESTARTS + 1))
    log "launching durable repair controller attempt=$CONTROLLER_RESTARTS"
    nohup setsid bash "$REPAIR_CONTROLLER" \
        --loop --status "$WATCHER_STATE_DIR/status.json" \
        --state-dir "$CONTROLLER_STATE_DIR" --root "$ROOT" \
        --execute-approved \
        >>"$CONTROLLER_LOG" 2>&1 </dev/null 9>&- &
    CONTROLLER_PID=$!
    CONTROLLER_STARTED_EPOCH=$(date +%s)
    write_pid_file "$CONTROLLER_PID_FILE" "$CONTROLLER_PID"
    local delay
    delay=$(backoff_for_attempt "$CONTROLLER_RESTARTS")
    CONTROLLER_NEXT_EPOCH=$((CONTROLLER_STARTED_EPOCH + delay))
}

preflight() {
    local failed=false
    for command in bash python flock nohup setsid readelf strings sha256sum awk \
        grep sed readlink; do
        if ! command -v "$command" >/dev/null 2>&1; then
            log "preflight missing command: $command"
            failed=true
        fi
    done
    if [[ ! -f "$SEQUENCE" ]]; then
        log "preflight missing sequence: $SEQUENCE"
        failed=true
    fi
    if [[ ! -f "$WATCHER" || ! -f "$RESUME_MARKER_VALIDATOR" ||
          ( "$ENABLE_REPAIR_CONTROLLER" == true &&
            ! -f "$REPAIR_CONTROLLER" ) ]]; then
        log "preflight missing watcher: $WATCHER"
        failed=true
    fi
    if [[ ! -f "$COMPAT_ENV" ]]; then
        log "preflight missing NVIDIA compatibility environment: $COMPAT_ENV"
        failed=true
    else
        # shellcheck source=/dev/null
        source "$COMPAT_ENV"
        if ! phase8_nvidia_compat_apply >/dev/null 2>&1; then
            log "preflight NVIDIA compatibility failed: $PHASE8_NVIDIA_COMPAT_ERROR"
            failed=true
        fi
    fi
    if [[ -f "$FINAL_REPORT" ]] && ! valid_final_report; then
        log "preflight canonical final exists but is not valid status=passed: $FINAL_REPORT"
        failed=true
    fi
    probe_gpu
    if [[ "$GPU_COMPAT_READY" != true ]]; then
        failed=true
    fi
    if [[ "$failed" == true ]]; then
        write_status PREFLIGHT_FAILED "required supervisor contract is incomplete"
        return 2
    fi
    write_status PREFLIGHT_OK \
        "dependencies valid; nvidia-smi is advisory and gpu_ok=$GPU_OK"
}

load_restart_state
if [[ "$PREFLIGHT_ONLY" == true ]]; then
    preflight
    log "preflight complete; sequence and watcher were not launched"
    exit 0
fi

if [[ ! -f "$SEQUENCE" || ! -f "$WATCHER" ||
      ! -f "$RESUME_MARKER_VALIDATOR" ||
      ( "$ENABLE_REPAIR_CONTROLLER" == true &&
        ! -f "$REPAIR_CONTROLLER" ) ]]; then
    probe_gpu
    write_status HARD_STOP "sequence, watcher, or repair controller contract file missing"
    log "required durable component missing; refusing to launch"
    exit 2
fi

probe_gpu
if supervisor_alert_present; then
    write_status HARD_STOP "SUPERVISOR_ALERT exists; manual review required"
    log "SUPERVISOR_ALERT exists at $SUPERVISOR_ALERT; no restart attempted"
    exit 5
fi
if valid_final_report; then
    write_status COMPLETE "canonical final report exists and status=passed; sequence skipped"
    log "canonical final report passed; sequence launch skipped"
    exit 0
fi

SEQUENCE_PID=$(find_component_pid "$SEQUENCE_PID_FILE" "$SEQUENCE" || true)
WATCHER_PID=$(find_component_pid "$WATCHER_PID_FILE" "$WATCHER" || true)
CONTROLLER_PID=$(find_component_pid "$CONTROLLER_PID_FILE" "$CONTROLLER_PROCESS_NEEDLE" || true)
if [[ "$ENABLE_REPAIR_CONTROLLER" != true ]]; then
    CONTROLLER_PID=""
fi
if [[ "$DRY_RUN" == true ]]; then
    detail="would supervise sequence(mode=resume) and read-only watcher; repair_controller_enabled=$ENABLE_REPAIR_CONTROLLER"
    [[ -n "$SEQUENCE_PID" ]] && detail+="; sequence already alive pid=$SEQUENCE_PID"
    [[ -n "$WATCHER_PID" ]] && detail+="; watcher already alive pid=$WATCHER_PID"
    [[ -n "$CONTROLLER_PID" ]] && detail+="; controller already alive pid=$CONTROLLER_PID"
    write_status DRY_RUN "$detail"
    log "$detail; nothing launched"
    exit 0
fi

log "supervisor started sequence_restarts=$SEQUENCE_RESTARTS watcher_restarts=$WATCHER_RESTARTS"
while true; do
    now=$(date +%s)
    probe_gpu
    RESUME_ALLOWED=false
    if resume_marker_authorized; then
        RESUME_ALLOWED=true
        SEQUENCE_NEXT_EPOCH=0
    fi

    if supervisor_alert_present; then
        write_status HARD_STOP "SUPERVISOR_ALERT exists; no further restarts"
        log "SUPERVISOR_ALERT exists; leaving all processes untouched"
        exit 5
    fi
    if valid_final_report; then
        write_status COMPLETE \
            "canonical final report exists and status=passed; no further restarts"
        log "canonical final report passed; supervision complete"
        exit 0
    fi

    SEQUENCE_PID=$(find_component_pid "$SEQUENCE_PID_FILE" "$SEQUENCE" || true)
    WATCHER_PID=$(find_component_pid "$WATCHER_PID_FILE" "$WATCHER" || true)
    CONTROLLER_PID=$(find_component_pid "$CONTROLLER_PID_FILE" "$CONTROLLER_PROCESS_NEEDLE" || true)
    if [[ "$ENABLE_REPAIR_CONTROLLER" != true ]]; then
        CONTROLLER_PID=""
    fi

    if hard_alert_present && [[ "$RESUME_ALLOWED" != true ]]; then
        # A watcher hard alert is an incident input for the durable controller,
        # not a reason to exit.  Preserve any existing sequence untouched and
        # suppress only sequence relaunches until the alert clears.
        log "hard ALERT present; sequence relaunch paused; watcher/controller remain supervised"
    elif [[ -n "$SEQUENCE_PID" ]]; then
        if (( SEQUENCE_STARTED_EPOCH > 0 &&
              now - SEQUENCE_STARTED_EPOCH >= STABLE_SECONDS )); then
            SEQUENCE_RESTARTS=0
            SEQUENCE_NEXT_EPOCH=0
            SEQUENCE_STARTED_EPOCH=0
        fi
    elif [[ "$GPU_COMPAT_READY" != true ]]; then
        # A failed userspace/version/NVML probe is never treated as idle and
        # never consumes the sequence restart budget.
        SEQUENCE_RESTARTS=0
        SEQUENCE_NEXT_EPOCH=0
        SEQUENCE_STARTED_EPOCH=0
    elif (( now >= SEQUENCE_NEXT_EPOCH )); then
        if (( SEQUENCE_RESTARTS >= MAX_SEQUENCE_RESTARTS )); then
            detail="sequence restart cap exceeded ($SEQUENCE_RESTARTS/$MAX_SEQUENCE_RESTARTS)"
            write_supervisor_alert sequence "$detail"
            write_status SUPERVISOR_ALERT "$detail; training was not killed"
            log "$detail; wrote $SUPERVISOR_ALERT and stopped restarting"
            exit 5
        fi
        launch_sequence
        if [[ "$RESUME_ALLOWED" == true ]]; then
            consume_resume_marker
            RESUME_ALLOWED=false
        fi
    fi

    if [[ -n "$WATCHER_PID" ]]; then
        if (( WATCHER_STARTED_EPOCH > 0 &&
              now - WATCHER_STARTED_EPOCH >= STABLE_SECONDS )); then
            WATCHER_RESTARTS=0
            WATCHER_NEXT_EPOCH=0
            WATCHER_STARTED_EPOCH=0
        fi
    elif (( now >= WATCHER_NEXT_EPOCH )); then
        if (( WATCHER_RESTARTS >= MAX_WATCHER_RESTARTS )); then
            detail="watcher restart cap exceeded ($WATCHER_RESTARTS/$MAX_WATCHER_RESTARTS)"
            write_supervisor_alert watcher "$detail"
            write_status SUPERVISOR_ALERT "$detail; training was not killed"
            log "$detail; wrote $SUPERVISOR_ALERT and stopped restarting"
            exit 5
        fi
        launch_watcher
    fi

    if [[ "$ENABLE_REPAIR_CONTROLLER" == true ]]; then
        if [[ -n "$CONTROLLER_PID" ]]; then
            if (( CONTROLLER_STARTED_EPOCH > 0 &&
                  now - CONTROLLER_STARTED_EPOCH >= STABLE_SECONDS )); then
                CONTROLLER_RESTARTS=0
                CONTROLLER_NEXT_EPOCH=0
                CONTROLLER_STARTED_EPOCH=0
            fi
        elif (( now >= CONTROLLER_NEXT_EPOCH )); then
            if (( CONTROLLER_RESTARTS >= MAX_CONTROLLER_RESTARTS )); then
                detail="repair controller restart cap exceeded ($CONTROLLER_RESTARTS/$MAX_CONTROLLER_RESTARTS)"
                write_supervisor_alert repair_controller "$detail"
                write_status SUPERVISOR_ALERT "$detail; training was not killed"
                log "$detail; wrote $SUPERVISOR_ALERT and stopped restarting"
                exit 5
            fi
            launch_controller
        fi
    fi

    if hard_alert_present && [[ -n "$SEQUENCE_PID" ]]; then
        write_status REPAIR_RESUME_VALIDATION \
            "sequence resumed from exact SOL-approved marker; awaiting watcher forward-progress validation"
    elif hard_alert_present; then
        write_status HARD_ALERT_ACTIVE \
            "watcher hard ALERT is active; sequence relaunch paused; watcher_pid=${WATCHER_PID:-none}; controller_pid=${CONTROLLER_PID:-none}"
    elif [[ -z "$SEQUENCE_PID" && "$GPU_COMPAT_READY" != true ]]; then
        write_status WAITING_FOR_GPU_DRIVER \
            "NVIDIA per-process compatibility is not ready; watcher_pid=${WATCHER_PID:-none}; controller_pid=${CONTROLLER_PID:-none}; sequence will auto-start after recovery"
    else
        write_status MONITORING \
            "sequence_pid=${SEQUENCE_PID:-none}; watcher_pid=${WATCHER_PID:-none}; controller_pid=${CONTROLLER_PID:-none}; gpu_ok=$GPU_OK"
    fi
    sleep "$POLL_SECONDS"
done
