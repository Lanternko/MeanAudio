#!/usr/bin/env bash
# Durable sequential Phase-8 Qwen bucket quarter backlog.
#
# Primary:
#   No-Q -> K=2 balanced (historical reuse) -> K=5 balanced -> K=10 balanced
# Backup (only after every primary report validates):
#   K=3 balanced -> K=5 fixed diagnostic -> K=10 fixed (historical reuse)
#
# This script does not launch in PREFLIGHT_ONLY=true or DRY_RUN=true mode.

set -euo pipefail

ROOT=/home/kojiek/MeanAudio
DATA=/mnt/HDD/kojiek/phase4_jamendo_data
LOG_ROOT=/home/kojiek/logs
STATE="$LOG_ROOT/phase8_qwen_bucket_quarter_backlog"
LOCK="$STATE/sequence.lock"
LOG="$LOG_ROOT/phase8_qwen_bucket_quarter_backlog_sequence.log"
FINAL_REPORT="$LOG_ROOT/phase8_qwen_bucket_quarter_backlog_FINAL_METRICS.json"
GRID="$DATA/phase8_qwen_meansim_bucket_grid.manifest.json"
MUSICCAPS="$DATA/musiccaps_test.tsv"
NOQ_TSV="$DATA/phase8_qwen_meansim_k2_balanced.tsv"
RUN_MODE="${EXPERIMENT_RUN_MODE:-fresh}"
POLL_SECONDS="${POLL_SECONDS:-60}"
PREFLIGHT_ONLY="${PREFLIGHT_ONLY:-false}"
DRY_RUN="${DRY_RUN:-false}"
GPU_CHECK_ONLY="${GPU_CHECK_ONLY:-false}"
NVIDIA_SMI_BIN="${NVIDIA_SMI_BIN:-nvidia-smi}"
# Entries are complete, NUL-normalized /proc cmdlines.  A basename/path
# substring is deliberately not sufficient to exempt a GPU process.
GPU_RESIDENT_ALLOWLIST="${GPU_RESIDENT_ALLOWLIST:-/home/kojiek/side_projects/reference-repos/Irodori-TTS/.venv/bin/python /home/kojiek/side_projects/apps/arale-persona-bot/server/tts_server_irodori.py}"
GPU_RESIDENT_MAX_MIB="${GPU_RESIDENT_MAX_MIB:-2048}"

case "$RUN_MODE" in fresh|resume) ;; *)
    echo "[FAIL] EXPERIMENT_RUN_MODE must be fresh or resume" >&2; exit 2 ;;
esac
case "$POLL_SECONDS" in
    ''|*[!0-9]*) echo "[FAIL] POLL_SECONDS must be a positive integer" >&2; exit 2 ;;
    0) echo "[FAIL] POLL_SECONDS must be greater than zero" >&2; exit 2 ;;
esac
case "$GPU_RESIDENT_MAX_MIB" in
    ''|*[!0-9]*) echo "[FAIL] GPU_RESIDENT_MAX_MIB must be a non-negative integer" >&2; exit 2 ;;
esac
for value_name in PREFLIGHT_ONLY DRY_RUN GPU_CHECK_ONLY; do
    case "${!value_name}" in true|false) ;; *)
        echo "[FAIL] $value_name must be true or false" >&2; exit 2 ;;
    esac
done

cd "$ROOT"

log() {
    echo "[QWEN-BACKLOG] $(date --iso-8601=seconds) $*" | tee -a "$LOG"
}

report_for() {
    case "$1" in
        noq) echo "$LOG_ROOT/phase8_qwen_bucket_quarter_noq_FINAL_METRICS.json" ;;
        *)
            local k="$2" strategy="$3"
            echo "$LOG_ROOT/phase8_qwen_bucket_quarter_k${k}_${strategy}_FINAL_METRICS.json"
            ;;
    esac
}

# Validate identity, data binding, training audit, models, full MusicCaps
# endpoints, and holdout metrics before treating any report as complete.
validate_report() {
    local kind="$1" k="$2" strategy="$3" report="$4"
    python - "$kind" "$k" "$strategy" "$report" "$GRID" "$MUSICCAPS" \
        "$NOQ_TSV" <<'PY'
import hashlib
import json
import math
import sys
from pathlib import Path

kind, k_raw, strategy, report_path, grid_path, musiccaps, noq_tsv = sys.argv[1:]
report_path, grid_path, musiccaps, noq_tsv = map(
    Path, (report_path, grid_path, musiccaps, noq_tsv)
)

def fail(message):
    raise SystemExit(f"[FAIL] {report_path}: {message}")

def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

def metric_endpoint(section, key):
    value = section.get(key)
    if not isinstance(value, dict):
        fail(f"missing metric endpoint {key}")
    required = {"clap_score", "aes_CE", "aes_CU", "aes_PC", "aes_PQ"}
    values = {name: value.get(name) for name in required}
    if not all(isinstance(item, (int, float)) and math.isfinite(item)
               for item in values.values()):
        fail(f"incomplete/nonfinite metric endpoint {key}: {values}")

if not report_path.is_file():
    fail("report missing")
try:
    payload = json.loads(report_path.read_text())
except Exception as exc:
    fail(f"invalid JSON: {exc}")
if payload.get("status") != "passed" or payload.get("scale") != "quarter":
    fail("status/scale mismatch")
prompts = payload.get("prompts", {})
if (
    prompts.get("path") != str(musiccaps)
    or prompts.get("rows") != 5521
    or prompts.get("sha256") != sha(musiccaps)
):
    fail("MusicCaps prompt binding mismatch")

audit_path = Path(payload.get("training_audit", ""))
if not audit_path.is_file():
    fail("training audit missing")
audit = json.loads(audit_path.read_text())
if audit.get("status") != "passed":
    fail("training audit not passed")
contract_path = Path(audit.get("contract", ""))
if not contract_path.is_file():
    fail("training contract missing")
contract = json.loads(contract_path.read_text())

models = payload.get("models", {})
if set(models) != {"stage1", "global"}:
    fail("model set mismatch")
for label, model in models.items():
    path = Path(model.get("path", ""))
    if not path.is_file() or model.get("sha256") != sha(path):
        fail(f"{label} model missing/hash drift")

stage1 = payload.get("stage1", {})
global_section = payload.get("global", {})
grid = json.loads(grid_path.read_text())
if kind == "noq":
    if (
        payload.get("experiment") != "phase8_qwen_bucket_quarter_noq"
        or payload.get("arm") != "noq"
        or payload.get("matched_bucket_arm") != "k2_balanced"
        or payload.get("q_conditioning") is not False
        or payload.get("train_rows") != 251599
        or payload.get("train_tsv", {}).get("path") != str(noq_tsv)
        or payload.get("train_tsv", {}).get("sha256") != sha(noq_tsv)
        or grid["outputs"]["k2_balanced"]["sha256"] != sha(noq_tsv)
        or audit.get("arm") != "noq"
        or audit.get("matched_bucket_arm") != "k2_balanced"
        or audit.get("stage1_iteration") != 100000
        or audit.get("stage2_iteration") != 150000
        or audit.get("stage1_use_q_conditioning") is not False
        or audit.get("stage2_use_q_conditioning") is not False
        or contract.get("train_tsv_sha256") != sha(noq_tsv)
        or contract.get("expected_rows") != 251599
    ):
        fail("No-Q identity/alignment/iteration contract mismatch")
    if stage1.get("protocol") != "MusicCaps 5521; FluxAudio FM25 CFG4.5; no_q":
        fail("No-Q Stage-1 protocol mismatch")
    if global_section.get("protocol") != "MusicCaps 5521; MeanFlow1 CFG0.5; no_q":
        fail("No-Q global protocol mismatch")
    metric_endpoint(stage1, "no_q")
    metric_endpoint(stage1, "holdout5009_no_q")
    metric_endpoint(global_section, "no_q")
    metric_endpoint(global_section, "holdout5009_no_q")
else:
    k = int(k_raw)
    expected_name = f"phase8_qwen_bucket_quarter_k{k}_{strategy}"
    if (
        payload.get("experiment") != expected_name
        or payload.get("k") != k
        or payload.get("strategy") != strategy
        or audit.get("stage1_iteration") != 100000
        or audit.get("stage2_iteration") != 150000
        or contract.get("train_tsv_sha256")
            != grid["outputs"][f"k{k}_{strategy}"]["sha256"]
        or stage1.get("protocol") != "MusicCaps 5521; FluxAudio FM25 CFG4.5"
        or global_section.get("protocol") != "MusicCaps 5521; MeanFlow1 CFG0.5"
    ):
        fail("bucket identity/data/iteration/protocol mismatch")
    metric_endpoint(stage1, "high_q9")
    metric_endpoint(stage1, "holdout5009_high_q9")
    metric_endpoint(global_section, "high_q9")
    metric_endpoint(global_section, "holdout5009_high_q9")
print(f"[OK] verified completed report: {report_path}")
PY
}

dry_run_plan() {
    cat <<EOF
[DRY RUN] Phase-8 Qwen quarter backlog; mode=$RUN_MODE
[DRY RUN] primary 1/4: No-Q; official Qwen K=2 balanced TSV/NPZ; train
[DRY RUN] primary 2/4: K=2 balanced; REUSE=k2_balanced_historical
[DRY RUN] primary 3/4: K=5 balanced; REUSE=none
[DRY RUN] primary 4/4: K=10 balanced; REUSE=none
[DRY RUN] backup 1/3: K=3 balanced; REUSE=none
[DRY RUN] backup 2/3: K=5 fixed; REUSE=none; diagnostic/strategy comparison only
[DRY RUN] backup 3/3: K=10 fixed; REUSE=k10_fixed_historical
[DRY RUN] GPU execution disabled
EOF
}

if [ "$DRY_RUN" = true ]; then
    dry_run_plan
    exit 0
fi

# shellcheck source=/dev/null
source /home/kojiek/venvs/dac/bin/activate
# shellcheck source=/dev/null
source "$ROOT/scripts/runtime/phase8_nvidia_compat_env.sh"
phase8_nvidia_compat_apply || {
    echo "[FAIL] NVIDIA 595.71.05 per-process compatibility preflight failed: $PHASE8_NVIDIA_COMPAT_ERROR" >&2
    exit 2
}
export PHASE8_NVIDIA_FUNCTIONAL_PREFLIGHT="${PHASE8_NVIDIA_FUNCTIONAL_PREFLIGHT:-true}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

for path in "$GRID" "$MUSICCAPS" "$NOQ_TSV" \
    "$ROOT/scripts/run_with_experiment_report.sh" \
    "$ROOT/scripts/training_pipelines/execute_phase8_qwen_bucket_arm_eval.sh" \
    "$ROOT/scripts/training_pipelines/execute_phase8_qwen_noq_arm_eval.sh"; do
    [ -e "$path" ] || { echo "[FAIL] missing $path" >&2; exit 2; }
done

run_preflight() {
    echo "[PREFLIGHT] No-Q official-aligned adapter"
    PREFLIGHT_ONLY=true EXPERIMENT_RUN_MODE="$RUN_MODE" \
        bash scripts/training_pipelines/execute_phase8_qwen_noq_arm_eval.sh

    local k strategy reuse
    while read -r k strategy reuse; do
        echo "[PREFLIGHT] K=$k strategy=$strategy reuse=$reuse"
        K="$k" STRATEGY="$strategy" SCALE=quarter REUSE="$reuse" \
            VALIDATE_ONLY=true EXPERIMENT_RUN_MODE="$RUN_MODE" \
            bash scripts/training_pipelines/execute_phase8_qwen_bucket_arm_eval.sh
    done <<'EOF'
2 balanced k2_balanced_historical
5 balanced none
10 balanced none
3 balanced none
5 fixed none
10 fixed k10_fixed_historical
EOF
}

run_preflight
if [ "$PREFLIGHT_ONLY" = true ]; then
    echo "[PREFLIGHT ONLY] all seven arms passed; no GPU process started."
    exit 0
fi

mkdir -p "$STATE" "$LOG_ROOT"
exec 9>"$LOCK"
flock -n 9 || {
    echo "[FAIL] Qwen quarter backlog sequence already running" >&2
    exit 3
}

# Return 0 only when the selected GPU is demonstrably clear.  NVML is primary.
# If NVML fails or emits malformed output, inspect /proc command lines and open
# NVIDIA device descriptors.  Any unreadable process state is an UNKNOWN
# blocker, so query failure can never be mistaken for idle.
gpu_clear() {
    python - "$NVIDIA_SMI_BIN" "$GPU_RESIDENT_ALLOWLIST" \
        "$GPU_RESIDENT_MAX_MIB" <<'PY'
import os
import subprocess
import sys
from pathlib import Path

nvidia_smi, allowlist_raw, max_mib_raw = sys.argv[1:]
allowlist = tuple(item for item in allowlist_raw.split(":") if item)
max_mib = int(max_mib_raw)

def is_allowed_resident(command):
    return bool(command and command in allowlist)

def command_for(pid):
    raw = (Path("/proc") / str(pid) / "cmdline").read_bytes()
    return raw.replace(b"\0", b" ").decode(errors="replace").strip()

try:
    result = subprocess.run(
        [
            nvidia_smi,
            "--query-compute-apps=pid,used_memory",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=15,
    )
    blockers = []
    for line in result.stdout.splitlines():
        if not line.strip():
            continue
        pieces = [piece.strip() for piece in line.split(",")]
        if len(pieces) != 2:
            raise ValueError(f"malformed NVML row: {line!r}")
        pid, memory = int(pieces[0]), int(pieces[1])
        try:
            command = command_for(pid)
        except OSError:
            command = "<process exited or cmdline unreadable>"
        if is_allowed_resident(command) and memory <= max_mib:
            print(f"IGNORED_RESIDENT=pid={pid} memory={memory}MiB cmd={command}")
            continue
        blockers.append(f"pid={pid} memory={memory}MiB cmd={command}")
    print("MODE=nvml")
    for blocker in blockers:
        print(blocker)
    raise SystemExit(1 if blockers else 0)
except (OSError, subprocess.SubprocessError, ValueError) as exc:
    nvml_error = f"{type(exc).__name__}: {exc}"

gpu_signatures = (
    "torchrun", "train.py", "eval.py", "phase4_eval.py",
    "accelerate launch", "deepspeed",
)
blockers = {}
unknown = []
proc = Path("/proc")
if not proc.is_dir():
    print(f"MODE=process-fallback NVML_ERROR={nvml_error}")
    print("UNKNOWN=/proc unavailable")
    raise SystemExit(2)

ancestors = {os.getpid()}
ancestor = os.getppid()
while ancestor > 1 and ancestor not in ancestors:
    ancestors.add(ancestor)
    try:
        stat = (proc / str(ancestor) / "stat").read_text()
        ancestor = int(stat.rsplit(")", 1)[1].split()[1])
    except (FileNotFoundError, PermissionError, OSError, ValueError, IndexError):
        break

for entry in proc.iterdir():
    if not entry.name.isdigit() or int(entry.name) in ancestors:
        continue
    pid = int(entry.name)
    try:
        command = command_for(pid)
    except FileNotFoundError:
        continue
    except (PermissionError, OSError) as exc:
        unknown.append(f"pid={pid} cmdline={type(exc).__name__}")
        continue
    if is_allowed_resident(command):
        unknown.append(
            f"pid={pid} allowlisted_resident_vram_unknown cmd={command}"
        )
        continue
    if any(signature in command for signature in gpu_signatures):
        blockers[pid] = f"pid={pid} cmd={command}"
        continue
    fd_dir = entry / "fd"
    try:
        fds = list(fd_dir.iterdir())
    except FileNotFoundError:
        continue
    except (PermissionError, OSError) as exc:
        # A readable command that is not GPU-like is not promoted to unknown
        # solely because its descriptors are protected.
        continue
    try:
        for fd in fds:
            try:
                target = os.readlink(fd)
            except FileNotFoundError:
                continue
            except (PermissionError, OSError):
                # The command line was readable and did not match a GPU-work
                # signature.  Protected individual descriptors (for example
                # systemd's fd 0) are not evidence of a hidden GPU process.
                continue
            if target.startswith("/dev/nvidia"):
                blockers[pid] = f"pid={pid} nvidia_fd={target} cmd={command}"
                break
    except FileNotFoundError:
        continue

print(f"MODE=process-fallback NVML_ERROR={nvml_error}")
for blocker in blockers.values():
    print(blocker)
for item in unknown:
    print(f"UNKNOWN={item}")
if unknown:
    raise SystemExit(2)
# The fallback has no authoritative per-process VRAM values.  Even when its
# command/device scan finds no blocker, it must not claim that the GPU is idle.
print("UNKNOWN=NVML VRAM data unavailable in fallback; refusing idle")
raise SystemExit(2)
PY
}

if [ "$GPU_CHECK_ONLY" = true ]; then
    gpu_clear
    exit $?
fi

wait_for_gpu() {
    local status blockers functional_output functional_status
    while true; do
        set +e
        blockers=$(gpu_clear 2>&1)
        status=$?
        set -e
        if [ "$status" -eq 0 ]; then
            set +e
            functional_output=$(
                PHASE8_NVIDIA_FUNCTIONAL_PREFLIGHT=true \
                    phase8_nvidia_compat_functional_preflight 2>&1
            )
            functional_status=$?
            set -e
            functional_output="${functional_output//$'\n'/; }"
            if [ "$functional_status" -eq 0 ]; then
                log "GPU clear and functional probe passed (${blockers//$'\n'/; }; ${functional_output})"
                return
            fi
            log "GPU functional probe failed; fail-closed wait: ${functional_output}"
            sleep "$POLL_SECONDS"
            continue
        fi
        if [ "$status" -eq 1 ]; then
            log "waiting for GPU blockers: ${blockers//$'\n'/; }"
        else
            log "GPU state unknown; fail-closed wait: ${blockers//$'\n'/; }"
        fi
        sleep "$POLL_SECONDS"
    done
}

run_noq() {
    local report wrapper_log
    report=$(report_for noq 0 none)
    if [ -f "$report" ]; then
        validate_report noq 0 none "$report"
        log "skip verified completed primary No-Q report=$report"
        return
    fi
    wait_for_gpu
    wrapper_log="$LOG_ROOT/phase8_qwen_bucket_quarter_noq_wrapper.log"
    log "primary arm starts: No-Q mode=$RUN_MODE"
    scripts/run_with_experiment_report.sh \
        --experiment phase8_qwen_bucket_quarter_noq \
        --report "$report" --log "$wrapper_log" \
        -- env EXPERIMENT_RUN_MODE="$RUN_MODE" \
            bash scripts/training_pipelines/execute_phase8_qwen_noq_arm_eval.sh \
        2>&1 | tee "$wrapper_log"
    validate_report noq 0 none "$report"
}

run_bucket() {
    local tier="$1" role="$2" k="$3" strategy="$4" reuse="$5"
    local report name wrapper_log
    report=$(report_for bucket "$k" "$strategy")
    name="phase8_qwen_bucket_quarter_k${k}_${strategy}"
    if [ -f "$report" ]; then
        validate_report bucket "$k" "$strategy" "$report"
        log "skip verified completed $tier arm=$name role=$role"
        return
    fi
    wait_for_gpu
    wrapper_log="$LOG_ROOT/${name}_wrapper.log"
    log "$tier arm starts: $name role=$role reuse=$reuse mode=$RUN_MODE"
    scripts/run_with_experiment_report.sh \
        --experiment "$name" --report "$report" --log "$wrapper_log" \
        -- env K="$k" STRATEGY="$strategy" SCALE=quarter REUSE="$reuse" \
            EXPERIMENT_RUN_MODE="$RUN_MODE" \
            bash scripts/training_pipelines/execute_phase8_qwen_bucket_arm_eval.sh \
        2>&1 | tee "$wrapper_log"
    validate_report bucket "$k" "$strategy" "$report"
}

# set -e and pipefail make a failed primary terminal.  The backup section is
# unreachable until every primary has returned with a validated report.
run_noq
run_bucket primary primary_resolution 2 balanced k2_balanced_historical
run_bucket primary primary_resolution 5 balanced none
run_bucket primary primary_resolution 10 balanced none

log "all four primary reports validated; backup chain is now eligible"
for primary_spec in "noq 0 none" "bucket 2 balanced" \
    "bucket 5 balanced" "bucket 10 balanced"; do
    # shellcheck disable=SC2086
    set -- $primary_spec
    validate_report "$1" "$2" "$3" "$(report_for "$1" "$2" "$3")"
done

run_bucket backup backup_resolution 3 balanced none
run_bucket backup diagnostic_backup 5 fixed none
run_bucket backup historical_reference 10 fixed k10_fixed_historical

python - "$FINAL_REPORT" "$LOG_ROOT" "$GRID" "$MUSICCAPS" <<'PY'
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

out, logs, grid, musiccaps = map(Path, sys.argv[1:])
arms = [
    ("primary", "primary_control", "noq",
     logs / "phase8_qwen_bucket_quarter_noq_FINAL_METRICS.json"),
    ("primary", "primary_resolution", "k2_balanced",
     logs / "phase8_qwen_bucket_quarter_k2_balanced_FINAL_METRICS.json"),
    ("primary", "primary_resolution", "k5_balanced",
     logs / "phase8_qwen_bucket_quarter_k5_balanced_FINAL_METRICS.json"),
    ("primary", "primary_resolution", "k10_balanced",
     logs / "phase8_qwen_bucket_quarter_k10_balanced_FINAL_METRICS.json"),
    ("backup", "backup_resolution", "k3_balanced",
     logs / "phase8_qwen_bucket_quarter_k3_balanced_FINAL_METRICS.json"),
    ("backup", "diagnostic_backup", "k5_fixed",
     logs / "phase8_qwen_bucket_quarter_k5_fixed_FINAL_METRICS.json"),
    ("backup", "historical_reference", "k10_fixed",
     logs / "phase8_qwen_bucket_quarter_k10_fixed_FINAL_METRICS.json"),
]
reports = []
for tier, role, arm, path in arms:
    if not path.is_file():
        raise SystemExit(f"[FAIL] finalization missing report: {path}")
    payload = json.loads(path.read_text())
    if payload.get("status") != "passed":
        raise SystemExit(f"[FAIL] finalization report not passed: {path}")
    reports.append({
        "tier": tier, "role": role, "arm": arm,
        "report": str(path), "payload": payload,
    })
result = {
    "schema_version": 1,
    "completed_at": datetime.now(timezone.utc).isoformat(),
    "status": "passed",
    "experiment": "phase8_qwen_bucket_quarter_backlog",
    "scale": "quarter",
    "grid_manifest": str(grid),
    "musiccaps": str(musiccaps),
    "stage1_updates": 100000,
    "stage2_updates": 50000,
    "train_rows": 251599,
    "primary_order": ["noq", "k2_balanced", "k5_balanced", "k10_balanced"],
    "backup_order": ["k3_balanced", "k5_fixed", "k10_fixed"],
    "claim_policy": {
        "primary_k_resolution_arms": ["k2_balanced", "k5_balanced", "k10_balanced"],
        "k5_fixed": (
            "diagnostic/backup strategy comparison against k5_balanced; "
            "excluded from the primary K-resolution claim"
        ),
    },
    "arms": reports,
}
tmp = out.with_suffix(out.suffix + ".tmp")
tmp.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
os.replace(tmp, out)
print(f"[COMPLETE] {out}")
PY
log "complete report=$FINAL_REPORT"
