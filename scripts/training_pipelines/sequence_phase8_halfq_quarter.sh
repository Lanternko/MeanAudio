#!/usr/bin/env bash
# Wait for other MeanAudio GPU work, then run the quarter No-Q vs half-Q chain.

set -euo pipefail

ROOT=/home/kojiek/MeanAudio
LOG_ROOT=/home/kojiek/logs
STATE="$LOG_ROOT/phase8_halfq_quarter"
LOG="$LOG_ROOT/phase8_halfq_quarter_sequence.log"
LOCK="$STATE/sequence.lock"
POLL_SECONDS="${POLL_SECONDS:-60}"

mkdir -p "$STATE" "$LOG_ROOT"
cd "$ROOT"
# shellcheck source=/dev/null
source /home/kojiek/venvs/dac/bin/activate

exec 9>"$LOCK"
if ! flock -n 9; then
    echo "[FAIL] half-Q quarter sequence already running" >&2
    exit 3
fi

log() {
    echo "[HALFQ-QUEUE] $(date --iso-8601=seconds) $*" | tee -a "$LOG"
}

gpu_blockers() {
    python - <<'PY'
import subprocess
from pathlib import Path

result = subprocess.run(
    [
        "nvidia-smi",
        "--query-compute-apps=pid,used_memory",
        "--format=csv,noheader,nounits",
    ],
    check=True,
    capture_output=True,
    text=True,
)
blockers = []
for line in result.stdout.splitlines():
    if not line.strip():
        continue
    pid_raw, memory_raw = [part.strip() for part in line.split(",", 1)]
    pid = int(pid_raw)
    proc = Path("/proc") / str(pid)
    try:
        command = (proc / "cmdline").read_bytes().replace(b"\0", b" ").decode(
            errors="replace"
        )
    except OSError:
        continue
    # The unrelated Irodori-TTS service is a known small resident process.
    if "Irodori-TTS" in command:
        continue
    blockers.append((pid, int(memory_raw), command))
for pid, memory, command in blockers:
    print(f"{pid}\t{memory} MiB\t{command}")
raise SystemExit(0 if not blockers else 1)
PY
}

log "CPU preflight starts"
PREFLIGHT_ONLY=true \
    bash scripts/training_pipelines/train_pipeline_phase8_halfq_quarter.sh \
    2>&1 | tee -a "$LOG"
log "CPU preflight passed"

while ! blockers=$(gpu_blockers); do
    log "waiting for GPU; blockers: ${blockers//$'\n'/; }"
    sleep "$POLL_SECONDS"
done

log "GPU clear; starting quarter baseline evaluation + half-Q training/evaluation"
PREFLIGHT_ONLY=false EXPERIMENT_RUN_MODE=fresh \
    bash scripts/training_pipelines/train_pipeline_phase8_halfq_quarter.sh \
    2>&1 | tee -a "$LOG"
log "complete"
