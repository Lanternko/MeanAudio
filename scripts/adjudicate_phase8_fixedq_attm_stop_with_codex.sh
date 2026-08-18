#!/usr/bin/env bash
# Read-only Codex SOL adjudication for the Fixed-Q9 vs matched-NoQ queue.
# This wrapper never sends a signal; it only writes a structured verdict.

set -euo pipefail

ROOT=/home/kojiek/MeanAudio
STATE=/home/kojiek/logs/phase8_fixedq_attm_monitor
SCHEMA="$ROOT/scripts/phase8_codex_stop_verdict.schema.json"
VERDICT="$STATE/codex_sol_verdict.json"
TRANSCRIPT="$STATE/codex_sol_adjudication.jsonl"
EVIDENCE="$STATE/codex_sol_evidence.txt"

mkdir -p "$STATE"
rm -f "$VERDICT"

if [ "${1:-}" = --dry-run ]; then
    echo "model=gpt-5.6-sol sandbox=read-only verdict=$VERDICT"
    exit 0
fi

{
    echo "captured_at=$(date --iso-8601=seconds)"
    "$ROOT/scripts/monitor_phase8_fixedq_attm_ft.py" --once || true
    echo "--- status ---"
    cat "$STATE/status.json" 2>/dev/null || true
    echo "--- alert ---"
    cat "$STATE/ALERT.json" 2>/dev/null || true
    echo "--- contracts and audits ---"
    cat /home/kojiek/logs/phase8_fixedq9_prior_ft100k_contract.json 2>/dev/null || true
    cat /home/kojiek/logs/phase8_matched_noq_ft100k_contract.json 2>/dev/null || true
    cat "$STATE"/*_FINAL_AUDIT.json 2>/dev/null || true
    echo "--- processes/tmux/GPU/disk ---"
    pgrep -af 'phase8_(fixedq9_prior|matched_noq)_ft100k|sequence_phase8_fixedq_attm' || true
    tmux ls 2>&1 || true
    nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu \
        --format=csv,noheader,nounits 2>&1 || true
    df -h / /mnt/HDD 2>&1 || true
    echo "--- current logs ---"
    tail -n 160 /home/kojiek/logs/phase8_fixedq9_prior_ft100k_stage2_ft100000.log 2>/dev/null || true
    tail -n 160 /home/kojiek/logs/phase8_matched_noq_ft100k_stage2_ft100000.log 2>/dev/null || true
    tail -n 160 /home/kojiek/logs/phase8_fixedq_attm_sequence.log 2>/dev/null || true
} >"$EVIDENCE"

{
cat <<'PROMPT'
You are the read-only Codex SOL stop adjudicator for the active MeanAudio
Phase-8 Fixed-Q9 vs matched-NoQ sequential experiment. Decide whether the
currently active process should be intentionally interrupted now. Do not edit,
signal, restart, resume, or change any process.

The queue contract is fixedq9 then matched-noq, each continuing the same clean
NoQ it600000 checkpoint to the preregistered it700000 endpoint with optimizer
reset, LR 3e-5, batch 8, seed 14159265, NoMask and single-cap. An isolated AMP
gradient NaN that recovers is not a stop reason. Loss near 0.98 is not a stop
reason. Stale, missing-process, and GPU-idle observations need two current
observations. Authorize stop only for a current verified condition that risks
invalidating the experiment or machine: contract/hash drift, persistent
nonfinite loss/gradients, repeated runtime failure/OOM, or critical disk risk.
If ambiguous, choose escalate with stop_authorized=false. If the process is
already absent, set stop_authorized=false.

Return only the schema-conforming verdict and report the directly observed
phase and iteration.

--- CURRENT EVIDENCE ---
PROMPT
cat "$EVIDENCE"
} | timeout --signal=TERM 300s codex exec --ephemeral --model gpt-5.6-sol \
    --sandbox read-only --cd "$ROOT" --output-schema "$SCHEMA" \
    --output-last-message "$VERDICT" --json - >"$TRANSCRIPT"

python - "$VERDICT" <<'PY'
import json
import sys
from pathlib import Path

payload = json.loads(Path(sys.argv[1]).read_text())
decision = payload.get("decision")
authorized = payload.get("stop_authorized")
if decision not in {"continue", "stop", "escalate"}:
    raise SystemExit("invalid Codex SOL decision")
if (decision == "stop") != (authorized is True):
    raise SystemExit("decision/stop_authorized mismatch")
print(f"Codex SOL verdict: {decision}; stop_authorized={authorized}")
PY
