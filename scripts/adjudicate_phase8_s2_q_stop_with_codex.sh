#!/usr/bin/env bash
# Read-only Codex SOL second opinion. This script never sends a signal.

set -euo pipefail
ROOT=/home/kojiek/MeanAudio
STATE=/home/kojiek/logs/phase8_s2_q_ablation_monitor
SCHEMA="$ROOT/scripts/phase8_codex_stop_verdict.schema.json"
VERDICT="$STATE/codex_sol_verdict.json"
EVIDENCE="$STATE/codex_sol_evidence.txt"
TRANSCRIPT="$STATE/codex_sol_adjudication.jsonl"
mkdir -p "$STATE"
rm -f "$VERDICT"

if [ "${1:-}" = --dry-run ]; then
    echo "model=gpt-5.6-sol sandbox=read-only verdict=$VERDICT"
    exit 0
fi

{
    echo "captured_at=$(date --iso-8601=seconds)"
    python "$ROOT/scripts/monitor_phase8_s2_q_ablation.py" --once || true
    echo "--- status ---"; cat "$STATE/status.json" 2>/dev/null || true
    echo "--- alert ---"; cat "$STATE/ALERT.json" 2>/dev/null || true
    echo "--- processes ---"; pgrep -af 'phase8_catalog_matched_s2_(realq|shuffledq)' || true
    echo "--- tmux ---"; tmux ls 2>&1 || true
    echo "--- gpu ---"; nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu --format=csv,noheader,nounits 2>&1 || true
    echo "--- disk ---"; df -h / 2>&1 || true
    echo "--- real tail ---"; tail -n 120 /home/kojiek/logs/phase8_catalog_matched_s2_realq_stage2_200000.log 2>/dev/null || true
    echo "--- shuffled tail ---"; tail -n 120 /home/kojiek/logs/phase8_catalog_matched_s2_shuffledq_stage2_200000.log 2>/dev/null || true
} >"$EVIDENCE"

{
cat <<'PROMPT'
You are the read-only Codex SOL stop adjudicator for the active MeanAudio
Phase-8 S2-only Q ablation sequence. Do not edit, signal, restart, or alter
anything. Decide whether the currently active process should be interrupted.

Contract: both arms reuse the same completed S1 NoQ checkpoint. Real-Q trains
S2 with the original per-row q_level; Shuffled-Q trains S2 with a fixed-seed
permutation of q_level only. Both are NoMask, single-cap, 400k→600k, and eval
MusicCaps at pre-registered q9 then q6. Real-Q completes before Shuffled-Q.

An isolated grad_norm NaN/Inf that subsequently recovers is an AMP skipped
step, not a stop reason. Require persistent/dense gradients (trailing >=2,
recent20 >=3, or recent100 >=10) or independent corruption. Loss near 1 is not
a failure. Stale/process/GPU incidents require a repeated observation. Only
authorize stop for a current verified contract drift, persistent non-finite
state, repeated OOM/runtime failure, critical disk risk, or equivalent danger.
If ambiguous, escalate with stop_authorized=false. If no process exists, there
is nothing to signal and stop_authorized must be false.

Return only the schema-conforming verdict.
--- CURRENT EVIDENCE ---
PROMPT
cat "$EVIDENCE"
} | timeout --signal=TERM 300s codex exec --ephemeral --model gpt-5.6-sol \
    --sandbox read-only --cd "$ROOT" --output-schema "$SCHEMA" \
    --output-last-message "$VERDICT" --json - >"$TRANSCRIPT"

python - "$VERDICT" <<'PY'
import json, sys
from pathlib import Path
p = json.loads(Path(sys.argv[1]).read_text())
if p.get("decision") == "stop" and p.get("stop_authorized") is not True:
    raise SystemExit("invalid stop verdict")
if p.get("decision") != "stop" and p.get("stop_authorized") is not False:
    raise SystemExit("invalid non-stop verdict")
print(f"Codex SOL verdict: {p['decision']}; stop_authorized={p['stop_authorized']}")
PY
