#!/usr/bin/env bash
# Read-only second opinion before anyone is allowed to stop Phase8 clean-NoQ.
# This script NEVER sends signals to training.  It only writes a structured
# Codex SOL verdict for the supervising Grok loop to consume.

set -euo pipefail

ROOT=/home/kojiek/MeanAudio
STATE_DIR=/home/kojiek/logs/phase8_catalog_matched_noq_monitor
SCHEMA="$ROOT/scripts/phase8_codex_stop_verdict.schema.json"
VERDICT="$STATE_DIR/codex_sol_verdict.json"
TRANSCRIPT="$STATE_DIR/codex_sol_adjudication.jsonl"
EVIDENCE="$STATE_DIR/codex_sol_evidence.txt"

mkdir -p "$STATE_DIR"
rm -f "$VERDICT"

if [ "${1:-}" = "--dry-run" ]; then
    echo "model=gpt-5.6-sol sandbox=read-only verdict=$VERDICT"
    exit 0
fi

{
    echo "captured_at=$(date --iso-8601=seconds)"
    "$ROOT/scripts/monitor_phase8_clean_noq.py" --once || true
    echo "--- status.json ---"
    cat "$STATE_DIR/status.json" 2>/dev/null || true
    echo "--- ALERT.json ---"
    cat "$STATE_DIR/ALERT.json" 2>/dev/null || true
    echo "--- contract audit ---"
    cat "$STATE_DIR/contract_audit.json" 2>/dev/null || true
    echo "--- training processes ---"
    pgrep -af "train.py.*phase8_catalog_matched_noq|torchrun.*phase8_catalog_matched_noq" || true
    echo "--- tmux ---"
    tmux ls 2>&1 || true
    echo "--- GPU ---"
    nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu \
        --format=csv,noheader,nounits 2>&1 || true
    echo "--- disk ---"
    df -h / 2>&1 || true
    echo "--- stage log tails ---"
    tail -n 120 /home/kojiek/logs/phase8_catalog_matched_noq_stage1_400000.log 2>/dev/null || true
    tail -n 120 /home/kojiek/logs/phase8_catalog_matched_noq_stage2_200000.log 2>/dev/null || true
} >"$EVIDENCE"

{
cat <<'PROMPT'
You are the read-only Codex SOL stop adjudicator for the currently running
MeanAudio Phase-8 catalog-matched clean-NoQ experiment.  Determine whether the
active training process should be intentionally interrupted now.  Do not edit
files, send signals, restart anything, or change tmux/process state.

The wrapper captured current evidence immediately before this review and
appended it below.  Treat that snapshot as primary; you may inspect more if the
read-only sandbox permits it.

Experiment contract: S1 and S2 use_q_conditioning=false; both stages NoMask and
multi_cap=false; eval no_q=True and no_text_attention_mask=True.  Do not alter
this contract.

Important policy:
- An isolated grad_norm NaN/Inf under AMP GradScaler, followed by finite loss
  and finite gradients, is a skipped optimizer update and is NOT a reason to
  stop.  Require persistent/dense non-finite gradients (trailing >=2,
  recent20 >=3, or recent100 >=10) or independent evidence of corruption.
- A loss plateau near 0.98-1.00 is expected and is not a reason to stop.
- Stale/process/GPU observations must be reproduced, not inferred from one
  old alert.
- Authorize STOP only for a current, verified condition where continuing risks
  invalidating the experiment or machine: contract drift, persistent
  non-finite loss/gradients, repeated OOM/runtime failure, critical disk risk,
  or equivalent strong evidence.
- If evidence is ambiguous or you cannot inspect it, choose ESCALATE and set
  stop_authorized=false.  If the process is already absent, there is nothing
  to signal: set stop_authorized=false.

Return only the required structured verdict, including the phase and iteration
you directly observed.  decision=stop is valid only when stop_authorized=true
and the evidence supports interrupting the active process now.  Otherwise
choose continue or escalate.

--- WRAPPER-CAPTURED CURRENT EVIDENCE ---
PROMPT
cat "$EVIDENCE"
} | timeout --signal=TERM 300s codex exec \
    --ephemeral \
    --model gpt-5.6-sol \
    --sandbox read-only \
    --cd "$ROOT" \
    --output-schema "$SCHEMA" \
    --output-last-message "$VERDICT" \
    --json - >"$TRANSCRIPT"

python - "$VERDICT" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
payload = json.loads(path.read_text())
decision = payload.get("decision")
authorized = payload.get("stop_authorized")
if decision not in {"continue", "stop", "escalate"}:
    raise SystemExit("invalid Codex SOL decision")
if decision == "stop" and authorized is not True:
    raise SystemExit("invalid Codex SOL verdict: stop without authorization")
if decision != "stop" and authorized is not False:
    raise SystemExit("invalid Codex SOL verdict: non-stop cannot authorize stop")
if not isinstance(payload.get("observed_phase"), (str, type(None))):
    raise SystemExit("invalid Codex SOL observed_phase")
if not isinstance(payload.get("observed_iteration"), (int, type(None))):
    raise SystemExit("invalid Codex SOL observed_iteration")
print(f"Codex SOL verdict: {decision}; stop_authorized={authorized}")
PY
