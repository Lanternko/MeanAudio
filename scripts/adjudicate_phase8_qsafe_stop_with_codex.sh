#!/usr/bin/env bash
# Read-only Codex SOL stop adjudication for Q-safe fine-tuning. Never signals.
set -euo pipefail
ROOT=/home/kojiek/MeanAudio
STATE=/home/kojiek/logs/phase8_qsafe_ft_monitor
SCHEMA="$ROOT/scripts/phase8_codex_stop_verdict.schema.json"
VERDICT="$STATE/codex_sol_verdict.json"
EVIDENCE="$STATE/codex_sol_evidence.txt"
TRANSCRIPT="$STATE/codex_sol_adjudication.jsonl"
mkdir -p "$STATE"; rm -f "$VERDICT"
if [ "${1:-}" = --dry-run ]; then echo "model=gpt-5.6-sol sandbox=read-only verdict=$VERDICT"; exit 0; fi
{
  echo "captured_at=$(date --iso-8601=seconds)"
  python "$ROOT/scripts/monitor_phase8_qsafe_ft.py" --once || true
  echo '--- status ---'; cat "$STATE/status.json" 2>/dev/null || true
  echo '--- alert ---'; cat "$STATE/ALERT.json" 2>/dev/null || true
  echo '--- processes ---'; pgrep -af 'phase8_qsafe_(realq|shuffledq)_ft100k' || true
  echo '--- tmux ---'; tmux ls 2>&1 || true
  echo '--- gpu/disk ---'; nvidia-smi 2>&1 || true; df -h / /mnt/HDD 2>&1 || true
  echo '--- sequence tail ---'; tail -n 160 /home/kojiek/logs/phase8_qsafe_ft_sequence.log 2>/dev/null || true
} >"$EVIDENCE"
{
cat <<'PROMPT'
You are the read-only Codex SOL stop adjudicator for the Phase-8 Q-safe
fine-tuning sequence. Do not edit, signal, restart, or change anything.

Contract: two sequential 100k MeanAudio continuations start from the same
completed NoQ S2 checkpoint at it=600k. In every online and EMA q_embed,
q0..9 are exact copies of trained q10 at initialization. Real-Q runs first;
fixed-seed Shuffled-Q runs second. Both use LR 3e-5 and finish at it=700k.

Isolated recovered AMP grad NaN/Inf is not a stop reason. Require current,
verified contract drift, persistent/dense nonfinite state, repeated OOM or
runtime failure, critical disk risk, or equivalent danger. A monitor nonzero
exit alone never authorizes stop. If uncertain or no active training exists,
return stop_authorized=false. Return only schema-conforming JSON.
--- EVIDENCE ---
PROMPT
cat "$EVIDENCE"
} | timeout --signal=TERM 300s codex exec --ephemeral --model gpt-5.6-sol \
  --sandbox read-only --cd "$ROOT" --output-schema "$SCHEMA" \
  --output-last-message "$VERDICT" --json - >"$TRANSCRIPT"
python - "$VERDICT" <<'PY'
import json,sys
from pathlib import Path
p=json.loads(Path(sys.argv[1]).read_text())
if (p.get('decision')=='stop') != (p.get('stop_authorized') is True): raise SystemExit('[FAIL] inconsistent verdict')
print(f"Codex SOL: {p['decision']}; stop_authorized={p['stop_authorized']}")
PY
