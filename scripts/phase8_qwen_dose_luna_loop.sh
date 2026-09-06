#!/usr/bin/env bash
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
AI_WORKTREE="${PHASE8_QWEN_AI_WORKTREE:-/home/kojiek/codex-worktrees/luna-phase8-qwen-probe}"
STATE="${PHASE8_QWEN_DOSE_STATE:-/home/kojiek/logs/phase8_qwen_dose_monitor}"
REPORTS="$STATE/ai_reports"
INTERVAL="${PHASE8_QWEN_DOSE_INTERVAL_SECONDS:-900}"
DURATION="${PHASE8_QWEN_DOSE_DURATION_SECONDS:-86400}"
CODEX_BIN="${CODEX_BIN:-/home/kojiek/.local/bin/codex}"
mkdir -p "$REPORTS" "$AI_WORKTREE/proposals"
deadline=$(( $(date +%s) + DURATION ))
cycle=0
while [ "$(date +%s)" -lt "$deadline" ]; do
  cycle=$((cycle + 1))
  stamp="$(date -u +%Y%m%dT%H%M%SZ)"
  rc=0
  /home/kojiek/venvs/dac/bin/python "$ROOT/scripts/phase8_qwen_dose_monitor.py" \
    --once --expect-active --state-dir "$STATE" --repo-root "$AI_WORKTREE" \
    --contract "$ROOT/docs/experiments/phase8_qwen_dose_contract.json" \
    >"$REPORTS/deterministic_${stamp}.log" 2>&1 || rc=$?
  if [ $(( (cycle - 1) % 4 )) -eq 0 ]; then
    timeout 12m "$CODEX_BIN" exec --ephemeral --model gpt-5.6-luna \
      -c 'model_reasoning_effort="xhigh"' -c 'approval_policy="never"' \
      --sandbox danger-full-access --cd "$AI_WORKTREE" \
      --output-last-message "$REPORTS/luna_${stamp}.md" \
      - < "$ROOT/docs/experiments/phase8_qwen_dose_luna_prompt.md" \
      >"$REPORTS/luna_${stamp}.log" 2>&1 || true
  fi
  if [ "$rc" -ne 0 ]; then
    timeout 12m "$CODEX_BIN" exec --ephemeral --model gpt-5.6-sol \
      -c 'model_reasoning_effort="high"' -c 'approval_policy="never"' \
      --sandbox danger-full-access --cd "$AI_WORKTREE" \
      --output-schema "$ROOT/scripts/phase8_qwen_sol_verdict.schema.json" \
      --output-last-message "$REPORTS/sol_${stamp}.json" \
      - < "$ROOT/docs/experiments/phase8_qwen_dose_sol_incident_prompt.md" \
      >"$REPORTS/sol_${stamp}.log" 2>&1 || true
  fi
  [ "$(date +%s)" -ge "$deadline" ] && break
  sleep "$INTERVAL"
done
printf '[COMPLETE] phase8 Qwen dose watcher cycles=%s\n' "$cycle"
