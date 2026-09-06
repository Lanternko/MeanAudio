#!/usr/bin/env bash
set -uo pipefail

# Durable three-hour AI watcher.  Luna diagnoses and writes proposal-only
# reports in the isolated worktree.  A failed deterministic monitor also asks
# Sol for adjudication; neither model may stop/change/relaunch the queue.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
AI_WORKTREE="${PHASE8_QWEN_AI_WORKTREE:-/home/kojiek/codex-worktrees/luna-phase8-qwen-probe}"
STATE_DIR="${PHASE8_QWEN_MONITOR_STATE_DIR:-/home/kojiek/logs/phase8_qwen_official_matched_monitor}"
REPORT_DIR="$STATE_DIR/ai_reports"
INTERVAL="${PHASE8_QWEN_LUNA_INTERVAL_SECONDS:-900}"
DURATION="${PHASE8_QWEN_LUNA_DURATION_SECONDS:-10800}"
CODEX_BIN="${CODEX_BIN:-/home/kojiek/.local/bin/codex}"

mkdir -p "$STATE_DIR" "$REPORT_DIR" "$AI_WORKTREE/proposals"
deadline=$(( $(date +%s) + DURATION ))
cycle=0

while [ "$(date +%s)" -lt "$deadline" ]; do
  cycle=$((cycle + 1))
  stamp="$(date -u +%Y%m%dT%H%M%SZ)"
  monitor_rc=0
  /home/kojiek/venvs/dac/bin/python "$ROOT/scripts/phase8_qwen_monitor.py" \
    --once --expect-active --state-dir "$STATE_DIR" --repo-root "$AI_WORKTREE" \
    --contract "$ROOT/docs/experiments/phase8_qwen_official_matched_contract.json" \
    >"$REPORT_DIR/deterministic_${stamp}.log" 2>&1 || monitor_rc=$?

  timeout 12m "$CODEX_BIN" exec --ephemeral --model gpt-5.6-luna \
    -c 'model_reasoning_effort="xhigh"' -c 'approval_policy="never"' \
    --sandbox danger-full-access --cd "$AI_WORKTREE" \
    --output-last-message "$REPORT_DIR/luna_${stamp}.md" \
    - < "$ROOT/docs/experiments/phase8_qwen_luna_3h_prompt.md" \
    >"$REPORT_DIR/luna_${stamp}.log" 2>&1 || true

  if [ "$monitor_rc" -ne 0 ]; then
    timeout 12m "$CODEX_BIN" exec --ephemeral --model gpt-5.6-sol \
      -c 'model_reasoning_effort="high"' -c 'approval_policy="never"' \
      --sandbox danger-full-access --cd "$AI_WORKTREE" \
      --output-schema "$ROOT/scripts/phase8_qwen_sol_verdict.schema.json" \
      --output-last-message "$REPORT_DIR/sol_${stamp}.md" \
      - < "$ROOT/docs/experiments/phase8_qwen_sol_adjudication_prompt.md" \
      >"$REPORT_DIR/sol_${stamp}.log" 2>&1 || true
  fi

  now_epoch="$(date +%s)"
  remaining=$((deadline - now_epoch))
  [ "$remaining" -le 0 ] && break
  wait_for="$INTERVAL"
  [ "$remaining" -lt "$wait_for" ] && wait_for="$remaining"
  sleep "$wait_for"
done

printf '[COMPLETE] phase8 Qwen Luna watcher cycles=%s\n' "$cycle"
