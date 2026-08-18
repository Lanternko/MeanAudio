#!/usr/bin/env bash
# Durable 3-hour Luna xhigh inspector. Sol high is invoked only for review,
# incidents, exact-resume recommendations, or committed repair proposals.

set -u

ROOT=/home/kojiek/MeanAudio
LUNA_WORKTREE=/home/kojiek/codex-worktrees/luna-phase8-fixedq-attm-loop
STATE=/home/kojiek/logs/phase8_fixedq_attm_luna_loop
LOCK="$STATE/loop.lock"
LUNA_SCHEMA="$ROOT/scripts/luna_phase8_fixedq_attm_report.schema.json"
SOL_SCHEMA="$ROOT/scripts/sol_phase8_fixedq_attm_review.schema.json"
LUNA_PROMPT="$ROOT/docs/experiments/phase8_fixedq_attm_luna_loop_prompt.md"
SOL_PROMPT="$ROOT/docs/experiments/phase8_fixedq_attm_sol_review_prompt.md"
LOOP_LOG="$STATE/loop.log"

mkdir -p "$STATE/runs"
[ -e "$LUNA_WORKTREE/.git" ] || {
    echo "[FAIL] missing isolated Luna worktree: $LUNA_WORKTREE" >&2
    exit 2
}
exec 9>"$LOCK"
if ! flock -n 9; then
    echo "[FAIL] Luna/Sol loop already running" >&2
    exit 3
fi

run_once() {
    local stamp run_dir luna_json luna_trace sol_json sol_trace review_required
    stamp=$(date -u +%Y%m%dT%H%M%SZ)
    run_dir="$STATE/runs/$stamp"
    luna_json="$run_dir/luna_report.json"
    luna_trace="$run_dir/luna_trace.jsonl"
    sol_json="$run_dir/sol_review.json"
    sol_trace="$run_dir/sol_trace.jsonl"
    mkdir -p "$run_dir"

    echo "[LOOP] $stamp Luna xhigh inspection starts" | tee -a "$LOOP_LOG"
    if ! codex exec --ephemeral --model gpt-5.6-luna \
        -c 'model_reasoning_effort="xhigh"' \
        -c 'approval_policy="never"' --sandbox danger-full-access \
        --cd "$LUNA_WORKTREE" --output-schema "$LUNA_SCHEMA" \
        --output-last-message "$luna_json" --json \
        "Read $LUNA_PROMPT completely and execute that inspection contract now." \
        >"$luna_trace" 2>&1; then
        echo "[LOOP] Luna invocation failed; preserving experiment" | tee -a "$LOOP_LOG"
        return 0
    fi

    review_required=$(python - "$luna_json" <<'PY'
import json
import sys
from pathlib import Path

payload = json.loads(Path(sys.argv[1]).read_text())
needs_review = payload["review_required"] or payload["status"] != "healthy"
print("true" if needs_review else "false")
PY
    ) || {
        echo "[LOOP] invalid Luna report; preserving experiment" | tee -a "$LOOP_LOG"
        return 0
    }

    cp "$luna_json" "$STATE/latest_luna_report.json"
    if [ "$review_required" != true ]; then
        echo "[LOOP] Luna reports healthy; Sol not needed" | tee -a "$LOOP_LOG"
        return 0
    fi

    echo "[LOOP] invoking Sol high audit" | tee -a "$LOOP_LOG"
    if ! codex exec --ephemeral --model gpt-5.6-sol \
        -c 'model_reasoning_effort="high"' \
        -c 'approval_policy="never"' --sandbox danger-full-access \
        --cd "$LUNA_WORKTREE" --output-schema "$SOL_SCHEMA" \
        --output-last-message "$sol_json" --json \
        "Read $SOL_PROMPT completely. Audit Luna report $luna_json and all current evidence now." \
        >"$sol_trace" 2>&1; then
        echo "[LOOP] Sol audit failed; no action authorized" | tee -a "$LOOP_LOG"
        return 0
    fi
    cp "$sol_json" "$STATE/latest_sol_review.json"
    python - "$sol_json" <<'PY' | tee -a "$LOOP_LOG"
import json
import sys
from pathlib import Path

payload = json.loads(Path(sys.argv[1]).read_text())
print(
    "[SOL] decision={decision} execution_authorized={authorized} action={action}".format(
        decision=payload["decision"],
        authorized=payload["execution_authorized"],
        action=payload["approved_action"],
    )
)
PY
}

while true; do
    run_once
    # Three hours, split into bounded one-minute waits.
    for _ in $(seq 1 180); do
        sleep 60
    done
done
