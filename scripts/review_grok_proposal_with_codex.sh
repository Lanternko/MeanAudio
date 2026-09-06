#!/usr/bin/env bash
# Review a Grok-authored branch/proposal with Codex SOL before execution.
# The reviewer is read-only.  Grok may execute only an exact approved command
# from a verdict tied to the still-current commit.

set -euo pipefail

if [ "$#" -ne 2 ]; then
    echo "usage: $0 /home/kojiek/grok-worktrees/<name> <proposal-file>" >&2
    exit 2
fi

ROOT=/home/kojiek/MeanAudio
WORKTREE=$(readlink -f "$1")
PROPOSAL=$(readlink -f "$2")
STATE_DIR=/home/kojiek/logs/grok_codex_reviews
SCHEMA="$ROOT/scripts/grok_codex_change_review.schema.json"
VERDICT="$STATE_DIR/latest_verdict.json"
TRANSCRIPT="$STATE_DIR/latest_review.jsonl"
EVIDENCE="$STATE_DIR/latest_evidence.txt"
DIFF_FILE="$STATE_DIR/latest_committed.diff"

case "$WORKTREE" in
    /home/kojiek/grok-worktrees/*) ;;
    *) echo "[FAIL] Grok proposals must use a separate worktree under /home/kojiek/grok-worktrees" >&2; exit 2 ;;
esac
case "$PROPOSAL" in
    "$WORKTREE"/*) ;;
    *) echo "[FAIL] proposal file must be inside the Grok worktree" >&2; exit 2 ;;
esac
[ -f "$PROPOSAL" ] || { echo "[FAIL] proposal not found: $PROPOSAL" >&2; exit 2; }

BRANCH=$(git -C "$WORKTREE" branch --show-current)
case "$BRANCH" in
    grok/*) ;;
    *) echo "[FAIL] proposal branch must be named grok/* (got $BRANCH)" >&2; exit 2 ;;
esac
if [ -n "$(git -C "$WORKTREE" status --porcelain)" ]; then
    echo "[FAIL] commit all proposal changes before Codex review" >&2
    exit 2
fi

COMMIT=$(git -C "$WORKTREE" rev-parse HEAD)
BASE=$(git -C "$WORKTREE" merge-base HEAD main)
mkdir -p "$STATE_DIR"
rm -f "$VERDICT"
git -C "$WORKTREE" diff "$BASE..$COMMIT" >"$DIFF_FILE"

{
    echo "reviewed_at=$(date --iso-8601=seconds)"
    echo "worktree=$WORKTREE"
    echo "branch=$BRANCH"
    echo "commit=$COMMIT"
    echo "base=$BASE"
    echo "--- proposal ---"
    sed -n '1,1200p' "$PROPOSAL"
    echo "--- diff stat ---"
    git -C "$WORKTREE" diff --stat "$BASE..$COMMIT"
    echo "--- committed diff (capped at 400 KiB; total bytes=$(stat -c %s "$DIFF_FILE")) ---"
    head -c 409600 "$DIFF_FILE"
} >"$EVIDENCE"

{
cat <<PROMPT
You are Codex SOL, the senior reviewer for a cheaper Grok scout/implementer.
Review the appended proposal and committed diff.  Do not edit, merge, launch,
stop, or mutate anything.  Your verdict is bound to commit $COMMIT.

Approve only if all of the following hold:
- the diagnosis and code change are technically sound and tested;
- a new experiment has a falsifiable objective, identifiable controlled
  variable, fixed baseline, exact train/eval commands, metrics, gates, resource
  budget, unique artifact prefix, and rollback/stop policy;
- it does not change, stop, contaminate, or compete with the active
  phase8_catalog_matched_noq run;
- it preserves Q/train/eval semantics and data provenance;
- the proposed execution command is exact, bounded, and safe.

Set execution_authorized=true only with decision=approve.  In that case return
exactly one approved_command copied or safely corrected from the proposal.
For revise/reject, execution_authorized=false and approved_command=null.  If
the diff is truncated or evidence is insufficient, choose revise.

--- GROK PROPOSAL EVIDENCE ---
PROMPT
cat "$EVIDENCE"
} | timeout --signal=TERM 600s codex exec \
    --ephemeral \
    --model gpt-5.6-sol \
    --sandbox read-only \
    --cd "$WORKTREE" \
    --output-schema "$SCHEMA" \
    --output-last-message "$VERDICT" \
    --json - >"$TRANSCRIPT"

python - "$VERDICT" "$COMMIT" <<'PY'
import json
import sys
from pathlib import Path

path, commit = Path(sys.argv[1]), sys.argv[2]
payload = json.loads(path.read_text())
if payload.get("reviewed_commit") != commit:
    raise SystemExit("[FAIL] Codex verdict does not match the reviewed commit")
decision = payload.get("decision")
authorized = payload.get("execution_authorized")
command = payload.get("approved_command")
if decision == "approve":
    if authorized is not True or not isinstance(command, str) or not command.strip():
        raise SystemExit("[FAIL] approve verdict lacks execution authorization/command")
else:
    if authorized is not False or command is not None:
        raise SystemExit("[FAIL] non-approve verdict cannot authorize execution")
print(f"Codex SOL change verdict: {decision}; commit={commit}; authorized={authorized}")
PY
