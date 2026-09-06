#!/usr/bin/env bash
# Restore the shared caption10s NPZ cache to its R-Matched binding.
#
# sequence_rich_shared_then_matched_full.sh rebinds
# /mnt/HDD/kojiek/phase8_qwen_official_matched_npz to R-Shared captions before
# the quarter control, and only rebinds it back when the promotion gate passes.
# On a non-promotion (or on a crash) the cache is left holding 214,614
# wrong-caption text features with nothing recording that fact.  Run this
# afterwards to put the cache back and leave an auditable state file.
#
# Usage:
#   bash scripts/restore_matched_binding_after_rich_shared.sh [--check-only]
set -euo pipefail

ROOT=/home/kojiek/MeanAudio
DATA=/mnt/HDD/kojiek/phase4_jamendo_data
PIPE=/home/kojiek/research/meanaudio_training/caption10s_pipeline
STATE=/home/kojiek/logs/rich_shared_then_matched_full
NPZ_DIR=/mnt/HDD/kojiek/phase8_qwen_official_matched_npz
CACHE_LIST="$DATA/phase8_qwen_official_matched_npz_cache_train.txt"
MATCHED_TSV="$DATA/phase8_qwen_caption10s_multisent_train.tsv"
SHARED_TSV="$DATA/caption_alignment_rich_shared_train.tsv"
REEXTRACT="$PIPE/reextract_text_inplace_caption10s.py"
AUDITOR="$ROOT/scripts/preprocess/audit_caption_npz_binding.py"
DETECTOR="$ROOT/scripts/preprocess/detect_caption_npz_binding.py"
BINDING_STATE="$STATE/cache_binding.json"
RESTORE_AUDIT="$STATE/restore_matched_npz_binding_audit.json"
EXPECTED_ROWS=251599

CHECK_ONLY=false
if [ "${1:-}" = "--check-only" ]; then CHECK_ONLY=true; shift; fi
[ "$#" -eq 0 ] || { echo "usage: $0 [--check-only]" >&2; exit 2; }

ts() { date --iso-8601=seconds; }
log() { echo "[$(ts)] $*"; }

# Fail closed while the owning sequence still holds the cache.  flock -n on the
# sequence's own lock file is the interlock: if we can take it, nobody is
# mutating the cache underneath us.
exec 8>"$STATE/sequence.lock"
if ! flock -n 8; then
    echo "[FAIL] rich_shared sequence is still running; it owns the NPZ cache." >&2
    echo "       Wait for it to finish, then re-run this script." >&2
    exit 3
fi

source /home/kojiek/venvs/dac/bin/activate
export CUDA_VISIBLE_DEVICES=0
cd "$ROOT"

for path in "$MATCHED_TSV" "$SHARED_TSV" "$CACHE_LIST" "$NPZ_DIR" \
            "$REEXTRACT" "$AUDITOR" "$DETECTOR"; do
    [ -e "$path" ] || { echo "[FAIL] missing required input: $path" >&2; exit 2; }
done

detect() {
    python "$DETECTOR" \
        --candidate "r_matched=$MATCHED_TSV" \
        --candidate "r_shared=$SHARED_TSV" \
        --cache-list "$CACHE_LIST" --npz-dir "$NPZ_DIR" \
        --samples 128 --report "$BINDING_STATE" >/dev/null
    python -c "import json,sys; print(json.load(open(sys.argv[1]))['binding'])" "$BINDING_STATE"
}

log "[DETECT] sampling current NPZ caption binding"
binding=$(detect || true)
[ -n "$binding" ] || binding=unknown
log "[DETECT] binding=$binding (state: $BINDING_STATE)"

if [ "$binding" = "r_matched" ]; then
    log "[OK] cache already bound to R-Matched; nothing to restore"
    exit 0
fi

if [ "$CHECK_ONLY" = true ]; then
    log "[CHECK-ONLY] cache is '$binding'; re-run without --check-only to restore"
    exit 1
fi

# 'unknown' means a partially rebound cache (e.g. the sequence was killed
# mid-rebind).  The re-extractor is per-row idempotent, so restoring is the
# correct action for both 'r_shared' and 'unknown'.
log "[RESTORE] rebinding cache to R-Matched captions"
python "$REEXTRACT" \
    --train_tsv "$MATCHED_TSV" --cache_list "$CACHE_LIST" --npz_dir "$NPZ_DIR" \
    --batch_size 32 \
    --progress_json "$STATE/restore_matched_reextract_progress.json" \
    --done_json "$STATE/restore_matched_reextract_done.json"

log "[AUDIT] full-corpus binding audit"
python "$AUDITOR" --tsv "$MATCHED_TSV" --cache-list "$CACHE_LIST" \
    --npz-dir "$NPZ_DIR" --report "$RESTORE_AUDIT" --expected-rows "$EXPECTED_ROWS"

binding=$(detect || true)
[ "$binding" = "r_matched" ] || {
    echo "[FAIL] cache still reports binding='$binding' after restore" >&2
    exit 2
}
log "[DONE] cache restored to R-Matched (audit: $RESTORE_AUDIT)"
