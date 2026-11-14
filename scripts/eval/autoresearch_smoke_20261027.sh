#!/bin/bash
# Timing smoke for the autoresearch inference-search spec (2026-10-27).
# NOT the search driver: it measures one trial's per-clip cost on the canonical
# path (eval.py via mc_mf25_eval.sh -> lvl30 rescore -> CLAP batch 1) for a
# 64-clip prefix (CFG0) and a 16-clip prefix (CFG3+neg, two forward passes/step).
# Output: runtime/autoresearch_smoke_20261027/{cells,timing.json,smoke.log}
set -eo pipefail
PY=/home/kojiek/venvs/dac/bin/python
ROOT=/home/kojiek/MeanAudio
OUT=$ROOT/runtime/autoresearch_smoke_20261027
LOG=$OUT/smoke.log
mkdir -p "$OUT"
exec > >(tee -a "$LOG") 2>&1
cd "$ROOT"
source "$ROOT/scripts/notify_lib.sh"
exec 9< <("$PY" "$HOME/gpu_queue/hold_lock.py" "$HOME/gpu_queue/gpu0.lock" --block --watch-pid $$)
read -r LOCKSTATE <&9
[ "$LOCKSTATE" = LOCKED ] || { echo "[FAIL] gpu0.lock: $LOCKSTATE"; exit 5; }
notify_on_exit "autoresearch_smoke_20261027" "$LOG"
export CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1

EXP=phase8_qwen_caption2p0_slot0clean_nmv2matched_noq_full_s14159265
CKPT=$HOME/exps_nvme/${EXP}_stage2_200000/${EXP}_stage2_200000_ema_final.pth
MC_TSV=/mnt/HDD/kojiek/phase4_jamendo_data/musiccaps_test.tsv
[ -f "$CKPT" ] || { echo "[FAIL] missing $CKPT"; exit 2; }

# Record-aware first-N rows (the canonical TSV has 5 captions whose quoted text
# spans a physical line break; head -n N+1 would be off by one past row 461).
mkdir -p "$OUT/tsv"
"$PY" - "$MC_TSV" "$OUT/tsv" <<'PY'
import csv, sys
from pathlib import Path
src, out = sys.argv[1], Path(sys.argv[2])
rows = list(csv.DictReader(open(src, encoding='utf-8', newline=''), delimiter='\t'))
for n in (16, 64):
    with open(out / f'mc_first{n}.tsv', 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]), delimiter='\t')
        w.writeheader(); w.writerows(rows[:n])
    print(f'[smoke] wrote {out / f"mc_first{n}.tsv"} rows={len(rows[:n])}')
PY

CELLS=$OUT/cells; mkdir -p "$CELLS"
now(){ date +%s.%N; }
secs(){ "$PY" -c "print(f'{$2-$1:.1f}')"; }

# ── 1) CFG0, 64 clips: generation + CLAP + AES + level ────────────────────
T0=$(now)
OUT_ROOT=$CELLS SMOKE_ROWS=64 bash scripts/eval/mc_mf25_eval.sh autoresearch_smoke "$CKPT" --no_q cfg0
T1=$(now); CFG0_64=$(secs "$T0" "$T1"); echo "[smoke] cfg0 64 clips gen+all-metrics = ${CFG0_64}s"
CELL=$CELLS/autoresearch_smoke_mc_mf25_cfg0_smoke64

# ── 2) CLAP-only on the same 64 raw clips (isolates CLAP cost) ────────────
T0=$(now)
"$PY" scripts/eval/eval_metrics.py --gen_dir "$CELL/audio" --tsv "$OUT/tsv/mc_first64.tsv" \
  --exp_name smoke_clap_raw --out_dir "$OUT/clap_raw" --skip_aes --skip_level
T1=$(now); CLAP_64=$(secs "$T0" "$T1"); echo "[smoke] CLAP batch1 64 clips = ${CLAP_64}s"

# ── 3) lvl30 rescore (CPU) + its standard metrics ─────────────────────────
T0=$(now)
"$PY" scripts/eval/level_match_rescore.py --cell_dir "$CELL" --tsv "$OUT/tsv/mc_first64.tsv" \
  --target -30 --workers 8 --out_root "$OUT/lvl30"
T1=$(now); LVL30_64=$(secs "$T0" "$T1"); echo "[smoke] lvl30 rescore 64 clips (+metrics) = ${LVL30_64}s"

# ── 4) CFG3+neg, 16 clips (two forwards per step) ─────────────────────────
T0=$(now)
OUT_ROOT=$CELLS SMOKE_ROWS=16 bash scripts/eval/mc_mf25_eval.sh autoresearch_smoke "$CKPT" --no_q cfg3neg
T1=$(now); CFG3_16=$(secs "$T0" "$T1"); echo "[smoke] cfg3+neg 16 clips gen+all-metrics = ${CFG3_16}s"

# Generation-only split for each cell from the wrapper's ISO timestamps.
"$PY" - "$OUT" "$CFG0_64" "$CLAP_64" "$LVL30_64" "$CFG3_16" <<'PY'
import datetime as dt, json, re, sys
from pathlib import Path
out = Path(sys.argv[1])
vals = dict(cfg0_64_total=float(sys.argv[2]), clap64=float(sys.argv[3]),
            lvl30_64=float(sys.argv[4]), cfg3_16_total=float(sys.argv[5]))
stamp = re.compile(r'^\[(\S+)\] \[mc_mf25 (\S+)\] (generate cfg=.*|generated \d+ clips)')
seen = {}
for line in (out / 'smoke.log').read_text().splitlines():
    m = stamp.match(line)
    if not m:
        continue
    ts = dt.datetime.fromisoformat(m.group(1).replace('Z', '+00:00')).timestamp()
    label, what = m.group(2), m.group(3)
    key = (label, 'start' if what.startswith('generate') else 'end')
    seen[key] = ts
for name, label in (('cfg0', 'autoresearch_smoke_mc_mf25_cfg0_smoke64'),
                    ('cfg3neg', 'autoresearch_smoke_mc_mf25_cfg3_neg_smoke16')):
    if (label, 'start') in seen and (label, 'end') in seen:
        vals[f'{name}_gen_s'] = round(seen[(label, 'end')] - seen[(label, 'start')], 1)
        vals[f'{name}_clips_logged'] = int(re.search(r'generated (\d+)', next(
            l for l in (out / 'smoke.log').read_text().splitlines() if label in l and 'generated' in l)).group(1))
cell0 = out / 'cells' / 'autoresearch_smoke_mc_mf25_cfg0_smoke64'
vals['disk_raw_cfg0_64_bytes'] = sum(p.stat().st_size for p in (cell0 / 'audio').glob('*.flac'))
lvl_audio = out / 'lvl30' / 'autoresearch_smoke_mc_mf25_cfg0_smoke64_lvl30' / 'audio'
vals['disk_lvl30_64_bytes'] = sum(p.stat().st_size for p in lvl_audio.glob('*.flac'))
(out / 'timing.json').write_text(json.dumps(vals, indent=1))
print(json.dumps(vals, indent=1))
PY
echo "[smoke] DONE"
