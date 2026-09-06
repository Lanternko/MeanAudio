#!/usr/bin/env bash
# Full-scale, official-Qwen-caption NoQ baseline.
#
# This is intentionally a new experiment namespace: it must not reuse the
# completed legacy-catalog NoQ run, because the caption/cache provenance here
# is the same official-Qwen mapping used by the current bucket ablations.

set -euo pipefail

ROOT=/home/kojiek/MeanAudio
DATA=/mnt/HDD/kojiek/phase4_jamendo_data
LOG_ROOT=/home/kojiek/logs
PREFIX=phase8_qwen_official_noq_full
TRAIN_TSV="$DATA/phase8_qwen_meansim_k2_balanced.tsv"
NPZ_DIR=/mnt/HDD/kojiek/phase8_qwen_official_matched_npz
GT_CACHE="$DATA/phase8_qwen_official_matched_npz_cache_train.txt"
GRID_MANIFEST="$DATA/phase8_qwen_meansim_bucket_grid.manifest.json"
NPZ_MANIFEST="$DATA/phase8_qwen_official_matched_npz_manifest.json"
CACHE_AUDIT="$DATA/phase8_qwen_official_matched_qwen_cache_audit.json"
MUSICCAPS="$DATA/musiccaps_test.tsv"
EVALUATOR=/home/kojiek/research/meanaudio_eval/phase4_eval.py
S1_UPDATES=400000
S2_UPDATES=200000
FINAL_IT=$((S1_UPDATES + S2_UPDATES))
EXPECTED_ROWS=251599
SEED=14159265
RUN_MODE="${EXPERIMENT_RUN_MODE:-fresh}"
PREFLIGHT_ONLY="${PREFLIGHT_ONLY:-false}"
CONTRACT="$LOG_ROOT/${PREFIX}_official_contract.json"
FINAL_AUDIT="$LOG_ROOT/${PREFIX}_FINAL_TRAIN_AUDIT.json"
FINAL_REPORT="$LOG_ROOT/${PREFIX}_FINAL_METRICS.json"
S1_EXP="${PREFIX}_stage1_${S1_UPDATES}"
S2_EXP="${PREFIX}_stage2_${S2_UPDATES}"
S2_EMA="$ROOT/exps/$S2_EXP/${S2_EXP}_ema_final.pth"
METRICS="$ROOT/eval_output/metrics/${S2_EXP}_musiccaps/metrics.txt"

case "$RUN_MODE" in fresh|resume) ;; *) echo "[FAIL] invalid EXPERIMENT_RUN_MODE=$RUN_MODE" >&2; exit 2;; esac
case "$PREFLIGHT_ONLY" in true|false) ;; *) echo "[FAIL] PREFLIGHT_ONLY must be true or false" >&2; exit 2;; esac

for path in "$TRAIN_TSV" "$NPZ_DIR" "$GT_CACHE" "$GRID_MANIFEST" "$NPZ_MANIFEST" \
            "$CACHE_AUDIT" "$MUSICCAPS" "$EVALUATOR"; do
    [ -e "$path" ] || { echo "[FAIL] missing required input: $path" >&2; exit 2; }
done

cd "$ROOT"
source /home/kojiek/venvs/dac/bin/activate
source "$ROOT/scripts/runtime/phase8_nvidia_compat_env.sh"
phase8_nvidia_compat_apply || { echo "[FAIL] NVIDIA preflight: $PHASE8_NVIDIA_COMPAT_ERROR" >&2; exit 2; }

# Bind this baseline to the exact caption ordering/cache audited for K=2.
python - "$TRAIN_TSV" "$GRID_MANIFEST" "$GT_CACHE" "$NPZ_DIR" "$NPZ_MANIFEST" "$CACHE_AUDIT" <<'PY'
import csv, hashlib, json, sys
from collections import Counter
from pathlib import Path

tsv, grid_file, cache, npz_dir, npz_manifest_file, cache_audit_file = map(Path, sys.argv[1:])
def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()
grid = json.loads(grid_file.read_text())
arm = grid.get('outputs', {}).get('k2_balanced', {})
rows = list(csv.DictReader(tsv.open(newline=''), delimiter='\t'))
names = [line.strip() for line in cache.read_text().splitlines() if line.strip()]
npz_manifest = json.loads(npz_manifest_file.read_text())
cache_audit = json.loads(cache_audit_file.read_text())
hist = Counter(int(row['q_level']) for row in rows)
expected_hist = Counter({int(k): int(v) for k, v in arm.get('q_histogram', {}).items()})
if (grid.get('status') != 'passed' or arm.get('sha256') != sha(tsv)
        or len(rows) != 251599 or len(names) != 251599 or hist != expected_hist
        or npz_manifest.get('status') != 'passed' or npz_manifest.get('completed_rows') != 251599
        or npz_manifest.get('cache_list_sha256') != sha(cache)
        or cache_audit.get('status') != 'passed' or cache_audit.get('semantic_gate', {}).get('status') != 'passed'):
    raise SystemExit('[FAIL] official Qwen TSV/cache/NPZ provenance gate failed')
print('[OK] official-Qwen NoQ full baseline bound to 251,599 audited rows')
PY

if [ "$RUN_MODE" = fresh ]; then
    conflicts=()
    for path in "$ROOT/exps/$S1_EXP" "$ROOT/exps/$S2_EXP" "$FINAL_AUDIT" "$FINAL_REPORT"; do
        [ -e "$path" ] && conflicts+=("$path")
    done
    [ "${#conflicts[@]}" -eq 0 ] || { printf '[FAIL] fresh artifacts exist: %s\n' "${conflicts[@]}" >&2; exit 2; }
fi

python - "$CONTRACT" "$TRAIN_TSV" "$GRID_MANIFEST" "$GT_CACHE" "$NPZ_MANIFEST" "$CACHE_AUDIT" <<'PY'
import hashlib, json, os, subprocess, sys
from datetime import datetime, timezone
from pathlib import Path

out, tsv, grid, cache, npz_manifest, cache_audit = map(Path, sys.argv[1:])
def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()
payload = {
    'schema_version': 1, 'created_at': datetime.now(timezone.utc).isoformat(),
    'experiment': 'phase8_qwen_official_noq_full', 'scale': 'full', 'arm': 'noq',
    'caption_semantics': 'official Qwen caption mapped to exact Jamendo track',
    'matched_bucket_arm': 'k2_balanced', 'q_semantics': 'q_level retained but ignored',
    'train_tsv': str(tsv), 'train_tsv_sha256': sha(tsv),
    'grid_manifest': str(grid), 'grid_manifest_sha256': sha(grid),
    'qwen_cache_list': str(cache), 'qwen_cache_list_sha256': sha(cache),
    'qwen_npz_manifest': str(npz_manifest), 'qwen_npz_manifest_sha256': sha(npz_manifest),
    'qwen_cache_audit': str(cache_audit), 'qwen_cache_audit_sha256': sha(cache_audit),
    'expected_rows': 251599, 'stage1_updates': 400000, 'stage2_updates': 200000,
    'stage2_final_iteration': 600000, 'stage1_use_q_conditioning': False,
    'stage2_use_q_conditioning': False, 'eval_q_mode': 'no_q',
    'seed': 14159265, 'learning_rate': 1e-4, 'batch_size': 8,
    'use_text_attention_mask': False, 'multi_cap': False,
}
payload['git_head'] = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
if out.exists():
    old = json.loads(out.read_text())
    changed = [k for k, v in payload.items() if k not in {'created_at', 'git_head'} and old.get(k) != v]
    if changed: raise SystemExit(f'[FAIL] immutable contract drift: {changed}')
else:
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix('.tmp'); tmp.write_text(json.dumps(payload, indent=2, sort_keys=True)+'\n'); os.replace(tmp, out)
print(f'[OK] contract={out}')
PY

if [ "$PREFLIGHT_ONLY" = true ]; then
    echo "[PREFLIGHT ONLY] $PREFIX; no checkpoint or GPU process started."
    exit 0
fi

export EXP_PREFIX="$PREFIX" TRAIN_TSV GT_CACHE SINGLECAP_NPZ="$NPZ_DIR" NPZ_MANIFEST EXPECTED_ROWS
export EXPERIMENT_REGIME=qwen_official_noq_full EXPERIMENT_RUN_MODE="$RUN_MODE"
export S1_ITERATIONS="$S1_UPDATES" S2_ITERATIONS="$S2_UPDATES"
export S1_USE_Q_CONDITIONING=false S2_USE_Q_CONDITIONING=false EVAL_Q_MODE=no_q
export USE_TEXT_ATTENTION_MASK=false RUN_PRIMARY_EVAL=true RUN_JAMENDO_EVAL=false
export EVAL_NUM_SAMPLES=5521 EVAL_SKIP_AES=false SEED TRAIN_SEED="$SEED"
bash "$ROOT/scripts/training_pipelines/train_pipeline_phase8_bugfix_rerun.sh"

python - "$FINAL_AUDIT" "$FINAL_REPORT" "$CONTRACT" "$S2_EMA" "$METRICS" "$ROOT/exps/$S1_EXP" "$ROOT/exps/$S2_EXP" <<'PY'
import hashlib, json, math, os, sys
from datetime import datetime, timezone
from pathlib import Path
import torch
from omegaconf import OmegaConf

audit_path, report_path, contract_path, ema_path, metrics_path, s1_dir, s2_dir = map(Path, sys.argv[1:])
contract = json.loads(contract_path.read_text())
issues = []
for directory, expected_it, expected_q, label in ((s1_dir, 400000, False, 's1'), (s2_dir, 600000, False, 's2')):
    ckpt = directory / f'{directory.name}_ckpt_last.pth'
    if not ckpt.is_file() or torch.load(ckpt, map_location='cpu', weights_only=False).get('it') != expected_it:
        issues.append(f'{label}_checkpoint_iteration')
    configs = sorted(directory.glob('train-*-hydra/config.yaml'))
    if not configs:
        issues.append(f'{label}_q_conditioning')
    else:
        config = OmegaConf.load(configs[-1])
        if OmegaConf.select(config, 'use_q_conditioning') is not expected_q:
            issues.append(f'{label}_q_conditioning')
        if OmegaConf.select(config, 'seed') != 14159265:
            issues.append(f'{label}_seed')
if not ema_path.is_file(): issues.append('missing_s2_ema')
values = {}
if not metrics_path.is_file():
    issues.append('missing_metrics')
else:
    for line in metrics_path.read_text().splitlines():
        if ': ' in line:
            key, value = line.split(': ', 1)
            if key in {'clap_score', 'aes_CE', 'aes_CU', 'aes_PC', 'aes_PQ'}:
                try: values[key] = float(value)
                except ValueError: issues.append(f'nonfinite_{key}')
    if set(values) != {'clap_score', 'aes_CE', 'aes_CU', 'aes_PC', 'aes_PQ'} or not all(math.isfinite(v) for v in values.values()):
        issues.append('invalid_metrics')
payload = {'schema_version': 1, 'completed_at': datetime.now(timezone.utc).isoformat(),
           'status': 'passed' if not issues else 'failed', 'issues': issues,
           'design': 'official_Qwen_NoQ_full', 'stage1_iteration': 400000,
           'stage2_iteration': 600000, 'stage1_use_q_conditioning': False,
           'stage2_use_q_conditioning': False, 'contract': str(contract_path),
           'model': str(ema_path), 'metrics': str(metrics_path)}
tmp = audit_path.with_suffix('.tmp'); tmp.write_text(json.dumps(payload, indent=2, sort_keys=True)+'\n'); os.replace(tmp, audit_path)
if issues: raise SystemExit('[FAIL] final audit: ' + ', '.join(issues))
def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''): h.update(block)
    return h.hexdigest()
report = {'schema_version': 1, 'completed_at': datetime.now(timezone.utc).isoformat(),
          'status': 'passed', 'experiment': contract['experiment'], 'scale': 'full',
          'arm': 'noq', 'design': 'official_Qwen_NoQ_full', 'training_audit': str(audit_path),
          'training_contract': {'path': str(contract_path), 'sha256': sha(contract_path)},
          'model': {'path': str(ema_path), 'sha256': sha(ema_path)},
          'global': {'protocol': 'MusicCaps 5521; MeanFlow1 CFG0.5', 'no_q': values}}
tmp = report_path.with_suffix('.tmp'); tmp.write_text(json.dumps(report, indent=2, sort_keys=True)+'\n'); os.replace(tmp, report_path)
print(f"[OK] final report={report_path}; CLAP={values['clap_score']:.4f}")
PY
