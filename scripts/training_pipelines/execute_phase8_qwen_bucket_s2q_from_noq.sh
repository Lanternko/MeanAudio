#!/usr/bin/env bash
# Controlled S2-only Q ablation: reuse the completed No-Q S1 checkpoint,
# initialize every Q row from its trained null row, then train/evaluate S2.
# This deliberately never alters the source checkpoint or existing bucket arms.

set -euo pipefail

ROOT=/home/kojiek/MeanAudio
DATA=/mnt/HDD/kojiek/phase4_jamendo_data
LOG_ROOT=/home/kojiek/logs
NPZ_DIR=/mnt/HDD/kojiek/phase8_qwen_official_matched_npz
GT_CACHE="$DATA/phase8_qwen_official_matched_npz_cache_train.txt"
GRID="$DATA/phase8_qwen_meansim_bucket_grid.manifest.json"
MUSICCAPS="$DATA/musiccaps_test.tsv"
HOLDOUT="$ROOT/smoke_data/phase8_qwen_bucket_grid_musiccaps_holdout_n5009.tsv"
EVALUATOR=/home/kojiek/research/meanaudio_eval/phase4_eval.py
SOURCE_PREFIX=phase8_qwen_bucket_quarter_noq
SOURCE_EXP="${SOURCE_PREFIX}_stage1_100000"
SOURCE_CKPT="$ROOT/exps/$SOURCE_EXP/${SOURCE_EXP}_ckpt_last.pth"
SOURCE_EMA="$ROOT/exps/$SOURCE_EXP/${SOURCE_EXP}_ema_final.pth"
SOURCE_AUDIT="$LOG_ROOT/${SOURCE_PREFIX}_FINAL_TRAIN_AUDIT.json"
SOURCE_CONTRACT="$LOG_ROOT/${SOURCE_PREFIX}_contract.json"
K="${K:?K is required}"
STRATEGY="${STRATEGY:-balanced}"
RUN_MODE="${EXPERIMENT_RUN_MODE:-fresh}"
PREFLIGHT_ONLY="${PREFLIGHT_ONLY:-false}"
EVAL_OUTPUT_ROOT="${EVAL_OUTPUT_ROOT:-$ROOT/eval_output_local}"
export EVAL_OUTPUT_ROOT

case "$K" in 2|3|5|10) ;; *) echo "[FAIL] K must be 2, 3, 5, or 10" >&2; exit 2;; esac
case "$STRATEGY" in balanced|fixed) ;; *) echo "[FAIL] STRATEGY must be balanced or fixed" >&2; exit 2;; esac
case "$RUN_MODE" in fresh|resume) ;; *) echo "[FAIL] invalid EXPERIMENT_RUN_MODE" >&2; exit 2;; esac
case "$PREFLIGHT_ONLY" in true|false) ;; *) echo "[FAIL] invalid PREFLIGHT_ONLY" >&2; exit 2;; esac

TRAIN_TSV="$DATA/phase8_qwen_meansim_k${K}_${STRATEGY}.tsv"
PREFIX="phase8_qwen_s2q_from_noq_quarter_k${K}_${STRATEGY}"
S2_EXP="${PREFIX}_stage2_50000"
S2_DIR="$ROOT/exps/$S2_EXP"
S2_CKPT="$S2_DIR/${S2_EXP}_ckpt_last.pth"
S2_EMA="$S2_DIR/${S2_EXP}_ema_final.pth"
INIT_AUDIT="$LOG_ROOT/${PREFIX}_Q_INIT_AUDIT.json"
TRAIN_AUDIT="$LOG_ROOT/${PREFIX}_FINAL_TRAIN_AUDIT.json"
CONTRACT="$LOG_ROOT/${PREFIX}_contract.json"
REPORT="$LOG_ROOT/${PREFIX}_FINAL_METRICS.json"
TRAIN_LOG="$LOG_ROOT/${S2_EXP}.log"
MIGRATE_LOG="$LOG_ROOT/${S2_EXP}_migrate.log"
HIGH_LABEL="${S2_EXP}_musiccaps_n5521_mf1_q9"
LOW_LABEL="${S2_EXP}_musiccaps_n5521_mf1_q0"
HIGH_HOLDOUT_LABEL="${HIGH_LABEL}_holdout5009"

cd "$ROOT"
source /home/kojiek/venvs/dac/bin/activate
source "$ROOT/scripts/runtime/phase8_nvidia_compat_env.sh"
phase8_nvidia_compat_apply || { echo "[FAIL] NVIDIA preflight: $PHASE8_NVIDIA_COMPAT_ERROR" >&2; exit 2; }
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

for path in "$TRAIN_TSV" "$GRID" "$GT_CACHE" "$NPZ_DIR" "$MUSICCAPS" "$HOLDOUT" \
    "$EVALUATOR" "$SOURCE_CKPT" "$SOURCE_EMA" "$SOURCE_AUDIT" "$SOURCE_CONTRACT"; do
    [ -e "$path" ] || { echo "[FAIL] missing input: $path" >&2; exit 2; }
done

python - "$GRID" "$TRAIN_TSV" "$K" "$STRATEGY" "$SOURCE_CKPT" "$SOURCE_AUDIT" "$SOURCE_CONTRACT" <<'PY'
import csv, hashlib, json, sys
from collections import Counter
from pathlib import Path
import torch
from omegaconf import OmegaConf

grid = Path(sys.argv[1])
tsv = Path(sys.argv[2])
k = int(sys.argv[3])
strategy = sys.argv[4]
source = Path(sys.argv[5])
audit_path = Path(sys.argv[6])
contract_path = Path(sys.argv[7])
payload = json.loads(grid.read_text())
arm = payload['outputs'][f'k{k}_{strategy}']
digest = hashlib.sha256(tsv.read_bytes()).hexdigest()
if payload.get('status') != 'passed' or arm.get('sha256') != digest:
    raise SystemExit('[FAIL] K TSV is not bound to passed grid')
with tsv.open(newline='') as f:
    rows = list(csv.DictReader(f, delimiter='\t'))
hist = Counter(int(row['q_level']) for row in rows)
if len(rows) != 251599 or hist != Counter({int(q): n for q, n in arm['q_histogram'].items()}):
    raise SystemExit('[FAIL] K TSV cardinality/histogram drift')
state = torch.load(source, map_location='cpu', weights_only=False)
if state.get('it') != 100000 or 'q_embed.weight' not in state.get('weights', {}):
    raise SystemExit('[FAIL] No-Q S1 checkpoint is not the expected 100k source')
audit = json.loads(audit_path.read_text())
contract = json.loads(contract_path.read_text())
if audit.get('status') != 'passed' or audit.get('stage1_use_q_conditioning') is not False:
    raise SystemExit('[FAIL] source audit is not a passed No-Q S1')
if (contract.get('stage1_use_q_conditioning') is not False
        or contract.get('stage2_use_q_conditioning') is not False):
    raise SystemExit('[FAIL] source contract is not No-Q')
configs = sorted(source.parent.glob('train-*-hydra/config.yaml'))
if not configs or OmegaConf.select(OmegaConf.load(configs[-1]), 'use_q_conditioning') is not False:
    raise SystemExit('[FAIL] source Hydra config is not No-Q')
print(f'[OK] source=NoQ@100k K={k} histogram={dict(sorted(hist.items()))}')
PY

if [ "$RUN_MODE" = fresh ]; then
    conflicts=()
    for path in "$S2_DIR" "$INIT_AUDIT" "$TRAIN_AUDIT" "$CONTRACT" "$REPORT" \
        "$TRAIN_LOG" "$MIGRATE_LOG" "$EVAL_OUTPUT_ROOT/$HIGH_LABEL" \
        "$EVAL_OUTPUT_ROOT/$LOW_LABEL" "$EVAL_OUTPUT_ROOT/metrics/$HIGH_LABEL" \
        "$EVAL_OUTPUT_ROOT/metrics/$LOW_LABEL" "$EVAL_OUTPUT_ROOT/metrics/$HIGH_HOLDOUT_LABEL"; do
        [ -e "$path" ] && conflicts+=("$path")
    done
    [ "${#conflicts[@]}" -eq 0 ] || { printf '[FAIL] fresh artifacts exist: %s\n' "${conflicts[@]}" >&2; exit 2; }
fi

if [ "$PREFLIGHT_ONLY" = true ]; then
    echo "[PREFLIGHT ONLY] $PREFIX; no artifacts, training, or evaluation started."
    exit 0
fi

python - "$CONTRACT" "$PREFIX" "$K" "$STRATEGY" "$TRAIN_TSV" "$SOURCE_CKPT" "$GRID" <<'PY'
import hashlib, json, os, subprocess, sys
from datetime import datetime, timezone
from pathlib import Path
out, prefix, k, strategy, tsv, source, grid = sys.argv[1:]
def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()
payload = {
    'schema_version': 1, 'created_at': datetime.now(timezone.utc).isoformat(),
    'experiment': prefix, 'design': 'NoQ_S1_to_Q_S2_only', 'k': int(k),
    'strategy': strategy, 'source_stage1_checkpoint': source,
    'source_stage1_sha256': sha(source), 'source_stage1_iteration': 100000,
    'stage1_use_q_conditioning': False, 'stage2_use_q_conditioning': True,
    'q_initialization': 'copy-null-q10-to-q0..q9-online-and-ema',
    'train_tsv': tsv, 'train_tsv_sha256': sha(tsv),
    'grid_manifest': grid, 'grid_manifest_sha256': sha(grid),
    'npz_dir': '/mnt/HDD/kojiek/phase8_qwen_official_matched_npz',
    'gt_cache': '/mnt/HDD/kojiek/phase4_jamendo_data/phase8_qwen_official_matched_npz_cache_train.txt',
    'expected_rows': 251599, 'seed': 14159265, 'batch_size': 8,
    'learning_rate': 1e-4, 'stage2_updates': 50000,
    'stage2_final_iteration': 150000, 'eval_high_q': 9, 'eval_low_q': 0,
}
payload['git_head'] = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
path = Path(out)
if path.exists():
    old = json.loads(path.read_text())
    changed = [x for x in payload if x not in {'created_at', 'git_head'} and old.get(x) != payload[x]]
    if changed: raise SystemExit(f'[FAIL] immutable contract drift: {changed}')
else:
    tmp = path.with_suffix('.tmp'); tmp.write_text(json.dumps(payload, indent=2, sort_keys=True)+'\n'); os.replace(tmp, path)
print(f'[OK] contract={path}')
PY

mkdir -p "$S2_DIR" "$LOG_ROOT" "$EVAL_OUTPUT_ROOT/metrics"
if [ ! -f "$S2_CKPT" ]; then
    python set_training_stage.py --stage 2
    python migrate_stage1_to_stage2_ckpt.py --s1_ckpt "$SOURCE_CKPT" --s2_out "$S2_CKPT" --q-init copy-null \
        2>&1 | tee "$MIGRATE_LOG"
    python - "$S2_CKPT" "$INIT_AUDIT" <<'PY'
import json, os, sys
from datetime import datetime, timezone
from pathlib import Path
import torch
state = torch.load(sys.argv[1], map_location='cpu', weights_only=False)
issues=[]
for key, weight in [('online', state['weights'].get('q_embed.weight')),
                    *[(key, value) for key, value in state['ema'].items() if key.endswith('q_embed.weight')]]:
    if weight is None or tuple(weight.shape)[0] != 11 or not torch.equal(weight[:10], weight[10].expand_as(weight[:10])):
        issues.append(str(key))
payload={'schema_version':1,'created_at':datetime.now(timezone.utc).isoformat(),'status':'passed' if not issues else 'failed','checkpoint':sys.argv[1],'iteration':state.get('it'),'q_init':'copy-null','nonidentical_q_rows':issues}
out=Path(sys.argv[2]); tmp=out.with_suffix('.tmp'); tmp.write_text(json.dumps(payload,indent=2,sort_keys=True)+'\n'); os.replace(tmp,out)
print(json.dumps(payload, indent=2)); raise SystemExit(bool(issues) or state.get('it') != 100000)
PY
fi

checkpoint_iteration() { python - "$1" <<'PY'
import sys, torch
from pathlib import Path
p=Path(sys.argv[1]); print(-1 if not p.is_file() else torch.load(p,map_location='cpu',weights_only=False).get('it'))
PY
}
s2_it=$(checkpoint_iteration "$S2_CKPT")
if [ "$s2_it" -lt 150000 ]; then
    python set_training_stage.py --stage 2
    torchrun --standalone --nproc_per_node=1 train.py \
        data=meanaudio model=meanaudio_s exp_id="$S2_EXP" num_iterations=150000 \
        "lr_schedule_steps=[999999,999999]" +use_q_conditioning=true batch_size=8 +accumulation_steps=1 \
        learning_rate=1e-4 seed=14159265 linear_warmup_steps=1000 num_workers=4 \
        save_weights_interval=150000 save_checkpoint_interval=150000 ++ema.checkpoint_every=150000 \
        +use_rope=False +use_wandb=False +use_text_attention_mask=false val_interval=999999 eval_interval=999999 save_eval_interval=999999 \
        "data.AudioCaps_npz.tsv=$TRAIN_TSV" "+data.AudioCaps_npz.gt_cache=$GT_CACHE" \
        "data.AudioCaps_val_npz.tsv=$TRAIN_TSV" "+data.AudioCaps_val_npz.gt_cache=$GT_CACHE" \
        "++data.AudioCaps_npz.npz_dir=$NPZ_DIR" "++data.AudioCaps_val_npz.npz_dir=$NPZ_DIR" ++multi_cap=False \
        2>&1 | tee "$TRAIN_LOG"
elif [ "$s2_it" -ne 150000 ]; then
    echo "[FAIL] unexpected S2 iteration=$s2_it" >&2; exit 2
fi
[ -f "$S2_EMA" ] || { echo "[FAIL] missing S2 EMA" >&2; exit 2; }

python - "$TRAIN_AUDIT" "$S2_CKPT" "$S2_DIR" "$CONTRACT" "$INIT_AUDIT" "$TRAIN_TSV" <<'PY'
import json, math, os, sys
from datetime import datetime, timezone
from pathlib import Path
import torch
from omegaconf import OmegaConf
out, ckpt, directory, contract, init, tsv = map(Path, sys.argv[1:])
state=torch.load(ckpt,map_location='cpu',weights_only=False); issues=[]
if state.get('it') != 150000: issues.append(f'iteration={state.get("it")}')
cfgs=sorted(directory.glob('train-*-hydra/config.yaml'))
if not cfgs: issues.append('missing_hydra_config')
else:
 cfg=OmegaConf.load(cfgs[-1])
 for key, expected in {'use_q_conditioning':True,'num_iterations':150000,'seed':14159265,'data.AudioCaps_npz.tsv':str(tsv)}.items():
  if OmegaConf.select(cfg,key) != expected: issues.append(f'{key}={OmegaConf.select(cfg,key)!r}')
if not init.is_file() or json.loads(init.read_text()).get('status') != 'passed': issues.append('copy_null_init_audit')
contract_payload=json.loads(contract.read_text())
payload={
 'schema_version':1, 'completed_at':datetime.now(timezone.utc).isoformat(),
 'status':'passed' if not issues else 'failed', 'issues':issues,
 'design':'NoQ_S1_to_Q_S2_only',
 'k':int(Path(contract).stem.split('_k')[-1].split('_')[0]),
 'strategy':contract_payload['strategy'],
 'source_stage1_iteration':100000, 'stage2_iteration':state.get('it'),
 'stage1_use_q_conditioning':False, 'stage2_use_q_conditioning':True,
 'q_initialization':'copy-null', 'contract':str(contract),
 'q_init_audit':str(init), 'hydra_config':str(cfgs[-1]) if cfgs else None,
}
tmp=out.with_suffix('.tmp'); tmp.write_text(json.dumps(payload,indent=2,sort_keys=True)+'\n'); os.replace(tmp,out); print(json.dumps(payload,indent=2)); raise SystemExit(bool(issues))
PY

run_eval() {
    local label="$1"
    local q="$2"
    local out="$EVAL_OUTPUT_ROOT/$label"
    local metrics="$EVAL_OUTPUT_ROOT/metrics/$label/metrics.txt"
    local log="$LOG_ROOT/${label}_eval.log"
    if [ ! -f "$metrics" ]; then
        mkdir -p "$out/audio"
        python eval.py --variant meanaudio_s --model_path "$S2_EMA" --output "$out/audio" --tsv "$MUSICCAPS" \
            --use_meanflow --num_steps 1 --encoder_name t5_clap --text_c_dim 512 --cfg_strength 0.5 \
            --quality_level "$q" --no_text_attention_mask --full_precision 2>&1 | tee "$log"
        [ "$(find "$out/audio" -maxdepth 1 -name '*.flac' -type f | wc -l)" -eq 5521 ] || { echo "[FAIL] incomplete audio $label" >&2; exit 2; }
        python "$EVALUATOR" --gen_dir "$out/audio" --out_dir "$EVAL_OUTPUT_ROOT/metrics" --tsv "$MUSICCAPS" --exp_name "$label" --num_samples 5521 2>&1 | tee -a "$log"
    fi
    [ -f "$metrics" ] || { echo "[FAIL] missing metrics $label" >&2; exit 2; }
}
run_eval "$HIGH_LABEL" 9
run_eval "$LOW_LABEL" 0
if [ ! -f "$EVAL_OUTPUT_ROOT/metrics/$HIGH_HOLDOUT_LABEL/metrics.txt" ]; then
    python "$EVALUATOR" --gen_dir "$EVAL_OUTPUT_ROOT/$HIGH_LABEL/audio" --out_dir "$EVAL_OUTPUT_ROOT/metrics" --tsv "$HOLDOUT" --exp_name "$HIGH_HOLDOUT_LABEL" --num_samples 5009 \
        2>&1 | tee "$LOG_ROOT/${HIGH_HOLDOUT_LABEL}_eval.log"
fi

python - "$REPORT" "$CONTRACT" "$TRAIN_AUDIT" "$S2_EMA" "$HIGH_LABEL" "$LOW_LABEL" "$HIGH_HOLDOUT_LABEL" "$MUSICCAPS" "$HOLDOUT" <<'PY'
import hashlib,json,math,os,sys
from datetime import datetime,timezone
from pathlib import Path
out,contract,audit,model,high,low,holdout,musiccaps,holdout_tsv=map(Path,sys.argv[1:])
root=Path(os.environ['EVAL_OUTPUT_ROOT'])/'metrics'; required={'clap_score','aes_CE','aes_CU','aes_PC','aes_PQ'}
def sha(p):
 h=hashlib.sha256();
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''): h.update(b)
 return h.hexdigest()
def metric(label):
 p=root/str(label)/'metrics.txt'; values={}
 for line in p.read_text().splitlines():
  if ':' in line:
   k,v=line.split(':',1)
   if k.strip() in required: values[k.strip()]=float(v)
 if set(values)!=required or not all(math.isfinite(v) for v in values.values()): raise SystemExit(f'[FAIL] invalid metrics {p}')
 return {'label':str(label),**values}
c=json.loads(contract.read_text()); h,l,ho=metric(high),metric(low),metric(holdout)
payload={'schema_version':1,'completed_at':datetime.now(timezone.utc).isoformat(),'status':'passed','experiment':c['experiment'],'design':c['design'],'k':c['k'],'strategy':c['strategy'],'source_stage1':{'checkpoint':c['source_stage1_checkpoint'],'sha256':c['source_stage1_sha256'],'iteration':100000,'q_conditioning':False},'training_audit':str(audit),'training_contract':{'path':str(contract),'sha256':sha(contract)},'model':{'path':str(model),'sha256':sha(model)},'global':{'protocol':'MusicCaps 5521; MeanFlow1 CFG0.5','high_q9':h,'supported_low':l,'q9_minus_low_clap':h['clap_score']-l['clap_score'],'holdout5009_high_q9':ho}}
tmp=out.with_suffix('.tmp'); tmp.write_text(json.dumps(payload,indent=2,sort_keys=True)+'\n'); os.replace(tmp,out); print(json.dumps(payload,indent=2,sort_keys=True))
PY
echo "[COMPLETE] $PREFIX report=$REPORT"
