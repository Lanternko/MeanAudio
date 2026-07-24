#!/usr/bin/env bash
# Train (or explicitly reuse) one bucket arm and evaluate both S1 and global S2.
# Invoke this file through run_with_experiment_report.sh.

set -euo pipefail

ROOT=/home/kojiek/MeanAudio
DATA=/mnt/HDD/kojiek/phase4_jamendo_data
LOG_ROOT=/home/kojiek/logs
EVALUATOR=/home/kojiek/research/meanaudio_eval/phase4_eval.py
GRID_MANIFEST="$DATA/phase8_qwen_meansim_bucket_grid.manifest.json"
PILOT_TSV="$ROOT/smoke_data/phase8_qwen_bucket_grid_musiccaps_seed14159265_n512.tsv"
FULL_TSV="$DATA/musiccaps_test.tsv"
HOLDOUT_TSV="$ROOT/smoke_data/phase8_qwen_bucket_grid_musiccaps_holdout_n5009.tsv"
K="${K:?}"
STRATEGY="${STRATEGY:?}"
SCALE="${SCALE:?}"
REUSE="${REUSE:-none}"
RUN_MODE="${EXPERIMENT_RUN_MODE:-fresh}"
VALIDATE_ONLY="${VALIDATE_ONLY:-false}"
case "$VALIDATE_ONLY" in true|false) ;; *)
    echo "[FAIL] VALIDATE_ONLY must be true or false" >&2; exit 2;;
esac

case "$SCALE" in
    pilot) S1_UPDATES=25000; S2_UPDATES=12500; PROMPTS="$PILOT_TSV"; N=512 ;;
    quarter) S1_UPDATES=100000; S2_UPDATES=50000; PROMPTS="$FULL_TSV"; N=5521 ;;
    *) echo "[FAIL] invalid SCALE=$SCALE" >&2; exit 2 ;;
esac
PREFIX="phase8_qwen_bucket_${SCALE}_k${K}_${STRATEGY}"
REPORT="$LOG_ROOT/${PREFIX}_FINAL_METRICS.json"

cd "$ROOT"
# shellcheck source=/dev/null
source /home/kojiek/venvs/dac/bin/activate
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

LOW_Q=$(python - "$GRID_MANIFEST" "$K" "$STRATEGY" <<'PY'
import json, sys
payload = json.load(open(sys.argv[1]))
arm = payload["outputs"][f"k{sys.argv[2]}_{sys.argv[3]}"]
value = arm["diagnostic_low_q"]
print("none" if value is None else value)
PY
)

HISTORICAL_REUSE=false
case "$REUSE" in
    none)
        K="$K" STRATEGY="$STRATEGY" SCALE="$SCALE" \
            PREFLIGHT_ONLY="$VALIDATE_ONLY" \
            EXPERIMENT_RUN_MODE="$RUN_MODE" \
            bash scripts/training_pipelines/train_phase8_qwen_bucket_arm.sh
        if [ "$VALIDATE_ONLY" = true ]; then
            echo "[VALIDATE ONLY] new arm preflight passed"
            exit 0
        fi
        S1_EXP="${PREFIX}_stage1_${S1_UPDATES}"
        S2_EXP="${PREFIX}_stage2_${S2_UPDATES}"
        AUDIT="$LOG_ROOT/${PREFIX}_FINAL_TRAIN_AUDIT.json"
        ;;
    k2_balanced_historical)
        [ "$SCALE:$K:$STRATEGY" = "quarter:2:balanced" ] || {
            echo "[FAIL] invalid K2 historical reuse request" >&2; exit 2;
        }
        HISTORICAL_REUSE=true
        S1_EXP=phase8_qwen_quarter_e2e_halfq_stage1_100000
        S2_EXP=phase8_qwen_quarter_e2e_halfq_stage2_50000
        AUDIT="$LOG_ROOT/phase8_qwen_quarter_e2e_halfq_FINAL_TRAIN_AUDIT.json"
        ;;
    k10_fixed_historical)
        [ "$SCALE:$K:$STRATEGY" = "quarter:10:fixed" ] || {
            echo "[FAIL] invalid K10 historical reuse request" >&2; exit 2;
        }
        HISTORICAL_REUSE=true
        S1_EXP=phase8_qwen_quarter_e2e_fullq_stage1_100000
        S2_EXP=phase8_qwen_quarter_e2e_fullq_stage2_50000
        AUDIT="$LOG_ROOT/phase8_qwen_quarter_e2e_fullq_FINAL_TRAIN_AUDIT.json"
        ;;
    *) echo "[FAIL] invalid REUSE=$REUSE" >&2; exit 2 ;;
esac

S1_EMA="$ROOT/exps/$S1_EXP/${S1_EXP}_ema_final.pth"
S2_EMA="$ROOT/exps/$S2_EXP/${S2_EXP}_ema_final.pth"
for path in "$AUDIT" "$S1_EMA" "$S2_EMA" "$PROMPTS"; do
    [ -f "$path" ] || { echo "[FAIL] missing arm artifact: $path" >&2; exit 2; }
done
python - "$AUDIT" "$GRID_MANIFEST" "$K" "$STRATEGY" "$SCALE" \
    "$HISTORICAL_REUSE" <<'PY'
import json, sys
from pathlib import Path
from omegaconf import OmegaConf
payload = json.load(open(sys.argv[1]))
if payload.get("status") != "passed":
    raise SystemExit("[FAIL] training audit is not passed")
if sys.argv[6] == "true":
    contract_path = Path(payload.get("contract", ""))
    if not contract_path.is_file():
        raise SystemExit("[FAIL] historical audit contract missing")
    contract = json.loads(contract_path.read_text())
    grid = json.load(open(sys.argv[2]))
    expected = grid["outputs"][f"k{sys.argv[3]}_{sys.argv[4]}"]["sha256"]
    if contract.get("train_tsv_sha256") != expected:
        raise SystemExit("[FAIL] historical checkpoint TSV SHA is not reuse-equivalent")
    expected_arm = "halfq" if sys.argv[3:5] == ["2", "balanced"] else "fullq"
    if (
        payload.get("arm") != expected_arm
        or payload.get("stage1_iteration") != 100000
        or payload.get("stage2_iteration") != 150000
        or payload.get("stage1_use_q_conditioning") is not True
        or payload.get("stage2_use_q_conditioning") is not True
    ):
        raise SystemExit("[FAIL] historical audit identity/iteration/Q mismatch")
    for label, model, iterations in (
        ("S1", "fluxaudio_s", 100000),
        ("S2", "meanaudio_s", 150000),
    ):
        cfg = OmegaConf.load(payload["hydra_configs"][label])
        checks = {
            "model": model,
            "num_iterations": iterations,
            "use_q_conditioning": True,
            "data.AudioCaps_npz.tsv": contract["train_tsv"],
        }
        for key, value in checks.items():
            if OmegaConf.select(cfg, key) != value:
                raise SystemExit(f"[FAIL] historical {label} config {key} mismatch")
else:
    expected_s1, expected_s2 = (
        (25000, 37500) if sys.argv[5] == "pilot" else (100000, 150000)
    )
    if (
        payload.get("scale") != sys.argv[5]
        or payload.get("k") != int(sys.argv[3])
        or payload.get("strategy") != sys.argv[4]
        or payload.get("q_conditioning") is not True
        or payload.get("stage1_iteration") != expected_s1
        or payload.get("stage2_iteration") != expected_s2
    ):
        raise SystemExit("[FAIL] new training audit identity/Q mismatch")
PY
if [ "$VALIDATE_ONLY" = true ]; then
    echo "[VALIDATE ONLY] historical reuse audit/model contract passed"
    exit 0
fi

run_eval() {
    local label="$1" variant="$2" model="$3" q="$4"
    local out="$ROOT/eval_output/$label"
    local metrics="$ROOT/eval_output/metrics/$label/metrics.txt"
    local eval_log="$LOG_ROOT/${label}_eval.log"
    local provenance="$out/provenance.json"
    local protocol=()
    local protocol_name
    if [ "$variant" = fluxaudio_s ]; then
        protocol=(--num_steps 25 --cfg_strength 4.5)
        protocol_name="FM25_CFG4.5"
    else
        protocol=(--use_meanflow --num_steps 1 --cfg_strength 0.5)
        protocol_name="MF1_CFG0.5"
    fi
    mkdir -p "$out/audio"
    bind_eval_provenance() {
        python - "$provenance" "$label" "$variant" "$model" "$PROMPTS" \
            "$N" "$q" "$protocol_name" <<'PY'
import hashlib, json, os, sys
from pathlib import Path
out, label, variant, model, prompts, n, q, protocol = sys.argv[1:]
def sha(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
payload = {
    "schema_version": 1, "label": label, "variant": variant,
    "model": model, "model_sha256": sha(model),
    "prompts": prompts, "prompts_sha256": sha(prompts),
    "rows": int(n), "quality_level": int(q), "protocol": protocol,
}
path = Path(out)
if path.exists():
    if json.loads(path.read_text()) != payload:
        raise SystemExit("[FAIL] evaluation provenance drift")
else:
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(tmp, path)
PY
    }
    if [ -f "$metrics" ]; then
        audio_n=$(find "$out/audio" -maxdepth 1 -type f -name '*.flac' | wc -l)
        [ "$audio_n" -eq "$N" ] || {
            echo "[FAIL] stale metrics audio count $audio_n/$N for $label" >&2; exit 2;
        }
        grep -F "Test TSV: $PROMPTS" "$metrics" >/dev/null || {
            echo "[FAIL] stale metrics prompt identity: $metrics" >&2; exit 2;
        }
        grep -F "Generated audio: $out/audio" "$metrics" >/dev/null || {
            echo "[FAIL] stale metrics audio identity: $metrics" >&2; exit 2;
        }
        grep -F "Test clips: $N" "$metrics" >/dev/null || {
            echo "[FAIL] stale metrics row count: $metrics" >&2; exit 2;
        }
        bind_eval_provenance
        echo "[SKIP] verified complete metrics: $metrics"
        return
    fi
    python eval.py --variant "$variant" --model_path "$model" \
        --output "$out/audio" --tsv "$PROMPTS" \
        --encoder_name t5_clap --text_c_dim 512 --no_text_attention_mask \
        --full_precision "${protocol[@]}" --quality_level "$q" \
        2>&1 | tee "$eval_log"
    audio_n=$(find "$out/audio" -maxdepth 1 -type f -name '*.flac' | wc -l)
    if [ "$audio_n" -ne "$N" ]; then
        echo "[FAIL] $label generated $audio_n/$N audio files" >&2
        exit 2
    fi
    bind_eval_provenance
    python "$EVALUATOR" --gen_dir "$out/audio" --tsv "$PROMPTS" \
        --exp_name "$label" --num_samples "$N" 2>&1 | tee -a "$eval_log"
    [ -f "$metrics" ] || { echo "[FAIL] no metrics for $label" >&2; exit 2; }
}

# Historical quarter labels deliberately point at already completed full-5521
# evaluations, avoiding regeneration. Missing supported-low endpoints are added.
if [ "$HISTORICAL_REUSE" = true ]; then
    S1_BASE="${S1_EXP}_musiccaps_fm25"
    S2_BASE="${S2_EXP}_musiccaps_mf1"
else
    S1_BASE="${S1_EXP}_musiccaps_n${N}_fm25"
    S2_BASE="${S2_EXP}_musiccaps_n${N}_mf1"
fi
S1_HIGH="${S1_BASE}_q9"
S2_HIGH="${S2_BASE}_q9"
run_eval "$S1_HIGH" fluxaudio_s "$S1_EMA" 9
run_eval "$S2_HIGH" meanaudio_s "$S2_EMA" 9
S1_LOW=
S2_LOW=
if [ "$LOW_Q" != none ]; then
    S1_LOW="${S1_BASE}_q${LOW_Q}"
    S2_LOW="${S2_BASE}_q${LOW_Q}"
    run_eval "$S1_LOW" fluxaudio_s "$S1_EMA" "$LOW_Q"
    run_eval "$S2_LOW" meanaudio_s "$S2_EMA" "$LOW_Q"
fi

S1_HOLDOUT=
S2_HOLDOUT=
score_holdout() {
    local source_label="$1"
    local label="${source_label}_holdout5009"
    local metrics="$ROOT/eval_output/metrics/$label/metrics.txt"
    local eval_log="$LOG_ROOT/${label}_eval.log"
    if [ ! -f "$metrics" ]; then
        python "$EVALUATOR" \
            --gen_dir "$ROOT/eval_output/$source_label/audio" \
            --tsv "$HOLDOUT_TSV" --exp_name "$label" --num_samples 5009 \
            2>&1 | tee "$eval_log"
    fi
    [ -f "$metrics" ] || { echo "[FAIL] no holdout metrics for $label" >&2; exit 2; }
    grep -F "Test TSV: $HOLDOUT_TSV" "$metrics" >/dev/null || {
        echo "[FAIL] holdout metric prompt identity mismatch" >&2; exit 2;
    }
    grep -F "Test clips: 5009" "$metrics" >/dev/null || {
        echo "[FAIL] holdout metric row count mismatch" >&2; exit 2;
    }
    echo "$label"
}
if [ "$SCALE" = quarter ]; then
    [ -f "$HOLDOUT_TSV" ] || { echo "[FAIL] missing holdout: $HOLDOUT_TSV" >&2; exit 2; }
    S1_HOLDOUT=$(score_holdout "$S1_HIGH" | tail -1)
    S2_HOLDOUT=$(score_holdout "$S2_HIGH" | tail -1)
fi

python - "$REPORT" "$GRID_MANIFEST" "$K" "$STRATEGY" "$SCALE" "$N" \
    "$PROMPTS" "$AUDIT" "$HISTORICAL_REUSE" "$S1_HIGH" "$S2_HIGH" \
    "$S1_LOW" "$S2_LOW" "$S1_HOLDOUT" "$S2_HOLDOUT" \
    "$S1_EMA" "$S2_EMA" <<'PY'
import hashlib
import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

(out, grid_path, k, strategy, scale, n, prompts, audit, reused,
 s1_high_label, s2_high_label, s1_low_label, s2_low_label,
 s1_holdout_label, s2_holdout_label, s1_model, s2_model) = sys.argv[1:]
metrics_root = Path("/home/kojiek/MeanAudio/eval_output/metrics")
required = {"clap_score", "aes_CE", "aes_CU", "aes_PC", "aes_PQ"}
def read(label):
    if not label:
        return None
    values = {}
    path = metrics_root / label / "metrics.txt"
    for line in path.read_text().splitlines():
        if ":" in line:
            key, value = line.split(":", 1)
            if key.strip() in required:
                values[key.strip()] = float(value)
    if set(values) != required or not all(math.isfinite(v) for v in values.values()):
        raise SystemExit(f"[FAIL] incomplete/nonfinite metrics: {path}: {values}")
    return {"label": label, **values}
def sha(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
grid = json.load(open(grid_path))
arm = grid["outputs"][f"k{k}_{strategy}"]
audit_payload = json.load(open(audit))
contract_path = audit_payload["contract"]
s1_high, s2_high = read(s1_high_label), read(s2_high_label)
s1_low, s2_low = read(s1_low_label), read(s2_low_label)
s1_holdout, s2_holdout = read(s1_holdout_label), read(s2_holdout_label)
payload = {
    "schema_version": 1,
    "completed_at": datetime.now(timezone.utc).isoformat(),
    "status": "passed",
    "experiment": f"phase8_qwen_bucket_{scale}_k{k}_{strategy}",
    "scale": scale,
    "k": int(k),
    "strategy": strategy,
    "prompts": {"path": prompts, "sha256": sha(prompts), "rows": int(n)},
    "bucket_support": {
        key: arm[key] for key in (
            "nominal_k", "occupied_k", "occupied_q_codes",
            "supported_k_at_1pct", "supported_q_codes_at_1pct",
            "nominal_low_q", "diagnostic_low_q", "high_q",
        )
    },
    "historical_checkpoint_reused": reused == "true",
    "training_audit": audit,
    "training_contract": {
        "path": contract_path,
        "sha256": sha(contract_path),
    },
    "models": {
        "stage1": {"path": s1_model, "sha256": sha(s1_model)},
        "global": {"path": s2_model, "sha256": sha(s2_model)},
    },
    "stage1": {
        "protocol": f"MusicCaps {n}; FluxAudio FM25 CFG4.5",
        "high_q9": s1_high,
        "supported_low": s1_low,
        "q9_minus_low_clap": (
            None if s1_low is None
            else s1_high["clap_score"] - s1_low["clap_score"]
        ),
        "holdout5009_high_q9": s1_holdout,
    },
    "global": {
        "protocol": f"MusicCaps {n}; MeanFlow1 CFG0.5",
        "high_q9": s2_high,
        "supported_low": s2_low,
        "q9_minus_low_clap": (
            None if s2_low is None
            else s2_high["clap_score"] - s2_low["clap_score"]
        ),
        "holdout5009_high_q9": s2_holdout,
    },
}
path = Path(out)
tmp = path.with_suffix(path.suffix + ".tmp")
tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
os.replace(tmp, path)
print(json.dumps(payload, indent=2, sort_keys=True))
PY
