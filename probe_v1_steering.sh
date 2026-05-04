#!/bin/bash
# ============================================================
# P9.5 V1 prompt-steering probe
# Generates 24 wav (4 prompt pairs × 3 seeds × 2 prompts each),
# then computes (A-B L2) / (noise L2) ratio per pair.
#
# Per memory project_p9_text_conditioning_dead.md:
#   single-cap models (P7 V1, P8): ratio 0.9-1.7 (prompt > noise)
#   multi-cap models (P9 V1/V2):   ratio 0.009-0.15 (noise > prompt → collapse)
#
# V2 launch gate (Codex 2026-05-04):
#   ratio > 0.2 in any pair → V2 worth running
#   ratio < 0.05 across all pairs → V2 will likely confirm collapse, save GPU
#
# Usage:
#   bash probe_v1_steering.sh [<model_ema_pth>]   # defaults to V1 ema_final
# ============================================================

set -eo pipefail
cd ~/MeanAudio
source ~/venvs/dac/bin/activate
export CUDA_VISIBLE_DEVICES=0

MODEL_EMA="${1:-$HOME/MeanAudio/exps/phase9_5_v1_stage2_200000/phase9_5_v1_stage2_200000_ema_final.pth}"

if [ ! -f "$MODEL_EMA" ]; then
    echo "[FAIL] model not found: $MODEL_EMA"
    echo "       wait for V1 Stage 2 to finish, then re-run"
    exit 1
fi

echo "Probing model: $MODEL_EMA"

OUT=eval_output/p9_5_v1_steering_probe
mkdir -p "$OUT/audio"

SEEDS=(42 123 456)
declare -A PROMPTS
PROMPTS[01_instrument_A]='Solo piano instrumental, calm and intimate.'
PROMPTS[01_instrument_B]='Fast EDM instrumental, strong kick drum, energetic synth lead.'
PROMPTS[02_vocals_A]='Instrumental soundtrack, no vocals, orchestral.'
PROMPTS[02_vocals_B]='Pop song with clear female vocals singing a melody.'
PROMPTS[03_drums_A]='Drumless ambient instrumental, no percussion, spacious pads.'
PROMPTS[03_drums_B]='Drum-heavy techno instrumental, strong kick and snare, 128 BPM.'
PROMPTS[04_density_A]='Minimal solo violin, sparse arrangement.'
PROMPTS[04_density_B]='Dense orchestral trailer music with full ensemble.'

PAIRS=(01_instrument 02_vocals 03_drums 04_density)

# infer.py NoQ workaround per memory feedback_infer_no_q_workaround.md:
#   infer.py 沒 --no_q → 用 --quality_level 10 (null token) for NoQ models
BASE_ARGS="--variant meanaudio_s --encoder_name t5_clap --text_c_dim 512 \
    --use_meanflow --num_steps 1 --cfg_strength 0.5 --full_precision \
    --quality_level 10 \
    --model_path $MODEL_EMA --output $OUT/audio"

echo "=== generating 24 wav (4 pairs × 3 seeds × 2 prompts) ==="
for pair in "${PAIRS[@]}"; do
    for seed in "${SEEDS[@]}"; do
        for AB in A B; do
            key="${pair}_${AB}"
            name="${pair}_${AB}_s${seed}"
            echo "  $name"
            python infer.py $BASE_ARGS \
                --seed $seed \
                --output_name "$name" \
                --prompt "${PROMPTS[$key]}"
        done
    done
done

echo "=== computing ratios ==="
python - <<'PYEOF'
import numpy as np, soundfile as sf
from pathlib import Path
from itertools import combinations

OUT = Path('eval_output/p9_5_v1_steering_probe/audio')
PAIRS = ['01_instrument', '02_vocals', '03_drums', '04_density']
SEEDS = [42, 123, 456]

def load(name):
    wav, _ = sf.read(str(OUT / f'{name}.wav'))
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    return wav

print(f'{"pair":<15} {"A-B L2":>10} {"noise L2":>10} {"ratio":>10}')
print('-' * 50)
ratios = []
for pair in PAIRS:
    # A-B L2: same seed, different prompt
    ab_l2s = []
    for s in SEEDS:
        a = load(f'{pair}_A_s{s}')
        b = load(f'{pair}_B_s{s}')
        n = min(len(a), len(b))
        ab_l2s.append(np.linalg.norm(a[:n] - b[:n]))

    # noise L2: same prompt (A), different seed
    noise_l2s = []
    for s1, s2 in combinations(SEEDS, 2):
        a1 = load(f'{pair}_A_s{s1}')
        a2 = load(f'{pair}_A_s{s2}')
        n = min(len(a1), len(a2))
        noise_l2s.append(np.linalg.norm(a1[:n] - a2[:n]))

    ab = float(np.mean(ab_l2s))
    noise = float(np.mean(noise_l2s))
    ratio = ab / max(noise, 1e-9)
    ratios.append((pair, ratio))
    print(f'{pair:<15} {ab:>10.3f} {noise:>10.3f} {ratio:>10.3f}')

print()
print('Reference ratios (memory project_p9_text_conditioning_dead):')
print('  P7 V1 Q (single-cap):    1.071–1.702 (prompt dominant)')
print('  P8 NoQ (single-cap):     0.913–1.723 (prompt dominant)')
print('  P9 V1 NoQ (multi-cap):   0.025–0.147 (noise dominant)')
print('  P9 V2 Q=8 (multi-cap):   0.012–0.056 (noise dominant)')
print()
max_ratio = max(r for _, r in ratios)
print(f'P9.5 V1 max ratio across pairs: {max_ratio:.3f}')
if max_ratio > 0.2:
    print('  → V2 launch GO (steering not collapsed)')
elif max_ratio > 0.05:
    print('  → V2 launch ambiguous, prefer CLAP-only gate (check vs P9 V1 baseline 0.0650)')
else:
    print('  → V2 launch SKIP recommended (steering collapsed like P9, V2 likely confirms failure)')
PYEOF
