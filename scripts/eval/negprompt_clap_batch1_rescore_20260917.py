#!/usr/bin/env python
"""062: re-score the core negprompt tables with per-file (batch 1) CLAP.

novocal_reeval (CFG 0, 12 arms) and negprompt_reeval_cfg3.0 (CFG 3 + fidelity8
negative prompt, 14 arms) scored CLAP in batches of 32 and then deleted the audio.
laion_clap pads differently above batch 8, so every CLAP in those tables is on a
different scale from phase4_eval.py (memory reference_clap_batch_size_sensitivity).
This job regenerates each arm with the exact original generation command, checks
that regeneration is faithful, scores CLAP one file at a time, and deletes the audio.

Faithfulness gate: AES is batch-independent and seed-42 generation is deterministic,
so a fixed 64-clip sample must reproduce the stored per-clip CE/CU/PC/PQ within
AES_TOLERANCE. If it does not, the arm fails loudly and nothing is written: a
batch-1 CLAP on different audio would not be a rescore.

Nothing in the original result files is modified. Results land in
OUT/<family>/<label>.json; OUT/summary.json tabulates b32 vs b1 side by side.
Resumable per arm: an existing result is skipped, partial audio is topped up.
"""
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path('/home/kojiek/MeanAudio')
sys.path.insert(0, str(ROOT / 'scripts/eval'))
from rescore_clap_batch1 import METHOD, load_model, load_rows, score_clap_batch1  # noqa: E402

EXPS = ROOT / 'exps'
TSV = Path('/mnt/HDD/kojiek/phase4_jamendo_data/musiccaps_test.tsv')
PYTHON = '/home/kojiek/venvs/dac/bin/python'
ARTIFACTS = Path('/home/kojiek/nvme_experiment_artifacts/meanaudio')
OUT = ARTIFACTS / 'negprompt_clap_batch1_rescore_20260917'
EXPECTED = 5521
AES_SAMPLE = 64
AES_TOLERANCE = 1e-3
NEGATIVE_PROMPT = ('low quality recording, noisy, amateur, distorted, muffled, '
                   'poor fidelity, hiss, lo-fi')

# family -> (source result dir, cfg_strength, negative prompt or None)
FAMILIES = {
    'cfg0': (ARTIFACTS / 'novocal_reeval', '0.0', None),
    'cfg3neg': (ARTIFACTS / 'negprompt_reeval_cfg3.0', '3.0', NEGATIVE_PROMPT),
}

# Same labels, checkpoints and q flags as negprompt_reeval_full_arms.ARMS.
CORE = [
    ('c2p0_slot0_full_noq',          'phase8_qwen_caption10s_multisent_noq_full_stage2_200000',            ['--no_q']),
    ('fulltrack_q3_full_q9',         'phase8_qwen_s2q_from_noq_full_k3_balanced_stage2_200000',            ['--quality_level', '9']),
    ('c2p0_fair013_worst_full',      'phase8_qwen_caption2p0_fair013_worstof3_noq_full_stage2_200000',     ['--no_q']),
    ('c2p0_slot0_q5_full_q9',        'phase8_qwen_caption2p0_s2q_from_noq_full_k5_balanced_stage2_200000', ['--quality_level', '9']),
    ('fulltrack_noq_full',           'phase8_qwen_official_noq_full_stage2_200000',                        ['--no_q']),
    ('c2p0_fair013_best_full',       'phase8_qwen_caption2p0_fair013_bestof3_noq_full_stage2_200000',      ['--no_q']),
    ('c2p0_slot0_q3_full_q9',        'phase8_qwen_caption2p0_s2q_from_noq_full_k3_balanced_stage2_200000', ['--quality_level', '9']),
    ('c2p0_slot2_full_noq',          'phase8_qwen_caption2p0_slot2_noq_full_stage2_200000',                ['--no_q']),
    ('c2p0_slot0_full_seed27182818', 'phase8_qwen_caption10s_multisent_noq_full_seed27182818_stage2_200000', ['--no_q']),
    ('c2p0_slot0_q5_full_q0',        'phase8_qwen_caption2p0_s2q_from_noq_full_k5_balanced_stage2_200000', ['--quality_level', '0']),
    ('c2p0_slot0_q3_full_q0',        'phase8_qwen_caption2p0_s2q_from_noq_full_k3_balanced_stage2_200000', ['--quality_level', '0']),
    ('p7v1_fullq_control_q9',        'phase7_v1_fullq_control_stage2_200000',                              ['--quality_level', '9']),
]
CFG3_ONLY = [
    ('c2p0_fair013_k3_full_q9',      'phase8_qwen_caption2p0_fair013_k3_balanced_full_stage2_200000',      ['--quality_level', '9']),
    ('a3_mfshort100k_direct_noq',    'mfshort100k_direct_noq_stage2_100000',                               ['--no_q']),
]


def jobs():
    """(family, label, exp_id, qflags) in run order: all CFG0 first, then CFG3+neg."""
    out = [('cfg0', *arm) for arm in CORE]
    out += [('cfg3neg', *arm) for arm in CORE + CFG3_ONLY]
    return out


def checkpoint(exp_id):
    return EXPS / exp_id / f'{exp_id}_ema_final.pth'


def source_result(family, label):
    return FAMILIES[family][0] / f'{label}.json'


def result_path(family, label):
    return OUT / family / f'{label}.json'


def audio_dir(family, label):
    return OUT / '_audio' / family / label


def generate(family, exp_id, qflags, target):
    _, cfg, negative = FAMILIES[family]
    target.mkdir(parents=True, exist_ok=True)
    if len(list(target.glob('*.flac'))) >= EXPECTED:
        print('  [skip gen] audio already complete', flush=True)
        return
    cmd = [PYTHON, 'eval.py', '--variant', 'meanaudio_s', '--model_path', str(checkpoint(exp_id)),
           '--output', str(target), '--tsv', str(TSV), '--use_meanflow',
           '--num_steps', '25', '--cfg_strength', cfg]
    if negative is not None:
        cmd += ['--negative_prompt', negative]
    cmd += ['--no_text_attention_mask', '--encoder_name', 't5_clap', '--text_c_dim', '512',
            '--seed', '42', '--full_precision'] + qflags
    t0 = time.time()
    if subprocess.run(cmd, cwd=ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT).returncode:
        raise SystemExit(f'[FAIL] generation failed: {family}/{exp_id}')
    got = len(list(target.glob('*.flac')))
    print(f'  [gen] {got} clips in {(time.time() - t0) / 60:.1f} min', flush=True)
    if got != EXPECTED:
        raise SystemExit(f'[FAIL] {got}/{EXPECTED} clips generated')


def aes_sample_check(rows, target, stored):
    """Score a fixed sample with AES and compare to the stored per-clip values."""
    import torch
    import audiobox_aesthetics.infer as aes_infer

    def read_wav(meta):
        wav, sr = sf.read(meta['path'], dtype='float32', always_2d=True)
        wav = torch.from_numpy(wav.T)
        if wav.shape[0] > 1:
            wav = wav.mean(0, keepdim=True)
        return wav, sr

    aes_infer.read_wav = read_wav
    from audiobox_aesthetics.infer import AesPredictor
    ids = sorted(r['id'] for r in rows)[::len(rows) // AES_SAMPLE][:AES_SAMPLE]
    predictor = AesPredictor(checkpoint_pth=None, batch_size=32)
    worst = 0.0
    for i in range(0, len(ids), 32):
        batch = ids[i:i + 32]
        res = predictor.forward([{'path': str(target / f'{cid}.flac')} for cid in batch])
        for cid, r in zip(batch, res):
            for key in ('CE', 'CU', 'PC', 'PQ'):
                worst = max(worst, abs(float(r[key]) - stored[cid][key]))
    del predictor
    torch.cuda.empty_cache()
    return {'n': len(ids), 'max_abs_diff': worst, 'tolerance': AES_TOLERANCE,
            'passed': worst <= AES_TOLERANCE}


def run_one(rows, family, label, exp_id, qflags):
    src = json.loads(source_result(family, label).read_text())
    stored = src['per_clip']
    if src.get('exp_id') != exp_id or list(src.get('q_flags') or []) != qflags:
        raise SystemExit(f'[FAIL] {family}/{label}: source result identity mismatch')
    target = audio_dir(family, label)
    generate(family, exp_id, qflags, target)
    check = aes_sample_check(rows, target, stored)
    print(f"  [aes check] max|diff| {check['max_abs_diff']:.2e} on {check['n']} clips", flush=True)
    if not check['passed']:
        raise SystemExit(f'[FAIL] {family}/{label}: regenerated audio does not reproduce stored AES')
    per = score_clap_batch1(rows, target)
    if len(per) != EXPECTED:
        raise SystemExit(f'[FAIL] {family}/{label}: CLAP scored {len(per)}/{EXPECTED}')
    ids = [r['id'] for r in rows]
    b1 = float(np.mean([per[i] for i in ids]))
    b32 = float(np.mean([stored[i]['clap'] for i in ids]))
    payload = {
        'family': family, 'label': label, 'exp_id': exp_id, 'q_flags': qflags,
        'cfg_strength': float(FAMILIES[family][1]), 'negative_prompt': FAMILIES[family][2],
        'source_result': str(source_result(family, label)),
        'source_result_sha256': hashlib.sha256(source_result(family, label).read_bytes()).hexdigest(),
        'method': METHOD, 'n': len(per), 'clap_batch1': b1, 'clap_batch32_stored': b32,
        'b32_minus_b1': b32 - b1, 'aes_reproduction_check': check, 'per_clip_clap_batch1': per,
    }
    out = result_path(family, label)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix('.tmp')
    tmp.write_text(json.dumps(payload))
    tmp.replace(out)
    print(f'  CLAP b1 {b1:.4f}  b32 {b32:.4f}  (b32-b1 {b32 - b1:+.4f})', flush=True)
    for f in target.glob('*.flac'):
        f.unlink()
    target.rmdir()


def summarize():
    table, missing = [], []
    for family, label, _, _ in jobs():
        path = result_path(family, label)
        if not path.exists():
            missing.append(f'{family}/{label}')
            continue
        r = json.loads(path.read_text())
        table.append({k: r[k] for k in ('family', 'label', 'exp_id', 'clap_batch1',
                                         'clap_batch32_stored', 'b32_minus_b1')}
                     | {'aes_max_abs_diff': r['aes_reproduction_check']['max_abs_diff']})
    return table, missing


def main():
    validate_only = '--validate-only' in sys.argv[1:]
    if not validate_only:
        rows = load_rows(TSV, EXPECTED)
        for family, label, exp_id, qflags in jobs():
            if result_path(family, label).exists():
                print(f'[done] {family}/{label}', flush=True)
                continue
            print(f'\n=== {family}/{label}  ({time.strftime("%H:%M:%S")})', flush=True)
            run_one(rows, family, label, exp_id, qflags)
    table, missing = summarize()
    if missing:
        print('INVALID: missing results: ' + ', '.join(missing), flush=True)
        return 2
    summary = {'document_kind': 'negprompt_clap_batch1_rescore_summary_v1', 'method': METHOD,
               'n_arms': len(table), 'arms': table}
    tmp = (OUT / 'summary.json').with_suffix('.tmp')
    tmp.write_text(json.dumps(summary, indent=2))
    tmp.replace(OUT / 'summary.json')
    for r in table:
        print(f"{r['family']:8s} {r['label']:30s} b1 {r['clap_batch1']:.4f}  "
              f"b32 {r['clap_batch32_stored']:.4f}  {r['b32_minus_b1']:+.4f}")
    print(f'PASS: {len(table)} arms', flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
