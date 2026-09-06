#!/usr/bin/env python
"""Re-evaluate full-scale Phase-8 arms on MusicCaps with a vocal / no-vocal split.

Generates the complete 5,521-row MusicCaps set once per arm, scores every clip
individually (Audiobox Aesthetics + CLAP), then aggregates the same per-clip
scores three ways: full set, no-vocal subset, vocal subset. Using one generation
for all three aggregations removes the noise-draw confound you would get from
generating the subset separately.

Protocol is the canonical one: MeanFlow 25, CFG 0, NoMask, seed 42, full precision.

Per-clip scores are written to <OUT>/<arm>.json so any later subset question can
be answered without regenerating audio. Generated audio is deleted after scoring.
"""
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path('/home/kojiek/MeanAudio')
EXPS = ROOT / 'exps'
TSV = Path('/mnt/HDD/kojiek/phase4_jamendo_data/musiccaps_test.tsv')
OUT = Path('/home/kojiek/nvme_experiment_artifacts/meanaudio/novocal_reeval')
AUDIO_ROOT = OUT / '_audio'
CLAP_CKPT = ROOT / 'weights/music_speech_audioset_epoch_15_esc_89.98.pt'
PYTHON = '/home/kojiek/venvs/dac/bin/python'
EXPECTED = 5521

VOCAL_RE = re.compile(
    r"\b(vocal|vocals|vocalist|vocalists|vocalisation|vocalization|singer|singers|"
    r"singing|sings|sung|sing|voice|voices|choir|chorus|chant|chanting|rap|rapper|"
    r"rapping|lyric|lyrics|acappella|harmonies|humming|hums|falsetto|soprano|tenor|"
    r"baritone|spoken|speaks|speech|narrate|narration|narrator)\b", re.I)

# (label, exp_id, q flag) — ordered by decision relevance.
ARMS = [
    ('c2p0_slot0_full_noq',        'phase8_qwen_caption10s_multisent_noq_full_stage2_200000',            ['--no_q']),
    ('fulltrack_q3_full_q9',       'phase8_qwen_s2q_from_noq_full_k3_balanced_stage2_200000',            ['--quality_level', '9']),
    ('c2p0_fair013_worst_full',    'phase8_qwen_caption2p0_fair013_worstof3_noq_full_stage2_200000',     ['--no_q']),
    ('c2p0_slot0_q5_full_q9',      'phase8_qwen_caption2p0_s2q_from_noq_full_k5_balanced_stage2_200000', ['--quality_level', '9']),
    ('fulltrack_noq_full',         'phase8_qwen_official_noq_full_stage2_200000',                        ['--no_q']),
    ('c2p0_fair013_best_full',     'phase8_qwen_caption2p0_fair013_bestof3_noq_full_stage2_200000',      ['--no_q']),
    ('c2p0_slot0_q3_full_q9',      'phase8_qwen_caption2p0_s2q_from_noq_full_k3_balanced_stage2_200000', ['--quality_level', '9']),
    ('c2p0_slot2_full_noq',        'phase8_qwen_caption2p0_slot2_noq_full_stage2_200000',                ['--no_q']),
    # Second training seed of c2p0_slot0_full_noq. Same corpus, same schedule, seed
    # 27182818 instead of 14159265 -- this is the only way to measure how much of the
    # spread in the historical arm table is training-seed noise rather than real effect.
    ('c2p0_slot0_full_seed27182818', 'phase8_qwen_caption10s_multisent_noq_full_seed27182818_stage2_200000', ['--no_q']),
    # --quality_level is part of the canonical protocol, so a q sweep on the
    # non-fulltrack Q arms is still a canonical measurement. Only q9 was ever run at
    # CFG 0; q0 is the cell the B-matrix design calls B5 and flags as a secondary
    # PQ>=6.9 target.
    ('c2p0_slot0_q5_full_q0',      'phase8_qwen_caption2p0_s2q_from_noq_full_k5_balanced_stage2_200000', ['--quality_level', '0']),
    ('c2p0_slot0_q3_full_q0',      'phase8_qwen_caption2p0_s2q_from_noq_full_k3_balanced_stage2_200000', ['--quality_level', '0']),
    # Phase 7 V1 full-Q control: LP-MusicCaps corpus, i.e. neither fulltrack nor
    # Caption 2.0. Historically the strongest CLAP model; its AES has never been
    # measured under the canonical CFG 0 protocol.
    ('p7v1_fullq_control_q9',      'phase7_v1_fullq_control_stage2_200000',                              ['--quality_level', '9']),
]


def load_rows():
    import csv
    with TSV.open(encoding='utf-8', newline='') as handle:
        rows = list(csv.DictReader(handle, delimiter='\t'))
    if len(rows) != EXPECTED:
        raise SystemExit(f'[FAIL] MusicCaps rows={len(rows)}/{EXPECTED}')
    return rows


def generate(exp_id, qflags, audio_dir):
    audio_dir.mkdir(parents=True, exist_ok=True)
    have = len(list(audio_dir.glob('*.flac')))
    if have >= EXPECTED:
        print(f'  [skip gen] {have} clips already present')
        return
    ckpt = EXPS / exp_id / f'{exp_id}_ema_final.pth'
    if not ckpt.is_file():
        raise SystemExit(f'[FAIL] missing checkpoint {ckpt}')
    cmd = [PYTHON, 'eval.py', '--variant', 'meanaudio_s', '--model_path', str(ckpt),
           '--output', str(audio_dir), '--tsv', str(TSV), '--use_meanflow',
           '--num_steps', '25', '--cfg_strength', '0.0', '--no_text_attention_mask',
           '--encoder_name', 't5_clap', '--text_c_dim', '512', '--seed', '42',
           '--full_precision'] + qflags
    t0 = time.time()
    proc = subprocess.run(cmd, cwd=ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
    if proc.returncode != 0:
        raise SystemExit(f'[FAIL] generation failed for {exp_id}')
    got = len(list(audio_dir.glob('*.flac')))
    print(f'  [gen] {got} clips in {(time.time()-t0)/60:.1f} min')
    if got < EXPECTED * 0.99:
        raise SystemExit(f'[FAIL] only {got}/{EXPECTED} clips generated')


def score(rows, audio_dir):
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

    present = [(r['id'], r['caption'], audio_dir / f"{r['id']}.flac") for r in rows]
    present = [(i, c, p) for i, c, p in present if p.exists()]
    per = {}

    predictor = AesPredictor(checkpoint_pth=None, batch_size=32)
    for i in range(0, len(present), 32):
        batch = present[i:i + 32]
        res = predictor.forward([{'path': str(p)} for _, _, p in batch])
        for (cid, _, _), r in zip(batch, res):
            per[cid] = {k: float(r[k]) for k in ('CE', 'CU', 'PC', 'PQ')}
    del predictor
    torch.cuda.empty_cache()

    import laion_clap
    model = laion_clap.CLAP_Module(enable_fusion=False, amodel='HTSAT-base')
    model.load_ckpt(str(CLAP_CKPT))
    model = model.eval().cuda()
    with torch.no_grad():
        for i in range(0, len(present), 32):
            batch = present[i:i + 32]
            ae = model.get_audio_embedding_from_filelist([str(p) for _, _, p in batch], use_tensor=True)
            te = model.get_text_embedding([c for _, c, _ in batch], use_tensor=True)
            sim = torch.nn.functional.cosine_similarity(ae, te, dim=-1)
            for (cid, _, _), s in zip(batch, sim):
                per[cid]['clap'] = float(s)
    del model
    torch.cuda.empty_cache()
    return per


def aggregate(per, ids):
    ids = [i for i in ids if i in per]
    keys = ('clap', 'CE', 'CU', 'PC', 'PQ')
    return {'n': len(ids), **{k: float(np.mean([per[i][k] for i in ids])) for k in keys}}


def main():
    only = set(sys.argv[1:])
    rows = load_rows()
    vocal_ids = [r['id'] for r in rows if VOCAL_RE.search(r['caption'])]
    novocal_ids = [r['id'] for r in rows if not VOCAL_RE.search(r['caption'])]
    print(f'MusicCaps {len(rows)}  no-vocal {len(novocal_ids)}  vocal {len(vocal_ids)}')
    OUT.mkdir(parents=True, exist_ok=True)

    for label, exp_id, qflags in ARMS:
        if only and label not in only:
            continue
        result_path = OUT / f'{label}.json'
        if result_path.exists():
            print(f'[done] {label} (result exists)')
            continue
        print(f'\n=== {label}  ({time.strftime("%H:%M:%S")})')
        audio_dir = AUDIO_ROOT / label
        generate(exp_id, qflags, audio_dir)
        per = score(rows, audio_dir)
        payload = {
            'label': label,
            'exp_id': exp_id,
            'q_flags': qflags,
            'protocol': 'MusicCaps 5521; MeanFlow 25; CFG 0; NoMask; seed 42; full precision',
            'tsv': str(TSV),
            'aggregates': {
                'full': aggregate(per, [r['id'] for r in rows]),
                'novocal': aggregate(per, novocal_ids),
                'vocal': aggregate(per, vocal_ids),
            },
            'per_clip': per,
        }
        tmp = result_path.with_suffix('.tmp')
        tmp.write_text(json.dumps(payload))
        tmp.replace(result_path)
        a = payload['aggregates']
        print(f"  full    n={a['full']['n']:4d} CLAP {a['full']['clap']:.4f} CE {a['full']['CE']:.4f} PQ {a['full']['PQ']:.4f}")
        print(f"  novocal n={a['novocal']['n']:4d} CLAP {a['novocal']['clap']:.4f} CE {a['novocal']['CE']:.4f} PQ {a['novocal']['PQ']:.4f}")
        print(f"  vocal   n={a['vocal']['n']:4d} CLAP {a['vocal']['clap']:.4f} CE {a['vocal']['CE']:.4f} PQ {a['vocal']['PQ']:.4f}")
        for f in audio_dir.glob('*.flac'):
            f.unlink()
        audio_dir.rmdir()
        print('  [cleanup] audio removed')

    print('\nALL ARMS DONE')


if __name__ == '__main__':
    main()
