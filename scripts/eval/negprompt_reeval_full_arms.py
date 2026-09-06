#!/usr/bin/env python
"""Re-evaluate every full-scale arm on MusicCaps under the negative-prompt protocol.

Secondary (non-canonical) protocol, preregistered here:

    MusicCaps 5521; MeanFlow 25; CFG 1.5; NoMask; seed 42; full precision;
    negative prompt = NEGATIVE_PROMPT below.

This is deliberately NOT the canonical CFG 0 contract, so it does not go through
scripts/caption10s_pipeline/eval_musiccaps_mf25.sh (which hard-refuses any label
that is not *_mf25_cfg0_*). Numbers produced here are not comparable to the
historical CFG 0 table cell for cell; they are only comparable to each other and,
per clip, to the CFG 0 per-clip scores in ../novocal_reeval/<arm>.json.

Rationale: the 512-prompt ablation (2026-08-28) showed roughly 70% of the PQ gain
is generic CFG extrapolation and roughly 30% is fidelity semantics, and that the
same intervention moves c2p0 by +0.67..1.01 PQ but fulltrack by only +0.12..0.18.
Whether the arm ordering survives the protocol change can only be settled by
running every arm, which is what this script does.

Per-clip scores land in <OUT>/<arm>.json; generated audio is deleted after
scoring. Each arm is resumable: an existing result file is skipped, and a partly
generated audio dir is topped up rather than restarted.
"""
import json
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
OUT = Path('/home/kojiek/nvme_experiment_artifacts/meanaudio/negprompt_reeval')
CFG0_REFERENCE = Path('/home/kojiek/nvme_experiment_artifacts/meanaudio/novocal_reeval')
CLAP_CKPT = ROOT / 'weights/music_speech_audioset_epoch_15_esc_89.98.pt'
PYTHON = '/home/kojiek/venvs/dac/bin/python'
EXPECTED = 5521

# Identical to the string used for N1 and for the 5,521-row slot0 confirmation.
NEGATIVE_PROMPT = ('low quality recording, noisy, amateur, distorted, muffled, '
                   'poor fidelity, hiss, lo-fi')
CFG_STRENGTH = '1.5'
PROTOCOL = (f'MusicCaps 5521; MeanFlow 25; CFG {CFG_STRENGTH}; NoMask; seed 42; '
            f'full precision; negative_prompt="{NEGATIVE_PROMPT}"')

VOCAL_RE = re.compile(
    r"\b(vocal|vocals|vocalist|vocalists|vocalisation|vocalization|singer|singers|"
    r"singing|sings|sung|sing|voice|voices|choir|chorus|chant|chanting|rap|rapper|"
    r"rapping|lyric|lyrics|acappella|harmonies|humming|hums|falsetto|soprano|tenor|"
    r"baritone|spoken|speaks|speech|narrate|narration|narrator)\b", re.I)

# Low-fidelity language in the MusicCaps prompt. The per-clip breakdown that
# motivated the fidelity negative prompt keyed on exactly this vocabulary, so the
# same split is reported here: the subset the intervention is supposed to help.
LOFI_RE = re.compile(
    r"\b(low[- ]?quality|poor[- ]?quality|bad[- ]?quality|noisy|noise|amateur|"
    r"amateurish|distort\w*|muffl\w*|lo[- ]?fi|hiss\w*|static|crackl\w*|hum|"
    r"buzz\w*|tinny|dull|muddy|clipp\w*|compress\w*|degrad\w*|artifact\w*)\b", re.I)

# (label, exp_id, q flag) -- same 12 cells as novocal_reeval_full_arms.py, in the
# same order, so the two tables line up row for row.
ARMS = [
    ('c2p0_slot0_full_noq',        'phase8_qwen_caption10s_multisent_noq_full_stage2_200000',            ['--no_q']),
    ('fulltrack_q3_full_q9',       'phase8_qwen_s2q_from_noq_full_k3_balanced_stage2_200000',            ['--quality_level', '9']),
    ('c2p0_fair013_worst_full',    'phase8_qwen_caption2p0_fair013_worstof3_noq_full_stage2_200000',     ['--no_q']),
    ('c2p0_slot0_q5_full_q9',      'phase8_qwen_caption2p0_s2q_from_noq_full_k5_balanced_stage2_200000', ['--quality_level', '9']),
    ('fulltrack_noq_full',         'phase8_qwen_official_noq_full_stage2_200000',                        ['--no_q']),
    ('c2p0_fair013_best_full',     'phase8_qwen_caption2p0_fair013_bestof3_noq_full_stage2_200000',      ['--no_q']),
    ('c2p0_slot0_q3_full_q9',      'phase8_qwen_caption2p0_s2q_from_noq_full_k3_balanced_stage2_200000', ['--quality_level', '9']),
    ('c2p0_slot2_full_noq',        'phase8_qwen_caption2p0_slot2_noq_full_stage2_200000',                ['--no_q']),
    ('c2p0_slot0_full_seed27182818', 'phase8_qwen_caption10s_multisent_noq_full_seed27182818_stage2_200000', ['--no_q']),
    ('c2p0_slot0_q5_full_q0',      'phase8_qwen_caption2p0_s2q_from_noq_full_k5_balanced_stage2_200000', ['--quality_level', '0']),
    ('c2p0_slot0_q3_full_q0',      'phase8_qwen_caption2p0_s2q_from_noq_full_k3_balanced_stage2_200000', ['--quality_level', '0']),
    ('p7v1_fullq_control_q9',      'phase7_v1_fullq_control_stage2_200000',                              ['--quality_level', '9']),
    # Arm 024, still training as this script was written. Skipped with a warning
    # until its EMA exists rather than failing the whole sweep.
    ('c2p0_fair013_k3_full_q9',    'phase8_qwen_caption2p0_fair013_k3_balanced_full_stage2_200000',      ['--quality_level', '9']),
]


def load_rows(tsv, expected):
    import csv
    with tsv.open(encoding='utf-8', newline='') as handle:
        rows = list(csv.DictReader(handle, delimiter='\t'))
    if expected and len(rows) != expected:
        raise SystemExit(f'[FAIL] rows={len(rows)}/{expected} in {tsv}')
    return rows


def generate(exp_id, qflags, audio_dir, tsv, expected):
    audio_dir.mkdir(parents=True, exist_ok=True)
    have = len(list(audio_dir.glob('*.flac')))
    if have >= expected:
        print(f'  [skip gen] {have} clips already present')
        return
    ckpt = EXPS / exp_id / f'{exp_id}_ema_final.pth'
    if not ckpt.is_file():
        raise SystemExit(f'[FAIL] missing checkpoint {ckpt}')
    cmd = [PYTHON, 'eval.py', '--variant', 'meanaudio_s', '--model_path', str(ckpt),
           '--output', str(audio_dir), '--tsv', str(tsv), '--use_meanflow',
           '--num_steps', '25', '--cfg_strength', CFG_STRENGTH,
           '--negative_prompt', NEGATIVE_PROMPT,
           '--no_text_attention_mask',
           '--encoder_name', 't5_clap', '--text_c_dim', '512', '--seed', '42',
           '--full_precision'] + qflags
    t0 = time.time()
    proc = subprocess.run(cmd, cwd=ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
    if proc.returncode != 0:
        raise SystemExit(f'[FAIL] generation failed for {exp_id}')
    got = len(list(audio_dir.glob('*.flac')))
    print(f'  [gen] {got} clips in {(time.time()-t0)/60:.1f} min')
    if got < expected * 0.99:
        raise SystemExit(f'[FAIL] only {got}/{expected} clips generated')


def check_saturation(audio_dir, sample=64):
    """cfg >= 1.0 is the branch that historically saturated waveforms, so record
    crest factor and clipping rather than assuming the 512-prompt check carries."""
    paths = sorted(audio_dir.glob('*.flac'))[:sample]
    crests, clipped = [], 0
    for path in paths:
        wav, _ = sf.read(path, dtype='float32', always_2d=True)
        wav = wav.mean(axis=1)
        rms = float(np.sqrt(np.mean(wav ** 2)))
        peak = float(np.max(np.abs(wav)))
        if rms > 0:
            crests.append(peak / rms)
        if peak >= 0.999:
            clipped += 1
    return {
        'n_sampled': len(paths),
        'crest_mean': float(np.mean(crests)) if crests else None,
        'crest_min': float(np.min(crests)) if crests else None,
        'clipped_fraction': clipped / len(paths) if paths else None,
    }


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

    # batch_size 32 to match novocal_reeval exactly; CLAP is batch-size sensitive
    # at roughly the same magnitude as the between-arm gaps.
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


def paired_delta(label, per):
    """Per-clip delta against the CFG 0 run of the same arm, same prompts and seed."""
    ref_path = CFG0_REFERENCE / f'{label}.json'
    if not ref_path.exists():
        return None
    ref = json.loads(ref_path.read_text()).get('per_clip', {})
    shared = [i for i in per if i in ref]
    if not shared:
        return None
    out = {'n_paired': len(shared)}
    for key in ('clap', 'CE', 'CU', 'PC', 'PQ'):
        diffs = np.array([per[i][key] - ref[i][key] for i in shared])
        out[key] = {
            'mean_delta': float(diffs.mean()),
            'sd': float(diffs.std(ddof=1)) if len(diffs) > 1 else None,
            'frac_improved': float((diffs > 0).mean()),
        }
    return out


def main():
    argv = [a for a in sys.argv[1:] if not a.startswith('--')]
    smoke = '--smoke' in sys.argv[1:]

    # --cfg N reruns the whole sweep at a different guidance strength, into its
    # own output dir so the CFG 1.5 table is never overwritten. The 1024-row
    # ablation matrix put the PQ optimum at 3.0 rather than 1.5 on c2p0_slot0,
    # and showed fulltrack degrading (crest_min 1.72) at the same point, so
    # whether the arm ordering is cfg-dependent needs the full set at 3.0 too.
    default_cfg = CFG_STRENGTH
    cfg = default_cfg
    for a in sys.argv[1:]:
        if a.startswith('--cfg='):
            cfg = a.split('=', 1)[1]
    globals()['CFG_STRENGTH'] = cfg
    globals()['PROTOCOL'] = PROTOCOL.replace(f'CFG {default_cfg}', f'CFG {cfg}')

    tsv, expected, out_dir = TSV, EXPECTED, OUT
    if cfg != default_cfg:
        out_dir = OUT.with_name(f'negprompt_reeval_cfg{cfg}')
        print(f'[cfg {cfg}] out={out_dir}')
    if smoke:
        tsv = Path('/tmp/claude-1005/-home-kojiek-MeanAudio/negprompt_smoke.tsv')
        expected = len(load_rows(tsv, 0))
        out_dir = OUT.with_name('negprompt_reeval_smoke')
        globals()['PROTOCOL'] = PROTOCOL.replace('MusicCaps 5521', f'SMOKE {expected} rows')
        print(f'[smoke] tsv={tsv} rows={expected} out={out_dir}')

    only = {a for a in argv if not a.startswith('--')}
    rows = load_rows(tsv, expected if not smoke else 0)
    vocal_ids = [r['id'] for r in rows if VOCAL_RE.search(r['caption'])]
    novocal_ids = [r['id'] for r in rows if not VOCAL_RE.search(r['caption'])]
    lofi_ids = [r['id'] for r in rows if LOFI_RE.search(r['caption'])]
    clean_ids = [r['id'] for r in rows if not LOFI_RE.search(r['caption'])]
    print(f'{tsv.name} {len(rows)}  no-vocal {len(novocal_ids)}  vocal {len(vocal_ids)}  '
          f'lofi-prompt {len(lofi_ids)}  clean-prompt {len(clean_ids)}')
    print(f'protocol: {PROTOCOL}')
    out_dir.mkdir(parents=True, exist_ok=True)

    for label, exp_id, qflags in ARMS:
        if only and label not in only:
            continue
        if not (EXPS / exp_id / f'{exp_id}_ema_final.pth').is_file():
            print(f'[skip] {label}: no EMA yet ({exp_id})')
            continue
        result_path = out_dir / f'{label}.json'
        if result_path.exists():
            print(f'[done] {label} (result exists)')
            continue
        print(f'\n=== {label}  ({time.strftime("%H:%M:%S")})')
        audio_dir = (out_dir / '_audio') / label
        generate(exp_id, qflags, audio_dir, tsv, expected)
        saturation = check_saturation(audio_dir)
        per = score(rows, audio_dir)
        payload = {
            'label': label,
            'exp_id': exp_id,
            'q_flags': qflags,
            'protocol': PROTOCOL,
            'cfg_strength': float(CFG_STRENGTH),
            'negative_prompt': NEGATIVE_PROMPT,
            'tsv': str(tsv),
            'saturation_check': saturation,
            'aggregates': {
                'full': aggregate(per, [r['id'] for r in rows]),
                'novocal': aggregate(per, novocal_ids),
                'vocal': aggregate(per, vocal_ids),
                'lofi_prompt': aggregate(per, lofi_ids),
                'clean_prompt': aggregate(per, clean_ids),
            },
            'paired_delta_vs_cfg0': paired_delta(label, per) if not smoke else None,
            'per_clip': per,
        }
        tmp = result_path.with_suffix('.tmp')
        tmp.write_text(json.dumps(payload))
        tmp.replace(result_path)
        a = payload['aggregates']
        for name in ('full', 'novocal', 'vocal', 'lofi_prompt', 'clean_prompt'):
            g = a[name]
            print(f"  {name:12s} n={g['n']:4d} CLAP {g['clap']:.4f} CE {g['CE']:.4f} PQ {g['PQ']:.4f}")
        print(f"  saturation crest_mean {saturation['crest_mean']} "
              f"crest_min {saturation['crest_min']} clipped {saturation['clipped_fraction']}")
        d = payload['paired_delta_vs_cfg0']
        if d:
            print(f"  vs CFG0 (n={d['n_paired']}): PQ {d['PQ']['mean_delta']:+.4f} "
                  f"({d['PQ']['frac_improved']*100:.1f}% improved)  "
                  f"CLAP {d['clap']['mean_delta']:+.4f}  CE {d['CE']['mean_delta']:+.4f}")
        for f in audio_dir.glob('*.flac'):
            f.unlink()
        audio_dir.rmdir()
        print('  [cleanup] audio removed')

    print('\nALL ARMS DONE')


if __name__ == '__main__':
    main()
