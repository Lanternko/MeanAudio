"""Ablation matrix over CFG strength and negative-prompt content.

Two questions the 12-arm negprompt_reeval sweep left open:

  1. How much of the gain is CFG itself rather than the negative text? The
     sweep only ran one point, cfg 1.5 with one fidelity-worded negative, so
     "guidance" and "this particular wording" are fully confounded there.
  2. Which negative wording is actually best, and does the answer depend on
     the arm? Our current wording is keyed on LP-MusicCaps boilerplate
     vocabulary and directly contradicts the 35.7% of MusicCaps prompts that
     themselves ask for low-fidelity audio, which is a metric-gaming risk.

The N_NONE cell is the control that settles (1): cfg >= 1.0 with the network's
stored empty_string_feat, i.e. textbook classifier-free guidance and the
upstream author's intended inference path. It is only reachable after the
eval.py fix that maps an empty --negative_prompt to negative_text=None; passing
[''] instead T5-encodes the empty string at inference, which is NOT the null
condition training used (whole-tensor cosine -0.158 against the stored feature).

Protocol: MusicCaps 1024-row seeded subset, MeanFlow 25 steps, NoMask, seed 42,
full precision, CLAP batch 32 (matches novocal_reeval / negprompt_reeval; CLAP
is batch-size sensitive at ~the same magnitude as between-arm gaps, so this is
comparable within this matrix and with those two sweeps, and NOT with any
per-file CLAP number).

Deltas are per-clip paired against the cfg 0 cell of the SAME arm on the SAME
subset, so nothing here is compared across subsets.

Resumable: an existing <cell>.json is skipped. Audio is deleted after scoring.
"""
import csv
import json
import random
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path('/home/kojiek/MeanAudio')
PYTHON = '/home/kojiek/venvs/dac/bin/python'
EXPS = Path('/home/kojiek/exps_nvme')
FULL_TSV = Path('/mnt/HDD/kojiek/phase4_jamendo_data/musiccaps_test.tsv')
OUT = Path('/home/kojiek/nvme_experiment_artifacts/meanaudio/negprompt_ablation')
SUBSET_TSV = OUT / 'musiccaps_subset1024.tsv'
AUDIO_ROOT = OUT / '_audio'
CLAP_CKPT = ROOT / 'weights/music_speech_audioset_epoch_15_esc_89.98.pt'

SUBSET_N = 1024
SUBSET_SEED = 20260830

# ---------------------------------------------------------------- negatives --
# N_NONE is a sentinel, not a string: it selects the stored null condition.
N_NONE = None
NEGATIVES = {
    # the control: pure CFG, no negative text at all
    'none': N_NONE,
    # what negprompt_reeval used, kept verbatim so the matrix ties back to it
    'fidelity': ('low quality recording, noisy, amateur, distorted, muffled, '
                 'poor fidelity, hiss, lo-fi'),
    # same direction, far fewer tokens: is the long list doing any work?
    'fidelity_short': 'low quality, noisy',
    # semantically contentful but fidelity-neutral
    'neutral': 'music',
    # off-domain: isolates "any text in the negative slot" from fidelity meaning
    'irrelevant': 'a photograph of a cat, a spreadsheet, printed text',
    # the direction check: pushing AWAY from high quality should not help
    'reversed': 'high quality recording, clean, professional, pristine, hi-fi',
    # non-fidelity audio failure mode
    'silence': 'silence, empty track, no sound',
}

ARMS = {
    'c2p0_slot0': ('phase8_qwen_caption10s_multisent_noq_full_stage2_200000', ['--no_q']),
    'fulltrack':  ('phase8_qwen_official_noq_full_stage2_200000', ['--no_q']),
}

CFGS_MAIN = ['1.0', '1.25', '1.5', '2.0', '3.0', '4.5']


def build_cells():
    """Ordered most-informative-first so a partial run still answers something."""
    cells = []
    # cfg 0 baselines on this subset -- every paired delta needs these
    for arm in ARMS:
        cells.append((f'{arm}__cfg0__none', arm, '0.0', 'none'))
    # Q1: CFG-only vs CFG+fidelity, across cfg, on both arms
    for cfg in CFGS_MAIN:
        for neg in ('none', 'fidelity'):
            for arm in ARMS:
                cells.append((f'{arm}__cfg{cfg}__{neg}', arm, cfg, neg))
    # Q2: negative content at cfg 1.5
    for neg in ('fidelity_short', 'neutral', 'irrelevant', 'reversed', 'silence'):
        for arm in ARMS:
            cells.append((f'{arm}__cfg1.5__{neg}', arm, '1.5', neg))
    return cells


def make_subset():
    if SUBSET_TSV.exists():
        return
    OUT.mkdir(parents=True, exist_ok=True)
    with FULL_TSV.open(encoding='utf-8', newline='') as fh:
        reader = csv.DictReader(fh, delimiter='\t')
        fields, rows = reader.fieldnames, list(reader)
    rng = random.Random(SUBSET_SEED)
    picked = rng.sample(rows, SUBSET_N)
    with SUBSET_TSV.open('w', encoding='utf-8', newline='') as fh:
        writer = csv.DictWriter(fh, fieldnames=fields, delimiter='\t')
        writer.writeheader()
        writer.writerows(picked)
    print(f'[subset] wrote {len(picked)} rows seed={SUBSET_SEED} -> {SUBSET_TSV}')


def load_rows():
    with SUBSET_TSV.open(encoding='utf-8', newline='') as fh:
        return list(csv.DictReader(fh, delimiter='\t'))


def generate(exp_id, qflags, cfg, neg_key, audio_dir):
    audio_dir.mkdir(parents=True, exist_ok=True)
    if len(list(audio_dir.glob('*.flac'))) >= SUBSET_N:
        print('  [skip gen] already present')
        return
    ckpt = EXPS / exp_id / f'{exp_id}_ema_final.pth'
    if not ckpt.is_file():
        raise SystemExit(f'[FAIL] missing checkpoint {ckpt}')
    cmd = [PYTHON, 'eval.py', '--variant', 'meanaudio_s', '--model_path', str(ckpt),
           '--output', str(audio_dir), '--tsv', str(SUBSET_TSV), '--use_meanflow',
           '--num_steps', '25', '--cfg_strength', cfg,
           '--no_text_attention_mask', '--encoder_name', 't5_clap',
           '--text_c_dim', '512', '--seed', '42', '--full_precision'] + qflags
    neg = NEGATIVES[neg_key]
    if neg is not None:
        cmd += ['--negative_prompt', neg]
    t0 = time.time()
    proc = subprocess.run(cmd, cwd=ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
    if proc.returncode != 0:
        raise SystemExit(f'[FAIL] generation failed: {exp_id} cfg={cfg} neg={neg_key}')
    got = len(list(audio_dir.glob('*.flac')))
    print(f'  [gen] {got} clips in {(time.time() - t0) / 60:.1f} min')
    if got < SUBSET_N * 0.99:
        raise SystemExit(f'[FAIL] only {got}/{SUBSET_N} clips')


def signal_stats(audio_dir, sample=256):
    """Guard against the failure mode where PQ rises because the audio merely got
    louder or brighter. Audiobox PQ is not level-invariant in practice, so a cell
    that gains PQ while also gaining several dB of RMS and a large spectral
    centroid shift has not been shown to gain *quality*. Also keeps the crest /
    clipping check, since cfg 3.0 and 4.5 are where saturation was seen before."""
    paths = sorted(audio_dir.glob('*.flac'))[:sample]
    crests, rmss, centroids = [], [], []
    clipped = 0
    for path in paths:
        wav, sr = sf.read(path, dtype='float32', always_2d=True)
        wav = wav.mean(axis=1)
        rms = float(np.sqrt(np.mean(wav ** 2)))
        peak = float(np.max(np.abs(wav)))
        if rms > 0:
            crests.append(peak / rms)
            rmss.append(20.0 * np.log10(rms))
        if peak >= 0.999:
            clipped += 1
        spec = np.abs(np.fft.rfft(wav * np.hanning(len(wav))))
        freqs = np.fft.rfftfreq(len(wav), 1.0 / sr)
        total = spec.sum()
        if total > 0:
            centroids.append(float((spec * freqs).sum() / total))
    return {'n_sampled': len(paths),
            'crest_mean': float(np.mean(crests)) if crests else None,
            'crest_min': float(np.min(crests)) if crests else None,
            'clipped_fraction': clipped / len(paths) if paths else None,
            'rms_db_mean': float(np.mean(rmss)) if rmss else None,
            'rms_db_sd': float(np.std(rmss)) if rmss else None,
            'spectral_centroid_hz_mean': float(np.mean(centroids)) if centroids else None}


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


LOFI_TERMS = ('low quality', 'noisy', 'poor', 'amateur', 'lo-fi', 'muffled', 'distorted')


def aggregate(per, rows):
    keys = ('clap', 'CE', 'CU', 'PC', 'PQ')

    def agg(ids):
        ids = [i for i in ids if i in per]
        return {'n': len(ids), **{k: float(np.mean([per[i][k] for i in ids])) for k in keys}}

    lofi = [r['id'] for r in rows if any(t in r['caption'].lower() for t in LOFI_TERMS)]
    clean = [r['id'] for r in rows if r['id'] not in set(lofi)]
    return {'full': agg([r['id'] for r in rows]),
            'lofi_prompt': agg(lofi),
            'clean_prompt': agg(clean)}, lofi


def paired_delta(arm, per):
    ref_path = OUT / f'{arm}__cfg0__none.json'
    if not ref_path.exists():
        return None
    ref = json.loads(ref_path.read_text()).get('per_clip', {})
    shared = [i for i in per if i in ref]
    if not shared:
        return None
    out = {'n_paired': len(shared)}
    for key in ('clap', 'CE', 'CU', 'PC', 'PQ'):
        d = np.array([per[i][key] - ref[i][key] for i in shared])
        out[key] = {'mean_delta': float(d.mean()),
                    'sd': float(d.std(ddof=1)) if len(d) > 1 else None,
                    'frac_improved': float((d > 0).mean())}
    return out


def main():
    # --smoke runs two real cells end to end (generate AND score) on a handful of
    # rows, in a throwaway output dir. The first launch of this matrix died after
    # generation on a wrong CLAP checkpoint path because the pre-flight check only
    # exercised eval.py; anything that only tests generation will miss that again.
    smoke = '--smoke' in sys.argv
    global SUBSET_N, SUBSET_TSV, OUT, AUDIO_ROOT
    if smoke:
        OUT = OUT.with_name('negprompt_ablation_smoke')
        AUDIO_ROOT = OUT / '_audio'
        SUBSET_TSV = OUT / 'smoke_subset.tsv'
        SUBSET_N = 8

    if not CLAP_CKPT.is_file():
        raise SystemExit(f'[FAIL] missing CLAP checkpoint {CLAP_CKPT}')
    for arm, (exp_id, _) in ARMS.items():
        ckpt = EXPS / exp_id / f'{exp_id}_ema_final.pth'
        if not ckpt.is_file():
            raise SystemExit(f'[FAIL] missing checkpoint for {arm}: {ckpt}')
    print('[preflight] CLAP checkpoint and all arm checkpoints present')

    make_subset()
    rows = load_rows()
    cells = build_cells()
    if smoke:
        cells = [c for c in cells if c[0].startswith('c2p0_slot0__cfg0')
                 or c[0] == 'c2p0_slot0__cfg1.5__fidelity']
    print(f'{len(cells)} cells, {len(rows)} rows each')
    for label, arm, cfg, neg_key in cells:
        result_path = OUT / f'{label}.json'
        if result_path.exists():
            print(f'[skip] {label}')
            continue
        exp_id, qflags = ARMS[arm]
        print(f'[cell] {label}')
        audio_dir = AUDIO_ROOT / label
        generate(exp_id, qflags, cfg, neg_key, audio_dir)
        sat = signal_stats(audio_dir)
        per = score(rows, audio_dir)
        aggs, lofi = aggregate(per, rows)
        payload = {'label': label, 'arm': arm, 'exp_id': exp_id,
                   'cfg_strength': float(cfg), 'negative_key': neg_key,
                   'negative_prompt': NEGATIVES[neg_key],
                   'subset': {'tsv': str(SUBSET_TSV), 'n': SUBSET_N, 'seed': SUBSET_SEED},
                   'signal_stats': sat,
                   'aggregates': aggs,
                   'lofi_ids': lofi,
                   'paired_delta_vs_cfg0': paired_delta(arm, per),
                   'per_clip': per}
        result_path.write_text(json.dumps(payload, indent=1))
        shutil.rmtree(audio_dir, ignore_errors=True)
        a = payload['aggregates']['full']
        print(f"  PQ {a['PQ']:.4f}  CLAP {a['clap']:.4f}  crest_min {sat['crest_min']:.2f}")
    print('\nALL CELLS DONE')


if __name__ == '__main__':
    sys.exit(main())
