"""Do MIR features add to AES in predicting human quality ratings of generated music?

Design + pre-registered gates: docs/experiments/mir_incremental_validity_musiceval_20260930.md

Stages (run in ~/venvs/dac):
  manifest  write <out>/manifest.tsv (set, key, path) for the feature extractors:
              musiceval  2,748 MusicEval clips (16 kHz mono, 5 expert raters, OVL/REL 1-5)
              pam        PAM human_eval music (4 TTM systems + real, 5 s)
              aesnat     AES-natural MusicCaps clips with human AES ratings (522)
              mc_*       1,000 MusicCaps prompts: real reference + nmv2 arm CFG0 / CFG3+neg
  aes       AES four axes + LUFS + duration for MusicEval -> <out>/musiceval_aes.tsv
            (PAM / AES-natural / MusicCaps arms reuse their existing per-clip AES)
  analyze   everything in the design doc -> <out>/summary.{txt,json}

Feature extractors (separate envs): mir_features_madmom.py, mir_features_essentia.py.
"""
import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / 'scripts' / 'eval'))

ME_ROOT = HERE / 'musiceval'
PAM_ROOT = HERE / 'pam_human_eval' / 'human_eval' / 'music'
REF_DIR = Path('/mnt/HDD/kojiek/musiccaps_reference')
EVAL_NVME = Path.home() / 'eval_output_nvme'
NMV2 = 'phase8_qwen_caption2p0_slot0nmv2_nmv2pair_noq_quarter_s14159265_mc_mf25_'
MC_ARMS = {
    'mc_real': (REF_DIR, '.wav', EVAL_NVME / 'musiccaps_reference_real' / 'musiccaps_reference_real'),
    'mc_nmv2_cfg0': (EVAL_NVME / f'{NMV2}cfg0' / 'audio', '.flac', EVAL_NVME / f'{NMV2}cfg0' / f'{NMV2}cfg0'),
    'mc_nmv2_cfg3neg': (EVAL_NVME / f'{NMV2}cfg3_neg' / 'audio', '.flac',
                        EVAL_NVME / f'{NMV2}cfg3_neg' / f'{NMV2}cfg3_neg'),
}
AXES = ['PQ', 'PC', 'CE', 'CU']
MADMOM = ['beat_act', 'pulse_clarity', 'ibi_cv', 'tempo_drift', 'downbeat_act', 'key_cnn_conf']
ESSENTIA = ['rhythm_conf', 'danceability', 'key_strength', 'dissonance', 'key_stability', 'chroma_entropy']
MIR = MADMOM + ESSENTIA
NUIS = ['log_dur', 'lufs']


def read_tsv(p):
    return list(csv.DictReader(open(p), delimiter='\t'))


def fnum(x):
    try:
        v = float(x)
        return v if np.isfinite(v) else np.nan
    except (TypeError, ValueError):
        return np.nan


# ── manifest ────────────────────────────────────────────
def stage_manifest(out):
    rows = []
    for line in open(ME_ROOT / 'sets' / 'total_mos_list.txt'):
        f = line.split(',')[0]
        rows.append(('musiceval', f[:-4], ME_ROOT / 'wav' / f))
    for r in read_tsv(HERE / 'output' / 'aes_human_corr_pam' / 'per_clip.tsv'):
        sysname, ytid = r['key'].split('__', 1)
        rows.append(('pam', r['key'], PAM_ROOT / sysname / f'{ytid}.wav'))
    by_ytid = {f.stem.rsplit('_', 1)[0]: f for f in REF_DIR.glob('*.wav')}
    for r in read_tsv(HERE / 'output' / 'aes_human_corr' / 'per_clip.tsv'):
        rows.append(('aesnat', r['ytid'], by_ytid[r['ytid']]))
    common = None
    for s, (d, ext, _) in MC_ARMS.items():
        ids = {p.stem for p in d.glob(f'*{ext}')}
        common = ids if common is None else common & ids
    ids = sorted(common)
    pick = sorted(np.random.default_rng(0).choice(ids, 1000, replace=False))
    for s, (d, ext, _) in MC_ARMS.items():
        rows += [(s, i, d / f'{i}{ext}') for i in pick]
    missing = [r for r in rows if not Path(r[2]).exists()]
    if missing:
        raise SystemExit(f'[FAIL] {len(missing)} missing, e.g. {missing[:3]}')
    with open(out / 'manifest.tsv', 'w') as f:
        f.write('set\tkey\tpath\n')
        for s, k, p in rows:
            f.write(f'{s}\t{k}\t{p}\n')
    from collections import Counter
    print(Counter(r[0] for r in rows), f'(MusicCaps common pool {len(ids)})')


# ── AES for MusicEval ───────────────────────────────────
def stage_aes(out):
    import soundfile as sf
    from eval_metrics import score_aes, _level_one
    rows = [r for r in read_tsv(out / 'manifest.tsv') if r['set'] == 'musiceval']
    per, failed = score_aes([r['path'] for r in rows])
    if failed:
        # MusicEval has one 349 s demo clip (S013_P013; next longest 87 s) that does not fit
        # in one AES forward; score its first 90 s instead and say so in the results doc.
        import tempfile
        tmp = Path(tempfile.mkdtemp())
        crops = {}
        for p, e in failed.items():
            w, sr = sf.read(p, dtype='float32')
            if len(w) <= 90 * sr:
                raise SystemExit(f'[FAIL] AES failed on {p}: {e}')
            crops[str(tmp / Path(p).name)] = p
            sf.write(str(tmp / Path(p).name), w[:90 * sr], sr)
            print(f'[WARN] AES on first 90 s of {Path(p).name} ({len(w) / sr:.0f} s): {e}')
        per2, failed2 = score_aes(list(crops), batch_size=1)
        if failed2:
            raise SystemExit(f'[FAIL] AES failed on cropped clips: {failed2}')
        per.update({crops[c]: v for c, v in per2.items()})
    with open(out / 'musiceval_aes.tsv', 'w') as f:
        f.write('key\t' + '\t'.join(AXES) + '\tlufs\tdur\n')
        for r in rows:
            lv = _level_one(r['path'])
            dur = sf.info(r['path']).duration
            f.write(f"{r['key']}\t" + '\t'.join(f"{per[r['path']][a]:.4f}" for a in AXES)
                    + f"\t{lv['lufs']:.3f}\t{dur:.3f}\n")
    print(f'AES done: {len(rows)} clips')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('stage', choices=['manifest', 'aes', 'analyze'])
    ap.add_argument('--out_dir', default=str(HERE / 'output' / 'musiceval_mir'))
    args = ap.parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    if args.stage == 'manifest':
        stage_manifest(out)
    elif args.stage == 'aes':
        stage_aes(out)
    else:
        from musiceval_mir_analysis import analyze
        analyze(out)


if __name__ == '__main__':
    main()
