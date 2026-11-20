#!/usr/bin/env python3
"""191 Part A helper: htdemucs stem energy fractions on CPU (2026-11-11).

Run with the venv that has demucs (~/venvs/karaoke-jp-align), CUDA hidden:
  CUDA_VISIBLE_DEVICES= ~/venvs/karaoke-jp-align/bin/python scripts/analysis/ce_human_residual_stems_20261111.py
Writes <cache>/stems_{mc,pam}.tsv: key, frac_drums, frac_bass, frac_other, frac_vocals
(fraction of summed stem energy; mono input duplicated to stereo, resampled to 44.1 kHz).
"""
import csv
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
EV = REPO / 'research/eval'
CACHE = Path.home() / 'eval_output_nvme/ce_human_residual_20261111'


def items():
    mc = {r['ytid'] for r in csv.DictReader(open(EV / 'output/aes_human_corr/per_clip.tsv'), delimiter='\t')}
    out = {'mc': {}, 'pam': {}}
    for f in sorted(Path('/mnt/HDD/kojiek/musiccaps_reference').glob('*.wav')):
        y = f.stem.rsplit('_', 1)[0]
        if y in mc:
            out['mc'][y] = f
    keys = {r['key'] for r in csv.DictReader(open(EV / 'output/aes_human_corr_pam/per_clip.tsv'), delimiter='\t')}
    for line in open(EV / 'aes_human_ratings/AES_PAM.jsonl'):
        rel = json.loads(line)['data_path'].split('/your_path/PAM/', 1)[1]
        p = rel.split('/')
        k = f'{p[2]}__{Path(p[-1]).stem}'
        if p[1] == 'music' and k in keys:
            out['pam'][k] = EV / 'pam_human_eval' / rel
    return out


def main():
    import soundfile as sf
    import torch
    import torchaudio
    from demucs.apply import apply_model
    from demucs.pretrained import get_model
    CACHE.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(16)
    model = get_model('htdemucs').eval()
    for name, d in items().items():
        dst = CACHE / f'stems_{name}.tsv'
        if dst.exists():
            continue
        rows = []
        for n, (k, f) in enumerate(d.items()):
            x, sr = sf.read(f, dtype='float32', always_2d=True)
            x = torch.from_numpy(x.T)
            if x.shape[0] == 1:
                x = x.repeat(2, 1)
            x = torchaudio.functional.resample(x[:2], sr, model.samplerate)
            ref = x.mean(0)
            xn = (x - ref.mean()) / (ref.std() + 1e-8)
            with torch.no_grad():
                s = apply_model(model, xn[None], shifts=0, split=True, overlap=0.25, device='cpu')[0]
            e = (s ** 2).sum((1, 2)).numpy()
            fr = e / max(e.sum(), 1e-12)
            rows.append([k] + [f'{fr[model.sources.index(t)]:.5f}' for t in ('drums', 'bass', 'other', 'vocals')])
            if n % 50 == 0:
                print(name, n, len(d), flush=True)
        tmp = dst.with_suffix('.tmp')
        with open(tmp, 'w') as fh:
            fh.write('key\tfrac_drums\tfrac_bass\tfrac_other\tfrac_vocals\n')
            fh.writelines('\t'.join(r) + '\n' for r in rows)
        tmp.rename(dst)
        print(name, 'done', len(rows), flush=True)


if __name__ == '__main__':
    sys.exit(main())
