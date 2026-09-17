#!/usr/bin/env python
"""Re-score CLAP one file at a time (batch 1) on a MusicCaps 5521 audio dir.

This is the phase4_eval.py::compute_clap_score path: laion_clap HTSAT-base,
get_audio_embedding_from_filelist on a single file, cosine against the caption.
The negprompt_reeval / novocal_reeval family scored CLAP in batches of 32, and
laion_clap pads differently above batch 8, so those numbers sit +0.004..+0.025
above the per-file ones and can flip rankings (memory
reference_clap_batch_size_sensitivity). Batch 1 is the reference: batch 8 agrees
with it to 2e-5, batch 32 does not.

Usage: rescore_clap_batch1.py <audio_dir> [<audio_dir> ...]
Writes <audio_dir>/../clap_batch1.json (mean + per-clip) next to each audio dir;
skips dirs that already have one. The library entry point score_clap_batch1() is
reused by the queued negprompt rescore job.
"""
import csv
import json
import sys
from pathlib import Path

TSV = Path('/mnt/HDD/kojiek/phase4_jamendo_data/musiccaps_test.tsv')
CLAP_CKPT = Path('/home/kojiek/MeanAudio/weights/music_speech_audioset_epoch_15_esc_89.98.pt')
EXPECTED = 5521
METHOD = 'laion_clap HTSAT-base, batch 1 (= phase4_eval.compute_clap_score), TSV row order'


def load_rows(tsv=TSV, expected=EXPECTED):
    with tsv.open(encoding='utf-8', newline='') as handle:
        rows = list(csv.DictReader(handle, delimiter='\t'))
    if expected and len(rows) != expected:
        raise SystemExit(f'[FAIL] rows={len(rows)}/{expected} in {tsv}')
    return rows


def load_model():
    import laion_clap
    model = laion_clap.CLAP_Module(enable_fusion=False, amodel='HTSAT-base')
    model.load_ckpt(str(CLAP_CKPT))
    return model.eval().cuda()


def score_clap_batch1(rows, audio_dir, model=None):
    """Per-clip CLAP for every row whose <id>.flac exists. Returns {id: score}."""
    import torch
    owned = model is None
    if owned:
        model = load_model()
    per = {}
    with torch.no_grad():
        for r in rows:
            path = Path(audio_dir) / f"{r['id']}.flac"
            if not path.exists():
                continue
            ae = model.get_audio_embedding_from_filelist([str(path)], use_tensor=True)
            te = model.get_text_embedding([r['caption']], use_tensor=True)
            per[r['id']] = float(torch.nn.functional.cosine_similarity(ae, te, dim=-1).item())
    if owned:
        del model
        torch.cuda.empty_cache()
    return per


def main():
    import numpy as np
    rows = load_rows()
    model = None
    for arg in sys.argv[1:]:
        audio_dir = Path(arg)
        out = audio_dir.parent / 'clap_batch1.json'
        if out.exists():
            print(f'[skip] {out} exists')
            continue
        if model is None:
            model = load_model()
        per = score_clap_batch1(rows, audio_dir, model)
        if len(per) != EXPECTED:
            raise SystemExit(f'[FAIL] {audio_dir}: {len(per)}/{EXPECTED} clips present')
        clap = float(np.mean(list(per.values())))
        tmp = out.with_suffix('.tmp')
        tmp.write_text(json.dumps({'audio_dir': str(audio_dir), 'n': len(per), 'clap_batch1': clap,
                                   'method': METHOD, 'per_clip': per}))
        tmp.replace(out)
        print(f'{clap:.4f}  n={len(per)}  {audio_dir}')


if __name__ == '__main__':
    main()
