"""Re-score CLAP with batch_size 32 on an existing MusicCaps 5521 audio dir.

The negprompt_reeval family (negprompt_reeval_full_arms.py and its cfg3.0 /
random_full_cfg3 outputs) scores CLAP in batches of 32. phase4_eval.py scores
file by file. laion_clap pads differently above batch 8, so the two disagree by
up to ~0.014 CLAP on the same audio while AES is bit-identical (memory
reference_clap_batch_size_sensitivity). Any CFG3+neg cell produced by
mc_mf25_cfg3neg_eval*.sh / an action's Step 6 must be re-scored here before it is
compared against a negprompt_reeval number (e.g. slot0 full 0.2605).

The CLAP block is copied from negprompt_reeval_full_arms.score(): same TSV row
order, same batch composition, same checkpoint.

Usage: rescore_clap_batch32.py <audio_dir> [<audio_dir> ...]
Writes <audio_dir>/../clap_batch32.json next to each audio dir; skips dirs that
already have one.
"""
import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

ROOT = Path('/home/kojiek/MeanAudio')
TSV = Path('/mnt/HDD/kojiek/phase4_jamendo_data/musiccaps_test.tsv')
CLAP_CKPT = ROOT / 'weights/music_speech_audioset_epoch_15_esc_89.98.pt'
EXPECTED = 5521


def main():
    dirs = [Path(a) for a in sys.argv[1:]]
    with TSV.open(encoding='utf-8', newline='') as handle:
        rows = list(csv.DictReader(handle, delimiter='\t'))
    assert len(rows) == EXPECTED, len(rows)

    import laion_clap
    model = laion_clap.CLAP_Module(enable_fusion=False, amodel='HTSAT-base')
    model.load_ckpt(str(CLAP_CKPT))
    model = model.eval().cuda()

    for audio_dir in dirs:
        out = audio_dir.parent / 'clap_batch32.json'
        if out.exists():
            print(f'[skip] {out}')
            continue
        present = [(r['id'], r['caption'], audio_dir / f"{r['id']}.flac") for r in rows]
        present = [(i, c, p) for i, c, p in present if p.exists()]
        if len(present) != EXPECTED:
            print(f'[FAIL] {audio_dir}: {len(present)}/{EXPECTED} clips')
            continue
        per = {}
        with torch.no_grad():
            for i in range(0, len(present), 32):
                batch = present[i:i + 32]
                ae = model.get_audio_embedding_from_filelist([str(p) for _, _, p in batch], use_tensor=True)
                te = model.get_text_embedding([c for _, c, _ in batch], use_tensor=True)
                sim = torch.nn.functional.cosine_similarity(ae, te, dim=-1)
                for (cid, _, _), s in zip(batch, sim):
                    per[cid] = float(s)
        clap = float(np.mean(list(per.values())))
        out.write_text(json.dumps({
            'audio_dir': str(audio_dir), 'n': len(per), 'clap_batch32': clap,
            'method': 'laion_clap HTSAT-base, batch 32, TSV row order (= negprompt_reeval_full_arms.score)',
            'written_at': datetime.now(timezone.utc).isoformat(), 'per_clip': per,
        }, indent=1) + '\n')
        print(f'{clap:.4f}  {audio_dir.parent.name}', flush=True)


if __name__ == '__main__':
    main()
