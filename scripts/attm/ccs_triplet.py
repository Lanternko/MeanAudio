#!/usr/bin/env python3
"""CCS over synthesised tag triplets -- ATTM Eq. 1-2 in its native form.

ccs.py had to take target concepts from MusicCaps' human aspect_list because
the prompts were natural captions. With synthesised triplets each clip has
exactly the 3 concepts it was asked for, which is what Eq. 2's 1/3N assumes:

    CCS = (1/3N) * sum_i sum_{t in T_i} D(x_i, t)

Reported three ways:
  raw          every triplet tag, no filter -- closest to the paper's Eq. 2
  verifiable   only tags that passed ATTM criterion 2 (recall >= 0.85), so a
               tag the judge cannot hear even when truly present is not
               charged against the generator
  corrected    verifiable, then chance-corrected (rate - fpr)/(1 - fpr) with
               the per-tag false-positive rates from ccs_specificity.json,
               closing the yes-bias hole documented on 2026-09-04
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent))
from ccs import ATTM, CANON, Judge, TAXONOMY_PATH, load_audio

TRIPLETS = ATTM / 'tag_triplets.json'
SPEC = ATTM / 'ccs_specificity.json'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--audio-dir', type=Path, required=True)
    ap.add_argument('--label', default=None)
    ap.add_argument('--limit', type=int, default=0)
    args = ap.parse_args()

    triplets = json.loads(TRIPLETS.read_text())['triplets']
    verifiable = set(json.loads(TAXONOMY_PATH.read_text())['verifiable_tags'])
    fpr = {t: d['fpr'] for t, d in json.loads(SPEC.read_text())['per_tag'].items()}
    label = args.label or args.audio_dir.name

    judge = Judge()
    per_tag = defaultdict(lambda: [0, 0])
    per_cat = defaultdict(lambda: [0, 0])
    raw = [0, 0]

    items = sorted(triplets)
    if args.limit:
        items = items[:args.limit]
    for n, cid in enumerate(items, 1):
        path = args.audio_dir / f'{cid}.flac'
        if not path.exists():
            continue
        wav = load_audio(path)
        for cat, tag in triplets[cid].items():
            y, no = judge.detect(wav, 16000, tag, cat)
            d = int(y > no)
            raw[0] += d
            raw[1] += 1
            per_tag[tag][0] += d
            per_tag[tag][1] += 1
            per_cat[cat][0] += d
            per_cat[cat][1] += 1
        if n % 100 == 0:
            print(f'  {n}/{len(items)}  running CCS_raw {raw[0]/max(raw[1],1):.4f}',
                  flush=True)

    def micro(tags):
        h = sum(per_tag[t][0] for t in tags)
        m = sum(per_tag[t][1] for t in tags)
        return (h / m if m else None), m

    vt = [t for t in per_tag if t in verifiable]
    ccs_raw = raw[0] / max(raw[1], 1)
    ccs_ver, n_ver = micro(vt)

    corr_num = corr_den = 0.0
    for t in vt:
        h, m = per_tag[t]
        f = fpr.get(t)
        if f is None or f >= 1.0:
            continue
        corr_num += m * max(0.0, (h / m - f) / (1 - f))
        corr_den += m
    ccs_corr = corr_num / corr_den if corr_den else None

    out = {
        'label': label,
        'judge': 'Qwen/Qwen2.5-Omni-3B',
        'n_clips': len([c for c in items if (args.audio_dir / f'{c}.flac').exists()]),
        'ccs_raw_all_tags': ccs_raw,
        'n_judgements_raw': raw[1],
        'ccs_verifiable': ccs_ver,
        'n_judgements_verifiable': n_ver,
        'ccs_corrected': ccs_corr,
        'per_category': {c: {'rate': h / m, 'n': m} for c, (h, m) in per_cat.items()},
        'per_tag': {t: {'rate': h / m, 'n': m, 'verifiable': t in verifiable,
                        'fpr': fpr.get(t)}
                    for t, (h, m) in sorted(per_tag.items())},
    }
    (ATTM / f'ccs_tri_{label}.json').write_text(json.dumps(out, indent=1))
    print(json.dumps({k: v for k, v in out.items()
                      if k not in ('per_tag',)}, indent=1), flush=True)


if __name__ == '__main__':
    main()
