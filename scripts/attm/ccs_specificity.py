#!/usr/bin/env python3
"""Specificity control for CCS -- the half ATTM's tag filter is missing.

ATTM criterion 2 keeps a tag if the judge detects it at recall >= 0.85 when it
is genuinely present. Nothing in that test punishes a judge that answers "Yes"
to everything: such a judge scores recall 1.000 on every tag, sails through the
filter, and then reports a CCS that carries no information at all.

Our calibration hit exactly that smell -- calming, relaxing, hypnotic and fun
all came back at recall 1.000.

This script measures the other half. For each real reference clip we ask the
judge about tags that are ABSENT from that clip's human aspect list, and record
how often it still says yes. Reported per tag:

    recall      P(yes | present)     from ccs_taxonomy.json
    fpr         P(yes | absent)      measured here
    youden_j    recall - fpr         0 means the tag carries no information

Tags are only worth scoring when recall is high AND fpr is low.
"""
import ast
import csv
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

ATTM = Path('/home/kojiek/nvme_experiment_artifacts/meanaudio/attm')
MUSICCAPS_CSV = ATTM / 'musiccaps-public.csv'
REF_DIR = ATTM / 'musiccaps_instrumental_ref'
TAX = ATTM / 'ccs_taxonomy.json'

# A tag is only a true negative if no near-synonym appears in the clip's
# aspects, otherwise a "yes" is correct and we would punish the judge unfairly.
SYNONYMS = [
    {'electric guitar', 'e-guitar', 'guitar', 'acoustic guitar', 'electric guitars'},
    {'bass', 'bass guitar', 'e-bass', 'bassline', 'groovy bass'},
    {'keyboard', 'keyboard harmony', 'piano', 'synth', 'synthesizer', 'organ'},
    {'acoustic drums', 'electronic drums', 'drums', 'percussion', 'drum machine'},
    {'strings', 'violin', 'cello', 'orchestra', 'string section'},
    {'classical', 'classical music', 'orchestral'},
    {'electronic music', 'electronic', 'edm', 'techno', 'house', 'dance'},
    {'calming', 'relaxing', 'soothing', 'meditative', 'mellow', 'easygoing', 'peaceful'},
    {'happy', 'cheerful', 'fun', 'playful', 'upbeat', 'lively', 'joyful'},
    {'energetic', 'spirited', 'intense', 'exciting', 'aggressive', 'epic'},
]


def blocked_by(tag):
    out = {tag}
    for grp in SYNONYMS:
        if tag in grp:
            out |= grp
    return out


def main():
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from ccs import Judge, VOCAL_RE, CANON, load_audio

    tax = json.loads(TAX.read_text())
    verifiable = tax['verifiable_tags']
    recall = tax['recall']

    rows = [r for r in csv.DictReader(MUSICCAPS_CSV.open())
            if not VOCAL_RE.search(r['caption'])]
    aspects = {f"{r['ytid']}_{r['start_s']}":
               {a.strip().lower() for a in ast.literal_eval(r['aspect_list'])}
               for r in rows}

    rng = random.Random(42)
    # build (clip, absent tag) probes, balanced across tags
    probes = defaultdict(list)
    for cid, asp in aspects.items():
        if not (REF_DIR / f'{cid}.wav').exists():
            continue
        for tag in verifiable:
            if not (blocked_by(tag) & asp):
                probes[tag].append(cid)
    per_tag = {t: rng.sample(v, min(40, len(v))) for t, v in probes.items()}
    total = sum(len(v) for v in per_tag.values())
    print(f'negative probes: {total} over {len(per_tag)} tags', flush=True)

    judge = Judge()
    by_clip = defaultdict(list)
    for tag, cids in per_tag.items():
        for cid in cids:
            by_clip[cid].append(tag)

    fp, n = Counter(), Counter()
    for i, (cid, tags) in enumerate(sorted(by_clip.items()), 1):
        wav = load_audio(REF_DIR / f'{cid}.wav')
        for tag in tags:
            y, no = judge.detect(wav, 16000, tag, CANON[tag])
            n[tag] += 1
            fp[tag] += int(y > no)
        if i % 200 == 0:
            print(f'  {i}/{len(by_clip)} clips', flush=True)

    out = {}
    for tag in verifiable:
        if n[tag] < 10:
            continue
        r, f = recall[tag], fp[tag] / n[tag]
        out[tag] = {'recall': round(r, 4), 'fpr': round(f, 4),
                    'youden_j': round(r - f, 4), 'n_neg': n[tag],
                    'category': CANON[tag]}
    usable = sorted([t for t, v in out.items() if v['youden_j'] >= 0.3],
                    key=lambda t: -out[t]['youden_j'])
    (ATTM / 'ccs_specificity.json').write_text(json.dumps({
        'judge': tax['judge'],
        'criterion': 'usable = recall >= 0.85 (ATTM) AND youden_j >= 0.3 (ours)',
        'per_tag': dict(sorted(out.items(), key=lambda kv: -kv[1]['youden_j'])),
        'usable_tags': usable}, indent=1))

    print(f'\n{"tag":22s} {"cat":10s} {"recall":>7s} {"fpr":>7s} {"J":>7s}')
    for t, v in sorted(out.items(), key=lambda kv: -kv[1]['youden_j']):
        flag = '' if v['youden_j'] >= 0.3 else '   <- uninformative'
        print(f'{t:22s} {v["category"]:10s} {v["recall"]:7.3f} {v["fpr"]:7.3f} '
              f'{v["youden_j"]:7.3f}{flag}')
    print(f'\nusable tags: {len(usable)}/{len(out)}')


if __name__ == '__main__':
    main()
