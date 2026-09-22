#!/usr/bin/env python
"""D1 probe: build the defect-prompt TSV.

Question: can MeanAudio be prompted INTO the defect region? 031 concluded the
negative-prompt gain comes from fidelity-domain vocabulary, not defect polarity.
One untested explanation is that the model has no defect direction at all,
because nothing in the training corpus is bad audio labelled as bad. If defect
prompts cannot move the output toward the programmatic-degradation reference
region, that explanation is supported at zero training cost.

Groups (N samples each, one row per sample so eval.py draws fresh noise):
  clean   3 bare musical stems (baseline)
  full    the same 3 stems + the whole defect stack
  axis    stem 1 + one defect axis each (noisy / distorted / muffled / lo-fi)
  pure    4 defect-only prompts, no music
"""
import argparse
import csv

STEMS = {
    'rock': 'a rock song with electric guitar, bass and drums',
    'piano': 'a solo piano piece',
    'edm': 'an electronic dance track with synth and a steady beat',
}
DEFECT_STACK = ('low quality recording, noisy, distorted, clipping, muffled, '
                'hiss, lo-fi, poor fidelity')
AXES = {
    'noisy': 'noisy recording with loud hiss and background static',
    'distorted': 'harsh digital distortion and clipping',
    'muffled': 'muffled low-bitrate recording with no high frequencies',
    'lofi': 'lo-fi amateur recording, poor fidelity',
}
PURE = {
    'white': 'white noise',
    'static': 'static noise and tape hiss, no music',
    'clip': 'harsh digital distortion and clipping, no music',
    'muffled': 'a muffled low-bitrate recording with no high frequencies, no music',
}


def prompts():
    """[(group, key, caption), ...] in a fixed order."""
    out = []
    for k, s in STEMS.items():
        out.append(('clean', k, s))
    for k, s in STEMS.items():
        out.append(('full', k, f'{DEFECT_STACK}. {s}'))
    for k, a in AXES.items():
        out.append(('axis', k, f'{STEMS["rock"]}, {a}'))
    for k, s in PURE.items():
        out.append(('pure', k, s))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=128, help='samples per prompt')
    ap.add_argument('--out', required=True)
    args = ap.parse_args()

    rows = [(f'd1_{g}_{k}_{i:03d}', c)
            for g, k, c in prompts() for i in range(args.n)]
    ids = [r[0] for r in rows]
    assert len(set(ids)) == len(ids)
    with open(args.out, 'w', encoding='utf-8', newline='') as f:
        w = csv.writer(f, delimiter='\t', lineterminator='\n')
        w.writerow(['id', 'caption'])
        w.writerows(rows)
    print(f'{args.out}: {len(rows)} rows, {len(prompts())} prompts x {args.n}')


if __name__ == '__main__':
    main()
