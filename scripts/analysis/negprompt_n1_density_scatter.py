"""Hypothesis N1: does the negative-prompt gain track the fidelity-vocabulary
density of the arm's own training captions?

QA-MDT (arXiv:2405.15863) explains its small negative-prompt effect by saying
prior work "relied on the rare instances of 'low quality' in the dataset". Our
c2p0 corpus is the opposite of rare -- 82.8% of its captions mention quality --
and our gain is two orders of magnitude larger. N1 turns that into a testable
claim: the gain should scale with how much fidelity language the model was
trained on.

Everything needed is already on disk: the paired cfg-0 deltas from the cfg 3.0
13-arm sweep, and the training TSV behind each arm. No GPU.

Three density metrics, all computed on the caption column of the arm's own
training TSV:

  quality_rate  \\bquality\\b -- the anchor. Reproduces the numbers already
                published in fidelity_stripped_caption_arm_2026_08_30.md
                (c2p0 82.8 / fulltrack 7.3 / fidstrip 10.2), so the definition
                is pinned to prior work rather than invented here.
  lofi_rate     LOFI_RE, verbatim from negprompt_reeval_full_arms.py -- the same
                vocabulary that defines the lofi_prompt eval split.
  negterm_rate  the eight words of the `fidelity` negative prompt itself. The
                most literal reading of N1: does y_neg's own vocabulary appear
                in training?

Two levels of aggregation, because the 12 arms are not 12 independent corpora:
six of them share the c2p0 slot0 caption pool and differ only in seed or in
whether Q was added at S2. Their spread is therefore a within-corpus noise
floor for this scatter, and the corpus-level correlation (n=6) is the honest
one to quote.

Usage:
    python scripts/analysis/negprompt_n1_density_scatter.py \\
        [--out-dir docs/experiments/results/phase8]
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path

csv.field_size_limit(10 ** 9)

SWEEP = Path('/home/kojiek/nvme_experiment_artifacts/meanaudio/negprompt_reeval_cfg3.0')

HDD = Path('/mnt/HDD/kojiek/phase4_jamendo_data')
C2P0 = Path('/home/kojiek/research/meanaudio_training/outputs/'
            'caption10s_pipeline/c2p0_qwen3cap_full')

# Caption corpora actually trained on. Arms that share a corpus point at the
# same entry; k3/k5 balanced differ from slot0 only in the q_level column, which
# carries no caption text, so they are the same corpus for this purpose.
CORPORA = {
    'c2p0_slot0':    HDD / 'phase8_qwen_caption10s_multisent_train.tsv',
    'c2p0_slot2':    C2P0 / 'phase8_caption2p0_slot2_train.tsv',
    'c2p0_f013_best':  C2P0 / 'phase8_caption2p0_fair013_bestof3_capidx_train.tsv',
    'c2p0_f013_worst': C2P0 / 'phase8_caption2p0_fair013_worstof3_capidx_train.tsv',
    'lpmc_p7v1':     HDD / '_QUARANTINED_phase7_v1_train.tsv',
    'fulltrack':     HDD / 'phase8_qwen_meansim_k3_balanced.tsv',
}

# Reference corpora: not arms, but they anchor the density scale. fidstrip is
# the not-yet-trained arm whose whole point is to sit at fulltrack's density,
# and MusicCaps is the eval prompt distribution.
REFERENCE = {
    'fidstrip (untrained)': Path('/home/kojiek/eval_tsvs_p100/'
                                 'phase8_caption2p0_fidstrip_train.tsv'),
    'MusicCaps (eval)':     HDD / 'musiccaps_test.tsv',
}

ARM_CORPUS = {
    'c2p0_slot0_full_noq':          'c2p0_slot0',
    'c2p0_slot0_full_seed27182818': 'c2p0_slot0',
    'c2p0_slot0_q3_full_q0':        'c2p0_slot0',
    'c2p0_slot0_q3_full_q9':        'c2p0_slot0',
    'c2p0_slot0_q5_full_q0':        'c2p0_slot0',
    'c2p0_slot0_q5_full_q9':        'c2p0_slot0',
    'c2p0_slot2_full_noq':          'c2p0_slot2',
    'c2p0_fair013_best_full':       'c2p0_f013_best',
    'c2p0_fair013_worst_full':      'c2p0_f013_worst',
    'p7v1_fullq_control_q9':        'lpmc_p7v1',
    'fulltrack_noq_full':           'fulltrack',
    'fulltrack_q3_full_q9':         'fulltrack',
}

QUALITY_RE = re.compile(r'\bquality\b', re.I)

# Verbatim from scripts/eval/negprompt_reeval_full_arms.py so the density axis
# and the lofi_prompt eval split key on the same vocabulary.
LOFI_RE = re.compile(
    r"\b(low[- ]?quality|poor[- ]?quality|bad[- ]?quality|noisy|noise|amateur|"
    r"amateurish|distort\w*|muffl\w*|lo[- ]?fi|hiss\w*|static|crackl\w*|hum|"
    r"buzz\w*|tinny|dull|muddy|clipp\w*|compress\w*|degrad\w*|artifact\w*)\b", re.I)

# The eight terms of the `fidelity` negative prompt, verbatim.
NEGTERM_RE = re.compile(
    r"\b(low quality recording|low[- ]quality|noisy|amateur|distorted|muffled|"
    r"poor fidelity|hiss|lo[- ]?fi)\b", re.I)

# High-fidelity language. Defined here from the phrase list in
# build_fidelity_stripped_captions.py; NOT claimed to reproduce the ad-hoc gate
# table in fidelity_stripped_caption_arm_2026_08_30.md.
HIFI_RE = re.compile(
    r"\b(high[- ]quality|high[- ]fidelity|excellent|professional(?:ly)?|pristine|"
    r"polished|immaculate|impeccable|crisp|clear|well[- ](?:balanced|mixed|produced|recorded)|"
    r"balanced)\b", re.I)


def caption_stats(path: Path) -> dict:
    n = words = 0
    hit = {'quality': 0, 'lofi': 0, 'negterm': 0, 'hifi': 0}
    with path.open(newline='', encoding='utf-8') as fh:
        for row in csv.DictReader(fh, delimiter='\t'):
            cap = row.get('caption') or ''
            n += 1
            words += len(cap.split())
            if QUALITY_RE.search(cap):
                hit['quality'] += 1
            if LOFI_RE.search(cap):
                hit['lofi'] += 1
            if NEGTERM_RE.search(cap):
                hit['negterm'] += 1
            if HIFI_RE.search(cap):
                hit['hifi'] += 1
    if not n:
        raise SystemExit(f'[FAIL] no rows in {path}')
    return {
        'path': str(path),
        'n_rows': n,
        'mean_words': words / n,
        'quality_rate': 100 * hit['quality'] / n,
        'lofi_rate': 100 * hit['lofi'] / n,
        'negterm_rate': 100 * hit['negterm'] / n,
        'hifi_rate': 100 * hit['hifi'] / n,
    }


def load_arms() -> dict:
    arms = {}
    for jf in sorted(SWEEP.glob('*.json')):
        blob = json.loads(jf.read_text())
        delta = blob.get('paired_delta_vs_cfg0')
        if not isinstance(delta, dict) or 'PQ' not in delta:
            continue
        label = blob['label']
        if label not in ARM_CORPUS:
            continue
        arms[label] = {
            'exp_id': blob.get('exp_id'),
            'corpus': ARM_CORPUS[label],
            'delta_pq': delta['PQ']['mean_delta'],
            'frac_improved': delta['PQ']['frac_improved'],
            'delta_clap': delta['clap']['mean_delta'],
            'pq_cfg3': blob['aggregates']['full']['PQ'],
        }
    return arms


def pearson(xs, ys):
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    sxx = sum((x - mx) ** 2 for x in xs)
    syy = sum((y - my) ** 2 for y in ys)
    if sxx <= 0 or syy <= 0:
        return float('nan'), float('nan'), float('nan')
    r = sxy / (sxx * syy) ** 0.5
    slope = sxy / sxx
    return r, r * r, slope


def rank(vals):
    order = sorted(range(len(vals)), key=lambda i: vals[i])
    out = [0.0] * len(vals)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and vals[order[j + 1]] == vals[order[i]]:
            j += 1
        avg = (i + j) / 2 + 1
        for k in range(i, j + 1):
            out[order[k]] = avg
        i = j + 1
    return out


def spearman(xs, ys):
    return pearson(rank(xs), rank(ys))[0]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--out-dir', default='docs/experiments/results/phase8')
    ap.add_argument('--metric', default='quality_rate',
                    choices=['quality_rate', 'lofi_rate', 'negterm_rate', 'hifi_rate'])
    args = ap.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print('[1/3] caption density per corpus')
    density = {}
    for name, path in CORPORA.items():
        if not path.exists():
            raise SystemExit(f'[FAIL] missing corpus {name}: {path}')
        density[name] = caption_stats(path)
        d = density[name]
        print(f'  {name:16s} n={d["n_rows"]:>7d} words={d["mean_words"]:5.1f} '
              f'quality={d["quality_rate"]:5.1f}% lofi={d["lofi_rate"]:5.1f}% '
              f'negterm={d["negterm_rate"]:5.1f}% hifi={d["hifi_rate"]:5.1f}%')

    reference = {}
    for name, path in REFERENCE.items():
        if path.exists():
            reference[name] = caption_stats(path)
            d = reference[name]
            print(f'  [ref] {name:22s} quality={d["quality_rate"]:5.1f}% '
                  f'lofi={d["lofi_rate"]:5.1f}%')

    print('[2/3] arms')
    arms = load_arms()
    missing = set(ARM_CORPUS) - set(arms)
    if missing:
        print(f'  [warn] no paired delta for: {sorted(missing)}')
    for label, a in sorted(arms.items(), key=lambda kv: -kv[1]['delta_pq']):
        print(f'  {label:30s} {a["corpus"]:16s} dPQ={a["delta_pq"]:+.4f}')

    metric = args.metric
    xs = [density[a['corpus']][metric] for a in arms.values()]
    ys = [a['delta_pq'] for a in arms.values()]
    r_arm, r2_arm, slope_arm = pearson(xs, ys)
    rho_arm = spearman(xs, ys)

    # Corpus level: one point per distinct training corpus, y = mean of its arms.
    by_corpus = {}
    for a in arms.values():
        by_corpus.setdefault(a['corpus'], []).append(a['delta_pq'])
    cx = [density[c][metric] for c in by_corpus]
    cy = [sum(v) / len(v) for v in by_corpus.values()]
    r_c, r2_c, slope_c = pearson(cx, cy)
    rho_c = spearman(cx, cy)

    within = {c: (max(v) - min(v)) for c, v in by_corpus.items() if len(v) > 1}

    # Leave-one-corpus-out. With six points and one of them (fulltrack) sitting
    # far from the rest, a single high-leverage corpus could be carrying the fit.
    corpus_names = list(by_corpus)
    loo = {}
    for drop in corpus_names:
        keep = [i for i, c in enumerate(corpus_names) if c != drop]
        r_i, _, _ = pearson([cx[i] for i in keep], [cy[i] for i in keep])
        loo[f'without_{drop}'] = r_i

    # Pre-registerable prediction for the fidelity-stripped arm: same captioner,
    # same everything, only the density moved. This is the arm that breaks the
    # captioner/style confound the six existing corpora cannot.
    mean_x = sum(cx) / len(cx)
    mean_y = sum(cy) / len(cy)
    prediction = None
    if 'fidstrip (untrained)' in reference:
        fx = reference['fidstrip (untrained)'][metric]
        prediction = {
            'arm': 'fidstrip (untrained, disk-blocked)',
            f'{metric}': fx,
            'predicted_delta_pq': mean_y + slope_c * (fx - mean_x),
            'basis': 'corpus-level least squares fit, n=6',
        }

    print('[3/3] correlation')
    print(f'  metric = {metric}')
    print(f'  arm level    n={len(xs)}  r={r_arm:+.4f}  R2={r2_arm:.4f}  '
          f'rho={rho_arm:+.4f}  slope={slope_arm:+.5f} PQ per %pt')
    print(f'  corpus level n={len(cx)}  r={r_c:+.4f}  R2={r2_c:.4f}  '
          f'rho={rho_c:+.4f}  slope={slope_c:+.5f} PQ per %pt')
    for c, w in within.items():
        print(f'  within-corpus spread ({c}, {len(by_corpus[c])} arms): {w:.4f} PQ')
    print('  leave-one-corpus-out r: ' +
          '  '.join(f'{k.replace("without_", "-")}={v:+.3f}' for k, v in loo.items()))
    if prediction:
        print(f'  prediction for fidstrip ({prediction[metric]:.1f}%): '
              f'dPQ = {prediction["predicted_delta_pq"]:+.3f}')

    result = {
        'metric': metric,
        'protocol': 'MusicCaps 5521; MeanFlow 25; CFG 3.0; negative=fidelity(8 terms); '
                    'seed 42; per-clip paired vs each arm own cfg 0',
        'density': density,
        'reference_density': reference,
        'arms': arms,
        'correlation': {
            'arm_level': {'n': len(xs), 'pearson_r': r_arm, 'r2': r2_arm,
                          'spearman_rho': rho_arm, 'slope_pq_per_pct': slope_arm},
            'corpus_level': {'n': len(cx), 'pearson_r': r_c, 'r2': r2_c,
                             'spearman_rho': rho_c, 'slope_pq_per_pct': slope_c},
        },
        'within_corpus_spread_pq': within,
        'leave_one_corpus_out_r': loo,
        'prediction': prediction,
    }
    jpath = out_dir / 'negprompt_n1_density.json'
    jpath.write_text(json.dumps(result, indent=2, ensure_ascii=False))
    print(f'  wrote {jpath}')

    try:
        plot(density, arms, by_corpus, metric, result, out_dir)
    except Exception as exc:  # plotting is a convenience, not the result
        print(f'  [warn] plot skipped: {exc}')
    return 0


def plot(density, arms, by_corpus, metric, result, out_dir):
    """Two panels, because the interesting result is the contrast.

    Left: quality_rate -- fidelity talk of any polarity. Right: negterm_rate --
    the negative prompt's own eight words. If N1 were literally true the right
    panel would be the strong one. It is not.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    label_map = {
        'c2p0_slot0': 'c2p0 slot0 (6 arms)',
        'c2p0_f013_best': 'fair013 best',
        'c2p0_f013_worst': 'fair013 worst',
        'c2p0_slot2': 'c2p0 slot2',
        'lpmc_p7v1': 'LP-MC (P7V1)',
        'fulltrack': 'fulltrack (2 arms)',
    }
    offsets = {
        'c2p0_slot0': (-24, 15), 'c2p0_f013_best': (-30, -17),
        'c2p0_f013_worst': (12, -13), 'c2p0_slot2': (-18, 13),
        'lpmc_p7v1': (10, -4), 'fulltrack': (10, 2),
    }
    panels = [
        ('quality_rate', 'any fidelity talk\n(\\b quality \\b)', '#0D6B60', 'SUPPORTED'),
        ('negterm_rate', "the negative prompt's own 8 terms", '#B4441C', 'FALSIFIED'),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(12.4, 5.2), dpi=160)
    for ax, (m, sub, colour, verdict) in zip(axes, panels):
        cx, cy = [], []
        for label, a in arms.items():
            ax.scatter(density[a['corpus']][m], a['delta_pq'], s=30,
                       color=colour, alpha=.35, edgecolors='none', zorder=3)
        for c, vals in by_corpus.items():
            x = density[c][m]
            y = sum(vals) / len(vals)
            cx.append(x)
            cy.append(y)
            ax.scatter(x, y, s=110, color=colour, zorder=4,
                       edgecolors='white', linewidths=1.4)
            ax.annotate(label_map.get(c, c), (x, y), textcoords='offset points',
                        xytext=offsets.get(c, (9, -3)), fontsize=8.2, color='#12191A')
        r, r2, slope = pearson(cx, cy)
        rho = spearman(cx, cy)
        mx, my = sum(cx) / len(cx), sum(cy) / len(cy)
        span = max(cx) - min(cx)
        lo, hi = min(cx) - .08 * span, max(cx) + .18 * span
        ax.plot([lo, hi], [my + slope * (lo - mx), my + slope * (hi - mx)],
                color=colour, lw=1.4, ls='--', zorder=2)
        ax.axhline(0, color='#D3DAD8', lw=1, zorder=1)
        ax.set_xlabel(f'training-caption {m.replace("_", " ")}  (% of captions)', fontsize=9.5)
        ax.set_title(f'{verdict}   ·   {sub}\n'
                     f'corpus fit  r={r:+.3f}   $R^2$={r2:.3f}   ρ={rho:+.3f}   (n=6)',
                     fontsize=10, loc='left', color='#12191A')
        ax.grid(True, color='#EBEEED', lw=.8)
        ax.set_axisbelow(True)
        for sp in ('top', 'right'):
            ax.spines[sp].set_visible(False)
    axes[0].set_ylabel('ΔPQ from `fidelity` negative prompt @ cfg 3.0', fontsize=10)
    ylo = min(a['delta_pq'] for a in arms.values()) - .08
    yhi = max(a['delta_pq'] for a in arms.values()) + .10
    for ax in axes:
        ax.set_ylim(ylo, yhi)
    noise = result['within_corpus_spread_pq'].get('c2p0_slot0')
    fig.suptitle('Hypothesis N1 — negative-prompt gain vs training-caption fidelity vocabulary\n'
                 f'MusicCaps n=5521 · MeanFlow 25 · cfg 3.0 · per-clip paired vs each arm own cfg 0'
                 f' · within-corpus spread (same captions, 6 arms) = {noise:.3f} PQ',
                 fontsize=11, x=.006, ha='left', y=.995)
    fig.tight_layout(rect=(0, 0, 1, .90))
    png = out_dir / 'negprompt_n1_density_scatter.png'
    fig.savefig(png)
    print(f'  wrote {png}')


if __name__ == '__main__':
    sys.exit(main())
