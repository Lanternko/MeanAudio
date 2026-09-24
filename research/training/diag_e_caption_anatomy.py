"""
E: Caption anatomy — pure CPU diagnostics
==========================================

E1 word/phrase frequency
  - 251K LP-MC vs 251K × 5 Qwen captions
  - top-50 unigrams, bigrams, trigrams 對比
  - boilerplate detection: 「features electric guitar」「creating a」「mood」 等

E2 multi-cap intra-audio diversity (Qwen 5 caps/id)
  - 用 char-level 4-gram Jaccard 算 5 個 caption 兩兩之間的相似度
  - 高 = 5 caption 彼此非常像（diversity 假，supervision noise 不夠）
  - 低 = 5 caption 真的不同
  - 對比 LP-MC：phase7_v1 是 random 1-of-4 已展平，沒法直接比；只能跟單一 LP-MC TSV 對

E3 unique caption ratio
  - 多少比例的 Qwen captions 是「跟另一個 audio 字面完全相同」
  - 高 = 重複句子多，supervision 弱

輸出：
  ~/research/meanaudio_training/diag_e_caption_anatomy.json
"""

import json
import csv
import random
from collections import Counter, defaultdict
from pathlib import Path

QWEN_JSONL = Path('/mnt/HDD/kojiek/phase4_jamendo_data/phase9_omni_captions.jsonl')
LPMC_TSV   = Path('/mnt/HDD/kojiek/phase4_jamendo_data/phase7_v1_train.tsv')
OUT_JSON   = Path('/home/kojiek/research/meanaudio_training/diag_e_caption_anatomy.json')
SEED       = 42


def load_qwen():
    d = {}
    with open(QWEN_JSONL) as f:
        for line in f:
            j = json.loads(line)
            d[j['id']] = j['captions']
    return d


def load_lpmc():
    d = {}
    with open(LPMC_TSV) as f:
        reader = csv.DictReader(f, delimiter='\t')
        for row in reader:
            d[row['id']] = row['caption']
    return d


def char_4gram_set(s, n=4):
    s = s.lower()
    return set(s[i:i+n] for i in range(len(s) - n + 1)) if len(s) >= n else {s}


def jaccard(a, b):
    if not a and not b:
        return 1.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union else 0.0


# ─── E1 word/phrase frequency ───────────────────────────────────────
def e1_word_freq(qwen, lpmc):
    print('\n=== E1: word/phrase frequency ===')
    rng = random.Random(SEED)

    # 取所有 Qwen captions（251K × 5 太大 → 抽 25K caption per source）
    lpmc_caps = list(lpmc.values())
    qwen_caps = [c for caps in qwen.values() for c in caps]
    rng.shuffle(lpmc_caps)
    rng.shuffle(qwen_caps)
    lpmc_sample = lpmc_caps[:25000]
    qwen_sample = qwen_caps[:25000]

    def tokens(s):
        return s.lower().replace('.', ' ').replace(',', ' ').split()

    def ngrams(toks, n):
        return [' '.join(toks[i:i+n]) for i in range(len(toks)-n+1)]

    def tabulate(caps):
        uni = Counter()
        bi  = Counter()
        tri = Counter()
        for c in caps:
            t = tokens(c)
            uni.update(t)
            bi.update(ngrams(t, 2))
            tri.update(ngrams(t, 3))
        return uni, bi, tri

    print(f'  tabulate LP-MC (n={len(lpmc_sample)})...')
    lu, lb, lt = tabulate(lpmc_sample)
    print(f'  tabulate Qwen  (n={len(qwen_sample)})...')
    qu, qb, qt = tabulate(qwen_sample)

    out = {
        'n_sample_lpmc': len(lpmc_sample),
        'n_sample_qwen': len(qwen_sample),
        'lpmc': {
            'top_unigram': lu.most_common(30),
            'top_bigram':  lb.most_common(30),
            'top_trigram': lt.most_common(30),
            'vocab_size_unigram': len(lu),
            'vocab_size_bigram':  len(lb),
        },
        'qwen': {
            'top_unigram': qu.most_common(30),
            'top_bigram':  qb.most_common(30),
            'top_trigram': qt.most_common(30),
            'vocab_size_unigram': len(qu),
            'vocab_size_bigram':  len(qb),
        },
    }
    print(f'  LP-MC vocab: {len(lu):,} unigram / {len(lb):,} bigram')
    print(f'  Qwen  vocab: {len(qu):,} unigram / {len(qb):,} bigram')
    print(f'\n  Qwen top 10 trigrams:')
    for tg, n in qt.most_common(10):
        print(f'    {n:5d}  "{tg}"')
    print(f'\n  LP-MC top 10 trigrams:')
    for tg, n in lt.most_common(10):
        print(f'    {n:5d}  "{tg}"')

    # 共 trigram 比率：Qwen 共 trigram 在多少 caption 內出現
    def coverage_at(top_n, ngram_counter, caps):
        topset = set(ng for ng, _ in ngram_counter.most_common(top_n))
        n_hit = 0
        for c in caps:
            t = tokens(c)
            if any(' '.join(t[i:i+3]) in topset for i in range(len(t)-2)):
                n_hit += 1
        return n_hit / max(1, len(caps))

    cov_lpmc_50 = coverage_at(50, lt, lpmc_sample)
    cov_qwen_50 = coverage_at(50, qt, qwen_sample)
    print(f'\n  fraction of caption containing ANY of top-50 trigrams:')
    print(f'    LP-MC: {cov_lpmc_50:.1%}')
    print(f'    Qwen:  {cov_qwen_50:.1%}    ← 高 = 重複套路句子多')
    out['top50_trigram_coverage'] = {'lpmc': round(cov_lpmc_50, 3), 'qwen': round(cov_qwen_50, 3)}
    return out


# ─── E2 multi-cap intra-audio diversity ─────────────────────────────
def e2_multicap_diversity(qwen):
    print('\n=== E2: Qwen multi-cap intra-audio diversity (5 caps × 251K) ===')
    rng = random.Random(SEED)
    sample_ids = rng.sample(list(qwen.keys()), 5000)

    pair_jaccards = []
    pair_count = 0
    for cid in sample_ids:
        caps = qwen[cid]
        sets = [char_4gram_set(c) for c in caps]
        for i in range(5):
            for j in range(i+1, 5):
                pair_jaccards.append(jaccard(sets[i], sets[j]))
                pair_count += 1

    import numpy as np
    arr = np.array(pair_jaccards)
    print(f'  n_pairs = {len(arr):,} (5C2 × {len(sample_ids)} = {pair_count:,})')
    print(f'  Jaccard 4-gram intra-audio mean = {arr.mean():.3f}')
    print(f'  median = {np.median(arr):.3f}  p25 = {np.percentile(arr, 25):.3f}  '
          f'p75 = {np.percentile(arr, 75):.3f}')
    print(f'  fraction pairs with Jaccard > 0.5 = {(arr > 0.5).mean():.2%}')
    print(f'  fraction pairs with Jaccard > 0.3 = {(arr > 0.3).mean():.2%}')
    print(f'  判讀：Jaccard 高 = 5 個 caption 彼此非常像 → diversity 假')
    return {
        'n_pairs': len(arr),
        'mean_jaccard_4gram': round(float(arr.mean()), 4),
        'median': round(float(np.median(arr)), 4),
        'p25': round(float(np.percentile(arr, 25)), 4),
        'p75': round(float(np.percentile(arr, 75)), 4),
        'frac_above_0_5': round(float((arr > 0.5).mean()), 4),
        'frac_above_0_3': round(float((arr > 0.3).mean()), 4),
    }


# ─── E3 duplicate caption ratio (cross-audio) ───────────────────────
def e3_duplicates(qwen, lpmc):
    print('\n=== E3: cross-audio duplicate captions ===')
    out = {}

    # LP-MC: 251K captions (1 per id)
    lpmc_caps = list(lpmc.values())
    lpmc_uniq = len(set(lpmc_caps))
    print(f'  LP-MC: {len(lpmc_caps):,} captions, {lpmc_uniq:,} unique '
          f'({lpmc_uniq/len(lpmc_caps):.1%})')
    out['lpmc'] = {'total': len(lpmc_caps), 'unique': lpmc_uniq,
                   'unique_rate': round(lpmc_uniq/len(lpmc_caps), 4)}

    # Qwen per-slot
    out['qwen_per_slot'] = []
    for s in range(5):
        caps = [v[s] for v in qwen.values()]
        u = len(set(caps))
        print(f'  Qwen slot {s}: {len(caps):,} captions, {u:,} unique ({u/len(caps):.1%})')
        out['qwen_per_slot'].append({'slot': s, 'total': len(caps), 'unique': u,
                                     'unique_rate': round(u/len(caps), 4)})

    # Qwen all (251K × 5)
    qwen_all = [c for v in qwen.values() for c in v]
    qa_uniq = len(set(qwen_all))
    print(f'  Qwen all 5: {len(qwen_all):,} captions, {qa_uniq:,} unique '
          f'({qa_uniq/len(qwen_all):.1%})')
    out['qwen_all'] = {'total': len(qwen_all), 'unique': qa_uniq,
                       'unique_rate': round(qa_uniq/len(qwen_all), 4)}
    return out


def main():
    print('讀 Qwen JSONL...')
    qwen = load_qwen()
    print(f'  {len(qwen):,} ids')
    print('讀 LP-MC TSV...')
    lpmc = load_lpmc()
    print(f'  {len(lpmc):,} ids')

    results = {}
    results['e1_word_freq']         = e1_word_freq(qwen, lpmc)
    results['e2_multicap_diversity'] = e2_multicap_diversity(qwen)
    results['e3_duplicates']         = e3_duplicates(qwen, lpmc)

    OUT_JSON.write_text(json.dumps(results, indent=2))
    print(f'\n→ {OUT_JSON}')


if __name__ == '__main__':
    main()
