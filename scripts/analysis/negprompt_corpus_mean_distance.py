#!/usr/bin/env python
"""Task A follow-up: does a negative prompt help more the closer it sits to the
training corpus' "average caption"?

Hypothesis (075 discussion): B = the model's guess under a text that looks like
the training captions' domain average; the CFG push A-B then removes the
generic "corpus-average" component. If so, dPQ should rise with the negative
prompt's similarity to the corpus mean, not with how "defect-ish" it is.

Data: the 17 distinct cfg3 negative prompts evaluated on the same checkpoint
(c2p0_slot0 = phase8_qwen_caption10s_multisent_noq_full_stage2_200000) and the
same MusicCaps subset1024 (negprompt_ablation + single_negprompt_cfg3_ablation).
Corpus: N random captions of that checkpoint's training TSV.

Predictors (cosine of the negative prompt to):
  P1 clap      mean of L2-normalised LAION-CLAP text embeddings of the corpus (primary)
  P2 t5_pool   mean of masked-mean-pooled flan-t5-large features of the corpus
  P3 t5_seq    mean of the padded 77x1024 T5 sequence, flattened (what a NoMask
               model actually attends to)
  P4 clap_eval mean CLAP embedding of the 1024 MusicCaps eval captions
Spearman rho vs dPQ (= PQ - PQ_none) with prompt-bootstrap CI and permutation p.

Usage: python negprompt_corpus_mean_distance.py [n_corpus=5000]
"""
import csv
import glob
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.stats import spearmanr

ROOT = Path('/home/kojiek/MeanAudio')
ART = Path('/home/kojiek/nvme_experiment_artifacts/meanaudio')
CORPUS_TSV = Path('/mnt/HDD/kojiek/phase4_jamendo_data/phase8_qwen_caption10s_multisent_train.tsv')
SUBSET_TSV = ART / 'negprompt_ablation' / 'musiccaps_subset1024.tsv'
OUT = ART / 'negprompt_corpus_mean_distance'
SEED = 20260926


def cells():
    fs = sorted(glob.glob(str(ART / 'negprompt_ablation' / 'c2p0_slot0__cfg3.0__*.json'))
                + glob.glob(str(ART / 'single_negprompt_cfg3_ablation' / 'c2p0_slot0__cfg3.0__*.json')))
    none_pq, by_prompt = None, {}
    for f in fs:
        if '__POS' in f:            # positive-slot edits are a different manipulation
            continue
        d = json.load(open(f))
        pq = d['aggregates']['full']['PQ']
        if d['negative_prompt'] is None:
            if f.endswith('__none.json') and 'single' not in f:
                none_pq = pq
            continue
        by_prompt.setdefault(d['negative_prompt'], (pq, Path(f).stem))
    assert none_pq is not None
    return none_pq, [(p, pq - none_pq, name) for p, (pq, name) in by_prompt.items()]


class Enc:
    def __init__(self):
        import laion_clap
        from transformers import AutoTokenizer, T5EncoderModel
        self.tok = AutoTokenizer.from_pretrained('google/flan-t5-large')
        self.t5 = T5EncoderModel.from_pretrained('google/flan-t5-large').eval().cuda()
        self.clap = laion_clap.CLAP_Module(enable_fusion=False, amodel='HTSAT-base').eval()
        self.clap.load_ckpt(str(ROOT / 'weights/music_speech_audioset_epoch_15_esc_89.98.pt'), verbose=False)
        self.clap.cuda()

    @torch.inference_mode()
    def __call__(self, texts):
        tk = self.tok(texts, max_length=77, padding='max_length', truncation=True, return_tensors='pt')
        ids, m = tk.input_ids.cuda(), tk.attention_mask.cuda()
        seq = self.t5(input_ids=ids, attention_mask=m)[0].float()                 # (B, 77, 1024)
        pool = (seq * m[..., None]).sum(1) / m.sum(1, keepdim=True)
        c = self.clap.get_text_embedding(texts, use_tensor=True).float()
        c = torch.nn.functional.normalize(c, dim=-1)
        return seq, pool, c


def mean_of(enc, texts, bs=64):
    seq_sum, pool_sum, clap_sum = 0, 0, 0
    for i in range(0, len(texts), bs):
        s, p, c = enc(texts[i:i + bs])
        seq_sum = seq_sum + s.sum(0)
        pool_sum = pool_sum + p.sum(0)
        clap_sum = clap_sum + c.sum(0)
    n = len(texts)
    return seq_sum / n, pool_sum / n, clap_sum / n


def cos(x, y):
    return float(torch.nn.functional.cosine_similarity(x.flatten()[None], y.flatten()[None]).item())


def rho_ci(x, y, n_boot=10000, n_perm=10000):
    rng = np.random.default_rng(SEED)
    x, y = np.asarray(x), np.asarray(y)
    rho = spearmanr(x, y).statistic
    boots = []
    for _ in range(n_boot):
        i = rng.integers(0, len(x), len(x))
        if len(set(x[i])) > 2 and len(set(y[i])) > 2:
            boots.append(spearmanr(x[i], y[i]).statistic)
    perm = np.array([spearmanr(x, rng.permutation(y)).statistic for _ in range(n_perm)])
    return {'rho': float(rho), 'ci95': [float(np.nanpercentile(boots, 2.5)), float(np.nanpercentile(boots, 97.5))],
            'p_perm_two_sided': float((np.abs(perm) >= abs(rho)).mean())}


def main():
    n_corpus = int(sys.argv[1]) if len(sys.argv) > 1 else 5000
    OUT.mkdir(parents=True, exist_ok=True)
    none_pq, rows = cells()
    print(f'none PQ {none_pq:.3f}; {len(rows)} distinct negative prompts', flush=True)
    corpus = [r['caption'] for r in csv.DictReader(open(CORPUS_TSV), delimiter='\t')]
    random.Random(SEED).shuffle(corpus)
    corpus = corpus[:n_corpus]
    evalcaps = [r['caption'] for r in csv.DictReader(open(SUBSET_TSV), delimiter='\t')]
    enc = Enc()
    c_seq, c_pool, c_clap = mean_of(enc, corpus)
    _, _, e_clap = mean_of(enc, evalcaps)
    table = []
    for prompt, dpq, name in rows:
        s, p, c = enc([prompt])
        table.append({'cell': name, 'negative_prompt': prompt, 'dPQ': dpq,
                      'clap': cos(c[0], c_clap), 't5_pool': cos(p[0], c_pool),
                      't5_seq': cos(s[0], c_seq), 'clap_eval': cos(c[0], e_clap)})
    table.sort(key=lambda r: -r['dPQ'])
    dpq = [r['dPQ'] for r in table]
    stats = {k: rho_ci([r[k] for r in table], dpq) for k in ('clap', 't5_pool', 't5_seq', 'clap_eval')}
    # robustness: drop the three fidelity8-family long prompts (dominant, near-duplicates)
    sub = [r for r in table if not r['negative_prompt'].startswith('low quality recording')]
    stats_sub = {k: rho_ci([r[k] for r in sub], [r['dPQ'] for r in sub]) for k in ('clap', 't5_pool', 't5_seq')}
    out = {'none_PQ': none_pq, 'n_corpus': len(corpus), 'n_eval_captions': len(evalcaps),
           'corpus_tsv': str(CORPUS_TSV), 'table': table, 'spearman_vs_dPQ': stats,
           'spearman_vs_dPQ_without_fidelity8_family': {'n': len(sub), **stats_sub}}
    path = OUT / f'negprompt_corpus_mean_distance_n{len(corpus)}.json'
    path.write_text(json.dumps(out, indent=2, ensure_ascii=False) + '\n')
    print(f'{"dPQ":>7} {"clap":>7} {"t5pool":>7} {"t5seq":>7} {"clapEv":>7}  prompt')
    for r in table:
        print(f'{r["dPQ"]:+7.3f} {r["clap"]:7.4f} {r["t5_pool"]:7.4f} {r["t5_seq"]:7.4f} '
              f'{r["clap_eval"]:7.4f}  {r["negative_prompt"][:60]}')
    for k, v in stats.items():
        print(f'{k:>10}: rho={v["rho"]:+.3f} CI[{v["ci95"][0]:+.3f},{v["ci95"][1]:+.3f}] p={v["p_perm_two_sided"]:.4f}')
    for k, v in stats_sub.items():
        print(f'{k:>10} (no fid8 family, n={len(sub)}): rho={v["rho"]:+.3f} '
              f'CI[{v["ci95"][0]:+.3f},{v["ci95"][1]:+.3f}] p={v["p_perm_two_sided"]:.4f}')
    print(f'wrote {path}')


if __name__ == '__main__':
    main()
