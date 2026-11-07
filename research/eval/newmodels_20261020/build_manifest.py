"""Build per-benchmark manifests (audio path, caption, human labels, existing AES/MEva scores). Run with venvs/dac python."""
import csv
import sys
from pathlib import Path

import pandas as pd

E = Path('/home/kojiek/MeanAudio/research/eval')
OUT = Path(__file__).resolve().parent
tsv = pd.read_csv('/mnt/HDD/kojiek/phase4_jamendo_data/musiccaps_test.tsv', sep='\t')
cap = {i.rsplit('_', 1)[0]: c for i, c in zip(tsv['id'], tsv['caption'])}
ref = Path('/mnt/HDD/kojiek/musiccaps_reference')

# 1) MusicCaps 522 (Meta AES-natural human ratings; 10 raters x 4 axes)
d = pd.read_csv(E / 'output/aes_human_corr/per_clip.tsv', sep='\t')
d['path'] = [str(next(ref.glob(f'{y}_*.wav'))) for y in d['ytid']]
d['caption'] = [cap[y] for y in d['ytid']]
d['key'] = d['ytid']
d['system'] = 'real'
d['prompt'] = d['ytid']
d = d.rename(columns={'raw_PQ': 'aes_PQ', 'raw_PC': 'aes_PC', 'raw_CE': 'aes_CE', 'raw_CU': 'aes_CU'})
d.drop(columns=[c for c in d if c.startswith('lufs23_')]).to_csv(OUT / 'bench_musiccaps522.tsv', sep='\t', index=False)

# 2) PAM human_eval music (500 = 4 generated systems + real, x100 prompts)
p = pd.read_csv(E / 'output/aes_human_corr_pam/per_clip.tsv', sep='\t')
sc = pd.read_csv(E / 'pam_human_eval/human_eval/music/scores.csv')
sc['key'] = sc['Model'] + '__' + sc['File Name']
txt = dict(zip(sc['key'], sc['Text']))
p['path'] = [str(E / 'pam_human_eval/human_eval/music' / s / (k.split('__', 1)[1] + '.wav')) for s, k in zip(p['system'], p['key'])]
p['prompt'] = [k.split('__', 1)[1] for k in p['key']]
p['caption'] = [txt[k] for k in p['key']]
mv = pd.read_csv('/home/kojiek/MeanAudio/runtime/meva_20261003/tables/pam.tsv', sep='\t')[['key', 'meva_raw']]
p = p.merge(mv, on='key', how='left')
p = p.rename(columns={'raw_PQ': 'aes_PQ', 'raw_PC': 'aes_PC', 'raw_CE': 'aes_CE', 'raw_CU': 'aes_CU', 'meva_raw': 'meva'})
p.drop(columns=[c for c in p if c.startswith('lufs23_')]).to_csv(OUT / 'bench_pam500.tsv', sep='\t', index=False)

# 3) MusicEval (2748 clips, 31 systems; shared = 25 systems x 100 prompts)
m = pd.read_csv(E / 'output/musiceval_mir/per_clip_musiceval.tsv', sep='\t')[
    ['f', 'ovl', 'rel', 'key', 'sys', 'prompt', 'PQ', 'PC', 'CE', 'CU', 'dur', 'shared']]
pi = dict(l.rstrip('\n').split('\t', 1) for l in open(E / 'musiceval/prompt_info.txt') if '\t' in l and not l.startswith('id'))
sp = {}
for nm in ('train', 'dev', 'test'):
    for l in open(E / f'musiceval/sets/{nm}_mos_list.txt'):
        sp[l.split(',')[0].strip()] = nm
m['split'] = [sp.get(f, 'none') for f in m['f']]
m['path'] = [str(E / 'musiceval/wav' / f) for f in m['f']]
dpi = dict(l.rstrip('\n').split('\t', 1) for l in open(E / 'musiceval/demo_prompt_info.txt') if '\t' in l and not l.startswith('id'))
m['caption'] = [pi.get(x) or dpi.get(f, '') for x, f in zip(m['prompt'], m['f'])]
me = pd.read_csv('/home/kojiek/MeanAudio/runtime/musiceval_full_20261003/per_clip.tsv', sep='\t')[['key', 'MEva']]
m = m.merge(me, on='key', how='left').rename(columns={'sys': 'system', 'MEva': 'meva', 'PQ': 'aes_PQ', 'PC': 'aes_PC', 'CE': 'aes_CE', 'CU': 'aes_CU', 'ovl': 'human_OVL', 'rel': 'human_REL'})
m.drop(columns=['f']).to_csv(OUT / 'bench_musiceval2748.tsv', sep='\t', index=False)
for n in ('musiccaps522', 'pam500', 'musiceval2748'):
    x = pd.read_csv(OUT / f'bench_{n}.tsv', sep='\t')
    print(n, len(x), 'missing path', sum(not Path(q).exists() for q in x['path']), 'empty caption', int((x['caption'].fillna('') == '').sum()))
