import json, itertools, math, sys
import numpy as np
R = json.load(open(sys.argv[1] if len(sys.argv) > 1 else 'scores.json'))
TON = list(range(60, 72))
idx = {(r['timbre'], r['filter'], r['drums'], r['score'], r['midi'], r['chord']): r for r in R}
rng = np.random.default_rng(0)
CHN = ['大三', '小三', 'sus4', '五度+八度', '減三', '大七']
TIM = ['e00osc', 'sine', 'piano', 'violin', 'guitar', 'organ', 'flute', 'strings']
SC = ['e00', 'short', 'grid', 'block', 'prog', 'root', 'rscale']
def v(t, f, d, s='e00', ch=0, m='CE', ton=TON):
    return np.array([idx[(t, f, d, s, n, ch)][m] for n in ton])
def ci(x):
    b = [np.mean(rng.choice(x, len(x))) for _ in range(2000)]
    return f'{np.mean(x):+.2f} [{np.percentile(b, 2.5):+.2f}, {np.percentile(b, 97.5):+.2f}]'
out = []
def P(s=''):
    print(s); out.append(s)
full = v('e00osc', 'abs', 1)
P('## 使用者指定的條件（C#4 單點／12 個主音 C4–B4 平均），CE｜PQ\n')
P('| 條件 | C#4 CE | 12 主音 CE | 12 主音 PQ |'); P('|---|---|---|---|')
def row(name, t, f, d, s='e00', ch=0):
    P(f'| {name} | {idx[(t, f, d, s, 61, ch)]["CE"]:.2f} | {v(t, f, d, s, ch).mean():.2f} | {v(t, f, d, s, ch, "PQ").mean():.2f} |')
row('e00 完整版（電子音＋低通＋鼓）', 'e00osc', 'abs', 1)
row('只留和弦：正弦波、無濾波、無鼓', 'sine', 'off', 0)
row('只留和弦：e00 振盪器、無濾波、無鼓', 'e00osc', 'off', 0)
row('和弦電子音：e00 振盪器＋低通、無鼓', 'e00osc', 'abs', 0)
P(f'| 只有鼓（無濾波） | {idx[("-", "off", 1, "none", 61, 0)]["CE"]:.2f} | – | {idx[("-", "off", 1, "none", 61, 0)]["PQ"]:.2f} |')
P(f'| 只有鼓（低通） | {idx[("-", "abs", 1, "none", 61, 0)]["CE"]:.2f} | – | {idx[("-", "abs", 1, "none", 61, 0)]["PQ"]:.2f} |')
P('\n## 別的和弦（e00 振盪器，12 主音平均 CE）\n')
P('| 和弦 | 低通無鼓 | 低通＋鼓 | 無濾波無鼓 | 無濾波＋鼓 |'); P('|---|---|---|---|---|')
for c in range(6):
    P(f'| {CHN[c]} | ' + ' | '.join(f'{v("e00osc", f, d, ch=c).mean():.2f}' for f, d in (('abs', 0), ('abs', 1), ('off', 0), ('off', 1))) + ' |')
P('\n## 別的音色（同一樂譜、大三和弦，12 主音平均 CE）\n')
P('| 音色 | 無濾波無鼓 | 無濾波＋鼓 | 低通無鼓 | 低通＋鼓 | vs e00 完整版 ΔCE（低通＋鼓） |'); P('|---|---|---|---|---|---|')
for t in TIM:
    P(f'| {t} | ' + ' | '.join(f'{v(t, f, d).mean():.2f}' for f, d in (('off', 0), ('off', 1), ('abs', 0), ('abs', 1))) + f' | {ci(v(t, "abs", 1) - full)} |')
P('\n## 音色 × 和弦（無濾波＋鼓，12 主音平均 CE）\n')
P('| 音色 | ' + ' | '.join(CHN) + ' |'); P('|' + '---|' * 7)
for t in TIM:
    P(f'| {t} | ' + ' | '.join(f'{v(t, "off", 1, ch=c).mean():.2f}' for c in range(6)) + ' |')
P('\n## 樂譜（大三，12 主音平均 CE；無濾波＋鼓）\n')
P('| 音色 | ' + ' | '.join(SC) + ' |'); P('|' + '---|' * 8)
for t in TIM:
    P(f'| {t} | ' + ' | '.join(f'{v(t, "off", 1, s).mean():.2f}' for s in SC) + ' |')
P('\n## 加鼓的效果（ΔCE＝有鼓−無鼓，無濾波，e00 樂譜大三）\n')
for t in TIM:
    P(f'- {t}：{ci(v(t, "off", 1) - v(t, "off", 0))}')
P('\n## Shapley：鋼琴和弦進行（prog、無濾波、無鼓）→ e00 完整版\n')
base = ['piano', 'off', 0, 'prog']; tgt = ['e00osc', 'abs', 1, 'e00']; NM = ['音色', '濾波', '鼓', '樂譜']
def cell(S):
    c = list(base)
    for j in S:
        c[j] = tgt[j]
    return v(*c)
tot = cell(range(4)) - cell([])
for j in range(4):
    phi = np.zeros(len(TON)); oth = [k for k in range(4) if k != j]
    for r in range(4):
        for S in itertools.combinations(oth, r):
            phi += math.factorial(r) * math.factorial(3 - r) / 24 * (cell(list(S) + [j]) - cell(S))
    P(f'- {NM[j]}：{ci(phi)}（占 {phi.mean() / tot.mean():.0%}）')
P(f'- 總差 {tot.mean():+.2f}（{cell([]).mean():.2f} → {cell(range(4)).mean():.2f}）')
P('\n## 音高：低通固定 2.26 kHz（abs）vs 跟音高移動（rel），e00 完整版\n')
P('| 八度 | abs | rel |'); P('|---|---|---|')
for o, rg in ((3, range(48, 60)), (4, range(60, 72)), (5, range(72, 84))):
    P(f'| {o} | {np.mean([idx[("e00osc", "abs", 1, "e00", n, 0)]["CE"] for n in rg]):.2f} | {np.mean([idx[("e00osc", "rel", 1, "e00", n, 0)]["CE"] for n in rg]):.2f} |')
open('summary.md', 'w').write('\n'.join(out) + '\n')
