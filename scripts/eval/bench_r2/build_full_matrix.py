#!/usr/bin/env python3
"""Assemble the full metric x human-rating matrix (2026-10-27) from the files that already exist.

Sources (read-only):
  <ROOT>/analysis/{odaq_g6,mc522_corr,songeval_corr,singmos_corr,aime_pairwise,arena_pairwise}.tsv
  <ROOT>/analysis/summary.json                     r2 preregistered decisions (AIME/Arena G1 = item-level Spearman)
  <ROOT>/analysis_extra/*.tsv                      written by jobs 159/160 (missing cells); absent before they run
  research/eval/newmodels_20261020/analysis_cells.json     round-1 cells for PAM400 / MusicEval
  docs/experiments/results/benchmark_screening_r2_human_ceiling_20261027.md   human ceilings (hand-copied below)

Cells are one of: value (rho or agreement + CI + n), pending (a queued job will write it), n/a (the human
rating or the caption the metric needs does not exist for that dataset).

  python -I build_full_matrix.py [--out-docs ...] [--refresh]

Writes docs/experiments/results/benchmark_screening_full_matrix_20261027.{md,json}.
Nothing here changes the round-2 preregistered gates or its summary json.
"""
import argparse
import csv
import json
import sys
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = Path('/home/kojiek/MeanAudio')
ROOT = Path('/mnt/seagate/benchmarks_r2')
ANA, EXTRA = ROOT / 'analysis', ROOT / 'analysis_extra'
R1_CELLS = REPO / 'research/eval/newmodels_20261020/analysis_cells.json'
OUT = REPO / 'docs/experiments/results'

SEED_R2, SEED_R1, NBOOT = 20261025, 20261020, 2000

# ── columns ───────────────────────────────────────────────────────────────
# key, short header, domain, human signal, statistic, ceiling, units, flag note
COLS = [
    dict(key='odaq', head='ODAQ', dom='訊號劣化', human='專家 MUSHRA（壓縮／分離瑕疵）',
         stat='G6a within-trial ρ', ceil=dict(v=0.993, how='√SB'),
         units=960, src='analysis/odaq_g6.tsv'),
    dict(key='mc522', head='MC522', dom='真實錄音', human='MusicCaps 522 首品質評分（10 人）',
         stat='G1 ρ（human_PQ）', ceil=dict(v=0.947, how='√SB'),
         units=522, src='analysis/mc522_corr.tsv'),
    dict(key='songeval', head='SongEval', dom='生成音樂（整首）', human='SongEval Musicality（4 標註者）',
         stat='G1 ρ（Musicality, VFULL）', ceil=dict(v=0.921, how='√SB'),
         units=800, src='analysis/songeval_corr.tsv'),
    dict(key='singmos', head='SingMOS-Pro', dom='歌唱', human='SingMOS MOS（官方 test）',
         stat='G1 ρ（mos）', ceil=dict(v=0.919, how='√SB'),
         units=3101, src='analysis/singmos_corr.tsv'),
    dict(key='aime', head='AIME', dom='生成音樂', human='13 系統成對偏好（12 票）',
         stat='G1 ρ（逐檔勝率）', ceil=dict(v=0.811, how='√SB'),
         units=1300, src='analysis/aime_pairwise.tsv'),
    dict(key='arena', head='Music Arena', dom='生成音樂', human='15 系統成對偏好',
         stat='G1 ρ（逐檔勝率）', ceil=dict(v=0.742, how='G3 BT'),
         units=3998, src='analysis/arena_pairwise.tsv'),
    dict(key='pam400', head='PAM400', dom='生成音樂（5 s）', human='PAM OVL（每檔 10 位 MTurk）',
         stat='ρ（round-1 協定）', ceil=None,
         units=400, src='analysis_extra/pam400_extra_cells.tsv'),
    dict(key='me', head='MusicEval', dom='生成音樂', human='MusicEval OVL（31 系統）',
         stat='ρ（round-1 協定）', ceil=None,
         units=2748, src='analysis_extra/me_extra_cells.tsv'),
    dict(key='musicprefs', head='MusicPrefs', dom='生成音樂（成對偏好）', human='2,515 對偏好（7 系統、2 軸）',
         stat='ρ（逐檔勝率）', ceil=None,
         units=5030, src='analysis_extra/musicprefs_pairwise.tsv'),
]
COLS[6]['ceiling_note'] = 'n/a（每檔是 10 人平均，沒有逐位評分可算 reliability）'
COLS[7]['ceiling_note'] = 'n/a（同上，只有平均值）'
COLS[8]['ceiling_note'] = '待 job 160 算 BT held-out'

# ── rows: (score column, label) ────────────────────────────────────────────
ROWS = [
    ('aes_PQ', 'AES PQ'),
    ('aes_CE', 'AES CE'),
    ('aes_CU', 'AES CU'),
    ('aes_PC', 'AES PC'),
    ('pam', 'PAM（零樣本 CLAP）'),
    ('meva', 'MEva'),
    ('muqeval_A1_MI', 'MuQ-Eval A1（MI）'),
    ('muqeval_A3a_MI', 'MuQ-Eval A3a（MI）'),
    ('cmi_empty_quality', 'CMI-RM（empty quality）'),
    ('se_mus', 'SongEval（se_mus）'),
    ('singmos', 'SingMOS predictor'),
    ('clap', 'CLAP（只描述）'),
]
# model name in score files, per row
MODEL = {'aes_PQ': 'aes', 'aes_CE': 'aes', 'aes_CU': 'aes', 'aes_PC': 'aes', 'pam': 'pam', 'meva': 'meva',
         'muqeval_A1_MI': 'muqeval_A1', 'muqeval_A3a_MI': 'muqeval_A3a', 'cmi_empty_quality': 'cmirm',
         'se_mus': 'songeval', 'singmos': 'singmos', 'clap': 'clap'}

# ── trained / overlap flags (docs/experiments/results/benchmark_screening_r2_overlap_audit_20261027.md) ──
TRAINED = {
    'meva': {'aime', 'songeval', 'arena', 'me', 'musicprefs'},
    'muqeval_A1_MI': {'me'},
    'muqeval_A3a_MI': {'me'},
    'cmi_empty_quality': {'me'},
    'se_mus': {'songeval'},
    'singmos': {'singmos'},
}

NO_CAPTION = {'odaq', 'singmos', 'songeval'}  # these datasets ship no text, so CLAP is undefined


def flag(metric, col):
    if metric == 'pam' and col == 'pam400':
        return 'd'  # PAM's own design/report data; descriptive only
    return 't' if col in TRAINED.get(metric, ()) else 'i'


def fnum(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def parse_ci(s):
    if not s or not s.strip().startswith('['):
        return None
    lo, hi = s.strip().strip('[]').split(',')
    return [float(lo), float(hi)]


def read_tsv(p):
    with open(p, newline='') as f:
        return list(csv.DictReader(f, delimiter='\t'))


def pick_ana(name, ds):
    """analysis_extra (job 159/160) wins over analysis when both exist."""
    for d in (EXTRA, ANA):
        p = d / name
        if p.exists():
            return p
    return None


def from_corr(ds, metric):
    p = pick_ana(f'{ds}_corr.tsv', ds)
    if p is None:
        return None
    for r in read_tsv(p):
        if r['col'] == metric and r['primary'] == '1':
            return dict(state='value', value=fnum(r['spearman']), ci=parse_ci(r['spearman_ci']),
                        n=int(r['n']), stat='G1 ρ', src=str(p.relative_to(ROOT)),
                        cell=f"{r['variant']} {r['loud']} / target {r['target']}")
    return None


def from_g6(metric):
    p = pick_ana('odaq_g6.tsv', 'odaq')
    if p is None:
        return None
    for r in read_tsv(p):
        if r['col'] == metric and (r['variant'], r['loud']) == ('native', 'raw'):
            return dict(state='value', value=fnum(r['g6a']), ci=parse_ci(r['g6a_ci']),
                        n=int(r['n_trials_g6a']), stat='G6a within-trial ρ', src=str(p.relative_to(ROOT)),
                        cell='native raw / 30 trials')
    return None


def from_pairwise(ds, metric):
    p = pick_ana(f'{ds}_pairwise.tsv', ds)
    if p is None:
        return None
    for r in read_tsv(p):
        if r['col'] != metric:
            continue
        return dict(state='value', value=fnum(r['spearman']), ci=parse_ci(r['spearman_ci']),
                    n=int(r['n_items']), stat='G1 ρ（逐檔勝率）', src=str(p.relative_to(ROOT)),
                    cell='native raw', agree=dict(value=fnum(r['agree_m']), ci=parse_ci(r['agree_m_ci']),
                                                  n_pairs=int(r['n_pairs']), m=fnum(r['m']),
                                                  forced=fnum(r['agree_forced']), tie_frac=fnum(r['tie_frac'])))
    return None


def r1_cell(bench, target, metric):
    if not R1_CELLS.exists():
        return None
    for r in json.loads(R1_CELLS.read_text()):
        if r['bench'] == bench and r['target'] == target and r['metric'] == metric:
            return dict(state='value', value=fnum(r['spearman'][0]), ci=r['spearman'][1:],
                        n=int(r['n']), stat='ρ（round-1 協定）',
                        src='research/eval/newmodels_20261020/analysis_cells.json',
                        cell=f'target {target}')
    return None


def extra_cell(ds, metric):
    p = EXTRA / f'{ds}_extra_cells.tsv'
    if not p.exists():
        return None
    for r in read_tsv(p):
        if r['metric'] != metric:
            continue
        # analyze_extra_cells writes each stat as "[point, ci_lo, ci_hi]"
        point, lo, hi = (float(x) for x in r['spearman'].strip('[]').split(','))
        return dict(state='value', value=point, ci=[lo, hi],
                    n=int(r['n']), stat='ρ（round-1 協定）', src=str(p.relative_to(ROOT)),
                    cell=f"target {r['target']}")
    return None


def musicprefs_cell(metric):
    p = EXTRA / 'musicprefs_pairwise.tsv'
    if not p.exists():
        return None
    out = []
    for r in read_tsv(p):
        if r['col'] != metric:
            continue
        out.append(dict(state='value', value=fnum(r.get('spearman')), ci=parse_ci(r.get('spearman_ci')),
                        n=int(r['n_items']), stat='ρ（逐檔勝率）', src=str(p.relative_to(ROOT)),
                        cell=f"{r['axis']} 軸 / 成對 {r['n_pairs']}",
                        agree=dict(value=fnum(r['agree_m']), ci=parse_ci(r['agree_m_ci']),
                                   n_pairs=int(r['n_pairs']), m=fnum(r['m']), forced=fnum(r['agree_forced']),
                                   tie_frac=fnum(r['tie_frac']), within_sys=fnum(r.get('within_sys')))))
    if not out:
        return None
    # musicality is the primary axis (the one the paper reports first); keep fidelity as a second read
    out.sort(key=lambda c: 0 if c['cell'].startswith('musicality') else 1)
    cell = dict(out[0])
    cell['axes'] = out
    return cell


PENDING_J6 = {('se_mus', 'odaq'), ('se_mus', 'mc522'), ('se_mus', 'aime'), ('se_mus', 'singmos'),
              ('se_mus', 'arena'), ('singmos', 'arena'), ('singmos', 'songeval'),
              ('se_mus', 'pam400'), ('singmos', 'pam400'), ('clap', 'pam400'),
              ('se_mus', 'me'), ('singmos', 'me'), ('clap', 'me')}


def get_cell(metric, col):
    if metric == 'clap' and col in NO_CAPTION:
        return dict(state='na', why='此資料集沒有人類文字，CLAP 無法定義')
    val = None
    if col == 'odaq':
        val = from_g6(metric)
    elif col in ('mc522', 'songeval', 'singmos'):
        val = from_corr(col, metric)
    elif col in ('aime', 'arena'):
        val = from_pairwise(col, metric)
    elif col == 'pam400':
        val = r1_cell('PAM400', 'pam_OVL', metric) or extra_cell('pam400', metric)
    elif col == 'me':
        val = r1_cell('ME', 'human_OVL', metric) or extra_cell('me', metric)
    elif col == 'musicprefs':
        val = musicprefs_cell(metric)
    if val is not None:
        val['flag'] = flag(metric, col)
        return val
    if (metric, col) in PENDING_J6:
        return dict(state='pending', job='159', why='排隊中（job 159_bench_r2_J6_extra）')
    if col == 'musicprefs':
        return dict(state='pending', job='160', why='排隊中（job 160_bench_r2_J7_musicprefs）')
    return dict(state='na', why='沒有這一格（該指標在此資料集缺分數且未排隊）')


def build():
    rows = []
    for metric, label in ROWS:
        cells = {c['key']: get_cell(metric, c['key']) for c in COLS}
        rows.append(dict(metric=metric, label=label, cells=cells))
    pairwise = {}
    for col in ('aime', 'arena', 'musicprefs'):
        ent = []
        for metric, label in ROWS:
            c = next(r for r in rows if r['metric'] == metric)['cells'][col]
            if c['state'] == 'value' and c.get('agree'):
                ent.append(dict(metric=metric, label=label, flag=c['flag'], **c['agree']))
        pairwise[col] = ent
    return dict(generated_at=datetime.now().strftime('%Y-%m-%d %H:%M'),
                seed_r2=SEED_R2, seed_r1=SEED_R1, nboot=NBOOT,
                root=str(ROOT), columns=[dict(c) for c in COLS], rows=rows, pairwise=pairwise)


# ── markdown ───────────────────────────────────────────────────────────────
def fmt_cell(c, dec=3):
    if c['state'] == 'value':
        v = c['value']
        s = 'n/a' if v is None else f'{v:.{dec}f}'
        if c.get('ci'):
            s += f" [{c['ci'][0]:.{dec}f}, {c['ci'][1]:.{dec}f}]"
        return s, c['flag']
    if c['state'] == 'pending':
        return '排隊中', '·'
    return '—', '·'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out-docs', default=str(OUT))
    a = ap.parse_args()
    data = build()
    out = Path(a.out_docs)
    out.mkdir(parents=True, exist_ok=True)
    js = out / 'benchmark_screening_full_matrix_20261027.json'
    js.write_text(json.dumps(data, indent=1, ensure_ascii=False, default=float))

    n_val = sum(1 for r in data['rows'] for c in r['cells'].values() if c['state'] == 'value')
    n_pen = sum(1 for r in data['rows'] for c in r['cells'].values() if c['state'] == 'pending')
    n_na = sum(1 for r in data['rows'] for c in r['cells'].values() if c['state'] == 'na')
    n_cells = len(data['rows']) * len(COLS)

    L = []
    w = L.append
    w('# 指標 × 人類評分：完整矩陣（2026-10-27）')
    w('')
    w('**這份文件把 round 2 海選從 4 個資料集擴成 9 個，但沒有改任何 r2 預登錄的門檻、種子或判定。**'
      'r2 的結論（六個指標排除、SingMOS 未定、本輪沒有候選）原樣成立；這裡多出來的格子是**事後描述**，'
      '不進 G1／G2／G3／G6 的判定，也不改 `bench_r2_*_summary.json`。')
    w('')
    w(f'* 產生時間：{data["generated_at"]}；由 `scripts/eval/bench_r2/build_full_matrix.py` 自動組表（重跑即更新）。')
    w(f'* 格子狀態：**{n_val} 格有數**、**{n_pen} 格排隊中**、**{n_na} 格不適用**（共 {n_cells} 格）。')
    w(f'* 種子：r2 `{SEED_R2}`／r1 `{SEED_R1}`；cluster bootstrap B={NBOOT}。')
    w('')
    w('## 0. 三種數字不要混著讀')
    w('')
    w('| 統計量 | 用在哪幾欄 | 意思是什麼 |')
    w('|---|---|---|')
    w('| G6a within-trial ρ | ODAQ（B1） | 同一 trial 內，指標排序是否跟專家排序一致（ODAQ 沒有逐檔絕對分數） |')
    w('| G1 ρ（逐檔／逐曲） | MC522、SongEval、SingMOS、AIME、Arena、PAM400、MusicEval、MusicPrefs | 指標分數 vs 人類分數（成對偏好資料集的人類分數＝該檔在成對比較裡的勝率） |')
    w('| 成對一致率 agree_m | AIME、Arena、MusicPrefs（另表） | 指標說「A 比 B 好」時跟人類同向的比例；差距 < m = 0.10×S 算平手給 0.5 |')
    w('')
    w('人類天花板（`benchmark_screening_r2_human_ceiling_20261027.md`）：G1 欄位用 √SB，成對欄位用 BT held-out。'
      '**「佔天花板」＝ ρ ÷ 天花板**，這才是可比的分數；不同資料集的天花板差很多（Arena 的 ρ 上限本來就低，因為每檔只有 1–2 票）。')
    w('')

    for title, keys in (('1. 主表 A — 生成音樂（人類聽生成系統）',
                         ['songeval', 'aime', 'arena', 'musicprefs', 'pam400', 'me']),
                        ('2. 主表 B — 訊號、真實錄音、歌唱', ['odaq', 'mc522', 'singmos'])):
        w(f'## {title}')
        w('')
        cols = [c for c in COLS if c['key'] in keys]
        w('| 指標 | ' + ' | '.join(f"{c['head']}" for c in cols) + ' |')
        w('|---|' + '---|' * len(cols))
        for r in data['rows']:
            cells = []
            for c in cols:
                s, fg = fmt_cell(r['cells'][c['key']], 2 if len(cols) > 3 else 3)
                cells.append(s if fg in ('i', '·') else f'{s} ({fg})')
            w(f"| {r['label']} | " + ' | '.join(cells) + ' |')
        w('| **人類天花板** | ' + ' | '.join(
            (f"{c['ceil']['v']:.2f} ({c['ceil']['how']})" if c.get('ceil') else 'n/a') for c in cols) + ' |')
        w('| **n** | ' + ' | '.join(f"{c['units']:,}" for c in cols) + ' |')
        w('')
        w('旗標：`i` 標籤獨立；`t` 訓練／重疊過（不算證據）；`d` 該指標自己的設計資料（只描述）。'
          '格內是 ρ [95% CI]；「排隊中」表示分數還沒算（job 159／160），不是 0 也不是沒過。')
        w('')

    w('## 3. 逐格明細（含 n、來源檔）')
    w('')
    for c in COLS:
        w(f"### {c['head']}（{c['dom']}）")
        w('')
        ceiling = f"{c['ceil']['v']:.3f}（{c['ceil']['how']}）" if c.get('ceil') else c.get('ceiling_note', 'n/a')
        w(f"* 人類評分：{c['human']}")
        w(f"* 統計量：{c['stat']}；人類天花板：{ceiling}；規劃單位數 {c['units']:,}")
        w('')
        nh = 'n（trial 數）' if c['key'] == 'odaq' else 'n（評分單位／item）'
        w(f'| 指標 | 值 | 95% CI | {nh} | 旗標 | 來源 |')
        w('|---|---|---|---|---|---|')
        for r in data['rows']:
            cc = r['cells'][c['key']]
            if cc['state'] == 'value':
                ci = cc.get('ci')
                cis = f'[{ci[0]:.3f}, {ci[1]:.3f}]' if ci else '—'
                w(f"| {r['label']} | {cc['value']:.3f} | {cis} | {cc['n']:,} | {cc.get('flag','')} | "
                  f"{cc.get('src','')} · {cc.get('cell','')} |")
            else:
                w(f"| {r['label']} | {cc['why']} | — | — | — | — |")
        w('')

    w('## 4. 成對偏好：一致率與人類天花板')
    w('')
    w('三個資料集的人類標籤是「A 比 B 好」，逐檔 ρ 只是它的投影；真正對得上標籤的是成對一致率。')
    w('')
    w('| 資料集 | 人類天花板 | 指標 | 一致率（m=0.10S） | 95% CI | 強制比較 | 平手率 m | 對數 | 旗標 |')
    w('|---|---|---|---|---|---|---|---|---|')
    ceil = {'aime': '0.685（BT held-out，G3）[0.675, 0.696]',
            'arena': '0.742（BT held-out，G3）[0.715, 0.765]',
            'musicprefs': '待 job 160 算（BT held-out）'}
    for col in ('aime', 'arena', 'musicprefs'):
        if not data['pairwise'][col]:
            w(f"| {col} | {ceil[col]} | （排隊中） | — | — | — | — | — | — |")
            continue
        for e in data['pairwise'][col]:
            ci = e.get('ci') or [float('nan'), float('nan')]
            w(f"| {col} | {ceil[col]} | {e['label']} | {e['value']:.3f} | [{ci[0]:.3f}, {ci[1]:.3f}] | "
              f"{e.get('forced', float('nan')):.3f} | {e.get('tie_frac', float('nan')):.3f} | {e['n_pairs']:,} | {e['flag']} |")
    w('')
    w('**Arena 的 ρ 為什麼一定低**：每檔只出現在 1–2 場比較，人類「分數」其實是 0/0.5/1 的勝率；'
      'G1 的 ρ 上限由 BT held-out 0.742 決定，不是 1.0。AES PQ 在 Arena 的 ρ 0.149 看起來很差，'
      '但它的成對一致率 0.584 高於 0.55 門檻——兩者並不矛盾（ρ 對只有兩三種取值的 target 很敏感）。')
    w('')

    w('## 5. 排隊中的格子（job 159／160）')
    w('')
    w('| job | 腳本 | contract | 補哪些格 | 預估 |')
    w('|---|---|---|---|---|')
    w('| 159 | `gpu_queue/p2/pending/159_bench_r2_J6_extra.sh` | `docs/experiments/bench_r2_J6_extra_contract.json` | SongEval predictor × ODAQ／MC522／AIME／SingMOS／Arena；SingMOS predictor × SongEval；SongEval／SingMOS／CLAP × PAM400／MusicEval；**另外重跑 arena 成對分析，補 SingMOS × Arena 那一列** | 1.5–2.5 h |')
    w('| 160 | `gpu_queue/p2/pending/160_bench_r2_J7_musicprefs.sh` | `docs/experiments/bench_r2_J7_musicprefs_contract.json` | MusicPrefs 整欄 9 個指標（5,030 檔、2,515 對、2 軸） | 2–3.5 h |')
    w('')
    w('編號從 **159** 起，因為 **153–158 已被 autoresearch 推論線預訂**（`docs/experiments/autoresearch_inference_clap_20261027.md` §10.3）。')
    w('SingMOS × Arena 的分數早在 job 152（J5c）就評完（`scores/arena/singmos__native_raw.tsv`，3,998 檔、0 缺），'
      '但 `analysis/arena_pairwise.tsv` 寫得比它早，所以表上暫時是「排隊中」；159 重跑 arena 分析就會補上。')
    w('兩支 job 都不改 r2 的格子；跑完後重跑 `build_full_matrix.py` 就會把「排隊中」換成數字。')
    w('')
    w('## 6. 不適用的格子')
    w('')
    w('* **CLAP × ODAQ／SingMOS／SongEval**：這三個資料集沒有人類文字（ODAQ 只有參考音、SingMOS 只有 MOS、SongEval 只有五維評分），'
      'CLAP 是文字–音訊對齊分數，沒有 caption 就沒有定義。')
    w('* **PAM400／MusicEval 的人類天花板**：釋出的是「每檔 10 人平均」，沒有逐位評分者資料，算不出 split-half reliability。')
    w('')
    w('## 7. 旗標表（誰訓練過誰）')
    w('')
    w('沿用 `benchmark_screening_r2_overlap_audit_20261027.md`；本表只多加 PAM400／MusicEval 兩欄，'
      '並把該檔的證據逐格列出。')
    w('')
    w('| 指標 | ' + ' | '.join(c['head'] for c in COLS) + ' |')
    w('|---|' + '---|' * len(COLS))
    for r in data['rows']:
        w(f"| {r['label']} | " + ' | '.join(r['cells'][c['key']].get('flag', '·') for c in COLS) + ' |')
    w('')
    w('* MEva 的五個訓練 benchmark 是 MusicEval／SongEval／MusicPref／AIME／Music Arena'
      '（`runtime/meva_20261003/paper_layout.txt:55`，pooled checkpoint），所以那五格都是 `t`；'
      'B1／B4／B5 不在其中 → `i`。')
    w('* MuQ-Eval A1／A3a、CMI-RM 只訓練在 MusicEval → MusicEval 欄 `t`，其餘 `i`。')
    w('* SongEval predictor 訓練在 SongEval → 只有 SongEval 欄 `t`；SingMOS predictor 訓練在 SingMOS-Pro → 只有 SingMOS 欄 `t`。')
    w('* PAM 是零樣本 CLAP，不是學出來的評分器；但 PAM400 是它論文裡選 prompt／調設計用的資料 → `d`（只描述）。')
    w('* AES 對 B2／B3／B4／B6 的**生成軌**是標籤＋音訊獨立；B1／B5 與真實錄音子集的音訊重疊仍無法核實（維持 `i`，但別寫「沒見過」）。')
    w('')
    w('## 8. 假設與限制')
    w('')
    w('1. **MusicPrefs 沒有授權宣告**：只供內部研究，不重新散布；音訊快取在 `~/eval_output_nvme/bench_r2_derived/musicprefs`（8.7 GB），'
      '不放進任何對外產物。')
    w('2. **PAM400／MusicEval 的 audio cache 走 symlink**（`derived/<ds>/` → 原始檔），沒有 re-encode；'
      'MusicPrefs 是從 parquet 一次性解成 float32 wav（44.1／32／16 kHz 原樣保留）。')
    w('3. **MusicPrefs 的 ρ 是事後投影**：人類標籤是成對偏好，逐檔 ρ 用「該檔的勝率」當 target，'
      '與 AIME／Arena 在 r2 的做法一致；主讀數仍以成對一致率為準。')
    w('4. **MusicEval 的 ρ 用全部 2,748 檔**（round-1 協定），成對一致率只在 `shared=True` 的子集上算。')
    w('5. 這份矩陣**不是**新的預登錄：新增的 12 格以上全部是描述性；任何「某指標在某格看起來過了」都不能回頭改 r2 判定。')
    w('')
    w('## 9. 重跑指令')
    w('')
    w('```bash')
    w('cd ~/MeanAudio')
    w('# 1) 只重組表格（不快取任何模型；讀 analysis/ + analysis_extra/）')
    w('/home/kojiek/venvs/dac/bin/python -I scripts/eval/bench_r2/build_full_matrix.py')
    w('')
    w('# 2) 格子跑完後（job 159／160）先驗證再重組')
    w('GPU_QUEUE_CONTRACT=docs/experiments/bench_r2_J6_extra_contract.json \\')
    w('  /home/kojiek/venvs/dac/bin/python -I scripts/eval/bench_r2/run_extra.py --validate-only')
    w('```')
    md = out / 'benchmark_screening_full_matrix_20261027.md'
    md.write_text('\n'.join(L) + '\n')
    print(f'wrote {md}')
    print(f'wrote {js}')
    print(f'cells: {n_val} value / {n_pen} pending / {n_na} n/a of {n_cells}')


if __name__ == '__main__':
    main()
