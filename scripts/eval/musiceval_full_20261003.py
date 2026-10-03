"""Frozen full-corpus MusicEval comparison: cached AES PQ/CE/CU + fixed MEva.

This is an in-domain descriptive evaluation, not an independent MEva test split.
"""
import argparse
import csv
import json
import math
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from meva_runtime import ROOT, Evaluator, sha
sys.path.insert(0, str(ROOT / 'scripts/experiment_harness'))
from notification_receipts import atomic_secure_json

RUN = ROOT / 'runtime/musiceval_full_20261003'
CONTRACT = ROOT / 'docs/experiments/musiceval_full_20261003_contract.json'
REPORT = ROOT / 'docs/experiments/results/musiceval_full_20261003.json'
NAMES = ['PQ', 'CE', 'CU', 'MEva']


def cached(row, binding):
    path = RUN / 'scores' / (row['key'] + '.json')
    if not path.exists():
        return None
    value = json.loads(path.read_text())
    if value.get('input_sha256') != row['audio_sha256'] or value.get('binding') != binding:
        raise ValueError('Stale score binding: ' + row['key'])
    if value.get('status') != 'scored' or not math.isfinite(value['meva_raw']):
        raise ValueError('Invalid score: ' + row['key'])
    return value


def preflight():
    c = json.loads(CONTRACT.read_text())
    bundle = Path(c['harn_bundle'])
    b = json.loads((bundle / 'contract.json').read_text())
    expected = next(a['sha256'] for a in b['corpus']['source_artifacts'] if a['path'] == str(CONTRACT))
    assert sha(CONTRACT) == expected, 'Contract changed'
    approval = json.loads((bundle / 'preflight.json').read_text())['approval_evidence']
    assert datetime.now(timezone.utc) < datetime.fromisoformat(approval['expires_at']), 'Approval expired'
    assert sha(c['approval_record']) == approval['channel_record_sha256'], 'Operator record changed'
    for item in c['raw_bindings']:
        assert sha(item['path']) == item['sha256'], 'Binding changed: ' + item['path']
    lock = json.loads((ROOT / 'docs/experiments/meva_095_model_lock.json').read_text())
    for item in lock['files']:
        assert sha(item['path']) == item['sha256'], 'MEva runtime/model changed: ' + item['path']
    from importlib.metadata import version
    for package, expected in c['package_versions'].items():
        assert version(package) == expected, 'Package changed: ' + package
    webhook = Path('/home/kojiek/.config/meanaudio/discord_webhook_url').stat()
    assert webhook.st_uid == os.geteuid() and webhook.st_mode & 0o777 == 0o600
    rows = json.loads((RUN / 'manifest.json').read_text())
    assert len(rows) == 2748 and len({r['key'] for r in rows}) == 2748
    for row in rows:
        assert sha(row['audio_path']) == row['audio_sha256'], 'Audio changed: ' + row['key']
    fs = os.statvfs(RUN)
    if fs.f_bavail * fs.f_frsize < c['storage']['hard_stop_free_bytes']:
        return 75
    print('PASS: all 2748 inputs, AES table, labels, model/runtime hashes and storage', flush=True)
    return 0


def score(rows, binding):
    import torch
    torch.set_num_threads(4)
    model = None
    for i, row in enumerate(rows):
        value = cached(row, binding)
        if value is None:
            assert sha(row['audio_path']) == row['audio_sha256']
            fs = os.statvfs(RUN)
            if fs.f_bavail * fs.f_frsize < 50 << 30:
                raise RuntimeError('Storage hard stop')
            if model is None:
                model = Evaluator()
            tick = time.monotonic()
            value = model.score(row['audio_path'])
            value.update(key=row['key'], input_sha256=row['audio_sha256'],
                         binding=binding, seconds=time.monotonic()-tick)
            atomic_secure_json(RUN / 'scores' / (row['key'] + '.json'), value)
        atomic_secure_json(RUN / 'progress.json', {'index': i+1, 'total': len(rows),
                           'last_key': row['key'], 'updated_epoch': time.time()})
        if (i+1) % 100 == 0:
            print(f'MusicEval {i+1}/{len(rows)} MEva={value["meva_raw"]:.4f}', flush=True)


def metrics(d):
    import numpy as np
    from scipy.stats import pearsonr, spearmanr
    h = d.MI.to_numpy()
    hp = h - d.groupby('system').MI.transform('mean').to_numpy()
    result = {}
    for name in NAMES:
        x = d[name].to_numpy()
        xp = x - d.groupby('system')[name].transform('mean').to_numpy()
        means = d.groupby('system')[[name, 'MI']].mean()
        result[name] = dict(n=len(d), mean=float(x.mean()), std=float(x.std()),
            min=float(x.min()), max=float(x.max()), pearson=float(pearsonr(x, h).statistic),
            spearman=float(spearmanr(x, h).statistic),
            within_system_pearson=float(pearsonr(xp, hp).statistic),
            system_mean_spearman=float(spearmanr(means[name], means.MI).statistic),
            alignment_pearson=float(pearsonr(x, d.TA).statistic))
    return result


def summarize(rows, binding):
    import numpy as np
    import pandas as pd
    from scipy.stats import pearsonr, spearmanr
    records = [cached(r, binding) for r in rows]
    assert all(v is not None for v in records), 'Incomplete corpus; no final results'
    d = pd.DataFrame([{**r, 'MEva': v['meva_raw']} for r, v in zip(rows, records)])
    assert len(d) == 2748 and np.isfinite(d[NAMES+['MI','TA']].to_numpy()).all()
    shared = d[d.system.isin(d.groupby('system').prompt.nunique().loc[lambda x: x == 100].index)]
    assert len(shared) == 2500 and shared.prompt.nunique() == 100
    left, right = [], []
    for _, group in shared.groupby('prompt'):
        ix = group.index.to_numpy()
        a, b = np.triu_indices(len(ix), k=1)
        left.extend(ix[a]); right.extend(ix[b])
    left, right = np.array(left), np.array(right)
    dh = d.MI.to_numpy()[left] - d.MI.to_numpy()[right]
    use = dh != 0
    pairs = {}
    for name in NAMES:
        dx = d[name].to_numpy()[left] - d[name].to_numpy()[right]
        hits = np.where(dx[use] == 0, .5, (dx[use] * dh[use] > 0).astype(float))
        pairs[name] = {'agreement': float(hits.mean()), 'n_pairs': int(use.sum()),
                       'human_ties_excluded': int((~use).sum()), 'model_ties_half_credit': True}
    # Paired cluster bootstrap: retain all systems for each resampled prompt.
    rng = np.random.default_rng(20261003)
    groups = [g.index.to_numpy() for _, g in d.groupby('prompt')]
    sample = []
    h = d.MI.to_numpy(); xs = d[NAMES].to_numpy()
    for _ in range(2000):
        ix = np.concatenate([groups[i] for i in rng.integers(0,len(groups),len(groups))])
        sample.append([[pearsonr(xs[ix,j],h[ix]).statistic,
                        spearmanr(xs[ix,j],h[ix]).statistic] for j in range(4)])
    sample = np.asarray(sample)
    full = metrics(d)
    for j, name in enumerate(NAMES):
        full[name]['pearson_ci95'] = np.quantile(sample[:,j,0],[.025,.975]).tolist()
        full[name]['spearman_ci95'] = np.quantile(sample[:,j,1],[.025,.975]).tolist()
    differences = {name: {'delta_spearman': full['MEva']['spearman']-full[name]['spearman'],
        'simultaneous_ci95': np.quantile(sample[:,3,1]-sample[:,j,1],[.05/6,1-.05/6]).tolist()}
        for j,name in enumerate(NAMES[:3])}
    report = dict(status='completed', n_scored=len(d), n_systems=d.system.nunique(),
        manifest_sha256=sha(RUN/'manifest.json'), model_binding=binding,
        scope='full MusicEval descriptive in-domain; not official held-out or external validity',
        aes_source='existing full-corpus frozen raw AES scores; no retraining or calibration',
        caveat='AES S013_P013 used first90s; MEva uses full349s. Report excluding this clip separately.',
        training_overlap='MusicEval belongs to MEva training benchmarks; per-clip overlap unknown',
        bootstrap={'unit':'prompt','repeats':2000,'seed':20261003}, full=full,
        shared_2500=metrics(shared), same_prompt_pairs_shared_2500=pairs,
        exclude_aes_90s_exception=metrics(d[d.key!='audiomos2025-track1-S013_P013']),
        meva_minus_aes=differences, aes_replaced=False)
    columns=['key','system','prompt','duration_sec','MI','TA',*NAMES,'audio_sha256']
    d[columns].to_csv(RUN/'per_clip.tsv',sep='\t',index=False)
    d.groupby('system')[['MI','TA',*NAMES]].mean().to_csv(RUN/'per_system.tsv',sep='\t')
    atomic_secure_json(REPORT, report)
    lines=['# MusicEval 完整 PQ / CE / CU / MEva 比較（2026-10-03）','',
        '2,748 首 / 31 系統。MEva 固定 pooled-small-f03；AES 沿用凍結的完整逐曲結果。',
        '此為訓練資料來源內的描述性比較，非 MEva 官方 held-out 重現，也非外部泛化驗證。','',
        '| 指標 | Pearson [95% CI] | Spearman [95% CI] | 系統內 Pearson | 同 prompt 勝負一致率 |',
        '|---|---|---|---|---|']
    for name in NAMES:
        v=full[name]
        fmt=lambda k: f'{v[k]:.3f} [{v[k+"_ci95"][0]:.3f}, {v[k+"_ci95"][1]:.3f}]'
        lines.append(f'| {name} | {fmt("pearson")} | {fmt("spearman")} | {v["within_system_pearson"]:.3f} | {pairs[name]["agreement"]:.1%} |')
    lines.extend(['','Pearson：分數升降相關；Spearman：排序相關，均非正確率。',
        '系統內：兩邊扣除各系統平均後的 Pearson。配對只用共用 100 prompts × 25 系統的 2,500 首；真人平手排除，模型平手算半分。',
        'CI 為 prompt cluster bootstrap 2,000 次；MEva−三個 AES 的差異 CI 有 Bonferroni 同時校正。','',
        '視窗例外：既有 AES 對 S013_P013（349秒）評前90秒；MEva 評完整音檔。JSON 另附排除此音檔的2,747首敏感度分析。',
        'MEva 可能見過本資料的訓練部分，因此高相關不可推論 PromptCC 或 PAM 的外部泛化。AES 維持使用。','',
        f'逐曲結果：`{RUN}/per_clip.tsv`；系統平均：`{RUN}/per_system.tsv`；完整JSON：`{REPORT}`。'])
    REPORT.with_suffix('.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps({'n':len(d),'full':full,'pairs':pairs},indent=2),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--preflight',action='store_true')
    parser.add_argument('--validate-only',action='store_true')
    args=parser.parse_args()
    rc=preflight()
    if rc or args.preflight:
        return rc
    c=json.loads(CONTRACT.read_text()); rows=json.loads((RUN/'manifest.json').read_text())
    if not args.validate_only:
        score(rows,c['model_binding'])
    summarize(rows,c['model_binding'])
    return 0


if __name__=='__main__':
    raise SystemExit(main())
