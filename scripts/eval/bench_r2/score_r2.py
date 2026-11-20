#!/usr/bin/env python3
"""Score the round-2 benchmark cache with the preregistered metrics (resumable, row level).

  dac venv : python score_r2.py --dataset odaq --job J1 [--models aes,pam,...] [--smoke 4]
  MEva venv: runtime/meva_20261003/venv/bin/python score_r2.py --dataset arena --job J5 --models meva

Heavy imports are lazy (the MEva venv has neither CLAP nor MuQ).  Existing scorers are imported, not copied:
newmetrics_{muqeval,cmirm,pam}, eval_metrics.{score_aes,score_clap}, metric_win_tie_126d (SongEval),
metric_win_tie_133 (MERT/MuQ embeddings), meva_runtime.Evaluator.

Output: <ROOT>/scores/<dataset>/<model>__<variant>_<loud>.tsv   (unit_id, item_id, n_win, <cols>)
        failures: ...<model>__<variant>_<loud>.err.tsv (unit_id, error); never silently dropped.
Windowed units (VFULL, long native audio scored by fixed-window models) are averaged: <col> = mean over
windows, plus <col>_med / <col>_first for VFULL.  MEva scores the whole track (the `full` file).
`--smoke N` writes to <ROOT>/scores_smoke and scores only the first N units of each plan entry.
"""
import argparse
import base64
import csv
import json
import math
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # scripts/eval
import importlib.util  # noqa: E402
_sp = importlib.util.spec_from_file_location('bench_r2_datasets', Path(__file__).with_name('bench_datasets.py'))
D = importlib.util.module_from_spec(_sp)  # not `import datasets`: would shadow the HuggingFace package
sys.modules['bench_r2_datasets'] = D
_sp.loader.exec_module(D)

EV = Path('/home/kojiek/MeanAudio/scripts/eval')
CAPTION_DS = {'mc522', 'aime', 'arena'}
MODELS_BY_DS = {
    'odaq': ['aes', 'meva', 'muqeval_A1', 'muqeval_A3a', 'cmirm', 'pam', 'singmos', 'embed'],
    'mc522': ['aes', 'meva', 'muqeval_A1', 'muqeval_A3a', 'cmirm', 'pam', 'clap', 'singmos'],
    'aime': ['aes', 'meva', 'muqeval_A1', 'muqeval_A3a', 'cmirm', 'pam', 'clap', 'singmos'],
    'singmos': ['aes', 'meva', 'muqeval_A1', 'muqeval_A3a', 'cmirm', 'pam', 'singmos'],
    'songeval': ['aes', 'meva', 'muqeval_A1', 'muqeval_A3a', 'cmirm', 'pam', 'songeval'],
    'arena': ['aes', 'meva', 'muqeval_A1', 'muqeval_A3a', 'cmirm', 'pam', 'clap'],
}
CHUNK = {'aes': 256, 'clap': 128}  # units per call for models whose loader is expensive per call
CROP_CLAP_SEC = 10.0


def log(*a):
    print(time.strftime('%T'), *a, flush=True)


def abspath(ds, name):
    return str(D.DER / ds / name)


# ───────────────────────── scorer back-ends: f(unit) -> list[dict] (one per window) ─────────────────────────
class Aes:
    cols = ['aes_PQ', 'aes_CE', 'aes_CU', 'aes_PC']

    def __init__(self, ds):
        self.ds = ds

    def score_units(self, units):
        from eval_metrics import score_aes
        paths = [abspath(self.ds, f) for u in units for f in u['files'].split(';')]
        per, failed = score_aes(paths, progress=False)
        out = {}
        for u in units:
            ws, bad = [], None
            for f in u['files'].split(';'):
                p = abspath(self.ds, f)
                if p in per:
                    ws.append({'aes_PQ': per[p]['PQ'], 'aes_CE': per[p]['CE'], 'aes_CU': per[p]['CU'],
                               'aes_PC': per[p]['PC']})
                else:
                    bad = failed.get(p, 'missing')
            out[u['unit_id']] = ws if bad is None else RuntimeError(f'aes failed: {bad}')
        return out


class PerWindow:
    """Wraps f(path, unit) -> dict for models without batching."""

    def __init__(self, ds, fn, cols):
        self.ds, self.fn, self.cols = ds, fn, cols

    def score_units(self, units):
        out = {}
        for u in units:
            try:
                out[u['unit_id']] = [self.fn(abspath(self.ds, f), u) for f in u['files'].split(';')]
            except Exception as e:
                out[u['unit_id']] = e
        return out


def make_muqeval(ds, model):
    from newmetrics_muqeval import score_muqeval
    v = model.split('_')[1]
    return PerWindow(ds, lambda p, u: {f'{model}_{k}': x for k, x in score_muqeval(p, None, variant=v).items()},
                     [f'{model}_MI', f'{model}_TA'])


def make_cmirm(ds):
    from newmetrics_cmirm import score_cmirm
    modes = ['empty'] + (['caption'] if ds in CAPTION_DS else [])

    def fn(p, u):
        o = {}
        for m in modes:
            r = score_cmirm(p, None, prompt_mode=m, caption=u['caption'])
            o[f'cmi_{m}_quality'], o[f'cmi_{m}_alignment'] = r['quality'], r['alignment']
        return o
    return PerWindow(ds, fn, [f'cmi_{m}_{k}' for m in modes for k in ('quality', 'alignment')])


def make_pam(ds):
    from newmetrics_pam import score_pam
    return PerWindow(ds, lambda p, u: {'pam': score_pam(p, None)['pam']}, ['pam'])


class Meva:
    cols = ['meva']

    def __init__(self, ds):
        self.ds = ds
        from meva_runtime import Evaluator
        self.m = Evaluator()

    def score_units(self, units):
        out = {}
        for u in units:
            f = u['full'] or u['files'].split(';')[0]
            try:
                out[u['unit_id']] = [{'meva': self.m.score(abspath(self.ds, f))['meva_raw']}]
            except Exception as e:
                out[u['unit_id']] = e
        return out


class SongEval:
    cols = ['se_coh', 'se_mem', 'se_nat', 'se_cla', 'se_mus']

    def __init__(self, ds):
        import torch
        import metric_win_tie_126d as S
        self.S, self.ds = S, ds
        self.dev = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.muq, self.gen = S.load_songeval(self.dev)

    def fixed30(self, path):
        import numpy as np
        w = self.S.load_24k(path)
        n = 30 * self.S.SE_SR
        if len(w) >= n:
            s = (len(w) - n) // 2
            return w[s:s + n]
        return np.concatenate([w, np.zeros(n - len(w), dtype=np.float32)])

    def score_units(self, units):
        out = {}
        for u in units:
            try:
                fs = u['files'].split(';')
                sc = []
                for i in range(0, len(fs), 8):
                    wavs = [self.fixed30(abspath(self.ds, f)) for f in fs[i:i + 8]]
                    sc += list(self.S.score_songeval_batch(wavs, self.muq, self.gen, self.dev))
                out[u['unit_id']] = [dict(zip(self.cols, map(float, s))) for s in sc]
            except Exception as e:
                out[u['unit_id']] = e
        return out


class Embed:
    """MERT-v1-95M and MuQ mean-pooled L2-normalised embeddings (24 kHz, 30 s chunks, tail < 5 s dropped)."""
    cols = ['mert_emb', 'muq_emb']
    string_cols = True

    def __init__(self, ds):
        import torch
        import metric_win_tie_133 as M
        self.M, self.ds = M, ds
        self.dev = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.feat, self.mert = M.load_mert(self.dev)
        self.muq = M.load_muq(self.dev)

    def score_units(self, units):
        import numpy as np
        out = {}
        n = 30 * self.M.EMBED_SR
        for u in units:
            try:
                f = u['full'] or u['files'].split(';')[0]
                w = self.M.load_24k(abspath(self.ds, f))
                chunks = [w[s:s + n] for s in range(0, len(w), n) if len(w[s:s + n]) >= 5 * self.M.EMBED_SR]
                chunks = chunks or [w]
                em = [self.M.embed_mert_batch([c.tolist()], self.feat, self.mert, self.dev)[0] for c in chunks]
                eq = [self.M.embed_muq_batch([c], self.muq, self.dev)[0] for c in chunks]
                row = {}
                for k, e in (('mert_emb', em), ('muq_emb', eq)):
                    v = np.mean(e, axis=0)
                    v = (v / max(np.linalg.norm(v), 1e-8)).astype(np.float16)
                    row[k] = base64.b64encode(v.tobytes()).decode()
                out[u['unit_id']] = [row]
            except Exception as e:
                out[u['unit_id']] = e
        return out


class Clap:
    cols = ['clap']

    def __init__(self, ds, tag):
        from eval_metrics import load_clap
        self.ds = ds
        self.model = load_clap()
        self.tmp = D.ROOT / 'tmp_clap' / tag
        self.tmp.mkdir(parents=True, exist_ok=True)

    def score_units(self, units):
        import soundfile as sf
        from eval_metrics import score_clap
        rows, out = [], {}
        for i, u in enumerate(units):
            try:
                a, sr = sf.read(abspath(self.ds, u['files'].split(';')[0]), dtype='float32')
                n = int(CROP_CLAP_SEC * sr)
                if len(a) > n:
                    s = (len(a) - n) // 2
                    a = a[s:s + n]
                sf.write(str(self.tmp / f'u{i}.flac'), a, sr, subtype='PCM_16')
                rows.append((f'u{i}', u['caption']))
            except Exception as e:
                out[u['unit_id']] = e
        per = score_clap(rows, self.tmp, model=self.model, progress=False)
        for i, u in enumerate(units):
            if u['unit_id'] in out:
                continue
            out[u['unit_id']] = [{'clap': per[f'u{i}']}] if f'u{i}' in per else RuntimeError('clap missing')
        for p in self.tmp.glob('u*.flac'):
            p.unlink()
        return out


def make_singmos(ds):
    """SingMOS-Pro predictor via torch.hub (local clone of South-Twilight/SingMOS@v1.1.2), 16 kHz, whole clip."""
    sys.path.insert(0, str(D.ROOT / 'pylib_singmos'))
    import numpy as np
    import soundfile as sf
    import torch
    import types
    import torchaudio
    # the v1.1.2 repo imports torchaudio's removed sox API at import time; the predictor never uses it
    torchaudio.set_audio_backend = lambda *a, **k: None
    se = types.ModuleType('torchaudio.sox_effects')
    se.apply_effects_tensor = se.apply_effects_file = None
    sys.modules['torchaudio.sox_effects'] = se
    torchaudio.sox_effects = se
    repo = Path.home() / '.cache/torch/hub/South-Twilight_SingMOS_v1.1.2'
    m = torch.hub.load(str(repo), 'singmos_pro', source='local', trust_repo=True).cuda().eval()

    @torch.no_grad()
    def fn(p, u):
        a, sr = sf.read(p, dtype='float32', always_2d=True)
        t = torch.from_numpy(a.mean(axis=1))
        if sr != 16000:
            t = torchaudio.functional.resample(t, sr, 16000)
        t = t.view(1, 1, -1).cuda()  # [B, 1, T]
        v = m(t, torch.tensor([t.shape[-1]]).cuda())
        return {'singmos': float(np.asarray(v.float().cpu()).reshape(-1)[0])}
    return PerWindow(ds, fn, ['singmos'])


def _takes_len(m):
    import inspect
    return len(inspect.signature(m.forward).parameters) >= 2


def build(model, ds, tag):
    if model == 'aes':
        return Aes(ds)
    if model == 'meva':
        return Meva(ds)
    if model.startswith('muqeval'):
        return make_muqeval(ds, model)
    if model == 'cmirm':
        return make_cmirm(ds)
    if model == 'pam':
        return make_pam(ds)
    if model == 'songeval':
        return SongEval(ds)
    if model == 'embed':
        return Embed(ds)
    if model == 'clap':
        return Clap(ds, tag)
    if model == 'singmos':
        return make_singmos(ds)
    raise SystemExit(f'unknown model {model}')


# ───────────────────────── driver ─────────────────────────
def aggregate(ws, cols, variant):
    row = {'n_win': len(ws)}
    for c in cols:
        if isinstance(ws[0][c], str):
            row[c] = ws[0][c]
            continue
        vals = sorted(w[c] for w in ws)
        row[c] = sum(w[c] for w in ws) / len(ws)
        if variant == 'VFULL':
            mid = len(vals) // 2
            row[c + '_med'] = vals[mid] if len(vals) % 2 else (vals[mid - 1] + vals[mid]) / 2
            row[c + '_first'] = ws[0][c]
    return row


def read_tsv_ids(path):
    if not Path(path).exists():
        return set(), None
    with open(path, newline='') as f:
        rd = csv.DictReader(f, delimiter='\t')
        ids = {r['unit_id'] for r in rd}
        return ids, rd.fieldnames


def load_index(ds):
    with open(D.DER / ds / 'index.tsv', newline='') as f:
        return {r['unit_id']: r for r in csv.DictReader(f, delimiter='\t')}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', required=True)
    ap.add_argument('--job', required=True)
    ap.add_argument('--models', default='')
    ap.add_argument('--smoke', type=int, default=0)
    a = ap.parse_args()
    ds = a.dataset
    models = a.models.split(',') if a.models else MODELS_BY_DS[ds]
    out_root = (D.ROOT / 'scores_smoke' if a.smoke else D.SCORES) / ds
    out_root.mkdir(parents=True, exist_ok=True)
    idx = load_index(ds)
    items = D.read_manifest(ds)
    for model in models:
        t_model = time.time()
        scorer = None
        for variant, loud, subset in D.PLAN[ds][a.job]:
            us = D.units(ds, variant, loud, subset, items)
            if a.smoke:
                us = us[:a.smoke]
            label = f'{model}__{D.variant_label(variant, loud)}'
            f_out, f_err = out_root / f'{label}.tsv', out_root / f'{label}.err.tsv'
            done, fields = read_tsv_ids(f_out)
            done_err, _ = read_tsv_ids(f_err)
            todo, n_inv = [], 0
            for u in us:
                ix = idx.get(u['unit_id'])
                if ix is None:
                    raise SystemExit(f'{u["unit_id"]} missing from index (run decode_cache.py)')
                if ix['invalid']:
                    n_inv += 1
                    continue
                if u['unit_id'] in done or u['unit_id'] in done_err:
                    continue
                todo.append(dict(u, files=ix['files'], full=ix['full']))
            log(f'{ds} {a.job} {label}: {len(us)} units, {len(done)} done, {len(done_err)} err, {n_inv} invalid, '
                f'{len(todo)} todo')
            if not todo:
                continue
            if scorer is None:
                scorer = build(model, ds, f'{ds}_{a.job}_{model}')
            step = CHUNK.get(model, 50)
            wr = we = None
            fo = open(f_out, 'a', newline='')
            fe = open(f_err, 'a', newline='')
            if not f_err.stat().st_size:
                csv.writer(fe, delimiter='\t').writerow(['unit_id', 'error'])
            t0, n_ok, n_bad = time.time(), 0, 0
            for i in range(0, len(todo), step):
                res = scorer.score_units(todo[i:i + step])
                for u in todo[i:i + step]:
                    r = res[u['unit_id']]
                    ok = isinstance(r, list) and r
                    if ok:
                        row = aggregate(r, scorer.cols, variant)
                        ok = all(math.isfinite(v) for v in row.values() if isinstance(v, float))
                    if not ok:
                        n_bad += 1
                        csv.writer(fe, delimiter='\t').writerow([u['unit_id'], repr(r)[:300] if not isinstance(r, list)
                                                                 else 'nonfinite'])
                        continue
                    row = dict(unit_id=u['unit_id'], item_id=u['item_id'], **row)
                    if wr is None:
                        if not f_out.stat().st_size:
                            fields = list(row)
                        wr = csv.DictWriter(fo, fieldnames=fields, delimiter='\t')
                        if not f_out.stat().st_size:
                            wr.writeheader()
                    wr.writerow(row)
                    n_ok += 1
                fo.flush()
                fe.flush()
                if (i // step) % 5 == 0:
                    log(f'  {label} {min(i + step, len(todo))}/{len(todo)} ok={n_ok} err={n_bad} '
                        f'{time.time() - t0:.0f}s')
            fo.close()
            fe.close()
            log(f'{label} done ok={n_ok} err={n_bad} {time.time() - t0:.0f}s')
        del scorer
        try:
            import torch
            torch.cuda.empty_cache()
        except Exception:
            pass
        log(f'MODEL {model} finished {time.time() - t_model:.0f}s')


if __name__ == '__main__':
    main()
