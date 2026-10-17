#!/usr/bin/env python3
"""128: frozen-input calibration and secondary re-adjudication of historical scores.

All transforms and random seeds come from 126a; historical level matching is 115's
function. The P2 supervisor owns launch, storage, notifications and termination.
Only completed, hash-bound score units may be resumed.
"""
import os
for _name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ.setdefault(_name, '1')
os.environ.pop('MKL_NUM_THREADS', None)

import argparse
import csv
import hashlib
import io
import json
import math
import multiprocessing as mp
import shutil
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parent))
import metric_win_tie_126a as A
import ttm_metrics_115 as P115

R = Path('/home/kojiek/MeanAudio')
GIB = 2 ** 30
MODES = ('C0', 'C1', 'C2', 'C3')
METRICS = ('PQ', 'CE', 'CU', 'PC', 'clap', 'meva')
JUDGED = ('PQ', 'CE', 'CU', 'clap')
TIES = ('gain-1', 'gain+1', 'gain-3', 'gain+3', 'gain-6', 'gain+6',
        'polarity', 'pcm16', 'resample44k', 'lead10ms', 'lead50ms', 'lead200ms')
LADDERS = {'noise': ['noise_snr40', 'noise_snr30', 'noise_snr20', 'noise_snr10'],
           'clip': ['clip3db', 'clip6db', 'clip12db'],
           'bitcrush': ['bitcrush8', 'bitcrush6'], 'clicks': ['clicks1ps', 'clicks5ps'],
           'dropout': ['dropout20ms_x3', 'dropout20ms_x10']}
VARIANTS = ['clean', 'clean_shuf', 'gain-1', 'gain+1', 'gain-3', 'gain-6', 'gain-10',
            'gain-20', 'gain+3', 'gain+6', 'polarity', 'pcm16', 'resample44k',
            'lead10ms', 'lead50ms', 'lead200ms', 'mp3_160k', 'mp3_64k',
            *[v for ladder in LADDERS.values() for v in ladder],
            'blast80ms_x1', 'blast80ms_x3', 'shuffle0.25s', 'reverse']
MIDI_CASES = ('clean', 'blastnote_x1', 'blastnote_x3', 'blastnote_x10', 'vel_x0.85',
              'gain-3', 'gain-10', 'lead10ms', 'polarity', 'wrong100pct', 'posshuffle100pct')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(4 << 20), b''):
            h.update(b)
    return h.hexdigest()


def hash_json(v):
    return hashlib.sha256(json.dumps(v, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f'.tmp.{os.getpid()}')
    with tmp.open('w') as f:
        json.dump(value, f, indent=2, sort_keys=True, allow_nan=False)
        f.write('\n'); f.flush(); os.fsync(f.fileno())
    os.replace(tmp, path)


def table(path, key='id'):
    with Path(path).open() as f:
        return {r[key]: r for r in csv.DictReader(f, delimiter='\t')}


def config():
    return json.loads(Path(os.environ['GPU_QUEUE_CONTRACT']).read_text())


def notify(c, event, text, status='start'):
    sys.path.insert(0, str(R / 'scripts/experiment_harness'))
    from notification_receipts import deliver_required
    deliver_required(contract_path=Path(os.environ['GPU_QUEUE_CONTRACT']),
                     launcher_path=Path(os.environ['GPU_QUEUE_JOB_SCRIPT']),
                     event=event, status=status, summary=text,
                     idempotency_key=f"{c['experiment_id']}:{c['run_id']}:{event}",
                     notifier=R / 'scripts/notify_experiment_webhook.py',
                     root=Path(c['notification_receipts']['root']))


def storage(c):
    for spec in c['filesystems']:
        free = shutil.disk_usage(spec['path']).free
        if free < spec['hard_floor_bytes']:
            print(f"storage hard stop: {spec['path']} free={free} floor={spec['hard_floor_bytes']}",flush=True)
            raise SystemExit(75)


def normalize(x, sr):
    import pyloudnorm
    level = float(pyloudnorm.Meter(sr).integrated_loudness(x))
    if not math.isfinite(level):
        raise ValueError('C1/C3 LUFS undefined; refuse silently substituting raw audio')
    y = x * 10 ** ((-30.0 - level) / 20)
    if not np.isfinite(y).all():
        raise ValueError('nonfinite canonicalized audio')
    return y


def members(x, sr, mode):
    y = normalize(x, sr) if mode in ('C1', 'C3') else x
    if mode in ('C0', 'C1'):
        return [y]
    n = len(y)
    return [np.pad(y[:n-k] if k else y, (k, 0)) * sign
            for ms, sign in ((0, 1), (40, -1), (80, 1), (120, -1))
            for k in [int(sr * ms / 1000)]]


def prepare_one(job):
    """One source -> exact historical variant quantization -> scoring views.

    WAV/FLOAT containers with .flac names preserve >0 dBFS C1/C3 samples. The
    existing file-based scorers sniff the container; no limiter or integer clip.
    """
    rec, system, variant, mode, tmp = job
    src, ident = Path(rec['path']), rec['id']
    if sha(src) != rec['sha256']:
        raise ValueError(f'input hash changed: {src}')
    x, sr = sf.read(src, dtype='float64')
    if x.ndim != 1:
        raise ValueError(f'expected historical mono input: {src}')
    if rec.get('raw_115'):
        p = Path(tmp) / f'{ident}.level.flac'
        P115._gain_one((str(src), str(p), -30.0))
        if rec.get('lvl30_sha256') and sha(p) != rec['lvl30_sha256']:
            raise ValueError(f'115 raw-to-lvl30 byte identity failed: {ident}')
        x, sr = sf.read(p, dtype='float64'); p.unlink()
    over, variant_capped = 0, False
    if variant in A.VARIANTS:
        var = A.VARIANTS[variant]
        if var['op']=='gain':
            variant_capped=bool(np.any(np.abs(x*10**(var['arg']/20))>.999))
        y, over = A.transform(x, var, A.rng_for(system, variant, ident), sr)
        buf = io.BytesIO()
        sf.write(buf, y, sr, format='FLAC', subtype=var['subtype'])
        buf.seek(0); x, sr = sf.read(buf, dtype='float64')
    paths, peaks = [], []
    for k, y in enumerate(members(x, sr, mode)):
        p = Path(tmp) / f'{ident}__e{k}.flac'
        sf.write(p, y, sr, format='WAV', subtype='FLOAT')
        paths.append(str(p)); peaks.append(float(np.max(np.abs(y))))
    return {'id': ident, 'paths': paths, 'over_fs': over, 'peaks': peaks,'variant_capped':variant_capped}


class Scorer:
    def __init__(self, c):
        self.c = c
        self.pool = ProcessPoolExecutor(max_workers=4, mp_context=mp.get_context('spawn'))
        from eval_metrics import load_clap
        self.clap = load_clap()

    def close(self):
        self.pool.shutdown()

    def unit(self, group, system, variant, mode, records):
        c = self.c; root = Path(c['storage']['path'])
        directory = root / group / system / variant / mode
        directory.mkdir(parents=True, exist_ok=True)
        identity = hash_json({'action': c['bindings']['action_sha256'], 'mode': mode,
                              'system': system, 'variant': variant, 'records': records})
        score_path, done = directory / 'scores.npz', directory / 'done.json'
        if done.exists():
            meta = json.loads(done.read_text())
            if meta['identity'] != identity or sha(score_path) != meta['score_sha256']:
                raise RuntimeError(f'resume unit identity changed: {directory}')
            with np.load(score_path) as z:
                return {k: z[k] for k in z.files}, meta
        storage(c)
        t0 = time.monotonic()
        tmp = Path(tempfile.mkdtemp(prefix='audio-', dir=c['storage']['transient_root']))
        child = None
        try:
            prepared = list(self.pool.map(prepare_one,
                [(r, system, variant, mode, str(tmp)) for r in records], chunksize=8))
            prepared_seconds = time.monotonic()-t0
            rows = [(Path(p).stem, r['caption']) for r, data in zip(records, prepared) for p in data['paths']]
            paths = [Path(p) for data in prepared for p in data['paths']]
            if mode in ('C0', 'C3'):
                mv_dir = directory / 'meva'; mv_dir.mkdir(exist_ok=True)
                with (directory / 'meva.log').open('w') as f:
                    child = subprocess.Popen([A.MEVA_PY, str(R / 'scripts/eval/meva_score_dir.py'),
                            '--audio_dir', str(tmp), '--out_dir', str(mv_dir)], stdout=f, stderr=subprocess.STDOUT)
            aes_clap_started = time.monotonic()
            from eval_metrics import score_aes, score_clap
            if variant == 'clean_shuf':
                import random
                random.Random(126).shuffle(paths)
            aes, failed = score_aes(paths, progress=False)
            if failed or len(aes) != len(paths):
                raise RuntimeError(f'AES incomplete: {directory}')
            aes = {Path(p).stem: v for p, v in aes.items()}
            clap = score_clap(rows, tmp, model=self.clap, progress=False) if mode != 'C1' else {}
            if mode != 'C1' and len(clap) != len(paths):
                raise RuntimeError('CLAP incomplete')
            aes_clap_seconds = time.monotonic()-aes_clap_started
            mv = {}
            if child is not None:
                if child.wait() != 0:
                    raise RuntimeError(f'MEva failed: {mv_dir}')
                mv = {i: float(v['meva_raw']) for i, v in table(mv_dir / 'meva_per_clip.tsv').items()}
                if len(mv) != len(paths):
                    raise RuntimeError('MEva incomplete')
            # MEva's own seconds starts after model construction. Fixed model load
            # latency must be charged per unit, not multiplied by N_a / 50.
            meva_compute = json.loads((mv_dir/'meva.json').read_text())['seconds'] if mv else 0.
            # File scorers run AES then CLAP while MEva runs concurrently.
            # Their elapsed time is recorded below independently of child wait.
            values, deviations = [], []
            for data in prepared:
                sub = []
                for p in data['paths']:
                    key = Path(p).stem
                    sub.append([aes[key][m] if m in aes[key] else
                                clap.get(key, np.nan) if m == 'clap' else mv.get(key, np.nan)
                                for m in METRICS])
                sub = np.asarray(sub)
                required = [0, 1, 2, 3] + ([4] if mode != 'C1' else []) + ([5] if mv else [])
                if not np.isfinite(sub[:, required]).all():
                    raise RuntimeError('nonfinite required score')
                values.append(sub.mean(0)); deviations.append(sub.std(0))
            out = {'ids': np.array([r['id'] for r in records]), 'values': np.array(values),
                   'sd': np.array(deviations), 'over_fs': np.array([d['over_fs'] for d in prepared]),
                   'variant_capped':np.array([d['variant_capped'] for d in prepared]),
                   'peak': np.array([max(d['peaks']) for d in prepared])}
            scratch = directory / 'scores.tmp.npz'
            np.savez(scratch, **out); os.replace(scratch, score_path)
            meta = {'identity': identity, 'n': len(records), 'seconds': time.monotonic()-t0,
                    'score_sha256': sha(score_path), 'mode': mode,
                    'members': 4 if mode in ('C2', 'C3') else 1,
                    'prepare_seconds':prepared_seconds,'meva_compute_seconds':meva_compute,
                    'aes_clap_seconds':aes_clap_seconds,
                    'scalable_seconds':prepared_seconds+max(aes_clap_seconds,meva_compute)}
            meta['setup_seconds']=max(0.,meta['seconds']-meta['scalable_seconds'])
            write(done, meta)
            write(root / 'progress.json', {'group': group, 'system': system, 'variant': variant,
                  'mode': mode, 'n': len(records), 'at': time.time()})
            A.log(f'{group}/{system}/{variant}/{mode} n={len(records)} {meta["seconds"]:.1f}s')
            return out, meta
        finally:
            if child is not None and child.poll() is None:
                child.terminate(); child.wait(timeout=30)
            shutil.rmtree(tmp)


def rates(delta, margin):
    return {'n': len(delta), 'tie': float(np.mean(np.abs(delta) <= margin)),
            'better': float(np.mean(delta > margin)), 'worse': float(np.mean(delta < -margin)),
            'mean_delta': float(np.mean(delta))}


def threshold(cells, ladder):
    return next((i for i, v in enumerate(ladder) if cells[v]['worse'] >= .8), len(ladder))


def select_modes(cells):
    """Cells[mode][system][metric][variant]; never infer a passing missing gate."""
    verdicts, chosen = {}, {}
    for metric in JUDGED:
        verdicts[metric] = {}
        for mode in MODES:
            if mode == 'C1' and metric == 'clap':
                continue
            failures = []
            for system in A.SYSTEMS:
                row, raw = cells[mode][system][metric], cells['C0'][system][metric]
                for v in TIES:
                    if row[v]['n'] == 0 or row[v]['tie'] < .8:
                        failures.append(f'{system}/{v}/tie')
                for name, ladder in LADDERS.items():
                    t0, t1 = threshold(raw, ladder), threshold(row, ladder)
                    if t1 > t0 + 1:
                        failures.append(f'{system}/{name}/threshold')
                    if t1 < len(ladder) and row[ladder[t1]]['better'] > raw[ladder[t1]]['better'] + .05:
                        failures.append(f'{system}/{name}/reversal')
            verdicts[metric][mode] = {'passed': not failures, 'failures': failures}
        passes = [m for m in MODES if verdicts[metric].get(m, {}).get('passed')]
        chosen[metric] = {'mode': passes[0] if passes else 'C3', 'sensitivity_only': not passes}
    return {'verdicts': verdicts, 'chosen': chosen}


# Identity gate, amendment 2026-10-17 (docs/experiments/harn/metric_invariance_128/identity_gate_amendment_20261017.json).
# AES moves by up to 0.005 when only the forward batch size changes (126a clean vs clean_shuf: all 22
# clips over 1e-4 sit in the two 11-clip tail batches), so a 1e-4 per-clip bound fails on machine noise.
# AES: per-clip max <= 0.01 and |mean signed diff| <= 0.001 (catches a small systematic offset such as
# a level error that stays under the per-clip bound). CLAP / MEva reproduce to ~5e-7 and keep 1e-4.
AES_AXES = ('PQ', 'CE', 'CU', 'PC')
IDENTITY = {'aes_max_abs': 0.01, 'aes_abs_mean_signed': 0.001, 'other_max_abs': 1e-4}


def check_c0(got, references, metrics, gate=IDENTITY):
    diffs, failures = {}, []
    for metric in metrics:
        j = METRICS.index(metric)
        d = [float(value)-float(references[str(i)][metric]) for i, value in zip(got['ids'], got['values'][:, j])]
        mx, mean = max(abs(x) for x in d), sum(d)/len(d)
        diffs[metric] = {'max_abs': mx, 'mean_signed': mean}
        if metric in AES_AXES:
            if mx > gate['aes_max_abs']: failures.append(f'{metric} max {mx:.6f} > {gate["aes_max_abs"]}')
            if abs(mean) > gate['aes_abs_mean_signed']:
                failures.append(f'{metric} |mean| {abs(mean):.6f} > {gate["aes_abs_mean_signed"]}')
        elif mx > gate['other_max_abs']:
            failures.append(f'{metric} max {mx:.2e} > {gate["other_max_abs"]}')
    if failures:
        raise RuntimeError(f'C0 identity failed: {failures}; diffs={diffs}')
    return diffs


def run_a(c, scorer, manifest):
    root = Path(c['storage']['path']); smoke_path = root / 'smoke.json'
    if not smoke_path.exists():
        timing, identity, setup = {}, {}, {}
        for system, records in manifest['calibration'].items():
            timing[system],setup[system] = {},{}
            for mode in MODES:
                got, meta = scorer.unit('smoke', system, 'clean', mode, records[:50])
                timing[system][mode] = meta['scalable_seconds'] / 50
                setup[system][mode] = meta['setup_seconds']
                if mode == 'C0':
                    ref = table(A.OUT / 'units/clean' / system / 'scores.tsv')
                    identity[system] = check_c0(got, ref, METRICS)
                    ref115 = table(manifest['reference_115'][system])
                    check_c0(got, ref115, ('PQ', 'CE', 'CU', 'PC', 'clap'))
        per_clip = sum(sum(v.values()) for v in timing.values())
        # Transform benchmark uses all 35 variants on 50 source clips. This also
        # charges MP3 roundtrip/correlation work instead of extrapolating clean.
        bench={}
        for variant in VARIANTS:
            tmp=Path(tempfile.mkdtemp(prefix='benchmark-',dir=c['storage']['transient_root']));started=time.monotonic()
            try:
                for s in manifest['calibration']:(tmp/s).mkdir()
                jobs=[(r,s,variant,'C3',str(tmp/s)) for s,rs in manifest['calibration'].items() for r in rs[:10]]
                list(scorer.pool.map(prepare_one,jobs,chunksize=2))
                bench[variant]=(time.monotonic()-started)/50
            finally:shutil.rmtree(tmp)
        extra_transform=max(0.,sum(bench.values())-len(VARIANTS)*bench['clean'])*5*4
        setup_cost=sum(sum(v.values()) for v in setup.values())*len(VARIANTS)
        # Include 25% for transforms / I/O and the complete MIDI block.
        estimates = {str(n): (per_clip*n*len(VARIANTS)+extra_transform*n+
                     setup_cost+per_clip/5*402*len(MIDI_CASES))*1.25 for n in (1000,600,400)}
        feasible = [n for n in (1000, 600, 400) if estimates[str(n)] <= 7 * 3600]
        smoke = {'identity': identity, 'timing': timing, 'setup_seconds':setup,
                 'transform_benchmark_seconds_per_clip':bench,'estimated_seconds': estimates,
                 'selected_n': feasible[0] if feasible else None, 'identity_gate': IDENTITY}
        write(smoke_path, smoke)
    smoke = json.loads(smoke_path.read_text())
    if smoke['selected_n'] is None:
        notify(c, 'smoke_budget_invalid', '128a: no registered N in {1000,600,400} fits 7h; no calibration promoted.', 'held')
        raise RuntimeError('registered smoke budget gate failed')
    notify(c, 'smoke_pass', f"128a C0 identity passed; frozen N_a={smoke['selected_n']}; starting calibration.")
    cells, clean_means = {}, {}
    n = smoke['selected_n']
    for mode in MODES:
        cells[mode], clean_means[mode] = {}, {}
        for system, all_records in manifest['calibration'].items():
            records = all_records[:n]
            raw = {}
            for variant in VARIANTS:
                raw[variant], _ = scorer.unit('calibration', system, variant, mode, records)
            clean = raw['clean']['values']
            clean_means[mode][system] = [None if not np.isfinite(v) else float(v) for v in clean.mean(0)]
            cells[mode][system] = {}
            for metric in JUDGED:
                if metric == 'clap' and mode == 'C1': continue
                j = METRICS.index(metric); margin = manifest['margins'][metric]
                cells[mode][system][metric] = {}
                for variant, got in raw.items():
                    keep = np.ones(n, dtype=bool)
                    if variant.startswith('gain+'):
                        keep = ~got['variant_capped']
                    cells[mode][system][metric][variant] = rates((got['values'][:, j]-clean[:, j])[keep], margin) if keep.any() else {'n': 0, 'tie': 0., 'better': 0., 'worse': 0., 'mean_delta': 0.}
    midi = {}
    for mode in MODES:
        midi[mode] = {}
        clean, _ = scorer.unit('midi', 'generaluser', 'source', mode, manifest['midi']['clean'])
        if mode == 'C0':
            refs = manifest['midi_reference']
            check_c0(clean, refs, ('PQ',))
        for case in MIDI_CASES:
            got = clean if case == 'clean' else scorer.unit('midi', 'generaluser', 'source_'+case, mode, manifest['midi'][case])[0]
            if list(got['ids']) != list(clean['ids']): raise RuntimeError('MIDI pairs not aligned')
            midi[mode][case] = rates(got['values'][:, 0]-clean['values'][:, 0], manifest['margins']['PQ'])
    result = {'status': 'completed', 'n_a': n, 'margins': manifest['margins'], 'cells': cells,
              'midi_pq': midi, 'clean_means': clean_means, **select_modes(cells),
              'q3_prediction_passed': midi['C1']['blastnote_x1']['better'] < .5}
    write(root / 'summary.json', result)
    notify(c, 'calibration_gate', '128a calibration complete; per-metric C* and sensitivity-only fallbacks persisted. 129 must verify this report before re-adjudication.')


def conclusion(deltas, pair_rates, margin):
    if (all(d > 0 for d in deltas) or all(d < 0 for d in deltas)) and all(abs(d) > margin for d in deltas) and all(r['better'] > r['worse'] for r in pair_rates):
        return 'still_holds'
    if abs(float(np.mean(deltas))) <= margin:
        return 'tie'
    return 'weakened'


def score_cells(c, scorer, manifest, part):
    root = Path(c['storage']['path'])
    calibration = json.loads(Path(c['calibration_summary']).read_text())
    cells = manifest['table_b'] if part == 'b' else manifest['negmf']
    results, reports = {}, {}
    for name, records in cells.items():
        results[name] = {}
        got, _ = scorer.unit('identity', name, 'clean', 'C0', records[:200])
        reference = manifest['table_references'].get(name)
        if reference:
            check_c0(got, table(reference), ('PQ', 'CE', 'CU', 'PC', 'clap'))
        else:
            # E1 has raw historical AES, not a 115 lvl30 score. Raw byte provenance
            # and the shared 115 transform are bound; do not relabel it as 115.
            if part != 'c': raise RuntimeError('missing required 115/118 identity scores')
        for mode in ('C1', 'C2', 'C3'):
            chunks = []
            for k in range(0, len(records), 1000):
                scored, _ = scorer.unit('table', name, f'chunk{k:05d}', mode, records[k:k+1000])
                chunks.append(scored)
            results[name][mode] = {key: np.concatenate([z[key] for z in chunks]) for key in chunks[0]}
        # C0 full scores are historical reference scores, with a live 200-clip gate.
        if reference:
            ref = table(reference)
            results[name]['C0'] = {'ids': np.array([r['id'] for r in records]),
                'values': np.array([[float(ref[r['id']].get(m, 'nan')) for m in METRICS] for r in records]),
                'sd': np.zeros((len(records),len(METRICS)))}
        else:
            chunks = [scorer.unit('table',name,f'chunk{k:05d}','C0',records[k:k+1000])[0] for k in range(0,len(records),1000)]
            results[name]['C0'] = {key: np.concatenate([z[key] for z in chunks]) for key in chunks[0]}
    for claim, metric, ours, baseline in manifest['claims'][part]:
        choice = calibration['chosen'][metric]; mode = choice['mode']; j = METRICS.index(metric)
        deltas, comparisons, noise = [], [], []
        for arm, base in zip(ours, baseline):
            x, y = results[arm][mode], results[base][mode]
            yi = {str(i): k for k, i in enumerate(y['ids'])}
            pairs = [(k, yi[str(i)]) for k,i in enumerate(x['ids']) if str(i) in yi]
            a,b = np.array(pairs).T
            delta = x['values'][a,j]-y['values'][b,j]
            combined = np.sqrt(results[arm]['C2']['sd'][a,j]**2+results[base]['C2']['sd'][b,j]**2)
            deltas.append(float(delta.mean())); comparisons.append(rates(delta,manifest['margins'][metric]))
            noise.append(float(np.mean(np.abs(delta)>2*combined)))
        reports[claim] = {'metric':metric,'mode':mode,'sensitivity_only':choice['sensitivity_only'],
            'deltas':deltas,'paired':comparisons,'above_2sd_fraction':noise,
            'verdict':conclusion(deltas,comparisons,manifest['margins'][metric])}
    rankings = {}
    from scipy.stats import kendalltau
    for metric in ('PQ','clap'):
        j=METRICS.index(metric); mode=calibration['chosen'][metric]['mode']
        x=[float(results[n]['C0']['values'][:,j].mean()) for n in cells]
        y=[float(results[n][mode]['values'][:,j].mean()) for n in cells]
        tau=float(kendalltau(x,y).statistic)
        rankings[metric]={'C0':dict(zip(cells,x)), 'Cstar':dict(zip(cells,y)), 'kendall_tau':tau if math.isfinite(tau) else None}
    write(root/'summary.json', {'status':'completed','part':part,'claims':reports,'rankings':rankings,
          'calibration_sha256':sha(c['calibration_summary']),
          'c2_ensemble_sd_means':{name:{metric:float(results[name]['C2']['sd'][:,METRICS.index(metric)].mean())
                                     for metric in ('PQ','CE','CU','PC','clap')} for name in cells}})
    if part == 'b':
        run_d(c, manifest, results, calibration)


def run_d(c, manifest, results, calibration):
    """Reference-side strata; never select clips by generated pulse clarity."""
    ref = table(manifest['real_mir'])
    common = sorted(set(ref) & {str(i) for i in results['musicgen_medium']['C0']['ids']})
    common = [i for i in common if ref[i].get('pulse_clarity','') not in ('','nan')]
    ordered = sorted(common, key=lambda i:(float(ref[i]['pulse_clarity']),i))
    # Ties in pulse clarity stay together, using reference quantile cutpoints.
    values=np.array([float(ref[i]['pulse_clarity']) for i in ordered]);q=np.quantile(values,[1/3,2/3])
    groups=[[i for i in ordered if int(np.searchsorted(q,float(ref[i]['pulse_clarity']),side='left'))==k] for k in range(3)]
    mode=calibration['chosen']['PQ']['mode']; per=[]
    mg=results['musicgen_medium'][mode]; lookup={str(i):v for i,v in zip(mg['ids'],mg['values'][:,0])}
    for seed in (14159265,16180339,27182818):
        s=results[f'meanaudio_s{seed}_cfg3_neg'][mode]
        own={str(i):v for i,v in zip(s['ids'],s['values'][:,0])}
        ids=[i for i in ordered if i in own and i in lookup]
        per.append({'seed':seed,'overall':float(np.mean([own[i]-lookup[i] for i in ids])),
                    'strata':[float(np.mean([own[i]-lookup[i] for i in g])) if g else None for g in groups]})
    overall=float(np.mean([p['overall'] for p in per]));means=[float(np.mean([p['strata'][k] for p in per])) if groups[k] else None for k in range(3)]
    reading='descriptive'
    if overall>0 and all(v is not None for v in means):
        if means[2]<.5*overall:reading='advantage_concentrated_in_weak_rhythm_prompts'
        elif all(.5*overall<=v<=1.5*overall for v in means):reading='advantage_independent_of_reference_rhythm'
    floors=json.loads(Path(manifest['floor_mir']).read_text())
    floor_entropy=[v['chroma_entropy'] for key,v in floors.items() if any(f'__{name}_' in key for name in ('sine1k','square100','square1k'))
                   and v.get('chroma_entropy') is not None]
    if not floor_entropy:raise RuntimeError('missing 126b sine/square MIR floors')
    low,high=min(floor_entropy),max(floor_entropy);floor_like={}
    for name,path in manifest['mir'].items():
        if name not in results:continue
        mir=table(path);scores=results[name][mode];pq={str(i):float(v) for i,v in zip(scores['ids'],scores['values'][:,0])}
        valid=[i for i,v in mir.items() if i in pq and v.get('pulse_clarity','') not in ('','nan') and v.get('chroma_entropy','') not in ('','nan')]
        selected=[i for i in valid if float(mir[i]['pulse_clarity'])<.1 and low<=float(mir[i]['chroma_entropy'])<=high]
        floor_like[name]={'n_valid':len(valid),'n_floor_like':len(selected),'fraction':len(selected)/len(valid) if valid else None,
                         'mean_pq':float(np.mean([pq[i] for i in selected])) if selected else None}
    write(Path(c['storage']['path'])/'summary_d.json',{'status':'completed','n_reference':len(ordered),
          'mode':mode,'cutpoints':q.tolist(),'stratum_counts':[len(g) for g in groups],
          'per_seed':per,'overall':overall,'stratum_means':means,'reading':reading,
          'floor_like_outputs':floor_like,'floor_entropy_range':[low,high]})


def preflight(c):
    storage(c)
    for path, expected in c['immutable_files'].items():
        if sha(path) != expected: raise RuntimeError(f'immutable binding changed: {path}')
    manifest=json.loads(Path(c['manifest']).read_text())
    groups=list(manifest['calibration'].values())+list(manifest['midi'].values())+list(manifest['table_b'].values())+list(manifest['negmf'].values())
    seen=set()
    for records in groups:
        for r in records:
            if r['path'] in seen:continue
            seen.add(r['path'])
            if sha(r['path'])!=r['sha256']:raise RuntimeError(f'input changed: {r["path"]}')
    if c['part'] in ('b','c'):
        summary=Path(c['calibration_summary'])
        if not summary.is_file():raise RuntimeError('scientific dependency 128a not complete')
        if json.loads(summary.read_text()).get('status')!='completed':raise RuntimeError('invalid calibration summary')
        for job,report in c['scientific_dependencies'].items():
            terminal=Path('/home/kojiek/gpu_queue/p2/done')/f'{job}.terminal.json'
            data=json.loads(terminal.read_text())
            evidence=data.get('evidence',{}).get('reports',[])
            match=[r for r in evidence if r['path']==report]
            if data.get('status')!='completed' or len(match)!=1 or sha(report)!=match[0]['sha256']:
                raise RuntimeError(f'unverified completed scientific predecessor: {job}')
    bundle=Path(c['harn_bundle'])
    typed=json.loads((bundle/'contract.json').read_text())
    for artifact in typed['corpus']['source_artifacts']:
        if sha(artifact['path'])!=artifact['sha256']:
            raise RuntimeError(f'schema contract artifact changed: {artifact["path"]}')
    subprocess.run([sys.executable,str(R/'scripts/validate_experiment_harness_documents.py'),
        '--contract',str(bundle/'contract.json'),'--preflight',str(bundle/'preflight.json'),
        '--ledger',str(bundle/'ledger.json'),'--queue',str(bundle/'queue.json')],check=True)
    storage(c)
    Path(c['storage']['transient_root']).mkdir(parents=True,exist_ok=True)
    print('128 preflight passed',flush=True)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--preflight',action='store_true');ap.add_argument('--validate-only',action='store_true')
    ap.add_argument('--part',choices=('a','b','c','d'),help='Must match bound contract; d reuses completed b scores on CPU.')
    args=ap.parse_args();c=config()
    if args.part and args.part!=c['part'] and not (args.part=='d' and c['part']=='b'):
        raise RuntimeError('--part must match the immutable contract')
    if args.part=='d':
        manifest=json.loads(Path(c['manifest']).read_text());calibration=json.loads(Path(c['calibration_summary']).read_text())
        mode=calibration['chosen']['PQ']['mode'];results={};root=Path(c['storage']['path'])
        for name,records in manifest['table_b'].items():
            refs=table(manifest['table_references'][name]);ids=[r['id'] for r in records]
            baseline={'ids':np.array(ids),'values':np.array([[float(refs[i].get(m,'nan')) for m in METRICS] for i in ids])}
            results[name]={'C0':baseline}
            if mode!='C0':
                chunks=[]
                for k in range(0,len(records),1000):
                    d=root/'table'/name/f'chunk{k:05d}'/mode
                    meta=json.loads((d/'done.json').read_text())
                    if sha(d/'scores.npz')!=meta['score_sha256']:raise RuntimeError('corrupt CPU-analysis input')
                    with np.load(d/'scores.npz') as z:chunks.append({key:z[key] for key in z.files})
                results[name][mode]={key:np.concatenate([z[key] for z in chunks]) for key in chunks[0]}
        run_d(c,manifest,results,calibration);return
    if args.preflight:preflight(c);return
    root=Path(c['storage']['path'])
    if args.validate_only:
        result=json.loads((root/'summary.json').read_text())
        if result.get('status') not in ('completed','deferred'):raise RuntimeError('incomplete summary')
        if result['status']=='deferred':
            if c['part']!='c' or result.get('elapsed_seconds',0)<=17*3600:raise RuntimeError('invalid deferral')
            return
        for done in root.glob('*/*/*/*/done.json'):
            meta=json.loads(done.read_text());score_path=done.parent/'scores.npz'
            if sha(score_path)!=meta['score_sha256']:raise RuntimeError(f'corrupt completed score unit: {done}')
            with np.load(score_path) as z:
                if len(z['ids'])!=meta['n'] or len(set(z['ids']))!=meta['n']:
                    raise RuntimeError('incomplete or duplicate scored ids')
                required=[0,1,2,3]+([4] if meta['mode']!='C1' else [])+([5] if meta['mode'] in ('C0','C3') else [])
                if z['values'].shape!=(meta['n'],len(METRICS)) or not np.isfinite(z['values'][:,required]).all():
                    raise RuntimeError('invalid scored metric matrix')
        if c['part']=='a':
            if set(result.get('chosen',{}))!=set(JUDGED):raise RuntimeError('missing calibration decisions')
            expected=len(VARIANTS)*len(A.SYSTEMS)*4
            if len(list((root/'calibration').glob('*/*/*/done.json')))!=expected:raise RuntimeError('missing calibration units')
            if len(list((root/'midi').glob('*/*/*/done.json')))!=len(MIDI_CASES)*4:raise RuntimeError('missing MIDI units')
        else:
            manifest=json.loads(Path(c['manifest']).read_text())
            selected=manifest['table_b'] if c['part']=='b' else manifest['negmf']
            expected=sum(math.ceil(len(rows)/1000)*3 for rows in selected.values())
            expected+=sum(math.ceil(len(rows)/1000) for name,rows in selected.items() if name not in manifest['table_references'])
            if len(list((root/'table').glob('*/*/*/done.json')))!=expected:
                raise RuntimeError('missing re-adjudication score units')
            if len(list((root/'identity').glob('*/*/*/done.json')))!=len(selected):
                raise RuntimeError('missing 200-clip identity units')
            if set(result.get('claims',{}))!={x[0] for x in manifest['claims'][c['part']]}:
                raise RuntimeError('missing re-adjudication claims')
            if c['part']=='b' and json.loads((root/'summary_d.json').read_text()).get('status')!='completed':
                raise RuntimeError('128d incomplete')
        return
    manifest=json.loads(Path(c['manifest']).read_text())
    if c['part']=='c':
        elapsed=time.time()-json.loads((Path(c['calibration_summary']).parent/'started.json').read_text())['at']
        if elapsed>17*3600:
            write(root/'summary.json',{'status':'deferred','reason':'128a+128b exceeded registered 17-hour budget','elapsed_seconds':elapsed})
            notify(c,'budget_deferred','130 / 128c deferred under registered 17h rule. No NegMF scores were recomputed.');return
    if c['part']=='a' and not (root/'started.json').exists():write(root/'started.json',{'at':time.time()})
    storage(c)
    tmp=Path(c['storage']['transient_root'])
    if tmp.is_symlink():raise RuntimeError('temporary scoring root must not be a symlink')
    # Only the supervisor holding this experiment's lock can start an action.
    # A killed unit has no completion evidence; discard only its registered scratch.
    if tmp.exists():shutil.rmtree(tmp)
    tmp.mkdir()
    scorer=Scorer(c)
    try:
        if c['part']=='a':run_a(c,scorer,manifest)
        else:score_cells(c,scorer,manifest,c['part'])
    finally:scorer.close()


if __name__=='__main__':main()
