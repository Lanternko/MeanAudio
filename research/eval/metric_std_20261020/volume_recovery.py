"""B: volume recovery check (pre-registered in docs/experiments/metric_standard_S_definition_20261020.md §4B).
Direct short job (~GPU < 30 min est.); outputs OUT/{waveform.tsv,scores.tsv,done.json}."""
import csv, json, math, random, shutil, subprocess, sys, time
from pathlib import Path
import numpy as np, soundfile as sf
REPO = Path('/home/kojiek/MeanAudio'); sys.path.insert(0, str(REPO / 'scripts/eval'))
EV = Path('/home/kojiek/eval_output_nvme'); OUT = EV / 'volume_recovery_20261020'
R115 = EV / 'ttm_metrics_115_20261009'
TSV = '/mnt/HDD/kojiek/phase4_jamendo_data/musiccaps_test.tsv'
MEVA_PY = REPO / 'runtime/meva_20261003/venv/bin/python'
N = 200; GS = (30, 60, 90)
PATHS = {'f32': ('FLOAT', 0), 'pcm24': ('PCM_24', 0), 'pcm16': ('PCM_16', 0), 'pcm16d': ('PCM_16', 1)}
METRICS = ('PQ', 'CE', 'CU', 'PC', 'meva', 'clap')

def log(m): print(time.strftime('%F %T'), m, flush=True)

def sample():
    caps = {}
    with open(TSV, encoding='utf-8', newline='') as f:
        for r in csv.DictReader(f, delimiter='\t'): caps[r['id']] = r['caption']
    ids = sorted(p.stem for p in (R115 / 'musiccaps_real/lvl30/audio').glob('*.flac')
                 if (R115 / 'meanaudio_cfg3_neg/lvl30/audio' / p.name).is_file() and p.stem in caps)
    rng = random.Random(20261020)
    cid = sorted(rng.sample(ids, N))
    items = json.loads((EV / 'metric_win_tie_126b_20261016/stimuli/items.json').read_text())
    midi = sorted((r for r in items if r['source'] == 'midi_generaluser' and r['kind'] == 'ref'), key=lambda r: r['key'])
    mid = rng.sample(midi, N)
    clips = []
    for i in cid:
        clips.append((f'real__{i}', str(R115 / 'musiccaps_real/lvl30/audio' / f'{i}.flac'), caps[i]))
        clips.append((f'ours__{i}', str(R115 / 'meanaudio_cfg3_neg/lvl30/audio' / f'{i}.flac'), caps[i]))
    for r in mid: clips.append((f'midi__{r["key"].split("__")[2]}', r['path'], None))
    return clips

def variants():
    v = [('clean', None, None, None)]
    for g in GS:
        for p in PATHS:
            v.append((f'g{g}_{p}_rec', g, p, True)); v.append((f'g{g}_{p}_unrec', g, p, False))
    return v

def store(x, g, p, rng, path):
    sub, dith = PATHS[p]
    y = x.astype(np.float64) * 10 ** (-g / 20)
    if dith: y = y + (rng.random(len(y)) + rng.random(len(y)) - 1) / 32768.0
    sf.write(path, y, 16000, format='WAV', subtype=sub)

def main():
    from eval_metrics import score_aes, score_clap, load_clap
    if shutil.disk_usage(OUT.parent).free < 30 * 2**30: raise SystemExit('disk')
    OUT.mkdir(exist_ok=True); tmp_root = OUT / 'tmp'; shutil.rmtree(tmp_root, ignore_errors=True)
    clips = sample(); ids = [c[0] for c in clips]; X = {}
    for k, p, _ in clips:
        x, sr = sf.read(p, dtype='float32'); assert sr == 16000 and x.ndim == 1, (p, sr, x.shape); X[k] = x
    json.dump({'n': len(clips), 'ids': ids}, open(OUT / 'sample.json', 'w'))
    clap_model = load_clap(); wf = []; sc_rows = []
    for name, g, p, rec in variants():
        d = tmp_root / name; d.mkdir(parents=True)
        for k in ids:
            x = X[k]
            if g is None:
                sf.write(d / f'{k}.flac', x, 16000, format='WAV', subtype='FLOAT'); continue
            rng = np.random.default_rng(zlib_hash(k, name))
            f = d / f'{k}.flac'; store(x, g, p, rng, f)
            if rec:
                y, _ = sf.read(f, dtype='float64'); y = y * 10 ** (g / 20)
                err = y - x.astype(np.float64)
                snr = 10 * math.log10(float(np.sum(x.astype(np.float64) ** 2)) / max(float(np.sum(err ** 2)), 1e-300))
                wf.append((name, k, snr, float(np.abs(err).max()), float(np.abs(y).max()), float(np.abs(x).max())))
                sf.write(f, y.astype(np.float32), 16000, format='WAV', subtype='FLOAT')  # scorer sees recovered float32
        mo = d / 'meva_out'; mo.mkdir()
        mp = subprocess.Popen([str(MEVA_PY), str(REPO / 'scripts/eval/meva_score_dir.py'), '--audio_dir', str(d), '--out_dir', str(mo)],
                              cwd=REPO, stdout=open(d / 'meva.log', 'w'), stderr=subprocess.STDOUT)
        aes, failed = score_aes([d / f'{k}.flac' for k in ids], progress=False)
        if failed or len(aes) != len(ids): raise SystemExit(f'{name}: AES {len(aes)}/{len(ids)}')
        aes = {Path(q).stem: v for q, v in aes.items()}
        crow = [(k, c) for k, _, c in clips if c is not None]
        clap = score_clap(crow, d, model=clap_model, progress=False)
        if mp.wait() != 0: raise SystemExit(f'{name}: meva failed')
        mv = {r['id']: float(r['meva_raw']) for r in csv.DictReader(open(mo / 'meva_per_clip.tsv'), delimiter='\t')}
        for k in ids:
            sc_rows.append((name, k, aes[k]['PQ'], aes[k]['CE'], aes[k]['CU'], aes[k]['PC'], mv[k], clap.get(k, float('nan'))))
        shutil.rmtree(d); log(f'done {name}')
        with open(OUT / 'scores.tsv', 'w', newline='') as fh:
            w = csv.writer(fh, delimiter='\t'); w.writerow(['variant', 'id', *METRICS]); w.writerows(sc_rows)
        with open(OUT / 'waveform.tsv', 'w', newline='') as fh:
            w = csv.writer(fh, delimiter='\t'); w.writerow(['variant', 'id', 'snr_db', 'max_err', 'max_rec', 'max_orig']); w.writerows(wf)
    shutil.rmtree(tmp_root, ignore_errors=True)
    (OUT / 'done.json').write_text(json.dumps({'n': len(clips), 'variants': len(variants())}))

def zlib_hash(k, name):
    import zlib; return zlib.crc32(f'{k}|{name}'.encode())

if __name__ == '__main__': main()
