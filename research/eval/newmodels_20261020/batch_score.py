"""Resumable per-clip scorer. usage: batch_score.py <model> ; model in muqeval_A1 muqeval_A3a cmirm pam
Writes scores_<col-set>_<bench>.tsv (key + score columns) next to this file; skips keys already present."""
import csv, sys, os, time
sys.path.insert(0, os.path.expanduser('~/MeanAudio/scripts/eval'))
HERE = os.path.dirname(os.path.abspath(__file__))
BENCH = {'mc522': 'bench_musiccaps522.tsv', 'pam500': 'bench_pam500.tsv', 'me2748': 'bench_musiceval2748.tsv'}
model = sys.argv[1]

if model.startswith('muqeval'):
    from newmetrics_muqeval import score_muqeval
    v = model.split('_')[1]
    jobs = {model: lambda p, c: {f'{model}_{k}': x for k, x in score_muqeval(p, None, variant=v).items()}}
elif model == 'pam':
    from newmetrics_pam import score_pam
    jobs = {model: lambda p, c: {'pam': score_pam(p)['pam']}}
elif model == 'cmirm':
    from newmetrics_cmirm import score_cmirm
    def f(p, c):
        o = {}
        for m in ('empty', 'caption'):
            r = score_cmirm(p, None, prompt_mode=m, caption=c)
            o[f'cmi_{m}_quality'] = r['quality']; o[f'cmi_{m}_alignment'] = r['alignment']
        return o
    jobs = {model: f}
fn = jobs[model]
for b, tsv in BENCH.items():
    rows = list(csv.DictReader(open(os.path.join(HERE, tsv)), delimiter='\t'))
    out = os.path.join(HERE, f'scores_{model}_{b}.tsv')
    done, cols = set(), None
    if os.path.exists(out):
        rd = list(csv.DictReader(open(out), delimiter='\t')); done = {r['key'] for r in rd}; cols = list(rd[0].keys()) if rd else None
    t0 = time.time(); n = 0
    fh = open(out, 'a', newline='')
    w = None
    for r in rows:
        if r['key'] in done: continue
        s = fn(r['path'], r['caption'])
        if w is None:
            cols = ['key'] + list(s.keys())
            w = csv.writer(fh, delimiter='\t')
            if os.path.getsize(out) == 0: w.writerow(cols)
        w.writerow([r['key']] + [f'{s[k]:.6f}' for k in cols[1:]]); n += 1
        if n % 100 == 0: fh.flush(); print(model, b, n, f'{time.time()-t0:.0f}s', flush=True)
    fh.close(); print('done', model, b, n, flush=True)
