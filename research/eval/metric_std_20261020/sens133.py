"""133 (MERT/MuQ cosim): D1-D3 undefined; sweep m = f * S_133 and count passing cells (same rule as metric_win_tie_133.judge_cell)."""
import csv, json
from pathlib import Path
import numpy as np
O = Path('/home/kojiek/eval_output_nvme/metric_win_tie_133_20261020')
E = Path('/home/kojiek/eval_output_nvme')
FS = [0.02, 0.05, 0.1, 0.2, 0.5, 1, 2, 5]
res = {}
for emb in ('mert', 'muq'):
    S = json.loads((O / f's_values_{emb}.json').read_text())[f'S_{emb}']
    for ds, items in (('126b', E / 'metric_win_tie_126b_20261016/stimuli/items.json'), ('126c', E / 'metric_win_tie_126c_20261019/stimuli/items.json')):
        cos = {r['key']: float(r[f'{emb}_cosim']) for r in csv.DictReader(open(O / f'{ds}_{emb}_cosim.tsv'), delimiter='\t')}
        by = {}
        for r in json.loads(items.read_text()):
            if r['base'] and r['kind'] in ('tie', 'lose') and r['key'] in cos:
                by.setdefault((r['case'], r['kind']), []).append(cos[r['key']])
        for f in FS:
            tie_p = tie_n = lose_p = lose_n = 0
            for (case, kind), v in by.items():
                v = np.array(v); v = v[np.isfinite(v)]
                t = 1 - f * S
                ok = np.mean(v >= t) if kind == 'tie' else np.mean(v < t)
                if kind == 'tie':
                    tie_n += 1; tie_p += ok >= 0.8
                else:
                    lose_n += 1; lose_p += ok >= 0.8
            res.setdefault(f'{ds}|{emb}', {})[str(f)] = {'S': S, 'm': f * S, 'tie_pass': f'{tie_p}/{tie_n}', 'lose_pass': f'{lose_p}/{lose_n}'}
Path('sens133.json').write_text(json.dumps(res, indent=1))
for k, v in res.items():
    print(k, 'S=%.4f' % list(v.values())[0]['S'])
    for f, x in v.items():
        print('  f=%-5s m=%.5f tie %s lose %s' % (f, x['m'], x['tie_pass'], x['lose_pass']))
