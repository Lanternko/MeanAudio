"""Inject ttm_score_distributions_126.json into page_template.html -> standalone HTML page."""
import json, sys
from pathlib import Path
here = Path(__file__).parent
d = json.load(open(sys.argv[1]))
d['metric_order'] = ['PQ', 'CE', 'CU', 'PC', 'MEva', 'clap']
for m in d['metrics'].values():
    for b in m['by_system'].values():
        b['density'] = [round(v, 4) for v in b['density']]
d.pop('scatter_ids', None)
for k, v in d['scatter'].items():
    for m in v:
        v[m] = [round(x, 3) for x in v[m]]
html = (here / 'page_template.html').read_text().replace('__DATA__', json.dumps(d, ensure_ascii=False, separators=(',', ':')))
Path(sys.argv[2]).write_text(html)
print(sys.argv[2], len(html))
