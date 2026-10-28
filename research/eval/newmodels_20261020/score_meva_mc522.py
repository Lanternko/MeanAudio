"""MEva (pinned f03 evaluator, same runtime as meva_score_dir.py) on the MC522 wavs. Run with runtime/meva_20261003/venv python."""
import csv, sys, os
sys.path.insert(0, os.path.expanduser('~/MeanAudio/scripts/eval'))
from meva_runtime import Evaluator
from meva_score_dir import check_binding
check_binding()
rows = list(csv.DictReader(open('bench_musiccaps522.tsv'), delimiter='\t'))
m = Evaluator()
with open('scores_meva_mc522.tsv', 'w') as fh:
    fh.write('key\tmeva\n')
    for r in rows: fh.write(f"{r['key']}\t{m.score(r['path'])['meva_raw']:.6f}\n")
print('done', len(rows))
