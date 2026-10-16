import os,json,hashlib
from pathlib import Path
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    c=json.loads(Path(os.environ['GPU_QUEUE_CONTRACT']).read_text());root=Path(c['runtime_root'])
    r=json.loads((root/'summary.json').read_text());assert r['status']=='completed' and r['diagnostic_only'] and not r['production_promotion']
    assert r['contract_sha256']==sha(os.environ['GPU_QUEUE_CONTRACT']) and r['cache_sha256']==sha(r['cache'])
    rows=[json.loads(l) for l in Path(r['cache']).read_text().splitlines()]
    expected={x['cell']+'__'+x['id']:x for x in c['diagnostic_sources']}
    assert len(rows)==r['n_sources']==len(expected) and {x['key'] for x in rows}==set(expected)
    for x in rows:assert x['source']==expected[x['key']]
    print('POSTFLIGHT DIAGNOSTIC PASS; production success NOT implied')
if __name__=='__main__':main()
