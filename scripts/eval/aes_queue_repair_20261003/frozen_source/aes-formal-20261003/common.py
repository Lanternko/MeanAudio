import os,json,hashlib,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parent; SRC=ROOT.parent/'aes-followup-20261002'; REPO=Path('/home/kojiek/MeanAudio'); PY='/home/kojiek/venvs/dac/bin/python'
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 return h.hexdigest()
def dump(p,d):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(d,ensure_ascii=False,indent=2,allow_nan=False)+'\n');t.replace(p)
def spec():return json.loads(Path(os.environ['GPU_QUEUE_CONTRACT']).read_text())
def notify(event,status,summary):
 sys.path.insert(0,str(REPO/'scripts/experiment_harness'))
 from notification_receipts import deliver_required
 c=spec();return deliver_required(contract_path=Path(os.environ['GPU_QUEUE_CONTRACT']),launcher_path=Path(os.environ['GPU_QUEUE_JOB_SCRIPT']),event=event,status=status,summary=summary,idempotency_key=f"{c['experiment_id']}:{c['run_id']}:{event}",notifier=Path(c['notification_receipts']['notifier']),root=Path(c['notification_receipts']['root']),python=Path(PY))
def load_records(path):
 path=Path(path)
 if not path.exists():return {}
 raw=path.read_bytes()
 if raw and not raw.endswith(b'\n'):
  offset=raw.rfind(b'\n')+1;tail=raw[offset:]
  try:json.loads(tail);raw+=b'\n'
  except json.JSONDecodeError:
   backup=path.with_name(path.name+'.partial-tail-'+hashlib.sha256(tail).hexdigest());backup.write_bytes(tail);raw=raw[:offset]
  path.write_bytes(raw)
 return {r['key']:r for r in map(json.loads,raw.decode().splitlines())}
