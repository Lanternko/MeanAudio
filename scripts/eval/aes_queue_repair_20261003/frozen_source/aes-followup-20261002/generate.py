import json,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parent
def sha(path):
 h=hashlib.sha256()
 with open(path,"rb") as f:
  for b in iter(lambda:f.read(8<<20),b""):h.update(b)
 return h.hexdigest()
def dump(path,value):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);temp=path.with_suffix(path.suffix+".tmp");temp.write_text(json.dumps(value,indent=2,allow_nan=False)+"\n");temp.replace(path)
