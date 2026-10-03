"""Read-only CPU regression for all16 original failing source/stem inputs."""
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import soundfile as sf
from probe import ROOT,interventions,stem_items,normalize,features
from generate import dump,sha

def main():
 c=json.loads((ROOT/'contract.json').read_text());old=Path(c['parent_root']);checks=[]
 for cell in c['cells']:
  lines=(old/'cells'/cell['id']/'probe.jsonl').read_text().splitlines();last=json.loads(lines[-1]);identifier=last['id'];out=ROOT/'cells'/cell['id'];y,sr=sf.read(out/'audio'/(identifier+'.flac'),dtype='float32');assert sr==16000
  # All16 failures occurred after stem separation and writing its immutable cache.
  assert (old/'cells'/cell['id']/'stems'/(identifier+'.npz')).exists()
  variants=[(a,z,b,None) for a,z,b in interventions(y,20261002)]+list(stem_items(y,identifier,out,SimpleNamespace(sources=['drums','bass','other','vocals','guitar','piano'])))
  finite=0;excluded=[];max_error=0.
  for case,z,baseline,residual in variants:
   try:
    zz,gain,lu=normalize(z.astype(np.float32));assert zz.dtype==np.float32 and np.isfinite(zz).all();max_error=max(max_error,abs(lu+23));assert abs(lu+23)<.005;finite+=1
   except ValueError as exc:excluded.append(dict(case=case,reason=str(exc)))
  checks.append(dict(cell=cell['id'],original_failing_prompt=identifier,variants_checked=len(variants),normalized_float32_finite=finite,excluded=excluded,max_lufs_error=max_error,source_sha256=sha(out/'audio'/(identifier+'.flac'))))
 dump(ROOT/'prior_failures_validation.json',dict(passed=True,n_cells=len(checks),checks=checks));print('Validated all16 prior failing inputs:',sum(x['variants_checked'] for x in checks),'variants')
if __name__=='__main__':main()
