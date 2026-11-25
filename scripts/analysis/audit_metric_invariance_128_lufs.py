#!/usr/bin/env python3
"""CPU diagnostic of the identity_r1 LUFS failure; does not alter scoring rules."""
import json,sys,math,datetime,argparse
from pathlib import Path
import numpy as np,soundfile as sf,pyloudnorm
R=Path('/home/kojiek/MeanAudio');sys.path.insert(0,str(R/'scripts/eval'));import metric_invariance_128 as M
O=Path('/mnt/seagate/kojiek_data/metric_invariance_128_20261017_identity_r1');mf=json.loads(Path('/mnt/seagate/kojiek_data/metric_invariance_128_20261017/inputs.json').read_text());B=R/'docs/experiments/harn/metric_invariance_128_identity_r1/incident'
with np.load(O/'a/smoke/musicgen/clean/C0/scores.npz') as z:got={k:z[k] for k in z.files}
identity={'126a':M.check_c0(got,M.table(M.A.OUT/'units/clean/musicgen/scores.tsv'),M.METRICS),'115':M.check_c0(got,M.table(mf['reference_115']['musicgen']),M.METRICS[:-1])}
issues=[];n=0
for system,records in mf['calibration'].items():
 for r in records[:50]:
  x,sr=sf.read(r['path']);meter=pyloudnorm.Meter(sr);v=float(meter.integrated_loudness(x));n+=1
  if not math.isfinite(v):
   peak=float(np.max(np.abs(x)));row={'system':system,'id':r['id'],'path':r['path'],'sha256':M.sha(r['path']),'peak':peak,'rms':float(np.sqrt(np.mean(x*x))),'standard_lufs':None}
   if peak:
    probe=x/peak;level=float(meter.integrated_loudness(probe));y=probe*10**((-30-level)/20)
    row.update({'proposed_peak_probe_lufs':level,'proposed_output_lufs':float(meter.integrated_loudness(y)),'proposed_output_peak':float(np.max(np.abs(y))),'proposed_is_finite':bool(np.isfinite(y).all())})
   issues.append(row)
report={'at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'fresh_gpu_c0_status':'passed','fresh_gpu_c0_identity':identity,'cpu_clean_smoke_clips_checked':n,'undefined_lufs':issues,'scope':'Diagnostic CPU only. Peak probe normalization is an unapproved protocol proposal, not applied to production or scored on GPU.'}
ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path);args=ap.parse_args()
if args.output:args.output.write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
