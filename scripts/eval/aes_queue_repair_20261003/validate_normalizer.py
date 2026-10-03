import sys,json
from pathlib import Path
import numpy as np,soundfile as sf
R=Path('/home/kojiek/MeanAudio');src=Path('/home/kojiek/Documents/Codex/2026-10-01/files-pasted-by-the-user-negmf/outputs/aes-followup-20261002');sys.path[:0]=[str(src),str(R/'scripts/eval/aes_queue_repair_20261003')];import probe
from normalize_candidate import normalize_candidate
c=json.loads((src/'contract.json').read_text());ids=['QUB_vpjogmo_170']+c['discovery_ids'][:31];worst=0;results=[]
for i in ids:
 y,sr=sf.read(src/'cells/N100_s27182818_mf25_cfg0_historical_null/audio'/(i+'.flac'),dtype='float32')
 if not np.isfinite(probe.loudness(y)):continue
 a,_,_=normalize_candidate(y,probe.normalize)
 for db in [-40,-6,6,40]:
  b,_,lu=normalize_candidate((y*np.float32(10**(db/20))).astype(np.float32),probe.normalize)
  delta=float(20*np.log10(np.linalg.norm(a.astype(float))/np.linalg.norm(b.astype(float))));relative=float(np.linalg.norm(a.astype(float)-b)/np.linalg.norm(a));assert abs(delta)<1e-5 and relative<1e-5 and abs(lu+23)<.005
  worst=max(worst,relative);results.append(dict(id=i,gain_db=db,rms_delta_db=delta,relative_error=relative))
y,_=sf.read(src/'cells/N100_s27182818_mf25_cfg0_historical_null/audio/QUB_vpjogmo_170.flac',dtype='float32');a=probe.normalize(y)[0];b=probe.normalize(y*np.float32(10**(-6/20)))[0];old=float(20*np.log10(np.linalg.norm(a)/np.linalg.norm(b)));assert old>.5
report={'status':'candidate_only_not_authorized_for_live_scores','old_outlier_rms_delta_db':old,'n_checks':len(results),'max_relative_waveform_error':worst,'checks':results};(R/'runtime/aes_queue_repair_20261003/normalizer_validation.json').write_text(json.dumps(report,indent=2)+'\n');print({k:v for k,v in report.items() if k!='checks'})
