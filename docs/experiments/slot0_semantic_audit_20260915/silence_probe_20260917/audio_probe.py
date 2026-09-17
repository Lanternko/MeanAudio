import json,numpy as np,soundfile as sf,librosa,csv,warnings; warnings.filterwarnings('ignore')
N='/home/kojiek/eval_output_nvme/'
D={'A_slot0':'c2p0_slot0_quarter_mc_mf25_cfg3_neg','A_nm':'phase8_qwen_caption2p0_slot0nm_noq_quarter_mc_mf25_cfg3_neg',
   'B_slot0':'phase8_qwen_caption2p0_slot0_noq_quarter_s27182818_mc_mf25_cfg3_neg','B_nm':'phase8_qwen_caption2p0_slot0nm_noq_quarter_s27182818_mc_mf25_cfg3_neg'}
P={k:json.load(open(N+v+'/clap_batch32.json'))['per_clip'] for k,v in D.items()}
cap={r['id']:r['caption'] for r in csv.DictReader(open('/mnt/HDD/kojiek/phase4_jamendo_data/musiccaps_test.tsv'),delimiter='\t')}
bad=['VCrnnx9jTqs_50','tOb0M2k3deo_30','RI71ebbU0PQ_0','0PMFAO4TIU4_30','G13NEVAm6-o_30','lUp9WnLYetg_30','yTq3Kr3jkvs_50','HVA9-fjtv6U_60','j8z9a9A8LV4_230']
def feats(p):
  y,sr=sf.read(p); y=y.astype(np.float32); y=y if y.ndim==1 else y.mean(1)
  rms=librosa.feature.rms(y=y)[0]; dur=len(y)/sr
  H,Pc=librosa.effects.hpss(y); eh,ep=np.sum(H**2),np.sum(Pc**2)
  on=librosa.onset.onset_detect(y=y,sr=sr,units='time')
  chroma=librosa.feature.chroma_stft(y=y,sr=sr); cmax=np.mean(chroma.max(0)/ (chroma.sum(0)+1e-9))
  flat=np.mean(librosa.feature.spectral_flatness(y=y))
  cent=np.mean(librosa.feature.spectral_centroid(y=y,sr=sr))
  f0,vf,vp=librosa.pyin(y,fmin=80,fmax=1000,sr=sr,frame_length=2048)
  return dict(dur=dur,rms_db=20*np.log10(np.sqrt(np.mean(y**2))+1e-9),silent_frac=float(np.mean(rms<0.01)),
    crest=float(np.max(np.abs(y))/(np.sqrt(np.mean(y**2))+1e-9)),perc_ratio=float(ep/(eh+ep+1e-9)),
    onsets_per_s=len(on)/dur,chroma_peak=float(cmax),flatness=float(flat),centroid=float(cent),voiced_frac=float(np.nanmean(vf)))
out={}
for i in bad:
  print('\n##',i,'|',cap[i][:100])
  print('  %-8s %6s %7s %6s %6s %6s %6s %6s %6s %6s %7s'%('arm','CLAP','rmsdB','silent','crest','perc%','onset/s','chroma','flat','voiced','centroid'))
  for k,v in D.items():
    f=feats(f'{N}{v}/audio/{i}.flac'); out.setdefault(i,{})[k]={**f,'clap':P[k][i]}
    print('  %-8s %6.3f %7.1f %6.2f %6.1f %6.2f %6.2f %6.2f %6.3f %6.2f %7.0f'%(k,P[k][i],f['rms_db'],f['silent_frac'],f['crest'],f['perc_ratio'],f['onsets_per_s'],f['chroma_peak'],f['flatness'],f['voiced_frac'],f['centroid']))
json.dump(out,open('/tmp/claude-1005/-home-kojiek-MeanAudio/248fe91c-9bfa-4256-8e2d-e8eaf16135aa/scratchpad/audio_probe.json','w'),indent=1)
