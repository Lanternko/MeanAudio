from pathlib import Path
import re,json,csv,hashlib,ast
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path('/home/kojiek/MeanAudio'); OUT=Path(__file__).parent
runs={}; metadata={}; rows=[]; bins=[]
for arm in ['quarter','full']:
 for stage,budget in [(1,100000 if arm=='quarter' else 400000),(2,50000 if arm=='quarter' else 200000)]:
  name=f'mf_dedup_noq_{arm}_stage{stage}_{budget}'; offset=0 if stage==1 else (100000 if arm=='quarter' else 400000)
  values={}; sources=[]
  for p in sorted((ROOT/'exps'/name).glob('train-*-rank0.log')):
   data=p.read_text();sources.append({'path':str(p.resolve()),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()})
   cfg=ast.literal_eval(next(l.split('All configuration: ',1)[1] for l in data.splitlines() if 'All configuration: ' in l))
   for line in data.splitlines():
    m=re.search(r'-train - it\s+(\d+):.*?loss:([\d.eE+-]+),.*?lr:([\d.eE+-]+)',line)
    if m: values[int(m[1])]=(float(m[2]),float(m[3]))
  a=np.array([(s-offset,v[0],v[1],s) for s,v in sorted(values.items())]);runs[(arm,stage)]=a
  metadata[name]={'sources':sources,'count':len(a),'local_first':a[0,0],'local_last':a[-1,0],'config':cfg}
  for s,l,lr,g in a:rows.append([arm,stage,int(g),int(s),l,lr])
  for end in range(25000,budget+1,25000):
   v=a[(a[:,0]>=end-25000)&(a[:,0]<end)]
   bins.append({'arm':arm,'stage':stage,'start':end-25000,'end':end,'mean_loss':float(v[:,1].mean()),'slope_per_25k':float(np.polyfit(v[:,0],v[:,1],1)[0]*25000)})
with (OUT/'scalars.csv').open('w') as f:
 w=csv.writer(f);w.writerow(['arm','stage','global_step','local_step','loss','lr']);w.writerows(rows)
(OUT/'provenance.json').write_text(json.dumps(metadata,indent=2))
(OUT/'windows.json').write_text(json.dumps(bins,indent=2))
plt.rcParams.update({'font.size':10})
fig,axs=plt.subplots(2,2,figsize=(13,8),gridspec_kw={'height_ratios':[2,1]})
for stage in [1,2]:
 ax=axs[0,stage-1];bx=axs[1,stage-1]
 for arm,color in [('full','#1565c0'),('quarter','#e87515')]:
  a=runs[arm,stage];ax.plot(a[:,0]/1000,a[:,1],color=color,alpha=.10,lw=.5)
  smooth=np.convolve(a[:,1],np.ones(100)/100,mode='valid')
  ax.plot(a[99:,0]/1000,smooth,color=color,label=f'{arm}: trailing 5k-step mean',lw=1.8)
  bb=[b for b in bins if b['arm']==arm and b['stage']==stage]
  bx.plot([b['end']/1000 for b in bb],[b['mean_loss'] for b in bb],'-o',color=color,label=arm,ms=4)
 q=100 if stage==1 else 50
 ax.axvline(q,color='gray',ls='--',lw=1);ax.set_title(f'Stage {stage}' + (' | starts from respective S1 checkpoint' if stage==2 else ' | both runs start from scratch'))
 ax.set_ylabel('Logged adaptive loss');ax.legend(fontsize=8);bx.set_ylabel('25k-step block mean');bx.set_xlabel('Stage-local training steps (thousands)');bx.legend()
 for p in [ax,bx]:p.grid(alpha=.2)
fig.suptitle('MF deduplicated corpus: quarter vs full training loss\nTraining loss only; lower is better; no validation-loss series',fontsize=14)
fig.tight_layout();fig.savefig(OUT/'loss_curves.png',dpi=170);fig.savefig(OUT/'loss_curves.pdf')
for b in bins:print(b)
for stage in [1,2]:
 for arm in ['quarter','full']:
  a=runs[arm,stage]; print('TAIL',arm,stage,a[a[:,0]>=a[-1,0]-9950,1].mean())
q=runs['quarter',1];f=runs['full',1][:len(q)];print('S1 shared prefix MAE',np.mean(abs(q[:,1]-f[:,1])))
