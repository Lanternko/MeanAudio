"""Outcome-independent paired listening kit; never synthesizes human ratings."""
import csv,json,hashlib,html
from pathlib import Path
import numpy as np
import soundfile as sf
from probe import normalize,loudness
from generate import ROOT,sha,dump

def main():
 c=json.loads((ROOT/'contract.json').read_text());old=Path(c['parent_root']);lookup={(r['family'],r['training_seed'],r['cfg_strength']):r['id'] for r in c['cells']};seeds=[14159265,16180339,27182818];metrics={}
 for cell in c['cells']:
  if cell['family'] not in ['control','N100']:continue
  metrics[cell['id']]={r['id']:r for r in csv.DictReader(open(old/'cells'/cell['id']/'metrics'/cell['id']/'per_clip.tsv'),delimiter='\t')}
 captions={r['id']:r['caption'] for r in csv.DictReader(open(c['tsv']),delimiter='\t')}
 candidates=sorted(c['probe_ids'],key=lambda i:hashlib.sha256(('human20261002'+i).encode()).hexdigest());eligible=[];excluded=[]
 for i in candidates:
  if all(m[i]['lufs'] and np.isfinite(float(m[i]['lufs'])) for m in metrics.values()):eligible.append(i)
  else:excluded.append(i)
 selected=eligible[:120];rng=np.random.default_rng(202610021);pairs=[(i,seeds[j%3],cfg) for j,i in enumerate(selected) for cfg in [3,0]];rng.shuffle(pairs)
 out=ROOT/'listening';out.mkdir(exist_ok=True);audio=out/'audio';audio.mkdir(exist_ok=True);public=[];truth=[]
 for j,(identifier,seed,cfg) in enumerate(pairs):
  arrays={};sources={};gains={}
  for family in ['control','N100']:
   source=old/'cells'/lookup[family,seed,cfg]/'audio'/(identifier+'.flac');y,sr=sf.read(source,dtype='float32');assert sr==16000
   z,g,lu=normalize(y);arrays[family]=z;sources[family]=dict(cell=lookup[family,seed,cfg],sha256=sha(source));gains[family]=g
  peak=max(np.max(np.abs(v)) for v in arrays.values());atten=min(1.,.98/max(float(peak),1e-12));target=-23+20*np.log10(atten)
  order=['control','N100'];rng.shuffle(order);record=dict(pair=f'pair_{j+1:03}',caption=captions[identifier]);answers={}
  for label,family in zip(['A','B'],order):
   path=audio/(record['pair']+'_'+label+'.wav');sf.write(path,(arrays[family]*atten).astype(np.float32),16000,subtype='PCM_24');z,_=sf.read(path,dtype='float32');assert np.max(np.abs(z))<=.981 and abs(loudness(z)-target)<.02
   record[label]='audio/'+path.name;answers[label]=dict(family=family,**sources[family],wav_sha256=sha(path),gain=gains[family]*atten,lufs=loudness(z))
  public.append(record);truth.append(dict(pair=record['pair'],prompt=identifier,training_seed=seed,cfg=cfg,common_target_lufs=target,answers=answers))
 dump(out/'pairs.json',public);dump(out/'private_answer_key.json',truth);dump(out/'DESIGN.json',dict(primary_pairs=240,unique_prompts=120,training_seeds=seeds,cfgs=[3,0],selection='outcome-independent SHA human20261002; common finite LUFS across all12 paired cells; prompt assigned seed round robin; same120 prompts in both CFG protocols; random pair order and A/B labels',excluded_nonfinite_lufs_ids=excluded,levels='same LUFS per pair, normally -23; both lowered together if needed to avoid clipping; PCM24 lossless WAV',analysis='paired wins and per-prompt cluster bootstrap; report CFG separately; both checkpoint labels concealed; no human responses received'))
 page='''<!doctype html><meta charset="utf-8"><title>音樂盲聽</title><style>body{font:18px sans-serif;max-width:800px;margin:40px auto;padding:20px}audio{width:100%}button,select{font-size:18px;margin:10px;padding:10px}label{display:block}#caption{color:#555}p{line-height:1.6}</style><h1>配對音樂盲聽</h1><p>先聽 A 與 B，分開判斷錄音清晰度和音樂本身。相同或無法確定也可選；尚未聽的配對請留空。</p><div id="position"></div><h2>A</h2><audio id="A" controls></audio><h2>B</h2><audio id="B" controls></audio><label>錄音清晰度 <select id="recording"><option value="">未評</option><option>A 較好</option><option>B 較好</option><option>相同</option><option>無法確定</option></select></label><label>整體音樂品質（旋律、和聲、節奏、連貫性） <select id="music"><option value="">未評</option><option>A 較好</option><option>B 較好</option><option>相同</option><option>無法確定</option></select></label><details><summary>完成音樂評分後，可查看描述並評估吻合程度</summary><p id="caption"></p><label>與描述的吻合程度 <select id="alignment"><option value="">未評</option><option>A 較好</option><option>B 較好</option><option>相同</option><option>無法確定</option></select></label></details><button id="prev">上一組</button><button id="next">下一組</button><button id="export">匯出評分 JSON</button><p>評分只保存在此瀏覽器；匯出後可供統計。請保持播放音量一致。</p><script>const pairs=PAIRS;const storage='aes-followup-listening-v1';let saved=JSON.parse(localStorage.getItem(storage)||'{}');let n=0;const fields=['recording','music','alignment'];function load(){const p=pairs[n];document.getElementById('position').textContent=`${n+1} / ${pairs.length} — ${p.pair}`;for(const a of ['A','B'])document.getElementById(a).src=p[a];document.getElementById('caption').textContent=p.caption;for(const f of fields)document.getElementById(f).value=(saved[p.pair]||{})[f]||'';document.querySelector('details').open=false}function save(){const r={};for(const f of fields)r[f]=document.getElementById(f).value;saved[pairs[n].pair]=r;localStorage.setItem(storage,JSON.stringify(saved))}for(const f of fields)document.getElementById(f).onchange=save;document.getElementById('prev').onclick=()=>{save();n=Math.max(0,n-1);load()};document.getElementById('next').onclick=()=>{save();n=Math.min(pairs.length-1,n+1);load()};document.getElementById('export').onclick=()=>{save();const u=URL.createObjectURL(new Blob([JSON.stringify({version:1,ratings:saved},null,2)],{type:'application/json'}));const a=document.createElement('a');a.href=u;a.download='blind-ratings.json';a.click();URL.revokeObjectURL(u)};load()</script>'''
 page=page.replace('PAIRS',json.dumps(public,ensure_ascii=False).replace('<','\\u003c'));(out/'index.html').write_text(page);print('Prepared',len(public),'blinded pairs; no ratings',flush=True)
if __name__=='__main__':main()
