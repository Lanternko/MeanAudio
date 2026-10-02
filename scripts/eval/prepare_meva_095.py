#!/usr/bin/env python3
"""Freeze retained PromptCC cell coverage and the PAM external validation inputs."""
import csv
import json
import subprocess
from pathlib import Path
from meva_runtime import ROOT, RUNTIME, sha
import sys
sys.path.insert(0,str(ROOT/'scripts/experiment_harness'))
from notification_receipts import atomic_secure_json

def main():
    manifest=RUNTIME/'manifest.json'
    if manifest.exists():
        raise SystemExit('Manifest already frozen; use a new registered run for changes')
    source=ROOT/'research/eval/output/aes_human_corr_pam/per_clip.tsv'
    rows=[]
    for r in csv.DictReader(source.open(),delimiter='\t'):
        system,prompt=r['key'].split('__',1)
        p=(ROOT/'research/eval/pam_human_eval/human_eval/music'/system/(prompt+'.wav')).resolve(strict=True)
        rows.append(dict(set='pam',key=r['key'],prompt_id=prompt,system=system,audio_path=str(p),
            audio_sha256=sha(p),human_ovl=float(r['pam_OVL']),aes_pq=float(r['raw_PQ']),aes_ce=float(r['raw_CE']),
            source_protocol='PAM native released 5s; OVL; no training on these labels'))
    coverage=[]
    # Keep source protocol names and byte provenance. Never regenerate or relabel historical audio.
    roots=[Path('/home/kojiek/eval_output_nvme'),Path('/home/kojiek/eval_output'),ROOT/'eval_output']
    seen=set()
    for source_root in roots:
        for d in sorted(source_root.iterdir()):
            if not d.is_dir() or '_mc_mf25_' not in d.name: continue
            if str(d.resolve()) in seen: continue
            seen.add(str(d.resolve()))
            reports=list(d.glob('*REPORT.json'))
            metric=d/d.name/'per_clip.tsv'
            audio=d/'audio'
            if len(reports)!=1 or not metric.exists():
                coverage.append({'cell':d.name,'status':'unavailable_source_report_or_per_clip','path':str(d)})
                continue
            report=json.loads(reports[0].read_text())
            if report.get('status') not in ['passed','completed']:
                coverage.append({'cell':d.name,'status':'unverified_historical_report','path':str(d)})
                continue
            entries=list(csv.DictReader(metric.open(),delimiter='\t'))
            paths=[audio/(r['id']+'.flac') for r in entries]
            missing=sum(not p.is_file() for p in paths)
            record={'cell':d.name,'path':str(d.resolve()),'source_report':str(reports[0].resolve()),
                    'source_report_sha256':sha(reports[0]),'per_clip_path':str(metric.resolve()),
                    'per_clip_sha256':sha(metric),'n_metrics':len(entries),'n_missing_audio':missing,
                    'cfg_strength':report.get('cfg_strength'),'negative_prompt':report.get('negative_prompt'),
                    'status':'included' if not missing else 'audio_missing_no_regeneration'}
            coverage.append(record)
            if missing: continue
            protocol=str(report.get('protocol','historical protocol; see source report'))
            for r,p in zip(entries,paths):
                rows.append(dict(set=d.name,key=r['id'],prompt_id=r['id'],system='PromptCC',
                                 audio_path=str(p.resolve()),audio_sha256=sha(p),aes_pq=float(r['PQ']),
                                 aes_ce=float(r['CE']),source_protocol=protocol))
            print('frozen',d.name,len(entries),flush=True)
    keys=[(r['set'],r['key']) for r in rows]
    if len(keys)!=len(set(keys)):raise ValueError('Duplicate cell/clip key')
    atomic_secure_json(manifest,rows)
    atomic_secure_json(RUNTIME/'coverage.json',{'status':'inventory_frozen','n_rows':len(rows),
                        'n_pam':sum(r['set']=='pam' for r in rows),'cells':coverage,
                        'limitations':'Only retained MF25 cells with passed reports and complete audio. No regeneration.'})
    files=[ROOT/'.external/MEva/training/hybrid/train_cnn.py',ROOT/'.external/MEva/lib/repro/audio_window.py',
           ROOT/'.external/musicdiscovery/sae_components/musicgen_hooked.py']
    files+=list((RUNTIME/'models').rglob('*.bin'))+list((RUNTIME/'models').rglob('*.pth'))
    files+=list((RUNTIME/'models/sae').iterdir())
    files+=list((RUNTIME/'hf-cache/hub').rglob('*.bin'))+list((RUNTIME/'hf-cache/hub').rglob('*.json'))
    files+=list((RUNTIME/'hf-cache/hub').rglob('*.model'))
    lock={'candidate':'pooled/small/f03_sae_only_cnn_clean_best.pth',
          'meva_revision':'93fdc17fd324b5a7ba8ad56292aa77538f7717c2',
          'checkpoint_revision':'c536169ac98b16449f3aa3be6355bf5620c1466b',
          'musicgen_revision':'4c8334b02c6ec4e8664a91979669a501ec497792',
          'musicdiscovery_revision':subprocess.check_output(['git','-C',str(ROOT/'.external/musicdiscovery'),'rev-parse','HEAD'],text=True).strip(),
          't5':json.loads((RUNTIME/'t5_revision.json').read_text()),
          'sae':'sae-4_k_32_11/facebook/musicgen-small', 'sae_dim':4096,'hook':'hook_layers.11',
          'score_scale':'raw pretrained output; no clipping, rescaling or label fitting',
          'precision':'MusicGen native fp16 autocast; SAE/CNN fp32; torch attention backend',
          'files':[{'path':str(p.resolve()),'sha256':sha(p)} for p in sorted(set(files))]}
    atomic_secure_json(ROOT/'docs/experiments/meva_095_model_lock.json',lock)
    freeze=subprocess.check_output([sys.executable,'-m','pip','freeze'],text=True)
    (RUNTIME/'requirements.resolved.txt').write_text(freeze)
    print('TOTAL',len(rows),'CELLS',sum(c['status']=='included' for c in coverage),flush=True)

if __name__=='__main__':main()
