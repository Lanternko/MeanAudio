#!/usr/bin/env python3
"""CPU scientific-gate fixtures for 128; no model or live queue writes."""
import copy
import csv
import json
import os
import sys
import tempfile
import unittest
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch, MagicMock

import numpy as np
import soundfile as sf

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'eval'))
import metric_invariance_128 as M
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'experiment_harness'))
import metric_invariance_128_guest as G
import register_metric_invariance_128 as REG


class ScientificGates(unittest.TestCase):
    def resume_fixture(self, root):
        records=[{'id':str(k),'caption':'test','path':'unused','sha256':'raw'} for k in range(4)]
        inputs={'table_b':{'system':records}};source=root/'source';dest=root/'dest'
        d=source/'table/system/chunk00000/C1';d.mkdir(parents=True)
        np.savez(d/'scores.npz',ids=np.array([r['id'] for r in records]),values=np.ones((4,6)))
        M.write(d/'done.json',{'identity':M.hash_json({'action':'action','mode':'C1','system':'system','variant':'chunk00000','records':records}),
                             'n':4,'score_sha256':M.sha(d/'scores.npz')})
        return records,inputs,source,dest,d

    def test_storage_resume_keeps_exact_units_and_skips_incomplete(self):
        with tempfile.TemporaryDirectory() as td:
            records,inputs,source,dest,d=self.resume_fixture(Path(td))
            partial=source/'table/system/chunk01000/C1';partial.mkdir(parents=True)
            (partial/'scores.tmp.npz').write_bytes(b'incomplete')
            copied=REG.copy_completed_units(source,dest,inputs,'action')
            self.assertEqual(len(copied),1)
            self.assertEqual(copied[0]['scores_sha256'],M.sha(d/'scores.npz'))
            scorer=M.Scorer.__new__(M.Scorer);scorer.c={'storage':{'path':str(dest)},'bindings':{'action_sha256':'action'}}
            got,meta=scorer.unit('table','system','chunk00000','C1',records)
            self.assertEqual(got['ids'].tolist(),[r['id'] for r in records])
            self.assertEqual(meta['n'],4)  # cache loads without any model initialization
            with self.assertRaises(RuntimeError):REG.copy_completed_units(source,dest,inputs,'action')

    def test_storage_resume_rejects_changed_scores_or_source(self):
        with tempfile.TemporaryDirectory() as td:
            records,inputs,source,dest,d=self.resume_fixture(Path(td))
            with self.assertRaises(RuntimeError):REG.copy_completed_units(source,dest,inputs,'different-action')
            (d/'scores.npz').write_bytes(b'corrupted')
            with self.assertRaises(RuntimeError):REG.copy_completed_units(source,dest,inputs,'action')

    def test_storage_resume_rejects_rehashed_wrong_ids(self):
        with tempfile.TemporaryDirectory() as td:
            records,inputs,source,dest,d=self.resume_fixture(Path(td))
            np.savez(d/'scores.npz',ids=np.array(['wrong']*4),values=np.ones((4,6)))
            meta=json.loads((d/'done.json').read_text());meta['score_sha256']=M.sha(d/'scores.npz');M.write(d/'done.json',meta)
            with self.assertRaises(RuntimeError):REG.copy_completed_units(source,dest,inputs,'action')

    def test_actual_200_clip_cu_failure_is_warning_for_pq_clap_claims(self):
        root=Path(__file__).resolve().parents[2]
        p=root/'docs/experiments/harn/metric_invariance_128_budget_r3/outcome_20261018/b_identity_200.tsv'
        with p.open() as f:rows=list(csv.DictReader(f,delimiter='\t'))
        got={'ids':np.array([r['id'] for r in rows]),'values':np.array([[float(r['score_'+m]) for m in M.METRICS] for r in rows])}
        refs={r['id']:{m:float(r['reference_'+m]) for m in M.METRICS[:-1]} for r in rows}
        audit=M.readjudication_identity(got,refs,['PQ','clap'])
        self.assertTrue(audit['passed']);self.assertEqual(audit['warning_metrics'],['CU'])
        self.assertAlmostEqual(audit['metrics']['CU']['max_abs'],.012890815734863281)
        with self.assertRaises(RuntimeError):M.check_c0(got,refs,M.METRICS[:-1])

    def test_claim_identity_keeps_primary_bounds_and_mean_bias(self):
        ids=[str(k) for k in range(50)]
        refs={i:{m:0. for m in M.METRICS[:-1]} for i in ids}
        for metric,value in [('PQ',.011),('clap',.0002)]:
            values=np.zeros((50,6));values[0,M.METRICS.index(metric)]=value
            audit=M.readjudication_identity({'ids':ids,'values':values},refs,['PQ','clap'])
            self.assertFalse(audit['passed']);self.assertIn(metric,audit['failed_blocking_metrics'])
        values=np.zeros((50,6));values[:,0]=.002
        audit=M.readjudication_identity({'ids':ids,'values':values},refs,['PQ'])
        self.assertFalse(audit['passed'])
        values[:]=0;values[0,4]=.0002
        audit=M.readjudication_identity({'ids':ids,'values':values},refs,['PQ'])
        self.assertTrue(audit['passed']);self.assertEqual(audit['warning_metrics'],['clap'])
        values[0,0]=np.nan
        with self.assertRaises(RuntimeError):M.readjudication_identity({'ids':ids,'values':values},refs,['PQ'])

    def test_ancillary_identity_warning_does_not_stop_score_cells(self):
        records=[{'id':str(k)} for k in range(225)]
        refs={r['id']:{m:7. for m in M.METRICS} for r in records}
        def unit(group,system,variant,mode,rs):
            values=np.full((len(rs),6),7.)
            if group=='identity':
                values[0,2]+=.02
                values[200:,0]+=2.  # context-only rows must not enter the 200-clip gate
            elif system=='ours':values+=.2
            return {'ids':np.array([r['id'] for r in rs]),'values':values,'sd':np.zeros_like(values),'lufs_fallback':np.zeros(len(rs),bool)},{}
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);calibration=root/'calibration.json'
            M.write(calibration,{'chosen':{m:{'mode':'C3','sensitivity_only':True} for m in M.JUDGED}})
            c={'storage':{'path':td},'calibration_summary':str(calibration),'part':'c',
               'protocol':{'readjudication_identity_policy':{'blocking_metrics':['PQ'],'context_batch_size':32},'identity_gate':M.IDENTITY}}
            manifest={'negmf':{'ours':records,'base':records},'table_references':{'ours':'ours-ref','base':'base-ref'},
                      'margins':{'PQ':.1},'claims':{'c':[['K4','PQ',['ours']*3,['base']*3]]}}
            scorer=MagicMock();scorer.unit.side_effect=unit
            with patch.object(M,'table',return_value=refs),patch.object(M,'notify') as notify:
                M.score_cells(c,scorer,manifest,'c')
            out=json.loads((root/'summary.json').read_text())
            self.assertEqual(out['status'],'completed')
            self.assertEqual(out['identity_checks']['ours']['warning_metrics'],['CU'])
            self.assertEqual(out['identity_checks']['ours']['checked_n'],200)
            self.assertEqual(out['identity_checks']['ours']['scored_context_n'],224)
            self.assertFalse(out['claims']['K4']['eligible_as_conclusion'])
            self.assertEqual(notify.call_count,2)
            self.assertEqual(json.loads((root/'identity_checks.json').read_text()),out['identity_checks'])

    def test_carried_budget_includes_completed_a_and_new_b_only(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/'b-started.json';M.write(p,{'at':100000.})
            c={'budget_carry':{'completed_calibration_seconds':7*3600,'readjudication_started_path':str(p)}}
            self.assertEqual(M.prior_phase_elapsed(c,100000+3600),8*3600)
            self.assertGreater(M.prior_phase_elapsed(c,100000+11*3600),17*3600)
            with self.assertRaises(RuntimeError):M.prior_phase_elapsed(c,99999.)
            c['budget_carry']['completed_calibration_seconds']=-1
            with self.assertRaises(RuntimeError):M.prior_phase_elapsed(c,100000.)
            old=Path(td)/'a';old.mkdir();M.write(old/'started.json',{'at':100.})
            self.assertEqual(M.prior_phase_elapsed({'calibration_summary':str(old/'summary.json')},200.),100.)

    def test_reused_calibration_detects_normalizer_change(self):
        source=Path(M.__file__).read_text();fingerprints=M.calibration_fingerprints(source)
        self.assertEqual(fingerprints,M.calibration_fingerprints(source+'\n# unrelated comment\n'))
        changed=source.replace('y = x * 10 ** ((-30.0 - level) / 20)','y = x * 10 ** ((-29.0 - level) / 20)',1)
        self.assertNotEqual(fingerprints,M.calibration_fingerprints(changed))

    def test_calibration_budget_comes_from_registered_contract(self):
        estimates={'1000':71299.05280564557,'600':44631.68567510432,'400':31298.002109833702}
        self.assertIsNone(M.select_calibration_n(estimates,{'protocol':{'budget_seconds_a':7*3600}}))
        self.assertEqual(M.select_calibration_n(estimates,{'protocol':{'budget_seconds_a':9*3600}}),400)
        for budget in (0,-1,float('nan'),float('inf')):
            with self.assertRaises(ValueError):M.select_calibration_n(estimates,{'protocol':{'budget_seconds_a':budget}})
        with self.assertRaises(KeyError):M.select_calibration_n(estimates,{'protocol':{}})

    def test_members_preserve_length_sign_and_truncation(self):
        x=np.linspace(-.2,.3,16000)
        ys,fallback=M.members(x,16000,'C2')
        self.assertFalse(fallback)
        self.assertEqual([len(y) for y in ys],[16000]*4)
        np.testing.assert_array_equal(ys[0],x)
        np.testing.assert_array_equal(ys[1][:640],0)
        np.testing.assert_array_equal(ys[1][640:],-x[:-640])

    def test_normalization_does_not_limit_peak(self):
        import pyloudnorm
        sr=16000;x=np.zeros(sr*10);x[100]=1.
        y,fallback=M.normalize(x,sr)
        self.assertFalse(fallback)
        self.assertAlmostEqual(pyloudnorm.Meter(sr).integrated_loudness(y),-30,places=8)
        self.assertGreater(float(np.max(np.abs(y))),1.)
        # Positive homogeneity is the gain-invariance construction.
        np.testing.assert_allclose(M.normalize(x*10,sr)[0],y,atol=1e-12)

    def test_silent_normalization_fails_closed(self):
        with self.assertRaises(ValueError):M.normalize(np.zeros(16000),16000)
        for bad in (np.array([]),np.full(16000,np.nan)):
            with self.assertRaises(ValueError):M.normalize(bad,16000)

    def test_quiet_lufs_fallback_amendment(self):
        import pyloudnorm
        sr=16000;t=np.arange(sr*10)/sr
        tone=.1*np.sin(2*np.pi*440*t)
        ref,fb=M.normalize(tone,sr);self.assertFalse(fb)
        # Below the -70 LUFS absolute gate: standard LUFS is -inf, fallback measures at unit peak.
        quiet=tone*1e-4
        self.assertEqual(pyloudnorm.Meter(sr).integrated_loudness(quiet),-np.inf)
        y,fb=M.normalize(quiet,sr);self.assertTrue(fb)
        self.assertAlmostEqual(pyloudnorm.Meter(sr).integrated_loudness(y),-30,places=8)
        np.testing.assert_allclose(y,ref,atol=1e-9)  # same waveform shape -> same canonical audio
        views,fb=M.members(quiet,sr,'C3');self.assertTrue(fb);self.assertEqual(len(views),4)
        u={'ids':np.array(['a','b','c']),'lufs_fallback':np.array([False,True,False])}
        self.assertEqual(M.fallback_ids(u,{'ids':np.array(['c'])},{'ids':np.array(['c']),'lufs_fallback':np.array([True])}),{'b','c'})

    def test_cpu_scan_classifies_and_cleans_isolated_audio(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);scratch=root/'scratch';scratch.mkdir()
            tone=.1*np.sin(2*np.pi*440*np.arange(16000)/16000)
            for label,x,expected in [('normal',tone,'ok'),('quiet',tone*1e-4,'fallback'),('zero',tone*0,'invalid')]:
                p=root/f'{label}.wav';sf.write(p,x,16000,subtype='FLOAT')
                rec={'id':'same-id','path':str(p),'sha256':M.sha(p)}
                row=M.scan_one((rec,label,'table',str(scratch)))
                self.assertEqual(row[3],expected)
                self.assertEqual(list(scratch.iterdir()),[])
            rec['sha256']='0'*64
            self.assertEqual(M.scan_one((rec,'zero','table',str(scratch)))[3],'invalid')

    def test_scan_gate_rejects_incomplete_and_stale_evidence(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);mf=root/'inputs.json'
            M.write(mf,{'calibration':{'system':[{}]},'midi':{c:[{}] for c in M.MIDI_CASES},
                        'table_b':{'system':[{}]},'negmf':{'system':[{}]}})
            paths={p:str(root/f'{p}.json') for p in 'abc'}
            reports={p:{'part':p,'status':'passed','n_invalid':0,'n_checked':len(M.VARIANTS)+len(M.MIDI_CASES) if p=='a' else 1,
                        'action_sha256':M.sha(M.__file__),'manifest_sha256':M.sha(mf)} for p in 'abc'}
            for p in 'abc':M.write(paths[p],reports[p])
            M.require_lufs_scans(paths,mf)
            for key,value in [('n_checked',0),('action_sha256','stale'),('manifest_sha256','stale'),('n_invalid',1)]:
                bad=dict(reports['b']);bad[key]=value;M.write(paths['b'],bad)
                with self.assertRaises(RuntimeError):M.require_lufs_scans(paths,mf)

    def test_preparation_failure_drains_writers_before_cleanup(self):
        with tempfile.TemporaryDirectory() as td:
            scratch=Path(td)/'scratch';scratch.mkdir()
            scorer=M.Scorer.__new__(M.Scorer)
            scorer.c={'storage':{'path':td,'transient_root':str(scratch)},
                      'bindings':{'action_sha256':'fixture'},'filesystems':[]}
            scorer.pool=MagicMock()
            # A worker fails while another still has a pending output write.
            def failed_map(*args,**kwargs):
                raise ValueError('LUFS undefined fixture')
            def drain(**kwargs):
                next(scratch.glob('audio-*')).joinpath('late.flac').write_bytes(b'late writer')
            scorer.pool.map.side_effect=failed_map
            scorer.pool.shutdown.side_effect=drain
            with self.assertRaisesRegex(ValueError,'LUFS undefined fixture'):
                scorer.unit('smoke','musicgen','clean','C1',[])
            scorer.pool.shutdown.assert_called_once_with(wait=True,cancel_futures=True)
            self.assertEqual(list(scratch.iterdir()),[])

    def test_transform_import_and_byte_identity(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/'clip.flac';x=np.sin(np.arange(16000)*.2)*.1
            sf.write(p,x,16000,subtype='PCM_24')
            rec={'id':'clip','path':str(p),'sha256':M.sha(p)}
            result=M.prepare_one((rec,'real','clean','C0',td))
            y,sr=sf.read(result['paths'][0],dtype='float32')
            original,_=sf.read(p,dtype='float32')
            np.testing.assert_array_equal(y,original)
            rec['sha256']='0'*64
            with self.assertRaises(ValueError):M.prepare_one((rec,'real','noise_snr10','C3',td))

    def test_margin_and_identity(self):
        r=M.rates(np.array([-.1,0,.1,.10001,-.10001]),.1)
        self.assertEqual(r['tie'],.6);self.assertEqual(r['better'],.2);self.assertEqual(r['worse'],.2)
        got={'ids':['a'],'values':np.array([[1,2,3,4,5,6]])}
        M.check_c0(got,{'a':{'PQ':1.00001}},['PQ'])
        with self.assertRaises(RuntimeError):M.check_c0(got,{'a':{'PQ':1.02}},['PQ'])

    def test_identity_gate_amendment(self):
        ids=[str(k) for k in range(50)]
        got={'ids':ids,'values':np.array([[7.,6.,7.,5.,.3,3.]]*50)}
        ref=lambda d:{i:{'PQ':7.+d(k),'CE':6.,'CU':7.,'PC':5.,'clap':.3,'meva':3.} for k,i in enumerate(ids)}
        # 2026-10-17 smoke: 18 tail clips off by up to 5.3e-4 with mixed sign -> passes now.
        M.check_c0(got,ref(lambda k:(5.3e-4 if k%2 else -4e-4) if k>=32 else 0.),['PQ','CE','CU','PC','clap'])
        with self.assertRaises(RuntimeError):  # one clip beyond the per-clip bound
            M.check_c0(got,ref(lambda k:.011 if k==49 else 0.),['PQ'])
        with self.assertRaises(RuntimeError):  # uniform small offset: per-clip ok, mean bias caught
            M.check_c0(got,ref(lambda k:.002),['PQ'])
        with self.assertRaises(RuntimeError):  # CLAP keeps 1e-4
            M.check_c0(got,{i:{'clap':.3002} for i in ids},['clap'])

    def test_mode_selection_and_missing_gate(self):
        cells={}
        for mode in M.MODES:
            cells[mode]={s:{m:{v:{'n':10,'tie':1.,'worse':.9,'better':0.}
                  for v in M.VARIANTS} for m in M.JUDGED} for s in M.A.SYSTEMS}
        self.assertEqual(M.select_modes(cells)['chosen']['PQ']['mode'],'C0')
        for s in M.A.SYSTEMS:cells['C0'][s]['PQ']['polarity']['tie']=.7
        self.assertEqual(M.select_modes(cells)['chosen']['PQ']['mode'],'C1')
        for mode in M.MODES:
            cells[mode]['real']['PQ']['polarity']['n']=0
        self.assertTrue(M.select_modes(cells)['chosen']['PQ']['sensitivity_only'])
        self.assertEqual(M.select_modes(cells)['chosen']['PQ']['mode'],'C3')

    def test_calibration_reports_lufs_fallback_sensitivity(self):
        n=5;ids=[f'x{k}' for k in range(n)]
        recs=[{'id':i} for i in ids]
        def unit(group,system,variant,mode,records):
            base=np.full((len(records),len(M.METRICS)),7.)
            name=variant.replace('source_','') if group=='midi' else variant
            if name in M.TIES or name in ('clean','clean_shuf','source'):d=0.
            else:d=-1.
            vals=base+d
            fb=np.array([r['id']=='x0' and mode in ('C1','C3') and system in ('real','generaluser') for r in records])
            if fb.any():vals[fb]+=5.  # the fallback clip disagrees; excluding it must change counts only
            return ({'ids':np.array([r['id'] for r in records]),'values':vals,'sd':np.zeros_like(vals),
                     'variant_capped':np.zeros(len(records),bool),'lufs_fallback':fb},{})
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);(root/'smoke.json').write_text(json.dumps({'selected_n':n}))
            manifest={'calibration':{s:recs for s in M.A.SYSTEMS},
                      'midi':{case:recs for case in M.MIDI_CASES},
                      'midi_reference':{i:{'PQ':7.} for i in ids},
                      'margins':{m:.1 for m in M.METRICS}}
            scorer=MagicMock();scorer.unit.side_effect=unit
            with patch.object(M,'notify'):M.run_a({'storage':{'path':td}},scorer,manifest)
            out=json.loads((root/'summary.json').read_text())
        sens=out['lufs_fallback_sensitivity']
        self.assertEqual(sens['excluded_ids']['real'],['x0'])
        self.assertEqual(sens['excluded_counts']['musicgen'],0)
        self.assertEqual(sens['midi_excluded_ids'],['x0'])
        self.assertEqual(out['cells']['C1']['real']['PQ']['gain-1']['n'],n)
        self.assertEqual(sens['cells']['C1']['real']['PQ']['gain-1']['n'],n-1)
        self.assertEqual(sens['cells']['C0']['real']['PQ']['gain-1']['n'],n-1)  # same cohort in every mode
        self.assertIn('robust_to_lufs_fallback',sens)

    def test_seed_readjudication(self):
        rates=[{'better':.8,'worse':.1}]*3
        self.assertEqual(M.conclusion([.2,.3,.2],rates,.1),'still_holds')
        self.assertEqual(M.conclusion([.01,.02,.01],rates,.1),'tie')
        self.assertEqual(M.conclusion([.2,.3,-.1],rates,.1),'weakened')

    def test_claim_fallback_exclusion_changes_verdict_and_blocks_conclusion(self):
        records=[{'id':str(k)} for k in range(4)]
        def unit(group,system,variant,mode,rs):
            values=np.zeros((len(rs),len(M.METRICS)))
            if system=='ours':
                values[:]=np.array([-.4 if r['id']=='0' else .2 for r in rs])[:,None]
            flags=np.array([system=='ours' and mode in ('C1','C3') and r['id']=='0' for r in rs])
            return {'ids':np.array([r['id'] for r in rs]),'values':values,'sd':np.zeros_like(values),'lufs_fallback':flags},{}
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);calibration=root/'calibration.json'
            M.write(calibration,{'chosen':{m:{'mode':'C3' if m=='clap' else 'C1','sensitivity_only':False} for m in M.JUDGED}})
            scorer=MagicMock();scorer.unit.side_effect=unit
            manifest={'negmf':{'ours':records,'base':records},'table_references':{},'margins':{'PQ':.1},
                      'claims':{'c':[['K4','PQ',['ours']*3,['base']*3]]}}
            M.score_cells({'storage':{'path':td},'calibration_summary':str(calibration),'part':'c'},scorer,manifest,'c')
            out=json.loads((root/'summary.json').read_text())
        claim=out['claims']['K4']
        self.assertEqual(claim['verdict'],'tie')
        self.assertEqual(claim['verdict_excl'],'still_holds')
        self.assertEqual(claim['n_pairs_excl'],[3]*3)
        self.assertFalse(claim['robust_to_lufs_fallback'])
        self.assertFalse(claim['eligible_as_conclusion'])
        self.assertEqual(out['lufs_fallback_counts'],{'ours':1,'base':0})

    def test_resume_accepts_only_matching_completed_bytes(self):
        with tempfile.TemporaryDirectory() as td:
            c={'storage':{'path':td},'bindings':{'action_sha256':'action'}}
            records=[{'id':'a','path':'unused','sha256':'source','caption':'caption'}]
            d=Path(td)/'group/system/clean/C0';d.mkdir(parents=True)
            score=d/'scores.npz';np.savez(score,ids=np.array(['a']),values=np.ones((1,6)))
            identity=M.hash_json({'action':'action','mode':'C0','system':'system','variant':'clean','records':records})
            M.write(d/'done.json',{'identity':identity,'score_sha256':M.sha(score),'n':1})
            scorer=M.Scorer.__new__(M.Scorer);scorer.c=c
            got,_=scorer.unit('group','system','clean','C0',records)
            self.assertEqual(got['ids'].tolist(),['a'])
            score.write_bytes(b'corrupt')
            with self.assertRaises(RuntimeError):scorer.unit('group','system','clean','C0',records)

    def test_storage_all_filesystems(self):
        c={'filesystems':[{'path':'/a','hard_floor_bytes':100,'warning_floor_bytes':200},
                           {'path':'/b','hard_floor_bytes':100,'warning_floor_bytes':200}]}
        class S:
            f_frsize=1
            def __init__(self,n):self.f_bavail=n
        with patch.object(G.os,'statvfs',side_effect=[S(500),S(99)]):
            self.assertEqual([r['state'] for r in G.storage_state(c)],['ok','hard_stop'])
        with patch.object(G.os,'statvfs',side_effect=[S(500),S(150)]):
            self.assertEqual([r['state'] for r in G.storage_state(c)],['ok','warning'])

    def test_action_storage_wait_has_retryable_exit(self):
        with patch.object(M.shutil,'disk_usage',return_value=type('Disk',(),{'free':99})()):
            with self.assertRaises(SystemExit) as caught:
                M.storage({'filesystems':[{'path':'/fixture','hard_floor_bytes':100}]})
            self.assertEqual(caught.exception.code,75)


class SupervisorBranches(unittest.TestCase):
    def run_supervisor(self, *, child_rc=0, notify_failure=False, interrupted=False, disk='ok', postflight_rc=0):
        with tempfile.TemporaryDirectory() as td, ExitStack() as stack:
            root=Path(td);script=root/'128_fixture.sh';script.write_text('#!/bin/bash\n')
            report=root/'summary.json';report.write_text('{"status":"completed"}')
            c={'storage':{'path':td},'filesystems':[], 'commands':{'preflight':['preflight'],
               'run':['run'],'postflight':['postflight']},'reports':[{'path':str(report)}],
               'watcher':{'stall_seconds':100},'experiment_id':'fixture','resume':{'autoresume':str(root/'resume.json')}}
            cp=root/'contract.json';cp.write_text(json.dumps(c))
            env={'GPU_QUEUE_JOB_SCRIPT':str(script),'GPU_QUEUE_CONTRACT':str(cp),
                 'P2_RUN_ID':'fixture-run','P2_CONTROL_DIR':str(root/'control')}
            stack.enter_context(patch.dict(os.environ,env))
            stack.enter_context(patch.object(G,'INTERRUPTED',interrupted))
            stack.enter_context(patch.object(G.signal,'signal'))
            stack.enter_context(patch.object(G,'accept_guest',return_value=(True,'ok')))
            stack.enter_context(patch.object(G,'pid_start_time',return_value='start'))
            def read(path):
                return {'pid':os.getpid(),'start_time':'start','job_id':script.stem,'run_id':'fixture-run'} if Path(path).name=='p2.running.json' else None
            stack.enter_context(patch.object(G,'read_json',side_effect=read))
            # Never alter a live queue idle receipt from a fixture.
            stack.enter_context(patch.object(G.Path,'unlink'))
            ctl=MagicMock();ctl.contract=c
            if notify_failure:ctl.pre_child_notifications.side_effect=RuntimeError('fixture notifier failure')
            stack.enter_context(patch.object(G,'Controller',return_value=ctl))
            stack.enter_context(patch.object(G,'storage_state',return_value=[{'state':disk}]))
            stack.enter_context(patch.object(G.time,'sleep',side_effect=RuntimeError('fixture polling boundary')))
            run=stack.enter_context(patch.object(G.subprocess,'run',side_effect=[type('RC',(),{'returncode':0})(),type('RC',(),{'returncode':postflight_rc})()]))
            child=MagicMock();child.pid=999999;child.poll.return_value=child_rc
            popen=stack.enter_context(patch.object(G.subprocess,'Popen',return_value=child))
            rc=G.main()
            return rc,ctl,popen,run

    def test_success_has_hash_evidence(self):
        rc,ctl,popen,_=self.run_supervisor()
        self.assertEqual(rc,0);popen.assert_called_once()
        self.assertEqual(ctl.terminal.call_args.args[0],'completed')
        self.assertTrue(ctl.terminal.call_args.kwargs['evidence']['eval_only'])
        self.assertEqual(len(ctl.terminal.call_args.kwargs['evidence']['reports'][0]['sha256']),64)

    def test_notifier_failure_prevents_child(self):
        rc,ctl,popen,_=self.run_supervisor(notify_failure=True)
        self.assertEqual(rc,2);popen.assert_not_called()
        self.assertEqual(ctl.terminal.call_args.args[0],'held')

    def test_disk_hard_stop_prevents_child(self):
        rc,ctl,popen,_=self.run_supervisor(disk='hard_stop')
        self.assertEqual(rc,2);popen.assert_not_called()
        self.assertEqual(ctl.notify.call_args.args[0],'disk_hard_stop')

    def test_child_137_interrupted(self):
        rc,ctl,_,_=self.run_supervisor(child_rc=137)
        self.assertEqual(rc,137);self.assertEqual(ctl.terminal.call_args.args[0],'interrupted')

    def test_postflight_invalid_never_completes(self):
        rc,ctl,_,_=self.run_supervisor(postflight_rc=2)
        self.assertEqual(rc,2);self.assertEqual(ctl.terminal.call_args.args[1],'held')
        self.assertEqual(ctl.terminal.call_args.kwargs['reason'],'postflight_invalid')

    def test_ordinary_failure_uses_scheduler_receipt_event(self):
        rc,ctl,_,_=self.run_supervisor(child_rc=1)
        self.assertEqual(rc,1)
        self.assertEqual(ctl.terminal.call_args.args[:3],('failed','failure','failure'))

    def test_signal_before_start_never_launches(self):
        rc,ctl,popen,_=self.run_supervisor(interrupted=True)
        self.assertEqual(rc,130);popen.assert_not_called()


if __name__=='__main__':unittest.main()
