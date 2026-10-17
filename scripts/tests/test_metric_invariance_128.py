#!/usr/bin/env python3
"""CPU scientific-gate fixtures for 128; no model or live queue writes."""
import copy
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


class ScientificGates(unittest.TestCase):
    def test_members_preserve_length_sign_and_truncation(self):
        x=np.linspace(-.2,.3,16000)
        ys=M.members(x,16000,'C2')
        self.assertEqual([len(y) for y in ys],[16000]*4)
        np.testing.assert_array_equal(ys[0],x)
        np.testing.assert_array_equal(ys[1][:640],0)
        np.testing.assert_array_equal(ys[1][640:],-x[:-640])

    def test_normalization_does_not_limit_peak(self):
        import pyloudnorm
        sr=16000;x=np.zeros(sr*10);x[100]=1.
        y=M.normalize(x,sr)
        self.assertAlmostEqual(pyloudnorm.Meter(sr).integrated_loudness(y),-30,places=8)
        self.assertGreater(float(np.max(np.abs(y))),1.)
        # Positive homogeneity is the gain-invariance construction.
        np.testing.assert_allclose(M.normalize(x*10,sr),y,atol=1e-12)

    def test_silent_normalization_fails_closed(self):
        with self.assertRaises(ValueError):M.normalize(np.zeros(16000),16000)

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

    def test_seed_readjudication(self):
        rates=[{'better':.8,'worse':.1}]*3
        self.assertEqual(M.conclusion([.2,.3,.2],rates,.1),'still_holds')
        self.assertEqual(M.conclusion([.01,.02,.01],rates,.1),'tie')
        self.assertEqual(M.conclusion([.2,.3,-.1],rates,.1),'weakened')

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
