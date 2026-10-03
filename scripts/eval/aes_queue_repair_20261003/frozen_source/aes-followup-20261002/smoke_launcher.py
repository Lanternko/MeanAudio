import os,fcntl,subprocess,json,time
from pathlib import Path
ROOT=Path(__file__).resolve().parent
lock=open('/home/kojiek/gpu_queue/gpu0.lock','a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
env=os.environ.copy();c=json.loads((ROOT/'contract.json').read_text());env.update(PYTHONPATH=c['isolated_packages'],HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',CUDA_VISIBLE_DEVICES='0')
pids=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True);assert set(map(int,pids.split()))<=set(c['allowed_existing_gpu_pids'])
start=time.monotonic();cell='control_s27182818_mf25_cfg3_fidelity8'
result=subprocess.run(['/home/kojiek/venvs/dac/bin/python','-u',str(ROOT/'probe.py'),cell,'--limit','4','--stage','confirmatory'],env=env)
assert result.returncode==0
(ROOT/'smoke_pass.json').write_text(json.dumps(dict(cell=cell,includes_previous_failed_prompt='5J6CjNc8Njo_30',requested_prompts=4,cases=32,seconds=time.monotonic()-start,returncode=0),indent=2)+'\n')
