"""Deliver formal registration receipts, then atomically append approved candidates."""
import os,sys,json,fcntl,subprocess,shutil
from pathlib import Path
from common import ROOT,REPO,PY,sha,dump
sys.path[:0]=['/home/kojiek/gpu_queue',str(REPO/'scripts/experiment_harness')]
from notification_receipts import deliver_required,validate_delivered_receipt
from lib_scheduler import accept_guest

def main():
 jobs=[]
 for number,name in [(95,'aes-negmf-formal-recovery-20261003'),(96,'aes-negmf-other-source-20261003')]:
  path=REPO/'docs/experiments'/f'{name}_contract.json';c=json.loads(path.read_text());launcher=Path(c['candidate_launcher']);env=os.environ.copy();env['GPU_QUEUE_CONTRACT']=str(path)
  subprocess.run(c['commands']['preflight']+['--allow-unregistered'],env=env,check=True)
  os.environ['GPU_QUEUE_CONTRACT']=str(path);ok,reason=accept_guest(launcher);assert ok,reason
  jobs.append((number,path,c,launcher))
 lock=open('/home/kojiek/gpu_queue/gpu0.lock','a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 for number,path,c,launcher in jobs:
  summary=('正式流程補報／註冊：前輪私有controller沒有Discord通知並已失聯；本次重驗88336原始音檔雜湊及評分，確認11個介入階段有效。095採用有效cache，只補跑未完成部分，再完成5009-prompt確認分析。CFG3 fidelity8為主要，CFG0維持歷史標籤。這是現在補報，不表示當時已通知。HARN/schema與33項queue測試已通過，通知接受後才可啟動。'
   if number==95 else '正式上架096 other-source實驗，排在095之後：6個CFG3 fidelity8模型、5009 holdout prompts、20條件、raw與等LUFS；拆解other聲部音量、頻段、phase與對齊。依據512-prompt探索結果，移除other讓N100 PQ優勢縮小約0.245；other包含音樂內容，不能稱為底噪。分析三seed配對、bootstrap與Holm；不是人耳品質或訓練中介的證明。')
  receipt=deliver_required(contract_path=path,launcher_path=launcher,event='queue_registration',status='start',summary=summary,idempotency_key=c['experiment_id']+':'+c['run_id']+':queue_registration',notifier=Path(c['notification_receipts']['notifier']),python=Path(PY),root=Path(c['notification_receipts']['root']))
  ok,why=validate_delivered_receipt(receipt,contract_path=path,launcher_path=launcher,event='queue_registration',status='start');assert ok,why
  dump(Path(c['harn_bundle'])/'queue_registration.json',dict(receipt=str(receipt),receipt_sha256=sha(receipt),contract_sha256=sha(path),launcher_sha256=sha(launcher)))
  env=os.environ.copy();env['GPU_QUEUE_CONTRACT']=str(path);subprocess.run(c['commands']['preflight'],env=env,check=True)
 for number,path,c,launcher in jobs:
  dest=Path('/home/kojiek/gpu_queue/p2/pending')/launcher.name
  assert not any((Path('/home/kojiek/gpu_queue/p2')/state/launcher.name).exists() for state in ['pending','running','held','done','failed','interrupted']),launcher.name
  temp=dest.with_suffix('.staging');shutil.copyfile(launcher,temp);temp.chmod(0o700);os.replace(temp,dest)
  print('REGISTERED AND APPENDED',dest,'SHA',sha(dest),flush=True)
 fcntl.flock(lock,fcntl.LOCK_UN)
if __name__=='__main__':main()
