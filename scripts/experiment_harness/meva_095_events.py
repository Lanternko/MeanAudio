"""Schema-v1 MEva event ledger; serialized parent/child updates and receipts."""
import fcntl
import hashlib
import json
import os
from pathlib import Path
from notification_receipts import atomic_secure_json, canonical_hash, deliver_required, utc_now

def append(contract, event_id, kind, verdict='none', relation=None, notification='not_applicable', state=None):
    root=Path(contract['harn_bundle'])
    with (root/'events.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        path=root/'ledger.json';ledger=json.loads(path.read_text());events=ledger['events']
        if any(e['event_id']==event_id for e in events):return
        event={'sequence':len(events)+1,'event_id':event_id,
          'idempotency_key':contract['experiment_id']+':'+event_id,'event_kind':kind,
          'occurred_at':utc_now(),'phase':'shadow-evaluation','verdict':verdict,
          'relates_to_event_id':relation,'notification_status':notification,
          'previous_event_sha256':events[-1]['event_sha256'] if events else None}
        event['event_sha256']=canonical_hash(event);events.append(event)
        atomic_secure_json(path,ledger)
        queue_path=root/'queue.json';queue=json.loads(queue_path.read_text())
        queue['updated_at']=utc_now();entry=queue['entries'][0]
        entry['bindings']['ledger_raw_sha256']=hashlib.sha256(path.read_bytes()).hexdigest()
        if state:
            entry['status']=state
            entry['assigned_resource']={'resource_type':'gpu','resource_id':'gpu0'} if state=='active' else None
            entry['terminal_notification_status']='delivered' if state in ['completed','failed','interrupted'] else 'not_applicable'
        atomic_secure_json(queue_path,queue)

def notify(contract, script, event, status, summary, kind='gate_result', verdict='pass', released=False):
    append(contract,event,kind,verdict,notification='pending')
    cfg=contract['notification_receipts']
    try:
        path=deliver_required(contract_path=Path(os.environ['GPU_QUEUE_CONTRACT']),
            launcher_path=script,event=event,status=status,summary=summary,
            idempotency_key=contract['experiment_id']+':'+event,
            notifier=Path(cfg['notifier']),root=Path(cfg['root']),
            extra_args=['--gpu-released'] if released else None)
    except Exception:
        append(contract,event+'-notification-failed','notification_delivery',relation=event,notification='failed')
        raise
    append(contract,event+'-notification-delivered','notification_delivery',relation=event,notification='delivered',
           state={'experiment_started':'active','experiment_completed':'completed','experiment_failed':'failed','experiment_interrupted':'interrupted'}.get(kind))
    return {'path':str(path),'event':event,'status':status}
