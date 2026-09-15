#!/usr/bin/env python3
"""CPU/API-only, resumable slot0 audit controller; never launches a GPU job.

SQLite records submission intent before network I/O. Completed responses are
durable before validation. Ambiguous batches reconcile via stored completion
metadata, then remain isolated rather than blocking unrelated work or replaying.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
import csv
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
import random
from pathlib import Path
import signal
import sqlite3
import sys
import threading
import time
import urllib.parse
import urllib.request

import slot0_semantic_audit as audit

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/experiment_harness"))
from notification_receipts import deliver_required


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def classify_error(error):
    if isinstance(error, audit.ApiFailure):
        if error.status == 429:
            return "retry_wait"
        if error.status in (401, 403, 404):
            return "credential_hold"
        if error.status >= 500 or error.status == 408:
            return "ambiguous"
        return "quarantined"
    if isinstance(error, (TimeoutError, OSError)):
        return "ambiguous"
    return "quarantined"


def reconcile(key, request_hash, rows, receipt):
    query = urllib.parse.urlencode({"metadata[audit_request]": request_hash, "limit": 2})
    req = urllib.request.Request("https://api.openai.com/v1/chat/completions?" + query,
                                headers={"Authorization": "Bearer " + key})
    with urllib.request.urlopen(req, timeout=30) as response:
        matches = json.load(response).get("data", [])
    if len(matches) != 1:
        return None  # An empty listing does NOT establish zero charges.
    completion = matches[0]
    if completion.get("metadata", {}).get("audit_request") != request_hash:
        raise ValueError("reconciliation metadata mismatch")
    audit.atomic(receipt, completion)
    return audit.parse_response(rows, completion)


class Controller:
    def __init__(self, spec_path, key, transport=audit.call_model, notifier=None):
        self.spec_path = Path(spec_path)
        self.spec = json.loads(self.spec_path.read_text())
        self.key, self.transport, self.notifier = key, transport, notifier
        self.out = Path(self.spec["output_dir"])
        self.out.mkdir(parents=True, exist_ok=True)
        (self.out / "receipts").mkdir(exist_ok=True)
        self.lock = (self.out / "controller.lock").open("a")
        try:
            fcntl.flock(self.lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except Exception:
            self.lock.close()
            raise
        self.db = sqlite3.connect(self.out / "state.sqlite")
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("PRAGMA synchronous=FULL")
        self.db.executescript("""
            CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT);
            CREATE TABLE IF NOT EXISTS jobs (
                id TEXT PRIMARY KEY, phase TEXT, payload TEXT, request_hash TEXT,
                status TEXT, attempts INTEGER DEFAULT 0, due REAL DEFAULT 0,
                receipt_sha TEXT, result TEXT, error TEXT, reserved_cost REAL DEFAULT 0);
            CREATE TABLE IF NOT EXISTS events (id TEXT PRIMARY KEY, status TEXT, summary TEXT,
                delivered INTEGER DEFAULT 0);
        """)
        self.stop = threading.Event()
        self.rate_lock = threading.Lock()
        self.next_request = 0.0
        self.started = time.time()
        self.last_progress = time.time()
        self.last_report = 0.0
        self.source_rows = []

    def close(self):
        self.db.close()
        self.lock.close()

    def meta(self, key, default=None):
        row = self.db.execute("SELECT value FROM meta WHERE key=?", (key,)).fetchone()
        return json.loads(row[0]) if row else default

    def set_meta(self, key, value):
        self.db.execute("INSERT OR REPLACE INTO meta VALUES (?,?)", (key, json.dumps(value)))
        self.db.commit()

    def notify(self, event, status, summary):
        self.db.execute("INSERT OR IGNORE INTO events(id,status,summary) VALUES (?,?,?)",
                        (event, status, summary))
        self.db.commit()
        self.set_meta("event_time:" + event, self.meta("event_time:" + event,
                      datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")))
        self.publish_harn("held" if status == "held" else "running")
        row = self.db.execute("SELECT status,summary,delivered FROM events WHERE id=?", (event,)).fetchone()
        if row[2]:
            return
        if self.notifier:
            self.notifier(event, row[0], row[1])
        else:
            deliver_required(contract_path=self.spec_path, launcher_path=Path(__file__),
                             event=event, status=row[0], summary=row[1],
                             idempotency_key=f'{self.spec["run_id"]}:{event}',
                             notifier=ROOT / "scripts/notify_experiment_webhook.py",
                             python=Path(sys.executable), root=self.out / "notification_receipts")
        self.db.execute("UPDATE events SET delivered=1 WHERE id=?", (event,))
        self.db.commit()

    def publish_harn(self, state):
        if "harn_bundle" not in self.spec:  # injected unit fixture, never production
            return
        source = Path(self.spec["harn_bundle"])
        target = self.out / "harn"
        target.mkdir(exist_ok=True)
        for name in ("contract.json", "preflight.json"):
            # Preserve raw bytes; the other documents bind these exact hashes.
            raw = (source / name).read_bytes()
            tmp = target / (name + ".tmp")
            with tmp.open("wb") as f:
                f.write(raw)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp, target / name)
        ledger = json.loads((source / "ledger.json").read_text())
        events = list(ledger["events"])
        mapping = {"start": "experiment_started", "preflight-pass": "preflight_passed",
                   "disk-warning": "disk_warning", "disk-hard-stop": "disk_hard_stop"}
        for event_id, event_status, summary, delivered in self.db.execute("SELECT id,status,summary,delivered FROM events ORDER BY rowid"):
            body = {"sequence": len(events) + 1, "event_id": event_id,
                    "idempotency_key": self.spec["run_id"] + ":" + event_id,
                    "event_kind": mapping.get(event_id, "queue_hold"),
                    "occurred_at": self.meta("event_time:" + event_id),
                    "phase": "audit-only", "verdict": "pass" if event_id == "preflight-pass" else "none",
                    "relates_to_event_id": None,
                    "notification_status": "delivered" if delivered else "pending",
                    "previous_event_sha256": events[-1]["event_sha256"]}
            body["event_sha256"] = audit.digest(json.dumps(body, sort_keys=True).encode())
            events.append(body)
        ledger["events"] = events
        audit.atomic(target / "ledger.json", ledger)
        queue = json.loads((source / "queue.json").read_text())
        queue["updated_at"] = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
        entry = queue["entries"][0]
        entry["status"] = "active" if state == "running" else "held"
        entry["assigned_resource"] = {"resource_type": "cpu_pool", "resource_id": "slot0-api-four-workers"} if state == "running" else None
        entry["bindings"]["ledger_raw_sha256"] = sha(target / "ledger.json")
        audit.atomic(target / "queue.json", queue)

    def status(self, state):
        counts = dict(self.db.execute("SELECT status,COUNT(*) FROM jobs GROUP BY status"))
        result = {"run_id": self.spec["run_id"], "state": state, "jobs": counts,
                  "pid": os.getpid(), "updated": time.time(), "last_progress": self.last_progress,
                  "source_rows": len(self.source_rows), "contract_sha256": sha(self.spec_path),
                  "known_cost_usd": self.meta("cost", 0),
                  "uncertain_and_inflight_reserve_usd": self.db.execute(
                      "SELECT COALESCE(SUM(reserved_cost),0) FROM jobs").fetchone()[0],
                  "training_corpus_released": False, "gpu_used": False}
        audit.atomic(self.out / "state.json", result)
        self.publish_harn(state)
        return result

    def storage(self, free=None):
        v = os.statvfs(self.out)
        free = v.f_bavail * v.f_frsize if free is None else free
        if free < self.spec["storage"]["hard_floor_bytes"]:
            self.notify("disk-hard-stop", "held", f"Free bytes {free}; API submission stopped.")
            return False
        if free < self.spec["storage"]["warning_floor_bytes"]:
            self.notify("disk-warning", "held", f"Storage warning: free bytes {free}.")
        return True

    def prepare(self):
        if self.spec.get("model") != audit.MODEL or self.spec["prompt_sha256"] != audit.digest(audit.PROMPT.encode()):
            raise ValueError("model/prompt binding mismatch")
        for artifact in self.spec["immutable_artifacts"]:
            if sha(artifact["path"]) != artifact["sha256"]:
                raise ValueError("immutable artifact hash mismatch: " + artifact["path"])
        old = self.meta("spec_hash")
        if old and old != sha(self.spec_path):
            raise ValueError("contract changed across resume")
        self.set_meta("spec_hash", sha(self.spec_path))
        if self.spec["approval"]["scope"] != "full_slot0_semantic_audit_exact_prefix_and_quarantine":
            raise ValueError("approval scope mismatch")
        for path in self.spec["quality_gate_reports"]:
            report = json.loads(Path(path).read_text())
            if report.get("status") != "passed" or report["binding"]["prompt_sha256"] != self.spec["prompt_sha256"]:
                raise ValueError("calibration/heldout gate failed")
        with open(self.spec["source"]["path"], newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f, delimiter="\t")
            self.columns = reader.fieldnames
            self.source_rows = list(reader)
        ids = [r["id"] for r in self.source_rows]
        if len(ids) != self.spec["source"]["rows"] or len(set(ids)) != len(ids) or not all(ids):
            raise ValueError("source ID coverage invalid")
        for row in self.source_rows:
            if any(value is None for value in row.values()) or None in row:
                raise ValueError("source TSV schema/null field defect")
        sample_path = self.out / "real_data_review_sample.json"
        sample_count = min(self.spec.get("independent_review_sample_rows", 3000), len(self.source_rows))
        sample = random.Random(self.spec.get("independent_review_seed", 2026091604)).sample(self.source_rows, sample_count)
        sample_document = {"source_sha256": self.spec["source"]["sha256"],
                           "seed": self.spec.get("independent_review_seed", 2026091604),
                           "selection": "uniform all-source sample before model labels; no regex enrichment",
                           "labels": "pending independent semantic review; not ground truth yet", "rows": sample}
        if sample_path.exists() and json.loads(sample_path.read_text()) != sample_document:
            raise ValueError("independent review sample changed")
        audit.atomic(sample_path, sample_document)
        for i in range(0, len(ids), self.spec["batch_size"]):
            rows = [{"id": r["id"], "caption": r["caption"]}
                    for r in self.source_rows[i:i + self.spec["batch_size"]]]
            self.add_job(f"audit-{i:06d}", "audit", rows)
        # Crashed inflight jobs are recoverable from local receipts or stored metadata,
        # never blindly retried. Other batches remain pending and eligible.
        self.db.execute("UPDATE jobs SET status='ambiguous' WHERE status='inflight'")
        self.db.commit()
        for event, status, summary in self.db.execute("SELECT id,status,summary FROM events WHERE delivered=0").fetchall():
            self.notify(event, status, summary)

    def add_job(self, job_id, phase, rows):
        payload = json.dumps(rows, ensure_ascii=False, separators=(",", ":"))
        request_hash = audit.digest((self.spec["run_id"] + self.spec["prompt_sha256"] + job_id + payload).encode())
        self.db.execute("INSERT OR IGNORE INTO jobs(id,phase,payload,request_hash,status) VALUES (?,?,?,?,?)",
                        (job_id, phase, payload, request_hash, "pending"))
        self.db.commit()

    def perform(self, job_id, payload, request_hash, recover=False):
        rows = json.loads(payload)
        receipt = self.out / "receipts" / (job_id + ".json")
        if receipt.exists():
            raw = json.loads(receipt.read_text())
            bound = raw.get("_audit_request_binding") or (raw.get("metadata") or {}).get("audit_request")
            if bound != request_hash:
                raise ValueError("local receipt request binding mismatch")
            return audit.parse_response(rows, raw)
        if recover:
            return reconcile(self.key, request_hash, rows, receipt)
        with self.rate_lock:
            delay = max(0, self.next_request - time.monotonic())
            if self.stop.wait(delay):
                return None
            self.next_request = time.monotonic() + self.spec["min_request_interval_seconds"]
        return self.transport(self.key, rows, receipt, {"audit_request": request_hash,
                                                      "run_id": self.spec["run_id"]})

    def completed(self, job_id, result):
        if result is None:
            self.db.execute("UPDATE jobs SET status='ambiguous',error='no_confirmed_response' WHERE id=?", (job_id,))
            self.db.commit()
            return
        receipt = self.out / "receipts" / (job_id + ".json")
        usage = result.get("usage", {})
        # Conservatively bill all input at uncached rate; never claim billing exactness.
        cost = (usage.get("prompt_tokens", 0) * self.spec["pricing"]["input_per_million"] +
                usage.get("completion_tokens", 0) * self.spec["pricing"]["output_per_million"]) / 1e6
        self.db.execute("UPDATE jobs SET status='done',result=?,receipt_sha=?,reserved_cost=0 WHERE id=?",
                        (json.dumps(result), sha(receipt), job_id))
        self.db.execute("INSERT OR REPLACE INTO meta VALUES ('cost',?)", (json.dumps(self.meta("cost", 0) + cost),))
        self.db.commit()
        self.last_progress = time.time()
        if job_id.startswith("audit-"):
            for decision in result["decisions"]:
                if decision["decision"] == "TRIM_SUFFIX":
                    self.add_job("verify-" + decision["id"], "verify", [
                        {"id": decision["id"], "caption": decision["retained"]}])

    def failed(self, job_id, error):
        state = classify_error(error)
        attempt = self.db.execute("SELECT attempts FROM jobs WHERE id=?", (job_id,)).fetchone()[0]
        if state == "retry_wait" and attempt >= self.spec["max_429_attempts"]:
            state = "quarantined"
        due = time.time() + max(getattr(error, "retry_after", 0), min(60, 2 ** attempt))
        detail = {"type": type(error).__name__, "http_status": getattr(error, "status", None),
                  "request_id": getattr(error, "request_id", None)}
        # A known HTTP rejection is not a completed response. Only 429 releases the
        # uncertain charge reservation automatically; other outcomes stay budgeted.
        self.db.execute("UPDATE jobs SET status=?,due=?,error=?,reserved_cost=CASE WHEN ? THEN 0 ELSE reserved_cost END WHERE id=?",
                        (state, due, json.dumps(detail), getattr(error, "status", None) == 429, job_id))
        self.db.commit()
        if state == "credential_hold":
            self.stop.set()
        # Batch errors are persisted individually; one event per failure category.
        self.notify("batch-" + state, "held", f"A batch entered {state}; independent eligible batches continue. See state.sqlite.")

    def reconcile_ambiguous(self):
        for job_id, payload, request_hash in self.db.execute(
                "SELECT id,payload,request_hash FROM jobs WHERE status='ambiguous'").fetchall():
            if self.stop.is_set():
                break
            try:
                result = self.perform(job_id, payload, request_hash, recover=True)
                if result:
                    self.completed(job_id, result)
            except Exception:
                # Read-only recovery failure cannot change the original charge status.
                continue

    def run(self):
        self.prepare()
        if not self.storage():
            self.status("storage_hold")
            return 2
        self.notify("preflight-pass", "start", "Pinned source/model/prompt and calibration gates passed; API-only audit, no GPU.")
        self.notify("start", "start", "Slot0 semantic audit started/resumed; four bounded workers, no GPU ownership.")
        self.set_meta("first_started", self.meta("first_started", time.time()))
        self.reconcile_ambiguous()
        inflight = {}
        with ThreadPoolExecutor(max_workers=self.spec["workers"]) as pool:
            while True:
                if time.time() - self.meta("first_started") > self.spec["max_wall_seconds"]:
                    self.stop.set()
                if time.time() - self.last_report >= 20:
                    self.status("running")
                    self.last_report = time.time()
                    if not self.storage():
                        self.stop.set()
                while not self.stop.is_set() and len(inflight) < self.spec["workers"]:
                    job = self.db.execute("SELECT id,payload,request_hash,attempts FROM jobs WHERE status IN ('pending','retry_wait') AND due<=? ORDER BY phase,id LIMIT 1",
                                          (time.time(),)).fetchone()
                    if job is None:
                        break
                    job_id, payload, request_hash, attempts = job
                    reserve = ((len(payload.encode()) + len(audit.PROMPT.encode()) + 2000) * self.spec["pricing"]["input_per_million"] +
                               4096 * self.spec["pricing"]["output_per_million"]) / 1e6
                    pending_cost = self.db.execute("SELECT COALESCE(SUM(reserved_cost),0) FROM jobs").fetchone()[0]
                    if self.meta("cost", 0) + pending_cost + reserve > self.spec["max_cost_usd"]:
                        self.notify("cost-hold", "held", "Registered API cost ceiling reached; submission stopped, receipts retained.")
                        self.stop.set()
                        break
                    if self.meta("post_count", 0) >= self.spec["max_api_posts"]:
                        self.stop.set()
                        break
                    self.db.execute("UPDATE jobs SET status='inflight',attempts=?,reserved_cost=? WHERE id=?",
                                    (attempts + 1, reserve, job_id))
                    self.set_meta("post_count", self.meta("post_count", 0) + 1)
                    self.db.commit()
                    future = pool.submit(self.perform, job_id, payload, request_hash)
                    inflight[future] = job_id
                if not inflight:
                    if self.stop.is_set():
                        break
                    future_jobs = self.db.execute("SELECT MIN(due) FROM jobs WHERE status='retry_wait'").fetchone()[0]
                    if future_jobs is None:
                        break
                    self.stop.wait(min(1, max(0.05, future_jobs - time.time())))
                    continue
                finished, _ = wait(inflight, timeout=1, return_when=FIRST_COMPLETED)
                for future in finished:
                    job_id = inflight.pop(future)
                    try:
                        self.completed(job_id, future.result())
                    except Exception as error:
                        self.failed(job_id, error)
        self.reconcile_ambiguous()
        report = self.export()
        state = "audit_complete_release_held" if report["audit_coverage_complete"] else "audit_incomplete"
        self.status(state)
        self.notify("audit-terminal", "held", f"Audit coverage {report['audited_rows']}/{report['source_rows']}; {report['unresolved_rows']} unresolved rows. No training corpus released.")
        return 0 if report["audit_coverage_complete"] else 2

    def export(self):
        decisions = {}
        verified = {}
        for job_id, phase, raw, receipt_sha in self.db.execute("SELECT id,phase,result,receipt_sha FROM jobs WHERE status='done'"):
            if sha(self.out / "receipts" / (job_id + ".json")) != receipt_sha:
                raise ValueError("receipt hash mismatch")
            for result in json.loads(raw)["decisions"]:
                target = decisions if phase == "audit" else verified
                if result["id"] in target:
                    raise ValueError("duplicate audit result")
                target[result["id"]] = result
        audit_tmp = self.out / "row_decisions.jsonl.tmp"
        quarantine_tmp = self.out / "quarantine.jsonl.tmp"
        regen_tmp = self.out / "regeneration_requests.tsv.tmp"
        unresolved = 0
        with audit_tmp.open("w") as a, quarantine_tmp.open("w") as q, regen_tmp.open("w") as regen:
            for source in self.source_rows:
                decision = decisions.get(source["id"])
                accepted = False
                if decision:
                    if decision["caption_sha256"] != audit.digest(source["caption"].encode()):
                        raise ValueError("source caption/result mismatch")
                    retained = audit.apply_decision(source["caption"], decision)
                    accepted = decision["decision"] == "KEEP"
                    if decision["decision"] == "TRIM_SUFFIX":
                        check = verified.get(source["id"])
                        accepted = bool(check and check["decision"] == "KEEP" and
                                        check["caption_sha256"] == audit.digest(retained.encode()))
                    if retained is not None and audit.structural_defects(retained):
                        accepted = False
                row = {"id": source["id"], "source_sha256": audit.digest(source["caption"].encode()),
                       "decision": decision, "resolved": accepted,
                       "review": verified.get(source["id"])}
                a.write(json.dumps(row, ensure_ascii=False) + "\n")
                if not accepted:
                    unresolved += 1
                    q.write(json.dumps({**row, "original": source}, ensure_ascii=False) + "\n")
                    if decision and decision["decision"] == "INVALID":
                        regen.write(source["id"] + "\tsemantic_invalid\n")
            for f in (a, q, regen):
                f.flush()
                os.fsync(f.fileno())
        for tmp in (audit_tmp, quarantine_tmp, regen_tmp):
            os.replace(tmp, tmp.with_suffix(""))
        report = {"run_id": self.spec["run_id"], "status": "held", "source_rows": len(self.source_rows),
                  "audited_rows": len(decisions), "unresolved_rows": unresolved,
                  "audit_coverage_complete": len(decisions) == len(self.source_rows),
                  "source_sha256": self.spec["source"]["sha256"],
                  "contract_sha256": sha(self.spec_path), "row_decisions_sha256": sha(self.out / "row_decisions.jsonl"),
                  "release_blockers": ["independent_real_data_miss_rate_review_pending"] +
                      (["unresolved_rows_require_review_or_same_prompt_regeneration"] if unresolved else []),
                  "training_corpus_released": False}
        audit.atomic(self.out / "full_gate_report.json", report)
        return report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--contract", type=Path, required=True)
    ap.add_argument("--key-file", type=Path, required=True)
    ap.add_argument("--export-only", action="store_true")
    args = ap.parse_args()
    controller = Controller(args.contract, audit.read_key_file(args.key_file))
    for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        signal.signal(sig, lambda *_: controller.stop.set())
    try:
        if args.export_only:
            controller.prepare()
            controller.export()
            return 0
        return controller.run()
    except Exception as error:
        controller.status("controller_hold")
        try:
            controller.notify("controller-hold", "held", "Controller exception: " + type(error).__name__ + "; inspect local state, no further submissions.")
        except Exception:
            pass
        # No raw urllib exception text (may contain a URL or credential).
        print(json.dumps({"status": "held", "error_type": type(error).__name__}), flush=True)
        return 2
    finally:
        controller.close()


if __name__ == "__main__":
    raise SystemExit(main())
