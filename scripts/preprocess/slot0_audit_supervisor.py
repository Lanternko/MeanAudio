#!/usr/bin/env python3
"""Independent, deterministic supervisor for the API-only audit child."""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import fcntl

import slot0_semantic_audit as A

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/experiment_harness"))
from notification_receipts import deliver_required


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--contract", type=Path, required=True)
    ap.add_argument("--key-file", type=Path, required=True)
    args = ap.parse_args()
    spec = json.loads(args.contract.read_text())
    out = Path(spec["output_dir"])
    out.mkdir(parents=True, exist_ok=True)
    lock = (out / "supervisor.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    child = None
    interrupted = False

    def notify(event, status, summary):
        deliver_required(contract_path=args.contract, launcher_path=Path(__file__),
                         event=event, status=status, summary=summary,
                         idempotency_key=spec["run_id"] + ":" + event,
                         notifier=ROOT / "scripts/notify_experiment_webhook.py",
                         python=Path(sys.executable), root=out / "supervisor_receipts")

    def on_signal(sig, frame):
        nonlocal interrupted
        interrupted = True
        if child and child.poll() is None:
            child.send_signal(sig)  # Popen-owned child only; never unrelated host processes.

    for sig in (signal.SIGHUP, signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, on_signal)
    binding = {"pid": os.getpid(), "boot_id": Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
               "start_ticks": Path(f"/proc/{os.getpid()}/stat").read_text().split()[21],
               "contract_sha256": A.digest(args.contract.read_bytes()), "state": "starting"}
    A.atomic(out / "supervisor.json", binding)
    # Offline bundle validation is required, and does not replace runtime checks.
    bundle = Path(spec["harn_bundle"])
    validation = subprocess.run([sys.executable, str(ROOT / "scripts/validate_experiment_harness_documents.py"),
                                 "--contract", str(bundle / "contract.json"), "--preflight", str(bundle / "preflight.json"),
                                 "--ledger", str(bundle / "ledger.json"), "--queue", str(bundle / "queue.json")],
                                capture_output=True, text=True)
    (out / "harn_validation.log").write_text(validation.stdout + validation.stderr)
    if validation.returncode:
        notify("harn-invalid", "held", "HARN document validation failed; API child was not started.")
        return 2
    with (out / "controller.log").open("a") as log:
        child = subprocess.Popen([sys.executable, str(ROOT / "scripts/preprocess/slot0_audit_controller.py"),
                                  "--contract", str(args.contract), "--key-file", str(args.key_file)],
                                 stdout=log, stderr=log)
        binding.update(child_pid=child.pid, state="running")
        A.atomic(out / "supervisor.json", binding)
        while child.poll() is None:
            state_path = out / "state.json"
            if state_path.exists():
                state = json.loads(state_path.read_text())
                if time.time() - state["last_progress"] > spec["watcher"]["stall_seconds"]:
                    notify("stall", "held", "No completed API batch within registered stall threshold; inspect state, no model diagnostic call.")
            free = os.statvfs(out).f_bavail * os.statvfs(out).f_frsize
            if free < spec["storage"]["hard_floor_bytes"]:
                notify("disk-hard-stop", "held", "Independent watcher detected hard storage floor; asking owned child to stop submissions.")
                child.terminate()
            time.sleep(10)
    binding.update(state="interrupted" if interrupted else "child_terminal", exit_code=child.returncode)
    A.atomic(out / "supervisor.json", binding)
    notify("supervisor-terminal", "interrupted" if interrupted else "held",
           f"API audit child exited {child.returncode}; full_gate_report.json records coverage and release blockers. GPU queue unchanged.")
    return child.returncode


if __name__ == "__main__":
    raise SystemExit(main())
