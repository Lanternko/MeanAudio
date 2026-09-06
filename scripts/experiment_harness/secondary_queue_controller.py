#!/usr/bin/env python3
"""Shared two-level resource-owning controller primitives for secondary evals."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from notification_receipts import atomic_secure_json, deliver_required, secure_read_json, utc_now


NOTIFIER = Path("/home/kojiek/MeanAudio/scripts/notify_experiment_webhook.py")
PYTHON = Path("/home/kojiek/venvs/dac/bin/python")


class Controller:
    def __init__(self, script: Path, contract_path: Path, state_root: Path):
        self.script = script.resolve()
        self.contract_path = contract_path.resolve()
        self.contract = json.loads(self.contract_path.read_text(encoding="utf-8"))
        self.state_root = state_root
        self.ledger_path = state_root / "controller_ledger.json"
        state_root.mkdir(mode=0o700, parents=True, exist_ok=True)
        os.chmod(state_root, 0o700)
        if not self.ledger_path.exists():
            atomic_secure_json(self.ledger_path, {
                "document_kind": "resource_owning_preflight_controller_ledger_v1",
                "experiment_id": self.contract["experiment_id"],
                "run_id": self.contract["run_id"],
                "events": [],
            })
        self.record("resource_owner_recorded", {
            "pid": os.getpid(),
            "p2_run_id": os.environ.get("P2_RUN_ID"),
            "p2_job_id": os.environ.get("P2_JOB_ID"),
            "controller_role": "resource_owning_preflight_controller",
        })

    def _load(self) -> dict[str, Any]:
        value = secure_read_json(self.ledger_path)
        if value.get("document_kind") != "resource_owning_preflight_controller_ledger_v1":
            raise RuntimeError("controller ledger kind mismatch")
        if value.get("experiment_id") != self.contract["experiment_id"]:
            raise RuntimeError("controller ledger experiment mismatch")
        return value

    def record(self, event: str, details: dict[str, Any] | None = None) -> None:
        ledger = self._load()
        ledger["events"].append({"at": utc_now(), "event": event, "details": details or {}})
        atomic_secure_json(self.ledger_path, ledger)

    def notify(self, event: str, status: str, summary: str, *, gpu_released: bool = False) -> dict[str, str]:
        key = f"{self.contract['experiment_id']}:{self.contract['run_id']}:{event}"
        extra = ["--gpu-released"] if gpu_released else []
        path = deliver_required(
            contract_path=self.contract_path,
            launcher_path=self.script,
            event=event,
            status=status,
            summary=summary,
            idempotency_key=key,
            notifier=NOTIFIER,
            python=PYTHON,
            root=Path(self.contract["notification_receipts"]["root"]),
            extra_args=extra,
        )
        self.record(f"notification_{event}_delivered", {"receipt": str(path)})
        return {"path": str(path), "event": event, "status": status}

    def terminal(self, status: str, event: str, notify_status: str, summary: str,
                 **extra: Any) -> None:
        receipt = self.notify(event, notify_status, summary, gpu_released=True)
        self.record("terminal_notification_delivered", {"status": status, **receipt})
        from lib_scheduler import atomic_json, now
        atomic_json(
            self.script.with_name(self.script.stem + ".terminal.json"),
            {"status": status, "written_at": now(), "notification_receipt": receipt, **extra},
        )
        self.record("terminal_json_committed", {"status": status})

    def pre_child_notifications(self, label: str) -> None:
        self.notify(
            "queue_handoff", "start",
            f"Starting {label}. Preflight passed. This is a start, not a hold.",
        )
        self.notify("start", "start", f"Launching the evaluation child for {label}. GPU is claimed.")
        self.record("evaluation_child_launch_authorized")
