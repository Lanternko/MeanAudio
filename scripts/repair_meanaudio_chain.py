#!/usr/bin/env python3
"""Read-only health check + *safe* auto-repairs for the MeanAudio chain.

Safe repairs only (never force-kills a healthy long train):
  - restart dead chain watcher
  - restart dead after_legacy_noq scheduler if still WAITING and legacy not DONE
  - restart dead final_audit loop if guard DONE but audit not PASSED/FAILED
  - clear stale soft state messages

Hard failures (NaN loss, OOM, gate fail, audit fail) are reported only —
do not auto-resume full training without human review when CLAUDE policy
requires investigation (e.g. clap_needs_review).
"""

from __future__ import annotations

import json
import os
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

LOG_ROOT = Path("/home/kojiek/logs")
STATE_DIR = LOG_ROOT / "meanaudio_chain_watch"
STATUS = STATE_DIR / "status.json"
REPAIR_LOG = STATE_DIR / "repair.log"
LEGACY_GUARD = LOG_ROOT / "phase8_legacy_repro_guard"
LEGACY_STATE = LEGACY_GUARD / "state.json"
AUDIT_STATUS = LEGACY_GUARD / "final_audit_loop_status.json"
NEXT_EXP = LEGACY_GUARD / "next_experiment_catalog_matched_noq.json"
WATCH_SCRIPT = Path("/home/kojiek/MeanAudio/scripts/watch_meanaudio_chain.py")
SCHEDULE_SCRIPT = Path(
    "/home/kojiek/MeanAudio/scripts/training_pipelines/schedule_catalog_matched_noq_after_legacy.sh"
)
AUDIT_SCRIPT = Path(
    "/home/kojiek/MeanAudio/scripts/training_pipelines/wait_and_audit_phase8_legacy_repro.sh"
)


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def log(msg: str) -> None:
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    line = f"[{now()}] {msg}"
    print(line, flush=True)
    with REPAIR_LOG.open("a") as handle:
        handle.write(line + "\n")


def read_json(path: Path) -> dict:
    try:
        data = json.loads(path.read_text())
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def tmux_has(name: str) -> bool:
    r = subprocess.run(
        ["tmux", "has-session", "-t", name],
        capture_output=True,
        check=False,
    )
    return r.returncode == 0


def tmux_new(name: str, command: str) -> None:
    if tmux_has(name):
        return
    subprocess.run(
        ["tmux", "new-session", "-d", "-s", name, command],
        check=False,
    )
    log(f"started tmux {name}")


def pgrep(pattern: str) -> bool:
    r = subprocess.run(
        ["pgrep", "-f", pattern],
        capture_output=True,
        check=False,
    )
    return r.returncode == 0


def repair_once() -> int:
    """Return number of repairs performed."""
    actions = 0
    status = read_json(STATUS)
    legacy = read_json(LEGACY_STATE)
    audit = read_json(AUDIT_STATUS)
    nxt = read_json(NEXT_EXP)

    # 1) chain watcher must always be alive
    if not pgrep("watch_meanaudio_chain.py"):
        log("REPAIR: chain watcher dead — restarting")
        tmux_new(
            "meanaudio_chain_watch",
            "source /home/kojiek/venvs/dac/bin/activate && "
            "python -u /home/kojiek/MeanAudio/scripts/watch_meanaudio_chain.py "
            "--interval 60 --heartbeat-every 900 2>&1 | "
            "tee -a /home/kojiek/logs/meanaudio_chain_watch/stdout.log",
        )
        actions += 1

    # 2) schedule must run until full NoQ is RUNNING or terminal state
    terminal = {"RUNNING", "BLOCKED", "FAILED", "GATE_FAILED"}
    nxt_status = nxt.get("status", "")
    if nxt_status not in terminal and nxt_status != "LAUNCHING":
        if not pgrep("schedule_catalog_matched_noq_after_legacy.sh"):
            # only restart if not already in a terminal handoff
            log(f"REPAIR: scheduler dead (next={nxt_status}) — restarting after_legacy_noq")
            tmux_new(
                "after_legacy_noq",
                f"bash {SCHEDULE_SCRIPT}",
            )
            actions += 1

    # 3) final audit loop if guard DONE but audit not finished
    if legacy.get("phase") == "DONE" and audit.get("phase") not in ("PASSED", "FAILED"):
        if not pgrep("wait_and_audit_phase8_legacy_repro.sh"):
            log("REPAIR: final audit loop missing while guard DONE — restarting")
            tmux_new(
                "phase8_legacy_final_audit",
                f"bash {AUDIT_SCRIPT}",
            )
            actions += 1

    # 4) report hard issues from status (no silent resume of failed full train)
    issues = status.get("issues") or []
    hard = [i for i in issues if i.get("severity") == "hard"]
    if hard:
        log(f"HARD issues present ({len(hard)}): " + "; ".join(
            f"{i.get('code')}:{i.get('detail','')[:120]}" for i in hard[:5]
        ))
        # Isolated AMP overflow recovery is owned by supervise_phase8_legacy_repro.
        # Here we only ensure supervisor is alive during legacy TRAINING.
        if legacy.get("phase") == "TRAINING" and not pgrep(
            "supervise_phase8_legacy_repro.sh"
        ):
            log("REPAIR: legacy supervisor missing during TRAINING — restarting")
            tmux_new(
                "phase8_legacy_supervisor",
                "bash /home/kojiek/MeanAudio/scripts/training_pipelines/"
                "supervise_phase8_legacy_repro.sh",
            )
            actions += 1

    if actions == 0:
        log(
            f"OK no repair needed | phase={status.get('phase')} "
            f"healthy={status.get('healthy')} next={nxt_status} "
            f"guard={legacy.get('phase')} audit={audit.get('phase')}"
        )
    return actions


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--loop", action="store_true")
    parser.add_argument("--interval", type=int, default=120)
    args = parser.parse_args()

    if not args.loop:
        repair_once()
        return

    log(f"repair loop start interval={args.interval}s")
    while True:
        try:
            repair_once()
        except Exception as exc:  # noqa: BLE001 — keep loop alive
            log(f"repair loop error: {exc}")
        time.sleep(max(30, args.interval))


if __name__ == "__main__":
    main()
