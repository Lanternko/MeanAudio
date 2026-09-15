#!/usr/bin/env python3
"""Run a contract preflight so its failure reason survives.

Why this exists: guests call ``subprocess.run(c['commands']['preflight'])`` with no
capture, so a rejected launch reaches the operator as ``preflight invalid rc=1`` and
nothing else. On 2026-09-12 job 051 was held because a training pipeline left
``meanaudio/model/mean_flow.py`` on Stage 2 while the contract pinned the Stage 1
hash; the validator said exactly that, and the message went to a closed pipe.
Identifying it needed a manual rerun of the validator.

Use this from new guests instead of a bare ``subprocess.run``::

    from preflight_capture import run_preflight
    rc = run_preflight(c['commands']['preflight'], STATE)

Behaviour is otherwise identical to the bare call: the return code is passed through
untouched, so the caller keeps its own ``rc == 75`` resource-wait handling and its own
terminal/held decisions. This only makes the reason legible; it never changes the
verdict. Existing guests are hash-pinned by their contracts, so adopt this in new
guests rather than retrofitting launched ones.
"""
from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Sequence

TAIL_CHARS = 4000


def run_preflight(command: Sequence[str], state_dir: Path, *,
                  log_name: str = "preflight_last.log") -> int:
    """Run ``command``, persist its output next to the guest state, return its rc.

    The log is rewritten per attempt and holds the whole transcript. On failure the
    tail is echoed to the guest's own stdout so the queue log carries the reason too.
    Capture problems are never allowed to change the verdict.
    """
    completed = subprocess.run(command, capture_output=True, text=True)
    transcript = (completed.stdout or "") + (completed.stderr or "")
    try:
        state_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
        (state_dir / log_name).write_text(transcript, encoding="utf-8")
    except OSError as exc:  # never mask the real preflight result
        print(f"[preflight_capture] could not persist log: {exc}", flush=True)
    if completed.returncode not in (0, 75) and transcript.strip():
        print(
            f"[preflight_capture] rc={completed.returncode}; tail of output:\n"
            f"{transcript.strip()[-TAIL_CHARS:]}",
            flush=True,
        )
    return completed.returncode
