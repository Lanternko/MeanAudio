#!/usr/bin/env python3
"""Discord notifier with non-terminal operational-state titles.

This is intentionally separate from notify_experiment_webhook.py. Existing
queue contracts bind that file by SHA-256, so changing it would invalidate
already delivered receipts and prepared queue entries.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import notify_experiment_webhook as base  # noqa: E402


STATUS_INFO = {
    "advisory": ("⚠️", "STORAGE GATE BLOCKED"),
    "recovered": ("▶️", "STORAGE GATE CLEARED"),
    "held": ("⏸️", "QUEUE HELD"),
}


def build_content(args: argparse.Namespace) -> str:
    emoji, title = STATUS_INFO[args.status]
    now = datetime.now().astimezone().isoformat(timespec="seconds")
    lines = [
        f"{emoji} **{title} · {args.experiment}**",
        "",
        f"**Host:** `{platform.node()}`  ·  **Time:** `{now}`",
        f"**Git:** `{base.git_identity(args.repo)}`",
    ]
    if args.exit_code is not None:
        lines.append(f"**Exit code:** `{args.exit_code}`")
    if args.summary:
        lines.append(f"**Summary:** {args.summary}")
    content = "\n".join(lines)
    if len(content) > base.MAX_CONTENT:
        content = content[: base.MAX_CONTENT - 32] + "\n…(report truncated)"
    return content


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--status", required=True, choices=tuple(STATUS_INFO))
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--summary")
    parser.add_argument("--exit-code", type=int)
    parser.add_argument("--repo", type=Path, default=Path("/home/kojiek/MeanAudio"))
    parser.add_argument("--webhook-file", type=Path, default=base.DEFAULT_WEBHOOK_FILE)
    parser.add_argument("--dry-run", action="store_true",
                        default=os.environ.get("MEANAUDIO_NOTIFY_DRY_RUN", "").lower() == "true")
    parser.add_argument("--receipt-managed", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--gpu-released", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    content = build_content(args)
    if args.dry_run:
        print(content)
        return 0
    try:
        webhook = base.read_webhook(args.webhook_file)
    except (OSError, ValueError) as exc:
        print(f"[NOTIFY PRE-REQUEST FAIL] {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    separator = "&" if "?" in webhook else "?"
    request = urllib.request.Request(
        webhook + separator + "wait=true",
        data=json.dumps({"content": content, "allowed_mentions": {"parse": []}}).encode(),
        headers={"Content-Type": "application/json", "User-Agent": "MeanAudio-Operational-Notifier/1.0"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=20) as response:
            response.read()
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        print(f"[NOTIFY FAIL] {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    print("[NOTIFY OK] operational event delivered")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
