#!/usr/bin/env python3
"""Send a bounded experiment completion/failure report to Discord.

The webhook URL is read from a local 0600 file and is never accepted as a
command-line argument, printed, or written into experiment artifacts.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path
from typing import Any

DEFAULT_WEBHOOK_FILE = Path("/home/kojiek/.config/meanaudio/discord_webhook_url")
MAX_CONTENT = 1_950
METRIC_KEYS = ("clap_score", "aes_CE", "aes_CU", "aes_PC", "aes_PQ")


def compact_duration(seconds: int | None) -> str:
    if seconds is None or seconds < 0:
        return "unknown"
    hours, remainder = divmod(seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours}h {minutes}m {secs}s"
    if minutes:
        return f"{minutes}m {secs}s"
    return f"{secs}s"


def git_identity(repo: Path) -> str:
    try:
        branch = subprocess.check_output(
            ["git", "branch", "--show-current"], cwd=repo, text=True
        ).strip()
        commit = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=repo, text=True
        ).strip()
        return f"{branch}@{commit}"
    except (OSError, subprocess.CalledProcessError):
        return "unavailable"


def metric_line(label: str, values: dict[str, Any]) -> str | None:
    found: list[str] = []
    for key in METRIC_KEYS:
        value = values.get(key)
        if isinstance(value, (int, float)):
            short = "CLAP" if key == "clap_score" else key.removeprefix("aes_")
            found.append(f"{short}={value:.4f}")
    return f"- {label}: " + ", ".join(found) if found else None


def report_lines(report: Path | None) -> list[str]:
    if report is None or not report.is_file():
        return []
    try:
        payload = json.loads(report.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return [f"- report artifact: `{report}` (not parseable as JSON)"]

    lines: list[str] = []
    stage1 = payload.get("stage1")
    if isinstance(stage1, dict):
        values = stage1.get("metrics", stage1)
        if isinstance(values, dict):
            line = metric_line("Stage 1", values)
            if line:
                lines.append(line)
    global_metrics = payload.get("global")
    if isinstance(global_metrics, dict):
        labels = {
            "quarter_noq_baseline": "Global baseline",
            "halfq_q9": "Global half-Q q9",
            "halfq_q0": "Global half-Q q0",
        }
        for key, label in labels.items():
            values = global_metrics.get(key)
            if isinstance(values, dict):
                line = metric_line(label, values)
                if line:
                    lines.append(line)
        for key in (
            "halfq_q9_minus_baseline_clap",
            "halfq_q0_minus_baseline_clap",
            "halfq_q9_minus_q0_clap",
        ):
            value = global_metrics.get(key)
            if isinstance(value, (int, float)):
                lines.append(f"- {key}: {value:+.6f}")
    if not lines:
        values = payload.get("metrics")
        if isinstance(values, dict):
            line = metric_line("Metrics", values)
            if line:
                lines.append(line)
    lines.append(f"- report artifact: `{report}`")
    return lines


def failure_tail(log_path: Path | None, line_count: int = 12) -> str | None:
    if log_path is None or not log_path.is_file():
        return None
    try:
        lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return None
    tail = "\n".join(lines[-line_count:]).strip()
    if not tail:
        return None
    tail = tail.replace("```", "'''")
    return tail[-900:]


def build_content(args: argparse.Namespace) -> str:
    status_info = {
        "success": ("✅", "COMPLETED"),
        "failure": ("❌", "FAILED"),
        "interrupted": ("⛔", "INTERRUPTED"),
        "test": ("🔔", "WEBHOOK TEST"),
    }
    emoji, title = status_info[args.status]
    now = datetime.now().astimezone().isoformat(timespec="seconds")
    duration = None
    if args.started_epoch is not None:
        duration = max(0, int(datetime.now().timestamp()) - args.started_epoch)
    lines = [
        f"{emoji} **MeanAudio experiment {title}**",
        f"**Experiment:** `{args.experiment}`",
        f"**Time:** `{now}`",
        f"**Host:** `{platform.node()}`",
        f"**Git:** `{git_identity(args.repo)}`",
        f"**Duration:** `{compact_duration(duration)}`",
    ]
    if args.exit_code is not None:
        lines.append(f"**Exit code:** `{args.exit_code}`")
    if args.summary:
        lines.append(f"**Summary:** {args.summary}")

    metrics = report_lines(args.report)
    if metrics:
        lines.extend(["", "**Report:**", *metrics])
    if args.status in {"failure", "interrupted"}:
        tail = failure_tail(args.log)
        if tail:
            lines.extend(["", "**Log tail:**", f"```text\n{tail}\n```"])

    content = "\n".join(lines)
    if len(content) > MAX_CONTENT:
        content = content[: MAX_CONTENT - 32] + "\n…(report truncated)"
    return content


def read_webhook(path: Path) -> str:
    mode = path.stat().st_mode & 0o777
    if mode & 0o077:
        raise ValueError(f"webhook file permissions must be 0600, got {mode:04o}")
    value = path.read_text(encoding="utf-8").strip()
    allowed = (
        "https://discord.com/api/webhooks/",
        "https://discordapp.com/api/webhooks/",
    )
    if not value.startswith(allowed):
        raise ValueError("webhook file does not contain a Discord webhook URL")
    return value


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--status",
        required=True,
        choices=("success", "failure", "interrupted", "test"),
    )
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--summary")
    parser.add_argument("--report", type=Path)
    parser.add_argument("--log", type=Path)
    parser.add_argument("--exit-code", type=int)
    parser.add_argument("--started-epoch", type=int)
    parser.add_argument("--repo", type=Path, default=Path("/home/kojiek/MeanAudio"))
    parser.add_argument(
        "--webhook-file",
        type=Path,
        default=Path(
            os.environ.get(
                "MEANAUDIO_DISCORD_WEBHOOK_FILE", str(DEFAULT_WEBHOOK_FILE)
            )
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=os.environ.get("MEANAUDIO_NOTIFY_DRY_RUN", "").lower() == "true",
    )
    args = parser.parse_args()

    content = build_content(args)
    if args.dry_run:
        print(content)
        return 0

    try:
        webhook = read_webhook(args.webhook_file)
        separator = "&" if "?" in webhook else "?"
        request = urllib.request.Request(
            webhook + separator + "wait=true",
            data=json.dumps(
                {"content": content, "allowed_mentions": {"parse": []}}
            ).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "User-Agent": "MeanAudio-Experiment-Reporter/1.0",
            },
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=20) as response:
            if response.status not in (200, 204):
                raise RuntimeError(f"unexpected Discord status {response.status}")
    except (OSError, ValueError, RuntimeError, urllib.error.URLError) as exc:
        print(f"[NOTIFY FAIL] {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2

    print(f"[NOTIFY OK] status={args.status} experiment={args.experiment}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
