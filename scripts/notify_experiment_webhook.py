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
NOQ_REFERENCE_REPORT = Path(
    "/home/kojiek/logs/phase8_qwen_bucket_quarter_noq_FINAL_METRICS.json"
)


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


def read_report(path: Path | None) -> dict[str, Any] | None:
    if path is None or not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def metric_values(section: Any) -> dict[str, float] | None:
    if not isinstance(section, dict):
        return None
    values = {key: section.get(key) for key in METRIC_KEYS}
    return values if all(isinstance(value, (int, float)) for value in values.values()) else None


def primary_global_metrics(payload: dict[str, Any]) -> tuple[str, dict[str, float]] | None:
    global_metrics = payload.get("global")
    if not isinstance(global_metrics, dict):
        return None
    for key, label in (
        ("high_q9", "q9"),
        ("no_q", "NoQ"),
        ("quarter_noq_baseline", "NoQ"),
    ):
        values = metric_values(global_metrics.get(key))
        if values:
            return label, values
    return None


def experiment_label(payload: dict[str, Any], fallback: str) -> str:
    k = payload.get("k")
    strategy = payload.get("strategy")
    design = payload.get("design")
    if design == "NoQ_S1_to_Q_S2_only" and isinstance(k, int):
        return f"NoQ S1 → K{k} {strategy or 'Q'} S2"
    if isinstance(k, int):
        return f"K{k} {strategy or 'Q'}"
    if payload.get("arm") == "noq" or "noq" in fallback.lower():
        return "NoQ"
    return fallback


def first_screen_report(report: Path | None, fallback_name: str) -> list[str]:
    """Return the concise result block placed before operational metadata."""
    payload = read_report(report)
    if payload is None:
        return []
    primary = primary_global_metrics(payload)
    if primary is None:
        return [f"**Result:** final report exists but has no recognized Global metric: `{report}`"]

    endpoint, values = primary
    global_metrics = payload["global"]
    lines = [
        f"**Design:** {experiment_label(payload, fallback_name)}",
        f"**Global {endpoint} CLAP:** `{values['clap_score']:.4f}`",
    ]
    if payload.get("design") == "NoQ_S1_to_Q_S2_only":
        reference = read_report(NOQ_REFERENCE_REPORT)
        reference_primary = primary_global_metrics(reference) if reference else None
        if reference_primary:
            _, reference_values = reference_primary
            delta = values["clap_score"] - reference_values["clap_score"]
            lines[-1] += f"  ·  **Δ NoQ:** `{delta:+.4f}` (NoQ `{reference_values['clap_score']:.4f}`)"
    holdout = metric_values(global_metrics.get("holdout5009_high_q9"))
    if holdout:
        lines.append(f"**Holdout q9 CLAP:** `{holdout['clap_score']:.4f}`")
    low = metric_values(global_metrics.get("supported_low"))
    steering = global_metrics.get("q9_minus_low_clap")
    if low and isinstance(steering, (int, float)):
        lines.append(
            f"**Q response:** q9 `{values['clap_score']:.4f}` → low "
            f"`{low['clap_score']:.4f}`  ·  **Δ:** `{steering:+.4f}`"
        )
    lines.append(
        "**Aesthetics (Global):** "
        f"CE `{values['aes_CE']:.4f}` · CU `{values['aes_CU']:.4f}` · "
        f"PC `{values['aes_PC']:.4f}` · PQ `{values['aes_PQ']:.4f}`"
    )
    return lines


def report_lines(report: Path | None) -> list[str]:
    if report is None or not report.is_file():
        return []
    payload = read_report(report)
    if payload is None:
        return [f"- report artifact: `{report}` (not parseable as JSON)"]

    lines: list[str] = []
    stage1 = payload.get("stage1")
    if isinstance(stage1, dict):
        values = stage1.get("metrics")
        if isinstance(values, dict):
            line = metric_line("Stage 1", values)
            if line:
                lines.append(line)
        else:
            labels = {
                "quarter_noq_baseline": "Stage 1 baseline",
                "halfq_q9": "Stage 1 half-Q q9",
                "halfq_q0": "Stage 1 half-Q q0",
                "fullq_q9": "Stage 1 full-Q q9",
                "fullq_q6": "Stage 1 full-Q q6",
                "high_q9": "Stage 1 q9",
                "supported_low": "Stage 1 supported-low",
                "holdout5009_high_q9": "Stage 1 holdout-5009 q9",
            }
            for key, label in labels.items():
                nested = stage1.get(key)
                if isinstance(nested, dict):
                    line = metric_line(label, nested)
                    if line:
                        lines.append(line)
        value = stage1.get("q9_minus_low_clap")
        if isinstance(value, (int, float)):
            lines.append(f"- Stage 1 q9-minus-supported-low CLAP: {value:+.6f}")
    global_metrics = payload.get("global")
    if isinstance(global_metrics, dict):
        labels = {
            "quarter_noq_baseline": "Global baseline",
            "halfq_q9": "Global half-Q q9",
            "halfq_q0": "Global half-Q q0",
            "fullq_q9": "Global full-Q q9",
            "fullq_q6": "Global full-Q q6",
            "high_q9": "Global q9",
            "supported_low": "Global supported-low",
            "holdout5009_high_q9": "Global holdout-5009 q9",
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
            "halfq_q9_minus_fullq_q9_clap",
            "fullq_q9_minus_q6_clap",
            "q9_minus_low_clap",
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
    lines = [f"{emoji} **{title} · {args.experiment}**"]
    if args.status == "success":
        first_screen = first_screen_report(args.report, args.experiment)
        if first_screen:
            lines.extend(["", "**RESULT — read this first**", *first_screen])
    lines.extend([
        "",
        f"**Duration:** `{compact_duration(duration)}`  ·  **Host:** `{platform.node()}`",
        f"**Git:** `{git_identity(args.repo)}`  ·  **Time:** `{now}`",
    ])
    if args.exit_code is not None:
        lines.append(f"**Exit code:** `{args.exit_code}`")
    if args.summary:
        lines.append(f"**Summary:** {args.summary}")

    metrics = report_lines(args.report)
    if metrics:
        lines.extend(["", "**Detailed endpoints:**", *metrics])
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
