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
import re
import stat
import subprocess
import sys
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent / "experiment_harness"))
from notification_receipts import deliver_required  # noqa: E402

DEFAULT_WEBHOOK_FILE = Path("/home/kojiek/.config/meanaudio/discord_webhook_url")
DEFAULT_MENTION_FILE = Path("/home/kojiek/.config/meanaudio/discord_mention_user_id")
MAX_CONTENT = 1_950
MAX_SECRET_BYTES = 4_096
METRIC_KEYS = ("clap_score", "aes_CE", "aes_CU", "aes_PC", "aes_PQ")
NOQ_REFERENCE_REPORT = Path(
    "/home/kojiek/logs/phase8_qwen_bucket_quarter_noq_FINAL_METRICS.json"
)


_SEAT_EXPERIMENT = re.compile(r"^p[12]-seat-[A-Za-z0-9][A-Za-z0-9_.-]*$")
_START_EVENT_SUFFIXES = (":start", ":queue_handoff")
_STATUS_TITLES = {
    "success": ("✅", "COMPLETED"),
    "failure": ("❌", "FAILED"),
    "interrupted": ("⛔", "INTERRUPTED"),
    "held": ("⏸️", "QUEUE HELD"),
    "idle": ("🟢", "QUEUE IDLE"),
    "start": ("▶️", "STARTED"),
    "test": ("🔔", "WEBHOOK TEST"),
}


def discord_title(status: str, experiment: str) -> tuple[str, str]:
    """Seating, queue_handoff, and start are progress, not holds."""
    if status == "start" or (
        status == "held"
        and (
            _SEAT_EXPERIMENT.fullmatch(experiment)
            or experiment.endswith(_START_EVENT_SUFFIXES)
        )
    ):
        return _STATUS_TITLES["start"]
    return _STATUS_TITLES[status]


def parse_queue_identity(experiment: str) -> dict[str, str]:
    for kind, pattern in (
        ("seat", r"(p[12])-seat-(.+)"),
        ("held", r"(p[12])-held-(.+)"),
        ("paused", r"(p[12])-paused-(.+)"),
        ("done", r"(p[12])-done-(.+)"),
        ("incident", r"(p[12])-incident-(.+)"),
    ):
        matched = re.fullmatch(pattern, experiment)
        if matched:
            return {"job": matched.group(2), "role": matched.group(1), "event": kind}
    if experiment == "gpu-queue-idle":
        return {"job": "gpu-queue", "role": "", "event": "idle"}
    parts = experiment.split(":")
    if len(parts) >= 3:
        run = parts[1]
        numbered = re.search(r"(?:^|-)(\d{3}[-_].+)$", run)
        job = numbered.group(1).replace("-", "_") if numbered else parts[0]
        return {"job": job, "role": "", "event": parts[-1], "run": run}
    return {"job": experiment, "role": "", "event": ""}


def lifecycle_for(title: str, event: str) -> str:
    if title == "STARTED" or event in {"seat", "start", "queue_handoff"}:
        return "starting"
    if title == "COMPLETED" or event in {"done", "success"}:
        return "finished"
    if title == "FAILED" or event in {"failure", "incident"}:
        return "failed"
    if title == "INTERRUPTED" or event in {"paused", "interrupted"}:
        return "paused"
    if title == "QUEUE IDLE" or event == "idle":
        return "idle"
    if title == "QUEUE HELD":
        return "blocked"
    return "unknown"


def what_happened(lifecycle: str, event: str) -> str:
    label = {
        "seat": "host seating",
        "queue_handoff": "queue handoff",
        "start": "child start",
        "held": "hold",
        "disk_warning": "disk warning",
        "paused": "P1 pause",
        "incident": "host incident",
    }.get(event, event or "status update")
    if lifecycle == "starting":
        return f"{label}: this experiment is starting. Not a hold, not a stop."
    if lifecycle == "blocked":
        return f"{label}: this experiment was blocked."
    if lifecycle == "failed":
        return f"{label}: this experiment failed."
    if lifecycle == "paused":
        return f"{label}: this experiment paused (P1 preemption or signal)."
    if lifecycle == "finished":
        return f"{label}: this experiment finished."
    if lifecycle == "idle":
        return "Queue idle: p1/p2 pending and running are empty."
    return label


def why_line(lifecycle: str, summary: str | None) -> str:
    text = (summary or "").strip()
    lowered = text.lower()
    if lifecycle == "starting":
        return "Does not apply — the job is starting, not stopping."
    if lifecycle in {"finished", "idle"}:
        return "Does not apply."
    if any(key in lowered for key in ("hard stop", "free_bytes", "storage", "disk")):
        kind = "Storage insufficient (disk warning or hard stop)."
    elif any(key in lowered for key in ("hash", "identity mismatch", "notifier")):
        kind = "Harness pin/hash mismatch — not storage, not a GPU race."
    elif "stall" in lowered:
        kind = "Watcher stall: no output progress."
    elif any(key in lowered for key in ("foreign", "smi", "probe")):
        kind = "GPU ownership race or nvidia-smi probe failed."
    elif "pause" in lowered:
        kind = "P1 preemption pause, not a failure."
    elif any(key in lowered for key in ("preflight", "contract")):
        kind = "Preflight/contract rejected the launch."
    elif lifecycle == "failed":
        kind = "Process exited non-zero."
    elif lifecycle == "paused":
        kind = "Interrupted."
    else:
        kind = "Hold/stop reason:"
    return f"{kind} {text}".strip()


def gpu_status_line() -> str:
    queue = Path(os.environ.get("GPU_QUEUE_ROOT", "/home/kojiek/gpu_queue"))
    owner = "no queue owner"
    try:
        rec = json.loads((queue / "gpu0.owner.json").read_text(encoding="utf-8"))
        if isinstance(rec, dict) and rec.get("job_id"):
            owner = f"owner `{rec.get('job_id')}` ({rec.get('role') or '?'})"
    except (OSError, json.JSONDecodeError, TypeError):
        pass
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=name,utilization.gpu,memory.used,memory.total",
                "--format=csv,noheader,nounits",
            ],
            timeout=3,
            text=True,
        ).strip().splitlines()[0]
        name, util, used, total = [part.strip() for part in out.split(",")]
        return f"{name} · {util}% util · {used}/{total} MiB · {owner}"
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired, ValueError, IndexError):
        return f"unavailable · {owner}"


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
    results = payload.get("results")
    if isinstance(results, dict):
        lines = []
        for label, section in results.items():
            values = metric_values(section)
            if values:
                lines.append(
                    f"**{label}:** CLAP `{values['clap_score']:.4f}` · "
                    f"CE `{values['aes_CE']:.4f}` · CU `{values['aes_CU']:.4f}` · "
                    f"PC `{values['aes_PC']:.4f}` · PQ `{values['aes_PQ']:.4f}`"
                )
        if lines:
            return lines
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
    emoji, title = discord_title(args.status, args.experiment)
    ident = parse_queue_identity(args.experiment)
    lifecycle = lifecycle_for(title, ident.get("event") or "")
    job = ident.get("job") or args.experiment
    now = datetime.now().astimezone().isoformat(timespec="seconds")
    duration = None
    if args.started_epoch is not None:
        duration = max(0, int(datetime.now().timestamp()) - args.started_epoch)
    lines = [f"{emoji} **{title} · {job}**"]
    if args.experiment != job:
        lines.append(f"**Queue key:** `{args.experiment}`")
    if args.status == "success":
        first_screen = first_screen_report(args.report, args.experiment)
        if first_screen:
            lines.extend(["", "**RESULT — read this first**", *first_screen])
    lines.extend([
        "",
        f"**Experiment:** `{job}`",
        f"**What happened:** {what_happened(lifecycle, ident.get('event') or '')}",
        f"**GPU:** {gpu_status_line()}",
        f"**Why:** {why_line(lifecycle, args.summary)}",
        "",
        f"**Duration:** `{compact_duration(duration)}`  ·  **Host:** `{platform.node()}`",
        f"**Git:** `{git_identity(args.repo)}`  ·  **Time:** `{now}`",
    ])
    if args.exit_code is not None:
        lines.append(f"**Exit code:** `{args.exit_code}`")
    if args.summary:
        lines.append(f"**Detail:** {args.summary}")

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


def _read_owned_secret_file(path: Path, *, max_bytes: int = MAX_SECRET_BYTES) -> str:
    """Read one small secret from the same verified, non-symlink file descriptor."""
    flags = os.O_RDONLY
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    if hasattr(os, "O_NONBLOCK"):
        flags |= os.O_NONBLOCK
    fd = os.open(path, flags)
    try:
        metadata = os.fstat(fd)
        if not stat.S_ISREG(metadata.st_mode):
            raise ValueError("secret path is not a regular file")
        if metadata.st_uid != os.geteuid():
            raise ValueError("secret file is not owned by the current user")
        mode = stat.S_IMODE(metadata.st_mode)
        if mode != 0o600:
            raise ValueError(f"secret file permissions must be 0600, got {mode:04o}")
        if metadata.st_size > max_bytes:
            raise ValueError(f"secret file exceeds {max_bytes} bytes")
        chunks: list[bytes] = []
        remaining = max_bytes + 1
        while remaining:
            chunk = os.read(fd, min(remaining, 1024))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        raw = b"".join(chunks)
        if len(raw) > max_bytes:
            raise ValueError(f"secret file exceeds {max_bytes} bytes")
        try:
            return raw.decode("utf-8").strip()
        except UnicodeDecodeError as exc:
            raise ValueError("secret file is not valid UTF-8") from exc
    finally:
        os.close(fd)


def read_webhook(path: Path) -> str:
    value = _read_owned_secret_file(path)
    allowed = (
        "https://discord.com/api/webhooks/",
        "https://discordapp.com/api/webhooks/",
    )
    if not value.startswith(allowed):
        raise ValueError("webhook file does not contain a Discord webhook URL")
    return value


def read_mention_user_id(path: Path) -> str | None:
    """Return an opt-in Discord user ID without ever guessing a recipient."""
    value = os.environ.get("MEANAUDIO_DISCORD_MENTION_USER_ID", "").strip()
    if not value and path.exists():
        value = _read_owned_secret_file(path)
    if not value:
        return None
    if not value.isdecimal():
        raise ValueError("Discord mention user ID must contain digits only")
    return value


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--status",
        required=True,
        choices=("success", "failure", "interrupted", "held", "idle", "start", "test"),
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
        "--mention-file",
        type=Path,
        default=Path(os.environ.get("MEANAUDIO_DISCORD_MENTION_FILE", str(DEFAULT_MENTION_FILE))),
    )
    parser.add_argument(
        "--gpu-released",
        action="store_true",
        help="Mark this as the terminal notification for an experiment chain.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=os.environ.get("MEANAUDIO_NOTIFY_DRY_RUN", "").lower() == "true",
    )
    parser.add_argument(
        "--receipt-managed",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    args = parser.parse_args()

    # Existing hosts issue this exact anchored key before seat_and_wait. Migrate
    # only that call shape into the authoritative global receipt ledger; every
    # other historical notifier invocation keeps its legacy behavior.
    seat = re.fullmatch(r"(p[12])-seat-([A-Za-z0-9][A-Za-z0-9_.-]*)", args.experiment)
    if seat and not args.receipt_managed:
        role, job = seat.groups()
        queue_root = Path(os.environ.get("GPU_QUEUE_ROOT", "/home/kojiek/gpu_queue"))
        candidates = [
            queue_root / role / "pending" / f"{job}.sh",
            queue_root / role / "running" / f"{job}.sh",
        ]
        launcher = next((path for path in candidates if path.is_file()), None)
        contract_path = None
        seat_contract: dict[str, Any] | None = None
        if launcher is not None:
            for line in launcher.read_text(encoding="utf-8", errors="replace").splitlines():
                stripped = line.strip().lstrip("#").strip()
                if stripped.startswith("GPU_QUEUE_CONTRACT="):
                    contract_path = Path(stripped.split("=", 1)[1].strip().strip("\"'"))
                    break
            if contract_path is not None and contract_path.is_file():
                try:
                    loaded = json.loads(contract_path.read_text(encoding="utf-8"))
                    seat_contract = loaded if isinstance(loaded, dict) else None
                except (OSError, json.JSONDecodeError):
                    seat_contract = None
        receipt_config = seat_contract.get("notification_receipts") if seat_contract else None
        if isinstance(receipt_config, dict) and receipt_config.get("required") is True:
            try:
                receipt_root = Path(receipt_config["root"])
                deliver_required(
                    contract_path=contract_path,
                    launcher_path=launcher,
                    event="seat_attempt",
                    status=args.status,
                    summary=args.summary or "",
                    idempotency_key=args.experiment,
                    notifier=Path(__file__).resolve(),
                    python=Path(sys.executable),
                    root=receipt_root,
                )
            except (OSError, RuntimeError, ValueError, KeyError, json.JSONDecodeError) as exc:
                print(f"[NOTIFY FAIL] {type(exc).__name__}: {exc}", file=sys.stderr)
                return 2
            print(f"[NOTIFY OK] durable seat_attempt experiment={args.experiment}")
            return 0

    content = build_content(args)
    should_mention = args.gpu_released or args.status == "idle"
    try:
        mention_user_id = read_mention_user_id(args.mention_file) if should_mention else None
    except (OSError, ValueError) as exc:
        print(f"[NOTIFY PRE-REQUEST FAIL] {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    prefix = f"<@{mention_user_id}>\n" if mention_user_id else ""
    if args.gpu_released:
        content = (
            prefix + content
            + "\n\n**GPU:** this experiment released its GPU reservation; "
            "the queue controller will re-check resources for the next eligible run."
        )
        if len(content) > MAX_CONTENT:
            content = content[: MAX_CONTENT - 32] + "\n…(report truncated)"
    elif args.status == "idle" and prefix:
        content = prefix + content
        if len(content) > MAX_CONTENT:
            content = content[: MAX_CONTENT - 32] + "\n…(report truncated)"
    if args.dry_run:
        print(content)
        return 0

    try:
        webhook = read_webhook(args.webhook_file)
    except (OSError, ValueError) as exc:
        print(f"[NOTIFY PRE-REQUEST FAIL] {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    try:
        separator = "&" if "?" in webhook else "?"
        request = urllib.request.Request(
            webhook + separator + "wait=true",
            data=json.dumps(
                {
                    "content": content,
                    "allowed_mentions": (
                        {"users": [mention_user_id]} if mention_user_id else {"parse": []}
                    ),
                }
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
