#!/usr/bin/env python3
"""Validate CFG0 runtime roots and children against shared-host replacement."""
from __future__ import annotations

import argparse
import os
import stat
from pathlib import Path


def validate_root_target(root_arg: Path, target: Path) -> None:
    root_arg = root_arg.absolute()
    root = root_arg.resolve(strict=True)
    if root != root_arg or root_arg.is_symlink():
        raise ValueError(f"registered root must not be a symlink: {root_arg}")
    root_info = root.lstat()
    if not root.is_dir() or not os.access(root, os.W_OK | os.X_OK):
        raise ValueError(f"unsafe registered output root: {root_arg} -> {root}")
    if root_info.st_uid != os.geteuid() or stat.S_IMODE(root_info.st_mode) != 0o700:
        raise ValueError(f"registered root must be current-user-owned mode 0700: {root}")
    for ancestor in (root, *root.parents):
        info = ancestor.lstat()
        if stat.S_ISLNK(info.st_mode) or info.st_mode & 0o022:
            raise ValueError(f"writable or symlinked runtime ancestor: {ancestor}")
    try:
        relative = target.relative_to(root_arg)
    except ValueError as exc:
        raise ValueError(f"output escapes registered root: {target}") from exc
    current = root
    for part in relative.parts:
        current = current / part
        if current.exists() or current.is_symlink():
            info = current.lstat()
            if (stat.S_ISLNK(info.st_mode) or info.st_uid != os.geteuid()
                    or info.st_mode & 0o077):
                raise ValueError(f"unsafe output component: {current}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--metrics-root", type=Path, required=True)
    parser.add_argument("--report-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--audio", type=Path, required=True)
    parser.add_argument("--metrics-dir", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    validate_root_target(args.output_root, args.out)
    validate_root_target(args.output_root, args.audio)
    validate_root_target(args.metrics_root, args.metrics_dir)
    validate_root_target(args.report_root, args.report)
    print("OUTPUT_PATH_OK")


if __name__ == "__main__":
    main()
