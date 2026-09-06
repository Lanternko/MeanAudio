#!/usr/bin/env python3
"""Deterministic transitive-runtime manifest verification (stdlib only)."""
from __future__ import annotations

import hashlib
import json
import os
import stat
from pathlib import Path
from typing import Any


TREE_BINDING_RULES = {
    "schema_version": 1,
    "excluded_directory_names": [
        ".hypothesis", ".mypy_cache", ".pytest_cache", ".pytype", ".ruff_cache", "__pycache__",
    ],
    "excluded_file_suffixes": [".pyc", ".pyo"],
    "rationale": "Interpreter/test caches are derived mutable state, are never approval authority, and must not cause runtime self-drift.",
}
CACHE_DIRECTORY_NAMES = frozenset(TREE_BINDING_RULES["excluded_directory_names"])
CACHE_FILE_SUFFIXES = tuple(TREE_BINDING_RULES["excluded_file_suffixes"])


def sha256_file(path: Path) -> str:
    fd = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    try:
        if not stat.S_ISREG(os.fstat(fd).st_mode):
            raise RuntimeError(f"runtime binding is not a regular file: {path}")
        digest = hashlib.sha256()
        while True:
            block = os.read(fd, 8 << 20)
            if not block:
                break
            digest.update(block)
        return digest.hexdigest()
    finally:
        os.close(fd)


def tree_sha256(root: Path) -> tuple[str, list[dict[str, str]], int]:
    if not root.is_dir() or root.is_symlink():
        raise RuntimeError(f"runtime tree is missing, non-directory, or symlinked: {root}")
    digest = hashlib.sha256()
    pth_files: list[dict[str, str]] = []
    count = 0
    for directory, directories, files in os.walk(root, topdown=True, followlinks=False):
        directories.sort()
        files.sort()
        base = Path(directory)
        retained = []
        for name in directories:
            if name in CACHE_DIRECTORY_NAMES:
                continue
            child = base / name
            if child.is_symlink():
                relative = child.relative_to(root).as_posix()
                target = os.readlink(child)
                digest.update(b"L\0" + relative.encode() + b"\0" + target.encode() + b"\n")
                count += 1
            else:
                retained.append(name)
        directories[:] = retained
        for name in files:
            if name.endswith(CACHE_FILE_SUFFIXES):
                continue
            path = base / name
            if path.is_symlink():
                relative = path.relative_to(root).as_posix()
                target = os.readlink(path)
                resolved = path.resolve(strict=True)
                file_hash = sha256_file(resolved)
                digest.update(b"L\0" + relative.encode() + b"\0" + target.encode() + b"\0" + file_hash.encode() + b"\n")
                count += 1
                if path.suffix == ".pth":
                    pth_files.append({"path": str(path), "sha256": file_hash})
                continue
            relative = path.relative_to(root).as_posix()
            file_hash = sha256_file(path)
            digest.update(relative.encode() + b"\0" + file_hash.encode() + b"\n")
            count += 1
            if path.suffix == ".pth":
                pth_files.append({"path": str(path), "sha256": file_hash})
    return digest.hexdigest(), pth_files, count


def canonical(value: Any) -> bytes:
    return json.dumps(value, separators=(",", ":"), sort_keys=True).encode()


def verify_manifest(path: Path, expected_sha256: str, required_roles: set[str]) -> dict[str, Any]:
    if sha256_file(path) != expected_sha256:
        raise RuntimeError("transitive runtime manifest file drift")
    manifest = json.loads(path.read_text())
    if manifest.get("document_kind") != "meanaudio_transitive_runtime_manifest" or manifest.get("schema_version") != 1:
        raise RuntimeError("invalid transitive runtime manifest")
    if manifest.get("tree_binding_rules") != TREE_BINDING_RULES:
        raise RuntimeError("runtime manifest cache-exclusion rules are missing or drifted")
    entries = manifest.get("entries")
    if not isinstance(entries, list):
        raise RuntimeError("runtime manifest entries are missing")
    roles = [entry.get("role") for entry in entries]
    if len(roles) != len(set(roles)) or not required_roles.issubset(set(roles)):
        raise RuntimeError("runtime manifest required roles are missing or duplicated")
    for entry in entries:
        target = Path(str(entry.get("path", "")))
        if entry.get("kind") == "file":
            if sha256_file(target) != entry.get("sha256"):
                raise RuntimeError(f"runtime file binding drift: {entry.get('role')} {target}")
        elif entry.get("kind") == "tree":
            observed, pth_files, count = tree_sha256(target)
            if observed != entry.get("sha256") or count != entry.get("file_count"):
                raise RuntimeError(f"runtime tree binding drift: {entry.get('role')} {target}")
            if pth_files != entry.get("pth_files", []):
                raise RuntimeError(f"runtime .pth set drift: {entry.get('role')} {target}")
        else:
            raise RuntimeError(f"unknown runtime manifest entry kind: {entry.get('kind')}")
    return manifest
