#!/usr/bin/env python3
"""Run an exact script fd with a closed, explicitly supplied import path.

This file is intended to be launched by the bound system interpreter with
``-I -S``.  It deliberately never imports ``site`` and never adds the current
working directory, a script directory, or a repository root to ``sys.path``.
"""
from __future__ import annotations

import argparse
import importlib.abc
import importlib.util
import os
import stat
import sys
from types import ModuleType


STDLIB_PATHS = (
    "/usr/lib/python312.zip",
    "/usr/lib/python3.12",
    "/usr/lib/python3.12/lib-dynload",
)


def regular_fd(fd: int, label: str) -> None:
    try:
        metadata = os.fstat(fd)
    except OSError as exc:
        raise SystemExit(f"invalid {label} fd") from exc
    if not stat.S_ISREG(metadata.st_mode):
        raise SystemExit(f"{label} fd is not a regular file")


def exact_directory(raw: str, label: str) -> str:
    if not raw or not os.path.isabs(raw) or os.path.islink(raw) or not os.path.isdir(raw):
        raise SystemExit(f"invalid exact {label} directory: {raw!r}")
    resolved = os.path.realpath(raw)
    if resolved != raw.rstrip("/"):
        raise SystemExit(f"non-canonical exact {label} directory: {raw!r}")
    return resolved


def read_regular_path(path: str, label: str) -> bytes:
    fd = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    try:
        regular_fd(fd, label)
        return read_fd(fd)
    finally:
        os.close(fd)


def read_fd(fd: int) -> bytes:
    chunks: list[bytes] = []
    offset = 0
    while True:
        block = os.pread(fd, 1 << 20, offset)
        if not block:
            return b"".join(chunks)
        chunks.append(block)
        offset += len(block)


class ExactSourceLoader(importlib.abc.Loader):
    """Compile an exact .py source directly, never consulting bytecode caches."""

    def __init__(self, fullname: str, source_path: str, *, package: bool) -> None:
        self.fullname = fullname
        self.source_path = source_path
        self.package = package

    def create_module(self, spec: object) -> None:
        return None

    def exec_module(self, module: ModuleType) -> None:
        module.__file__ = self.source_path
        module.__cached__ = None
        if self.package:
            module.__package__ = self.fullname
            module.__path__ = [os.path.dirname(self.source_path)]
        else:
            module.__package__ = self.fullname.rpartition(".")[0]
        source = read_regular_path(self.source_path, f"source module {self.fullname}")
        exec(compile(source, self.source_path, "exec"), module.__dict__, module.__dict__)


class ExactSourceFinder(importlib.abc.MetaPathFinder):
    """Prefer exact source under approved roots and reject legacy .pyc authority."""

    def __init__(self, roots: list[str], package_roots: dict[str, str]) -> None:
        self.roots = roots
        self.package_roots = package_roots

    @staticmethod
    def spec(fullname: str, base: str, relative: list[str]) -> object | None:
        package_init = os.path.join(base, *relative, "__init__.py")
        module_source = os.path.join(base, *relative) + ".py"
        if os.path.isfile(package_init) and not os.path.islink(package_init):
            loader = ExactSourceLoader(fullname, package_init, package=True)
            return importlib.util.spec_from_file_location(
                fullname, package_init, loader=loader,
                submodule_search_locations=[os.path.dirname(package_init)],
            )
        if os.path.isfile(module_source) and not os.path.islink(module_source):
            return importlib.util.spec_from_file_location(
                fullname, module_source, loader=ExactSourceLoader(fullname, module_source, package=False),
            )
        legacy = [
            os.path.join(base, *relative) + suffix for suffix in (".pyc", ".pyo")
        ] + [
            os.path.join(base, *relative, "__init__" + suffix) for suffix in (".pyc", ".pyo")
        ]
        if any(os.path.lexists(path) for path in legacy):
            raise ImportError(f"bytecode-only import authority rejected: {fullname}")
        return None

    def find_spec(self, fullname: str, path: object = None, target: object = None) -> object | None:
        for package_name, package_dir in self.package_roots.items():
            if fullname.startswith(package_name + "."):
                relative = fullname.split(".")[1:]
                return self.spec(fullname, package_dir, relative)
        relative = fullname.split(".")
        for root in self.roots:
            spec = self.spec(fullname, root, relative)
            if spec is not None:
                return spec
        return None


class BytecodeOnlyDenyFinder(importlib.abc.MetaPathFinder):
    """Reject legacy source-less bytecode in approved site-package roots."""

    def __init__(self, roots: list[str]) -> None:
        self.roots = roots

    def find_spec(self, fullname: str, path: object = None, target: object = None) -> None:
        relative = fullname.split(".")
        for root in self.roots:
            package_source = os.path.join(root, *relative, "__init__.py")
            module_source = os.path.join(root, *relative) + ".py"
            if os.path.isfile(package_source) or os.path.isfile(module_source):
                continue
            legacy = [
                os.path.join(root, *relative) + suffix for suffix in (".pyc", ".pyo")
            ] + [
                os.path.join(root, *relative, "__init__" + suffix) for suffix in (".pyc", ".pyo")
            ]
            if any(os.path.lexists(candidate) for candidate in legacy):
                raise ImportError(f"bytecode-only import authority rejected: {fullname}")
        return None


def preload_package(name: str, init_fd: int, package_dir: str) -> ModuleType:
    if not name.isidentifier() or "." in name:
        raise SystemExit("invalid preload package name")
    regular_fd(init_fd, "package init")
    package_dir = exact_directory(package_dir, "package")
    source_path = os.path.join(package_dir, "__init__.py")
    loader = ExactSourceLoader(name, source_path, package=True)
    spec = importlib.util.spec_from_file_location(
        name, source_path, loader=loader, submodule_search_locations=[package_dir],
    )
    if spec is None:
        raise SystemExit("could not create exact package spec")
    spec.submodule_search_locations = [package_dir]
    module = importlib.util.module_from_spec(spec)
    module.__file__ = source_path
    module.__path__ = [package_dir]
    sys.modules[name] = module
    try:
        source = read_fd(init_fd)
        exec(compile(source, source_path, "exec"), module.__dict__, module.__dict__)
    except BaseException:
        sys.modules.pop(name, None)
        raise
    return module


def read_script(fd: int) -> bytes:
    regular_fd(fd, "script")
    return read_fd(fd)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--site-packages", action="append", default=[])
    parser.add_argument("--package-name")
    parser.add_argument("--package-init-fd", type=int)
    parser.add_argument("--package-dir")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--script-fd", type=int)
    source.add_argument("--stdin-script", action="store_true")
    parser.add_argument("--display", required=True)
    parser.add_argument("script_args", nargs=argparse.REMAINDER)
    args = parser.parse_args()

    if args.script_args[:1] == ["--"]:
        args.script_args = args.script_args[1:]
    package_values = (args.package_name, args.package_init_fd, args.package_dir)
    if any(value is not None for value in package_values) != all(value is not None for value in package_values):
        raise SystemExit("package preload requires name, init fd, and exact directory")

    sites = [exact_directory(path, "site-packages") for path in args.site_packages]
    sys.path[:] = [*STDLIB_PATHS, *sites]
    sys.dont_write_bytecode = True
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    if sys.pycache_prefix != "/dev/null":
        raise SystemExit("isolated bootstrap requires -X pycache_prefix=/dev/null")
    package_roots = {}
    if all(value is not None for value in package_values):
        package_roots[args.package_name] = exact_directory(args.package_dir, "package")
    # Site roots remain exact sys.path entries so native-extension/package
    # semantics are preserved.  Approval-bound workspace packages use the
    # source-only finder, and evaluator/main scripts are compiled from held
    # descriptors below, so neither can acquire authority from bytecode.
    sys.meta_path.insert(0, ExactSourceFinder([], package_roots))
    sys.meta_path.insert(1, BytecodeOnlyDenyFinder(sites))
    if all(value is not None for value in package_values):
        preload_package(args.package_name, args.package_init_fd, args.package_dir)

    source_bytes = sys.stdin.buffer.read() if args.stdin_script else read_script(args.script_fd)
    sys.argv[:] = [args.display, *args.script_args]
    namespace = {
        "__name__": "__main__", "__file__": args.display, "__package__": None,
        "__cached__": None, "__builtins__": __builtins__,
    }
    exec(compile(source_bytes, args.display, "exec"), namespace, namespace)


if __name__ == "__main__":
    main()
