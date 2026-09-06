#!/usr/bin/env python3
"""Offline structural and consistency validation for experiment-harness JSON.

This validator is intentionally not a runtime authorization mechanism. The CLI
checks raw-byte hash bindings for the files it parses, but it does not authenticate
approval records, acquire locks, prove process identity, or execute any action.
Runtime must independently rehash immediately before any authorized action.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import stat
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Mapping, Sequence

try:
    import jsonschema
    from jsonschema import Draft202012Validator, FormatChecker
except ImportError:  # pragma: no cover - exercised by an environment without the dependency
    jsonschema = None
    Draft202012Validator = None  # type: ignore[assignment,misc]
    FormatChecker = None  # type: ignore[assignment,misc]


BUNDLE_ID = "harn-schema-v1"
NOTICE = "STRUCTURAL VALIDATION ONLY — SUCCESS IS NOT AUTHORIZATION."
RUNTIME_BOUNDARY = (
    "Reserved to runtime immediately before action: authenticity, independent raw-file "
    "rehashing, atomic fsync/rename persistence, locks, live process identity, full "
    "symlink/TOCTOU defenses, and execution."
)
MAX_FILE_BYTES = 4 * 1024 * 1024
MAX_DEPTH = 64
MAX_NODES = 200_000
MAX_STRING_LENGTH = 16_384
MAX_KEY_LENGTH = 256
MAX_ARRAY_LENGTH = 100_000
MAX_ERRORS = 50
SCHEMA_FILES = {
    "contract": "experiment-contract-v1.schema.json",
    "preflight": "preflight-report-v1.schema.json",
    "ledger": "event-ledger-v1.schema.json",
    "queue": "queue-state-v1.schema.json",
}
SCHEMA_DIR = Path(__file__).resolve().parents[1] / "docs" / "experiments" / "schemas"
_SECRET_KEY_RE = re.compile(
    r"(?:^|[_-])(?:password|passwd|secret|api[_-]?key|access[_-]?token|auth[_-]?token|"
    r"github[_-]?token|slack[_-]?token|discord[_-]?token|bearer|webhook(?:[_-]?url)?|"
    r"private[_-]?key|client[_-]?secret|credentials?)(?:$|[_-])",
    re.IGNORECASE,
)
_EXACT_SECRET_KEYS = {
    "authorization", "credential", "credentials", "password", "passwd", "secret", "token"
}
_SECRET_KEY_EDGE_RE = re.compile(
    r"(?:^(?:authorization|credential|credentials)[_-]|[_-]token$)", re.IGNORECASE
)
_DISCORD_WEBHOOK_RE = re.compile(
    r"https?://(?:canary\.|ptb\.)?(?:discord(?:app)?\.com)/api(?:/v\d+)?/webhooks/",
    re.IGNORECASE,
)
_DISCORD_TOKEN_RE = re.compile(
    r"(?:\bmfa\.[A-Za-z0-9_-]{20,}\b|\b[A-Za-z0-9_-]{20,}\.[A-Za-z0-9_-]{6}\."
    r"[A-Za-z0-9_-]{20,}\b)"
)
_SLACK_WEBHOOK_RE = re.compile(
    r"https?://hooks\.slack(?:-gov)?\.com/services/[A-Za-z0-9/_-]+", re.IGNORECASE
)
_GITHUB_TOKEN_RE = re.compile(
    r"(?:\bgh[pousr]_[A-Za-z0-9]{20,}\b|\bgithub_pat_[A-Za-z0-9_]{20,}\b)",
    re.IGNORECASE,
)
_BEARER_TOKEN_RE = re.compile(
    r"\bBearer[ \t]+[A-Za-z0-9._~+/=-]{8,}(?=$|[^A-Za-z0-9._~+/=-])", re.IGNORECASE
)
_SHELL_NAMES = {"bash", "dash", "fish", "ksh", "sh", "zsh"}
_PYTHON_RE = re.compile(r"^(?:python|python\d+(?:\.\d+)?)$")
_INLINE_FLAGS = {
    "perl": {"-e"},
    "ruby": {"-e"},
    "node": {"-e", "--eval"},
    "php": {"-r", "--run"},
    "lua": {"-e"},
    "lua5.1": {"-e"},
    "lua5.2": {"-e"},
    "lua5.3": {"-e"},
    "lua5.4": {"-e"},
    "r": {"-e", "--expression"},
    "rscript": {"-e", "--expression"},
    "pwsh": {"-command", "-c"},
    "powershell": {"-command", "-c"},
}
_AWK_NAMES = {"awk", "gawk", "mawk", "nawk"}


class DocumentLoadError(ValueError):
    """Raised for a bounded, non-secret document loading failure."""


class DuplicateKeyError(DocumentLoadError):
    pass


@dataclass(frozen=True)
class LoadedDocument:
    value: Any
    raw_sha256: str


@dataclass(frozen=True)
class ValidationIssue:
    document: str
    code: str
    path: str
    detail: str

    def render(self) -> str:
        return f"{self.document}:{self.path}: [{self.code}] {self.detail}"


def _reject_constant(value: str) -> None:
    raise DocumentLoadError(f"non-finite JSON number {value!r} is forbidden")


def _finite_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed):
        raise DocumentLoadError("non-finite or overflowing JSON number is forbidden")
    return parsed


def _unique_object(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise DuplicateKeyError("duplicate JSON object key is forbidden")
        result[key] = value
    return result


def _walk_limits(value: Any) -> None:
    nodes = 0
    stack: list[tuple[Any, int, str]] = [(value, 1, "$")]
    while stack:
        current, depth, location = stack.pop()
        nodes += 1
        if nodes > MAX_NODES:
            raise DocumentLoadError("JSON node-count limit exceeded")
        if depth > MAX_DEPTH:
            raise DocumentLoadError("JSON nesting-depth limit exceeded")
        if isinstance(current, str):
            if len(current) > MAX_STRING_LENGTH:
                raise DocumentLoadError(f"string-length limit exceeded at {location}")
        elif isinstance(current, list):
            if len(current) > MAX_ARRAY_LENGTH:
                raise DocumentLoadError(f"array-length limit exceeded at {location}")
            stack.extend((item, depth + 1, f"{location}[{index}]")
                         for index, item in enumerate(current))
        elif isinstance(current, dict):
            for key, item in current.items():
                if len(key) > MAX_KEY_LENGTH:
                    raise DocumentLoadError(f"object-key length limit exceeded at {location}")
                stack.append((item, depth + 1, f"{location}.*"))


def _secret_material_detail(value: str) -> str | None:
    if _DISCORD_WEBHOOK_RE.search(value):
        return "Discord webhook material is forbidden"
    if _DISCORD_TOKEN_RE.search(value):
        return "Discord token-like material is forbidden"
    if _SLACK_WEBHOOK_RE.search(value):
        return "Slack webhook material is forbidden"
    if _GITHUB_TOKEN_RE.search(value):
        return "GitHub token-like material is forbidden"
    if _BEARER_TOKEN_RE.search(value):
        return "bearer token-like material is forbidden"
    return None


def _normalized_key(value: str) -> str:
    with_word_boundaries = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", value)
    return re.sub(r"[^a-z0-9]+", "_", with_word_boundaries.casefold()).strip("_")


def _secret_issue(value: Any) -> tuple[str, str] | None:
    stack: list[tuple[Any, str]] = [(value, "$")]
    while stack:
        current, location = stack.pop()
        if isinstance(current, dict):
            for key, item in current.items():
                child = f"{location}.{key}"
                normalized_key = _normalized_key(key)
                if (normalized_key in _EXACT_SECRET_KEYS or _SECRET_KEY_RE.search(normalized_key)
                        or _SECRET_KEY_EDGE_RE.search(normalized_key)):
                    return location, "secret-like key name is forbidden"
                key_material = _secret_material_detail(key)
                if key_material:
                    return location, key_material
                stack.append((item, child))
        elif isinstance(current, list):
            stack.extend((item, f"{location}[{index}]")
                         for index, item in enumerate(current))
        elif isinstance(current, str):
            detail = _secret_material_detail(current)
            if detail:
                return location, detail
    return None


def load_json_document_with_hash(path: Path) -> LoadedDocument:
    """Parse and hash the same bounded regular-file bytes without following a final symlink."""
    flags = os.O_RDONLY
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    if hasattr(os, "O_NONBLOCK"):
        flags |= os.O_NONBLOCK
    try:
        fd = os.open(path, flags)
    except OSError as exc:
        raise DocumentLoadError(f"cannot open document as a regular file ({exc.strerror})") from None
    try:
        metadata = os.fstat(fd)
        if not stat.S_ISREG(metadata.st_mode):
            raise DocumentLoadError("document is not a regular file")
        if metadata.st_size > MAX_FILE_BYTES:
            raise DocumentLoadError("document exceeds the byte-size limit")
        chunks: list[bytes] = []
        remaining = MAX_FILE_BYTES + 1
        while remaining:
            chunk = os.read(fd, min(remaining, 64 * 1024))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        raw = b"".join(chunks)
        if len(raw) > MAX_FILE_BYTES or (remaining == 0 and os.read(fd, 1)):
            raise DocumentLoadError("document exceeds the byte-size limit")
    finally:
        os.close(fd)
    try:
        text = raw.decode("utf-8", errors="strict")
    except UnicodeDecodeError:
        raise DocumentLoadError("document is not strict UTF-8") from None
    try:
        value = json.loads(
            text,
            object_pairs_hook=_unique_object,
            parse_constant=_reject_constant,
            parse_float=_finite_float,
        )
    except DuplicateKeyError:
        raise
    except DocumentLoadError:
        raise
    except (json.JSONDecodeError, RecursionError):
        raise DocumentLoadError("document is not exactly one complete JSON value") from None
    _walk_limits(value)
    secret = _secret_issue(value)
    if secret:
        raise DocumentLoadError(f"{secret[1]} at {secret[0]}")
    return LoadedDocument(value=value, raw_sha256=hashlib.sha256(raw).hexdigest())


def load_json_document(path: Path) -> Any:
    """Compatibility loader for callers that do not require raw-byte bindings."""
    return load_json_document_with_hash(path).value


def load_schemas(schema_dir: Path = SCHEMA_DIR) -> dict[str, Mapping[str, Any]]:
    if jsonschema is None:
        raise RuntimeError("jsonschema >=4.10,<5 is required")
    result: dict[str, Mapping[str, Any]] = {}
    for name, filename in SCHEMA_FILES.items():
        schema = load_json_document(schema_dir / filename)
        if not isinstance(schema, dict):
            raise RuntimeError(f"schema {filename} is not an object")
        Draft202012Validator.check_schema(schema)
        for node in _iter_values(schema):
            if isinstance(node, dict):
                for keyword in ("$ref", "$dynamicRef"):
                    if keyword not in node:
                        continue
                    reference = node[keyword]
                    if not isinstance(reference, str) or not reference.startswith("#/"):
                        raise RuntimeError(f"schema {filename} contains a non-local {keyword}")
        result[name] = schema
    return result


def _iter_values(root: Any) -> Iterable[Any]:
    stack = [root]
    while stack:
        value = stack.pop()
        yield value
        if isinstance(value, dict):
            stack.extend(value.values())
        elif isinstance(value, list):
            stack.extend(value)


def _json_path(parts: Iterable[Any]) -> str:
    result = "$"
    for part in parts:
        result += f"[{part}]" if isinstance(part, int) else f".{part}"
    return result


def _schema_issues(name: str, document: Any, schema: Mapping[str, Any]) -> list[ValidationIssue]:
    validator = Draft202012Validator(schema, format_checker=FormatChecker())
    errors = sorted(
        validator.iter_errors(document),
        key=lambda error: tuple(str(part) for part in error.absolute_path),
    )
    return [
        ValidationIssue(
            name,
            f"schema.{error.validator}",
            _json_path(error.absolute_path),
            f"document does not satisfy the {error.validator!r} constraint",
        )
        for error in errors[:MAX_ERRORS]
    ]


def _issue(document: str, code: str, path: str, detail: str) -> ValidationIssue:
    return ValidationIssue(document, code, path, detail)


def _duplicates(values: Iterable[Any]) -> set[Any]:
    seen: set[Any] = set()
    duplicates: set[Any] = set()
    for value in values:
        if value in seen:
            duplicates.add(value)
        seen.add(value)
    return duplicates


def _parse_datetime(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError("timezone is required")
    return parsed.astimezone(timezone.utc)


def _has_parent_reference(value: str) -> bool:
    return ".." in PurePosixPath(value).parts


def _path_issue(value: str) -> str | None:
    if not value.startswith("/"):
        return "path must be absolute"
    if _has_parent_reference(value):
        return "path traversal component is forbidden"
    if "\x00" in value or "\n" in value or "\r" in value:
        return "control character is forbidden in a path"
    return None


def _contains_inline_flag(arguments: Sequence[str], flags: set[str]) -> bool:
    for argument in arguments:
        lowered = argument.lower()
        for flag in flags:
            lowered_flag = flag.lower()
            if lowered == lowered_flag or lowered.startswith(f"{lowered_flag}="):
                return True
            if len(lowered_flag) == 2 and lowered_flag.startswith("-") and lowered.startswith(
                    lowered_flag) and len(lowered) > 2:
                return True
    return False


def _awk_has_absolute_script(arguments: Sequence[str]) -> bool:
    for index, argument in enumerate(arguments):
        if argument == "-f" and index + 1 < len(arguments):
            return _path_issue(arguments[index + 1]) is None
        if argument.startswith("--file="):
            return _path_issue(argument.split("=", 1)[1]) is None
    return False


def _known_interpreter_requires_script(name: str) -> bool:
    return name in _SHELL_NAMES or _PYTHON_RE.fullmatch(name) is not None or name in _INLINE_FLAGS


def _has_absolute_positional_script(arguments: Sequence[str]) -> bool:
    positional = next((argument for argument in arguments if not argument.startswith("-")), None)
    return positional is not None and _path_issue(positional) is None


def _semantic_contract(contract: Mapping[str, Any]) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    phases = contract["phases"]
    commands = contract["commands"]
    filesystems = contract["filesystems"]
    phase_ids = [phase["phase_id"] for phase in phases]
    action_ids = [command["action_id"] for command in commands]
    filesystem_paths = [filesystem["path"] for filesystem in filesystems]
    if _duplicates(phase_ids):
        issues.append(_issue("contract", "contract.duplicate_phase", "$.phases",
                             "phase identifiers must be unique"))
    if _duplicates(action_ids):
        issues.append(_issue("contract", "contract.duplicate_action", "$.commands",
                             "action identifiers must be unique"))
    if _duplicates(filesystem_paths):
        issues.append(_issue("contract", "contract.duplicate_filesystem", "$.filesystems",
                             "filesystem paths must be unique"))

    known_actions = set(action_ids)
    for index, phase in enumerate(phases):
        for field in ("action_id", "resume_action_id"):
            if phase[field] not in known_actions:
                issues.append(_issue("contract", "contract.unknown_action",
                                     f"$.phases[{index}].{field}",
                                     "phase references an unregistered action"))
    for index, filesystem in enumerate(filesystems):
        if filesystem["warning_floor_bytes"] < filesystem["hard_floor_bytes"]:
            issues.append(_issue("contract", "contract.storage_floor_order",
                                 f"$.filesystems[{index}].warning_floor_bytes",
                                 "warning floor must be at least the hard-stop floor"))

    corpus = contract["corpus"]
    if corpus["kind"] == "generated" and "generated_corpus_full_gate" not in contract["required_preflight_checks"]:
        issues.append(_issue("contract", "contract.generated_gate_missing",
                             "$.required_preflight_checks",
                             "generated corpus requires the full-corpus preflight gate"))

    repair = contract["repair"]
    if repair["enabled"]:
        envelope = repair["envelope"]
        refs = list(envelope["test_action_ids"])
        refs.extend([envelope["apply_action_id"], envelope["rollback_action_id"], envelope["resume_action_id"]])
        if any(reference not in known_actions for reference in refs):
            issues.append(_issue("contract", "contract.repair_unknown_action", "$.repair.envelope",
                                 "repair envelope references an unregistered action"))

    for path, value in _contract_paths(contract):
        problem = _path_issue(value)
        if problem:
            issues.append(_issue("contract", "contract.unsafe_path", path, problem))

    for index, command in enumerate(commands):
        argv = command["argv"]
        executable = PurePosixPath(argv[0]).name.lower()
        if not argv[0].startswith("/"):
            issues.append(_issue("contract", "contract.relative_executable",
                                 f"$.commands[{index}].argv[0]",
                                 "executable must be an absolute path"))
        if executable == "env":
            issues.append(_issue("contract", "contract.environment_wrapper",
                                 f"$.commands[{index}].argv[0]",
                                 "environment-wrapper commands are forbidden; use the closed environment object"))
        for executable_index, candidate_executable in enumerate(argv):
            candidate_name = PurePosixPath(candidate_executable).name.lower()
            tail = [argument.lower() for argument in argv[executable_index + 1:]]
            if candidate_name in _SHELL_NAMES and any(
                argument == "--command" or argument.startswith("--command=")
                or re.fullmatch(r"-[^-]*c[^-]*", argument)
                for argument in tail
            ):
                issues.append(_issue("contract", "contract.inline_code",
                                     f"$.commands[{index}].argv",
                                     "inline shell code is forbidden"))
                break
            if _PYTHON_RE.fullmatch(candidate_name) and _contains_inline_flag(tail, {"-c"}):
                issues.append(_issue("contract", "contract.inline_code",
                                     f"$.commands[{index}].argv",
                                     "inline Python code is forbidden"))
                break
            if candidate_name in _INLINE_FLAGS and _contains_inline_flag(
                    tail, _INLINE_FLAGS[candidate_name]):
                issues.append(_issue("contract", "contract.inline_code",
                                     f"$.commands[{index}].argv",
                                     "inline interpreter code is forbidden"))
                break
            if (_known_interpreter_requires_script(candidate_name)
                    and not _has_absolute_positional_script(argv[executable_index + 1:])):
                issues.append(_issue("contract", "contract.interpreter_script_required",
                                     f"$.commands[{index}].argv",
                                     "known interpreter actions require an absolute positional script file"))
                break
            if candidate_name in _AWK_NAMES and not _awk_has_absolute_script(argv[executable_index + 1:]):
                issues.append(_issue("contract", "contract.inline_code",
                                     f"$.commands[{index}].argv",
                                     "awk actions require an absolute program file supplied with -f"))
                break
            if candidate_name in _AWK_NAMES and (
                _contains_inline_flag(tail, {"-e"})
                or any(argument == "--source" or argument.startswith("--source=")
                       for argument in tail)
            ):
                issues.append(_issue("contract", "contract.inline_code",
                                     f"$.commands[{index}].argv",
                                     "inline awk source is forbidden"))
                break
        for arg_index, argument in enumerate(argv):
            if "\x00" in argument or "\n" in argument or "\r" in argument:
                issues.append(_issue("contract", "contract.command_control_character",
                                     f"$.commands[{index}].argv[{arg_index}]",
                                     "command arguments cannot contain control characters"))
            candidate = argument.split("=", 1)[1] if argument.startswith("--") and "=" in argument else argument
            if _has_parent_reference(candidate):
                issues.append(_issue("contract", "contract.command_path_traversal",
                                     f"$.commands[{index}].argv[{arg_index}]",
                                     "path traversal in command arguments is forbidden"))
            elif "/" in candidate:
                if not candidate.startswith("/"):
                    issues.append(_issue("contract", "contract.relative_command_path",
                                         f"$.commands[{index}].argv[{arg_index}]",
                                         "path-like command arguments must be absolute"))
    return issues


def _contract_paths(contract: Mapping[str, Any]) -> Iterable[tuple[str, str]]:
    for index, item in enumerate(contract["filesystems"]):
        yield f"$.filesystems[{index}].path", item["path"]
    for index, command in enumerate(contract["commands"]):
        yield f"$.commands[{index}].working_directory", command["working_directory"]
    for index, phase in enumerate(contract["phases"]):
        for output_index, path in enumerate(phase["output_paths"]):
            yield f"$.phases[{index}].output_paths[{output_index}]", path
        for field in ("input_artifacts", "completion_evidence"):
            for artifact_index, artifact in enumerate(phase[field]):
                yield f"$.phases[{index}].{field}[{artifact_index}].path", artifact["path"]
    corpus = contract["corpus"]
    if corpus["kind"] == "non_generated":
        for index, artifact in enumerate(corpus["source_artifacts"]):
            yield f"$.corpus.source_artifacts[{index}].path", artifact["path"]
    else:
        yield "$.corpus.corpus_artifact.path", corpus["corpus_artifact"]["path"]
        yield "$.corpus.full_gate_report.path", corpus["full_gate_report"]["path"]
    if contract["repair"]["enabled"]:
        for index, path in enumerate(contract["repair"]["envelope"]["writable_paths"]):
            yield f"$.repair.envelope.writable_paths[{index}]", path


def _semantic_preflight(
    contract: Mapping[str, Any], preflight: Mapping[str, Any], now: datetime
) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    evidence = preflight["approval_evidence"]
    for field in ("experiment_id", "run_id"):
        if preflight[field] != contract[field] or evidence[field] != contract[field]:
            issues.append(_issue("preflight", "preflight.identity_mismatch", f"$.{field}",
                                 "contract, report, and approval identities must match"))
    if evidence["trusted_channel"] not in contract["approval_requirement"]["trusted_channels"]:
        issues.append(_issue("preflight", "preflight.untrusted_channel",
                             "$.approval_evidence.trusted_channel",
                             "approval channel is not registered by the contract"))

    expected_bindings = dict(contract["bindings"])
    expected_bindings["contract_raw_sha256"] = preflight["contract_raw_sha256"]
    expected_bindings["repair_envelope_sha256"] = (
        contract["repair"]["envelope"]["envelope_sha256"] if contract["repair"]["enabled"] else None
    )
    for field, expected in expected_bindings.items():
        if evidence["bindings"].get(field) != expected:
            issues.append(_issue("preflight", "preflight.approval_binding_mismatch",
                                 f"$.approval_evidence.bindings.{field}",
                                 "approval evidence does not bind the registered contract value"))

    try:
        issued = _parse_datetime(evidence["issued_at"])
        expires = _parse_datetime(evidence["expires_at"])
        created = _parse_datetime(preflight["created_at"])
        if issued >= expires:
            issues.append(_issue("preflight", "preflight.approval_interval",
                                 "$.approval_evidence.expires_at",
                                 "approval expiry must be later than issuance"))
        if issued > now or created > now:
            issues.append(_issue("preflight", "preflight.future_timestamp", "$.created_at",
                                 "approval/report timestamps cannot be in the future"))
        if created < issued or created >= expires:
            issues.append(_issue("preflight", "preflight.approval_report_interval", "$.created_at",
                                 "preflight report must be created inside the approval interval"))
        if expires <= now:
            issues.append(_issue("preflight", "preflight.approval_expired",
                                 "$.approval_evidence.expires_at", "approval evidence is expired"))
    except (TypeError, ValueError):
        pass  # The format checker already reports malformed timestamps.

    expected_checks = set(contract["required_preflight_checks"])
    check_ids = [check["check_id"] for check in preflight["checks"]]
    if set(check_ids) != expected_checks or len(check_ids) != len(expected_checks):
        issues.append(_issue("preflight", "preflight.check_set_mismatch", "$.checks",
                             "preflight checks must exactly cover the contract requirement"))
    for index, check in enumerate(preflight["checks"]):
        if check["verdict"] != "pass":
            issues.append(_issue("preflight", "preflight.check_not_passed",
                                 f"$.checks[{index}].verdict", "required preflight check did not pass"))
        try:
            observed = _parse_datetime(check["observed_at"])
            valid_until = _parse_datetime(check["valid_until"])
            if observed > now:
                issues.append(_issue("preflight", "preflight.future_check",
                                     f"$.checks[{index}].observed_at",
                                     "preflight observation cannot be in the future"))
            if observed > _parse_datetime(preflight["created_at"]):
                issues.append(_issue("preflight", "preflight.check_after_report",
                                     f"$.checks[{index}].observed_at",
                                     "preflight observation cannot postdate its report"))
            if valid_until <= now or valid_until <= observed:
                issues.append(_issue("preflight", "preflight.stale_check",
                                     f"$.checks[{index}].valid_until",
                                     "preflight check is stale or has an invalid interval"))
        except (TypeError, ValueError):
            pass

    expected_storage = {item["path"]: item for item in contract["filesystems"]}
    observed_paths = [item["path"] for item in preflight["storage"]]
    if set(observed_paths) != set(expected_storage) or len(observed_paths) != len(expected_storage):
        issues.append(_issue("preflight", "preflight.storage_set_mismatch", "$.storage",
                             "storage measurements must exactly cover registered filesystems"))
    for index, measurement in enumerate(preflight["storage"]):
        model = expected_storage.get(measurement["path"])
        if model is None:
            continue
        for field in ("hard_floor_bytes", "peak_additional_bytes", "transient_bytes", "recovery_reserve_bytes"):
            if measurement[field] != model[field]:
                issues.append(_issue("preflight", "preflight.storage_model_mismatch",
                                     f"$.storage[{index}].{field}",
                                     "storage measurement does not match the contract model"))
        required = max(
            measurement["hard_floor_bytes"],
            measurement["peak_additional_bytes"] + measurement["transient_bytes"]
            + measurement["recovery_reserve_bytes"],
        )
        expected_verdict = "pass" if measurement["free_bytes"] >= required else "fail"
        if measurement["verdict"] != expected_verdict:
            issues.append(_issue("preflight", "preflight.storage_verdict_mismatch",
                                 f"$.storage[{index}].verdict",
                                 "storage verdict does not match the modeled byte threshold"))
        if expected_verdict != "pass":
            issues.append(_issue("preflight", "preflight.insufficient_storage",
                                 f"$.storage[{index}].free_bytes",
                                 "free bytes are below the required storage threshold"))
        try:
            measured = _parse_datetime(measurement["measured_at"])
            if measured > now or measured > _parse_datetime(preflight["created_at"]):
                issues.append(_issue("preflight", "preflight.storage_timestamp",
                                     f"$.storage[{index}].measured_at",
                                     "storage measurement timestamp is inconsistent"))
        except (TypeError, ValueError):
            pass

    if preflight["derived_verdict"] != "pass":
        issues.append(_issue("preflight", "preflight.derived_not_passed", "$.derived_verdict",
                             "derived preflight verdict must pass for a valid bundle"))
    return issues


def _semantic_ledger(ledger: Mapping[str, Any]) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    events = ledger["events"]
    event_ids = [event["event_id"] for event in events]
    idempotency_keys = [event["idempotency_key"] for event in events]
    if _duplicates(event_ids):
        issues.append(_issue("ledger", "ledger.duplicate_event_id", "$.events",
                             "event identifiers must be unique"))
    if _duplicates(idempotency_keys):
        issues.append(_issue("ledger", "ledger.duplicate_idempotency_key", "$.events",
                             "idempotency keys must be unique"))
    if events[0]["event_kind"] != "contract_registered":
        issues.append(_issue("ledger", "ledger.first_event", "$.events[0].event_kind",
                             "the first event must register the contract"))

    known_events: dict[str, Mapping[str, Any]] = {}
    seen_preflight = False
    seen_resources = False
    seen_start = False
    terminal_index: int | None = None
    delivered_notifications: set[str] = set()
    prior_time: datetime | None = None
    for index, event in enumerate(events):
        expected_sequence = index + 1
        if event["sequence"] != expected_sequence:
            issues.append(_issue("ledger", "ledger.sequence_gap", f"$.events[{index}].sequence",
                                 "event sequence must be contiguous and start at one"))
        expected_previous = None if index == 0 else events[index - 1]["event_sha256"]
        if event["previous_event_sha256"] != expected_previous:
            issues.append(_issue("ledger", "ledger.hash_chain", f"$.events[{index}].previous_event_sha256",
                                 "previous-event hash does not match the preceding event"))
        relation = event["relates_to_event_id"]
        if relation is not None and relation not in known_events:
            issues.append(_issue("ledger", "ledger.forward_or_unknown_relation",
                                 f"$.events[{index}].relates_to_event_id",
                                 "event relationships must reference an earlier event"))
        try:
            occurred = _parse_datetime(event["occurred_at"])
            if prior_time is not None and occurred < prior_time:
                issues.append(_issue("ledger", "ledger.time_order", f"$.events[{index}].occurred_at",
                                     "event timestamps must be nondecreasing"))
            prior_time = occurred
        except (TypeError, ValueError):
            pass

        kind = event["event_kind"]
        if kind == "preflight_passed":
            seen_preflight = True
            if event["verdict"] != "pass":
                issues.append(_issue("ledger", "ledger.preflight_verdict", f"$.events[{index}].verdict",
                                     "preflight_passed must have a pass verdict"))
        elif kind == "resources_acquired":
            if not seen_preflight:
                issues.append(_issue("ledger", "ledger.resources_before_preflight", f"$.events[{index}]",
                                     "resources cannot be acquired before preflight passes"))
            seen_resources = True
        elif kind == "experiment_started":
            if not (seen_preflight and seen_resources):
                issues.append(_issue("ledger", "ledger.start_order", f"$.events[{index}]",
                                     "experiment start requires passed preflight and owned resources"))
            if seen_start:
                issues.append(_issue("ledger", "ledger.duplicate_start", f"$.events[{index}]",
                                     "experiment may start only once"))
            seen_start = True
        elif kind == "gate_result" and event["verdict"] == "none":
            issues.append(_issue("ledger", "ledger.gate_verdict", f"$.events[{index}].verdict",
                                 "gate result requires pass, fail, or invalid"))
        elif kind == "notification_delivery":
            if relation is None or event["notification_status"] == "not_applicable":
                issues.append(_issue("ledger", "ledger.notification_record",
                                     f"$.events[{index}]",
                                     "notification delivery requires an earlier related event and delivery status"))
            if event["notification_status"] == "delivered" and relation is not None:
                delivered_notifications.add(relation)
        elif kind == "promotion_started":
            related = known_events.get(relation) if relation is not None else None
            if related is None or related["event_kind"] != "gate_result" or related["verdict"] != "pass":
                issues.append(_issue("ledger", "ledger.promotion_gate", f"$.events[{index}]",
                                     "promotion must reference an earlier passing gate result"))
            if relation not in delivered_notifications:
                issues.append(_issue("ledger", "ledger.promotion_notification_order", f"$.events[{index}]",
                                     "promotion requires delivered notification for its gate result"))

        if kind in {"experiment_completed", "experiment_failed", "experiment_interrupted"}:
            if not seen_start:
                issues.append(_issue("ledger", "ledger.terminal_before_start", f"$.events[{index}]",
                                     "terminal event requires a prior experiment start"))
            if terminal_index is not None:
                issues.append(_issue("ledger", "ledger.multiple_terminal", f"$.events[{index}]",
                                     "only one terminal lifecycle event is allowed"))
            terminal_index = index
        elif terminal_index is not None and kind != "notification_delivery":
            issues.append(_issue("ledger", "ledger.event_after_terminal", f"$.events[{index}]",
                                 "only terminal-notification delivery may follow a terminal event"))
        known_events[event["event_id"]] = event
    return issues


def _semantic_queue(
    contract: Mapping[str, Any], preflight: Mapping[str, Any], ledger: Mapping[str, Any],
    queue: Mapping[str, Any]
) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    entries = queue["entries"]
    entry_ids = [entry["entry_id"] for entry in entries]
    positions = [entry["position"] for entry in entries]
    if _duplicates(entry_ids):
        issues.append(_issue("queue", "queue.duplicate_entry", "$.entries",
                             "queue entry identifiers must be unique"))
    if _duplicates(positions) or sorted(positions) != list(range(1, len(entries) + 1)):
        issues.append(_issue("queue", "queue.position_sequence", "$.entries",
                             "queue positions must be unique and contiguous from one"))
    known = set(entry_ids)
    graph: dict[str, list[str]] = {}
    active_resources: set[tuple[str, str]] = set()
    for index, entry in enumerate(entries):
        dependencies = entry["dependencies"]
        graph[entry["entry_id"]] = dependencies
        if any(dependency not in known for dependency in dependencies):
            issues.append(_issue("queue", "queue.unknown_dependency",
                                 f"$.entries[{index}].dependencies",
                                 "queue dependency does not exist"))
        if entry["entry_id"] in dependencies:
            issues.append(_issue("queue", "queue.self_dependency",
                                 f"$.entries[{index}].dependencies",
                                 "queue entry cannot depend on itself"))
        terminal = entry["status"] in {"completed", "failed", "interrupted"}
        if terminal and entry["terminal_notification_status"] != "delivered":
            issues.append(_issue("queue", "queue.terminal_notification",
                                 f"$.entries[{index}].terminal_notification_status",
                                 "terminal queue state requires delivered notification"))
        if not terminal and entry["terminal_notification_status"] != "not_applicable":
            issues.append(_issue("queue", "queue.nonterminal_notification",
                                 f"$.entries[{index}].terminal_notification_status",
                                 "nonterminal queue state cannot claim a terminal notification"))
        if entry["status"] == "active":
            resource = entry["assigned_resource"]
            if resource is None:
                issues.append(_issue("queue", "queue.active_without_resource",
                                     f"$.entries[{index}].assigned_resource",
                                     "active queue entry must own a resource"))
            else:
                identity = (resource["resource_type"], resource["resource_id"])
                if identity in active_resources:
                    issues.append(_issue("queue", "queue.resource_conflict",
                                         f"$.entries[{index}].assigned_resource",
                                         "active entries cannot share one resource"))
                active_resources.add(identity)

    indegree = {node: 0 for node in graph}
    dependents: dict[str, list[str]] = {node: [] for node in graph}
    for node, dependencies in graph.items():
        for dependency in dependencies:
            if dependency in graph:
                indegree[node] += 1
                dependents[dependency].append(node)
    ready = [node for node, degree in indegree.items() if degree == 0]
    processed = 0
    while ready:
        node = ready.pop()
        processed += 1
        for dependent in dependents[node]:
            indegree[dependent] -= 1
            if indegree[dependent] == 0:
                ready.append(dependent)
    if processed != len(graph):
        issues.append(_issue("queue", "queue.dependency_cycle", "$.entries",
                             "queue dependency graph must be acyclic"))

    matching = [entry for entry in entries
                if entry["experiment_id"] == contract["experiment_id"]
                and entry["run_id"] == contract["run_id"]]
    if len(matching) != 1:
        issues.append(_issue("queue", "queue.bundle_entry", "$.entries",
                             "exactly one queue entry must bind the supplied run"))
    else:
        matched_entry = matching[0]
        bindings = matched_entry["bindings"]
        comparisons = {
            "contract_raw_sha256": preflight["contract_raw_sha256"],
            "preflight_report_raw_sha256": ledger["bindings"]["preflight_report_raw_sha256"],
            "schema_bundle_sha256": contract["bindings"]["schema_bundle_sha256"],
        }
        for field, expected in comparisons.items():
            if bindings[field] != expected:
                issues.append(_issue("queue", "queue.binding_mismatch",
                                     f"$.entries.bindings.{field}",
                                     "queue binding does not match the supplied document bundle"))
        terminal_kind = {
            "completed": "experiment_completed",
            "failed": "experiment_failed",
            "interrupted": "experiment_interrupted",
        }.get(matched_entry["status"])
        if terminal_kind is not None:
            terminal_events = [event for event in ledger["events"] if event["event_kind"] == terminal_kind]
            delivered_for_terminal = {
                event["relates_to_event_id"] for event in ledger["events"]
                if event["event_kind"] == "notification_delivery"
                and event["notification_status"] == "delivered"
            }
            if len(terminal_events) != 1:
                issues.append(_issue("queue", "queue.ledger_terminal_mismatch", "$.entries.status",
                                     "queue terminal status must match exactly one ledger terminal event"))
            elif terminal_events[0]["event_id"] not in delivered_for_terminal:
                issues.append(_issue("queue", "queue.ledger_terminal_notification", "$.entries.status",
                                     "ledger must record delivered notification for the terminal event"))
    return issues


def _raw_binding_issues(
    contract: Mapping[str, Any],
    preflight: Mapping[str, Any],
    ledger: Mapping[str, Any],
    queue: Mapping[str, Any],
    raw_hashes: Mapping[str, str],
) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    required_names = {"contract", "preflight", "ledger", "queue"}
    if not required_names.issubset(raw_hashes):
        return [_issue("bundle", "raw.hashes_required", "$",
                       "launch-ready validation requires hashes from the same bytes parsed for all documents")]

    contract_hash = raw_hashes["contract"]
    for document, path, declared in (
        ("preflight", "$.contract_raw_sha256", preflight["contract_raw_sha256"]),
        ("preflight", "$.approval_evidence.bindings.contract_raw_sha256",
         preflight["approval_evidence"]["bindings"]["contract_raw_sha256"]),
        ("ledger", "$.bindings.contract_raw_sha256", ledger["bindings"]["contract_raw_sha256"]),
    ):
        if declared != contract_hash:
            issues.append(_issue(document, "raw.contract_binding", path,
                                 "declared contract hash does not match the exact parsed file bytes"))

    if ledger["bindings"]["preflight_report_raw_sha256"] != raw_hashes["preflight"]:
        issues.append(_issue("ledger", "raw.preflight_binding",
                             "$.bindings.preflight_report_raw_sha256",
                             "declared preflight hash does not match the exact parsed file bytes"))

    matching = [
        (index, entry) for index, entry in enumerate(queue["entries"])
        if entry["experiment_id"] == contract["experiment_id"]
        and entry["run_id"] == contract["run_id"]
    ]
    if len(matching) == 1:
        index, entry = matching[0]
        expected = {
            "contract_raw_sha256": (raw_hashes["contract"], "raw.contract_binding"),
            "preflight_report_raw_sha256": (raw_hashes["preflight"], "raw.preflight_binding"),
            "ledger_raw_sha256": (raw_hashes["ledger"], "raw.ledger_binding"),
        }
        for field, (actual_hash, code) in expected.items():
            if entry["bindings"][field] != actual_hash:
                issues.append(_issue(
                    "queue", code, f"$.entries[{index}].bindings.{field}",
                    "declared raw document hash does not match the exact parsed file bytes",
                ))
    return issues


def validate_documents(
    contract: Mapping[str, Any],
    preflight: Mapping[str, Any],
    ledger: Mapping[str, Any],
    queue: Mapping[str, Any],
    *,
    now: datetime | None = None,
    schemas: Mapping[str, Mapping[str, Any]] | None = None,
    raw_hashes: Mapping[str, str] | None = None,
    require_raw_hashes: bool = False,
) -> list[ValidationIssue]:
    """Validate documents without implying authorization.

    In-memory validation does not verify raw-file bindings by default. Launch-ready
    callers must set ``require_raw_hashes`` and supply hashes obtained from the exact
    parsed bytes; ``validate_paths`` is the supported filesystem entry point.
    """
    if jsonschema is None:
        raise RuntimeError("jsonschema >=4.10,<5 is required")
    schemas = schemas or load_schemas()
    documents = {"contract": contract, "preflight": preflight, "ledger": ledger, "queue": queue}
    issues: list[ValidationIssue] = []
    for name, document in documents.items():
        issues.extend(_schema_issues(name, document, schemas[name]))
    if issues:
        return issues[:MAX_ERRORS]

    now = now or datetime.now(timezone.utc)
    if now.tzinfo is None:
        raise ValueError("validation time must be timezone-aware")
    now = now.astimezone(timezone.utc)

    for name, document in documents.items():
        if document["schema_bundle_id"] != BUNDLE_ID:
            issues.append(_issue(name, "bundle.identifier", "$.schema_bundle_id",
                                 "document uses an incompatible schema bundle"))
        for field in ("experiment_id", "run_id"):
            if name != "queue" and document[field] != contract[field]:
                issues.append(_issue(name, "bundle.identity", f"$.{field}",
                                     "document identity does not match the contract"))
    if ledger["bindings"]["contract_raw_sha256"] != preflight["contract_raw_sha256"]:
        issues.append(_issue("ledger", "bundle.contract_hash", "$.bindings.contract_raw_sha256",
                             "ledger and preflight contract hash bindings differ"))
    if ledger["bindings"]["schema_bundle_sha256"] != contract["bindings"]["schema_bundle_sha256"]:
        issues.append(_issue("ledger", "bundle.schema_hash", "$.bindings.schema_bundle_sha256",
                             "ledger and contract schema hash bindings differ"))
    if require_raw_hashes and raw_hashes is None:
        issues.append(_issue("bundle", "raw.hashes_required", "$",
                             "launch-ready validation requires exact parsed-byte hashes"))
    elif raw_hashes is not None:
        issues.extend(_raw_binding_issues(contract, preflight, ledger, queue, raw_hashes))

    issues.extend(_semantic_contract(contract))
    issues.extend(_semantic_preflight(contract, preflight, now))
    issues.extend(_semantic_ledger(ledger))
    issues.extend(_semantic_queue(contract, preflight, ledger, queue))
    return issues[:MAX_ERRORS]


def validate_paths(
    contract_path: Path, preflight_path: Path, ledger_path: Path, queue_path: Path,
    *, now: datetime | None = None
) -> list[ValidationIssue]:
    documents: dict[str, Any] = {}
    raw_hashes: dict[str, str] = {}
    for name, path in {
        "contract": contract_path,
        "preflight": preflight_path,
        "ledger": ledger_path,
        "queue": queue_path,
    }.items():
        try:
            loaded = load_json_document_with_hash(path)
            documents[name] = loaded.value
            raw_hashes[name] = loaded.raw_sha256
        except DocumentLoadError as exc:
            return [_issue(name, "loader.rejected", "$", str(exc))]
    return validate_documents(
        documents["contract"], documents["preflight"], documents["ledger"], documents["queue"],
        now=now, raw_hashes=raw_hashes, require_raw_hashes=True,
    )


def _argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", required=True, type=Path)
    parser.add_argument("--preflight", required=True, type=Path)
    parser.add_argument("--ledger", required=True, type=Path)
    parser.add_argument("--queue", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    print(NOTICE)
    print(RUNTIME_BOUNDARY)
    if jsonschema is None:
        print("ERROR: jsonschema >=4.10,<5 is required", file=sys.stderr)
        return 2
    arguments = _argument_parser().parse_args(argv)
    try:
        issues = validate_paths(arguments.contract, arguments.preflight, arguments.ledger, arguments.queue)
    except (RuntimeError, OSError) as exc:
        print(f"ERROR: validator setup failed: {exc}", file=sys.stderr)
        return 2
    if issues:
        for issue in issues:
            print(issue.render(), file=sys.stderr)
        if len(issues) >= MAX_ERRORS:
            print(f"ERROR: stopped after {MAX_ERRORS} bounded errors", file=sys.stderr)
        return 1
    print("All four documents satisfy the offline structural and consistency checks.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
