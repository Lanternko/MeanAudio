#!/usr/bin/env python3
"""Validate slot3 CFG0 registration readiness and protected runtime state."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import install_slot3_cfg0_interceptor as installer
import register_slot3_fair013_cfg0 as reg


FOUR_CELL_CONTRACT = reg.ROOT / "docs/experiments/caption2p0_quarter_cfg0_rerun_contract.json"
FOUR_CELL_PROPOSAL = reg.ROOT / "docs/experiments/caption2p0_quarter_cfg0_queue_entry.pending.json"


def validate_priority_order(order: dict[str, Any], template: dict[str, Any]) -> None:
    if order.get("mutation_authorized") is not False:
        raise reg.RegistrationError("priority proposal must not authorize live queue mutation")
    items = order.get("durable_evaluation_order")
    if not isinstance(items, list) or len(items) != 4:
        raise reg.RegistrationError("priority proposal requires one four-cell bundle plus three arms")
    first = items[0]
    if first.get("kind") != "existing_four_cell_bundle" or first.get("proposal") != str(FOUR_CELL_PROPOSAL):
        raise reg.RegistrationError("four-cell bundle is not first")
    if first.get("proposal_sha256") != reg.sha256_path(FOUR_CELL_PROPOSAL):
        raise reg.RegistrationError("four-cell queue proposal hash drift")
    contract = reg.load_json(FOUR_CELL_CONTRACT)
    expected_cells = contract.get("execution", {}).get("order")
    if first.get("cells") != expected_cells or len(expected_cells or []) != 4:
        raise reg.RegistrationError("four-cell order binding drift")
    arm_ids = [item.get("arm_id") for item in items[1:]]
    if arm_ids != [arm["arm_id"] for arm in template["arms"]]:
        raise reg.RegistrationError("three fair013 arm order drift")
    for item in items[1:]:
        protocol = item.get("protocol", "")
        if "MeanFlow 25" not in protocol or "CFG 0" not in protocol or "CFG 4.5" in protocol:
            raise reg.RegistrationError("priority arm is not CFG0/MF25")
    if order.get("preserved_parent_remainder") != [
        "after K3 registration: train fair013 best-of-3",
        "after best registration: train fair013 worst-of-3",
        "after worst registration: restore final slot0 cache",
        "emit FAIR013_CHAIN_DONE",
    ]:
        raise reg.RegistrationError("parent remainder is not preserved exactly")


def validate_manifest(manifest: dict[str, Any], template: dict[str, Any]) -> None:
    generated = reg.runtime_manifest_payload()
    if manifest != generated:
        raise reg.RegistrationError("transitive runtime manifest drift")
    source_paths = {entry["path"] for entry in manifest.get("sources", [])}
    required = {str(path) for path in (
        reg.HOOK, reg.COORDINATOR, reg.VALIDATOR, reg.INSTALLER, reg.NOTIFIER,
        reg.PYTHON, reg.BASH, reg.ENV, reg.TEMPLATE, reg.ORDER, reg.PLAN, reg.SECURITY_REVIEW,
    )}
    if source_paths != required:
        raise reg.RegistrationError("runtime manifest source closure drift")
    if manifest.get("fixed_child_environment") != template.get("fixed_child_environment"):
        raise reg.RegistrationError("fixed child environment drift")
    if manifest.get("hmac_domains_hex") != template.get("hmac", {}).get("domains_hex"):
        raise reg.RegistrationError("HMAC domain drift")
    if manifest.get("allowed_writes") != [str(reg.RUNTIME), str(reg.LIVE_EVALUATOR)]:
        raise reg.RegistrationError("runtime allowed-write scope drift")
    forbidden = manifest.get("forbidden_execution_paths", [])
    if str(reg.ROOT / "eval.py") not in forbidden or str(reg.LIVE_EVALUATOR) not in forbidden:
        raise reg.RegistrationError("forbidden execution paths incomplete")


def validate_pending_readiness() -> dict[str, Any]:
    template = reg.load_json(reg.TEMPLATE)
    reg.validate_template(template)
    reg.validate_security_receipt(reg.load_json(reg.SECURITY_REVIEW), reg.sha256_path(reg.PLAN))
    reg.validate_runtime_ancestry(reg.RUNTIME, reg.RUNTIME_TRUST_ANCHOR)
    if reg.sha256_path(Path(template["scientific_protocol"]["tsv"])) != template["scientific_protocol"]["tsv_sha256"]:
        raise reg.RegistrationError("MusicCaps TSV hash drift")
    order = reg.load_json(reg.ORDER)
    validate_priority_order(order, template)
    manifest = reg.load_json(reg.MANIFEST)
    validate_manifest(manifest, template)
    installer.validate_descriptor(require_approved=False)
    if reg.load_json(reg.APPROVAL).get("approval_status") != "pending":
        raise reg.RegistrationError("activation approval must remain pending during readiness")
    return {
        "status": "REGISTRATION_READINESS_OK_ACTIVATION_PENDING",
        "arms": [arm["arm_id"] for arm in template["arms"]],
        "runtime_manifest_sha256": reg.sha256_path(reg.MANIFEST),
        "replacement_hook_sha256": reg.sha256_path(reg.HOOK),
        "gpu_minutes": 0,
        "live_queue_mutation": False,
    }


def validate_runtime(root: Path, *, require_complete: bool) -> dict[str, Any]:
    template = reg.load_json(reg.TEMPLATE)
    reg.validate_template(template)
    reg.validate_security_receipt(reg.load_json(reg.SECURITY_REVIEW), reg.sha256_path(reg.PLAN))
    paths = reg.StorePaths.under(root)
    reg.validate_runtime_ancestry(root, paths.trust_anchor)
    reg._check_owned_mode(root, 0o700, directory=True)
    key = reg.create_or_load_key(paths.key)
    manifest_sha256 = reg.sha256_path(reg.MANIFEST)
    state = reg.load_state(
        root, key, reg.sha256_bytes(reg.canonical(template)), manifest_sha256,
        reg.sha256_path(reg.APPROVAL),
    )
    unsigned = reg.verify_signed(state, key, reg.STATE_DOMAIN)
    sequence = unsigned["sequence"]
    if require_complete and sequence != 3:
        raise reg.RegistrationError("registration runtime is incomplete")
    for index, accepted in enumerate(unsigned["accepted_arms"]):
        arm = template["arms"][index]
        if accepted["arm_id"] != arm["arm_id"]:
            raise reg.RegistrationError("runtime arm order drift")
        contract_path = root / arm["resolved_contract"]
        contract = reg.load_json(contract_path)
        contract_unsigned = reg.verify_signed(contract, key, reg.CONTRACT_DOMAIN)
        if contract_unsigned.get("canonical_label") != arm["canonical_label"]:
            raise reg.RegistrationError("resolved canonical label drift")
        protocol = contract_unsigned.get("protocol", {})
        if protocol.get("cfg_strength") != 0 or protocol.get("num_steps") != 25:
            raise reg.RegistrationError("resolved protocol is not CFG0/MF25")
        if accepted["resolved_contract_sha256"] != reg.sha256_bytes(reg.canonical(contract)):
            raise reg.RegistrationError("resolved contract hash drift")
        event_path = paths.outbox / f"{arm['sequence']:02d}_{arm['arm_id']}.json"
        event = reg.load_json(event_path)
        event_unsigned = reg.verify_signed(event, key, reg.NOTIFICATION_DOMAIN)
        if event_unsigned.get("delivery_status") != "delivered" or event_unsigned.get("arm_id") != arm["arm_id"]:
            raise reg.RegistrationError("required hold event is not delivered")
    if reg.runtime_bytes(root) > template["interceptor"]["maximum_runtime_bytes"]:
        raise reg.RegistrationError("runtime exceeds 16 MiB")
    return {"status": "RUNTIME_OK", "registered_arms": sequence, "complete": sequence == 3}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--readiness", action="store_true")
    parser.add_argument("--runtime-root", type=Path)
    parser.add_argument("--require-complete", action="store_true")
    args = parser.parse_args()
    if args.runtime_root is not None:
        result = validate_runtime(args.runtime_root, require_complete=args.require_complete)
    else:
        result = validate_pending_readiness()
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, reg.RegistrationError, json.JSONDecodeError) as exc:
        print(f"SLOT3_CFG0_VALIDATION_HOLD {type(exc).__name__}: {exc}", file=sys.stderr)
        raise SystemExit(125)
