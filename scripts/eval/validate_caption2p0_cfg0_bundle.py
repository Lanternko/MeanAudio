#!/usr/bin/env python3
"""Validate all four terminal reports for the Caption 2.0 quarter CFG0 rerun."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from validate_caption2p0_cfg0_report import validate


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", type=Path, required=True)
    args = parser.parse_args()
    payload = json.loads(args.contract.read_text())
    expected = payload["execution"]["order"]
    cells = {cell["cell_id"]: cell for cell in payload["cells"]}
    if set(cells) != set(expected) or len(expected) != 4:
        raise SystemExit("invalid four-cell order")
    for arm in expected:
        validate(args.contract, arm, Path(cells[arm]["report"]))
    print("CFG0_BUNDLE_OK cells=4")


if __name__ == "__main__":
    main()
