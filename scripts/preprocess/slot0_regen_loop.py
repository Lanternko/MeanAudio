#!/usr/bin/env python3
"""Regenerate every locally flagged slot0 row until the local LLM accepts it.

Operator flow (2026-09-15): local LLM flags rows -> regenerate with the original
captioner + prompt -> local LLM re-audits -> repeat for rows still failing ->
small Luna spot check on the result (slot0_audit_crosscheck.py).

Per attempt k (resumable, every artifact written atomically):
  attempt_k_ids.txt         ids still pending
  attempt_k_candidates.jsonl  one candidate per id (dac venv, Qwen2.5-Omni-3B)
  attempt_k_to_audit.jsonl  structurally valid candidates
  attempt_k_audit.json      local LLM decisions (vllm venv); only KEEP is accepted
A row still failing after --max-attempts is listed in unresolved.tsv and left OUT
of the assembled TSV (never silently kept or dropped). The original source TSV
is never modified; the assembled TSV is a candidate, not a released corpus.
"""
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/preprocess"))
import slot0_contamination_a as audit  # noqa: E402  (digest)
from slot0_audit_crosscheck import load_local  # noqa: E402

DAC_PY = Path.home() / "venvs/dac/bin/python"
VLLM_PY = Path.home() / "venvs/vllm/bin/python"


def run(cmd: list) -> None:
    print("[run]", " ".join(map(str, cmd)), flush=True)
    subprocess.run(list(map(str, cmd)), check=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--local-full", type=Path, required=True)
    ap.add_argument("--source", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--max-attempts", type=int, default=6)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    local, quarantined = load_local(args.local_full)
    flagged = sorted([i for i, d in local.items() if d["decision"] != "KEEP"] + [q["id"] for q in quarantined])
    reasons = {i: local[i]["decision"] if i in local else "QUARANTINED" for i in flagged}
    print(json.dumps({"flagged": len(flagged)}), flush=True)

    # Resume = deterministic replay of existing attempt files; accepted.json is output only.
    accepted_path = args.out / "accepted.json"
    accepted: dict[str, dict] = {}
    last_failure: dict[str, dict] = {}
    pending = list(flagged)
    for k in range(1, args.max_attempts + 1):
        if not pending:
            break
        ids_path = args.out / f"attempt_{k}_ids.txt"
        if ids_path.exists():
            if ids_path.read_text().split() != pending:
                raise ValueError(f"resume mismatch at attempt {k}: pending ids differ")
        else:
            ids_path.write_text("\n".join(pending) + "\n")
        cand_path = args.out / f"attempt_{k}_candidates.jsonl"
        if not cand_path.exists():
            run([DAC_PY, ROOT / "scripts/caption10s_pipeline/gen_slot0_regen_candidates.py",
                 "--ids", ids_path, "--attempt", k, "--out_jsonl", cand_path])
        cands = [json.loads(l) for l in cand_path.read_text().splitlines() if l.strip()]
        if sorted(c["id"] for c in cands) != sorted(pending):
            raise ValueError(f"attempt {k}: candidate coverage mismatch")
        valid = [c for c in cands if c["caption"]]
        for c in cands:
            if not c["caption"]:
                last_failure[c["id"]] = {"attempt": k, "why": "structural:" + ",".join(c["structural_defects"]),
                                         "candidate": c["raw"]}
        to_audit = args.out / f"attempt_{k}_to_audit.jsonl"
        audit_path = args.out / f"attempt_{k}_audit.json"
        if not audit_path.exists():
            to_audit.write_text("".join(json.dumps({"id": c["id"], "caption": c["caption"]}, ensure_ascii=False) + "\n"
                                        for c in valid))
            run([VLLM_PY, ROOT / "scripts/preprocess/slot0_audit_local.py", "candidates",
                 "--candidates", to_audit, "--out", audit_path])
        result = json.loads(audit_path.read_text())
        by_id = {c["id"]: c for c in valid}
        for d in result["decisions"]:
            c = by_id[d["id"]]
            if d["decision"] == "KEEP" and d["caption_sha256"] == audit.digest(c["caption"].encode()):
                accepted[d["id"]] = {"caption": c["caption"], "attempt": k, "seed": c["seed"],
                                     "model_revision": c["model_revision"], "local_reason": d["reason"]}
            else:
                last_failure[d["id"]] = {"attempt": k, "why": f"local:{d['decision']}:{d['reason']}",
                                         "candidate": c["caption"]}
        for q in result["quarantined"]:
            last_failure[q["id"]] = {"attempt": k, "why": "local_quarantine", "candidate": by_id[q["id"]]["caption"]}
        accepted_path.write_text(json.dumps(accepted, ensure_ascii=False, indent=1))
        pending = [i for i in flagged if i not in accepted]
        print(json.dumps({"attempt": k, "accepted_total": len(accepted), "still_pending": len(pending)}), flush=True)

    unresolved = [i for i in flagged if i not in accepted]
    with (args.out / "unresolved.tsv").open("w", newline="") as fh:
        w = csv.writer(fh, delimiter="\t")
        w.writerow(["id", "original_local_decision", "last_attempt", "last_failure", "last_candidate"])
        for i in unresolved:
            f = last_failure.get(i, {})
            w.writerow([i, reasons[i], f.get("attempt"), f.get("why"), f.get("candidate")])

    csv.field_size_limit(10**9)
    flagged_set = set(flagged)
    assembled = args.out / "slot0_regen_candidate_corpus.tsv"
    tmp = assembled.with_suffix(".tsv.tmp")
    n_rows = n_changed = 0
    with args.source.open(newline="") as src, tmp.open("w", newline="") as dst, \
            (args.out / "provenance.jsonl").open("w") as prov:
        reader = csv.DictReader(src, delimiter="\t")
        writer = csv.DictWriter(dst, fieldnames=reader.fieldnames, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        for row in reader:
            if row["id"] in flagged_set:
                if row["id"] not in accepted:
                    continue  # listed in unresolved.tsv
                a = accepted[row["id"]]
                prov.write(json.dumps({"id": row["id"], "original_sha256": audit.digest(row["caption"].encode()),
                                       "original_local_decision": reasons[row["id"]], **a},
                                      ensure_ascii=False) + "\n")
                row = {**row, "caption": a["caption"]}
                n_changed += 1
            writer.writerow(row)
            n_rows += 1
    tmp.replace(assembled)
    summary = {"status": "regen_complete_not_released", "source_rows": len(local) + len(quarantined),
               "flagged": len(flagged), "regenerated_accepted": len(accepted), "unresolved": len(unresolved),
               "assembled_rows": n_rows, "changed_rows": n_changed,
               "accepted_by_attempt": {k: sum(1 for a in accepted.values() if a["attempt"] == k)
                                       for k in range(1, args.max_attempts + 1)}}
    (args.out / "summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
