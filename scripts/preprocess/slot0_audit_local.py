#!/usr/bin/env python3
"""Slot0 semantic audit on a local vLLM model (replaces paid Luna API calls).

Reuses PROMPT / SCHEMA / exact-prefix edit safety from slot0_semantic_audit.py
unchanged (that file is hash-pinned by full_v1 and must not be edited).
Never writes a training corpus: outputs are per-row decisions only.

  calibrate  --fixtures calibration_v5.json         (fixture accuracy)
  agree      --luna-db full_v1/state.sqlite          (agreement vs Luna rows)
  full       --source <slot0 TSV>                    (resumable chunked audit)
  candidates --candidates regen.jsonl --out x.json   (audit regenerated captions)

Failure policy: a batch whose output fails validation is re-run one row at a
time; a row that still fails is quarantined with its error. Nothing is dropped.
TRIM_SUFFIX prefixes are re-audited by the SAME model (trim_selfcheck_keep). That is a
consistency check, not independent verification; acceptance needs the Luna cross-check
and human labels (slot0_audit_crosscheck.py).
"""
from __future__ import annotations

import argparse
import collections
import csv
import fcntl
import hashlib
import importlib.util
import json
import sqlite3
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("audit", HERE / "slot0_semantic_audit.py")
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)

DEFAULT_MODEL = str(next(Path.home().glob(
    ".cache/huggingface/hub/models--Qwen--Qwen2.5-7B-Instruct/snapshots/*")))
BATCH = 8
CHUNK_BATCHES = 1000


class Engine:
    def __init__(self, model: str, max_model_len: int, gpu_mem: float):
        from vllm import LLM, SamplingParams
        from vllm.sampling_params import StructuredOutputsParams
        self.model = model
        self.llm = LLM(model=model, max_model_len=max_model_len, gpu_memory_utilization=gpu_mem,
                       enable_prefix_caching=True, seed=0)
        self.params = SamplingParams(temperature=0.0, max_tokens=2048, seed=0,
                                     structured_outputs=StructuredOutputsParams(json=audit.SCHEMA))
        # parse_response() checks the resolved model name against audit.MODEL.
        audit.MODEL = model
        self.usage = collections.Counter()

    def _generate(self, batches: list[list[dict]]) -> list[dict | Exception]:
        convs = [[{"role": "system", "content": audit.PROMPT},
                  {"role": "user", "content": json.dumps(
                      [{"id": r["id"], "caption": r["caption"]} for r in rows], ensure_ascii=False)}]
                 for rows in batches]
        outs = self.llm.chat(convs, self.params, use_tqdm=False)
        results = []
        for rows, out in zip(batches, outs):
            gen = out.outputs[0]
            usage = {"prompt_tokens": len(out.prompt_token_ids), "completion_tokens": len(gen.token_ids)}
            self.usage.update(usage)
            fake = {"id": "local-" + audit.digest(json.dumps(rows, ensure_ascii=False).encode())[:16],
                    "model": self.model, "usage": usage,
                    "choices": [{"finish_reason": gen.finish_reason,
                                 "message": {"content": gen.text, "refusal": None}}]}
            try:
                results.append(audit.parse_response(rows, fake))
            except Exception as error:  # noqa: BLE001 - recorded, never swallowed
                results.append(error)
        return results

    def audit_rows(self, rows: list[dict]) -> tuple[list[dict], list[dict]]:
        """Return (decisions, quarantined) covering every input row exactly once."""
        batches = [rows[i:i + BATCH] for i in range(0, len(rows), BATCH)]
        decisions, retry = [], []
        for batch, res in zip(batches, self._generate(batches)):
            if isinstance(res, Exception):
                retry.extend(batch)
            else:
                decisions.extend(res["decisions"])
        quarantined = []
        if retry:
            for row, res in zip(retry, self._generate([[r] for r in retry])):
                if isinstance(res, Exception):
                    quarantined.append({"id": row["id"], "error": f"{type(res).__name__}: {res}"[:300],
                                        "caption_sha256": audit.digest(row["caption"].encode())})
                else:
                    decisions.extend(res["decisions"])
        # Same-model self-check of trims (not independent; repeats the same blind spots).
        trims = [d for d in decisions if d["decision"] == "TRIM_SUFFIX"]
        if trims:
            checks = self._generate([[{"id": d["id"], "caption": d["retained"]}] for d in trims])
            for d, res in zip(trims, checks):
                ok = not isinstance(res, Exception) and res["decisions"][0]["decision"] == "KEEP"
                d["trim_selfcheck_keep"] = ok
        order = {r["id"]: i for i, r in enumerate(rows)}
        decisions.sort(key=lambda d: order[d["id"]])
        assert len(decisions) + len(quarantined) == len(rows)
        return decisions, quarantined


def binding(args, extra: dict) -> dict:
    return {"implementation_sha256": audit.digest(Path(__file__).read_bytes()),
            "audit_module_sha256": audit.digest((HERE / "slot0_semantic_audit.py").read_bytes()),
            "prompt_sha256": audit.digest(audit.PROMPT.encode()), "model": args.model,
            "batch_size": BATCH, "temperature": 0.0, "max_model_len": args.max_model_len,
            "training_corpus_written": False, **extra}


def cmd_calibrate(args, engine: Engine) -> int:
    rows = json.loads(args.fixtures.read_text())
    decisions, quarantined = engine.audit_rows(rows)
    by_id = {d["id"]: d for d in decisions}
    errors = []
    for row in rows:
        d = by_id.get(row["id"])
        actual = d["decision"] if d else "QUARANTINED"
        if actual not in row["expected_decisions"]:
            errors.append({"id": row["id"], "expected": row["expected_decisions"], "actual": actual,
                           "reason": d["reason"] if d else None})
        if d and "expected_retained" in row and d["retained"] != row["expected_retained"]:
            errors.append({"id": row["id"], "error": "wrong trim boundary"})
    report = {"status": "passed" if not errors else "failed", "rows": len(rows), "errors": errors,
              "quarantined": quarantined, "usage": dict(engine.usage),
              "binding": binding(args, {"source_sha256": audit.digest(args.fixtures.read_bytes())}),
              "meaning": "fixture accuracy only; not a full-corpus quality approval"}
    audit.atomic(args.out / f"calibration_{args.fixtures.stem}.json", report)
    print(json.dumps({"fixtures": args.fixtures.name, "status": report["status"],
                      "errors": len(errors)}), flush=True)
    return int(bool(errors))


def cmd_agree(args, engine: Engine) -> int:
    db = sqlite3.connect(f"file:{args.luna_db}?mode=ro", uri=True)
    luna, rows = {}, []
    for payload, result in db.execute(
            "SELECT payload, result FROM jobs WHERE phase='audit' AND status='done' ORDER BY id"):
        rows.extend(json.loads(payload))
        for d in json.loads(result)["decisions"]:
            luna[d["id"]] = d
    t0 = time.time()
    decisions, quarantined = engine.audit_rows(rows)
    elapsed = time.time() - t0
    confusion = collections.Counter()
    disagreements = []
    for d in decisions:
        ref = luna[d["id"]]
        confusion[(ref["decision"], d["decision"])] += 1
        if ref["decision"] != d["decision"] or ref["retained"] != d["retained"]:
            disagreements.append({"id": d["id"], "luna": ref["decision"], "local": d["decision"],
                                  "luna_reason": ref["reason"], "local_reason": d["reason"],
                                  "same_retained": ref["retained"] == d["retained"]})
    non_keep_ref = [i for i, d in luna.items() if d["decision"] != "KEEP"]
    local = {d["id"]: d for d in decisions}
    caught = sum(1 for i in non_keep_ref if i in local and local[i]["decision"] != "KEEP")
    report = {"rows": len(rows), "quarantined": len(quarantined),
              "exact_decision_agreement": sum(v for (a, b), v in confusion.items() if a == b) / len(decisions),
              "luna_non_keep": len(non_keep_ref), "local_also_non_keep": caught,
              "local_non_keep_where_luna_keep": sum(v for (a, b), v in confusion.items() if a == "KEEP" and b != "KEEP"),
              "confusion_luna_to_local": {f"{a}->{b}": v for (a, b), v in sorted(confusion.items())},
              "elapsed_seconds": elapsed, "rows_per_second": len(rows) / elapsed,
              "usage": dict(engine.usage), "disagreements": disagreements,
              "quarantined_rows": quarantined,
              "binding": binding(args, {"luna_db": str(args.luna_db)}),
              "meaning": "agreement with Luna is not ground truth; Luna itself is unvalidated at scale; not a quality gate"}
    audit.atomic(args.out / "agreement_vs_luna_full_v1.json", report)
    print(json.dumps({k: report[k] for k in ("rows", "exact_decision_agreement", "luna_non_keep",
                                              "local_also_non_keep", "local_non_keep_where_luna_keep",
                                              "quarantined", "rows_per_second")}), flush=True)
    return 0


def cmd_full(args, engine: Engine) -> int:
    manifest = json.loads(args.manifest.read_text())
    if audit.digest(args.source.read_bytes()) != manifest["sha256"]:
        raise ValueError("source TSV sha256 differs from source_manifest.json")
    csv.field_size_limit(10**9)
    with args.source.open(newline="") as fh:
        rows = [{"id": r["id"], "caption": r["caption"]} for r in csv.DictReader(fh, delimiter="\t")]
    if len(rows) != manifest["rows"] or len({r["id"] for r in rows}) != len(rows):
        raise ValueError("source row count / id uniqueness mismatch")
    bind = binding(args, {"source_sha256": manifest["sha256"], "rows": len(rows),
                          "chunk_rows": CHUNK_BATCHES * BATCH})
    bpath = args.out / "binding.json"
    if bpath.exists() and json.loads(bpath.read_text()) != bind:
        raise ValueError("resume binding changed; use a new output directory")
    audit.atomic(bpath, bind)
    step = CHUNK_BATCHES * BATCH
    n_chunks = (len(rows) + step - 1) // step
    t0 = time.time()
    for k in range(n_chunks):
        part = args.out / f"chunk_{k:04d}.json"
        if part.exists():
            continue
        chunk = rows[k * step:(k + 1) * step]
        decisions, quarantined = engine.audit_rows(chunk)
        audit.atomic(part, {"chunk": k, "first_id": chunk[0]["id"], "rows": len(chunk),
                            "decisions": decisions, "quarantined": quarantined})
        done = min((k + 1) * step, len(rows))
        audit.atomic(args.out / "state.json", {"status": "running", "chunks_done": k + 1,
                     "chunks_total": n_chunks, "rows_done": done, "usage": dict(engine.usage),
                     "session_elapsed_seconds": time.time() - t0, "updated": time.time()})
        print(json.dumps({"chunk": k + 1, "of": n_chunks, "rows_done": done,
                          "elapsed_s": round(time.time() - t0)}), flush=True)
    counts, quarantine_total, seen = collections.Counter(), 0, 0
    for k in range(n_chunks):
        data = json.loads((args.out / f"chunk_{k:04d}.json").read_text())
        seen += len(data["decisions"]) + len(data["quarantined"])
        quarantine_total += len(data["quarantined"])
        for d in data["decisions"]:
            key = d["decision"] + ("" if d["decision"] != "TRIM_SUFFIX" else
                                   ("_selfcheck_keep" if d.get("trim_selfcheck_keep") else "_selfcheck_fail"))
            counts[key] += 1
    if seen != len(rows):
        raise ValueError(f"coverage mismatch: {seen} != {len(rows)}")
    audit.atomic(args.out / "state.json", {"status": "audit_complete_not_released", "rows": seen,
                 "decisions": dict(counts), "quarantined": quarantine_total, "updated": time.time()})
    print(json.dumps({"status": "audit_complete_not_released", "decisions": dict(counts),
                      "quarantined": quarantine_total}), flush=True)
    return 0


def cmd_candidates(args, engine: Engine) -> int:
    """Audit regenerated candidates (jsonl with id, caption). Accept = KEEP only."""
    rows = [json.loads(l) for l in args.candidates.read_text().splitlines() if l.strip()]
    rows = [{"id": r["id"], "caption": r["caption"]} for r in rows]
    decisions, quarantined = engine.audit_rows(rows) if rows else ([], [])
    audit.atomic(args.out, {"rows": len(rows), "decisions": decisions, "quarantined": quarantined,
                            "usage": dict(engine.usage),
                            "binding": binding(args, {"candidates_sha256": audit.digest(args.candidates.read_bytes())})})
    print(json.dumps({"candidates": len(rows), "keep": sum(d["decision"] == "KEEP" for d in decisions),
                      "quarantined": len(quarantined)}), flush=True)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["calibrate", "agree", "full", "candidates"])
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--max-model-len", type=int, default=8192)
    ap.add_argument("--gpu-mem", type=float, default=0.85)
    ap.add_argument("--fixtures", type=Path, nargs="*", default=[])
    ap.add_argument("--luna-db", type=Path)
    ap.add_argument("--source", type=Path)
    ap.add_argument("--manifest", type=Path)
    ap.add_argument("--candidates", type=Path)
    args = ap.parse_args()
    lock_dir = args.out.parent if args.mode == "candidates" else args.out
    lock_dir.mkdir(parents=True, exist_ok=True)
    with (lock_dir / f"lock_{args.mode}").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        engine = Engine(args.model, args.max_model_len, args.gpu_mem)
        if args.mode == "calibrate":
            return max((cmd_calibrate(argparse.Namespace(**{**vars(args), "fixtures": f}), engine)
                        for f in args.fixtures), default=0)
        if args.mode == "agree":
            return cmd_agree(args, engine)
        if args.mode == "candidates":
            return cmd_candidates(args, engine)
        return cmd_full(args, engine)


if __name__ == "__main__":
    raise SystemExit(main())
