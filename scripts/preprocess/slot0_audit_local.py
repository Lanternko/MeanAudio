#!/usr/bin/env python3
"""Slot0 contamination screen (definition A) on a local vLLM model.

Definition, prompt and schema: slot0_contamination_a.py (KEEP / FLAG + category;
every FLAG row is regenerated, so no edit spans). Never writes a training corpus.

  probe       --probe probe_defA_v1.json      recall / false-flag per stratum + gate
  full        --source <slot0 TSV>            resumable chunked screen of every row
  candidates  --candidates regen.jsonl --out x.json   screen regenerated captions

Failure policy: a batch whose output fails validation is re-run one row at a
time; a row that still fails is quarantined (treated as FLAG downstream).
"""
from __future__ import annotations

import argparse
import collections
import csv
import fcntl
import json
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import slot0_contamination_a as A  # noqa: E402

DEFAULT_MODEL = str(next(Path.home().glob(
    ".cache/huggingface/hub/models--Qwen--Qwen2.5-32B-Instruct-AWQ/snapshots/*")))
BATCH = 8
CHUNK_BATCHES = 1000


def atomic(path: Path, value) -> None:
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(value, f, ensure_ascii=False, indent=1)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


class Engine:
    def __init__(self, model: str, max_model_len: int, gpu_mem: float):
        from vllm import LLM, SamplingParams
        from vllm.sampling_params import StructuredOutputsParams
        self.model = model
        self.llm = LLM(model=model, max_model_len=max_model_len, gpu_memory_utilization=gpu_mem,
                       enable_prefix_caching=True, seed=0)
        self.params = SamplingParams(temperature=0.0, max_tokens=1536, seed=0,
                                     structured_outputs=StructuredOutputsParams(json=A.SCHEMA_A))
        self.usage = collections.Counter()

    def _generate(self, batches: list[list[dict]]) -> list:
        convs = [[{"role": "system", "content": A.PROMPT_A},
                  {"role": "user", "content": A.user_message(rows)}] for rows in batches]
        outs = self.llm.chat(convs, self.params, use_tqdm=False)
        results = []
        for rows, out in zip(batches, outs):
            gen = out.outputs[0]
            self.usage.update({"prompt_tokens": len(out.prompt_token_ids), "completion_tokens": len(gen.token_ids)})
            try:
                results.append(A.parse_a(rows, gen.text, gen.finish_reason))
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
                decisions.extend(res)
        quarantined = []
        if retry:
            for row, res in zip(retry, self._generate([[r] for r in retry])):
                if isinstance(res, Exception):
                    quarantined.append({"id": row["id"], "error": f"{type(res).__name__}: {res}"[:300],
                                        "caption_sha256": A.digest(row["caption"].encode())})
                else:
                    decisions.extend(res)
        order = {r["id"]: i for i, r in enumerate(rows)}
        decisions.sort(key=lambda d: order[d["id"]])
        assert len(decisions) + len(quarantined) == len(rows)
        return decisions, quarantined


def binding(args, extra: dict) -> dict:
    return {"implementation_sha256": A.digest(Path(__file__).read_bytes()),
            "definition_module_sha256": A.digest((HERE / "slot0_contamination_a.py").read_bytes()),
            "prompt_sha256": A.digest(A.PROMPT_A.encode()), "model": args.model,
            "batch_size": BATCH, "temperature": 0.0, "max_model_len": args.max_model_len,
            "training_corpus_written": False, **extra}


def cmd_probe(args, engine: Engine) -> int:
    items = json.loads(args.probe.read_text())
    t0 = time.time()
    decisions, quarantined = engine.audit_rows([{"id": x["id"], "caption": x["caption"]} for x in items])
    elapsed = time.time() - t0
    got = {d["id"]: d for d in decisions}
    per = collections.defaultdict(lambda: {"n": 0, "flagged": 0, "misses": []})
    for x in items:
        d = got.get(x["id"])
        flagged = d is None or d["decision"] == "FLAG"  # quarantine counts as FLAG (it would be regenerated)
        s = per[x["stratum"]]
        s["n"] += 1
        s["flagged"] += int(flagged)
        if flagged != (x["label"] == "FLAG"):
            s["misses"].append({"id": x["id"], "caption": x["caption"][:300],
                                "got": d["decision"] if d else "QUARANTINED",
                                "category": d["category"] if d else None, "reason": d["reason"] if d else None})
    flag_strata = [k for k in per if k.endswith("_flag")]
    keep_strata = [k for k in per if k.endswith("_keep")]
    recall = {k: per[k]["flagged"] / per[k]["n"] for k in flag_strata}
    flag_rate = {k: per[k]["flagged"] / per[k]["n"] for k in keep_strata}
    recall_all = sum(per[k]["flagged"] for k in flag_strata) / sum(per[k]["n"] for k in flag_strata)
    # Gate (fixed before running 32B): overall recall >= 0.85 and every flag stratum >= 0.70;
    # false flags on random Luna-KEEP rows <= 3%. Hard negatives are reported, not gated.
    gate = {"recall_all": recall_all, "recall_by_stratum": recall, "flag_rate_on_keep": flag_rate,
            "passed": (recall_all >= 0.85 and min(recall.values()) >= 0.70
                       and flag_rate.get("luna_random_keep", 1.0) <= 0.03)}
    report = {"gate": gate, "strata": per, "quarantined": quarantined, "elapsed_seconds": elapsed,
              "rows_per_second": len(items) / elapsed, "usage": dict(engine.usage),
              "binding": binding(args, {"probe_sha256": A.digest(args.probe.read_bytes())}),
              "meaning": "probe of the local screen; labels are hand/Luna/pattern derived, not ground truth"}
    atomic(args.out / "probe_report.json", report)
    print("PROBE_GATE", json.dumps({**gate, "rows_per_second": round(len(items) / elapsed, 1)}), flush=True)
    return 0 if gate["passed"] else 3


def cmd_full(args, engine: Engine) -> int:
    manifest = json.loads(args.manifest.read_text())
    if A.digest(args.source.read_bytes()) != manifest["sha256"]:
        raise ValueError("source TSV sha256 differs from source_manifest.json")
    csv.field_size_limit(10**9)
    with args.source.open(newline="") as fh:
        rows = [{"id": r["id"], "caption": r["caption"]} for r in csv.DictReader(fh, delimiter="\t")]
    if len(rows) != manifest["rows"] or len({r["id"] for r in rows}) != len(rows):
        raise ValueError("source row count / id uniqueness mismatch")
    bind = binding(args, {"source_sha256": manifest["sha256"], "rows": len(rows), "chunk_rows": CHUNK_BATCHES * BATCH})
    bpath = args.out / "binding.json"
    if bpath.exists() and json.loads(bpath.read_text()) != bind:
        raise ValueError("resume binding changed; use a new output directory")
    atomic(bpath, bind)
    step = CHUNK_BATCHES * BATCH
    n_chunks = (len(rows) + step - 1) // step
    t0 = time.time()
    for k in range(n_chunks):
        part = args.out / f"chunk_{k:04d}.json"
        if part.exists():
            continue
        chunk = rows[k * step:(k + 1) * step]
        decisions, quarantined = engine.audit_rows(chunk)
        atomic(part, {"chunk": k, "first_id": chunk[0]["id"], "rows": len(chunk),
                      "decisions": decisions, "quarantined": quarantined})
        done = min((k + 1) * step, len(rows))
        atomic(args.out / "state.json", {"status": "running", "chunks_done": k + 1, "chunks_total": n_chunks,
                                         "rows_done": done, "usage": dict(engine.usage),
                                         "session_elapsed_seconds": time.time() - t0, "updated": time.time()})
        print(json.dumps({"chunk": k + 1, "of": n_chunks, "rows_done": done,
                          "elapsed_s": round(time.time() - t0)}), flush=True)
    counts, categories, quarantine_total, seen = collections.Counter(), collections.Counter(), 0, 0
    for k in range(n_chunks):
        data = json.loads((args.out / f"chunk_{k:04d}.json").read_text())
        seen += len(data["decisions"]) + len(data["quarantined"])
        quarantine_total += len(data["quarantined"])
        for d in data["decisions"]:
            counts[d["decision"]] += 1
            categories[d["category"]] += 1
    if seen != len(rows):
        raise ValueError(f"coverage mismatch: {seen} != {len(rows)}")
    atomic(args.out / "state.json", {"status": "audit_complete_not_released", "rows": seen,
                                     "decisions": dict(counts), "categories": dict(categories),
                                     "quarantined": quarantine_total, "updated": time.time()})
    print(json.dumps({"status": "audit_complete_not_released", "decisions": dict(counts),
                      "categories": dict(categories), "quarantined": quarantine_total}), flush=True)
    return 0


def cmd_candidates(args, engine: Engine) -> int:
    rows = [json.loads(l) for l in args.candidates.read_text().splitlines() if l.strip()]
    rows = [{"id": r["id"], "caption": r["caption"]} for r in rows]
    decisions, quarantined = engine.audit_rows(rows) if rows else ([], [])
    atomic(args.out, {"rows": len(rows), "decisions": decisions, "quarantined": quarantined,
                      "usage": dict(engine.usage),
                      "binding": binding(args, {"candidates_sha256": A.digest(args.candidates.read_bytes())})})
    print(json.dumps({"candidates": len(rows), "keep": sum(d["decision"] == "KEEP" for d in decisions),
                      "quarantined": len(quarantined)}), flush=True)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["probe", "full", "candidates"])
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--max-model-len", type=int, default=6144)
    ap.add_argument("--gpu-mem", type=float, default=0.88)
    ap.add_argument("--probe", type=Path)
    ap.add_argument("--source", type=Path)
    ap.add_argument("--manifest", type=Path)
    ap.add_argument("--candidates", type=Path)
    args = ap.parse_args()
    lock_dir = args.out.parent if args.mode == "candidates" else args.out
    lock_dir.mkdir(parents=True, exist_ok=True)
    with (lock_dir / f"lock_{args.mode}").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        engine = Engine(args.model, args.max_model_len, args.gpu_mem)
        return {"probe": cmd_probe, "full": cmd_full, "candidates": cmd_candidates}[args.mode](args, engine)


if __name__ == "__main__":
    raise SystemExit(main())
