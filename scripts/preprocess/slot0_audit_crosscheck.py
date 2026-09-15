#!/usr/bin/env python3
"""Luna spot check of the regenerated slot0 candidate corpus (operator flow step 3).

  sample : two random strata from the assembled corpus (seed-fixed)
             R = rows regenerated and accepted by the local LLM (up to --per-stratum)
             U = rows the local LLM kept untouched            (up to --per-stratum)
  run    : Luna reviews the sampled captions; hard cost cap; --execute required
           to spend. full_v1 decisions are reused only when the caption hash matches.
  report : Luna KEEP rate per stratum with exact Clopper-Pearson 95% CI.
           PASS if every stratum's KEEP rate >= --pass-rate; otherwise
           REPORT_TO_OPERATOR with the failing examples.

Luna is a second text-only reviewer, not ground truth; audio fidelity is not
checked. A PASS means "an independent reviewer rarely disagrees on this sample",
not "zero contamination". Never releases a corpus.
"""
from __future__ import annotations

import argparse
import collections
import csv
import importlib.util
import json
import math
import random
import sqlite3
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("audit", HERE / "slot0_semantic_audit.py")
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)

BATCH = 8
PRICE_IN, PRICE_OUT = 0.2e-6, 1.2e-6          # USD / token, from full_v1 contract
EST_COST_PER_BATCH = 0.00075                    # observed ~0.00065 in full_v1, padded
AMBIGUOUS_RESERVE = 0.002                       # budgeted per failed/unknown request
SEED = 2026091505


# ---------------------------------------------------------------- statistics
def _binom_cdf(k: int, n: int, p: float) -> float:
    if p <= 0:
        return 1.0
    if p >= 1:
        return 0.0 if k < n else 1.0
    lp, lq = math.log(p), math.log1p(-p)
    terms = [math.lgamma(n + 1) - math.lgamma(i + 1) - math.lgamma(n - i + 1) + i * lp + (n - i) * lq
             for i in range(k + 1)]
    m = max(terms)
    return min(1.0, math.exp(m) * sum(math.exp(t - m) for t in terms))


def _cp_upper(x: int, n: int, tail: float) -> float:
    """p with P(X <= x | n, p) = tail (CDF is decreasing in p)."""
    if x >= n:
        return 1.0
    lo, hi = 0.0, 1.0
    for _ in range(100):
        mid = (lo + hi) / 2
        if _binom_cdf(x, n, mid) > tail:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def clopper_pearson(x: int, n: int, alpha: float = 0.05) -> tuple[float, float]:
    lower = 0.0 if x == 0 else 1 - _cp_upper(n - x, n, alpha / 2)
    return lower, _cp_upper(x, n, alpha / 2)


# ---------------------------------------------------------------- helpers
def load_local(full_dir: Path) -> tuple[dict, list]:
    decisions, quarantined = {}, []
    for part in sorted(full_dir.glob("chunk_*.json")):
        data = json.loads(part.read_text())
        for d in data["decisions"]:
            decisions[d["id"]] = d
        quarantined.extend(data["quarantined"])
    state = json.loads((full_dir / "state.json").read_text())
    if state.get("status") != "audit_complete_not_released":
        raise ValueError("local full audit is not complete")
    return decisions, quarantined


def load_tsv(path: Path) -> dict:
    csv.field_size_limit(10**9)
    with path.open(newline="") as fh:
        return {r["id"]: r["caption"] for r in csv.DictReader(fh, delimiter="\t")}


def load_luna_prior(db_path: Path) -> dict:
    db = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    out = {}
    for (result,) in db.execute("SELECT result FROM jobs WHERE phase='audit' AND status='done'"):
        for d in json.loads(result)["decisions"]:
            out[d["id"]] = d
    return out


# ---------------------------------------------------------------- sample
def cmd_sample(args) -> int:
    summary = json.loads((args.regen_dir / "summary.json").read_text())
    if summary["status"] != "regen_complete_not_released":
        raise ValueError("regeneration loop not complete")
    corpus_path = args.regen_dir / "slot0_regen_candidate_corpus.tsv"
    corpus = load_tsv(corpus_path)
    regenerated = sorted(json.loads((args.regen_dir / "accepted.json").read_text()))
    local, _ = load_local(args.local_full)
    untouched = sorted(i for i in corpus if i in local and local[i]["decision"] == "KEEP")
    rng = random.Random(SEED)
    strata = {"R_regenerated": sorted(rng.sample(regenerated, min(args.per_stratum, len(regenerated)))),
              "U_untouched": sorted(rng.sample(untouched, min(args.per_stratum, len(untouched))))}
    args.out.mkdir(parents=True, exist_ok=True)
    audit.atomic(args.out / "sample.json", {
        "seed": SEED, "corpus": str(corpus_path), "corpus_sha256": audit.digest(corpus_path.read_bytes()),
        "population": {"R_regenerated": len(regenerated), "U_untouched": len(untouched)},
        "strata": strata})
    print(json.dumps({k: len(v) for k, v in strata.items()}))
    return 0


# ---------------------------------------------------------------- run
def cmd_run(args) -> int:
    sample = json.loads((args.out / "sample.json").read_text())
    corpus_path = Path(sample["corpus"])
    if audit.digest(corpus_path.read_bytes()) != sample["corpus_sha256"]:
        raise ValueError("corpus changed since sampling")
    corpus = load_tsv(corpus_path)
    prior = load_luna_prior(args.luna_db)
    ids = sorted({i for s in sample["strata"].values() for i in s})
    reusable = {i for i in ids if i in prior and prior[i]["caption_sha256"] == audit.digest(corpus[i].encode())}
    need = [i for i in ids if i not in reusable]
    batches = [need[j:j + BATCH] for j in range(0, len(need), BATCH)]
    bdir = args.out / "luna_batches"
    bdir.mkdir(exist_ok=True)
    remaining = sum(1 for idx in range(len(batches)) if not (bdir / f"b{idx:05d}.json").exists()
                    and not (bdir / f"b{idx:05d}.failed.json").exists())
    projected = remaining * EST_COST_PER_BATCH
    spent_path = args.out / "luna_cost.json"
    spent = json.loads(spent_path.read_text())["usd"] if spent_path.exists() else 0.0
    print(json.dumps({"rows_needing_api": len(need), "reused_from_full_v1": len(reusable),
                      "batches": len(batches), "projected_usd": round(projected, 3),
                      "already_spent_usd": round(spent, 4), "cap_usd": args.cap_usd}), flush=True)
    if spent + projected > args.cap_usd:
        raise SystemExit(f"[HOLD] projected {spent + projected:.2f} USD exceeds cap {args.cap_usd}")
    if not args.execute:
        print("[DRY-RUN] no API call made; pass --execute to spend", flush=True)
        return 0
    key = audit.read_key_file(args.key_file)
    lock = threading.Lock()
    state = {"usd": spent, "failed": 0}

    def work(idx_ids):
        idx, batch_ids = idx_ids
        part = bdir / f"b{idx:05d}.json"
        failed_marker = part.with_suffix(".failed.json")
        if part.exists() or failed_marker.exists():
            return  # never auto-resubmit a failed/ambiguous request (possible double spend)
        rows = [{"id": i, "caption": corpus[i]} for i in batch_ids]
        for attempt in range(4):
            with lock:
                if state["usd"] + AMBIGUOUS_RESERVE > args.cap_usd:
                    return
            try:
                res = audit.call_model(key, rows, part.with_suffix(".receipt.json"))
            except audit.ApiFailure as e:
                if e.status == 429 and attempt < 3:
                    time.sleep(max(e.retry_after, 2 ** (attempt + 1)))
                    continue
                err = {"type": "ApiFailure", "status": e.status}
            except Exception as e:  # noqa: BLE001
                err = {"type": type(e).__name__}
            else:
                u = res.get("usage", {})
                with lock:
                    state["usd"] += u.get("prompt_tokens", 0) * PRICE_IN + u.get("completion_tokens", 0) * PRICE_OUT
                    audit.atomic(spent_path, {"usd": state["usd"]})
                audit.atomic(part, res)
                return
            with lock:
                state["usd"] += AMBIGUOUS_RESERVE
                state["failed"] += 1
                audit.atomic(spent_path, {"usd": state["usd"]})
            audit.atomic(failed_marker, {"error": err, "ids": batch_ids, "time": time.time()})
            return

    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(work, enumerate(batches)))
    print(json.dumps({"status": "run_finished", "usd_accounted": round(state["usd"], 4),
                      "failed_batches": state["failed"]}), flush=True)
    return 0


# ---------------------------------------------------------------- report
def cmd_report(args) -> int:
    sample = json.loads((args.out / "sample.json").read_text())
    corpus = load_tsv(Path(sample["corpus"]))
    luna = {i: d for i, d in load_luna_prior(args.luna_db).items()
            if i in corpus and d["caption_sha256"] == audit.digest(corpus[i].encode())}
    for part in (args.out / "luna_batches").glob("b*.json"):
        if not part.name.endswith((".receipt.json", ".failed.json")):
            for d in json.loads(part.read_text())["decisions"]:
                luna[d["id"]] = d
    strata, passed = {}, True
    for name, ids in sample["strata"].items():
        done = [i for i in ids if i in luna]
        keep = sum(1 for i in done if luna[i]["decision"] == "KEEP")
        lo, hi = clopper_pearson(keep, len(done)) if done else (None, None)
        rate = keep / len(done) if done else None
        ok = bool(done) and len(done) == len(ids) and rate >= args.pass_rate
        passed &= ok
        strata[name] = {"population": sample["population"][name], "sampled": len(ids), "luna_reviewed": len(done),
                        "luna_keep": keep, "keep_rate": rate, "keep_rate_ci95": [lo, hi], "pass": ok,
                        "luna_decisions": dict(collections.Counter(luna[i]["decision"] for i in done)),
                        "failures": [{"id": i, "luna": luna[i]["decision"], "reason": luna[i]["reason"],
                                      "caption": corpus[i]} for i in done if luna[i]["decision"] != "KEEP"]}
    verdict = "PASS" if passed else "REPORT_TO_OPERATOR"
    report = {"verdict": verdict, "pass_rate_threshold": args.pass_rate, "strata": strata,
              "meaning": ("Luna KEEP rate on a random sample; Luna is a text-only second reviewer, not ground "
                          "truth; audio fidelity unchecked; PASS is not a zero-contamination claim")}
    audit.atomic(args.out / "spotcheck_report.json", report)
    print(verdict, json.dumps({n: {k: s[k] for k in ("sampled", "luna_reviewed", "keep_rate", "keep_rate_ci95", "pass")}
                               for n, s in strata.items()}), flush=True)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["sample", "run", "report"])
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--local-full", type=Path)
    ap.add_argument("--regen-dir", type=Path)
    ap.add_argument("--luna-db", type=Path)
    ap.add_argument("--key-file", type=Path, default=Path.home() / ".config/meanaudio/luna_api_key")
    ap.add_argument("--per-stratum", type=int, default=500)
    ap.add_argument("--pass-rate", type=float, default=0.98)
    ap.add_argument("--cap-usd", type=float, default=2.0)
    ap.add_argument("--execute", action="store_true", help="run: actually call the paid API")
    args = ap.parse_args()
    return {"sample": cmd_sample, "run": cmd_run, "report": cmd_report}[args.mode](args)


if __name__ == "__main__":
    raise SystemExit(main())
