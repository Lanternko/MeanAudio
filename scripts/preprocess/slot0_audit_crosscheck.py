#!/usr/bin/env python3
"""Luna spot check of the regenerated slot0 corpus, definition A (operator flow step 3).

Paired design (the contamination base rate is ~0.2%, so a fixed KEEP-rate
threshold cannot tell a working screen from a useless one):
  S = 6,000 random ids from the source. Luna (same definition-A prompt as the
      local screen) reviews each ORIGINAL caption, and additionally the CLEANED
      caption for ids that were regenerated. Untouched ids have identical text,
      so one review serves both.
  R = up to 500 random regenerated-and-accepted rows (quality of regenerations).
  P = the probe set (checks Luna itself against the labels; report only).

report verdict:
  base      = Luna FLAG count on originals in S
  residual  = Luna FLAG count on cleaned captions in S (unresolved ids count as
              FLAG only if their original was flagged; they are excluded from the corpus)
  PASS if base >= 5, residual <= max(1, floor(0.25 * base)), and R KEEP rate >= 0.98.
  INCONCLUSIVE if base < 5 (too few contaminated rows in the sample to judge).
  otherwise REPORT_TO_OPERATOR with the failing captions.
Luna is a text-only second reviewer, not ground truth; audio fidelity unchecked.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import slot0_contamination_a as A  # noqa: E402

BATCH = 8
PRICE_IN, PRICE_OUT = 0.2e-6, 1.2e-6          # USD / token (gpt-5.6-luna, from full_v1 contract)
EST_COST_PER_BATCH = 0.0008                     # padded
AMBIGUOUS_RESERVE = 0.002
SEED = 2026091507
KEY_FILE = Path.home() / ".config/meanaudio/luna_api_key"


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


def load_tsv(path: Path) -> dict:
    csv.field_size_limit(10**9)
    with path.open(newline="") as fh:
        return {r["id"]: r["caption"] for r in csv.DictReader(fh, delimiter="\t")}


def load_local(full_dir: Path) -> tuple[dict, list]:
    decisions, quarantined = {}, []
    for part in sorted(full_dir.glob("chunk_*.json")):
        data = json.loads(part.read_text())
        for d in data["decisions"]:
            decisions[d["id"]] = d
        quarantined.extend(data["quarantined"])
    if json.loads((full_dir / "state.json").read_text()).get("status") != "audit_complete_not_released":
        raise ValueError("local full audit is not complete")
    return decisions, quarantined


def review_key(i: str, text: str) -> str:
    return f"{i}#{A.digest(text.encode())[:16]}"


# ---------------------------------------------------------------- sample
def cmd_sample(args) -> int:
    if json.loads((args.regen_dir / "summary.json").read_text())["status"] != "regen_complete_not_released":
        raise ValueError("regeneration loop not complete")
    source = load_tsv(args.source)
    corpus_path = args.regen_dir / "slot0_regen_candidate_corpus.tsv"
    corpus = load_tsv(corpus_path)
    accepted = json.loads((args.regen_dir / "accepted.json").read_text())
    local, quarantined = load_local(args.local_full)
    flagged = {i for i, d in local.items() if d["decision"] != "KEEP"} | {q["id"] for q in quarantined}
    rng = random.Random(SEED)
    s_ids = sorted(rng.sample(sorted(source), args.paired_n))
    r_ids = sorted(rng.sample(sorted(accepted), min(args.regen_n, len(accepted))))
    reviews = {}  # key -> {"id", "caption"}
    for i in s_ids:
        reviews[review_key(i, source[i])] = {"id": i, "caption": source[i]}
        if i in corpus and corpus[i] != source[i]:
            reviews[review_key(i, corpus[i])] = {"id": i, "caption": corpus[i]}
    for i in r_ids:
        reviews[review_key(i, corpus[i])] = {"id": i, "caption": corpus[i]}
    probe = json.loads(args.probe.read_text())
    for x in probe:
        reviews[review_key(x["id"], x["caption"])] = {"id": x["id"], "caption": x["caption"]}
    args.out.mkdir(parents=True, exist_ok=True)
    A_path = args.out / "sample.json"
    tmp = A_path.with_suffix(".tmp")
    tmp.write_text(json.dumps({
        "seed": SEED, "source_sha256": A.digest(args.source.read_bytes()),
        "corpus": str(corpus_path), "corpus_sha256": A.digest(corpus_path.read_bytes()),
        "probe": str(args.probe), "probe_sha256": A.digest(args.probe.read_bytes()),
        "S": s_ids, "R": r_ids, "S_locally_flagged": sorted(set(s_ids) & flagged),
        "population": {"source": len(source), "regenerated": len(accepted), "local_flagged": len(flagged)},
        "reviews": reviews}, ensure_ascii=False))
    tmp.replace(A_path)
    print(json.dumps({"S": len(s_ids), "S_locally_flagged": len(set(s_ids) & flagged), "R": len(r_ids),
                      "probe": len(probe), "unique_reviews": len(reviews)}))
    return 0


# ---------------------------------------------------------------- run
def cmd_run(args) -> int:
    sample = json.loads((args.out / "sample.json").read_text())
    keys = sorted(sample["reviews"])
    batches = [keys[j:j + BATCH] for j in range(0, len(keys), BATCH)]
    bdir = args.out / "luna_batches"
    bdir.mkdir(exist_ok=True)
    remaining = sum(1 for idx in range(len(batches)) if not (bdir / f"b{idx:05d}.json").exists()
                    and not (bdir / f"b{idx:05d}.failed.json").exists())
    spent_path = args.out / "luna_cost.json"
    spent = json.loads(spent_path.read_text())["usd"] if spent_path.exists() else 0.0
    projected = remaining * EST_COST_PER_BATCH
    print(json.dumps({"reviews": len(keys), "batches": len(batches), "remaining": remaining,
                      "projected_usd": round(projected, 3), "already_spent_usd": round(spent, 4),
                      "cap_usd": args.cap_usd}), flush=True)
    if spent + projected > args.cap_usd:
        raise SystemExit(f"[HOLD] projected {spent + projected:.2f} USD exceeds cap {args.cap_usd}")
    if not args.execute:
        print("[DRY-RUN] no API call made; pass --execute to spend", flush=True)
        return 0
    key = A.read_key_file(KEY_FILE)
    lock = threading.Lock()
    state = {"usd": spent, "failed": 0}

    def work(idx_keys):
        idx, bkeys = idx_keys
        part = bdir / f"b{idx:05d}.json"
        failed = part.with_suffix(".failed.json")
        if part.exists() or failed.exists():
            return  # never auto-resubmit a failed/ambiguous request (possible double spend)
        # Luna sees unique per-batch ids; map back through the review key.
        rows = [{"id": f"r{n}", "caption": sample["reviews"][k]["caption"]} for n, k in enumerate(bkeys)]
        for attempt in range(4):
            with lock:
                if state["usd"] + AMBIGUOUS_RESERVE > args.cap_usd:
                    return
            try:
                res = A.luna_call_a(key, rows, part.with_suffix(".receipt.json"))
            except A.ApiFailure as e:
                if e.status == 429 and attempt < 3:
                    time.sleep(max(e.retry_after, 2 ** (attempt + 1)))
                    continue
                err = {"type": "ApiFailure", "status": e.status}
            except Exception as e:  # noqa: BLE001
                err = {"type": type(e).__name__, "detail": str(e)[:200]}
            else:
                u = res["usage"]
                with lock:
                    state["usd"] += u.get("prompt_tokens", 0) * PRICE_IN + u.get("completion_tokens", 0) * PRICE_OUT
                    spent_path.write_text(json.dumps({"usd": state["usd"]}))
                out = {k: {"decision": d["decision"], "category": d["category"], "reason": d["reason"]}
                       for k, d in zip(bkeys, res["decisions"])}
                part.write_text(json.dumps(out, ensure_ascii=False))
                return
            with lock:
                state["usd"] += AMBIGUOUS_RESERVE
                state["failed"] += 1
                spent_path.write_text(json.dumps({"usd": state["usd"]}))
            failed.write_text(json.dumps({"error": err, "keys": bkeys, "time": time.time()}))
            return

    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(work, enumerate(batches)))
    print(json.dumps({"status": "run_finished", "usd_accounted": round(state["usd"], 4),
                      "failed_batches": state["failed"]}), flush=True)
    return 0


# ---------------------------------------------------------------- report
def cmd_report(args) -> int:
    sample = json.loads((args.out / "sample.json").read_text())
    source = load_tsv(args.source)
    corpus = load_tsv(Path(sample["corpus"]))
    luna = {}
    for part in (args.out / "luna_batches").glob("b*.json"):
        if not part.name.endswith((".receipt.json", ".failed.json")):
            luna.update(json.loads(part.read_text()))
    rev = lambda i, text: luna.get(review_key(i, text))  # noqa: E731
    flagged_local = set(sample["S_locally_flagged"])

    missing, base, residual, local_caught, base_rows, residual_rows = 0, 0, 0, 0, [], []
    for i in sample["S"]:
        o = rev(i, source[i])
        if o is None:
            missing += 1
            continue
        if o["decision"] == "FLAG":
            base += 1
            local_caught += int(i in flagged_local)
            base_rows.append({"id": i, "category": o["category"], "local_flagged": i in flagged_local,
                              "caption": source[i][:300]})
        if i not in corpus:  # unresolved, excluded from corpus
            continue
        c = rev(i, corpus[i])
        if c is None:
            missing += 1
            continue
        if c["decision"] == "FLAG":
            residual += 1
            residual_rows.append({"id": i, "category": c["category"], "regenerated": corpus[i] != source[i],
                                  "caption": corpus[i][:300]})
    r_done = [i for i in sample["R"] if rev(i, corpus[i]) is not None]
    r_keep = sum(1 for i in r_done if rev(i, corpus[i])["decision"] == "KEEP")
    r_rate = r_keep / len(r_done) if r_done else None

    probe = json.loads(Path(sample["probe"]).read_text())
    p_agree = [rev(x["id"], x["caption"])["decision"] == x["label"] for x in probe if rev(x["id"], x["caption"])]

    n_s = len(sample["S"]) - missing
    if missing:
        verdict = "INCOMPLETE"
    elif base < 5:
        verdict = "INCONCLUSIVE"
    elif residual <= max(1, math.floor(0.25 * base)) and r_rate is not None and r_rate >= 0.98:
        verdict = "PASS"
    else:
        verdict = "REPORT_TO_OPERATOR"
    report = {
        "verdict": verdict, "missing_reviews": missing,
        "paired": {"n": n_s, "base_flags": base, "base_rate_ci95": clopper_pearson(base, n_s) if n_s else None,
                   "residual_flags": residual, "residual_rate_ci95": clopper_pearson(residual, n_s) if n_s else None,
                   "local_recall_on_luna_flags": (local_caught / base) if base else None,
                   "base_rows": base_rows, "residual_rows": residual_rows},
        "regenerated": {"n": len(r_done), "keep": r_keep, "keep_rate": r_rate,
                        "failures": [{"id": i, **rev(i, corpus[i]), "caption": corpus[i][:300]}
                                     for i in r_done if rev(i, corpus[i])["decision"] != "KEEP"]},
        "luna_probe_agreement": (sum(p_agree) / len(p_agree)) if p_agree else None,
        "rule": "PASS iff base>=5 and residual<=max(1,floor(0.25*base)) and regenerated KEEP rate>=0.98",
        "meaning": "Luna (definition A, text only) is a second reviewer, not ground truth; audio fidelity unchecked"}
    (args.out / "spotcheck_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=1))
    print(verdict, json.dumps({"base": base, "residual": residual, "n": n_s,
                               "local_recall": report["paired"]["local_recall_on_luna_flags"],
                               "regen_keep_rate": r_rate, "luna_probe_agreement": report["luna_probe_agreement"]}),
          flush=True)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["sample", "run", "report"])
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--source", type=Path)
    ap.add_argument("--local-full", type=Path)
    ap.add_argument("--regen-dir", type=Path)
    ap.add_argument("--probe", type=Path)
    ap.add_argument("--paired-n", type=int, default=6000)
    ap.add_argument("--regen-n", type=int, default=500)
    ap.add_argument("--cap-usd", type=float, default=2.0)
    ap.add_argument("--execute", action="store_true", help="run: actually call the paid API")
    args = ap.parse_args()
    return {"sample": cmd_sample, "run": cmd_run, "report": cmd_report}[args.mode](args)


if __name__ == "__main__":
    raise SystemExit(main())
