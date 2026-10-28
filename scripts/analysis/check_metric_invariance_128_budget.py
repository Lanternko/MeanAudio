#!/usr/bin/env python3
"""Recompute quiet_r2's smoke estimates without models; hypothetical budgets do not authorize runs."""
import argparse
import json
import math
from pathlib import Path


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--smoke',type=Path,default=Path(__file__).resolve().parents[2]/'docs/experiments/harn/metric_invariance_128_quiet_r2/budget_incident/smoke.json')
    ap.add_argument('--budget-hours',type=float,default=7)
    args=ap.parse_args()
    s=json.loads(args.smoke.read_text())
    per_clip=sum(sum(v.values()) for v in s['timing'].values())
    bench=s['transform_benchmark_seconds_per_clip'];variants=len(bench);systems=len(s['timing'])
    modes=len(next(iter(s['timing'].values())))
    extra=max(0.,sum(bench.values())-variants*bench['clean'])*systems*modes
    setup=sum(sum(v.values()) for v in s['setup_seconds'].values())*variants
    estimates={str(n):(per_clip*n*variants+extra*n+setup+per_clip/systems*402*11)*1.25 for n in (1000,600,400)}
    assert all(math.isclose(v,s['estimated_seconds'][n],rel_tol=1e-12) for n,v in estimates.items())
    feasible=[n for n in (1000,600,400) if estimates[str(n)]<=args.budget_hours*3600]
    print(json.dumps({'estimates_reproduced':True,'estimated_hours':{n:v/3600 for n,v in estimates.items()},
        'hypothetical_budget_hours':args.budget_hours,'selected_n_under_this_budget':feasible[0] if feasible else None,
        'note':'Offline calculation only. Registered quiet_r2 gate remains 7h; a different budget requires approval and a new contract.'},indent=2))


if __name__=='__main__':main()
