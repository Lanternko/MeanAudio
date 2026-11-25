#!/usr/bin/env python
"""CPU unit tests for the inference-time CLAP search harness.

    /home/kojiek/venvs/dac/bin/python -m pytest -q \
        scripts/eval/autoresearch_clap_20261027/test_unit.py
"""
import sys
import tempfile
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import common                                   # noqa: E402
import patch_meanflow as pm                     # noqa: E402
import trial                                    # noqa: E402

BASE = {'lvl30': {'clap': 0.2300, 'PQ': 7.60, 'peak_capped_n': 2},
        'raw': {'silent_n': 0, 'clipped_n': 0, 'crest_mean': 6.30},
        'fad': {'value': 5.00}}
LVL_OK = {'clap': 0.2400, 'pq': 7.70, 'lufs': -30.0, 'peak_capped_n': 2,
          'lufs_off_0p05_n': 0}
RAW_OK = {'silent_n': 0, 'clipped_n': 0, 'crest_mean': 6.30, 'lufs': -18.0}
DL_OK = {'lvl30_clap': 0.01, 'lvl30_pq': 0.10, 'fad': 0.10}


def cfg(**kw):
    base = {'id': 't', 'cfg': 3.0, 'prompt': 'stock', 'interval': None, 'schedule': None,
            'geometry': 'vanilla', 'steps': 25, 'grid': 'uniform'}
    base.update(kw)
    return base


# ── F3 interval ──────────────────────────────────────────
def test_interval_boundaries():
    spec = pm.spec_from_config(cfg(interval=[0.333, 0.667]))
    assert pm.cfg_at(0.333, spec) == 0.0          # lo is exclusive
    assert pm.cfg_at(0.3334, spec) == 3.0
    assert pm.cfg_at(0.667, spec) == 3.0          # hi is inclusive
    assert pm.cfg_at(0.6671, spec) == 0.0
    assert pm.cfg_at(0.0, spec) == 0.0            # t=0 is outside (0.333, 0.667]
    spec_early = pm.spec_from_config(cfg(interval=[0.667, 1.0]))
    assert pm.cfg_at(1.0, spec_early) == 3.0      # hi is inclusive at the first step
    assert pm.cfg_at(0.667, spec_early) == 0.0


def test_interval_on_real_grid():
    spec = pm.spec_from_config(cfg(interval=[0.5, 1.0]))
    ts = [float(t) for t in pm.grid_values(25, 'uniform')[:-1]]
    on = [t for t in ts if pm.cfg_at(t, spec) == 3.0]
    assert on == sorted([t for t in ts if t > 0.5], reverse=True)
    assert 0.52 in [round(t, 2) for t in on] and 0.48 not in on


def test_interval_and_schedule_exclusive():
    try:
        pm.spec_from_config(cfg(interval=[0.5, 1.0], schedule='ramp_up'))
    except SystemExit:
        return
    raise AssertionError('interval+schedule must be rejected')


# ── F4 schedule normalization ────────────────────────────
def test_schedule_mean_guidance_is_three():
    for sched in pm.SCHEDULES:
        for grid in pm.GRIDS:
            ws = pm.schedule_weights(sched, 25, grid)
            assert abs(sum(ws) / len(ws) - 3.0) < 1e-9, (sched, grid)
            assert min(ws) >= 0.0                    # r5: mean-3 scale, c<1 = conditional only
            assert max(ws) > 3.0
            spec = pm.spec_from_config(cfg(schedule=sched, grid=grid))
            ts = pm.grid_values(25, grid)[:-1]
            for t, w in zip(ts, ws):
                assert abs(pm.cfg_at(float(t), spec) - w) < 1e-6


def test_schedule_zero_at_the_endpoint_but_mean_still_three():
    for sched in ('ramp_up', 'sqrt_up', 'bump_mid'):
        ws = pm.schedule_weights(sched, 25, 'uniform')
        assert ws[0] == 0.0                     # t=1 endpoint: conditional-only, = vanilla c=1
        assert abs(sum(ws) / len(ws) - 3.0) < 1e-9
    dn = pm.schedule_weights('ramp_dn', 25, 'uniform')
    assert dn[0] > dn[-1] > 0.0


def test_schedule_shapes_are_ordered():
    up = pm.schedule_weights('ramp_up', 25, 'uniform')      # r = 1-t
    dn = pm.schedule_weights('ramp_dn', 25, 'uniform')      # r = t
    assert up[0] < up[-1] and dn[0] > dn[-1]
    assert abs(sum(up) - sum(dn)) < 1e-9


def test_schedule_differs_from_stock_pointwise():
    ws = pm.schedule_weights('ramp_up', 25, 'uniform')
    assert any(abs(w - 3.0) > 0.5 for w in ws)


# ── F5 geometry ──────────────────────────────────────────
def test_apg_eta0_projects_out_parallel():
    a = torch.tensor([[3.0, 4.0]])
    b = torch.tensor([[-3.0, -4.0]])             # delta = 2a, purely parallel -> v = a
    assert torch.allclose(pm.guidance(a, b, 3.0, 'apg', eta=0.0), a)
    b2 = torch.tensor([[0.0, 0.0]])              # delta == a -> pure parallel -> v == a
    assert torch.allclose(pm.guidance(a, b2, 3.0, 'apg', eta=0.0), a)


def test_apg_hand_computation():
    a = torch.tensor([[3.0, 4.0]])               # unit 0.6, 0.8
    b = torch.tensor([[7.0, 1.0]])               # delta = (-4, 3), orthogonal to a
    delta = torch.tensor([[-4.0, 3.0]])
    v = pm.guidance(a, b, 3.0, 'apg', eta=0.0)
    assert torch.allclose(v, a + 2.0 * delta)
    v_half = pm.guidance(a, b, 3.0, 'apg', eta=0.5)
    assert torch.allclose(v_half, v)             # par = 0, eta has nothing to scale
    b3 = torch.tensor([[3.0, 0.0]])              # delta = (0, 4): mixed
    delta3 = a - b3
    par = a * ((delta3 * a).sum() / (a * a).sum())
    assert torch.allclose(pm.guidance(a, b3, 3.0, 'apg', eta=0.0),
                          a + 2.0 * (delta3 - par))
    assert torch.allclose(pm.guidance(a, b3, 3.0, 'apg', eta=0.5),
                          a + 2.0 * (delta3 - par + 0.5 * par))


def test_adg_hand_computation():
    a = torch.tensor([[3.0, 4.0]])               # |a| = 5
    b = torch.tensor([[0.0, 0.0]])
    vanilla = torch.tensor([[9.0, 12.0]])        # 3a - 2b, |v| = 15
    v = pm.guidance(a, b, 3.0, 'adg', gamma=0.5)
    assert torch.allclose(v, (0.5 + 0.5 * 5.0 / 15.0) * vanilla)      # (6, 8)
    assert torch.allclose(pm.guidance(a, b, 3.0, 'adg', gamma=1.0), a)  # norm pulled to |a|
    assert torch.allclose(pm.guidance(a, b, 3.0, 'adg', gamma=0.0), vanilla)


def test_apg_eta1_matches_vanilla_numerically():
    torch.manual_seed(0)
    a = torch.randn(4, 8, 16, dtype=torch.float32)
    b = torch.randn(4, 8, 16, dtype=torch.float32)
    van = pm.guidance(a, b, 3.0, 'vanilla')
    apg = pm.guidance(a, b, 3.0, 'apg', eta=1.0)
    adg0 = pm.guidance(a, b, 3.0, 'adg', gamma=0.0)
    assert (apg - van).abs().max() / van.abs().max() < 1e-5
    assert (adg0 - van).abs().max() / van.abs().max() < 1e-6
    apg0 = pm.guidance(a, b, 3.0, 'apg', eta=0.0)
    assert (apg0 - van).abs().max() / van.abs().max() > 1e-3   # genuinely different


def test_cfg_below_one_is_conditional_only():
    a = torch.randn(2, 4)
    b = torch.randn(2, 4)
    assert torch.equal(pm.guidance(a, b, 0.0), a)
    assert torch.equal(pm.guidance(a, b, 0.5, 'apg', eta=0.5), a)


# ── F7 grid ──────────────────────────────────────────────
def test_grids_are_decreasing_with_exact_endpoints():
    for grid in pm.GRIDS:
        ts = pm.grid_values(25, grid)
        assert float(ts[0]) == 1.0 and float(ts[-1]) == 0.0
        assert bool((ts[:-1] > ts[1:]).all())
        assert len(ts) == 26
    uni = pm.grid_values(25, 'uniform')
    assert torch.equal(uni, torch.linspace(1, 0, 26, dtype=torch.float64))


# ── spec / plan plumbing ─────────────────────────────────
def test_twenty_of_38_configs_patch_nothing():
    cfgs = common.load_prereg()['search_space_rung1']['configs']
    unpatched = [c for c in cfgs
                 if not pm.needs_ode_patch(pm.spec_from_config(c))
                 and not pm.needs_grid_patch(pm.spec_from_config(c))]
    assert len(cfgs) == 38 and len(unpatched) == 20
    assert {c['family'] for c in unpatched} == {'F1', 'F2', 'F6', 'C'}


def test_effective_seed_and_trial_dir():
    plan1 = {'round': 'r1', 'split': 'dev256', 'checkpoints': ['a'], 'seeds': [42], 'configs': []}
    c1 = cfg(id='C1', seed=7)
    assert common.effective_seed(plan1, c1) == 7
    assert common.trial_dir(Path('/s'), plan1, c1, 'a', 7).name == 'C1'
    plan4 = {'round': 'r4', 'split': 'heldout1024', 'checkpoints': ['a', 'b'],
             'seeds': [42, 7], 'configs': []}
    assert common.trial_dir(Path('/s'), plan4, c1, 'a', 7).name == 'C1__a__s7'


def test_config_sha_covers_checkpoint_seed_split_and_cfg():
    c = cfg(id='F3')
    s = common.config_sha(c, 'ck', 42, 'dev256')
    assert s == common.config_sha(dict(c), 'ck', 42, 'dev256')
    assert s != common.config_sha(c, 'ck', 7, 'dev256')
    assert s != common.config_sha(c, 'ck2', 42, 'dev256')
    assert s != common.config_sha(dict(c, cfg=3.5), 'ck', 42, 'dev256')
    assert s != common.config_sha(c, 'ck', 42, 'dev1024')


def test_baseline_cfgkey_splits_at_one():
    assert common.baseline_cfgkey(cfg(cfg=0.0)) == 'cfg0'
    assert common.baseline_cfgkey(cfg(cfg=1.5)) == 'cfg3_neg'


# ── gates ────────────────────────────────────────────────
def _gates(**over):
    lvl = dict(LVL_OK, **{k: v for k, v in over.items() if k in LVL_OK})
    raw = dict(RAW_OK, **{k: v for k, v in over.items() if k in RAW_OK})
    dl = dict(DL_OK, **{k: v for k, v in over.items() if k in DL_OK})
    problems = over.get('problems', [])
    return trial.compute_gates(lvl, raw, BASE, dl, 256, problems)


def test_all_gates_pass():
    gates, status = _gates()
    assert status == 'pass'
    assert all(gates[g] is True for g in ('G2', 'G3', 'G4', 'G5', 'G6', 'G7'))
    assert gates['G0'] is None and gates['G1'] is None


def test_g2_fad_budget_is_pending_not_reject():
    gates, status = _gates(fad=0.31)
    assert gates['G2'] is False and status == 'fad_pending'


def test_g3_pq_drop_rejects():
    gates, status = _gates(lvl30_pq=-0.11)
    assert gates['G3'] is False and status == 'reject'


def test_g4_silence_rejects():
    gates, status = _gates(silent_n=4)
    assert gates['G4'] is False and status == 'reject'


def test_g5_clipping_rejects():
    gates, status = _gates(clipped_n=1)
    assert gates['G5'] is False and status == 'reject'
    gates, status = _gates(crest_mean=5.79)
    assert gates['G5'] is False and status == 'reject'


def test_g6_loudness_alignment_rejects():
    gates, status = _gates(peak_capped_n=6)
    assert gates['G6'] is False and status == 'reject'
    gates, status = _gates(lufs_off_0p05_n=3)
    assert gates['G6'] is False and status == 'reject'


def test_g7_problems_reject():
    gates, status = _gates(problems=['missing metrics'])
    assert gates['G7'] is False and status == 'reject'


def test_missing_fad_is_pending_and_missing_pq_is_reject():
    gates, status = _gates(fad=None)
    assert gates['G2'] is False and status == 'fad_pending'
    gates, status = _gates(lvl30_pq=None)
    assert gates['G3'] is False and status == 'reject'


# ── queue contract integrity ─────────────────────────────
def test_generated_contracts_bind_their_launcher_and_action():
    import hashlib
    import json
    docs = common.ROOT / 'docs/experiments'
    contracts = sorted(docs.glob('autoresearch_*_contract.json'))
    assert contracts, 'no generated contract to check'
    for cp in contracts:
        c = json.loads(cp.read_text())
        b = c['bindings']
        assert c['launch_allowed'] is True and c['queue_role'] == 'p2'
        assert b['action'] == str(common.HERE / 'run_round.py')
        assert b['action_sha256'] == common.sha256(b['action'])
        assert c['commands']['run'][-1] == b['action']
        assert c['fixed_protocol']['prereg_sha256'] == common.sha256(common.PREREG)
        launcher = common.ROOT / 'scripts/queue_candidates' / c['queue_name']
        body = launcher.read_bytes()
        assert hashlib.sha256(body).hexdigest() == b['launcher_sha256'], cp.name
        text = body.decode()
        assert 'df -B1 --output=avail /home/kojiek' in text and 'exit 3' in text
        assert 'python -I' in text and b['harn'] in text
        assert launcher.stat().st_mode & 0o111


def test_contract_launcher_path_points_at_pending():
    import json
    c = json.loads((common.ROOT / 'docs/experiments/autoresearch_153_p0_contract.json').read_text())
    assert c['bindings']['launcher'] == '/home/kojiek/gpu_queue/p2/pending/153_autoresearch_p0.sh'
    assert c['fixed_protocol']['mode'] == 'phase0'
    assert c['resume']['pause_progress'].endswith('pause_progress_153.json')


# ── split integrity ──────────────────────────────────────
def test_split_shas_match_prereg_and_nesting():
    import subprocess
    with tempfile.TemporaryDirectory() as td:
        subprocess.run([common.PY, str(HERE / 'split_subsets.py'), '--out', td],
                       check=True, capture_output=True)
        man = common.read_json(Path(td) / 'manifest.json')
        prereg = common.load_prereg()['splits']
        assert man['source_sha256'] == prereg['source_sha256']
        assert man['n_records'] == prereg['n_records'] == 5521
        for name in ('dev1024', 'heldout1024', 'rest3473'):
            assert man['splits'][name]['sha256'] == prereg[name]['sha256']
            assert man['splits'][name]['n'] == prereg[name]['n']
        dev256 = common.rows_for_tsv(Path(td) / 'dev256.tsv')
        dev1024 = common.rows_for_tsv(Path(td) / 'dev1024.tsv')
        assert len(dev256) == 256
        assert [i for i, _ in dev256] == [i for i, _ in dev1024][:256]
        assert man['splits']['dev256']['ids_sha256'] == common.sha256_text(
            '\n'.join(i for i, _ in dev256))
