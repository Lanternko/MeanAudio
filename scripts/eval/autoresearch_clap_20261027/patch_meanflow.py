"""Monkeypatch layer for the inference-time CLAP search (F3/F4/F5/F7).

Only `MeanAudio.ode_wrapper` and `MeanFlow.run_t0_to_t1` are replaced, and only
for configs that actually change them. `spec_from_config()` maps a configs.json
entry to the spec consumed here; a vanilla spec patches nothing at all, so the
copy gate and 20 of the 38 rung-1 configs run eval.py verbatim.

Formulas are taken from scripts/eval/guidance_geometry_adg_apg_20260922.py (073)
without modification; the F4 normalization is the r5 addendum
`w(t) = 3.0 * s(t) / mean_t s(t)` with the mean over the trial's real step points.
"""
from __future__ import annotations

import contextlib
import math

import torch

SCHEDULES = ('ramp_up', 'ramp_dn', 'bump_mid', 'sqrt_up')
GEOMETRIES = ('vanilla', 'adg', 'apg')
GRIDS = ('uniform', 'cos', 'smoothstep')

# schedule shape r(t); each is mapped to w(t) = 3.0 * r(t) / mean_step r
_SHAPES = {
    'ramp_up': lambda t: 1.0 - t,
    'ramp_dn': lambda t: t,
    'bump_mid': lambda t: 4.0 * t * (1.0 - t),
    'sqrt_up': lambda t: math.sqrt(max(0.0, 1.0 - t)),
}


# ── spec construction ────────────────────────────────────
def spec_from_config(cfg, steps=None, grid=None) -> dict:
    """Configs.json entry -> patch spec.

    `interval`, `schedule` and `geometry` are independent so a rung-3 combo
    (F3 interval x F5 geometry) is expressible; interval and schedule are
    mutually exclusive (no registered config uses both).
    """
    interval = cfg.get('interval')
    if interval is not None:
        if len(interval) != 2 or not (0.0 <= float(interval[0]) < float(interval[1]) <= 1.0):
            raise SystemExit(f'[FAIL] bad interval {interval!r} in {cfg.get("id")}')
        interval = (float(interval[0]), float(interval[1]))
    schedule = cfg.get('schedule')
    if schedule is not None and schedule not in SCHEDULES:
        raise SystemExit(f'[FAIL] unknown schedule {schedule!r}')
    if interval is not None and schedule is not None:
        raise SystemExit(f'[FAIL] {cfg.get("id")}: interval and schedule are exclusive')
    geometry = cfg.get('geometry', 'vanilla') or 'vanilla'
    if geometry not in GEOMETRIES:
        raise SystemExit(f'[FAIL] unknown geometry {geometry!r}')
    grid = str(grid if grid is not None else cfg.get('grid', 'uniform'))
    if grid not in GRIDS:
        raise SystemExit(f'[FAIL] unknown grid {grid!r}')
    spec = {
        'id': cfg.get('id'),
        'cfg': float(cfg.get('cfg', 3.0)),
        'interval': interval,
        'schedule': schedule,
        'geometry': geometry,
        'adg_gamma': float(cfg.get('adg_gamma', 0.0)),
        'apg_eta': float(cfg.get('apg_eta', 0.0)),
        'steps': int(steps if steps is not None else cfg.get('steps', 25)),
        'grid': grid,
    }
    if geometry == 'adg' and not (0.0 <= spec['adg_gamma'] <= 1.0):
        raise SystemExit(f'[FAIL] adg_gamma out of range: {spec["adg_gamma"]}')
    if geometry == 'apg' and not (0.0 <= spec['apg_eta'] <= 1.0):
        raise SystemExit(f'[FAIL] apg_eta out of range: {spec["apg_eta"]}')
    if schedule is not None:
        spec['schedule_mean'] = schedule_mean(schedule, spec['steps'], spec['grid'])
    return spec


def needs_ode_patch(spec) -> bool:
    return spec['interval'] is not None or spec['schedule'] is not None \
        or spec['geometry'] != 'vanilla'


def needs_grid_patch(spec) -> bool:
    return spec['grid'] != 'uniform'


# ── time grids (shared by the patch and the F4 normalization) ────────────
def grid_values(steps, grid, dtype=torch.float64):
    """steps+1 strictly decreasing values from 1.0 to 0.0 (inclusive endpoints)."""
    if grid == 'uniform':
        base = torch.linspace(1, 0, steps + 1, dtype=dtype)
        return base
    if grid in ('cos', 'smoothstep'):
        p = torch.linspace(0, 1, steps + 1, dtype=dtype)
        if grid == 'cos':
            ts = 0.5 * (1.0 + torch.cos(math.pi * p))
        else:
            ts = 1.0 - p * p * (3.0 - 2.0 * p)
        ts[0], ts[-1] = 1.0, 0.0
        return ts
    raise SystemExit(f'[FAIL] unknown grid {grid!r}')


def schedule_weights(schedule, steps, grid='uniform'):
    """w(t_i) = 3.0 * r(t_i) / mean_i r(t_i) over the steps the trial really takes."""
    if schedule not in _SHAPES:
        raise SystemExit(f'[FAIL] unknown schedule {schedule!r}')
    shape = _SHAPES[schedule]
    ts = grid_values(steps, grid)[:-1]                       # step start times
    vals = [shape(float(t)) for t in ts]
    mean = sum(vals) / len(vals)
    if mean <= 0.0:
        raise SystemExit(f'[FAIL] schedule {schedule} has zero mean on this grid')
    return [3.0 * v / mean for v in vals]


def schedule_mean(schedule, steps, grid='uniform'):
    shape = _SHAPES[schedule]
    ts = grid_values(steps, grid)[:-1]
    vals = [shape(float(t)) for t in ts]
    return sum(vals) / len(vals)


def cfg_at(t, spec) -> float:
    """Guidance strength for the step that starts at time t."""
    t = float(t)
    if spec['interval'] is not None:
        lo, hi = spec['interval']
        return spec['cfg'] if lo < t <= hi else 0.0
    if spec['schedule'] is not None:
        r = _SHAPES[spec['schedule']](t)
        return 3.0 * r / spec['schedule_mean']
    return spec['cfg']


# ── guidance geometry ────────────────────────────────────
def _norm(x, axis='global'):
    dims = tuple(range(1, x.ndim)) if axis == 'global' else (-1,)
    return x.pow(2).sum(dim=dims, keepdim=True).sqrt().clamp_min(1e-12)


def guidance(a, b, c, geometry='vanilla', gamma=0.0, eta=0.0):
    """One guidance step from the conditional (a) and unconditional (b) flows."""
    if c < 1.0:
        return a
    delta = a - b
    if geometry == 'apg':
        a_hat = a / _norm(a, 'global')
        dims = tuple(range(1, a.ndim))
        par = (delta * a_hat).sum(dim=dims, keepdim=True) * a_hat
        return a + (c - 1.0) * ((delta - par) + eta * par)
    v = a + (c - 1.0) * delta                       # eval.py: c*a + (1-c)*b
    if geometry == 'adg':
        v = v * ((1.0 - gamma) + gamma * _norm(a, 'global') / _norm(v, 'global'))
    return v


def make_ode_wrapper(spec):
    def ode_wrapper(self, t, r, latent, conditions, empty_conditions, cfg_strength, q=None):
        t = t * torch.ones(len(latent), device=latent.device, dtype=latent.dtype)
        r = r * torch.ones(len(latent), device=latent.device, dtype=latent.dtype)
        if q is None:
            q = torch.full((len(latent),), 10, dtype=torch.long, device=latent.device)
        c = cfg_at(float(t[0]), spec)
        if c < 1.0:
            return self.predict_flow(latent, t, r, conditions, q)
        a = self.predict_flow(latent, t, r, conditions, q)
        b = self.predict_flow(latent, t, r, empty_conditions, q)
        return guidance(a, b, c, spec['geometry'], spec['adg_gamma'], spec['apg_eta'])
    return ode_wrapper


def make_run_t0_to_t1(spec):
    def run_t0_to_t1(self, fn, x0):
        if spec['grid'] == 'uniform':
            steps = torch.linspace(1, 0, self.steps + 1).to(device=x0.device, dtype=x0.dtype)
        else:
            steps = grid_values(self.steps, spec['grid']).to(device=x0.device, dtype=x0.dtype)
        for ti, t in enumerate(steps[:-1]):
            t = t.expand(x0.shape[0])
            next_t = steps[ti + 1].expand(x0.shape[0])
            u_flow = fn(t=t, r=next_t, x=x0)
            dt = (t - next_t).mean()
            x0 = x0 - dt * u_flow
        return x0
    return run_t0_to_t1


@contextlib.contextmanager
def patch(spec):
    """Patch the two methods for the duration of the block; vanilla patches nothing."""
    from meanaudio.model.networks import MeanAudio
    from meanaudio.model.mean_flow import MeanFlow
    saved = {}
    try:
        if needs_ode_patch(spec):
            saved['ode'] = MeanAudio.ode_wrapper
            MeanAudio.ode_wrapper = make_ode_wrapper(spec)
        if needs_grid_patch(spec):
            saved['grid'] = MeanFlow.run_t0_to_t1
            MeanFlow.run_t0_to_t1 = make_run_t0_to_t1(spec)
        yield spec
    finally:
        if 'ode' in saved:
            MeanAudio.ode_wrapper = saved['ode']
        if 'grid' in saved:
            MeanFlow.run_t0_to_t1 = saved['grid']
