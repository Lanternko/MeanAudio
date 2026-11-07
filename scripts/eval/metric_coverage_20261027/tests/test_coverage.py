#!/usr/bin/env python3
"""CPU unit tests for the 161-177 coverage jobs (§4.5).

Covers the judging boundaries and the 80% rule, resume skipping, sample-manifest
determinism and the startup prerequisite checks.  No model is loaded and nothing
outside tmp_path is written.
"""
import json
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
sys.dont_write_bytecode = True

import common as C          # noqa: E402
import judge_coverage as J  # noqa: E402
import score_coverage as SC # noqa: E402


# ── judging boundaries / the 80% rule ────────────────────────────────────────
def test_tie_boundary_is_inclusive():
    m = 0.5
    u = J.judge_unit([m, -m, 0.0], m, 'tie')
    assert u['tie'] == 1.0 and u['pass'] is True and u['worse'] == 0.0


def test_tie_80pct_is_inclusive():
    m = 0.2
    at = J.judge_unit([m] * 8 + [10 * m] * 2, m, 'tie')
    below = J.judge_unit([m] * 7 + [10 * m] * 3, m, 'tie')
    assert at['tie'] == pytest.approx(0.8) and at['pass'] is True
    assert below['tie'] == pytest.approx(0.7) and below['pass'] is False


def test_lose_is_strict_and_needs_a_direction():
    m = 0.1
    # exactly -m is a tie, not "worse"
    u = J.judge_unit([-m, -m, -2 * m], m, 'lose')
    assert u['worse'] == pytest.approx(1 / 3) and u['pass'] is False
    ok = J.judge_unit([-2 * m] * 8 + [-0.5 * m] * 2, m, 'lose')
    assert ok['worse'] == pytest.approx(0.8) and ok['pass'] is True
    # AES PC has no lose direction in 126a: never a lose pass
    und = J.judge_unit([-2 * m] * 10, m, 'lose', directed=False)
    assert und['pass'] is None


def test_degenerate_all_zero_is_not_a_pass():
    u = J.judge_unit([0.0] * 5, 0.2, 'tie')
    assert u['degenerate'] is True and u['pass'] is False
    assert u['status'] == 'degenerate'
    cell = J.cell_from_units([u, J.judge_unit([0.0] * 5, 0.2, 'lose')])
    assert cell['label'] == 'degenerate'


def test_inconclusive_without_s():
    u = J.judge_unit([0.1, 0.2], None, 'tie')
    assert u['pass'] is None and u['status'] == 'inconclusive'
    assert J.cell_from_units([u])['label'] == 'inconclusive'


def test_cell_pass_requires_both_units():
    ok_t = J.judge_unit([0.01] * 10, 0.2, 'tie')
    ok_l = J.judge_unit([-10.0] * 10, 0.2, 'lose')
    bad_l = J.judge_unit([0.01] * 10, 0.2, 'lose')
    assert J.cell_from_units([ok_t, ok_l])['label'] == 'pass'
    assert J.cell_from_units([ok_t, bad_l])['label'] == 'partial'
    # only one unit measured = "只跑了一部分 unit" -> partial (§5.4)
    assert J.cell_from_units([bad_l])['label'] == 'partial'


def test_ood_units_are_annotated_and_excluded():
    good = J.judge_unit([0.01] * 4, 0.2, 'tie')
    ood = J.judge_unit([0.01] * 4, 0.2, 'tie', ood='OOD-MIDI')
    cell = J.cell_from_units([good, ood])
    assert cell['label'] == 'pass' and cell['excluded_units'] == 1
    assert cell['ood_units'] == ['OOD-MIDI']
    only = J.cell_from_units([ood])
    assert only['label'] == 'ood_only' and only['judged_label'] == 'pass'


def test_combine_dims_does_not_average():
    assert J.combine_dims(['pass', 'pass']) == 'pass'
    assert J.combine_dims(['pass', 'fail']) == 'partial'
    assert J.combine_dims(['pass', 'inconclusive']) == 'partial'
    assert J.combine_dims(['inconclusive', 'inconclusive']) == 'inconclusive'
    assert J.combine_dims(['n/a', 'n/a']) == 'n/a'


def test_muq_s_fallback_is_flagged():
    old = C.S_133_SUMMARY
    C.S_133_SUMMARY = Path('/nonexistent/metric_win_tie_133_summary.json')
    try:
        info = J.s_info('muq', '126a', 'muq_cosim')
    finally:
        C.S_133_SUMMARY = old
    assert info['S'] == pytest.approx(0.108679)
    assert info['m'] == pytest.approx(0.0108679)
    assert info['S_fallback'] is True and info['S_source'] == '133'


# ── resume skipping ──────────────────────────────────────────────────────────
class _Scorer:
    cols = ['pam']
    batch = 2

    def __init__(self):
        self.calls = 0
        self.keys = []

    def score_batch(self, items):
        self.calls += 1
        self.keys += [it['key'] for it in items]
        return [{'pam': 1.0} for _ in items]


def test_resume_skips_already_scored_keys(tmp_path):
    dummy = tmp_path / 'dummy.wav'
    dummy.write_bytes(b'0')
    items = [{'key': f'k{i}', 'path': str(dummy), 'kind': 'tie'} for i in range(5)]
    out = tmp_path / 'shard.tsv'
    s = _Scorer()
    assert SC.score_group('pam', '126b', items, out, s, tmp_path)[0] == 5
    first_calls = s.calls
    assert first_calls == 3  # batch 2
    # second pass: every key is already in the shard -> no model call
    assert SC.score_group('pam', '126b', items, out, s, tmp_path)[0] == 0
    assert s.calls == first_calls
    assert sorted(C.read_tsv(out)) == [f'k{i}' for i in range(5)]


# ── sample manifest determinism ─────────────────────────────────────────────
def test_sample_manifest_regenerates_and_is_prefixed():
    a = C.build_sample_manifest()
    b = C.build_sample_manifest()
    assert C.sha256_json(a) == C.sha256_json(b)
    p = C.sample_manifest_path()
    assert p.is_file()
    committed = json.loads(p.read_text())
    assert C.sha256_json(committed) == C.sha256_json(a)
    assert len(a['clip_ids']) == 512
    assert len(a['clip_ids_1024']) == 1024
    assert a['clip_ids_1024'][:512] == a['clip_ids']
    assert a['pam_clip_ids'] == a['clip_ids'][:256]


# ── startup prerequisite checks ─────────────────────────────────────────────
def test_prerequisites_171_fail_immediately(tmp_path, monkeypatch):
    monkeypatch.setattr(C, 'OUT_ROOT', tmp_path)
    monkeypatch.setattr(C, 'S_FILE', tmp_path / 's_values')
    monkeypatch.setattr(C, 'sample_manifest_path', lambda: tmp_path / 'manifest.json')
    need = SC.prerequisites(171)
    (tmp_path / 's_values').mkdir()
    for p in need:
        Path(p).parent.mkdir(parents=True, exist_ok=True)
        Path(p).write_text('{}')
    assert all(Path(p).exists() for p in need)
    assert SC.require(need) is None
    Path(need[-1]).unlink()
    with pytest.raises(SystemExit) as exc:
        SC.require(need)
    assert exc.value.code == 4


def test_require_lists_every_missing_path(tmp_path, capsys):
    with pytest.raises(SystemExit):
        SC.require([tmp_path / 'a', tmp_path / 'b'])
    err = capsys.readouterr().err
    assert 'PREREQUISITE MISSING' in err and str(tmp_path / 'a') in err


# ── generated contracts (skipped until make_contracts.py has run) ───────────
CONTRACTS = sorted((C.DOCS).glob('metric_coverage_*_contract.json'))


@pytest.mark.skipif(not CONTRACTS, reason='contracts not generated yet')
def test_contracts_are_valid_and_bind_their_launcher():
    import hashlib
    for p in CONTRACTS:
        d = json.loads(p.read_text())
        assert d['queue_role'] == 'p2' and d['launch_allowed'] is True
        assert d['commands']['preflight'][:2] == ['/home/kojiek/venvs/dac/bin/python', '-I']
        assert d['watcher']['stall_seconds'] >= 1800
        cand = C.REPO / 'scripts/queue_candidates' / d['queue_name']
        pending = Path(d['bindings']['launcher'])
        src = pending if pending.is_file() else cand
        assert src.is_file(), src
        assert hashlib.sha256(src.read_bytes()).hexdigest() == \
            d['bindings']['launcher_sha256']
