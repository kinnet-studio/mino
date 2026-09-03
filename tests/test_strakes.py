import numpy as np
import pytest

from mino.core import LoftError, LoftParams, loft
from mino.core.rails import prepare_rails
from mino.core.strakes import chain_loft, find_strake_count, mid_rails, subdivide, subdivide_sections
from tests.cases import CASES


def _twisted(**overrides):
    case = CASES["twisted"]()
    params = LoftParams(**{**case["params"], **overrides})
    res = loft(case["points_a"], case["points_b"], params, case["tangents_a"], case["tangents_b"])
    return case, params, res


def _line_point_distance(p, a, b):
    d = b - a
    return np.linalg.norm(np.cross(p - a, d)) / np.linalg.norm(d)


def test_mid_rails_lie_on_rulings():
    case, params, res = _twisted()
    ra, rb = prepare_rails(case["points_a"], case["points_b"], params.samples,
                           case["tangents_a"], case["tangents_b"])
    mids = mid_rails(ra, rb, res.rulings, count=3)
    assert len(mids) == 3
    for m in mids:
        assert m.shape == (len(res.rulings), 3)
        for (i, j), p in zip(res.rulings, m):
            assert _line_point_distance(p, ra.points[i], rb.points[j]) < 1e-9
    # fractions increase from A toward B
    i0, j0 = res.rulings[10]
    d = [np.linalg.norm(m[10] - ra.points[i0]) for m in mids]
    assert d[0] < d[1] < d[2]


def test_subdivide_halves_twist_roughly():
    case, params, res = _twisted()
    strips = subdivide(case["points_a"], case["points_b"], case["tangents_a"], case["tangents_b"],
                       params, res.rulings, strakes=2)
    assert len(strips) == 2
    worst = max(s.report.max_twist for s in strips)
    assert worst < 0.7 * res.report.max_twist


def test_find_strake_count_reaches_tolerance():
    # measured: 8 strakes leave 5.2 deg on the twisted case, so use an 8 deg tolerance (6 strakes pass at 7.5)
    case, params, res = _twisted(twist_tolerance=8.0)
    count, worst, strips = find_strake_count(case["points_a"], case["points_b"], case["tangents_a"],
                                             case["tangents_b"], params, res.rulings, max_strakes=8)
    assert count == 6
    assert len(strips) == count
    assert worst <= params.twist_tolerance
    assert all(s.report.failing_ruling_count == 0 for s in strips)


def test_find_strake_count_visits_doubling_then_bracket(monkeypatch):
    import mino.core.strakes as strakes_mod
    visited = []
    real = strakes_mod.subdivide

    def spy(*args, **kwargs):
        visited.append(args[6] if len(args) > 6 else kwargs["strakes"])
        return real(*args, **kwargs)

    monkeypatch.setattr(strakes_mod, "subdivide", spy)
    case, params, res = _twisted(twist_tolerance=8.0)
    count, _, _ = find_strake_count(case["points_a"], case["points_b"], case["tangents_a"],
                                    case["tangents_b"], params, res.rulings, max_strakes=8)
    assert count == 6
    assert visited == [2, 4, 8, 5, 6]


def test_find_strake_count_reports_failure_when_capped():
    case, params, res = _twisted()
    count, worst, strips = find_strake_count(case["points_a"], case["points_b"], case["tangents_a"],
                                             case["tangents_b"], params, res.rulings, max_strakes=8)
    assert count is None
    assert 5.0 < worst < 6.0
    assert len(strips) == 8


def test_chain_loft_shares_rails():
    case = CASES["cylinder"]()
    params = LoftParams()
    n = params.samples
    mid = case["points_a"].copy()
    mid[:, 2] = 0.5
    strips = chain_loft([(case["points_a"], case["tangents_a"]), (mid, None), (case["points_b"], case["tangents_b"])], params)
    assert len(strips) == 2
    assert np.allclose(strips[0].verts[n:], strips[1].verts[:n])
    assert all(s.failing_ranges == [] for s in strips)


def test_subdivide_sections_shape():
    case, params, res = _twisted()
    secs = subdivide_sections(case["points_a"], case["points_b"], case["tangents_a"], case["tangents_b"],
                              params, res.rulings, strakes=4)
    assert len(secs) == 5
    assert secs[0][1] is not None and secs[-1][1] is not None
    assert all(t is None for _, t in secs[1:-1])
    assert all(p.shape[1] == 3 for p, _ in secs)


def test_chain_loft_and_subdivide_reject_bad_input():
    case, params, res = _twisted()
    with pytest.raises(LoftError):
        chain_loft([(case["points_a"], None)], params)
    with pytest.raises(LoftError):
        subdivide(case["points_a"], case["points_b"], None, None, params, res.rulings, strakes=1)


def test_subdivide_strips_share_rails_with_planarize_on():
    case, params, res = _twisted()
    strips = subdivide(case["points_a"], case["points_b"], case["tangents_a"], case["tangents_b"],
                       params, res.rulings, strakes=3)
    n = params.samples
    for left, right in zip(strips, strips[1:]):
        assert np.allclose(left.verts[n:], right.verts[:n], atol=1e-12)


def test_loft_pins_whole_rails_when_asked():
    case, params, res = _twisted(samples=24)
    ra, rb = prepare_rails(case["points_a"], case["points_b"], params.samples, case["tangents_a"], case["tangents_b"])
    pinned = loft(case["points_a"], case["points_b"], params, case["tangents_a"], case["tangents_b"], pin_a=True, pin_b=True)
    n = params.samples
    assert np.allclose(pinned.verts[:n], ra.points) and np.allclose(pinned.verts[n:], rb.points)
    free = loft(case["points_a"], case["points_b"], params, case["tangents_a"], case["tangents_b"])
    assert not np.allclose(free.verts[:n], ra.points)
