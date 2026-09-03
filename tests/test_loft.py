import json

import numpy as np
import pytest

from devloft.core import LoftError, LoftParams, loft
from devloft.core.export import result_to_dict
from tests.cases import CASES, OFFSET_STEPS


def _run(name, **overrides):
    case = CASES[name]()
    params = LoftParams(**{**case["params"], **overrides})
    res = loft(case["points_a"], case["points_b"], params, case["tangents_a"], case["tangents_b"])
    return case, params, res


def _line_point_distance(p, a, b):
    d = b - a
    return np.linalg.norm(np.cross(p - a, d)) / np.linalg.norm(d)


def test_cylinder_pairs_by_index_with_zero_twist():
    _, params, res = _run("cylinder")
    n = params.samples
    assert res.rulings == [(i, i) for i in range(n)]
    assert res.ruling_twist.max() < 0.01
    assert not res.face_split.any()
    assert res.failing_ranges == []
    assert res.report.area_unfolded == pytest.approx(res.report.area_3d, rel=1e-3)
    assert res.verts.shape == (2 * n, 3)
    assert len(res.layout) == len(res.faces)


def test_cylinder_without_tangents_still_zero_twist():
    case = CASES["cylinder"]()
    res = loft(case["points_a"], case["points_b"], LoftParams())
    assert res.ruling_twist.max() < 0.05
    assert res.failing_ranges == []


def test_cone_rulings_converge_to_apex():
    _, params, res = _run("cone")
    n = params.samples
    apex = np.array([0.0, 0.0, 2.0])
    for i, j in res.rulings:
        assert _line_point_distance(apex, res.verts[i], res.verts[n + j]) < 1e-6
    assert res.ruling_twist.max() < 0.01
    assert not res.face_split.any()


def test_offset_cylinder_leans_rulings():
    _, params, res = _run("offset_cylinder")
    n = params.samples
    middle = [(i, j) for i, j in res.rulings if 12 <= i <= n - 12]
    assert middle and all(j - i == -OFFSET_STEPS for i, j in middle)
    mid_twist = [tw for (i, j), tw in zip(res.rulings, res.ruling_twist) if 12 <= i <= n - 12]
    assert max(mid_twist) < 0.1
    assert res.failing_ranges
    assert res.failing_ranges[0][0] == 0
    assert res.failing_ranges[-1][1] == len(res.rulings) - 1
    assert res.rulings[0] == (0, 0) and res.rulings[-1] == (n - 1, n - 1)


def test_twisted_flags_high_twist_end():
    _, params, res = _run("twisted")
    assert res.report.max_twist > params.twist_tolerance
    assert res.failing_ranges
    assert res.failing_ranges[-1][1] == len(res.rulings) - 1
    assert res.report.area_unfolded == pytest.approx(res.report.area_3d, rel=1e-2)
    assert len(res.face_twist) == len(res.faces) == len(res.face_planarity) == len(res.face_split)


def test_twisted_coarse_splits_quads_when_not_planarized():
    _, _, res = _run("twisted", samples=24, planarize=False)
    assert res.report.split_quad_count > 0
    assert res.face_split.sum() == 2 * res.report.split_quad_count


def test_planarize_keeps_rail_endpoints():
    case = CASES["twisted"]()
    params = LoftParams(samples=24)
    res = loft(case["points_a"], case["points_b"], params, case["tangents_a"], case["tangents_b"])
    n = params.samples
    assert np.allclose(res.verts[0], case["points_a"][0])
    assert np.allclose(res.verts[n - 1], case["points_a"][-1])
    assert np.allclose(res.verts[n], case["points_b"][0])
    assert np.allclose(res.verts[2 * n - 1], case["points_b"][-1])


def test_tie_breakers_run():
    for mode in ("shortest", "plane"):
        _, _, res = _run("cylinder", tie_breaker=mode, tie_weight=1.0)
        assert res.ruling_twist.max() < 0.01


def test_invalid_rails_raise():
    with pytest.raises(LoftError):
        loft(np.zeros((5, 3)), np.ones((5, 3)))


def test_result_to_dict_is_json_serializable():
    case, params, res = _run("twisted", samples=12)
    d = result_to_dict(res, case["points_a"], case["points_b"], "twisted", params)
    s = json.dumps(d)
    back = json.loads(s)
    assert back["name"] == "twisted"
    assert len(back["verts"]) == 24
    assert len(back["faces"]) == len(res.faces)
    assert len(back["layout"]) == len(res.faces)
    assert back["report"]["ruling_count"] == len(res.rulings)
    assert back["params"]["samples"] == 12
    assert len(back["rails"]["a"]) == len(case["points_a"])
