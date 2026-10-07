import json

import numpy as np
import pytest

from mino.core import LoftParams, loft
from mino.core.diagnose import (crease_runs, diagnose, diagnosis_from_dict, diagnosis_to_dict,
                                format_diagnosis, probe_window)
from mino.core.rails import prepare_rails
from tests.cases import CASES


def _run(name, **overrides):
    case = CASES[name]()
    params = LoftParams(**{**case["params"], **overrides})
    res = loft(case["points_a"], case["points_b"], params, case["tangents_a"], case["tangents_b"])
    return case, params, res


def _diag(name, **overrides):
    case, params, res = _run(name, **overrides)
    return params, res, diagnose(case["points_a"], case["points_b"], case["tangents_a"], case["tangents_b"], params, res)


def test_ok_when_developable():
    params, res, d = _diag("cylinder")
    assert d.failing_ranges == [] and d.window_fix is None and d.strakes_needed is None
    assert [s.kind for s in d.suggestions] == ["ok"]
    assert format_diagnosis(d) == ["All rulings within tolerance"]


def test_crease_runs_counts_pairs():
    assert crease_runs(np.array([False, True, True, True, True, False, True, True])) == 2
    assert crease_runs(np.zeros(5, dtype=bool)) == 0


def test_probe_window_on_ellipse():
    case, params, res = _run("ellipse")
    ra, rb = prepare_rails(case["points_a"], case["points_b"], params.samples, case["tangents_a"], case["tangents_b"])
    assert probe_window(ra, rb, params, 2) > params.twist_tolerance
    assert probe_window(ra, rb, params, 8) <= params.twist_tolerance


def test_window_fix_on_ellipse():
    params, res, d = _diag("ellipse")
    assert res.failing_ranges
    assert d.window_fix == 8
    assert d.suggestions[0].kind == "window" and d.suggestions[0].params == {"window": 8}
    assert "Window 8" in d.suggestions[0].text
    # the fix really works
    case = CASES["ellipse"]()
    fixed = loft(case["points_a"], case["points_b"], LoftParams(window=8), case["tangents_a"], case["tangents_b"])
    assert fixed.failing_ranges == []


def test_twisted_is_unsolved_at_default_tolerance():
    params, res, d = _diag("twisted")
    assert d.window_fix is None
    assert d.strakes_needed is None
    kinds = [s.kind for s in d.suggestions]
    assert kinds[0] == "unsolved" and kinds.count("dart") == len(res.failing_ranges)
    assert kinds[-1] == "creases"
    assert d.darts[-1].kind == "gusset"
    assert "gusset" in [s for s in d.suggestions if s.kind == "dart"][0].text
    assert [s.rank for s in d.suggestions] == sorted(s.rank for s in d.suggestions)


def test_twisted_subdivides_at_eight_degrees():
    params, res, d = _diag("twisted", twist_tolerance=8.0)
    assert d.strakes_needed == 6
    sub = [s for s in d.suggestions if s.kind == "subdivide"][0]
    assert sub.params == {"strakes": 6} and "6 strakes" in sub.text


def test_diagnosis_round_trips_through_json():
    params, res, d = _diag("twisted")
    data = json.loads(json.dumps(diagnosis_to_dict(d)))
    back = diagnosis_from_dict(data)
    assert back.failing_ranges == d.failing_ranges
    assert [s.kind for s in back.suggestions] == [s.kind for s in d.suggestions]
    assert back.darts[0].ruling == d.darts[0].ruling
    assert np.isclose(back.darts[0].wedge_deg, d.darts[0].wedge_deg)


def test_probe_window_measures_the_quad_strip():
    # at Window 8 the ellipse's worst ruling sits in a fan, so spread and grid twist differ
    case, params, res = _run("ellipse", window=8)
    grid = _run("ellipse", window=8, quads=False)[2]
    assert res.report.max_twist < grid.report.max_twist - 0.1
    ra, rb = prepare_rails(case["points_a"], case["points_b"], params.samples, case["tangents_a"], case["tangents_b"])
    assert probe_window(ra, rb, params, 8) == pytest.approx(res.report.max_twist, rel=1e-12)
