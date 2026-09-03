import numpy as np
from mino.core.types import LoftParams, Rail, Report
from mino.core.errors import LoftError


def test_default_params():
    p = LoftParams()
    assert p.samples == 60
    assert p.window == 8
    assert p.twist_tolerance == 5.0
    assert p.tie_breaker == "none"
    assert p.planarize is True
    assert p.planar_tolerance == 0.01
    assert p.planarize_max_nudge == 0.05


def test_rail_holds_arrays():
    r = Rail(points=np.zeros((3, 3)), tangents=np.zeros((3, 3)))
    assert r.points.shape == (3, 3)


def test_loft_error_is_exception():
    assert issubclass(LoftError, Exception)


def test_consistent_creases_default():
    assert LoftParams().consistent_creases is True
