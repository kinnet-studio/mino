import numpy as np
import pytest

from mino.core import LoftParams, loft
from mino.core.rails import prepare_rails
from mino.core.relax import numeric_gradient, relax_objective, strip_normals_b
from tests.cases import CASES


def _rails_and_path(name, **case_kwargs):
    case = CASES[name](**case_kwargs)
    params = LoftParams(**case["params"])
    res = loft(case["points_a"], case["points_b"], params, case["tangents_a"], case["tangents_b"])
    ra, rb = prepare_rails(case["points_a"], case["points_b"], params.samples,
                           case["tangents_a"], case["tangents_b"])
    return case, params, res, ra, rb, np.asarray(res.rulings, dtype=int)


def test_strip_normals_are_unit_and_perpendicular_to_tangent_and_ruling():
    case, params, res, ra, rb, path = _rails_and_path("twisted")
    normals = strip_normals_b(ra, rb, path)
    assert normals.shape == rb.points.shape
    assert np.allclose(np.linalg.norm(normals, axis=1), 1.0)
    first = {}
    for i, j in path:
        first.setdefault(int(j), int(i))
    for j, i in first.items():
        r = rb.points[j] - ra.points[i]
        assert abs(np.dot(normals[j], rb.tangents[j])) < 1e-9
        assert abs(np.dot(normals[j], r)) < 1e-9


def test_objective_is_zero_when_within_target_and_unmoved():
    case, params, res, ra, rb, path = _rails_and_path("cylinder")
    normals = strip_normals_b(ra, rb, path)
    delta = np.zeros(len(rb.points))
    f, twists = relax_objective(delta, ra, rb, normals, path, target=5.0, smoothness=1.0)
    assert f == 0.0
    # B tangents are re-derived by central difference, so twists only approximate the loft's
    # exact-tangent values; the two end rulings use one-sided differences and drift about 1.5 deg
    assert np.allclose(twists[1:-1], res.ruling_twist[1:-1], atol=1.0)
    assert np.allclose(twists, res.ruling_twist, atol=2.0)


def test_objective_counts_twist_over_target_squared():
    case, params, res, ra, rb, path = _rails_and_path("twisted")
    normals = strip_normals_b(ra, rb, path)
    delta = np.zeros(len(rb.points))
    f, twists = relax_objective(delta, ra, rb, normals, path, target=4.5, smoothness=1.0)
    assert np.allclose(twists, res.ruling_twist, atol=1.0)
    expected = float((np.maximum(0.0, twists - 4.5) ** 2).sum())
    assert np.isclose(f, expected)


def test_smoothness_gradient_matches_chain_laplacian():
    case, params, res, ra, rb, path = _rails_and_path("cylinder")
    normals = strip_normals_b(ra, rb, path)
    n = len(rb.points)
    rng = np.random.default_rng(1)
    delta = 1e-3 * rng.normal(size=n)
    lam = 2.5
    # target far above any twist: only the smoothness term is active
    f = lambda d: relax_objective(d, ra, rb, normals, path, target=1e6, smoothness=lam)[0]
    free = np.ones(n, dtype=bool)
    free[0] = free[-1] = False
    g = numeric_gradient(f, delta, free, h=1e-7)
    lap = np.zeros(n)
    lap[1:-1] = 2 * delta[1:-1] - delta[:-2] - delta[2:]
    lap[0] = delta[0] - delta[1]
    lap[-1] = delta[-1] - delta[-2]
    analytic = 2.0 * lam * lap
    analytic[~free] = 0.0
    assert np.allclose(g, analytic, atol=1e-6)
    assert g[0] == 0.0 and g[-1] == 0.0


def test_numeric_gradient_on_quadratic():
    f = lambda x: float((x ** 2).sum())
    x = np.array([1.0, -2.0, 0.5])
    g = numeric_gradient(f, x, np.array([True, False, True]), h=1e-6)
    assert np.allclose(g, [2.0, 0.0, 1.0], atol=1e-6)
