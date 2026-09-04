import numpy as np

from mino.core import LoftParams, loft
from mino.core.rails import central_difference, normalize_rows, prepare_rails
from mino.core.relax import moved_tangents_b, numeric_gradient, relax_objective, strip_normals_b
from mino.core.types import Rail
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
    # nothing moved, so the supplied tangents are kept and the twists match the loft's; arccos
    # near zero twist amplifies machine epsilon to about 1e-6 deg, hence the margin
    assert np.allclose(twists, res.ruling_twist, atol=1e-4)


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


from mino.core.relax import RelaxResult, relax_rail_b


def _relax(name, scale=None, **kwargs):
    case = CASES[name]() if scale is None else CASES[name](scale=scale)
    params = LoftParams(**case["params"])
    return case, params, relax_rail_b(case["points_a"], case["points_b"], case["tangents_a"],
                                      case["tangents_b"], params, **kwargs)


def test_developable_strip_is_left_alone():
    case, params, r = _relax("cylinder")
    assert isinstance(r, RelaxResult)
    assert r.max_move_used < 1e-9
    assert r.iterations_run == 0
    ra, rb = prepare_rails(case["points_a"], case["points_b"], params.samples,
                           case["tangents_a"], case["tangents_b"])
    assert np.allclose(r.points_b, rb.points, atol=1e-9)
    assert r.result.failing_ranges == []
    # same tangent convention on both sides, so a developable strip measures the same twice
    assert abs(r.twist_after - r.twist_before) < 1e-3


def test_bounds_and_pins_are_respected():
    case, params, r = _relax("twisted", max_move=0.05)
    bound = 0.05 * r.mean_ruling
    assert np.all(np.abs(r.delta) <= bound + 1e-12)
    assert r.delta[0] == 0.0 and r.delta[-1] == 0.0
    assert np.isclose(r.max_move_used, bound, atol=1e-9)   # the bound is active on this case
    ra, rb = prepare_rails(case["points_a"], case["points_b"], params.samples,
                           case["tangents_a"], case["tangents_b"])
    assert np.allclose(np.linalg.norm(r.points_b - rb.points, axis=1), np.abs(r.delta), atol=1e-9)


def test_unpinned_endpoints_may_move():
    case, params, r = _relax("twisted", max_move=0.05, pin_endpoints=False)
    assert abs(r.delta[0]) > 0.0 or abs(r.delta[-1]) > 0.0


def test_mild_case_reaches_tolerance_within_bound():
    # Spike (spec section 7): twist_after about 4.6 deg, max_move_used about 0.065 of the mean ruling.
    case, params, r = _relax("twisted", scale=0.3, max_move=0.15)
    assert r.twist_before > params.twist_tolerance
    assert r.result.failing_ranges == []
    assert r.twist_after <= params.twist_tolerance
    assert r.max_move_used / r.mean_ruling < 0.1
    # BB descent lands near, not on, the hinge target (objective 161 -> under 1 in 800 steps)
    assert r.objective[-1] < 0.01 * r.objective[0]


def test_full_twisted_case_improves_monotonically():
    case, params, r = _relax("twisted", max_move=0.15)
    assert r.twist_after < r.twist_before
    assert r.twist_before > 25.0                     # about 31 deg; documents the input
    assert r.twist_after < 25.0                      # spike measured about 22 deg
    assert all(b <= a for a, b in zip(r.objective, r.objective[1:]))
    assert 0 < r.iterations_run <= 800
    assert np.isclose(r.max_move_used, 0.15 * r.mean_ruling, atol=1e-9)


def test_result_matches_reloft_of_returned_rail():
    case, params, r = _relax("twisted", scale=0.3, max_move=0.15)
    again = loft(case["points_a"], r.points_b, params, case["tangents_a"], r.tangents_b)
    assert np.allclose(again.verts, r.result.verts)
    assert again.rulings == r.result.rulings


def test_zero_max_move_is_a_noop():
    case, params, r = _relax("twisted", max_move=0.0)
    assert r.iterations_run == 0 and r.max_move_used == 0.0
    assert abs(r.twist_after - r.twist_before) < 1e-3


def test_relaxation_is_scale_invariant():
    # the smoothness term is dimensionless, so the same design in millimetres relaxes the same way
    case = CASES["twisted"](scale=0.3)
    params = LoftParams(**case["params"])
    unit = relax_rail_b(case["points_a"], case["points_b"], case["tangents_a"], case["tangents_b"],
                        params, max_move=0.15)
    big = relax_rail_b(1000.0 * case["points_a"], 1000.0 * case["points_b"], case["tangents_a"],
                       case["tangents_b"], params, max_move=0.15)
    assert big.result.failing_ranges == []
    # BB trajectories diverge under round-off before convergence; the pre-fix scaled case sat at 7 deg
    assert abs(big.twist_after - unit.twist_after) < 0.5
    assert abs(big.max_move_used / big.mean_ruling - unit.max_move_used / unit.mean_ruling) < 0.01


def test_exact_tangents_survive_low_sample_counts():
    # central-difference end tangents alone would invent about 7 deg of twist at 8 samples
    case = CASES["cylinder"]()
    params = LoftParams(**{**case["params"], "samples": 8})
    r = relax_rail_b(case["points_a"], case["points_b"], case["tangents_a"], case["tangents_b"], params)
    assert r.twist_before < params.twist_tolerance
    assert r.iterations_run == 0 and r.max_move_used < 1e-9
    assert abs(r.twist_after - r.twist_before) < 1e-3
    assert r.result.failing_ranges == []


def test_moved_tangents_reduce_to_central_difference_without_exact_tangents():
    case = CASES["twisted"]()
    params = LoftParams(**case["params"])
    ra, rb = prepare_rails(case["points_a"], case["points_b"], params.samples, None, None)
    moved = rb.points + 0.01 * np.random.default_rng(3).normal(size=rb.points.shape)
    assert np.allclose(moved_tangents_b(rb, moved), normalize_rows(central_difference(moved)), atol=1e-12)
    assert np.allclose(moved_tangents_b(rb, rb.points), rb.tangents, atol=1e-12)


def test_degenerate_ruling_counts_as_ninety_degrees():
    a = Rail(np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]]), np.array([[1.0, 0.0, 0.0]] * 3))
    b = Rail(np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 1.0], [2.0, 0.0, 1.0]]), np.array([[1.0, 0.0, 0.0]] * 3))
    path = np.array([[0, 0], [1, 1], [2, 2]])
    normals = strip_normals_b(a, b, path)
    f, twists = relax_objective(np.zeros(3), a, b, normals, path, target=5.0, smoothness=1.0)
    assert np.isinf(twists[0]) and np.isfinite(twists[1:]).all()
    assert np.isclose(f, (90.0 - 5.0) ** 2 + float((np.maximum(0.0, twists[1:] - 5.0) ** 2).sum()))


def test_stop_reason_names_why_the_loop_ended():
    _, _, noop = _relax("cylinder")
    assert noop.stop_reason == "noop"
    _, _, capped = _relax("twisted", max_move=0.15, iterations=20)
    assert capped.stop_reason == "iterations" and capped.iterations_run == 20
    _, _, zero = _relax("twisted", max_move=0.0)
    assert zero.stop_reason == "noop"


def test_time_budget_stops_the_loop_early():
    import time
    t0 = time.monotonic()
    _, _, r = _relax("twisted", scale=0.3, max_move=0.15, max_seconds=0.05)
    elapsed = time.monotonic() - t0
    assert r.stop_reason == "time"
    assert 0 < r.iterations_run < 100
    assert elapsed < 2.0            # the budget bounds the loop, the two lofts around it are cheap
    assert all(b <= a for a, b in zip(r.objective, r.objective[1:]))


def test_zero_time_budget_means_unlimited():
    _, _, r = _relax("twisted", max_move=0.15, iterations=30, max_seconds=0.0)
    assert r.stop_reason == "iterations" and r.iterations_run == 30


def test_progress_callback_sees_every_accepted_step():
    seen = []
    _, _, r = _relax("twisted", max_move=0.15, iterations=25, progress=lambda it, total: seen.append((it, total)))
    assert seen == [(k, 25) for k in range(1, r.iterations_run + 1)]
