import numpy as np
import pytest

from mino.core.errors import LoftError
from mino.core.rails import (bend_profile, central_difference, normalize_rows, prepare_rails, resample,
                             shared_positions)


def test_resample_equal_arc_length_spacing():
    # collinear but unevenly spaced input: arc length equals Euclidean distance,
    # so equal arc-length spacing must give equal segment lengths
    pts = np.array([[0, 0, 0], [1, 0, 0], [1.2, 0, 0], [3, 0, 0]], float)
    rail = resample(pts, 7)
    seg = np.linalg.norm(np.diff(rail.points, axis=0), axis=1)
    assert rail.points.shape == (7, 3)
    assert np.allclose(seg, 0.5)
    assert np.allclose(rail.points[0], pts[0])
    assert np.allclose(rail.points[-1], pts[-1])


def test_resample_corner_lands_on_polyline():
    # a corner: samples at arc lengths 0, 1, 2 hit the corner exactly and stay on the path
    pts = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0]], float)
    rail = resample(pts, 3)
    assert np.allclose(rail.points, pts)


def test_resample_removes_duplicates():
    pts = np.array([[0, 0, 0], [0, 0, 0], [1, 0, 0], [1, 0, 0], [2, 0, 0]], float)
    rail = resample(pts, 3)
    assert np.allclose(rail.points, [[0, 0, 0], [1, 0, 0], [2, 0, 0]])


def test_resample_tangents_are_unit_and_follow_curve():
    pts = np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0]], float)
    rail = resample(pts, 5)
    assert np.allclose(np.linalg.norm(rail.tangents, axis=1), 1.0)
    assert np.allclose(rail.tangents, [[1, 0, 0]] * 5)


def test_resample_uses_supplied_tangents():
    pts = np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0]], float)
    tans = np.array([[0, 2, 0], [0, 2, 0], [0, 2, 0]], float)
    rail = resample(pts, 4, tans)
    assert np.allclose(rail.tangents, [[0, 1, 0]] * 4)


def test_resample_rejects_degenerate():
    with pytest.raises(LoftError):
        resample(np.zeros((3, 3)), 5)
    with pytest.raises(LoftError):
        resample(np.array([[0, 0, 0], [1, 0, 0]], float), 1)


def test_central_difference_ends_are_second_order():
    # ends use the three-point one-sided formula, scaled like the interior (2h * f')
    pts = np.array([[0, 0, 0], [1, 0, 0], [3, 0, 0]], float)
    t = central_difference(pts)
    assert np.allclose(t, [[1, 0, 0], [3, 0, 0], [5, 0, 0]])


def test_central_difference_two_points_falls_back_to_the_chord():
    pts = np.array([[0, 0, 0], [2, 0, 0]], float)
    assert np.allclose(central_difference(pts), [[2, 0, 0], [2, 0, 0]])


def test_end_tangent_of_sampled_circle_is_close_to_analytic():
    th = np.linspace(0.0, np.pi, 20)
    pts = np.column_stack([np.cos(th), np.sin(th), np.zeros_like(th)])
    t = normalize_rows(central_difference(pts))
    exact0 = np.array([0.0, 1.0, 0.0])
    err = np.degrees(np.arccos(np.clip(np.dot(t[0], exact0), -1.0, 1.0)))
    assert err < 0.5   # the one-sided chord was off by about 4.7 deg at this sampling


def test_normalize_rows_keeps_zero():
    v = normalize_rows(np.array([[0, 0, 0], [0, 3, 0]], float))
    assert np.allclose(v, [[0, 0, 0], [0, 1, 0]])


def test_prepare_rails_flips_reversed_b():
    a = np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0]], float)
    b_reversed = np.array([[2, 1, 0], [1, 1, 0], [0, 1, 0]], float)
    tans_b = np.array([[-1, 0, 0]] * 3, float)
    ra, rb = prepare_rails(a, b_reversed, 3, tangents_b=tans_b)
    assert np.allclose(rb.points[0], [0, 1, 0])
    assert np.allclose(rb.tangents, [[1, 0, 0]] * 3)
    assert ra.points.shape == rb.points.shape == (3, 3)


L_RAIL = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0]], float)


def _arc_then_line(n=400):
    """Quarter arc of radius 1, then a straight of length 3, tangent-continuous at (1, 1, 0)."""
    th = np.linspace(0.0, np.pi / 2, n)
    arc = np.column_stack([np.sin(th), 1 - np.cos(th), np.zeros(n)])
    u = np.linspace(0.0, 3.0, n)[1:]
    return np.vstack([arc, np.column_stack([np.ones_like(u), 1 + u, np.zeros_like(u)])])


def _true_tangent(p):
    a = np.arctan2(p[:, 0], 1 - p[:, 1])
    arc = np.column_stack([np.cos(a), np.sin(a), np.zeros(len(p))])
    return np.where((p[:, 1] <= 1)[:, None], arc, [0.0, 1.0, 0.0])


def _outline_error(curve, poly):
    """Largest distance from the curve's points to the polyline."""
    best = np.full(len(curve), np.inf)
    for a, b in zip(poly[:-1], poly[1:]):
        d = b - a
        t = np.clip((curve - a) @ d / (d @ d), 0.0, 1.0)
        best = np.minimum(best, np.linalg.norm(curve - (a + t[:, None] * d), axis=1))
    return float(best.max())


def test_bend_profile_spreads_a_corner_over_its_half_segments():
    t, theta = bend_profile(L_RAIL)
    assert np.allclose(t, [0, 0.25, 0.75, 1])
    assert np.allclose(theta, [0, 0, np.pi / 2, np.pi / 2])


def test_bend_profile_of_one_segment_is_flat():
    t, theta = bend_profile(np.array([[0, 0, 0], [2, 0, 0]], float))
    assert np.allclose(t, [0, 0.5, 1]) and not theta.any()


def test_shared_positions_follow_the_bending_rail():
    straight = np.array([[0, 0, 1], [3, 0, 1]], float)
    t = shared_positions(L_RAIL, straight, 21, 0.6)
    assert t[0] == 0.0 and t[-1] == 1.0 and np.all(np.diff(t) > 0)
    gaps = np.diff(t)
    assert gaps[9] < 0.05 - 1e-6 < gaps[0]  # dense around the corner, sparse at the ends; even is 1/20


def test_shared_positions_are_even_for_straight_rails():
    a = np.array([[0, 0, 0], [2, 0, 0]], float)
    assert np.allclose(shared_positions(a, a + [0, 1, 0], 11, 0.6), np.linspace(0, 1, 11))


def test_resample_puts_samples_at_given_positions():
    line = np.array([[0, 0, 0], [4, 0, 0]], float)
    rail = resample(line, 4, positions=np.array([0.0, 0.1, 0.5, 1.0]))
    assert np.allclose(rail.points[:, 0], [0, 0.4, 2, 4])


def test_adaptive_clusters_samples_around_a_corner():
    adapt, _ = prepare_rails(L_RAIL, L_RAIL + [0, 0, 1], 21, adaptive=0.6)
    gaps = np.linalg.norm(np.diff(adapt.points, axis=0), axis=1)
    k = int(np.argmin(np.linalg.norm(adapt.points - [1, 0, 0], axis=1)))
    assert gaps[k - 1:k + 1].max() < 0.1 - 1e-6  # even spacing is 2/20
    assert gaps.min() > 1e-9


def test_adaptive_straight_rails_match_even_spacing():
    a = np.array([[0, 0, 0], [2, 0, 0]], float)
    for x, y in zip(prepare_rails(a, a + [0, 1, 0], 11, adaptive=0.6), prepare_rails(a, a + [0, 1, 0], 11)):
        assert np.allclose(x.points, y.points)


def test_adaptive_halves_outline_error_on_arc_then_line():
    p = _arc_then_line()
    even, _ = prepare_rails(p, p + [0, 0, 1], 16)
    adapt, _ = prepare_rails(p, p + [0, 0, 1], 16, adaptive=0.6)
    assert _outline_error(p, adapt.points) < 0.5 * _outline_error(p, even.points)


def test_adaptive_tangents_stay_within_a_degree():
    p = _arc_then_line()
    ra, _ = prepare_rails(p, p + [0, 0, 1], 30, adaptive=0.6)
    cos = np.einsum("ij,ij->i", ra.tangents, _true_tangent(ra.points))
    assert np.degrees(np.arccos(np.clip(cos, -1, 1))).max() < 1.0


@pytest.mark.parametrize("adaptive", [-0.1, 1.0])
def test_adaptive_out_of_range_raises(adaptive):
    with pytest.raises(LoftError, match="adaptive must be at least 0 and below 1"):
        prepare_rails(L_RAIL, L_RAIL + [0, 0, 1], 10, adaptive=adaptive)


def test_adaptive_survives_a_hairpin_and_repeated_points():
    hair = np.array([[0, 0, 0], [1, 0, 0], [1, 0, 0], [0, 0.001, 0]], float)
    for rail in prepare_rails(hair, hair + [0, 0, 1], 30, adaptive=0.9):
        assert np.linalg.norm(np.diff(rail.points, axis=0), axis=1).min() > 1e-9
        assert np.isfinite(rail.tangents).all()
        assert np.allclose(np.linalg.norm(rail.tangents, axis=1), 1.0)


def test_adaptive_on_a_collapsed_rail_raises_cleanly():
    import warnings
    line = np.array([[0, 0, 0], [1, 0, 0]], float)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with pytest.raises(LoftError, match="2 distinct points"):
            prepare_rails(np.zeros((4, 3)), line, 10, adaptive=0.5)
