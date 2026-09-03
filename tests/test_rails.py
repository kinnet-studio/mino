import numpy as np
import pytest

from mino.core.errors import LoftError
from mino.core.rails import central_difference, normalize_rows, prepare_rails, resample


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


def test_central_difference_ends():
    pts = np.array([[0, 0, 0], [1, 0, 0], [3, 0, 0]], float)
    t = central_difference(pts)
    assert np.allclose(t, [[1, 0, 0], [3, 0, 0], [2, 0, 0]])


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
