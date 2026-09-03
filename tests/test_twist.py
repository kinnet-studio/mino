import numpy as np

from mino.core.rails import Rail
from mino.core.twist import ruling_lengths, twist_matrix


def _rail(points, tangents):
    return Rail(np.asarray(points, float), np.asarray(tangents, float))


def test_parallel_tangents_give_zero_twist():
    a = _rail([[0, 0, 0], [1, 0, 0], [2, 0, 0]], [[1, 0, 0]] * 3)
    b = _rail([[0, 0, 1], [1, 0, 1], [2, 0, 1]], [[1, 0, 0]] * 3)
    tw = twist_matrix(a, b, window=2)
    assert np.allclose(np.diagonal(tw), 0.0)


def test_ninety_degree_case():
    # ruling along z; tangent A along x, tangent B along y -> normals differ by 90 deg
    a = _rail([[0, 0, 0]], [[1, 0, 0]])
    b = _rail([[0, 0, 1]], [[0, 1, 0]])
    tw = twist_matrix(a, b, window=0)
    assert np.isclose(tw[0, 0], 90.0)


def test_twist_is_folded_for_flipped_tangent():
    a = _rail([[0, 0, 0]], [[1, 0, 0]])
    b = _rail([[0, 0, 1]], [[-1, 0, 0]])
    tw = twist_matrix(a, b, window=0)
    assert np.isclose(tw[0, 0], 0.0)


def test_band_and_invalid_are_inf():
    a = _rail([[0, 0, 0], [1, 0, 0], [2, 0, 0]], [[1, 0, 0]] * 3)
    b = _rail([[0, 0, 1], [1, 0, 1], [3, 0, 0]], [[1, 0, 0]] * 3)
    tw = twist_matrix(a, b, window=1)
    assert np.isinf(tw[0, 2]) and np.isinf(tw[2, 0])   # outside band
    assert np.isinf(tw[2, 2])                          # ruling parallel to tangent
    assert np.isfinite(tw[1, 1])


def test_ruling_lengths():
    a = _rail([[0, 0, 0], [1, 0, 0]], [[1, 0, 0]] * 2)
    b = _rail([[0, 0, 2], [1, 0, 2]], [[1, 0, 0]] * 2)
    L = ruling_lengths(a, b)
    assert np.allclose(np.diagonal(L), 2.0)
    assert np.isclose(L[0, 1], np.sqrt(5))
