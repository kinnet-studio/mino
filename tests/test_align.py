import numpy as np
import pytest

from devloft.core.align import STEP_PENALTY, align, tie_matrix
from devloft.core.errors import LoftError
from devloft.core.types import Rail


def _is_monotone(path):
    for (i, j), (i2, j2) in zip(path, path[1:]):
        di, dj = i2 - i, j2 - j
        if (di, dj) not in {(1, 0), (0, 1), (1, 1)}:
            return False
    return True


def test_diagonal_when_cheapest():
    cost = np.full((4, 4), 10.0)
    np.fill_diagonal(cost, 0.0)
    path = align(cost)
    assert path == [(0, 0), (1, 1), (2, 2), (3, 3)]


def test_leans_to_cheap_offset_band():
    n = 8
    cost = np.full((n, n), 20.0)
    for i in range(2, n):
        cost[i, i - 2] = 0.0
    path = align(cost)
    assert _is_monotone(path)
    assert path[0] == (0, 0) and path[-1] == (n - 1, n - 1)
    middle = [(i, j) for i, j in path if 2 <= i <= n - 3]
    assert all(j == i - 2 for i, j in middle)


def test_step_penalty_prefers_quads_on_ties():
    cost = np.zeros((3, 3))
    path = align(cost)
    assert path == [(0, 0), (1, 1), (2, 2)]
    assert STEP_PENALTY > 0


def test_routes_around_inf():
    cost = np.zeros((3, 3))
    cost[1, 1] = np.inf
    path = align(cost)
    assert _is_monotone(path)
    assert (1, 1) not in path


def test_raises_when_no_path():
    cost = np.zeros((3, 3))
    cost[1, :] = np.inf
    with pytest.raises(LoftError) as e:
        align(cost)
    assert "1" in str(e.value)


def test_tie_matrix_modes():
    a = Rail(np.array([[0, 0, 0], [1, 0, 0]], float), np.array([[1, 0, 0]] * 2, float))
    b = Rail(np.array([[0, 0, 1], [1, 0, 1]], float), np.array([[1, 0, 0]] * 2, float))
    assert np.allclose(tie_matrix(a, b, "none", (0, 0, 1)), 0.0)
    short = tie_matrix(a, b, "shortest", (0, 0, 1))
    assert np.allclose(np.diagonal(short), 1.0)
    assert short[0, 1] > 1.0
    plane = tie_matrix(a, b, "plane", (0, 0, 1))
    assert np.allclose(np.diagonal(plane), 0.0)
    assert plane[0, 1] > 0.0
    with pytest.raises(LoftError):
        tie_matrix(a, b, "bogus", (0, 0, 1))


def test_raises_naming_unreachable_row_when_rows_have_holes():
    # every row has a finite cell, but no monotone path connects them
    cost = np.full((3, 3), np.inf)
    cost[0, 0] = 0.0
    cost[1, 2] = 0.0
    cost[2, 0] = 0.0
    with pytest.raises(LoftError) as e:
        align(cost)
    assert "1" in str(e.value)
