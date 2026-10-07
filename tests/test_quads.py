import numpy as np

from mino.core.quads import SPREAD, quad_strip, rail_at, spread_fans
from mino.core.rails import prepare_rails
from mino.core.twist import twist_matrix
from tests.cases import CASES


def _random_path(rng, n, m):
    i = j = 0
    path = [(0, 0)]
    while (i, j) != (n - 1, m - 1):
        moves = [(di, dj) for di, dj in ((1, 1), (1, 0), (0, 1)) if i + di < n and j + dj < m]
        di, dj = moves[rng.integers(len(moves))]
        i, j = i + di, j + dj
        path.append((i, j))
    return path


def _cylinder_rails(n=12):
    case = CASES["cylinder"]()
    return prepare_rails(case["points_a"], case["points_b"], n, case["tangents_a"], case["tangents_b"])


def test_spread_fans_known_path():
    path = [(0, 0), (0, 1), (0, 2), (1, 3), (2, 3), (3, 4)]
    assert np.allclose(spread_fans(path), [[0, 0], [0.125, 1], [0.25, 2], [1, 2.75], [2, 3.25], [3, 4]])


def test_spread_fans_random_paths_strictly_increase():
    rng = np.random.default_rng(7)
    for _ in range(300):
        n, m = (int(v) for v in rng.integers(2, 40, size=2))
        path = _random_path(rng, n, m)
        out, p = spread_fans(path), np.array(path, float)
        assert out.shape == p.shape
        assert np.all(np.diff(out, axis=0) > 0)
        assert np.array_equal(out[0], p[0]) and np.array_equal(out[-1], p[-1])
        assert np.abs(out - p).max() <= SPREAD + 1e-12


def test_spread_fans_leaves_fan_free_paths_alone():
    path = [(k, k) for k in range(10)]
    assert np.array_equal(spread_fans(path), np.array(path, float))


def test_rail_at_integer_indices_returns_samples():
    ra, _ = _cylinder_rails()
    pts, tans = rail_at(ra, np.arange(12, dtype=float))
    assert np.array_equal(pts, ra.points) and np.allclose(tans, ra.tangents)


def test_rail_at_interpolates_between_samples():
    ra, _ = _cylinder_rails()
    pts, tans = rail_at(ra, np.array([0.5]))
    assert np.allclose(pts[0], 0.5 * (ra.points[0] + ra.points[1]))
    assert np.isclose(np.linalg.norm(tans[0]), 1.0)


def test_quad_strip_on_a_diagonal_path_is_the_grid():
    ra, rb = _cylinder_rails()
    n = 12
    verts, faces, twist = quad_strip(ra, rb, [(k, k) for k in range(n)])
    assert np.array_equal(verts, np.vstack([ra.points, rb.points]))
    assert faces == [(k, k + 1, n + k + 1, n + k) for k in range(n - 1)]
    assert np.allclose(twist, np.diagonal(twist_matrix(ra, rb, 1)), atol=1e-9)
