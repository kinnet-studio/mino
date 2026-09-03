import numpy as np
import pytest

from mino.core import LoftParams, loft
from mino.core.dart import angle_deficit_total, dart_mesh, dart_proposal, refined_grid
from mino.core.rails import prepare_rails
from tests.cases import CASES


def _twisted():
    case = CASES["twisted"]()
    params = LoftParams()
    res = loft(case["points_a"], case["points_b"], params, case["tangents_a"], case["tangents_b"])
    ra, rb = prepare_rails(case["points_a"], case["points_b"], params.samples,
                           case["tangents_a"], case["tangents_b"])
    return params, res, ra, rb


def test_refined_grid_shape_and_rows():
    params, res, ra, rb = _twisted()
    grid = refined_grid(ra, rb, res.rulings, mid_rails=3)
    assert grid.shape == (5, len(res.rulings), 3)
    for k, (i, j) in enumerate(res.rulings):
        assert np.allclose(grid[0, k], ra.points[i])
        assert np.allclose(grid[4, k], rb.points[j])
        assert np.allclose(grid[2, k], 0.5 * (ra.points[i] + rb.points[j]))


def test_angle_deficit_zero_on_plane_and_cylinder():
    xs = np.linspace(0, 1, 12)
    plane = np.array([[[x, y, 0.0] for x in xs] for y in xs])
    assert abs(angle_deficit_total(plane)) < 1e-9
    cyl = np.array([[[np.cos(th), np.sin(th), z] for th in np.linspace(0, np.pi, 30)]
                    for z in np.linspace(0, 1, 10)])
    assert abs(angle_deficit_total(cyl)) < 1e-9


def test_angle_deficit_matches_sphere_solid_angle():
    rows, cols = 40, 80
    phi = np.linspace(0, np.pi / 4, rows)
    lam = np.linspace(0, np.pi / 2, cols)
    sphere = np.array([[[np.cos(f) * np.cos(l), np.cos(f) * np.sin(l), np.sin(f)] for l in lam] for f in phi])
    deficit = angle_deficit_total(sphere)
    solid_angle = (np.pi / 2) * np.sin(np.pi / 4)
    assert deficit > 0
    # interior vertices miss a half-cell band along the boundary: a few percent low
    assert abs(deficit - solid_angle) / solid_angle < 0.06


def test_angle_deficit_negative_on_saddle():
    xs = np.linspace(-1, 1, 30)
    saddle = np.array([[[x, y, x * x - y * y] for x in xs] for y in xs])
    assert angle_deficit_total(saddle) < -1.0


def test_angle_deficit_respects_column_range():
    xs = np.linspace(-1, 1, 30)
    saddle = np.array([[[x, y, x * x - y * y] for x in xs] for y in xs])
    full = angle_deficit_total(saddle)
    part = angle_deficit_total(saddle, cols_range=(5, 15))
    assert part < 0 and abs(part) < abs(full)


def test_dart_proposal_on_twisted_is_a_gusset_at_max_twist():
    params, res, ra, rb = _twisted()
    rng = res.failing_ranges[-1]
    prop = dart_proposal(ra, rb, res.rulings, res.ruling_twist, rng, mid_rails=3)
    k1, k2 = rng
    assert prop.range == rng
    assert prop.ruling == k1 + int(np.argmax(res.ruling_twist[k1:k2 + 1]))
    assert prop.kind == "gusset" and prop.wedge_deg < -5.0


def test_dart_mesh_has_seam_path_and_clean_topology():
    params, res, ra, rb = _twisted()
    prop = dart_proposal(ra, rb, res.rulings, res.ruling_twist, res.failing_ranges[-1], mid_rails=3)
    dm = dart_mesh(ra, rb, res.rulings, res.ruling_twist, prop, mid_rails=3,
                   planar_tolerance=params.planar_tolerance, dart_from="B")
    assert len(dm.faces) == len(dm.face_twist)
    assert all(len(f) in (3, 4) for f in dm.faces)
    assert all(len(set(f)) == len(f) for f in dm.faces)
    assert max(v for f in dm.faces for v in f) < len(dm.verts)
    # no duplicate vertices
    assert len({tuple(np.round(v, 9)) for v in dm.verts}) == len(dm.verts)
    # seam: rows - 2 edges forming one path from the B row to row 1
    assert len(dm.seam_edges) == 3
    for (a, b), (c, d) in zip(dm.seam_edges, dm.seam_edges[1:]):
        assert b == c
    assert all(a != b for a, b in dm.seam_edges)
    # seam starts on rail B (z near cos(phi) side) when dart_from == "B"
    start = dm.verts[dm.seam_edges[0][0]]
    i, j = res.rulings[prop.ruling]
    assert np.allclose(start, rb.points[j])


def test_dart_mesh_from_a_starts_on_rail_a():
    params, res, ra, rb = _twisted()
    prop = dart_proposal(ra, rb, res.rulings, res.ruling_twist, res.failing_ranges[-1])
    dm = dart_mesh(ra, rb, res.rulings, res.ruling_twist, prop, dart_from="A")
    i, j = res.rulings[prop.ruling]
    assert np.allclose(dm.verts[dm.seam_edges[0][0]], ra.points[i])
