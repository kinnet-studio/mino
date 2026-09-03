import numpy as np

from mino.core.mesh import mesh_area
from mino.core.unfold import unfold_strip


def _box_strip():
    # three planar quads folded around a box edge: unfolds to a 3x1 rectangle
    v = np.array([
        [0, 0, 0], [0, 1, 0],
        [1, 0, 0], [1, 1, 0],
        [1, 0, 1], [1, 1, 1],
        [0, 0, 1], [0, 1, 1],
    ], float)
    faces = [(0, 2, 3, 1), (2, 4, 5, 3), (4, 6, 7, 5)]
    return v, faces


def test_unfold_planar_strip_preserves_area_and_edges():
    v, faces = _box_strip()
    area, layout = unfold_strip(v, faces)
    assert np.isclose(area, mesh_area(v, faces))
    assert len(layout) == 3 and all(p.shape == (4, 2) for p in layout)
    # shared edge of faces 0 and 1 lands on the same 2D points
    f0, f1 = faces[0], faces[1]
    for vi in (2, 3):
        assert np.allclose(layout[0][f0.index(vi)], layout[1][f1.index(vi)], atol=1e-9)
    # faces lie on opposite sides of the shared edge: total spans 3 units
    pts = np.vstack(layout)
    ext = pts.max(axis=0) - pts.min(axis=0)
    assert np.isclose(sorted(ext)[1], 3.0) and np.isclose(sorted(ext)[0], 1.0)


def test_unfold_triangles():
    v = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]], float)
    faces = [(0, 1, 2), (1, 3, 2)]
    area, layout = unfold_strip(v, faces)
    assert np.isclose(area, 1.0)


def test_unfold_bent_quad_loses_area():
    v = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0.5], [0, 1, 0], [2, 0, 0], [2, 1, 0.5]], float)
    faces = [(0, 1, 2, 3), (1, 4, 5, 2)]
    area, _ = unfold_strip(v, faces)
    assert area < mesh_area(v, faces)


def test_flatten_face_survives_degenerate_first_edge():
    from mino.core.unfold import _flatten_face
    P = np.array([[0, 0, 0], [0, 0, 0], [1, 1, 0], [1, 0, 0]], float)
    local = _flatten_face(P)
    assert local.shape == (4, 2)
    # the non-degenerate vertices keep their true pairwise distances
    assert np.isclose(np.linalg.norm(local[2] - local[3]), 1.0)
    assert np.isclose(np.linalg.norm(local[3] - local[0]), 1.0)
    assert np.isclose(np.linalg.norm(local[2] - local[0]), np.sqrt(2))
    assert np.allclose(_flatten_face(np.zeros((3, 3))), 0.0)
