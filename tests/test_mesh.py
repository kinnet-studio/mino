import numpy as np

from devloft.core.mesh import (best_diagonal, build_faces, face_planarity, mesh_area, planarize,
                               quad_planarity, split_quads)


def test_build_faces_quads_and_triangles():
    path = [(0, 0), (1, 1), (2, 1), (2, 2)]
    faces = build_faces(path, n_a=3)
    assert faces == [(0, 1, 4, 3), (1, 2, 4), (2, 5, 4)]


def test_planarity_zero_for_flat_quad():
    v = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], float)
    assert quad_planarity(v, (0, 1, 2, 3)) == 0.0
    assert np.allclose(face_planarity(v, [(0, 1, 2, 3), (0, 1, 2)]), [0.0, 0.0])


def test_planarity_positive_for_bent_quad():
    v = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0.2], [0, 1, 0]], float)
    p = quad_planarity(v, (0, 1, 2, 3))
    assert 0.05 < p < 0.2


def test_planarize_flattens_unpinned_quad():
    v = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0.2], [0, 1, 0]], float)
    out = planarize(v, [(0, 1, 2, 3)], pinned=[], tolerance=1e-9, iterations=5, max_nudge=1.0)
    assert quad_planarity(out, (0, 1, 2, 3)) < 1e-9
    assert not np.allclose(out[2], v[2])


def test_planarize_pins_and_improves():
    v = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0.2], [0, 1, 0]], float)
    out = planarize(v, [(0, 1, 2, 3)], pinned=[0], tolerance=1e-9, iterations=20, max_nudge=1.0)
    assert np.allclose(out[0], v[0])
    assert quad_planarity(out, (0, 1, 2, 3)) < 0.5 * quad_planarity(v, (0, 1, 2, 3))


def test_planarize_caps_nudge():
    v = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0.2], [0, 1, 0]], float)
    out = planarize(v, [(0, 1, 2, 3)], pinned=[], tolerance=1e-6, iterations=20, max_nudge=0.01)
    assert np.linalg.norm(out - v, axis=1).max() <= 0.01 + 1e-9


def test_split_quads_over_tolerance():
    v = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0.2], [0, 1, 0],
                  [2, 0, 0], [2, 1, 0], [1, 1, 0]], float)
    faces = [(0, 1, 2, 3), (1, 4, 5, 6)]
    pl = face_planarity(v, faces)
    assert pl[0] > 0.01 and pl[1] == 0.0
    out, src, split = split_quads(v, faces, pl, tolerance=0.5)
    assert out == faces and list(src) == [0, 1] and not split.any()
    out, src, split = split_quads(v, faces, pl, tolerance=0.01)
    assert len(out) == 3 and all(len(f) == 3 for f in out[:2]) and len(out[2]) == 4
    assert list(src) == [0, 0, 1]
    assert list(split) == [True, True, False]
    assert set(out[0]) | set(out[1]) == {0, 1, 2, 3}


def test_split_quads_keep_strip_adjacency():
    # raising different vertices exercises both diagonal choices; every
    # consecutive pair of output faces must still share exactly two vertices
    for raised in (1, 2, 5, 6):
        v = np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0], [3, 0, 0],
                      [0, 1, 0], [1, 1, 0], [2, 1, 0], [3, 1, 0]], float)
        v[raised, 2] = 0.3
        faces = [(0, 1, 5, 4), (1, 2, 6, 5), (2, 3, 7, 6)]
        out, src, split = split_quads(v, faces, face_planarity(v, faces), tolerance=0.01)
        assert split.any()
        for f, g in zip(out, out[1:]):
            assert len(set(f) & set(g)) == 2


def test_best_diagonal_avoids_degenerate_triangle_when_possible():
    # v0 and v2 (opposite quad corners) coincide, so the diagonal split at
    # (v0, v2) collapses both its triangles to zero area; the other split
    # at (v1, v3) stays non-degenerate. best_diagonal must not return a
    # degenerate triangle when a fully non-degenerate option exists.
    v = np.array([[0, 0, 0], [1, 0, 0], [0, 0, 0], [0, 1, 0.5]], float)

    def is_degenerate(tri):
        p0, p1, p2 = v[list(tri)]
        return np.linalg.norm(np.cross(p1 - p0, p2 - p0)) < 1e-12

    tris = best_diagonal(v, (0, 1, 2, 3))
    degenerate = [is_degenerate(t) for t in tris]
    assert not any(degenerate)


def test_mesh_area():
    v = np.array([[0, 0, 0], [2, 0, 0], [2, 1, 0], [0, 1, 0]], float)
    assert np.isclose(mesh_area(v, [(0, 1, 2, 3)]), 2.0)
    assert np.isclose(mesh_area(v, [(0, 1, 2)]), 1.0)
