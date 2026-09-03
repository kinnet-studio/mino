import numpy as np

from mino.core.mesh import (best_diagonal, build_faces, face_planarity, mesh_area, planarize,
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


def _split_pairs(out, split):
    """Yield (first_tri, second_tri) for each split quad in strip order."""
    k = 0
    while k < len(out):
        if split[k]:
            yield out[k], out[k + 1]
            k += 2
        else:
            k += 1


def _orientation(first_tri, n_a):
    # option A's first triangle (v0, v2, v3) holds one A vertex; option B's (v0, v1, v3) holds two
    return "A" if sum(v < n_a for v in first_tri) == 1 else "B"


def test_split_quads_consistent_uses_one_diagonal_per_run():
    n_a = 6
    v = np.array([[i, 0.0, 0.0] for i in range(n_a)] + [[i, 1.0, 0.0] for i in range(n_a)], float)
    # bend alternate vertices so per-quad choices would disagree
    v[7, 2] = 0.25
    v[3, 2] = -0.25
    v[10, 2] = 0.25
    faces = [(i, i + 1, n_a + i + 1, n_a + i) for i in range(n_a - 1)]
    pl = face_planarity(v, faces)
    out, src, split = split_quads(v, faces, pl, tolerance=0.01, consistent=True)
    assert split.sum() >= 4
    orientations = [_orientation(a, n_a) for a, b in _split_pairs(out, split)]
    assert len(set(orientations)) == 1
    # adjacency invariant survives
    for f, g in zip(out, out[1:]):
        assert len(set(f) & set(g)) == 2


def test_split_quads_consistent_picks_lower_total_dihedral():
    from mino.core.mesh import _dihedral
    n_a = 4
    v = np.array([[i, 0.0, 0.0] for i in range(n_a)] + [[i, 1.0, 0.0] for i in range(n_a)], float)
    v[5, 2] = 0.3  # bends quads 0 and 1 only: one run of two split quads
    faces = [(i, i + 1, n_a + i + 1, n_a + i) for i in range(n_a - 1)]
    pl = face_planarity(v, faces)
    assert [k for k in range(len(faces)) if pl[k] > 0.01] == [0, 1]
    out, src, split = split_quads(v, faces, pl, tolerance=0.01, consistent=True)
    chosen = sum(_dihedral(v, pair) for pair in _split_pairs(out, split))
    # the other orientation over the same run costs at least as much
    def other(face):
        v0, v1, v2, v3 = face
        a = [(v0, v2, v3), (v0, v1, v2)]
        b = [(v0, v1, v3), (v1, v2, v3)]
        return a, b
    runs = [faces[k] for k in range(len(faces)) if pl[k] > 0.01]
    cost_a = sum(_dihedral(v, other(f)[0]) for f in runs)
    cost_b = sum(_dihedral(v, other(f)[1]) for f in runs)
    assert chosen <= max(cost_a, cost_b) + 1e-9
    assert np.isclose(chosen, min(cost_a, cost_b))


def test_split_quads_default_is_per_quad():
    v = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0.2], [0, 1, 0]], float)
    faces = [(0, 1, 2, 3)]
    a, _, _ = split_quads(v, faces, face_planarity(v, faces), tolerance=0.01)
    b, _, _ = split_quads(v, faces, face_planarity(v, faces), tolerance=0.01, consistent=False)
    assert a == b
