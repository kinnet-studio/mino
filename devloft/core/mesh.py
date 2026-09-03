"""Face construction, planarity metric, planarize pass, and quad splitting."""
from __future__ import annotations

import numpy as np


def build_faces(path, n_a: int):
    faces = []
    for (i, j), (i2, j2) in zip(path, path[1:]):
        if i2 == i + 1 and j2 == j + 1:
            faces.append((i, i + 1, n_a + j + 1, n_a + j))
        elif i2 == i + 1:
            faces.append((i, i + 1, n_a + j))
        else:
            faces.append((i, n_a + j + 1, n_a + j))
    return faces


def quad_planarity(verts: np.ndarray, face) -> float:
    """Closest-approach distance of the two diagonals over their mean length."""
    p0, p1, p2, p3 = verts[list(face)]
    d1 = p2 - p0
    d2 = p3 - p1
    n = np.cross(d1, d2)
    ln = np.linalg.norm(n)
    mean = 0.5 * (np.linalg.norm(d1) + np.linalg.norm(d2))
    if ln < 1e-12 or mean < 1e-12:
        return 0.0
    return float(abs(np.dot(p1 - p0, n)) / ln / mean)


def face_planarity(verts: np.ndarray, faces) -> np.ndarray:
    return np.array([quad_planarity(verts, f) if len(f) == 4 else 0.0 for f in faces], dtype=float)


def planarize(verts, faces, pinned, tolerance, iterations, max_nudge):
    verts = np.asarray(verts, dtype=float).copy()
    orig = verts.copy()
    quads = [f for f in faces if len(f) == 4]
    pinned = list(pinned)
    for _ in range(iterations):
        pl = face_planarity(verts, quads)
        if len(pl) == 0 or pl.max() <= tolerance:
            break
        acc = np.zeros_like(verts)
        cnt = np.zeros(len(verts))
        for f in quads:
            idx = list(f)
            P = verts[idx]
            n = np.cross(P[2] - P[0], P[3] - P[1])
            ln = np.linalg.norm(n)
            if ln < 1e-12:
                continue
            n = n / ln
            c = P.mean(axis=0)
            proj = P - np.outer((P - c) @ n, n)
            acc[idx] += proj
            cnt[idx] += 1
        moved = cnt > 0
        new = verts.copy()
        new[moved] = acc[moved] / cnt[moved, None]
        new[pinned] = orig[pinned]
        disp = new - orig
        d = np.linalg.norm(disp, axis=1)
        over = d > max_nudge
        if over.any():
            new[over] = orig[over] + disp[over] * (max_nudge / d[over])[:, None]
        verts = new
    return verts


def _tri_normal(verts, tri):
    p0, p1, p2 = verts[list(tri)]
    n = np.cross(p1 - p0, p2 - p0)
    ln = np.linalg.norm(n)
    return n / ln if ln > 1e-12 else np.zeros(3)


def _dihedral(verts, tris):
    n0, n1 = _tri_normal(verts, tris[0]), _tri_normal(verts, tris[1])
    if not n0.any() or not n1.any():
        # A degenerate (zero-area) triangle has no normal; treat this
        # diagonal option as maximally bad so it never wins over a
        # non-degenerate alternative.
        return float("inf")
    return float(np.degrees(np.arccos(np.clip(np.dot(n0, n1), -1.0, 1.0))))


def best_diagonal(verts, face):
    """Pick the flatter diagonal of a strip quad `(v0, v1, v2, v3) = (A[i], A[i+1], B[j+1], B[j])`.

    Ordering rule: the first emitted triangle must contain the incoming
    ruling `(v0, v3)` and the second must contain the outgoing ruling
    `(v1, v2)`, so consecutive faces in the strip keep sharing an edge after
    a split. Winding stays consistent between the two options.
    """
    v0, v1, v2, v3 = face
    opt_a = [(v0, v2, v3), (v0, v1, v2)]
    opt_b = [(v0, v1, v3), (v1, v2, v3)]
    return opt_a if _dihedral(verts, opt_a) <= _dihedral(verts, opt_b) else opt_b


def split_quads(verts, faces, planarity, tolerance):
    out, src, split = [], [], []
    for k, f in enumerate(faces):
        if len(f) == 4 and planarity[k] > tolerance:
            out.extend(best_diagonal(verts, f))
            src.extend([k, k])
            split.extend([True, True])
        else:
            out.append(tuple(f))
            src.append(k)
            split.append(False)
    return out, np.array(src, dtype=int), np.array(split, dtype=bool)


def mesh_area(verts, faces) -> float:
    total = 0.0
    for f in faces:
        P = verts[list(f)]
        for k in range(1, len(f) - 1):
            total += 0.5 * np.linalg.norm(np.cross(P[k] - P[0], P[k + 1] - P[0]))
    return float(total)
