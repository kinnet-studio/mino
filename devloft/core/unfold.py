"""Sequential flattening of a strip for validation (each face treated as rigid)."""
from __future__ import annotations

import numpy as np

from .rails import normalize_rows


def _polygon_normal(P: np.ndarray) -> np.ndarray:
    n = np.zeros(3)
    for k in range(len(P)):
        n += np.cross(P[k], P[(k + 1) % len(P)])
    return normalize_rows(n)


def _flatten_face(P: np.ndarray) -> np.ndarray:
    c = P.mean(axis=0)
    n = _polygon_normal(P - c)

    # Find first non-degenerate edge
    u = None
    for k in range(len(P)):
        edge = P[(k + 1) % len(P)] - P[k]
        edge = edge - n * np.dot(edge, n)
        if np.linalg.norm(edge) > 1e-12:
            u = normalize_rows(edge)
            break

    if u is None:  # fully degenerate face
        return np.zeros((len(P), 2))

    v = np.cross(n, u)
    Q = P - c
    return np.column_stack([Q @ u, Q @ v])


def _reflect_across(pts, a, b):
    d = b - a
    d = d / (np.linalg.norm(d) or 1.0)
    q = pts - a
    return a + 2.0 * np.outer(q @ d, d) - q


def _place(local, ia, ib, ta, tb, prev_centroid):
    a, b = local[ia], local[ib]
    ang = np.arctan2(tb[1] - ta[1], tb[0] - ta[0]) - np.arctan2(b[1] - a[1], b[0] - a[0])
    c, s = np.cos(ang), np.sin(ang)
    rot = np.array([[c, -s], [s, c]])
    placed = (local - a) @ rot.T + ta
    e = tb - ta

    def side(p):
        return e[0] * (p[1] - ta[1]) - e[1] * (p[0] - ta[0])

    if side(placed.mean(axis=0)) * side(prev_centroid) > 0:
        placed = _reflect_across(placed, ta, tb)
    return placed


def _shoelace(p: np.ndarray) -> float:
    x, y = p[:, 0], p[:, 1]
    return float(0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


def unfold_strip(verts, faces):
    """Lay the strip flat face by face. Returns (area_2d, layout)."""
    verts = np.asarray(verts, dtype=float)
    layout = []
    prev_face, prev_2d = None, None
    for f in faces:
        f = tuple(f)
        local = _flatten_face(verts[list(f)])
        if prev_face is None:
            placed = local
        else:
            shared = [v for v in f if v in prev_face]
            if len(shared) < 2:
                raise ValueError("consecutive faces must share an edge")
            a, b = shared[0], shared[1]
            placed = _place(local, f.index(a), f.index(b),
                            prev_2d[prev_face.index(a)], prev_2d[prev_face.index(b)],
                            prev_2d.mean(axis=0))
        layout.append(placed)
        prev_face, prev_2d = f, placed
    return float(sum(_shoelace(p) for p in layout)), layout
