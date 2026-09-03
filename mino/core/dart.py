"""Dart (gusset) proposal and refined mesh for a non-developable region.

The refined grid lies on the ruled surface spanned by the chosen rulings. A
ruled surface has non-positive Gaussian curvature, so the wedge is normally
negative: a gusset to insert, not a dart to remove.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .mesh import quad_planarity
from .types import Rail


@dataclass
class DartProposal:
    range: tuple            # failing ruling range (k1, k2)
    ruling: int             # cut ruling: max twist within the range
    wedge_deg: float        # positive = dart (remove wedge), negative = gusset (insert)
    kind: str               # "dart" | "gusset"


@dataclass
class DartMesh:
    verts: np.ndarray
    faces: list
    face_twist: np.ndarray
    seam_edges: list        # (v_from, v_to) pairs, a path from the cut's open end to its tip


def refined_grid(rail_a: Rail, rail_b: Rail, rulings, mid_rails: int) -> np.ndarray:
    rows = mid_rails + 2
    A = np.array([rail_a.points[i] for i, _ in rulings], dtype=float)
    B = np.array([rail_b.points[j] for _, j in rulings], dtype=float)
    t = np.linspace(0.0, 1.0, rows)[:, None, None]
    return A[None] + t * (B - A)[None]


def _corner_angle(p, q, r) -> float:
    u, v = q - p, r - p
    lu, lv = np.linalg.norm(u), np.linalg.norm(v)
    if lu < 1e-12 or lv < 1e-12:
        return 0.0
    return float(np.arccos(np.clip(np.dot(u, v) / (lu * lv), -1.0, 1.0)))


def _tri_area(P, tri) -> float:
    a, b, c = (P[v] for v in tri)
    return 0.5 * float(np.linalg.norm(np.cross(b - a, c - a)))


def _quad_triangles(P, quad):
    """Two triangles along the shorter diagonal, degenerate ones dropped."""
    v0, v1, v2, v3 = quad
    if np.linalg.norm(P[v2] - P[v0]) <= np.linalg.norm(P[v3] - P[v1]):
        cand = [(v0, v1, v2), (v0, v2, v3)]
    else:
        cand = [(v0, v1, v3), (v1, v2, v3)]
    return [t for t in cand if _tri_area(P, t) > 1e-14]


def _grid_quads(rows, cols):
    idx = lambda r, c: r * cols + c
    return [(idx(r, c), idx(r, c + 1), idx(r + 1, c + 1), idx(r + 1, c))
            for r in range(rows - 1) for c in range(cols - 1)]


def angle_deficit_total(grid: np.ndarray, cols_range=None) -> float:
    """Sum of 2π minus corner-angle sums over interior grid vertices (radians)."""
    rows, cols, _ = grid.shape
    c0, c1 = (0, cols - 1) if cols_range is None else cols_range
    P = grid.reshape(-1, 3)
    sums = np.zeros(rows * cols)
    for quad in _grid_quads(rows, cols):
        for a, b, c in _quad_triangles(P, quad):
            sums[a] += _corner_angle(P[a], P[b], P[c])
            sums[b] += _corner_angle(P[b], P[c], P[a])
            sums[c] += _corner_angle(P[c], P[a], P[b])
    total = 0.0
    for r in range(1, rows - 1):
        for c in range(c0 + 1, c1):
            total += 2.0 * np.pi - sums[r * cols + c]
    return float(total)


def dart_proposal(rail_a, rail_b, rulings, ruling_twist, failing_range, mid_rails: int = 3) -> DartProposal:
    k1, k2 = failing_range
    tw = np.asarray(ruling_twist, dtype=float)[k1:k2 + 1]
    tw = np.where(np.isfinite(tw), tw, -1.0)
    ruling = k1 + int(np.argmax(tw))
    grid = refined_grid(rail_a, rail_b, rulings, mid_rails)
    cols = grid.shape[1]
    theta = angle_deficit_total(grid, (max(k1 - 1, 0), min(k2 + 1, cols - 1)))
    wedge = float(np.degrees(theta))
    return DartProposal(range=(k1, k2), ruling=ruling, wedge_deg=wedge,
                        kind="dart" if wedge >= 0.0 else "gusset")


def _merge_duplicates(P, faces, face_twist, edges):
    """Merge coincident vertices; drop faces that collapse and zero-length edges."""
    key_to_new, old_to_new, verts = {}, [], []
    for p in P:
        key = tuple(np.round(p, 9))
        if key not in key_to_new:
            key_to_new[key] = len(verts)
            verts.append(p)
        old_to_new.append(key_to_new[key])
    out_faces, out_twist = [], []
    for f, tw in zip(faces, face_twist):
        mapped = []
        for v in f:
            nv = old_to_new[v]
            if not mapped or mapped[-1] != nv:
                mapped.append(nv)
        if len(mapped) > 1 and mapped[0] == mapped[-1]:
            mapped.pop()
        if len(set(mapped)) >= 3:
            out_faces.append(tuple(mapped))
            out_twist.append(tw)
    out_edges = [(old_to_new[a], old_to_new[b]) for a, b in edges if old_to_new[a] != old_to_new[b]]
    return np.array(verts, dtype=float), out_faces, np.array(out_twist, dtype=float), out_edges


def dart_mesh(rail_a, rail_b, rulings, ruling_twist, proposal: DartProposal, mid_rails: int = 3,
              planar_tolerance: float = 0.01, dart_from: str = "B") -> DartMesh:
    grid = refined_grid(rail_a, rail_b, rulings, mid_rails)
    rows, cols, _ = grid.shape
    P = grid.reshape(-1, 3)
    idx = lambda r, c: r * cols + c
    tw = np.asarray(ruling_twist, dtype=float)
    faces, face_twist = [], []
    for r in range(rows - 1):
        for c in range(cols - 1):
            quad = (idx(r, c), idx(r, c + 1), idx(r + 1, c + 1), idx(r + 1, c))
            t = float(max(tw[c], tw[c + 1]))
            corners = [P[v] for v in quad]
            distinct = len({tuple(np.round(p, 9)) for p in corners})
            if distinct < 3:
                continue
            if distinct == 4 and quad_planarity(P, quad) <= planar_tolerance:
                faces.append(quad)
                face_twist.append(t)
            else:
                for tri in _quad_triangles(P, quad):
                    faces.append(tri)
                    face_twist.append(t)
    k = proposal.ruling
    if dart_from == "B":
        seam = [(idx(r, k), idx(r - 1, k)) for r in range(rows - 1, 1, -1)]
    else:
        seam = [(idx(r, k), idx(r + 1, k)) for r in range(0, rows - 2)]
    verts, faces, face_twist, seam = _merge_duplicates(P, faces, face_twist, seam)
    return DartMesh(verts=verts, faces=faces, face_twist=face_twist, seam_edges=seam)
