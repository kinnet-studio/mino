"""Rail relaxation: move rail B a bounded amount along the strip normal until the loft is developable."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .rails import central_difference, normalize_rows
from .twist import paired_twist
from .types import Rail

INF_TWIST = 90.0  # degrees counted for an invalid ruling inside the objective


def strip_normals_b(rail_a: Rail, rail_b: Rail, path) -> np.ndarray:
    """Unit normal per B point: T_B[j] x R for the first ruling in `path` that uses j."""
    n = len(rail_b.points)
    normals = np.zeros((n, 3))
    seen = np.zeros(n, dtype=bool)
    for i, j in path:
        if not seen[j]:
            seen[j] = True
            r = rail_b.points[j] - rail_a.points[i]
            normals[j] = normalize_rows(np.cross(rail_b.tangents[j], r))
    return normals


def relax_objective(delta, rail_a: Rail, rail_b: Rail, normals, path, target: float, smoothness: float):
    """(F, twists) for rail B moved by delta along `normals`, twist evaluated on the fixed `path`."""
    path = np.asarray(path, dtype=int)
    i, j = path[:, 0], path[:, 1]
    moved = rail_b.points + np.asarray(delta, dtype=float)[:, None] * normals
    tangents = normalize_rows(central_difference(moved))
    twists = paired_twist(rail_a.points[i], rail_a.tangents[i], moved[j], tangents[j])
    finite = np.where(np.isfinite(twists), twists, INF_TWIST)
    hinge = np.maximum(0.0, finite - target)
    smooth = float((np.diff(delta) ** 2).sum())
    return float((hinge ** 2).sum() + smoothness * smooth), twists


def numeric_gradient(f, x, free, h: float) -> np.ndarray:
    """Central-difference gradient of scalar f at x on the indices where `free` is True."""
    x = np.asarray(x, dtype=float)
    g = np.zeros_like(x)
    for k in np.flatnonzero(free):
        e = np.zeros_like(x)
        e[k] = h
        g[k] = (f(x + e) - f(x - e)) / (2.0 * h)
    return g
