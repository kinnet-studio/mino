"""Rail preparation: dedupe, arc-length resampling, tangents, orientation."""
from __future__ import annotations

import numpy as np

from .errors import LoftError
from .types import Rail


def normalize_rows(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=float)
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    n = np.where(n < 1e-12, 1.0, n)
    return v / n


def dedupe_indices(points: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    keep = [0]
    for k in range(1, len(points)):
        if np.linalg.norm(points[k] - points[keep[-1]]) > eps:
            keep.append(k)
    return np.array(keep, dtype=int)


def central_difference(points: np.ndarray) -> np.ndarray:
    """Tangent directions scaled like 2h * f': central inside, three-point one-sided at the ends.

    The second-order end formula keeps the end tangent of a curved rail within a
    fraction of a degree; the plain chord was off by half a sample step.
    """
    t = np.empty_like(points)
    t[1:-1] = points[2:] - points[:-2]
    if len(points) < 3:
        t[0] = t[-1] = points[-1] - points[0]
        return t
    t[0] = -3.0 * points[0] + 4.0 * points[1] - points[2]
    t[-1] = 3.0 * points[-1] - 4.0 * points[-2] + points[-3]
    return t


def resample(points, samples: int, tangents=None) -> Rail:
    if samples < 2:
        raise LoftError("samples must be at least 2")
    pts = np.asarray(points, dtype=float).reshape(-1, 3)
    if len(pts) == 0:
        raise LoftError("a rail needs at least 2 distinct points")
    keep = dedupe_indices(pts)
    pts = pts[keep]
    if len(pts) < 2:
        raise LoftError("a rail needs at least 2 distinct points")
    tans = None
    if tangents is not None:
        tans = np.asarray(tangents, dtype=float).reshape(-1, 3)
        if len(tans) != len(np.asarray(points, dtype=float).reshape(-1, 3)):
            raise LoftError("tangents must match points one-to-one")
        tans = tans[keep]

    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    targets = np.linspace(0.0, cum[-1], samples)
    out = np.column_stack([np.interp(targets, cum, pts[:, k]) for k in range(3)])
    if tans is None:
        t = central_difference(out)
    else:
        t = np.column_stack([np.interp(targets, cum, tans[:, k]) for k in range(3)])
    return Rail(points=out, tangents=normalize_rows(t))


def prepare_rails(points_a, points_b, samples: int, tangents_a=None, tangents_b=None):
    """Resample both rails to `samples` points, reversing B if it runs opposite to A."""
    pa = np.asarray(points_a, dtype=float).reshape(-1, 3)
    pb = np.asarray(points_b, dtype=float).reshape(-1, 3)
    if len(pa) < 2 or len(pb) < 2:
        raise LoftError("each rail needs at least 2 points")
    tb = None if tangents_b is None else np.asarray(tangents_b, dtype=float).reshape(-1, 3)
    if np.linalg.norm(pb[-1] - pa[0]) < np.linalg.norm(pb[0] - pa[0]):
        pb = pb[::-1].copy()
        if tb is not None:
            tb = -tb[::-1]
    return resample(pa, samples, tangents_a), resample(pb, samples, tb)
