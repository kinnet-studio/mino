"""Quad strips: give every ruling its own rail points by spreading fans."""
from __future__ import annotations

import numpy as np

from .rails import normalize_rows
from .twist import paired_twist
from .types import Rail

SPREAD = 0.25  # samples a fan's shared end may move along its rail


def spread_fans(path) -> np.ndarray:
    """Ruling ends as fractional sample indices, strictly increasing on both rails.

    In each run of identical single-rail steps (a fan), the end the run holds fixed is spread
    linearly from -SPREAD to +SPREAD; a run touching the first or last ruling spreads inward
    only, so the strip's corners stay put. Rulings outside fans keep their sample indices.
    """
    ends = np.asarray(path, dtype=float).copy()
    steps = np.diff(np.asarray(path, dtype=int), axis=0)
    last = len(steps)
    k = 0
    while k < last:
        step = steps[k]
        if step[0] == 1 and step[1] == 1:
            k += 1
            continue
        e = k
        while e + 1 < last and (steps[e + 1] == step).all():
            e += 1
        held = 0 if step[0] == 0 else 1
        lo = 0.0 if k == 0 else -SPREAD
        hi = 0.0 if e + 1 == last else SPREAD
        ends[k:e + 2, held] += np.linspace(lo, hi, e - k + 2)
        k = e + 1
    return ends


def rail_at(rail: Rail, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Points and unit tangents at fractional sample indices, linear between samples.

    Written as p[i]·(1−u) + p[i+1]·u so an integer index returns its sample exactly.
    """
    x = np.asarray(x, dtype=float)
    i = np.clip(np.floor(x).astype(int), 0, len(rail.points) - 2)
    u = (x - i)[:, None]
    points = rail.points[i] * (1.0 - u) + rail.points[i + 1] * u
    tangents = rail.tangents[i] * (1.0 - u) + rail.tangents[i + 1] * u
    return points, normalize_rows(tangents)


def quad_strip(rail_a: Rail, rail_b: Rail, path) -> tuple[np.ndarray, list, np.ndarray]:
    """Vertices (A then B), one quad per pair of consecutive rulings, and each ruling's twist."""
    ends = spread_fans(path)
    pa, ta = rail_at(rail_a, ends[:, 0])
    pb, tb = rail_at(rail_b, ends[:, 1])
    k = len(ends)
    faces = [(m, m + 1, k + m + 1, k + m) for m in range(k - 1)]
    return np.vstack([pa, pb]), faces, paired_twist(pa, ta, pb, tb)
