"""Subdivide a loft into narrower strips (strakes) and chain lofts through sections."""
from __future__ import annotations

import numpy as np

from . import loft
from .errors import LoftError
from .rails import prepare_rails
from .types import LoftParams, Rail, StripResult


def mid_rails(rail_a: Rail, rail_b: Rail, rulings, count: int) -> list[np.ndarray]:
    """Polylines at fractions r/(count+1) along the chosen rulings, from A toward B."""
    ends_a = np.array([rail_a.points[i] for i, _ in rulings], dtype=float)
    ends_b = np.array([rail_b.points[j] for _, j in rulings], dtype=float)
    return [ends_a + (r / (count + 1)) * (ends_b - ends_a) for r in range(1, count + 1)]


def chain_loft(sections, params: LoftParams) -> list[StripResult]:
    """Loft each consecutive pair of (points, tangents) sections."""
    if len(sections) < 2:
        raise LoftError("need at least two sections to loft")
    strips = []
    for (pa, ta), (pb, tb) in zip(sections, sections[1:]):
        strips.append(loft(pa, pb, params, ta, tb))
    return strips


def subdivide_sections(points_a, points_b, tangents_a, tangents_b, params: LoftParams, rulings, strakes: int):
    """A, the strakes-1 mid-rails on the given rulings, and B, as (points, tangents) sections."""
    if strakes < 2:
        raise LoftError("strakes must be at least 2")
    rail_a, rail_b = prepare_rails(points_a, points_b, params.samples, tangents_a, tangents_b)
    mids = mid_rails(rail_a, rail_b, rulings, strakes - 1)
    return ([(np.asarray(points_a, dtype=float), tangents_a)]
            + [(m, None) for m in mids]
            + [(np.asarray(points_b, dtype=float), tangents_b)])


def subdivide(points_a, points_b, tangents_a, tangents_b, params: LoftParams, rulings, strakes: int):
    """Loft `strakes` narrower strips between A and B through mid-rails on the given rulings."""
    return chain_loft(subdivide_sections(points_a, points_b, tangents_a, tangents_b, params, rulings, strakes), params)


def find_strake_count(points_a, points_b, tangents_a, tangents_b, params: LoftParams, rulings,
                      max_strakes: int = 8):
    """Smallest strake count whose strips all pass the twist tolerance.

    Returns (count, worst_twist, strips); count is None when even max_strakes fails,
    in which case worst_twist and strips describe the max_strakes attempt.
    """
    worst, strips = float("inf"), []
    for strakes in range(2, max_strakes + 1):
        strips = subdivide(points_a, points_b, tangents_a, tangents_b, params, rulings, strakes)
        worst = max(s.report.max_twist for s in strips)
        if worst <= params.twist_tolerance:
            return strakes, worst, strips
    return None, worst, strips
