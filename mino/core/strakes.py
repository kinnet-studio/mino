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
    """Loft each consecutive pair of (points, tangents) sections.

    Shared rails (mid-rails between strakes) are pinned in both strips that share them,
    so only outer rails A and B may be nudged by planarization.
    """
    if len(sections) < 2:
        raise LoftError("need at least two sections to loft")
    strips = []
    for k, ((pa, ta), (pb, tb)) in enumerate(zip(sections, sections[1:])):
        pin_a = k > 0
        pin_b = k < len(sections) - 2
        strips.append(loft(pa, pb, params, ta, tb, pin_a=pin_a, pin_b=pin_b))
    return strips


def subdivide_sections(points_a, points_b, tangents_a, tangents_b, params: LoftParams, rulings, strakes: int):
    """A, the strakes-1 mid-rails on the given rulings, and B, as (points, tangents) sections."""
    if strakes < 2:
        raise LoftError("strakes must be at least 2")
    rail_a, rail_b = prepare_rails(points_a, points_b, params.samples, tangents_a, tangents_b, params.adaptive)
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

    Bracketed doubling: try 2, 4, 8, ... up to max_strakes. If none passes,
    return (None, worst at the largest tried, its strips). When a count passes,
    scan every count between the last failing doubling and the passing one in
    ascending order and return the first that passes, so the minimum is exact.
    Worst twist is not monotone in the strake count, so no bisection.
    """
    def attempt(strakes):
        strips = subdivide(points_a, points_b, tangents_a, tangents_b, params, rulings, strakes)
        return max(s.report.max_twist for s in strips), strips

    tol = params.twist_tolerance
    last_fail = 1
    strakes = 2
    worst, strips = float("inf"), []
    while True:
        strakes = min(strakes, max_strakes)
        worst, strips = attempt(strakes)
        if worst <= tol:
            for candidate in range(last_fail + 1, strakes):
                cand_worst, cand_strips = attempt(candidate)
                if cand_worst <= tol:
                    return candidate, cand_worst, cand_strips
            return strakes, worst, strips
        if strakes >= max_strakes:
            return None, worst, strips
        last_fail = strakes
        strakes *= 2
