"""Monotone (non-crossing) ruling selection by dynamic programming."""
from __future__ import annotations

import numpy as np

from .errors import LoftError
from .rails import normalize_rows
from .twist import ruling_lengths
from .types import Rail

STEP_PENALTY = 0.5  # degrees, added to non-diagonal moves


def tie_matrix(rail_a: Rail, rail_b: Rail, mode: str, plane_normal) -> np.ndarray:
    n = len(rail_a.points)
    if mode == "none":
        return np.zeros((n, n))
    L = ruling_lengths(rail_a, rail_b)
    if mode == "shortest":
        mean = float(np.mean(np.diagonal(L)))
        return L / (mean if mean > 1e-12 else 1.0)
    if mode == "plane":
        nrm = normalize_rows(np.asarray(plane_normal, dtype=float).reshape(3))
        R = rail_b.points[None, :, :] - rail_a.points[:, None, :]
        Ls = np.where(L < 1e-12, 1.0, L)
        Rn = R / Ls[..., None]
        return 1.0 - np.abs(Rn @ nrm)
    raise LoftError(f"unknown tie_breaker {mode!r}")


def _explain_failure(cost: np.ndarray, D: np.ndarray) -> str:
    finite_rows = np.isfinite(cost).any(axis=1)
    if not finite_rows.all():
        i = int(np.flatnonzero(~finite_rows)[0])
        return (f"no valid ruling from rail A point {i} within the window "
                f"(ruling parallel to a tangent or zero length); widen the window or check the rails")
    # Find the first row where no monotone path reaches
    reachable_rows = np.isfinite(D).any(axis=1)
    if not reachable_rows.all():
        i = int(np.flatnonzero(~reachable_rows)[0])
        return f"no monotone ruling path reaches rail A point {i} within the window; try a larger window"
    return "no monotone ruling path within the window; try a larger window"


def align(cost: np.ndarray) -> list[tuple[int, int]]:
    """Cheapest monotone path from (0,0) to (N-1,M-1) using moves (1,1), (1,0), (0,1)."""
    n, m = cost.shape
    D = np.full((n, m), np.inf)
    back = np.full((n, m), -1, dtype=np.int8)  # 0 diag, 1 from (i-1,j), 2 from (i,j-1)
    D[0, 0] = cost[0, 0]
    for i in range(n):
        for j in np.flatnonzero(np.isfinite(cost[i])):
            if i == 0 and j == 0:
                continue
            best, arg = np.inf, -1
            if i > 0 and j > 0 and D[i - 1, j - 1] < best:
                best, arg = D[i - 1, j - 1], 0
            if i > 0 and D[i - 1, j] + STEP_PENALTY < best:
                best, arg = D[i - 1, j] + STEP_PENALTY, 1
            if j > 0 and D[i, j - 1] + STEP_PENALTY < best:
                best, arg = D[i, j - 1] + STEP_PENALTY, 2
            if np.isfinite(best):
                D[i, j] = cost[i, j] + best
                back[i, j] = arg
    if not np.isfinite(D[n - 1, m - 1]):
        raise LoftError(_explain_failure(cost, D))
    path = []
    i, j = n - 1, m - 1
    while True:
        path.append((i, j))
        if i == 0 and j == 0:
            break
        b = back[i, j]
        if b == 0:
            i, j = i - 1, j - 1
        elif b == 1:
            i -= 1
        else:
            j -= 1
    path.reverse()
    return path
