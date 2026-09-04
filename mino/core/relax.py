"""Rail relaxation: move rail B a bounded amount along the strip normal until the loft is developable."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import loft
from .rails import central_difference, normalize_rows, prepare_rails
from .twist import paired_twist
from .types import LoftParams, Rail, StripResult

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


STALL_WINDOW = 20      # steps over which a relative decrease below STALL_REL counts as a stall
STALL_REL = 1e-6
MAX_HALVINGS = 30


@dataclass
class RelaxResult:
    points_b: np.ndarray
    delta: np.ndarray
    max_move_used: float
    mean_ruling: float
    twist_before: float
    twist_after: float
    result: StripResult
    iterations_run: int
    objective: list = field(default_factory=list)


def relax_rail_b(points_a, points_b, tangents_a, tangents_b, params: LoftParams | None = None,
                 max_move: float = 0.05, smoothness: float = 1.0, margin: float = 0.5,
                 iterations: int = 400, pin_endpoints: bool = True) -> RelaxResult:
    """Move rail B along the strip normal, bounded by max_move x mean ruling, to reduce twist."""
    params = params or LoftParams()
    base = loft(points_a, points_b, params, tangents_a, tangents_b)
    rail_a, rail_b = prepare_rails(points_a, points_b, params.samples, tangents_a, tangents_b)
    path = np.asarray(base.rulings, dtype=int)
    n = len(rail_b.points)

    lengths = np.linalg.norm(rail_b.points[path[:, 1]] - rail_a.points[path[:, 0]], axis=1)
    mean_ruling = float(lengths.mean())
    bound = max(0.0, float(max_move)) * mean_ruling
    normals = strip_normals_b(rail_a, rail_b, path)
    target = max(0.0, params.twist_tolerance - margin)

    free = np.ones(n, dtype=bool)
    if pin_endpoints:
        free[0] = free[-1] = False

    def objective(d):
        return relax_objective(d, rail_a, rail_b, normals, path, target, smoothness)[0]

    def project(d):
        d = np.clip(d, -bound, bound)
        d[~free] = 0.0
        return d

    delta = np.zeros(n)
    history = [objective(delta)]
    iterations_run = 0

    if bound > 0.0 and history[0] > 0.0:
        h = 1e-4 * bound
        grad = numeric_gradient(objective, delta, free, h)
        alpha = bound / max(float(np.linalg.norm(grad)), 1e-12)
        for it in range(iterations):
            if history[-1] == 0.0 or not np.any(grad):
                break
            step, accepted = alpha, None
            for _ in range(MAX_HALVINGS):
                cand = project(delta - step * grad)
                fc = objective(cand)
                if fc < history[-1]:
                    accepted = (cand, fc)
                    break
                step *= 0.5
            if accepted is None:
                break
            cand, fc = accepted
            new_grad = numeric_gradient(objective, cand, free, h)
            s, y = cand - delta, new_grad - grad
            sy = float(s @ y)
            alpha = float(s @ s) / sy if sy > 1e-18 else 2.0 * step
            delta, grad = cand, new_grad
            history.append(fc)
            iterations_run = it + 1
            if len(history) > STALL_WINDOW:
                before = history[-1 - STALL_WINDOW]
                if (before - history[-1]) / max(before, 1e-12) < STALL_REL:
                    break

    moved = rail_b.points + delta[:, None] * normals
    result = loft(points_a, moved, params, tangents_a, None)
    return RelaxResult(
        points_b=moved, delta=delta, max_move_used=float(np.abs(delta).max()),
        mean_ruling=mean_ruling, twist_before=base.report.max_twist,
        twist_after=result.report.max_twist, result=result,
        iterations_run=iterations_run, objective=history,
    )
