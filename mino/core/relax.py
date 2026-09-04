"""Rail relaxation: move rail B a bounded amount along the strip normal until the loft is developable."""
from __future__ import annotations

import time
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


def moved_tangents_b(rail_b: Rail, moved: np.ndarray) -> np.ndarray:
    """Tangents of the moved rail: the supplied tangents plus the central-difference change.

    Returns rail_b.tangents exactly when nothing moved, so exact (Bezier) tangents are kept;
    when rail_b.tangents came from central differences this reduces to those of `moved`.
    """
    before = normalize_rows(central_difference(rail_b.points))
    after = normalize_rows(central_difference(moved))
    return normalize_rows(rail_b.tangents + after - before)


def relax_objective(delta, rail_a: Rail, rail_b: Rail, normals, path, target: float, smoothness: float,
                    length_scale: float = 1.0):
    """(F, twists) for rail B moved by delta along `normals`, twist evaluated on the fixed `path`.

    smoothness is dimensionless: the move differences are measured in units of `length_scale`.
    """
    path = np.asarray(path, dtype=int)
    i, j = path[:, 0], path[:, 1]
    moved = rail_b.points + np.asarray(delta, dtype=float)[:, None] * normals
    tangents = moved_tangents_b(rail_b, moved)
    twists = paired_twist(rail_a.points[i], rail_a.tangents[i], moved[j], tangents[j])
    finite = np.where(np.isfinite(twists), twists, INF_TWIST)
    hinge = np.maximum(0.0, finite - target)
    smooth = smoothness * float((np.diff(delta) ** 2).sum()) / length_scale ** 2
    return float((hinge ** 2).sum() + smooth), twists


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
    tangents_b: np.ndarray  # tangents of the moved rail B, (N, 3)
    delta: np.ndarray
    max_move_used: float
    mean_ruling: float
    twist_before: float
    twist_after: float
    result: StripResult
    iterations_run: int
    objective: list = field(default_factory=list)
    stop_reason: str = "noop"  # noop | converged | stalled | line_search | iterations | time


def relax_rail_b(points_a, points_b, tangents_a, tangents_b, params: LoftParams | None = None,
                 max_move: float = 0.05, smoothness: float = 1.0, margin: float = 0.5,
                 iterations: int = 800, pin_endpoints: bool = True,
                 max_seconds: float = 0.0, progress=None) -> RelaxResult:
    """Move rail B along the strip normal, bounded by max_move x mean ruling, to reduce twist.

    max_seconds > 0 stops the loop once that wall-clock budget is spent (0 means no limit);
    progress(step, iterations) is called after every accepted step.
    """
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

    length_scale = mean_ruling if mean_ruling > 0 else 1.0

    def objective(d):
        return relax_objective(d, rail_a, rail_b, normals, path, target, smoothness,
                               length_scale=length_scale)[0]

    def project(d):
        d = np.clip(d, -bound, bound)
        d[~free] = 0.0
        return d

    delta = np.zeros(n)
    history = [objective(delta)]
    iterations_run = 0
    stop_reason = "noop"

    if bound > 0.0 and history[0] > 0.0:
        deadline = time.monotonic() + max_seconds if max_seconds > 0 else None
        h = 1e-4 * bound
        grad = numeric_gradient(objective, delta, free, h)
        alpha = bound / max(float(np.linalg.norm(grad)), 1e-12)
        stop_reason = "iterations"
        for it in range(iterations):
            if history[-1] == 0.0 or not np.any(grad):
                stop_reason = "converged"
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
                stop_reason = "line_search"
                break
            cand, fc = accepted
            new_grad = numeric_gradient(objective, cand, free, h)
            s, y = cand - delta, new_grad - grad
            sy = float(s @ y)
            # Spec says fall back when s.y <= 0; a relative epsilon also rejects a near-zero
            # denominator that would produce a step no amount of halving walks back.
            if sy > 1e-12 * float(np.linalg.norm(s)) * float(np.linalg.norm(y)):
                alpha = float(s @ s) / sy
            else:
                alpha = 2.0 * step
            delta, grad = cand, new_grad
            history.append(fc)
            iterations_run = it + 1
            if progress is not None:
                progress(iterations_run, iterations)
            if fc == 0.0:
                stop_reason = "converged"
                break
            if len(history) > STALL_WINDOW:
                before = history[-1 - STALL_WINDOW]
                if (before - history[-1]) / max(before, 1e-12) < STALL_REL:
                    stop_reason = "stalled"
                    break
            if deadline is not None and time.monotonic() > deadline:
                stop_reason = "time"
                break

    moved = rail_b.points + delta[:, None] * normals
    tangents_b = moved_tangents_b(rail_b, moved)
    result = loft(points_a, moved, params, tangents_a, tangents_b)
    return RelaxResult(
        points_b=moved, tangents_b=tangents_b, delta=delta, max_move_used=float(np.abs(delta).max()),
        mean_ruling=mean_ruling, twist_before=base.report.max_twist,
        twist_after=result.report.max_twist, result=result,
        iterations_run=iterations_run, objective=history, stop_reason=stop_reason,
    )
