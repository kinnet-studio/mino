"""Rail preparation: dedupe, arc-length resampling, tangents, orientation."""
from __future__ import annotations

import numpy as np

from .errors import LoftError
from .types import Rail

FLOAT32_EPS = 2.0 ** -23  # relative precision of the float32 coordinates Blender evaluates curves in
BEND_NOISE = 0.01         # radians of summed rounding noise the bending measure may pick up
STRAIGHT_TURNING = 0.05   # radians of summed turning below which both rails count as straight


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


def bend_profile(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Cumulative turning (radians) along a deduplicated polyline, at normalized arc length.

    Knots sit at both ends and at every segment midpoint, so the turning at a vertex is
    spread over the two half-segments that meet there.
    """
    pts = np.asarray(points, dtype=float)
    seg = np.diff(pts, axis=0)
    lengths = np.linalg.norm(seg, axis=1)
    cum = np.concatenate([[0.0], np.cumsum(lengths)])
    d = seg / lengths[:, None]
    turn = np.arccos(np.clip(np.einsum("ij,ij->i", d[:-1], d[1:]), -1.0, 1.0))
    knots = np.concatenate([[0.0], 0.5 * (cum[:-1] + cum[1:]), [cum[-1]]]) / cum[-1]
    theta = np.concatenate([[0.0, 0.0], np.cumsum(turn), [turn.sum()]])
    return knots, theta


def bend_points(points: np.ndarray, samples: int) -> int:
    """How many evenly spaced points to measure a deduplicated rail's bending on.

    Four per sample resolves where bends start and end. Coordinates rounded to float32 read as
    about FLOAT32_EPS x scale / s of turning per vertex at spacing s, which sums to
    FLOAT32_EPS x scale x length / s^2 along the rail (radians of it on a straight Bezier rail at
    16x density); the spacing stays coarse enough that this is at most BEND_NOISE.
    """
    length = float(np.linalg.norm(np.diff(points, axis=0), axis=1).sum())
    scale = float(np.abs(points).max())
    noise_cap = np.sqrt(length * BEND_NOISE / (FLOAT32_EPS * scale))
    return int(min(max(noise_cap, 16), 4 * samples))


def _rail_bend(points, samples: int) -> tuple[np.ndarray, np.ndarray]:
    pts = np.asarray(points, dtype=float).reshape(-1, 3)
    pts = pts[dedupe_indices(pts)]
    if len(pts) < 2:
        raise LoftError("a rail needs at least 2 distinct points")
    return bend_profile(resample(pts, bend_points(pts, samples)).points)


def shared_positions(points_a, points_b, samples: int, adaptive: float) -> np.ndarray:
    """Normalized positions for both rails: a share `adaptive` follows their summed bending.

    Bending is measured on each rail resampled evenly to bend_points(...) points, so rounding in
    the input is not mistaken for bending; below STRAIGHT_TURNING in all the rails count as straight.
    """
    ta, va = _rail_bend(points_a, samples)
    tb, vb = _rail_bend(points_b, samples)
    total = va[-1] + vb[-1]
    if total < STRAIGHT_TURNING:
        return np.linspace(0.0, 1.0, samples)
    t = np.union1d(ta, tb)
    u = (1.0 - adaptive) * t + adaptive * (np.interp(t, ta, va) + np.interp(t, tb, vb)) / total
    out = np.interp(np.linspace(0.0, 1.0, samples), u, t)
    out[0], out[-1] = 0.0, 1.0
    return out


def resample(points, samples: int, tangents=None, positions=None) -> Rail:
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
    if positions is None:
        targets = np.linspace(0.0, cum[-1], samples)
    else:
        targets = np.asarray(positions, dtype=float) * cum[-1]
    out = np.column_stack([np.interp(targets, cum, pts[:, k]) for k in range(3)])
    if tans is None:
        t = central_difference(out)
    else:
        t = np.column_stack([np.interp(targets, cum, tans[:, k]) for k in range(3)])
    return Rail(points=out, tangents=normalize_rows(t))


def prepare_rails(points_a, points_b, samples: int, tangents_a=None, tangents_b=None, adaptive: float = 0.0):
    """Resample both rails to `samples` points, reversing B if it runs opposite to A.

    adaptive > 0 places that share of the samples by the rails' summed bending, at the same
    normalized positions on both rails; 0 spaces them evenly by arc length.
    """
    if not 0.0 <= adaptive < 1.0:
        raise LoftError("adaptive must be at least 0 and below 1")
    pa = np.asarray(points_a, dtype=float).reshape(-1, 3)
    pb = np.asarray(points_b, dtype=float).reshape(-1, 3)
    if len(pa) < 2 or len(pb) < 2:
        raise LoftError("each rail needs at least 2 points")
    tb = None if tangents_b is None else np.asarray(tangents_b, dtype=float).reshape(-1, 3)
    if np.linalg.norm(pb[-1] - pa[0]) < np.linalg.norm(pb[0] - pa[0]):
        pb = pb[::-1].copy()
        if tb is not None:
            tb = -tb[::-1]
    positions = shared_positions(pa, pb, samples, adaptive) if adaptive > 0 else None
    return resample(pa, samples, tangents_a, positions), resample(pb, samples, tb, positions)
