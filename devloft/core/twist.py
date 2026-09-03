"""Twist scoring of candidate rulings between two rails."""
from __future__ import annotations

import numpy as np

from .types import Rail

MIN_RULING_LENGTH = 1e-9
MIN_SIN = float(np.sin(np.radians(1.0)))  # ruling within 1 deg of a tangent is invalid


def ruling_lengths(rail_a: Rail, rail_b: Rail) -> np.ndarray:
    R = rail_b.points[None, :, :] - rail_a.points[:, None, :]
    return np.linalg.norm(R, axis=2)


def twist_matrix(rail_a: Rail, rail_b: Rail, window: int) -> np.ndarray:
    """Twist angle in degrees for every (i, j) ruling from A[i] to B[j].

    twist = angle between n_A = T_A x R and n_B = T_B x R, folded into [0, 90].
    Entries are inf when the ruling is degenerate, nearly parallel to a tangent,
    or outside the band |i - j| <= window.
    """
    A = rail_a.points[:, None, :]
    B = rail_b.points[None, :, :]
    R = B - A
    L = np.linalg.norm(R, axis=2)
    Ls = np.where(L < MIN_RULING_LENGTH, 1.0, L)
    Rn = R / Ls[..., None]

    TA = np.broadcast_to(rail_a.tangents[:, None, :], R.shape)
    TB = np.broadcast_to(rail_b.tangents[None, :, :], R.shape)
    nA = np.cross(TA, Rn)
    nB = np.cross(TB, Rn)
    lA = np.linalg.norm(nA, axis=2)
    lB = np.linalg.norm(nB, axis=2)
    bad_angle = (lA < MIN_SIN) | (lB < MIN_SIN)
    denom = np.where(bad_angle, 1.0, lA * lB)
    cosang = np.abs(np.einsum("ijk,ijk->ij", nA, nB)) / denom
    twist = np.degrees(np.arccos(np.clip(cosang, 0.0, 1.0)))

    i = np.arange(len(rail_a.points))[:, None]
    j = np.arange(len(rail_b.points))[None, :]
    invalid = (L < MIN_RULING_LENGTH) | bad_angle | (np.abs(i - j) > window)
    twist[invalid] = np.inf
    return twist
