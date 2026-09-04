"""Twist scoring of candidate rulings between two rails."""
from __future__ import annotations

import numpy as np

from .types import Rail

MIN_RULING_LENGTH = 1e-9
MIN_SIN = float(np.sin(np.radians(1.0)))  # ruling within 1 deg of a tangent is invalid


def ruling_lengths(rail_a: Rail, rail_b: Rail) -> np.ndarray:
    R = rail_b.points[None, :, :] - rail_a.points[:, None, :]
    return np.linalg.norm(R, axis=2)


def paired_twist(points_a, tangents_a, points_b, tangents_b) -> np.ndarray:
    """Twist in degrees for K rulings pairing row k of A with row k of B.

    twist = angle between n_A = T_A x R and n_B = T_B x R, folded into [0, 90].
    inf when the ruling is degenerate or nearly parallel to a tangent.
    """
    R = np.asarray(points_b, dtype=float) - np.asarray(points_a, dtype=float)
    L = np.linalg.norm(R, axis=1)
    Rn = R / np.where(L < MIN_RULING_LENGTH, 1.0, L)[:, None]
    nA = np.cross(np.asarray(tangents_a, dtype=float), Rn)
    nB = np.cross(np.asarray(tangents_b, dtype=float), Rn)
    lA = np.linalg.norm(nA, axis=1)
    lB = np.linalg.norm(nB, axis=1)
    bad_angle = (lA < MIN_SIN) | (lB < MIN_SIN)
    denom = np.where(bad_angle, 1.0, lA * lB)
    cosang = np.abs(np.einsum("ij,ij->i", nA, nB)) / denom
    twist = np.degrees(np.arccos(np.clip(cosang, 0.0, 1.0)))
    twist[(L < MIN_RULING_LENGTH) | bad_angle] = np.inf
    return twist


def twist_matrix(rail_a: Rail, rail_b: Rail, window: int) -> np.ndarray:
    """Twist angle in degrees for every (i, j) ruling from A[i] to B[j].

    Entries are inf when the ruling is degenerate, nearly parallel to a tangent,
    or outside the band |i - j| <= window.
    """
    na, nb = len(rail_a.points), len(rail_b.points)
    ii, jj = np.meshgrid(np.arange(na), np.arange(nb), indexing="ij")
    flat = paired_twist(rail_a.points[ii.ravel()], rail_a.tangents[ii.ravel()],
                        rail_b.points[jj.ravel()], rail_b.tangents[jj.ravel()])
    twist = flat.reshape(na, nb)
    twist[np.abs(ii - jj) > window] = np.inf
    return twist
