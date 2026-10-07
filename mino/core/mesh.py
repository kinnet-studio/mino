"""Face construction, planarity metric, planarize pass, and quad splitting."""
from __future__ import annotations

import numpy as np

# Planarize keeps each rail smooth: at every rail vertex, the move leaves the spacing-weighted
# interpolation of its neighbours' moves by at most MAX_KINK x h_l h_r / (h_l + h_r), h_l and h_r the
# rail segments meeting there. With even spacing h that is |d[v-1] - 2 d[v] + d[v+1]| <= MAX_KINK h,
# and either way about 6 degrees of added bend.
MAX_KINK = 0.1


def build_faces(path, n_a: int):
    faces = []
    for (i, j), (i2, j2) in zip(path, path[1:]):
        if i2 == i + 1 and j2 == j + 1:
            faces.append((i, i + 1, n_a + j + 1, n_a + j))
        elif i2 == i + 1:
            faces.append((i, i + 1, n_a + j))
        else:
            faces.append((i, n_a + j + 1, n_a + j))
    return faces


def quad_planarity(verts: np.ndarray, face) -> float:
    """Closest-approach distance of the two diagonals over their mean length."""
    p0, p1, p2, p3 = verts[list(face)]
    d1 = p2 - p0
    d2 = p3 - p1
    n = np.cross(d1, d2)
    ln = np.linalg.norm(n)
    mean = 0.5 * (np.linalg.norm(d1) + np.linalg.norm(d2))
    if ln < 1e-12 or mean < 1e-12:
        return 0.0
    return float(abs(np.dot(p1 - p0, n)) / ln / mean)


def face_planarity(verts: np.ndarray, faces) -> np.ndarray:
    return np.array([quad_planarity(verts, f) if len(f) == 4 else 0.0 for f in faces], dtype=float)


def limit_kinks(disp, points, fixed, max_nudge, sweeps=50):
    """Move one rail's displacements so no free vertex bends the rail by more than about 6 degrees,
    and |disp[v]| <= max_nudge everywhere.

    points are the rail's vertices before the move; their spacing weighs the bound (see MAX_KINK),
    so tightly packed vertices, as in a spread fan, cannot zigzag. disp is (m, k): k = 3 for free 3D
    moves, k = 1 for distances along fixed directions. In place.
    Each kink is fixed by moving its own vertex toward the interpolation of its neighbours. Kinks of
    one parity move disjoint vertices, so each half-sweep is an exact projection; alternating the
    halves with the nudge cap converges because disp = 0 satisfies every bound.
    """
    m = len(disp)
    seg = np.linalg.norm(np.diff(np.asarray(points, dtype=float), axis=0), axis=1)
    for _ in range(sweeps):
        worst = 0.0
        for start in (1, 2):
            v = np.arange(start, m - 1, 2)
            v = v[~fixed[v]]
            hl, hr = seg[v - 1], seg[v]
            k = (hr[:, None] * disp[v - 1] + hl[:, None] * disp[v + 1]) / (hl + hr)[:, None] - disp[v]
            kn = np.linalg.norm(k, axis=1)
            limit = MAX_KINK * hl * hr / (hl + hr)
            over = kn > limit
            if over.any():
                v, k, kn, lim = v[over], k[over], kn[over], limit[over]
                worst = max(worst, float(((kn - lim) / lim).max()))
                disp[v] += k * (1.0 - lim / kn)[:, None]
        d = np.linalg.norm(disp, axis=1)
        out = d > max_nudge
        if out.any():
            worst = max(worst, float(((d[out] - max_nudge) / d[out]).max()))
            disp[out] *= (max_nudge / d[out])[:, None]
        if worst < 1e-9:
            break
    return disp


def planarize(verts, faces, pinned, tolerance, iterations, max_nudge, rails=()):
    """Move each vertex to the mean of its quads' projections onto their own planes.

    rails: index chains (rail A, rail B) whose displacements must stay smooth: at any vertex the
    move adds at most about 6 degrees of bend (MAX_KINK, weighted by spacing). Without this, a vertex in a
    single quad (next to a triangle) takes that quad's whole correction on every iteration and
    spikes out of the surface until max_nudge stops it.

    Returns the iterate (the input included) with the fewest quads over tolerance, then the lowest
    total planarity, so flattening one quad never buys a split elsewhere.
    """
    verts = np.asarray(verts, dtype=float).copy()
    orig = verts.copy()
    quads = [f for f in faces if len(f) == 4]
    pinned = list(pinned)
    if len(set(pinned)) >= len(verts):
        return verts
    fixed = np.zeros(len(verts), dtype=bool)
    fixed[pinned] = True
    chains = [np.asarray(list(r), dtype=int) for r in rails]
    chains = [chain for chain in chains if len(chain) >= 3]
    pl = face_planarity(verts, quads)
    best, best_key = verts, (int((pl > tolerance).sum()), float(pl.sum()))
    for _ in range(iterations):
        if len(pl) == 0 or pl.max() <= tolerance:
            break
        acc = np.zeros_like(verts)
        cnt = np.zeros(len(verts))
        for f in quads:
            idx = list(f)
            P = verts[idx]
            n = np.cross(P[2] - P[0], P[3] - P[1])
            ln = np.linalg.norm(n)
            if ln < 1e-12:
                continue
            n = n / ln
            c = P.mean(axis=0)
            proj = P - np.outer((P - c) @ n, n)
            acc[idx] += proj
            cnt[idx] += 1
        moved = cnt > 0
        new = verts.copy()
        new[moved] = acc[moved] / cnt[moved, None]
        new[pinned] = orig[pinned]
        disp = new - orig
        d = np.linalg.norm(disp, axis=1)
        over = d > max_nudge
        if over.any():
            new[over] = orig[over] + disp[over] * (max_nudge / d[over])[:, None]
        for chain in chains:
            disp = limit_kinks(new[chain] - orig[chain], orig[chain], fixed[chain], max_nudge)
            new[chain] = orig[chain] + disp
        verts = new
        pl = face_planarity(verts, quads)
        key = (int((pl > tolerance).sum()), float(pl.sum()))
        if key < best_key:
            best, best_key = verts, key
    return best


def _tri_normal(verts, tri):
    p0, p1, p2 = verts[list(tri)]
    n = np.cross(p1 - p0, p2 - p0)
    ln = np.linalg.norm(n)
    return n / ln if ln > 1e-12 else np.zeros(3)


def _dihedral(verts, tris):
    n0, n1 = _tri_normal(verts, tris[0]), _tri_normal(verts, tris[1])
    if not n0.any() or not n1.any():
        # A degenerate (zero-area) triangle has no normal; treat this
        # diagonal option as maximally bad so it never wins over a
        # non-degenerate alternative.
        return float("inf")
    return float(np.degrees(np.arccos(np.clip(np.dot(n0, n1), -1.0, 1.0))))


def _option_a(face):
    v0, v1, v2, v3 = face
    return [(v0, v2, v3), (v0, v1, v2)]


def _option_b(face):
    v0, v1, v2, v3 = face
    return [(v0, v1, v3), (v1, v2, v3)]


def best_diagonal(verts, face):
    """Pick the flatter diagonal of a strip quad `(v0, v1, v2, v3) = (A[i], A[i+1], B[j+1], B[j])`.

    Ordering rule: the first emitted triangle must contain the incoming
    ruling `(v0, v3)` and the second must contain the outgoing ruling
    `(v1, v2)`, so consecutive faces in the strip keep sharing an edge after
    a split. Winding stays consistent between the two options.
    """
    opt_a, opt_b = _option_a(face), _option_b(face)
    return opt_a if _dihedral(verts, opt_a) <= _dihedral(verts, opt_b) else opt_b


def _consistent_choices(verts, faces, needs_split):
    """One diagonal orientation per run of consecutive split quads."""
    choice = {}
    k = 0
    while k < len(faces):
        if not needs_split[k]:
            k += 1
            continue
        run = [k]
        while k + 1 < len(faces) and needs_split[k + 1]:
            k += 1
            run.append(k)
        cost_a = sum(_dihedral(verts, _option_a(faces[q])) for q in run)
        cost_b = sum(_dihedral(verts, _option_b(faces[q])) for q in run)
        if not np.isfinite(cost_a) and not np.isfinite(cost_b):
            k += 1
            continue
        pick = _option_a if cost_a <= cost_b else _option_b
        for q in run:
            choice[q] = pick(faces[q])
        k += 1
    return choice


def split_quads(verts, faces, planarity, tolerance, consistent=False):
    needs_split = [len(f) == 4 and planarity[k] > tolerance for k, f in enumerate(faces)]
    choice = _consistent_choices(verts, faces, needs_split) if consistent else {}
    out, src, split = [], [], []
    for k, f in enumerate(faces):
        if needs_split[k]:
            out.extend(choice.get(k) or best_diagonal(verts, f))
            src.extend([k, k])
            split.extend([True, True])
        else:
            out.append(tuple(f))
            src.append(k)
            split.append(False)
    return out, np.array(src, dtype=int), np.array(split, dtype=bool)


def mesh_area(verts, faces) -> float:
    total = 0.0
    for f in faces:
        P = verts[list(f)]
        for k in range(1, len(f) - 1):
            total += 0.5 * np.linalg.norm(np.cross(P[k] - P[0], P[k + 1] - P[0]))
    return float(total)
