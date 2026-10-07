"""Pure-numpy developable loft. No Blender imports here."""
from __future__ import annotations

import numpy as np

from .align import align, tie_matrix
from .errors import LoftError
from .mesh import build_faces, face_planarity, mesh_area, planarize, split_quads
from .quads import quad_strip
from .rails import prepare_rails
from .report import failing_ranges, format_report
from .twist import twist_matrix
from .types import LoftParams, Rail, Report, StripResult
from .unfold import unfold_strip

__all__ = ["LoftError", "LoftParams", "Rail", "Report", "StripResult", "loft", "format_report"]


def loft(points_a, points_b, params: LoftParams | None = None,
         tangents_a=None, tangents_b=None, *, pin_a: bool = False, pin_b: bool = False) -> StripResult:
    params = params or LoftParams()
    n = params.samples
    rail_a, rail_b = prepare_rails(points_a, points_b, n, tangents_a, tangents_b, params.adaptive)

    twist = twist_matrix(rail_a, rail_b, params.window)
    cost = twist + params.tie_weight * tie_matrix(rail_a, rail_b, params.tie_breaker, params.plane_normal)
    path = align(cost)

    if params.quads:
        verts, faces, ruling_twist = quad_strip(rail_a, rail_b, path)
        m = len(path)
        ruling_verts = [(k, m + k) for k in range(m)]
    else:
        ruling_twist = np.array([twist[i, j] for i, j in path], dtype=float)
        verts = np.vstack([rail_a.points, rail_b.points])
        faces = build_faces(path, n)
        m = n
        ruling_verts = [(i, n + j) for i, j in path]
    face_twist = np.array([max(ruling_twist[k], ruling_twist[k + 1]) for k in range(len(faces))])

    if params.planarize:
        lengths = [np.linalg.norm(verts[b] - verts[a]) for a, b in ruling_verts]
        max_nudge = params.planarize_max_nudge * float(np.mean(lengths))
        pinned = [0, m - 1, m, 2 * m - 1]
        if pin_a:
            pinned += list(range(0, m))
        if pin_b:
            pinned += list(range(m, 2 * m))
        verts = planarize(verts, faces, pinned=pinned,
                          tolerance=params.planar_tolerance,
                          iterations=params.planarize_iterations, max_nudge=max_nudge,
                          rails=(range(0, m), range(m, 2 * m)))

    planarity = face_planarity(verts, faces)
    out_faces, src, split = split_quads(verts, faces, planarity, params.planar_tolerance,
                                        consistent=params.consistent_creases)
    quad_count = sum(1 for f in faces if len(f) == 4)
    split_quad_count = int(split.sum() // 2)

    area_3d = mesh_area(verts, out_faces)
    area_2d, layout = unfold_strip(verts, out_faces)
    ranges = failing_ranges(ruling_twist, params.twist_tolerance)
    finite = ruling_twist[np.isfinite(ruling_twist)]
    report = Report(
        ruling_count=len(path),
        max_twist=float(finite.max()) if len(finite) else float("inf"),
        mean_twist=float(finite.mean()) if len(finite) else float("inf"),
        failing_ruling_count=int(sum(b - a + 1 for a, b in ranges)),
        twist_tolerance=params.twist_tolerance,
        quad_count=quad_count,
        split_quad_count=split_quad_count,
        area_3d=area_3d,
        area_unfolded=area_2d,
    )
    return StripResult(
        verts=verts, faces=out_faces, rulings=path, ruling_twist=ruling_twist,
        face_twist=face_twist[src], face_planarity=planarity[src], face_split=split,
        failing_ranges=ranges, report=report, layout=layout, ruling_verts=ruling_verts,
    )
