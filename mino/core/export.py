"""JSON-friendly export of a StripResult for the browser viewer."""
from __future__ import annotations

from dataclasses import asdict

import numpy as np


def _finite_list(values):
    return [float(v) if np.isfinite(v) else None for v in np.asarray(values, dtype=float)]


def result_to_dict(result, points_a, points_b, name, params) -> dict:
    p = asdict(params)
    p["plane_normal"] = [float(x) for x in p["plane_normal"]]
    report = asdict(result.report)
    for k, v in report.items():
        if isinstance(v, float) and not np.isfinite(v):
            report[k] = None
    return {
        "name": name,
        "params": p,
        "verts": np.asarray(result.verts, dtype=float).tolist(),
        "faces": [list(map(int, f)) for f in result.faces],
        "rulings": [[int(i), int(j)] for i, j in result.rulings],
        "ruling_verts": [[int(a), int(b)] for a, b in result.ruling_verts],
        "ruling_twist": _finite_list(result.ruling_twist),
        "face_twist": _finite_list(result.face_twist),
        "face_planarity": _finite_list(result.face_planarity),
        "face_split": [bool(x) for x in result.face_split],
        "failing_ranges": [[int(a), int(b)] for a, b in result.failing_ranges],
        "report": report,
        "layout": [np.asarray(p2, dtype=float).tolist() for p2 in result.layout],
        "rails": {
            "a": np.asarray(points_a, dtype=float).tolist(),
            "b": np.asarray(points_b, dtype=float).tolist(),
        },
    }
