"""Developability diagnosis: ranked, numeric suggestions for a loft result."""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace

import numpy as np

from .align import align, tie_matrix
from .dart import DartProposal, dart_proposal
from .rails import prepare_rails
from .strakes import find_strake_count
from .twist import twist_matrix
from .types import LoftParams, Rail, StripResult

RANK = {"ok": 0, "window": 1, "subdivide": 2, "unsolved": 2, "dart": 3, "creases": 4}


@dataclass
class Suggestion:
    kind: str
    text: str
    params: dict
    rank: int


@dataclass
class Diagnosis:
    failing_ranges: list
    max_twist: float
    window_fix: int | None
    strakes_needed: int | None
    strakes_worst_twist: float
    darts: list
    crease_runs: int
    suggestions: list


def probe_window(rail_a: Rail, rail_b: Rail, params: LoftParams, window: int) -> float:
    twist = twist_matrix(rail_a, rail_b, window)
    cost = twist + params.tie_weight * tie_matrix(rail_a, rail_b, params.tie_breaker, params.plane_normal)
    path = align(cost)
    return float(max(twist[i, j] for i, j in path))


def crease_runs(face_split) -> int:
    runs, inside = 0, False
    for flag in np.asarray(face_split, dtype=bool):
        if flag and not inside:
            runs += 1
        inside = bool(flag)
    return runs


def diagnose(points_a, points_b, tangents_a, tangents_b, params: LoftParams, result: StripResult,
             max_strakes: int = 8) -> Diagnosis:
    ranges = list(result.failing_ranges)
    runs = crease_runs(result.face_split)
    if not ranges:
        return Diagnosis(ranges, result.report.max_twist, None, None, result.report.max_twist, [], runs,
                         [Suggestion("ok", "All rulings within tolerance", {}, RANK["ok"])])
    tol = params.twist_tolerance
    n = params.samples
    rail_a, rail_b = prepare_rails(points_a, points_b, n, tangents_a, tangents_b, params.adaptive)

    window_fix = None
    for w in sorted({min(2 * params.window, n - 1), min(4 * params.window, n - 1)}):
        if w <= params.window:
            continue
        if probe_window(rail_a, rail_b, params, w) <= tol:
            window_fix = w
            break

    search_params = replace(params, window=window_fix) if window_fix else params
    count, worst, _ = find_strake_count(points_a, points_b, tangents_a, tangents_b, search_params,
                                        result.rulings, max_strakes)
    darts = [dart_proposal(rail_a, rail_b, result.rulings, result.ruling_twist, r) for r in ranges]

    suggestions = []
    if window_fix:
        suggestions.append(Suggestion("window", f"Re-loft with Window {window_fix}: all rulings within tolerance",
                                      {"window": window_fix}, RANK["window"]))
    if count:
        suggestions.append(Suggestion("subdivide", f"Subdivide into {count} strakes (worst twist {worst:.1f}°)",
                                      {"strakes": count}, RANK["subdivide"]))
    else:
        suggestions.append(Suggestion("unsolved",
                                      f"{max_strakes} strakes still leave {worst:.1f}° twist: cut a gusset or relax rail B",
                                      {"strakes": max_strakes}, RANK["unsolved"]))
    for d in darts:
        suggestions.append(Suggestion("dart",
                                      f"Cut a {d.kind} at ruling {d.ruling} ({abs(d.wedge_deg):.1f}° wedge, refined mesh)",
                                      {"ruling": d.ruling, "wedge_deg": d.wedge_deg, "kind": d.kind}, RANK["dart"]))
    if params.consistent_creases:
        text = f"{runs} crease run(s), diagonals consistent"
    else:
        text = f"{runs} crease run(s) zigzag; enable Consistent Creases"
    suggestions.append(Suggestion("creases", text, {"runs": runs}, RANK["creases"]))
    suggestions.sort(key=lambda s: s.rank)
    return Diagnosis(ranges, result.report.max_twist, window_fix, count, worst, darts, runs, suggestions)


def format_diagnosis(d: Diagnosis) -> list[str]:
    return [s.text for s in d.suggestions]


def diagnosis_to_dict(d: Diagnosis) -> dict:
    return {
        "failing_ranges": [[int(a), int(b)] for a, b in d.failing_ranges],
        "max_twist": float(d.max_twist),
        "window_fix": d.window_fix,
        "strakes_needed": d.strakes_needed,
        "strakes_worst_twist": float(d.strakes_worst_twist),
        "darts": [{"range": [int(p.range[0]), int(p.range[1])], "ruling": int(p.ruling),
                   "wedge_deg": float(p.wedge_deg), "kind": p.kind} for p in d.darts],
        "crease_runs": int(d.crease_runs),
        "suggestions": [asdict(s) for s in d.suggestions],
    }


def diagnosis_from_dict(data: dict) -> Diagnosis:
    return Diagnosis(
        failing_ranges=[(int(a), int(b)) for a, b in data["failing_ranges"]],
        max_twist=float(data["max_twist"]),
        window_fix=data.get("window_fix"),
        strakes_needed=data.get("strakes_needed"),
        strakes_worst_twist=float(data["strakes_worst_twist"]),
        darts=[DartProposal(range=(int(p["range"][0]), int(p["range"][1])), ruling=int(p["ruling"]),
                            wedge_deg=float(p["wedge_deg"]), kind=p["kind"]) for p in data["darts"]],
        crease_runs=int(data["crease_runs"]),
        suggestions=[Suggestion(**s) for s in data["suggestions"]],
    )
