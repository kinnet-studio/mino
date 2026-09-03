"""Failure ranges and the one-line status summary."""
from __future__ import annotations

import numpy as np

from .types import Report


def failing_ranges(ruling_twist, tolerance: float):
    bad = ~(np.asarray(ruling_twist, dtype=float) <= tolerance)  # inf counts as bad
    ranges, start = [], None
    for k, b in enumerate(bad):
        if b and start is None:
            start = k
        if not b and start is not None:
            ranges.append((start, k - 1))
            start = None
    if start is not None:
        ranges.append((start, len(bad) - 1))
    return ranges


def _fmt_range(r):
    a, b = r
    return f"{a}" if a == b else f"{a}-{b}"


def format_report(report: Report, ranges) -> str:
    mx = "inf" if not np.isfinite(report.max_twist) else f"{report.max_twist:.1f}"
    parts = [f"Mino: {report.ruling_count} rulings", f"max twist {mx} deg"]
    if ranges:
        parts.append(f"{report.failing_ruling_count} over {report.twist_tolerance:.1f} deg at rulings "
                     + ", ".join(_fmt_range(r) for r in ranges))
    else:
        parts.append(f"all within tolerance {report.twist_tolerance:.1f} deg")
    parts.append(f"{report.split_quad_count}/{report.quad_count} quads split")
    parts.append(f"area {report.area_3d:.4g} -> {report.area_unfolded:.4g} unfolded")
    return ", ".join(parts)
