import numpy as np

from mino.core.report import failing_ranges, format_report
from mino.core.types import Report


def test_failing_ranges_groups_runs_and_inf():
    tw = np.array([0, 6, 7, 0, 0, np.inf, 2, 9])
    assert failing_ranges(tw, 5.0) == [(1, 2), (5, 5), (7, 7)]
    assert failing_ranges(np.zeros(3), 5.0) == []


def test_format_report_mentions_ranges():
    r = Report(ruling_count=60, max_twist=14.2, mean_twist=3.0, failing_ruling_count=7,
               twist_tolerance=5.0, quad_count=59, split_quad_count=4, area_3d=12.3, area_unfolded=12.29)
    s = format_report(r, [(31, 37)])
    assert s.startswith("Mino: 60 rulings")
    assert "14.2" in s and "31-37" in s and "4/59" in s
    ok = format_report(Report(10, 0.1, 0.05, 0, 5.0, 9, 0, 1.0, 1.0), [])
    assert "within tolerance" in ok
