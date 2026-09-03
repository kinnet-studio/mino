"""Rail sets used by tests and by tools/viewer/export_cases.py."""
import numpy as np


def arc(radius, z, angle_start, angle_end, n, phase=0.0):
    th = np.linspace(angle_start, angle_end, n) + phase
    pts = np.column_stack([radius * np.cos(th), radius * np.sin(th), np.full(n, float(z))])
    tans = np.column_stack([-np.sin(th), np.cos(th), np.zeros(n)])
    return pts, tans


def cylinder(n=200):
    a, ta = arc(1.0, 0.0, 0.0, np.pi, n)
    b, tb = arc(1.0, 1.0, 0.0, np.pi, n)
    return dict(points_a=a, points_b=b, tangents_a=ta, tangents_b=tb, params={})


def cone(n=200):
    a, ta = arc(1.0, 0.0, 0.0, np.pi, n)
    b, tb = arc(0.5, 1.0, 0.0, np.pi, n)
    return dict(points_a=a, points_b=b, tangents_a=ta, tangents_b=tb, params={})


OFFSET_STEPS = 4
OFFSET_SAMPLES = 60


def offset_cylinder(n=200):
    delta = OFFSET_STEPS * np.pi / (OFFSET_SAMPLES - 1)
    a, ta = arc(1.0, 0.0, 0.0, np.pi, n)
    b, tb = arc(1.0, 1.0, 0.0, np.pi, n, phase=delta)
    return dict(points_a=a, points_b=b, tangents_a=ta, tangents_b=tb,
                params=dict(samples=OFFSET_SAMPLES, window=8))


def twisted(n=200):
    t = np.linspace(0.0, 4.0, n)
    phi = 1.2 * (t / 4.0) ** 2
    dphi = 0.15 * t
    a = np.column_stack([t, np.zeros(n), np.zeros(n)])
    ta = np.column_stack([np.ones(n), np.zeros(n), np.zeros(n)])
    b = np.column_stack([t, np.sin(phi), np.cos(phi)])
    tb = np.column_stack([np.ones(n), np.cos(phi) * dphi, -np.sin(phi) * dphi])
    return dict(points_a=a, points_b=b, tangents_a=ta, tangents_b=tb, params={})


CASES = {
    "cylinder": cylinder,
    "cone": cone,
    "offset_cylinder": offset_cylinder,
    "twisted": twisted,
}
