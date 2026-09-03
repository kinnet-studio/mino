from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class Rail:
    points: np.ndarray    # (N, 3), resampled by arc length
    tangents: np.ndarray  # (N, 3), unit vectors


@dataclass
class LoftParams:
    samples: int = 60
    window: int = 8
    twist_tolerance: float = 5.0
    tie_breaker: str = "none"  # "none" | "shortest" | "plane"
    tie_weight: float = 0.1
    plane_normal: tuple = (0.0, 0.0, 1.0)
    planarize: bool = True
    planar_tolerance: float = 0.01
    planarize_iterations: int = 10


@dataclass
class Report:
    ruling_count: int
    max_twist: float
    mean_twist: float
    failing_ruling_count: int
    twist_tolerance: float
    quad_count: int
    split_quad_count: int
    area_3d: float
    area_unfolded: float


@dataclass
class StripResult:
    verts: np.ndarray
    faces: list
    rulings: list
    ruling_twist: np.ndarray
    face_twist: np.ndarray
    face_planarity: np.ndarray
    face_split: np.ndarray
    failing_ranges: list
    report: Report
    layout: list = field(default_factory=list)
