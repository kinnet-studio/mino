# DevLoft PoC Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A Blender extension that lofts a developable strip between two rail curves, flags non-developable regions by twist, and ships with a browser viewer for the test outputs.

**Architecture:** `devloft/core` is pure numpy geometry (rails, twist, DP alignment, faces, planarize, unfold, report). `devloft/blender` is a thin bpy layer (inputs, operator, output mesh with attributes, panel). `tools/viewer` is a single three.js page fed by a generated `data.js`.

**Tech Stack:** Python 3.11, numpy, pytest, uv, bpy 4.5 wheel (headless tests), three.js r128 from cdnjs.

**Spec:** `docs/superpowers/specs/2026-09-03-devloft-poc-design.md`

## Global Constraints

- `devloft/core/**` must never import `bpy` or `mathutils`. numpy and stdlib only.
- Extension manifest: `schema_version = "1.0.0"`, `blender_version_min = "4.2.0"`, `type = "add-on"`, license `SPDX:GPL-3.0-or-later`.
- No wheel dependencies in the extension; numpy is bundled with Blender.
- Vertex layout in results: rail A is indices `0..N-1`, rail B is `N..2N-1`.
- Twist is in degrees, folded into `[0, 90]`. Invalid rulings have twist `inf`.
- Non-diagonal DP steps cost an extra `0.5` degrees.
- Default `LoftParams`: samples 60, window 8, twist_tolerance 5.0, tie_breaker "none", tie_weight 0.1, plane_normal (0,0,1), planarize True, planar_tolerance 0.01, planarize_iterations 10.
- Output mesh attributes: face FLOAT `twist`, face FLOAT `planarity`, face FLOAT_COLOR `twist_color`, face BOOLEAN `split`.
- Commit after every task with the trailer lines shown in Task 1.
- Run all tests with `uv run pytest -q` from the repo root.

Deviations from the spec, decided while planning (keep the spec's intent):
- Split quads keep the source quad's planarity value on both triangles (more informative than 0); `face_split` marks them.
- Planarize caps each vertex's total displacement at `planar_tolerance * mean ruling length` so nudges stay tiny.
- `StripResult` gains a `layout` field: the 2D unfolded polygon per face, used by the viewer.
- `Report` gains a `twist_tolerance` field so `format_report` needs no extra argument.
- No `bl_info` in the package; the manifest is authoritative. Legacy-add-on install is not supported.
- A fourth test case, `offset_cylinder`, proves that the DP leans rulings when that removes twist.

---

### Task 1: Project scaffold, types, errors

**Files:**
- Create: `pyproject.toml`, `.gitignore`, `devloft/__init__.py`, `devloft/core/__init__.py`, `devloft/core/errors.py`, `devloft/core/types.py`, `tests/__init__.py`, `tests/conftest.py`, `tests/test_types.py`

**Interfaces:**
- Produces: `LoftError`, dataclasses `Rail`, `LoftParams`, `Report`, `StripResult` (fields below). Later tasks import them from `devloft.core.types` and `devloft.core.errors`.

- [ ] **Step 1: Write pyproject and gitignore**

`pyproject.toml`:
```toml
[project]
name = "devloft"
version = "0.1.0"
description = "Developable loft between two rails for Blender"
requires-python = ">=3.11"
dependencies = ["numpy>=1.24"]

[dependency-groups]
dev = ["pytest>=8", "bpy==4.5.*"]

[tool.uv]
package = false

[tool.pytest.ini_options]
testpaths = ["tests"]
```

`.gitignore`:
```
.venv/
__pycache__/
*.pyc
dist/
.pytest_cache/
```

- [ ] **Step 2: Create the venv and install deps**

Run: `uv python pin 3.11 && uv sync`
Expected: `.venv` created, pytest, numpy and bpy 4.5.x installed. If bpy fails to resolve, run `uv sync --no-group dev && uv pip install pytest "bpy==4.5.*"` and report the error text.

- [ ] **Step 3: Write the failing test**

`tests/__init__.py`: empty file.

`tests/conftest.py`:
```python
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
```

`tests/test_types.py`:
```python
import numpy as np
from devloft.core.types import LoftParams, Rail, Report
from devloft.core.errors import LoftError


def test_default_params():
    p = LoftParams()
    assert p.samples == 60
    assert p.window == 8
    assert p.twist_tolerance == 5.0
    assert p.tie_breaker == "none"
    assert p.planarize is True
    assert p.planar_tolerance == 0.01


def test_rail_holds_arrays():
    r = Rail(points=np.zeros((3, 3)), tangents=np.zeros((3, 3)))
    assert r.points.shape == (3, 3)


def test_loft_error_is_exception():
    assert issubclass(LoftError, Exception)
```

- [ ] **Step 4: Run test to verify it fails**

Run: `uv run pytest tests/test_types.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'devloft'` or similar.

- [ ] **Step 5: Write the package files**

`devloft/__init__.py`:
```python
"""DevLoft: developable loft between two rails.

The core package is importable without Blender. The Blender layer is only
imported inside register()/unregister() so tests can run on plain Python.
"""

__version__ = "0.1.0"


def register():
    from .blender import register as _register
    _register()


def unregister():
    from .blender import unregister as _unregister
    _unregister()
```

`devloft/core/errors.py`:
```python
class LoftError(Exception):
    """Raised when the rails cannot be lofted. The message is shown to the user."""
```

`devloft/core/types.py`:
```python
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
```

`devloft/core/__init__.py` (placeholder for now, filled in Task 7):
```python
from .errors import LoftError
from .types import LoftParams, Rail, Report, StripResult

__all__ = ["LoftError", "LoftParams", "Rail", "Report", "StripResult"]
```

- [ ] **Step 6: Run test to verify it passes**

Run: `uv run pytest tests/test_types.py -q`
Expected: 3 passed.

- [ ] **Step 7: Commit**

```bash
git add pyproject.toml .gitignore .python-version uv.lock devloft tests
git commit -m "Scaffold devloft package with core types

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_012kESBz7SJpRjCw2UpaQv9E"
```

---

### Task 2: Rails: dedupe, resample, tangents, orientation

**Files:**
- Create: `devloft/core/rails.py`, `tests/test_rails.py`

**Interfaces:**
- Consumes: `Rail`, `LoftError`.
- Produces:
  - `normalize_rows(v: np.ndarray) -> np.ndarray`
  - `central_difference(points: np.ndarray) -> np.ndarray`
  - `resample(points, samples: int, tangents=None) -> Rail`
  - `prepare_rails(points_a, points_b, samples, tangents_a=None, tangents_b=None) -> tuple[Rail, Rail]`

- [ ] **Step 1: Write the failing tests**

`tests/test_rails.py`:
```python
import numpy as np
import pytest

from devloft.core.errors import LoftError
from devloft.core.rails import central_difference, normalize_rows, prepare_rails, resample


def test_resample_equal_arc_length_spacing():
    # collinear but unevenly spaced input: arc length equals Euclidean distance,
    # so equal arc-length spacing must give equal segment lengths
    pts = np.array([[0, 0, 0], [1, 0, 0], [1.2, 0, 0], [3, 0, 0]], float)
    rail = resample(pts, 7)
    seg = np.linalg.norm(np.diff(rail.points, axis=0), axis=1)
    assert rail.points.shape == (7, 3)
    assert np.allclose(seg, 0.5)
    assert np.allclose(rail.points[0], pts[0])
    assert np.allclose(rail.points[-1], pts[-1])


def test_resample_corner_lands_on_polyline():
    # a corner: samples at arc lengths 0, 1, 2 hit the corner exactly and stay on the path
    pts = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0]], float)
    rail = resample(pts, 3)
    assert np.allclose(rail.points, pts)


def test_resample_removes_duplicates():
    pts = np.array([[0, 0, 0], [0, 0, 0], [1, 0, 0], [1, 0, 0], [2, 0, 0]], float)
    rail = resample(pts, 3)
    assert np.allclose(rail.points, [[0, 0, 0], [1, 0, 0], [2, 0, 0]])


def test_resample_tangents_are_unit_and_follow_curve():
    pts = np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0]], float)
    rail = resample(pts, 5)
    assert np.allclose(np.linalg.norm(rail.tangents, axis=1), 1.0)
    assert np.allclose(rail.tangents, [[1, 0, 0]] * 5)


def test_resample_uses_supplied_tangents():
    pts = np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0]], float)
    tans = np.array([[0, 2, 0], [0, 2, 0], [0, 2, 0]], float)
    rail = resample(pts, 4, tans)
    assert np.allclose(rail.tangents, [[0, 1, 0]] * 4)


def test_resample_rejects_degenerate():
    with pytest.raises(LoftError):
        resample(np.zeros((3, 3)), 5)
    with pytest.raises(LoftError):
        resample(np.array([[0, 0, 0], [1, 0, 0]], float), 1)


def test_central_difference_ends():
    pts = np.array([[0, 0, 0], [1, 0, 0], [3, 0, 0]], float)
    t = central_difference(pts)
    assert np.allclose(t, [[1, 0, 0], [3, 0, 0], [2, 0, 0]])


def test_normalize_rows_keeps_zero():
    v = normalize_rows(np.array([[0, 0, 0], [0, 3, 0]], float))
    assert np.allclose(v, [[0, 0, 0], [0, 1, 0]])


def test_prepare_rails_flips_reversed_b():
    a = np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0]], float)
    b_reversed = np.array([[2, 1, 0], [1, 1, 0], [0, 1, 0]], float)
    tans_b = np.array([[-1, 0, 0]] * 3, float)
    ra, rb = prepare_rails(a, b_reversed, 3, tangents_b=tans_b)
    assert np.allclose(rb.points[0], [0, 1, 0])
    assert np.allclose(rb.tangents, [[1, 0, 0]] * 3)
    assert ra.points.shape == rb.points.shape == (3, 3)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_rails.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'devloft.core.rails'`.

- [ ] **Step 3: Write the implementation**

`devloft/core/rails.py`:
```python
"""Rail preparation: dedupe, arc-length resampling, tangents, orientation."""
from __future__ import annotations

import numpy as np

from .errors import LoftError
from .types import Rail


def normalize_rows(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=float)
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    n = np.where(n < 1e-12, 1.0, n)
    return v / n


def dedupe_indices(points: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    keep = [0]
    for k in range(1, len(points)):
        if np.linalg.norm(points[k] - points[keep[-1]]) > eps:
            keep.append(k)
    return np.array(keep, dtype=int)


def central_difference(points: np.ndarray) -> np.ndarray:
    t = np.empty_like(points)
    t[1:-1] = points[2:] - points[:-2]
    t[0] = points[1] - points[0]
    t[-1] = points[-1] - points[-2]
    return t


def resample(points, samples: int, tangents=None) -> Rail:
    if samples < 2:
        raise LoftError("samples must be at least 2")
    pts = np.asarray(points, dtype=float).reshape(-1, 3)
    if len(pts) == 0:
        raise LoftError("a rail needs at least 2 distinct points")
    keep = dedupe_indices(pts)
    pts = pts[keep]
    if len(pts) < 2:
        raise LoftError("a rail needs at least 2 distinct points")
    tans = None
    if tangents is not None:
        tans = np.asarray(tangents, dtype=float).reshape(-1, 3)
        if len(tans) != len(np.asarray(points, dtype=float).reshape(-1, 3)):
            raise LoftError("tangents must match points one-to-one")
        tans = tans[keep]

    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    targets = np.linspace(0.0, cum[-1], samples)
    out = np.column_stack([np.interp(targets, cum, pts[:, k]) for k in range(3)])
    if tans is None:
        t = central_difference(out)
    else:
        t = np.column_stack([np.interp(targets, cum, tans[:, k]) for k in range(3)])
    return Rail(points=out, tangents=normalize_rows(t))


def prepare_rails(points_a, points_b, samples: int, tangents_a=None, tangents_b=None):
    """Resample both rails to `samples` points, reversing B if it runs opposite to A."""
    pa = np.asarray(points_a, dtype=float).reshape(-1, 3)
    pb = np.asarray(points_b, dtype=float).reshape(-1, 3)
    if len(pa) < 2 or len(pb) < 2:
        raise LoftError("each rail needs at least 2 points")
    tb = None if tangents_b is None else np.asarray(tangents_b, dtype=float).reshape(-1, 3)
    if np.linalg.norm(pb[-1] - pa[0]) < np.linalg.norm(pb[0] - pa[0]):
        pb = pb[::-1].copy()
        if tb is not None:
            tb = -tb[::-1]
    return resample(pa, samples, tangents_a), resample(pb, samples, tb)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_rails.py -q`
Expected: 9 passed.

- [ ] **Step 5: Commit**

```bash
git add devloft/core/rails.py tests/test_rails.py
git commit -m "Add rail resampling and orientation

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_012kESBz7SJpRjCw2UpaQv9E"
```

---

### Task 3: Twist matrix

**Files:**
- Create: `devloft/core/twist.py`, `tests/test_twist.py`

**Interfaces:**
- Consumes: `Rail`.
- Produces:
  - `twist_matrix(rail_a: Rail, rail_b: Rail, window: int) -> np.ndarray` shape `(N, N)`, degrees, `inf` where invalid or outside band `|i-j| <= window`.
  - `ruling_lengths(rail_a: Rail, rail_b: Rail) -> np.ndarray` shape `(N, N)`.

- [ ] **Step 1: Write the failing tests**

`tests/test_twist.py`:
```python
import numpy as np

from devloft.core.rails import Rail
from devloft.core.twist import ruling_lengths, twist_matrix


def _rail(points, tangents):
    return Rail(np.asarray(points, float), np.asarray(tangents, float))


def test_parallel_tangents_give_zero_twist():
    a = _rail([[0, 0, 0], [1, 0, 0], [2, 0, 0]], [[1, 0, 0]] * 3)
    b = _rail([[0, 0, 1], [1, 0, 1], [2, 0, 1]], [[1, 0, 0]] * 3)
    tw = twist_matrix(a, b, window=2)
    assert np.allclose(np.diagonal(tw), 0.0)


def test_ninety_degree_case():
    # ruling along z; tangent A along x, tangent B along y -> normals differ by 90 deg
    a = _rail([[0, 0, 0]], [[1, 0, 0]])
    b = _rail([[0, 0, 1]], [[0, 1, 0]])
    tw = twist_matrix(a, b, window=0)
    assert np.isclose(tw[0, 0], 90.0)


def test_twist_is_folded_for_flipped_tangent():
    a = _rail([[0, 0, 0]], [[1, 0, 0]])
    b = _rail([[0, 0, 1]], [[-1, 0, 0]])
    tw = twist_matrix(a, b, window=0)
    assert np.isclose(tw[0, 0], 0.0)


def test_band_and_invalid_are_inf():
    a = _rail([[0, 0, 0], [1, 0, 0], [2, 0, 0]], [[1, 0, 0]] * 3)
    b = _rail([[0, 0, 1], [1, 0, 1], [3, 0, 0]], [[1, 0, 0]] * 3)
    tw = twist_matrix(a, b, window=1)
    assert np.isinf(tw[0, 2]) and np.isinf(tw[2, 0])   # outside band
    assert np.isinf(tw[2, 2])                          # ruling parallel to tangent
    assert np.isfinite(tw[1, 1])


def test_ruling_lengths():
    a = _rail([[0, 0, 0], [1, 0, 0]], [[1, 0, 0]] * 2)
    b = _rail([[0, 0, 2], [1, 0, 2]], [[1, 0, 0]] * 2)
    L = ruling_lengths(a, b)
    assert np.allclose(np.diagonal(L), 2.0)
    assert np.isclose(L[0, 1], np.sqrt(5))
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_twist.py -q`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Write the implementation**

`devloft/core/twist.py`:
```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_twist.py -q`
Expected: 5 passed.

- [ ] **Step 5: Commit**

```bash
git add devloft/core/twist.py tests/test_twist.py
git commit -m "Add ruling twist matrix

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_012kESBz7SJpRjCw2UpaQv9E"
```

---
### Task 4: Alignment: tie-breaker costs and monotone DP

**Files:**
- Create: `devloft/core/align.py`, `tests/test_align.py`

**Interfaces:**
- Consumes: `Rail`, `LoftError`, `normalize_rows`, `ruling_lengths`.
- Produces:
  - `STEP_PENALTY = 0.5`
  - `tie_matrix(rail_a: Rail, rail_b: Rail, mode: str, plane_normal) -> np.ndarray` shape `(N, N)`, unitless.
  - `align(cost: np.ndarray) -> list[tuple[int, int]]` monotone path from `(0,0)` to `(N-1,N-1)`; raises `LoftError` if none exists.

- [ ] **Step 1: Write the failing tests**

`tests/test_align.py`:
```python
import numpy as np
import pytest

from devloft.core.align import STEP_PENALTY, align, tie_matrix
from devloft.core.errors import LoftError
from devloft.core.types import Rail


def _is_monotone(path):
    for (i, j), (i2, j2) in zip(path, path[1:]):
        di, dj = i2 - i, j2 - j
        if (di, dj) not in {(1, 0), (0, 1), (1, 1)}:
            return False
    return True


def test_diagonal_when_cheapest():
    cost = np.full((4, 4), 10.0)
    np.fill_diagonal(cost, 0.0)
    path = align(cost)
    assert path == [(0, 0), (1, 1), (2, 2), (3, 3)]


def test_leans_to_cheap_offset_band():
    n = 8
    cost = np.full((n, n), 20.0)
    for i in range(2, n):
        cost[i, i - 2] = 0.0
    path = align(cost)
    assert _is_monotone(path)
    assert path[0] == (0, 0) and path[-1] == (n - 1, n - 1)
    middle = [(i, j) for i, j in path if 2 <= i <= n - 3]
    assert all(j == i - 2 for i, j in middle)


def test_step_penalty_prefers_quads_on_ties():
    cost = np.zeros((3, 3))
    path = align(cost)
    assert path == [(0, 0), (1, 1), (2, 2)]
    assert STEP_PENALTY > 0


def test_routes_around_inf():
    cost = np.zeros((3, 3))
    cost[1, 1] = np.inf
    path = align(cost)
    assert _is_monotone(path)
    assert (1, 1) not in path


def test_raises_when_no_path():
    cost = np.zeros((3, 3))
    cost[1, :] = np.inf
    with pytest.raises(LoftError) as e:
        align(cost)
    assert "1" in str(e.value)


def test_tie_matrix_modes():
    a = Rail(np.array([[0, 0, 0], [1, 0, 0]], float), np.array([[1, 0, 0]] * 2, float))
    b = Rail(np.array([[0, 0, 1], [1, 0, 1]], float), np.array([[1, 0, 0]] * 2, float))
    assert np.allclose(tie_matrix(a, b, "none", (0, 0, 1)), 0.0)
    short = tie_matrix(a, b, "shortest", (0, 0, 1))
    assert np.allclose(np.diagonal(short), 1.0)
    assert short[0, 1] > 1.0
    plane = tie_matrix(a, b, "plane", (0, 0, 1))
    assert np.allclose(np.diagonal(plane), 0.0)
    assert plane[0, 1] > 0.0
    with pytest.raises(LoftError):
        tie_matrix(a, b, "bogus", (0, 0, 1))
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_align.py -q`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Write the implementation**

`devloft/core/align.py`:
```python
"""Monotone (non-crossing) ruling selection by dynamic programming."""
from __future__ import annotations

import numpy as np

from .errors import LoftError
from .rails import normalize_rows
from .twist import ruling_lengths
from .types import Rail

STEP_PENALTY = 0.5  # degrees, added to non-diagonal moves


def tie_matrix(rail_a: Rail, rail_b: Rail, mode: str, plane_normal) -> np.ndarray:
    n = len(rail_a.points)
    if mode == "none":
        return np.zeros((n, n))
    L = ruling_lengths(rail_a, rail_b)
    if mode == "shortest":
        mean = float(np.mean(np.diagonal(L)))
        return L / (mean if mean > 1e-12 else 1.0)
    if mode == "plane":
        nrm = normalize_rows(np.asarray(plane_normal, dtype=float).reshape(3))
        R = rail_b.points[None, :, :] - rail_a.points[:, None, :]
        Ls = np.where(L < 1e-12, 1.0, L)
        Rn = R / Ls[..., None]
        return 1.0 - np.abs(Rn @ nrm)
    raise LoftError(f"unknown tie_breaker {mode!r}")


def _explain_failure(cost: np.ndarray) -> str:
    finite_rows = np.isfinite(cost).any(axis=1)
    if not finite_rows.all():
        i = int(np.flatnonzero(~finite_rows)[0])
        return (f"no valid ruling from rail A point {i} within the window "
                f"(ruling parallel to a tangent or zero length); widen the window or check the rails")
    return "no monotone ruling path within the window; try a larger window"


def align(cost: np.ndarray) -> list[tuple[int, int]]:
    """Cheapest monotone path from (0,0) to (N-1,M-1) using moves (1,1), (1,0), (0,1)."""
    n, m = cost.shape
    D = np.full((n, m), np.inf)
    back = np.full((n, m), -1, dtype=np.int8)  # 0 diag, 1 from (i-1,j), 2 from (i,j-1)
    D[0, 0] = cost[0, 0]
    for i in range(n):
        for j in np.flatnonzero(np.isfinite(cost[i])):
            if i == 0 and j == 0:
                continue
            best, arg = np.inf, -1
            if i > 0 and j > 0 and D[i - 1, j - 1] < best:
                best, arg = D[i - 1, j - 1], 0
            if i > 0 and D[i - 1, j] + STEP_PENALTY < best:
                best, arg = D[i - 1, j] + STEP_PENALTY, 1
            if j > 0 and D[i, j - 1] + STEP_PENALTY < best:
                best, arg = D[i, j - 1] + STEP_PENALTY, 2
            if np.isfinite(best):
                D[i, j] = cost[i, j] + best
                back[i, j] = arg
    if not np.isfinite(D[n - 1, m - 1]):
        raise LoftError(_explain_failure(cost))
    path = []
    i, j = n - 1, m - 1
    while True:
        path.append((i, j))
        if i == 0 and j == 0:
            break
        b = back[i, j]
        if b == 0:
            i, j = i - 1, j - 1
        elif b == 1:
            i -= 1
        else:
            j -= 1
    path.reverse()
    return path
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_align.py -q`
Expected: 6 passed.

- [ ] **Step 5: Commit**

```bash
git add devloft/core/align.py tests/test_align.py
git commit -m "Add monotone DP ruling alignment with tie-breakers

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_012kESBz7SJpRjCw2UpaQv9E"
```

---

### Task 5: Faces, planarity, planarize, split

**Files:**
- Create: `devloft/core/mesh.py`, `tests/test_mesh.py`

**Interfaces:**
- Produces:
  - `build_faces(path: list[tuple[int,int]], n_a: int) -> list[tuple[int, ...]]` one face per consecutive ruling pair, quads `(A[i], A[i+1], B[j+1], B[j])`.
  - `quad_planarity(verts, face) -> float`
  - `face_planarity(verts, faces) -> np.ndarray` (0 for triangles)
  - `planarize(verts, faces, pinned, tolerance, iterations, max_nudge) -> np.ndarray` new verts
  - `split_quads(verts, faces, planarity, tolerance) -> tuple[list, np.ndarray, np.ndarray]` = (new faces, source face index per new face, split flag per new face)
  - `mesh_area(verts, faces) -> float`

- [ ] **Step 1: Write the failing tests**

`tests/test_mesh.py`:
```python
import numpy as np

from devloft.core.mesh import (build_faces, face_planarity, mesh_area, planarize,
                               quad_planarity, split_quads)


def test_build_faces_quads_and_triangles():
    path = [(0, 0), (1, 1), (2, 1), (2, 2)]
    faces = build_faces(path, n_a=3)
    assert faces == [(0, 1, 4, 3), (1, 2, 4), (2, 5, 4)]


def test_planarity_zero_for_flat_quad():
    v = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], float)
    assert quad_planarity(v, (0, 1, 2, 3)) == 0.0
    assert np.allclose(face_planarity(v, [(0, 1, 2, 3), (0, 1, 2)]), [0.0, 0.0])


def test_planarity_positive_for_bent_quad():
    v = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0.2], [0, 1, 0]], float)
    p = quad_planarity(v, (0, 1, 2, 3))
    assert 0.05 < p < 0.2


def test_planarize_flattens_unpinned_quad():
    v = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0.2], [0, 1, 0]], float)
    out = planarize(v, [(0, 1, 2, 3)], pinned=[], tolerance=1e-9, iterations=5, max_nudge=1.0)
    assert quad_planarity(out, (0, 1, 2, 3)) < 1e-9
    assert not np.allclose(out[2], v[2])


def test_planarize_pins_and_improves():
    v = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0.2], [0, 1, 0]], float)
    out = planarize(v, [(0, 1, 2, 3)], pinned=[0], tolerance=1e-9, iterations=20, max_nudge=1.0)
    assert np.allclose(out[0], v[0])
    assert quad_planarity(out, (0, 1, 2, 3)) < 0.5 * quad_planarity(v, (0, 1, 2, 3))


def test_planarize_caps_nudge():
    v = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0.2], [0, 1, 0]], float)
    out = planarize(v, [(0, 1, 2, 3)], pinned=[], tolerance=1e-6, iterations=20, max_nudge=0.01)
    assert np.linalg.norm(out - v, axis=1).max() <= 0.01 + 1e-9


def test_split_quads_over_tolerance():
    v = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0.2], [0, 1, 0],
                  [2, 0, 0], [2, 1, 0], [1, 1, 0]], float)
    faces = [(0, 1, 2, 3), (1, 4, 5, 6)]
    pl = face_planarity(v, faces)
    assert pl[0] > 0.01 and pl[1] == 0.0
    out, src, split = split_quads(v, faces, pl, tolerance=0.5)
    assert out == faces and list(src) == [0, 1] and not split.any()
    out, src, split = split_quads(v, faces, pl, tolerance=0.01)
    assert len(out) == 3 and all(len(f) == 3 for f in out[:2]) and len(out[2]) == 4
    assert list(src) == [0, 0, 1]
    assert list(split) == [True, True, False]
    assert set(out[0]) | set(out[1]) == {0, 1, 2, 3}


def test_mesh_area():
    v = np.array([[0, 0, 0], [2, 0, 0], [2, 1, 0], [0, 1, 0]], float)
    assert np.isclose(mesh_area(v, [(0, 1, 2, 3)]), 2.0)
    assert np.isclose(mesh_area(v, [(0, 1, 2)]), 1.0)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_mesh.py -q`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Write the implementation**

`devloft/core/mesh.py`:
```python
"""Face construction, planarity metric, planarize pass, and quad splitting."""
from __future__ import annotations

import numpy as np


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


def planarize(verts, faces, pinned, tolerance, iterations, max_nudge):
    verts = np.asarray(verts, dtype=float).copy()
    orig = verts.copy()
    quads = [f for f in faces if len(f) == 4]
    pinned = list(pinned)
    for _ in range(iterations):
        pl = face_planarity(verts, quads)
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
        verts = new
    return verts


def _tri_normal(verts, tri):
    p0, p1, p2 = verts[list(tri)]
    n = np.cross(p1 - p0, p2 - p0)
    ln = np.linalg.norm(n)
    return n / ln if ln > 1e-12 else np.zeros(3)


def _dihedral(verts, tris):
    n0, n1 = _tri_normal(verts, tris[0]), _tri_normal(verts, tris[1])
    return float(np.degrees(np.arccos(np.clip(np.dot(n0, n1), -1.0, 1.0))))


def best_diagonal(verts, face):
    v0, v1, v2, v3 = face
    opt_a = [(v0, v1, v2), (v0, v2, v3)]
    opt_b = [(v0, v1, v3), (v1, v2, v3)]
    return opt_a if _dihedral(verts, opt_a) <= _dihedral(verts, opt_b) else opt_b


def split_quads(verts, faces, planarity, tolerance):
    out, src, split = [], [], []
    for k, f in enumerate(faces):
        if len(f) == 4 and planarity[k] > tolerance:
            out.extend(best_diagonal(verts, f))
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_mesh.py -q`
Expected: 8 passed.

- [ ] **Step 5: Commit**

```bash
git add devloft/core/mesh.py tests/test_mesh.py
git commit -m "Add face building, planarity, planarize and split

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_012kESBz7SJpRjCw2UpaQv9E"
```

---

### Task 6: Unfold check and report

**Files:**
- Create: `devloft/core/unfold.py`, `devloft/core/report.py`, `tests/test_unfold.py`, `tests/test_report.py`

**Interfaces:**
- Consumes: `Report`, `normalize_rows`.
- Produces:
  - `unfold_strip(verts, faces) -> tuple[float, list[np.ndarray]]` = (2D area, per-face `(n, 2)` coordinates). Consecutive faces must share exactly two vertices.
  - `failing_ranges(ruling_twist: np.ndarray, tolerance: float) -> list[tuple[int, int]]`
  - `format_report(report: Report, ranges) -> str`

- [ ] **Step 1: Write the failing tests**

`tests/test_unfold.py`:
```python
import numpy as np

from devloft.core.mesh import mesh_area
from devloft.core.unfold import unfold_strip


def _box_strip():
    # three planar quads folded around a box edge: unfolds to a 3x1 rectangle
    v = np.array([
        [0, 0, 0], [0, 1, 0],
        [1, 0, 0], [1, 1, 0],
        [1, 0, 1], [1, 1, 1],
        [0, 0, 1], [0, 1, 1],
    ], float)
    faces = [(0, 2, 3, 1), (2, 4, 5, 3), (4, 6, 7, 5)]
    return v, faces


def test_unfold_planar_strip_preserves_area_and_edges():
    v, faces = _box_strip()
    area, layout = unfold_strip(v, faces)
    assert np.isclose(area, mesh_area(v, faces))
    assert len(layout) == 3 and all(p.shape == (4, 2) for p in layout)
    # shared edge of faces 0 and 1 lands on the same 2D points
    f0, f1 = faces[0], faces[1]
    for vi in (2, 3):
        assert np.allclose(layout[0][f0.index(vi)], layout[1][f1.index(vi)], atol=1e-9)
    # faces lie on opposite sides of the shared edge: total spans 3 units
    pts = np.vstack(layout)
    ext = pts.max(axis=0) - pts.min(axis=0)
    assert np.isclose(sorted(ext)[1], 3.0) and np.isclose(sorted(ext)[0], 1.0)


def test_unfold_triangles():
    v = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]], float)
    faces = [(0, 1, 2), (1, 3, 2)]
    area, layout = unfold_strip(v, faces)
    assert np.isclose(area, 1.0)


def test_unfold_bent_quad_loses_area():
    v = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0.5], [0, 1, 0], [2, 0, 0], [2, 1, 0.5]], float)
    faces = [(0, 1, 2, 3), (1, 4, 5, 2)]
    area, _ = unfold_strip(v, faces)
    assert area < mesh_area(v, faces)
```

`tests/test_report.py`:
```python
import numpy as np

from devloft.core.report import failing_ranges, format_report
from devloft.core.types import Report


def test_failing_ranges_groups_runs_and_inf():
    tw = np.array([0, 6, 7, 0, 0, np.inf, 2, 9])
    assert failing_ranges(tw, 5.0) == [(1, 2), (5, 5), (7, 7)]
    assert failing_ranges(np.zeros(3), 5.0) == []


def test_format_report_mentions_ranges():
    r = Report(ruling_count=60, max_twist=14.2, mean_twist=3.0, failing_ruling_count=7,
               twist_tolerance=5.0, quad_count=59, split_quad_count=4, area_3d=12.3, area_unfolded=12.29)
    s = format_report(r, [(31, 37)])
    assert s.startswith("DevLoft: 60 rulings")
    assert "14.2" in s and "31-37" in s and "4/59" in s
    ok = format_report(Report(10, 0.1, 0.05, 0, 5.0, 9, 0, 1.0, 1.0), [])
    assert "within tolerance" in ok
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_unfold.py tests/test_report.py -q`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Write the implementations**

`devloft/core/unfold.py`:
```python
"""Sequential flattening of a strip for validation (each face treated as rigid)."""
from __future__ import annotations

import numpy as np

from .rails import normalize_rows


def _polygon_normal(P: np.ndarray) -> np.ndarray:
    n = np.zeros(3)
    for k in range(len(P)):
        n += np.cross(P[k], P[(k + 1) % len(P)])
    return normalize_rows(n)


def _flatten_face(P: np.ndarray) -> np.ndarray:
    c = P.mean(axis=0)
    n = _polygon_normal(P - c)
    u = P[1] - P[0]
    u = u - n * np.dot(u, n)
    u = normalize_rows(u)
    v = np.cross(n, u)
    Q = P - c
    return np.column_stack([Q @ u, Q @ v])


def _reflect_across(pts, a, b):
    d = b - a
    d = d / (np.linalg.norm(d) or 1.0)
    q = pts - a
    return a + 2.0 * np.outer(q @ d, d) - q


def _place(local, ia, ib, ta, tb, prev_centroid):
    a, b = local[ia], local[ib]
    ang = np.arctan2(tb[1] - ta[1], tb[0] - ta[0]) - np.arctan2(b[1] - a[1], b[0] - a[0])
    c, s = np.cos(ang), np.sin(ang)
    rot = np.array([[c, -s], [s, c]])
    placed = (local - a) @ rot.T + ta
    e = tb - ta

    def side(p):
        return e[0] * (p[1] - ta[1]) - e[1] * (p[0] - ta[0])

    if side(placed.mean(axis=0)) * side(prev_centroid) > 0:
        placed = _reflect_across(placed, ta, tb)
    return placed


def _shoelace(p: np.ndarray) -> float:
    x, y = p[:, 0], p[:, 1]
    return float(0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


def unfold_strip(verts, faces):
    """Lay the strip flat face by face. Returns (area_2d, layout)."""
    verts = np.asarray(verts, dtype=float)
    layout = []
    prev_face, prev_2d = None, None
    for f in faces:
        f = tuple(f)
        local = _flatten_face(verts[list(f)])
        if prev_face is None:
            placed = local
        else:
            shared = [v for v in f if v in prev_face]
            if len(shared) < 2:
                raise ValueError("consecutive faces must share an edge")
            a, b = shared[0], shared[1]
            placed = _place(local, f.index(a), f.index(b),
                            prev_2d[prev_face.index(a)], prev_2d[prev_face.index(b)],
                            prev_2d.mean(axis=0))
        layout.append(placed)
        prev_face, prev_2d = f, placed
    return float(sum(_shoelace(p) for p in layout)), layout
```

`devloft/core/report.py`:
```python
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
    parts = [f"DevLoft: {report.ruling_count} rulings", f"max twist {mx} deg"]
    if ranges:
        parts.append(f"{report.failing_ruling_count} over {report.twist_tolerance:.1f} deg at rulings "
                     + ", ".join(_fmt_range(r) for r in ranges))
    else:
        parts.append(f"all within tolerance {report.twist_tolerance:.1f} deg")
    parts.append(f"{report.split_quad_count}/{report.quad_count} quads split")
    parts.append(f"area {report.area_3d:.4g} -> {report.area_unfolded:.4g} unfolded")
    return ", ".join(parts)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_unfold.py tests/test_report.py -q`
Expected: 5 passed.

- [ ] **Step 5: Commit**

```bash
git add devloft/core/unfold.py devloft/core/report.py tests/test_unfold.py tests/test_report.py
git commit -m "Add unfold validation and report formatting

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_012kESBz7SJpRjCw2UpaQv9E"
```

---
### Task 7: `loft()` entry point, test cases, JSON export

**Files:**
- Modify: `devloft/core/__init__.py`
- Create: `devloft/core/export.py`, `tests/cases.py`, `tests/test_loft.py`

**Interfaces:**
- Consumes: everything from Tasks 2 to 6.
- Produces:
  - `devloft.core.loft(points_a, points_b, params: LoftParams | None = None, tangents_a=None, tangents_b=None) -> StripResult`
  - `devloft.core.export.result_to_dict(result, points_a, points_b, name, params) -> dict` JSON-serializable (inf becomes None).
  - `tests.cases.CASES: dict[str, Callable[[], dict]]` with keys `cylinder`, `cone`, `offset_cylinder`, `twisted`. Each dict has `points_a`, `points_b`, `tangents_a`, `tangents_b` (arrays or None) and `params` (kwargs for `LoftParams`).

- [ ] **Step 1: Write the test cases module**

`tests/cases.py`:
```python
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
```

- [ ] **Step 2: Write the failing tests**

`tests/test_loft.py`:
```python
import json

import numpy as np
import pytest

from devloft.core import LoftError, LoftParams, loft
from devloft.core.export import result_to_dict
from tests.cases import CASES, OFFSET_STEPS


def _run(name, **overrides):
    case = CASES[name]()
    params = LoftParams(**{**case["params"], **overrides})
    res = loft(case["points_a"], case["points_b"], params, case["tangents_a"], case["tangents_b"])
    return case, params, res


def _line_point_distance(p, a, b):
    d = b - a
    return np.linalg.norm(np.cross(p - a, d)) / np.linalg.norm(d)


def test_cylinder_pairs_by_index_with_zero_twist():
    _, params, res = _run("cylinder")
    n = params.samples
    assert res.rulings == [(i, i) for i in range(n)]
    assert res.ruling_twist.max() < 0.01
    assert not res.face_split.any()
    assert res.failing_ranges == []
    assert res.report.area_unfolded == pytest.approx(res.report.area_3d, rel=1e-3)
    assert res.verts.shape == (2 * n, 3)
    assert len(res.layout) == len(res.faces)


def test_cylinder_without_tangents_still_zero_twist():
    case = CASES["cylinder"]()
    res = loft(case["points_a"], case["points_b"], LoftParams())
    assert res.ruling_twist.max() < 0.05
    assert res.failing_ranges == []


def test_cone_rulings_converge_to_apex():
    _, params, res = _run("cone")
    n = params.samples
    apex = np.array([0.0, 0.0, 2.0])
    for i, j in res.rulings:
        assert _line_point_distance(apex, res.verts[i], res.verts[n + j]) < 1e-6
    assert res.ruling_twist.max() < 0.01
    assert not res.face_split.any()


def test_offset_cylinder_leans_rulings():
    _, params, res = _run("offset_cylinder")
    n = params.samples
    middle = [(i, j) for i, j in res.rulings if 12 <= i <= n - 12]
    assert middle and all(j - i == -OFFSET_STEPS for i, j in middle)
    mid_twist = [tw for (i, j), tw in zip(res.rulings, res.ruling_twist) if 12 <= i <= n - 12]
    assert max(mid_twist) < 0.1
    assert res.failing_ranges
    assert res.failing_ranges[0][0] == 0
    assert res.failing_ranges[-1][1] == len(res.rulings) - 1
    assert res.rulings[0] == (0, 0) and res.rulings[-1] == (n - 1, n - 1)


def test_twisted_flags_high_twist_end():
    _, params, res = _run("twisted")
    assert res.report.max_twist > params.twist_tolerance
    assert res.failing_ranges
    assert res.failing_ranges[-1][1] == len(res.rulings) - 1
    assert res.report.area_unfolded == pytest.approx(res.report.area_3d, rel=1e-2)
    assert len(res.face_twist) == len(res.faces) == len(res.face_planarity) == len(res.face_split)


def test_twisted_coarse_splits_quads_when_not_planarized():
    _, _, res = _run("twisted", samples=24, planarize=False)
    assert res.report.split_quad_count > 0
    assert res.face_split.sum() == 2 * res.report.split_quad_count


def test_planarize_keeps_rail_endpoints():
    case = CASES["twisted"]()
    params = LoftParams(samples=24)
    res = loft(case["points_a"], case["points_b"], params, case["tangents_a"], case["tangents_b"])
    n = params.samples
    assert np.allclose(res.verts[0], case["points_a"][0])
    assert np.allclose(res.verts[n - 1], case["points_a"][-1])
    assert np.allclose(res.verts[n], case["points_b"][0])
    assert np.allclose(res.verts[2 * n - 1], case["points_b"][-1])


def test_tie_breakers_run():
    for mode in ("shortest", "plane"):
        _, _, res = _run("cylinder", tie_breaker=mode, tie_weight=1.0)
        assert res.ruling_twist.max() < 0.01


def test_invalid_rails_raise():
    with pytest.raises(LoftError):
        loft(np.zeros((5, 3)), np.ones((5, 3)))


def test_result_to_dict_is_json_serializable():
    case, params, res = _run("twisted", samples=12)
    d = result_to_dict(res, case["points_a"], case["points_b"], "twisted", params)
    s = json.dumps(d)
    back = json.loads(s)
    assert back["name"] == "twisted"
    assert len(back["verts"]) == 24
    assert len(back["faces"]) == len(res.faces)
    assert len(back["layout"]) == len(res.faces)
    assert back["report"]["ruling_count"] == len(res.rulings)
    assert back["params"]["samples"] == 12
    assert len(back["rails"]["a"]) == len(case["points_a"])
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `uv run pytest tests/test_loft.py -q`
Expected: FAIL with `ImportError: cannot import name 'loft'`.

- [ ] **Step 4: Write `loft()` and the exporter**

`devloft/core/__init__.py`:
```python
"""Pure-numpy developable loft. No Blender imports here."""
from __future__ import annotations

import numpy as np

from .align import align, tie_matrix
from .errors import LoftError
from .mesh import build_faces, face_planarity, mesh_area, planarize, split_quads
from .rails import prepare_rails
from .report import failing_ranges, format_report
from .twist import twist_matrix
from .types import LoftParams, Rail, Report, StripResult
from .unfold import unfold_strip

__all__ = ["LoftError", "LoftParams", "Rail", "Report", "StripResult", "loft", "format_report"]


def loft(points_a, points_b, params: LoftParams | None = None,
         tangents_a=None, tangents_b=None) -> StripResult:
    params = params or LoftParams()
    n = params.samples
    rail_a, rail_b = prepare_rails(points_a, points_b, n, tangents_a, tangents_b)

    twist = twist_matrix(rail_a, rail_b, params.window)
    cost = twist + params.tie_weight * tie_matrix(rail_a, rail_b, params.tie_breaker, params.plane_normal)
    path = align(cost)
    ruling_twist = np.array([twist[i, j] for i, j in path], dtype=float)

    verts = np.vstack([rail_a.points, rail_b.points])
    faces = build_faces(path, n)
    face_twist = np.array([max(ruling_twist[k], ruling_twist[k + 1]) for k in range(len(faces))])

    if params.planarize:
        lengths = [np.linalg.norm(verts[n + j] - verts[i]) for i, j in path]
        max_nudge = params.planar_tolerance * float(np.mean(lengths))
        verts = planarize(verts, faces, pinned=[0, n - 1, n, 2 * n - 1],
                          tolerance=params.planar_tolerance,
                          iterations=params.planarize_iterations, max_nudge=max_nudge)

    planarity = face_planarity(verts, faces)
    out_faces, src, split = split_quads(verts, faces, planarity, params.planar_tolerance)
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
        failing_ranges=ranges, report=report, layout=layout,
    )
```

`devloft/core/export.py`:
```python
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
```

- [ ] **Step 5: Run the whole suite**

Run: `uv run pytest -q`
Expected: all passed. If `test_offset_cylinder_leans_rulings` fails on the sign of `j - i`, print `res.rulings` and check `tests/cases.py`: B is phase-shifted by `+delta`, so B[j] matches A[i] when `j = i - OFFSET_STEPS`. Do not loosen the assertion to `abs()`.

- [ ] **Step 6: Commit**

```bash
git add devloft/core/__init__.py devloft/core/export.py tests/cases.py tests/test_loft.py
git commit -m "Add loft entry point, test rail cases and JSON export

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_012kESBz7SJpRjCw2UpaQv9E"
```

---
### Task 8: Browser viewer

**Files:**
- Create: `tools/viewer/export_cases.py`, `tools/viewer/index.html`, `tools/viewer/data.js` (generated), `tools/viewer/*.json` (generated), `tests/test_export_cases.py`

**Interfaces:**
- Consumes: `loft`, `LoftParams`, `result_to_dict`, `tests.cases.CASES`.
- Produces: `tools/viewer/data.js` defining `window.DEVLOFT_CASES` as a list of `result_to_dict` outputs; `export_cases.main(out_dir: Path) -> list[Path]`.

- [ ] **Step 1: Write the failing test**

`tests/test_export_cases.py`:
```python
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools" / "viewer"))

import export_cases  # noqa: E402


def test_export_writes_data_js_and_json(tmp_path):
    written = export_cases.main(tmp_path)
    names = {p.name for p in written}
    assert "data.js" in names
    assert {"cylinder.json", "cone.json", "offset_cylinder.json", "twisted.json"} <= names
    js = (tmp_path / "data.js").read_text()
    assert js.startswith("window.DEVLOFT_CASES = [")
    payload = json.loads(js[len("window.DEVLOFT_CASES = "):].rstrip().rstrip(";"))
    assert [c["name"] for c in payload] == ["cylinder", "cone", "offset_cylinder", "twisted"]
    cyl = json.loads((tmp_path / "cylinder.json").read_text())
    assert cyl["report"]["failing_ruling_count"] == 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_export_cases.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'export_cases'`.

- [ ] **Step 3: Write the exporter**

`tools/viewer/export_cases.py`:
```python
"""Run the core on the test rail sets and write viewer data.

Usage: uv run python tools/viewer/export_cases.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from devloft.core import LoftParams, loft  # noqa: E402
from devloft.core.export import result_to_dict  # noqa: E402
from tests.cases import CASES  # noqa: E402


def main(out_dir: Path = HERE) -> list[Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    payload = []
    for name, build in CASES.items():
        case = build()
        params = LoftParams(**case["params"])
        result = loft(case["points_a"], case["points_b"], params, case["tangents_a"], case["tangents_b"])
        data = result_to_dict(result, case["points_a"], case["points_b"], name, params)
        payload.append(data)
        path = out_dir / f"{name}.json"
        path.write_text(json.dumps(data))
        written.append(path)
        print(f"{name}: {data['report']}")
    js = out_dir / "data.js"
    js.write_text("window.DEVLOFT_CASES = " + json.dumps(payload) + ";\n")
    written.append(js)
    return written


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run the test, then generate the real data**

Run: `uv run pytest tests/test_export_cases.py -q && uv run python tools/viewer/export_cases.py`
Expected: 1 passed, four report lines printed, `tools/viewer/data.js` and four `.json` files created.

- [ ] **Step 5: Write the viewer page**

`tools/viewer/index.html`:
```html
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>DevLoft viewer</title>
<style>
  html, body { margin: 0; height: 100%; font: 13px/1.4 -apple-system, system-ui, sans-serif; background: #1e1e22; color: #ddd; }
  #app { display: flex; height: 100%; }
  #sidebar { width: 320px; padding: 12px; box-sizing: border-box; overflow-y: auto; background: #26262b; border-right: 1px solid #333; }
  #view { flex: 1; position: relative; }
  #view canvas { display: block; }
  h1 { font-size: 15px; margin: 0 0 10px; }
  h2 { font-size: 12px; margin: 14px 0 6px; color: #aaa; text-transform: uppercase; letter-spacing: .04em; }
  button { display: block; width: 100%; text-align: left; margin: 2px 0; padding: 5px 8px; background: #333; color: #ddd; border: 1px solid #444; border-radius: 4px; cursor: pointer; }
  button.active { background: #4a6cf7; border-color: #4a6cf7; color: #fff; }
  label { display: block; margin: 3px 0; }
  #legend { height: 12px; border-radius: 3px; background: linear-gradient(90deg, #22cc44, #ffdd22 50%, #ee2222); }
  #legend-labels { display: flex; justify-content: space-between; color: #aaa; font-size: 11px; }
  pre { white-space: pre-wrap; background: #1a1a1e; padding: 8px; border-radius: 4px; font-size: 11px; }
  #flat { width: 100%; background: #1a1a1e; border-radius: 4px; }
  input[type=file] { width: 100%; font-size: 11px; }
  .hint { color: #888; font-size: 11px; }
</style>
<script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
<script src="data.js"></script>
</head>
<body>
<div id="app">
  <div id="sidebar">
    <h1>DevLoft viewer</h1>
    <h2>Cases</h2>
    <div id="cases"></div>
    <input type="file" id="file" accept=".json">
    <p class="hint">Drag to orbit, shift-drag or right-drag to pan, wheel to zoom.</p>
    <h2>Show</h2>
    <label><input type="checkbox" id="showFaces" checked> faces (colored by twist)</label>
    <label><input type="checkbox" id="showWire" checked> wireframe</label>
    <label><input type="checkbox" id="showRulings" checked> rulings</label>
    <label><input type="checkbox" id="showRails" checked> input rails</label>
    <h2>Twist legend</h2>
    <div id="legend"></div>
    <div id="legend-labels"><span>0</span><span id="legend-mid">tol</span><span id="legend-max">2 tol</span></div>
    <h2>Report</h2>
    <pre id="report"></pre>
    <h2>Flat pattern</h2>
    <canvas id="flat" width="296" height="220"></canvas>
  </div>
  <div id="view"></div>
</div>
<script>
(function () {
  if (typeof THREE === 'undefined') {
    document.getElementById('view').innerHTML = '<p style="padding:20px">three.js failed to load (offline?).</p>';
    return;
  }
  const cases = (window.DEVLOFT_CASES || []).slice();
  const view = document.getElementById('view');
  const renderer = new THREE.WebGLRenderer({ antialias: true });
  renderer.setPixelRatio(window.devicePixelRatio || 1);
  view.appendChild(renderer.domElement);
  const scene = new THREE.Scene();
  scene.background = new THREE.Color(0x1e1e22);
  const camera = new THREE.PerspectiveCamera(45, 1, 0.01, 1000);
  scene.add(new THREE.AmbientLight(0xffffff, 0.55));
  const sun = new THREE.DirectionalLight(0xffffff, 0.7);
  sun.position.set(2, 4, 3);
  scene.add(sun);
  scene.add(new THREE.GridHelper(4, 8, 0x444444, 0x333333));

  // Blender is Z-up, three.js is Y-up.
  const toThree = p => [p[0], p[2], -p[1]];

  const orbit = { theta: 0.8, phi: 1.05, dist: 5, target: new THREE.Vector3() };
  function updateCamera() {
    const { theta, phi, dist, target } = orbit;
    camera.position.set(
      target.x + dist * Math.sin(phi) * Math.cos(theta),
      target.y + dist * Math.cos(phi),
      target.z + dist * Math.sin(phi) * Math.sin(theta));
    camera.lookAt(target);
  }
  function resize() {
    const w = view.clientWidth, h = view.clientHeight;
    renderer.setSize(w, h);
    camera.aspect = w / h;
    camera.updateProjectionMatrix();
  }
  window.addEventListener('resize', () => { resize(); render(); });

  let drag = null;
  const el = renderer.domElement;
  el.addEventListener('contextmenu', e => e.preventDefault());
  el.addEventListener('mousedown', e => { drag = { x: e.clientX, y: e.clientY, pan: e.button === 2 || e.shiftKey }; });
  window.addEventListener('mouseup', () => { drag = null; });
  window.addEventListener('mousemove', e => {
    if (!drag) return;
    const dx = e.clientX - drag.x, dy = e.clientY - drag.y;
    drag.x = e.clientX; drag.y = e.clientY;
    if (drag.pan) {
      const right = new THREE.Vector3().setFromMatrixColumn(camera.matrix, 0);
      const up = new THREE.Vector3().setFromMatrixColumn(camera.matrix, 1);
      const k = orbit.dist * 0.0015;
      orbit.target.addScaledVector(right, -dx * k).addScaledVector(up, dy * k);
    } else {
      orbit.theta += dx * 0.006;
      orbit.phi = Math.min(Math.PI - 0.05, Math.max(0.05, orbit.phi + dy * 0.006));
    }
    updateCamera(); render();
  });
  el.addEventListener('wheel', e => {
    e.preventDefault();
    orbit.dist *= Math.exp(e.deltaY * 0.001);
    updateCamera(); render();
  }, { passive: false });

  function colorFor(twist, tol) {
    if (twist === null || twist === undefined) return [1, 0, 1];
    const t = twist / tol;
    const g = [0.13, 0.8, 0.27], y = [1, 0.87, 0.13], r = [0.93, 0.13, 0.13];
    const mix = (a, b, s) => a.map((v, i) => v + (b[i] - v) * s);
    if (t <= 1) return mix(g, y, t);
    return mix(y, r, Math.min(1, t - 1));
  }

  let group = null;
  let layers = {};
  function buildCase(c) {
    if (group) scene.remove(group);
    group = new THREE.Group();
    layers = {};
    const tol = c.params.twist_tolerance;
    const pos = [], col = [], wire = [];
    c.faces.forEach((f, k) => {
      const rgb = colorFor(c.face_twist[k], tol);
      const tris = f.length === 4 ? [[f[0], f[1], f[2]], [f[0], f[2], f[3]]] : [f];
      tris.forEach(t => t.forEach(vi => { pos.push(...toThree(c.verts[vi])); col.push(...rgb); }));
      for (let e = 0; e < f.length; e++) {
        wire.push(...toThree(c.verts[f[e]]), ...toThree(c.verts[f[(e + 1) % f.length]]));
      }
    });
    const geom = new THREE.BufferGeometry();
    geom.setAttribute('position', new THREE.Float32BufferAttribute(pos, 3));
    geom.setAttribute('color', new THREE.Float32BufferAttribute(col, 3));
    geom.computeVertexNormals();
    layers.faces = new THREE.Mesh(geom, new THREE.MeshLambertMaterial({
      vertexColors: true, side: THREE.DoubleSide, polygonOffset: true, polygonOffsetFactor: 1, polygonOffsetUnits: 1 }));
    const wgeom = new THREE.BufferGeometry();
    wgeom.setAttribute('position', new THREE.Float32BufferAttribute(wire, 3));
    layers.wire = new THREE.LineSegments(wgeom, new THREE.LineBasicMaterial({ color: 0x111111 }));

    const n = c.params.samples;
    const rp = [], rc = [];
    c.rulings.forEach(([i, j], k) => {
      const rgb = colorFor(c.ruling_twist[k], tol);
      rp.push(...toThree(c.verts[i]), ...toThree(c.verts[n + j]));
      rc.push(...rgb, ...rgb);
    });
    const rgeom = new THREE.BufferGeometry();
    rgeom.setAttribute('position', new THREE.Float32BufferAttribute(rp, 3));
    rgeom.setAttribute('color', new THREE.Float32BufferAttribute(rc, 3));
    layers.rulings = new THREE.LineSegments(rgeom, new THREE.LineBasicMaterial({ vertexColors: true }));

    const rails = new THREE.Group();
    [c.rails.a, c.rails.b].forEach((pts, idx) => {
      const curve = new THREE.CatmullRomCurve3(pts.map(p => new THREE.Vector3(...toThree(p))));
      const tube = new THREE.TubeGeometry(curve, Math.max(8, pts.length), 0.006 * orbitScale(c), 6, false);
      rails.add(new THREE.Mesh(tube, new THREE.MeshBasicMaterial({ color: idx === 0 ? 0x66aaff : 0xffaa66 })));
    });
    layers.rails = rails;

    Object.values(layers).forEach(o => group.add(o));
    scene.add(group);
    fitCamera(c);
    applyVisibility();
    document.getElementById('legend-mid').textContent = tol.toFixed(1) + '°';
    document.getElementById('legend-max').textContent = (2 * tol).toFixed(1) + '°+';
    document.getElementById('report').textContent = reportText(c);
    drawFlat(c);
    render();
  }
  function orbitScale(c) {
    const b = new THREE.Box3();
    c.verts.forEach(p => b.expandByPoint(new THREE.Vector3(...toThree(p))));
    return b.getSize(new THREE.Vector3()).length() || 1;
  }
  function fitCamera(c) {
    const b = new THREE.Box3();
    c.verts.forEach(p => b.expandByPoint(new THREE.Vector3(...toThree(p))));
    b.getCenter(orbit.target);
    orbit.dist = b.getSize(new THREE.Vector3()).length() * 1.4 || 5;
    updateCamera();
  }
  function reportText(c) {
    const r = c.report;
    const fmt = v => (v === null ? 'inf' : Number(v).toFixed(2));
    const ranges = c.failing_ranges.map(([a, b]) => (a === b ? `${a}` : `${a}-${b}`)).join(', ') || 'none';
    return [
      `${c.name}`,
      `samples ${c.params.samples}, window ${c.params.window}, tolerance ${c.params.twist_tolerance}°`,
      `rulings: ${r.ruling_count}`,
      `max twist: ${fmt(r.max_twist)}°   mean: ${fmt(r.mean_twist)}°`,
      `failing rulings: ${r.failing_ruling_count}  at ${ranges}`,
      `quads split: ${r.split_quad_count}/${r.quad_count}`,
      `area 3D: ${fmt(r.area_3d)}   unfolded: ${fmt(r.area_unfolded)}`,
    ].join('\n');
  }
  function drawFlat(c) {
    const cv = document.getElementById('flat');
    const ctx = cv.getContext('2d');
    ctx.clearRect(0, 0, cv.width, cv.height);
    if (!c.layout.length) return;
    let minx = Infinity, miny = Infinity, maxx = -Infinity, maxy = -Infinity;
    c.layout.forEach(poly => poly.forEach(([x, y]) => {
      minx = Math.min(minx, x); maxx = Math.max(maxx, x); miny = Math.min(miny, y); maxy = Math.max(maxy, y);
    }));
    const pad = 8;
    const s = Math.min((cv.width - 2 * pad) / (maxx - minx || 1), (cv.height - 2 * pad) / (maxy - miny || 1));
    const ox = pad + ((cv.width - 2 * pad) - s * (maxx - minx)) / 2;
    const oy = pad + ((cv.height - 2 * pad) - s * (maxy - miny)) / 2;
    const tol = c.params.twist_tolerance;
    c.layout.forEach((poly, k) => {
      const [r, g, b] = colorFor(c.face_twist[k], tol).map(v => Math.round(v * 255));
      ctx.beginPath();
      poly.forEach(([x, y], i) => {
        const px = ox + (x - minx) * s, py = cv.height - (oy + (y - miny) * s);
        if (i === 0) ctx.moveTo(px, py); else ctx.lineTo(px, py);
      });
      ctx.closePath();
      ctx.fillStyle = `rgb(${r},${g},${b})`;
      ctx.fill();
      ctx.strokeStyle = '#111';
      ctx.lineWidth = 0.6;
      ctx.stroke();
    });
  }
  function applyVisibility() {
    if (!layers.faces) return;
    layers.faces.visible = document.getElementById('showFaces').checked;
    layers.wire.visible = document.getElementById('showWire').checked;
    layers.rulings.visible = document.getElementById('showRulings').checked;
    layers.rails.visible = document.getElementById('showRails').checked;
  }
  ['showFaces', 'showWire', 'showRulings', 'showRails'].forEach(id =>
    document.getElementById(id).addEventListener('change', () => { applyVisibility(); render(); }));

  function render() { renderer.render(scene, camera); }

  function rebuildButtons(activeIdx) {
    const box = document.getElementById('cases');
    box.innerHTML = '';
    cases.forEach((c, i) => {
      const b = document.createElement('button');
      b.textContent = c.name;
      if (i === activeIdx) b.classList.add('active');
      b.onclick = () => { rebuildButtons(i); buildCase(c); };
      box.appendChild(b);
    });
  }
  document.getElementById('file').addEventListener('change', e => {
    const f = e.target.files[0];
    if (!f) return;
    const reader = new FileReader();
    reader.onload = () => {
      try {
        const data = JSON.parse(reader.result);
        data.name = data.name || f.name;
        cases.push(data);
        rebuildButtons(cases.length - 1);
        buildCase(data);
      } catch (err) {
        alert('Could not read JSON: ' + err.message);
      }
    };
    reader.readAsText(f);
  });

  resize();
  if (cases.length) { rebuildButtons(0); buildCase(cases[0]); } else { render(); }
})();
</script>
</body>
</html>
```

- [ ] **Step 6: Open the page and check it visually**

Run: `open tools/viewer/index.html`
Expected: four case buttons; cylinder shows all-green faces and parallel rulings; cone shows converging rulings; offset_cylinder shows leaning rulings with red ends; twisted shows a green-to-red gradient and a flat pattern at the bottom of the sidebar. If the page is blank, open the browser console and report the error.

- [ ] **Step 7: Commit**

```bash
git add tools/viewer tests/test_export_cases.py
git commit -m "Add browser viewer and case exporter

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_012kESBz7SJpRjCw2UpaQv9E"
```

---
### Task 9: Blender layer: manifest, inputs, output, operator, panel, headless tests

**Files:**
- Create: `devloft/blender_manifest.toml`, `devloft/blender/__init__.py`, `devloft/blender/inputs.py`, `devloft/blender/output.py`, `devloft/blender/operator.py`, `devloft/blender/panel.py`, `tests/test_blender.py`

**Interfaces:**
- Consumes: `loft`, `LoftParams`, `LoftError`, `format_report`, `result_to_dict`.
- Produces:
  - `inputs.get_rails(context, samples) -> tuple[tuple[np.ndarray, np.ndarray | None], tuple[np.ndarray, np.ndarray | None]]` = ((points_a, tangents_a), (points_b, tangents_b)) in world space.
  - `inputs.chains_from_edges(coords: list, edges: list[tuple[int,int]]) -> list[list[int]]` ordered vertex-index chains (pure Python, unit-testable without bpy).
  - `output.create_strip_object(context, result, name, twist_tolerance) -> bpy.types.Object`
  - `output.twist_colors(face_twist: np.ndarray, tolerance: float) -> np.ndarray` shape `(F, 4)`.
  - Operator id `devloft.loft`, class `DEVLOFT_OT_loft`; panel `DEVLOFT_PT_panel`.

- [ ] **Step 1: Write the failing tests**

`tests/test_blender.py`:
```python
import math

import numpy as np
import pytest

bpy = pytest.importorskip("bpy")  # devloft.blender imports bpy at package level

import devloft  # noqa: E402
from devloft.blender.inputs import chains_from_edges  # noqa: E402
from devloft.blender.output import twist_colors  # noqa: E402
from devloft.core.errors import LoftError  # noqa: E402


def test_chains_from_edges_two_open_chains():
    coords = [(i, 0, 0) for i in range(4)] + [(i, 1, 0) for i in range(3)]
    edges = [(0, 1), (1, 2), (2, 3), (4, 5), (5, 6)]
    chains = chains_from_edges(coords, edges)
    assert sorted(chains) == [[0, 1, 2, 3], [4, 5, 6]]


def test_chains_from_edges_rejects_closed_and_branching():
    with pytest.raises(LoftError):
        chains_from_edges([(0, 0, 0)] * 3, [(0, 1), (1, 2), (2, 0)])
    with pytest.raises(LoftError):
        chains_from_edges([(0, 0, 0)] * 4, [(0, 1), (1, 2), (1, 3)])


def test_twist_colors_ramp():
    c = twist_colors(np.array([0.0, 5.0, 10.0, 50.0, np.inf]), 5.0)
    assert c.shape == (5, 4)
    assert c[0][1] > c[0][0]            # green at 0
    assert c[1][0] > 0.9 and c[1][1] > 0.8   # yellow at tol
    assert c[2][0] > 0.9 and c[2][1] < 0.2   # red at 2 tol
    assert np.allclose(c[3], c[2])
    assert np.allclose(c[:, 3], 1.0)


@pytest.fixture
def fresh_scene():
    bpy.ops.wm.read_factory_settings(use_empty=True)
    try:
        devloft.register()
    except ValueError:
        pass  # already registered
    yield bpy.context


def _make_arc_curve(name, radius, z, n=5):
    cu = bpy.data.curves.new(name, "CURVE")
    cu.dimensions = "3D"
    sp = cu.splines.new("BEZIER")
    sp.bezier_points.add(n - 1)
    step = math.pi / (n - 1)
    handle = radius * (4.0 / 3.0) * math.tan(step / 4.0)
    for k, bp in enumerate(sp.bezier_points):
        th = k * step
        co = (radius * math.cos(th), radius * math.sin(th), z)
        t = (-math.sin(th), math.cos(th), 0.0)
        bp.handle_left_type = bp.handle_right_type = "FREE"
        bp.co = co
        bp.handle_left = tuple(c - handle * tv for c, tv in zip(co, t))
        bp.handle_right = tuple(c + handle * tv for c, tv in zip(co, t))
    obj = bpy.data.objects.new(name, cu)
    bpy.context.collection.objects.link(obj)
    return obj


def _select(objs, active):
    for o in bpy.context.view_layer.objects:
        o.select_set(False)
    for o in objs:
        o.select_set(True)
    bpy.context.view_layer.objects.active = active


def test_operator_on_curves(fresh_scene):
    a = _make_arc_curve("A", 1.0, 0.0)
    b = _make_arc_curve("B", 1.0, 1.0)
    _select([a, b], a)
    result = bpy.ops.devloft.loft(samples=30)
    assert result == {"FINISHED"}
    obj = bpy.data.objects["DevLoft"]
    me = obj.data
    assert len(me.vertices) == 60
    assert 29 <= len(me.polygons) <= 60
    for attr in ("twist", "planarity", "twist_color", "split"):
        assert attr in me.attributes
    twist = [d.value for d in me.attributes["twist"].data]
    assert max(twist) < 1.0
    assert bpy.context.view_layer.objects.active == obj


def test_operator_on_edit_mode_chains(fresh_scene):
    me = bpy.data.meshes.new("rails")
    verts = [(i * 0.5, 0.0, 0.0) for i in range(6)] + [(i * 0.5, 0.0, 1.0) for i in range(6)]
    edges = [(i, i + 1) for i in range(5)] + [(6 + i, 7 + i) for i in range(5)]
    me.from_pydata(verts, edges, [])
    me.update()
    for e in me.edges:
        e.select = True
    obj = bpy.data.objects.new("rails", me)
    bpy.context.collection.objects.link(obj)
    _select([obj], obj)
    bpy.ops.object.mode_set(mode="EDIT")
    result = bpy.ops.devloft.loft(samples=12)
    assert result == {"FINISHED"}
    assert bpy.context.mode == "OBJECT"
    out = bpy.data.objects["DevLoft"]
    assert len(out.data.vertices) == 24
    assert "twist" in out.data.attributes


def test_operator_errors_on_bad_selection(fresh_scene):
    a = _make_arc_curve("A", 1.0, 0.0)
    _select([a], a)
    assert bpy.ops.devloft.loft() == {"CANCELLED"}
    assert "DevLoft" not in bpy.data.objects
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_blender.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'devloft.blender'` (or all skipped if bpy is missing, which means Task 1 Step 2 did not install it).

- [ ] **Step 3: Write the manifest**

`devloft/blender_manifest.toml`:
```toml
schema_version = "1.0.0"

id = "devloft"
version = "0.1.0"
name = "DevLoft"
tagline = "Developable loft between two rails for papercraft"
maintainer = "Vincent Chang"
type = "add-on"
blender_version_min = "4.2.0"
license = ["SPDX:GPL-3.0-or-later"]
tags = ["Mesh", "Modeling"]
```

- [ ] **Step 4: Write inputs.py**

`devloft/blender/inputs.py`:
```python
"""Read two rails from the Blender selection as world-space polylines."""
from __future__ import annotations

import math

import numpy as np

from ..core.errors import LoftError


def chains_from_edges(coords, edges):
    """Group edges into ordered open chains of vertex indices.

    Pure Python so it can be unit-tested without bpy. Raises LoftError for
    closed loops or branching selections.
    """
    adj: dict[int, list[int]] = {}
    for a, b in edges:
        adj.setdefault(a, []).append(b)
        adj.setdefault(b, []).append(a)
    for v, nb in adj.items():
        if len(nb) > 2:
            raise LoftError(f"selected edges branch at vertex {v}; select two simple edge chains")
    seen = set()
    chains = []
    ends = sorted(v for v, nb in adj.items() if len(nb) == 1)
    for start in ends:
        if start in seen:
            continue
        chain = [start]
        seen.add(start)
        prev, cur = None, start
        while True:
            nxt = [w for w in adj[cur] if w != prev]
            if not nxt:
                break
            prev, cur = cur, nxt[0]
            if cur in seen:
                break
            chain.append(cur)
            seen.add(cur)
        chains.append(chain)
    if len(seen) != len(adj):
        raise LoftError("a selected edge loop is closed; rails must be open chains")
    return chains


def _bezier_rail(obj, samples):
    from mathutils.geometry import interpolate_bezier

    spline = obj.data.splines[0]
    if spline.use_cyclic_u:
        raise LoftError(f"'{obj.name}' is a closed curve; rails must be open")
    bps = spline.bezier_points
    nseg = len(bps) - 1
    if nseg < 1:
        raise LoftError(f"'{obj.name}' needs at least 2 control points")
    res = max(2, math.ceil(samples * 4 / nseg) + 1)
    mw = obj.matrix_world
    rot = mw.to_3x3()
    pts, tans = [], []
    for k in range(nseg):
        p0, p1 = bps[k], bps[k + 1]
        seg = interpolate_bezier(p0.co, p0.handle_right, p1.handle_left, p1.co, res)
        for idx, q in enumerate(seg):
            if k > 0 and idx == 0:
                continue
            t = idx / (res - 1)
            d = (3 * (1 - t) ** 2 * (p0.handle_right - p0.co)
                 + 6 * (1 - t) * t * (p1.handle_left - p0.handle_right)
                 + 3 * t ** 2 * (p1.co - p1.handle_left))
            pts.append(mw @ q)
            tans.append(rot @ d)
    return np.array([list(p) for p in pts], float), np.array([list(t) for t in tans], float)


def _poly_rail(obj):
    spline = obj.data.splines[0]
    if spline.use_cyclic_u:
        raise LoftError(f"'{obj.name}' is a closed curve; rails must be open")
    mw = obj.matrix_world
    return np.array([list(mw @ p.co.xyz) for p in spline.points], float), None


def _evaluated_rail(obj, context):
    dg = context.evaluated_depsgraph_get()
    ev = obj.evaluated_get(dg)
    me = ev.to_mesh()
    try:
        coords = [tuple(v.co) for v in me.vertices]
        edges = [tuple(e.vertices) for e in me.edges]
    finally:
        ev.to_mesh_clear()
    chains = chains_from_edges(coords, edges)
    if len(chains) != 1:
        raise LoftError(f"'{obj.name}' evaluates to {len(chains)} chains; expected 1")
    from mathutils import Vector

    mw = obj.matrix_world
    return np.array([list(mw @ Vector(coords[i])) for i in chains[0]], float), None


def curve_rail(obj, context, samples):
    cu = obj.data
    if len(cu.splines) != 1:
        raise LoftError(f"'{obj.name}' has {len(cu.splines)} splines; separate them so each rail has one")
    kind = cu.splines[0].type
    if kind == "BEZIER":
        return _bezier_rail(obj, samples)
    if kind == "POLY":
        return _poly_rail(obj)
    return _evaluated_rail(obj, context)


def edit_mode_rails(obj):
    import bmesh

    bm = bmesh.from_edit_mesh(obj.data)
    bm.verts.ensure_lookup_table()
    edges = [(e.verts[0].index, e.verts[1].index) for e in bm.edges if e.select]
    if not edges:
        raise LoftError("select the edges of two rail chains")
    coords = [tuple(v.co) for v in bm.verts]
    chains = chains_from_edges(coords, edges)
    if len(chains) != 2:
        raise LoftError(f"{len(chains)} edge chains selected; expected 2")
    chains.sort(key=min)
    mw = obj.matrix_world
    rails = []
    for chain in chains:
        rails.append((np.array([list(mw @ bm.verts[i].co) for i in chain], float), None))
    return rails[0], rails[1]


def get_rails(context, samples):
    obj = context.active_object
    if context.mode == "EDIT_MESH" and obj is not None and obj.type == "MESH":
        return edit_mode_rails(obj)
    curves = [o for o in context.selected_objects if o.type == "CURVE"]
    if len(curves) != 2:
        raise LoftError(f"select exactly two curve objects ({len(curves)} selected), "
                        "or two edge chains in Edit Mode")
    if obj in curves:
        a = obj
        b = curves[0] if curves[1] is obj else curves[1]
    else:
        a, b = curves
    return curve_rail(a, context, samples), curve_rail(b, context, samples)
```

- [ ] **Step 5: Write output.py**

`devloft/blender/output.py`:
```python
"""Create the result mesh object with twist attributes."""
from __future__ import annotations

import numpy as np

GREEN = np.array([0.13, 0.8, 0.27, 1.0])
YELLOW = np.array([1.0, 0.87, 0.13, 1.0])
RED = np.array([0.93, 0.13, 0.13, 1.0])


def twist_colors(face_twist, tolerance):
    tw = np.asarray(face_twist, dtype=float)
    t = np.where(np.isfinite(tw), tw / max(tolerance, 1e-9), 2.0)
    t = np.clip(t, 0.0, 2.0)
    out = np.empty((len(tw), 4))
    low = t <= 1.0
    out[low] = GREEN + (YELLOW - GREEN) * t[low, None]
    out[~low] = YELLOW + (RED - YELLOW) * (t[~low, None] - 1.0)
    return out


def create_strip_object(context, result, name, twist_tolerance):
    import bpy

    me = bpy.data.meshes.new(name)
    me.from_pydata(result.verts.tolist(), [], [list(map(int, f)) for f in result.faces])
    me.update()

    twist = me.attributes.new("twist", "FLOAT", "FACE")
    twist.data.foreach_set("value", np.where(np.isfinite(result.face_twist), result.face_twist, 90.0).astype(np.float32))
    plan = me.attributes.new("planarity", "FLOAT", "FACE")
    plan.data.foreach_set("value", result.face_planarity.astype(np.float32))
    col = me.attributes.new("twist_color", "FLOAT_COLOR", "FACE")
    col.data.foreach_set("color", twist_colors(result.face_twist, twist_tolerance).astype(np.float32).ravel())
    split = me.attributes.new("split", "BOOLEAN", "FACE")
    split.data.foreach_set("value", result.face_split.astype(bool))
    me.color_attributes.active_color = col
    me.update()

    obj = bpy.data.objects.new(name, me)
    context.collection.objects.link(obj)
    for o in context.view_layer.objects:
        o.select_set(False)
    obj.select_set(True)
    context.view_layer.objects.active = obj
    return obj
```

- [ ] **Step 6: Write operator.py, panel.py and the package init**

`devloft/blender/operator.py`:
```python
from __future__ import annotations

import json

import bpy
from bpy.props import BoolProperty, EnumProperty, FloatProperty, FloatVectorProperty, IntProperty, StringProperty

from ..core import LoftError, LoftParams, format_report, loft
from ..core.export import result_to_dict
from . import inputs, output


class DEVLOFT_OT_loft(bpy.types.Operator):
    """Loft a developable strip between two rail curves or edge chains"""
    bl_idname = "devloft.loft"
    bl_label = "Developable Loft (two rails)"
    bl_options = {"REGISTER", "UNDO"}

    samples: IntProperty(name="Samples", default=60, min=8, max=400,
                         description="Points per rail after resampling")
    window: IntProperty(name="Window", default=8, min=1, max=100,
                        description="How far rulings may lean, in samples")
    twist_tolerance: FloatProperty(name="Twist Tolerance", default=5.0, min=0.0, max=90.0, subtype="NONE",
                                   description="Rulings with more twist (degrees) are flagged")
    tie_breaker: EnumProperty(name="Tie Breaker", default="none", items=[
        ("none", "None", "Minimum twist only"),
        ("shortest", "Shortest Ruling", "Prefer shorter rulings"),
        ("plane", "Plane Direction", "Prefer rulings parallel to a plane normal"),
    ])
    tie_weight: FloatProperty(name="Tie Weight", default=0.1, min=0.0, max=10.0)
    plane_normal: FloatVectorProperty(name="Plane Normal", default=(0.0, 0.0, 1.0), subtype="XYZ")
    planarize: BoolProperty(name="Planarize", default=True,
                            description="Nudge vertices so near-planar quads become planar")
    planar_tolerance: FloatProperty(name="Planar Tolerance", default=0.01, min=0.0, max=0.5,
                                    description="Relative diagonal offset; quads above this are split")
    export_json: StringProperty(name="Export JSON", default="", subtype="FILE_PATH",
                                description="Optional path to write viewer JSON")

    def execute(self, context):
        params = LoftParams(
            samples=self.samples, window=self.window, twist_tolerance=self.twist_tolerance,
            tie_breaker=self.tie_breaker, tie_weight=self.tie_weight,
            plane_normal=tuple(self.plane_normal), planarize=self.planarize,
            planar_tolerance=self.planar_tolerance,
        )
        try:
            (pa, ta), (pb, tb) = inputs.get_rails(context, params.samples)
            if context.mode == "EDIT_MESH":
                bpy.ops.object.mode_set(mode="OBJECT")
            result = loft(pa, pb, params, ta, tb)
        except LoftError as exc:
            self.report({"ERROR"}, str(exc))
            return {"CANCELLED"}
        output.create_strip_object(context, result, "DevLoft", params.twist_tolerance)
        if self.export_json:
            path = bpy.path.abspath(self.export_json)
            with open(path, "w", encoding="utf-8") as fh:
                json.dump(result_to_dict(result, pa, pb, "blender", params), fh)
        summary = format_report(result.report, result.failing_ranges)
        self.report({"INFO"}, summary)
        if result.failing_ranges:
            ranges = ", ".join(f"{a}-{b}" if a != b else str(a) for a, b in result.failing_ranges)
            self.report({"WARNING"}, f"Not developable at rulings {ranges}; split the panel there")
        return {"FINISHED"}

    def draw(self, context):
        col = self.layout.column()
        col.prop(self, "samples")
        col.prop(self, "window")
        col.prop(self, "twist_tolerance")
        col.prop(self, "tie_breaker")
        if self.tie_breaker != "none":
            col.prop(self, "tie_weight")
        if self.tie_breaker == "plane":
            col.prop(self, "plane_normal")
        col.prop(self, "planarize")
        col.prop(self, "planar_tolerance")
        col.prop(self, "export_json")
```

`devloft/blender/panel.py`:
```python
import bpy

from .operator import DEVLOFT_OT_loft


class DEVLOFT_PT_panel(bpy.types.Panel):
    bl_label = "DevLoft"
    bl_idname = "DEVLOFT_PT_panel"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "DevLoft"

    def draw(self, context):
        col = self.layout.column(align=True)
        col.operator(DEVLOFT_OT_loft.bl_idname, icon="MOD_CURVE")
        col.separator()
        col.label(text="Select two curves (active = rail A),")
        col.label(text="or two edge chains in Edit Mode.")
        col.label(text="Settings: Adjust Last Operation panel.")
        col.separator()
        col.label(text="View twist: Solid shading > Color > Attribute")
```

`devloft/blender/__init__.py`:
```python
import bpy

from . import operator, panel

_classes = (operator.DEVLOFT_OT_loft, panel.DEVLOFT_PT_panel)


def _menu(self, context):
    self.layout.operator(operator.DEVLOFT_OT_loft.bl_idname, icon="MOD_CURVE")


def register():
    for cls in _classes:
        bpy.utils.register_class(cls)
    bpy.types.VIEW3D_MT_add.append(_menu)


def unregister():
    bpy.types.VIEW3D_MT_add.remove(_menu)
    for cls in reversed(_classes):
        bpy.utils.unregister_class(cls)
```

- [ ] **Step 7: Run the Blender tests**

Run: `uv run pytest tests/test_blender.py -q -p no:cacheprovider`
Expected: 6 passed. The first bpy import takes several seconds.

If `bpy.ops.object.mode_set` fails in background mode with a context error, wrap the call in the test and in `execute` with `context.temp_override(active_object=obj, object=obj, selected_objects=[obj])` and rerun. If `me.color_attributes.active_color = col` raises, drop that line; it is cosmetic.

- [ ] **Step 8: Run the full suite**

Run: `uv run pytest -q`
Expected: all passed.

- [ ] **Step 9: Commit**

```bash
git add devloft/blender_manifest.toml devloft/blender tests/test_blender.py
git commit -m "Add Blender operator, inputs, output mesh and panel

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_012kESBz7SJpRjCw2UpaQv9E"
```

---

### Task 10: Packaging script and README

**Files:**
- Create: `make_zip.py`, `README.md`, `tests/test_make_zip.py`

**Interfaces:**
- Produces: `make_zip.build(dist_dir: Path) -> Path` returning the zip path `dist/devloft-<version>.zip`, with `blender_manifest.toml` at the zip root.

- [ ] **Step 1: Write the failing test**

`tests/test_make_zip.py`:
```python
import zipfile

import make_zip


def test_zip_contains_manifest_and_package(tmp_path):
    path = make_zip.build(tmp_path)
    assert path.name == "devloft-0.1.0.zip"
    with zipfile.ZipFile(path) as z:
        names = z.namelist()
    assert "blender_manifest.toml" in names
    assert "__init__.py" in names
    assert "core/__init__.py" in names
    assert "blender/operator.py" in names
    assert not any("__pycache__" in n for n in names)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_make_zip.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'make_zip'`.

- [ ] **Step 3: Write the script**

`make_zip.py`:
```python
"""Build dist/devloft-<version>.zip for Install from Disk in Blender 4.2+."""
from __future__ import annotations

import tomllib
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "devloft"


def build(dist_dir: Path = ROOT / "dist") -> Path:
    manifest = tomllib.loads((SRC / "blender_manifest.toml").read_text())
    dist_dir = Path(dist_dir)
    dist_dir.mkdir(parents=True, exist_ok=True)
    out = dist_dir / f"{manifest['id']}-{manifest['version']}.zip"
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        for p in sorted(SRC.rglob("*")):
            if p.is_dir() or "__pycache__" in p.parts or p.suffix == ".pyc":
                continue
            z.write(p, p.relative_to(SRC).as_posix())
    return out


if __name__ == "__main__":
    print(build())
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_make_zip.py -q`
Expected: 1 passed.

- [ ] **Step 5: Write the README**

`README.md`:
```markdown
# DevLoft for Blender

Developable loft between two rail curves, for papercraft. Select two curves
(or two edge chains in Edit Mode), run **Developable Loft (two rails)**, and
get a strip mesh whose faces are planar where the rails allow it. Each face
carries a `twist` attribute and a `twist_color` (green = developable, red =
not) so you can see where a panel must be split.

## Install

    uv run python make_zip.py

Then in Blender 4.2+: Preferences > Get Extensions > drop-down > Install from
Disk, choose `dist/devloft-0.1.0.zip`. The operator appears in the 3D
Viewport sidebar under the **DevLoft** tab and in the Add menu.

To see the twist colors: Solid shading > Color > Attribute.

## Parameters

- Samples: points per rail after resampling.
- Window: how far a ruling may lean, in samples.
- Twist Tolerance: degrees; rulings above it are flagged and colored.
- Tie Breaker: none, shortest ruling, or plane direction, with a weight.
- Planarize: nudge vertices so near-planar quads become exactly planar.
- Planar Tolerance: quads still above it are split into two triangles.
- Export JSON: optional path; drop the file on `tools/viewer/index.html`.

## Develop

    uv sync
    uv run pytest -q
    uv run python tools/viewer/export_cases.py
    open tools/viewer/index.html

Design: `docs/superpowers/specs/2026-09-03-devloft-poc-design.md`.
```

- [ ] **Step 6: Build the zip and run everything**

Run: `uv run python make_zip.py && uv run pytest -q`
Expected: prints `dist/devloft-0.1.0.zip`; all tests pass.

- [ ] **Step 7: Commit**

```bash
git add make_zip.py README.md tests/test_make_zip.py
git commit -m "Add extension zip builder and README

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_012kESBz7SJpRjCw2UpaQv9E"
```
