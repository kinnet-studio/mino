# Mino Diagnosis and Remedies Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** When a loft is not developable, show ranked numeric suggestions in the Mino panel for the selected result and provide buttons that re-loft with a larger window, subdivide into strakes, or cut a dart, plus consistent crease directions.

**Architecture:** Result objects carry their inputs as JSON custom properties, so remedies act on "this result" without the source curves. Three new pure-numpy core modules (`strakes`, `dart`, `diagnose`) plus a `consistent` option in `split_quads`; a `remedies.py` Blender module with three operators and a pure `diagnosis_rows` helper the panel draws from.

**Tech Stack:** Python 3.11, numpy, pytest, bpy 4.5 wheel (headless tests) via `uv run --no-sync pytest -q`.

**Spec:** `docs/superpowers/specs/2026-09-03-mino-diagnosis-and-remedies-design.md`

Deviations from the spec decided while planning: stored-input helpers live in a new `mino/blender/state.py` (the spec listed them under output/operator); `subdivide_sections` is added beside `subdivide` so operators can store per-strip inputs; `dart_proposal` takes no `planar_tolerance` (the wedge does not depend on it); the twisted test case is not solved by 8 strakes at the default 5° tolerance (measured 5.2°), so the diagnosis tests assert the `unsolved` path there and the `subdivide` path at 8°.

## Global Constraints

- `mino/core/**` must never import `bpy` or `mathutils`. numpy and stdlib only.
- Vertex layout in a StripResult: rail A is indices `0..N-1`, rail B is `N..2N-1`. Strip faces are ordered along the strip; consecutive faces share exactly the two vertices of a ruling.
- Stored custom properties on every result object, all strings: `mino_rails` (JSON `{"a","ta","b","tb"}`, tangents may be null), `mino_params` (JSON of `LoftParams`), `mino_report` (the format_report line), `mino_diagnosis` (JSON of Diagnosis, or `""` when diagnosis is off).
- Suggestion kinds: `ok`, `window`, `subdivide`, `unsolved`, `dart`, `creases`. Ranks: window 1, subdivide/unsolved 2, dart 3, creases 4, ok 0.
- `LoftParams` gains `consistent_creases: bool = True`.
- A ruled surface has non-positive Gaussian curvature, so on Mino's ruling-based surface the wedge is a gusset (negative deficit) or zero. Panel text uses "gusset" when `wedge_deg < 0` and "dart" when `>= 0`.
- Remedy operators run in Object Mode only, read the active object's stored inputs, and create new objects; they never modify the source object.
- Commit after every task with the two trailer lines:
  `Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>` and `Claude-Session: https://claude.ai/code/session_012kESBz7SJpRjCw2UpaQv9E`.
- The bpy tests need `bpy.ops.wm.read_factory_settings(use_empty=True)` and `mino.register()` per test (see the `fresh_scene` fixture in `tests/test_blender.py`).

Existing code you build on (read these files first when a task touches them): `mino/core/__init__.py` (`loft`), `mino/core/mesh.py` (`split_quads`, `best_diagonal`, `_dihedral`), `mino/core/rails.py` (`prepare_rails`), `mino/core/twist.py` (`twist_matrix`), `mino/core/align.py` (`align`, `tie_matrix`), `mino/core/report.py`, `mino/blender/operator.py`, `mino/blender/output.py`, `mino/blender/inputs.py`, `mino/blender/panel.py`, `mino/blender/__init__.py`, `tests/cases.py`, `tests/test_blender.py`.

---

### Task 1: Consistent creases

**Files:**
- Modify: `mino/core/mesh.py` (`best_diagonal`, `split_quads`), `mino/core/types.py` (`LoftParams`), `mino/core/__init__.py` (`loft`), `mino/blender/operator.py` (new property, draw)
- Test: `tests/test_mesh.py`, `tests/test_types.py`, `tests/test_loft.py`

**Interfaces:**
- Consumes: `_dihedral(verts, tris) -> float` (inf for a degenerate triangle), `best_diagonal(verts, face)`.
- Produces: `split_quads(verts, faces, planarity, tolerance, consistent=False) -> (faces, src, split)`; `LoftParams.consistent_creases`; `crease_orientation(first_tri, n_a) -> str` is NOT provided, tests classify by counting A-vertices.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_mesh.py`:
```python
def _split_pairs(out, split):
    """Yield (first_tri, second_tri) for each split quad in strip order."""
    k = 0
    while k < len(out):
        if split[k]:
            yield out[k], out[k + 1]
            k += 2
        else:
            k += 1


def _orientation(first_tri, n_a):
    # option A's first triangle (v0, v2, v3) holds one A vertex; option B's (v0, v1, v3) holds two
    return "A" if sum(v < n_a for v in first_tri) == 1 else "B"


def test_split_quads_consistent_uses_one_diagonal_per_run():
    n_a = 6
    v = np.array([[i, 0.0, 0.0] for i in range(n_a)] + [[i, 1.0, 0.0] for i in range(n_a)], float)
    # bend alternate vertices so per-quad choices would disagree
    v[7, 2] = 0.25
    v[3, 2] = -0.25
    v[10, 2] = 0.25
    faces = [(i, i + 1, n_a + i + 1, n_a + i) for i in range(n_a - 1)]
    pl = face_planarity(v, faces)
    out, src, split = split_quads(v, faces, pl, tolerance=0.01, consistent=True)
    assert split.sum() >= 4
    orientations = [_orientation(a, n_a) for a, b in _split_pairs(out, split)]
    assert len(set(orientations)) == 1
    # adjacency invariant survives
    for f, g in zip(out, out[1:]):
        assert len(set(f) & set(g)) == 2


def test_split_quads_consistent_picks_lower_total_dihedral():
    from mino.core.mesh import _dihedral
    n_a = 4
    v = np.array([[i, 0.0, 0.0] for i in range(n_a)] + [[i, 1.0, 0.0] for i in range(n_a)], float)
    v[5, 2] = 0.3  # bends quads 0 and 1 only: one run of two split quads
    faces = [(i, i + 1, n_a + i + 1, n_a + i) for i in range(n_a - 1)]
    pl = face_planarity(v, faces)
    assert [k for k in range(len(faces)) if pl[k] > 0.01] == [0, 1]
    out, src, split = split_quads(v, faces, pl, tolerance=0.01, consistent=True)
    chosen = sum(_dihedral(v, pair) for pair in _split_pairs(out, split))
    # the other orientation over the same run costs at least as much
    def other(face):
        v0, v1, v2, v3 = face
        a = [(v0, v2, v3), (v0, v1, v2)]
        b = [(v0, v1, v3), (v1, v2, v3)]
        return a, b
    runs = [faces[k] for k in range(len(faces)) if pl[k] > 0.01]
    cost_a = sum(_dihedral(v, other(f)[0]) for f in runs)
    cost_b = sum(_dihedral(v, other(f)[1]) for f in runs)
    assert chosen <= max(cost_a, cost_b) + 1e-9
    assert np.isclose(chosen, min(cost_a, cost_b))


def test_split_quads_default_is_per_quad():
    v = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0.2], [0, 1, 0]], float)
    faces = [(0, 1, 2, 3)]
    a, _, _ = split_quads(v, faces, face_planarity(v, faces), tolerance=0.01)
    b, _, _ = split_quads(v, faces, face_planarity(v, faces), tolerance=0.01, consistent=False)
    assert a == b
```

Append to `tests/test_types.py`:
```python
def test_consistent_creases_default():
    assert LoftParams().consistent_creases is True
```

Append to `tests/test_loft.py`:
```python
def test_consistent_creases_on_twisted_strip():
    _, params, res = _run("twisted", samples=24, planarize=False, consistent_creases=True)
    n = params.samples
    k = 0
    orientations = []
    run = []
    while k < len(res.faces):
        if res.face_split[k]:
            first = res.faces[k]
            run.append("A" if sum(v < n for v in first) == 1 else "B")
            k += 2
        else:
            if run:
                orientations.append(run)
                run = []
            k += 1
    if run:
        orientations.append(run)
    assert orientations
    for r in orientations:
        assert len(set(r)) == 1
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run --no-sync pytest tests/test_mesh.py tests/test_types.py tests/test_loft.py -q`
Expected: FAIL with `TypeError: split_quads() got an unexpected keyword argument 'consistent'`, `AttributeError` for `consistent_creases`, and `TypeError` from `LoftParams(**...)`.

- [ ] **Step 3: Implement**

In `mino/core/mesh.py` replace `best_diagonal` and `split_quads` with:
```python
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
```

In `mino/core/types.py` add to `LoftParams` after `planarize_max_nudge`:
```python
    consistent_creases: bool = True
```

In `mino/core/__init__.py` change the split call to:
```python
    out_faces, src, split = split_quads(verts, faces, planarity, params.planar_tolerance,
                                        consistent=params.consistent_creases)
```

In `mino/blender/operator.py` add after `planarize_max_nudge`:
```python
    consistent_creases: BoolProperty(name="Consistent Creases", default=True,
                                     description="Use one crease direction per run of split quads")
```
pass `consistent_creases=self.consistent_creases` into `LoftParams(...)`, and add `col.prop(self, "consistent_creases")` after `planarize_max_nudge` in `draw`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run --no-sync pytest -q`
Expected: all passed (62 existing plus 5 new).

- [ ] **Step 5: Commit**

```bash
git add mino/core/mesh.py mino/core/types.py mino/core/__init__.py mino/blender/operator.py tests/test_mesh.py tests/test_types.py tests/test_loft.py
git commit -m "Add consistent crease direction per run of split quads

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_012kESBz7SJpRjCw2UpaQv9E"
```

---

### Task 2: Strakes: mid-rails, subdivide, strake search, chaining

**Files:**
- Create: `mino/core/strakes.py`, `tests/test_strakes.py`

**Interfaces:**
- Consumes: `loft(points_a, points_b, params, tangents_a, tangents_b) -> StripResult`, `prepare_rails(points_a, points_b, samples, tangents_a, tangents_b) -> (Rail, Rail)`, `LoftError`.
- Produces:
  - `mid_rails(rail_a: Rail, rail_b: Rail, rulings, count: int) -> list[np.ndarray]` polylines at fractions `r/(count+1)`.
  - `chain_loft(sections: list[tuple[points, tangents | None]], params) -> list[StripResult]`.
  - `subdivide_sections(points_a, points_b, tangents_a, tangents_b, params, rulings, strakes: int) -> list[tuple[points, tangents | None]]` the A, mid-rail and B sections.
  - `subdivide(points_a, points_b, tangents_a, tangents_b, params, rulings, strakes: int) -> list[StripResult]` = `chain_loft(subdivide_sections(...), params)`.
  - `find_strake_count(points_a, points_b, tangents_a, tangents_b, params, rulings, max_strakes=8) -> tuple[int | None, float, list[StripResult]]`.

- [ ] **Step 1: Write the failing tests**

`tests/test_strakes.py`:
```python
import numpy as np
import pytest

from mino.core import LoftError, LoftParams, loft
from mino.core.rails import prepare_rails
from mino.core.strakes import chain_loft, find_strake_count, mid_rails, subdivide, subdivide_sections
from tests.cases import CASES


def _twisted(**overrides):
    case = CASES["twisted"]()
    params = LoftParams(**{**case["params"], **overrides})
    res = loft(case["points_a"], case["points_b"], params, case["tangents_a"], case["tangents_b"])
    return case, params, res


def _line_point_distance(p, a, b):
    d = b - a
    return np.linalg.norm(np.cross(p - a, d)) / np.linalg.norm(d)


def test_mid_rails_lie_on_rulings():
    case, params, res = _twisted()
    ra, rb = prepare_rails(case["points_a"], case["points_b"], params.samples,
                           case["tangents_a"], case["tangents_b"])
    mids = mid_rails(ra, rb, res.rulings, count=3)
    assert len(mids) == 3
    for m in mids:
        assert m.shape == (len(res.rulings), 3)
        for (i, j), p in zip(res.rulings, m):
            assert _line_point_distance(p, ra.points[i], rb.points[j]) < 1e-9
    # fractions increase from A toward B
    i0, j0 = res.rulings[10]
    d = [np.linalg.norm(m[10] - ra.points[i0]) for m in mids]
    assert d[0] < d[1] < d[2]


def test_subdivide_halves_twist_roughly():
    case, params, res = _twisted()
    strips = subdivide(case["points_a"], case["points_b"], case["tangents_a"], case["tangents_b"],
                       params, res.rulings, strakes=2)
    assert len(strips) == 2
    worst = max(s.report.max_twist for s in strips)
    assert worst < 0.7 * res.report.max_twist


def test_find_strake_count_reaches_tolerance():
    # measured: 8 strakes leave 5.2 deg on the twisted case, so use an 8 deg tolerance (6 strakes pass at 7.5)
    case, params, res = _twisted(twist_tolerance=8.0)
    count, worst, strips = find_strake_count(case["points_a"], case["points_b"], case["tangents_a"],
                                             case["tangents_b"], params, res.rulings, max_strakes=8)
    assert count == 6
    assert len(strips) == count
    assert worst <= params.twist_tolerance
    assert all(s.report.failing_ruling_count == 0 for s in strips)


def test_find_strake_count_reports_failure_when_capped():
    case, params, res = _twisted()
    count, worst, strips = find_strake_count(case["points_a"], case["points_b"], case["tangents_a"],
                                             case["tangents_b"], params, res.rulings, max_strakes=8)
    assert count is None
    assert 5.0 < worst < 6.0
    assert len(strips) == 8


def test_chain_loft_shares_rails():
    case = CASES["cylinder"]()
    params = LoftParams(planarize=False)
    n = params.samples
    mid = case["points_a"].copy()
    mid[:, 2] = 0.5
    strips = chain_loft([(case["points_a"], case["tangents_a"]), (mid, None), (case["points_b"], case["tangents_b"])], params)
    assert len(strips) == 2
    assert np.allclose(strips[0].verts[n:], strips[1].verts[:n])
    assert all(s.failing_ranges == [] for s in strips)


def test_subdivide_sections_shape():
    case, params, res = _twisted()
    secs = subdivide_sections(case["points_a"], case["points_b"], case["tangents_a"], case["tangents_b"],
                              params, res.rulings, strakes=4)
    assert len(secs) == 5
    assert secs[0][1] is not None and secs[-1][1] is not None
    assert all(t is None for _, t in secs[1:-1])
    assert all(p.shape[1] == 3 for p, _ in secs)


def test_chain_loft_and_subdivide_reject_bad_input():
    case, params, res = _twisted()
    with pytest.raises(LoftError):
        chain_loft([(case["points_a"], None)], params)
    with pytest.raises(LoftError):
        subdivide(case["points_a"], case["points_b"], None, None, params, res.rulings, strakes=1)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run --no-sync pytest tests/test_strakes.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'mino.core.strakes'`.

- [ ] **Step 3: Implement**

`mino/core/strakes.py`:
```python
"""Subdivide a loft into narrower strips (strakes) and chain lofts through sections."""
from __future__ import annotations

import numpy as np

from . import loft
from .errors import LoftError
from .rails import prepare_rails
from .types import LoftParams, Rail, StripResult


def mid_rails(rail_a: Rail, rail_b: Rail, rulings, count: int) -> list[np.ndarray]:
    """Polylines at fractions r/(count+1) along the chosen rulings, from A toward B."""
    ends_a = np.array([rail_a.points[i] for i, _ in rulings], dtype=float)
    ends_b = np.array([rail_b.points[j] for _, j in rulings], dtype=float)
    return [ends_a + (r / (count + 1)) * (ends_b - ends_a) for r in range(1, count + 1)]


def chain_loft(sections, params: LoftParams) -> list[StripResult]:
    """Loft each consecutive pair of (points, tangents) sections."""
    if len(sections) < 2:
        raise LoftError("need at least two sections to loft")
    strips = []
    for (pa, ta), (pb, tb) in zip(sections, sections[1:]):
        strips.append(loft(pa, pb, params, ta, tb))
    return strips


def subdivide_sections(points_a, points_b, tangents_a, tangents_b, params: LoftParams, rulings, strakes: int):
    """A, the strakes-1 mid-rails on the given rulings, and B, as (points, tangents) sections."""
    if strakes < 2:
        raise LoftError("strakes must be at least 2")
    rail_a, rail_b = prepare_rails(points_a, points_b, params.samples, tangents_a, tangents_b)
    mids = mid_rails(rail_a, rail_b, rulings, strakes - 1)
    return ([(np.asarray(points_a, dtype=float), tangents_a)]
            + [(m, None) for m in mids]
            + [(np.asarray(points_b, dtype=float), tangents_b)])


def subdivide(points_a, points_b, tangents_a, tangents_b, params: LoftParams, rulings, strakes: int):
    """Loft `strakes` narrower strips between A and B through mid-rails on the given rulings."""
    return chain_loft(subdivide_sections(points_a, points_b, tangents_a, tangents_b, params, rulings, strakes), params)


def find_strake_count(points_a, points_b, tangents_a, tangents_b, params: LoftParams, rulings,
                      max_strakes: int = 8):
    """Smallest strake count whose strips all pass the twist tolerance.

    Returns (count, worst_twist, strips); count is None when even max_strakes fails,
    in which case worst_twist and strips describe the max_strakes attempt.
    """
    worst, strips = float("inf"), []
    for strakes in range(2, max_strakes + 1):
        strips = subdivide(points_a, points_b, tangents_a, tangents_b, params, rulings, strakes)
        worst = max(s.report.max_twist for s in strips)
        if worst <= params.twist_tolerance:
            return strakes, worst, strips
    return None, worst, strips
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run --no-sync pytest tests/test_strakes.py -q`
Expected: 7 passed. If `test_subdivide_halves_twist_roughly` fails on the 0.7 factor, print both twists in the report and stop; do not change the factor.

- [ ] **Step 5: Commit**

```bash
git add mino/core/strakes.py tests/test_strakes.py
git commit -m "Add strake subdivision and section chaining

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_012kESBz7SJpRjCw2UpaQv9E"
```

---
### Task 3: Dart: refined grid, Gauss–Bonnet wedge, seam mesh

**Files:**
- Create: `mino/core/dart.py`, `tests/test_dart.py`

**Interfaces:**
- Consumes: `Rail`, `quad_planarity(verts, face)` from `mino.core.mesh`.
- Produces:
  - `refined_grid(rail_a, rail_b, rulings, mid_rails: int) -> np.ndarray` shape `(mid_rails + 2, len(rulings), 3)`, row 0 on A, last row on B.
  - `angle_deficit_total(grid, cols_range=None) -> float` radians, summed over interior vertices (rows 1..rows-2, columns strictly inside `cols_range`).
  - `DartProposal(range, ruling, wedge_deg, kind)`; `dart_proposal(rail_a, rail_b, rulings, ruling_twist, failing_range, mid_rails=3) -> DartProposal`.
  - `DartMesh(verts, faces, face_twist, seam_edges)`; `dart_mesh(rail_a, rail_b, rulings, ruling_twist, proposal, mid_rails=3, planar_tolerance=0.01, dart_from="B") -> DartMesh`.

Calibration facts (measured on the current core, keep the assertions as written): on the twisted case the failing range's deficit is about −18°, i.e. a gusset; the sphere patch below measures 3.7 percent low; a cylinder grid measures 1e-14.

- [ ] **Step 1: Write the failing tests**

`tests/test_dart.py`:
```python
import numpy as np
import pytest

from mino.core import LoftParams, loft
from mino.core.dart import angle_deficit_total, dart_mesh, dart_proposal, refined_grid
from mino.core.rails import prepare_rails
from tests.cases import CASES


def _twisted():
    case = CASES["twisted"]()
    params = LoftParams()
    res = loft(case["points_a"], case["points_b"], params, case["tangents_a"], case["tangents_b"])
    ra, rb = prepare_rails(case["points_a"], case["points_b"], params.samples,
                           case["tangents_a"], case["tangents_b"])
    return params, res, ra, rb


def test_refined_grid_shape_and_rows():
    params, res, ra, rb = _twisted()
    grid = refined_grid(ra, rb, res.rulings, mid_rails=3)
    assert grid.shape == (5, len(res.rulings), 3)
    for k, (i, j) in enumerate(res.rulings):
        assert np.allclose(grid[0, k], ra.points[i])
        assert np.allclose(grid[4, k], rb.points[j])
        assert np.allclose(grid[2, k], 0.5 * (ra.points[i] + rb.points[j]))


def test_angle_deficit_zero_on_plane_and_cylinder():
    xs = np.linspace(0, 1, 12)
    plane = np.array([[[x, y, 0.0] for x in xs] for y in xs])
    assert abs(angle_deficit_total(plane)) < 1e-9
    cyl = np.array([[[np.cos(th), np.sin(th), z] for th in np.linspace(0, np.pi, 30)]
                    for z in np.linspace(0, 1, 10)])
    assert abs(angle_deficit_total(cyl)) < 1e-9


def test_angle_deficit_matches_sphere_solid_angle():
    rows, cols = 40, 80
    phi = np.linspace(0, np.pi / 4, rows)
    lam = np.linspace(0, np.pi / 2, cols)
    sphere = np.array([[[np.cos(f) * np.cos(l), np.cos(f) * np.sin(l), np.sin(f)] for l in lam] for f in phi])
    deficit = angle_deficit_total(sphere)
    solid_angle = (np.pi / 2) * np.sin(np.pi / 4)
    assert deficit > 0
    # interior vertices miss a half-cell band along the boundary: a few percent low
    assert abs(deficit - solid_angle) / solid_angle < 0.06


def test_angle_deficit_negative_on_saddle():
    xs = np.linspace(-1, 1, 30)
    saddle = np.array([[[x, y, x * x - y * y] for x in xs] for y in xs])
    assert angle_deficit_total(saddle) < -1.0


def test_angle_deficit_respects_column_range():
    xs = np.linspace(-1, 1, 30)
    saddle = np.array([[[x, y, x * x - y * y] for x in xs] for y in xs])
    full = angle_deficit_total(saddle)
    part = angle_deficit_total(saddle, cols_range=(5, 15))
    assert part < 0 and abs(part) < abs(full)


def test_dart_proposal_on_twisted_is_a_gusset_at_max_twist():
    params, res, ra, rb = _twisted()
    rng = res.failing_ranges[-1]
    prop = dart_proposal(ra, rb, res.rulings, res.ruling_twist, rng, mid_rails=3)
    k1, k2 = rng
    assert prop.range == rng
    assert prop.ruling == k1 + int(np.argmax(res.ruling_twist[k1:k2 + 1]))
    assert prop.kind == "gusset" and prop.wedge_deg < -5.0


def test_dart_mesh_has_seam_path_and_clean_topology():
    params, res, ra, rb = _twisted()
    prop = dart_proposal(ra, rb, res.rulings, res.ruling_twist, res.failing_ranges[-1], mid_rails=3)
    dm = dart_mesh(ra, rb, res.rulings, res.ruling_twist, prop, mid_rails=3,
                   planar_tolerance=params.planar_tolerance, dart_from="B")
    assert len(dm.faces) == len(dm.face_twist)
    assert all(len(f) in (3, 4) for f in dm.faces)
    assert all(len(set(f)) == len(f) for f in dm.faces)
    assert max(v for f in dm.faces for v in f) < len(dm.verts)
    # no duplicate vertices
    assert len({tuple(np.round(v, 9)) for v in dm.verts}) == len(dm.verts)
    # seam: rows - 2 edges forming one path from the B row to row 1
    assert len(dm.seam_edges) == 3
    for (a, b), (c, d) in zip(dm.seam_edges, dm.seam_edges[1:]):
        assert b == c
    assert all(a != b for a, b in dm.seam_edges)
    # seam starts on rail B (z near cos(phi) side) when dart_from == "B"
    start = dm.verts[dm.seam_edges[0][0]]
    i, j = res.rulings[prop.ruling]
    assert np.allclose(start, rb.points[j])


def test_dart_mesh_from_a_starts_on_rail_a():
    params, res, ra, rb = _twisted()
    prop = dart_proposal(ra, rb, res.rulings, res.ruling_twist, res.failing_ranges[-1])
    dm = dart_mesh(ra, rb, res.rulings, res.ruling_twist, prop, dart_from="A")
    i, j = res.rulings[prop.ruling]
    assert np.allclose(dm.verts[dm.seam_edges[0][0]], ra.points[i])
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run --no-sync pytest tests/test_dart.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'mino.core.dart'`.

- [ ] **Step 3: Implement**

`mino/core/dart.py`:
```python
"""Dart (gusset) proposal and refined mesh for a non-developable region.

The refined grid lies on the ruled surface spanned by the chosen rulings. A
ruled surface has non-positive Gaussian curvature, so the wedge is normally
negative: a gusset to insert, not a dart to remove.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .mesh import quad_planarity
from .types import Rail


@dataclass
class DartProposal:
    range: tuple            # failing ruling range (k1, k2)
    ruling: int             # cut ruling: max twist within the range
    wedge_deg: float        # positive = dart (remove wedge), negative = gusset (insert)
    kind: str               # "dart" | "gusset"


@dataclass
class DartMesh:
    verts: np.ndarray
    faces: list
    face_twist: np.ndarray
    seam_edges: list        # (v_from, v_to) pairs, a path from the cut's open end to its tip


def refined_grid(rail_a: Rail, rail_b: Rail, rulings, mid_rails: int) -> np.ndarray:
    rows = mid_rails + 2
    A = np.array([rail_a.points[i] for i, _ in rulings], dtype=float)
    B = np.array([rail_b.points[j] for _, j in rulings], dtype=float)
    t = np.linspace(0.0, 1.0, rows)[:, None, None]
    return A[None] + t * (B - A)[None]


def _corner_angle(p, q, r) -> float:
    u, v = q - p, r - p
    lu, lv = np.linalg.norm(u), np.linalg.norm(v)
    if lu < 1e-12 or lv < 1e-12:
        return 0.0
    return float(np.arccos(np.clip(np.dot(u, v) / (lu * lv), -1.0, 1.0)))


def _tri_area(P, tri) -> float:
    a, b, c = (P[v] for v in tri)
    return 0.5 * float(np.linalg.norm(np.cross(b - a, c - a)))


def _quad_triangles(P, quad):
    """Two triangles along the shorter diagonal, degenerate ones dropped."""
    v0, v1, v2, v3 = quad
    if np.linalg.norm(P[v2] - P[v0]) <= np.linalg.norm(P[v3] - P[v1]):
        cand = [(v0, v1, v2), (v0, v2, v3)]
    else:
        cand = [(v0, v1, v3), (v1, v2, v3)]
    return [t for t in cand if _tri_area(P, t) > 1e-14]


def _grid_quads(rows, cols):
    idx = lambda r, c: r * cols + c
    return [(idx(r, c), idx(r, c + 1), idx(r + 1, c + 1), idx(r + 1, c))
            for r in range(rows - 1) for c in range(cols - 1)]


def angle_deficit_total(grid: np.ndarray, cols_range=None) -> float:
    """Sum of 2π minus corner-angle sums over interior grid vertices (radians)."""
    rows, cols, _ = grid.shape
    c0, c1 = (0, cols - 1) if cols_range is None else cols_range
    P = grid.reshape(-1, 3)
    sums = np.zeros(rows * cols)
    for quad in _grid_quads(rows, cols):
        for a, b, c in _quad_triangles(P, quad):
            sums[a] += _corner_angle(P[a], P[b], P[c])
            sums[b] += _corner_angle(P[b], P[c], P[a])
            sums[c] += _corner_angle(P[c], P[a], P[b])
    total = 0.0
    for r in range(1, rows - 1):
        for c in range(c0 + 1, c1):
            total += 2.0 * np.pi - sums[r * cols + c]
    return float(total)


def dart_proposal(rail_a, rail_b, rulings, ruling_twist, failing_range, mid_rails: int = 3) -> DartProposal:
    k1, k2 = failing_range
    tw = np.asarray(ruling_twist, dtype=float)[k1:k2 + 1]
    tw = np.where(np.isfinite(tw), tw, -1.0)
    ruling = k1 + int(np.argmax(tw))
    grid = refined_grid(rail_a, rail_b, rulings, mid_rails)
    cols = grid.shape[1]
    theta = angle_deficit_total(grid, (max(k1 - 1, 0), min(k2 + 1, cols - 1)))
    wedge = float(np.degrees(theta))
    return DartProposal(range=(k1, k2), ruling=ruling, wedge_deg=wedge,
                        kind="dart" if wedge >= 0.0 else "gusset")


def _merge_duplicates(P, faces, face_twist, edges):
    """Merge coincident vertices; drop faces that collapse and zero-length edges."""
    key_to_new, old_to_new, verts = {}, [], []
    for p in P:
        key = tuple(np.round(p, 9))
        if key not in key_to_new:
            key_to_new[key] = len(verts)
            verts.append(p)
        old_to_new.append(key_to_new[key])
    out_faces, out_twist = [], []
    for f, tw in zip(faces, face_twist):
        mapped = []
        for v in f:
            nv = old_to_new[v]
            if not mapped or mapped[-1] != nv:
                mapped.append(nv)
        if len(mapped) > 1 and mapped[0] == mapped[-1]:
            mapped.pop()
        if len(set(mapped)) >= 3:
            out_faces.append(tuple(mapped))
            out_twist.append(tw)
    out_edges = [(old_to_new[a], old_to_new[b]) for a, b in edges if old_to_new[a] != old_to_new[b]]
    return np.array(verts, dtype=float), out_faces, np.array(out_twist, dtype=float), out_edges


def dart_mesh(rail_a, rail_b, rulings, ruling_twist, proposal: DartProposal, mid_rails: int = 3,
              planar_tolerance: float = 0.01, dart_from: str = "B") -> DartMesh:
    grid = refined_grid(rail_a, rail_b, rulings, mid_rails)
    rows, cols, _ = grid.shape
    P = grid.reshape(-1, 3)
    idx = lambda r, c: r * cols + c
    tw = np.asarray(ruling_twist, dtype=float)
    faces, face_twist = [], []
    for r in range(rows - 1):
        for c in range(cols - 1):
            quad = (idx(r, c), idx(r, c + 1), idx(r + 1, c + 1), idx(r + 1, c))
            t = float(max(tw[c], tw[c + 1]))
            corners = [P[v] for v in quad]
            distinct = len({tuple(np.round(p, 9)) for p in corners})
            if distinct < 3:
                continue
            if distinct == 4 and quad_planarity(P, quad) <= planar_tolerance:
                faces.append(quad)
                face_twist.append(t)
            else:
                for tri in _quad_triangles(P, quad):
                    faces.append(tri)
                    face_twist.append(t)
    k = proposal.ruling
    if dart_from == "B":
        seam = [(idx(r, k), idx(r - 1, k)) for r in range(rows - 1, 1, -1)]
    else:
        seam = [(idx(r, k), idx(r + 1, k)) for r in range(0, rows - 2)]
    verts, faces, face_twist, seam = _merge_duplicates(P, faces, face_twist, seam)
    return DartMesh(verts=verts, faces=faces, face_twist=face_twist, seam_edges=seam)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run --no-sync pytest tests/test_dart.py -q`
Expected: 8 passed.

- [ ] **Step 5: Commit**

```bash
git add mino/core/dart.py tests/test_dart.py
git commit -m "Add dart proposal with Gauss-Bonnet wedge and seam mesh

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_012kESBz7SJpRjCw2UpaQv9E"
```

---

### Task 4: Diagnosis

**Files:**
- Create: `mino/core/diagnose.py`, `tests/test_diagnose.py`
- Modify: `tests/cases.py` (add the `ellipse` case)

**Interfaces:**
- Consumes: `prepare_rails`, `twist_matrix(rail_a, rail_b, window)`, `align(cost)`, `tie_matrix(rail_a, rail_b, mode, plane_normal)`, `find_strake_count(...)`, `dart_proposal(...)`, `DartProposal`.
- Produces:
  - `Suggestion(kind, text, params, rank)`, `Diagnosis(failing_ranges, max_twist, window_fix, strakes_needed, strakes_worst_twist, darts, crease_runs, suggestions)`.
  - `probe_window(rail_a, rail_b, params, window) -> float` max twist along the DP path at that window.
  - `crease_runs(face_split) -> int`.
  - `diagnose(points_a, points_b, tangents_a, tangents_b, params, result, max_strakes=8) -> Diagnosis`.
  - `format_diagnosis(d) -> list[str]`; `diagnosis_to_dict(d) -> dict`; `diagnosis_from_dict(data) -> Diagnosis`.

Calibration facts: the ellipse case below needs a lean of 5 samples; with window 2 it has 32 failing rulings, window 4 leaves 4, window 8 leaves none. The twisted case at the default 5° tolerance is not solved by 8 strakes (worst 5.2°); at 8° tolerance 6 strakes pass.

- [ ] **Step 1: Add the ellipse case**

Append to `tests/cases.py` before `CASES`, and add `"ellipse": ellipse,` to `CASES`:
```python
def ellipse(n=200, a=1.5):
    """Circle to a stretched ellipse: developable only if rulings lean by up to ~5 samples."""
    th = np.linspace(0.0, np.pi, n)
    pa = np.column_stack([np.cos(th), np.sin(th), np.zeros(n)])
    ta = np.column_stack([-np.sin(th), np.cos(th), np.zeros(n)])
    pb = np.column_stack([a * np.cos(th), np.sin(th), np.ones(n)])
    tb = np.column_stack([-a * np.sin(th), np.cos(th), np.zeros(n)])
    return dict(points_a=pa, points_b=pb, tangents_a=ta, tangents_b=tb, params=dict(window=2))
```
Then run `uv run --no-sync pytest tests/test_export_cases.py -q`; it fails because the viewer test expects four case names. Update that test's expected list to `["cylinder", "cone", "offset_cylinder", "twisted", "ellipse"]` and the set of json names to include `"ellipse.json"`, then run `uv run --no-sync python tools/viewer/export_cases.py` to regenerate `tools/viewer/data.js` and `tools/viewer/ellipse.json` (commit them with this task).

- [ ] **Step 2: Write the failing tests**

`tests/test_diagnose.py`:
```python
import json

import numpy as np

from mino.core import LoftParams, loft
from mino.core.diagnose import (crease_runs, diagnose, diagnosis_from_dict, diagnosis_to_dict,
                                format_diagnosis, probe_window)
from mino.core.rails import prepare_rails
from tests.cases import CASES


def _run(name, **overrides):
    case = CASES[name]()
    params = LoftParams(**{**case["params"], **overrides})
    res = loft(case["points_a"], case["points_b"], params, case["tangents_a"], case["tangents_b"])
    return case, params, res


def _diag(name, **overrides):
    case, params, res = _run(name, **overrides)
    return params, res, diagnose(case["points_a"], case["points_b"], case["tangents_a"], case["tangents_b"], params, res)


def test_ok_when_developable():
    params, res, d = _diag("cylinder")
    assert d.failing_ranges == [] and d.window_fix is None and d.strakes_needed is None
    assert [s.kind for s in d.suggestions] == ["ok"]
    assert format_diagnosis(d) == ["All rulings within tolerance"]


def test_crease_runs_counts_pairs():
    assert crease_runs(np.array([False, True, True, True, True, False, True, True])) == 2
    assert crease_runs(np.zeros(5, dtype=bool)) == 0


def test_probe_window_on_ellipse():
    case, params, res = _run("ellipse")
    ra, rb = prepare_rails(case["points_a"], case["points_b"], params.samples, case["tangents_a"], case["tangents_b"])
    assert probe_window(ra, rb, params, 2) > params.twist_tolerance
    assert probe_window(ra, rb, params, 8) <= params.twist_tolerance


def test_window_fix_on_ellipse():
    params, res, d = _diag("ellipse")
    assert res.failing_ranges
    assert d.window_fix == 8
    assert d.suggestions[0].kind == "window" and d.suggestions[0].params == {"window": 8}
    assert "Window 8" in d.suggestions[0].text
    # the fix really works
    case = CASES["ellipse"]()
    fixed = loft(case["points_a"], case["points_b"], LoftParams(window=8), case["tangents_a"], case["tangents_b"])
    assert fixed.failing_ranges == []


def test_twisted_is_unsolved_at_default_tolerance():
    params, res, d = _diag("twisted")
    assert d.window_fix is None
    assert d.strakes_needed is None
    kinds = [s.kind for s in d.suggestions]
    assert kinds[0] == "unsolved" and kinds.count("dart") == len(res.failing_ranges)
    assert kinds[-1] == "creases"
    assert d.darts[-1].kind == "gusset"
    assert "gusset" in [s for s in d.suggestions if s.kind == "dart"][0].text
    assert [s.rank for s in d.suggestions] == sorted(s.rank for s in d.suggestions)


def test_twisted_subdivides_at_eight_degrees():
    params, res, d = _diag("twisted", twist_tolerance=8.0)
    assert d.strakes_needed == 6
    sub = [s for s in d.suggestions if s.kind == "subdivide"][0]
    assert sub.params == {"strakes": 6} and "6 strakes" in sub.text


def test_diagnosis_round_trips_through_json():
    params, res, d = _diag("twisted")
    data = json.loads(json.dumps(diagnosis_to_dict(d)))
    back = diagnosis_from_dict(data)
    assert back.failing_ranges == d.failing_ranges
    assert [s.kind for s in back.suggestions] == [s.kind for s in d.suggestions]
    assert back.darts[0].ruling == d.darts[0].ruling
    assert np.isclose(back.darts[0].wedge_deg, d.darts[0].wedge_deg)
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `uv run --no-sync pytest tests/test_diagnose.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'mino.core.diagnose'`.

- [ ] **Step 4: Implement**

`mino/core/diagnose.py`:
```python
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
    rail_a, rail_b = prepare_rails(points_a, points_b, n, tangents_a, tangents_b)

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
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run --no-sync pytest tests/test_diagnose.py tests/test_export_cases.py -q`
Expected: 8 passed. Then `uv run --no-sync pytest -q` all green.

- [ ] **Step 6: Commit**

```bash
git add mino/core/diagnose.py tests/test_diagnose.py tests/cases.py tests/test_export_cases.py tools/viewer/data.js tools/viewer/ellipse.json
git commit -m "Add developability diagnosis with ranked suggestions

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_012kESBz7SJpRjCw2UpaQv9E"
```

---
### Task 5: Blender: stored inputs, chaining, diagnosis on loft

**Files:**
- Create: `mino/blender/state.py`
- Modify: `mino/blender/inputs.py` (add `get_sections`, `order_sections`), `mino/blender/output.py` (add `create_result_object`, factor `_write_twist_attributes`), `mino/blender/operator.py` (chaining, new properties, stored inputs)
- Test: `tests/test_blender.py` (append)

**Interfaces:**
- Consumes: `chain_loft(sections, params)`, `diagnose(...)`, `diagnosis_to_dict`, `format_report`, `LoftParams`.
- Produces:
  - `state.params_to_dict(params) -> dict`, `state.params_from_dict(data) -> LoftParams` (unknown keys ignored).
  - `state.store_inputs(obj, points_a, tangents_a, points_b, tangents_b, params, result, diagnosis | None)`.
  - `state.load_inputs(obj) -> (points_a, tangents_a, points_b, tangents_b, params)` raising `LoftError` when `mino_rails` is missing.
  - `state.load_diagnosis(obj) -> Diagnosis | None`.
  - `inputs.get_sections(context, samples) -> list[tuple[np.ndarray, np.ndarray | None]]` (two for Edit Mode, two or more for curves, ordered by greedy nearest centroid from the active curve).
  - `output.create_result_object(context, name, points_a, tangents_a, points_b, tangents_b, params, result, diagnose_flag: bool, strake: int = 0) -> (obj, Diagnosis | None)`.
  - Operator properties `consistent_creases` (Task 1) and `diagnose` (default True).

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_blender.py`:
```python
import json  # noqa: E402  (add to the imports at the top of the file)


def test_loft_stores_inputs_and_diagnosis(fresh_scene):
    a = _make_arc_curve("A", 1.0, 0.0)
    b = _make_arc_curve("B", 1.0, 1.0)
    _select([a, b], a)
    assert bpy.ops.mino.loft(samples=30) == {"FINISHED"}
    obj = bpy.data.objects["Mino"]
    rails = json.loads(obj["mino_rails"])
    assert set(rails) == {"a", "ta", "b", "tb"}
    assert len(rails["a"]) >= 30 * 4 and len(rails["ta"]) == len(rails["a"])
    params = json.loads(obj["mino_params"])
    assert params["samples"] == 30 and params["consistent_creases"] is True
    assert obj["mino_report"].startswith("Mino: 30 rulings")
    diag = json.loads(obj["mino_diagnosis"])
    assert [s["kind"] for s in diag["suggestions"]] == ["ok"]
    assert "strake" in obj.data.attributes
    assert all(d.value == 0 for d in obj.data.attributes["strake"].data)


def test_loft_with_diagnose_off_stores_empty_diagnosis(fresh_scene):
    a = _make_arc_curve("A", 1.0, 0.0)
    b = _make_arc_curve("B", 1.0, 1.0)
    _select([a, b], a)
    assert bpy.ops.mino.loft(samples=20, diagnose=False) == {"FINISHED"}
    assert bpy.data.objects["Mino"]["mino_diagnosis"] == ""


def test_three_curves_chain_into_two_strips(fresh_scene):
    a = _make_arc_curve("A", 1.0, 0.0)
    b = _make_arc_curve("B", 1.0, 1.0)
    c = _make_arc_curve("C", 1.0, 2.0)
    _select([a, c, b], a)  # selection order deliberately scrambled
    assert bpy.ops.mino.loft(samples=20) == {"FINISHED"}
    first, second = bpy.data.objects["Mino"], bpy.data.objects["Mino.001"]
    assert len(first.data.vertices) == 40 and len(second.data.vertices) == 40
    assert {d.value for d in first.data.attributes["strake"].data} == {0}
    assert {d.value for d in second.data.attributes["strake"].data} == {1}
    r1, r2 = json.loads(first["mino_rails"]), json.loads(second["mino_rails"])
    assert np.allclose(np.array(r1["b"]), np.array(r2["a"]))
    # the middle curve (z = 1) is the shared rail
    assert np.allclose(np.array(r1["b"])[:, 2], 1.0)


def test_load_inputs_round_trip(fresh_scene):
    from mino.blender.state import load_inputs, params_from_dict, params_to_dict
    a = _make_arc_curve("A", 1.0, 0.0)
    b = _make_arc_curve("B", 1.0, 1.0)
    _select([a, b], a)
    bpy.ops.mino.loft(samples=24, window=5)
    pa, ta, pb, tb, params = load_inputs(bpy.data.objects["Mino"])
    assert pa.shape[1] == 3 and ta.shape == pa.shape and pb.shape[1] == 3
    assert params.samples == 24 and params.window == 5
    assert params_from_dict({**params_to_dict(params), "bogus": 1}).window == 5
    with pytest.raises(LoftError):
        load_inputs(a)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run --no-sync pytest tests/test_blender.py -q -k "stores or diagnose_off or three_curves or load_inputs"`
Expected: FAIL with `KeyError: 'mino_rails'`, `TypeError` for the unknown `diagnose` property, `LoftError` for three curves, and `ModuleNotFoundError` for `mino.blender.state`.

- [ ] **Step 3: Write state.py**

`mino/blender/state.py`:
```python
"""Stored inputs on Mino result objects, so remedies can act on 'this result'."""
from __future__ import annotations

import json
from dataclasses import asdict, fields

import numpy as np

from ..core import LoftParams, format_report
from ..core.diagnose import Diagnosis, diagnosis_from_dict, diagnosis_to_dict
from ..core.errors import LoftError


def params_to_dict(params: LoftParams) -> dict:
    data = asdict(params)
    data["plane_normal"] = [float(x) for x in data["plane_normal"]]
    return data


def params_from_dict(data: dict) -> LoftParams:
    names = {f.name for f in fields(LoftParams)}
    kwargs = {k: v for k, v in data.items() if k in names}
    if "plane_normal" in kwargs:
        kwargs["plane_normal"] = tuple(float(x) for x in kwargs["plane_normal"])
    return LoftParams(**kwargs)


def _opt(arr):
    return None if arr is None else np.asarray(arr, dtype=float).tolist()


def store_inputs(obj, points_a, tangents_a, points_b, tangents_b, params, result, diagnosis):
    obj["mino_rails"] = json.dumps({"a": _opt(points_a), "ta": _opt(tangents_a),
                                    "b": _opt(points_b), "tb": _opt(tangents_b)})
    obj["mino_params"] = json.dumps(params_to_dict(params))
    obj["mino_report"] = format_report(result.report, result.failing_ranges)
    obj["mino_diagnosis"] = json.dumps(diagnosis_to_dict(diagnosis)) if diagnosis is not None else ""


def _arr(value):
    return None if value is None else np.asarray(value, dtype=float)


def load_inputs(obj):
    if "mino_rails" not in obj:
        raise LoftError(f"'{obj.name}' is not a Mino result object")
    rails = json.loads(obj["mino_rails"])
    params = params_from_dict(json.loads(obj["mino_params"]))
    return _arr(rails["a"]), _arr(rails["ta"]), _arr(rails["b"]), _arr(rails["tb"]), params


def load_diagnosis(obj) -> Diagnosis | None:
    raw = obj.get("mino_diagnosis", "")
    return diagnosis_from_dict(json.loads(raw)) if raw else None
```

- [ ] **Step 4: Add sections to inputs.py**

Append to `mino/blender/inputs.py`:
```python
def order_sections(rails_by_name: dict, first: str) -> list[str]:
    """Greedy nearest-centroid chain of section names starting at `first`."""
    centroids = {name: np.asarray(pts, dtype=float).mean(axis=0) for name, (pts, _) in rails_by_name.items()}
    ordered, remaining = [first], [n for n in rails_by_name if n != first]
    while remaining:
        last = centroids[ordered[-1]]
        nxt = min(remaining, key=lambda n: float(np.linalg.norm(centroids[n] - last)))
        ordered.append(nxt)
        remaining.remove(nxt)
    return ordered


def get_sections(context, samples):
    """Two rails from Edit Mode chains, or two or more curves ordered from the active one."""
    obj = context.active_object
    if context.mode == "EDIT_MESH" and obj is not None and obj.type == "MESH":
        a, b = edit_mode_rails(obj)
        return [a, b]
    curves = [o for o in context.selected_objects if o.type == "CURVE"]
    if len(curves) < 2:
        raise LoftError(f"select at least two curve objects ({len(curves)} selected), "
                        "or two edge chains in Edit Mode")
    rails_by_name = {o.name: curve_rail(o, context, samples) for o in curves}
    first = obj.name if obj in curves else curves[0].name
    return [rails_by_name[name] for name in order_sections(rails_by_name, first)]
```
Keep `get_rails` as it is (it still serves two-curve callers).

- [ ] **Step 5: Extend output.py**

In `mino/blender/output.py` add `import json` is not needed; add these functions and refactor `create_strip_object` to use `_write_twist_attributes`:
```python
def _write_twist_attributes(me, face_twist, twist_tolerance):
    """Face 'twist' float and corner 'twist_color' set as the active color."""
    import numpy as np
    twist = me.attributes.new("twist", "FLOAT", "FACE")
    twist.data.foreach_set("value", np.where(np.isfinite(face_twist), face_twist, 90.0).astype(np.float32))
    col = me.attributes.new("twist_color", "FLOAT_COLOR", "CORNER")
    face_colors = twist_colors(face_twist, twist_tolerance)
    corner_colors = np.repeat(face_colors, [len(p.loop_indices) for p in me.polygons], axis=0)
    col.data.foreach_set("color", corner_colors.astype(np.float32).ravel())
    me.color_attributes.active_color = col
    me.color_attributes.render_color_index = me.color_attributes.active_color_index


def _link_and_select(context, obj):
    context.collection.objects.link(obj)
    if context.mode == "OBJECT":
        for o in context.view_layer.objects:
            o.select_set(False)
        obj.select_set(True)
        context.view_layer.objects.active = obj


def create_result_object(context, name, points_a, tangents_a, points_b, tangents_b, params, result,
                         diagnose_flag: bool, strake: int = 0):
    """Strip object plus stored inputs and (optionally) a diagnosis."""
    from ..core.diagnose import diagnose
    from .state import store_inputs

    obj = create_strip_object(context, result, name, params.twist_tolerance)
    attr = obj.data.attributes.new("strake", "INT", "FACE")
    attr.data.foreach_set("value", [int(strake)] * len(obj.data.polygons))
    diagnosis = None
    if diagnose_flag:
        diagnosis = diagnose(points_a, points_b, tangents_a, tangents_b, params, result)
    store_inputs(obj, points_a, tangents_a, points_b, tangents_b, params, result, diagnosis)
    return obj, diagnosis
```
In `create_strip_object`, replace the `twist` and `twist_color` blocks with a call `_write_twist_attributes(me, result.face_twist, twist_tolerance)` placed after `planarity` and `split` are written and before the final `me.update()`, and replace the link/select block with `_link_and_select(context, obj)`. Behavior is unchanged; `tests/test_blender.py::test_operator_on_curves` still passes.

- [ ] **Step 6: Update the operator**

In `mino/blender/operator.py`:
- import `from ..core.strakes import chain_loft` and `from .output import create_result_object`.
- add properties after `consistent_creases`:
```python
    diagnose: BoolProperty(name="Diagnose", default=True,
                           description="Store ranked suggestions for non-developable regions on the result")
```
- replace the body of `execute` from the `try:` onward with:
```python
        try:
            sections = inputs.get_sections(context, params.samples)
            results = chain_loft(sections, params)
        except LoftError as exc:
            self.report({"ERROR"}, str(exc))
            return {"CANCELLED"}
        objs = []
        for k, (((pa, ta), (pb, tb)), result) in enumerate(zip(zip(sections, sections[1:]), results)):
            obj, diagnosis = create_result_object(context, "Mino", pa, ta, pb, tb, params, result,
                                                  self.diagnose, strake=k)
            objs.append(obj)
        if len(results) == 1:
            result = results[0]
            (pa, ta), (pb, tb) = sections
            if self.export_json:
                path = bpy.path.abspath(self.export_json)
                try:
                    with open(path, "w", encoding="utf-8") as fh:
                        json.dump(result_to_dict(result, pa, pb, "blender", params), fh)
                except OSError as exc:
                    self.report({"WARNING"}, f"Could not write {path}: {exc}")
            self.report({"INFO"}, format_report(result.report, result.failing_ranges))
            if result.failing_ranges:
                ranges = ", ".join(f"{a}-{b}" if a != b else str(a) for a, b in result.failing_ranges)
                self.report({"WARNING"}, f"Not developable at rulings {ranges}; see the Mino panel for suggestions")
        else:
            worst = max(r.report.max_twist for r in results)
            bad = sum(1 for r in results if r.failing_ranges)
            self.report({"INFO"}, f"Mino: {len(results)} strips, worst twist {worst:.1f} deg, "
                                  f"{bad} not developable")
        return {"FINISHED"}
```
- in `draw`, add `col.prop(self, "diagnose")` after `consistent_creases`.
- update the `poll` message and the docstring to say "two or more curves".

- [ ] **Step 7: Run the tests**

Run: `uv run --no-sync pytest tests/test_blender.py -q` then `uv run --no-sync pytest -q`.
Expected: all passed. The existing `test_operator_errors_on_bad_selection` still raises (one curve selected).

- [ ] **Step 8: Commit**

```bash
git add mino/blender/state.py mino/blender/inputs.py mino/blender/output.py mino/blender/operator.py tests/test_blender.py
git commit -m "Store loft inputs on results, chain sections, run diagnosis

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_012kESBz7SJpRjCw2UpaQv9E"
```

---

### Task 6: Remedy operators, Diagnosis panel, README

**Files:**
- Create: `mino/blender/remedies.py`, `tests/test_blender_remedies.py`
- Modify: `mino/blender/output.py` (add `create_dart_object`), `mino/blender/panel.py`, `mino/blender/__init__.py`, `README.md`

**Interfaces:**
- Consumes: `state.load_inputs`, `state.load_diagnosis`, `output.create_result_object`, `loft`, `prepare_rails`, `subdivide_sections`, `chain_loft`, `dart_proposal`, `dart_mesh`, `DartMesh`.
- Produces: operators `mino.reloft` (`MINO_OT_reloft`), `mino.subdivide` (`MINO_OT_subdivide`), `mino.dart` (`MINO_OT_dart`); `remedies.diagnosis_rows(obj) -> list[tuple[str, str | None, dict]]`; `output.create_dart_object(context, dart_mesh, name, twist_tolerance) -> obj`.

- [ ] **Step 1: Write the failing tests**

`tests/test_blender_remedies.py`:
```python
import json

import numpy as np
import pytest

bpy = pytest.importorskip("bpy")

import mino  # noqa: E402
from mino.blender.remedies import diagnosis_rows  # noqa: E402
from tests.cases import CASES  # noqa: E402


@pytest.fixture
def fresh_scene():
    bpy.ops.wm.read_factory_settings(use_empty=True)
    try:
        mino.register()
    except ValueError:
        pass
    yield bpy.context


def _make_poly_curve(name, points):
    cu = bpy.data.curves.new(name, "CURVE")
    cu.dimensions = "3D"
    sp = cu.splines.new("POLY")
    sp.points.add(len(points) - 1)
    for p, xyz in zip(sp.points, points):
        p.co = (float(xyz[0]), float(xyz[1]), float(xyz[2]), 1.0)
    obj = bpy.data.objects.new(name, cu)
    bpy.context.collection.objects.link(obj)
    return obj


def _select(objs, active):
    for o in bpy.context.view_layer.objects:
        o.select_set(False)
    for o in objs:
        o.select_set(True)
    bpy.context.view_layer.objects.active = active


def _twisted_loft(samples=40):
    case = CASES["twisted"](n=60)
    a = _make_poly_curve("A", case["points_a"])
    b = _make_poly_curve("B", case["points_b"])
    _select([a, b], a)
    assert bpy.ops.mino.loft(samples=samples) == {"FINISHED"}
    obj = bpy.data.objects["Mino"]
    _select([obj], obj)
    return obj


def test_diagnosis_rows_maps_kinds_to_operators():
    diag = {
        "failing_ranges": [[10, 20]], "max_twist": 20.0, "window_fix": 16, "strakes_needed": 4,
        "strakes_worst_twist": 4.0, "crease_runs": 1,
        "darts": [{"range": [10, 20], "ruling": 15, "wedge_deg": -12.5, "kind": "gusset"}],
        "suggestions": [
            {"kind": "window", "text": "w", "params": {"window": 16}, "rank": 1},
            {"kind": "subdivide", "text": "s", "params": {"strakes": 4}, "rank": 2},
            {"kind": "dart", "text": "d", "params": {"ruling": 15, "wedge_deg": -12.5, "kind": "gusset"}, "rank": 3},
            {"kind": "creases", "text": "c", "params": {"runs": 1}, "rank": 4},
        ],
    }
    rows = diagnosis_rows({"mino_diagnosis": json.dumps(diag)})
    assert [r[1] for r in rows] == ["mino.reloft", "mino.subdivide", "mino.dart", None]
    assert rows[0][2] == {"window": 16} and rows[1][2] == {"strakes": 4} and rows[2][2] == {"ruling": 15}
    assert "gusset" in rows[2][0] and "12.5" in rows[2][0]
    assert diagnosis_rows({"mino_diagnosis": ""}) == [("Diagnosis off for this loft", None, {})]
    assert diagnosis_rows({}) == [("Diagnosis off for this loft", None, {})]


def test_reloft_with_window_creates_new_object(fresh_scene):
    obj = _twisted_loft()
    assert bpy.ops.mino.reloft(window=16) == {"FINISHED"}
    new = bpy.data.objects["Mino.reloft"]
    assert json.loads(new["mino_params"])["window"] == 16
    assert json.loads(new["mino_params"])["samples"] == 40
    assert "mino_diagnosis" in new


def test_subdivide_creates_strake_objects(fresh_scene):
    obj = _twisted_loft()
    assert bpy.ops.mino.subdivide(strakes=3) == {"FINISHED"}
    names = [f"Mino.strake.{k}" for k in range(3)]
    for k, name in enumerate(names):
        strip = bpy.data.objects[name]
        assert {d.value for d in strip.data.attributes["strake"].data} == {k}
        assert "mino_rails" in strip
    r0, r1 = json.loads(bpy.data.objects[names[0]]["mino_rails"]), json.loads(bpy.data.objects[names[1]]["mino_rails"])
    assert np.allclose(np.array(r0["b"]), np.array(r1["a"]))


def test_dart_creates_seam_edges(fresh_scene):
    obj = _twisted_loft()
    assert bpy.ops.mino.dart() == {"FINISHED"}
    dart = bpy.data.objects["Mino.dart"]
    me = dart.data
    assert "seam" in me.attributes and me.attributes["seam"].domain == "EDGE"
    seams = [e for e in me.edges if e.use_seam]
    assert len(seams) == 3
    assert sum(d.value for d in me.attributes["seam"].data) == 3
    assert "twist" in me.attributes and me.color_attributes.active_color is not None
    assert len(me.vertices) > 2 * 40


def test_remedies_require_a_mino_object(fresh_scene):
    case = CASES["cylinder"](n=30)
    a = _make_poly_curve("A", case["points_a"])
    _select([a], a)
    assert not bpy.ops.mino.subdivide.poll()
    assert not bpy.ops.mino.dart.poll()
    assert not bpy.ops.mino.reloft.poll()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run --no-sync pytest tests/test_blender_remedies.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'mino.blender.remedies'`.

- [ ] **Step 3: Add create_dart_object to output.py**

Append to `mino/blender/output.py`:
```python
def create_dart_object(context, dart, name, twist_tolerance):
    """Refined mesh with the cut ruling marked as seams (attribute and use_seam)."""
    import bpy

    me = bpy.data.meshes.new(name)
    me.from_pydata(dart.verts.tolist(), [list(map(int, e)) for e in dart.seam_edges],
                   [list(map(int, f)) for f in dart.faces])
    me.validate()
    me.update()
    _write_twist_attributes(me, dart.face_twist, twist_tolerance)
    by_key = {tuple(sorted(e.vertices)): e.index for e in me.edges}
    flags = [False] * len(me.edges)
    for a, b in dart.seam_edges:
        idx = by_key.get(tuple(sorted((int(a), int(b)))))
        if idx is not None:
            flags[idx] = True
            me.edges[idx].use_seam = True
    seam = me.attributes.new("seam", "BOOLEAN", "EDGE")
    seam.data.foreach_set("value", flags)
    me.update()
    obj = bpy.data.objects.new(name, me)
    _link_and_select(context, obj)
    return obj
```

- [ ] **Step 4: Write remedies.py**

`mino/blender/remedies.py`:
```python
"""Remedy operators acting on the active Mino result object, and the panel rows."""
from __future__ import annotations

import json
from dataclasses import replace

import bpy
from bpy.props import EnumProperty, FloatProperty, IntProperty

from ..core import LoftError, loft
from ..core.dart import dart_mesh, dart_proposal
from ..core.diagnose import diagnosis_from_dict
from ..core.rails import prepare_rails
from ..core.strakes import chain_loft, subdivide_sections
from . import output
from .state import load_diagnosis, load_inputs

OFF_ROW = ("Diagnosis off for this loft", None, {})


def diagnosis_rows(obj):
    """(text, operator idname or None, property values) per suggestion, from stored JSON."""
    raw = obj.get("mino_diagnosis", "") if hasattr(obj, "get") else ""
    if not raw:
        return [OFF_ROW]
    d = diagnosis_from_dict(json.loads(raw))
    rows = []
    for s in d.suggestions:
        if s.kind == "window":
            rows.append((f"Re-loft with Window {s.params['window']}", "mino.reloft", {"window": int(s.params["window"])}))
        elif s.kind == "subdivide":
            rows.append((f"Subdivide into {s.params['strakes']} strakes", "mino.subdivide", {"strakes": int(s.params["strakes"])}))
        elif s.kind == "unsolved":
            rows.append((s.text, "mino.subdivide", {"strakes": int(s.params["strakes"])}))
        elif s.kind == "dart":
            rows.append((f"Cut {s.params['kind']} at ruling {s.params['ruling']} ({abs(float(s.params['wedge_deg'])):.1f}°)",
                         "mino.dart", {"ruling": int(s.params["ruling"])}))
        else:
            rows.append((s.text, None, {}))
    return rows


class _MinoRemedy:
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context):
        obj = context.active_object
        ok = context.mode == "OBJECT" and obj is not None and "mino_rails" in obj
        if not ok:
            cls.poll_message_set("Select a Mino result object in Object Mode")
        return ok


class MINO_OT_reloft(_MinoRemedy, bpy.types.Operator):
    """Loft the stored rails again with changed settings"""
    bl_idname = "mino.reloft"
    bl_label = "Re-loft"

    window: IntProperty(name="Window", default=0, min=0, max=100, description="0 keeps the stored value")
    samples: IntProperty(name="Samples", default=0, min=0, max=400, description="0 keeps the stored value")
    twist_tolerance: FloatProperty(name="Twist Tolerance", default=0.0, min=0.0, max=90.0,
                                   description="0 keeps the stored value")

    def execute(self, context):
        obj = context.active_object
        try:
            pa, ta, pb, tb, params = load_inputs(obj)
            overrides = {k: v for k, v in (("window", self.window), ("samples", self.samples),
                                           ("twist_tolerance", self.twist_tolerance)) if v}
            params = replace(params, **overrides)
            result = loft(pa, pb, params, ta, tb)
        except LoftError as exc:
            self.report({"ERROR"}, str(exc))
            return {"CANCELLED"}
        new, _ = output.create_result_object(context, f"{obj.name}.reloft", pa, ta, pb, tb, params, result, True)
        self.report({"INFO"}, new["mino_report"])
        return {"FINISHED"}


class MINO_OT_subdivide(_MinoRemedy, bpy.types.Operator):
    """Split the loft into narrower developable strips (strakes)"""
    bl_idname = "mino.subdivide"
    bl_label = "Subdivide into Strakes"

    strakes: IntProperty(name="Strakes", default=0, min=0, max=8,
                         description="Number of strips; 0 uses the diagnosis suggestion")

    def execute(self, context):
        obj = context.active_object
        try:
            pa, ta, pb, tb, params = load_inputs(obj)
            diag = load_diagnosis(obj)
            strakes = self.strakes or (diag.strakes_needed if diag and diag.strakes_needed else 2)
            base = loft(pa, pb, params, ta, tb)
            sections = subdivide_sections(pa, pb, ta, tb, params, base.rulings, strakes)
            strips = chain_loft(sections, params)
        except LoftError as exc:
            self.report({"ERROR"}, str(exc))
            return {"CANCELLED"}
        for k, (((sa, sta), (sb, stb)), strip) in enumerate(zip(zip(sections, sections[1:]), strips)):
            output.create_result_object(context, f"{obj.name}.strake.{k}", sa, sta, sb, stb, params, strip, True, strake=k)
        worst = max(s.report.max_twist for s in strips)
        bad = sum(1 for s in strips if s.failing_ranges)
        self.report({"INFO"}, f"Mino: {strakes} strakes, worst twist {worst:.1f} deg, {bad} not developable")
        return {"FINISHED"}


class MINO_OT_dart(_MinoRemedy, bpy.types.Operator):
    """Refine the strip and mark a dart (gusset) cut where twist peaks"""
    bl_idname = "mino.dart"
    bl_label = "Cut Dart"

    ruling: IntProperty(name="Ruling", default=-1, min=-1, max=800,
                        description="Cut ruling; -1 uses the diagnosis suggestion")
    mid_rails: IntProperty(name="Mid Rails", default=3, min=1, max=6)
    dart_from: EnumProperty(name="Open End", default="B", items=[
        ("B", "Rail B", "The cut opens at rail B and its tip stops short of rail A"),
        ("A", "Rail A", "The cut opens at rail A and its tip stops short of rail B"),
    ])

    def execute(self, context):
        obj = context.active_object
        try:
            pa, ta, pb, tb, params = load_inputs(obj)
            diag = load_diagnosis(obj)
            base = loft(pa, pb, params, ta, tb)
            ra, rb = prepare_rails(pa, pb, params.samples, ta, tb)
            if self.ruling >= 0:
                ruling = min(self.ruling, len(base.rulings) - 1)
            elif diag and diag.darts:
                ruling = diag.darts[0].ruling
            else:
                ruling = int(base.ruling_twist.argmax())
            rng = next((r for r in base.failing_ranges if r[0] <= ruling <= r[1]), (ruling, ruling))
            proposal = dart_proposal(ra, rb, base.rulings, base.ruling_twist, rng, self.mid_rails)
            proposal = replace(proposal, ruling=ruling)
            dm = dart_mesh(ra, rb, base.rulings, base.ruling_twist, proposal, self.mid_rails,
                           params.planar_tolerance, self.dart_from)
        except LoftError as exc:
            self.report({"ERROR"}, str(exc))
            return {"CANCELLED"}
        output.create_dart_object(context, dm, f"{obj.name}.dart", params.twist_tolerance)
        self.report({"INFO"}, f"Mino: {proposal.kind} at ruling {ruling}, "
                              f"{abs(proposal.wedge_deg):.1f} deg wedge, {len(dm.faces)} faces")
        return {"FINISHED"}


classes = (MINO_OT_reloft, MINO_OT_subdivide, MINO_OT_dart)
```

- [ ] **Step 5: Panel, registration, README**

`mino/blender/panel.py`, replace `draw`:
```python
    def draw(self, context):
        from .remedies import diagnosis_rows

        layout = self.layout
        col = layout.column(align=True)
        col.operator(MINO_OT_loft.bl_idname, icon="MOD_CURVE")
        col.separator()
        col.label(text="Select two or more curves (active = first),")
        col.label(text="or two edge chains in Edit Mode.")
        col.label(text="Settings: Adjust Last Operation panel.")
        col.separator()
        col.label(text="View twist: Solid shading > Color > Attribute")

        obj = context.active_object
        if obj is None or "mino_rails" not in obj:
            return
        box = layout.box()
        box.label(text="Diagnosis", icon="INFO")
        box.label(text=obj.get("mino_report", ""))
        for text, idname, props in diagnosis_rows(obj):
            row = box.row()
            if idname is None:
                row.label(text=text)
                continue
            op = row.operator(idname, text=text)
            for key, value in props.items():
                setattr(op, key, value)
```

`mino/blender/__init__.py`: import `remedies` and set `_classes = (operator.MINO_OT_loft, *remedies.classes, panel.MINO_PT_panel)`.

`README.md`: after the Parameters section add:
```markdown
## When a loft is not developable

Select the Mino result. The sidebar shows a Diagnosis with ranked
suggestions and a button for each:

- Re-loft with a larger Window when leaning rulings alone fix it.
- Subdivide into N strakes: mid-rails on the chosen rulings, one object
  per strip, each developable within tolerance.
- Cut a gusset (or dart): a refined mesh with a seam along the worst
  ruling and the wedge angle you should expect in the flat pattern. A
  ruled surface has non-positive curvature, so this is normally a gusset.
- Creases: with Consistent Creases on, split diagonals run in one
  direction per run.

Every remedy creates new objects next to the original. Selecting more than
two curves lofts them in order as a chain of strips.
```

- [ ] **Step 6: Run the tests**

Run: `uv run --no-sync pytest tests/test_blender_remedies.py -q` then `uv run --no-sync pytest -q`.
Expected: all passed. If `test_dart_creates_seam_edges` finds fewer than 3 `use_seam` edges, check that `me.validate()` did not drop the seam edges; print `len(me.edges)` and the `by_key` misses in the report and stop.

- [ ] **Step 7: Commit**

```bash
git add mino/blender/remedies.py mino/blender/output.py mino/blender/panel.py mino/blender/__init__.py README.md tests/test_blender_remedies.py
git commit -m "Add remedy operators and the Diagnosis panel

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_012kESBz7SJpRjCw2UpaQv9E"
```
