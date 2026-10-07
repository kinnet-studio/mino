# Adaptive Sampling and Quad Strips Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Place rail samples by bending (Adaptive) and build the strip as quads between consecutive rulings (Quads), keeping triangles only where a quad cannot be made planar.

**Architecture:** Adaptive sampling is a shared position map computed in `mino/core/rails.py` and applied to both rails before alignment. Quads is a new stage in `mino/core/quads.py` that runs after alignment: it spreads each fan's shared end by at most ¼ sample, interpolates the vertices along the rails, and re-measures twist. `result.rulings` stays the sample-index path, so strakes, darts, relaxation and diagnosis keep working on samples.

**Tech Stack:** Python 3.11, numpy, pytest, Blender 4.2+ (`bpy==4.5.*` in the dev group for the Blender tests).

**Spec:** `docs/superpowers/specs/2026-10-07-mino-adaptive-sampling-and-quads-design.md`

## Global Constraints

- `mino/core` stays pure numpy: no `bpy` or `mathutils` imports. No new dependencies.
- `LoftParams.adaptive: float = 0.0`; `LoftParams.quads: bool = True`.
- `adaptive == 0` takes today's code path exactly (no positions passed to `resample`).
- Core validation: `LoftError("adaptive must be at least 0 and below 1")` unless `0 <= adaptive < 1`.
- `SPREAD = 0.25` samples.
- Bezier input oversampling: 4 today, 16 when `adaptive > 0`.
- UI copy, verbatim: Adaptive: FloatProperty `name="Adaptive"`, default 0.0, min 0.0, max 0.9, description "Share of samples placed where the rails bend; 0 spaces them evenly". Quads: BoolProperty `name="Quads"`, default True, description "Give every ruling its own rail points so faces are quads; quads that cannot be made flat are still split".
- Tangents computed from samples keep today's `central_difference`; `Rail` and `relax.moved_tangents_b` do not change (spec 3.4, 3.7).
- Commit messages: one imperative sentence like the existing history ("Add …", "Keep …"), ending with the session's attribution trailer.
- Full suite: `uv run pytest -q`, about 2 minutes, 143 passing at baseline.

## Review Focus

1. Sharp corners, hairpins and repeated points at high Adaptive: samples stay distinct and tangents stay finite unit vectors. Test in Task 1.
2. Subdivide into strakes with Quads on (and with Adaptive): neighbouring strips meet on the shared rail with matching end corners. Test in Task 4.
3. Mino results stored before this change (no `adaptive`/`quads` keys): Re-loft works and uses the defaults. Test in Task 6.
4. Planarize on thin fan quads: rail vertices still move smoothly (MAX_KINK). Test in Task 4.
5. A "Re-loft with Window N" suggestion in Quads mode matches the loft it produces. Test in Task 4.

---

### Task 1: Shared adaptive positions

**Files:**
- Modify: `mino/core/rails.py`
- Test: `tests/test_rails.py`

**Interfaces:**
- Produces:
  - `bend_profile(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]`: normalized knots `t` and cumulative turning `theta` (radians) of a deduplicated polyline, per spec 3.2.
  - `shared_positions(points_a, points_b, samples: int, adaptive: float) -> np.ndarray`: `(samples,)` normalized positions shared by both rails, per spec 3.3. Deduplicates each rail with `dedupe_indices` first.
  - `resample(points, samples, tangents=None, positions=None) -> Rail`: with `positions`, targets are `positions * L`.
  - `prepare_rails(points_a, points_b, samples, tangents_a=None, tangents_b=None, adaptive=0.0)`: validates `adaptive`, and when it is above 0 computes `shared_positions` on A and the oriented B and passes them to both `resample` calls.

- [ ] **Step 1: Write the failing tests** in `tests/test_rails.py` (add `bend_profile`, `shared_positions` to the import).

```python
L_RAIL = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0]], float)


def _arc_then_line(n=400):
    """Quarter arc of radius 1, then a straight of length 3, tangent-continuous at (1, 1, 0)."""
    th = np.linspace(0.0, np.pi / 2, n)
    arc = np.column_stack([np.sin(th), 1 - np.cos(th), np.zeros(n)])
    u = np.linspace(0.0, 3.0, n)[1:]
    return np.vstack([arc, np.column_stack([np.ones_like(u), 1 + u, np.zeros_like(u)])])


def _true_tangent(p):
    a = np.arctan2(p[:, 0], 1 - p[:, 1])
    arc = np.column_stack([np.cos(a), np.sin(a), np.zeros(len(p))])
    return np.where((p[:, 1] <= 1)[:, None], arc, [0.0, 1.0, 0.0])


def _outline_error(curve, poly):
    """Largest distance from the curve's points to the polyline."""
    best = np.full(len(curve), np.inf)
    for a, b in zip(poly[:-1], poly[1:]):
        d = b - a
        t = np.clip((curve - a) @ d / (d @ d), 0.0, 1.0)
        best = np.minimum(best, np.linalg.norm(curve - (a + t[:, None] * d), axis=1))
    return float(best.max())


def test_bend_profile_spreads_a_corner_over_its_half_segments():
    t, theta = bend_profile(L_RAIL)
    assert np.allclose(t, [0, 0.25, 0.75, 1])
    assert np.allclose(theta, [0, 0, np.pi / 2, np.pi / 2])


def test_bend_profile_of_one_segment_is_flat():
    t, theta = bend_profile(np.array([[0, 0, 0], [2, 0, 0]], float))
    assert np.allclose(t, [0, 0.5, 1]) and not theta.any()


def test_shared_positions_follow_the_bending_rail():
    straight = np.array([[0, 0, 1], [3, 0, 1]], float)
    t = shared_positions(L_RAIL, straight, 21, 0.6)
    assert t[0] == 0.0 and t[-1] == 1.0 and np.all(np.diff(t) > 0)
    gaps = np.diff(t)
    assert gaps[9] < 0.05 - 1e-6 < gaps[0]  # dense around the corner, sparse at the ends; even is 1/20


def test_shared_positions_are_even_for_straight_rails():
    a = np.array([[0, 0, 0], [2, 0, 0]], float)
    assert np.allclose(shared_positions(a, a + [0, 1, 0], 11, 0.6), np.linspace(0, 1, 11))


def test_resample_puts_samples_at_given_positions():
    line = np.array([[0, 0, 0], [4, 0, 0]], float)
    rail = resample(line, 4, positions=np.array([0.0, 0.1, 0.5, 1.0]))
    assert np.allclose(rail.points[:, 0], [0, 0.4, 2, 4])


def test_adaptive_clusters_samples_around_a_corner():
    adapt, _ = prepare_rails(L_RAIL, L_RAIL + [0, 0, 1], 21, adaptive=0.6)
    gaps = np.linalg.norm(np.diff(adapt.points, axis=0), axis=1)
    k = int(np.argmin(np.linalg.norm(adapt.points - [1, 0, 0], axis=1)))
    assert gaps[k - 1:k + 1].max() < 0.1 - 1e-6  # even spacing is 2/20
    assert gaps.min() > 1e-9


def test_adaptive_straight_rails_match_even_spacing():
    a = np.array([[0, 0, 0], [2, 0, 0]], float)
    for x, y in zip(prepare_rails(a, a + [0, 1, 0], 11, adaptive=0.6), prepare_rails(a, a + [0, 1, 0], 11)):
        assert np.allclose(x.points, y.points)


def test_adaptive_halves_outline_error_on_arc_then_line():
    p = _arc_then_line()
    even, _ = prepare_rails(p, p + [0, 0, 1], 16)
    adapt, _ = prepare_rails(p, p + [0, 0, 1], 16, adaptive=0.6)
    assert _outline_error(p, adapt.points) < 0.5 * _outline_error(p, even.points)


def test_adaptive_tangents_stay_within_a_degree():
    p = _arc_then_line()
    ra, _ = prepare_rails(p, p + [0, 0, 1], 30, adaptive=0.6)
    cos = np.einsum("ij,ij->i", ra.tangents, _true_tangent(ra.points))
    assert np.degrees(np.arccos(np.clip(cos, -1, 1))).max() < 1.0


@pytest.mark.parametrize("adaptive", [-0.1, 1.0])
def test_adaptive_out_of_range_raises(adaptive):
    with pytest.raises(LoftError, match="adaptive must be at least 0 and below 1"):
        prepare_rails(L_RAIL, L_RAIL + [0, 0, 1], 10, adaptive=adaptive)


def test_adaptive_survives_a_hairpin_and_repeated_points():
    hair = np.array([[0, 0, 0], [1, 0, 0], [1, 0, 0], [0, 0.001, 0]], float)
    for rail in prepare_rails(hair, hair + [0, 0, 1], 30, adaptive=0.9):
        assert np.linalg.norm(np.diff(rail.points, axis=0), axis=1).min() > 1e-9
        assert np.isfinite(rail.tangents).all()
        assert np.allclose(np.linalg.norm(rail.tangents, axis=1), 1.0)
```

- [ ] **Step 2: Run them to verify they fail**

Run: `uv run pytest tests/test_rails.py -q`
Expected: collection error, `ImportError: cannot import name 'bend_profile'`.

- [ ] **Step 3: Implement `bend_profile`, `shared_positions`, `resample(..., positions=None)` and `prepare_rails(..., adaptive=0.0)` in `mino/core/rails.py`**

Formulas are in spec 3.2 and 3.3. `shared_positions` evaluates `U` on `np.union1d` of both rails' knots, inverts it with `np.interp(np.linspace(0, 1, samples), U, T)`, then sets the first and last entries to exactly 0 and 1. If the summed turning is below `1e-9`, it returns `np.linspace(0, 1, samples)`. `prepare_rails` validates `adaptive` before anything else.

- [ ] **Step 4: Run them to verify they pass**

Run: `uv run pytest tests/test_rails.py -q`
Expected: all pass, the existing rail tests included.

- [ ] **Step 5: Commit**

```bash
git add mino/core/rails.py tests/test_rails.py
git commit -m "Add shared adaptive sample positions to rail preparation"
```

### Task 2: Thread Adaptive through the loft and remedies

**Files:**
- Modify: `mino/core/types.py` (`LoftParams.adaptive`), `mino/core/__init__.py`, `mino/core/strakes.py`, `mino/core/diagnose.py`, `mino/core/relax.py`, `mino/blender/remedies.py` (dart remedy)
- Test: `tests/test_types.py`, `tests/test_strakes.py`, `tests/test_loft.py`

**Interfaces:**
- Consumes: `prepare_rails(..., adaptive=...)` from Task 1.
- Produces: `LoftParams.adaptive: float = 0.0`. Every `prepare_rails` call in `mino/` passes `params.adaptive`.

- [ ] **Step 1: Write the failing tests**

`tests/test_types.py`, inside `test_default_params`:
```python
    assert p.adaptive == 0.0
```

`tests/test_strakes.py`:
```python
def test_subdivide_sections_use_adaptive_rails():
    case, params, res = _twisted(adaptive=0.6)
    secs = subdivide_sections(case["points_a"], case["points_b"], case["tangents_a"], case["tangents_b"],
                              params, res.rulings, strakes=2)
    ra, rb = prepare_rails(case["points_a"], case["points_b"], params.samples, case["tangents_a"],
                           case["tangents_b"], adaptive=0.6)
    i, j = np.array(res.rulings).T
    assert np.allclose(secs[1][0], 0.5 * (ra.points[i] + rb.points[j]))
```

`tests/test_loft.py`:
```python
@pytest.mark.parametrize("name", sorted(CASES))
def test_adaptive_loft_runs_on_every_case(name):
    _, params, res = _run(name, adaptive=0.6)
    assert np.isfinite(res.verts).all()
    assert res.report.ruling_count == len(res.rulings)
    if name in ("cylinder", "cone"):
        assert res.ruling_twist.max() < 0.01 and res.failing_ranges == []
```

- [ ] **Step 2: Run them to verify they fail**

Run: `uv run pytest tests/test_types.py tests/test_strakes.py tests/test_loft.py -q -k "default_params or adaptive"`
Expected: FAIL, `TypeError: LoftParams.__init__() got an unexpected keyword argument 'adaptive'` (and `AttributeError` in `test_default_params`).

- [ ] **Step 3: Add `LoftParams.adaptive: float = 0.0`** after `samples` in `mino/core/types.py`, and pass `adaptive=params.adaptive` to each `prepare_rails` call: `loft`, `strakes.subdivide_sections`, `diagnose.diagnose`, `relax.relax_rail_b`, and `MINO_OT_dart.execute` in `mino/blender/remedies.py`.

- [ ] **Step 4: Verify every call site and run the suite**

Run: `grep -n "prepare_rails(" mino --include=*.py`
Expected: every call except the definition passes `params.adaptive`.

Run: `uv run pytest -q`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add mino tests
git commit -m "Pass Adaptive to every rail preparation"
```

### Task 3: Fan spreading and the quad strip

**Files:**
- Create: `mino/core/quads.py`
- Test: `tests/test_quads.py`

**Interfaces:**
- Consumes: `Rail` (`mino/core/types.py`), `paired_twist(points_a, tangents_a, points_b, tangents_b)` (`mino/core/twist.py`), `normalize_rows` (`mino/core/rails.py`).
- Produces:
  - `SPREAD = 0.25`
  - `spread_fans(path) -> np.ndarray`: `(K, 2)` float, per spec 4.2.
  - `rail_at(rail: Rail, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]`: points and unit tangents at fractional sample indices; integer `x` returns the samples exactly.
  - `quad_strip(rail_a: Rail, rail_b: Rail, path) -> tuple[np.ndarray, list, np.ndarray]`: `(verts (2K, 3), faces [(k, k+1, K+k+1, K+k)], twist (K,))`.

- [ ] **Step 1: Write the failing tests** in `tests/test_quads.py`

```python
import numpy as np

from mino.core.quads import SPREAD, quad_strip, rail_at, spread_fans
from mino.core.rails import prepare_rails
from mino.core.twist import twist_matrix
from tests.cases import CASES


def _random_path(rng, n, m):
    i = j = 0
    path = [(0, 0)]
    while (i, j) != (n - 1, m - 1):
        moves = [(di, dj) for di, dj in ((1, 1), (1, 0), (0, 1)) if i + di < n and j + dj < m]
        di, dj = moves[rng.integers(len(moves))]
        i, j = i + di, j + dj
        path.append((i, j))
    return path


def _cylinder_rails(n=12):
    case = CASES["cylinder"]()
    return prepare_rails(case["points_a"], case["points_b"], n, case["tangents_a"], case["tangents_b"])


def test_spread_fans_known_path():
    path = [(0, 0), (0, 1), (0, 2), (1, 3), (2, 3), (3, 4)]
    assert np.allclose(spread_fans(path), [[0, 0], [0.125, 1], [0.25, 2], [1, 2.75], [2, 3.25], [3, 4]])


def test_spread_fans_random_paths_strictly_increase():
    rng = np.random.default_rng(7)
    for _ in range(300):
        n, m = (int(v) for v in rng.integers(2, 40, size=2))
        path = _random_path(rng, n, m)
        out, p = spread_fans(path), np.array(path, float)
        assert out.shape == p.shape
        assert np.all(np.diff(out, axis=0) > 0)
        assert np.array_equal(out[0], p[0]) and np.array_equal(out[-1], p[-1])
        assert np.abs(out - p).max() <= SPREAD + 1e-12


def test_spread_fans_leaves_fan_free_paths_alone():
    path = [(k, k) for k in range(10)]
    assert np.array_equal(spread_fans(path), np.array(path, float))


def test_rail_at_integer_indices_returns_samples():
    ra, _ = _cylinder_rails()
    pts, tans = rail_at(ra, np.arange(12, dtype=float))
    assert np.array_equal(pts, ra.points) and np.allclose(tans, ra.tangents)


def test_rail_at_interpolates_between_samples():
    ra, _ = _cylinder_rails()
    pts, tans = rail_at(ra, np.array([0.5]))
    assert np.allclose(pts[0], 0.5 * (ra.points[0] + ra.points[1]))
    assert np.isclose(np.linalg.norm(tans[0]), 1.0)


def test_quad_strip_on_a_diagonal_path_is_the_grid():
    ra, rb = _cylinder_rails()
    n = 12
    verts, faces, twist = quad_strip(ra, rb, [(k, k) for k in range(n)])
    assert np.array_equal(verts, np.vstack([ra.points, rb.points]))
    assert faces == [(k, k + 1, n + k + 1, n + k) for k in range(n - 1)]
    assert np.allclose(twist, np.diagonal(twist_matrix(ra, rb, 1)), atol=1e-9)
```

- [ ] **Step 2: Run them to verify they fail**

Run: `uv run pytest tests/test_quads.py -q`
Expected: collection error, `ModuleNotFoundError: No module named 'mino.core.quads'`.

- [ ] **Step 3: Implement `mino/core/quads.py`**

Module docstring: "Quad strips: give every ruling its own rail points by spreading fans." `spread_fans` walks the steps and finds each maximal run of identical non-diagonal steps. It adds the linear `lo`→`hi` offset of spec 4.2 to the coordinate the run holds fixed. `rail_at` takes `i = clip(floor(x), 0, n-2)` and `u = x - i`, and interpolates as `p[i]·(1−u) + p[i+1]·u`, not `p[i] + u·(p[i+1]−p[i])`, so that integer `x` returns the sample bit for bit (the `array_equal` tests depend on it). Tangents are interpolated the same way, then normalized.

- [ ] **Step 4: Run them to verify they pass**

Run: `uv run pytest tests/test_quads.py -q`
Expected: 6 passed.

- [ ] **Step 5: Commit**

```bash
git add mino/core/quads.py tests/test_quads.py
git commit -m "Add fan spreading and quad strip construction"
```

### Task 4: Quads in the loft and diagnosis

**Files:**
- Modify: `mino/core/types.py` (`LoftParams.quads`, `StripResult.ruling_verts`), `mino/core/__init__.py`, `mino/core/diagnose.py` (`probe_window`)
- Test: `tests/test_loft.py`, `tests/test_diagnose.py`, `tests/test_strakes.py`, `tests/test_types.py`

**Interfaces:**
- Consumes: `quad_strip` from Task 3.
- Produces:
  - `LoftParams.quads: bool = True`.
  - `StripResult.ruling_verts: list = field(default_factory=list)`, declared after `layout`. It holds `(k, K + k)` per ruling with quads on, and `(i, n + j)` per path entry with quads off.
  - Each rail has `m = len(verts) // 2` vertices: `K` with quads on, `n` with them off.

- [ ] **Step 1: Write the failing tests**

`tests/test_types.py`, inside `test_default_params`:
```python
    assert p.quads is True
```

`tests/test_loft.py` (`_rail_kinks` already exists there):
```python
@pytest.mark.parametrize("name", ["ellipse", "offset_cylinder"])
def test_quads_replace_fan_triangles(name):
    quad, grid = _run(name)[2], _run(name, quads=False)[2]
    assert all(len(f) == 4 for f in quad.faces) and not quad.face_split.any()
    assert quad.report.max_twist == pytest.approx(grid.report.max_twist, rel=1e-9)


def test_quads_on_twisted_leave_only_split_triangles():
    res = _run("twisted")[2]
    assert all(len(f) == 4 or res.face_split[k] for k, f in enumerate(res.faces))


@pytest.mark.parametrize("name", ["cylinder", "cone"])
def test_quads_match_the_grid_without_fans(name):
    quad, grid = _run(name)[2], _run(name, quads=False)[2]
    assert np.array_equal(quad.verts, grid.verts) and quad.faces == grid.faces


@pytest.mark.parametrize("name", ["cylinder", "cone", "ellipse", "offset_cylinder"])
def test_quads_unfold_to_the_same_area(name):
    res = _run(name)[2]
    assert res.report.area_unfolded == pytest.approx(res.report.area_3d, rel=1e-3)


@pytest.mark.parametrize("quads", [True, False])
def test_ruling_verts_pair_each_ruling(quads):
    _, params, res = _run("ellipse", quads=quads)
    m = len(res.verts) // 2
    expected = ([(k, m + k) for k in range(len(res.rulings))] if quads
                else [(i, params.samples + j) for i, j in res.rulings])
    assert res.ruling_verts == expected


@pytest.mark.parametrize("name", ["ellipse", "twisted"])
def test_planarize_keeps_quad_rails_smooth(name):
    on, off = _run(name, samples=24)[2], _run(name, samples=24, planarize=False)[2]
    m = len(on.verts) // 2
    assert _rail_kinks(on.verts[:m], off.verts[:m]).max() <= MAX_KINK + 1e-9
    assert _rail_kinks(on.verts[m:], off.verts[m:]).max() <= MAX_KINK + 1e-9
```

`tests/test_diagnose.py`:
```python
def test_probe_window_measures_the_quad_strip():
    case, params, res = _run("ellipse")
    ra, rb = prepare_rails(case["points_a"], case["points_b"], params.samples, case["tangents_a"], case["tangents_b"])
    assert probe_window(ra, rb, params, params.window) == pytest.approx(res.report.max_twist, rel=1e-12)
```
(add `import pytest`).

`tests/test_strakes.py`:
```python
def _distance_to_polyline(points, poly):
    best = np.full(len(points), np.inf)
    for a, b in zip(poly[:-1], poly[1:]):
        d = b - a
        t = np.clip((points - a) @ d / (d @ d), 0.0, 1.0)
        best = np.minimum(best, np.linalg.norm(points - (a + t[:, None] * d), axis=1))
    return best


@pytest.mark.parametrize("adaptive", [0.0, 0.6])
def test_quad_strakes_meet_on_the_shared_rail(adaptive):
    case, params, res = _twisted(adaptive=adaptive)
    secs = subdivide_sections(case["points_a"], case["points_b"], case["tangents_a"], case["tangents_b"],
                              params, res.rulings, strakes=3)
    strips = chain_loft(secs, params)
    for (mid, _), left, right in zip(secs[1:-1], strips, strips[1:]):
        ml, mr = len(left.verts) // 2, len(right.verts) // 2
        width = np.mean(np.linalg.norm(left.verts[ml:] - left.verts[:ml], axis=1))
        assert _distance_to_polyline(left.verts[ml:], mid).max() < 1e-3 * width
        assert _distance_to_polyline(right.verts[:mr], mid).max() < 1e-3 * width
        assert np.allclose(left.verts[ml], right.verts[0], atol=1e-12)
        assert np.allclose(left.verts[-1], right.verts[mr - 1], atol=1e-12)
```

- [ ] **Step 2: Run them to verify they fail**

Run: `uv run pytest tests/test_types.py tests/test_loft.py tests/test_diagnose.py tests/test_strakes.py -q -k "default_params or quad or ruling_verts"`
Expected: FAIL, `TypeError ... unexpected keyword argument 'quads'` / `AttributeError ... 'ruling_verts'`.

- [ ] **Step 3: Implement the quads branch in `loft`** (`mino/core/__init__.py`) per spec 4.4.

With `params.quads`: `verts, faces, ruling_twist = quad_strip(rail_a, rail_b, path)`, and `m = len(path)`. Without it, today's grid code runs with `m = n`. After the branch, every use of `n` takes `m` instead: the planarize ruling lengths (measured from `ruling_verts`), `pinned`, `pin_a`/`pin_b` and the `rails` chains. Add `LoftParams.quads` after `consistent_creases`, and set `ruling_verts` on the result.

- [ ] **Step 4: Make `probe_window` measure the quad strip when `params.quads`** (`mino/core/diagnose.py`): `float(np.max(quad_strip(rail_a, rail_b, path)[2]))`, otherwise today's expression.

- [ ] **Step 5: Run the new tests, then the full suite**

Run: `uv run pytest tests/test_types.py tests/test_loft.py tests/test_diagnose.py tests/test_strakes.py -q -k "default_params or quad or ruling_verts"`
Expected: all pass.

Run: `uv run pytest -q`
Expected: some existing tests fail because they rely on the grid layout. A test qualifies if it indexes `verts` by `samples`, compares rail vertices to samples, or counts the grid's fan or split faces. Likely candidates:
- `test_loft.py`: `test_planarize_keeps_rail_endpoints`, `test_planarize_reduces_splits_on_coarse_ellipse`, `test_planarize_keeps_rails_smooth`, `test_consistent_creases_on_twisted_strip`, `test_result_to_dict_is_json_serializable`
- `test_strakes.py`: `test_chain_loft_shares_rails`, `test_subdivide_strips_share_rails_with_planarize_on`, `test_loft_pins_whole_rails_when_asked`

Fix only a test that fails AND relies on the grid layout, by adding `quads=False` to its params, and change nothing else in it. Any other failure is a bug in Steps 3–4: fix the code, not the test.

Run: `uv run pytest -q`
Expected: all pass.

- [ ] **Step 6: Commit**

```bash
git add mino/core tests
git commit -m "Build the strip as quads between consecutive rulings"
```

### Task 5: Export and viewer

**Files:**
- Modify: `mino/core/export.py`, `tools/viewer/index.html` (ruling drawing, around line 151), `tools/viewer/*.json`, `tools/viewer/data.js` (regenerated)
- Test: `tests/test_loft.py`, `tests/test_export_cases.py`

**Interfaces:**
- Consumes: `StripResult.ruling_verts` from Task 4.
- Produces: export key `"ruling_verts": [[a, b], ...]` (ints).

- [ ] **Step 1: Write the failing tests**

`tests/test_loft.py`, in `test_result_to_dict_is_json_serializable`:
```python
    assert back["ruling_verts"] == [list(p) for p in res.ruling_verts]
```
`tests/test_export_cases.py`, in `test_export_writes_data_js_and_json`:
```python
    assert len(cyl["ruling_verts"]) == len(cyl["rulings"])
```

- [ ] **Step 2: Run them to verify they fail**

Run: `uv run pytest tests/test_loft.py tests/test_export_cases.py -q -k "result_to_dict or export"`
Expected: FAIL with `KeyError: 'ruling_verts'`.

- [ ] **Step 3: Add `"ruling_verts"` to `result_to_dict`**, and draw viewer rulings from `c.ruling_verts`, falling back to `c.rulings.map(([i, j]) => [i, n + j])` when the key is missing.

- [ ] **Step 4: Regenerate the viewer cases and run the tests**

Run: `uv run python tools/viewer/export_cases.py && uv run pytest tests/test_loft.py tests/test_export_cases.py -q`
Expected: the JSON and data.js files are rewritten, and the tests pass.

- [ ] **Step 5: Commit**

```bash
git add mino/core/export.py tools/viewer tests
git commit -m "Export ruling vertices and draw viewer rulings from them"
```

### Task 6: Blender operators, input density and README

**Files:**
- Modify: `mino/blender/operator.py`, `mino/blender/inputs.py`, `mino/blender/remedies.py` (`PARAM_PROPS`, `MINO_OT_reloft`), `README.md`
- Test: `tests/test_blender.py`, `tests/test_blender_remedies.py`

**Interfaces:**
- Consumes: `LoftParams.adaptive`, `LoftParams.quads`.
- Produces:
  - `_bezier_rail(obj, samples, oversample=4)`, `curve_rail(obj, context, samples, oversample=4)`, `get_sections(context, samples, adaptive=0.0)`, where `oversample = 16 if adaptive > 0 else 4`.
  - Operator properties `adaptive` and `quads` on `MINO_OT_loft` and `MINO_OT_reloft`, copy per Global Constraints.

- [ ] **Step 1: Write the failing tests**

`tests/test_blender.py`:
```python
def test_loft_adaptive_densifies_bezier_input_and_stores_params(fresh_scene):
    a = _make_arc_curve("A", 1.0, 0.0)
    b = _make_arc_curve("B", 1.0, 1.0)
    _select([a, b], a)
    assert bpy.ops.mino.loft(samples=30, adaptive=0.5) == {"FINISHED"}
    obj = bpy.data.objects["Mino"]
    assert len(json.loads(obj["mino_rails"])["a"]) >= 30 * 16
    params = json.loads(obj["mino_params"])
    assert params["adaptive"] == 0.5 and params["quads"] is True
```

`tests/test_blender_remedies.py`:
```python
def _ellipse_loft(**props):
    case = CASES["ellipse"]()
    a = _make_poly_curve("A", case["points_a"])
    b = _make_poly_curve("B", case["points_b"])
    _select([a, b], a)
    assert bpy.ops.mino.loft(samples=60, window=2, **props) == {"FINISHED"}
    return bpy.data.objects["Mino"]


def test_quads_loft_has_only_quads(fresh_scene):
    obj = _ellipse_loft()
    assert all(len(p.vertices) == 4 for p in obj.data.polygons)


def test_grid_loft_keeps_fan_triangles(fresh_scene):
    obj = _ellipse_loft(quads=False)
    assert any(len(p.vertices) == 3 for p in obj.data.polygons)


def test_reloft_overrides_adaptive_and_quads(fresh_scene):
    _twisted_loft()
    assert bpy.ops.mino.reloft(adaptive=0.5, quads=False) == {"FINISHED"}
    params = json.loads(bpy.data.objects["Mino.reloft"]["mino_params"])
    assert params["adaptive"] == 0.5 and params["quads"] is False


def test_reloft_of_a_result_stored_without_new_params(fresh_scene):
    obj = _twisted_loft()
    stored = json.loads(obj["mino_params"])
    del stored["adaptive"], stored["quads"]
    obj["mino_params"] = json.dumps(stored)
    assert bpy.ops.mino.reloft() == {"FINISHED"}
    params = json.loads(bpy.data.objects["Mino.reloft"]["mino_params"])
    assert params["adaptive"] == 0.0 and params["quads"] is True
```
(`CASES` and `_select` are already imported or defined in that file. If they aren't, import them the way `_twisted_loft` uses them.)

- [ ] **Step 2: Run them to verify they fail**

Run: `uv run pytest tests/test_blender.py tests/test_blender_remedies.py -q -k "adaptive or quads or grid_loft or without_new"`
Expected: FAIL, `TypeError: Converting py args to operator properties: keyword "adaptive" unrecognized`.

- [ ] **Step 3: Implement**

- `MINO_OT_loft`: add both properties and pass them into `LoftParams`. Call `inputs.get_sections(context, params.samples, params.adaptive)`. In `draw`, put `adaptive` after `samples` and `quads` before `planarize`.
- `inputs.py`: thread `oversample` through `curve_rail` into `_bezier_rail`, so `res = max(2, math.ceil(samples * oversample / nseg) + 1)`.
- `remedies.py`: add both properties to `MINO_OT_reloft`, and add `"adaptive"` and `"quads"` to `PARAM_PROPS`.

- [ ] **Step 4: Update README.md, Parameters section**

Add two lines after Samples and before Planarize:
- "Adaptive: share of samples placed where the rails bend (0 = even spacing). 0.5–0.6 smooths curved outlines several times over for the same Samples; higher values thin samples on straight and gently curving parts. Window still counts samples, so it covers less distance where samples are dense. NURBS rails are read at the curve's own Resolution U, which bounds the detail."
- "Quads: give every ruling its own rail points so every face is a quad between two rulings; quads that cannot be made flat are still split into two triangles."

- [ ] **Step 5: Run the Blender tests, then the full suite**

Run: `uv run pytest tests/test_blender.py tests/test_blender_remedies.py tests/test_blender_relax.py -q`
Expected: all pass.

Run: `uv run pytest -q`
Expected: all pass.

- [ ] **Step 6: Commit**

```bash
git add mino/blender README.md tests
git commit -m "Add Adaptive and Quads to the loft and Re-loft operators"
```
