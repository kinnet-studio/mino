# Mino Rail Relaxation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a "Relax rail B" remedy that moves rail B by a bounded amount along the strip normal so the loft becomes developable, and hands the moved rail back as a new curve object plus a new loft object.

**Architecture:** One new pure-numpy core module, `mino/core/relax.py`, holds the objective, a numeric gradient, and a projected Barzilai-Borwein descent over one scalar per B point. A new paired-twist helper in `mino/core/twist.py` evaluates twist on a fixed ruling pairing (the existing matrix function is refactored onto it). The Blender side adds `MINO_OT_relax` to `mino/blender/remedies.py`, a curve-object writer to `mino/blender/output.py`, and a "Relax rail B" row to `diagnosis_rows`.

**Tech Stack:** Python 3.11, numpy, pytest, bpy 4.5 wheel (headless tests). Run everything with `uv run --no-sync pytest -q` from the repo root (plain `uv run` re-syncs the venv; `--no-sync` only works inside the project directory).

**Spec:** `docs/superpowers/specs/2026-09-03-mino-rail-relaxation-design.md` (read section 2 for the formulation and section 7 for the spike numbers the tests rely on).

Deviations from the spec decided while planning: `RelaxResult` gains two fields, `objective` (the monotone list of objective values, needed by the monotonicity test) and `mean_ruling` (needed by the operator's percentage report); the core re-loft uses the original A points and tangents so the stored inputs on the relaxed object reproduce its mesh exactly; the mild test case is `CASES["twisted"]` with a new `scale` argument instead of a separate case; the operator report uses `->` instead of the arrow glyph.

## Global Constraints

- `mino/core/**` must never import `bpy` or `mathutils`. numpy and stdlib only.
- Vertex layout in a StripResult: rail A is indices `0..N-1`, rail B is `N..2N-1`.
- Stored custom properties on every result object, all strings: `mino_rails` (JSON `{"a","ta","b","tb"}`, tangents may be null), `mino_params`, `mino_report`, `mino_diagnosis` (`""` when diagnosis is off).
- Twist tolerance and margin are in degrees. `target = max(0, twist_tolerance - margin)`.
- Relaxation moves each B point by a scalar `delta[j]` along its strip normal; `|delta[j]| <= max_move * mean ruling length`; pinned endpoints have `delta = 0`.
- The objective sequence must be monotone non-increasing (accept a step only if the objective decreased).
- Remedy operators run in Object Mode only, read the active object's stored inputs, and create new objects; they never modify the source object.
- Operator defaults: `max_move` 0.05 (0..0.5), `smoothness` 1.0 (0..10), `margin` 0.5 (0..5), `iterations` 400 (10..1000), `pin_endpoints` True, `diagnose` True.
- New object names: `<name>.railB.relaxed` (POLY curve) and `<name>.relaxed` (loft result).
- Commit after every task with the two trailer lines:
  `Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>` and `Claude-Session: https://claude.ai/code/session_01Me1BZW7kSiuLCU5ebALpDX`.
- bpy tests use the `fresh_scene` fixture pattern: `bpy.ops.wm.read_factory_settings(use_empty=True)` then `mino.register()` (ignore `ValueError` if already registered).

Existing code you build on (read these files first when a task touches them): `mino/core/__init__.py` (`loft`), `mino/core/twist.py` (`twist_matrix`, `MIN_RULING_LENGTH`, `MIN_SIN`), `mino/core/rails.py` (`prepare_rails`, `central_difference`, `normalize_rows`), `mino/core/types.py`, `mino/core/strakes.py` (how a core module imports `loft`), `mino/blender/remedies.py`, `mino/blender/output.py`, `mino/blender/state.py`, `tests/cases.py`, `tests/test_blender_remedies.py`.

---

### Task 1: Paired twist helper

**Files:**
- Modify: `mino/core/twist.py`
- Test: `tests/test_twist.py`

**Interfaces:**
- Consumes: `MIN_RULING_LENGTH`, `MIN_SIN` from `mino/core/twist.py`.
- Produces: `paired_twist(points_a, tangents_a, points_b, tangents_b) -> np.ndarray` of shape `(K,)`, degrees in `[0, 90]`, `inf` where the ruling is shorter than `MIN_RULING_LENGTH` or within 1° of either tangent. Row `k` of each array pairs with row `k` of the others. `twist_matrix` keeps its signature and behaviour.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_twist.py`:

```python
from mino.core.twist import paired_twist


def test_paired_twist_matches_matrix_entries():
    rng = np.random.default_rng(0)
    pa = np.cumsum(rng.normal(size=(6, 3)), axis=0)
    pb = np.cumsum(rng.normal(size=(6, 3)), axis=0) + np.array([0.0, 0.0, 3.0])
    ta = np.diff(pa, axis=0, append=pa[-1:] + (pa[-1:] - pa[-2:-1]))
    tb = np.diff(pb, axis=0, append=pb[-1:] + (pb[-1:] - pb[-2:-1]))
    a = _rail(pa, ta / np.linalg.norm(ta, axis=1, keepdims=True))
    b = _rail(pb, tb / np.linalg.norm(tb, axis=1, keepdims=True))
    full = twist_matrix(a, b, window=5)
    pairs = [(0, 0), (1, 2), (3, 1), (5, 5)]
    i = np.array([p[0] for p in pairs])
    j = np.array([p[1] for p in pairs])
    got = paired_twist(a.points[i], a.tangents[i], b.points[j], b.tangents[j])
    assert np.allclose(got, full[i, j])


def test_paired_twist_flags_degenerate_rulings():
    pa = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    ta = np.array([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    pb = np.array([[0.0, 0.0, 0.0], [3.0, 0.0, 0.0]])   # zero-length, then parallel to tangent
    tb = np.array([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    got = paired_twist(pa, ta, pb, tb)
    assert np.isinf(got).all()
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run --no-sync pytest tests/test_twist.py -q`
Expected: ImportError, `cannot import name 'paired_twist'`.

- [ ] **Step 3: Implement `paired_twist` and refactor `twist_matrix` onto it**

Replace the body of `mino/core/twist.py` from `def twist_matrix` to the end with:

```python
def paired_twist(points_a, tangents_a, points_b, tangents_b) -> np.ndarray:
    """Twist in degrees for K rulings pairing row k of A with row k of B.

    twist = angle between n_A = T_A x R and n_B = T_B x R, folded into [0, 90].
    inf when the ruling is degenerate or nearly parallel to a tangent.
    """
    R = np.asarray(points_b, dtype=float) - np.asarray(points_a, dtype=float)
    L = np.linalg.norm(R, axis=1)
    Rn = R / np.where(L < MIN_RULING_LENGTH, 1.0, L)[:, None]
    nA = np.cross(np.asarray(tangents_a, dtype=float), Rn)
    nB = np.cross(np.asarray(tangents_b, dtype=float), Rn)
    lA = np.linalg.norm(nA, axis=1)
    lB = np.linalg.norm(nB, axis=1)
    bad_angle = (lA < MIN_SIN) | (lB < MIN_SIN)
    denom = np.where(bad_angle, 1.0, lA * lB)
    cosang = np.abs(np.einsum("ij,ij->i", nA, nB)) / denom
    twist = np.degrees(np.arccos(np.clip(cosang, 0.0, 1.0)))
    twist[(L < MIN_RULING_LENGTH) | bad_angle] = np.inf
    return twist


def twist_matrix(rail_a: Rail, rail_b: Rail, window: int) -> np.ndarray:
    """Twist angle in degrees for every (i, j) ruling from A[i] to B[j].

    Entries are inf when the ruling is degenerate, nearly parallel to a tangent,
    or outside the band |i - j| <= window.
    """
    na, nb = len(rail_a.points), len(rail_b.points)
    ii, jj = np.meshgrid(np.arange(na), np.arange(nb), indexing="ij")
    flat = paired_twist(rail_a.points[ii.ravel()], rail_a.tangents[ii.ravel()],
                        rail_b.points[jj.ravel()], rail_b.tangents[jj.ravel()])
    twist = flat.reshape(na, nb)
    twist[np.abs(ii - jj) > window] = np.inf
    return twist
```

Keep `ruling_lengths`, `MIN_RULING_LENGTH` and `MIN_SIN` as they are.

- [ ] **Step 4: Run the whole suite to verify the refactor changed nothing**

Run: `uv run --no-sync pytest -q`
Expected: all tests pass (the loft, diagnose, strakes and viewer-export tests exercise `twist_matrix`).

- [ ] **Step 5: Commit**

```bash
git add mino/core/twist.py tests/test_twist.py
git commit -m "Add paired_twist and build twist_matrix on it

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01Me1BZW7kSiuLCU5ebALpDX"
```

---

### Task 2: Relaxation objective, normals and numeric gradient

**Files:**
- Create: `mino/core/relax.py`
- Modify: `tests/cases.py` (`twisted` gains `scale`)
- Test: `tests/test_relax.py`

**Interfaces:**
- Consumes: `paired_twist` (Task 1), `central_difference`, `normalize_rows`, `prepare_rails` from `mino/core/rails.py`, `Rail` from `mino/core/types.py`.
- Produces:
  - `strip_normals_b(rail_a: Rail, rail_b: Rail, path) -> np.ndarray` of shape `(N, 3)`, unit normals; for each B index `j` the normal is `normalize(T_B[j] x (B[j] - A[i]))` for the first `(i, j)` in `path` that uses `j`.
  - `relax_objective(delta, rail_a, rail_b, normals, path, target, smoothness) -> tuple[float, np.ndarray]`: `(F, twists)` where `twists` is the paired twist on `path` for the moved rail (may contain `inf`), and `F = sum(max(0, t - target)^2) + smoothness * sum(diff(delta)^2)` with `inf` twists counted as 90°.
  - `numeric_gradient(f, x, free, h) -> np.ndarray`: central differences with step `h` on the indices where `free` is True, zero elsewhere.
  - `CASES["twisted"](n=200, scale=1.2)`: `scale=0.3` is the mild case (max twist about 8.5°).

- [ ] **Step 1: Give the twisted case a scale**

In `tests/cases.py` change `twisted` to:

```python
def twisted(n=200, scale=1.2):
    """B is a slowly rolling copy of A; scale=0.3 is the mild case (max twist about 8.5 deg)."""
    t = np.linspace(0.0, 4.0, n)
    phi = scale * (t / 4.0) ** 2
    dphi = 2.0 * scale * t / 16.0
    a = np.column_stack([t, np.zeros(n), np.zeros(n)])
    ta = np.column_stack([np.ones(n), np.zeros(n), np.zeros(n)])
    b = np.column_stack([t, np.sin(phi), np.cos(phi)])
    tb = np.column_stack([np.ones(n), np.cos(phi) * dphi, -np.sin(phi) * dphi])
    return dict(points_a=a, points_b=b, tangents_a=ta, tangents_b=tb, params={})
```

`2.0 * 1.2 * t / 16.0` equals the old `0.15 * t`, so the default case is unchanged. Run `uv run --no-sync pytest -q` and confirm everything still passes.

- [ ] **Step 2: Write the failing tests**

Create `tests/test_relax.py`:

```python
import numpy as np
import pytest

from mino.core import LoftParams, loft
from mino.core.rails import prepare_rails
from mino.core.relax import numeric_gradient, relax_objective, strip_normals_b
from tests.cases import CASES


def _rails_and_path(name, **case_kwargs):
    case = CASES[name](**case_kwargs)
    params = LoftParams(**case["params"])
    res = loft(case["points_a"], case["points_b"], params, case["tangents_a"], case["tangents_b"])
    ra, rb = prepare_rails(case["points_a"], case["points_b"], params.samples,
                           case["tangents_a"], case["tangents_b"])
    return case, params, res, ra, rb, np.asarray(res.rulings, dtype=int)


def test_strip_normals_are_unit_and_perpendicular_to_tangent_and_ruling():
    case, params, res, ra, rb, path = _rails_and_path("twisted")
    normals = strip_normals_b(ra, rb, path)
    assert normals.shape == rb.points.shape
    assert np.allclose(np.linalg.norm(normals, axis=1), 1.0)
    first = {}
    for i, j in path:
        first.setdefault(int(j), int(i))
    for j, i in first.items():
        r = rb.points[j] - ra.points[i]
        assert abs(np.dot(normals[j], rb.tangents[j])) < 1e-9
        assert abs(np.dot(normals[j], r)) < 1e-9


def test_objective_is_zero_when_within_target_and_unmoved():
    case, params, res, ra, rb, path = _rails_and_path("cylinder")
    normals = strip_normals_b(ra, rb, path)
    delta = np.zeros(len(rb.points))
    f, twists = relax_objective(delta, ra, rb, normals, path, target=5.0, smoothness=1.0)
    assert f == 0.0
    assert np.allclose(twists, res.ruling_twist)


def test_objective_counts_twist_over_target_squared():
    case, params, res, ra, rb, path = _rails_and_path("twisted")
    normals = strip_normals_b(ra, rb, path)
    delta = np.zeros(len(rb.points))
    f, twists = relax_objective(delta, ra, rb, normals, path, target=4.5, smoothness=1.0)
    expected = float((np.maximum(0.0, res.ruling_twist - 4.5) ** 2).sum())
    assert np.isclose(f, expected)


def test_smoothness_gradient_matches_chain_laplacian():
    case, params, res, ra, rb, path = _rails_and_path("cylinder")
    normals = strip_normals_b(ra, rb, path)
    n = len(rb.points)
    rng = np.random.default_rng(1)
    delta = 1e-3 * rng.normal(size=n)
    lam = 2.5
    # target far above any twist: only the smoothness term is active
    f = lambda d: relax_objective(d, ra, rb, normals, path, target=1e6, smoothness=lam)[0]
    free = np.ones(n, dtype=bool)
    free[0] = free[-1] = False
    g = numeric_gradient(f, delta, free, h=1e-7)
    lap = np.zeros(n)
    lap[1:-1] = 2 * delta[1:-1] - delta[:-2] - delta[2:]
    lap[0] = delta[0] - delta[1]
    lap[-1] = delta[-1] - delta[-2]
    analytic = 2.0 * lam * lap
    analytic[~free] = 0.0
    assert np.allclose(g, analytic, atol=1e-6)
    assert g[0] == 0.0 and g[-1] == 0.0


def test_numeric_gradient_on_quadratic():
    f = lambda x: float((x ** 2).sum())
    x = np.array([1.0, -2.0, 0.5])
    g = numeric_gradient(f, x, np.array([True, False, True]), h=1e-6)
    assert np.allclose(g, [2.0, 0.0, 1.0], atol=1e-6)
```

- [ ] **Step 3: Run the tests to verify they fail**

Run: `uv run --no-sync pytest tests/test_relax.py -q`
Expected: ModuleNotFoundError, `No module named 'mino.core.relax'`.

- [ ] **Step 4: Implement the helpers**

Create `mino/core/relax.py`:

```python
"""Rail relaxation: move rail B a bounded amount along the strip normal until the loft is developable."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .rails import central_difference, normalize_rows
from .twist import paired_twist
from .types import Rail

INF_TWIST = 90.0  # degrees counted for an invalid ruling inside the objective


def strip_normals_b(rail_a: Rail, rail_b: Rail, path) -> np.ndarray:
    """Unit normal per B point: T_B[j] x R for the first ruling in `path` that uses j."""
    n = len(rail_b.points)
    normals = np.zeros((n, 3))
    seen = np.zeros(n, dtype=bool)
    for i, j in path:
        if not seen[j]:
            seen[j] = True
            r = rail_b.points[j] - rail_a.points[i]
            normals[j] = normalize_rows(np.cross(rail_b.tangents[j], r))
    return normals


def relax_objective(delta, rail_a: Rail, rail_b: Rail, normals, path, target: float, smoothness: float):
    """(F, twists) for rail B moved by delta along `normals`, twist evaluated on the fixed `path`."""
    path = np.asarray(path, dtype=int)
    i, j = path[:, 0], path[:, 1]
    moved = rail_b.points + np.asarray(delta, dtype=float)[:, None] * normals
    tangents = normalize_rows(central_difference(moved))
    twists = paired_twist(rail_a.points[i], rail_a.tangents[i], moved[j], tangents[j])
    finite = np.where(np.isfinite(twists), twists, INF_TWIST)
    hinge = np.maximum(0.0, finite - target)
    smooth = float((np.diff(delta) ** 2).sum())
    return float((hinge ** 2).sum() + smoothness * smooth), twists


def numeric_gradient(f, x, free, h: float) -> np.ndarray:
    """Central-difference gradient of scalar f at x on the indices where `free` is True."""
    x = np.asarray(x, dtype=float)
    g = np.zeros_like(x)
    for k in np.flatnonzero(free):
        e = np.zeros_like(x)
        e[k] = h
        g[k] = (f(x + e) - f(x - e)) / (2.0 * h)
    return g
```

(`dataclass` and `field` are imported now because Task 3 adds `RelaxResult` to this file.)

- [ ] **Step 5: Run the tests to verify they pass**

Run: `uv run --no-sync pytest tests/test_relax.py tests/test_loft.py -q`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add mino/core/relax.py tests/test_relax.py tests/cases.py
git commit -m "Add relaxation objective, strip normals and numeric gradient

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01Me1BZW7kSiuLCU5ebALpDX"
```

---

### Task 3: Projected Barzilai-Borwein solver, `relax_rail_b`

**Files:**
- Modify: `mino/core/relax.py`
- Test: `tests/test_relax.py`

**Interfaces:**
- Consumes: `strip_normals_b`, `relax_objective`, `numeric_gradient` (Task 2); `loft` from `mino/core/__init__.py`; `prepare_rails`.
- Produces:

```python
@dataclass
class RelaxResult:
    points_b: np.ndarray       # moved rail B, (N, 3), in the resampled order used by the loft
    delta: np.ndarray          # signed moves along the strip normal, (N,)
    max_move_used: float       # max |delta|, world units
    mean_ruling: float         # mean ruling length of the base loft, world units
    twist_before: float        # max twist of the base loft
    twist_after: float         # max twist of the re-loft
    result: StripResult        # loft(points_a, points_b) with the original A
    iterations_run: int
    objective: list            # objective value after each accepted step, starts with the initial value

def relax_rail_b(points_a, points_b, tangents_a, tangents_b, params,
                 max_move=0.05, smoothness=1.0, margin=0.5, iterations=400,
                 pin_endpoints=True) -> RelaxResult
```

`relax_rail_b(...).result` equals `loft(points_a, relax.points_b, params, tangents_a, None)`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_relax.py`:

```python
from mino.core.relax import RelaxResult, relax_rail_b


def _relax(name, scale=None, **kwargs):
    case = CASES[name]() if scale is None else CASES[name](scale=scale)
    params = LoftParams(**case["params"])
    return case, params, relax_rail_b(case["points_a"], case["points_b"], case["tangents_a"],
                                      case["tangents_b"], params, **kwargs)


def test_developable_strip_is_left_alone():
    case, params, r = _relax("cylinder")
    assert isinstance(r, RelaxResult)
    assert r.max_move_used < 1e-9
    assert r.iterations_run == 0
    ra, rb = prepare_rails(case["points_a"], case["points_b"], params.samples,
                           case["tangents_a"], case["tangents_b"])
    assert np.allclose(r.points_b, rb.points, atol=1e-9)
    # the re-loft derives B tangents by central difference, so twist can differ slightly
    assert r.result.failing_ranges == []
    assert abs(r.twist_after - r.twist_before) < 1.0


def test_bounds_and_pins_are_respected():
    case, params, r = _relax("twisted", max_move=0.05)
    bound = 0.05 * r.mean_ruling
    assert np.all(np.abs(r.delta) <= bound + 1e-12)
    assert r.delta[0] == 0.0 and r.delta[-1] == 0.0
    assert np.isclose(r.max_move_used, bound, atol=1e-9)   # the bound is active on this case
    ra, rb = prepare_rails(case["points_a"], case["points_b"], params.samples,
                           case["tangents_a"], case["tangents_b"])
    assert np.allclose(np.linalg.norm(r.points_b - rb.points, axis=1), np.abs(r.delta), atol=1e-9)


def test_unpinned_endpoints_may_move():
    case, params, r = _relax("twisted", max_move=0.05, pin_endpoints=False)
    assert abs(r.delta[0]) > 0.0 or abs(r.delta[-1]) > 0.0


def test_mild_case_reaches_tolerance_within_bound():
    # Spike (spec section 7): twist_after about 4.6 deg, max_move_used about 0.065 of the mean ruling.
    case, params, r = _relax("twisted", scale=0.3, max_move=0.15)
    assert r.twist_before > params.twist_tolerance
    assert r.result.failing_ranges == []
    assert r.twist_after <= params.twist_tolerance
    assert r.max_move_used / r.mean_ruling < 0.1
    assert r.objective[-1] == 0.0


def test_full_twisted_case_improves_monotonically():
    case, params, r = _relax("twisted", max_move=0.15)
    assert r.twist_after < r.twist_before
    assert r.twist_before > 25.0                     # about 31 deg; documents the input
    assert r.twist_after < 25.0                      # spike measured about 22 deg
    assert all(b <= a for a, b in zip(r.objective, r.objective[1:]))
    assert 0 < r.iterations_run <= 400
    assert np.isclose(r.max_move_used, 0.15 * r.mean_ruling, atol=1e-9)


def test_result_matches_reloft_of_returned_rail():
    case, params, r = _relax("twisted", scale=0.3, max_move=0.15)
    again = loft(case["points_a"], r.points_b, params, case["tangents_a"], None)
    assert np.allclose(again.verts, r.result.verts)
    assert again.rulings == r.result.rulings


def test_zero_max_move_is_a_noop():
    case, params, r = _relax("twisted", max_move=0.0)
    assert r.iterations_run == 0 and r.max_move_used == 0.0
    assert abs(r.twist_after - r.twist_before) < 1.0
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run --no-sync pytest tests/test_relax.py -q`
Expected: ImportError, `cannot import name 'RelaxResult'`.

- [ ] **Step 3: Implement the solver**

Add to `mino/core/relax.py`. Import `loft` and `prepare_rails` at the top of the file the way `mino/core/strakes.py` does:

```python
from . import loft
from .rails import central_difference, normalize_rows, prepare_rails
from .types import LoftParams, Rail, StripResult
```

Then append:

```python
STALL_WINDOW = 20      # steps over which a relative decrease below STALL_REL counts as a stall
STALL_REL = 1e-6
MAX_HALVINGS = 30


@dataclass
class RelaxResult:
    points_b: np.ndarray
    delta: np.ndarray
    max_move_used: float
    mean_ruling: float
    twist_before: float
    twist_after: float
    result: StripResult
    iterations_run: int
    objective: list = field(default_factory=list)


def relax_rail_b(points_a, points_b, tangents_a, tangents_b, params: LoftParams | None = None,
                 max_move: float = 0.05, smoothness: float = 1.0, margin: float = 0.5,
                 iterations: int = 400, pin_endpoints: bool = True) -> RelaxResult:
    """Move rail B along the strip normal, bounded by max_move x mean ruling, to reduce twist."""
    params = params or LoftParams()
    base = loft(points_a, points_b, params, tangents_a, tangents_b)
    rail_a, rail_b = prepare_rails(points_a, points_b, params.samples, tangents_a, tangents_b)
    path = np.asarray(base.rulings, dtype=int)
    n = len(rail_b.points)

    lengths = np.linalg.norm(rail_b.points[path[:, 1]] - rail_a.points[path[:, 0]], axis=1)
    mean_ruling = float(lengths.mean())
    bound = max(0.0, float(max_move)) * mean_ruling
    normals = strip_normals_b(rail_a, rail_b, path)
    target = max(0.0, params.twist_tolerance - margin)

    free = np.ones(n, dtype=bool)
    if pin_endpoints:
        free[0] = free[-1] = False

    def objective(d):
        return relax_objective(d, rail_a, rail_b, normals, path, target, smoothness)[0]

    def project(d):
        d = np.clip(d, -bound, bound)
        d[~free] = 0.0
        return d

    delta = np.zeros(n)
    history = [objective(delta)]
    iterations_run = 0

    if bound > 0.0 and history[0] > 0.0:
        h = 1e-4 * bound
        grad = numeric_gradient(objective, delta, free, h)
        alpha = bound / max(float(np.linalg.norm(grad)), 1e-12)
        for it in range(iterations):
            if history[-1] == 0.0 or not np.any(grad):
                break
            step, accepted = alpha, None
            for _ in range(MAX_HALVINGS):
                cand = project(delta - step * grad)
                fc = objective(cand)
                if fc < history[-1]:
                    accepted = (cand, fc)
                    break
                step *= 0.5
            if accepted is None:
                break
            cand, fc = accepted
            new_grad = numeric_gradient(objective, cand, free, h)
            s, y = cand - delta, new_grad - grad
            sy = float(s @ y)
            alpha = float(s @ s) / sy if sy > 1e-18 else 2.0 * step
            delta, grad = cand, new_grad
            history.append(fc)
            iterations_run = it + 1
            if len(history) > STALL_WINDOW:
                before = history[-1 - STALL_WINDOW]
                if (before - history[-1]) / max(before, 1e-12) < STALL_REL:
                    break

    moved = rail_b.points + delta[:, None] * normals
    result = loft(points_a, moved, params, tangents_a, None)
    return RelaxResult(
        points_b=moved, delta=delta, max_move_used=float(np.abs(delta).max()),
        mean_ruling=mean_ruling, twist_before=base.report.max_twist,
        twist_after=result.report.max_twist, result=result,
        iterations_run=iterations_run, objective=history,
    )
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run --no-sync pytest tests/test_relax.py -q`
Expected: PASS. The mild and full twisted tests take about two seconds each.

If `test_mild_case_reaches_tolerance_within_bound` fails, do NOT loosen the assertion. Print `r.twist_after`, `r.max_move_used / r.mean_ruling`, `r.iterations_run` and `r.objective[-1]`, stop, and report those numbers (spec section 4).

- [ ] **Step 5: Run the full suite**

Run: `uv run --no-sync pytest -q`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add mino/core/relax.py tests/test_relax.py
git commit -m "Add relax_rail_b: projected Barzilai-Borwein descent on rail B

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01Me1BZW7kSiuLCU5ebALpDX"
```

---

### Task 4: `mino.relax` operator, curve output, panel row and README

**Files:**
- Modify: `mino/blender/output.py` (add `create_curve_object`)
- Modify: `mino/blender/remedies.py` (add `MINO_OT_relax`, extend `diagnosis_rows` and `classes`)
- Modify: `tests/test_blender_remedies.py` (`test_diagnosis_rows_maps_kinds_to_operators` expects the relax row)
- Modify: `README.md` (remedy list)
- Test: `tests/test_blender_relax.py`

**Interfaces:**
- Consumes: `relax_rail_b`, `RelaxResult` (Task 3); `load_inputs` from `mino/blender/state.py`; `output.create_result_object(context, name, points_a, tangents_a, points_b, tangents_b, params, result, diagnose_flag)`; `_MinoRemedy` poll mixin in `remedies.py`.
- Produces: `create_curve_object(context, name, points) -> bpy.types.Object` (POLY spline, 3D, world space, not selected); operator `mino.relax` with properties `max_move`, `smoothness`, `margin`, `iterations`, `pin_endpoints`, `diagnose`; `diagnosis_rows` appends `("Relax rail B (moves ≤ 5% of ruling)", "mino.relax", {})` when the stored diagnosis has failing ranges.

- [ ] **Step 1: Update the existing `diagnosis_rows` test**

In `tests/test_blender_remedies.py`, inside `test_diagnosis_rows_maps_kinds_to_operators`, change:

```python
    assert [r[1] for r in rows] == ["mino.reloft", "mino.subdivide", "mino.dart", None]
```

to:

```python
    assert [r[1] for r in rows] == ["mino.reloft", "mino.subdivide", "mino.dart", None, "mino.relax"]
    assert rows[-1][0].startswith("Relax rail B") and rows[-1][2] == {}
```

and append a second check after the `diagnosis_rows({}) == ...` line:

```python
    ok = {**diag, "failing_ranges": [], "suggestions": [{"kind": "ok", "text": "o", "params": {}, "rank": 0}]}
    assert [r[1] for r in diagnosis_rows({"mino_diagnosis": json.dumps(ok)})] == [None]
```

- [ ] **Step 2: Write the failing Blender tests**

Create `tests/test_blender_relax.py`:

```python
import json

import numpy as np
import pytest

bpy = pytest.importorskip("bpy")
from mathutils import Matrix  # noqa: E402

import mino  # noqa: E402
from mino.core.rails import prepare_rails  # noqa: E402
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


def _loft(case, samples=40):
    a = _make_poly_curve("A", case["points_a"])
    b = _make_poly_curve("B", case["points_b"])
    _select([a, b], a)
    assert bpy.ops.mino.loft(samples=samples) == {"FINISHED"}
    obj = bpy.data.objects["Mino"]
    _select([obj], obj)
    return obj


def test_relax_creates_curve_and_loft_objects(fresh_scene):
    obj = _loft(CASES["twisted"](n=60, scale=0.3))
    assert bpy.ops.mino.relax(max_move=0.15) == {"FINISHED"}

    curve = bpy.data.objects["Mino.railB.relaxed"]
    assert curve.type == "CURVE"
    spline = curve.data.splines[0]
    assert spline.type == "POLY" and len(spline.points) == 40
    assert curve.matrix_world == Matrix.Identity(4)

    new = bpy.data.objects["Mino.relaxed"]
    rails = json.loads(new["mino_rails"])
    assert len(rails["b"]) == 40 and rails["tb"] is None
    assert np.allclose(np.array(rails["a"]), np.array(json.loads(obj["mino_rails"])["a"]))
    curve_pts = np.array([p.co[:3] for p in spline.points])
    assert np.allclose(curve_pts, np.array(rails["b"]))
    assert json.loads(new["mino_params"]) == json.loads(obj["mino_params"])
    assert "mino_diagnosis" in new and new["mino_diagnosis"] != ""
    assert bpy.context.view_layer.objects.active is new

    stored = json.loads(obj["mino_rails"])
    _, rb = prepare_rails(stored["a"], stored["b"], 40, stored["ta"], stored["tb"])
    assert not np.allclose(curve_pts, rb.points)          # the rail actually moved
    assert np.linalg.norm(curve_pts - rb.points, axis=1).max() < 0.15 * 1.05  # mean ruling is about 1


def test_relax_leaves_developable_rail_in_place(fresh_scene):
    obj = _loft(CASES["cylinder"](n=60))
    assert bpy.ops.mino.relax() == {"FINISHED"}
    new = bpy.data.objects["Mino.relaxed"]
    assert "all within tolerance" in new["mino_report"]
    curve_pts = np.array([p.co[:3] for p in bpy.data.objects["Mino.railB.relaxed"].data.splines[0].points])
    stored = json.loads(obj["mino_rails"])
    _, rb = prepare_rails(stored["a"], stored["b"], 40, stored["ta"], stored["tb"])
    assert np.allclose(curve_pts, rb.points, atol=1e-9)   # nothing moved


def test_relax_can_skip_diagnosis(fresh_scene):
    obj = _loft(CASES["twisted"](n=60))
    assert bpy.ops.mino.relax(diagnose=False) == {"FINISHED"}
    assert bpy.data.objects["Mino.relaxed"]["mino_diagnosis"] == ""


def test_relax_requires_a_mino_object(fresh_scene):
    case = CASES["cylinder"](n=30)
    a = _make_poly_curve("A", case["points_a"])
    _select([a], a)
    assert not bpy.ops.mino.relax.poll()
```

- [ ] **Step 3: Run the tests to verify they fail**

Run: `uv run --no-sync pytest tests/test_blender_relax.py tests/test_blender_remedies.py -q`
Expected: the relax tests fail with `AttributeError: ... has no attribute "relax"` (or a poll failure), and `test_diagnosis_rows_maps_kinds_to_operators` fails on the list comparison.

- [ ] **Step 4: Add `create_curve_object` to `mino/blender/output.py`**

Append:

```python
def create_curve_object(context, name, points):
    """POLY curve through `points` (world space, identity transform). Linked, not selected."""
    import bpy

    pts = np.asarray(points, dtype=float).reshape(-1, 3)
    cu = bpy.data.curves.new(name, "CURVE")
    cu.dimensions = "3D"
    sp = cu.splines.new("POLY")
    sp.points.add(len(pts) - 1)
    sp.points.foreach_set("co", np.column_stack([pts, np.ones(len(pts))]).ravel())
    obj = bpy.data.objects.new(name, cu)
    context.collection.objects.link(obj)
    return obj
```

- [ ] **Step 5: Add the operator and the panel row to `mino/blender/remedies.py`**

Add the import near the other core imports:

```python
from ..core.relax import relax_rail_b
```

In `diagnosis_rows`, after the `for s in d.suggestions:` loop and before `return rows`, add:

```python
    if d.failing_ranges:
        rows.append(("Relax rail B (moves ≤ 5% of ruling)", "mino.relax", {}))
```

Add the operator class after `MINO_OT_dart`:

```python
class MINO_OT_relax(_MinoRemedy, bpy.types.Operator):
    """Move rail B a bounded amount so the loft becomes developable, and return the moved rail as a curve"""
    bl_idname = "mino.relax"
    bl_label = "Relax Rail B"

    max_move: FloatProperty(name="Max Move", default=0.05, min=0.0, max=0.5,
                            description="Largest move of any rail B point, as a fraction of the mean ruling length")
    smoothness: FloatProperty(name="Smoothness", default=1.0, min=0.0, max=10.0,
                              description="Weight of the term that keeps neighbouring moves similar")
    margin: FloatProperty(name="Margin", default=0.5, min=0.0, max=5.0,
                          description="Degrees below Twist Tolerance the solver aims for")
    iterations: IntProperty(name="Iterations", default=400, min=10, max=1000)
    pin_endpoints: BoolProperty(name="Pin Endpoints", default=True,
                                description="Keep the two ends of rail B where they are")
    diagnose: BoolProperty(name="Diagnose", default=True,
                           description="Store ranked suggestions for non-developable regions on the result")

    def execute(self, context):
        obj = context.active_object
        try:
            pa, ta, pb, tb, params = load_inputs(obj)
            res = relax_rail_b(pa, pb, ta, tb, params, max_move=self.max_move, smoothness=self.smoothness,
                               margin=self.margin, iterations=self.iterations, pin_endpoints=self.pin_endpoints)
        except LoftError as exc:
            self.report({"ERROR"}, str(exc))
            return {"CANCELLED"}
        output.create_curve_object(context, f"{obj.name}.railB.relaxed", res.points_b)
        output.create_result_object(context, f"{obj.name}.relaxed", pa, ta, res.points_b, None, params,
                                    res.result, self.diagnose)
        pct = 100.0 * res.max_move_used / res.mean_ruling if res.mean_ruling > 0 else 0.0
        self.report({"INFO"}, f"Mino: relaxed rail B, max twist {res.twist_before:.1f} -> {res.twist_after:.1f} deg, "
                              f"largest move {res.max_move_used:.3g} ({pct:.0f}% of mean ruling)")
        if res.result.failing_ranges:
            self.report({"WARNING"}, "Mino: still over tolerance; raise Max Move or subdivide into strakes")
        return {"FINISHED"}
```

Update the registration tuple at the bottom:

```python
classes = (MINO_OT_reloft, MINO_OT_subdivide, MINO_OT_dart, MINO_OT_relax)
```

- [ ] **Step 6: Run the Blender tests to verify they pass**

Run: `uv run --no-sync pytest tests/test_blender_relax.py tests/test_blender_remedies.py -q`
Expected: PASS.

- [ ] **Step 7: Update the README**

In `README.md`, in the list under "## When a loft is not developable", add after the "Cut a gusset (or dart)" bullet:

```markdown
- Relax rail B: move rail B by at most Max Move (a fraction of the mean
  ruling length) along the strip so the loft becomes developable. You get
  the moved rail as a new curve `<name>.railB.relaxed` to accept or reject,
  plus the re-lofted strip `<name>.relaxed`. If the twist is still over
  tolerance, raise Max Move or subdivide instead.
```

- [ ] **Step 8: Run the full suite and build the zip**

Run: `uv run --no-sync pytest -q && uv run --no-sync python make_zip.py`
Expected: all tests pass; `dist/mino-0.1.0.zip` is rebuilt (the `test_make_zip` test already checks the zip contents include every module under `mino/`, so the new file is covered).

- [ ] **Step 9: Commit**

```bash
git add mino/blender/output.py mino/blender/remedies.py tests/test_blender_relax.py tests/test_blender_remedies.py README.md
git commit -m "Add the Relax Rail B remedy operator and panel row

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01Me1BZW7kSiuLCU5ebALpDX"
```

---

## Self-review notes

Spec coverage: section 2 formulation is Tasks 2 and 3 (normals, objective with target and smoothness, BB solver with projection, halving, stall and zero stops, re-loft); section 3 operator, properties, object names, report and warning, panel row are Task 4; section 4 tests map one-to-one (gradient sanity, bounds and pins, cylinder unchanged, mild case, full twisted case, Blender object creation, `diagnosis_rows` row). Section 5 layout matches the files above. The paired-twist helper in Task 1 is the "twist formula from the loft core" the spec refers to.

Type consistency: `relax_rail_b` returns `RelaxResult` with fields `points_b, delta, max_move_used, mean_ruling, twist_before, twist_after, result, iterations_run, objective`; Task 4 reads `points_b, result, twist_before, twist_after, max_move_used, mean_ruling`. `create_result_object` is called with `tangents_b=None` for the relaxed object, matching the spec's "no B tangents".
