# Mino: rail relaxation. Design (spec 2 of 2)

Date: 2026-09-03
Status: approved design, pre-implementation
Builds on: spec 1 (`2026-09-03-mino-diagnosis-and-remedies-design.md`), which provides the stored inputs on result objects and the Diagnosis panel.

## 1. Goal

When subdividing or creasing is not acceptable, change the design
minimally instead: move rail B by a bounded amount so the loft between A
and the moved B is developable, and hand the moved rail back as a new curve
the user can accept or reject. Rail A is the fixed design edge.

## 2. Formulation (`mino/core/relax.py`)

Inputs: resampled rails `A`, `B` (N points each, tangents), the chosen
rulings `path = [(i, j), ...]` from the current loft, `LoftParams`, and:

- `max_move`: bound on each point's displacement, as a fraction of the mean ruling length (default 0.05).
- `smoothness`: weight λ of the smoothness term (default 1.0).
- `iterations`: gradient steps (default 200).
- `pin_endpoints`: keep `B[0]` and `B[N-1]` fixed (default True).

Variables: `δ ∈ R^N`, one scalar per B point, moving `B'[j] = B[j] + δ_j·n_j`
where `n_j` is the local strip normal at B: `normalize(T_B[j] × R)` with `R`
the ruling of the first path entry that uses `j`. Bounds `|δ_j| ≤ d` where
`d = max_move · mean ruling length`; pinned endpoints have `δ = 0`.

Tangents of `B'` are recomputed by central difference each evaluation.
Twist is evaluated on the fixed pairing `path` (no DP inside the loop):
`twist_k(δ)` for each `(i, j)` in `path` using the twist formula from the
loft core on `A[i]`, `T_A[i]`, `B'[j]`, `T_B'[j]`.

Objective:
```
F(δ) = Σ_k max(0, twist_k(δ) − tol)²  +  λ · Σ_j (δ_{j+1} − δ_j)²
```
with `tol = params.twist_tolerance` in degrees.

Solver: projected gradient descent with central finite-difference
gradients (step `1e-4·d`), backtracking line search (halve the step until
`F` decreases, at most 20 halvings), projection onto the bounds after each
step. Stop at `iterations` or when the relative decrease of `F` over 10
steps is below `1e-6`. Cost per gradient: 2N twist evaluations, each
vectorized O(N). At N = 60 this is well under a second in numpy.

After the loop, run the full `loft(A, B')` (DP included) to get the final
result; the DP may now choose better rulings than `path`.

API:
```python
@dataclass
class RelaxResult:
    points_b: np.ndarray       # moved rail B, N×3
    delta: np.ndarray          # signed moves, N
    max_move_used: float       # max |delta|
    twist_before: float        # max twist on path before
    twist_after: float         # max twist of the re-loft
    result: StripResult        # loft(A, B')
    iterations_run: int

def relax_rail_b(points_a, points_b, tangents_a, tangents_b, params,
                 max_move=0.05, smoothness=1.0, iterations=200,
                 pin_endpoints=True) -> RelaxResult
```
`relax_rail_b` first runs `loft` to obtain the rails and `path`, then
optimizes, then re-lofts.

## 3. Blender layer

`mino.relax` operator, Object Mode, active object must carry
`mino_rails`. Properties: `max_move` 0..0.5 default 0.05, `smoothness`
0..10 default 1.0, `iterations` 10..1000 default 200, `pin_endpoints`
default True. It:

1. Reads the stored rails and params.
2. Runs `relax_rail_b`.
3. Creates a POLY curve object `<name>.railB.relaxed` through `points_b` (world space, identity transform), plus a new loft object `<name>.relaxed` from A and B' with stored inputs and diagnosis like any Mino result.
4. Reports: "Relaxed rail B: max twist {before:.1f}° → {after:.1f}°, largest move {m:.3g} ({pct:.0f}% of mean ruling)". If `twist_after` is still over tolerance it adds a WARNING suggesting a larger `max_move` or subdivide.

Panel: the Diagnosis box gains a row "Relax rail B (moves ≤ {max_move})" with
a button whenever failing ranges exist; it is rank 5 after the spec 1
suggestions and is added by `diagnosis_rows` when the relax operator is
registered.

## 4. Testing

Core:
- Gradient sanity: on a small random δ the finite-difference gradient of the smoothness term matches its analytic gradient `2λ·L δ` (L the path Laplacian) within 1e-6.
- Bounds and pins: every `|δ_j| ≤ d`, endpoints exactly 0 when pinned.
- Cylinder: relaxing a developable strip leaves `B` unchanged within 1e-9 (gradient is zero when no ruling exceeds tolerance).
- Mildly twisted pair (the twisted case with `phi = 0.3·(t/4)²`, max twist about 8.5°): with `max_move = 0.15` the re-loft has no failing rulings. The implementer records the achieved `twist_after` and `max_move_used` in the test as documentation; if tolerance is not reached, the implementer stops and reports the numbers rather than loosening the assertion.
- Full twisted case: `twist_after < twist_before` and the objective decreases monotonically (line search guarantee).

Blender (headless bpy):
- `mino.relax` on a twisted pair creates the curve object and the relaxed loft object, both with the expected names, and the loft object carries `mino_rails`.
- `diagnosis_rows` includes the relax row when failing ranges exist.

## 5. Layout

```
mino/core/relax.py          RelaxResult, relax_rail_b, helpers
mino/blender/remedies.py    MINO_OT_relax added; diagnosis_rows gains the relax row
tests/test_relax.py tests/test_blender_relax.py
```

## 6. Open risks

- Local minima: gradient descent on a non-convex twist objective can stall. The smoothness term and small steps make the common bow-flare case behave; if it stalls, the report shows the residual twist honestly.
- Moving along the strip normal changes ruling lengths slightly; the DP re-loft absorbs that.
- Numeric gradients cost 2N evaluations per step; at samples 400 and 200 iterations this is a few seconds. Acceptable; the operator reports progress only at the end.
