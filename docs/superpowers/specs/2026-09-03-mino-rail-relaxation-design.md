# Mino: rail relaxation. Design (spec 2 of 2)

Date: 2026-09-03
Status: approved design, pre-implementation. Amended 2026-09-04 after a throwaway solver spike (see section 7) and again after the final branch review (see section 8).
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
- `margin`: degrees below `twist_tolerance` that the solver aims for (default 0.5, min 0), so the re-loft lands strictly inside tolerance rather than on its edge.
- `iterations`: gradient steps (default 800; a pinned end couples its tangent to two neighbours through the second-order end formula, which roughly doubles the steps the mild case needs).
- `pin_endpoints`: keep `B[0]` and `B[N-1]` fixed (default True).
- `max_seconds`: wall-clock budget for the loop, 0 for no limit (default 0 in the core, 10 in the operator).
- `progress`: optional callback `progress(step, iterations)` invoked after every accepted step.

Variables: `δ ∈ R^N`, one scalar per B point, moving `B'[j] = B[j] + δ_j·n_j`
where `n_j` is the local strip normal at B: `normalize(T_B[j] × R)` with `R`
the ruling of the first path entry that uses `j`. Bounds `|δ_j| ≤ d` where
`d = max_move · mean ruling length`; pinned endpoints have `δ = 0`.

Tangents of `B'` are the supplied tangents plus the central-difference
change: `T_B'[j] = normalize(T_B[j] + c(B')[j] − c(B)[j])` where `c(P)` is
the normalized central difference of the points `P`. When nothing moves
this returns `T_B` exactly, so exact (Bezier) tangents are kept; when `T_B`
itself came from central differences it reduces to `c(B')`. Twist is
evaluated on the fixed pairing `path` (no DP inside the loop):
`twist_k(δ)` for each `(i, j)` in `path` using the twist formula from the
loft core on `A[i]`, `T_A[i]`, `B'[j]`, `T_B'[j]`.

Objective:
```
F(δ) = Σ_k max(0, twist_k(δ) − target)²  +  λ · Σ_j (δ_{j+1} − δ_j)² / ℓ²
```
with `target = max(0, params.twist_tolerance − margin)` in degrees and
`ℓ` the mean ruling length, so `λ` is dimensionless and the remedy behaves
the same at metre and millimetre scale.

Solver: projected gradient descent with central finite-difference
gradients (step `1e-4·d`; a gradient costs 2·(N−2) objective evaluations,
each vectorized O(N)). The step length is the Barzilai-Borwein estimate
`α = (s·s)/(s·y)` from the previous step `s = δ_new − δ_old` and gradient
change `y = g_new − g_old` (first step: `d / |g|`; if `s·y ≤ 0` reuse twice
the last accepted step). Each step is projected onto the bounds and pinned
entries are zeroed, then accepted only if `F` decreased; otherwise the step
is halved, at most 30 times. The sequence of `F` values is therefore
monotone. Stop when `F` reaches 0 (every ruling on the path is within
`target`), when the line search cannot find a decrease, when the relative
decrease of `F` over 20 steps is below `1e-6`, or at `iterations`.
At N = 60 with 800 iterations this takes about six seconds in numpy; the operator's time budget (section 3) bounds it.

Why not plain normalized-gradient descent: the twist depends on the
tangents of `B'`, which are second differences of `δ`, so the problem is
badly conditioned and a fixed-direction line search crawls (measured: 200
steps used a fifth of the allowed move and left 6.2° on the 8.5° case).
Why the margin: the hinge is flat at the tolerance, so the optimum sits
exactly on it and the re-loft reads a hair over (5.05° at a 5° tolerance).

After the loop, run the full `loft(A, B', T_A, T_B')` (DP included) with
the moved tangents defined above to get the final result; the DP may now
choose better rulings than `path`. Measuring the result with the same
tangent convention as the baseline means a developable strip relaxes to
itself with `twist_after == twist_before`.

API:
```python
@dataclass
class RelaxResult:
    points_b: np.ndarray       # moved rail B, N×3
    tangents_b: np.ndarray     # tangents of the moved rail B, N×3 (see above)
    delta: np.ndarray          # signed moves, N
    max_move_used: float       # max |delta|
    twist_before: float        # max twist on path before
    twist_after: float         # max twist of the re-loft
    result: StripResult        # loft(A, B')
    iterations_run: int
    stop_reason: str           # noop | converged | stalled | line_search | iterations | time

def relax_rail_b(points_a, points_b, tangents_a, tangents_b, params,
                 max_move=0.05, smoothness=1.0, margin=0.5, iterations=800,
                 pin_endpoints=True, max_seconds=0.0, progress=None) -> RelaxResult
```
`relax_rail_b` first runs `loft` to obtain the rails and `path`, then
optimizes, then re-lofts.

## 3. Blender layer

`mino.relax` operator, Object Mode, active object must carry
`mino_rails`. Properties: `max_move` 0..0.5 default 0.05, `smoothness`
0..10 default 1.0, `margin` 0..5 default 0.5, `iterations` 10..1000
default 800 (max 2000), `max_seconds` 0..120 default 10, `pin_endpoints`
default True, and `diagnose` default True like the other remedies. It drives
Blender's cursor progress from the solver's callback and, when the budget
stopped the loop, appends "stopped at the {max_seconds} s budget after N
steps" to its report. It:

1. Reads the stored rails and params.
2. Runs `relax_rail_b`.
3. Creates a POLY curve object `<name>.railB.relaxed` through `points_b` (world space, identity transform), plus a new loft object `<name>.relaxed` from A and B' with stored inputs and diagnosis like any Mino result. The stored inputs are the original A points and tangents, the moved B' points, and the moved B' tangents from `RelaxResult.tangents_b`, so a re-loft from the stored inputs reproduces the relaxed mesh exactly.
4. Reports: "Relaxed rail B: max twist {before:.1f}° → {after:.1f}°, largest move {m:.3g} ({pct:.0f}% of mean ruling)". If `twist_after` is still over tolerance it adds a WARNING suggesting a larger `max_move` or subdivide.

Panel: the Diagnosis box gains a row "Relax rail B (moves ≤ 5% of ruling)"
with a button whenever the stored diagnosis has failing ranges; it is rank 5
after the spec 1 suggestions and is appended by `diagnosis_rows` from the
stored diagnosis, not stored as a suggestion, so results made before this
feature also show it. The button runs `mino.relax` with its defaults.

## 4. Testing

Core:
- Gradient sanity: on a small random δ the finite-difference gradient of the smoothness term matches its analytic gradient `2λ·L δ` (L the 1D chain Laplacian over the N points, pinned rows zeroed) within 1e-6.
- Bounds and pins: every `|δ_j| ≤ d`, endpoints exactly 0 when pinned.
- Cylinder: relaxing a developable strip leaves `B` unchanged within 1e-9 (gradient is zero when no ruling exceeds tolerance) and `twist_after` equals `twist_before` within 1e-3 degrees (resampling round-off); the same holds at `samples = 8`, the operator minimum, where central-difference end tangents alone would invent about 7° of twist.
- Scale invariance: the mild case scaled by 1000 relaxes to the same twist and the same move fraction as at unit scale.
- Degenerate rulings count as 90° inside the objective: a hand-built path with one zero-length ruling gives `F = (90 − target)²`.
- Mildly twisted pair (the twisted case with `phi = 0.3·(t/4)²`, max twist about 8.5°): with `max_move = 0.15` the re-loft has no failing rulings. The spike measured `twist_after` about 4.6° and `max_move_used` about 0.065 with the default margin; the test asserts no failing rulings and `max_move_used < 0.1`, and records the achieved numbers as documentation. If tolerance is not reached, the implementer stops and reports the numbers rather than loosening the assertion.
- Full twisted case (max twist about 31°): `twist_after < twist_before` (the spike measured about 22° at `max_move = 0.15`), the objective decreases monotonically (line search guarantee), and `max_move_used` equals `max_move` within 1e-9 (the bound is active).

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
- Numeric gradients cost about 2N evaluations per step; at samples 60 and 800 iterations this is about six seconds, at samples 400 it would take minutes, so the operator's 10 s default budget bounds it and the cursor shows progress. The early stops (objective zero, stall) usually end it far sooner.

## 7. Spike record (2026-09-04)

A throwaway prototype of section 2 was run before planning. With the
original solver (normalized gradient, backtracking, 200 steps) the 8.5°
case stalled at 6.2°. Adam and Barzilai-Borwein steps both drove the
objective to zero within about 0.07 of the mean ruling length; BB keeps
the monotone guarantee so it was chosen. Moving B in full 3D instead of
along the strip normal did not help, so the scalar-per-point formulation
stands. The margin was added because the hinge optimum sits exactly on
the tolerance. Nothing from the spike is kept as code.

## 8. Final review amendments (2026-09-04)

The whole-branch review measured two defects in the section 2 formulation
as first written. The smoothness term carried units of length², so at
millimetre scale it swamped the hinge term and the mild case stayed at 7°;
dividing by the mean ruling length squared makes `λ` dimensionless. The
baseline twist used the caller's tangents while the objective and re-loft
used central differences, whose one-sided end formula invents about
90/(N−1) degrees on a curved rail; at `samples = 8` that moved a
developable Bezier-railed strip and then warned about it. Defining the
moved tangents as the supplied tangents plus the central-difference change
keeps both measurements on one convention.
