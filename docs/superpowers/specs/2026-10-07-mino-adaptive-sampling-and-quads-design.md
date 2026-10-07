# Mino: adaptive sampling and quad strips. Design

Date: 2026-10-07
Status: approved design, pre-implementation. Amended 2026-10-07 after calibrating tangents and outline error (section 3.7): tangents keep today's formula, so `Rail` and relaxation are unchanged.
Builds on: the PoC spec (`2026-09-03-devloft-poc-design.md`) for the loft pipeline, and spec 1 (`2026-09-03-mino-diagnosis-and-remedies-design.md`) for stored inputs and remedies.

## 1. Goal

Papercraft is the main use. Two changes to how rail samples become mesh
vertices:

1. **Adaptive sampling**: place more of the Samples where the rails bend
   and fewer on straight parts, so the strip outline is smoother for the
   same vertex count.
2. **Quad strips**: every face is a quad between two consecutive rulings.
   Triangles remain only where a quad cannot be made planar (the strip is
   not developable there), which is where paper needs a fold anyway.

Both are opt-in in effect: Adaptive defaults to 0, which is exactly
today's spacing. Quads defaults to on (section 4 shows it is never worse on
the test cases).

## 2. Where triangles come from today

Measured on `tests/cases.py` with default parameters:

| case | quads | fan triangles | split triangles |
|---|---|---|---|
| cylinder, cone | 59 | 0 | 0 |
| ellipse | 55 | 8 | 0 |
| offset_cylinder | 55 | 8 | 0 |
| twisted | 28 | 2 | 60 |

- **Fan triangles**: the alignment DP moves `(1,1)`, `(1,0)` or `(0,1)` on
  the sample grid. Where rulings lean, a run of single-rail steps makes
  several rulings share one rail point, and `build_faces` emits a fan of
  triangles. This is an artifact of the grid.
- **Split triangles**: quads still over Planar Tolerance after planarize
  are split along a diagonal by `split_quads`. These mark the
  non-developable regions and stay (decision: option A).

## 3. Adaptive sampling (`mino/core/rails.py`)

### 3.1 Parameter

`LoftParams.adaptive: float = 0.0`, the share of samples placed by
bending. `prepare_rails` raises `LoftError` unless `0 <= adaptive < 1`
(at 1, straight parts would get no samples). The operator caps it at 0.9.

### 3.2 Bending profile of one rail

For a deduplicated polyline `p_0 … p_M` (M ≥ 1) with segment lengths
`ℓ_k`, cumulative lengths `c_0 = 0 … c_M = L`, unit segment directions
`d_k` and turning angles `θ_v = arccos(clip(d_{v−1}·d_v))` at interior
vertices `v = 1 … M−1`:

- knots (arc length): `0, m_0, m_1, …, m_{M−1}, L` with segment midpoints `m_k = (c_k + c_{k+1}) / 2`
- values: `0, 0, θ_1, θ_1+θ_2, …, Σθ, Σθ`, so `Θ(m_k) = Σ_{v=1..k} θ_v`

`Θ` is linear between knots, so the turning at vertex `v` is spread over
the two half-segments that meet there. A sharp corner on a POLY rail gets
samples clustered around it, never stacked on one point. Knots are
divided by `L` to give normalized positions `t ∈ [0, 1]`.
`bend_profile(points) -> (t_knots, theta)`.

### 3.3 Shared positions

Rails A and B must keep the same count N, and sample `i` on A should keep
facing sample `i` on B so Window keeps its meaning. Both rails therefore
use the same normalized positions:

```
U(t) = (1 − w)·t + w·(Θ_A(t) + Θ_B(t)) / (Θ_A,tot + Θ_B,tot)
```

with `w = adaptive`. `U` is evaluated on the union of both rails' knots,
where it is piecewise linear and strictly increasing for `w < 1`, so
`t_i = interp(i/(N−1), U, t)` inverts it exactly. `t_0 = 0` and
`t_{N−1} = 1` are set exactly. If `Θ_A,tot + Θ_B,tot < 1e-9` (both rails
straight) the positions are `linspace(0, 1, N)`. Summing absolute turning
(rather than normalizing each rail to its own total) keeps a rail with a
few degrees of wiggle from claiming as many samples as one with a 90°
bend. `shared_positions(points_a, points_b, samples, adaptive) -> (N,)`.

### 3.4 Resampling and tangents

- `resample(points, samples, tangents=None, positions=None)`: with
  `positions` (normalized, strictly increasing) the targets are
  `positions · L`; without, today's code path runs unchanged. Supplied
  tangents are interpolated at the targets as today.
- Tangents computed from the samples keep today's `central_difference`
  (neighbour chord inside, three-point one-sided at the ends) in both
  modes. A three-point formula that accounts for uneven spacing was
  measured and rejected (3.7): it overshoots wherever spacing jumps, which
  adaptive spacing does at every bend-to-straight transition. `Rail` and
  `relax.moved_tangents_b` are therefore unchanged.
- `prepare_rails(points_a, points_b, samples, tangents_a=None, tangents_b=None, adaptive=0.0)`:
  after orienting B, if `adaptive > 0` it computes shared positions on the
  deduplicated rails and resamples both with them.
- Callers pass `params.adaptive`: `loft`, `strakes.subdivide_sections`,
  `diagnose.diagnose`, `relax.relax_rail_b` and the dart remedy in
  `blender/remedies.py`.

### 3.5 Blender input density

`inputs._bezier_rail` evaluates `ceil(samples · oversample / segments) + 1`
points per Bezier segment. `oversample` is 4 today and becomes 16 when
`adaptive > 0`, so clustered samples land on the curve rather than on
chords of the input polyline. `get_sections(context, samples, adaptive=0.0)`
passes it through `curve_rail`. POLY rails and Edit Mode chains are used
as given. NURBS rails come from Blender's evaluated curve, so their
Resolution U still bounds the detail (README).

### 3.6 Window

Window stays in samples. With adaptive spacing, the same Window covers
less distance where samples are dense. Documented in the README; no
behavioural change.

### 3.7 Evidence (throwaway calibration)

Dense test curves resampled with the shared positions of 3.3, rail B a
translated copy. Outline error is the largest distance from the true curve
to the resampled polyline; tangent error the largest angle between a
computed and the true tangent.

| curve, Samples | outline error w=0 / 0.6 / 0.9 | tangent error w=0.6: chord / uneven three-point |
|---|---|---|
| ellipse 3:1, 30 | 0.0048 / 0.0008 / 0.0019 | 0.17° / 0.12° |
| sine (2 periods), 30 | 0.0335 / 0.0147 / 0.0259 | 1.65° / 4.29° |
| quarter arc + straight, 30 | 0.0031 / 0.0007 / 0.0004 | 0.76° / 1.32° |
| quarter arc + straight, 16 | 0.0116 / 0.0025 / 0.0016 | 0.25° / 3.32° |
| tanh step, 60 | 0.0021 / 0.0002 / 0.0004 | 0.74° / 0.79° |

Adaptive 0.6 cuts outline error 2–10× on every curve; 0.9 over-concentrates
samples on some (sine, ellipse) and is worse than 0.6 there, so the README
suggests 0.5–0.6. On the sine, w=0.9 thins the samples at the inflections
enough that the tangent error there exceeds even spacing's (9.34° vs 7.97°
at 16 samples, 1.48° vs 0.58° at 60), while w=0.6 stays at or below it
(4.85° and 0.57°); the README notes this.

## 4. Quad strips (`mino/core/quads.py`, new)

### 4.1 Parameter

`LoftParams.quads: bool = True`.

### 4.2 Spreading fans

`spread_fans(path) -> np.ndarray` of shape `(K, 2)`, float, `K = len(path)`.
Steps are `s_k = path[k+1] − path[k]`. For each maximal run of identical
non-diagonal steps covering vertices `k … e+1` (`r = e − k + 1` steps),
the coordinate the run holds fixed (i for `(0,1)` steps, j for `(1,0)`)
gets an offset rising linearly from `lo` at vertex `k` to `hi` at vertex
`e+1`:

- `lo = −SPREAD`, or 0 if the run starts at the first vertex
- `hi = +SPREAD`, or 0 if the run ends at the last vertex
- `SPREAD = 0.25` samples

Strictly increasing in both coordinates, by construction: every offset
lies in `[−SPREAD, SPREAD]`, so a coordinate that advances by 1 still
advances by at least `1 − 2·SPREAD = 0.5`; a held coordinate advances by
`(hi − lo)/r ≥ SPREAD/r > 0` (no run can start at the first vertex and end
at the last, since the path reaches `(N−1, N−1)`). Vertices outside fans
keep integer coordinates, so their rulings sit exactly on samples. The
strip's four corners never move.

### 4.3 Strip

- `rail_at(rail, x) -> (points, tangents)`: linear interpolation at
  fractional indices `x` along the resampled rail, tangents renormalized.
  Integer `x` returns the sample exactly.
- `quad_strip(rail_a, rail_b, path) -> (verts, faces, twist)`:
  `ends = spread_fans(path)`, `verts = [A(ends[:,0]); B(ends[:,1])]`
  (`2K` vertices), faces `(k, k+1, K+k+1, K+k)` for `k < K−1` (the vertex
  order `best_diagonal` expects), and twist from `paired_twist` on the
  spread rulings.

### 4.4 Loft pipeline (`mino/core/__init__.py`)

After `path = align(cost)`:

- `quads` on: `verts, faces, ruling_twist = quad_strip(...)`; vertices per rail `m = K`.
- `quads` off: today's grid twist, `verts = [A; B]`, `build_faces(path, n)`; `m = n`.

Everything after that is shared, with `m` in place of `n`:

- ruling lengths are measured between each ruling's two vertices,
- `pinned = [0, m−1, m, 2m−1]`, plus `range(0, m)` / `range(m, 2m)` for `pin_a` / `pin_b`,
- planarize `rails = (range(0, m), range(m, 2m))`,
- face twist, split, unfold and the report are unchanged.

`StripResult.rulings` stays the sample-index DP path, so strakes
(`mid_rails`), darts, relaxation and diagnosis keep working on samples.
`ruling_twist`, `face_twist`, failing ranges and the report describe the
spread rulings, i.e. the mesh actually produced; ruling `k` is still
`path[k]`, so indices line up. New field
`StripResult.ruling_verts: list` holds each ruling's two mesh vertices:
`(k, K+k)` for quads, `(i, n+j)` for the grid.

### 4.5 Diagnosis

`probe_window` measures twist the way the loft will: when
`params.quads`, the twist of `quad_strip` on the probed path. A "Re-loft
with Window N" suggestion then matches the result it produces. Strake
search already calls `loft` and needs no change.

### 4.6 Strakes

Each strip spreads its own fans. Neighbouring strakes can therefore place
their vertices on a shared mid-rail up to ¼ sample apart. Both lie on the
same rail polyline, so the glue edges meet; their lengths differ only by
the sagitta of a quarter-sample chord. Pinned rails are interpolated
points and stay pinned.

### 4.7 Evidence (throwaway probe, before planarize)

| case | quads | over Planar Tolerance | worst planarity | worst twist (grid) |
|---|---|---|---|---|
| cylinder, cone | 59 | 0 | 0.0000 | 0.00 (0.00) |
| offset_cylinder | 63 | 0 | 0.0007 | 12.47 (12.47) |
| ellipse | 63 | 0 | 0.0052 | 11.09 (11.09) |
| twisted | 60 | 31 | 0.0184 | 30.96 (30.96) |

Extra smoothing of the whole ruling map (rejected option 2) gave no
planarity gain and moves every vertex off its sample.

## 5. Blender, export and viewer

- `MINO_OT_loft` and `MINO_OT_reloft` gain:
  - `adaptive`: FloatProperty "Adaptive", default 0.0, 0.0–0.9, "Share of samples placed where the rails bend; 0 spaces them evenly". Drawn after Samples.
  - `quads`: BoolProperty "Quads", default True, "Give every ruling its own rail points so faces are quads; quads that cannot be made flat are still split". Drawn before Planarize.
- `PARAM_PROPS` gains both, so Re-loft can override them.
- Stored params round-trip through `asdict` with no change. Objects made
  before this change load with the defaults, so re-lofting one gives quads.
- `result_to_dict` adds `"ruling_verts"`. The viewer draws rulings from
  `ruling_verts`, falling back to `[i, samples + j]` when the field is
  missing (older JSON). `tools/viewer/export_cases.py` regenerates the
  sample cases.
- The dart (gusset) mesh is unchanged: it builds its own refined grid from
  `result.rulings`.
- README: Adaptive and Quads in Parameters, suggesting 0.5–0.6 and noting
  that high values thin samples on straight and gently curving parts
  (3.7); the Window note (3.6); NURBS Resolution U (3.5).

## 6. Errors

- `adaptive` outside `[0, 1)`: `LoftError("adaptive must be at least 0 and below 1")`.
- A spread ruling can be degenerate (twist `inf`) like any grid ruling;
  the report and `output.py` already handle non-finite twist.

## 7. Testing (pytest, written first)

Rails:
- `adaptive=0` takes today's code path (no positions passed), so existing rail tests stand unchanged.
- An L-shaped POLY rail with `adaptive=0.6`: the samples nearest the corner are closer together than even spacing; all samples distinct.
- Straight rails with `adaptive=0.6` stay evenly spaced.
- `shared_positions` gives one array for both rails, from 0 to 1, strictly increasing, and its spacing follows the rail that bends when the other is straight.
- `resample` with positions puts samples at those fractions of the length.
- A quarter arc joined to a straight segment, 16 samples: outline error with `adaptive=0.6` is below half the even-spacing error (measured 0.0025 vs 0.0116).
- The same rail without supplied tangents, 30 samples, `adaptive=0.6`: tangents within 1° of the true ones (measured 0.76°).
- `adaptive` of −0.1 and 1.0 raise `LoftError`.

Quads:
- `spread_fans`: on many random monotone paths, both coordinates strictly increase, endpoints are unchanged, every offset is at most 0.25, and a fan-free path comes back unchanged.
- `rail_at` at integer indices returns the samples exactly.
- Loft with `quads=True`:
  - ellipse and offset_cylinder: every face a quad, none split, worst twist equal to the grid loft's.
  - twisted: no fan triangles; every triangle comes from a split.
  - cylinder and cone: vertices and faces identical to `quads=False`.
  - `area_unfolded ≈ area_3d` on the developable cases.
  - `ruling_verts` has one pair per ruling, all in range.
- `probe_window` agrees with the loft's worst twist when `quads` is on.
- Existing tests that depend on fan triangles pin `quads=False`.

Export and Blender:
- `result_to_dict` carries `ruling_verts`; the regenerated viewer cases load.
- Blender tests (run when `bpy` is importable): the loft and Re-loft operators accept `adaptive` and `quads`; a quads loft of two arcs gives a mesh with only quads.
