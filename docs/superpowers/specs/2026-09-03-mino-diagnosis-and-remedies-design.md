# Mino: developability diagnosis and remedies. Design (spec 1 of 2)

Date: 2026-09-03
Status: approved design, pre-implementation
Builds on: `2026-09-03-devloft-poc-design.md` (the loft core and Blender layer, now named Mino)
Companion: `2026-09-03-mino-rail-relaxation-design.md` (spec 2)

## 1. Goal

When a loft is not developable, Mino currently reports the failing ruling
ranges and stops. This spec adds a Diagnosis: ranked, numeric suggestions
for what to do, shown in the Mino sidebar panel for the selected result
object, each with a button that runs the remedy on that result's inputs.

Remedies in this spec: re-loft with a larger window, subdivide into
strakes, cut a dart, consistent creases. Rail relaxation is spec 2.

## 2. Facts the design rests on

- Twist at a ruling depends only on the rail tangents at its endpoints.
  Cutting a strip along a ruling never lowers twist. Narrowing a strip
  across its width does: with a mid-rail on the same rulings, each half has
  roughly half the twist.
- A surface with non-zero Gaussian curvature has no distortion-free flat
  pattern. Every remedy subdivides, creases, or changes the shape.
- A dart is not a developable-strip tool. It refines the failing region
  with interior vertices so paper can bend both ways, places one cut where
  distortion concentrates, and reports the wedge angle. Unfolder flattens
  the result with the dart. The panel says so.

## 3. Where "the last loft" lives

The loft operator stores its inputs on the result object as string custom
properties, so any Mino object is self-describing:

- `mino_rails`: JSON `{"a": [[x,y,z],...], "ta": [...] | null, "b": [...], "tb": [...] | null}` in world space, the raw polylines handed to the core.
- `mino_params`: JSON of `LoftParams` via `dataclasses.asdict`.
- `mino_report`: the `format_report` line.
- `mino_diagnosis`: JSON of the Diagnosis (section 5).

The panel reads the active object; if it has `mino_rails` the Diagnosis
section is shown. Remedy operators read these properties, never the
original curves, so they work after the curves are edited or deleted, in
Object Mode only (they create objects). Several results can coexist.

## 4. Multi-section loft (chaining)

`mino.loft` accepts two or more selected curves. With more than two, the
active curve is the first section and the rest are ordered greedily by
nearest centroid from the previous section. Each consecutive pair is
lofted with the same parameters, producing one object per strip named
`Mino`, `Mino.001`, ... each with its own stored inputs and diagnosis, and a
face integer attribute `strake` (0-based strip index). Edit Mode input stays
two chains only.

Core: `chain_loft(sections: list[tuple[points, tangents | None]], params) -> list[StripResult]` in `mino/core/strakes.py`.

## 5. Diagnosis (`mino/core/diagnose.py`)

```python
@dataclass
class Suggestion:
    kind: str      # "ok" | "window" | "subdivide" | "dart" | "creases" | "unsolved"
    text: str      # one line for the panel
    params: dict   # kind-specific numbers the button pre-fills
    rank: int      # 1 is shown first

@dataclass
class Diagnosis:
    failing_ranges: list[tuple[int, int]]
    max_twist: float
    window_fix: int | None        # smallest probed window that passes
    strakes_needed: int | None    # smallest strip count that passes, None if > max
    strakes_worst_twist: float    # worst twist at strakes_needed or at the max tried
    darts: list[DartProposal]     # one per failing range
    crease_runs: int              # runs of consecutive split quads
    suggestions: list[Suggestion]
```

`diagnose(points_a, points_b, tangents_a, tangents_b, params, result, max_strakes=8) -> Diagnosis`:

1. No failing ranges: one suggestion `ok`, "All rulings within tolerance", and stop.
2. Window probe: for `w` in (2·window, 4·window), capped at `samples - 1`, rerun `twist_matrix` and `align` only and take the max twist along the path. The first `w` under tolerance becomes `window_fix`; suggestion rank 1: "Re-loft with Window {w}: all rulings within tolerance".
3. Strakes: `find_strake_count` (section 6) with the probed window if found, else the current one. Rank 2: "Subdivide into {n} strakes (worst twist {x:.1f}°)". If none up to `max_strakes` passes: rank 2 `unsolved`: "{max} strakes still leave {x:.1f}° twist: cut a dart or relax rail B".
4. Darts: one `DartProposal` per failing range (section 8). Rank 3: "Cut a dart at ruling {k} ({angle:.1f}° wedge, refined mesh)" or "...gusset..." when the deficit is negative.
5. Creases: rank 4, informational: "{n} crease runs, diagonals consistent" or "... zigzag; enable Consistent creases".

Suggestions are sorted by rank. `format_diagnosis(d) -> list[str]` returns the lines.

## 6. Subdivide into strakes (`mino/core/strakes.py`)

Mid-rails lie on the surface Mino already showed the user: for each
chosen ruling `(i, j)` in `result.rulings`, with `A`, `B` the resampled rails
before planarize, the point at fraction `t` is `A[i] + t·(B[j] − A[i])`.
The mid-rail at `t` is that polyline over all rulings (duplicates from
triangle steps are removed by `resample`). Tangents are not supplied;
central differences apply.

- `mid_rails(rail_a, rail_b, rulings, count) -> list[np.ndarray]` returns `count` polylines at `t = 1/(count+1), ..., count/(count+1)`.
- `subdivide(points_a, points_b, tangents_a, tangents_b, params, rulings, strakes) -> list[StripResult]` lofts A→M1, M1→M2, ..., Mk→B with `strakes = k+1` strips. Mid-rails get the resampled points as their input polylines.
- `find_strake_count(points_a, points_b, tangents_a, tangents_b, params, rulings, max_strakes) -> tuple[int | None, float, list[StripResult]]`: a bracketed doubling search. It tries 2, 4, 8 (capped at `max_strakes`); if none passes it returns `(None, worst at the largest tried, its strips)`. When a count passes, it scans every count in the bracket between the last failing doubling and the passing one in ascending order and returns the first that passes, so the reported minimum is exact. Worst twist is not monotone in strake count, so a bisection is not used. The search is the bulk of a diagnosis (about 1 s at 60 samples with the linear scan; the bracket cuts it by roughly 2.6x).

Note the re-lofts choose their own rulings; the mid-rails only fix the
surface. Twist typically halves per doubling, so 8 strakes covers
about 3 halvings.

## 7. Consistent creases (`mino/core/mesh.py`)

`split_quads(verts, faces, planarity, tolerance, consistent=False)`. With
`consistent=True`, after choosing each quad's diagonal individually, group
consecutive split quads (adjacent in `faces` order with no unsplit face
between) into runs. For each run compute the total dihedral of option A
(`(v0,v2,v3),(v0,v1,v2)`) and option B (`(v0,v1,v3),(v1,v2,v3)`) summed
over the run, and apply the winner to every quad in the run. Strip
adjacency (first triangle carries the incoming ruling) is preserved by
both options. `LoftParams.consistent_creases: bool = True`; the loft
operator exposes it as "Consistent Creases". `Diagnosis.crease_runs` counts
the runs.

## 8. Dart (`mino/core/dart.py`)

```python
@dataclass
class DartProposal:
    range: tuple[int, int]   # failing ruling range
    ruling: int              # cut ruling: max twist within the range
    wedge_deg: float         # positive = dart (remove wedge), negative = gusset (insert)
    kind: str                # "dart" | "gusset"
```

Refined grid: rows `r = 0..m+1` at fractions `t_r = r/(m+1)` (row 0 is A,
row m+1 is B), columns are the rulings `k` of the whole strip;
`P[r][k] = A[i_k] + t_r·(B[j_k] − A[i_k])`. Faces between consecutive rows
and columns are quads; each is kept if its planarity is under
`planar_tolerance`, else split along its shorter diagonal. Default `m = 3`,
operator range 1..6.

Wedge angle: discrete Gauss–Bonnet. At every interior grid vertex (rows
1..m, columns `k1..k2` of the failing range, i.e. strictly inside the window
widened by one column on each side) the angle deficit is
`2π − Σ incident face corner angles`, computed on the triangulated faces.
`wedge = Σ deficits` in degrees. Positive means the flat piece needs a
wedge removed (dart), negative means inserted (gusset). This approximates
the total Gaussian curvature of the ruled surface spanned by the chosen
rulings; the band along the boundary is not counted, so the estimate is a
few percent low at `m = 3`.

Cut: along the column of `ruling`, from the B row down to row 1 (the tip is
one row short of A), or mirrored when `dart_from == "A"`. Those edges get a
boolean edge attribute `seam` and `use_seam = True` in Blender so Unfolder
and Export Paper Model cut there.

- `dart_proposal(rail_a, rail_b, rulings, ruling_twist, failing_range, mid_rails=3, planar_tolerance) -> DartProposal`
- `dart_mesh(rail_a, rail_b, rulings, ruling_twist, proposal, mid_rails, planar_tolerance, dart_from="B") -> DartMesh` with `verts`, `faces`, `face_twist` (max of the two bounding rulings), `seam_edges: list[tuple[int, int]]`.

## 9. Re-loft with window (`mino.reloft`)

Runs `loft` on the stored rails with the stored params and any overrides
given as operator properties (window, samples, twist_tolerance, ...),
creating a new object next to the original. The diagnosis button
pre-fills `window = window_fix`.

## 10. Blender layer

Operators, all in Object Mode, all reading the active Mino object's
stored inputs and creating new objects (never replacing):

- `mino.reloft`: properties mirror the loft's. Any property the caller did not set (`self.properties.is_property_set`) takes its value from `mino_params`; `invoke` seeds the unset ones so the redo panel shows real values. Every remedy operator also has a `diagnose` toggle (default True).
- `mino.subdivide`: `strakes` int 2..8; when unset it takes `strakes_needed` from the stored diagnosis (or 2), seeded by `invoke` for the redo panel. Creates one object per strip named `<name>.strake.<k>` with `strake` attribute, each with its own stored inputs and diagnosis. Reports the worst twist.
- `mino.dart`: `ruling` int (unset: the first dart proposal's ruling, else the max-twist ruling), `mid_rails` int 1..6 default 3, `dart_from` enum A/B default B. Creates `<name>.dart` with `twist` face attribute, `seam` edge attribute, `use_seam` set. Reports the wedge angle and kind.
- `mino.loft` gains `consistent_creases` (default True) and `diagnose` (default True; when off no diagnosis is stored and the panel says so).

Panel `MINO_PT_panel` gains a Diagnosis box when the active object has
`mino_rails`: the report line, then one row per suggestion: the text and,
for kinds window / subdivide / dart, a button whose operator properties are
pre-filled from `params`. `ok` and `creases` rows have no button. A helper
`diagnosis_rows(obj) -> list[tuple[str, str | None, dict]]` (text, operator
idname or None, property values) is pure Python over the stored JSON so it
can be unit-tested without drawing.

The loft operator also stores `mino_rails`, `mino_params`, `mino_report`,
`mino_diagnosis` on every object it creates (including chained strips).

## 11. Testing

Core (numpy only):
- `diagnose` on the cylinder returns the single `ok` suggestion.
- `diagnose` on the offset cylinder with window 2 finds `window_fix` (the offset needs 4) and rank-1 window suggestion.
- `find_strake_count` on the twisted case returns a count in 2..8 whose strips all pass; the mid-rails lie on the rulings (each mid-rail point is collinear with its ruling endpoints).
- Consistent creases: on the twisted case at samples 24 with planarize off, every run of split quads uses one diagonal orientation; strip adjacency test still passes.
- Gauss–Bonnet: `angle_deficit_total` on a planar grid is 0, on a cylinder grid is 0, on a unit-sphere patch (longitude 0..π/2, latitude 0..π/4, grid 40×80) is within 6 percent of the solid angle `(π/2)·sin(π/4)`, and on a saddle grid is negative.
- `dart_proposal` on the twisted case picks the max-twist ruling in the range and a positive or negative wedge (sign documented in the test by the surface).
- `dart_mesh` seam edges form a single path from the B row to row 1 along the cut column; consecutive faces share edges (unfold still works).
- `chain_loft` with three sections returns two strips whose shared rail matches.

Blender (headless bpy):
- After `mino.loft`, the object has all four properties and they parse.
- `mino.reloft` with `window` creates a second object.
- `mino.subdivide` on a twisted pair creates `strakes` objects with the `strake` attribute.
- `mino.dart` creates an object whose `seam` edge attribute and `use_seam` mark at least one edge.
- Three selected curves produce two strip objects.
- `diagnosis_rows` returns rows with the expected operator ids.

## 12. Layout

```
mino/core/diagnose.py   Diagnosis, Suggestion, diagnose, format_diagnosis
mino/core/strakes.py    mid_rails, subdivide, find_strake_count, chain_loft
mino/core/dart.py       DartProposal, DartMesh, angle_deficit_total, dart_proposal, dart_mesh
mino/core/mesh.py       split_quads(consistent=)
mino/core/types.py      LoftParams.consistent_creases
mino/blender/remedies.py   MINO_OT_reloft, MINO_OT_subdivide, MINO_OT_dart, diagnosis_rows
mino/blender/operator.py   stores properties, chaining, new options
mino/blender/panel.py      Diagnosis box
tests/test_diagnose.py test_strakes.py test_dart.py test_blender_remedies.py
```

## 13. Open risks

- Strake search cost: up to 7 extra lofts per diagnosis. At samples 60 it is well under a second; at 400 it can take several seconds, hence the `diagnose` toggle.
- Mid-rails from triangle-step rulings can produce short segments; `resample` dedupes exact duplicates only, so near-duplicates slightly bias arc-length spacing. Acceptable for a PoC.
- The wedge angle counts interior vertices only; users should read it as an estimate of a few percent precision.
