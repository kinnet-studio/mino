# DevLoft for Blender: developable strip PoC. Design

Date: 2026-09-03
Status: approved design, pre-implementation
Source brief: `~/Desktop/devloft-blender-poc-brief.md`

## 1. Goal

A Blender extension that lofts a developable strip between two rail curves,
in the spirit of Rhino's DevLoft. The user selects two rails, runs one
operator, and gets an ordinary mesh whose faces are planar wherever the
rails allow it, plus per-ruling twist feedback that shows where the panel
is not developable and must be split.

Downstream tools (Unfolder, PolyZamboni) consume plain meshes, so the output
carries no custom data beyond standard mesh attributes.

## 2. Non-goals for the PoC

- Multi-strip chaining across many sections.
- Planar merge and rotate-diagonal cleanup passes.
- The hair `Curves` object type.
- Guaranteeing developability when the rails do not permit it. The tool
  reports failure regions; it does not modify the rails to fix them.

## 3. Architecture

Three parts with one-way dependencies:

```
tools/viewer   ->  devloft/core   <-  devloft/blender
(three.js)         (numpy only)       (bpy, thin)
```

- `devloft/core/` contains all geometry. It imports only numpy and the
  standard library, never `bpy` or `mathutils`. It is testable with pytest
  on any Python.
- `devloft/blender/` adapts Blender data to the core and back: reads rails
  from the selection, exposes the operator and panel, writes the result
  mesh and its attributes.
- `tools/viewer/` is a standalone HTML page that renders exported strips so
  results can be inspected without Blender.

numpy is bundled with Blender, so the extension declares no wheel
dependencies.

## 4. Core data model

```python
@dataclass
class Rail:
    points: np.ndarray    # (N, 3) float64, resampled by arc length
    tangents: np.ndarray  # (N, 3) unit vectors

@dataclass
class LoftParams:
    samples: int = 60          # N, points per rail after resampling
    window: int = 8            # band half-width around the diagonal
    twist_tolerance: float = 5.0   # degrees, ruling is "failing" above this
    tie_breaker: str = "none"  # "none" | "shortest" | "plane"
    tie_weight: float = 0.1    # weight of the tie-breaker cost term
    plane_normal: tuple = (0.0, 0.0, 1.0)  # used when tie_breaker == "plane"
    planarize: bool = True
    planar_tolerance: float = 0.01  # relative diagonal offset, see 5.4
    planarize_iterations: int = 10
    planarize_max_nudge: float = 0.05  # fraction of mean ruling length

@dataclass
class StripResult:
    verts: np.ndarray           # (V, 3), rail A points then rail B points
    faces: list[tuple[int, ...]]  # index tuples, quads or triangles
    rulings: list[tuple[int, int]]   # (i, j) pairs along the strip
    ruling_twist: np.ndarray    # degrees, one per ruling
    face_twist: np.ndarray      # degrees, max twist of the face's rulings
    face_planarity: np.ndarray  # relative diagonal offset, 0 for triangles
    face_split: np.ndarray      # bool, True where a quad was triangulated
    failing_ranges: list[tuple[int, int]]  # inclusive ruling index ranges over tolerance
    report: Report

@dataclass
class Report:
    max_twist: float
    mean_twist: float
    failing_ruling_count: int
    ruling_count: int
    split_quad_count: int
    quad_count: int
    area_3d: float
    area_unfolded: float
```

Vertex indexing: rail A occupies indices `0..N-1`, rail B occupies
`N..2N-1`. Faces reference these directly. The planarize pass may move
vertices slightly; `verts` holds the moved positions.

## 5. Core algorithm

### 5.1 Rails (`core/rails.py`)

Input is a polyline `(K, 3)` and optional tangents `(K, 3)`. Processing:

1. Remove consecutive duplicate points.
2. Compute cumulative arc length and resample to `samples` points at equal
   arc-length spacing using linear interpolation.
3. Tangents: if supplied, interpolate them the same way and renormalize.
   Otherwise use central differences on the resampled points, one-sided at
   the ends.
4. Orientation: rail B is reversed when `|B[-1] - A[0]| < |B[0] - A[0]|`.
   Both rails are resampled to the same `samples` count.

Callers that have exact tangents (Bezier curves) supply them. The bpy layer
samples Bezier segments densely (`samples * 4` points per rail) before
handing them to the core so that linear resampling introduces negligible
error.

### 5.2 Twist (`core/twist.py`)

For ruling `R = B[j] - A[i]` with tangents `T_A`, `T_B`:

```
n_A = normalize(T_A x R)
n_B = normalize(T_B x R)
twist = angle(n_A, n_B) in degrees, folded into [0, 90]
```

Folding into `[0, 90]` means `n` and `-n` are treated as the same normal,
so the sign convention of the tangents does not matter. A ruling whose
length is below `1e-9` or that is within 1 degree of parallel to either
tangent is invalid and receives cost `inf`.

`twist_matrix(rail_a, rail_b, window)` computes the full `(N, N)` cost
matrix, `inf` outside the band `|i - j| <= window`.

### 5.3 Alignment (`core/align.py`)

A dynamic-time-warping pass over the band:

- State `(i, j)` means the ruling from `A[i]` to `B[j]` is used.
- Moves: `(i+1, j+1)` diagonal, `(i+1, j)` and `(i, j+1)` steps.
- Cost of entering `(i, j)`: `twist[i, j] + tie_weight * tie[i, j]` where
  `tie` is `0` for "none", normalized ruling length (length divided by the
  mean ruling length on the diagonal) for "shortest", and
  `1 - |dot(normalize(R), plane_normal)|` for "plane".
- Non-diagonal steps carry an additional constant penalty of `0.5` degrees
  so the solver does not prefer triangle fans over quads for free.
- Start at `(0, 0)`, end at `(N-1, N-1)`. Backtrack to recover the path.
- If the entire band is `inf` at some row or column, the DP still finds a
  path because non-diagonal moves can route around an invalid cell. If no
  finite path exists at all, the operator raises `LoftError` with a message
  naming the first rail index where every candidate is invalid.

The path is a monotone sequence, so consecutive rulings never cross.

### 5.4 Faces (`core/mesh.py`)

Walk the ruling path. For consecutive rulings `(i, j)` and `(i', j')`:

- Diagonal step (`i' = i+1`, `j' = j+1`): quad `(A[i], A[i+1], B[j+1], B[j])`.
- Step along A only: triangle `(A[i], A[i+1], B[j])`.
- Step along B only: triangle `(A[i], B[j+1], B[j])`.

Face winding is consistent so normals point the same way along the strip.

Planarity of a quad is the distance between its two diagonals (closest
approach of the two lines) divided by the mean diagonal length. Zero means
exactly planar. Triangles have planarity `0`.

Planarize pass (optional): for `planarize_iterations` rounds, compute each
quad's best-fit plane through its centroid using the normal of the cross
product of the diagonals, project its four vertices onto that plane, and
accumulate the projected positions. Each vertex moves to the average of its
proposals. Rail endpoints (`A[0]`, `A[N-1]`, `B[0]`, `B[N-1]`) are pinned.
Each vertex's total displacement is capped at `planarize_max_nudge` times the
mean ruling length (default 0.05), a separate parameter from
`planar_tolerance` so the cap does not forbid the very move the pass needs.
The pass stops early when the max planarity is below `planar_tolerance`. On a
genuinely twisted strip it cannot flatten every quad; it is best effort.

After planarization, any quad still above `planar_tolerance` is split into
two triangles along the diagonal with the smaller dihedral angle between
the two resulting triangles. `face_split` records this.

`face_twist` is the max of the twist of the rulings bounding the face.

### 5.5 Unfold check (`core/unfold.py`)

A sequential flattening used for validation only. Each face is treated as
rigid: it is projected onto its own best-fit plane, then placed in 2D by
aligning the ruling edge it shares with the previous face (rotation only).
Because the strip is a chain, every face shares exactly one ruling with its
predecessor. Return the 2D area and the per-face layout. Since quads over
`planar_tolerance` are split before unfolding, the unfolded area matches the
3D area within that tolerance by construction; the check confirms the
emitted faces are flat enough to lay out rigidly. It does not detect 2D
overlap of the flat pattern. Unfolder remains the real test.

### 5.6 Report (`core/report.py`)

`failing_ranges` are maximal runs of ruling indices with twist above
`twist_tolerance`. `Report` aggregates the numbers listed in section 4.
`format_report(report, failing_ranges)` returns a one-line summary for the
Blender status bar, for example:

```
DevLoft: 60 rulings, max twist 14.2 deg, 7 over 5 deg at rulings 31-37, 4/59 quads split, area 12.30 -> 12.29 unfolded
```

## 6. Blender layer

### 6.1 Inputs (`blender/inputs.py`)

`get_rails(context, samples) -> tuple[np.ndarray, np.ndarray | None] x 2`

Object Mode, two selected Curve objects:
- The first spline of each curve is used. Curves with more than one spline
  raise `LoftError` asking the user to separate them.
- Bezier splines: each segment is evaluated with
  `mathutils.geometry.interpolate_bezier` at `samples * 4 / segment_count`
  points, capped so the whole rail has at least `samples * 4` points.
  Tangents come from the derivative of the Bezier polynomial at the same
  parameters.
- Poly splines: control points are used directly, no tangents supplied.
- NURBS splines: the object's evaluated mesh from the depsgraph gives the
  polyline, no tangents supplied.
- Coordinates are transformed to world space with `matrix_world`.

Edit Mode, one mesh object with selected edges:
- The selected edges must form exactly two disjoint open edge chains.
  Chains are found by walking selected edges from endpoint vertices with
  one selected edge. Anything else raises `LoftError` describing what was
  found (for example "3 chains selected, expected 2" or "selected loop is
  closed").
- Vertex positions are used directly, no tangents supplied.
- The two chains are ordered so that the one containing the lowest vertex
  index is rail A.

Rail selection order: for curves, the active object is rail A and the other
selected curve is rail B.

### 6.2 Operator (`blender/operator.py`)

`DEVLOFT_OT_loft`, label "Developable Loft (two rails)", registered in the
3D Viewport Add menu, the Object Mode and Edit Mesh context menus, and the
DevLoft panel.
Properties mirror `LoftParams`:

- `samples` int, 8..400, default 60
- `window` int, 1..100, default 8
- `twist_tolerance` float degrees, 0..90, default 5
- `tie_breaker` enum none / shortest / plane, default none
- `tie_weight` float, 0..10, default 0.1
- `plane_normal` float vector, default (0, 0, 1)
- `planarize` bool, default True
- `planar_tolerance` float, 0..0.5, default 0.01
- `planarize_max_nudge` float, 0..0.5, default 0.05 (fraction of mean ruling length)

`execute` (no mode switching, so Adjust Last Operation can re-run it):
1. `get_rails` -> two polylines with optional tangents.
2. `core.loft(rail_a, rail_b, params)` -> `StripResult`.
3. `output.create_strip_object(context, result, name="DevLoft")`.
4. `self.report({'INFO'}, format_report(...))`; if `failing_ranges` is
   non-empty, also `self.report({'WARNING'}, ...)` listing the ranges.
5. `LoftError` is caught and reported as `{'ERROR'}` with its message; the
   operator returns `{'CANCELLED'}`.

The operator supports redo through the Adjust Last Operation panel.

### 6.3 Output (`blender/output.py`)

Creates a new mesh object linked to the active collection, in world space
(identity transform), with:

- Face float attribute `twist` (degrees).
- Face float attribute `planarity`.
- Color attribute `twist_color` (FLOAT_COLOR, domain CORNER, one value per
  face corner so faces stay hard-edged): green at 0, yellow at
  `twist_tolerance`, red at `2 * twist_tolerance` and above. Blender only
  treats POINT and CORNER color attributes as color attributes, so FACE would
  never appear in Solid shading. It is set as the active color so Solid
  shading with Color set to Attribute shows it, and it survives export.
- Face boolean attribute `split` marking triangulated quads.
- Edges along rulings marked as seams is NOT done; seams belong to the
  user's unfolding decisions.

In Object Mode the new object becomes active and selected. In Edit Mode the
operator does not switch modes: the new object is linked but the user stays
in Edit Mode of the source mesh (switching modes inside a REGISTER/UNDO
operator breaks redo). The input geometry stays untouched.

### 6.4 Panel (`blender/panel.py`)

Sidebar tab "DevLoft" in the 3D Viewport, showing the operator button and
a brief note on selection requirements. Settings are edited in the Adjust
Last Operation panel after running.

### 6.5 Packaging

`devloft/blender_manifest.toml` with `schema_version = "1.0.0"`,
`blender_version_min = "4.2.0"`, `type = "add-on"`, license GPL-3.0-or-later.
`devloft/__init__.py` registers the operator and panel. A `make_zip.py`
script at the repo root produces `dist/devloft-<version>.zip` installable
through Preferences > Get Extensions > Install from Disk, and also works as
a legacy add-on zip.

## 7. Viewer (`tools/viewer/`)

- `export_cases.py` runs the core on the three test rail sets from
  `tests/cases.py` and writes `tools/viewer/data.js` defining
  `window.DEVLOFT_CASES = [...]`. Each case holds verts, faces, rulings,
  ruling twist, face twist, face split flags, the report, and the input
  rails. It also writes plain `.json` files next to it.
- `index.html` loads three.js from cdnjs (pinned version) and `data.js`.
  It renders rails as thick lines, rulings as lines colored by the same
  green/yellow/red ramp, faces with per-face color, orbit controls, a
  wireframe toggle, a case selector, a legend with the tolerance, and the
  report text. A file input loads any JSON with the same shape, which the
  Blender operator can also write when a debug property is set.
- The page opens from disk with no server because all data is in `data.js`.

## 8. Testing

Core tests (`tests/test_core_*.py`, pytest, numpy only):

- Cylinder: two circular arcs offset along the axis. All rulings pair
  `i == i`, twist below 0.01 degrees, no splits, unfolded area within 0.1
  percent of 3D area.
- Cone: two concentric arcs of different radius at different heights.
  Rulings converge toward the apex (extend consecutive rulings and check
  they meet near a common point), twist below 0.01 degrees, no splits.
- Twisted pair: rail B is rail A rotated about the strip's long axis by a
  twist that grows along its length. The DP path has rulings leaning forward
  (mean `j - i` differs from zero), `failing_ranges` is non-empty and
  covers the high-twist end, and the unfolded area still matches within
  1 percent because split quads are honest triangles.
- Unit tests for resampling (equal spacing, tangent normalization,
  reversal), twist (parallel rulings give 0, known 90 degree case), DP
  (monotone path, band respected, `LoftError` when nothing is valid),
  planarity metric (planar quad gives 0), and planarize (converges on a
  slightly bent quad, pinned endpoints stay put).

Blender tests (`tests/test_blender_*.py`, run with the bpy 4.5 wheel in a
Python 3.11 venv managed by uv, skipped if `bpy` is not importable):

- Build two Bezier curve objects, run the operator, assert a new mesh
  object exists with the expected attributes and face counts.
- Build a mesh with two selected edge chains in Edit Mode, run the operator,
  same assertions.
- Selection errors produce `CANCELLED` and an error report.

## 9. Repository layout

```
devloft/
  blender_manifest.toml
  __init__.py
  core/
    __init__.py   loft() entry point
    rails.py twist.py align.py mesh.py unfold.py report.py types.py
  blender/
    __init__.py inputs.py operator.py output.py panel.py
tests/
  cases.py          the three rail sets
  test_core_*.py
  test_blender_*.py
tools/viewer/
  index.html
  export_cases.py
  data.js           generated, committed so the page works from a clone
make_zip.py
pyproject.toml      uv project, dev deps: pytest, numpy, bpy==4.5.*
README.md
docs/superpowers/specs/
```

## 10. Open risks

- Rails with very different lengths make equal-count resampling pair
  points at different arc-length fractions. The band and DP handle
  moderate mismatch; extreme mismatch is out of scope for the PoC.
- Planarize moves vertices off the rails by tiny amounts. This is accepted
  per the brief; the default tolerance keeps the offset well below paper
  thickness at typical model scales.
- The bpy wheel is Blender 4.5 while the user's Blender may be 4.2 or 5.x.
  The APIs used (curves, bmesh, attributes, extension manifest) are stable
  across that range.
