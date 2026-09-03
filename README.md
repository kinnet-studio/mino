# Mino

A developable loft (in the spirit of Rhino's DevLoft) for Blender, by Kinnet Studio.

Developable loft between two rail curves, for papercraft. Select two curves
(or two edge chains in Edit Mode), run **Developable Loft (two rails)**, and
get a strip mesh whose faces are planar where the rails allow it. Each face
carries a `twist` attribute and a `twist_color` (green = developable, red =
not) so you can see where a panel must be split.

## Install

    uv run python make_zip.py

Then in Blender 4.2+: Preferences > Get Extensions > drop-down > Install from
Disk, choose `dist/mino-0.1.0.zip`. The operator appears in the 3D
Viewport sidebar under the **Mino** tab and in the Add menu.
After an Edit Mode run, press Tab to return to Object Mode and select the
new Mino object.

To see the twist colors: Solid shading > Color > Attribute.

## Parameters

- Samples: points per rail after resampling.
- Window: how far a ruling may lean, in samples.
- Twist Tolerance: degrees; rulings above it are flagged and colored.
- Tie Breaker: none, shortest ruling, or plane direction, with a weight.
- Plane Normal: the reference plane's normal, used when Tie Breaker is set to plane direction.
- Planarize: nudge vertices (at most Max Nudge × mean ruling length) so near-planar quads become planar where the rails allow it.
- Planar Tolerance: quads still above it are split into two triangles.
- Max Nudge: cap on vertex movement during planarize, as a fraction of the mean ruling length.
- Export JSON: optional path; open `tools/viewer/index.html` and load the file with the Choose File button.

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

## Develop

    uv sync
    uv run pytest -q
    uv run python tools/viewer/export_cases.py
    open tools/viewer/index.html

Design: `docs/superpowers/specs/2026-09-03-mino-poc-design.md`.
