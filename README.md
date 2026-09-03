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

## Develop

    uv sync
    uv run pytest -q
    uv run python tools/viewer/export_cases.py
    open tools/viewer/index.html

Design: `docs/superpowers/specs/2026-09-03-mino-poc-design.md`.
