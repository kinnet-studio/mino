# DevLoft for Blender

Developable loft between two rail curves, for papercraft. Select two curves
(or two edge chains in Edit Mode), run **Developable Loft (two rails)**, and
get a strip mesh whose faces are planar where the rails allow it. Each face
carries a `twist` attribute and a `twist_color` (green = developable, red =
not) so you can see where a panel must be split.

## Install

    uv run python make_zip.py

Then in Blender 4.2+: Preferences > Get Extensions > drop-down > Install from
Disk, choose `dist/devloft-0.1.0.zip`. The operator appears in the 3D
Viewport sidebar under the **DevLoft** tab and in the Add menu.

To see the twist colors: Solid shading > Color > Attribute.

## Parameters

- Samples: points per rail after resampling.
- Window: how far a ruling may lean, in samples.
- Twist Tolerance: degrees; rulings above it are flagged and colored.
- Tie Breaker: none, shortest ruling, or plane direction, with a weight.
- Planarize: nudge vertices so near-planar quads become exactly planar.
- Planar Tolerance: quads still above it are split into two triangles.
- Export JSON: optional path; drop the file on `tools/viewer/index.html`.

## Develop

    uv sync
    uv run pytest -q
    uv run python tools/viewer/export_cases.py
    open tools/viewer/index.html

Design: `docs/superpowers/specs/2026-09-03-devloft-poc-design.md`.
