"""Read two rails from the Blender selection as world-space polylines."""
from __future__ import annotations

import math

import numpy as np

from ..core.errors import LoftError


def chains_from_edges(coords, edges):
    """Group edges into ordered open chains of vertex indices.

    Pure Python so it can be unit-tested without bpy. Raises LoftError for
    closed loops or branching selections.
    """
    adj: dict[int, list[int]] = {}
    for a, b in edges:
        adj.setdefault(a, []).append(b)
        adj.setdefault(b, []).append(a)
    for v, nb in adj.items():
        if len(nb) > 2:
            raise LoftError(f"selected edges branch at vertex {v}; select two simple edge chains")
    seen = set()
    chains = []
    ends = sorted(v for v, nb in adj.items() if len(nb) == 1)
    for start in ends:
        if start in seen:
            continue
        chain = [start]
        seen.add(start)
        prev, cur = None, start
        while True:
            nxt = [w for w in adj[cur] if w != prev]
            if not nxt:
                break
            prev, cur = cur, nxt[0]
            if cur in seen:
                break
            chain.append(cur)
            seen.add(cur)
        chains.append(chain)
    if len(seen) != len(adj):
        raise LoftError("a selected edge loop is closed; rails must be open chains")
    return chains


def _bezier_rail(obj, samples):
    from mathutils.geometry import interpolate_bezier

    spline = obj.data.splines[0]
    if spline.use_cyclic_u:
        raise LoftError(f"'{obj.name}' is a closed curve; rails must be open")
    bps = spline.bezier_points
    nseg = len(bps) - 1
    if nseg < 1:
        raise LoftError(f"'{obj.name}' needs at least 2 control points")
    res = max(2, math.ceil(samples * 4 / nseg) + 1)
    mw = obj.matrix_world
    rot = mw.to_3x3()
    pts, tans = [], []
    for k in range(nseg):
        p0, p1 = bps[k], bps[k + 1]
        seg = interpolate_bezier(p0.co, p0.handle_right, p1.handle_left, p1.co, res)
        for idx, q in enumerate(seg):
            if k > 0 and idx == 0:
                continue
            t = idx / (res - 1)
            d = (3 * (1 - t) ** 2 * (p0.handle_right - p0.co)
                 + 6 * (1 - t) * t * (p1.handle_left - p0.handle_right)
                 + 3 * t ** 2 * (p1.co - p1.handle_left))
            pts.append(mw @ q)
            tans.append(rot @ d)
    return np.array([list(p) for p in pts], float), np.array([list(t) for t in tans], float)


def _poly_rail(obj):
    spline = obj.data.splines[0]
    if spline.use_cyclic_u:
        raise LoftError(f"'{obj.name}' is a closed curve; rails must be open")
    mw = obj.matrix_world
    return np.array([list(mw @ p.co.xyz) for p in spline.points], float), None


def _evaluated_rail(obj, context):
    dg = context.evaluated_depsgraph_get()
    ev = obj.evaluated_get(dg)
    me = ev.to_mesh()
    try:
        coords = [tuple(v.co) for v in me.vertices]
        edges = [tuple(e.vertices) for e in me.edges]
    finally:
        ev.to_mesh_clear()
    chains = chains_from_edges(coords, edges)
    if len(chains) != 1:
        raise LoftError(f"'{obj.name}' evaluates to {len(chains)} chains; expected 1")
    from mathutils import Vector

    mw = obj.matrix_world
    return np.array([list(mw @ Vector(coords[i])) for i in chains[0]], float), None


def curve_rail(obj, context, samples):
    cu = obj.data
    if len(cu.splines) != 1:
        raise LoftError(f"'{obj.name}' has {len(cu.splines)} splines; separate them so each rail has one")
    kind = cu.splines[0].type
    if kind == "BEZIER":
        return _bezier_rail(obj, samples)
    if kind == "POLY":
        return _poly_rail(obj)
    return _evaluated_rail(obj, context)


def edit_mode_rails(obj):
    import bmesh

    bm = bmesh.from_edit_mesh(obj.data)
    bm.verts.ensure_lookup_table()
    edges = [(e.verts[0].index, e.verts[1].index) for e in bm.edges if e.select]
    if not edges:
        raise LoftError("select the edges of two rail chains")
    coords = [tuple(v.co) for v in bm.verts]
    chains = chains_from_edges(coords, edges)
    if len(chains) != 2:
        raise LoftError(f"{len(chains)} edge chains selected; expected 2")
    chains.sort(key=min)
    mw = obj.matrix_world
    rails = []
    for chain in chains:
        rails.append((np.array([list(mw @ bm.verts[i].co) for i in chain], float), None))
    return rails[0], rails[1]


def get_rails(context, samples):
    obj = context.active_object
    if context.mode == "EDIT_MESH" and obj is not None and obj.type == "MESH":
        return edit_mode_rails(obj)
    curves = [o for o in context.selected_objects if o.type == "CURVE"]
    if len(curves) != 2:
        raise LoftError(f"select exactly two curve objects ({len(curves)} selected), "
                        "or two edge chains in Edit Mode")
    if obj in curves:
        a = obj
        b = curves[0] if curves[1] is obj else curves[1]
    else:
        a, b = curves
    return curve_rail(a, context, samples), curve_rail(b, context, samples)
