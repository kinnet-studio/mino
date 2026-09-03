"""Create the result mesh object with twist attributes."""
from __future__ import annotations

import numpy as np

GREEN = np.array([0.13, 0.8, 0.27, 1.0])
YELLOW = np.array([1.0, 0.87, 0.13, 1.0])
RED = np.array([0.93, 0.13, 0.13, 1.0])


def twist_colors(face_twist, tolerance):
    tw = np.asarray(face_twist, dtype=float)
    t = np.where(np.isfinite(tw), tw / max(tolerance, 1e-9), 2.0)
    t = np.clip(t, 0.0, 2.0)
    out = np.empty((len(tw), 4))
    low = t <= 1.0
    out[low] = GREEN + (YELLOW - GREEN) * t[low, None]
    out[~low] = YELLOW + (RED - YELLOW) * (t[~low, None] - 1.0)
    return out


def _write_twist_attributes(me, face_twist, twist_tolerance):
    """Face 'twist' float and corner 'twist_color' set as the active color."""
    import numpy as np
    twist = me.attributes.new("twist", "FLOAT", "FACE")
    # 90.0 is a defensive sentinel for non-finite twist; the DP alignment
    # only ever enters finite-cost cells, so face_twist can never actually
    # be inf here, but foreach_set requires a finite float32 array.
    twist.data.foreach_set("value", np.where(np.isfinite(face_twist), face_twist, 90.0).astype(np.float32))
    # Blender only surfaces POINT/CORNER FLOAT_COLOR attributes in
    # mesh.color_attributes (and therefore in Solid shading's Color >
    # Attribute dropdown); a FACE-domain color attribute is invisible there.
    # me.update() has already built me.polygons, so loop_indices exist.
    col = me.attributes.new("twist_color", "FLOAT_COLOR", "CORNER")
    face_colors = twist_colors(face_twist, twist_tolerance)
    corner_colors = np.repeat(face_colors, [len(p.loop_indices) for p in me.polygons], axis=0)
    col.data.foreach_set("color", corner_colors.astype(np.float32).ravel())
    me.color_attributes.active_color = col
    me.color_attributes.render_color_index = me.color_attributes.active_color_index


def _link_and_select(context, obj):
    context.collection.objects.link(obj)
    # Selecting/activating objects is only valid in Object Mode; the operator
    # no longer force-switches modes (a mode switch inside a REGISTER/UNDO
    # operator leaves the mode un-restored on undo, breaking Adjust Last
    # Operation). In Edit Mode we just link the new object and leave
    # selection/active-object state alone.
    if context.mode == "OBJECT":
        for o in context.view_layer.objects:
            o.select_set(False)
        obj.select_set(True)
        context.view_layer.objects.active = obj


def create_strip_object(context, result, name, twist_tolerance):
    import bpy

    me = bpy.data.meshes.new(name)
    me.from_pydata(result.verts.tolist(), [], [list(map(int, f)) for f in result.faces])
    me.update()

    plan = me.attributes.new("planarity", "FLOAT", "FACE")
    plan.data.foreach_set("value", result.face_planarity.astype(np.float32))
    split = me.attributes.new("split", "BOOLEAN", "FACE")
    split.data.foreach_set("value", result.face_split.astype(bool))
    _write_twist_attributes(me, result.face_twist, twist_tolerance)
    me.update()

    obj = bpy.data.objects.new(name, me)
    _link_and_select(context, obj)
    return obj


def create_result_object(context, name, points_a, tangents_a, points_b, tangents_b, params, result,
                         diagnose_flag: bool, strake: int = 0):
    """Strip object plus stored inputs and (optionally) a diagnosis."""
    from ..core.diagnose import diagnose
    from .state import store_inputs

    obj = create_strip_object(context, result, name, params.twist_tolerance)
    attr = obj.data.attributes.new("strake", "INT", "FACE")
    attr.data.foreach_set("value", [int(strake)] * len(obj.data.polygons))
    diagnosis = None
    if diagnose_flag:
        diagnosis = diagnose(points_a, points_b, tangents_a, tangents_b, params, result)
    store_inputs(obj, points_a, tangents_a, points_b, tangents_b, params, result, diagnosis)
    return obj, diagnosis
