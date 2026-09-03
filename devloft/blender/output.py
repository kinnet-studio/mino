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


def create_strip_object(context, result, name, twist_tolerance):
    import bpy

    me = bpy.data.meshes.new(name)
    me.from_pydata(result.verts.tolist(), [], [list(map(int, f)) for f in result.faces])
    me.update()

    twist = me.attributes.new("twist", "FLOAT", "FACE")
    twist.data.foreach_set("value", np.where(np.isfinite(result.face_twist), result.face_twist, 90.0).astype(np.float32))
    plan = me.attributes.new("planarity", "FLOAT", "FACE")
    plan.data.foreach_set("value", result.face_planarity.astype(np.float32))
    col = me.attributes.new("twist_color", "FLOAT_COLOR", "FACE")
    col.data.foreach_set("color", twist_colors(result.face_twist, twist_tolerance).astype(np.float32).ravel())
    split = me.attributes.new("split", "BOOLEAN", "FACE")
    split.data.foreach_set("value", result.face_split.astype(bool))
    me.color_attributes.active_color = col
    me.update()

    obj = bpy.data.objects.new(name, me)
    context.collection.objects.link(obj)
    for o in context.view_layer.objects:
        o.select_set(False)
    obj.select_set(True)
    context.view_layer.objects.active = obj
    return obj
