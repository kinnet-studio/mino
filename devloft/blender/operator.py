from __future__ import annotations

import json

import bpy
from bpy.props import BoolProperty, EnumProperty, FloatProperty, FloatVectorProperty, IntProperty, StringProperty

from ..core import LoftError, LoftParams, format_report, loft
from ..core.export import result_to_dict
from . import inputs, output


class DEVLOFT_OT_loft(bpy.types.Operator):
    """Loft a developable strip between two rail curves or edge chains"""
    bl_idname = "devloft.loft"
    bl_label = "Developable Loft (two rails)"
    bl_options = {"REGISTER", "UNDO"}

    samples: IntProperty(name="Samples", default=60, min=8, max=400,
                         description="Points per rail after resampling")
    window: IntProperty(name="Window", default=8, min=1, max=100,
                        description="How far rulings may lean, in samples")
    twist_tolerance: FloatProperty(name="Twist Tolerance", default=5.0, min=0.0, max=90.0, subtype="NONE",
                                   description="Rulings with more twist (degrees) are flagged")
    tie_breaker: EnumProperty(name="Tie Breaker", default="none", items=[
        ("none", "None", "Minimum twist only"),
        ("shortest", "Shortest Ruling", "Prefer shorter rulings"),
        ("plane", "Plane Direction", "Prefer rulings parallel to a plane normal"),
    ])
    tie_weight: FloatProperty(name="Tie Weight", default=0.1, min=0.0, max=10.0)
    plane_normal: FloatVectorProperty(name="Plane Normal", default=(0.0, 0.0, 1.0), subtype="XYZ")
    planarize: BoolProperty(name="Planarize", default=True,
                            description="Nudge vertices so near-planar quads become planar")
    planar_tolerance: FloatProperty(name="Planar Tolerance", default=0.01, min=0.0, max=0.5,
                                    description="Relative diagonal offset; quads above this are split")
    export_json: StringProperty(name="Export JSON", default="", subtype="FILE_PATH",
                                description="Optional path to write viewer JSON")

    def execute(self, context):
        params = LoftParams(
            samples=self.samples, window=self.window, twist_tolerance=self.twist_tolerance,
            tie_breaker=self.tie_breaker, tie_weight=self.tie_weight,
            plane_normal=tuple(self.plane_normal), planarize=self.planarize,
            planar_tolerance=self.planar_tolerance,
        )
        try:
            (pa, ta), (pb, tb) = inputs.get_rails(context, params.samples)
            if context.mode == "EDIT_MESH":
                bpy.ops.object.mode_set(mode="OBJECT")
            result = loft(pa, pb, params, ta, tb)
        except LoftError as exc:
            self.report({"ERROR"}, str(exc))
            return {"CANCELLED"}
        output.create_strip_object(context, result, "DevLoft", params.twist_tolerance)
        if self.export_json:
            path = bpy.path.abspath(self.export_json)
            with open(path, "w", encoding="utf-8") as fh:
                json.dump(result_to_dict(result, pa, pb, "blender", params), fh)
        summary = format_report(result.report, result.failing_ranges)
        self.report({"INFO"}, summary)
        if result.failing_ranges:
            ranges = ", ".join(f"{a}-{b}" if a != b else str(a) for a, b in result.failing_ranges)
            self.report({"WARNING"}, f"Not developable at rulings {ranges}; split the panel there")
        return {"FINISHED"}

    def draw(self, context):
        col = self.layout.column()
        col.prop(self, "samples")
        col.prop(self, "window")
        col.prop(self, "twist_tolerance")
        col.prop(self, "tie_breaker")
        if self.tie_breaker != "none":
            col.prop(self, "tie_weight")
        if self.tie_breaker == "plane":
            col.prop(self, "plane_normal")
        col.prop(self, "planarize")
        col.prop(self, "planar_tolerance")
        col.prop(self, "export_json")
