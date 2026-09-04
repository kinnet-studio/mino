"""Remedy operators acting on the active Mino result object, and the panel rows."""
from __future__ import annotations

import json
from dataclasses import replace

import bpy
from bpy.props import BoolProperty, EnumProperty, FloatProperty, FloatVectorProperty, IntProperty

from ..core import LoftError, loft
from ..core.dart import dart_mesh, dart_proposal
from ..core.diagnose import diagnosis_from_dict
from ..core.rails import prepare_rails
from ..core.relax import relax_rail_b
from ..core.strakes import chain_loft, subdivide_sections
from . import output
from .state import load_diagnosis, load_inputs

OFF_ROW = ("Diagnosis off for this loft", None, {})

PARAM_PROPS = ("samples", "window", "twist_tolerance", "tie_breaker", "tie_weight", "plane_normal",
               "planarize", "planar_tolerance", "planarize_max_nudge", "consistent_creases")


def effective_params(op, stored):
    """Stored LoftParams overridden by every property the caller actually set."""
    overrides = {}
    for name in PARAM_PROPS:
        if op.properties.is_property_set(name):
            value = getattr(op, name)
            overrides[name] = tuple(value) if name == "plane_normal" else value
    return replace(stored, **overrides)


def seed_from_params(op, stored):
    """Copy stored values into unset properties so the redo panel shows real numbers."""
    for name in PARAM_PROPS:
        if not op.properties.is_property_set(name):
            setattr(op, name, getattr(stored, name))


def diagnosis_rows(obj):
    """(text, operator idname or None, property values) per suggestion, from stored JSON."""
    raw = obj.get("mino_diagnosis", "") if hasattr(obj, "get") else ""
    if not raw:
        return [OFF_ROW]
    d = diagnosis_from_dict(json.loads(raw))
    rows = []
    for s in d.suggestions:
        if s.kind == "window":
            window = min(int(s.params["window"]), 400)
            rows.append((f"Re-loft with Window {window}", "mino.reloft", {"window": window}))
        elif s.kind == "subdivide":
            rows.append((f"Subdivide into {s.params['strakes']} strakes", "mino.subdivide", {"strakes": int(s.params["strakes"])}))
        elif s.kind == "unsolved":
            rows.append((s.text, "mino.subdivide", {"strakes": int(s.params["strakes"])}))
        elif s.kind == "dart":
            rows.append((f"Cut {s.params['kind']} at ruling {s.params['ruling']} ({abs(float(s.params['wedge_deg'])):.1f}°)",
                         "mino.dart", {"ruling": int(s.params["ruling"])}))
        else:
            rows.append((s.text, None, {}))
    if d.failing_ranges:
        rows.append(("Relax rail B (moves ≤ 5% of ruling)", "mino.relax", {}))
    return rows


class _MinoRemedy:
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context):
        obj = context.active_object
        ok = context.mode == "OBJECT" and obj is not None and "mino_rails" in obj
        if not ok:
            cls.poll_message_set("Select a Mino result object in Object Mode")
        return ok


class MINO_OT_reloft(_MinoRemedy, bpy.types.Operator):
    """Loft the stored rails again with changed settings"""
    bl_idname = "mino.reloft"
    bl_label = "Re-loft"

    samples: IntProperty(name="Samples", default=60, min=8, max=400,
                         description="Points per rail after resampling")
    window: IntProperty(name="Window", default=8, min=1, max=400,
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
    planarize_max_nudge: FloatProperty(name="Max Nudge", default=0.05, min=0.0, max=0.5,
                                       description="Cap on vertex movement during planarize, "
                                                    "as a fraction of the mean ruling length")
    consistent_creases: BoolProperty(name="Consistent Creases", default=True,
                                     description="Use one crease direction per run of split quads")
    diagnose: BoolProperty(name="Diagnose", default=True,
                           description="Store ranked suggestions for non-developable regions on the result")

    def invoke(self, context, event):
        obj = context.active_object
        try:
            pa, ta, pb, tb, params = load_inputs(obj)
        except LoftError as exc:
            self.report({"ERROR"}, str(exc))
            return {"CANCELLED"}
        seed_from_params(self, params)
        return self.execute(context)

    def execute(self, context):
        obj = context.active_object
        try:
            pa, ta, pb, tb, stored = load_inputs(obj)
            params = effective_params(self, stored)
            result = loft(pa, pb, params, ta, tb)
        except LoftError as exc:
            self.report({"ERROR"}, str(exc))
            return {"CANCELLED"}
        new, _ = output.create_result_object(context, f"{obj.name}.reloft", pa, ta, pb, tb, params, result,
                                             self.diagnose)
        self.report({"INFO"}, new["mino_report"])
        return {"FINISHED"}


class MINO_OT_subdivide(_MinoRemedy, bpy.types.Operator):
    """Split the loft into narrower developable strips (strakes)"""
    bl_idname = "mino.subdivide"
    bl_label = "Subdivide into Strakes"

    strakes: IntProperty(name="Strakes", default=2, min=2, max=8,
                         description="Number of strips")
    diagnose: BoolProperty(name="Diagnose", default=True,
                           description="Store ranked suggestions for non-developable regions on each strake")

    def invoke(self, context, event):
        obj = context.active_object
        if not self.properties.is_property_set("strakes"):
            diag = load_diagnosis(obj)
            if diag and diag.strakes_needed:
                self.strakes = diag.strakes_needed
        return self.execute(context)

    def execute(self, context):
        obj = context.active_object
        try:
            pa, ta, pb, tb, params = load_inputs(obj)
            diag = load_diagnosis(obj)
            strakes = (self.strakes if self.properties.is_property_set("strakes")
                      else (diag.strakes_needed if diag and diag.strakes_needed else 2))
            base = loft(pa, pb, params, ta, tb)
            sections = subdivide_sections(pa, pb, ta, tb, params, base.rulings, strakes)
            strips = chain_loft(sections, params)
        except LoftError as exc:
            self.report({"ERROR"}, str(exc))
            return {"CANCELLED"}
        for k, (((sa, sta), (sb, stb)), strip) in enumerate(zip(zip(sections, sections[1:]), strips)):
            output.create_result_object(context, f"{obj.name}.strake.{k}", sa, sta, sb, stb, params, strip,
                                        self.diagnose, strake=k)
        worst = max(s.report.max_twist for s in strips)
        bad = sum(1 for s in strips if s.failing_ranges)
        self.report({"INFO"}, f"Mino: {strakes} strakes, worst twist {worst:.1f} deg, {bad} not developable")
        return {"FINISHED"}


class MINO_OT_dart(_MinoRemedy, bpy.types.Operator):
    """Refine the strip and mark a dart (gusset) cut where twist peaks"""
    bl_idname = "mino.dart"
    bl_label = "Cut Dart"

    ruling: IntProperty(name="Ruling", default=0, min=0, max=800,
                        description="Cut ruling")
    mid_rails: IntProperty(name="Mid Rails", default=3, min=1, max=6)
    dart_from: EnumProperty(name="Open End", default="B", items=[
        ("B", "Rail B", "The cut opens at rail B and its tip stops short of rail A"),
        ("A", "Rail A", "The cut opens at rail A and its tip stops short of rail B"),
    ])

    def invoke(self, context, event):
        obj = context.active_object
        if not self.properties.is_property_set("ruling"):
            diag = load_diagnosis(obj)
            if diag and diag.darts:
                self.ruling = diag.darts[0].ruling
        return self.execute(context)

    def execute(self, context):
        obj = context.active_object
        try:
            pa, ta, pb, tb, params = load_inputs(obj)
            diag = load_diagnosis(obj)
            base = loft(pa, pb, params, ta, tb)
            ra, rb = prepare_rails(pa, pb, params.samples, ta, tb)
            if self.properties.is_property_set("ruling"):
                ruling = min(self.ruling, len(base.rulings) - 1)
            elif diag and diag.darts:
                ruling = diag.darts[0].ruling
            else:
                ruling = int(base.ruling_twist.argmax())
            rng = next((r for r in base.failing_ranges if r[0] <= ruling <= r[1]), (ruling, ruling))
            proposal = dart_proposal(ra, rb, base.rulings, base.ruling_twist, rng, self.mid_rails)
            proposal = replace(proposal, ruling=ruling)
            dm = dart_mesh(ra, rb, base.rulings, base.ruling_twist, proposal, self.mid_rails,
                           params.planar_tolerance, self.dart_from)
        except LoftError as exc:
            self.report({"ERROR"}, str(exc))
            return {"CANCELLED"}
        output.create_dart_object(context, dm, f"{obj.name}.dart", params.twist_tolerance)
        self.report({"INFO"}, f"Mino: {proposal.kind} at ruling {ruling}, "
                              f"{abs(proposal.wedge_deg):.1f} deg wedge, {len(dm.faces)} faces")
        return {"FINISHED"}


class MINO_OT_relax(_MinoRemedy, bpy.types.Operator):
    """Move rail B a bounded amount so the loft becomes developable, and return the moved rail as a curve"""
    bl_idname = "mino.relax"
    bl_label = "Relax Rail B"

    max_move: FloatProperty(name="Max Move", default=0.05, min=0.0, max=0.5,
                            description="Largest move of any rail B point, as a fraction of the mean ruling length")
    smoothness: FloatProperty(name="Smoothness", default=1.0, min=0.0, max=10.0,
                              description="Weight of the term that keeps neighbouring moves similar")
    margin: FloatProperty(name="Margin", default=0.5, min=0.0, max=5.0,
                          description="Degrees below Twist Tolerance the solver aims for")
    iterations: IntProperty(name="Iterations", default=400, min=10, max=1000)
    pin_endpoints: BoolProperty(name="Pin Endpoints", default=True,
                                description="Keep the two ends of rail B where they are")
    diagnose: BoolProperty(name="Diagnose", default=True,
                           description="Store ranked suggestions for non-developable regions on the result")

    def execute(self, context):
        obj = context.active_object
        try:
            pa, ta, pb, tb, params = load_inputs(obj)
            res = relax_rail_b(pa, pb, ta, tb, params, max_move=self.max_move, smoothness=self.smoothness,
                               margin=self.margin, iterations=self.iterations, pin_endpoints=self.pin_endpoints)
        except LoftError as exc:
            self.report({"ERROR"}, str(exc))
            return {"CANCELLED"}
        output.create_curve_object(context, f"{obj.name}.railB.relaxed", res.points_b)
        output.create_result_object(context, f"{obj.name}.relaxed", pa, ta, res.points_b, None, params,
                                    res.result, self.diagnose)
        pct = 100.0 * res.max_move_used / res.mean_ruling if res.mean_ruling > 0 else 0.0
        self.report({"INFO"}, f"Mino: relaxed rail B, max twist {res.twist_before:.1f} -> {res.twist_after:.1f} deg, "
                              f"largest move {res.max_move_used:.3g} ({pct:.0f}% of mean ruling)")
        if res.result.failing_ranges:
            self.report({"WARNING"}, "Mino: still over tolerance; raise Max Move or subdivide into strakes")
        return {"FINISHED"}


classes = (MINO_OT_reloft, MINO_OT_subdivide, MINO_OT_dart, MINO_OT_relax)
