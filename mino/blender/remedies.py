"""Remedy operators acting on the active Mino result object, and the panel rows."""
from __future__ import annotations

import json
from dataclasses import replace

import bpy
from bpy.props import EnumProperty, FloatProperty, IntProperty

from ..core import LoftError, loft
from ..core.dart import dart_mesh, dart_proposal
from ..core.diagnose import diagnosis_from_dict
from ..core.rails import prepare_rails
from ..core.strakes import chain_loft, subdivide_sections
from . import output
from .state import load_diagnosis, load_inputs

OFF_ROW = ("Diagnosis off for this loft", None, {})


def diagnosis_rows(obj):
    """(text, operator idname or None, property values) per suggestion, from stored JSON."""
    raw = obj.get("mino_diagnosis", "") if hasattr(obj, "get") else ""
    if not raw:
        return [OFF_ROW]
    d = diagnosis_from_dict(json.loads(raw))
    rows = []
    for s in d.suggestions:
        if s.kind == "window":
            rows.append((f"Re-loft with Window {s.params['window']}", "mino.reloft", {"window": int(s.params["window"])}))
        elif s.kind == "subdivide":
            rows.append((f"Subdivide into {s.params['strakes']} strakes", "mino.subdivide", {"strakes": int(s.params["strakes"])}))
        elif s.kind == "unsolved":
            rows.append((s.text, "mino.subdivide", {"strakes": int(s.params["strakes"])}))
        elif s.kind == "dart":
            rows.append((f"Cut {s.params['kind']} at ruling {s.params['ruling']} ({abs(float(s.params['wedge_deg'])):.1f}°)",
                         "mino.dart", {"ruling": int(s.params["ruling"])}))
        else:
            rows.append((s.text, None, {}))
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

    window: IntProperty(name="Window", default=0, min=0, max=100, description="0 keeps the stored value")
    samples: IntProperty(name="Samples", default=0, min=0, max=400, description="0 keeps the stored value")
    twist_tolerance: FloatProperty(name="Twist Tolerance", default=0.0, min=0.0, max=90.0,
                                   description="0 keeps the stored value")

    def execute(self, context):
        obj = context.active_object
        try:
            pa, ta, pb, tb, params = load_inputs(obj)
            overrides = {k: v for k, v in (("window", self.window), ("samples", self.samples),
                                           ("twist_tolerance", self.twist_tolerance)) if v}
            params = replace(params, **overrides)
            result = loft(pa, pb, params, ta, tb)
        except LoftError as exc:
            self.report({"ERROR"}, str(exc))
            return {"CANCELLED"}
        new, _ = output.create_result_object(context, f"{obj.name}.reloft", pa, ta, pb, tb, params, result, True)
        self.report({"INFO"}, new["mino_report"])
        return {"FINISHED"}


class MINO_OT_subdivide(_MinoRemedy, bpy.types.Operator):
    """Split the loft into narrower developable strips (strakes)"""
    bl_idname = "mino.subdivide"
    bl_label = "Subdivide into Strakes"

    strakes: IntProperty(name="Strakes", default=0, min=0, max=8,
                         description="Number of strips; 0 uses the diagnosis suggestion")

    def execute(self, context):
        obj = context.active_object
        try:
            pa, ta, pb, tb, params = load_inputs(obj)
            diag = load_diagnosis(obj)
            strakes = self.strakes or (diag.strakes_needed if diag and diag.strakes_needed else 2)
            base = loft(pa, pb, params, ta, tb)
            sections = subdivide_sections(pa, pb, ta, tb, params, base.rulings, strakes)
            strips = chain_loft(sections, params)
        except LoftError as exc:
            self.report({"ERROR"}, str(exc))
            return {"CANCELLED"}
        for k, (((sa, sta), (sb, stb)), strip) in enumerate(zip(zip(sections, sections[1:]), strips)):
            output.create_result_object(context, f"{obj.name}.strake.{k}", sa, sta, sb, stb, params, strip, True, strake=k)
        worst = max(s.report.max_twist for s in strips)
        bad = sum(1 for s in strips if s.failing_ranges)
        self.report({"INFO"}, f"Mino: {strakes} strakes, worst twist {worst:.1f} deg, {bad} not developable")
        return {"FINISHED"}


class MINO_OT_dart(_MinoRemedy, bpy.types.Operator):
    """Refine the strip and mark a dart (gusset) cut where twist peaks"""
    bl_idname = "mino.dart"
    bl_label = "Cut Dart"

    ruling: IntProperty(name="Ruling", default=-1, min=-1, max=800,
                        description="Cut ruling; -1 uses the diagnosis suggestion")
    mid_rails: IntProperty(name="Mid Rails", default=3, min=1, max=6)
    dart_from: EnumProperty(name="Open End", default="B", items=[
        ("B", "Rail B", "The cut opens at rail B and its tip stops short of rail A"),
        ("A", "Rail A", "The cut opens at rail A and its tip stops short of rail B"),
    ])

    def execute(self, context):
        obj = context.active_object
        try:
            pa, ta, pb, tb, params = load_inputs(obj)
            diag = load_diagnosis(obj)
            base = loft(pa, pb, params, ta, tb)
            ra, rb = prepare_rails(pa, pb, params.samples, ta, tb)
            if self.ruling >= 0:
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


classes = (MINO_OT_reloft, MINO_OT_subdivide, MINO_OT_dart)
