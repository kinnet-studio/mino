import bpy

from .operator import MINO_OT_loft


class MINO_PT_panel(bpy.types.Panel):
    bl_label = "Mino"
    bl_idname = "MINO_PT_panel"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "Mino"

    def draw(self, context):
        from .remedies import diagnosis_rows

        layout = self.layout
        col = layout.column(align=True)
        col.operator(MINO_OT_loft.bl_idname, icon="MOD_CURVE")
        col.separator()
        col.label(text="Select two or more curves (active = first),")
        col.label(text="or two edge chains in Edit Mode.")
        col.label(text="Settings: Adjust Last Operation panel.")
        col.separator()
        col.label(text="View twist: Solid shading > Color > Attribute")

        obj = context.active_object
        if obj is None or "mino_rails" not in obj:
            return
        box = layout.box()
        box.label(text="Diagnosis", icon="INFO")
        box.label(text=obj.get("mino_report", ""))
        try:
            rows = diagnosis_rows(obj)
        except Exception:
            box.label(text="Diagnosis unavailable (stored data unreadable)")
            return
        for text, idname, props in rows:
            row = box.row()
            if idname is None:
                row.label(text=text)
                continue
            op = row.operator(idname, text=text)
            for key, value in props.items():
                setattr(op, key, value)
