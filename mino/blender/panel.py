import bpy

from .operator import MINO_OT_loft


class MINO_PT_panel(bpy.types.Panel):
    bl_label = "Mino"
    bl_idname = "MINO_PT_panel"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "Mino"

    def draw(self, context):
        col = self.layout.column(align=True)
        col.operator(MINO_OT_loft.bl_idname, icon="MOD_CURVE")
        col.separator()
        col.label(text="Select two curves (active = rail A),")
        col.label(text="or two edge chains in Edit Mode.")
        col.label(text="Settings: Adjust Last Operation panel.")
        col.separator()
        col.label(text="View twist: Solid shading > Color > Attribute")
