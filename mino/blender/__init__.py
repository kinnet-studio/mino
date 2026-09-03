import bpy

from . import operator, panel

_classes = (operator.MINO_OT_loft, panel.MINO_PT_panel)


def _menu(self, context):
    self.layout.operator(operator.MINO_OT_loft.bl_idname, icon="MOD_CURVE")


def register():
    for cls in _classes:
        bpy.utils.register_class(cls)
    bpy.types.VIEW3D_MT_add.append(_menu)
    bpy.types.VIEW3D_MT_object_context_menu.append(_menu)
    bpy.types.VIEW3D_MT_edit_mesh_context_menu.append(_menu)


def unregister():
    bpy.types.VIEW3D_MT_edit_mesh_context_menu.remove(_menu)
    bpy.types.VIEW3D_MT_object_context_menu.remove(_menu)
    bpy.types.VIEW3D_MT_add.remove(_menu)
    for cls in reversed(_classes):
        bpy.utils.unregister_class(cls)
