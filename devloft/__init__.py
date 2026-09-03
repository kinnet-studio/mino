"""DevLoft: developable loft between two rails.

The core package is importable without Blender. The Blender layer is only
imported inside register()/unregister() so tests can run on plain Python.
"""

__version__ = "0.1.0"


def register():
    from .blender import register as _register
    _register()


def unregister():
    from .blender import unregister as _unregister
    _unregister()
