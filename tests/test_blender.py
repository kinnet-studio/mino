import math

import numpy as np
import pytest

bpy = pytest.importorskip("bpy")  # devloft.blender imports bpy at package level

import devloft  # noqa: E402
from devloft.blender.inputs import chains_from_edges  # noqa: E402
from devloft.blender.output import twist_colors  # noqa: E402
from devloft.core.errors import LoftError  # noqa: E402


def test_chains_from_edges_two_open_chains():
    coords = [(i, 0, 0) for i in range(4)] + [(i, 1, 0) for i in range(3)]
    edges = [(0, 1), (1, 2), (2, 3), (4, 5), (5, 6)]
    chains = chains_from_edges(coords, edges)
    assert sorted(chains) == [[0, 1, 2, 3], [4, 5, 6]]


def test_chains_from_edges_rejects_closed_and_branching():
    with pytest.raises(LoftError):
        chains_from_edges([(0, 0, 0)] * 3, [(0, 1), (1, 2), (2, 0)])
    with pytest.raises(LoftError):
        chains_from_edges([(0, 0, 0)] * 4, [(0, 1), (1, 2), (1, 3)])


def test_twist_colors_ramp():
    c = twist_colors(np.array([0.0, 5.0, 10.0, 50.0, np.inf]), 5.0)
    assert c.shape == (5, 4)
    assert c[0][1] > c[0][0]            # green at 0
    assert c[1][0] > 0.9 and c[1][1] > 0.8   # yellow at tol
    assert c[2][0] > 0.9 and c[2][1] < 0.2   # red at 2 tol
    assert np.allclose(c[3], c[2])
    assert np.allclose(c[:, 3], 1.0)


@pytest.fixture
def fresh_scene():
    bpy.ops.wm.read_factory_settings(use_empty=True)
    try:
        devloft.register()
    except ValueError:
        pass  # already registered
    yield bpy.context


def _make_arc_curve(name, radius, z, n=5):
    cu = bpy.data.curves.new(name, "CURVE")
    cu.dimensions = "3D"
    sp = cu.splines.new("BEZIER")
    sp.bezier_points.add(n - 1)
    step = math.pi / (n - 1)
    handle = radius * (4.0 / 3.0) * math.tan(step / 4.0)
    for k, bp in enumerate(sp.bezier_points):
        th = k * step
        co = (radius * math.cos(th), radius * math.sin(th), z)
        t = (-math.sin(th), math.cos(th), 0.0)
        bp.handle_left_type = bp.handle_right_type = "FREE"
        bp.co = co
        bp.handle_left = tuple(c - handle * tv for c, tv in zip(co, t))
        bp.handle_right = tuple(c + handle * tv for c, tv in zip(co, t))
    obj = bpy.data.objects.new(name, cu)
    bpy.context.collection.objects.link(obj)
    return obj


def _select(objs, active):
    for o in bpy.context.view_layer.objects:
        o.select_set(False)
    for o in objs:
        o.select_set(True)
    bpy.context.view_layer.objects.active = active


def test_operator_on_curves(fresh_scene):
    a = _make_arc_curve("A", 1.0, 0.0)
    b = _make_arc_curve("B", 1.0, 1.0)
    _select([a, b], a)
    result = bpy.ops.devloft.loft(samples=30)
    assert result == {"FINISHED"}
    obj = bpy.data.objects["DevLoft"]
    me = obj.data
    assert len(me.vertices) == 60
    assert 29 <= len(me.polygons) <= 60
    for attr in ("twist", "planarity", "twist_color", "split"):
        assert attr in me.attributes
    twist = [d.value for d in me.attributes["twist"].data]
    assert max(twist) < 1.0
    assert bpy.context.view_layer.objects.active == obj
    assert me.color_attributes.active_color is not None
    assert me.color_attributes.active_color.name == "twist_color"
    assert me.attributes["twist_color"].domain == "CORNER"
    corner_colors = np.array([tuple(d.color) for d in me.attributes["twist_color"].data])
    assert corner_colors.shape == (sum(len(p.loop_indices) for p in me.polygons), 4)
    # every corner of a zero-twist cylinder face is the green end of the ramp
    assert np.allclose(corner_colors[:, 1], 0.8, atol=0.05) and np.all(corner_colors[:, 0] < 0.3)
    planarity = [d.value for d in me.attributes["planarity"].data]
    assert all(0.0 <= v <= 1.0 for v in planarity)
    split = [d.value for d in me.attributes["split"].data]
    assert set(split) <= {True, False} and not any(split)


def test_operator_on_edit_mode_chains(fresh_scene):
    me = bpy.data.meshes.new("rails")
    verts = [(i * 0.5, 0.0, 0.0) for i in range(6)] + [(i * 0.5, 0.0, 1.0) for i in range(6)]
    edges = [(i, i + 1) for i in range(5)] + [(6 + i, 7 + i) for i in range(5)]
    me.from_pydata(verts, edges, [])
    me.update()
    for e in me.edges:
        e.select = True
    obj = bpy.data.objects.new("rails", me)
    bpy.context.collection.objects.link(obj)
    _select([obj], obj)
    bpy.ops.object.mode_set(mode="EDIT")
    result = bpy.ops.devloft.loft(samples=12)
    assert result == {"FINISHED"}
    assert bpy.context.mode == "EDIT_MESH"
    out = bpy.data.objects["DevLoft"]
    assert len(out.data.vertices) == 24
    assert "twist" in out.data.attributes
    bpy.ops.object.mode_set(mode="OBJECT")


def test_operator_errors_on_bad_selection(fresh_scene):
    # bpy.ops raises RuntimeError (not a plain {"CANCELLED"} return) when an
    # operator reports {'ERROR'} and cancels; this is standard bpy.ops
    # behavior, not a headless-only quirk (reproduced with a minimal
    # unrelated operator during development).
    a = _make_arc_curve("A", 1.0, 0.0)
    _select([a], a)
    with pytest.raises(RuntimeError):
        bpy.ops.devloft.loft()
    assert "DevLoft" not in bpy.data.objects
