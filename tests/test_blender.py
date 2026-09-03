import json
import math

import numpy as np
import pytest

bpy = pytest.importorskip("bpy")  # mino.blender imports bpy at package level

import mino  # noqa: E402
from mino.blender.inputs import chains_from_edges  # noqa: E402
from mino.blender.output import twist_colors  # noqa: E402
from mino.core.errors import LoftError  # noqa: E402


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
        mino.register()
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


def _make_poly_arc_curve(name, radius, z, n=12):
    cu = bpy.data.curves.new(name, "CURVE")
    cu.dimensions = "3D"
    sp = cu.splines.new("POLY")
    sp.points.add(n - 1)
    for k in range(n):
        th = k * math.pi / (n - 1)
        x, y = radius * math.cos(th), radius * math.sin(th)
        sp.points[k].co = (x, y, z, 1.0)
    obj = bpy.data.objects.new(name, cu)
    bpy.context.collection.objects.link(obj)
    return obj


def _make_nurbs_arc_curve(name, radius, z, n=12):
    cu = bpy.data.curves.new(name, "CURVE")
    cu.dimensions = "3D"
    sp = cu.splines.new("NURBS")
    sp.points.add(n - 1)
    for k in range(n):
        th = k * math.pi / (n - 1)
        x, y = radius * math.cos(th), radius * math.sin(th)
        sp.points[k].co = (x, y, z, 1.0)
    sp.use_endpoint_u = True
    sp.order_u = 4
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
    result = bpy.ops.mino.loft(samples=30)
    assert result == {"FINISHED"}
    obj = bpy.data.objects["Mino"]
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
    result = bpy.ops.mino.loft(samples=12)
    assert result == {"FINISHED"}
    assert bpy.context.mode == "EDIT_MESH"
    out = bpy.data.objects["Mino"]
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
        bpy.ops.mino.loft()
    assert "Mino" not in bpy.data.objects


def test_operator_on_poly_curves(fresh_scene):
    a = _make_poly_arc_curve("A", 1.0, 0.0)
    b = _make_poly_arc_curve("B", 1.0, 1.0)
    _select([a, b], a)
    result = bpy.ops.mino.loft(samples=20)
    assert result == {"FINISHED"}
    obj = bpy.data.objects["Mino"]
    me = obj.data
    assert len(me.vertices) == 40
    assert "twist" in me.attributes
    twist = [d.value for d in me.attributes["twist"].data]
    assert max(twist) < 1.0


def test_operator_on_nurbs_curves(fresh_scene):
    a = _make_nurbs_arc_curve("A", 1.0, 0.0)
    b = _make_nurbs_arc_curve("B", 1.0, 1.0)
    _select([a, b], a)
    try:
        result = bpy.ops.mino.loft(samples=20)
    except RuntimeError:
        a.data.resolution_u = 12
        b.data.resolution_u = 12
        result = bpy.ops.mino.loft(samples=20)
    assert result == {"FINISHED"}
    obj = bpy.data.objects["Mino"]
    assert len(obj.data.vertices) == 40


def test_operator_rejects_multi_spline_curve(fresh_scene):
    a = _make_poly_arc_curve("A", 1.0, 0.0)
    b = _make_poly_arc_curve("B", 1.0, 1.0)
    a.data.splines.new("POLY")
    _select([a, b], a)
    with pytest.raises(RuntimeError):
        bpy.ops.mino.loft(samples=20)
    assert "Mino" not in bpy.data.objects


def test_loft_stores_inputs_and_diagnosis(fresh_scene):
    a = _make_arc_curve("A", 1.0, 0.0)
    b = _make_arc_curve("B", 1.0, 1.0)
    _select([a, b], a)
    assert bpy.ops.mino.loft(samples=30) == {"FINISHED"}
    obj = bpy.data.objects["Mino"]
    rails = json.loads(obj["mino_rails"])
    assert set(rails) == {"a", "ta", "b", "tb"}
    assert len(rails["a"]) >= 30 * 4 and len(rails["ta"]) == len(rails["a"])
    params = json.loads(obj["mino_params"])
    assert params["samples"] == 30 and params["consistent_creases"] is True
    assert obj["mino_report"].startswith("Mino: 30 rulings")
    diag = json.loads(obj["mino_diagnosis"])
    assert [s["kind"] for s in diag["suggestions"]] == ["ok"]
    assert "strake" in obj.data.attributes
    assert all(d.value == 0 for d in obj.data.attributes["strake"].data)


def test_loft_with_diagnose_off_stores_empty_diagnosis(fresh_scene):
    a = _make_arc_curve("A", 1.0, 0.0)
    b = _make_arc_curve("B", 1.0, 1.0)
    _select([a, b], a)
    assert bpy.ops.mino.loft(samples=20, diagnose=False) == {"FINISHED"}
    assert bpy.data.objects["Mino"]["mino_diagnosis"] == ""


def test_three_curves_chain_into_two_strips(fresh_scene):
    a = _make_arc_curve("A", 1.0, 0.0)
    b = _make_arc_curve("B", 1.0, 1.0)
    c = _make_arc_curve("C", 1.0, 2.0)
    _select([a, c, b], a)  # selection order deliberately scrambled
    assert bpy.ops.mino.loft(samples=20) == {"FINISHED"}
    first, second = bpy.data.objects["Mino"], bpy.data.objects["Mino.001"]
    assert len(first.data.vertices) == 40 and len(second.data.vertices) == 40
    assert {d.value for d in first.data.attributes["strake"].data} == {0}
    assert {d.value for d in second.data.attributes["strake"].data} == {1}
    r1, r2 = json.loads(first["mino_rails"]), json.loads(second["mino_rails"])
    assert np.allclose(np.array(r1["b"]), np.array(r2["a"]))
    # the middle curve (z = 1) is the shared rail
    assert np.allclose(np.array(r1["b"])[:, 2], 1.0)
    assert bpy.data.objects["Mino"].select_get() and bpy.data.objects["Mino.001"].select_get()
    assert bpy.context.view_layer.objects.active == bpy.data.objects["Mino"]


def test_load_inputs_round_trip(fresh_scene):
    from mino.blender.state import load_inputs, params_from_dict, params_to_dict
    a = _make_arc_curve("A", 1.0, 0.0)
    b = _make_arc_curve("B", 1.0, 1.0)
    _select([a, b], a)
    bpy.ops.mino.loft(samples=24, window=5)
    pa, ta, pb, tb, params = load_inputs(bpy.data.objects["Mino"])
    assert pa.shape[1] == 3 and ta.shape == pa.shape and pb.shape[1] == 3
    assert params.samples == 24 and params.window == 5
    assert params_from_dict({**params_to_dict(params), "bogus": 1}).window == 5
    with pytest.raises(LoftError):
        load_inputs(a)
