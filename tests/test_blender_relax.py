import json

import numpy as np
import pytest

bpy = pytest.importorskip("bpy")
from mathutils import Matrix  # noqa: E402

import mino  # noqa: E402
from mino.core.rails import prepare_rails  # noqa: E402
from tests.cases import CASES  # noqa: E402


@pytest.fixture
def fresh_scene():
    bpy.ops.wm.read_factory_settings(use_empty=True)
    try:
        mino.register()
    except ValueError:
        pass
    yield bpy.context


def _make_poly_curve(name, points):
    cu = bpy.data.curves.new(name, "CURVE")
    cu.dimensions = "3D"
    sp = cu.splines.new("POLY")
    sp.points.add(len(points) - 1)
    for p, xyz in zip(sp.points, points):
        p.co = (float(xyz[0]), float(xyz[1]), float(xyz[2]), 1.0)
    obj = bpy.data.objects.new(name, cu)
    bpy.context.collection.objects.link(obj)
    return obj


def _select(objs, active):
    for o in bpy.context.view_layer.objects:
        o.select_set(False)
    for o in objs:
        o.select_set(True)
    bpy.context.view_layer.objects.active = active


def _loft(case, samples=40):
    a = _make_poly_curve("A", case["points_a"])
    b = _make_poly_curve("B", case["points_b"])
    _select([a, b], a)
    assert bpy.ops.mino.loft(samples=samples) == {"FINISHED"}
    obj = bpy.data.objects["Mino"]
    _select([obj], obj)
    return obj


def test_relax_creates_curve_and_loft_objects(fresh_scene):
    obj = _loft(CASES["twisted"](n=60, scale=0.3))
    assert bpy.ops.mino.relax(max_move=0.15) == {"FINISHED"}

    curve = bpy.data.objects["Mino.railB.relaxed"]
    assert curve.type == "CURVE"
    spline = curve.data.splines[0]
    assert spline.type == "POLY" and len(spline.points) == 40
    assert curve.matrix_world == Matrix.Identity(4)
    assert not curve.select_get()

    new = bpy.data.objects["Mino.relaxed"]
    rails = json.loads(new["mino_rails"])
    assert len(rails["b"]) == 40 and len(rails["tb"]) == 40
    assert np.allclose(np.array(rails["a"]), np.array(json.loads(obj["mino_rails"])["a"]))
    curve_pts = np.array([p.co[:3] for p in spline.points])
    assert np.allclose(curve_pts, np.array(rails["b"]))
    assert json.loads(new["mino_params"]) == json.loads(obj["mino_params"])
    assert "mino_diagnosis" in new and new["mino_diagnosis"] != ""
    assert bpy.context.view_layer.objects.active == new

    stored = json.loads(obj["mino_rails"])
    _, rb = prepare_rails(stored["a"], stored["b"], 40, stored["ta"], stored["tb"])
    assert not np.allclose(curve_pts, rb.points)          # the rail actually moved
    assert np.linalg.norm(curve_pts - rb.points, axis=1).max() < 0.15 * 1.05  # mean ruling is about 1


def test_relax_leaves_developable_rail_in_place(fresh_scene):
    obj = _loft(CASES["cylinder"](n=60))
    assert bpy.ops.mino.relax() == {"FINISHED"}
    new = bpy.data.objects["Mino.relaxed"]
    assert "all within tolerance" in new["mino_report"]
    curve_pts = np.array([p.co[:3] for p in bpy.data.objects["Mino.railB.relaxed"].data.splines[0].points])
    stored = json.loads(obj["mino_rails"])
    _, rb = prepare_rails(stored["a"], stored["b"], 40, stored["ta"], stored["tb"])
    assert np.allclose(curve_pts, rb.points, atol=1e-9)   # nothing moved


def test_relax_can_skip_diagnosis(fresh_scene):
    obj = _loft(CASES["twisted"](n=60))
    assert bpy.ops.mino.relax(diagnose=False) == {"FINISHED"}
    assert bpy.data.objects["Mino.relaxed"]["mino_diagnosis"] == ""


def test_relax_requires_a_mino_object(fresh_scene):
    case = CASES["cylinder"](n=30)
    a = _make_poly_curve("A", case["points_a"])
    _select([a], a)
    assert not bpy.ops.mino.relax.poll()


def test_relax_accepts_a_time_budget(fresh_scene):
    obj = _loft(CASES["twisted"](n=60))
    assert bpy.ops.mino.relax(max_seconds=0.01) == {"FINISHED"}
    assert "Mino.relaxed" in bpy.data.objects and "Mino.railB.relaxed" in bpy.data.objects
