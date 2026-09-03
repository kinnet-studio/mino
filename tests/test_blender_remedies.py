import json

import numpy as np
import pytest

bpy = pytest.importorskip("bpy")

import mino  # noqa: E402
from mino.blender.remedies import diagnosis_rows  # noqa: E402
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


def _twisted_loft(samples=40):
    case = CASES["twisted"](n=60)
    a = _make_poly_curve("A", case["points_a"])
    b = _make_poly_curve("B", case["points_b"])
    _select([a, b], a)
    assert bpy.ops.mino.loft(samples=samples) == {"FINISHED"}
    obj = bpy.data.objects["Mino"]
    _select([obj], obj)
    return obj


def test_diagnosis_rows_maps_kinds_to_operators():
    diag = {
        "failing_ranges": [[10, 20]], "max_twist": 20.0, "window_fix": 16, "strakes_needed": 4,
        "strakes_worst_twist": 4.0, "crease_runs": 1,
        "darts": [{"range": [10, 20], "ruling": 15, "wedge_deg": -12.5, "kind": "gusset"}],
        "suggestions": [
            {"kind": "window", "text": "w", "params": {"window": 16}, "rank": 1},
            {"kind": "subdivide", "text": "s", "params": {"strakes": 4}, "rank": 2},
            {"kind": "dart", "text": "d", "params": {"ruling": 15, "wedge_deg": -12.5, "kind": "gusset"}, "rank": 3},
            {"kind": "creases", "text": "c", "params": {"runs": 1}, "rank": 4},
        ],
    }
    rows = diagnosis_rows({"mino_diagnosis": json.dumps(diag)})
    assert [r[1] for r in rows] == ["mino.reloft", "mino.subdivide", "mino.dart", None]
    assert rows[0][2] == {"window": 16} and rows[1][2] == {"strakes": 4} and rows[2][2] == {"ruling": 15}
    assert "gusset" in rows[2][0] and "12.5" in rows[2][0]
    assert diagnosis_rows({"mino_diagnosis": ""}) == [("Diagnosis off for this loft", None, {})]
    assert diagnosis_rows({}) == [("Diagnosis off for this loft", None, {})]


def test_reloft_with_window_creates_new_object(fresh_scene):
    obj = _twisted_loft()
    assert bpy.ops.mino.reloft(window=16) == {"FINISHED"}
    new = bpy.data.objects["Mino.reloft"]
    assert json.loads(new["mino_params"])["window"] == 16
    assert json.loads(new["mino_params"])["samples"] == 40
    assert json.loads(new["mino_params"])["planarize"] is True
    assert "mino_diagnosis" in new


def test_reloft_can_turn_off_planarize_and_diagnose(fresh_scene):
    obj = _twisted_loft()
    assert bpy.ops.mino.reloft(planarize=False, diagnose=False) == {"FINISHED"}
    new = bpy.data.objects["Mino.reloft"]
    params = json.loads(new["mino_params"])
    assert params["planarize"] is False and params["window"] == 8
    assert new["mino_diagnosis"] == ""


def test_subdivide_defaults_to_diagnosis_count(fresh_scene):
    obj = _twisted_loft()
    diag = json.loads(obj["mino_diagnosis"])
    expected = diag["strakes_needed"] or 2
    assert bpy.ops.mino.subdivide() == {"FINISHED"}
    assert f"Mino.strake.{expected - 1}" in bpy.data.objects
    assert f"Mino.strake.{expected}" not in bpy.data.objects


def test_load_inputs_rejects_missing_params(fresh_scene):
    from mino.blender.state import load_inputs
    from mino.core.errors import LoftError
    obj = _twisted_loft()
    del obj["mino_params"]
    with pytest.raises(LoftError):
        load_inputs(obj)


def test_subdivide_creates_strake_objects(fresh_scene):
    obj = _twisted_loft()
    assert bpy.ops.mino.subdivide(strakes=3) == {"FINISHED"}
    names = [f"Mino.strake.{k}" for k in range(3)]
    for k, name in enumerate(names):
        strip = bpy.data.objects[name]
        assert {d.value for d in strip.data.attributes["strake"].data} == {k}
        assert "mino_rails" in strip
    r0, r1 = json.loads(bpy.data.objects[names[0]]["mino_rails"]), json.loads(bpy.data.objects[names[1]]["mino_rails"])
    assert np.allclose(np.array(r0["b"]), np.array(r1["a"]))


def test_dart_creates_seam_edges(fresh_scene):
    obj = _twisted_loft()
    assert bpy.ops.mino.dart() == {"FINISHED"}
    dart = bpy.data.objects["Mino.dart"]
    me = dart.data
    assert "seam" in me.attributes and me.attributes["seam"].domain == "EDGE"
    seams = [e for e in me.edges if e.use_seam]
    assert len(seams) == 3
    assert sum(d.value for d in me.attributes["seam"].data) == 3
    assert "twist" in me.attributes and me.color_attributes.active_color is not None
    assert len(me.vertices) > 2 * 40


def test_remedies_require_a_mino_object(fresh_scene):
    case = CASES["cylinder"](n=30)
    a = _make_poly_curve("A", case["points_a"])
    _select([a], a)
    assert not bpy.ops.mino.subdivide.poll()
    assert not bpy.ops.mino.dart.poll()
    assert not bpy.ops.mino.reloft.poll()
