"""Stored inputs on Mino result objects, so remedies can act on 'this result'."""
from __future__ import annotations

import json
from dataclasses import asdict, fields

import numpy as np

from ..core import LoftParams, format_report
from ..core.diagnose import Diagnosis, diagnosis_from_dict, diagnosis_to_dict
from ..core.errors import LoftError


def params_to_dict(params: LoftParams) -> dict:
    data = asdict(params)
    data["plane_normal"] = [float(x) for x in data["plane_normal"]]
    return data


def params_from_dict(data: dict) -> LoftParams:
    names = {f.name for f in fields(LoftParams)}
    kwargs = {k: v for k, v in data.items() if k in names}
    if "plane_normal" in kwargs:
        kwargs["plane_normal"] = tuple(float(x) for x in kwargs["plane_normal"])
    return LoftParams(**kwargs)


def _opt(arr):
    return None if arr is None else np.asarray(arr, dtype=float).tolist()


def store_inputs(obj, points_a, tangents_a, points_b, tangents_b, params, result, diagnosis):
    obj["mino_rails"] = json.dumps({"a": _opt(points_a), "ta": _opt(tangents_a),
                                    "b": _opt(points_b), "tb": _opt(tangents_b)})
    obj["mino_params"] = json.dumps(params_to_dict(params))
    obj["mino_report"] = format_report(result.report, result.failing_ranges)
    obj["mino_diagnosis"] = json.dumps(diagnosis_to_dict(diagnosis)) if diagnosis is not None else ""


def _arr(value):
    return None if value is None else np.asarray(value, dtype=float)


def load_inputs(obj):
    if "mino_rails" not in obj:
        raise LoftError(f"'{obj.name}' is not a Mino result object")
    if "mino_params" not in obj:
        raise LoftError(f"'{obj.name}' has no stored parameters")
    rails = json.loads(obj["mino_rails"])
    params = params_from_dict(json.loads(obj["mino_params"]))
    return _arr(rails["a"]), _arr(rails["ta"]), _arr(rails["b"]), _arr(rails["tb"]), params


def load_diagnosis(obj) -> Diagnosis | None:
    raw = obj.get("mino_diagnosis", "")
    return diagnosis_from_dict(json.loads(raw)) if raw else None
