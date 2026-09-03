"""Run the core on the test rail sets and write viewer data.

Usage: uv run python tools/viewer/export_cases.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mino.core import LoftParams, loft  # noqa: E402
from mino.core.export import result_to_dict  # noqa: E402
from tests.cases import CASES  # noqa: E402


def main(out_dir: Path = HERE) -> list[Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    payload = []
    for name, build in CASES.items():
        case = build()
        params = LoftParams(**case["params"])
        result = loft(case["points_a"], case["points_b"], params, case["tangents_a"], case["tangents_b"])
        data = result_to_dict(result, case["points_a"], case["points_b"], name, params)
        payload.append(data)
        path = out_dir / f"{name}.json"
        path.write_text(json.dumps(data))
        written.append(path)
        print(f"{name}: {data['report']}")
    js = out_dir / "data.js"
    js.write_text("window.MINO_CASES = " + json.dumps(payload) + ";\n")
    written.append(js)
    return written


if __name__ == "__main__":
    main()
