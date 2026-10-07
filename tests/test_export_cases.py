import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools" / "viewer"))

import export_cases  # noqa: E402


def test_export_writes_data_js_and_json(tmp_path):
    written = export_cases.main(tmp_path)
    names = {p.name for p in written}
    assert "data.js" in names
    assert {"cylinder.json", "cone.json", "offset_cylinder.json", "twisted.json", "ellipse.json"} <= names
    js = (tmp_path / "data.js").read_text()
    assert js.startswith("window.MINO_CASES = [")
    payload = json.loads(js[len("window.MINO_CASES = "):].rstrip().rstrip(";"))
    assert [c["name"] for c in payload] == ["cylinder", "cone", "offset_cylinder", "twisted", "ellipse"]
    cyl = json.loads((tmp_path / "cylinder.json").read_text())
    assert cyl["report"]["failing_ruling_count"] == 0
    assert len(cyl["ruling_verts"]) == len(cyl["rulings"])
