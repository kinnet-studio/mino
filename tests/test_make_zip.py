import zipfile

import make_zip


def test_zip_contains_manifest_and_package(tmp_path):
    path = make_zip.build(tmp_path)
    assert path.name == "mino-0.1.0.zip"
    with zipfile.ZipFile(path) as z:
        names = z.namelist()
    assert "blender_manifest.toml" in names
    assert "__init__.py" in names
    assert "core/__init__.py" in names
    assert "blender/operator.py" in names
    assert not any("__pycache__" in n for n in names)
