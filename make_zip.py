"""Build dist/mino-<version>.zip for Install from Disk in Blender 4.2+."""
from __future__ import annotations

import tomllib
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "mino"


def build(dist_dir: Path = ROOT / "dist") -> Path:
    manifest = tomllib.loads((SRC / "blender_manifest.toml").read_text())
    dist_dir = Path(dist_dir)
    dist_dir.mkdir(parents=True, exist_ok=True)
    out = dist_dir / f"{manifest['id']}-{manifest['version']}.zip"
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        for p in sorted(SRC.rglob("*")):
            if p.is_dir() or "__pycache__" in p.parts or p.suffix == ".pyc":
                continue
            z.write(p, p.relative_to(SRC).as_posix())
    return out


if __name__ == "__main__":
    print(build())
