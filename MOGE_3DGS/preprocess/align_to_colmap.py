"""Align cached MoGe inverse-depth PNGs to COLMAP sparse depth.

Thin wrapper around the gaussian-splatting submodule's `make_depth_scale.py`,
which fits a per-image (scale, offset) by least-squares against
COLMAP-reprojected sparse-point depths, then writes the result to:

    <scene>/sparse/0/depth_params.json

That JSON is what the submodule's scene loader picks up: for each training
camera, the cached inverse-depth PNG is multiplied by `scale` and shifted by
`offset` before being handed to the depth loss. Frames whose fitted scale is
> 5x or < 0.2x the median are flagged unreliable and excluded from the depth
loss (the loader injects `med_scale` at load time — we don't have to).

Usage:
    python MOGE_3DGS/preprocess/align_to_colmap.py \
        --scene /path/to/colmap_scene \
        --depths-dir depths_moge \
        [--model-type bin]
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
_GS_DIR = _REPO_ROOT / "SRC" / "third_party" / "gaussian-splatting"
_MAKE_SCALE = _GS_DIR / "utils" / "make_depth_scale.py"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", required=True, type=Path,
                    help="COLMAP scene root (must contain sparse/0/).")
    ap.add_argument("--depths-dir", default="depths_moge",
                    help="Cached MoGe depth subdir under --scene.")
    ap.add_argument("--model-type", choices=("bin", "txt"), default="bin",
                    help="COLMAP model file extension (default: bin).")
    args = ap.parse_args()

    if not _MAKE_SCALE.is_file():
        sys.exit(f"missing upstream helper: {_MAKE_SCALE}")
    sparse_dir = args.scene / "sparse" / "0"
    if not sparse_dir.is_dir():
        sys.exit(f"COLMAP sparse dir not found: {sparse_dir}")
    depths_abs = (args.scene / args.depths_dir).resolve()
    if not depths_abs.is_dir():
        sys.exit(f"depths dir not found: {depths_abs} — run cache_moge_depth.py first.")

    # make_depth_scale.py imports `read_write_model` as a top-level module from
    # the same `utils/` folder it lives in — so we run it from that cwd.
    cmd = [
        sys.executable,
        str(_MAKE_SCALE),
        "--base_dir", str(args.scene.resolve()),
        "--depths_dir", str(depths_abs),
        "--model_type", args.model_type,
    ]
    print("[align_to_colmap] $ " + " ".join(cmd))
    env = dict(os.environ)
    # Ensure the helper can `from read_write_model import *`.
    pp = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = str(_MAKE_SCALE.parent) + (os.pathsep + pp if pp else "")
    r = subprocess.run(cmd, cwd=str(_MAKE_SCALE.parent), env=env)
    if r.returncode != 0:
        sys.exit(f"make_depth_scale.py exited with code {r.returncode}")

    out = sparse_dir / "depth_params.json"
    if not out.is_file():
        sys.exit(f"depth_params.json was not produced at {out}")
    print(f"[align_to_colmap] wrote {out}")


if __name__ == "__main__":
    main()
