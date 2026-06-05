"""Publish trained Mip-360 scenes into the SERVIS DATA/ layout.

Each Mip-360 scene already ships everything SERVIS needs except the renderer
asset: a PINHOLE `sparse/0` (poses + points) and full-res `images/` plus
official downsamples `images_{2,4,8}/`. This script does the two remaining
steps, per scene:

  1. Downscale the reconstruction to a chosen pyramid level N (2/4/8) by
     rewriting `sparse/0` camera intrinsics to match the `images_N/` pixel
     dims, and repointing `images/` at `images_N/`. Poses and points3D are
     metric and resolution-independent, so they are copied unchanged. SERVIS
     then loads/servos at the reduced resolution with no code change.
  2. Publish `gs.ply` from the trained Gaussians at
     `output/point_cloud/iteration_<ITER>/point_cloud.ply`.

It is reversible (originals saved to `sparse_full/` and `images_full/`) and
idempotent (downscaling always derives from `sparse_full/` when present, so you
can re-run with a different --level). Scenes whose `output/` has not been synced
from the cluster yet are skipped with a message instead of failing.

Usage:
    python make_mip360_scenes.py --level 4 \
        --scenes bicycle bonsai counter flowers garden room stump treehill
    python make_mip360_scenes.py --scenes garden --level 2 --copy
    python make_mip360_scenes.py --all                  # every scene with images_N
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import pycolmap
import numpy as np
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = PROJECT_ROOT / "DATA"


def first_image_size(images_dir: Path):
    """Return (width, height) of the first readable image in a directory."""
    for entry in sorted(images_dir.iterdir()):
        if entry.suffix.lower() in {".jpg", ".jpeg", ".png"}:
            with Image.open(entry) as im:
                return im.size  # (W, H)
    raise FileNotFoundError(f"No image files in {images_dir}")


def downscale_sparse(scene_dir: Path, level: int):
    """Rewrite sparse/0 to the resolution of images_<level>/, reversibly."""
    full_sparse = scene_dir / "sparse_full"
    live_sparse = scene_dir / "sparse"

    # Always derive from the pristine full-res model so re-runs are clean.
    if not full_sparse.exists():
        shutil.copytree(live_sparse, full_sparse)
        print(f"  backed up full-res sparse -> {full_sparse.relative_to(scene_dir)}/")
    src_model = full_sparse / "0"

    images_n = scene_dir / f"images_{level}"
    if not images_n.is_dir():
        raise FileNotFoundError(f"Missing {images_n} (needed for --level {level})")

    full_images = scene_dir / "images_full"
    live_images = scene_dir / "images"
    if not full_images.exists() and live_images.is_dir() and not live_images.is_symlink():
        live_images.rename(full_images)
        print(f"  backed up full-res images -> {full_images.relative_to(scene_dir)}/")

    # Exact scale from real pixel dims (official downsamples round per-axis).
    fw, fh = first_image_size(full_images if full_images.exists() else live_images)
    nw, nh = first_image_size(images_n)
    sx, sy = nw / fw, nh / fh
    print(f"  level {level}: {fw}x{fh} -> {nw}x{nh}  (sx={sx:.4f} sy={sy:.4f})")

    rec = pycolmap.Reconstruction(src_model)
    for cam in rec.cameras.values():
        if cam.model.name != "PINHOLE":
            raise NotImplementedError(
                f"{scene_dir.name}: expected PINHOLE camera, got {cam.model.name}"
            )
        fx, fy, cx, cy = (np.asarray(cam.params, dtype=float))
        cam.params = np.array([fx * sx, fy * sy, cx * sx, cy * sy], dtype=float)
        cam.width = nw
        cam.height = nh

    # Write fresh sparse/0.
    out_model = live_sparse / "0"
    if live_sparse.is_symlink():
        live_sparse.unlink()
    if live_sparse.exists():
        shutil.rmtree(live_sparse)
    out_model.mkdir(parents=True)
    rec.write(out_model)

    # Repoint images/ at the downsampled set (filenames match COLMAP names).
    if live_images.is_symlink() or live_images.exists():
        if live_images.is_symlink():
            live_images.unlink()
        elif live_images.is_dir():
            shutil.rmtree(live_images)
    live_images.symlink_to(Path(f"images_{level}"))
    print(f"  images/ -> images_{level}/   sparse/0 rewritten")


def publish_gs(scene_dir: Path, iteration: int, copy: bool):
    """Link or copy the trained Gaussians to <scene>/gs.ply."""
    src = scene_dir / "output" / "point_cloud" / f"iteration_{iteration}" / "point_cloud.ply"
    if not src.is_file():
        print(
            f"  [skip gs.ply] no trained output at "
            f"{src.relative_to(scene_dir)} — sync output/ from the cluster first",
            file=sys.stderr,
        )
        return False

    dst = scene_dir / "gs.ply"
    if dst.is_symlink() or dst.exists():
        dst.unlink()
    if copy:
        shutil.copyfile(src, dst)
        print(f"  gs.ply copied from iteration_{iteration}")
    else:
        dst.symlink_to(src.relative_to(scene_dir))
        print(f"  gs.ply -> {src.relative_to(scene_dir)}")
    return True


def discover_scenes():
    return [
        d.name
        for d in sorted(DATA_ROOT.iterdir())
        if d.is_dir() and (d / "sparse" / "0").is_dir() and list(d.glob("images_*"))
    ]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scenes", nargs="+", help="Scene names under DATA/.")
    ap.add_argument("--all", action="store_true", help="All scenes with images_*/ pyramids.")
    ap.add_argument("--level", type=int, default=4, choices=(1, 2, 4, 8),
                    help="Downsample level (matches images_N/). 1 = keep full res.")
    ap.add_argument("--iteration", type=int, default=30000, help="GS iteration to publish.")
    ap.add_argument("--copy", action="store_true", help="Copy gs.ply instead of symlinking.")
    ap.add_argument("--no-downscale", action="store_true", help="Only publish gs.ply.")
    args = ap.parse_args()

    scenes = args.scenes or (discover_scenes() if args.all else None)
    if not scenes:
        ap.error("pass --scenes <names...> or --all")

    for name in scenes:
        scene_dir = DATA_ROOT / name
        print(f"== {name} ==")
        if not (scene_dir / "sparse").exists() and not (scene_dir / "sparse_full").exists():
            print(f"  [skip] no sparse/ in {scene_dir}", file=sys.stderr)
            continue
        if not args.no_downscale and args.level != 1:
            downscale_sparse(scene_dir, args.level)
        publish_gs(scene_dir, args.iteration, args.copy)


if __name__ == "__main__":
    main()
