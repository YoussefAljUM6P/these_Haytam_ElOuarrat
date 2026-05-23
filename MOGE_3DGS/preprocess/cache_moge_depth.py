"""Run MoGe on every image in a COLMAP scene → cache 16-bit inverse-depth PNGs.

Output layout (matches the gaussian-splatting depth-supervision convention):
    <scene>/<out_dir>/<image_stem>.png         uint16 inverse-depth, [0, 65535]
    <scene>/<out_dir>/_meta.json                per-image inv_min, inv_max, valid_count
    <scene>/<out_dir>/masks/<image_stem>.png    uint8 valid mask (optional)
    <scene>/<out_dir>/depth_raw/<stem>.npz     metric depth + mask (optional, --save-raw)

Normalization (exact same protocol the submodule expects):
    inv_depth_pixel = 1 / depth           (with masked / infinite pixels = 0)
    inv_norm        = inv_depth_pixel / max_inv     in [0, 1]
    png_uint16      = round(inv_norm * (2**16 - 1)) in [0, 65535]
The downstream loader reads the PNG, divides by 2^16, and applies the
per-image (scale, offset) from depth_params.json. We *do not* clip MoGe
depth here — alignment to COLMAP happens in align_to_colmap.py.

Usage:
    python MOGE_3DGS/preprocess/cache_moge_depth.py \
        --scene /path/to/colmap_scene \
        --out-dir depths_moge \
        [--model-version v2] [--resolution-level 9] [--fp16] [--save-raw]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from tqdm import tqdm

_REPO_ROOT = Path(__file__).resolve().parents[2]
_MOGE_DIR = _REPO_ROOT / "SRC" / "third_party" / "MoGe"
if str(_MOGE_DIR) not in sys.path:
    sys.path.insert(0, str(_MOGE_DIR))

from moge.model import import_model_class_by_version  # noqa: E402


DEFAULT_PRETRAINED = {
    "v1": "Ruicheng/moge-vitl",
    "v2": "Ruicheng/moge-2-vitl-normal",
}
IMG_SUFFIXES = (".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG")


def list_images(images_dir: Path) -> list[Path]:
    out = []
    for p in sorted(images_dir.rglob("*")):
        if p.is_file() and p.suffix in IMG_SUFFIXES:
            out.append(p)
    return out


def to_inv_depth_uint16(depth: np.ndarray, mask: np.ndarray | None) -> tuple[np.ndarray, float]:
    """Convert metric depth (H, W) → uint16 inverse-depth PNG payload.

    Returns (uint16_image, inv_max) where inv_max is the value used to
    normalize. Invalid pixels (mask == False or depth <= 0 or not finite)
    are written as 0 — the downstream loader treats those as invalid via
    its own mask handling.
    """
    valid = np.isfinite(depth) & (depth > 0)
    if mask is not None:
        valid = valid & mask
    inv = np.zeros_like(depth, dtype=np.float32)
    inv[valid] = 1.0 / depth[valid]
    if valid.sum() > 0:
        inv_max = float(np.percentile(inv[valid], 99.5))
        if inv_max <= 0:
            inv_max = float(inv[valid].max())
    else:
        inv_max = 1.0
    if inv_max <= 0:
        inv_max = 1.0
    inv_norm = np.clip(inv / inv_max, 0.0, 1.0)
    return (inv_norm * 65535.0 + 0.5).astype(np.uint16), inv_max


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", required=True, type=Path,
                    help="COLMAP scene root (must contain images/).")
    ap.add_argument("--images-subdir", default="images",
                    help="Image subdir under --scene (default: images).")
    ap.add_argument("--out-dir", default="depths_moge",
                    help="Output subdir under --scene (default: depths_moge).")
    ap.add_argument("--model-version", choices=("v1", "v2"), default="v2")
    ap.add_argument("--pretrained", default=None,
                    help="HF model id or path; defaults per --model-version.")
    ap.add_argument("--resolution-level", type=int, default=9)
    ap.add_argument("--num-tokens", type=int, default=None)
    ap.add_argument("--fp16", action="store_true")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--save-raw", action="store_true",
                    help="Also dump metric depth + mask as .npz alongside PNGs.")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    images_dir = args.scene / args.images_subdir
    if not images_dir.is_dir():
        sys.exit(f"images dir not found: {images_dir}")
    out_root = args.scene / args.out_dir
    masks_dir = out_root / "masks"
    raw_dir = out_root / "depth_raw"
    out_root.mkdir(parents=True, exist_ok=True)
    masks_dir.mkdir(parents=True, exist_ok=True)
    if args.save_raw:
        raw_dir.mkdir(parents=True, exist_ok=True)

    pretrained = args.pretrained or DEFAULT_PRETRAINED[args.model_version]
    device = torch.device(args.device)
    print(f"[cache_moge] loading {pretrained} on {device}")
    model = (
        import_model_class_by_version(args.model_version)
        .from_pretrained(pretrained)
        .to(device)
        .eval()
    )
    if args.fp16:
        model.half()

    images = list_images(images_dir)
    if not images:
        sys.exit(f"no images found under {images_dir}")
    print(f"[cache_moge] {len(images)} images -> {out_root}")

    meta: dict[str, dict] = {}
    for img_path in tqdm(images, desc="MoGe"):
        rel = img_path.relative_to(images_dir).with_suffix(".png")
        out_png = out_root / rel
        mask_png = masks_dir / rel
        raw_npz = raw_dir / rel.with_suffix(".npz") if args.save_raw else None
        if out_png.exists() and not args.overwrite:
            continue
        out_png.parent.mkdir(parents=True, exist_ok=True)
        mask_png.parent.mkdir(parents=True, exist_ok=True)
        if raw_npz is not None:
            raw_npz.parent.mkdir(parents=True, exist_ok=True)

        bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if bgr is None:
            print(f"[cache_moge] skip unreadable: {img_path}")
            continue
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        H, W = rgb.shape[:2]
        img_t = torch.from_numpy(rgb).float().div(255.0).permute(2, 0, 1).to(device)

        out = model.infer(
            img_t,
            resolution_level=args.resolution_level,
            num_tokens=args.num_tokens,
            use_fp16=args.fp16,
            apply_mask=False,
        )
        depth = out["depth"].detach().float().cpu().numpy()
        mask_t = out.get("mask")
        mask_np = (mask_t.detach().cpu().numpy() > 0.5) if mask_t is not None else None
        if mask_np is not None and mask_np.dtype != bool:
            mask_np = mask_np.astype(bool)

        # MoGe sometimes returns at a slightly different resolution after token
        # rounding; resize back to the original image grid.
        if depth.shape != (H, W):
            depth = cv2.resize(depth, (W, H), interpolation=cv2.INTER_NEAREST)
            if mask_np is not None:
                mask_np = cv2.resize(
                    mask_np.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST
                ).astype(bool)

        u16, inv_max = to_inv_depth_uint16(depth, mask_np)
        cv2.imwrite(str(out_png), u16)
        if mask_np is not None:
            cv2.imwrite(str(mask_png), (mask_np.astype(np.uint8) * 255))
        if raw_npz is not None:
            np.savez_compressed(raw_npz, depth=depth.astype(np.float32),
                                mask=(mask_np if mask_np is not None else np.ones_like(depth, dtype=bool)))

        valid_n = int(mask_np.sum()) if mask_np is not None else int(np.isfinite(depth).sum())
        meta[str(rel.with_suffix(""))] = {
            "inv_max": inv_max,
            "valid_pixels": valid_n,
            "H": int(H),
            "W": int(W),
        }

    with open(out_root / "_meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(f"[cache_moge] wrote {len(meta)} entries to {out_root}/_meta.json")


if __name__ == "__main__":
    main()
