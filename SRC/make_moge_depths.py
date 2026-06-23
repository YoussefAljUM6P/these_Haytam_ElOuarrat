"""Generate mono inverse-depth maps with MoGe-2 for 3DGS depth regularization.

Drop-in replacement for Depth-Anything-V2's `run.py` in the official 3DGS
depth-regularization recipe. It writes one grayscale PNG of normalized
*inverse depth* per input image (brightest = nearest), exactly the artifact
`utils/make_depth_scale.py` and `train.py -d <dir>` expect.

Pipeline (mirrors the Inria docs, MoGe-2 substituted for Depth-Anything-V2):

    python SRC/make_moge_depths.py --img-path <scene>/images --outdir <scene>/depths
    python SRC/third_party/gaussian-splatting/utils/make_depth_scale.py \
        --base_dir <scene> --depths_dir <scene>/depths
    python SRC/third_party/gaussian-splatting/train.py -s <scene> -d depths

Why inverse depth: GS compares the rendered inverse depth
(`render_pkg["depth"]`) against this map, and `make_depth_scale.py` aligns it
to the COLMAP sparse points via `invcolmapdepth = 1/z`. MoGe-2 returns *metric
depth*, so we invert it. The absolute scale is irrelevant — `make_depth_scale`
re-fits a per-image scale+offset — so we min-max normalize each map on its own.

Invalid pixels: MoGe-2 flags sky / uncertain regions. This GS version applies
the depth loss over the whole image (no per-pixel mask, see scene/cameras.py),
so we set invalid pixels to the *minimum* valid inverse depth (treated as
"far"), which is harmless, rather than leaving NaN/inf to poison the loss.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".JPG", ".JPEG", ".PNG")
DEFAULT_MODEL = "Ruicheng/moge-2-vitl-normal"


def _load_moge(model_id: str):
    """Load MoGe-2 once, resolving the vendored package like SRC/depth.py does."""
    moge_root = Path(__file__).resolve().parent / "third_party" / "moge"
    if not moge_root.exists():
        moge_root = Path(__file__).resolve().parent / "third_party" / "MoGe"
    if str(moge_root) not in sys.path:
        sys.path.insert(0, str(moge_root))

    # MoGe disables cudnn on this setup to dodge CUDNN_STATUS_NOT_INITIALIZED.
    torch.backends.cudnn.enabled = False
    from moge.model.v2 import MoGeModel

    return MoGeModel.from_pretrained(model_id).to("cuda").eval()


def _inverse_depth_png(model, image_path: Path) -> np.ndarray:
    """Return a uint16 grayscale inverse-depth map for one image."""
    bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise IOError(f"could not read image: {image_path}")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

    image_t = torch.as_tensor(rgb, dtype=torch.float32, device="cuda").permute(2, 0, 1)
    with torch.inference_mode():
        out = model.infer(image_t)

    depth = out["depth"].detach().cpu().numpy().astype(np.float32)
    # Valid = finite, positive depth, and (if MoGe gives one) inside its mask.
    valid = np.isfinite(depth) & (depth > 0)
    mask = out.get("mask")
    if mask is not None:
        valid &= mask.detach().cpu().numpy().astype(bool)
    if not valid.any():
        raise RuntimeError(f"MoGe produced no valid depth for {image_path}")

    invdepth = np.zeros_like(depth)
    invdepth[valid] = 1.0 / depth[valid]

    lo = float(invdepth[valid].min())
    hi = float(invdepth[valid].max())
    span = max(hi - lo, 1e-12)
    norm = (invdepth - lo) / span          # valid -> [0, 1]
    norm[~valid] = 0.0                       # invalid/sky -> far (min disparity)
    norm = np.clip(norm, 0.0, 1.0)
    return (norm * float(2**16 - 1)).astype(np.uint16)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--img-path", required=True,
                    help="directory of input images (or a single image file)")
    ap.add_argument("--outdir", required=True, help="output directory for depth PNGs")
    ap.add_argument("--model", default=DEFAULT_MODEL, help="MoGe-2 HF model id")
    ap.add_argument("--pattern", default=None,
                    help="glob to select inputs in --img-path (e.g. '*.color.jpg'); "
                         "default = all common image extensions. Use this when the "
                         "directory also contains depth PNGs you must not feed to MoGe.")
    args = ap.parse_args(argv)

    img_path = Path(args.img_path)
    if img_path.is_file():
        images = [img_path]
    elif args.pattern:
        images = sorted(p for p in img_path.glob(args.pattern) if p.is_file())
    else:
        images = sorted(p for p in img_path.iterdir()
                        if p.is_file() and p.suffix in IMAGE_EXTS)
    if not images:
        print(f"No images found under {img_path}")
        return 1

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    print(f"Loading {args.model} ...")
    model = _load_moge(args.model)

    print(f"Generating {len(images)} inverse-depth maps -> {outdir}")
    for i, image_path in enumerate(images, 1):
        png = _inverse_depth_png(model, image_path)
        # make_depth_scale.py looks up {depths_dir}/{image_stem}.png
        out_path = outdir / f"{image_path.stem}.png"
        cv2.imwrite(str(out_path), png)
        print(f"  [{i}/{len(images)}] {image_path.name} -> {out_path.name}")

    print("Done. Next: make_depth_scale.py, then train.py -d <depths>.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
