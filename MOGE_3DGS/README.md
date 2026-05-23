# MOGE_3DGS

Experiments: train a 3D Gaussian Splatting scene using MoGe-inferred depth
maps as supervision for the rendered depth channel.

The aim of this folder is **not** to ship a production trainer — it is to
produce a clean comparison between vanilla 3DGS and several MoGe-supervised
variants, so we can argue (one way or the other) whether monocular-depth
supervision actually helps 3DGS reconstruction quality.

## Layout

```
MOGE_3DGS/
├── README.md                  this file
├── preprocess/                MoGe inference + COLMAP alignment (TODO)
├── train/
│   ├── train.py               standalone 3DGS trainer with switchable depth loss
│   └── depth_losses.py        L1 / Pearson / masked / online-aligned variants
├── eval/                      held-out PSNR/SSIM/LPIPS + depth metrics (TODO)
├── configs/                   per-variant CLI presets (TODO)
└── scripts/                   end-to-end orchestration (TODO)
```

The trainer reuses the gaussian-splatting submodule at
`SRC/third_party/gaussian-splatting/` for `Scene`, `GaussianModel`,
`gaussian_renderer.render`, and friends. We add only the training loop and
the depth-loss switch.

## Variants implemented

See `train/depth_losses.py`:

| Variant                  | What it does                                                          |
|--------------------------|-----------------------------------------------------------------------|
| `none`                   | Photometric only (vanilla 3DGS baseline).                             |
| `l1_inv`                 | Raw L1 on inverse depth — no alignment (designed to show it breaks). |
| `l1_inv_aligned`         | L1 after submodule's precomputed per-image scale/offset (depth_params).|
| `l1_inv_solve_align`     | L1 after **online** per-view affine LS alignment of GT to render.     |
| `pearson_inv`            | 1 − Pearson on inverse depth (fully scale-free).                      |
| `pearson_depth`          | 1 − Pearson on depth (1 / invdepth).                                  |
| `masked_l1_inv_aligned`  | Edge-aware-weighted aligned L1 (down-weight near depth discontinuities).|

## Quickstart

After the preprocessing pipeline lands (cache MoGe + align to COLMAP):

```bash
# vanilla baseline
python MOGE_3DGS/train/train.py \
    -s /path/to/scene -m runs/scene/vanilla \
    --depth-loss none --iterations 30000

# best-effort MoGe-supervised variant
python MOGE_3DGS/train/train.py \
    -s /path/to/scene -m runs/scene/aligned_l1 \
    -d depths_moge \
    --depth-loss l1_inv_aligned --lambda-d 0.1 --depth-loss-until 20000
```

The scene directory must follow the gaussian-splatting layout:

```
<source>/images/...
<source>/sparse/0/{cameras,images,points3D}.bin
<source>/depths_moge/<name>.png       16-bit inverse-depth PNGs (from MoGe)
<source>/sparse/0/depth_params.json   per-image {scale, offset, med_scale}
```
