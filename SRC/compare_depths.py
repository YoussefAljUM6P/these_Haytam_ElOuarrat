import argparse
import csv
import json
import random
from pathlib import Path

import numpy as np
from PIL import Image

from dataset import load_colmap
from depth import estimate_depth_moge
from scenes.gs import GSScene


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def read_colmap_depth(path):
    with open(path, "rb") as fid:
        width, height, channels = np.genfromtxt(
            fid, delimiter="&", max_rows=1, usecols=(0, 1, 2), dtype=int
        )
        fid.seek(0)
        num_delimiter = 0
        while True:
            byte = fid.read(1)
            if byte == b"":
                raise ValueError(f"Malformed COLMAP depth map header: {path}")
            if byte == b"&":
                num_delimiter += 1
                if num_delimiter >= 3:
                    break
        array = np.fromfile(fid, np.float32)
    array = array.reshape((width, height, channels), order="F")
    return np.transpose(array, (1, 0, 2)).squeeze().astype(np.float32, copy=False)


def colmap_depth_dirs(scene_dir, explicit_dir=None):
    if explicit_dir is not None:
        return [Path(explicit_dir)]

    scene_dir = Path(scene_dir)
    return [
        scene_dir / "mesh_v2" / "dense" / "stereo" / "depth_maps",
        scene_dir / "mesh" / "mvs" / "stereo" / "depth_maps",
        scene_dir / "dense" / "stereo" / "depth_maps",
        scene_dir / "stereo" / "depth_maps",
    ]


def find_colmap_depth_path(scene_dir, image_name, depth_kind, explicit_dir=None):
    names = [image_name, Path(image_name).name]
    for depth_dir in colmap_depth_dirs(scene_dir, explicit_dir):
        for name in names:
            candidate = depth_dir / f"{name}.{depth_kind}.bin"
            if candidate.exists():
                return candidate
    return None


def resize_depth_nearest(depth, target_shape):
    target_h, target_w = target_shape
    if depth.shape == (target_h, target_w):
        return depth.astype(np.float32, copy=False)

    image = Image.fromarray(depth.astype(np.float32, copy=False), mode="F")
    image = image.resize((target_w, target_h), Image.Resampling.NEAREST)
    return np.array(image, dtype=np.float32)


def load_real_rgb(rgb_path, target_shape):
    target_h, target_w = target_shape
    image = Image.open(rgb_path).convert("RGB").resize((target_w, target_h))
    return np.array(image, dtype=np.float32) / 255.0


def valid_depth_mask(pred_depth, gt_depth):
    return (
        np.isfinite(pred_depth)
        & np.isfinite(gt_depth)
        & (pred_depth > 0.0)
        & (gt_depth > 0.0)
    )


def compute_metric_values(pred, gt):
    diff = pred - gt
    abs_diff = np.abs(diff)
    ratio = np.maximum(pred / gt, gt / pred)

    return {
        "mae": float(abs_diff.mean()),
        "rmse": float(np.sqrt((diff ** 2).mean())),
        "median_abs_error": float(np.median(abs_diff)),
        "bias": float(diff.mean()),
        "abs_rel": float((abs_diff / gt).mean()),
        "sq_rel": float(((diff ** 2) / gt).mean()),
        "delta_1_25": float((ratio < 1.25).mean()),
        "delta_1_25_2": float((ratio < 1.25 ** 2).mean()),
        "delta_1_25_3": float((ratio < 1.25 ** 3).mean()),
        "pred_mean": float(pred.mean()),
        "gt_mean": float(gt.mean()),
    }


def prefixed_metrics(prefix, metrics):
    return {f"{prefix}_{key}": value for key, value in metrics.items()}


def compute_metrics(pred_depth, gt_depth):
    mask = valid_depth_mask(pred_depth, gt_depth)
    valid_count = int(mask.sum())
    total_count = int(mask.size)
    if valid_count == 0:
        raise ValueError("No overlapping valid depth pixels to compare")

    pred = pred_depth[mask].astype(np.float64)
    gt = gt_depth[mask].astype(np.float64)
    raw_metrics = compute_metric_values(pred, gt)

    pred_median = float(np.median(pred))
    gt_median = float(np.median(gt))
    if pred_median <= 0.0 or not np.isfinite(pred_median):
        raise ValueError("Predicted depth median is invalid; cannot compute scale diagnostic")
    median_scale = gt_median / pred_median
    aligned_metrics = compute_metric_values(pred * median_scale, gt)

    metrics = {
        "valid_pixels": valid_count,
        "total_pixels": total_count,
        "valid_fraction": float(valid_count / total_count),
        "median_scale_pred_to_gt": float(median_scale),
    }
    metrics.update(raw_metrics)
    metrics.update(prefixed_metrics("median_aligned", aligned_metrics))
    return metrics


def save_rgb(image, path):
    image_u8 = np.clip(image * 255.0, 0, 255).astype(np.uint8)
    Image.fromarray(image_u8).save(path)


def save_depth_vis(depth, path, valid_mask=None):
    if valid_mask is None:
        valid_mask = np.isfinite(depth) & (depth > 0.0)

    vis = np.zeros(depth.shape, dtype=np.uint8)
    if valid_mask.any():
        values = depth[valid_mask]
        lo, hi = np.percentile(values, [2, 98])
        if hi <= lo:
            hi = lo + 1e-6
        normalized = np.clip((depth - lo) / (hi - lo), 0.0, 1.0)
        vis[valid_mask] = (normalized[valid_mask] * 255.0).astype(np.uint8)

    Image.fromarray(vis).save(path)


def save_mask(mask, path):
    Image.fromarray((mask.astype(np.uint8) * 255)).save(path)


def save_sample_outputs(output_dir, prefix, rendered, real_rgb, pred_depth, gt_depth):
    output_dir.mkdir(parents=True, exist_ok=True)
    mask = valid_depth_mask(pred_depth, gt_depth)
    abs_error = np.zeros_like(pred_depth, dtype=np.float32)
    abs_error[mask] = np.abs(pred_depth[mask] - gt_depth[mask])

    save_rgb(rendered, output_dir / f"{prefix}_render.png")
    if real_rgb is not None:
        save_rgb(real_rgb, output_dir / f"{prefix}_real.png")
    save_depth_vis(pred_depth, output_dir / f"{prefix}_moge_depth.png")
    save_depth_vis(gt_depth, output_dir / f"{prefix}_gt_depth.png")
    save_depth_vis(abs_error, output_dir / f"{prefix}_abs_error.png", mask)
    save_mask(mask, output_dir / f"{prefix}_valid_mask.png")

    np.save(output_dir / f"{prefix}_moge_depth.npy", pred_depth.astype(np.float32))
    np.save(output_dir / f"{prefix}_gt_depth.npy", gt_depth.astype(np.float32))


def evaluate_colmap_item(
    output_dir,
    scene,
    item,
    depth_kind,
    pair_frame=None,
):
    camera, rgb_path, depth_path = item
    rendered = scene.render(camera)
    pred_depth = estimate_depth_moge(rendered)
    gt_depth = resize_depth_nearest(read_colmap_depth(depth_path), pred_depth.shape)
    real_rgb = load_real_rgb(rgb_path, rendered.shape[:2])

    frame = rgb_path.name
    prefix = f"gs_colmap_{depth_path.name.replace('.' + depth_kind + '.bin', '')}"
    save_sample_outputs(output_dir, prefix, rendered, real_rgb, pred_depth, gt_depth)

    metrics = compute_metrics(pred_depth, gt_depth)
    metrics.update({
        "source": "gs_colmap",
        "frame": frame,
        "pair_frame": pair_frame,
        "rgb_path": str(rgb_path),
        "gt_depth_path": str(depth_path),
        "colmap_depth_kind": depth_kind,
        "rendered_path": str(output_dir / f"{prefix}_render.png"),
        "moge_depth_path": str(output_dir / f"{prefix}_moge_depth.npy"),
        "gt_depth_npy_path": str(output_dir / f"{prefix}_gt_depth.npy"),
    })
    return metrics

def sample_colmap(
    scene_dir,
    output_dir,
    rng,
    depth_kind,
    colmap_depth_dir,
):
    candidates = []
    for camera, rgb_path in load_colmap(scene_dir):
        depth_path = find_colmap_depth_path(
            scene_dir, rgb_path.name, depth_kind, explicit_dir=colmap_depth_dir
        )
        if depth_path is not None:
            candidates.append((camera, rgb_path, depth_path))

    if not candidates:
        raise RuntimeError(
            f"No COLMAP {depth_kind!r} dense depth maps matched COLMAP images in {scene_dir}"
        )

    camera, rgb_path, depth_path = rng.choice(candidates)
    scene = GSScene(Path(scene_dir) / "gs.ply")
    metrics = evaluate_colmap_item(
        output_dir,
        scene,
        (camera, rgb_path, depth_path),
        depth_kind,
    )
    metrics["source"] = "colmap"
    return metrics


def colmap_depth_candidates(scene_dir, depth_kind, colmap_depth_dir):
    candidates = {}
    for camera, rgb_path in load_colmap(scene_dir):
        depth_path = find_colmap_depth_path(
            scene_dir, rgb_path.name, depth_kind, explicit_dir=colmap_depth_dir
        )
        if depth_path is not None:
            candidates[rgb_path.name] = (camera, rgb_path, depth_path)
    return candidates


def summarize_metrics(rows):
    metric_keys = [
        "valid_fraction",
        "mae",
        "rmse",
        "median_abs_error",
        "bias",
        "abs_rel",
        "sq_rel",
        "delta_1_25",
        "delta_1_25_2",
        "delta_1_25_3",
        "median_scale_pred_to_gt",
        "median_aligned_mae",
        "median_aligned_rmse",
        "median_aligned_median_abs_error",
        "median_aligned_bias",
        "median_aligned_abs_rel",
        "median_aligned_sq_rel",
        "median_aligned_delta_1_25",
        "median_aligned_delta_1_25_2",
        "median_aligned_delta_1_25_3",
    ]
    summary = {}
    for key in metric_keys:
        values = np.array([row[key] for row in rows], dtype=np.float64)
        summary[key] = {
            "mean": float(values.mean()),
            "std": float(values.std(ddof=0)),
        }
    return summary


def summarize_metrics_by_source(rows):
    summary = {}
    for source in sorted({row["source"] for row in rows}):
        source_rows = [row for row in rows if row["source"] == source]
        summary[source] = summarize_metrics(source_rows)
    return summary


def report_row(row):
    return (
        f"| {row['source']} | {row['frame']} | "
        f"{row.get('pair_frame', '') or ''} | "
        f"{row['valid_fraction'] * 100.0:.2f} | "
        f"{row['mae']:.4f} | "
        f"{row['rmse']:.4f} | "
        f"{row['abs_rel']:.4f} | "
        f"{row['median_aligned_mae']:.4f} | "
        f"{row['median_aligned_rmse']:.4f} | "
        f"{row['median_aligned_abs_rel']:.4f} | "
        f"{row['median_scale_pred_to_gt']:.4f} |"
    )


def report_summary_row(source, rows):
    valid = np.array([row["valid_fraction"] for row in rows], dtype=np.float64)
    mae = np.array([row["mae"] for row in rows], dtype=np.float64)
    rmse = np.array([row["rmse"] for row in rows], dtype=np.float64)
    abs_rel = np.array([row["abs_rel"] for row in rows], dtype=np.float64)
    aligned_mae = np.array([row["median_aligned_mae"] for row in rows], dtype=np.float64)
    aligned_rmse = np.array([row["median_aligned_rmse"] for row in rows], dtype=np.float64)
    aligned_abs_rel = np.array([row["median_aligned_abs_rel"] for row in rows], dtype=np.float64)

    return (
        f"| {source} | {len(rows)} | "
        f"{valid.mean() * 100.0:.2f} | "
        f"{mae.mean():.4f} | "
        f"{rmse.mean():.4f} | "
        f"{abs_rel.mean():.4f} | "
        f"{aligned_mae.mean():.4f} | "
        f"{aligned_rmse.mean():.4f} | "
        f"{aligned_abs_rel.mean():.4f} |"
    )


def write_report(output_dir, rows, calibration=None):
    lines = [
        "# Depth Comparison Report",
        "",
        "Scene-scale metrics are paper-facing absolute depth differences. "
        "Median-aligned metrics are diagnostic relative-shape errors after one "
        "per-image scalar is applied to MoGe2 depth.",
        "",
    ]

    lines.extend([
        "## Per-Sample Metrics",
        "",
        "| Source | Frame | Pair Frame | Valid % | Depth MAE | Depth RMSE | AbsRel | Median-Aligned MAE | Median-Aligned RMSE | Median-Aligned AbsRel | Per-Image Scale |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ])
    lines.extend(report_row(row) for row in rows)

    lines.extend([
        "",
        "## Mean By Source",
        "",
        "| Source | Samples | Valid % | Depth MAE | Depth RMSE | AbsRel | Median-Aligned MAE | Median-Aligned RMSE | Median-Aligned AbsRel |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ])
    for source in sorted({row["source"] for row in rows}):
        lines.append(report_summary_row(source, [row for row in rows if row["source"] == source]))

    lines.append("")
    (output_dir / "depth_report.md").write_text("\n".join(lines))


def write_metrics(output_dir, rows, calibration=None):
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "metrics.json", "w") as f:
        json.dump(
            {
                "calibration": calibration or {},
                "samples": rows,
                "summary": summarize_metrics(rows),
                "summary_by_source": summarize_metrics_by_source(rows),
            },
            f,
            indent=2,
        )

    fieldnames = sorted({key for row in rows for key in row.keys()})
    with open(output_dir / "metrics.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    write_report(output_dir, rows, calibration=calibration)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare MoGe2 depth on rendered views against dataset depth."
    )
    parser.add_argument(
        "--scene-dir",
        type=Path,
        default=PROJECT_ROOT / "DATA" / "kitchen",
        help="COLMAP scene directory containing sparse/0, images/, gs.ply, and dense depth maps.",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=1,
        help="Number of random samples per selected source.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "depth_comparison",
        help="Directory for visualizations, depth arrays, and metrics.",
    )
    parser.add_argument(
        "--colmap-depth-kind",
        choices=("geometric", "photometric"),
        default="geometric",
        help="COLMAP dense depth map kind. No fallback to the other kind is used.",
    )
    parser.add_argument(
        "--colmap-depth-dir",
        type=Path,
        default=None,
        help="Explicit COLMAP dense depth_maps directory.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    rng = random.Random(args.seed)
    rows = []
    for _ in range(args.num_samples):
        rows.append(
            sample_colmap(
                args.scene_dir,
                args.output_dir,
                rng,
                args.colmap_depth_kind,
                args.colmap_depth_dir,
            )
        )

    write_metrics(args.output_dir, rows)

    for row in rows:
        print(
            f"{row['source']} {row['frame']}"
            f"{' pair=' + row['pair_frame'] if row.get('pair_frame') else ''}: "
            f"MAE={row['mae']:.4f} RMSE={row['rmse']:.4f} "
            f"AbsRel={row['abs_rel']:.4f} "
            f"AlignedMAE={row['median_aligned_mae']:.4f} "
            f"Scale={row['median_scale_pred_to_gt']:.4f} "
            f"valid={row['valid_fraction']:.2%}"
        )
    print(f"Saved outputs to {args.output_dir}")


if __name__ == "__main__":
    main()
