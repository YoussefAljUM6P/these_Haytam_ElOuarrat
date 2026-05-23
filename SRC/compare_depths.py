import argparse
import csv
import json
import random
from pathlib import Path

import numpy as np
from evo.core.geometry import umeyama_alignment
from PIL import Image

from dataset import load_colmap, load_scannet
from depth import estimate_depth_moge
from scenes.gs import GSScene
from scenes.mesh import MeshScene


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def parse_scannet_depth_shift(scene_dir):
    info_path = Path(scene_dir) / "info.txt"
    with open(info_path, "r") as f:
        for line in f:
            line = line.strip()
            if line.startswith("m_depthShift"):
                return float(line.split("=")[1].strip())
    raise ValueError(f"No m_depthShift entry in {info_path}")


def load_scannet_depth(depth_path, depth_shift, target_shape):
    raw = np.array(Image.open(depth_path), dtype=np.float32)
    depth = raw / depth_shift
    return resize_depth_nearest(depth, target_shape)


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
        "mae_m": float(abs_diff.mean()),
        "rmse_m": float(np.sqrt((diff ** 2).mean())),
        "median_abs_error_m": float(np.median(abs_diff)),
        "bias_m": float(diff.mean()),
        "abs_rel": float((abs_diff / gt).mean()),
        "sq_rel": float(((diff ** 2) / gt).mean()),
        "delta_1_25": float((ratio < 1.25).mean()),
        "delta_1_25_2": float((ratio < 1.25 ** 2).mean()),
        "delta_1_25_3": float((ratio < 1.25 ** 3).mean()),
        "pred_mean_m": float(pred.mean()),
        "gt_mean_m": float(gt.mean()),
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


def sample_scannet(scene_dir, output_dir, rng):
    data = [
        item for item in load_scannet(scene_dir)
        if item[2].exists()
    ]
    if not data:
        raise RuntimeError(f"No ScanNet frames with depth maps in {scene_dir}")

    camera, rgb_path, depth_path = rng.choice(data)
    scene = MeshScene(Path(scene_dir) / "mesh.ply")
    rendered = scene.render(camera)
    pred_depth = estimate_depth_moge(rendered)
    gt_depth = load_scannet_depth(depth_path, parse_scannet_depth_shift(scene_dir), pred_depth.shape)
    real_rgb = load_real_rgb(rgb_path, rendered.shape[:2])

    prefix = f"scannet_{depth_path.stem.replace('.depth', '')}"
    save_sample_outputs(output_dir, prefix, rendered, real_rgb, pred_depth, gt_depth)

    metrics = compute_metrics(pred_depth, gt_depth)
    metrics.update({
        "source": "scannet",
        "frame": depth_path.stem.replace(".depth", ""),
        "rgb_path": str(rgb_path),
        "gt_depth_path": str(depth_path),
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
    colmap_scale_info=None,
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
        colmap_scale_info=colmap_scale_info,
    )
    metrics["source"] = "colmap"
    return metrics


def scannet_depth_candidates(scene_dir):
    return {
        rgb_path.name: (camera, rgb_path, depth_path)
        for camera, rgb_path, depth_path in load_scannet(scene_dir)
        if depth_path.exists()
    }


def colmap_depth_candidates(scene_dir, depth_kind, colmap_depth_dir):
    candidates = {}
    for camera, rgb_path in load_colmap(scene_dir):
        depth_path = find_colmap_depth_path(
            scene_dir, rgb_path.name, depth_kind, explicit_dir=colmap_depth_dir
        )
        if depth_path is not None:
            candidates[rgb_path.name] = (camera, rgb_path, depth_path)
    return candidates


def estimate_similarity_umeyama(source_points, target_points):
    source = np.asarray(source_points, dtype=np.float64)
    target = np.asarray(target_points, dtype=np.float64)
    if source.shape != target.shape or source.ndim != 2 or source.shape[1] != 3:
        raise ValueError("Expected matched source/target point arrays with shape (N, 3)")
    if source.shape[0] < 3:
        raise ValueError("At least 3 matched points are required for Sim(3) scale")
    if np.sum((source - source.mean(axis=0)) ** 2) <= 0.0:
        raise ValueError("COLMAP camera centers are degenerate; cannot estimate scale")

    rotation, translation, scale = umeyama_alignment(source.T, target.T, with_scale=True)
    aligned = (scale * (rotation @ source.T)).T + translation
    errors = np.linalg.norm(aligned - target, axis=1)

    return {
        "scale": float(scale),
        "rotation": rotation.astype(np.float32),
        "translation": translation.astype(np.float32),
        "median_alignment_error_m": float(np.median(errors)),
        "mean_alignment_error_m": float(errors.mean()),
        "max_alignment_error_m": float(errors.max()),
    }


def estimate_colmap_to_scannet_scale(scene_dir):
    scannet = {
        rgb_path.name: camera.T_world_cam[:3, 3]
        for camera, rgb_path, depth_path in load_scannet(scene_dir)
        if depth_path.exists()
    }
    colmap = {
        rgb_path.name: camera.T_world_cam[:3, 3]
        for camera, rgb_path in load_colmap(scene_dir)
    }
    frame_names = sorted(scannet.keys() & colmap.keys())
    if len(frame_names) < 3:
        raise RuntimeError(
            "Need at least 3 same-frame ScanNet/COLMAP camera poses to estimate "
            "COLMAP-to-ScanNet scale"
        )

    colmap_centers = np.stack([colmap[name] for name in frame_names])
    scannet_centers = np.stack([scannet[name] for name in frame_names])
    sim3 = estimate_similarity_umeyama(colmap_centers, scannet_centers)

    return {
        "source": "trajectory_umeyama",
        "frame_count": len(frame_names),
        "colmap_to_metric_scale": sim3["scale"],
        "median_alignment_error_m": sim3["median_alignment_error_m"],
        "mean_alignment_error_m": sim3["mean_alignment_error_m"],
        "max_alignment_error_m": sim3["max_alignment_error_m"],
    }


def evaluate_scannet_item(scene_dir, output_dir, scene, item, pair_frame=None):
    camera, rgb_path, depth_path = item
    rendered = scene.render(camera)
    pred_depth = estimate_depth_moge(rendered)
    gt_depth = load_scannet_depth(depth_path, parse_scannet_depth_shift(scene_dir), pred_depth.shape)
    real_rgb = load_real_rgb(rgb_path, rendered.shape[:2])

    frame = depth_path.stem.replace(".depth", "")
    prefix = f"mesh_scannet_{frame}"
    save_sample_outputs(output_dir, prefix, rendered, real_rgb, pred_depth, gt_depth)

    metrics = compute_metrics(pred_depth, gt_depth)
    metrics.update({
        "source": "mesh_scannet",
        "frame": frame,
        "pair_frame": pair_frame,
        "rgb_path": str(rgb_path),
        "gt_depth_path": str(depth_path),
        "rendered_path": str(output_dir / f"{prefix}_render.png"),
        "moge_depth_path": str(output_dir / f"{prefix}_moge_depth.npy"),
        "gt_depth_npy_path": str(output_dir / f"{prefix}_gt_depth.npy"),
    })
    return metrics


def evaluate_colmap_item(
    output_dir,
    scene,
    item,
    depth_kind,
    pair_frame=None,
    colmap_scale_info=None,
):
    camera, rgb_path, depth_path = item
    rendered = scene.render(camera)
    pred_depth = estimate_depth_moge(rendered)
    colmap_to_metric_scale = (
        1.0 if colmap_scale_info is None
        else colmap_scale_info["colmap_to_metric_scale"]
    )
    gt_depth = (
        resize_depth_nearest(read_colmap_depth(depth_path), pred_depth.shape)
        * colmap_to_metric_scale
    )
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
        "colmap_scale_source": "none" if colmap_scale_info is None else colmap_scale_info["source"],
        "colmap_to_metric_scale": colmap_to_metric_scale,
        "rendered_path": str(output_dir / f"{prefix}_render.png"),
        "moge_depth_path": str(output_dir / f"{prefix}_moge_depth.npy"),
        "gt_depth_npy_path": str(output_dir / f"{prefix}_gt_depth.npy"),
    })
    return metrics


def sample_paired(scene_dir, output_dir, rng, depth_kind, colmap_depth_dir, colmap_scale_info):
    scannet = scannet_depth_candidates(scene_dir)
    colmap = colmap_depth_candidates(scene_dir, depth_kind, colmap_depth_dir)
    frame_names = sorted(scannet.keys() & colmap.keys())
    if not frame_names:
        raise RuntimeError(
            "No same-frame candidates found with both ScanNet depth and "
            f"COLMAP {depth_kind!r} dense depth"
        )

    frame_name = rng.choice(frame_names)
    mesh_scene = MeshScene(Path(scene_dir) / "mesh.ply")
    gs_scene = GSScene(Path(scene_dir) / "gs.ply")

    return [
        evaluate_scannet_item(
            scene_dir,
            output_dir,
            mesh_scene,
            scannet[frame_name],
            pair_frame=frame_name,
        ),
        evaluate_colmap_item(
            output_dir,
            gs_scene,
            colmap[frame_name],
            depth_kind,
            pair_frame=frame_name,
            colmap_scale_info=colmap_scale_info,
        ),
    ]


def summarize_metrics(rows):
    metric_keys = [
        "valid_fraction",
        "mae_m",
        "rmse_m",
        "median_abs_error_m",
        "bias_m",
        "abs_rel",
        "sq_rel",
        "delta_1_25",
        "delta_1_25_2",
        "delta_1_25_3",
        "median_scale_pred_to_gt",
        "median_aligned_mae_m",
        "median_aligned_rmse_m",
        "median_aligned_median_abs_error_m",
        "median_aligned_bias_m",
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
        f"{row['mae_m']:.4f} | "
        f"{row['rmse_m']:.4f} | "
        f"{row['abs_rel']:.4f} | "
        f"{row['median_aligned_mae_m']:.4f} | "
        f"{row['median_aligned_rmse_m']:.4f} | "
        f"{row['median_aligned_abs_rel']:.4f} | "
        f"{row['median_scale_pred_to_gt']:.4f} |"
    )


def report_summary_row(source, rows):
    valid = np.array([row["valid_fraction"] for row in rows], dtype=np.float64)
    mae = np.array([row["mae_m"] for row in rows], dtype=np.float64)
    rmse = np.array([row["rmse_m"] for row in rows], dtype=np.float64)
    abs_rel = np.array([row["abs_rel"] for row in rows], dtype=np.float64)
    aligned_mae = np.array([row["median_aligned_mae_m"] for row in rows], dtype=np.float64)
    aligned_rmse = np.array([row["median_aligned_rmse_m"] for row in rows], dtype=np.float64)
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
        "Scene-scaled metrics are the paper-facing absolute metric errors. "
        "Median-aligned metrics are diagnostic relative-shape errors after one "
        "per-image scalar is applied to MoGe2 depth.",
        "",
    ]

    if calibration and calibration.get("colmap_to_scannet"):
        scale_info = calibration["colmap_to_scannet"]
        lines.extend([
            "## Scene Scale Calibration",
            "",
            f"- Source: `{scale_info['source']}`",
            f"- Paired camera centers: `{scale_info['frame_count']}`",
            f"- COLMAP-to-ScanNet scale: `{scale_info['colmap_to_metric_scale']:.8f}`",
            f"- Median trajectory alignment error: `{scale_info['median_alignment_error_m']:.4f} m`",
            f"- Mean trajectory alignment error: `{scale_info['mean_alignment_error_m']:.4f} m`",
            "",
        ])

    lines.extend([
        "## Per-Sample Metrics",
        "",
        "| Source | Frame | Pair Frame | Valid % | Metric MAE m | Metric RMSE m | Metric AbsRel | Median-Aligned MAE m | Median-Aligned RMSE m | Median-Aligned AbsRel | Per-Image Scale |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ])
    lines.extend(report_row(row) for row in rows)

    lines.extend([
        "",
        "## Mean By Source",
        "",
        "| Source | Samples | Valid % | Metric MAE m | Metric RMSE m | Metric AbsRel | Median-Aligned MAE m | Median-Aligned RMSE m | Median-Aligned AbsRel |",
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
        description="Compare MoGe2 metric depth on rendered views against dataset depth."
    )
    parser.add_argument(
        "--scene-dir",
        type=Path,
        default=PROJECT_ROOT / "DATA" / "kitchen",
        help="Scene directory containing ScanNet/COLMAP data.",
    )
    parser.add_argument(
        "--source",
        choices=("scannet", "colmap", "both"),
        default="scannet",
        help="Dataset depth source to compare against. 'both' samples paired frame names.",
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
    parser.add_argument(
        "--no-colmap-metric-scale",
        action="store_true",
        help=(
            "Disable scene-level COLMAP-to-ScanNet metric scale calibration. "
            "By default, COLMAP depths are scaled to ScanNet meters."
        ),
    )
    return parser.parse_args()


def main():
    args = parse_args()
    rng = random.Random(args.seed)
    rows = []
    calibration = {}
    colmap_scale_info = None

    if args.source in ("colmap", "both") and not args.no_colmap_metric_scale:
        colmap_scale_info = estimate_colmap_to_scannet_scale(args.scene_dir)
        calibration["colmap_to_scannet"] = colmap_scale_info
        print(
            "COLMAP-to-ScanNet scale "
            f"{colmap_scale_info['colmap_to_metric_scale']:.6f} "
            f"from {colmap_scale_info['frame_count']} paired camera centers "
            f"(median trajectory error "
            f"{colmap_scale_info['median_alignment_error_m']:.4f} m)"
        )

    if args.source == "both":
        for _ in range(args.num_samples):
            rows.extend(
                sample_paired(
                    args.scene_dir,
                    args.output_dir,
                    rng,
                    args.colmap_depth_kind,
                    args.colmap_depth_dir,
                    colmap_scale_info,
                )
            )
    else:
        for _ in range(args.num_samples):
            if args.source == "scannet":
                rows.append(sample_scannet(args.scene_dir, args.output_dir, rng))
            elif args.source == "colmap":
                rows.append(
                    sample_colmap(
                        args.scene_dir,
                        args.output_dir,
                        rng,
                        args.colmap_depth_kind,
                        args.colmap_depth_dir,
                        colmap_scale_info=colmap_scale_info,
                    )
                )
            else:
                raise ValueError(f"Unknown source: {args.source}")

    write_metrics(args.output_dir, rows, calibration=calibration)

    for row in rows:
        print(
            f"{row['source']} {row['frame']}"
            f"{' pair=' + row['pair_frame'] if row.get('pair_frame') else ''}: "
            f"MAE={row['mae_m']:.4f}m RMSE={row['rmse_m']:.4f}m "
            f"AbsRel={row['abs_rel']:.4f} "
            f"AlignedMAE={row['median_aligned_mae_m']:.4f}m "
            f"Scale={row['median_scale_pred_to_gt']:.4f} "
            f"valid={row['valid_fraction']:.2%}"
        )
    print(f"Saved outputs to {args.output_dir}")


if __name__ == "__main__":
    main()
