import argparse
from pathlib import Path

import numpy as np
from PIL import Image

from dataset import load_colmap
from features import FeatureMatcher, filter_matches
from scenes.mesh import MeshScene
from viz import save_match_visualization


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def load_real_rgb(rgb_path, camera):
    real_rgb = Image.open(rgb_path).convert("RGB")
    real_rgb = real_rgb.resize((camera.W, camera.H))
    return np.asarray(real_rgb, dtype=np.float32) / 255.0


def frame_label(rgb_path):
    name = Path(rgb_path).name
    if name.endswith(".color.jpg"):
        return name.split(".color.jpg")[0]
    return Path(name).stem


def mesh_label(mesh_path):
    mesh_path = Path(mesh_path)
    return f"{mesh_path.parent.name}_{mesh_path.stem}"


def ply_declares_color(mesh_path):
    mesh_path = Path(mesh_path)
    if mesh_path.suffix.lower() != ".ply":
        return True

    properties = set()
    with mesh_path.open("rb") as f:
        for raw_line in f:
            line = raw_line.decode("ascii", errors="ignore").strip()
            if line.startswith("property "):
                parts = line.split()
                if parts:
                    properties.add(parts[-1])
            if line == "end_header":
                break

    common_rgb = {"red", "green", "blue"}
    short_rgb = {"r", "g", "b"}
    return common_rgb.issubset(properties) or short_rgb.issubset(properties)


def save_correspondence(mesh_path, camera, rgb_path, matcher, output_dir, camera_source):
    output_dir.mkdir(parents=True, exist_ok=True)

    if not ply_declares_color(mesh_path):
        print(
            f"WARNING: {mesh_path} has no PLY RGB color properties; "
            "the render will be geometry-only gray."
        )

    scene = MeshScene(mesh_path)
    rendered = scene.render(camera)
    real = load_real_rgb(rgb_path, camera)

    kpts_render, kpts_real = matcher.match(rendered, real)
    kpts_render_f, kpts_real_f, H_matrix, _, _, inliers = filter_matches(
        kpts_render, kpts_real, camera
    )

    kpts_render_removed = kpts_render[~inliers]
    kpts_real_removed = kpts_real[~inliers]
    matches_removed = [(i, i) for i in range(len(kpts_render_removed))]
    matches_kept = [(i, i) for i in range(len(kpts_render_f))]

    frame_name = frame_label(rgb_path)
    out_path = (
        output_dir
        / f"{mesh_label(mesh_path)}_{frame_name}_{camera_source}_{matcher.method}_matches.png"
    )

    save_match_visualization(
        rendered,
        real,
        kpts_render_removed,
        kpts_real_removed,
        matches_removed,
        kpts_render_f,
        kpts_real_f,
        matches_kept,
        out_path,
    )

    return {
        "path": out_path,
        "num_matches": len(kpts_render),
        "num_inliers": len(kpts_render_f),
        "homography_ok": H_matrix is not None,
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description="Render mesh views and draw feature correspondences to original images."
    )
    parser.add_argument(
        "--scene-dir",
        type=Path,
        default=PROJECT_ROOT / "DATA" / "kitchen",
        help="COLMAP-style scene directory with sparse/0 and images/.",
    )
    parser.add_argument(
        "--mesh",
        type=Path,
        nargs="+",
        default=None,
        help="One or more mesh .ply paths. Defaults to <scene-dir>/mesh.ply.",
    )
    parser.add_argument(
        "--frame-index",
        type=int,
        default=0,
        help="Index into the selected camera/image list.",
    )
    parser.add_argument(
        "--camera-source",
        choices=("auto", "colmap"),
        default="auto",
        help="Pose/image source. auto resolves to colmap."
    )
    parser.add_argument(
        "--method",
        choices=("xfeat", "sift"),
        default="xfeat",
        help="Feature matcher to use.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "RUNS" / "mesh_feature_correspondence",
        help="Directory for match overlay images.",
    )
    return parser.parse_args()


def resolve_camera_source(scene_dir, mesh_paths, requested):
    if requested == "auto":
        return "colmap"
    return requested


def load_frames(scene_dir, camera_source):
    if camera_source == "colmap":
        return sorted(load_colmap(scene_dir), key=lambda item: str(item[1]))
    raise ValueError(f"Unknown camera source: {camera_source!r}")


def main():
    args = parse_args()
    scene_dir = args.scene_dir.resolve()
    mesh_paths = args.mesh or [scene_dir / "mesh.ply"]

    camera_source = resolve_camera_source(scene_dir, mesh_paths, args.camera_source)
    frames = load_frames(scene_dir, camera_source)
    if not frames:
        raise RuntimeError(f"No {camera_source} frames found in {scene_dir}")
    if args.frame_index < 0 or args.frame_index >= len(frames):
        raise IndexError(
            f"--frame-index {args.frame_index} is outside the valid range 0..{len(frames) - 1}"
        )

    camera, rgb_path = frames[args.frame_index]
    matcher = FeatureMatcher(method=args.method)

    print(
        f"Frame {args.frame_index}: {frame_label(rgb_path)} "
        f"({camera.W}x{camera.H}), camera_source={camera_source}, matcher={args.method}"
    )

    for mesh_path in mesh_paths:
        mesh_path = mesh_path.resolve()
        result = save_correspondence(
            mesh_path,
            camera,
            rgb_path,
            matcher,
            args.output_dir,
            camera_source,
        )
        print(
            f"{mesh_path}: {result['num_matches']} matches, "
            f"{result['num_inliers']} RANSAC inliers, "
            f"H={'ok' if result['homography_ok'] else 'failed'} -> {result['path']}"
        )


if __name__ == "__main__":
    main()
