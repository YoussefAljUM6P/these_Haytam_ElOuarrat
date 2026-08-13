"""Inspect a single pose: render the scene at frame N and compare to the real RGB.

Use via the CLI:
    python cli.py inspect --scene kitchen --index 89
    python cli.py inspect --scene kitchen --index 89 --features sift
    python cli.py inspect --scene living --index 42 --renderer gs --features xfeat
"""

from datetime import datetime
from pathlib import Path

from scene_assets import has_nerfstudio_checkpoint


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_ROOT = PROJECT_ROOT / "DATA"
RUNS_ROOT = PROJECT_ROOT / "RUNS"
INSPECT_ROOT = RUNS_ROOT / "inspect"

RENDERER_CHOICES = ("mesh", "gs", "nerf")
FEATURE_CHOICES = ("none", "sift", "xfeat")


def add_arguments(parser):
    parser.add_argument(
        "--scene",
        required=True,
        help="Scene folder under DATA/ (e.g. kitchen).",
    )
    parser.add_argument(
        "--index",
        required=True,
        type=int,
        help="1-based logical frame index to render and compare.",
    )
    parser.add_argument(
        "--renderer",
        choices=RENDERER_CHOICES,
        default=None,
        help="Renderer to use. Auto-picked if scene has exactly one renderable asset.",
    )
    parser.add_argument(
        "--features",
        choices=FEATURE_CHOICES,
        default="none",
        help="Overlay feature matches between render and real image. Default: none.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output PNG path. Default: RUNS/inspect/<scene>_<frame>_<renderer>_<features>_<stamp>.png.",
    )


def _detect_renderers(scene_dir):
    available = []
    if (scene_dir / "mesh.ply").exists():
        available.append("mesh")
    if (scene_dir / "gs.ply").exists():
        available.append("gs")
    if has_nerfstudio_checkpoint(scene_dir):
        available.append("nerf")
    return available


def _resolve_renderer(scene_dir, requested):
    available = _detect_renderers(scene_dir)
    if not available:
        raise RuntimeError(
            f"Scene {scene_dir.name!r} has no renderable assets "
            f"(expected mesh.ply, gs.ply, or nerf/)."
        )
    if requested is None:
        if len(available) > 1:
            raise RuntimeError(
                f"Scene {scene_dir.name!r} has multiple renderers {available}; "
                f"pick one with --renderer."
            )
        return available[0]
    if requested not in available:
        raise RuntimeError(
            f"Renderer {requested!r} not available for scene {scene_dir.name!r}. "
            f"Available: {available}"
        )
    return requested


def run(args):
    import numpy as np

    from runners.servo_frames import (
        load_rgb,
        load_scene_and_frames,
        resolve_frame_from_index,
    )
    from viz import save_match_visualization, save_side_by_side

    scene_name = args.scene
    scene_dir = DATA_ROOT / scene_name
    if not scene_dir.exists():
        raise RuntimeError(f"Scene directory not found: {scene_dir}")

    renderer = _resolve_renderer(scene_dir, args.renderer)
    feature_method = args.features

    scene, frame_index = load_scene_and_frames(scene_dir, renderer)
    frame_id = resolve_frame_from_index(frame_index, renderer, args.index)
    record = frame_index[frame_id]
    camera = record["camera"]
    rgb_path = record["rgb_path"]

    rendered = scene.render(camera)
    real = load_rgb(rgb_path, camera.W, camera.H)

    if args.output is not None:
        out_path = Path(args.output)
        if not out_path.suffix:
            out_path = out_path.with_suffix(".png")
        if out_path.parent == Path("."):
            INSPECT_ROOT.mkdir(parents=True, exist_ok=True)
            out_path = INSPECT_ROOT / out_path.name
        elif not out_path.is_absolute():
            out_path = (Path.cwd() / out_path).resolve()
    else:
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        INSPECT_ROOT.mkdir(parents=True, exist_ok=True)
        out_path = (
            INSPECT_ROOT
            / f"{scene_name}_{frame_id}_{renderer}_{feature_method}_{stamp}.png"
        )
    out_path.parent.mkdir(parents=True, exist_ok=True)

    info = {
        "scene": scene_name,
        "frame": frame_id,
        "rgb_path": str(rgb_path),
        "renderer": renderer,
        "features": feature_method,
        "image_size": [int(camera.W), int(camera.H)],
    }

    if feature_method == "none":
        save_side_by_side(rendered, real, out_path)
    else:
        from features import FeatureMatcher, filter_matches

        matcher = FeatureMatcher(method=feature_method)
        kpts_render, kpts_real = matcher.match(rendered, real)
        kpts_render_f, kpts_real_f, H_matrix, _, _, inliers = filter_matches(
            kpts_render, kpts_real, camera
        )

        if kpts_render.shape[0] > 0 and inliers.shape[0] == kpts_render.shape[0]:
            kpts_render_removed = kpts_render[~inliers]
            kpts_real_removed = kpts_real[~inliers]
        else:
            empty = np.zeros((0, 2), dtype=np.float32)
            kpts_render_removed = empty
            kpts_real_removed = empty.copy()

        matches_removed = [(i, i) for i in range(len(kpts_render_removed))]
        matches_kept = [(i, i) for i in range(len(kpts_render_f))]

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

        info.update({
            "num_matches": int(kpts_render.shape[0]),
            "num_inliers": int(kpts_render_f.shape[0]),
            "homography_ok": H_matrix is not None,
        })

    print(f"scene      : {info['scene']}")
    print(f"frame      : {info['frame']}")
    print(f"rgb        : {info['rgb_path']}")
    print(f"renderer   : {info['renderer']}")
    print(f"features   : {info['features']}")
    if feature_method != "none":
        print(
            f"matches    : {info['num_matches']} total, "
            f"{info['num_inliers']} RANSAC inliers, "
            f"H={'ok' if info['homography_ok'] else 'failed'}"
        )
    print(f"saved      : {out_path}")
    return 0
