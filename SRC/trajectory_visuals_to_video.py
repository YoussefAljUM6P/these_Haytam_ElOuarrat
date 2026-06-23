"""Compile saved trajectory visualizations into a video.

Examples:
  python SRC/trajectory_visuals_to_video.py --run-dir RUNS/trajectory/20260524-153012_living_gs_ibvs_intrinsic_sift
  python SRC/trajectory_visuals_to_video.py --latest --fps 8
"""

import argparse
import csv
import re
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parent.parent
RUNS_ROOT = PROJECT_ROOT / "RUNS"

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from runners.trajectory import read_tum


def natural_key(path):
    parts = re.split(r"(\d+)", str(path))
    return [int(part) if part.isdigit() else part.lower() for part in parts]


def latest_trajectory_run():
    base = RUNS_ROOT / "trajectory"
    if not base.exists():
        raise FileNotFoundError(f"No trajectory runs found under {base}")

    candidates = []
    for entry in base.iterdir():
        if not entry.is_dir():
            continue
        if entry.name in {"mesh", "gs", "nerf"}:
            candidates.extend(path for path in entry.iterdir() if path.is_dir())
        else:
            candidates.append(entry)

    if not candidates:
        raise FileNotFoundError(f"No trajectory run directories found under {base}")
    return max(candidates, key=lambda path: path.stat().st_mtime)


def scene_dir_for(run_dir, scene):
    direct = run_dir / scene
    nested = run_dir / "scenes" / scene
    if nested.exists() or not direct.exists():
        return nested
    return direct


def iter_scene_dirs(run_dir):
    if (run_dir / "summary.json").exists() and (run_dir / "sim_traj.tum").exists():
        return [run_dir]
    scenes_dir = run_dir / "scenes"
    if scenes_dir.exists():
        return sorted(
            (path for path in scenes_dir.iterdir() if path.is_dir()),
            key=natural_key,
        )
    return sorted(
        (path for path in run_dir.iterdir() if path.is_dir()),
        key=natural_key,
    )


def resolve_existing_run_dir(run_dir):
    resolved = Path(run_dir).resolve()
    if resolved.exists():
        return resolved

    # Recover from paste errors like:
    #   RUNS/trajectory/gs/home/user/project/RUNS/trajectory/gs/<run>
    # where an absolute path lost its leading slash and was appended to a
    # relative prefix.
    parts = Path(run_dir).parts
    for index in range(len(parts)):
        candidate = Path("/", *parts[index:])
        if candidate.exists():
            print(f"Using repaired run directory: {candidate}")
            return candidate.resolve()

    raise FileNotFoundError(f"Trajectory run directory not found: {resolved}")


def resolve_frame_path(path_text, csv_path, run_dir):
    path = Path(path_text)
    if path.is_absolute():
        return path

    csv_relative = (csv_path.parent / path).resolve()
    if csv_relative.exists():
        return csv_relative
    return (run_dir / path).resolve()


def collect_manifest_frames(run_dir, scene=None):
    frames = []
    if (run_dir / "per_task_errors.csv").exists():
        csv_paths = [run_dir / "per_task_errors.csv"]
    else:
        csv_paths = sorted(
            list((run_dir / "scenes").glob("*/per_task_errors.csv"))
            + list(run_dir.glob("*/per_task_errors.csv")),
            key=natural_key,
        )

    for csv_path in csv_paths:
        scene_name = csv_path.parent.name
        if scene is not None and scene_name != scene:
            continue

        with open(csv_path, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                viz_path = row.get("viz_path", "").strip()
                if not viz_path:
                    continue
                path = resolve_frame_path(viz_path, csv_path, run_dir)
                if not path.exists():
                    print(f"WARNING: skipping missing visualization: {path}")
                    continue
                task_index = int(row.get("task_index", len(frames)))
                frames.append((scene_name, task_index, path))

    frames.sort(key=lambda item: (item[0], item[1], natural_key(item[2])))
    return [path for _, _, path in frames]


def collect_glob_frames(run_dir, scene=None, pattern="visualizations/*.png"):
    if scene is not None:
        search_roots = [scene_dir_for(run_dir, scene)]
    elif (run_dir / "visualizations").exists():
        search_roots = [run_dir]
    else:
        search_roots = iter_scene_dirs(run_dir)

    frames = []
    for root in search_roots:
        frames.extend(root.glob(pattern))
        if pattern == "visualizations/*.png":
            frames.extend(root.glob("visualizations_from_traj/*.png"))
    return sorted((path for path in frames if path.is_file()), key=natural_key)


def collect_frames(run_dir, scene=None, pattern="visualizations/*.png"):
    frames = collect_manifest_frames(run_dir, scene=scene)
    if frames:
        return frames

    frames = collect_glob_frames(run_dir, scene=scene, pattern=pattern)
    if frames:
        return frames

    scene_hint = f" for scene {scene!r}" if scene is not None else ""
    raise FileNotFoundError(
        f"No saved trajectory visualization PNGs found in {run_dir}{scene_hint}"
    )


def read_tum_poses(path):
    _, poses = read_tum(path)
    return poses


def render_missing_scene_visuals(
    scene_dir,
    pattern="task_{task_index:04d}_final_vs_target.png",
    max_frames=None,
):
    from runners.servo_frames import (
        PROJECT_ROOT as MAIN_PROJECT_ROOT,
        load_rgb as load_target_rgb,
        load_scene_and_frames,
    )
    from servo import copy_camera_with_pose
    from viz import save_current_desired_error_visualization, save_side_by_side

    scene_dir = Path(scene_dir)
    summary_path = scene_dir / "summary.json"
    csv_path = scene_dir / "per_task_errors.csv"
    sim_tum_path = scene_dir / "sim_traj.tum"
    if not summary_path.exists() or not csv_path.exists() or not sim_tum_path.exists():
        return []

    import json

    with open(summary_path) as f:
        summary = json.load(f)

    scene_name = summary.get("scene", scene_dir.name)
    renderer = summary.get("renderer")
    controller = summary.get("controller")
    if not renderer:
        raise RuntimeError(f"Cannot render missing visuals: {summary_path} has no renderer")

    data_scene_dir = MAIN_PROJECT_ROOT / "DATA" / scene_name
    scene, frame_index = load_scene_and_frames(data_scene_dir, renderer)
    sim_poses = read_tum_poses(sim_tum_path)

    with open(csv_path, newline="") as f:
        rows = list(csv.DictReader(f))
    if len(sim_poses) < len(rows) + 1:
        raise RuntimeError(
            f"{scene_dir}: sim_traj.tum has {len(sim_poses)} poses but "
            f"{len(rows)} task rows need {len(rows) + 1}"
        )

    out_dir = scene_dir / "visualizations_from_traj"
    out_dir.mkdir(parents=True, exist_ok=True)
    frame_paths = []

    for row in rows:
        if max_frames is not None and len(frame_paths) >= int(max_frames):
            break
        task_index = int(row["task_index"])
        target_frame = row["target_frame"]
        if target_frame not in frame_index:
            raise RuntimeError(
                f"{scene_dir}: target frame {target_frame!r} not found in loaded {renderer} frames"
            )

        frame = frame_index[target_frame]
        target_camera = frame["camera"]
        final_camera = copy_camera_with_pose(target_camera, sim_poses[task_index + 1])
        target_image = load_target_rgb(
            frame["rgb_path"],
            final_camera.W,
            final_camera.H,
        )
        rendered = scene.render(final_camera)
        out_path = out_dir / pattern.format(
            task_index=task_index,
            target_frame=target_frame,
        )
        if controller == "photometric":
            save_current_desired_error_visualization(
                rendered,
                target_image,
                out_path,
                current_label="final",
                desired_label="desired",
            )
        else:
            save_side_by_side(rendered, target_image, out_path)
        frame_paths.append(out_path)

    return frame_paths


def render_missing_visuals(run_dir, scene=None, max_frames=None):
    if (run_dir / "summary.json").exists() and (run_dir / "sim_traj.tum").exists():
        scene_dirs = [run_dir]
    elif scene is not None:
        scene_dirs = [scene_dir_for(run_dir, scene)]
    else:
        scene_dirs = iter_scene_dirs(run_dir)

    frame_paths = []
    for scene_dir in scene_dirs:
        remaining = None
        if max_frames is not None:
            remaining = int(max_frames) - len(frame_paths)
            if remaining <= 0:
                break
        scene_frames = render_missing_scene_visuals(
            scene_dir,
            max_frames=remaining,
        )
        if scene_frames:
            frame_paths.extend(scene_frames)
            print(f"Rendered {len(scene_frames)} visualizations in {scene_dir}")
    return frame_paths


def load_rgb(path):
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)


def fit_on_canvas(image, width, height):
    h, w = image.shape[:2]
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    y0 = (height - h) // 2
    x0 = (width - w) // 2
    canvas[y0:y0 + h, x0:x0 + w] = image
    return canvas


def even(value):
    value = int(value)
    return value if value % 2 == 0 else value + 1


def write_video(frame_paths, output_path, fps, codec):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    sizes = []
    for path in frame_paths:
        with Image.open(path) as image:
            sizes.append(image.size)

    width = even(max(size[0] for size in sizes))
    height = even(max(size[1] for size in sizes))
    fourcc = cv2.VideoWriter_fourcc(*codec)
    writer = cv2.VideoWriter(str(output_path), fourcc, float(fps), (width, height))
    if not writer.isOpened():
        raise RuntimeError(
            f"Could not open video writer for {output_path} with codec {codec!r}"
        )

    try:
        for path in frame_paths:
            frame = fit_on_canvas(load_rgb(path), width, height)
            writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    finally:
        writer.release()

    return width, height


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--run-dir",
        type=Path,
        help="Trajectory run root, e.g. RUNS/trajectory/<run_id>.",
    )
    group.add_argument(
        "--latest",
        action="store_true",
        help="Use the newest run under RUNS/trajectory/.",
    )
    parser.add_argument(
        "--scene",
        help="Optional scene subdirectory to compile, e.g. kitchen.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Output video path. Defaults to <run-dir>/trajectory_visualizations.mp4.",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=6.0,
        help="Output frames per second.",
    )
    parser.add_argument(
        "--codec",
        default="mp4v",
        help="OpenCV fourcc codec. mp4v works for .mp4 on most installs.",
    )
    parser.add_argument(
        "--pattern",
        default="visualizations/*.png",
        help="Fallback glob under each scene dir when per_task_errors.csv is absent.",
    )
    parser.add_argument(
        "--render-missing",
        action="store_true",
        help=(
            "If no saved task visualizations exist, rebuild final-vs-target images "
            "from sim_traj.tum and per_task_errors.csv before writing the video."
        ),
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        help="Optional cap on frames to encode or reconstruct.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    run_dir = latest_trajectory_run() if args.latest or args.run_dir is None else args.run_dir
    run_dir = resolve_existing_run_dir(run_dir)

    try:
        frame_paths = collect_frames(run_dir, scene=args.scene, pattern=args.pattern)
    except FileNotFoundError:
        if not args.render_missing:
            raise
        frame_paths = render_missing_visuals(
            run_dir,
            scene=args.scene,
            max_frames=args.max_frames,
        )
        if not frame_paths:
            raise FileNotFoundError(
                f"No saved or reconstructable trajectory visualization frames found in {run_dir}"
            )
    if args.max_frames is not None:
        frame_paths = frame_paths[: int(args.max_frames)]
    output_path = args.output or (run_dir / "trajectory_visualizations.mp4")

    width, height = write_video(
        frame_paths,
        output_path=output_path,
        fps=args.fps,
        codec=args.codec,
    )
    print(
        f"Wrote {output_path} from {len(frame_paths)} frames "
        f"at {args.fps:g} fps ({width}x{height})"
    )


if __name__ == "__main__":
    main()
