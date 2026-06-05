"""Offline re-rendering of a previous trajectory run.

Given a trajectory scene directory produced by `cli.py trajectory`, rebuild
the scene + camera intrinsics from `summary.json`, parse `sim_traj.tum`
(and optionally `gt_traj.tum`), and render each pose to disk.

Usage:
    python SRC/render_offline.py <run_scene_dir> [--gt] [--side-by-side]
                                                 [--every N] [--out NAME]

Outputs:
    <run_scene_dir>/<out>/sim_pose_NNNN.png       (--out defaults to 'renders_offline')
    <run_scene_dir>/<out>/gt_pose_NNNN.png        (if --gt)
    <run_scene_dir>/<out>/sbs_NNNN.png            (if --side-by-side: sim vs target RGB)
"""

import argparse
import json
import sys
from pathlib import Path

# Make sibling imports work when run as a script.
_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from camera import Camera
from runners.servo_frames import load_rgb, load_scene_and_frames, save_rgb
from runners.trajectory import read_tum
from viz import save_side_by_side


def build_camera(intrinsics, T_world_cam):
    return Camera(
        T_world_cam=T_world_cam,
        fx=intrinsics["fx"],
        fy=intrinsics["fy"],
        cx=intrinsics["cx"],
        cy=intrinsics["cy"],
        H=int(intrinsics["height"]),
        W=int(intrinsics["width"]),
    )


def load_per_task_targets(csv_path):
    """Return list of dicts keyed by task_index with target_rgb paths."""
    import csv
    rows = {}
    if not csv_path.exists():
        return rows
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                idx = int(row.get("task_index", ""))
            except ValueError:
                continue
            rows[idx] = row
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_scene_dir", type=Path,
                        help="Path like RUNS/trajectory/<run>/scenes/<scene>/")
    parser.add_argument("--gt", action="store_true",
                        help="Also render gt_traj.tum poses.")
    parser.add_argument("--side-by-side", action="store_true",
                        help="Save sim render alongside target RGB (per task).")
    parser.add_argument("--every", type=int, default=1,
                        help="Render every Nth pose (default 1 = all).")
    parser.add_argument("--out", type=str, default="renders_offline",
                        help="Output subdirectory name under run_scene_dir.")
    parser.add_argument("--scene-dir", type=Path, default=None,
                        help="Override scene_dir from summary.json (e.g. moved DATA).")
    args = parser.parse_args()

    run_scene_dir = args.run_scene_dir.resolve()
    summary_path = run_scene_dir / "summary.json"
    sim_tum_path = run_scene_dir / "sim_traj.tum"
    gt_tum_path = run_scene_dir / "gt_traj.tum"

    if not summary_path.exists():
        raise FileNotFoundError(f"Missing {summary_path}")
    if not sim_tum_path.exists():
        raise FileNotFoundError(f"Missing {sim_tum_path}")

    with open(summary_path) as f:
        summary = json.load(f)

    renderer = summary["renderer"]
    intrinsics = summary["camera"]
    scene_dir = args.scene_dir or Path(summary.get("scene_dir", ""))
    if not scene_dir.exists():
        raise FileNotFoundError(
            f"scene_dir from summary does not exist: {scene_dir}. "
            f"Pass --scene-dir to override."
        )

    out_dir = run_scene_dir / args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading {renderer} scene from {scene_dir} ...")
    scene, _frame_index = load_scene_and_frames(scene_dir, renderer)

    print(f"Reading sim trajectory: {sim_tum_path}")
    _, sim_poses = read_tum(sim_tum_path)

    _, gt_poses = ([], [])
    if args.gt and gt_tum_path.exists():
        print(f"Reading gt trajectory:  {gt_tum_path}")
        _, gt_poses = read_tum(gt_tum_path)
        if len(gt_poses) != len(sim_poses):
            print(
                f"warning: gt traj has {len(gt_poses)} poses, "
                f"sim has {len(sim_poses)} -- pairing only the shared prefix"
            )

    targets = load_per_task_targets(run_scene_dir / "per_task_errors.csv")

    every = max(1, int(args.every))

    # The first TUM row (timestamp 0.0) is the initial pose, not a mini-task
    # final. Tasks are indexed from 0 starting at TUM index 1.
    n = len(sim_poses)
    for i in range(n):
        if i == 0:
            tag = "init"
            target_idx = None
        else:
            task_idx = i - 1
            if task_idx % every != 0:
                continue
            tag = f"task{task_idx:04d}"
            target_idx = task_idx

        sim_T = sim_poses[i]
        cam = build_camera(intrinsics, sim_T)
        sim_render = scene.render(cam)
        save_rgb(out_dir / f"sim_{tag}.png", sim_render)

        if args.gt and i < len(gt_poses):
            gt_cam = build_camera(intrinsics, gt_poses[i])
            gt_render = scene.render(gt_cam)
            save_rgb(out_dir / f"gt_{tag}.png", gt_render)

        if args.side_by_side and target_idx is not None and target_idx in targets:
            target_rgb_path = targets[target_idx].get("target_rgb", "")
            if target_rgb_path:
                try:
                    target_img = load_rgb(
                        target_rgb_path, cam.W, cam.H,
                    )
                    save_side_by_side(
                        sim_render, target_img,
                        out_dir / f"sbs_{tag}.png",
                    )
                except FileNotFoundError as e:
                    print(f"sbs skipped for {tag}: {e}")

        print(f"  rendered {tag}")

    print(f"Wrote {n} renders to {out_dir}")


if __name__ == "__main__":
    main()
