"""Compiled trajectory of mini servo tasks across one or more datasets.

For each consecutive pair (i, i + STRIDE) in a dataset:
  - servo from the previous task's final pose to the *real* RGB at frame i+STRIDE
  - the final pose becomes the start pose of the next mini task
Append the final pose to a sim trajectory, the GT pose of frame i+STRIDE to a
GT trajectory, write both as TUM files and evaluate with evo (APE/RPE + plots).

Use via the CLI:
    python cli.py trajectory --config <path> [--set k=v ...] [--resume[=path]]
"""

import csv
import json
import math
import sys
import traceback
from pathlib import Path

from run_layout import (
    color,
    run_folder,
    run_folder_name,
    write_command,
    write_json as write_run_json,
    write_run_readme,
)
from runners.servo_frames import (
    PROJECT_ROOT,
    RUNS_ROOT,
    camera_metadata,
    controller_error_display,
    adaptive_motion_limits,
    depth_preflight,
    format_stat,
    frame_id_from_path,
    needs_intrinsic_depth_preflight,
    load_rgb,
    rotation_error_from_pose,
    sorted_frame_ids,
    translation_error_from_pose,
)

# Heavy deps (numpy / scipy / scenes / controllers / evo / matplotlib) are
# imported inside the function that uses them so `cli.py` can register this
# subcommand without pulling them in.


DATASETS = ["living"]
RENDERER = "gs"
GS_MODEL = "standard"  # "standard" -> gs.ply, "moge" -> gs_moge.ply
NERF_RENDER_SCALE = 0.25
GS_RENDER_SCALE = 1.0
MESH_RENDER_SCALE = 1.0
STRIDE = 1
MINI_ITERATIONS = 500
DT = 1.0
DEPTH_MODE = "intrinsic"
FEATURE_METHOD = "sift"
RUN_TAG = "GS_SIFT_INTRINSIC"
GAIN_IBVS = 0.75
GAIN_PHOTO = 0.75
MIN_FEATURES = 3
RATIO = 1
START_INDEX = 1
MAX_PAIRS = None
# Split the capture into "takes" (smooth passes separated by camera teleports)
# and run them back-to-back as one trajectory: at each take boundary the camera
# resets to the GT pose of that take's first frame instead of servoing across
# the jump. Still one run, one TUM pair, one evo evaluation. See SRC/takes.py.
USE_TAKES = True
TAKES_JUMP_FACTOR = 5.0
STOP_RESIDUAL_PX = 0.5      # IBVS: RMS reprojection error (px)
STOP_MSE_PER_PX = 2.0e-6    # legacy photometric MSE; used when STOP_SSD is None
STOP_SSD = None             # photometric: ViSP-style SSD threshold, or None
# Per-task divergence abort thresholds. Set pose thresholds to None to disable.
DIVERGE_RESIDUAL_PX = 1000.0  # IBVS: feature error (px) above which a task is "diverged"
DIVERGE_MSE_PER_PX = 0.01     # photometric: image MSE/px above which a task is "diverged"
DIVERGE_TRANSLATION_ERROR = 0.5  # final translation gap above which a task is "diverged"
DIVERGE_ROTATION_ERROR_DEG = 30.0  # final rotation error above which a task is "diverged"
MIN_INTERACTION_RANK = 6
MAX_INTERACTION_CONDITION = 1.0e8
MAX_TRANSLATION_STEP = 0.5
MAX_ROTATION_STEP_DEG = 30.0
HARD_TRANSLATION_STEP = 5.0
HARD_ROTATION_STEP_DEG = 180.0
ADAPTIVE_STEP_FRACTION = 0.25
CONTINUE_ON_TASK_FAILURE = False
RPE_DELTA = 1
SAVE_TASK_VIZ = True
TASK_VIZ_EVERY = 1
# Live per-iteration progress shown in the terminal while each mini-task runs
# (iter NNN/MMM ...), updated in place. "auto" => only when stdout is a TTY, so
# redirected logs keep just the final per-task lines. True forces it, False off.
SHOW_STEP_PROGRESS = "auto"

# Dynamic iteration schedule for IBVS launches (paper eq. 7):
#     N(w) = N1                if w > 1
#          = N1 / 3            if w < 1/3
#          = round(w * N1)     otherwise
# where w(i) = ||n_r_i - n_q_i||_2 / ||n_r_1 - n_q_1||_2 over filtered
# correspondences (pixels). Disabled for non-IBVS controllers.
DYNAMIC_IBVS_ITERS = True


def _show_step_progress() -> bool:
    """Whether to print the live per-iteration progress line.

    ``SHOW_STEP_PROGRESS`` is "auto" (only on an interactive TTY, so redirected
    cluster logs keep just the final per-task lines), True (always) or False.
    """
    if SHOW_STEP_PROGRESS == "auto":
        return sys.stdout.isatty()
    return bool(SHOW_STEP_PROGRESS)


def _dynamic_n_iter(w: float, n1: int) -> int:
    """Paper eq. 7. N1 is the first-launch iter count."""
    n1 = int(n1)
    if n1 <= 0:
        return 0
    if not (w == w):  # NaN guard
        return n1
    if w > 1.0:
        return n1
    if w < 1.0 / 3.0:
        return max(1, n1 // 3)
    return max(1, int(round(w * n1)))

def _is_cuda_out_of_memory(exc):
    markers = (
        "cuda out of memory",
        "cuda_error_out_of_memory",
        "outofmemoryerror",
        "cumemcreate",
        "cublas_status_alloc_failed",
    )
    seen = set()
    current = exc
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        text = f"{type(current).__name__}: {current}".lower()
        if any(marker in text for marker in markers):
            return True
        current = getattr(current, "__cause__", None) or getattr(current, "__context__", None)
    return False


CONTROLLER = "ibvs"
SIGMA_BLUR = 1.0
USE_GZN = True
GRAD_PERCENTILE = 50.0
PHOTOMETRIC_MAX_PIXELS = 50_000
USE_HUBER = True
HUBER_K = None


def add_arguments(parser):
    parser.add_argument(
        "--config",
        help="JSON experiment config, resolved relative to CONFIGS/ if needed.",
    )
    parser.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help=(
            "Override a config value. May be repeated, e.g. "
            "--set renderer=mesh --set datasets=kitchen."
        ),
    )
    parser.add_argument(
        "--resume",
        nargs="?",
        const="auto",
        default=None,
        help=(
            "Resume an interrupted run. With no value, auto-picks the most "
            "recent run dir matching the current tag. Pass a path to target a "
            "specific run dir."
        ),
    )
    parser.add_argument(
        "--diverge-mse-per-px",
        type=float,
        default=None,
        help=(
            "Photometric trajectory abort threshold on final image MSE per pixel. "
            "Overrides config/--set diverge_mse_per_px."
        ),
    )
    parser.add_argument(
        "--diverge-residual-px",
        type=float,
        default=None,
        help=(
            "IBVS trajectory abort threshold on final feature residual in pixels. "
            "Overrides config/--set diverge_residual_px."
        ),
    )
    parser.add_argument(
        "--diverge-translation-error",
        type=float,
        default=None,
        help=(
            "Trajectory abort threshold on final translation error. "
            "Overrides config/--set diverge_translation_error."
        ),
    )
    parser.add_argument(
        "--diverge-rotation-error-deg",
        type=float,
        default=None,
        help=(
            "Trajectory abort threshold on final rotation error in degrees. "
            "Overrides config/--set diverge_rotation_error_deg."
        ),
    )
    parser.add_argument(
        "--continue-on-task-failure",
        action="store_true",
        default=None,
        help=(
            "When a mini-task fails or diverges, record a cut to its desired "
            "pose and continue with the next task instead of aborting."
        ),
    )


def make_frame_index(records):
    index = {}
    for record in records:
        if len(record) == 3:
            camera, rgb_path, _ = record
        elif len(record) == 2:
            camera, rgb_path = record
        else:
            raise ValueError(f"Unexpected frame record arity: {len(record)}")
        index[frame_id_from_path(rgb_path)] = {
            "camera": camera,
            "rgb_path": rgb_path,
        }
    return index


def scale_camera(camera, scale):
    from camera import Camera

    scale = float(scale)
    if scale == 1.0:
        return camera

    height = max(1, int(round(camera.H * scale)))
    width = max(1, int(round(camera.W * scale)))
    return Camera(
        camera.T_world_cam,
        camera.fx * scale,
        camera.fy * scale,
        camera.cx * scale,
        camera.cy * scale,
        height,
        width,
    )


def scale_frame_records(records, scale):
    scale = float(scale)
    if scale == 1.0:
        return records

    scaled = []
    for record in records:
        if len(record) == 3:
            camera, rgb_path, depth_path = record
            scaled.append((scale_camera(camera, scale), rgb_path, depth_path))
        elif len(record) == 2:
            camera, rgb_path = record
            scaled.append((scale_camera(camera, scale), rgb_path))
        else:
            raise ValueError(f"Unexpected frame record arity: {len(record)}")
    return scaled


def build_trajectory_tasks(frame_ids, takes=None):
    """Chained servo tasks, optionally partitioned into takes.

    ``takes`` is a list of frame-id lists (one per capture pass); pass None for
    the legacy single-sequence behavior. The first task of every take is marked
    ``reset`` so the runner starts it from the GT pose of that take's first
    frame rather than chaining across the camera jump. ``START_INDEX`` offsets
    into the first segment; ``MAX_PAIRS`` caps the total task count.
    """
    if takes:
        segments = [list(take) for take in takes]
    else:
        segments = [list(frame_ids)]

    start_pos = max(0, int(START_INDEX) - 1)
    if segments and start_pos:
        segments[0] = segments[0][start_pos:]

    tasks = []
    for segment in segments:
        first_task = len(tasks)
        for src_pos in range(0, len(segment) - STRIDE, STRIDE):
            tasks.append({
                "src_frame": segment[src_pos],
                "target_frame": segment[src_pos + STRIDE],
                "reset": False,
            })
        if len(tasks) > first_task:
            tasks[first_task]["reset"] = True

    if MAX_PAIRS is not None:
        tasks = tasks[: int(MAX_PAIRS)]

    return tasks


def _apply_scale(records, scale, label):
    records = scale_frame_records(records, scale)
    if float(scale) != 1.0:
        print(f"Using {label} render scale {float(scale):g}")
    return records


def load_trajectory_scene_and_frames(scene_dir):
    from dataset import load_colmap

    records = load_colmap(scene_dir)
    if RENDERER == "mesh":
        records = _apply_scale(records, MESH_RENDER_SCALE, "mesh")
        from scenes.mesh import MeshScene
        mesh_path = scene_dir / "mesh.ply"
        if not mesh_path.exists():
            raise FileNotFoundError(f"Mesh file not found: {mesh_path}")
        print(f"Using mesh {mesh_path}")
        scene = MeshScene(mesh_path)
    elif RENDERER == "gs":
        records = _apply_scale(records, GS_RENDER_SCALE, "gs")
        from scenes.gs import GSScene
        ply_path = scene_dir / ("gs_moge.ply" if GS_MODEL == "moge" else "gs.ply")
        if not ply_path.exists():
            raise FileNotFoundError(f"GS model not found: {ply_path}")
        if GS_MODEL == "moge":
            print(f"Using MoGe depth-supervised GS {ply_path}")
        scene = GSScene(ply_path)
    elif RENDERER == "nerf":
        records = _apply_scale(records, NERF_RENDER_SCALE, "NeRF")
        from scenes.nerf import NeRFScene
        scene = NeRFScene(scene_dir)
    else:
        raise ValueError(f"Unknown renderer {RENDERER!r}")

    frame_index = make_frame_index(records)
    if not frame_index:
        raise RuntimeError(f"No frames loaded for {RENDERER} from {scene_dir}")
    return scene, frame_index


def pose_to_tum_row(timestamp, T_world_cam):
    from scipy.spatial.transform import Rotation

    t = T_world_cam[:3, 3]
    R = T_world_cam[:3, :3]
    q = Rotation.from_matrix(R).as_quat()
    return (
        f"{timestamp:.6f} "
        f"{t[0]:.9f} {t[1]:.9f} {t[2]:.9f} "
        f"{q[0]:.9f} {q[1]:.9f} {q[2]:.9f} {q[3]:.9f}\n"
    )


def write_tum(path, timestamps, poses):
    import numpy as np

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for ts, T in zip(timestamps, poses):
            f.write(pose_to_tum_row(float(ts), np.asarray(T, dtype=np.float64)))


def poses_to_evo_trajectory(timestamps, poses):
    import numpy as np
    from evo.core.trajectory import PoseTrajectory3D
    from scipy.spatial.transform import Rotation

    positions = np.array([np.asarray(T)[:3, 3] for T in poses], dtype=np.float64)
    quats_xyzw = np.array(
        [Rotation.from_matrix(np.asarray(T)[:3, :3]).as_quat() for T in poses],
        dtype=np.float64,
    )
    quats_wxyz = quats_xyzw[:, [3, 0, 1, 2]]
    ts = np.asarray(timestamps, dtype=np.float64)
    return PoseTrajectory3D(positions, quats_wxyz, ts)


def stat_dict(metric):
    return {k: float(v) for k, v in metric.get_all_statistics().items()}


def evaluate_and_plot(timestamps, sim_poses, gt_poses, out_dir, scene_name):
    from evo.core import metrics as evo_metrics
    from evo.core import sync
    from evo.core.metrics import PoseRelation, Unit
    from evo.tools import plot
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    traj_sim = poses_to_evo_trajectory(timestamps, sim_poses)
    traj_gt = poses_to_evo_trajectory(timestamps, gt_poses)

    traj_gt_synced, traj_sim_synced = sync.associate_trajectories(
        traj_gt, traj_sim, max_diff=1e-3,
    )
    pair = (traj_gt_synced, traj_sim_synced)

    ape_t = evo_metrics.APE(PoseRelation.translation_part)
    ape_t.process_data(pair)
    ape_r = evo_metrics.APE(PoseRelation.rotation_angle_deg)
    ape_r.process_data(pair)

    rpe_t = evo_metrics.RPE(
        PoseRelation.translation_part,
        delta=float(RPE_DELTA),
        delta_unit=Unit.frames,
        rel_delta_tol=0.0,
        all_pairs=False,
    )
    rpe_t.process_data(pair)
    rpe_r = evo_metrics.RPE(
        PoseRelation.rotation_angle_deg,
        delta=float(RPE_DELTA),
        delta_unit=Unit.frames,
        rel_delta_tol=0.0,
        all_pairs=False,
    )
    rpe_r.process_data(pair)

    metrics_out = {
        "ape_translation": stat_dict(ape_t),
        "ape_rotation_deg": stat_dict(ape_r),
        "rpe_translation": stat_dict(rpe_t),
        "rpe_rotation_deg": stat_dict(rpe_r),
        "num_poses": int(traj_gt_synced.num_poses),
    }

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    fig_xyz = plt.figure(figsize=(10, 8))
    ax_xyz = plot.prepare_axis(fig_xyz, plot.PlotMode.xyz)
    plot.traj(ax_xyz, plot.PlotMode.xyz, traj_gt_synced,
              style="-", color="green", label="GT")
    plot.traj(ax_xyz, plot.PlotMode.xyz, traj_sim_synced,
              style="--", color="red", label="sim")
    ax_xyz.legend()
    ax_xyz.set_title(f"{scene_name}: sim vs GT (3D)")
    fig_xyz.savefig(out_dir / "trajectory_xyz.png", dpi=120, bbox_inches="tight")
    plt.close(fig_xyz)

    fig_xy = plt.figure(figsize=(8, 8))
    ax_xy = plot.prepare_axis(fig_xy, plot.PlotMode.xy)
    plot.traj(ax_xy, plot.PlotMode.xy, traj_gt_synced,
              style="-", color="green", label="GT")
    plot.traj(ax_xy, plot.PlotMode.xy, traj_sim_synced,
              style="--", color="red", label="sim")
    ax_xy.legend()
    ax_xy.set_title(f"{scene_name}: sim vs GT (top-down xy)")
    fig_xy.savefig(out_dir / "trajectory_xy.png", dpi=120, bbox_inches="tight")
    plt.close(fig_xy)

    fig_err = plt.figure(figsize=(10, 4))
    ax_err = fig_err.add_subplot(111)
    ax_err.plot(traj_sim_synced.timestamps, ape_t.error,
                color="red", label="APE trans")
    ax_err.set_xlabel("task index")
    ax_err.set_ylabel("APE trans")
    ax_err.set_title(f"{scene_name}: APE translation per task")
    ax_err.grid(True, alpha=0.3)
    ax_err.legend()
    fig_err.savefig(out_dir / "ape_translation.png",
                    dpi=120, bbox_inches="tight")
    plt.close(fig_err)

    fig_rot = plt.figure(figsize=(10, 4))
    ax_rot = fig_rot.add_subplot(111)
    ax_rot.plot(traj_sim_synced.timestamps, ape_r.error,
                color="orange", label="APE rot (deg)")
    ax_rot.set_xlabel("task index")
    ax_rot.set_ylabel("APE rot (deg)")
    ax_rot.set_title(f"{scene_name}: APE rotation per task")
    ax_rot.grid(True, alpha=0.3)
    ax_rot.legend()
    fig_rot.savefig(out_dir / "ape_rotation.png",
                    dpi=120, bbox_inches="tight")
    plt.close(fig_rot)

    print(f"\n=== {scene_name} evo metrics ({metrics_out['num_poses']} poses) ===")
    for name in ("ape_translation", "ape_rotation_deg",
                 "rpe_translation", "rpe_rotation_deg"):
        stats = metrics_out[name]
        print(
            f"  {name}: "
            f"rmse={stats.get('rmse', float('nan')):.6f} "
            f"mean={stats.get('mean', float('nan')):.6f} "
            f"median={stats.get('median', float('nan')):.6f} "
            f"std={stats.get('std', float('nan')):.6f} "
            f"min={stats.get('min', float('nan')):.6f} "
            f"max={stats.get('max', float('nan')):.6f}"
        )

    return metrics_out


PER_TASK_FIELDS = [
    "task_index",
    "sequence_reset",
    "src_frame",
    "target_frame",
    "src_rgb",
    "target_rgb",
    "iterations_planned",
    "iterations_run",
    "dynamic_w",
    "feature_err_px",
    "final_residual",
    "diverged",
    "cut_to_desired",
    "failure_type",
    "failure_message",
    "stop_reason",
    "depth_valid_pixels",
    "depth_finite_ratio",
    "initial_translation_gap",
    "initial_rotation_gap_deg",
    "motion_max_translation_step",
    "motion_max_rotation_step_deg",
    "pre_cut_translation_gap",
    "pre_cut_rotation_error_deg",
    "translation_gap",
    "rotation_error_deg",
    "fps",
    "render_fps",
    "iter_ms_mean",
    "render_ms_mean",
    "controller_ms_mean",
    "loop_s",
    "viz_path",
]


def append_task_row(path, row):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists() or path.stat().st_size == 0
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=PER_TASK_FIELDS)
        if write_header:
            writer.writeheader()
        writer.writerow({k: row.get(k, "") for k in PER_TASK_FIELDS})


def read_per_task_csv(path):
    path = Path(path)
    if not path.exists():
        return []
    rows = []
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def _finite_float(value):
    if value in (None, ""):
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(out):
        return None
    return out


def _positive_int(value):
    out = _finite_float(value)
    if out is None or out <= 0:
        return 0
    return int(out)


def summarize_task_timing(rows):
    weighted_keys = ("iter_ms_mean", "render_ms_mean", "controller_ms_mean")
    weighted_sums = {key: 0.0 for key in weighted_keys}
    weighted_counts = {key: 0 for key in weighted_keys}

    total_iters = 0
    total_loop_s = 0.0
    fps_weighted_sum = 0.0
    fps_weighted_count = 0
    num_timed_tasks = 0

    for row in rows:
        iterations = _positive_int(row.get("iterations_run"))
        if iterations <= 0:
            continue

        total_iters += iterations
        row_has_timing = False

        loop_s = _finite_float(row.get("loop_s"))
        if loop_s is not None and loop_s > 0.0:
            total_loop_s += loop_s
            row_has_timing = True

        fps = _finite_float(row.get("fps"))
        if fps is not None and fps > 0.0:
            fps_weighted_sum += fps * iterations
            fps_weighted_count += iterations
            row_has_timing = True

        for key in weighted_keys:
            value = _finite_float(row.get(key))
            if value is None or value < 0.0:
                continue
            weighted_sums[key] += value * iterations
            weighted_counts[key] += iterations
            row_has_timing = True

        if row_has_timing:
            num_timed_tasks += 1

    timing = {
        "num_tasks": int(len(rows)),
        "num_timed_tasks": int(num_timed_tasks),
        "iterations_run": int(total_iters),
        "loop_s": float(total_loop_s),
    }

    for key in weighted_keys:
        count = weighted_counts[key]
        timing[key] = float(weighted_sums[key] / count) if count else 0.0

    if total_loop_s > 0.0:
        fps = float(total_iters / total_loop_s)
    elif fps_weighted_count:
        fps = float(fps_weighted_sum / fps_weighted_count)
    else:
        fps = 0.0

    timing["fps"] = fps
    timing["render_fps"] = (
        float(1000.0 / timing["render_ms_mean"])
        if timing["render_ms_mean"] > 0.0
        else 0.0
    )
    timing["avg_fps"] = timing["fps"]
    timing["avg_render_ms"] = timing["render_ms_mean"]
    return timing


def read_tum(path):
    import numpy as np
    from scipy.spatial.transform import Rotation

    path = Path(path)
    timestamps = []
    poses = []
    with open(path) as f:
        for line in f:
            parts = line.strip().split()
            if not parts or parts[0].startswith("#"):
                continue
            if len(parts) != 8:
                raise ValueError(f"Bad TUM row in {path}: {line!r}")
            ts, tx, ty, tz, qx, qy, qz, qw = (float(p) for p in parts)
            T = np.eye(4, dtype=np.float32)
            T[:3, :3] = Rotation.from_quat([qx, qy, qz, qw]).as_matrix().astype(np.float32)
            T[:3, 3] = [tx, ty, tz]
            timestamps.append(ts)
            poses.append(T)
    return timestamps, poses


def _format_summary_value(value):
    if value is None:
        return "-"
    try:
        value = float(value)
    except (TypeError, ValueError):
        return "-"
    if value != value:
        return "-"
    if value in (float("inf"), float("-inf")):
        return "inf" if value > 0.0 else "-inf"
    mag = abs(value)
    if mag == 0.0:
        return "0"
    if mag < 1.0e-3:
        return f"{value:.2e}"
    if mag < 1.0:
        return f"{value:.6f}".rstrip("0").rstrip(".")
    if mag < 10.0:
        return f"{value:.4f}".rstrip("0").rstrip(".")
    return f"{value:.2f}"


def _summary_metric(scene_summary, family, key):
    if not isinstance(scene_summary, dict):
        return None
    metrics = scene_summary.get("metrics", {})
    if not isinstance(metrics, dict):
        return None
    values = metrics.get(family, {})
    if not isinstance(values, dict):
        return None
    return values.get(key)


def _summary_timing(scene_summary, key):
    if not isinstance(scene_summary, dict):
        return None
    timing = scene_summary.get("timing", {})
    if not isinstance(timing, dict):
        return None
    return timing.get(key)


def _summary_status(scene_summary):
    if not isinstance(scene_summary, dict):
        return "failed"
    if "error" in scene_summary and "metrics" not in scene_summary:
        return "failed"
    metrics = scene_summary.get("metrics", {})
    if isinstance(metrics, dict) and metrics.get("error"):
        return "eval_failed"
    return "ok"


def _summary_error(scene_summary):
    if not isinstance(scene_summary, dict):
        return "invalid summary"
    if "error" in scene_summary and "metrics" not in scene_summary:
        return str(scene_summary["error"])
    metrics = scene_summary.get("metrics", {})
    if isinstance(metrics, dict) and metrics.get("error"):
        return str(metrics["error"])
    return ""


def write_trajectory_summary_markdown(path, overall):
    rows = [[
        "Dataset",
        "Status",
        "Tasks",
        "APE trans RMSE",
        "APE trans mean",
        "APE rot RMSE deg",
        "RPE trans RMSE",
        "RPE rot RMSE deg",
        "Avg FPS",
        "Avg render ms",
        "Error",
    ]]
    for scene_name, scene_summary in overall.items():
        rows.append([
            scene_name,
            _summary_status(scene_summary),
            (
                str(scene_summary.get("num_tasks", "-"))
                if isinstance(scene_summary, dict)
                else "-"
            ),
            _format_summary_value(
                _summary_metric(scene_summary, "ape_translation", "rmse")
            ),
            _format_summary_value(
                _summary_metric(scene_summary, "ape_translation", "mean")
            ),
            _format_summary_value(
                _summary_metric(scene_summary, "ape_rotation_deg", "rmse")
            ),
            _format_summary_value(
                _summary_metric(scene_summary, "rpe_translation", "rmse")
            ),
            _format_summary_value(
                _summary_metric(scene_summary, "rpe_rotation_deg", "rmse")
            ),
            _format_summary_value(_summary_timing(scene_summary, "fps")),
            _format_summary_value(_summary_timing(scene_summary, "render_ms_mean")),
            _summary_error(scene_summary),
        ])

    widths = [max(len(str(row[i])) for row in rows) for i in range(len(rows[0]))]

    def fmt(row):
        return "| " + " | ".join(
            str(cell).ljust(widths[i]) for i, cell in enumerate(row)
        ) + " |"

    lines = [
        "# Trajectory Comparative Summary",
        "",
        "Translation values are in COLMAP scene scale.",
        "",
        fmt(rows[0]),
        "| " + " | ".join("-" * width for width in widths) + " |",
    ]
    lines.extend(fmt(row) for row in rows[1:])
    lines.append("")
    Path(path).write_text("\n".join(lines))


def trajectory_scene_dir(run_root, scene_name, resume=False):
    # Flat layout: each scene's run folder holds its artifacts directly.
    return Path(run_root)


def trajectory_feature_matcher():
    # Only feature-based servoing (FBVS / IBVS) carries a feature matcher in the
    # run-folder name; photometric controllers don't.
    return FEATURE_METHOD if CONTROLLER == "ibvs" else None


def resolved_trajectory_config(tag, run_root):
    return {
        "kind": "trajectory",
        "run_root": str(run_root),
        "run_tag": tag,
        "datasets": list(DATASETS),
        "renderer": RENDERER,
        "nerf_render_scale": float(NERF_RENDER_SCALE),
        "gs_render_scale": float(GS_RENDER_SCALE),
        "mesh_render_scale": float(MESH_RENDER_SCALE),
        "stride": int(STRIDE),
        "mini_iterations": int(MINI_ITERATIONS),
        "dt": float(DT),
        "depth_mode": DEPTH_MODE,
        "feature_method": FEATURE_METHOD,
        "controller": CONTROLLER,
        "gain_ibvs": float(GAIN_IBVS),
        "gain_photo": float(GAIN_PHOTO),
        "min_features": int(MIN_FEATURES),
        "ratio": int(RATIO),
        "start_index": int(START_INDEX),
        "max_pairs": MAX_PAIRS,
        "use_takes": bool(USE_TAKES),
        "takes_jump_factor": float(TAKES_JUMP_FACTOR),
        "stop_residual_px": float(STOP_RESIDUAL_PX),
        "stop_mse_per_px": float(STOP_MSE_PER_PX),
        "stop_ssd": None if STOP_SSD is None else float(STOP_SSD),
        "diverge_residual_px": float(DIVERGE_RESIDUAL_PX),
        "diverge_mse_per_px": float(DIVERGE_MSE_PER_PX),
        "diverge_translation_error": (
            None
            if DIVERGE_TRANSLATION_ERROR is None
            else float(DIVERGE_TRANSLATION_ERROR)
        ),
        "diverge_rotation_error_deg": (
            None
            if DIVERGE_ROTATION_ERROR_DEG is None
            else float(DIVERGE_ROTATION_ERROR_DEG)
        ),
        "min_interaction_rank": int(MIN_INTERACTION_RANK),
        "max_interaction_condition": (
            None
            if MAX_INTERACTION_CONDITION is None
            else float(MAX_INTERACTION_CONDITION)
        ),
        "max_translation_step": (
            None if MAX_TRANSLATION_STEP is None else float(MAX_TRANSLATION_STEP)
        ),
        "max_rotation_step_deg": (
            None if MAX_ROTATION_STEP_DEG is None else float(MAX_ROTATION_STEP_DEG)
        ),
        "hard_translation_step": (
            None if HARD_TRANSLATION_STEP is None else float(HARD_TRANSLATION_STEP)
        ),
        "hard_rotation_step_deg": (
            None if HARD_ROTATION_STEP_DEG is None else float(HARD_ROTATION_STEP_DEG)
        ),
        "adaptive_step_fraction": float(ADAPTIVE_STEP_FRACTION),
        "continue_on_task_failure": bool(CONTINUE_ON_TASK_FAILURE),
        "rpe_delta": int(RPE_DELTA),
        "save_task_viz": bool(SAVE_TASK_VIZ),
        "task_viz_every": int(TASK_VIZ_EVERY),
        "dynamic_ibvs_iters": bool(DYNAMIC_IBVS_ITERS),
        "sigma_blur": float(SIGMA_BLUR),
        "use_gzn": bool(USE_GZN),
        "grad_percentile": float(GRAD_PERCENTILE),
        "photometric_max_pixels": int(PHOTOMETRIC_MAX_PIXELS),
        "use_huber": bool(USE_HUBER),
        "huber_k": None if HUBER_K is None else float(HUBER_K),
    }


def find_latest_resumable_run(scene_name, renderer, controller, depth, feature_matcher):
    """Latest flat run folder for this exact config that has progress to resume.

    Matches RUNS/<NAME> and its '-NN' collision variants, where NAME is the
    SCENE-RENDERER-CONTROLLER-DEPTH[-FEATUREMATCHER] folder name.
    """
    name = run_folder_name(
        scene_name, controller, depth, feature_matcher, renderer=renderer
    )
    if not RUNS_ROOT.exists():
        return None
    candidates = [
        entry
        for entry in RUNS_ROOT.iterdir()
        if entry.is_dir()
        and (entry.name == name or entry.name.startswith(name + "-"))
        and (entry / "per_task_errors.csv").exists()
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def run_scene(scene_name, run_root, resume=False):
    import numpy as np

    from controllers import IBVSController
    from features import FeatureMatcher
    from photometric import PhotometricControllerTorch
    from servo import copy_camera_with_pose, run_servo_loop
    from viz import save_current_desired_error_visualization, save_side_by_side

    def _residual(rendered, target, camera):
        if CONTROLLER == "ibvs" and isinstance(controller, IBVSController):
            return float(
                controller.feature_error_px(rendered, target, camera)
            )
        return float(
            np.mean(
                (np.asarray(rendered, dtype=np.float64)
                 - np.asarray(target, dtype=np.float64)) ** 2
            )
        )

    def _diverge_threshold():
        return DIVERGE_RESIDUAL_PX if CONTROLLER == "ibvs" else DIVERGE_MSE_PER_PX

    def _residual_label():
        return "feat_err_px" if CONTROLLER == "ibvs" else "mse_per_px"

    def _fmt_residual(value):
        return (
            f"{value:.2f}" if CONTROLLER == "ibvs" else f"{value:.3e}"
        )

    scene_dir = PROJECT_ROOT / "DATA" / scene_name
    if not scene_dir.exists():
        raise RuntimeError(f"Scene directory not found: {scene_dir}")

    scene, frame_index = load_trajectory_scene_and_frames(scene_dir)
    frame_ids = sorted_frame_ids(frame_index)

    takes = None
    if USE_TAKES:
        from takes import ensure_takes
        takes = ensure_takes(
            scene_dir, frame_index, jump_factor=TAKES_JUMP_FACTOR
        )

    tasks = build_trajectory_tasks(frame_ids, takes)
    if not tasks:
        raise RuntimeError(
            f"{scene_name}: no valid frame pairs at stride {STRIDE} "
            f"from START_INDEX={START_INDEX}"
        )
    num_takes = len(takes) if takes else 1

    initial_frame = frame_index[tasks[0]["src_frame"]]
    initial_camera = initial_frame["camera"]

    matcher = FeatureMatcher(method=FEATURE_METHOD)
    if CONTROLLER == "ibvs":
        controller = IBVSController(
            matcher=matcher,
            gain=GAIN_IBVS,
            min_features=MIN_FEATURES,
            scene=scene,
            use_intrinsic_depth=(DEPTH_MODE == "intrinsic"),
            ratio=RATIO,
            stop_residual_px=STOP_RESIDUAL_PX,
            min_interaction_rank=MIN_INTERACTION_RANK,
            max_interaction_condition=MAX_INTERACTION_CONDITION,
        )
    elif CONTROLLER == "photometric":
        import torch
        controller = PhotometricControllerTorch(
            scene=scene,
            target_camera=None,
            gain=GAIN_PHOTO,
            sigma_blur=SIGMA_BLUR,
            use_gzn=USE_GZN,
            grad_percentile=GRAD_PERCENTILE,
            max_pixels=PHOTOMETRIC_MAX_PIXELS,
            use_huber=USE_HUBER,
            huber_k=HUBER_K,
            use_intrinsic_depth=(DEPTH_MODE == "intrinsic"),
            method="lm",
            stop_mse_per_px=STOP_MSE_PER_PX,
            stop_ssd=STOP_SSD,
            min_interaction_rank=MIN_INTERACTION_RANK,
            max_interaction_condition=MAX_INTERACTION_CONDITION,
            device="cuda" if torch.cuda.is_available() else "cpu",
        )
    else:
        raise ValueError(f"Unknown CONTROLLER={CONTROLLER!r}")

    scene_out = trajectory_scene_dir(run_root, scene_name, resume=resume)
    scene_out.mkdir(parents=True, exist_ok=True)
    viz_dir = scene_out / "visualizations" if SAVE_TASK_VIZ else None
    if viz_dir is not None:
        viz_dir.mkdir(parents=True, exist_ok=True)

    sim_tum = scene_out / "sim_traj.tum"
    gt_tum = scene_out / "gt_traj.tum"
    csv_path = scene_out / "per_task_errors.csv"

    resume_offset = 0
    sim_poses = [initial_camera.T_world_cam.copy()]
    gt_poses = [initial_camera.T_world_cam.copy()]
    timestamps = [0.0]
    per_task_rows = []

    if resume and sim_tum.exists() and gt_tum.exists() and csv_path.exists():
        prev_rows = read_per_task_csv(csv_path)
        sim_ts, sim_loaded = read_tum(sim_tum)
        _, gt_loaded = read_tum(gt_tum)
        expected = len(prev_rows) + 1
        if len(sim_loaded) != expected or len(gt_loaded) != expected:
            raise RuntimeError(
                f"{scene_name}: resume mismatch — csv rows={len(prev_rows)}, "
                f"sim_tum poses={len(sim_loaded)}, gt_tum poses={len(gt_loaded)}"
            )
        sim_poses = list(sim_loaded)
        gt_poses = list(gt_loaded)
        timestamps = list(sim_ts)
        per_task_rows = prev_rows
        resume_offset = len(prev_rows)
        print(
            f"[{scene_name}] resume: {resume_offset} tasks already done, "
            f"continuing from task {resume_offset}"
        )

    if resume_offset >= len(tasks):
        print(f"[{scene_name}] all {len(tasks)} tasks complete; running eval only")
        current_camera = copy_camera_with_pose(initial_camera, sim_poses[-1])
    else:
        current_camera = copy_camera_with_pose(initial_camera, sim_poses[-1])
        if resume_offset == 0:
            write_tum(sim_tum, timestamps, sim_poses)
            write_tum(gt_tum, timestamps, gt_poses)

    baseline_feature_err_px = None  # ||n_r_1 - n_q_1||_2 for dynamic schedule

    for task_idx, task in enumerate(tasks):
        if task_idx < resume_offset:
            continue
        src_frame_id = task["src_frame"]
        tgt_frame_id = task["target_frame"]
        src_frame = frame_index[src_frame_id]
        tgt_frame = frame_index[tgt_frame_id]
        target_camera = tgt_frame["camera"]

        if task["reset"]:
            current_camera = copy_camera_with_pose(
                src_frame["camera"],
                src_frame["camera"].T_world_cam.copy(),
            )
            baseline_feature_err_px = None  # restart paper schedule on reset

        target_image = load_rgb(
            tgt_frame["rgb_path"],
            current_camera.W,
            current_camera.H,
        )
        if isinstance(controller, PhotometricControllerTorch):
            controller.set_target_camera(target_camera)

        initial_translation_gap = translation_error_from_pose(
            current_camera.T_world_cam,
            target_camera.T_world_cam,
        )
        initial_rotation_gap = rotation_error_from_pose(
            current_camera.T_world_cam,
            target_camera.T_world_cam,
        )
        motion_max_translation_step, motion_max_rotation_step_deg = adaptive_motion_limits(
            initial_translation_gap,
            initial_rotation_gap,
            max_translation_step=MAX_TRANSLATION_STEP,
            max_rotation_step_deg=MAX_ROTATION_STEP_DEG,
            adaptive_step_fraction=ADAPTIVE_STEP_FRACTION,
        )

        # Paper eq. 7: dynamically choose iter count for this IBVS launch.
        iters_for_task = int(MINI_ITERATIONS)
        dynamic_w = float("nan")
        feature_err_px = float("nan")
        if DYNAMIC_IBVS_ITERS and isinstance(controller, IBVSController):
            pre_render = scene.render(current_camera)
            feature_err_px = controller.feature_error_px(
                pre_render, target_image, current_camera,
            )
            if baseline_feature_err_px is None or baseline_feature_err_px <= 1e-6:
                baseline_feature_err_px = feature_err_px
                dynamic_w = 1.0
            else:
                dynamic_w = feature_err_px / baseline_feature_err_px
            iters_for_task = _dynamic_n_iter(dynamic_w, MINI_ITERATIONS)

        step_callback = None
        if _show_step_progress():
            pair_tag = f"[{scene_name}] task={task_idx:04d} {src_frame_id}->{tgt_frame_id}"

            def step_callback(item, _target_T=target_camera.T_world_cam,
                              _init_gap=initial_translation_gap, _tag=pair_tag,
                              _total=iters_for_task):
                # In-place, ephemeral progress line: overwritten each iteration
                # (\r + clear-to-EOL) and never written to the run logs.
                it = int(item["iteration"]) + 1
                t_gap = translation_error_from_pose(item["T_world_cam"], _target_T)
                info = item.get("controller_info", {})
                _, err_value = controller_error_display(info)
                sys.stdout.write(
                    f"\r\033[K{_tag} iter{it:03d}/{_total:03d} "
                    f"Error={err_value} "
                    f"t_gap={format_stat(t_gap, 6)}"
                )
                sys.stdout.flush()
                return False

        depth_preflight_info = None
        try:
            if needs_intrinsic_depth_preflight(CONTROLLER, DEPTH_MODE):
                depth_preflight_info = depth_preflight(
                    scene,
                    target_camera,
                    renderer=RENDERER,
                    frame_id=tgt_frame_id,
                )

            result = run_servo_loop(
                scene,
                current_camera,
                target_image,
                controller,
                iterations=iters_for_task,
                dt=DT,
                visualization_dir=None,
                matcher=matcher,
                feature_method=FEATURE_METHOD,
                iteration_callback=step_callback,
                viz_iter=0,
                max_translation_step=motion_max_translation_step,
                max_rotation_step_deg=motion_max_rotation_step_deg,
                hard_translation_step=HARD_TRANSLATION_STEP,
                hard_rotation_step_deg=HARD_ROTATION_STEP_DEG,
            )
        except Exception as e:
            if step_callback is not None:
                sys.stdout.write("\r\033[K")
                sys.stdout.flush()
            traceback.print_exc()
            failure_info = {
                "task_index": int(task_idx),
                "src_frame": src_frame_id,
                "target_frame": tgt_frame_id,
                "error_type": type(e).__name__,
                "error_message": str(e),
            }
            try:
                with open(scene_out / "failure.json", "w") as f:
                    json.dump(failure_info, f, indent=2)
            except OSError:
                pass

            if _is_cuda_out_of_memory(e):
                print(
                    f"[{scene_name}] task {task_idx:04d} hit CUDA out-of-memory "
                    f"({type(e).__name__}: {e}); stopping scene instead of "
                    f"cutting through remaining tasks. Last successful task: "
                    f"{per_task_rows[-1]['task_index'] if per_task_rows else 'none'}. "
                    f"Proceeding to evaluation on completed tasks."
                )
                break

            if not CONTINUE_ON_TASK_FAILURE:
                print(
                    f"[{scene_name}] task {task_idx:04d} FAILED ({type(e).__name__}: {e}); "
                    f"aborting scene. Last successful task: "
                    f"{per_task_rows[-1]['task_index'] if per_task_rows else 'none'}. "
                    f"Proceeding to evaluation on completed tasks."
                )
                break

            gt_T = target_camera.T_world_cam.copy()
            sim_poses.append(gt_T.copy())
            gt_poses.append(gt_T.copy())
            timestamps.append(float(task_idx + 1))
            cut_row = {
                "task_index": task_idx,
                "sequence_reset": int(bool(task["reset"])),
                "src_frame": src_frame_id,
                "target_frame": tgt_frame_id,
                "src_rgb": str(frame_index[src_frame_id]["rgb_path"]),
                "target_rgb": str(tgt_frame["rgb_path"]),
                "iterations_planned": int(iters_for_task),
                "iterations_run": 0,
                "dynamic_w": float(dynamic_w),
                "feature_err_px": float(feature_err_px),
                "final_residual": float("nan"),
                "diverged": 0,
                "cut_to_desired": 1,
                "failure_type": type(e).__name__,
                "failure_message": str(e),
                "stop_reason": f"cut_after_{type(e).__name__}",
                "depth_valid_pixels": (
                    "" if depth_preflight_info is None
                    else int(depth_preflight_info["valid_pixels"])
                ),
                "depth_finite_ratio": (
                    "" if depth_preflight_info is None
                    else float(depth_preflight_info["finite_ratio"])
                ),
                "initial_translation_gap": float(initial_translation_gap),
                "initial_rotation_gap_deg": float(initial_rotation_gap),
                "motion_max_translation_step": motion_max_translation_step,
                "motion_max_rotation_step_deg": motion_max_rotation_step_deg,
                "pre_cut_translation_gap": "",
                "pre_cut_rotation_error_deg": "",
                "translation_gap": 0.0,
                "rotation_error_deg": 0.0,
                "fps": 0.0,
                "render_fps": 0.0,
                "iter_ms_mean": 0.0,
                "render_ms_mean": 0.0,
                "controller_ms_mean": 0.0,
                "loop_s": 0.0,
                "viz_path": "",
            }
            per_task_rows.append(cut_row)
            append_task_row(csv_path, cut_row)
            write_tum(sim_tum, timestamps, sim_poses)
            write_tum(gt_tum, timestamps, gt_poses)
            current_camera = copy_camera_with_pose(target_camera, gt_T)
            baseline_feature_err_px = None
            print(
                f"[{scene_name}] task {task_idx:04d} FAILED ({type(e).__name__}: {e}); "
                f"cutting to {tgt_frame_id} desired pose and continuing."
            )
            continue

        if step_callback is not None:
            # Drop the ephemeral progress line; the final per-task summary below
            # takes its place on a clean line.
            sys.stdout.write("\r\033[K")
            sys.stdout.flush()

        final_camera = result["camera"]
        sim_T = final_camera.T_world_cam.copy()
        gt_T = target_camera.T_world_cam.copy()

        translation_err = translation_error_from_pose(sim_T, gt_T)
        rotation_err = rotation_error_from_pose(sim_T, gt_T)

        sim_poses.append(sim_T)
        gt_poses.append(gt_T)
        timestamps.append(float(task_idx + 1))

        task_viz_path = None
        if (
            viz_dir is not None
            and TASK_VIZ_EVERY > 0
            and task_idx % int(TASK_VIZ_EVERY) == 0
        ):
            final_render = result.get("rendered")
            if final_render is None:
                final_render = scene.render(final_camera)
            task_viz_path = viz_dir / f"task_{task_idx:04d}_final_vs_target.png"
            if CONTROLLER == "photometric":
                save_current_desired_error_visualization(
                    final_render,
                    target_image,
                    task_viz_path,
                    current_label="final",
                    desired_label="desired",
                )
            else:
                save_side_by_side(final_render, target_image, task_viz_path)

        timing = result.get("timing", {}) or {}
        task_row = {
            "task_index": task_idx,
            "sequence_reset": int(bool(task["reset"])),
            "src_frame": src_frame_id,
            "target_frame": tgt_frame_id,
            "src_rgb": str(frame_index[src_frame_id]["rgb_path"]),
            "target_rgb": str(tgt_frame["rgb_path"]),
            "iterations_planned": int(iters_for_task),
            "iterations_run": int(len(result["history"])),
            "dynamic_w": float(dynamic_w),
            "feature_err_px": float(feature_err_px),
            "cut_to_desired": 0,
            "failure_type": "",
            "failure_message": "",
            "stop_reason": result["stop_reason"],
            "depth_valid_pixels": (
                "" if depth_preflight_info is None
                else int(depth_preflight_info["valid_pixels"])
            ),
            "depth_finite_ratio": (
                "" if depth_preflight_info is None
                else float(depth_preflight_info["finite_ratio"])
            ),
            "initial_translation_gap": float(initial_translation_gap),
            "initial_rotation_gap_deg": float(initial_rotation_gap),
            "motion_max_translation_step": motion_max_translation_step,
            "motion_max_rotation_step_deg": motion_max_rotation_step_deg,
            "pre_cut_translation_gap": "",
            "pre_cut_rotation_error_deg": "",
            "translation_gap": float(translation_err),
            "rotation_error_deg": float(rotation_err),
            "fps": float(timing.get("fps", 0.0)),
            "render_fps": float(timing.get("render_fps", 0.0)),
            "iter_ms_mean": float(timing.get("iter_ms_mean", 0.0)),
            "render_ms_mean": float(timing.get("render_ms_mean", 0.0)),
            "controller_ms_mean": float(timing.get("controller_ms_mean", 0.0)),
            "loop_s": float(timing.get("loop_s", 0.0)),
            "viz_path": str(task_viz_path) if task_viz_path is not None else "",
        }
        per_task_rows.append(task_row)

        final_rendered = result.get("rendered")
        if final_rendered is None:
            final_rendered = scene.render(final_camera)
        final_resid = _residual(final_rendered, target_image, final_camera)

        last_history = result["history"][-1] if result["history"] else {}
        last_info = last_history.get("controller_info", {})
        fault_reason = last_info.get("fault_reason")

        # A controller fault (e.g. no usable pixels/features found because the
        # camera renders empty geometry) or a non-finite residual is a failure,
        # not convergence. NaN slips past `final_resid > threshold`, so flag it
        # explicitly and route it through the diverged cut/abort path below
        # instead of silently recording a frozen task.
        threshold = _diverge_threshold()
        no_features = bool(fault_reason) or not np.isfinite(final_resid)
        residual_diverged = final_resid > threshold
        failure_reasons = []
        if residual_diverged:
            failure_reasons.append(
                f"{_residual_label()}={_fmt_residual(final_resid)} > "
                f"{_fmt_residual(threshold)}"
            )
        if (
            DIVERGE_TRANSLATION_ERROR is not None
            and np.isfinite(translation_err)
            and translation_err > float(DIVERGE_TRANSLATION_ERROR)
        ):
            failure_reasons.append(
                f"tr_error={format_stat(translation_err, 6)} > "
                f"{format_stat(DIVERGE_TRANSLATION_ERROR, 6)}"
            )
        if (
            DIVERGE_ROTATION_ERROR_DEG is not None
            and np.isfinite(rotation_err)
            and rotation_err > float(DIVERGE_ROTATION_ERROR_DEG)
        ):
            failure_reasons.append(
                f"rot_error={format_stat(rotation_err, 3)}deg > "
                f"{format_stat(DIVERGE_ROTATION_ERROR_DEG, 3)}deg"
            )
        diverged = no_features or bool(failure_reasons)

        task_row["final_residual"] = float(final_resid)
        task_row["diverged"] = int(bool(diverged))

        diverge_tag = " DIVERGED" if diverged else ""
        _, err_value = controller_error_display(last_info)
        # E/F: final feature-error magnitude per matched feature.
        num_feats = last_info.get("num_inlier_matches") or 0
        err_per_feature = final_resid / num_feats if num_feats else float("nan")

        ident = color(f"[{scene_name}] task={task_idx:04d}", "cyan")
        err_str = color(
            f"Error={err_value} E/F={format_stat(err_per_feature, 3)}",
            "yellow",
        )
        gap_str = color(
            f"t_gap={format_stat(translation_err, 6)} "
            f"rot_gap={format_stat(rotation_err, 2)}deg",
            "blue",
        )
        timing_str = color(
            f"fps={task_row['fps']:.1f} "
            f"render_ms={task_row['render_ms_mean']:.2f}",
            "magenta",
        )
        print(
            f"{ident} {src_frame_id}->{tgt_frame_id} "
            f"iters={len(result['history'])}/{MINI_ITERATIONS} "
            f"{err_str} {gap_str} {timing_str}"
            f"{color(diverge_tag, 'red', 'bold')}"
        )

        if diverged:
            if no_features:
                fail_type = fault_reason or "no_features"
                fail_desc = f"controller fault ({fail_type})"
            else:
                fail_type = "diverged"
                fail_desc = "; ".join(failure_reasons)
            if CONTINUE_ON_TASK_FAILURE:
                task_row["cut_to_desired"] = 1
                task_row["failure_type"] = fail_type
                task_row["failure_message"] = fail_desc
                task_row["stop_reason"] = f"cut_after_{fail_type}"
                task_row["pre_cut_translation_gap"] = float(translation_err)
                task_row["pre_cut_rotation_error_deg"] = float(rotation_err)
                sim_poses[-1] = gt_T.copy()
                translation_err = 0.0
                rotation_err = 0.0
                task_row["translation_gap"] = float(translation_err)
                task_row["rotation_error_deg"] = float(rotation_err)
                write_tum(sim_tum, timestamps, sim_poses)
                write_tum(gt_tum, timestamps, gt_poses)
                append_task_row(csv_path, task_row)
                current_camera = copy_camera_with_pose(target_camera, gt_T)
                baseline_feature_err_px = None
                print(
                    f"[{scene_name}] FAILED at task {task_idx:04d}: {fail_desc}. "
                    f"Cutting to {tgt_frame_id} desired pose and continuing."
                )
                continue

            task_row["failure_type"] = fail_type
            task_row["failure_message"] = fail_desc
            append_task_row(csv_path, task_row)
            print(
                f"[{scene_name}] FAILED at task {task_idx:04d}: {fail_desc}. Aborting run."
            )
            raise SystemExit(1)

        write_tum(sim_tum, timestamps, sim_poses)
        write_tum(gt_tum, timestamps, gt_poses)
        append_task_row(csv_path, task_row)
        current_camera = copy_camera_with_pose(target_camera, sim_T)

    metrics = {}
    if len(per_task_rows) >= 1:
        try:
            metrics = evaluate_and_plot(
                timestamps, sim_poses, gt_poses, scene_out, scene_name,
            )
        except Exception as e:
            traceback.print_exc()
            print(f"[{scene_name}] evaluate_and_plot failed: {type(e).__name__}: {e}")
            metrics = {"error": f"{type(e).__name__}: {e}"}
    else:
        print(f"[{scene_name}] no tasks completed; skipping evo evaluation")
        metrics = {"error": "no_completed_tasks"}

    # Completed-task breakdown: a task either converged on its own or failed
    # (diverged / errored) and was corrected by cutting to the desired pose.
    def _is_corrected(row):
        return str(row.get("cut_to_desired", 0)) in {"1", "True", "true"}

    num_corrected = sum(_is_corrected(row) for row in per_task_rows)
    num_converged = len(per_task_rows) - num_corrected
    timing = summarize_task_timing(per_task_rows)
    timing_str = color(
        f"avg_fps={timing['fps']:.1f} "
        f"avg_render_ms={timing['render_ms_mean']:.2f}",
        "magenta",
    )
    print(
        f"\n=== {scene_name} completed tasks ({len(per_task_rows)} total) ===\n"
        f"  converged: {num_converged}\n"
        f"  failed (corrected): {num_corrected}\n"
        f"  timing: {timing_str}"
    )

    summary = {
        "scene": scene_name,
        "scene_dir": str(scene_dir),
        "renderer": RENDERER,
        "controller": CONTROLLER,
        "nerf_render_scale": float(NERF_RENDER_SCALE),
        "gs_render_scale": float(GS_RENDER_SCALE),
        "mesh_render_scale": float(MESH_RENDER_SCALE),
        "mesh_path": (
            str(scene_dir / "mesh.ply") if RENDERER == "mesh" else None
        ),
        "stride": int(STRIDE),
        "mini_iterations": int(MINI_ITERATIONS),
        "depth_mode": DEPTH_MODE,
        "feature_method": FEATURE_METHOD,
        "gain_ibvs": float(GAIN_IBVS),
        "gain_photo": float(GAIN_PHOTO),
        "ratio": int(RATIO),
        "start_index": int(START_INDEX),
        "max_pairs": MAX_PAIRS,
        "stop_residual_px": float(STOP_RESIDUAL_PX),
        "stop_mse_per_px": float(STOP_MSE_PER_PX),
        "stop_ssd": (None if STOP_SSD is None else float(STOP_SSD)),
        "diverge_residual_px": float(DIVERGE_RESIDUAL_PX),
        "diverge_mse_per_px": float(DIVERGE_MSE_PER_PX),
        "diverge_translation_error": (
            None
            if DIVERGE_TRANSLATION_ERROR is None
            else float(DIVERGE_TRANSLATION_ERROR)
        ),
        "diverge_rotation_error_deg": (
            None
            if DIVERGE_ROTATION_ERROR_DEG is None
            else float(DIVERGE_ROTATION_ERROR_DEG)
        ),
        "min_interaction_rank": int(MIN_INTERACTION_RANK),
        "max_interaction_condition": (
            None
            if MAX_INTERACTION_CONDITION is None
            else float(MAX_INTERACTION_CONDITION)
        ),
        "max_translation_step": (
            None if MAX_TRANSLATION_STEP is None else float(MAX_TRANSLATION_STEP)
        ),
        "max_rotation_step_deg": (
            None if MAX_ROTATION_STEP_DEG is None else float(MAX_ROTATION_STEP_DEG)
        ),
        "hard_translation_step": (
            None if HARD_TRANSLATION_STEP is None else float(HARD_TRANSLATION_STEP)
        ),
        "hard_rotation_step_deg": (
            None if HARD_ROTATION_STEP_DEG is None else float(HARD_ROTATION_STEP_DEG)
        ),
        "adaptive_step_fraction": float(ADAPTIVE_STEP_FRACTION),
        "continue_on_task_failure": bool(CONTINUE_ON_TASK_FAILURE),
        "save_task_viz": bool(SAVE_TASK_VIZ),
        "task_viz_every": int(TASK_VIZ_EVERY),
        "use_takes": bool(USE_TAKES),
        "num_takes": int(num_takes),
        "num_tasks": len(per_task_rows),
        "num_cuts": num_corrected,
        "num_converged": num_converged,
        "num_failed_corrected": num_corrected,
        "timing": timing,
        "camera": camera_metadata(initial_camera),
        "sim_traj": str(sim_tum),
        "gt_traj": str(gt_tum),
        "per_task_csv": str(scene_out / "per_task_errors.csv"),
        "metrics": metrics,
    }
    with open(scene_out / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    return summary


def run(args):
    from experiment_config import (
        TRAJECTORY_CONFIG_KEYS,
        apply_config,
        format_applied_config,
        load_cli_config,
    )

    applied_config = load_cli_config(
        args.config,
        args.set,
        TRAJECTORY_CONFIG_KEYS,
        "trajectory",
    )
    diverge_mse_per_px = getattr(args, "diverge_mse_per_px", None)
    diverge_residual_px = getattr(args, "diverge_residual_px", None)
    diverge_translation_error = getattr(args, "diverge_translation_error", None)
    diverge_rotation_error_deg = getattr(args, "diverge_rotation_error_deg", None)
    if diverge_mse_per_px is not None:
        if diverge_mse_per_px <= 0.0:
            raise ValueError("--diverge-mse-per-px must be > 0")
        applied_config["diverge_mse_per_px"] = diverge_mse_per_px
    if diverge_residual_px is not None:
        if diverge_residual_px <= 0.0:
            raise ValueError("--diverge-residual-px must be > 0")
        applied_config["diverge_residual_px"] = diverge_residual_px
    if diverge_translation_error is not None:
        if diverge_translation_error <= 0.0:
            raise ValueError("--diverge-translation-error must be > 0")
        applied_config["diverge_translation_error"] = diverge_translation_error
    if diverge_rotation_error_deg is not None:
        if diverge_rotation_error_deg <= 0.0:
            raise ValueError("--diverge-rotation-error-deg must be > 0")
        applied_config["diverge_rotation_error_deg"] = diverge_rotation_error_deg
    if getattr(args, "continue_on_task_failure", None) is not None:
        applied_config["continue_on_task_failure"] = bool(args.continue_on_task_failure)
    apply_config(applied_config, globals(), TRAJECTORY_CONFIG_KEYS)
    if applied_config:
        print(f"Applied trajectory config: {format_applied_config(applied_config)}")

    if RUN_TAG:
        tag = RUN_TAG
    elif CONTROLLER == "ibvs":
        tag = f"{FEATURE_METHOD}_stride{STRIDE}_iters{MINI_ITERATIONS}"
    else:
        tag = f"{CONTROLLER}_stride{STRIDE}_iters{MINI_ITERATIONS}"

    resume = args.resume is not None
    explicit_resume_root = None
    if resume and args.resume != "auto":
        explicit_resume_root = Path(args.resume).resolve()
        if not explicit_resume_root.exists():
            raise FileNotFoundError(f"--resume path not found: {explicit_resume_root}")

    matcher = trajectory_feature_matcher()

    overall = {}
    multi_dataset = len(DATASETS) > 1
    for scene_name in DATASETS:
        scene_resume = resume
        if not resume:
            run_root = run_folder(
                RUNS_ROOT,
                scene_name,
                CONTROLLER,
                DEPTH_MODE,
                matcher,
                renderer=RENDERER,
            )
        elif explicit_resume_root is not None:
            run_root = explicit_resume_root
            print(f"Resuming run: {run_root}")
        else:  # --resume auto
            run_root = find_latest_resumable_run(
                scene_name, RENDERER, CONTROLLER, DEPTH_MODE, matcher
            )
            if run_root is None:
                print(
                    f"--resume: no resumable run found under RUNS/ for "
                    f"{run_folder_name(scene_name, CONTROLLER, DEPTH_MODE, matcher, renderer=RENDERER)}. "
                    f"Starting a fresh run."
                )
                scene_resume = False
                run_root = run_folder(
                    RUNS_ROOT,
                    scene_name,
                    CONTROLLER,
                    DEPTH_MODE,
                    matcher,
                    renderer=RENDERER,
                )
            else:
                print(f"Resuming run: {run_root}")

        run_root.mkdir(parents=True, exist_ok=True)
        write_run_json(
            run_root / "config.resolved.json",
            resolved_trajectory_config(tag, run_root),
        )
        write_command(run_root / "command.txt")

        try:
            result = run_scene(scene_name, run_root, resume=scene_resume)
        except SystemExit as e:
            if not multi_dataset:
                raise
            traceback.print_exc()
            result = {"error": f"SystemExit({e.code})"}
        except Exception as e:
            traceback.print_exc()
            result = {"error": str(e)}
        overall[scene_name] = result

        # Each scene's folder is self-contained: write its details straight in.
        scene_summary = {scene_name: result}
        write_run_json(run_root / "trajectory_summary.json", scene_summary)
        write_run_json(run_root / "summary.json", scene_summary)
        write_run_json(
            run_root / "config.resolved.json",
            resolved_trajectory_config(tag, run_root),
        )
        write_command(run_root / "command.txt")
        write_trajectory_summary_markdown(run_root / "summary.md", scene_summary)
        write_trajectory_summary_markdown(
            run_root / "trajectory_summary.md", scene_summary
        )
        write_run_readme(
            run_root / "README.md",
            "SERVIS Trajectory Run",
            fields=[
                ("scene", scene_name),
                ("renderer", RENDERER),
                ("controller", CONTROLLER),
                ("depth", DEPTH_MODE),
                ("feature", FEATURE_METHOD),
                ("tag", tag),
            ],
            artifacts=[
                ("summary table", "summary.md"),
                ("machine summary", "summary.json"),
                ("resolved config", "config.resolved.json"),
                ("trajectories", "sim_traj.tum, gt_traj.tum"),
                ("per-task errors", "per_task_errors.csv"),
            ],
        )
        print(f"\nWrote {run_root}")
        print(f"Summary table: {run_root / 'summary.md'}")
