"""Single frame-to-frame servo runner.

Use via the CLI:
    python cli.py servo-frames --config <path> [--set k=v ...]

Heavy imports (controllers, scenes, photometric) are deferred to `run()` so
this module can be imported by `cli.py` without pulling them in.
"""

import csv
import json
import math
import re
from pathlib import Path

from run_layout import (
    unique_run_root,
    write_command,
    write_json as write_run_json,
    write_run_readme,
)


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
RUNS_ROOT = PROJECT_ROOT / "RUNS"

# Edit these values to define the frame-to-frame servo experiment.
SCENE_DIR = PROJECT_ROOT / "DATA" / "kitchen"
RENDERER = "mesh"  # "mesh", "gs", or "nerf"
START_INDEX = 1
INDEX_AWAY = 1
TARGET_INDEX = None
ITERATIONS = 100
DT = 1.0
DEPTH_MODE = "intrinsic"  # "learned" = MoGe2, "intrinsic" = scene.render_depth()
FEATURE_METHOD = "sift"
VIZ_ITER = 1
GAIN_IBVS = 0.75
GAIN_PHOTO = 0.005
MIN_FEATURES = 3
RATIO = 1
RUN_NAME = None
STOP_RESIDUAL_PX = 0.5      # IBVS: RMS reprojection error (px)
STOP_MSE_PER_PX = 2.0e-6    # legacy photometric MSE; used when STOP_SSD is None
STOP_SSD = None             # photometric: ViSP-style SSD threshold, or None
MIN_INTERACTION_RANK = 6
MAX_INTERACTION_CONDITION = 1.0e8

CONTROLLER = "ibvs"  # "ibvs", "photometric", or "photometric_torch"
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
            "--set renderer=mesh --set scene=kitchen."
        ),
    )


def normalize_frame_id(value):
    text = str(value)
    match = re.search(r"(\d+)", text)
    if match is None:
        raise ValueError(f"Could not parse frame id from {value!r}")
    return f"frame-{int(match.group(1)):06d}"


def frame_id_from_path(path):
    return normalize_frame_id(Path(path).name)


def frame_number(frame_id):
    return int(normalize_frame_id(frame_id).split("-")[1])


def load_rgb(rgb_path, width, height):
    import numpy as np
    from PIL import Image

    image = Image.open(rgb_path).convert("RGB")
    image = image.resize((int(width), int(height)))
    return np.asarray(image, dtype=np.float32) / 255.0


def save_rgb(path, image):
    import numpy as np
    from PIL import Image

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    image_u8 = (np.asarray(image) * 255.0).clip(0, 255).astype(np.uint8)
    Image.fromarray(image_u8).save(path)


def make_frame_index(records, renderer):
    index = {}
    for record in records:
        if len(record) == 3:
            camera, rgb_path, _ = record
        elif len(record) == 2:
            camera, rgb_path = record
        else:
            raise ValueError(
                f"Unexpected record arity {len(record)} for renderer {renderer!r}"
            )
        index[frame_id_from_path(rgb_path)] = {
            "camera": camera,
            "rgb_path": rgb_path,
        }
    return index


def available_frame_hint(frame_index, limit=8):
    frame_ids = sorted(frame_index, key=frame_number)
    if not frame_ids:
        return "none"
    head = ", ".join(frame_ids[:limit])
    tail = ", ".join(frame_ids[-limit:])
    if len(frame_ids) <= 2 * limit:
        return ", ".join(frame_ids)
    return f"{head}, ..., {tail}"


def sorted_frame_ids(frame_index):
    return sorted(frame_index, key=frame_number)


def resolve_target_index(start_index, target_index, index_away):
    if target_index is not None:
        return int(target_index)
    return int(start_index) + int(index_away)


def resolve_frame_from_index(frame_index, renderer, logical_index):
    logical_index = int(logical_index)
    if logical_index < 1:
        raise ValueError("Frame indexes are 1-based; use START_INDEX >= 1")

    if renderer == "mesh":
        frame_id = f"frame-{logical_index:06d}"
        if frame_id not in frame_index:
            raise RuntimeError(
                f"Missing logical index {logical_index} as {frame_id} for mesh. "
                f"Available frames include: {available_frame_hint(frame_index)}"
            )
        return frame_id

    frame_ids = sorted_frame_ids(frame_index)
    position = logical_index - 1
    if position >= len(frame_ids):
        raise RuntimeError(
            f"Missing logical index {logical_index} for {renderer}; "
            f"only {len(frame_ids)} frames are loaded. "
            f"Available frames include: {available_frame_hint(frame_index)}"
        )
    return frame_ids[position]


def load_scene_and_frames(scene_dir, renderer):
    from dataset import load_colmap

    records = load_colmap(scene_dir)
    if renderer == "mesh":
        from scenes.mesh import MeshScene
        scene = MeshScene(scene_dir / "mesh.ply")
    elif renderer == "gs":
        from scenes.gs import GSScene
        scene = GSScene(scene_dir / "gs.ply")
    elif renderer == "nerf":
        from scenes.nerf import NeRFScene
        scene = NeRFScene(scene_dir)
    else:
        raise ValueError(f"Unknown renderer {renderer!r}")

    frame_index = make_frame_index(records, renderer)
    if not frame_index:
        raise RuntimeError(f"No frames loaded for {renderer} from {scene_dir}")
    return scene, frame_index


def depth_preflight(
    scene,
    camera,
    *,
    renderer,
    frame_id,
    min_depth=1.0e-4,
    min_valid_pixels=6,
):
    import numpy as np

    render_depth = getattr(scene, "render_depth", None)
    if not callable(render_depth):
        raise RuntimeError(
            f"renderer {renderer} has no render_depth() for COLMAP frame {frame_id}"
        )

    try:
        depth = np.asarray(render_depth(camera), dtype=np.float32)
    except Exception as exc:
        raise RuntimeError(
            f"renderer {renderer} failed depth preflight for COLMAP frame "
            f"{frame_id}: {type(exc).__name__}: {exc}"
        ) from exc

    finite = np.isfinite(depth)
    positive = finite & (depth > float(min_depth))
    valid_pixels = int(np.count_nonzero(positive))
    total_pixels = int(depth.size)
    finite_pixels = int(np.count_nonzero(finite))
    finite_ratio = (float(finite_pixels) / float(total_pixels)) if total_pixels else 0.0

    if valid_pixels:
        valid_depths = depth[positive]
        min_valid_depth = float(valid_depths.min())
        max_valid_depth = float(valid_depths.max())
    else:
        min_valid_depth = float("nan")
        max_valid_depth = float("nan")

    info = {
        "renderer": str(renderer),
        "frame_id": str(frame_id),
        "valid_pixels": valid_pixels,
        "total_pixels": total_pixels,
        "finite_pixels": finite_pixels,
        "finite_ratio": finite_ratio,
        "min_depth": min_valid_depth,
        "max_depth": max_valid_depth,
        "min_depth_threshold": float(min_depth),
        "min_valid_pixels": int(min_valid_pixels),
    }

    if valid_pixels < int(min_valid_pixels):
        raise RuntimeError(
            f"renderer {renderer} produced {valid_pixels} valid depth pixels "
            f"for COLMAP frame {frame_id} "
            f"(need >= {int(min_valid_pixels)}, finite_ratio={finite_ratio:.6f}, "
            f"min_depth={format_stat(min_valid_depth, 6)}, "
            f"max_depth={format_stat(max_valid_depth, 6)})"
        )

    return info


def needs_intrinsic_depth_preflight(controller_kind, depth_mode):
    return (
        str(depth_mode) == "intrinsic"
        and str(controller_kind) in ("photometric", "photometric_torch")
    )


def rotation_error_from_pose(T_world_cam, target_T_world_cam):
    import numpy as np

    R_current = T_world_cam[:3, :3]
    R_target = target_T_world_cam[:3, :3]
    R_delta = R_target.T @ R_current
    cos_angle = (np.trace(R_delta) - 1.0) * 0.5
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_angle)))


def rotation_error_deg(camera, target_camera):
    return rotation_error_from_pose(camera.T_world_cam, target_camera.T_world_cam)


def translation_error_from_pose(T_world_cam, target_T_world_cam):
    import numpy as np

    delta = T_world_cam[:3, 3] - target_T_world_cam[:3, 3]
    return float(np.linalg.norm(delta))


def translation_gap(camera, target_camera):
    return translation_error_from_pose(camera.T_world_cam, target_camera.T_world_cam)


def format_stat(value, digits=4):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return "-"
    if not math.isfinite(value):
        return "-"
    return f"{value:.{int(digits)}f}"


def format_sci(value):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return "-"
    if math.isnan(value):
        return "-"
    if math.isinf(value):
        return "inf" if value > 0 else "-inf"
    return f"{value:.2e}"


def short_stop_reason(reason):
    if reason in (None, "", "None"):
        return "-"
    text = str(reason)
    prefixes = ("measurement_invalid_", "velocity_invalid_")
    for prefix in prefixes:
        if text.startswith(prefix):
            return text[len(prefix):]
    return text


def controller_error_display(info):
    if info.get("residual_rms_px") is not None:
        return "err_px", format_stat(info.get("residual_rms_px"), 3)
    if info.get("stop_ssd") is not None:
        return "ssd", format_sci(info.get("stop_ssd"))
    if info.get("raw_image_mse_per_px") is not None:
        return "raw_mse", format_sci(info.get("raw_image_mse_per_px"))
    return "err", format_sci(info.get("residual_norm"))


def gap_closed_percent(initial_gap, final_gap):
    try:
        initial_gap = float(initial_gap)
        final_gap = float(final_gap)
    except (TypeError, ValueError):
        return float("nan")
    if not math.isfinite(initial_gap) or not math.isfinite(final_gap):
        return float("nan")
    if abs(initial_gap) <= 1e-12:
        return float("nan")
    return 100.0 * (initial_gap - final_gap) / initial_gap


def format_percent(value, digits=1):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return "-"
    if not math.isfinite(value):
        return "-"
    return f"{value:.{int(digits)}f}%"


def public_controller_info(info):
    public = dict(info or {})
    return public


def write_history_csv(path, history):
    import numpy as np

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "iteration",
        "num_matches",
        "num_inliers",
        "feature_mode",
        "controller_inliers",
        "residual_norm",
        "residual_norm_px",
        "residual_rms_px",
        "residual_ssd",
        "residual_mse_per_px",
        "weighted_residual_ssd",
        "raw_image_ssd",
        "raw_image_mse_per_px",
        "stop_ssd",
        "stop_ssd_threshold",
        "interaction_rank",
        "interaction_condition",
        "interaction_min_singular",
        "interaction_max_singular",
        "diff_max",
        "diff_mean",
        "translation_gap",
        "gap_closed_pct",
        "rotation_error_deg",
        "cached_features",
        "dropped_features",
        "mean_depth",
        "min_depth",
        "max_depth",
        "velocity_norm",
        "velocity_limited",
        "velocity_scale",
        "translation_step",
        "rotation_step_deg",
        "raw_translation_step",
        "raw_rotation_step_deg",
        "vx",
        "vy",
        "vz",
        "wx",
        "wy",
        "wz",
        "render_ms",
        "controller_ms",
        "iter_ms",
        "step_accepted",
        "stop_reason",
    ]

    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for item in history:
            info = public_controller_info(item.get("controller_info", {}))
            velocity = np.asarray(item["velocity"], dtype=np.float32)
            writer.writerow({
                "iteration": item["iteration"],
                "num_matches": item.get("num_matches", ""),
                "num_inliers": item.get("num_inliers", ""),
                "feature_mode": item.get("feature_mode", ""),
                "controller_inliers": info.get("num_inlier_matches", ""),
                "residual_norm": info.get("residual_norm", ""),
                "residual_norm_px": info.get("residual_norm_px", ""),
                "residual_rms_px": info.get("residual_rms_px", ""),
                "residual_ssd": info.get("residual_ssd", ""),
                "residual_mse_per_px": info.get("residual_mse_per_px", ""),
                "weighted_residual_ssd": info.get("weighted_residual_ssd", ""),
                "raw_image_ssd": info.get("raw_image_ssd", ""),
                "raw_image_mse_per_px": info.get("raw_image_mse_per_px", ""),
                "stop_ssd": info.get("stop_ssd", ""),
                "stop_ssd_threshold": info.get("stop_ssd_threshold", ""),
                "interaction_rank": info.get("interaction_rank", ""),
                "interaction_condition": info.get("interaction_condition", ""),
                "interaction_min_singular": info.get("interaction_min_singular", ""),
                "interaction_max_singular": info.get("interaction_max_singular", ""),
                "diff_max": item.get("diff_max", ""),
                "diff_mean": item.get("diff_mean", ""),
                "translation_gap": item.get("translation_gap", ""),
                "gap_closed_pct": item.get("gap_closed_pct", ""),
                "rotation_error_deg": item.get("rotation_error_deg", ""),
                "cached_features": info.get("num_cached_features", ""),
                "dropped_features": info.get("num_dropped_features", ""),
                "mean_depth": info.get("mean_depth", ""),
                "min_depth": info.get("min_depth", ""),
                "max_depth": info.get("max_depth", ""),
                "velocity_norm": info.get("velocity_norm", ""),
                "velocity_limited": int(bool(item.get("velocity_limited", False))),
                "velocity_scale": item.get("velocity_scale", ""),
                "translation_step": item.get("translation_step", ""),
                "rotation_step_deg": item.get("rotation_step_deg", ""),
                "raw_translation_step": item.get("raw_translation_step", ""),
                "raw_rotation_step_deg": item.get("raw_rotation_step_deg", ""),
                "vx": float(velocity[0]),
                "vy": float(velocity[1]),
                "vz": float(velocity[2]),
                "wx": float(velocity[3]),
                "wy": float(velocity[4]),
                "wz": float(velocity[5]),
                "render_ms": item.get("render_ms", ""),
                "controller_ms": item.get("controller_ms", ""),
                "iter_ms": item.get("iter_ms", ""),
                "step_accepted": int(bool(item.get("step_accepted", False))),
                "stop_reason": item.get("stop_reason", ""),
            })


def history_for_json(history):
    import numpy as np

    rows = []
    for item in history:
        rows.append({
            "iteration": int(item["iteration"]),
            "T_world_cam": np.asarray(
                item["T_world_cam"],
                dtype=np.float32,
            ).tolist(),
            "next_T_world_cam": np.asarray(
                item["next_T_world_cam"],
                dtype=np.float32,
            ).tolist(),
            "raw_velocity": np.asarray(
                item.get("raw_velocity", item["velocity"]),
                dtype=np.float32,
            ).tolist(),
            "velocity": np.asarray(item["velocity"], dtype=np.float32).tolist(),
            "step_accepted": bool(item.get("step_accepted", False)),
            "velocity_limited": bool(item.get("velocity_limited", False)),
            "velocity_scale": item.get("velocity_scale"),
            "translation_step": item.get("translation_step"),
            "rotation_step_deg": item.get("rotation_step_deg"),
            "raw_translation_step": item.get("raw_translation_step"),
            "raw_rotation_step_deg": item.get("raw_rotation_step_deg"),
            "max_translation_step": item.get("max_translation_step"),
            "max_rotation_step_deg": item.get("max_rotation_step_deg"),
            "hard_translation_step": item.get("hard_translation_step"),
            "hard_rotation_step_deg": item.get("hard_rotation_step_deg"),
            "num_matches": item.get("num_matches"),
            "num_inliers": item.get("num_inliers"),
            "feature_mode": item.get("feature_mode"),
            "translation_gap": item.get("translation_gap"),
            "gap_closed_pct": item.get("gap_closed_pct"),
            "rotation_error_deg": item.get("rotation_error_deg"),
            "visualization_path": item.get("visualization_path"),
            "diff_max": item.get("diff_max"),
            "diff_mean": item.get("diff_mean"),
            "stop_reason": item.get("stop_reason"),
            "controller_info": public_controller_info(item.get("controller_info", {})),
        })
    return rows


def controller_short_name(controller):
    name = controller.__class__.__name__
    if name.lower().endswith("controller"):
        name = name[: -len("Controller")]
    return name.lower() or "controller"


def format_gain(gain):
    text = f"{float(gain):.3g}"
    return text.replace(".", "p")


def make_run_tag(gain, feature_method, ratio):
    return f"g{format_gain(gain)}_{feature_method}_r{int(ratio)}"


def make_trial_name(start_index, target_index, depth_mode):
    if RUN_NAME is not None:
        return str(RUN_NAME)
    return f"{int(start_index)}-to-{int(target_index)}_{depth_mode}"


def make_run_dir(controller, scene_name, renderer, start_index, target_index, depth_mode):
    trial_dir = RUNS_ROOT / "servo_frames"
    active_gain = GAIN_PHOTO if CONTROLLER != "ibvs" else GAIN_IBVS
    run_tag = RUN_NAME or make_run_tag(active_gain, FEATURE_METHOD, RATIO)
    output_dir = unique_run_root(
        RUNS_ROOT,
        "servo_frames",
        [
            scene_name,
            renderer,
            controller_short_name(controller),
            depth_mode,
            FEATURE_METHOD,
            f"{int(start_index)}-to-{int(target_index)}",
            run_tag,
        ],
    )
    return trial_dir, output_dir


TRIAL_INDEX_FIELDS = [
    "run_dir",
    "timestamp",
    "controller",
    "renderer",
    "scene",
    "depth_mode",
    "feature_method",
    "gain",
    "ratio",
    "start_index",
    "target_index",
    "iterations_run",
    "stop_reason",
    "initial_translation_gap",
    "final_translation_gap",
    "translation_gap_closed_pct",
    "initial_rotation_error_deg",
    "final_rotation_error_deg",
]


def append_trial_index(trial_dir, row):
    path = Path(trial_dir) / "_index.csv"
    write_header = not path.exists()
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=TRIAL_INDEX_FIELDS)
        if write_header:
            writer.writeheader()
        writer.writerow({k: row.get(k, "") for k in TRIAL_INDEX_FIELDS})


def camera_metadata(camera):
    return {
        "fx": camera.fx,
        "fy": camera.fy,
        "cx": camera.cx,
        "cy": camera.cy,
        "height": camera.H,
        "width": camera.W,
    }


def _summary_cell(value):
    return str(value).replace("|", "\\|")


def write_servo_summary_markdown(path, summary):
    rows = [
        ("Scene", Path(summary["scene_dir"]).name),
        ("Renderer", summary["renderer"]),
        ("Controller", summary["controller"]),
        ("Frames", f"{summary['start_frame']} -> {summary['target_frame']}"),
        ("Depth", summary["depth"]),
        ("Feature", summary["feature_method"]),
        ("Iterations", f"{summary['iterations_run']}/{summary['iterations']}"),
        ("Stop", summary["stop_reason"]),
        ("Translation gap", f"{format_stat(summary['initial_translation_gap'], 6)} -> {format_stat(summary['final_translation_gap'], 6)}"),
        ("Gap closed", format_percent(summary["translation_gap_closed_pct"])),
        ("Rotation gap", f"{format_stat(summary['initial_rotation_error_deg'], 4)}deg -> {format_stat(summary['final_rotation_error_deg'], 4)}deg"),
    ]
    lines = [
        "# Servo Frame Summary",
        "",
        "Translation values are in COLMAP scene scale.",
        "",
        "| Field | Value |",
        "| ----- | ----- |",
    ]
    lines.extend(f"| {_summary_cell(k)} | {_summary_cell(v)} |" for k, v in rows)
    lines.append("")
    Path(path).write_text("\n".join(lines))


def experiment_config(
    controller_name,
    trial_dir,
    start_index,
    target_index,
    start_frame,
    target_frame,
    output_dir,
):
    return {
        "script": str(Path(__file__).resolve()),
        "project_root": str(PROJECT_ROOT),
        "runs_root": str(RUNS_ROOT),
        "trial_dir": str(trial_dir),
        "output_dir": str(output_dir),
        "controller": controller_name,
        "renderer": RENDERER,
        "scene_dir": str(SCENE_DIR),
        "frame_selection": "logical_index",
        "frame_index_base": 1,
        "start_index": int(start_index),
        "target_index": int(target_index),
        "index_away": int(INDEX_AWAY),
        "start_frame": start_frame,
        "target_frame": target_frame,
        "target_index_override": TARGET_INDEX,
        "iterations": int(ITERATIONS),
        "dt": float(DT),
        "depth_mode": DEPTH_MODE,
        "feature_method": FEATURE_METHOD,
        "viz_iter": int(VIZ_ITER),
        "gain": float(GAIN_PHOTO if CONTROLLER != "ibvs" else GAIN_IBVS),
        "gain_ibvs": float(GAIN_IBVS),
        "gain_photo": float(GAIN_PHOTO),
        "min_features": int(MIN_FEATURES),
        "ratio": int(RATIO),
        "run_name": RUN_NAME,
        "stop_residual_px": float(STOP_RESIDUAL_PX),
        "stop_mse_per_px": float(STOP_MSE_PER_PX),
        "stop_ssd": (None if STOP_SSD is None else float(STOP_SSD)),
        "min_interaction_rank": int(MIN_INTERACTION_RANK),
        "max_interaction_condition": (
            None
            if MAX_INTERACTION_CONDITION is None
            else float(MAX_INTERACTION_CONDITION)
        ),
        "controller_kind": CONTROLLER,
        "sigma_blur": float(SIGMA_BLUR),
        "use_gzn": bool(USE_GZN),
        "grad_percentile": float(GRAD_PERCENTILE),
        "photometric_max_pixels": int(PHOTOMETRIC_MAX_PIXELS),
        "use_huber": bool(USE_HUBER),
        "huber_k": (None if HUBER_K is None else float(HUBER_K)),
    }


def run(args):
    from controllers import IBVSController, PhotometricController
    from experiment_config import (
        SERVO_FRAMES_CONFIG_KEYS,
        apply_config,
        format_applied_config,
        load_cli_config,
    )
    from features import FeatureMatcher
    from photometric import PhotometricControllerTorch
    from servo import run_servo_loop
    from viz import save_current_desired_error_visualization, save_error_evolution

    applied_config = load_cli_config(
        args.config,
        args.set,
        SERVO_FRAMES_CONFIG_KEYS,
        "servo_frames",
    )
    apply_config(applied_config, globals(), SERVO_FRAMES_CONFIG_KEYS)
    if applied_config:
        print(f"Applied servo config: {format_applied_config(applied_config)}")

    if DEPTH_MODE not in ("learned", "intrinsic"):
        raise ValueError("DEPTH_MODE must be 'learned' or 'intrinsic'")

    scene, frame_index = load_scene_and_frames(SCENE_DIR, RENDERER)
    start_index = int(START_INDEX)
    target_index = resolve_target_index(start_index, TARGET_INDEX, INDEX_AWAY)
    start_frame = resolve_frame_from_index(frame_index, RENDERER, start_index)
    target_frame = resolve_frame_from_index(frame_index, RENDERER, target_index)

    start = frame_index[start_frame]
    target = frame_index[target_frame]
    start_camera = start["camera"]
    target_camera = target["camera"]

    depth_preflight_info = None
    if needs_intrinsic_depth_preflight(CONTROLLER, DEPTH_MODE):
        depth_preflight_info = depth_preflight(
            scene,
            target_camera,
            renderer=RENDERER,
            frame_id=target_frame,
        )
        print(
            f"Depth preflight {RENDERER} {target_frame}: "
            f"valid={depth_preflight_info['valid_pixels']}/"
            f"{depth_preflight_info['total_pixels']} "
            f"range={format_stat(depth_preflight_info['min_depth'], 6)}.."
            f"{format_stat(depth_preflight_info['max_depth'], 6)}"
        )

    matcher = FeatureMatcher(method=FEATURE_METHOD)
    if CONTROLLER == "ibvs":
        controller = IBVSController(
            matcher=matcher,
            gain=GAIN_IBVS,
            min_features=MIN_FEATURES,
            scene=scene,
            use_intrinsic_depth=DEPTH_MODE == "intrinsic",
            ratio=RATIO,
            stop_residual_px=STOP_RESIDUAL_PX,
            min_interaction_rank=MIN_INTERACTION_RANK,
            max_interaction_condition=MAX_INTERACTION_CONDITION,
        )
    elif CONTROLLER == "photometric":
        controller = PhotometricController(
            scene=scene,
            target_camera=target_camera,
            gain=GAIN_PHOTO,
            sigma_blur=SIGMA_BLUR,
            use_gzn=USE_GZN,
            grad_percentile=GRAD_PERCENTILE,
            max_pixels=PHOTOMETRIC_MAX_PIXELS,
            use_huber=USE_HUBER,
            huber_k=HUBER_K,
            use_intrinsic_depth=DEPTH_MODE == "intrinsic",
            stop_mse_per_px=STOP_MSE_PER_PX,
            stop_ssd=STOP_SSD,
            min_interaction_rank=MIN_INTERACTION_RANK,
            max_interaction_condition=MAX_INTERACTION_CONDITION,
        )
    elif CONTROLLER == "photometric_torch":
        controller = PhotometricControllerTorch(
            scene=scene,
            target_camera=target_camera,
            gain=GAIN_PHOTO,
            sigma_blur=SIGMA_BLUR,
            use_gzn=USE_GZN,
            grad_percentile=GRAD_PERCENTILE,
            max_pixels=PHOTOMETRIC_MAX_PIXELS,
            use_huber=USE_HUBER,
            huber_k=HUBER_K,
            use_intrinsic_depth=DEPTH_MODE == "intrinsic",
            method="lm",
            stop_mse_per_px=STOP_MSE_PER_PX,
            stop_ssd=STOP_SSD,
            min_interaction_rank=MIN_INTERACTION_RANK,
            max_interaction_condition=MAX_INTERACTION_CONDITION,
        )
    else:
        raise ValueError(f"Unknown CONTROLLER={CONTROLLER!r}")
    controller_name = controller.__class__.__name__

    trial_dir, output_dir = make_run_dir(
        controller,
        SCENE_DIR.name,
        RENDERER,
        start_index,
        target_index,
        DEPTH_MODE,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    logs_dir = output_dir / "logs"
    visualizations_dir = output_dir / "visualizations"
    logs_dir.mkdir(parents=True, exist_ok=True)
    visualizations_dir.mkdir(parents=True, exist_ok=True)

    target_image = load_rgb(target["rgb_path"], start_camera.W, start_camera.H)
    initial_render = scene.render(start_camera)
    save_rgb(output_dir / "target.png", target_image)
    save_rgb(output_dir / "initial_render.png", initial_render)

    initial_translation_gap = translation_gap(start_camera, target_camera)
    initial_rotation_error = rotation_error_deg(start_camera, target_camera)
    target_T_world_cam = target_camera.T_world_cam.copy()

    def record_iteration_metrics(item):
        translation_gap = translation_error_from_pose(
            item["T_world_cam"],
            target_T_world_cam,
        )
        rotation_error = rotation_error_from_pose(
            item["T_world_cam"],
            target_T_world_cam,
        )
        item["translation_gap"] = translation_gap
        item["rotation_error_deg"] = rotation_error
        closed_pct = gap_closed_percent(initial_translation_gap, translation_gap)
        item["gap_closed_pct"] = closed_pct

        info = item.get("controller_info", {})
        err_label, err_value = controller_error_display(info)
        inliers = info.get("num_inlier_matches", 0)
        print(
            f"it={item['iteration']:04d} "
            f"ok={int(bool(item.get('step_accepted', False)))} "
            f"mode={info.get('feature_mode', '-')} "
            f"{err_label}={err_value} "
            f"t_gap={format_stat(translation_gap, 6)} "
            f"rot_gap={format_stat(rotation_error, 3)}deg "
            f"closed={format_percent(closed_pct)} "
            f"inliers={inliers} "
            f"stop={short_stop_reason(item.get('stop_reason'))}"
        )

    result = run_servo_loop(
        scene,
        start_camera,
        target_image,
        controller,
        iterations=ITERATIONS,
        dt=DT,
        visualization_dir=visualizations_dir / "matches",
        matcher=matcher,
        feature_method=FEATURE_METHOD,
        iteration_callback=record_iteration_metrics,
        viz_iter=VIZ_ITER,
    )

    final_camera = result["camera"]
    final_render = result["rendered"]
    save_rgb(output_dir / "final_render.png", final_render)
    final_photometric_viz = {}
    if CONTROLLER in ("photometric", "photometric_torch"):
        final_photometric_viz = save_current_desired_error_visualization(
            final_render,
            target_image,
            visualizations_dir / "final_desired_error.png",
            current_label="final",
            desired_label="desired",
        )
    write_history_csv(output_dir / "history.csv", result["history"])
    write_history_csv(logs_dir / "history.csv", result["history"])
    save_error_evolution(
        result["history"],
        visualizations_dir / "error_evolution.png",
    )
    save_error_evolution(
        result["history"],
        logs_dir / "error_evolution.png",
    )

    final_translation_gap = translation_gap(final_camera, target_camera)
    final_rotation_error = rotation_error_deg(final_camera, target_camera)
    summary = {
        "config": experiment_config(
            controller_name,
            trial_dir,
            start_index,
            target_index,
            start_frame,
            target_frame,
            output_dir,
        ),
        "controller": controller_name,
        "renderer": RENDERER,
        "trial_dir": str(trial_dir),
        "scene_dir": str(SCENE_DIR),
        "frame_selection": "logical_index",
        "frame_index_base": 1,
        "start_index": int(start_index),
        "target_index": int(target_index),
        "index_away": int(INDEX_AWAY),
        "start_frame": start_frame,
        "target_frame": target_frame,
        "start_rgb": str(start["rgb_path"]),
        "target_rgb": str(target["rgb_path"]),
        "depth": DEPTH_MODE,
        "feature_method": FEATURE_METHOD,
        "viz_iter": int(VIZ_ITER),
        "logs_dir": str(logs_dir),
        "visualizations_dir": str(visualizations_dir),
        "error_evolution_plot": str(logs_dir / "error_evolution.png"),
        "iterations": ITERATIONS,
        "dt": DT,
        "camera": camera_metadata(start_camera),
        "start_T_world_cam": start_camera.T_world_cam.tolist(),
        "target_T_world_cam": target_camera.T_world_cam.tolist(),
        "final_T_world_cam": final_camera.T_world_cam.tolist(),
        "final_desired_error_viz": final_photometric_viz.get(
            "visualization_path",
        ),
        "depth_preflight": depth_preflight_info,
        "initial_translation_gap": initial_translation_gap,
        "final_translation_gap": final_translation_gap,
        "translation_gap_closed_pct": gap_closed_percent(
            initial_translation_gap,
            final_translation_gap,
        ),
        "initial_rotation_error_deg": initial_rotation_error,
        "final_rotation_error_deg": final_rotation_error,
        "iterations_run": len(result["history"]),
        "stop_reason": result["stop_reason"],
        "stop_iteration": result["stop_iteration"],
        "timing": result.get("timing", {}),
        "history": history_for_json(result["history"]),
    }

    write_run_json(output_dir / "summary.json", summary)
    write_run_json(output_dir / "config.resolved.json", summary["config"])
    write_run_json(output_dir / "config.json", summary["config"])
    write_command(output_dir / "command.txt")
    write_servo_summary_markdown(output_dir / "summary.md", summary)
    write_run_readme(
        output_dir / "README.md",
        "SERVIS Servo Frame Run",
        fields=[
            ("scene", SCENE_DIR.name),
            ("renderer", RENDERER),
            ("controller", controller_name),
            ("frames", f"{start_frame} -> {target_frame}"),
            ("depth", DEPTH_MODE),
            ("feature", FEATURE_METHOD),
        ],
        artifacts=[
            ("summary table", "summary.md"),
            ("machine summary", "summary.json"),
            ("resolved config", "config.resolved.json"),
            ("history", "history.csv"),
            ("visualizations", "visualizations/"),
        ],
    )
    write_run_json(logs_dir / "summary.json", summary)
    write_run_json(logs_dir / "config.json", summary["config"])

    append_trial_index(trial_dir, {
        "run_dir": output_dir.name,
        "timestamp": output_dir.name.split("_", 1)[0],
        "controller": controller_name,
        "renderer": RENDERER,
        "scene": SCENE_DIR.name,
        "depth_mode": DEPTH_MODE,
        "feature_method": FEATURE_METHOD,
        "gain_ibvs": float(GAIN_IBVS),
        "gain_photo": float(GAIN_PHOTO),
        "ratio": int(RATIO),
        "start_index": int(start_index),
        "target_index": int(target_index),
        "iterations_run": len(result["history"]),
        "stop_reason": result["stop_reason"],
        "fps": float(result.get("timing", {}).get("fps", 0.0)),
        "render_fps": float(result.get("timing", {}).get("render_fps", 0.0)),
        "iter_ms_mean": float(result.get("timing", {}).get("iter_ms_mean", 0.0)),
        "render_ms_mean": float(result.get("timing", {}).get("render_ms_mean", 0.0)),
        "initial_translation_gap": initial_translation_gap,
        "final_translation_gap": final_translation_gap,
        "translation_gap_closed_pct": gap_closed_percent(
            initial_translation_gap,
            final_translation_gap,
        ),
        "initial_rotation_error_deg": initial_rotation_error,
        "final_rotation_error_deg": final_rotation_error,
    })

    last = result["history"][-1] if result["history"] else {}
    info = last.get("controller_info", {})
    timing = result.get("timing", {})
    print(
        f"Servo {controller_name} {RENDERER}: index {start_index} -> {target_index} "
        f"({start_frame} -> {target_frame}), "
        f"depth={DEPTH_MODE}, "
        f"iterations={len(result['history'])}/{ITERATIONS}, "
        f"stop={result['stop_reason']}, "
        f"fps={timing.get('fps', 0.0):.1f} "
        f"(iter_ms={timing.get('iter_ms_mean', 0.0):.2f}, "
        f"render_ms={timing.get('render_ms_mean', 0.0):.2f}, "
        f"ctrl_ms={timing.get('controller_ms_mean', 0.0):.2f})"
    )
    print(
        f"Translation gap: {format_stat(initial_translation_gap, 6)} -> "
        f"{format_stat(final_translation_gap, 6)}"
    )
    print(
        f"Rotation gap: {format_stat(initial_rotation_error, 4)}deg -> "
        f"{format_stat(final_rotation_error, 4)}deg"
    )
    if info:
        print(
            f"Last iteration: {info['num_inlier_matches']} controller inliers, "
            f"residual={info['residual_norm']:.6f}, "
            f"|v|={info['velocity_norm']:.6f}"
        )
    print(f"Wrote {output_dir}")
