"""Single frame-to-frame servo runner.

Use via the CLI:
    python cli.py servo-frames --config <path> [--set k=v ...]

Heavy imports (controllers, scenes, photometric) are deferred to `run()` so
this module can be imported by `cli.py` without pulling them in.
"""

import csv
import json
import re
from datetime import datetime
from pathlib import Path


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
STOP_MSE_PER_PX = 2.0e-6    # photometric: mean(e^2) per pixel on [0, 1] floats

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


def translation_error_m(camera, target_camera):
    return translation_error_from_pose(camera.T_world_cam, target_camera.T_world_cam)


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
        "translation_error_m",
        "rotation_error_deg",
        "cached_features",
        "dropped_features",
        "mean_depth_m",
        "min_depth_m",
        "max_depth_m",
        "velocity_norm",
        "vx",
        "vy",
        "vz",
        "wx",
        "wy",
        "wz",
        "render_ms",
        "controller_ms",
        "iter_ms",
        "stop_reason",
    ]

    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for item in history:
            info = item.get("controller_info", {})
            velocity = np.asarray(item["velocity"], dtype=np.float32)
            writer.writerow({
                "iteration": item["iteration"],
                "num_matches": item.get("num_matches", ""),
                "num_inliers": item.get("num_inliers", ""),
                "feature_mode": item.get("feature_mode", ""),
                "controller_inliers": info.get("num_inlier_matches", ""),
                "residual_norm": info.get("residual_norm", ""),
                "translation_error_m": item.get("translation_error_m", ""),
                "rotation_error_deg": item.get("rotation_error_deg", ""),
                "cached_features": info.get("num_cached_features", ""),
                "dropped_features": info.get("num_dropped_features", ""),
                "mean_depth_m": info.get("mean_depth_m", ""),
                "min_depth_m": info.get("min_depth_m", ""),
                "max_depth_m": info.get("max_depth_m", ""),
                "velocity_norm": info.get("velocity_norm", ""),
                "vx": float(velocity[0]),
                "vy": float(velocity[1]),
                "vz": float(velocity[2]),
                "wx": float(velocity[3]),
                "wy": float(velocity[4]),
                "wz": float(velocity[5]),
                "render_ms": item.get("render_ms", ""),
                "controller_ms": item.get("controller_ms", ""),
                "iter_ms": item.get("iter_ms", ""),
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
            "velocity": np.asarray(item["velocity"], dtype=np.float32).tolist(),
            "num_matches": item.get("num_matches"),
            "num_inliers": item.get("num_inliers"),
            "feature_mode": item.get("feature_mode"),
            "translation_error_m": item.get("translation_error_m"),
            "rotation_error_deg": item.get("rotation_error_deg"),
            "visualization_path": item.get("visualization_path"),
            "stop_reason": item.get("stop_reason"),
            "controller_info": item.get("controller_info", {}),
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
    trial_dir = (
        RUNS_ROOT
        / controller_short_name(controller)
        / renderer
        / scene_name
        / make_trial_name(start_index, target_index, depth_mode)
    )
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    active_gain = GAIN_PHOTO if CONTROLLER != "ibvs" else GAIN_IBVS
    tag = make_run_tag(active_gain, FEATURE_METHOD, RATIO)
    base = f"{timestamp}_{tag}"
    candidate = trial_dir / base
    suffix = 2
    while candidate.exists():
        candidate = trial_dir / f"{base}_{suffix:02d}"
        suffix += 1
    return trial_dir, candidate


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
    "initial_translation_error_m",
    "final_translation_error_m",
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
        "gain_ibvs": float(GAIN_IBVS),
        "gain_photo": float(GAIN_PHOTO),
        "min_features": int(MIN_FEATURES),
        "ratio": int(RATIO),
        "run_name": RUN_NAME,
        "stop_residual_px": float(STOP_RESIDUAL_PX),
        "stop_mse_per_px": float(STOP_MSE_PER_PX),
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
    from viz import save_error_evolution

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

    initial_translation_error = translation_error_m(start_camera, target_camera)
    initial_rotation_error = rotation_error_deg(start_camera, target_camera)
    target_T_world_cam = target_camera.T_world_cam.copy()

    def record_iteration_metrics(item):
        translation_error = translation_error_from_pose(
            item["T_world_cam"],
            target_T_world_cam,
        )
        rotation_error = rotation_error_from_pose(
            item["T_world_cam"],
            target_T_world_cam,
        )
        item["translation_error_m"] = translation_error
        item["rotation_error_deg"] = rotation_error

        info = item.get("controller_info", {})
        residual = info.get("residual_norm", float("nan"))
        inliers = info.get("num_inlier_matches", 0)
        print(
            f"iter {item['iteration']:04d}: "
            f"mode={info.get('feature_mode')} "
            f"error_norm={residual:.6f} "
            f"pose_distance={translation_error * 1000.0:.4f}mm "
            f"rotation_distance={rotation_error:.4f}deg "
            f"cached={info.get('num_cached_features', 0)} "
            f"dropped={info.get('num_dropped_features', 0)} "
            f"inliers={inliers}"
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

    final_translation_error = translation_error_m(final_camera, target_camera)
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
        "initial_translation_error_m": initial_translation_error,
        "final_translation_error_m": final_translation_error,
        "initial_rotation_error_deg": initial_rotation_error,
        "final_rotation_error_deg": final_rotation_error,
        "iterations_run": len(result["history"]),
        "stop_reason": result["stop_reason"],
        "stop_iteration": result["stop_iteration"],
        "timing": result.get("timing", {}),
        "history": history_for_json(result["history"]),
    }

    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    with open(output_dir / "config.json", "w") as f:
        json.dump(summary["config"], f, indent=2)
    with open(logs_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    with open(logs_dir / "config.json", "w") as f:
        json.dump(summary["config"], f, indent=2)

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
        "initial_translation_error_m": initial_translation_error,
        "final_translation_error_m": final_translation_error,
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
        f"Translation error: {initial_translation_error * 1000.0:.4f}mm -> "
        f"{final_translation_error * 1000.0:.4f}mm"
    )
    print(
        f"Rotation error: {initial_rotation_error:.4f}deg -> "
        f"{final_rotation_error:.4f}deg"
    )
    if info:
        print(
            f"Last iteration: {info['num_inlier_matches']} controller inliers, "
            f"residual={info['residual_norm']:.6f}, "
            f"|v|={info['velocity_norm']:.6f}"
        )
    print(f"Wrote {output_dir}")
