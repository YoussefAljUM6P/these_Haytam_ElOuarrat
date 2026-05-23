from __future__ import annotations
from pathlib import Path
from time import perf_counter
from typing import Optional, Union, Tuple, List, Dict, Callable, Any

import cv2
import numpy as np
from numpy.typing import NDArray
import scipy.linalg

from camera import Camera
from features import FeatureMatcher, filter_matches
from viz import save_match_visualization


def se3_exp(twist: NDArray[np.float64]) -> NDArray[np.float32]:
    twist = np.asarray(twist, dtype=np.float64)
    if twist.shape != (6,):
        raise ValueError(f"twist must have shape (6,), got {twist.shape}")
    v = twist[:3]
    w = twist[3:]
    xi = np.zeros((4, 4), dtype=np.float64)
    xi[0, 1] = -w[2]; xi[0, 2] =  w[1]
    xi[1, 0] =  w[2]; xi[1, 2] = -w[0]
    xi[2, 0] = -w[1]; xi[2, 1] =  w[0]
    xi[:3, 3] = v
    return scipy.linalg.expm(xi).astype(np.float32)


def copy_camera_with_pose(camera: Camera, T_world_cam: NDArray[np.float32]) -> Camera:
    return Camera(
        T_world_cam,
        camera.fx,
        camera.fy,
        camera.cx,
        camera.cy,
        camera.H,
        camera.W,
    )


def apply_camera_velocity(camera: Camera, velocity: NDArray[np.float32], dt: float = 1.0) -> Camera:
    velocity = np.asarray(velocity, dtype=np.float32)
    if velocity.shape != (6,):
        raise ValueError(f"velocity must have shape (6,), got {velocity.shape}")

    # Body/camera-frame velocity: update T_world_cam by right-multiplying.
    delta_T = se3_exp(velocity * float(dt))
    T_world_cam = camera.T_world_cam @ delta_T
    return copy_camera_with_pose(camera, T_world_cam.astype(np.float32))


class FixedVelocityController:
    def __init__(self, velocity: NDArray[np.float32]) -> None:
        self.velocity = np.asarray(velocity, dtype=np.float32)
        if self.velocity.shape != (6,):
            raise ValueError(f"velocity must have shape (6,), got {self.velocity.shape}")

    def __call__(self, rendered: NDArray[np.float32], target: NDArray[np.float32], camera: Camera, iteration: int) -> NDArray[np.float32]:
        return self.velocity.copy()


def save_iteration_matches(rendered: NDArray[np.float32], target: NDArray[np.float32], camera: Camera, matcher: FeatureMatcher, output_path: Union[str, Path]) -> Dict[str, Any]:
    kpts1, kpts2 = matcher.match(rendered, target)
    kpts1_kept, kpts2_kept, _, _, _, _ = filter_matches(kpts1, kpts2, camera)
    matches_kept = [(i, i) for i in range(len(kpts1_kept))]
    empty = np.zeros((0, 2), dtype=np.float32)

    save_match_visualization(
        rendered,
        target,
        empty,
        empty.copy(),
        [],
        kpts1_kept,
        kpts2_kept,
        matches_kept,
        output_path,
        draw_removed=False,
    )

    return {
        "num_matches": int(len(kpts1)),
        "num_inliers": int(len(kpts1_kept)),
        "visualization_path": str(output_path),
    }


def save_photometric_visualization(rendered: NDArray[np.float32], target: NDArray[np.float32], visualization: Optional[Dict[str, Any]], output_path: Union[str, Path]) -> Dict[str, Any]:
    """Three-panel viz for photometric servoing: rendered | target | |diff| heatmap.

    Diff is computed on grayscale intensities in [0, 1]; mapped to JET colormap
    over [0, max(|diff|)] of the current frame, and a fixed-range overlay over
    [0, 0.5] is also drawn alongside so the absolute scale is visible.
    """
    def to_uint8_rgb(image):
        arr = np.asarray(image, dtype=np.float32)
        if arr.ndim == 2:
            arr = np.stack([arr] * 3, axis=-1)
        arr = np.clip(arr, 0.0, 1.0)
        return (arr * 255.0 + 0.5).astype(np.uint8)

    def to_gray01(image):
        arr = np.asarray(image, dtype=np.float32)
        if arr.ndim == 2:
            return np.clip(arr, 0.0, 1.0)
        return cv2.cvtColor(np.clip(arr, 0.0, 1.0), cv2.COLOR_RGB2GRAY)

    gray_cur = to_gray01(rendered)
    gray_tgt = to_gray01(target)
    diff = np.abs(gray_cur - gray_tgt)
    diff_max = float(diff.max()) if diff.size else 0.0

    diff_norm = (diff / max(diff_max, 1e-6) * 255.0).astype(np.uint8)
    heat_auto = cv2.applyColorMap(diff_norm, cv2.COLORMAP_JET)
    heat_auto = cv2.cvtColor(heat_auto, cv2.COLOR_BGR2RGB)

    rendered_rgb = to_uint8_rgb(rendered)
    target_rgb = to_uint8_rgb(target)

    H = rendered_rgb.shape[0]
    label_strip = 22
    panels = []
    for img, label in (
        (rendered_rgb, "rendered"),
        (target_rgb, "target"),
        (heat_auto, f"|I-I*|  max={diff_max:.3f}"),
    ):
        panel = np.zeros((H + label_strip, img.shape[1], 3), dtype=np.uint8)
        cv2.putText(
            panel, label, (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
            (255, 255, 255), 1, cv2.LINE_AA,
        )
        panel[label_strip:, :, :] = img
        panels.append(panel)

    out = np.concatenate(panels, axis=1)
    out_bgr = cv2.cvtColor(out, cv2.COLOR_RGB2BGR)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), out_bgr)

    info = visualization or {}
    return {
        "num_matches": int(info.get("num_pixels_used", 0)),
        "num_inliers": int(info.get("num_pixels_used", 0)),
        "feature_mode": info.get("feature_mode", "photometric"),
        "visualization_path": str(output_path),
        "diff_max": diff_max,
        "diff_mean": float(diff.mean()) if diff.size else 0.0,
    }


def save_controller_matches(rendered: NDArray[np.float32], target: NDArray[np.float32], visualization: Dict[str, Any], output_path: Union[str, Path]) -> Dict[str, Any]:
    kpts_current = np.asarray(
        visualization.get("kpts_current", np.zeros((0, 2), dtype=np.float32)),
        dtype=np.float32,
    ).reshape(-1, 2)
    kpts_target = np.asarray(
        visualization.get("kpts_target", np.zeros((0, 2), dtype=np.float32)),
        dtype=np.float32,
    ).reshape(-1, 2)
    if kpts_current.shape != kpts_target.shape:
        raise RuntimeError(
            f"Controller visualization keypoints are not paired: "
            f"{kpts_current.shape} vs {kpts_target.shape}"
        )

    matches_kept = [(i, i) for i in range(len(kpts_current))]
    empty = np.zeros((0, 2), dtype=np.float32)
    save_match_visualization(
        rendered,
        target,
        empty,
        empty.copy(),
        [],
        kpts_current,
        kpts_target,
        matches_kept,
        output_path,
        draw_removed=False,
    )

    return {
        "num_matches": int(visualization.get("num_raw_matches", len(kpts_current))),
        "num_inliers": int(len(kpts_current)),
        "feature_mode": visualization.get("feature_mode"),
        "visualization_path": str(output_path),
    }


def run_servo_loop(
    scene: Any,
    initial_camera: Camera,
    target_image: NDArray[np.float32],
    controller: Callable[..., NDArray[np.float32]],
    iterations: int,
    dt: float = 1.0,
    visualization_dir: Optional[Union[str, Path]] = None,
    matcher: Optional[FeatureMatcher] = None,
    feature_method: str = "xfeat",
    iteration_callback: Optional[Callable[[Dict[str, Any]], bool]] = None,
    viz_iter: Optional[int] = 1,
) -> Dict[str, Any]:
    if iterations < 0:
        raise ValueError("iterations must be non-negative")
    if viz_iter is not None and int(viz_iter) < 0:
        raise ValueError("viz_iter must be >= 0 or None")

    if visualization_dir is not None:
        visualization_dir = Path(visualization_dir)
        visualization_dir.mkdir(parents=True, exist_ok=True)
        if matcher is None:
            matcher = FeatureMatcher(method=feature_method)

    camera = copy_camera_with_pose(initial_camera, initial_camera.T_world_cam.copy())
    history = []
    stop_reason = None
    stop_iteration = None

    loop_t0 = perf_counter()
    render_total = 0.0
    controller_total = 0.0
    viz_total = 0.0

    for iteration in range(iterations):
        iter_t0 = perf_counter()

        t0 = perf_counter()
        rendered = scene.render(camera)
        render_dt = perf_counter() - t0
        render_total += render_dt

        t0 = perf_counter()
        velocity = np.asarray(
            controller(rendered, target_image, camera, iteration),
            dtype=np.float32,
        )
        controller_dt = perf_counter() - t0
        controller_total += controller_dt

        controller_info = getattr(controller, "last_info", {})
        next_camera = apply_camera_velocity(camera, velocity, dt=dt)

        match_info = {}
        should_save_viz = (
            visualization_dir is not None
            and viz_iter is not None
            and int(viz_iter) > 0
            and iteration % int(viz_iter) == 0
        )
        viz_t0 = perf_counter()
        if should_save_viz:
            controller_visualization = getattr(controller, "last_visualization", None)
            is_photometric = (
                controller_visualization is not None
                and controller_visualization.get("iteration") == iteration
                and controller_visualization.get("feature_mode") == "photometric"
            )
            if is_photometric:
                output_path = visualization_dir / f"iter_{iteration:04d}_photometric.png"
                match_info = save_photometric_visualization(
                    rendered,
                    target_image,
                    controller_visualization,
                    output_path,
                )
            else:
                output_path = visualization_dir / f"iter_{iteration:04d}_matches.png"
                if (
                    controller_visualization is not None
                    and controller_visualization.get("iteration") == iteration
                ):
                    match_info = save_controller_matches(
                        rendered,
                        target_image,
                        controller_visualization,
                        output_path,
                    )
                else:
                    match_info = save_iteration_matches(
                        rendered,
                        target_image,
                        camera,
                        matcher,
                        output_path,
                    )
        viz_dt = perf_counter() - viz_t0
        viz_total += viz_dt

        iter_dt = perf_counter() - iter_t0
        history_item = {
            "iteration": iteration,
            "T_world_cam": camera.T_world_cam.copy(),
            "velocity": velocity.copy(),
            "next_T_world_cam": next_camera.T_world_cam.copy(),
            "controller_info": dict(controller_info),
            "render_ms": render_dt * 1000.0,
            "controller_ms": controller_dt * 1000.0,
            "viz_ms": viz_dt * 1000.0,
            "iter_ms": iter_dt * 1000.0,
            **match_info,
        }
        callback_stop = False
        if iteration_callback is not None:
            callback_stop = bool(iteration_callback(history_item))

        # Each controller may define its own should_stop() returning a reason
        # string or None. This is the only termination path besides
        # iteration_callback and the iteration cap.
        controller_stop_reason = None
        controller_should_stop = getattr(controller, "should_stop", None)
        if callable(controller_should_stop):
            controller_stop_reason = controller_should_stop()

        if controller_stop_reason:
            stop_reason = str(controller_stop_reason)
            stop_iteration = int(iteration)
            history_item["stop_reason"] = stop_reason
        elif callback_stop:
            stop_reason = "callback"
            stop_iteration = int(iteration)
            history_item["stop_reason"] = stop_reason

        history.append(history_item)

        if stop_reason is not None:
            break

        camera = next_camera

    if stop_reason is None:
        stop_reason = "max_iterations"

    loop_dt = perf_counter() - loop_t0
    n_iters = len(history)
    iter_times = [float(h.get("iter_ms", 0.0)) for h in history]
    render_times = [float(h.get("render_ms", 0.0)) for h in history]
    controller_times = [float(h.get("controller_ms", 0.0)) for h in history]

    def _mean(xs):
        return float(sum(xs) / len(xs)) if xs else 0.0

    timing = {
        "n_iters": int(n_iters),
        "loop_s": float(loop_dt),
        "iter_ms_mean": _mean(iter_times),
        "iter_ms_max": (max(iter_times) if iter_times else 0.0),
        "render_ms_mean": _mean(render_times),
        "render_ms_total": float(render_total * 1000.0),
        "controller_ms_mean": _mean(controller_times),
        "controller_ms_total": float(controller_total * 1000.0),
        "viz_ms_total": float(viz_total * 1000.0),
        "fps": (float(n_iters) / loop_dt) if loop_dt > 0 else 0.0,
        "render_fps": (float(n_iters) / render_total) if render_total > 0 else 0.0,
    }

    return {
        "camera": camera,
        "rendered": scene.render(camera),
        "history": history,
        "stop_reason": stop_reason,
        "stop_iteration": stop_iteration,
        "timing": timing,
    }
