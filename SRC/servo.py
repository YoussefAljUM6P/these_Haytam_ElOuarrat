from __future__ import annotations
from pathlib import Path
from time import perf_counter
from typing import Optional, Union, Tuple, List, Dict, Callable, Any

import numpy as np
from numpy.typing import NDArray
import scipy.linalg

from camera import Camera
from features import FeatureMatcher, filter_matches
from viz import save_current_desired_error_visualization, save_match_visualization


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
    """Save current/desired frames with the photometric error heatmap adjacent."""
    viz_info = save_current_desired_error_visualization(
        rendered,
        target,
        output_path,
        current_label="current",
        desired_label="desired",
    )
    info = visualization or {}
    return {
        "num_matches": int(info.get("num_pixels_used", 0)),
        "num_inliers": int(info.get("num_pixels_used", 0)),
        "feature_mode": info.get("feature_mode", "photometric"),
        **viz_info,
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
    max_translation_step: Optional[float] = 0.5,
    max_rotation_step_deg: Optional[float] = 30.0,
    hard_translation_step: Optional[float] = 5.0,
    hard_rotation_step_deg: Optional[float] = 180.0,
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

    def _positive_limit(name: str, value: Optional[float]) -> Optional[float]:
        if value is None:
            return None
        value = float(value)
        if not np.isfinite(value) or value <= 0.0:
            raise ValueError(f"{name} must be positive or None")
        return value

    max_translation_step = _positive_limit(
        "max_translation_step", max_translation_step,
    )
    max_rotation_step_deg = _positive_limit(
        "max_rotation_step_deg", max_rotation_step_deg,
    )
    hard_translation_step = _positive_limit(
        "hard_translation_step", hard_translation_step,
    )
    hard_rotation_step_deg = _positive_limit(
        "hard_rotation_step_deg", hard_rotation_step_deg,
    )
    if (
        max_translation_step is not None
        and hard_translation_step is not None
        and hard_translation_step < max_translation_step
    ):
        raise ValueError("hard_translation_step must be >= max_translation_step")
    if (
        max_rotation_step_deg is not None
        and hard_rotation_step_deg is not None
        and hard_rotation_step_deg < max_rotation_step_deg
    ):
        raise ValueError("hard_rotation_step_deg must be >= max_rotation_step_deg")

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

    def _controller_stop_reason() -> Optional[str]:
        controller_should_stop = getattr(controller, "should_stop", None)
        if not callable(controller_should_stop):
            return None
        reason = controller_should_stop()
        return None if reason is None else str(reason)

    def _history_velocity(value: NDArray[np.float32]) -> NDArray[np.float32]:
        if value.shape == (6,):
            return value.copy()
        out = np.full(6, np.nan, dtype=np.float32)
        flat = value.reshape(-1)
        n = min(int(flat.size), 6)
        if n:
            out[:n] = flat[:n]
        return out

    def _step_metrics(value: NDArray[np.float32]) -> Tuple[float, float]:
        if value.shape != (6,) or not np.isfinite(value).all():
            return float("nan"), float("nan")
        step = value.astype(np.float64) * float(dt)
        translation_step = float(np.linalg.norm(step[:3]))
        rotation_step_deg = float(np.degrees(np.linalg.norm(step[3:])))
        return translation_step, rotation_step_deg

    def _motion_info(
        raw_velocity: NDArray[np.float32],
        applied_velocity: NDArray[np.float32],
        *,
        velocity_limited: bool = False,
        velocity_scale: float = 1.0,
    ) -> Dict[str, Any]:
        raw_t, raw_r = _step_metrics(raw_velocity)
        applied_t, applied_r = _step_metrics(applied_velocity)
        return {
            "velocity_limited": bool(velocity_limited),
            "velocity_scale": float(velocity_scale),
            "raw_translation_step": raw_t,
            "raw_rotation_step_deg": raw_r,
            "translation_step": applied_t,
            "rotation_step_deg": applied_r,
            "max_translation_step": max_translation_step,
            "max_rotation_step_deg": max_rotation_step_deg,
            "hard_translation_step": hard_translation_step,
            "hard_rotation_step_deg": hard_rotation_step_deg,
        }

    def _limit_velocity(
        raw_velocity: NDArray[np.float32],
    ) -> Tuple[NDArray[np.float32], Dict[str, Any], Optional[str]]:
        applied_velocity = raw_velocity.copy()
        raw_translation_step, raw_rotation_step_deg = _step_metrics(raw_velocity)

        exceeds_hard_translation = (
            hard_translation_step is not None
            and raw_translation_step > hard_translation_step
        )
        exceeds_hard_rotation = (
            hard_rotation_step_deg is not None
            and raw_rotation_step_deg > hard_rotation_step_deg
        )
        if exceeds_hard_translation or exceeds_hard_rotation:
            zero_velocity = np.zeros(6, dtype=np.float32)
            return (
                zero_velocity,
                _motion_info(raw_velocity, zero_velocity),
                "velocity_invalid_exceeds_hard_limit",
            )

        scales = []
        if (
            max_translation_step is not None
            and raw_translation_step > max_translation_step
        ):
            scales.append(max_translation_step / raw_translation_step)
        if (
            max_rotation_step_deg is not None
            and raw_rotation_step_deg > max_rotation_step_deg
        ):
            scales.append(max_rotation_step_deg / raw_rotation_step_deg)

        velocity_scale = min(scales) if scales else 1.0
        velocity_limited = velocity_scale < 1.0
        if velocity_limited:
            applied_velocity = (applied_velocity * velocity_scale).astype(np.float32)

        return (
            applied_velocity,
            _motion_info(
                raw_velocity,
                applied_velocity,
                velocity_limited=velocity_limited,
                velocity_scale=velocity_scale,
            ),
            None,
        )

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
        controller_stop_reason = _controller_stop_reason()
        stop_reason_for_iteration = controller_stop_reason
        step_accepted = False
        raw_velocity = velocity
        applied_velocity = np.zeros(6, dtype=np.float32)
        motion_info = _motion_info(raw_velocity, applied_velocity)
        next_camera = copy_camera_with_pose(camera, camera.T_world_cam.copy())

        if stop_reason_for_iteration is None:
            velocity_fault_reason = None
            if velocity.shape != (6,):
                velocity_fault_reason = "velocity_invalid_shape"
            elif not np.isfinite(velocity).all():
                velocity_fault_reason = "velocity_invalid_nonfinite"

            if velocity_fault_reason is not None:
                stop_reason_for_iteration = velocity_fault_reason
                controller_info = dict(controller_info)
                controller_info["fault_reason"] = velocity_fault_reason
                controller_info["velocity_shape"] = tuple(int(x) for x in velocity.shape)
            else:
                applied_velocity, motion_info, limit_fault_reason = _limit_velocity(
                    raw_velocity,
                )
                if limit_fault_reason is not None:
                    stop_reason_for_iteration = limit_fault_reason
                    controller_info = dict(controller_info)
                    controller_info["fault_reason"] = stop_reason_for_iteration
                else:
                    try:
                        next_camera = apply_camera_velocity(
                            camera,
                            applied_velocity,
                            dt=dt,
                        )
                        step_accepted = True
                    except Exception as exc:
                        stop_reason_for_iteration = "velocity_invalid_pose_update"
                        applied_velocity = np.zeros(6, dtype=np.float32)
                        motion_info = _motion_info(raw_velocity, applied_velocity)
                        controller_info = dict(controller_info)
                        controller_info["fault_reason"] = stop_reason_for_iteration
                        controller_info["fault_detail"] = f"{type(exc).__name__}: {exc}"
                        next_camera = copy_camera_with_pose(
                            camera,
                            camera.T_world_cam.copy(),
                        )

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
            "raw_velocity": _history_velocity(raw_velocity),
            "velocity": applied_velocity.copy(),
            "next_T_world_cam": next_camera.T_world_cam.copy(),
            "step_accepted": bool(step_accepted),
            "controller_info": dict(controller_info),
            "render_ms": render_dt * 1000.0,
            "controller_ms": controller_dt * 1000.0,
            "viz_ms": viz_dt * 1000.0,
            "iter_ms": iter_dt * 1000.0,
            **motion_info,
            **match_info,
            "stop_reason": stop_reason_for_iteration,
        }
        callback_stop = False
        if iteration_callback is not None:
            callback_stop = bool(iteration_callback(history_item))

        if stop_reason_for_iteration:
            stop_reason = str(stop_reason_for_iteration)
            stop_iteration = int(iteration)
            history_item["stop_reason"] = stop_reason
        elif callback_stop:
            stop_reason = "callback"
            stop_iteration = int(iteration)
            history_item["stop_reason"] = stop_reason
            history_item["velocity"] = np.zeros(6, dtype=np.float32)
            history_item["next_T_world_cam"] = camera.T_world_cam.copy()
            history_item["translation_step"] = 0.0
            history_item["rotation_step_deg"] = 0.0
            history_item["step_accepted"] = False

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
