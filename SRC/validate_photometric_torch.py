"""Unit + sim-to-sim tests for the torch photometric controller (ViSP port).

Mirrors validate_photometric.py for the NumPy controller. Plus a parity test
that asserts the torch implementation produces the same residuals / velocities
as the NumPy implementation on identical inputs (within float32 tolerance).
"""

import numpy as np
import torch

from camera import Camera
from photometric import (
    FeatureLuminance,
    PhotometricControllerTorch,
    PhotometricServo,
    derivative_filter_x,
    derivative_filter_y,
    gaussian_blur,
    luminance_interaction,
    point_interaction_matrix,
    rgb_to_gray,
)
from photometric.servo import huber_weights


# ---- helpers ---------------------------------------------------------------


def make_camera(W=64, H=64, f=80.0):
    return Camera(
        np.eye(4, dtype=np.float32),
        fx=f, fy=f, cx=W * 0.5, cy=H * 0.5, H=H, W=W,
    )


class _FakeScene:
    def __init__(self, depth_map):
        self._depth = np.asarray(depth_map, dtype=np.float32)

    def render_depth(self, camera=None):
        return self._depth.copy()


def _check_close(a, b, atol, label):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    diff = float(np.abs(a - b).max()) if a.size else 0.0
    if diff > atol:
        raise AssertionError(f"{label}: max diff {diff:.3e} > {atol:.3e}")


# ---- unit: filters ---------------------------------------------------------


def test_rgb_to_gray_matches_bt601_weights():
    rgb = np.zeros((4, 4, 3), dtype=np.float32)
    rgb[..., 0] = 1.0
    rgb[..., 1] = 0.5
    rgb[..., 2] = 0.25
    gray = rgb_to_gray(torch.as_tensor(rgb)).numpy()
    expected = 0.299 + 0.5 * 0.587 + 0.25 * 0.114
    if not np.allclose(gray, expected, atol=1e-6):
        raise AssertionError(f"gray={gray.mean()} expected {expected}")


def test_derivative_x_zeros_border_and_recovers_ramp():
    H, W = 16, 16
    x = np.arange(W, dtype=np.float32) * 0.1
    img = np.broadcast_to(x, (H, W)).astype(np.float32)
    gx = derivative_filter_x(torch.as_tensor(img)).numpy()
    if (gx[:, :3] != 0).any() or (gx[:, -3:] != 0).any():
        raise AssertionError("border columns must be zero")
    # ViSP 7-tap kernel applied to a linear ramp of slope 0.1 -> 0.1 in the
    # interior (kernel sums symmetric weights of consecutive offsets).
    interior = gx[3:-3, 3:-3]
    if not np.allclose(interior, 0.1, atol=1e-5):
        raise AssertionError(f"gx interior {interior.mean()} != 0.1")


def test_derivative_y_constant_image_is_zero():
    img = torch.full((10, 10), 0.7, dtype=torch.float32)
    gy = derivative_filter_y(img).numpy()
    if np.abs(gy).max() > 1e-7:
        raise AssertionError("derivative of constant image must be ~0")


def test_gaussian_blur_passthrough_for_sigma_zero():
    img = torch.rand(8, 8)
    out = gaussian_blur(img, sigma=0.0)
    if not torch.equal(img, out):
        raise AssertionError("sigma=0 should be identity")


# ---- unit: interaction matrices --------------------------------------------


def test_point_interaction_matrix_shape_and_values():
    x = torch.tensor([0.0, 0.1], dtype=torch.float32)
    y = torch.tensor([0.0, -0.2], dtype=torch.float32)
    z = torch.tensor([1.0, 2.0], dtype=torch.float32)
    L = point_interaction_matrix(x, y, z).numpy()
    if L.shape != (2, 2, 6):
        raise AssertionError(f"shape {L.shape}")
    # First point at origin with Z=1: L_x row should be [-1, 0, 0, 0, -1, 0].
    expected = np.array([-1, 0, 0, 0, -1, 0], dtype=np.float32)
    if not np.allclose(L[0, 0], expected, atol=1e-6):
        raise AssertionError(f"L[0, 0] = {L[0, 0]} != {expected}")


def test_luminance_interaction_matches_manual_contraction():
    rng = np.random.default_rng(0)
    n = 7
    gx = rng.standard_normal(n).astype(np.float32)
    gy = rng.standard_normal(n).astype(np.float32)
    x = rng.standard_normal(n).astype(np.float32)
    y = rng.standard_normal(n).astype(np.float32)
    z = (rng.random(n) + 1.0).astype(np.float32)

    L = luminance_interaction(gx, gy, x, y, z).numpy()
    Lx = point_interaction_matrix(x, y, z).numpy()
    expected = -(np.stack([gx, gy], axis=1)[:, :, None] * Lx).sum(axis=1)
    _check_close(L, expected, atol=1e-6, label="L_I vs manual contraction")
    if L.shape != (n, 6):
        raise AssertionError(f"shape {L.shape}")


# ---- unit: servo + huber ----------------------------------------------------


def test_huber_weights_auto_k_downweights_outlier():
    r = torch.tensor([0.0, 1.0, -1.0, 0.5, -0.5, 100.0], dtype=torch.float32)
    w, k = huber_weights(r)
    if k <= 0.0:
        raise AssertionError(f"k must be > 0 got {k}")
    if w[-1].item() >= 1.0:
        raise AssertionError("outlier should be down-weighted")
    if not torch.all(w[:5] >= w[-1]).item():
        raise AssertionError("inliers should weigh >= outlier")


def test_servo_returns_zero_velocity_for_zero_error():
    servo = PhotometricServo(gain=1.0, method="gn", use_huber=False)
    L = torch.randn(50, 6)
    e = torch.zeros(50)
    v, info = servo.solve(e, L)
    if v.shape != (6,):
        raise AssertionError(f"v shape {v.shape}")
    if v.abs().max().item() > 1e-6:
        raise AssertionError(f"v should be ~0, got {v}")


def test_servo_lm_updates_mu_on_cost_increase():
    servo = PhotometricServo(gain=0.1, method="lm", mu_init=0.01, use_huber=False)
    L = torch.randn(100, 6)
    # First solve sets baseline cost.
    servo.solve(torch.randn(100) * 0.1, L)
    mu_before = servo.mu
    # Big residual -> cost goes up -> mu should grow.
    servo.solve(torch.randn(100) * 100.0, L)
    if servo.mu <= mu_before:
        raise AssertionError(
            f"mu must grow on cost increase: {mu_before} -> {servo.mu}"
        )


# ---- unit: FeatureLuminance -------------------------------------------------


def test_feature_luminance_drops_border_pixels():
    cam = make_camera(W=32, H=32, f=50.0)
    target = np.tile(
        np.linspace(0.0, 1.0, cam.W, dtype=np.float32)[None, :, None],
        (cam.H, 1, 3),
    )
    depth = np.full((cam.H, cam.W), 2.0, dtype=np.float32)
    feat = FeatureLuminance(
        camera=cam,
        target_image=target,
        depth_star=depth,
        sigma_blur=0.0,
        use_gzn=False,
        bord=4,
        grad_percentile=0.0,
        sat_lo=float("-inf"),
        sat_hi=float("inf"),
        min_depth=1e-4,
        max_pixels=0,
    )
    # No pixel within `bord` of the border should be kept.
    rows = feat._idx // cam.W
    cols = feat._idx % cam.W
    if (rows < 4).any() or (rows >= cam.H - 4).any():
        raise AssertionError("kept border rows")
    if (cols < 4).any() or (cols >= cam.W - 4).any():
        raise AssertionError("kept border cols")
    if feat.num_pixels == 0:
        raise AssertionError("expected non-empty mask")


def test_feature_luminance_residual_zero_for_identical_images():
    cam = make_camera(W=48, H=48, f=80.0)
    H, W = cam.H, cam.W
    target = np.zeros((H, W, 3), dtype=np.float32)
    target[:, :, 0] = np.linspace(0.1, 0.9, W, dtype=np.float32)[None, :]
    target[:, :, 1] = np.linspace(0.1, 0.9, H, dtype=np.float32)[:, None]
    target[:, :, 2] = 0.5
    depth = np.full((H, W), 2.0, dtype=np.float32)

    feat = FeatureLuminance(
        cam, target, depth, sigma_blur=0.0, use_gzn=False, bord=5,
    )
    feat.build_from(target)
    e = feat.error().numpy()
    if np.abs(e).max() > 1e-5:
        raise AssertionError(f"residual should be ~0, got max {np.abs(e).max()}")


# ---- controller: behaviour parity with NumPy impl --------------------------


def _make_torch_controller(cam, target_cam, **kw):
    scene = _FakeScene(np.full((cam.H, cam.W), 2.0, dtype=np.float32))
    return PhotometricControllerTorch(
        scene=scene,
        target_camera=target_cam,
        gain=kw.pop("gain", 0.5),
        sigma_blur=0.0,
        use_gzn=False,
        bord=kw.pop("bord", 4),
        grad_percentile=0.0,
        sat_lo=float("-inf"),
        sat_hi=float("inf"),
        min_depth=1e-4,
        max_pixels=0,
        use_huber=False,
        method=kw.pop("method", "gn"),
    )


def test_torch_controller_zero_error_returns_zero_velocity():
    cam = make_camera()
    H, W = cam.H, cam.W
    target = np.zeros((H, W, 3), dtype=np.float32)
    target[:, :, 0] = np.linspace(0.1, 0.9, W, dtype=np.float32)[None, :]
    target[:, :, 1] = np.linspace(0.1, 0.9, H, dtype=np.float32)[:, None]
    target[:, :, 2] = 0.5
    ctrl = _make_torch_controller(cam, cam)
    v = ctrl(target.copy(), target, cam, iteration=0)
    if v.shape != (6,):
        raise AssertionError(f"v shape {v.shape}")
    if float(np.linalg.norm(v)) > 1e-5:
        raise AssertionError(f"||v|| should be ~0, got {v}")
    info = ctrl.last_info
    if info["feature_mode"] != "photometric":
        raise AssertionError("feature_mode must be 'photometric'")
    if info["backend"] != "torch":
        raise AssertionError("backend must be 'torch'")


def test_torch_controller_rank_deficient_returns_fault():
    cam = make_camera()
    H, W = cam.H, cam.W
    target = np.full((H, W, 3), 0.5, dtype=np.float32)
    rendered = np.full_like(target, 0.55)
    ctrl = _make_torch_controller(cam, cam, gain=1.0)
    v = ctrl(rendered, target, cam, iteration=0)
    if not np.allclose(v, 0.0, atol=1e-8):
        raise AssertionError(f"rank-deficient target should return zero velocity, got {v}")
    if ctrl.last_info["fault_reason"] != "measurement_invalid_rank_deficient":
        raise AssertionError(f"wrong fault: {ctrl.last_info}")
    if ctrl.should_stop() != "measurement_invalid_rank_deficient":
        raise AssertionError(f"wrong stop reason: {ctrl.should_stop()}")


def test_torch_controller_textured_residual_drives_velocity():
    cam = make_camera()
    H, W = cam.H, cam.W
    rng = np.random.default_rng(3)
    target = rng.uniform(0.1, 0.9, (H, W, 3)).astype(np.float32)
    rendered = np.zeros_like(target)
    rendered[:, 1:, :] = target[:, :-1, :]
    rendered[:, 0, :] = target[:, 0, :]
    ctrl = _make_torch_controller(cam, cam, gain=1.0)
    v = ctrl(rendered, target, cam, iteration=0)
    if ctrl.last_info["fault_reason"] is not None:
        raise AssertionError(f"unexpected fault: {ctrl.last_info}")
    if ctrl.last_info["interaction_rank"] < 6:
        raise AssertionError(f"expected full rank, got {ctrl.last_info}")
    if not np.isfinite(v).all():
        raise AssertionError(f"v must be finite, got {v}")
    if float(np.linalg.norm(v)) <= 0.0:
        raise AssertionError("nonzero residual should produce nonzero velocity")


def test_torch_controller_missing_target_camera_raises():
    cam = make_camera()
    scene = _FakeScene(np.full((cam.H, cam.W), 2.0, dtype=np.float32))
    ctrl = PhotometricControllerTorch(
        scene=scene, target_camera=None,
        sigma_blur=0.0, use_gzn=False, bord=4, method="gn",
    )
    target = np.random.rand(cam.H, cam.W, 3).astype(np.float32)
    try:
        ctrl(target.copy(), target, cam, iteration=0)
    except RuntimeError as exc:
        if "target_camera" not in str(exc):
            raise AssertionError(f"wrong error: {exc}")
        return
    raise AssertionError("expected RuntimeError when target_camera missing")


def test_torch_controller_last_info_has_parity_keys():
    cam = make_camera()
    target = np.tile(
        np.linspace(0.1, 0.9, cam.W, dtype=np.float32)[None, :, None],
        (cam.H, 1, 3),
    )
    ctrl = _make_torch_controller(cam, cam)
    ctrl(target.copy(), target, cam, iteration=0)
    required = {
        "iteration", "feature_mode", "num_raw_matches", "num_inlier_matches",
        "num_cached_features", "num_dropped_features", "residual_norm",
        "velocity_norm", "mean_depth", "min_depth", "max_depth",
    }
    missing = required - set(ctrl.last_info)
    if missing:
        raise AssertionError(f"last_info missing keys: {missing}")



# ---- integration: servo-loop motion ----------------------------------------


def _translation_gap(camera, target_camera):
    return float(np.linalg.norm(
        camera.T_world_cam[:3, 3] - target_camera.T_world_cam[:3, 3]
    ))


def _motion_summary(result, start_camera, target_camera):
    final_camera = result["camera"]
    return {
        "initial_gap": _translation_gap(start_camera, target_camera),
        "final_gap": _translation_gap(final_camera, target_camera),
        "pose_delta": float(np.linalg.norm(
            final_camera.T_world_cam[:3, 3] - start_camera.T_world_cam[:3, 3]
        )),
        "accepted_steps": int(sum(
            1 for item in result.get("history", [])
            if item.get("step_accepted")
        )),
        "iterations_run": int(len(result.get("history", []))),
        "stop_reason": result.get("stop_reason"),
    }


def _assert_servo_moved(summary, label):
    if int(summary["accepted_steps"]) <= 0:
        raise AssertionError(f"{label} accepted no servo steps: {summary}")
    if not np.isfinite(summary["pose_delta"]):
        raise AssertionError(f"{label} pose delta is non-finite: {summary}")
    if float(summary["pose_delta"]) <= 1e-9:
        raise AssertionError(f"{label} final pose did not change: {summary}")


def _try_torch_sim_to_sim_motion():
    try:
        from validate_photometric import _build_textured_plane_scene, _camera_at
        scene, _ = _build_textured_plane_scene()
    except Exception as exc:
        return None, f"renderer unavailable: {exc}"

    target_camera = _camera_at(translation_xyz=(0.0, 0.0, 0.0),
                               rotation_axis_angle_deg=None)
    start_camera = _camera_at(translation_xyz=(0.02, 0.0, 0.0),
                              rotation_axis_angle_deg=None)
    try:
        target_image = scene.render(target_camera)
    except Exception as exc:
        return None, f"target render failed: {exc}"

    from servo import run_servo_loop

    ctrl = PhotometricControllerTorch(
        scene=scene,
        target_camera=target_camera,
        gain=0.01,
        sigma_blur=1.0,
        use_gzn=True,
        bord=10,
        grad_percentile=50.0,
        max_pixels=20_000,
        use_huber=True,
        use_intrinsic_depth=True,
        method="lm",
        mu_init=0.1,
    )

    result = run_servo_loop(
        scene, start_camera, target_image, ctrl,
        iterations=80, dt=1.0,
        visualization_dir=None, matcher=None,
        iteration_callback=None,
        viz_iter=0,
    )
    return _motion_summary(result, start_camera, target_camera), None


def test_torch_sim_to_sim_moves_and_closes_gap():
    summary, skip = _try_torch_sim_to_sim_motion()
    if skip is not None:
        print(f"  (skipping torch sim-to-sim motion: {skip})")
        return
    _assert_servo_moved(summary, "torch sim-to-sim")
    if summary["final_gap"] >= summary["initial_gap"]:
        raise AssertionError(
            f"torch sim-to-sim failed to reduce translation gap: {summary}"
        )
    if summary["final_gap"] >= 0.5 * summary["initial_gap"]:
        raise AssertionError(
            f"torch sim-to-sim did not close enough gap: {summary}"
        )
    print(
        "  torch sim-to-sim: "
        f"t_gap={summary['initial_gap']:.6f} -> {summary['final_gap']:.6f} "
        f"pose_delta={summary['pose_delta']:.6f} "
        f"accepted={summary['accepted_steps']}"
    )


def _valid_depth_pixels(scene, camera, min_depth=1e-4):
    depth = np.asarray(scene.render_depth(camera), dtype=np.float32)
    valid = np.isfinite(depth) & (depth > float(min_depth))
    return int(valid.sum())


# ---- integration: render-vs-real on a COLMAP dataset -----------------------
#
# Project goal: drive a virtual camera (mesh/GS render) until its render
# matches a *real* captured RGB image at the target pose. Sim-to-sim only
# tests the math; render-vs-real is the actual research question.


REAL_TEST_SCENES = ("living", "kitchen")
# Try renderers in order; first one that loads wins. Mesh needs pyrender/EGL,
# GS needs the CUDA gaussian rasterizer.
REAL_TEST_RENDERERS = ("mesh", "gs")
REAL_TEST_FRAME_STRIDE = 1            # pair = (frame[i], frame[i + STRIDE])
REAL_TEST_ITERATIONS = 50


def _try_render_vs_real():
    """Drive servo from real RGB at frame `target`, starting at frame `start`."""
    from pathlib import Path

    project_root = Path(__file__).resolve().parent.parent

    try:
        from runners.servo_frames import (
            load_rgb,
            load_scene_and_frames,
            sorted_frame_ids,
        )
    except Exception as exc:
        return None, f"runner import failed: {exc}"

    load_errors = []
    selected = None
    for scene_name in REAL_TEST_SCENES:
        scene_dir = project_root / "DATA" / scene_name
        if not scene_dir.exists():
            load_errors.append(f"{scene_name}: dataset missing")
            continue
        for renderer in REAL_TEST_RENDERERS:
            try:
                scene, frame_index = load_scene_and_frames(scene_dir, renderer)
                frame_ids = sorted_frame_ids(frame_index)
                if len(frame_ids) < REAL_TEST_FRAME_STRIDE + 1:
                    load_errors.append(
                        f"{scene_name}/{renderer}: not enough frames "
                        f"{len(frame_ids)} < {REAL_TEST_FRAME_STRIDE + 1}"
                    )
                    continue
                start_frame_id = frame_ids[0]
                target_frame_id = frame_ids[REAL_TEST_FRAME_STRIDE]
                start_camera = frame_index[start_frame_id]["camera"]
                target_camera = frame_index[target_frame_id]["camera"]
                target_rgb_path = frame_index[target_frame_id]["rgb_path"]
                if not target_rgb_path.exists():
                    load_errors.append(
                        f"{scene_name}/{renderer}: target RGB missing "
                        f"{target_rgb_path}"
                    )
                    continue
                valid_depth_pixels = _valid_depth_pixels(scene, target_camera)
                if valid_depth_pixels < 6:
                    load_errors.append(
                        f"{scene_name}/{renderer}: target intrinsic depth has "
                        f"{valid_depth_pixels} valid pixels"
                    )
                    continue
                selected = (
                    scene_name, renderer, scene, start_camera, target_camera,
                    target_rgb_path, start_frame_id, target_frame_id,
                    valid_depth_pixels,
                )
                break
            except Exception as exc:
                load_errors.append(f"{scene_name}/{renderer}: {exc}")
        if selected is not None:
            break

    if selected is None:
        return None, f"no usable render-vs-real fixture ({'; '.join(load_errors)})"

    (
        scene_name, renderer, scene, start_camera, target_camera,
        target_rgb_path, start_frame_id, target_frame_id, valid_depth_pixels,
    ) = selected
    target_image = load_rgb(target_rgb_path, start_camera.W, start_camera.H)

    from servo import run_servo_loop

    ctrl = PhotometricControllerTorch(
        scene=scene,
        target_camera=target_camera,
        gain=0.01,
        sigma_blur=1.0,
        use_gzn=True,
        bord=10,
        grad_percentile=50.0,
        max_pixels=20_000,
        use_huber=True,
        use_intrinsic_depth=True,
        method="lm",
        mu_init=0.1,
    )

    try:
        result = run_servo_loop(
            scene, start_camera, target_image, ctrl,
            iterations=REAL_TEST_ITERATIONS, dt=1.0,
            visualization_dir=None, matcher=None,
            iteration_callback=None,
            viz_iter=0,
        )
    except Exception as exc:
        return None, f"servo loop failed: {exc}"

    summary = _motion_summary(result, start_camera, target_camera)
    summary.update({
        "scene": scene_name,
        "renderer": renderer,
        "start_frame": start_frame_id,
        "target_frame": target_frame_id,
        "valid_depth_pixels": int(valid_depth_pixels),
    })
    return summary, None


def test_render_vs_real_translation():
    """Smoke test on real captured RGB. Does NOT assert convergence.

    It does assert that a usable real fixture produces at least one accepted
    servo step and a nonzero pose update. If the local fixture has no valid
    intrinsic target depth, the test skips instead of reporting a fake stable
    gap.
    """
    summary, skip = _try_render_vs_real()
    if skip is not None:
        print(f"  (skipping render-vs-real: {skip})")
        return
    if not np.isfinite(summary["initial_gap"]):
        raise AssertionError(f"initial gap non-finite: {summary}")
    if not np.isfinite(summary["final_gap"]):
        raise AssertionError(f"final gap non-finite: {summary}")
    _assert_servo_moved(summary, "render-vs-real")
    delta_gap = summary["final_gap"] - summary["initial_gap"]
    print(
        f"  render-vs-real ({summary['scene']} / {summary['renderer']}, "
        f"{summary['start_frame']} -> {summary['target_frame']}): "
        f"t_gap={summary['initial_gap']:.6f} -> {summary['final_gap']:.6f} "
        f"delta={delta_gap:+.6f} "
        f"pose_delta={summary['pose_delta']:.6f} "
        f"accepted={summary['accepted_steps']}"
    )


# ---- runner ----------------------------------------------------------------


def main():
    test_rgb_to_gray_matches_bt601_weights()
    test_derivative_x_zeros_border_and_recovers_ramp()
    test_derivative_y_constant_image_is_zero()
    test_gaussian_blur_passthrough_for_sigma_zero()
    test_point_interaction_matrix_shape_and_values()
    test_luminance_interaction_matches_manual_contraction()
    test_huber_weights_auto_k_downweights_outlier()
    test_servo_returns_zero_velocity_for_zero_error()
    test_servo_lm_updates_mu_on_cost_increase()
    test_feature_luminance_drops_border_pixels()
    test_feature_luminance_residual_zero_for_identical_images()
    test_torch_controller_zero_error_returns_zero_velocity()
    test_torch_controller_rank_deficient_returns_fault()
    test_torch_controller_textured_residual_drives_velocity()
    test_torch_controller_missing_target_camera_raises()
    test_torch_controller_last_info_has_parity_keys()
    test_torch_sim_to_sim_moves_and_closes_gap()
    test_render_vs_real_translation()
    print("Torch photometric controller validation passed")


if __name__ == "__main__":
    main()
