"""Unit + sim-to-sim tests for PhotometricController."""

import numpy as np

from camera import Camera
from controllers import (
    PhotometricController,
    build_pixel_mask,
    gaussian_blur,
    gzn_transform,
    huber_weights,
    image_gradient,
    image_to_gray,
    normalize_points,
    photometric_interaction,
    pixel_grid_normalized,
    point_interaction_matrix,
)


# ---- helpers ---------------------------------------------------------------


def make_camera(W=64, H=64, f=80.0):
    return Camera(
        np.eye(4, dtype=np.float32),
        fx=f,
        fy=f,
        cx=W * 0.5,
        cy=H * 0.5,
        H=H,
        W=W,
    )


class _FakeScene:
    """Scene-like stub: returns a fixed depth map regardless of camera."""

    def __init__(self, depth_map):
        self._depth = np.asarray(depth_map, dtype=np.float32)

    def render_depth(self, camera=None):
        return self._depth.copy()


# ---- unit: preprocessing ---------------------------------------------------


def test_image_to_gray_shape_and_dtype():
    rgb = np.random.rand(8, 12, 3).astype(np.float32)
    gray = image_to_gray(rgb)
    if gray.shape != (8, 12):
        raise AssertionError(f"gray shape {gray.shape}")
    if gray.dtype != np.float32:
        raise AssertionError(f"gray dtype {gray.dtype}")


def test_gradient_sobel_ramp():
    H, W = 16, 16
    x = np.arange(W, dtype=np.float32) * 0.1
    gray = np.broadcast_to(x, (H, W)).astype(np.float32)
    gx, gy = image_gradient(gray)
    # Interior pixels should see derivative ~ 0.1 in x, ~ 0 in y.
    if not np.allclose(gx[2:-2, 2:-2], 0.1, atol=1e-5):
        raise AssertionError(f"gx interior not 0.1: {gx[2:-2, 2:-2].mean()}")
    if not np.allclose(gy[2:-2, 2:-2], 0.0, atol=1e-5):
        raise AssertionError(f"gy interior not 0: {np.abs(gy[2:-2, 2:-2]).max()}")


def test_gaussian_blur_passthrough_for_zero_sigma():
    img = np.random.rand(10, 10).astype(np.float32)
    out = gaussian_blur(img, sigma=0.0)
    if not np.array_equal(img, out):
        raise AssertionError("sigma=0 should return identical array")


def test_gzn_zero_mean_unit_std():
    gray = np.random.rand(32, 32).astype(np.float32)
    out = gzn_transform(gray, sigma=0.0)
    if abs(float(out.mean())) > 1e-5:
        raise AssertionError(f"gzn mean not zero: {out.mean()}")
    if abs(float(out.std()) - 1.0) > 1e-5:
        raise AssertionError(f"gzn std not unit: {out.std()}")


def test_pixel_grid_normalized_matches_normalize_points():
    cam = make_camera(W=16, H=8, f=50.0)
    xn, yn = pixel_grid_normalized(cam)
    pix = np.array([[3.0, 5.0], [9.0, 1.0]], dtype=np.float32)
    ref = normalize_points(pix, cam)
    got = np.stack([xn[5, 3], yn[5, 3]]), np.stack([xn[1, 9], yn[1, 9]])
    if not np.allclose(got[0], ref[0], atol=1e-6):
        raise AssertionError(f"grid mismatch at (3,5): {got[0]} vs {ref[0]}")
    if not np.allclose(got[1], ref[1], atol=1e-6):
        raise AssertionError(f"grid mismatch at (9,1): {got[1]} vs {ref[1]}")


def test_photometric_interaction_matches_manual_contraction():
    rng = np.random.default_rng(0)
    n = 5
    gx = rng.standard_normal(n).astype(np.float32)
    gy = rng.standard_normal(n).astype(np.float32)
    xn = rng.standard_normal(n).astype(np.float32)
    yn = rng.standard_normal(n).astype(np.float32)
    z = (rng.random(n) + 1.0).astype(np.float32)

    L_photo = photometric_interaction(gx, gy, xn, yn, z)
    Lx = point_interaction_matrix(np.stack([xn, yn], axis=1), z).reshape(n, 2, 6)
    expected = -(np.stack([gx, gy], axis=1)[:, :, None] * Lx).sum(axis=1)
    if not np.allclose(L_photo, expected, atol=1e-6):
        raise AssertionError("photometric_interaction != manual contraction")
    if L_photo.shape != (n, 6):
        raise AssertionError(f"shape {L_photo.shape}")


def test_build_pixel_mask_drops_invalid():
    I = np.full((8, 8), 0.5, dtype=np.float32)
    gx = np.ones_like(I)
    gy = np.zeros_like(I)
    Z = np.full_like(I, 2.0)
    Z[0, 0] = -1.0           # invalid depth
    I[0, 1] = 0.99            # saturated (above 0.98)
    gx[0, 2] = 0.0; gy[0, 2] = 0.0  # zero gradient pixel

    mask = build_pixel_mask(
        I, gx, gy, Z,
        grad_percentile=10.0,
        sat_lo=0.02, sat_hi=0.98,
        min_depth=1e-4,
    )
    if mask[0, 0] or mask[0, 1]:
        raise AssertionError("mask should drop invalid depth + saturated pixel")
    if mask[0, 2]:
        raise AssertionError("mask should drop zero-gradient pixel via percentile")
    if mask.sum() == 0:
        raise AssertionError("mask should keep some pixels")


def test_huber_auto_k_from_mad():
    r = np.array([0.0, 1.0, -1.0, 0.5, -0.5, 100.0], dtype=np.float32)
    w, k = huber_weights(r, k=None)
    if k <= 0.0:
        raise AssertionError(f"huber k must be > 0, got {k}")
    if w[-1] >= 1.0:
        raise AssertionError("outlier should be down-weighted")
    if not np.all(w[:5] >= w[-1]):
        raise AssertionError("inliers should weigh >= outlier")


# ---- unit: controller behaviour --------------------------------------------


def _make_controller_with_constant_depth(cam, target, gain=0.5, **kwargs):
    depth = np.full((cam.H, cam.W), 2.0, dtype=np.float32)
    scene = _FakeScene(depth)
    ctrl = PhotometricController(
        scene=scene,
        target_camera=cam,
        gain=gain,
        sigma_blur=0.0,
        use_gzn=False,
        grad_percentile=10.0,
        max_pixels=0,
        use_huber=False,
        **kwargs,
    )
    return ctrl


def test_missing_target_camera_raises():
    cam = make_camera()
    scene = _FakeScene(np.full((cam.H, cam.W), 2.0, dtype=np.float32))
    ctrl = PhotometricController(scene=scene, target_camera=None,
                                 use_gzn=False, sigma_blur=0.0)
    target = np.random.rand(cam.H, cam.W, 3).astype(np.float32)
    rendered = target.copy()
    try:
        ctrl(rendered, target, cam, iteration=0)
    except RuntimeError as exc:
        if "target_camera" not in str(exc):
            raise AssertionError(f"wrong error: {exc}")
        return
    raise AssertionError("expected RuntimeError when target_camera missing")


def test_zero_error_returns_zero_velocity():
    cam = make_camera()
    # Make sure the target has gradients so the mask retains pixels.
    H, W = cam.H, cam.W
    target = np.zeros((H, W, 3), dtype=np.float32)
    target[:, :, 0] = np.linspace(0.1, 0.9, W, dtype=np.float32)[None, :]
    target[:, :, 1] = np.linspace(0.1, 0.9, H, dtype=np.float32)[:, None]
    target[:, :, 2] = 0.5

    ctrl = _make_controller_with_constant_depth(cam, target)
    rendered = target.copy()
    v = ctrl(rendered, target, cam, iteration=0)
    if v.shape != (6,):
        raise AssertionError(f"v shape {v.shape}")
    if float(np.linalg.norm(v)) > 1e-5:
        raise AssertionError(f"||v|| should be ~0 for zero error, got {v}")
    info = ctrl.last_info
    if info["feature_mode"] != "photometric":
        raise AssertionError("feature_mode must be 'photometric'")
    if info["num_inlier_matches"] <= 0:
        raise AssertionError("controller used 0 pixels")


def test_rank_deficient_photometric_returns_fault():
    cam = make_camera()
    H, W = cam.H, cam.W
    base = np.linspace(0.1, 0.9, W, dtype=np.float32)
    target = np.zeros((H, W, 3), dtype=np.float32)
    target[:, :, 0] = base[None, :]
    target[:, :, 1] = base[None, :]
    target[:, :, 2] = base[None, :]

    rendered = np.zeros_like(target)
    rendered[:, 1:, :] = target[:, :-1, :]
    rendered[:, 0, :] = target[:, 0, :]

    ctrl = _make_controller_with_constant_depth(cam, target, gain=1.0)
    v = ctrl(rendered, target, cam, iteration=0)
    if not np.allclose(v, 0.0, atol=1e-8):
        raise AssertionError(f"rank-deficient target should return zero velocity, got {v}")
    if ctrl.last_info["fault_reason"] != "measurement_invalid_rank_deficient":
        raise AssertionError(f"wrong fault: {ctrl.last_info}")
    if ctrl.should_stop() != "measurement_invalid_rank_deficient":
        raise AssertionError(f"wrong stop reason: {ctrl.should_stop()}")


def test_textured_residual_drives_nonzero_velocity():
    cam = make_camera()
    H, W = cam.H, cam.W
    rng = np.random.default_rng(3)
    target = rng.uniform(0.1, 0.9, (H, W, 3)).astype(np.float32)

    rendered = np.zeros_like(target)
    rendered[:, 1:, :] = target[:, :-1, :]
    rendered[:, 0, :] = target[:, 0, :]

    ctrl = _make_controller_with_constant_depth(cam, target, gain=1.0)
    v = ctrl(rendered, target, cam, iteration=0)
    if ctrl.last_info["fault_reason"] is not None:
        raise AssertionError(f"unexpected fault: {ctrl.last_info}")
    if ctrl.last_info["interaction_rank"] < 6:
        raise AssertionError(f"expected full rank, got {ctrl.last_info}")
    if not np.isfinite(v).all():
        raise AssertionError(f"v must be finite, got {v}")
    if float(np.linalg.norm(v)) <= 0.0:
        raise AssertionError("nonzero residual should produce nonzero velocity")


def test_target_change_invalidates_cache():
    cam = make_camera()
    H, W = cam.H, cam.W
    t1 = np.tile(np.linspace(0.1, 0.9, W, dtype=np.float32)[None, :, None], (H, 1, 3))
    t2 = t1[:, ::-1, :].copy()  # different buffer, different content

    ctrl = _make_controller_with_constant_depth(cam, t1)
    ctrl(t1.copy(), t1, cam, iteration=0)
    first_idx = ctrl._cached_target_id
    ctrl(t2.copy(), t2, cam, iteration=1)
    second_idx = ctrl._cached_target_id
    if first_idx == second_idx:
        raise AssertionError("cache should refresh when target buffer changes")


def test_last_info_has_ibvs_parity_keys():
    cam = make_camera()
    target = np.tile(
        np.linspace(0.1, 0.9, cam.W, dtype=np.float32)[None, :, None],
        (cam.H, 1, 3),
    )
    ctrl = _make_controller_with_constant_depth(cam, target)
    ctrl(target.copy(), target, cam, iteration=0)
    required = {
        "iteration", "feature_mode", "num_raw_matches", "num_inlier_matches",
        "num_cached_features", "num_dropped_features", "residual_norm",
        "velocity_norm", "mean_depth", "min_depth", "max_depth",
    }
    missing = required - set(ctrl.last_info)
    if missing:
        raise AssertionError(f"last_info missing keys: {missing}")


# ---- integration: sim-to-sim with a textured plane mesh --------------------


def _build_textured_plane_scene():
    """Return (MeshScene, plane_z) — a checker-textured plane at z = plane_z."""
    import trimesh
    from scenes.mesh import MeshScene

    grid = 32                  # vertices per side
    extent = 2.0               # plane half-size in metres
    plane_z = 2.0

    lin = np.linspace(-extent, extent, grid, dtype=np.float32)
    xx, yy = np.meshgrid(lin, lin)
    zz = np.full_like(xx, plane_z)
    verts = np.stack([xx.ravel(), yy.ravel(), zz.ravel()], axis=1)

    # Two-triangle quad per cell. Wind so the normal points back along -z
    # (towards the camera at the origin, which looks down +z in CV convention).
    faces = []
    for i in range(grid - 1):
        for j in range(grid - 1):
            a = i * grid + j
            b = a + 1
            c = a + grid
            d = c + 1
            faces.append([a, d, b])
            faces.append([a, c, d])
    faces = np.asarray(faces, dtype=np.int64)

    # Checker vertex colours (high spatial frequency = good for photometric).
    cu = ((np.arange(grid) // 2) % 2).astype(np.uint8)
    cv = ((np.arange(grid) // 2) % 2).astype(np.uint8)
    cgrid = (cu[None, :] ^ cv[:, None]).astype(np.uint8)
    base = np.where(cgrid == 0, 60, 220).astype(np.uint8)
    vert_colors = np.zeros((verts.shape[0], 4), dtype=np.uint8)
    flat = base.ravel()
    vert_colors[:, 0] = flat
    vert_colors[:, 1] = flat
    vert_colors[:, 2] = flat
    vert_colors[:, 3] = 255

    mesh = trimesh.Trimesh(vertices=verts, faces=faces,
                           vertex_colors=vert_colors, process=False)

    import tempfile, os
    fd, path = tempfile.mkstemp(suffix=".ply")
    os.close(fd)
    mesh.export(path)
    scene = MeshScene(path)
    return scene, plane_z


def _camera_at(translation_xyz, rotation_axis_angle_deg, W=128, H=128, f=160.0):
    from scipy.spatial.transform import Rotation
    T = np.eye(4, dtype=np.float32)
    if rotation_axis_angle_deg is not None:
        axis, deg = rotation_axis_angle_deg
        R = Rotation.from_rotvec(np.deg2rad(deg) * np.asarray(axis)).as_matrix()
        T[:3, :3] = R
    T[:3, 3] = np.asarray(translation_xyz, dtype=np.float32)
    return Camera(T, fx=f, fy=f, cx=W * 0.5, cy=H * 0.5, H=H, W=W)


def _try_sim_to_sim():
    """Returns scene-scale translation gap for a small task,
    or None if the renderer (EGL) is unavailable in this environment."""
    try:
        scene, _ = _build_textured_plane_scene()
    except Exception as exc:  # pyrender/EGL issues -> skip
        return None, None, str(exc)

    # Target camera looks at the plane (plane at z=+2 in world; camera at origin
    # looking down +z by convention used by MeshScene).
    target_cam = _camera_at(translation_xyz=(0.0, 0.0, 0.0),
                            rotation_axis_angle_deg=None)
    # Start: small translation perturbation along camera-x.
    start_cam = _camera_at(translation_xyz=(0.02, 0.0, 0.0),
                           rotation_axis_angle_deg=None)

    try:
        target_image = scene.render(target_cam)
    except Exception as exc:
        return None, None, str(exc)

    from servo import run_servo_loop
    # Phase-1 GN diverges with the paper's default gain (RR-6631 §4 is explicit
    # about this — they introduce LM to handle it). Use a small step size for
    # the GN-only smoke test; LM tuning is Phase-2 work.
    ctrl = PhotometricController(
        scene=scene,
        target_camera=target_cam,
        gain=0.005,
        sigma_blur=1.0,
        use_gzn=True,
        grad_percentile=50.0,
        max_pixels=20_000,
        use_huber=True,
        huber_k=None,
        use_intrinsic_depth=True,
    )

    initial_err = float(np.linalg.norm(
        start_cam.T_world_cam[:3, 3] - target_cam.T_world_cam[:3, 3]
    ))
    result = run_servo_loop(
        scene, start_cam, target_image, ctrl,
        iterations=200, dt=1.0,
        visualization_dir=None, matcher=None,
        iteration_callback=None,
        viz_iter=0,
    )
    final_cam = result["camera"]
    final_err = float(np.linalg.norm(
        final_cam.T_world_cam[:3, 3] - target_cam.T_world_cam[:3, 3]
    ))
    return initial_err, final_err, None


def test_sim_to_sim_translation():
    initial, final, skip = _try_sim_to_sim()
    if skip is not None:
        print(f"  (skipping sim-to-sim, renderer unavailable: {skip})")
        return
    if final >= initial:
        raise AssertionError(
            f"photometric controller failed to reduce translation error: "
            f"{initial:.4f} -> {final:.4f}"
        )
    if final >= 0.5 * initial:
        raise AssertionError(
            f"photometric controller did not converge enough: "
            f"{initial:.4f} -> {final:.4f} (need final < 0.5 * initial)"
        )
    print(f"  sim-to-sim t_gap: {initial:.6f} -> {final:.6f}")


# ---- runner ----------------------------------------------------------------


def main():
    test_image_to_gray_shape_and_dtype()
    test_gradient_sobel_ramp()
    test_gaussian_blur_passthrough_for_zero_sigma()
    test_gzn_zero_mean_unit_std()
    test_pixel_grid_normalized_matches_normalize_points()
    test_photometric_interaction_matches_manual_contraction()
    test_build_pixel_mask_drops_invalid()
    test_huber_auto_k_from_mad()
    test_missing_target_camera_raises()
    test_zero_error_returns_zero_velocity()
    test_rank_deficient_photometric_returns_fault()
    test_textured_residual_drives_nonzero_velocity()
    test_target_change_invalidates_cache()
    test_last_info_has_ibvs_parity_keys()
    test_sim_to_sim_translation()
    print("Photometric controller validation passed")


if __name__ == "__main__":
    main()
