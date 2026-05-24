"""Image-based visual servo controller (Chaumette & Hutchinson, 2006, Part I).

Implements the basic IBVS scheme:
    e   = s - s*                                     (paper eq. 1)
    s   = stacked normalized image points (x, y)     (paper eq. 6)
    L_x = per-point interaction matrix               (paper eq. 11)
    v_c = -lambda * pinv(L_e) * e                    (paper eq. 5)
"""

from __future__ import annotations
from typing import Optional, Union, Tuple, List, Dict, Callable, Any

import cv2
import numpy as np
from numpy.typing import NDArray
from scipy.stats import median_abs_deviation
import torch

from depth import get_depth
from features import FeatureMatcher, filter_matches
from camera import Camera



def normalize_points(kpts: NDArray[np.float32], camera: Camera) -> NDArray[np.float32]:
    """Pixels -> normalized image coordinates (x, y) = ((u-cu)/f, (v-cv)/f)."""
    kpts = np.asarray(kpts, dtype=np.float32).reshape(-1, 2)
    center = np.array([camera.cx, camera.cy], dtype=np.float32)
    inv_focal = np.array([1.0 / camera.fx, 1.0 / camera.fy], dtype=np.float32)
    return (kpts - center) * inv_focal


def sample_depth_nearest(depth: NDArray[np.float32], kpts: NDArray[np.float32], min_depth: float = 1e-4) -> Tuple[NDArray[np.float32], NDArray[bool]]:
    depth = np.asarray(depth, dtype=np.float32)
    kpts = np.asarray(kpts, dtype=np.float32).reshape(-1, 2)

    xs = np.rint(kpts[:, 0]).astype(np.int32)
    ys = np.rint(kpts[:, 1]).astype(np.int32)
    valid = (
        (xs >= 0)
        & (xs < depth.shape[1])
        & (ys >= 0)
        & (ys < depth.shape[0])
    )

    values = np.zeros(len(kpts), dtype=np.float32)
    values[valid] = depth[ys[valid], xs[valid]]
    valid &= np.isfinite(values) & (values > min_depth)
    return values, valid


def backproject_pixels(kpts: NDArray[np.float32], depths: NDArray[np.float32], camera: Camera) -> NDArray[np.float32]:
    kpts = np.asarray(kpts, dtype=np.float32).reshape(-1, 2)
    depths = np.asarray(depths, dtype=np.float32).reshape(-1)
    if len(kpts) != len(depths):
        raise ValueError("kpts and depths must have the same length")

    norm = normalize_points(kpts, camera)
    points_cam = np.empty((len(kpts), 3), dtype=np.float32)
    points_cam[:, :2] = norm * depths[:, None]
    points_cam[:, 2] = depths
    return points_cam


def camera_points_to_world(points_cam: NDArray[np.float32], camera: Camera) -> NDArray[np.float32]:
    points_cam = np.asarray(points_cam, dtype=np.float32).reshape(-1, 3)
    R = camera.T_world_cam[:3, :3]
    t = camera.T_world_cam[:3, 3]
    return (points_cam @ R.T + t).astype(np.float32, copy=False)


def project_world_points(points_world: NDArray[np.float32], camera: Camera, min_depth: float = 1e-4) -> Tuple[NDArray[np.float32], NDArray[np.float32], NDArray[bool]]:
    points_world = np.asarray(points_world, dtype=np.float32).reshape(-1, 3)
    R = camera.T_cam_world[:3, :3]
    t = camera.T_cam_world[:3, 3]
    points_cam = points_world @ R.T + t
    depths = points_cam[:, 2]

    valid = np.isfinite(points_cam).all(axis=1) & (depths > float(min_depth))
    pixels = np.zeros((len(points_world), 2), dtype=np.float32)
    safe_depth = np.where(valid, depths, 1.0)
    pixels[:, 0] = camera.fx * (points_cam[:, 0] / safe_depth) + camera.cx
    pixels[:, 1] = camera.fy * (points_cam[:, 1] / safe_depth) + camera.cy
    pixels[~valid] = 0.0
    valid &= (
        np.isfinite(pixels).all(axis=1)
        & (pixels[:, 0] >= 0.0)
        & (pixels[:, 0] < camera.W)
        & (pixels[:, 1] >= 0.0)
        & (pixels[:, 1] < camera.H)
    )
    return pixels, depths.astype(np.float32, copy=False), valid


def point_interaction_matrix(points: NDArray[np.float32], depths: NDArray[np.float32]) -> NDArray[np.float32]:
    """Stack per-point L_x (paper eq. 11). Input points are normalized (x, y)."""
    points = np.asarray(points, dtype=np.float32).reshape(-1, 2)
    depths = np.asarray(depths, dtype=np.float32).reshape(-1)
    if len(points) != len(depths):
        raise ValueError("points and depths must have the same length")

    L = np.zeros((2 * len(points), 6), dtype=np.float32)
    x = points[:, 0]
    y = points[:, 1]
    inv_z = 1.0 / depths

    L[0::2, 0] = -inv_z
    L[0::2, 2] = x * inv_z
    L[0::2, 3] = x * y
    L[0::2, 4] = -(1.0 + x * x)
    L[0::2, 5] = y

    L[1::2, 1] = -inv_z
    L[1::2, 2] = y * inv_z
    L[1::2, 3] = 1.0 + y * y
    L[1::2, 4] = -x * y
    L[1::2, 5] = -x

    return L


# ----------------------------------------------------------------------------
# Photometric helpers (Collewet & Marchand RR-6631; Rodriguez et al. Sensors 2020)
# ----------------------------------------------------------------------------


def image_to_gray(rgb_float01: NDArray[np.float32]) -> NDArray[np.float32]:
    """RGB float in [0, 1] -> grayscale float32 (H, W)."""
    rgb = np.asarray(rgb_float01, dtype=np.float32)
    if rgb.ndim == 2:
        return rgb.astype(np.float32, copy=False)
    if rgb.ndim != 3 or rgb.shape[2] not in (3, 4):
        raise ValueError(f"image_to_gray expected (H, W, 3|4), got {rgb.shape}")
    if rgb.shape[2] == 4:
        rgb = rgb[..., :3]
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)


def gaussian_blur(image: NDArray[np.float32], sigma: float) -> NDArray[np.float32]:
    """Isotropic Gaussian blur. Passthrough for sigma <= 0."""
    image = np.asarray(image, dtype=np.float32)
    if float(sigma) <= 0.0:
        return image
    return cv2.GaussianBlur(image, ksize=(0, 0), sigmaX=float(sigma))


def gzn_transform(gray: NDArray[np.float32], sigma: float) -> NDArray[np.float32]:
    """Gaussian + Zero-mean Normalization (Rodriguez Sensors 2020, eq. 1+2)."""
    gray = np.asarray(gray, dtype=np.float32)
    blurred = gaussian_blur(gray, sigma)
    mean = float(blurred.mean())
    std = float(blurred.std())
    return (blurred - mean) / (std + 1e-8)


def image_gradient(gray: NDArray[np.float32]) -> Tuple[NDArray[np.float32], NDArray[np.float32]]:
    """Sobel ksize=3 spatial gradient, scaled to true derivative magnitude."""
    gray = np.asarray(gray, dtype=np.float32)
    # Sobel ksize=3 kernel sum on one side = 4; for symmetric derivative -> /8
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3) * (1.0 / 8.0)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3) * (1.0 / 8.0)
    return gx, gy


def pixel_grid_normalized(camera: Camera) -> Tuple[NDArray[np.float32], NDArray[np.float32]]:
    """(H, W) meshgrids of normalized image coords x=(u-cx)/fx, y=(v-cy)/fy."""
    u = np.arange(camera.W, dtype=np.float32)
    v = np.arange(camera.H, dtype=np.float32)
    uu, vv = np.meshgrid(u, v)
    xn = (uu - camera.cx) / camera.fx
    yn = (vv - camera.cy) / camera.fy
    return xn.astype(np.float32, copy=False), yn.astype(np.float32, copy=False)


def photometric_interaction(gx: NDArray[np.float32], gy: NDArray[np.float32], xn: NDArray[np.float32], yn: NDArray[np.float32], z: NDArray[np.float32]) -> NDArray[np.float32]:
    """Per-pixel luminance interaction matrix (RR-6631 eq. 9).

        L_I_i = -[gx_i, gy_i] @ L_x(xn_i, yn_i, z_i)

    Inputs are 1-D arrays of length N (already masked & flattened).
    Returns (N, 6) float32.
    """
    gx = np.asarray(gx, dtype=np.float32).reshape(-1)
    gy = np.asarray(gy, dtype=np.float32).reshape(-1)
    xn = np.asarray(xn, dtype=np.float32).reshape(-1)
    yn = np.asarray(yn, dtype=np.float32).reshape(-1)
    z = np.asarray(z, dtype=np.float32).reshape(-1)
    n = gx.size
    if not (gy.size == n and xn.size == n and yn.size == n and z.size == n):
        raise ValueError("photometric_interaction: input length mismatch")

    points = np.stack([xn, yn], axis=1)
    Lx = point_interaction_matrix(points, z).reshape(n, 2, 6)
    grads = np.stack([gx, gy], axis=1)
    L = -np.einsum('ni,nij->nj', grads, Lx)
    return L.astype(np.float32, copy=False)


def build_pixel_mask(
    I_star: NDArray[np.float32],
    gx_star: NDArray[np.float32],
    gy_star: NDArray[np.float32],
    Z_star: NDArray[np.float32],
    grad_percentile: float = 50.0,
    sat_lo: float = 0.02,
    sat_hi: float = 0.98,
    min_depth: float = 1e-4,
) -> NDArray[bool]:
    """Boolean mask of pixels usable for photometric servo on the desired image.

    Drops: invalid/shallow depth, saturated intensity, low-gradient pixels.
    `grad_percentile` keeps the top (100 - p)% by |grad| within the depth-and-
    intensity-valid region.
    """
    I_star = np.asarray(I_star, dtype=np.float32)
    gx_star = np.asarray(gx_star, dtype=np.float32)
    gy_star = np.asarray(gy_star, dtype=np.float32)
    Z_star = np.asarray(Z_star, dtype=np.float32)

    valid = np.isfinite(Z_star) & (Z_star > float(min_depth))
    valid &= np.isfinite(I_star)
    # sat_lo / sat_hi semantics only make sense in [0, 1] raw intensity space;
    # if the input is GZN-normalized (sat range meaningless) the caller can
    # pass sat_lo=-inf, sat_hi=inf to disable.
    if np.isfinite(sat_lo):
        valid &= I_star >= float(sat_lo)
    if np.isfinite(sat_hi):
        valid &= I_star <= float(sat_hi)

    grad_mag = np.sqrt(gx_star * gx_star + gy_star * gy_star)
    if valid.any():
        thresh = float(np.percentile(grad_mag[valid], float(grad_percentile)))
    else:
        thresh = 0.0
    valid &= grad_mag >= thresh
    return valid


def huber_weights(residuals: NDArray[np.float32], k: Optional[float] = None) -> Tuple[NDArray[np.float32], float]:
    """Huber re-weighting w_i = 1 if |r| <= k else k / |r|.

    If `k` is None, derive k = 1.345 * 1.4826 * MAD(residuals).
    """
    r = np.asarray(residuals, dtype=np.float32).reshape(-1)
    abs_r = np.abs(r)
    if k is None:
        k = 1.345 * max(float(median_abs_deviation(r, scale="normal")), 1e-6)
    k = float(k)
    w = np.ones_like(abs_r)
    big = abs_r > k
    w[big] = k / np.maximum(abs_r[big], 1e-12)
    return w.astype(np.float32, copy=False), k


DEFAULT_MIN_INTERACTION_RANK = 6
DEFAULT_MAX_INTERACTION_CONDITION = 1.0e8
INTERACTION_RANK_REL_TOL = 1.0e-12
INTERACTION_RANK_ABS_TOL = 1.0e-12


def validate_interaction_guard(
    min_rank: int = DEFAULT_MIN_INTERACTION_RANK,
    max_condition: Optional[float] = DEFAULT_MAX_INTERACTION_CONDITION,
) -> Tuple[int, Optional[float]]:
    min_rank = int(min_rank)
    if min_rank < 1 or min_rank > 6:
        raise ValueError("min_interaction_rank must be in [1, 6]")
    if max_condition is None:
        return min_rank, None
    max_condition = float(max_condition)
    if not np.isfinite(max_condition) or max_condition <= 0.0:
        raise ValueError("max_interaction_condition must be positive or None")
    return min_rank, max_condition


def interaction_matrix_diagnostics(
    L: NDArray[np.float32],
    min_rank: int = DEFAULT_MIN_INTERACTION_RANK,
    max_condition: Optional[float] = DEFAULT_MAX_INTERACTION_CONDITION,
) -> Dict[str, Any]:
    """Return rank/conditioning diagnostics for an effective interaction matrix."""
    min_rank, max_condition = validate_interaction_guard(min_rank, max_condition)
    arr = np.asarray(L, dtype=np.float64)
    rows = int(arr.shape[0]) if arr.ndim >= 1 else 0
    cols = int(arr.shape[1]) if arr.ndim == 2 else 0
    info = {
        "interaction_rows": rows,
        "interaction_cols": cols,
        "interaction_rank": 0,
        "interaction_min_rank": int(min_rank),
        "interaction_condition": float("inf"),
        "interaction_max_condition": max_condition,
        "interaction_min_singular": 0.0,
        "interaction_max_singular": 0.0,
        "interaction_rank_tolerance": float("nan"),
        "interaction_svd_failed": False,
    }

    if arr.ndim != 2 or rows == 0 or cols == 0 or not np.isfinite(arr).all():
        return info

    try:
        singular_values = np.linalg.svd(arr, compute_uv=False)
    except np.linalg.LinAlgError:
        info["interaction_svd_failed"] = True
        return info

    if singular_values.size == 0:
        return info

    max_singular = float(singular_values[0])
    rank_tolerance = max(
        INTERACTION_RANK_ABS_TOL,
        INTERACTION_RANK_REL_TOL * max_singular,
    )
    rank = int(np.count_nonzero(singular_values > rank_tolerance))
    required_index = int(min_rank) - 1
    min_required_singular = (
        float(singular_values[required_index])
        if required_index < singular_values.size
        else 0.0
    )
    condition = (
        max_singular / min_required_singular
        if min_required_singular > 0.0
        else float("inf")
    )

    info.update({
        "interaction_rank": rank,
        "interaction_condition": float(condition),
        "interaction_min_singular": float(min_required_singular),
        "interaction_max_singular": float(max_singular),
        "interaction_rank_tolerance": float(rank_tolerance),
    })
    return info


def interaction_fault_reason(info: Dict[str, Any]) -> Optional[str]:
    if bool(info.get("interaction_svd_failed", False)):
        return "measurement_invalid_svd_failed"
    rank = int(info.get("interaction_rank", 0))
    min_rank = int(info.get("interaction_min_rank", DEFAULT_MIN_INTERACTION_RANK))
    if rank < min_rank:
        return "measurement_invalid_rank_deficient"
    max_condition = info.get("interaction_max_condition")
    condition = float(info.get("interaction_condition", float("inf")))
    if (
        max_condition is not None
        and np.isfinite(condition)
        and condition > float(max_condition)
    ):
        return "measurement_invalid_ill_conditioned"
    if max_condition is not None and not np.isfinite(condition):
        return "measurement_invalid_ill_conditioned"
    return None


class IBVSController:
    """Paper IBVS with optional feature-cache refresh ratio.

    `ratio` controls when features are re-matched:
        ratio = 1  -> re-match every iteration (paper default)
        ratio = N  -> re-match every Nth iteration; reproject cached 3D points between
        ratio = 0  -> match once on first call, reproject forever
    """

    def __init__(
        self,
        matcher: Optional[FeatureMatcher] = None,
        feature_method: str = "xfeat",
        gain: float = 0.5,
        min_features: int = 3,
        min_depth: float = 1e-4,
        depth_provider: Optional[Callable[[NDArray[np.float32]], NDArray[np.float32]]] = None,
        scene: Optional[Any] = None,
        use_intrinsic_depth: bool = False,
        ratio: int = 1,
        damping: float = 0.01,
        adaptive_gain: bool = True,
        velocity_alpha: float = 0.8,
        stop_residual_px: float = 0.5,
        min_interaction_rank: int = DEFAULT_MIN_INTERACTION_RANK,
        max_interaction_condition: Optional[float] = DEFAULT_MAX_INTERACTION_CONDITION,
    ) -> None:
        if int(ratio) < 0:
            raise ValueError("ratio must be >= 0")
        self.matcher = matcher if matcher is not None else FeatureMatcher(method=feature_method)
        self.gain = float(gain)
        self.min_features = int(min_features)
        self.min_depth = float(min_depth)
        self.depth_provider = depth_provider
        self.scene = scene
        self.use_intrinsic_depth = bool(use_intrinsic_depth)
        self.ratio = int(ratio)
        self.damping = float(damping)
        self.adaptive_gain = bool(adaptive_gain)
        self.velocity_alpha = float(velocity_alpha)
        # Stop when per-feature mean pixel error (RMS over 2N components) drops
        # below this threshold. Default 0.5 px ~ subpixel agreement.
        self.stop_residual_px = float(stop_residual_px)
        (
            self.min_interaction_rank,
            self.max_interaction_condition,
        ) = validate_interaction_guard(
            min_interaction_rank,
            max_interaction_condition,
        )

        self.cached_points_world = None
        self.cached_target_kpts = None
        self.cache_refresh_iteration = None
        self.last_info = {}
        self.last_visualization = {}
        self.prev_velocity = None

    def _depth(self, rendered: NDArray[np.float32]) -> NDArray[np.float32]:
        if self.depth_provider is not None:
            return np.asarray(self.depth_provider(rendered), dtype=np.float32)
        return get_depth(
            rendered,
            scene=self.scene,
            use_intrinsic=self.use_intrinsic_depth,
        )

    def _should_refresh(self, iteration: int) -> bool:
        if self.cached_points_world is None:
            return True
        if self.ratio == 0:
            return False
        return int(iteration) % self.ratio == 0

    def _refresh(self, rendered: NDArray[np.float32], target: NDArray[np.float32], camera: Camera, iteration: int) -> Dict[str, Any]:
        kpts_current, kpts_target = self.matcher.match(rendered, target)
        num_raw = len(kpts_current)
        kpts_current, kpts_target, _, _, _, _ = filter_matches(
            kpts_current,
            kpts_target,
            camera,
        )

        depth_map = self._depth(rendered)
        depths, valid = sample_depth_nearest(
            depth_map,
            kpts_current,
            min_depth=self.min_depth,
        )
        kpts_current = kpts_current[valid]
        kpts_target = kpts_target[valid]
        depths = depths[valid]

        if len(kpts_current) < self.min_features:
            self.cached_points_world = None
            self.cached_target_kpts = None
            self.cache_refresh_iteration = int(iteration)
            return {
                "mode": "refresh",
                "num_raw_matches": int(num_raw),
                "kpts_current": kpts_current,
                "kpts_target": kpts_target,
                "depths": depths,
                "num_dropped_features": int(max(num_raw - len(kpts_current), 0)),
                "fault_reason": "measurement_invalid_not_enough_features",
            }

        points_cam = backproject_pixels(kpts_current, depths, camera)
        self.cached_points_world = camera_points_to_world(points_cam, camera)
        self.cached_target_kpts = kpts_target.copy()
        self.cache_refresh_iteration = int(iteration)

        return {
            "mode": "refresh",
            "num_raw_matches": int(num_raw),
            "kpts_current": kpts_current,
            "kpts_target": kpts_target,
            "depths": depths,
            "num_dropped_features": int(max(num_raw - len(kpts_current), 0)),
            "fault_reason": None,
        }

    def _reproject(self, rendered: NDArray[np.float32], camera: Camera) -> Dict[str, Any]:
        cached_before = len(self.cached_points_world)
        kpts_current, depths_geom, valid = project_world_points(
            self.cached_points_world,
            camera,
            min_depth=self.min_depth,
        )
        kpts_current = kpts_current[valid]
        depths = depths_geom[valid]
        kpts_target = self.cached_target_kpts[valid].copy()

        # With intrinsic (exact) depth from the scene, sample current depth at
        # the reprojected pixels instead of using the geometric Zc. Closer to the
        # paper's L_x (uses current observed Z) and catches occlusion / surface
        # changes. With learned depth we trust the cached 3D point's geometric
        # Zc more than a noisy depth map.
        if self.use_intrinsic_depth and len(kpts_current) > 0:
            depth_map = self._depth(rendered)
            measured, valid_z = sample_depth_nearest(
                depth_map,
                kpts_current,
                min_depth=self.min_depth,
            )
            kpts_current = kpts_current[valid_z]
            kpts_target = kpts_target[valid_z]
            depths = measured[valid_z]

        if len(kpts_current) < self.min_features:
            return {
                "mode": "reproject",
                "num_raw_matches": int(cached_before),
                "kpts_current": kpts_current,
                "kpts_target": kpts_target,
                "depths": depths,
                "num_dropped_features": int(cached_before - len(kpts_current)),
                "fault_reason": "measurement_invalid_not_enough_features",
            }

        return {
            "mode": "reproject",
            "num_raw_matches": int(cached_before),
            "kpts_current": kpts_current,
            "kpts_target": kpts_target,
            "depths": depths,
            "num_dropped_features": int(cached_before - len(kpts_current)),
            "fault_reason": None,
        }

    def __call__(self, rendered: NDArray[np.float32], target: NDArray[np.float32], camera: Camera, iteration: int) -> NDArray[np.float32]:
        if self._should_refresh(iteration):
            state = self._refresh(rendered, target, camera, iteration)
        else:
            state = self._reproject(rendered, camera)

        kpts_current = state["kpts_current"]
        kpts_target = state["kpts_target"]
        depths = state["depths"]

        fault_reason = state.get("fault_reason")
        if len(kpts_current) > 0:
            # Normalize image coordinates (paper eq. 6).
            x = normalize_points(kpts_current, camera)
            x_star = normalize_points(kpts_target, camera)
            error = (x - x_star).reshape(-1)
            pixel_error = (kpts_current - kpts_target).reshape(-1)
        else:
            x = np.zeros((0, 2), dtype=np.float32)
            error = np.zeros((0,), dtype=np.float32)
            pixel_error = np.zeros((0,), dtype=np.float32)

        interaction_info = interaction_matrix_diagnostics(
            np.zeros((0, 6), dtype=np.float32),
            self.min_interaction_rank,
            self.max_interaction_condition,
        )
        if fault_reason is None:
            # Build L_e from current points + current depths (paper eq. 11).
            L = point_interaction_matrix(x, depths)
            if not np.isfinite(error).all() or not np.isfinite(L).all():
                fault_reason = "measurement_invalid_nonfinite"
                velocity = np.zeros(6, dtype=np.float32)
            else:
                interaction_info = interaction_matrix_diagnostics(
                    L,
                    self.min_interaction_rank,
                    self.max_interaction_condition,
                )
                interaction_fault = interaction_fault_reason(interaction_info)
                if interaction_fault is not None:
                    fault_reason = interaction_fault
                    velocity = np.zeros(6, dtype=np.float32)
                else:
                    # Control law: v_c = -lambda * pinv(L_e) * e (paper eq. 5).
                    velocity = (-self.gain * (np.linalg.pinv(L) @ error)).astype(np.float32)
        else:
            velocity = np.zeros(6, dtype=np.float32)

        n_components = int(error.size)
        residual_norm = float(np.linalg.norm(error)) if n_components else float("nan")
        pixel_residual_norm = (
            float(np.linalg.norm(pixel_error)) if pixel_error.size else float("nan")
        )
        residual_rms_px = (
            pixel_residual_norm / float(np.sqrt(int(pixel_error.size)))
            if pixel_error.size else float("nan")
        )
        cached_features = (
            0 if self.cached_points_world is None else int(len(self.cached_points_world))
        )
        if depths.size:
            mean_d = float(np.mean(depths))
            min_d = float(np.min(depths))
            max_d = float(np.max(depths))
        else:
            mean_d = min_d = max_d = float("nan")

        self.last_info = {
            "iteration": int(iteration),
            "feature_mode": state["mode"],
            "ratio": self.ratio,
            "cache_refresh_iteration": self.cache_refresh_iteration,
            "num_raw_matches": state["num_raw_matches"],
            "num_inlier_matches": int(len(kpts_current)),
            "num_cached_features": cached_features,
            "num_dropped_features": state["num_dropped_features"],
            "measurement_valid": fault_reason is None,
            "fault_reason": fault_reason,
            "residual_norm": residual_norm,
            "residual_norm_px": pixel_residual_norm,
            "residual_rms_px": residual_rms_px,
            **interaction_info,
            "velocity_norm": float(np.linalg.norm(velocity)),
            "mean_depth": mean_d,
            "min_depth": min_d,
            "max_depth": max_d,
        }
        self.last_visualization = {
            "iteration": int(iteration),
            "feature_mode": state["mode"],
            "num_raw_matches": state["num_raw_matches"],
            "kpts_current": kpts_current.copy(),
            "kpts_target": kpts_target.copy(),
        }
        return velocity

    def should_stop(self) -> Optional[str]:
        """Stop when RMS feature reprojection error <= stop_residual_px."""
        fault_reason = self.last_info.get("fault_reason")
        if fault_reason:
            return str(fault_reason)
        rms = self.last_info.get("residual_rms_px")
        if rms is None or not np.isfinite(float(rms)):
            return None
        if float(rms) <= self.stop_residual_px:
            return "ibvs_rms_below_threshold"
        return None

    def feature_error_px(
        self,
        rendered: NDArray[np.float32],
        target: NDArray[np.float32],
        camera: Camera,
    ) -> float:
        """Return ||kpts_current - kpts_target||_2 over filtered correspondences.

        Used by the trajectory runner to compute the paper's w(i) ratio for
        dynamic iteration scheduling (eq. 7). Does not require depth — only
        2D pixel coordinates. Returns 0.0 if fewer than `min_features` match.
        """
        kpts_current, kpts_target = self.matcher.match(rendered, target)
        kpts_current, kpts_target, _, _, _, _ = filter_matches(
            kpts_current, kpts_target, camera,
        )
        if len(kpts_current) < self.min_features:
            return 0.0
        return float(np.linalg.norm((kpts_current - kpts_target).reshape(-1)))


class PhotometricController:
    """Direct photometric visual servo (Collewet & Marchand RR-6631).

    Uses raw pixel intensities as the visual feature; no matching/tracking.
    Error e = I - I*; control law v = -gain * (H + eps I)^-1 L^T e with the
    analytic luminance interaction matrix L_I = -grad(I)^T L_x. Optionally
    applies Gaussian Zero-mean Normalization (Rodriguez Sensors 2020) for
    robustness to global illumination changes, and Huber re-weighting for
    robustness to occluders/specularities. Depth is taken at the desired pose
    once and cached (`depth_mode="desired_only"`); requires `target_camera`
    via constructor or `set_target_camera()` before the first call.

    Phase 1 scope: Gauss-Newton solve, single scale, desired_only depth.
    LM / ESM / multi-scale pyramid are deferred (see plan file).
    """

    def __init__(
        self,
        scene: Any,
        target_camera: Optional[Camera] = None,
        gain: float = 0.5,
        sigma_blur: float = 1.0,
        use_gzn: bool = True,
        grad_percentile: float = 50.0,
        sat_lo: float = 0.02,
        sat_hi: float = 0.98,
        min_depth: float = 1e-4,
        max_pixels: int = 50_000,
        use_huber: bool = True,
        huber_k: Optional[float] = None,
        use_intrinsic_depth: bool = True,
        depth_provider: Optional[Callable[[NDArray[np.float32]], NDArray[np.float32]]] = None,
        seed: int = 0,
        stop_mse_per_px: float = 2.0e-6,
        stop_ssd: Optional[float] = None,
        min_interaction_rank: int = DEFAULT_MIN_INTERACTION_RANK,
        max_interaction_condition: Optional[float] = DEFAULT_MAX_INTERACTION_CONDITION,
    ) -> None:
        if scene is None and depth_provider is None:
            raise ValueError(
                "PhotometricController needs `scene` (with render_depth) "
                "or an explicit `depth_provider`."
            )
        self.scene = scene
        self.target_camera = target_camera
        self.gain = float(gain)
        self.sigma_blur = float(sigma_blur)
        self.use_gzn = bool(use_gzn)
        self.grad_percentile = float(grad_percentile)
        self.sat_lo = float(sat_lo)
        self.sat_hi = float(sat_hi)
        self.min_depth = float(min_depth)
        self.max_pixels = int(max_pixels) if max_pixels else 0
        self.use_huber = bool(use_huber)
        self.huber_k = None if huber_k is None else float(huber_k)
        self.use_intrinsic_depth = bool(use_intrinsic_depth)
        self.depth_provider = depth_provider
        # ViSP-style stop is SSD on the feature error vector. If stop_ssd is
        # None, derive the SSD threshold from the legacy per-pixel MSE knob.
        self.stop_mse_per_px = float(stop_mse_per_px)
        self.stop_ssd = None if stop_ssd is None else float(stop_ssd)
        (
            self.min_interaction_rank,
            self.max_interaction_condition,
        ) = validate_interaction_guard(
            min_interaction_rank,
            max_interaction_condition,
        )

        self._rng = np.random.default_rng(int(seed))

        # Camera-grid cache.
        self._camera_grid_key = None
        self._x_norm_full = None
        self._y_norm_full = None

        # Desired-side cache (per target).
        self._cached_target_id = None
        self._I_star_flat = None
        self._I_star_raw_flat = None
        self._Z_star_flat_masked = None
        self._xn_flat_masked = None
        self._yn_flat_masked = None
        self._mask_idx = None
        self._num_total_valid = 0

        self.last_info = {}
        self.last_visualization = {}

    # -- public ----------------------------------------------------------------

    def set_target_camera(self, camera: Camera) -> None:
        """Provide / replace the desired-pose camera. Invalidates the cache."""
        self.target_camera = camera
        self._cached_target_id = None

    def __call__(self, rendered: NDArray[np.float32], target: NDArray[np.float32], camera: Camera, iteration: int) -> NDArray[np.float32]:
        self._ensure_camera_grid(camera)

        if self._cached_target_id != id(target) or int(iteration) == 0:
            self._build_target_cache(target)

        e, L, depths_used, raw_error = self._compute_residual_and_L(rendered)
        control_error = e.copy()

        huber_k_active = None
        if self.use_huber and e.size > 0:
            w, huber_k_active = huber_weights(e, k=self.huber_k)
            sqrt_w = np.sqrt(w)
            e = e * sqrt_w
            L = L * sqrt_w[:, None]

        interaction_info = interaction_matrix_diagnostics(
            L,
            self.min_interaction_rank,
            self.max_interaction_condition,
        )
        fault_reason = None
        if e.size < 6:
            fault_reason = "measurement_invalid_not_enough_pixels"
            velocity = np.zeros(6, dtype=np.float32)
        elif not np.isfinite(e).all() or not np.isfinite(L).all():
            fault_reason = "measurement_invalid_nonfinite"
            velocity = np.zeros(6, dtype=np.float32)
        else:
            interaction_fault = interaction_fault_reason(interaction_info)
            if interaction_fault is not None:
                fault_reason = interaction_fault
                velocity = np.zeros(6, dtype=np.float32)
            else:
                H = L.T @ L
                b = L.T @ e
                H_reg = H + 1e-9 * np.eye(6, dtype=np.float32)
                try:
                    step = np.linalg.solve(H_reg, b)
                except np.linalg.LinAlgError:
                    step = np.linalg.pinv(L) @ e
                velocity = (-self.gain * step).astype(np.float32)

        n_used = int(control_error.size)
        control_residual_norm = (
            float(np.linalg.norm(control_error)) if control_error.size else float("nan")
        )
        control_residual_ssd = (
            float(np.square(control_error).sum()) if control_error.size else float("nan")
        )
        control_residual_mse = (
            float(np.square(control_error).mean()) if control_error.size else float("nan")
        )
        weighted_residual_norm = float(np.linalg.norm(e)) if e.size else float("nan")
        weighted_residual_ssd = float(np.square(e).sum()) if e.size else float("nan")
        weighted_residual_mse = float(np.square(e).mean()) if e.size else float("nan")
        raw_image_ssd = (
            float(np.square(raw_error).sum()) if raw_error.size else float("nan")
        )
        raw_image_mse = (
            float(np.square(raw_error).mean()) if raw_error.size else float("nan")
        )
        stop_ssd_threshold = self._stop_ssd_threshold(n_used)
        if depths_used.size:
            mean_d = float(depths_used.mean())
            min_d = float(depths_used.min())
            max_d = float(depths_used.max())
        else:
            mean_d = min_d = max_d = float("nan")

        self.last_info = {
            "iteration": int(iteration),
            "feature_mode": "photometric",
            "controller": "photometric",
            "num_raw_matches": n_used,
            "num_inlier_matches": n_used,
            "num_cached_features": n_used,
            "num_dropped_features": int(self._num_total_valid - n_used),
            "measurement_valid": fault_reason is None,
            "fault_reason": fault_reason,
            "residual_norm": control_residual_norm,
            "residual_ssd": control_residual_ssd,
            "residual_mse_per_px": control_residual_mse,
            "control_residual_norm": control_residual_norm,
            "control_residual_ssd": control_residual_ssd,
            "control_residual_mse_per_px": control_residual_mse,
            "weighted_residual_norm": weighted_residual_norm,
            "weighted_residual_ssd": weighted_residual_ssd,
            "weighted_residual_mse_per_px": weighted_residual_mse,
            "raw_image_ssd": raw_image_ssd,
            "raw_image_mse_per_px": raw_image_mse,
            "stop_ssd": control_residual_ssd,
            "stop_ssd_threshold": stop_ssd_threshold,
            "stop_mse_per_px_threshold": self.stop_mse_per_px,
            **interaction_info,
            "velocity_norm": float(np.linalg.norm(velocity)),
            "mean_depth": mean_d,
            "min_depth": min_d,
            "max_depth": max_d,
            "num_pixels_used": n_used,
            "use_gzn": self.use_gzn,
            "huber_k_active": (
                float(huber_k_active) if huber_k_active is not None else None
            ),
        }
        self.last_visualization = {
            "iteration": int(iteration),
            "feature_mode": "photometric",
            "num_pixels_used": n_used,
            "residual_norm": control_residual_norm,
            "residual_ssd": control_residual_ssd,
        }
        return velocity

    def should_stop(self) -> Optional[str]:
        """ViSP-style stop: SSD of the feature error vector below threshold."""
        fault_reason = self.last_info.get("fault_reason")
        if fault_reason:
            return str(fault_reason)
        ssd = self.last_info.get("stop_ssd")
        threshold = self.last_info.get("stop_ssd_threshold")
        if (
            ssd is None
            or threshold is None
            or not np.isfinite(float(ssd))
            or not np.isfinite(float(threshold))
        ):
            return None
        if float(ssd) <= float(threshold):
            return "photometric_ssd_below_threshold"
        return None

    # -- internals -------------------------------------------------------------

    def _stop_ssd_threshold(self, num_pixels: int) -> float:
        if self.stop_ssd is not None:
            return float(self.stop_ssd)
        if int(num_pixels) <= 0:
            return float("nan")
        return float(self.stop_mse_per_px) * float(num_pixels)

    def _depth(self, image, camera=None):
        if self.depth_provider is not None:
            return np.asarray(self.depth_provider(image), dtype=np.float32)
        if self.use_intrinsic_depth:
            if self.scene is None:
                raise RuntimeError("scene required for intrinsic depth")
            return np.asarray(
                self.scene.render_depth(camera), dtype=np.float32
            )
        return get_depth(image, scene=self.scene, use_intrinsic=False)

    def _ensure_camera_grid(self, camera: Camera) -> None:
        key = (camera.fx, camera.fy, camera.cx, camera.cy, camera.H, camera.W)
        if self._camera_grid_key == key:
            return
        self._x_norm_full, self._y_norm_full = pixel_grid_normalized(camera)
        self._camera_grid_key = key
        # Intrinsics changed -> any cached target state is stale.
        self._cached_target_id = None

    def _preprocess(self, rgb: NDArray[np.float32]) -> NDArray[np.float32]:
        gray = image_to_gray(rgb)
        if self.use_gzn:
            return gzn_transform(gray, self.sigma_blur)
        return gaussian_blur(gray, self.sigma_blur)

    def _build_target_cache(self, target: NDArray[np.float32]) -> None:
        if self.target_camera is None:
            raise RuntimeError(
                "PhotometricController requires a target_camera (pass via "
                "constructor or call set_target_camera(target_camera) before "
                "run_servo_loop)."
            )
        if self._x_norm_full is None:
            raise RuntimeError(
                "Camera grid not initialised; _ensure_camera_grid must be "
                "called before _build_target_cache."
            )

        I_star_raw = image_to_gray(target)
        I_star = self._preprocess(target)
        if I_star_raw.shape != I_star.shape:
            raise RuntimeError(
                f"raw target shape {I_star_raw.shape} != feature image shape {I_star.shape}"
            )
        gx_star, gy_star = image_gradient(I_star)
        Z_star = self._depth(target, camera=self.target_camera)
        if Z_star.shape != I_star.shape:
            raise RuntimeError(
                f"depth shape {Z_star.shape} != target image shape {I_star.shape}"
            )

        # In GZN-space sat bounds are meaningless; disable for the mask.
        if self.use_gzn:
            sat_lo, sat_hi = float("-inf"), float("inf")
        else:
            sat_lo, sat_hi = self.sat_lo, self.sat_hi

        mask = build_pixel_mask(
            I_star, gx_star, gy_star, Z_star,
            grad_percentile=self.grad_percentile,
            sat_lo=sat_lo,
            sat_hi=sat_hi,
            min_depth=self.min_depth,
        )
        mask_flat = mask.reshape(-1)
        idx = np.flatnonzero(mask_flat).astype(np.int64)
        self._num_total_valid = int(idx.size)

        if self.max_pixels and idx.size > self.max_pixels:
            sel = self._rng.choice(idx.size, size=self.max_pixels, replace=False)
            idx = np.sort(idx[sel])

        flat = lambda a: a.reshape(-1)
        self._I_star_flat = flat(I_star)[idx].astype(np.float32, copy=False)
        self._I_star_raw_flat = flat(I_star_raw)[idx].astype(np.float32, copy=False)
        self._Z_star_flat_masked = flat(Z_star)[idx].astype(np.float32, copy=False)
        self._xn_flat_masked = flat(self._x_norm_full)[idx].astype(np.float32, copy=False)
        self._yn_flat_masked = flat(self._y_norm_full)[idx].astype(np.float32, copy=False)
        self._mask_idx = idx
        self._cached_target_id = id(target)

    def _compute_residual_and_L(self, rendered: NDArray[np.float32]) -> Tuple[NDArray[np.float32], NDArray[np.float32], NDArray[np.float32], NDArray[np.float32]]:
        I_cur_raw = image_to_gray(rendered)
        I_cur = self._preprocess(rendered)
        if I_cur_raw.shape != I_cur.shape:
            raise RuntimeError(
                f"raw current shape {I_cur_raw.shape} != feature image shape {I_cur.shape}"
            )
        gx_cur, gy_cur = image_gradient(I_cur)

        idx = self._mask_idx
        if idx is None or idx.size == 0:
            empty = np.zeros((0,), dtype=np.float32)
            return empty, np.zeros((0, 6), dtype=np.float32), empty, empty

        I_cur_m = I_cur.reshape(-1)[idx]
        I_cur_raw_m = I_cur_raw.reshape(-1)[idx]
        gx_m = gx_cur.reshape(-1)[idx]
        gy_m = gy_cur.reshape(-1)[idx]

        e = (I_cur_m - self._I_star_flat).astype(np.float32, copy=False)
        raw_error = (I_cur_raw_m - self._I_star_raw_flat).astype(
            np.float32,
            copy=False,
        )
        L = photometric_interaction(
            gx_m, gy_m,
            self._xn_flat_masked, self._yn_flat_masked,
            self._Z_star_flat_masked,
        )
        return e, L, self._Z_star_flat_masked, raw_error
