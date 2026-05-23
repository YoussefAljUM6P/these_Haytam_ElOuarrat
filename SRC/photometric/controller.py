"""SERVIS-compatible callable wrapping `FeatureLuminance` + `PhotometricServo`.

Shape matches `controllers.PhotometricController` so it plugs into
`servo.run_servo_loop` without changes elsewhere. Differences vs the NumPy
controller:

  * end-to-end torch (CPU or GPU);
  * ViSP 7-tap derivative kernel via `filter.derivative_filter_x/y`;
  * Levenberg-Marquardt control law (matches ViSP example) instead of GN-only.
"""

from __future__ import annotations
from typing import Optional, Union, Tuple, List, Dict, Callable, Any

import numpy as np
from numpy.typing import NDArray
import torch

from depth import get_depth
from camera import Camera


from .feature import FeatureLuminance
from .servo import PhotometricServo


def _to_numpy(t: Union[torch.Tensor, NDArray[Any]]) -> NDArray[Any]:
    if isinstance(t, torch.Tensor):
        return t.detach().cpu().numpy()
    return np.asarray(t)


class PhotometricControllerTorch:
    """Direct photometric VS, torch-backed port of ViSP `vpFeatureLuminance`.

    The desired-side state (I*, Z*, grid, mask) is cached on the first call
    (or when `target` changes identity) inside a `FeatureLuminance`. The
    `PhotometricServo` instance owns the LM state.
    """

    def __init__(
        self,
        scene: Any,
        target_camera: Optional[Camera] = None,
        gain: float = 0.5,
        sigma_blur: float = 1.0,
        use_gzn: bool = True,
        bord: int = 10,
        grad_percentile: float = 0.0,
        sat_lo: float = 0.02,
        sat_hi: float = 0.98,
        min_depth: float = 1e-4,
        max_pixels: int = 50_000,
        use_huber: bool = True,
        huber_k: Optional[float] = None,
        use_intrinsic_depth: bool = True,
        depth_provider: Optional[Callable[[NDArray[np.float32]], NDArray[np.float32]]] = None,
        method: str = "lm",
        mu_init: float = 0.01,
        damping: float = 1e-9,
        device: Optional[str] = None,
        seed: int = 0,
        stop_mse_per_px: float = 2.0e-6,
    ) -> None:
        if scene is None and depth_provider is None:
            raise ValueError(
                "PhotometricControllerTorch needs `scene` (with render_depth) "
                "or an explicit `depth_provider`."
            )
        self.scene = scene
        self.target_camera = target_camera
        self.sigma_blur = float(sigma_blur)
        self.use_gzn = bool(use_gzn)
        self.bord = int(bord)
        self.grad_percentile = float(grad_percentile)
        self.sat_lo = float(sat_lo)
        self.sat_hi = float(sat_hi)
        self.min_depth = float(min_depth)
        self.max_pixels = int(max_pixels) if max_pixels else 0
        self.use_intrinsic_depth = bool(use_intrinsic_depth)
        self.depth_provider = depth_provider
        self.seed = int(seed)
        self.device = torch.device(device) if device is not None else torch.device("cpu")
        # Stop when mean(e^2) per pixel falls below this threshold (ViSP-style
        # stop, expressed resolution-free on whatever intensity scale the
        # feature uses; for [0,1] floats the ViSP equivalent of `SSD>10000` on
        # [0,255] over 320x240 is ~2e-6).
        self.stop_mse_per_px = float(stop_mse_per_px)

        self.servo = PhotometricServo(
            gain=gain,
            damping=damping,
            method=method,
            mu_init=mu_init,
            use_huber=use_huber,
            huber_k=huber_k,
        )

        self._feature = None
        self._cached_target_id = None
        self.last_info = {}
        self.last_visualization = {}

    # -- public API --------------------------------------------------------

    def should_stop(self) -> Optional[str]:
        """ViSP-style stop: residual mean square per pixel <= threshold."""
        mse = self.last_info.get("residual_mse_per_px")
        if mse is None or not np.isfinite(float(mse)):
            return None
        if float(mse) <= self.stop_mse_per_px:
            return "photometric_mse_below_threshold"
        return None

    def set_target_camera(self, camera: Camera) -> None:
        self.target_camera = camera
        self._cached_target_id = None
        self._feature = None
        self.servo.reset()

    def __call__(self, rendered: NDArray[np.float32], target: NDArray[np.float32], camera: Camera, iteration: int) -> NDArray[np.float32]:
        if self._cached_target_id != id(target) or self._feature is None:
            self._build_feature(target)

        feature = self._feature
        feature.build_from(rendered)
        e = feature.error()
        L = feature.interaction()
        velocity_t, solver_info = self.servo.solve(e, L)
        velocity = _to_numpy(velocity_t).astype(np.float32)

        depths = feature.Z_star
        n_used = feature.num_pixels
        if depths.numel():
            mean_d = float(depths.mean().item())
            min_d = float(depths.min().item())
            max_d = float(depths.max().item())
        else:
            mean_d = min_d = max_d = float("nan")

        self.last_info = {
            "iteration": int(iteration),
            "feature_mode": "photometric",
            "controller": "photometric_torch",
            "num_raw_matches": n_used,
            "num_inlier_matches": n_used,
            "num_cached_features": n_used,
            "num_dropped_features": int(feature.num_total_valid - n_used),
            "residual_norm": solver_info["residual_norm"],
            "residual_ssd": solver_info["residual_ssd"],
            "residual_mse_per_px": solver_info["residual_mse_per_px"],
            "velocity_norm": float(np.linalg.norm(velocity)),
            "mean_depth_m": mean_d,
            "min_depth_m": min_d,
            "max_depth_m": max_d,
            "num_pixels_used": n_used,
            "use_gzn": self.use_gzn,
            "huber_k_active": solver_info["huber_k"],
            "lm_mu": solver_info["mu"],
            "lm_cost": solver_info["cost"],
            "method": solver_info["method"],
            "backend": "torch",
        }
        self.last_visualization = {
            "iteration": int(iteration),
            "feature_mode": "photometric",
            "num_pixels_used": n_used,
            "residual_norm": solver_info["residual_norm"],
        }
        return velocity

    # -- internals ---------------------------------------------------------

    def _build_feature(self, target: NDArray[np.float32]) -> None:
        if self.target_camera is None:
            raise RuntimeError(
                "PhotometricControllerTorch requires a target_camera (pass via "
                "constructor or call set_target_camera() before run_servo_loop)."
            )

        depth_np = self._depth(target, camera=self.target_camera)

        if self.use_gzn:
            sat_lo, sat_hi = float("-inf"), float("inf")
        else:
            sat_lo, sat_hi = self.sat_lo, self.sat_hi

        self._feature = FeatureLuminance(
            camera=self.target_camera,
            target_image=target,
            depth_star=depth_np,
            sigma_blur=self.sigma_blur,
            use_gzn=self.use_gzn,
            bord=self.bord,
            grad_percentile=self.grad_percentile,
            sat_lo=sat_lo,
            sat_hi=sat_hi,
            min_depth=self.min_depth,
            max_pixels=self.max_pixels,
            device=self.device,
            seed=self.seed,
        )
        self._cached_target_id = id(target)
        self.servo.reset()

    def _depth(self, image: NDArray[np.float32], camera: Optional[Camera] = None) -> NDArray[np.float32]:
        if self.depth_provider is not None:
            return np.asarray(self.depth_provider(image), dtype=np.float32)
        if self.use_intrinsic_depth:
            if self.scene is None:
                raise RuntimeError("scene required for intrinsic depth")
            return np.asarray(self.scene.render_depth(camera), dtype=np.float32)
        return get_depth(image, scene=self.scene, use_intrinsic=False)
