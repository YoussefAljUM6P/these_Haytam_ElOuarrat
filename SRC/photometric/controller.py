"""Photometric luminance visual servoing controller.

This is the SERVIS in-repo implementation of the dense photometric controller
from Collewet and Marchand's photometric visual servoing formulation and the
Gaussian Splatting PVVS loop:

    e       = I(r) - I*
    L_I(x) = -grad(I)^T L_x(x, y, Z)
    v       = -lambda (H + mu diag(H))^-1 L_I^T e

The returned twist is a camera/body-frame velocity compatible with
servo.apply_camera_velocity(), which right-multiplies T_world_cam.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from numpy.typing import NDArray

from camera import Camera
from controllers import (
    DEFAULT_MAX_INTERACTION_CONDITION,
    DEFAULT_MIN_INTERACTION_RANK,
    interaction_fault_reason,
    interaction_matrix_diagnostics,
    validate_interaction_guard,
)
from depth import get_depth


ArrayLike = Union[NDArray[Any], torch.Tensor]


def _image_device(image: ArrayLike, fallback: torch.device) -> torch.device:
    if isinstance(image, torch.Tensor):
        return image.device
    return fallback


def _as_tensor(image: ArrayLike, device: torch.device) -> torch.Tensor:
    if isinstance(image, torch.Tensor):
        return image.detach().to(device=device, dtype=torch.float32)
    return torch.as_tensor(np.asarray(image), dtype=torch.float32, device=device)


def _as_numpy(image: ArrayLike) -> NDArray[np.float32]:
    if isinstance(image, torch.Tensor):
        return image.detach().cpu().numpy().astype(np.float32, copy=False)
    return np.asarray(image, dtype=np.float32)


def _luminance(image: torch.Tensor) -> torch.Tensor:
    image = image.to(torch.float32)
    if image.ndim == 2:
        return image
    if image.ndim != 3:
        raise ValueError(f"image must be HxW or HxWxC, got shape {tuple(image.shape)}")
    if image.shape[-1] == 1:
        return image[..., 0]
    if image.shape[-1] >= 3:
        coeffs = torch.tensor(
            [0.299, 0.587, 0.114],
            dtype=image.dtype,
            device=image.device,
        )
        return (image[..., :3] * coeffs).sum(dim=-1)
    raise ValueError(f"unsupported image channel count in shape {tuple(image.shape)}")


def _gaussian_kernel1d(sigma: float, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    radius = max(1, int(round(3.0 * float(sigma))))
    x = torch.arange(-radius, radius + 1, dtype=dtype, device=device)
    kernel = torch.exp(-(x * x) / (2.0 * float(sigma) * float(sigma)))
    return kernel / kernel.sum().clamp_min(1e-12)


def _gaussian_blur(gray: torch.Tensor, sigma: float) -> torch.Tensor:
    if sigma <= 0.0:
        return gray
    if gray.shape[0] < 2 or gray.shape[1] < 2:
        return gray
    kernel = _gaussian_kernel1d(float(sigma), gray.device, gray.dtype)
    radius = int(kernel.numel() // 2)
    img = gray[None, None, :, :]
    kx = kernel.view(1, 1, 1, -1)
    ky = kernel.view(1, 1, -1, 1)
    img = F.pad(img, (radius, radius, 0, 0), mode="replicate")
    img = F.conv2d(img, kx)
    img = F.pad(img, (0, 0, radius, radius), mode="replicate")
    img = F.conv2d(img, ky)
    return img[0, 0]


def _image_gradients(gray: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    img = gray[None, None, :, :]
    kx = torch.tensor(
        [-0.5, 0.0, 0.5],
        dtype=gray.dtype,
        device=gray.device,
    ).view(1, 1, 1, 3)
    ky = torch.tensor(
        [-0.5, 0.0, 0.5],
        dtype=gray.dtype,
        device=gray.device,
    ).view(1, 1, 3, 1)
    grad_u = F.conv2d(F.pad(img, (1, 1, 0, 0), mode="replicate"), kx)[0, 0]
    grad_v = F.conv2d(F.pad(img, (0, 0, 1, 1), mode="replicate"), ky)[0, 0]
    return grad_u, grad_v


def _huber_weights(residuals: torch.Tensor, k: Optional[float]) -> Tuple[torch.Tensor, float]:
    r = residuals.flatten()
    abs_r = r.abs()
    if k is None:
        med = r.median()
        mad = (r - med).abs().median()
        k_t = (1.345 * 1.4826) * torch.clamp(mad, min=1e-6)
    else:
        k_t = torch.tensor(float(k), dtype=r.dtype, device=r.device)
    weights = torch.ones_like(abs_r)
    weights = torch.where(abs_r > k_t, k_t / abs_r.clamp_min(1e-12), weights)
    return weights, float(k_t.detach().cpu().item())


def _empty_interaction_info(
    min_rank: int,
    max_condition: Optional[float],
) -> Dict[str, Any]:
    return interaction_matrix_diagnostics(
        np.zeros((0, 6), dtype=np.float32),
        min_rank,
        max_condition,
    )


class PhotometricControllerTorch:
    """Dense photometric visual-servo controller using Torch math.

    The controller conforms to the SERVIS controller protocol used by
    servo.run_servo_loop. It accepts either NumPy images or HWC Torch tensors;
    when a scene exposes render_tensor(), the servo loop can keep the rendered
    image on the GPU and only the final 6D velocity is copied back to NumPy.
    """

    accepts_torch_tensors = True

    def __init__(
        self,
        scene: Optional[Any] = None,
        target_camera: Optional[Camera] = None,
        gain: float = 0.75,
        sigma_blur: float = 1.0,
        use_gzn: bool = True,
        grad_percentile: float = 50.0,
        max_pixels: int = 50_000,
        use_huber: bool = True,
        huber_k: Optional[float] = None,
        use_intrinsic_depth: bool = False,
        depth_provider: Optional[Callable[..., NDArray[np.float32]]] = None,
        method: str = "lm",
        damping: float = 1.0e-2,
        mu_init: Optional[float] = None,
        min_depth: float = 1.0e-4,
        min_pixels: Optional[int] = None,
        stop_mse_per_px: float = 2.0e-6,
        stop_ssd: Optional[float] = None,
        stop_plateau_iters: Optional[int] = None,
        stop_plateau_ratio: float = 0.99,
        min_interaction_rank: int = DEFAULT_MIN_INTERACTION_RANK,
        max_interaction_condition: Optional[float] = DEFAULT_MAX_INTERACTION_CONDITION,
        device: Optional[str] = None,
        **_: Any,
    ) -> None:
        if method not in {"lm", "gn"}:
            raise ValueError(f"method must be 'lm' or 'gn', got {method!r}")
        if max_pixels is not None and int(max_pixels) < 0:
            raise ValueError("max_pixels must be >= 0")
        if sigma_blur < 0.0:
            raise ValueError("sigma_blur must be >= 0")
        if not (0.0 <= float(grad_percentile) < 100.0):
            raise ValueError("grad_percentile must be in [0, 100)")

        self.scene = scene
        self.target_camera = target_camera
        self.gain = float(gain)
        self.sigma_blur = float(sigma_blur)
        self.use_gzn = bool(use_gzn)
        self.grad_percentile = float(grad_percentile)
        self.max_pixels = int(max_pixels or 0)
        self.use_huber = bool(use_huber)
        self.huber_k = None if huber_k is None else float(huber_k)
        self.use_intrinsic_depth = bool(use_intrinsic_depth)
        self.depth_provider = depth_provider
        self.method = method
        self.damping = float(mu_init if mu_init is not None else damping)
        self.min_depth = float(min_depth)
        self.stop_mse_per_px = float(stop_mse_per_px)
        self.stop_ssd_threshold = None if stop_ssd is None else float(stop_ssd)
        self.stop_plateau_iters = int(stop_plateau_iters) if stop_plateau_iters is not None else None
        self.stop_plateau_ratio = float(stop_plateau_ratio)
        self.best_mse = float("inf")
        self.plateau_count = 0
        (
            self.min_interaction_rank,
            self.max_interaction_condition,
        ) = validate_interaction_guard(
            min_interaction_rank,
            max_interaction_condition,
        )
        self.min_pixels = (
            max(6, int(self.min_interaction_rank))
            if min_pixels is None
            else int(min_pixels)
        )
        if self.min_pixels < 1:
            raise ValueError("min_pixels must be >= 1")

        self.device = torch.device(device) if device is not None else torch.device("cpu")
        self.last_info: Dict[str, Any] = {}
        self.last_visualization: Dict[str, Any] = {}

    def set_target_camera(self, camera: Camera) -> None:
        self.target_camera = camera

    def should_stop(self) -> Optional[str]:
        fault_reason = self.last_info.get("fault_reason")
        if fault_reason:
            return str(fault_reason)

        stop_ssd = self.last_info.get("stop_ssd")
        stop_threshold = self.last_info.get("stop_ssd_threshold")
        if (
            stop_threshold is not None
            and stop_ssd is not None
            and np.isfinite(float(stop_threshold))
            and np.isfinite(float(stop_ssd))
            and float(stop_ssd) <= float(stop_threshold)
        ):
            return "photometric_ssd_below_threshold"

        mse = self.last_info.get("raw_image_mse_per_px")
        if (
            mse is not None
            and np.isfinite(float(mse))
        ):
            mse = float(mse)
            if mse <= self.stop_mse_per_px:
                return "photometric_mse_below_threshold"

            if self.stop_plateau_iters is not None:
                if mse < self.best_mse * self.stop_plateau_ratio:
                    self.best_mse = mse
                    self.plateau_count = 0
                else:
                    self.plateau_count += 1
                    if self.plateau_count >= self.stop_plateau_iters:
                        return "plateau"
        return None

    def __call__(
        self,
        rendered: ArrayLike,
        target: ArrayLike,
        camera: Camera,
        iteration: int,
    ) -> NDArray[np.float32]:
        device = _image_device(rendered, self.device)
        current_raw = _luminance(_as_tensor(rendered, device)).clamp(0.0, 1.0)
        target_raw = _luminance(_as_tensor(target, device)).clamp(0.0, 1.0)
        if tuple(current_raw.shape) != tuple(target_raw.shape):
            raise ValueError(
                f"rendered and target luminance shapes differ: "
                f"{tuple(current_raw.shape)} vs {tuple(target_raw.shape)}"
            )
        if current_raw.shape != (int(camera.H), int(camera.W)):
            raise ValueError(
                f"image shape {tuple(current_raw.shape)} does not match camera "
                f"HxW {(int(camera.H), int(camera.W))}"
            )

        current = _gaussian_blur(current_raw, self.sigma_blur)
        target_lum = _gaussian_blur(target_raw, self.sigma_blur)
        depth = self._depth(rendered, camera, device)
        measurement = self._build_measurement(current, target_lum, current_raw, target_raw, depth, camera)
        velocity, solver_info = self._solve(measurement)

        raw_velocity = velocity.detach().cpu().numpy().astype(np.float32, copy=False)
        self._record_info(iteration, measurement, solver_info, raw_velocity)
        return raw_velocity

    def _depth(self, rendered: ArrayLike, camera: Camera, device: torch.device) -> torch.Tensor:
        if self.depth_provider is not None:
            try:
                depth_np = self.depth_provider(_as_numpy(rendered), camera)
            except TypeError:
                depth_np = self.depth_provider(_as_numpy(rendered))
        elif self.use_intrinsic_depth:
            render_depth = None if self.scene is None else getattr(self.scene, "render_depth", None)
            if not callable(render_depth):
                raise NotImplementedError("Intrinsic depth requires a scene with render_depth(camera)")
            depth_np = render_depth(camera)
        else:
            depth_np = get_depth(_as_numpy(rendered), scene=self.scene, use_intrinsic=False)

        depth = _as_tensor(depth_np, device)
        if depth.ndim != 2:
            raise ValueError(f"depth must be HxW, got shape {tuple(depth.shape)}")
        return depth

    def _build_measurement(
        self,
        current: torch.Tensor,
        target: torch.Tensor,
        current_raw: torch.Tensor,
        target_raw: torch.Tensor,
        depth: torch.Tensor,
        camera: Camera,
    ) -> Dict[str, Any]:
        H, W = current.shape
        if tuple(depth.shape) != (H, W):
            raise ValueError(f"depth shape {tuple(depth.shape)} does not match image {(H, W)}")

        grad_u, grad_v = _image_gradients(current)
        base_mask = (
            torch.isfinite(current)
            & torch.isfinite(target)
            & torch.isfinite(depth)
            & torch.isfinite(grad_u)
            & torch.isfinite(grad_v)
            & (depth > self.min_depth)
        )
        if H > 2 and W > 2:
            base_mask[0, :] = False
            base_mask[-1, :] = False
            base_mask[:, 0] = False
            base_mask[:, -1] = False

        control_current = current
        control_target = target
        grad_scale = torch.tensor(1.0, dtype=current.dtype, device=current.device)
        if self.use_gzn and torch.count_nonzero(base_mask).item() > 0:
            c_vals = current[base_mask]
            t_vals = target[base_mask]
            c_std = c_vals.std(unbiased=False).clamp_min(1e-6)
            t_std = t_vals.std(unbiased=False).clamp_min(1e-6)
            control_current = (current - c_vals.mean()) / c_std
            control_target = (target - t_vals.mean()) / t_std
            grad_scale = c_std

        residual = control_current - control_target
        raw_residual = current_raw - target_raw
        grad_x = (grad_u / grad_scale) * float(camera.fx)
        grad_y = (grad_v / grad_scale) * float(camera.fy)
        grad_mag = torch.sqrt(grad_x * grad_x + grad_y * grad_y)

        mask = base_mask & torch.isfinite(residual) & torch.isfinite(raw_residual)
        if torch.count_nonzero(mask).item() > 0:
            valid_grad = grad_mag[mask]
            threshold = torch.quantile(valid_grad, self.grad_percentile / 100.0)
            mask = mask & (grad_mag > threshold)
        else:
            threshold = torch.tensor(float("nan"), dtype=current.dtype, device=current.device)

        idx = torch.nonzero(mask.reshape(-1), as_tuple=False).flatten()
        num_total_valid = int(idx.numel())
        if self.max_pixels > 0 and idx.numel() > self.max_pixels:
            step = int(np.ceil(float(idx.numel()) / float(self.max_pixels)))
            idx = idx[::step][: self.max_pixels]

        if idx.numel() == 0:
            empty = torch.empty((0,), dtype=current.dtype, device=current.device)
            return {
                "e": empty,
                "raw_e": empty,
                "L": torch.empty((0, 6), dtype=current.dtype, device=current.device),
                "depths": empty,
                "grad_mag": empty,
                "num_total_valid": num_total_valid,
                "num_pixels_used": 0,
                "num_dropped_features": num_total_valid,
                "grad_threshold": float(threshold.detach().cpu().item()) if torch.isfinite(threshold) else float("nan"),
            }

        ys, xs = torch.meshgrid(
            torch.arange(H, dtype=current.dtype, device=current.device),
            torch.arange(W, dtype=current.dtype, device=current.device),
            indexing="ij",
        )
        x = ((xs.reshape(-1)[idx] - float(camera.cx)) / float(camera.fx)).to(current.dtype)
        y = ((ys.reshape(-1)[idx] - float(camera.cy)) / float(camera.fy)).to(current.dtype)
        z = depth.reshape(-1)[idx].to(current.dtype)
        ix = grad_x.reshape(-1)[idx].to(current.dtype)
        iy = grad_y.reshape(-1)[idx].to(current.dtype)
        e = residual.reshape(-1)[idx].to(current.dtype)
        raw_e = raw_residual.reshape(-1)[idx].to(current.dtype)

        inv_z = 1.0 / z.clamp_min(self.min_depth)
        zeros = torch.zeros_like(x)
        Lx0 = torch.stack([
            -inv_z,
            zeros,
            x * inv_z,
            x * y,
            -(1.0 + x * x),
            y,
        ], dim=1)
        Lx1 = torch.stack([
            zeros,
            -inv_z,
            y * inv_z,
            1.0 + y * y,
            -x * y,
            -x,
        ], dim=1)
        L = -(ix[:, None] * Lx0 + iy[:, None] * Lx1)

        return {
            "e": e,
            "raw_e": raw_e,
            "L": L,
            "depths": z,
            "grad_mag": grad_mag.reshape(-1)[idx],
            "num_total_valid": num_total_valid,
            "num_pixels_used": int(idx.numel()),
            "num_dropped_features": int(max(num_total_valid - int(idx.numel()), 0)),
            "grad_threshold": float(threshold.detach().cpu().item()) if torch.isfinite(threshold) else float("nan"),
        }

    def _solve(self, measurement: Dict[str, Any]) -> Tuple[torch.Tensor, Dict[str, Any]]:
        e = measurement["e"].flatten().to(torch.float64)
        raw_e = measurement["raw_e"].flatten().to(torch.float64)
        L = measurement["L"].to(torch.float64)
        device = L.device

        interaction_info = _empty_interaction_info(
            self.min_interaction_rank,
            self.max_interaction_condition,
        )
        fault_reason = None
        if e.numel() < self.min_pixels:
            fault_reason = "measurement_invalid_not_enough_pixels"
        elif L.ndim != 2 or L.shape != (e.numel(), 6):
            fault_reason = "measurement_invalid_interaction_shape"
        elif not torch.isfinite(e).all() or not torch.isfinite(raw_e).all() or not torch.isfinite(L).all():
            fault_reason = "measurement_invalid_nonfinite"

        if fault_reason is not None:
            return torch.zeros(6, dtype=torch.float32, device=device), self._solver_info(
                e,
                raw_e,
                e,
                L,
                interaction_info,
                fault_reason,
                None,
            )

        if self.use_huber:
            weights, huber_k = _huber_weights(e, self.huber_k)
            sqrt_w = torch.sqrt(weights.to(torch.float64))
            e_eff = e * sqrt_w
            L_eff = L * sqrt_w[:, None]
        else:
            huber_k = None
            e_eff = e
            L_eff = L

        interaction_info = interaction_matrix_diagnostics(
            L_eff.detach().cpu().numpy().astype(np.float64, copy=False),
            self.min_interaction_rank,
            self.max_interaction_condition,
        )
        fault_reason = interaction_fault_reason(interaction_info)
        if fault_reason is not None:
            return torch.zeros(6, dtype=torch.float32, device=device), self._solver_info(
                e,
                raw_e,
                e_eff,
                L_eff,
                interaction_info,
                fault_reason,
                huber_k,
            )

        H = L_eff.T @ L_eff
        g = L_eff.T @ e_eff
        if self.method == "gn":
            reg = 0.0
        else:
            reg = self.damping
        diag = torch.diagonal(H).abs().clamp_min(1.0e-12)
        H_reg = H + float(reg) * torch.diag(diag)
        H_reg = H_reg + 1.0e-12 * torch.eye(6, dtype=H_reg.dtype, device=device)
        try:
            try:
                step = torch.linalg.solve(H_reg, g)
            except RuntimeError:
                # The 6x6 normal-equations solve is trivial, but cuSOLVER can
                # still fail to initialise (CUSOLVER_STATUS_INTERNAL_ERROR on
                # cusolverDnCreate) when the GPU is memory-starved -- e.g. the
                # 6GB laptop card with a NeRF renderer resident. That is a
                # GPU-resource failure, not a singular system, so retry the tiny
                # solve on CPU instead of failing the whole task.
                step = torch.linalg.solve(H_reg.detach().cpu(), g.detach().cpu()).to(device)
            velocity = (-self.gain * step).to(torch.float32)
            if torch.isfinite(velocity).all():
                solve_fault = None
            else:
                velocity = torch.zeros(6, dtype=torch.float32, device=device)
                solve_fault = "measurement_invalid_nonfinite_velocity"
        except RuntimeError:
            # Even the CPU retry failed (genuinely pathological system); keep the
            # original cut-and-continue behaviour rather than aborting the run.
            velocity = torch.zeros(6, dtype=torch.float32, device=device)
            solve_fault = "measurement_invalid_solve_failed"

        return velocity, self._solver_info(
            e,
            raw_e,
            e_eff,
            L_eff,
            interaction_info,
            solve_fault,
            huber_k,
        )

    def _solver_info(
        self,
        e: torch.Tensor,
        raw_e: torch.Tensor,
        e_eff: torch.Tensor,
        L_eff: torch.Tensor,
        interaction_info: Dict[str, Any],
        fault_reason: Optional[str],
        huber_k: Optional[float],
    ) -> Dict[str, Any]:
        residual_ssd = float(e.pow(2).sum().detach().cpu().item()) if e.numel() else float("nan")
        residual_mse = float(e.pow(2).mean().detach().cpu().item()) if e.numel() else float("nan")
        raw_ssd = float(raw_e.pow(2).sum().detach().cpu().item()) if raw_e.numel() else float("nan")
        raw_mse = float(raw_e.pow(2).mean().detach().cpu().item()) if raw_e.numel() else float("nan")
        weighted_ssd = float(e_eff.pow(2).sum().detach().cpu().item()) if e_eff.numel() else float("nan")
        weighted_mse = float(e_eff.pow(2).mean().detach().cpu().item()) if e_eff.numel() else float("nan")
        stop_threshold = self._stop_ssd_threshold(int(raw_e.numel()))
        return {
            "measurement_valid": fault_reason is None,
            "fault_reason": fault_reason,
            "residual_norm": float(e.norm().detach().cpu().item()) if e.numel() else float("nan"),
            "residual_ssd": residual_ssd,
            "residual_mse_per_px": residual_mse,
            "weighted_residual_norm": float(e_eff.norm().detach().cpu().item()) if e_eff.numel() else float("nan"),
            "weighted_residual_ssd": weighted_ssd,
            "weighted_residual_mse_per_px": weighted_mse,
            "raw_image_ssd": raw_ssd,
            "raw_image_mse_per_px": raw_mse,
            "stop_ssd": raw_ssd,
            "stop_ssd_threshold": stop_threshold,
            "stop_mse_per_px_threshold": self.stop_mse_per_px,
            "huber_k": huber_k,
            "method": self.method,
            "lm_mu": 0.0 if self.method == "gn" else self.damping,
            "lm_cost": 0.5 * weighted_ssd if np.isfinite(weighted_ssd) else float("nan"),
            **interaction_info,
        }

    def _stop_ssd_threshold(self, n_pixels: int) -> float:
        if self.stop_ssd_threshold is not None:
            return float(self.stop_ssd_threshold)
        if n_pixels <= 0:
            return float("nan")
        return float(self.stop_mse_per_px) * float(n_pixels)

    def _record_info(
        self,
        iteration: int,
        measurement: Dict[str, Any],
        solver_info: Dict[str, Any],
        velocity: NDArray[np.float32],
    ) -> None:
        depths = measurement["depths"]
        grad_mag = measurement["grad_mag"]
        if depths.numel():
            mean_depth = float(depths.mean().detach().cpu().item())
            min_depth = float(depths.min().detach().cpu().item())
            max_depth = float(depths.max().detach().cpu().item())
        else:
            mean_depth = min_depth = max_depth = float("nan")
        if grad_mag.numel():
            mean_grad = float(grad_mag.mean().detach().cpu().item())
            max_grad = float(grad_mag.max().detach().cpu().item())
        else:
            mean_grad = max_grad = float("nan")

        n_used = int(measurement["num_pixels_used"])
        self.last_info = {
            "iteration": int(iteration),
            "feature_mode": "photometric",
            "controller": "photometric_torch",
            "backend": "torch",
            "num_raw_matches": int(measurement["num_total_valid"]),
            "num_inlier_matches": n_used,
            "num_cached_features": n_used,
            "num_dropped_features": int(measurement["num_dropped_features"]),
            "num_pixels_used": n_used,
            "mean_depth": mean_depth,
            "min_depth": min_depth,
            "max_depth": max_depth,
            "mean_gradient": mean_grad,
            "max_gradient": max_grad,
            "gradient_threshold": float(measurement["grad_threshold"]),
            "use_gzn": bool(self.use_gzn),
            "use_huber": bool(self.use_huber),
            "huber_k_active": solver_info.get("huber_k"),
            "velocity_norm": float(np.linalg.norm(velocity)),
            **solver_info,
        }
        self.last_visualization = {
            "iteration": int(iteration),
            "feature_mode": "photometric",
            "num_pixels_used": n_used,
            "residual_norm": solver_info.get("residual_norm"),
            "residual_ssd": solver_info.get("residual_ssd"),
            "raw_image_mse_per_px": solver_info.get("raw_image_mse_per_px"),
        }


PhotometricController = PhotometricControllerTorch
PhotometricControllerViSP = PhotometricControllerTorch

__all__ = [
    "PhotometricController",
    "PhotometricControllerTorch",
    "PhotometricControllerViSP",
]
