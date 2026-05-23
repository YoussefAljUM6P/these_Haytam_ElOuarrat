"""ViSP `vpFeatureLuminance` port (torch).

A FeatureLuminance is built once against the desired image + per-pixel depth at
the desired pose. It caches the normalized image grid (x_n, y_n), the masked
pixel indices, the desired-image intensities I*, and the depth Z at those
pixels. At each iteration `buildFrom(image)` loads the current intensities I
and gradients Ix, Iy, and `interaction()` returns the (N, 6) row-stacked
luminance interaction matrix.
"""

from __future__ import annotations
from typing import Optional, Union, Tuple, List, Dict, Callable, Any

import numpy as np
from numpy.typing import NDArray
import torch

from camera import Camera

from .filter import (
    border_mask,
    derivative_filter_x,
    derivative_filter_y,
    gaussian_blur,
    rgb_to_gray,
)
from .interaction import luminance_interaction


def _gzn(gray: torch.Tensor, sigma: float) -> torch.Tensor:
    """Gaussian + zero-mean normalization (Rodriguez Sensors 2020)."""
    blurred = gaussian_blur(gray, sigma)
    mean = blurred.mean()
    std = blurred.std().clamp(min=1e-8)
    return (blurred - mean) / std


def _preprocess(image: Union[torch.Tensor, NDArray[np.float32]], sigma_blur: float, use_gzn: bool) -> torch.Tensor:
    gray = rgb_to_gray(image)
    if use_gzn:
        return _gzn(gray, sigma_blur)
    return gaussian_blur(gray, sigma_blur)


def _pixel_grid_normalized(H: int, W: int, fx: float, fy: float, cx: float, cy: float, device: torch.device, dtype: torch.dtype = torch.float32) -> Tuple[torch.Tensor, torch.Tensor]:
    u = torch.arange(W, device=device, dtype=dtype)
    v = torch.arange(H, device=device, dtype=dtype)
    vv, uu = torch.meshgrid(v, u, indexing="ij")
    xn = (uu - cx) / fx
    yn = (vv - cy) / fy
    return xn, yn


class FeatureLuminance:
    """Cached desired-image side of photometric VS.

    Parameters mirror `vpFeatureLuminance::init`. `bord` is the number of
    outermost pixels to discard (ViSP default = 10) so derivative artefacts
    don't pollute the residual.
    """

    def __init__(
        self,
        camera: Camera,
        target_image: Union[torch.Tensor, NDArray[np.float32]],
        depth_star: Union[torch.Tensor, NDArray[np.float32]],
        sigma_blur: float = 1.0,
        use_gzn: bool = True,
        bord: int = 10,
        grad_percentile: float = 0.0,
        sat_lo: float = float("-inf"),
        sat_hi: float = float("inf"),
        min_depth: float = 1e-4,
        max_pixels: int = 0,
        device: Optional[Union[str, torch.device]] = None,
        seed: int = 0,
    ) -> None:
        self.device = torch.device(device) if device is not None else torch.device("cpu")
        self.sigma_blur = float(sigma_blur)
        self.use_gzn = bool(use_gzn)
        self.bord = int(bord)
        self.grad_percentile = float(grad_percentile)
        self.sat_lo = float(sat_lo)
        self.sat_hi = float(sat_hi)
        self.min_depth = float(min_depth)
        self.max_pixels = int(max_pixels) if max_pixels else 0
        self._rng = torch.Generator(device="cpu")
        self._rng.manual_seed(int(seed))

        I_star = _preprocess(target_image, self.sigma_blur, self.use_gzn).to(self.device)
        gx_star = derivative_filter_x(I_star)
        gy_star = derivative_filter_y(I_star)
        Z_star = torch.as_tensor(depth_star, dtype=torch.float32, device=self.device)
        if Z_star.shape != I_star.shape:
            raise ValueError(
                f"depth_star shape {tuple(Z_star.shape)} != image shape {tuple(I_star.shape)}"
            )

        H, W = I_star.shape
        xn_full, yn_full = _pixel_grid_normalized(
            H, W, camera.fx, camera.fy, camera.cx, camera.cy, self.device
        )

        bmask = border_mask(H, W, self.bord, device=self.device)
        valid = bmask & torch.isfinite(Z_star) & (Z_star > self.min_depth)
        valid &= torch.isfinite(I_star)
        if torch.isfinite(torch.tensor(self.sat_lo)):
            valid &= I_star >= self.sat_lo
        if torch.isfinite(torch.tensor(self.sat_hi)):
            valid &= I_star <= self.sat_hi

        grad_mag = torch.sqrt(gx_star * gx_star + gy_star * gy_star)
        if self.grad_percentile > 0.0 and valid.any():
            thresh = torch.quantile(
                grad_mag[valid].to(torch.float32),
                self.grad_percentile / 100.0,
            )
            valid &= grad_mag >= thresh

        idx = torch.nonzero(valid.flatten(), as_tuple=False).flatten().to(torch.long)
        self._num_total_valid = int(idx.numel())

        if self.max_pixels and idx.numel() > self.max_pixels:
            perm = torch.randperm(idx.numel(), generator=self._rng)[: self.max_pixels]
            idx = torch.sort(idx[perm.to(idx.device)]).values

        flat = lambda a: a.reshape(-1)
        self._idx = idx
        self._I_star = flat(I_star)[idx].contiguous()
        self._Z_star = flat(Z_star)[idx].contiguous()
        self._xn = flat(xn_full)[idx].contiguous()
        self._yn = flat(yn_full)[idx].contiguous()

        self._I_cur = None
        self._gx_cur = None
        self._gy_cur = None
        self._shape = (H, W)

    @property
    def num_pixels(self) -> int:
        return int(self._idx.numel())

    @property
    def num_total_valid(self) -> int:
        return self._num_total_valid

    @property
    def shape(self) -> Tuple[int, int]:
        return self._shape

    @property
    def I_star(self) -> torch.Tensor:
        return self._I_star

    @property
    def Z_star(self) -> torch.Tensor:
        return self._Z_star

    def build_from(self, image: Union[torch.Tensor, NDArray[np.float32]]) -> torch.Tensor:
        I_cur = _preprocess(image, self.sigma_blur, self.use_gzn).to(self.device)
        if I_cur.shape != self._shape:
            raise ValueError(
                f"image shape {tuple(I_cur.shape)} != expected {self._shape}"
            )
        self._I_cur = I_cur.reshape(-1)[self._idx]
        self._gx_cur = derivative_filter_x(I_cur).reshape(-1)[self._idx]
        self._gy_cur = derivative_filter_y(I_cur).reshape(-1)[self._idx]
        return self._I_cur

    def error(self) -> torch.Tensor:
        if self._I_cur is None:
            raise RuntimeError("FeatureLuminance.error() called before build_from()")
        return self._I_cur - self._I_star

    def interaction(self) -> torch.Tensor:
        """L_I built from current Ix, Iy and cached x*, y*, Z* (desired-side depth)."""
        if self._gx_cur is None:
            raise RuntimeError("FeatureLuminance.interaction() called before build_from()")
        return luminance_interaction(
            self._gx_cur, self._gy_cur, self._xn, self._yn, self._Z_star
        )
