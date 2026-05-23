"""Depth-supervision losses for 3DGS + MoGe experiments.

All losses operate on rendered *inverse-depth* (the convention of the
gaussian-splatting fork's rasterizer) and a GT *inverse-depth* map.

Variants:
    none                  : zero (photometric-only baseline).
    l1_inv                : raw L1 on inverse depth, no alignment.
    l1_inv_aligned        : L1 on inverse depth after gaussian-splatting's
                            precomputed per-image scale/offset (in Camera.invdepthmap).
    l1_inv_solve_align    : same as above but resolves (scale, shift) every step
                            via closed-form LS against the current render
                            (per-view affine align, online).
    pearson_inv           : 1 - Pearson correlation on inverse depth (scale-free).
    pearson_depth         : 1 - Pearson correlation on depth (1 / invdepth, scale-free).
    masked_l1_inv_aligned : aligned L1 weighted by edge-aware kernel
                            (down-weight near depth discontinuities).
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _to_2d(x: torch.Tensor) -> torch.Tensor:
    """Squeeze any leading singleton dims to a [H, W] tensor."""
    while x.dim() > 2:
        x = x.squeeze(0)
    return x


def _valid_mask(invdepth_gt: torch.Tensor, depth_mask: torch.Tensor | None) -> torch.Tensor:
    """Build a finite, positive-invdepth mask, combined with the optional caller mask."""
    m = torch.isfinite(invdepth_gt) & (invdepth_gt > 0)
    if depth_mask is not None:
        m = m & (_to_2d(depth_mask) > 0)
    return m


def _solve_affine(pred: torch.Tensor, gt: torch.Tensor, mask: torch.Tensor) -> tuple[float, float]:
    """Closed-form least squares: argmin_{s, t} || s * gt + t - pred ||^2 over mask.

    Returns (s, t) as python floats (detached); used to align GT to current render
    before computing L1. Both inputs must be 2D, mask must be boolean.
    """
    if mask.sum() < 16:
        return 1.0, 0.0
    g = gt[mask].detach()
    p = pred[mask].detach()
    g_mean, p_mean = g.mean(), p.mean()
    gc, pc = g - g_mean, p - p_mean
    denom = (gc * gc).sum()
    if denom < 1e-8:
        return 1.0, 0.0
    s = (gc * pc).sum() / denom
    t = p_mean - s * g_mean
    return float(s.item()), float(t.item())


def _pearson(pred: torch.Tensor, gt: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """1 - Pearson correlation coefficient over masked pixels."""
    if mask.sum() < 16:
        return pred.new_zeros(())
    p = pred[mask]
    g = gt[mask]
    p = p - p.mean()
    g = g - g.mean()
    num = (p * g).sum()
    den = torch.sqrt((p * p).sum() * (g * g).sum() + 1e-8)
    return 1.0 - num / den


def _edge_weight(invdepth_gt: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """exp(-|grad invdepth|) — down-weight pixels near depth discontinuities.

    Returns a [H, W] tensor in [0, 1]; non-mask pixels set to 0.
    """
    d = invdepth_gt.clone()
    d[~mask] = 0.0
    dx = torch.zeros_like(d)
    dy = torch.zeros_like(d)
    dx[:, 1:] = (d[:, 1:] - d[:, :-1]).abs()
    dy[1:, :] = (d[1:, :] - d[:-1, :]).abs()
    grad = dx + dy
    if mask.sum() > 0:
        scale = grad[mask].mean().clamp_min(1e-6)
    else:
        scale = torch.tensor(1.0, device=d.device)
    w = torch.exp(-grad / scale)
    w = w * mask.float()
    return w


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def depth_loss(
    variant: str,
    rendered_invdepth: torch.Tensor,
    gt_invdepth: torch.Tensor | None,
    depth_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Compute a depth-supervision loss term.

    Args:
        variant: one of the variants listed at the top of this module.
        rendered_invdepth: [1, H, W] or [H, W] inverse depth from rasterizer.
        gt_invdepth: [1, H, W] or [H, W] GT inverse depth (already scale/offset
                     aligned via depth_params if using l1_inv_aligned).
                     Can be None when variant == "none".
        depth_mask: optional [1, H, W] binary mask of valid pixels.

    Returns:
        Scalar loss tensor (on rendered_invdepth.device).
    """
    if variant == "none" or gt_invdepth is None:
        return rendered_invdepth.new_zeros(())

    pred = _to_2d(rendered_invdepth)
    gt = _to_2d(gt_invdepth)
    if pred.shape != gt.shape:
        raise ValueError(f"shape mismatch: pred {pred.shape} vs gt {gt.shape}")
    mask = _valid_mask(gt, depth_mask)

    if variant == "l1_inv":
        if mask.sum() == 0:
            return pred.new_zeros(())
        return (pred - gt).abs()[mask].mean()

    if variant == "l1_inv_aligned":
        # Caller already applied depth_params (scale, offset); raw L1.
        if mask.sum() == 0:
            return pred.new_zeros(())
        return (pred - gt).abs()[mask].mean()

    if variant == "l1_inv_solve_align":
        s, t = _solve_affine(pred, gt, mask)
        if mask.sum() == 0:
            return pred.new_zeros(())
        gt_aligned = s * gt + t
        return (pred - gt_aligned).abs()[mask].mean()

    if variant == "pearson_inv":
        return _pearson(pred, gt, mask)

    if variant == "pearson_depth":
        eps = 1e-4
        pred_d = 1.0 / pred.clamp_min(eps)
        gt_d = 1.0 / gt.clamp_min(eps)
        return _pearson(pred_d, gt_d, mask)

    if variant == "masked_l1_inv_aligned":
        if mask.sum() == 0:
            return pred.new_zeros(())
        w = _edge_weight(gt, mask)
        diff = (pred - gt).abs()
        denom = w.sum().clamp_min(1.0)
        return (diff * w).sum() / denom

    raise ValueError(f"unknown depth-loss variant: {variant!r}")


VARIANTS = (
    "none",
    "l1_inv",
    "l1_inv_aligned",
    "l1_inv_solve_align",
    "pearson_inv",
    "pearson_depth",
    "masked_l1_inv_aligned",
)
