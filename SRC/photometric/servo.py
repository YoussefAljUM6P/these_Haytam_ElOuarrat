"""Servo control law: GN and LM, ported from ViSP `vpServo`.

ViSP `photometricVisualServoing.cpp` uses Levenberg-Marquardt:

    H        = L^T diag(w) L
    v        = -lambda * (H + mu * diag(H))^-1 L^T diag(w) e

with mu updated each step based on whether the cost decreased (`mu /= 2` on
success, `mu *= 10` on failure). For GN, set `mu = 0`.

Huber re-weighting (`w_i = 1` for |r| <= k else `k / |r|`) is optional and
mirrors `vpRobust::TUKEY/HUBER` style weighting at first order.
"""

from __future__ import annotations

import torch


DEFAULT_MIN_INTERACTION_RANK = 6
DEFAULT_MAX_INTERACTION_CONDITION = 1.0e8
INTERACTION_RANK_REL_TOL = 1.0e-12
INTERACTION_RANK_ABS_TOL = 1.0e-12


def validate_interaction_guard(min_rank=DEFAULT_MIN_INTERACTION_RANK, max_condition=DEFAULT_MAX_INTERACTION_CONDITION):
    min_rank = int(min_rank)
    if min_rank < 1 or min_rank > 6:
        raise ValueError("min_interaction_rank must be in [1, 6]")
    if max_condition is None:
        return min_rank, None
    max_condition = float(max_condition)
    if not torch.isfinite(torch.tensor(max_condition)) or max_condition <= 0.0:
        raise ValueError("max_interaction_condition must be positive or None")
    return min_rank, max_condition


def interaction_matrix_diagnostics(L, min_rank=DEFAULT_MIN_INTERACTION_RANK, max_condition=DEFAULT_MAX_INTERACTION_CONDITION):
    min_rank, max_condition = validate_interaction_guard(min_rank, max_condition)
    rows = int(L.shape[0]) if L.ndim >= 1 else 0
    cols = int(L.shape[1]) if L.ndim == 2 else 0
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

    if L.ndim != 2 or rows == 0 or cols == 0 or not torch.isfinite(L).all().item():
        return info

    try:
        singular_values = torch.linalg.svdvals(L.to(torch.float64))
    except RuntimeError:
        info["interaction_svd_failed"] = True
        return info

    if singular_values.numel() == 0:
        return info

    singular_values = singular_values.detach().cpu()
    max_singular = float(singular_values[0].item())
    rank_tolerance = max(
        INTERACTION_RANK_ABS_TOL,
        INTERACTION_RANK_REL_TOL * max_singular,
    )
    rank = int((singular_values > rank_tolerance).sum().item())
    required_index = int(min_rank) - 1
    min_required_singular = (
        float(singular_values[required_index].item())
        if required_index < singular_values.numel()
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


def interaction_fault_reason(info):
    if bool(info.get("interaction_svd_failed", False)):
        return "measurement_invalid_svd_failed"
    rank = int(info.get("interaction_rank", 0))
    min_rank = int(info.get("interaction_min_rank", DEFAULT_MIN_INTERACTION_RANK))
    if rank < min_rank:
        return "measurement_invalid_rank_deficient"
    max_condition = info.get("interaction_max_condition")
    condition = float(info.get("interaction_condition", float("inf")))
    if max_condition is not None and condition > float(max_condition):
        return "measurement_invalid_ill_conditioned"
    return None


def huber_weights(residuals, k=None):
    """Huber weights w_i = 1 if |r| <= k else k/|r|.

    If `k` is None, derive k = 1.345 * 1.4826 * MAD(r). Returns (w, k_used).
    """
    r = residuals.flatten().to(torch.float32)
    abs_r = r.abs()
    if k is None:
        med = r.median()
        mad = (r - med).abs().median()
        k_t = (1.345 * 1.4826) * torch.clamp(mad, min=1e-6)
    else:
        k_t = torch.tensor(float(k), dtype=r.dtype, device=r.device)
    w = torch.ones_like(abs_r)
    big = abs_r > k_t
    w = torch.where(big, k_t / abs_r.clamp(min=1e-12), w)
    return w, float(k_t.item() if torch.is_tensor(k_t) else k_t)


class PhotometricServo:
    """Photometric VS control law.

    `solve(e, L)` returns the 6-DOF velocity for residual `e` (N,) and
    luminance interaction matrix `L` (N, 6).

    Parameters:
        gain        : positive scalar -> v = -gain * step
        damping     : Tikhonov base damping added to diag(H) regardless of mu
        method      : "gn" (mu=0 always) or "lm" (adaptive mu).
        mu_init     : starting Marquardt mu
        mu_dec      : factor applied to mu after a successful step (mu *= mu_dec)
        mu_inc      : factor applied to mu after a failed step (mu *= mu_inc)
        use_huber   : apply Huber re-weighting before forming the normal equations
        huber_k     : fixed k for Huber; None -> auto-MAD
    """

    def __init__(
        self,
        gain=0.5,
        damping=1e-9,
        method="lm",
        mu_init=0.01,
        mu_dec=0.5,
        mu_inc=10.0,
        mu_min=1e-9,
        mu_max=1e6,
        use_huber=True,
        huber_k=None,
        min_interaction_rank=DEFAULT_MIN_INTERACTION_RANK,
        max_interaction_condition=DEFAULT_MAX_INTERACTION_CONDITION,
    ):
        if method not in ("gn", "lm"):
            raise ValueError(f"method must be 'gn' or 'lm', got {method!r}")
        self.gain = float(gain)
        self.damping = float(damping)
        self.method = method
        self.mu = float(mu_init)
        self.mu_dec = float(mu_dec)
        self.mu_inc = float(mu_inc)
        self.mu_min = float(mu_min)
        self.mu_max = float(mu_max)
        self.use_huber = bool(use_huber)
        self.huber_k = None if huber_k is None else float(huber_k)
        (
            self.min_interaction_rank,
            self.max_interaction_condition,
        ) = validate_interaction_guard(
            min_interaction_rank,
            max_interaction_condition,
        )

        self._prev_cost = None
        self.last_huber_k = None

    def solve(self, e, L):
        e = e.to(torch.float32).flatten()
        L = L.to(torch.float32)

        fault_reason = None
        if L.ndim != 2 or L.shape[1] != 6:
            fault_reason = "measurement_invalid_interaction_shape"
        elif e.numel() < 6:
            fault_reason = "measurement_invalid_not_enough_pixels"
        elif L.shape[0] != e.numel():
            fault_reason = "measurement_invalid_length_mismatch"
        elif not torch.isfinite(e).all() or not torch.isfinite(L).all():
            fault_reason = "measurement_invalid_nonfinite"

        if fault_reason is not None:
            ssd = float(e.pow(2).sum().item()) if e.numel() else float("nan")
            mse = ssd / int(e.numel()) if e.numel() else float("nan")
            residual_norm = float(e.norm().item()) if e.numel() else float("nan")
            interaction_info = interaction_matrix_diagnostics(
                L,
                self.min_interaction_rank,
                self.max_interaction_condition,
            )
            return torch.zeros(6, dtype=torch.float32, device=e.device), {
                "residual_norm": residual_norm,
                "residual_ssd": ssd,
                "residual_mse_per_px": mse,
                "control_residual_norm": residual_norm,
                "control_residual_ssd": ssd,
                "control_residual_mse_per_px": mse,
                "weighted_residual_norm": float("nan"),
                "weighted_residual_ssd": float("nan"),
                "weighted_residual_mse_per_px": float("nan"),
                "cost": 0.5 * ssd if e.numel() else float("nan"),
                "mu": self.mu,
                "num_pixels": int(e.numel()),
                "huber_k": None,
                "method": self.method,
                "measurement_valid": False,
                "fault_reason": fault_reason,
                **interaction_info,
            }

        if self.use_huber:
            w, k_used = huber_weights(e, k=self.huber_k)
            self.last_huber_k = k_used
            sqrt_w = torch.sqrt(w)
            e_w = e * sqrt_w
            L_w = L * sqrt_w.unsqueeze(1)
        else:
            self.last_huber_k = None
            e_w = e
            L_w = L

        weighted_residual_norm = float(e_w.norm().item())
        weighted_residual_ssd = float(e_w.pow(2).sum().item())
        weighted_residual_mse = float(e_w.pow(2).mean().item())
        cost = 0.5 * weighted_residual_ssd

        interaction_info = interaction_matrix_diagnostics(
            L_w,
            self.min_interaction_rank,
            self.max_interaction_condition,
        )
        fault_reason = interaction_fault_reason(interaction_info)

        control_residual_norm = float(e.norm().item())
        control_residual_ssd = float(e.pow(2).sum().item())
        control_residual_mse = float(e.pow(2).mean().item())

        if fault_reason is not None:
            return torch.zeros(6, dtype=torch.float32, device=e.device), {
                "residual_norm": control_residual_norm,
                "residual_ssd": control_residual_ssd,
                "residual_mse_per_px": control_residual_mse,
                "control_residual_norm": control_residual_norm,
                "control_residual_ssd": control_residual_ssd,
                "control_residual_mse_per_px": control_residual_mse,
                "weighted_residual_norm": weighted_residual_norm,
                "weighted_residual_ssd": weighted_residual_ssd,
                "weighted_residual_mse_per_px": weighted_residual_mse,
                "cost": cost,
                "mu": self.mu,
                "num_pixels": int(e.numel()),
                "huber_k": self.last_huber_k,
                "method": self.method,
                "measurement_valid": False,
                "fault_reason": fault_reason,
                **interaction_info,
            }

        H = L_w.t() @ L_w
        b = L_w.t() @ e_w

        if self.method == "lm":
            self._update_mu(cost)
            damping = self.damping + self.mu
        else:
            damping = self.damping

        diag_H = torch.diagonal(H)
        H_reg = H + damping * torch.diag(diag_H + 1e-12)

        try:
            step = torch.linalg.solve(H_reg, b)
        except RuntimeError:
            step = torch.linalg.pinv(L_w) @ e_w

        velocity = (-self.gain * step).to(torch.float32)

        info = {
            "residual_norm": control_residual_norm,
            "residual_ssd": control_residual_ssd,
            "residual_mse_per_px": control_residual_mse,
            "control_residual_norm": control_residual_norm,
            "control_residual_ssd": control_residual_ssd,
            "control_residual_mse_per_px": control_residual_mse,
            "weighted_residual_norm": weighted_residual_norm,
            "weighted_residual_ssd": weighted_residual_ssd,
            "weighted_residual_mse_per_px": weighted_residual_mse,
            "cost": cost,
            "mu": self.mu,
            "num_pixels": int(e.numel()),
            "huber_k": self.last_huber_k,
            "method": self.method,
            "measurement_valid": True,
            "fault_reason": None,
            **interaction_info,
        }
        return velocity, info

    def _update_mu(self, cost):
        if self._prev_cost is None:
            self._prev_cost = cost
            return
        if cost < self._prev_cost:
            self.mu = max(self.mu_min, self.mu * self.mu_dec)
        else:
            self.mu = min(self.mu_max, self.mu * self.mu_inc)
        self._prev_cost = cost

    def reset(self):
        self._prev_cost = None
        self.last_huber_k = None
