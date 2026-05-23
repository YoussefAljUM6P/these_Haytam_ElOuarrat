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

        self._prev_cost = None
        self.last_huber_k = None

    def solve(self, e, L):
        if e.numel() < 6 or L.shape[1] != 6:
            ssd = float(e.pow(2).sum().item()) if e.numel() else 0.0
            mse = ssd / max(int(e.numel()), 1)
            return torch.zeros(6, dtype=torch.float32, device=e.device), {
                "residual_norm": float(e.norm().item()) if e.numel() else 0.0,
                "residual_ssd": ssd,
                "residual_mse_per_px": mse,
                "cost": 0.5 * ssd,
                "mu": self.mu,
                "num_pixels": int(e.numel()),
                "huber_k": None,
                "method": self.method,
            }

        e = e.to(torch.float32)
        L = L.to(torch.float32)

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

        cost = 0.5 * float(e_w.pow(2).sum().item())

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
            "residual_norm": float(e.norm().item()),
            "residual_ssd": float(e.pow(2).sum().item()),
            "residual_mse_per_px": float(e.pow(2).mean().item()),
            "cost": cost,
            "mu": self.mu,
            "num_pixels": int(e.numel()),
            "huber_k": self.last_huber_k,
            "method": self.method,
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
