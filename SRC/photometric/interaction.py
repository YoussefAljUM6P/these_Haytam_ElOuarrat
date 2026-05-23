"""Interaction matrices (torch).

`point_interaction_matrix` is the classic IBVS 2x6 L_x derived from the
projection model (Chaumette & Hutchinson 2006, eq. 11). Used by both feature-
point IBVS and photometric VS (the latter contracts L_x against image
gradients to obtain L_I).

`luminance_interaction` mirrors ViSP `vpFeatureLuminance::interaction` and
returns one row of L per pixel:

    L_I_i = -[Ix_i, Iy_i] @ L_x(x_i, y_i, Z_i)
"""

from __future__ import annotations
from typing import Optional, Union, Tuple, List, Dict, Callable, Any
from numpy.typing import NDArray
import numpy as np

import torch


def point_interaction_matrix(x: Union[torch.Tensor, NDArray[Any]], y: Union[torch.Tensor, NDArray[Any]], z: Union[torch.Tensor, NDArray[Any]]) -> torch.Tensor:
    """Per-point IBVS interaction matrix.

    Inputs are 1-D tensors of normalized image coords (x, y) and depth Z.
    Returns (N, 2, 6) where rows 0/1 are the dx/dy interaction rows.
    """
    x = torch.as_tensor(x).flatten().to(torch.float32)
    y = torch.as_tensor(y).flatten().to(torch.float32)
    z = torch.as_tensor(z).flatten().to(torch.float32)
    if not (x.shape == y.shape == z.shape):
        raise ValueError(
            f"shape mismatch x={tuple(x.shape)} y={tuple(y.shape)} z={tuple(z.shape)}"
        )

    inv_z = 1.0 / z
    zero = torch.zeros_like(x)
    one = torch.ones_like(x)

    row_x = torch.stack(
        [-inv_z, zero, x * inv_z, x * y, -(one + x * x), y],
        dim=1,
    )
    row_y = torch.stack(
        [zero, -inv_z, y * inv_z, one + y * y, -x * y, -x],
        dim=1,
    )
    L = torch.stack([row_x, row_y], dim=1)  # (N, 2, 6)
    return L


def luminance_interaction(gx: Union[torch.Tensor, NDArray[Any]], gy: Union[torch.Tensor, NDArray[Any]], x: Union[torch.Tensor, NDArray[Any]], y: Union[torch.Tensor, NDArray[Any]], z: Union[torch.Tensor, NDArray[Any]]) -> torch.Tensor:
    """Per-pixel luminance interaction matrix L_I_i = -[gx gy] L_x.

    All inputs are 1-D tensors of equal length. Returns (N, 6) float32.
    """
    gx = torch.as_tensor(gx).flatten().to(torch.float32)
    gy = torch.as_tensor(gy).flatten().to(torch.float32)
    Lx = point_interaction_matrix(x, y, z)  # (N, 2, 6)
    grads = torch.stack([gx, gy], dim=1)  # (N, 2)
    L = -(grads.unsqueeze(-1) * Lx).sum(dim=1)  # (N, 6)
    return L
