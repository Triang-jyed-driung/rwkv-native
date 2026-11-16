"""Scaled low-precision operands, FP32 accumulators and activation gradients."""

import torch

from .._extensions import extension
from .gradient_pack import combine, pack


def strict_mm(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    return extension("adjoints").mm(x, w)


def projection_backward(x, w, g, mode="store", *, need_dx=True, need_dw=True):
    if mode not in ("fp32", "store"):
        raise ValueError("adjoint_mode must be fp32 or store")
    if mode == "fp32" or not g.numel():
        dx = strict_mm(g, w.float().T) if need_dx else None
        dw = None
        if need_dw:
            dw = (
                strict_mm(x.float().T, g) if g.numel() else torch.zeros_like(w, dtype=torch.float32)
            )
        return dx, dw
    packed, repeated, coefficients, scale = pack(x.contiguous(), g.contiguous(), need_dw)
    parts = 2 if w.dtype == torch.float16 else 3
    dx = combine(strict_mm(packed, w.T), scale, parts) if need_dx else None
    dw = extension("adjoints").run(repeated, packed, coefficients) if need_dw else None
    return dx, dw
