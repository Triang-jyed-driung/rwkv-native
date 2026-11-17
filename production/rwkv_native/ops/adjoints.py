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
    if x.dtype == torch.bfloat16 and g.shape[0] >= 4096:
        # Independently scale BF16 operands before FP16 conversion. GEMMs and
        # activation gradients remain FP32; convert the parameter gradient once.
        xh, gh, wh, factors = extension("scaled_half").pack(
            x.contiguous() if need_dw else x.new_empty(0), g.contiguous(),
            w if need_dx else w.new_empty(0))
        dx = strict_mm(gh, wh.T).mul_(factors[8]) if need_dx else None
        dw = (extension("scaled_half").dw(xh, gh, factors[6:8], g.new_empty(0)).to(w.dtype)
              if need_dw else None)
        return dx, dw
    packed, repeated, coefficients, scale = pack(x.contiguous(), g.contiguous(), need_dw)
    parts = 2
    dx = None
    if need_dx:
        if (g.shape[0] >= 4096 and x.shape[1] == 4*g.shape[1]
                and g.shape[1] in (2560,4096)):
            # Large FFN down adjoints otherwise allocate parts*M*K FP32
            # products plus another M*K output. Accumulate correction then high
            # in the original correction order into one FP32 matrix.
            rows = g.shape[0]
            dx = g.new_empty((rows,x.shape[1]))
            for index in reversed(range(parts)):
                extension("adjoints").mm_into(packed[index*rows:(index+1)*rows],w.T,dx,
                                               index != parts-1)
            dx.div_(scale)
        else:
            dx = combine(strict_mm(packed, w.T), scale, parts)
    dw = extension("adjoints").run(repeated, packed, coefficients) if need_dw else None
    return dx, dw
