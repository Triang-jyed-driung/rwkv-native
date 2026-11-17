"""Fused FP32 readout and its CUDA adjoint."""

import torch
from torch.autograd.function import once_differentiable

from .._extensions import extension


class _Readout(torch.autograd.Function):
    @staticmethod
    def forward(ctx, heads, *xs):
        ctx.heads = heads
        ctx.save_for_backward(*xs)
        return extension("core").readout_forward(list(xs), heads)

    @staticmethod
    @once_differentiable
    def backward(ctx, grad):
        inputs = ctx.saved_tensors
        gradients = extension("core").readout_backward(
            list(inputs), grad.contiguous(), ctx.heads
        )
        return None, *gradients


def readout(x, r, k, v, scale, weight, bias, gate, heads, *, packed=False):
    # Parameters are converted in registers inside the CUDA forward.
    xs = [t.contiguous() for t in (x, r, k, v, scale.flatten(), weight, bias, gate)]
    if packed:
        if torch.is_grad_enabled():
            raise ValueError("packed readout requires gradients to be disabled")
        # Preserve FP32 normalization and gating, then store the exact operand
        # needed by the next projection without an intermediate FP32 buffer.
        return extension("core").readout_packed_forward(xs, heads)
    return _Readout.apply(heads, *xs)
