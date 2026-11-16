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
            [value.float() for value in inputs], grad.contiguous(), ctx.heads
        )
        return None, *[grad.to(value.dtype) for grad, value in zip(gradients, inputs)]


def readout(x, r, k, v, scale, weight, bias, gate, heads):
    # Parameters are converted in registers inside the CUDA forward.
    xs = [t.contiguous() for t in (x, r, k, v, scale.flatten(), weight, bias, gate)]
    return _Readout.apply(heads, *xs)
