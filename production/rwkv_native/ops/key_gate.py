"""Fused FP32 key normalization and adaptation, with a CUDA adjoint."""

import torch
from torch.autograd.function import once_differentiable

from .._extensions import extension


class _KeyGate(torch.autograd.Function):
    @staticmethod
    def forward(ctx, heads, *xs):
        ctx.heads = heads
        ctx.save_for_backward(*xs)
        ctx.set_materialize_grads(False)
        return tuple(extension("core").key_gate_forward(list(xs), heads))

    @staticmethod
    @once_differentiable
    def backward(ctx, *grads):
        xs = ctx.saved_tensors
        grads = [grad.contiguous() if grad is not None else xs[0].new_empty(0) for grad in grads]
        gradients = extension("core").key_gate_backward(list(xs), grads, ctx.heads)
        return None, *gradients


def key_gate(key, scale, bias, raw, adapt, heads):
    return _KeyGate.apply(
        heads,
        key.contiguous(),
        scale.flatten().contiguous(),
        bias.flatten().contiguous(),
        raw.contiguous(),
        adapt.flatten().contiguous(),
    )
