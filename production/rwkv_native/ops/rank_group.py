"""Grouped low-rank expansion with real FP16 operands and FP32 adjoints."""

import torch
from torch.autograd.function import once_differentiable

from .._extensions import extension
from .adjoints import projection_backward
from .normalization import SIGMOID, TANH, activation_backward


class _Expand(torch.autograd.Function):
    @staticmethod
    def forward(ctx, mode, *xs):
        count = len(xs) // 2
        raw = xs[:count]
        weights = xs[count:]
        kernel = extension("rank_group")
        packed = kernel.pack([value.contiguous() for value in raw])
        operands = [value.reshape(-1, value.shape[-1]) for value in packed]
        flat = kernel.run(
            operands,
            [weight.T.contiguous() for weight in weights],
            16 if operands[0].shape[0] <= 16 else 64,
        )
        outputs = flat.reshape(count, flat.numel() // count).unbind(0)
        ctx.save_for_backward(*packed, raw[0], raw[2], *weights)
        ctx.count = count
        ctx.mode = mode
        ctx.shapes = [x.shape for x in raw]
        ctx.set_materialize_grads(False)
        return tuple(y.view(*x.shape[:-1], w.shape[1]) for x, w, y in zip(raw, weights, outputs))

    @staticmethod
    @once_differentiable
    def backward(ctx, *grads):
        count = ctx.count
        saved = ctx.saved_tensors
        packed = saved[:count]
        decay_input, gate_input = saved[count : count + 2]
        weights = saved[count + 2 :]
        dxs = []
        dws = []
        for i, (x, w, g) in enumerate(zip(packed, weights, grads)):
            if g is None:
                dxs.append(None)
                dws.append(None)
                continue
            dx, dw = projection_backward(
                x.reshape(-1, x.shape[-1]), w, g.reshape(-1, w.shape[1]).contiguous(), ctx.mode
            )
            dx = dx.view(ctx.shapes[i])
            if i in (0, 2):
                dx = activation_backward(
                    decay_input if i == 0 else gate_input, dx, TANH if i == 0 else SIGMOID
                )
            dxs.append(dx)
            dws.append(dw.to(w.dtype))
        return (None, *dxs, *dws)


def expand_low_rank(raw, weights, mode="store"):
    return _Expand.apply(mode, *raw, *weights)
