"""None-training MMA backward; forward and exposed state stay FP32.

Adapted from wkv7_fast (MIT; see third_party/LICENSE.wkv7_fast). The single
packing kernel makes BF16 vectors and FP16 checkpoints required by the MMA
backward. Output gradients are written directly in FP32.
"""
import torch
from torch.autograd.function import once_differentiable
from .._extensions import extension

class _MMAWKV(torch.autograd.Function):
    @staticmethod
    def forward(ctx, r, w, k, v, a, b, state, mask):
        xs = [x.contiguous() for x in (r, w, k, v, a, b)]
        from .wkv_training import forward
        y, final, cp = forward(xs,state,mask,16,True)
        ctx.save_for_backward(*xs, mask, cp)
        ctx.set_materialize_grads(False)
        return y, final

    @staticmethod
    @once_differentiable
    def backward(ctx, dy, ds):
        *xs, mask, cp = ctx.saved_tensors
        dy = torch.zeros_like(xs[0]) if dy is None else dy.contiguous()
        ds = xs[0].new_empty(0) if ds is None else ds.contiguous()
        return *extension("wkv_mma_backward").run(xs, mask.contiguous(), cp, dy, ds), None


