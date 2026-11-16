"""Autograd bridge for a shared training/inference recurrence.

Inputs: six [B,T,H,64] tensors (BF16, FP16 or FP32); FP32 state [B,H,K,V] with K=V=64.
attention_mask: boolean [B,T]. False tokens neither read nor update the state.
Output: y in input dtype, final state in FP32. The input state is never mutated.
Backward replays checkpointed forward segments, avoiding unstable state inversion.
Only first-order differentiation is currently implemented.
"""

import torch
from torch.autograd.function import once_differentiable

from .._extensions import extension


class _WKV(torch.autograd.Function):
    @staticmethod
    def forward(ctx, r, w, k, v, a, b, state, mask, checkpoint_size):
        needs_backward = any(ctx.needs_input_grad[:7])
        y, final, checkpoints = extension("core").forward(
            r, w, k, v, a, b, state, mask, checkpoint_size, needs_backward
        )
        if needs_backward:
            ctx.save_for_backward(r, w, k, v, a, b, mask, checkpoints)
            ctx.checkpoint_size = checkpoint_size
        ctx.set_materialize_grads(False)
        return y, final

    @staticmethod
    @once_differentiable
    def backward(ctx, dy, ds):
        r, w, k, v, a, b, mask, checkpoints = ctx.saved_tensors
        if dy is None:
            dy = torch.zeros_like(r)
        if ds is None:
            ds = torch.empty(0, device=r.device, dtype=torch.float32)
        gradients = extension("core").backward(
            r,
            w,
            k,
            v,
            a,
            b,
            mask,
            checkpoints,
            dy.contiguous(),
            ds.contiguous(),
            ctx.checkpoint_size,
        )
        return (*gradients, None, None)


def wkv7(r, w, k, v, a, b, state=None, attention_mask=None, *, checkpoint_size=16):
    """RWKV-7 with FP32 [B,H,K,V] state; padding and mask holes are skipped."""
    if r.ndim != 4 or r.shape[-1] != 64 or min(r.shape) < 1:
        raise ValueError("inputs must have nonempty shape [B,T,H,64]")
    if not r.is_cuda or r.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise ValueError("inputs must be CUDA FP16, BF16 or FP32")
    for x in (w, k, v, a, b):
        if x.shape != r.shape or x.dtype != r.dtype or x.device != r.device:
            raise ValueError("all six inputs must share shape, dtype and device")
    B, _, H, D = r.shape
    if state is None:
        state = torch.zeros((B, H, D, D), device=r.device, dtype=torch.float32)
    if state.shape != (B, H, D, D) or state.dtype != torch.float32 or state.device != r.device:
        raise ValueError("state must be FP32 [B,H,64,64] on the input device")
    if attention_mask is None:
        attention_mask = torch.ones(r.shape[:2], device=r.device, dtype=torch.bool)
    if (
        attention_mask.shape != r.shape[:2]
        or attention_mask.dtype != torch.bool
        or attention_mask.device != r.device
    ):
        raise ValueError("attention_mask must be bool [B,T] on the input device")
    if not isinstance(checkpoint_size, int) or not 1 <= checkpoint_size <= 256:
        raise ValueError("checkpoint_size must be an integer between 1 and 256")
    return _WKV.apply(
        *(x.contiguous() for x in (r, w, k, v, a, b, state, attention_mask)), checkpoint_size
    )
