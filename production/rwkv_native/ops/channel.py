"""Fuse attention output projection, residual, normalization, shift and FFN key."""

import torch
from torch.autograd.function import once_differentiable

from .._extensions import extension
from .adjoints import projection_backward
from .gemm import project_rows
from .normalization import norm_shift_backward


class _ChannelFrontend(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        x,
        mixed,
        output_weight,
        previous,
        norm_weight,
        norm_bias,
        mix_coefficient,
        key_weight,
        mask,
        save,
        mode,
        backends,
    ):
        output_backend, key_backend = backends
        attention_operand = mixed.reshape(-1, mixed.shape[-1]).to(output_weight.dtype).contiguous()
        attention_output = project_rows(attention_operand, output_weight, output_backend)
        packed, normalized, state, residual = extension("frontend").run(
            [x, attention_output, previous, norm_weight, norm_bias, mix_coefficient], mask, save
        )
        key = project_rows(packed, key_weight, key_backend).view(*x.shape[:2], key_weight.shape[1])
        if save:
            ctx.save_for_backward(
                attention_operand,
                output_weight,
                residual,
                previous,
                norm_weight,
                mix_coefficient,
                key_weight,
                mask,
                packed,
                normalized,
            )
        ctx.mode = mode
        ctx.set_materialize_grads(False)
        return key, state, residual

    @staticmethod
    @once_differentiable
    def backward(ctx, key_grad, state_grad, residual_grad):
        (
            attention_operand,
            output_weight,
            residual,
            previous,
            norm_weight,
            mix_coefficient,
            key_weight,
            mask,
            packed,
            normalized,
        ) = ctx.saved_tensors
        if key_grad is None:
            mixed_grad = torch.zeros_like(residual)
            key_weight_grad = torch.zeros_like(key_weight)
        else:
            mixed_grad, key_weight_grad = projection_backward(
                packed, key_weight, key_grad.reshape(-1, key_weight.shape[1]).float(), ctx.mode
            )
            mixed_grad = mixed_grad.view_as(residual)
            key_weight_grad = key_weight_grad.to(key_weight.dtype)
        state_grad = torch.zeros_like(previous) if state_grad is None else state_grad.contiguous()
        dx, previous_grad, norm_weight_grad, norm_bias_grad, coefficient_grad = norm_shift_backward(
            residual,
            normalized,
            previous,
            norm_weight,
            [mix_coefficient],
            [mixed_grad.contiguous()],
            state_grad,
            mask,
        )
        if residual_grad is not None:
            dx = dx + residual_grad
        attention_grad, output_weight_grad = projection_backward(
            attention_operand, output_weight, dx.reshape(-1, dx.shape[-1]), ctx.mode
        )
        return (
            dx,
            attention_grad.view_as(residual),
            output_weight_grad.to(output_weight.dtype),
            previous_grad,
            norm_weight_grad,
            norm_bias_grad,
            coefficient_grad,
            key_weight_grad,
            None,
            None,
            None,
            None,
        )


def channel_frontend(
    x,
    mixed,
    output_weight,
    previous,
    norm_weight,
    norm_bias,
    mix_coefficient,
    key_weight,
    mask,
    mode,
    backends,
):
    inputs = [
        x.contiguous(),
        mixed.contiguous(),
        output_weight,
        previous.contiguous(),
        norm_weight.flatten().contiguous(),
        norm_bias.flatten().contiguous(),
        mix_coefficient.flatten().contiguous(),
        key_weight,
    ]
    save = torch.is_grad_enabled() and any(value.requires_grad for value in inputs)
    return _ChannelFrontend.apply(*inputs, mask.contiguous(), save, mode, backends)
