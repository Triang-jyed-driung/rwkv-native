"""FP32 normalization, activations and shared token-shift adjoints."""

import torch
from torch.autograd.function import once_differentiable

from .._extensions import extension

RELU_SQUARED, TANH, SIGMOID = 1, 2, 3


def layer_norm_backward(value, weight, grad):
    dx, dw, db = extension("core").layer_norm_backward(
        value, weight.float(), grad.contiguous(), 1e-5
    )
    return dx, dw.to(weight.dtype), db.to(weight.dtype)


def norm_shift_backward(value, normalized, previous, weight, coefficients, grads, state_grad, mask):
    """Shared attention/channel adjoint; masked decode preserves the old shift."""
    decode = value.shape[1] == 1
    normalized_grad, previous_grad, *coefficient_grads = extension("core").mix_backward(
        normalized,
        previous,
        [coefficient.float() for coefficient in coefficients],
        grads,
        torch.where(mask, state_grad, 0) if decode else state_grad,
    )
    if decode:
        previous_grad = previous_grad + torch.where(mask, 0, state_grad)
    value_grad, weight_grad, bias_grad = layer_norm_backward(value, weight, normalized_grad)
    return (
        value_grad,
        previous_grad,
        weight_grad,
        bias_grad,
        *[grad.to(coefficient.dtype) for grad, coefficient in zip(coefficient_grads, coefficients)],
    )


class _LayerNorm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, value, weight, bias):
        ctx.save_for_backward(value, weight)
        return extension("ops").norm(value, weight, bias)

    @staticmethod
    @once_differentiable
    def backward(ctx, grad):
        return layer_norm_backward(*ctx.saved_tensors, grad)


def layer_norm(value, weight, bias):
    return _LayerNorm.apply(value.float().contiguous(), weight, bias)


def activation_backward(value, grad, kind):
    return extension("core").activation_backward(value, grad.contiguous(), kind)


class _Activation(torch.autograd.Function):
    @staticmethod
    def forward(ctx, value, kind):
        ctx.save_for_backward(value)
        ctx.kind = kind
        return extension("core").activation_forward(value, kind)

    @staticmethod
    @once_differentiable
    def backward(ctx, grad):
        return activation_backward(ctx.saved_tensors[0], grad, ctx.kind), None


def activate(value, kind):
    return _Activation.apply(value.float().contiguous(), kind)


class _ValueBlend(torch.autograd.Function):
    @staticmethod
    def forward(ctx, value, first, bias, raw):
        ctx.save_for_backward(value, first, bias, raw)
        return extension("ops").value_blend(value, first, bias, raw)

    @staticmethod
    @once_differentiable
    def backward(ctx, grad):
        value, first, bias, raw = ctx.saved_tensors
        value_grad, first_grad, bias_grad, raw_grad = extension("core").vres_backward(
            value, first, bias.float(), raw, grad.contiguous()
        )
        return value_grad, first_grad, bias_grad.to(bias.dtype), raw_grad


def value_blend(value, first, bias, raw):
    return _ValueBlend.apply(
        value.contiguous(), first.contiguous(), bias.flatten().contiguous(), raw.contiguous()
    )
