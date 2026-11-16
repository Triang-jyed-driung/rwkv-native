"""Fused attention normalization, token mixing and input projections."""

import torch
from torch.autograd.function import once_differentiable

from .._extensions import extension
from .adjoints import projection_backward
from .gemm import project_rows
from .normalization import norm_shift_backward

# R,K,V,W1,A1,G1,(V1) select branches of the checkpoint's R,W,K,V,A,G mixes.
PROJECTION_BRANCHES = (0, 2, 3, 1, 4, 5, 3)


def _pack_attention(inputs, mask, save):
    if inputs[0].shape[1] > 1:
        normalized = extension("ops").norm(inputs[0], inputs[2], inputs[3])
        packed, state = extension("attention").prefill(normalized, inputs[1], inputs[4:])
        return packed, normalized if save else normalized.new_empty(0), state
    # B1 uses independent CTAs per branch, with the same normalization order.
    return extension("attention").run(inputs, mask, save, inputs[0].shape[0] == 1)


class _AttentionProjections(torch.autograd.Function):
    @staticmethod
    def forward(ctx, mask, mode, save, backends, *inputs):
        x, previous, norm_weight, norm_bias, *tail = inputs
        coefficients, weights = tail[:6], tail[6:]
        packed, normalized, state = _pack_attention(
            [x, previous, norm_weight, norm_bias, *coefficients], mask, save
        )
        operands = [
            packed[branch].reshape(-1, x.shape[-1])
            for branch in PROJECTION_BRANCHES[: len(weights)]
        ]
        expected_widths = (2048, 2048, 2048, 96, 96, 256, 64)[: len(weights)]
        if (
            all(backend == "pre_group" for backend in backends)
            and tuple(weight.shape[1] for weight in weights) == expected_widths
        ):
            flat = extension("pre_group").run(
                operands, [weight.T.contiguous() for weight in weights], -1
            )
            outputs = flat.split(
                [operand.shape[0] * weight.shape[1] for operand, weight in zip(operands, weights)]
            )
        else:
            outputs = [
                project_rows(operand, weight, backend)
                for operand, weight, backend in zip(operands, weights, backends)
            ]
        if save:
            ctx.save_for_backward(
                x, previous, norm_weight, normalized, *coefficients, mask, packed, *weights
            )
        ctx.mode = mode
        ctx.set_materialize_grads(False)
        return (
            *[
                output.view(*x.shape[:2], weight.shape[1])
                for output, weight in zip(outputs, weights)
            ],
            state,
        )

    @staticmethod
    @once_differentiable
    def backward(ctx, *gradients):
        x, previous, norm_weight, normalized, *saved = ctx.saved_tensors
        coefficients, mask, packed, weights = saved[:6], saved[6], saved[7], saved[8:]
        mix_grads, weight_grads = [None] * 6, []
        for branch, weight, grad in zip(PROJECTION_BRANCHES, weights, gradients[:-1]):
            if grad is None:
                weight_grads.append(torch.zeros_like(weight))
                continue
            mixed_grad, weight_grad = projection_backward(
                packed[branch].reshape(-1, weight.shape[0]),
                weight,
                grad.reshape(-1, weight.shape[1]).contiguous(),
                ctx.mode,
            )
            mixed_grad = mixed_grad.view_as(x)
            mix_grads[branch] = (
                mixed_grad if mix_grads[branch] is None else mix_grads[branch] + mixed_grad
            )
            weight_grads.append(weight_grad.to(weight.dtype))
        mix_grads = [x.new_empty(0) if grad is None else grad.contiguous() for grad in mix_grads]
        state_grad = (
            torch.zeros_like(previous) if gradients[-1] is None else gradients[-1].contiguous()
        )
        input_grads = norm_shift_backward(
            x, normalized, previous, norm_weight, coefficients, mix_grads, state_grad, mask
        )
        return None, None, None, None, *input_grads, *weight_grads


def attention_frontend(
    x, previous, norm_weight, norm_bias, coefficients, weights, mask, mode, backends
):
    """Masked T>1 prefill uses the model's explicit token-shift composition."""
    inputs = [
        x.contiguous(),
        previous.contiguous(),
        norm_weight.flatten().contiguous(),
        norm_bias.flatten().contiguous(),
        *[coefficient.flatten().contiguous() for coefficient in coefficients],
        *weights,
    ]
    save = torch.is_grad_enabled() and any(value.requires_grad for value in inputs)
    *projected, state = _AttentionProjections.apply(
        mask.contiguous(), mode, save, backends, *inputs
    )
    return *projected[:3], projected[3:], state
