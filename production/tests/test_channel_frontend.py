import pytest
import torch

from rwkv_native.model import token_shift
from rwkv_native.ops.channel import channel_frontend
from rwkv_native.ops.gemm import matmul
from rwkv_native.ops.normalization import layer_norm


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("batch,tokens", [(1, 1), (4, 1), (1, 7), (4, 7), (48, 7)])
def test_channel_frontend_composition_and_adjoints(dtype, batch, tokens):
    torch.manual_seed(904)
    c = 64
    x = torch.randn(batch, tokens, c, device="cuda") * 0.1
    mixed = torch.randn_like(x) * 0.1
    output_weight = (torch.randn(c, c, device="cuda", dtype=dtype) * 0.01).T
    previous = torch.randn(batch, c, device="cuda") * 0.01
    norm_weight = torch.randn(c, device="cuda", dtype=dtype) * 0.01 + 1
    norm_bias = torch.randn(c, device="cuda", dtype=dtype) * 0.01
    mix_coefficient = torch.rand(c, device="cuda", dtype=dtype)
    key_weight = (torch.randn(256, c, device="cuda", dtype=dtype) * 0.01).T
    mask = torch.ones(batch, tokens, device="cuda", dtype=torch.bool)
    if tokens == 1:
        mask[::3] = False
    plain = [
        t.detach().clone().requires_grad_()
        for t in (
            x,
            mixed,
            output_weight,
            previous,
            norm_weight,
            norm_bias,
            mix_coefficient,
            key_weight,
        )
    ]
    fused = [t.detach().clone().requires_grad_() for t in plain]
    x, mixed, output_weight, previous, norm_weight, norm_bias, mix_coefficient, key_weight = plain
    residual = x + matmul(mixed, output_weight, backend="fixed")
    normalized = layer_norm(residual, norm_weight, norm_bias)
    past, state = token_shift(normalized, previous, mask)
    key = matmul(normalized + (past - normalized) * mix_coefficient, key_weight, backend="fixed")
    expected = (key, state, residual)
    actual = channel_frontend(*fused, mask, "store", ("fixed", "fixed"))
    for a, b in zip(actual, expected):
        torch.testing.assert_close(a, b, atol=0, rtol=0)
    grads = [torch.randn_like(t) * 0.01 for t in expected]
    ga = torch.autograd.grad(expected, plain, grads)
    gb = torch.autograd.grad(actual, fused, grads)
    for a, b in zip(ga, gb):
        torch.testing.assert_close(a, b, atol=2e-5, rtol=0.02 if dtype == torch.bfloat16 else 0.003)
