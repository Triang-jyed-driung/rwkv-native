import pytest
import torch

from rwkv_native.ops.gradient_pack import combine, pack


def reference(x, g, need_dw):
    _, exponent = torch.frexp(g.abs().amax())
    scale = torch.ldexp(g.new_ones(()), (7 - exponent).clamp(-120, 120))
    scaled = g * scale
    high = scaled.to(x.dtype)
    low = (scaled - high.float()).to(x.dtype)
    parts = [high, low]
    return (
        torch.cat(parts, 0),
        torch.cat([x] * len(parts), 0) if need_dw else x.new_empty((0, x.shape[1])),
        torch.stack((scale.reciprocal(), torch.zeros_like(scale))),
        scale,
    )


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("shape", [(1, 8, 13), (48, 71, 67), (96, 2048, 2048), (48, 64, 65536)])
@pytest.mark.parametrize("need_dw", [False, True])
def test_pack_matches_torch(dtype, shape, need_dw):
    torch.manual_seed(1278)
    m, k, n = shape
    x = torch.randn(m, k, device="cuda", dtype=dtype)
    g = torch.randn(m, n, device="cuda") * 0.00003
    old_x = x.clone()
    old_g = g.clone()
    for a, e in zip(pack(x, g, need_dw), reference(x, g, need_dw)):
        torch.testing.assert_close(a, e, atol=0, rtol=0)
    torch.testing.assert_close(x, old_x, atol=0, rtol=0)
    torch.testing.assert_close(g, old_g, atol=0, rtol=0)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("exponent", [-140, -120, -30, 0, 120])
def test_scale_extremes_and_combine(dtype, exponent):
    x = torch.ones(48, 17, device="cuda", dtype=dtype)
    g = torch.ldexp(torch.randn(48, 31, device="cuda"), torch.tensor(exponent, device="cuda"))
    actual = pack(x, g, True)
    expected = reference(x, g, True)
    for a, e in zip(actual, expected):
        torch.testing.assert_close(a, e, atol=0, rtol=0)
    parts = 2 if dtype == torch.float16 else 3
    products = torch.randn(48 * parts, 17, device="cuda") * 0.001
    chunks = products.split(48, 0)
    correction = chunks[1] if parts == 2 else chunks[1] + chunks[2]
    target = (chunks[0] + correction) / expected[-1]
    torch.testing.assert_close(combine(products, actual[-1], parts), target, atol=0, rtol=0)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_zero_gradients(dtype):
    x = torch.zeros(96, 64, device="cuda", dtype=dtype)
    g = torch.zeros(96, 64, device="cuda")
    for a, e in zip(pack(x, g, True), reference(x, g, True)):
        torch.testing.assert_close(a, e, atol=0, rtol=0)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_ieee_exponents_and_nonfinite_scale(dtype):
    x = torch.ones(1, 1, device="cuda", dtype=dtype)
    values = [0.0, float("inf"), float("nan")] + [2.0**exponent for exponent in range(-149, 128)]
    for value in values:
        g = torch.tensor([[value]], device="cuda", dtype=torch.float32)
        for a, e in zip(pack(x, g, False), reference(x, g, False)):
            torch.testing.assert_close(a, e, atol=0, rtol=0, equal_nan=True)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_unaligned_contiguous_inputs(dtype):
    x = torch.randn(48 * 71 + 1, device="cuda", dtype=dtype)[1:].view(48, 71)
    g = torch.randn(48 * 67 + 1, device="cuda")[1:].view(48, 67)
    assert x.data_ptr() % 16 and g.data_ptr() % 16
    for a, e in zip(pack(x, g, True), reference(x, g, True)):
        torch.testing.assert_close(a, e, atol=0, rtol=0)
