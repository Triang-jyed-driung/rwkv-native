import pytest
import torch

from rwkv_native._extensions import extension
from rwkv_native.ops.rank_group import expand_low_rank


@pytest.mark.parametrize("count", [3, 4])
@pytest.mark.parametrize("rows", [1, 4, 96, 1025])
def test_grouped_expansion_tiles_pack_and_fp64(count, rows):
    torch.manual_seed(941)
    ranks = (96, 96, 256, 64)[:count]
    xs = [torch.randn(rows, r, device="cuda") * 0.1 for r in ranks]
    weights = [torch.randn(2048, r, device="cuda", dtype=torch.float16) * 0.01 for r in ranks]
    ext = extension("rank_group")
    packed = ext.pack(xs)
    for g, (x, p) in enumerate(zip(xs, packed)):
        expected = extension("core").activation_forward(x, 2 if g == 0 else 3) if g in (0, 2) else x
        torch.testing.assert_close(p, expected.half(), atol=0, rtol=0)
    small = [y.view(rows, 2048) for y in ext.run(packed, weights, 16).split(rows * 2048)]
    large = [y.view(rows, 2048) for y in ext.run(packed, weights, 64).split(rows * 2048)]
    singles = [
        y.view(1, 2048)
        for y in ext.run([x[-1:].contiguous() for x in packed], weights, 16).split(2048)
    ]
    for x, w, s, large_result, one in zip(packed, weights, small, large, singles):
        torch.testing.assert_close(s, large_result, atol=0, rtol=0)
        torch.testing.assert_close(s[-1:], one, atol=0, rtol=0)
        torch.testing.assert_close(s.double(), x.double() @ w.double().T, atol=3e-6, rtol=3e-5)


@pytest.mark.parametrize("mode", ["store", "fp32"])
def test_grouped_expansion_adjoints(mode):
    torch.manual_seed(942)
    ranks = (96, 96, 256, 64)
    xs = [(torch.randn(2, 7, r, device="cuda") * 0.1).requires_grad_() for r in ranks]
    weights = [
        (torch.randn(2048, r, device="cuda", dtype=torch.float16) * 0.01).T.requires_grad_()
        for r in ranks
    ]
    ys = expand_low_rank(xs, weights, mode)
    grads = [torch.randn_like(y) * 1e-4 for y in ys]
    actual = torch.autograd.grad(ys, (*xs, *weights), grads)
    packed = extension("rank_group").pack(xs)
    for i, (x, w, g, p) in enumerate(zip(xs, weights, grads, packed)):
        dx = (g.double().reshape(-1, 2048) @ w.detach().double().T).reshape_as(x)
        if i in (0, 2):
            a = extension("core").activation_forward(x.detach(), 2 if i == 0 else 3).double()
            dx *= 1 - a * a if i == 0 else a * (1 - a)
        dw = p.double().reshape(-1, ranks[i]).T @ g.double().reshape(-1, 2048)
        torch.testing.assert_close(actual[i].double(), dx, atol=3e-9, rtol=5e-5)
        torch.testing.assert_close(actual[i + 4], dw.half(), atol=2e-7, rtol=0.003)
