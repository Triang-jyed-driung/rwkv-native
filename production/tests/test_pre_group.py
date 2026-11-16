import pytest
import torch

from rwkv_native._extensions import extension
from rwkv_native.ops.gemm import matmul


@pytest.mark.parametrize("count", [6, 7])
@pytest.mark.parametrize("rows", [1, 17, 96, 257])
def test_pre_group_tiles_partitions_single_and_fp64(count, rows):
    torch.manual_seed(951)
    sizes = (2048, 2048, 2048, 96, 96, 256, 64)[:count]
    xs = [torch.randn(rows, 2048, device="cuda", dtype=torch.float16) * 0.1 for _ in sizes]
    weights = [torch.randn(n, 2048, device="cuda", dtype=torch.float16) * 0.01 for n in sizes]
    ext = extension("pre_group")
    expected = ext.run(xs, weights, 16)
    for mode in (0, 64, -1):
        torch.testing.assert_close(ext.run(xs, weights, mode), expected, atol=0, rtol=0)
    offset = 0
    for x, w, n in zip(xs, weights, sizes):
        actual = expected[offset : offset + rows * n].view(rows, n)
        offset += rows * n
        torch.testing.assert_close(actual.double(), x.double() @ w.double().T, atol=3e-6, rtol=3e-5)
        torch.testing.assert_close(ext.run([x], [w], -1).view(rows, n), actual, atol=0, rtol=0)
        torch.testing.assert_close(ext.run([x[:1]], [w], -1).view(1, n), actual[:1], atol=0, rtol=0)


@pytest.mark.parametrize("n", [96, 2048])
@pytest.mark.parametrize("mode", ["store", "fp32"])
def test_pre_projection_adjoints(n, mode):
    torch.manual_seed(952)
    x = (torch.randn(17, 2048, device="cuda") * 0.1).requires_grad_()
    w = (torch.randn(n, 2048, device="cuda", dtype=torch.float16) * 0.01).T.requires_grad_()
    y = matmul(x, w, backend="pre_group", adjoint_mode=mode)
    g = torch.randn_like(y) * 1e-5
    dx, dw = torch.autograd.grad(y, (x, w), g)
    torch.testing.assert_close(
        y.double(), x.detach().half().double() @ w.detach().double(), atol=3e-6, rtol=3e-5
    )
    torch.testing.assert_close(
        dx.double(), g.double() @ w.detach().double().T, atol=5e-10, rtol=3e-5
    )
    torch.testing.assert_close(
        dw, (x.detach().half().double().T @ g.double()).half(), atol=2e-7, rtol=0.003
    )
