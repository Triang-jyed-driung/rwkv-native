import itertools

import pytest
import torch
from conftest import tiny_weights

from rwkv_native import FastRWKV7
from rwkv_native.model import State
from rwkv_native.ops.adjoints import strict_mm
from rwkv_native.ops.gemm import matmul, project_rows


def states_close(a, b, order=None):
    for name in ("attention", "channel", "wkv"):
        for x, y in zip(getattr(a, name), getattr(b, name)):
            torch.testing.assert_close(x, y if order is None else y[order], atol=0, rtol=0)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_strict_mm_ignores_global_precision_flags(dtype):
    torch.manual_seed(1191)
    flags = (
        torch.backends.cuda.matmul.allow_tf32,
        torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction,
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,
    )
    try:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = True
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = True
        x = torch.randn(17, 129, device="cuda", dtype=dtype)
        w = torch.randn(129, 31, device="cuda", dtype=dtype)
        ref = x.double() @ w.double()
        for xc, wc in itertools.product((False, True), repeat=2):
            a = x.T.contiguous().T if xc else x
            b = w.T.contiguous().T if wc else w
            torch.testing.assert_close(strict_mm(a, b).double(), ref, atol=2e-5, rtol=2e-5)
        assert torch.backends.cuda.matmul.allow_tf32
        assert torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction
        assert torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction
    finally:
        torch.backends.cuda.matmul.allow_tf32 = flags[0]
        torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = flags[1]
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = flags[2]


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize(
    "shape", [(0, 8, 13), (1, 8, 13), (48, 71, 67), (96, 64, 128), (769, 33, 7)]
)
def test_fixed_padding_chunking_and_adjoints(dtype, shape):
    torch.manual_seed(1192)
    m, k, n = shape
    x = torch.randn(m, k, device="cuda", requires_grad=True)
    w = (torch.randn(n, k, device="cuda") * 0.1).to(dtype).T.requires_grad_()
    y = matmul(x, w, backend="fixed")
    high = x.detach().to(dtype)
    ref = high.double() @ w.detach().double()
    torch.testing.assert_close(y.double(), ref, atol=2e-5, rtol=2e-5)
    if m:
        order = torch.randperm(m, device="cuda")
        torch.testing.assert_close(
            project_rows(high[order], w, "fixed"), y.detach()[order], atol=0, rtol=0
        )
        for i in (0, m - 1):
            torch.testing.assert_close(
                project_rows(high[i : i + 1], w, "fixed"), y.detach()[i : i + 1], atol=0, rtol=0
            )
    g = torch.randn_like(y) * 1e-5
    dx, dw = torch.autograd.grad(y, (x, w), g)
    rdx = g.double() @ w.detach().double().T
    rdw = high.double().T @ g.double()
    torch.testing.assert_close(dx.double(), rdx, atol=2e-10, rtol=2e-5)
    torch.testing.assert_close(
        dw,
        rdw.to(dtype),
        atol=2e-7 if dtype == torch.float16 else 2e-6,
        rtol=0.003 if dtype == torch.float16 else 0.02,
    )


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_fixed_model_masks_batch_reordering_and_optimizer(dtype):
    model = FastRWKV7(tiny_weights(), dtype=dtype, backend="fixed")
    ids = (torch.arange(7, device="cuda")[None] + torch.arange(48, device="cuda")[:, None]) % 128
    mask = torch.ones_like(ids, dtype=torch.bool)
    mask[:, 0] = False
    mask[::2, 3] = False
    mask[0] = False
    initial = model.initial_state(48)
    for group in ("attention", "channel", "wkv"):
        for x in getattr(initial, group):
            x.normal_(0, 0.01)
    with torch.no_grad():
        full, final = model(ids, initial, mask)
        parts = []
        state = initial
        for t in range(7):
            out, state = model(ids[:, t : t + 1], state, mask[:, t : t + 1])
            parts.append(out)
        torch.testing.assert_close(torch.cat(parts, 1)[mask], full[mask], atol=0, rtol=0)
        states_close(state, final)
        order = torch.randperm(48, device="cuda")
        shuffled = State(
            *[[x[order] for x in getattr(initial, g)] for g in ("attention", "channel", "wkv")]
        )
        out, state = model(ids[order], shuffled, mask[order])
        torch.testing.assert_close(out, full[order], atol=0, rtol=0)
        states_close(state, final, order)
        padded = torch.nn.functional.pad(ids, (7, 3), value=-999)
        pm = torch.nn.functional.pad(mask, (7, 3), value=False)
        out, state = model(padded, initial, pm)
        torch.testing.assert_close(out[pm], full[mask], atol=0, rtol=0)
        states_close(state, final)
    out, state = model(ids, initial, mask)
    torch.testing.assert_close(out, full, atol=0, rtol=0)
    states_close(state, final)
    opt = torch.optim.SGD(model.parameters(), lr=0.01)
    before = model.weight("head.weight").detach().clone()
    loss = torch.nn.functional.cross_entropy(out[mask], ids[mask])
    loss.backward()
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
    opt.step()
    assert not torch.equal(before, model.weight("head.weight"))
    model.to(dtype=torch.bfloat16 if dtype == torch.float16 else torch.float16)
    assert not model.lt_shapes


def test_metadata_mismatch_selects_whole_shape_fallback(monkeypatch):
    from rwkv_native.ops import gemm

    class Plan:
        def info(self):
            return {"id": -2}

    class Library:
        def candidates(self, *args):
            return [Plan()]

    monkeypatch.setattr(
        gemm,
        "selections",
        lambda equivalent=False: (
            {} if equivalent else {"64,64": {"index": 0, "algorithm": {"id": -1}}}
        ),
    )
    monkeypatch.setattr(gemm, "extension", lambda kind: Library())
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device=None: (12, 0))
    with pytest.warns(RuntimeWarning, match="every row count"):
        model = FastRWKV7(tiny_weights())
    assert not model.lt_shapes
    assert model.backend_report[(64, 64)]["backend"] == "fixed CUDA"


def test_other_architecture_policy_uses_fixed_cuda(monkeypatch):
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device=None: (8, 0))
    model = FastRWKV7(tiny_weights())
    assert not model.lt_shapes
    assert all(v["backend"] == "fixed CUDA" for v in model.backend_report.values())


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_strict_mm_preserves_large_partial_sum_cancellation(dtype):
    x = torch.ones(1, 16385, device="cuda", dtype=dtype)
    w = torch.empty(16385, 1, device="cuda", dtype=dtype)
    w[:8192] = 128
    w[8192:16384] = -128
    w[-1] = 1
    torch.testing.assert_close(strict_mm(x, w), torch.ones(1, 1, device="cuda"), atol=0, rtol=0)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_all_masked_fixed_model_keeps_state_and_state_gradients(dtype):
    model = FastRWKV7(tiny_weights(), dtype=dtype, backend="fixed")
    ids = torch.full((2, 3), -999, device="cuda")
    mask = torch.zeros_like(ids, dtype=torch.bool)
    initial = model.initial_state(2)
    leaves = []
    for name in ("attention", "channel", "wkv"):
        for x in getattr(initial, name):
            x.normal_(0, 0.01).requires_grad_()
            leaves.append(x)
    logits, final = model(ids, initial, mask)
    assert logits.count_nonzero() == 0
    states_close(final, initial)
    loss = logits.sum() + sum(
        x.sum() for name in ("attention", "channel", "wkv") for x in getattr(final, name)
    )
    loss.backward()
    for x in leaves:
        torch.testing.assert_close(x.grad, torch.ones_like(x), atol=0, rtol=0)
    assert all(p.grad is None or p.grad.count_nonzero() == 0 for p in model.parameters())


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_fixed_projection_unaligned_contiguous_operands(dtype):
    x = torch.randn(48 * 64 + 1, device="cuda", dtype=dtype)[1:].view(48, 64)
    w = torch.randn(64 * 128 + 1, device="cuda", dtype=dtype)[1:].view(128, 64).T
    assert x.is_contiguous() and x.data_ptr() % 16
    actual = project_rows(x, w, "fixed")
    torch.testing.assert_close(actual, project_rows(x.clone(), w.clone(), "fixed"), atol=0, rtol=0)
    torch.testing.assert_close(actual.double(), x.double() @ w.double(), atol=2e-5, rtol=2e-5)
