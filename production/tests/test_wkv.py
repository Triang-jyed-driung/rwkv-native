import pytest
import torch
from wkv_reference import wkv7_reference

from rwkv_native import wkv7


@pytest.mark.parametrize("batch,heads", [(1, 40), (1, 64), (2, 40)])
@pytest.mark.parametrize("length", [255, 256, 257, 513])
def test_register_prefetch_matches_training_and_chunked_forward(batch, heads, length):
    from rwkv_native._extensions import extension

    xs, state, mask = inputs(torch.float32, length, B=batch, H=heads)
    # Keep valid tokens in the one-sequence cases, including an incomplete
    # prefetch group. Masked NaNs must not contaminate live state or outputs.
    mask[0] = torch.arange(length, device="cuda") % 7 != 0
    for x in xs:
        x[~mask] = float("nan")
    original_state = state.clone()
    core = extension("core")
    training, training_state, _ = core.forward(*xs, state, mask, 16, True)
    inference, inference_state, _ = core.forward(*xs, state, mask, 16, False)
    assert torch.equal(training, inference)
    assert torch.equal(training_state, inference_state)
    with torch.no_grad():
        split, current = [], state
        for start in range(0, length, 127):
            stop = min(start + 127, length)
            y, current = wkv7(*(x[:, start:stop] for x in xs), current, mask[:, start:stop])
            split.append(y)
        assert torch.equal(torch.cat(split, dim=1), inference)
        assert torch.equal(current, inference_state)
        for sample in range(batch):
            y, final = wkv7(*(x[sample:sample + 1] for x in xs),
                            state[sample:sample + 1], mask[sample:sample + 1])
            assert torch.equal(y, inference[sample:sample + 1])
            assert torch.equal(final, inference_state[sample:sample + 1])
    assert torch.equal(state, original_state)


def inputs(dtype, T, B=2, H=2):
    torch.manual_seed(1234)
    shape = (B, T, H, 64)
    xs = [torch.randn(shape, device="cuda", dtype=dtype) * 0.2 for _ in range(6)]
    xs[4] = -torch.nn.functional.normalize(xs[4].float(), dim=-1).to(dtype)
    xs[5] = -xs[4] * torch.sigmoid(xs[5])
    state = torch.randn(B, H, 64, 64, device="cuda") * 0.1
    mask = torch.rand(B, T, device="cuda") > 0.35
    mask[0] = False  # Entirely padded sequences must preserve state and adjoints.
    return xs, state, mask


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
@pytest.mark.parametrize("T", [1, 15, 16, 17, 65])
def test_forward_backward_against_float64(dtype, T):
    xs, state, mask = inputs(dtype, T)
    actual_inputs = [x.requires_grad_() for x in [*xs, state]]
    expected_inputs = [x.detach().double().requires_grad_() for x in actual_inputs]
    y, final = wkv7(*xs, state, mask)
    yr, sr = wkv7_reference(*expected_inputs, mask)
    torch.testing.assert_close(final.double(), sr, atol=2e-6, rtol=2e-5)
    tol = 0.008 if dtype == torch.bfloat16 else 0.001 if dtype == torch.float16 else 2e-5
    torch.testing.assert_close(y.double(), yr, atol=2e-6, rtol=tol)
    dy, ds = torch.randn_like(y), torch.randn_like(final)
    grads = torch.autograd.grad((y, final), actual_inputs, (dy, ds))
    refs = torch.autograd.grad((yr, sr), expected_inputs, (dy.double(), ds.double()))
    for i, (g, ref) in enumerate(zip(grads, refs)):
        error = (g.double() - ref).abs().max().item()
        limit = (2e-5 if i == 6 else tol) * max(ref.abs().max().item(), 1e-2)
        assert error < limit, (i, error, limit)
    assert torch.count_nonzero(y[~mask]) == 0
    for g in grads[:6]:
        assert torch.count_nonzero(g[~mask]) == 0
    torch.testing.assert_close(final[0], state[0], rtol=0, atol=0)
    torch.testing.assert_close(grads[-1][0], ds[0], rtol=0, atol=0)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_training_prefill_decode_and_chunk_gradients(dtype):
    xs, state, mask = inputs(dtype, 33)
    xs = [x.requires_grad_() for x in xs]
    state.requires_grad_()
    full, final = wkv7(*xs, state, mask)
    streamed, s = [], state
    for start, stop in [(0, 1), (1, 8), (8, 19), (19, 33)]:
        y, s = wkv7(*(x[:, start:stop] for x in xs), s, mask[:, start:stop])
        streamed.append(y)
    torch.testing.assert_close(torch.cat(streamed, 1), full, rtol=0, atol=0)
    torch.testing.assert_close(s, final, rtol=0, atol=0)
    dy, ds = torch.randn_like(full), torch.randn_like(final)
    g1 = torch.autograd.grad((full, final), (*xs, state), (dy, ds))
    g2 = torch.autograd.grad((torch.cat(streamed, 1), s), (*xs, state), (dy, ds))
    for g, other in zip(g1, g2):
        torch.testing.assert_close(
            g, other, rtol=0.008 if dtype == torch.bfloat16 else 0.001, atol=2e-6
        )
    with torch.no_grad():
        s = state.detach()
        decoded = []
        for t in range(33):
            y, s = wkv7(*(x[:, t : t + 1] for x in xs), s, mask[:, t : t + 1])
            decoded.append(y)
    torch.testing.assert_close(torch.cat(decoded, 1), full, rtol=0, atol=0)
    torch.testing.assert_close(s, final, rtol=0, atol=0)


def test_padding_nan_is_not_read_and_state_only_loss():
    xs, state, mask = inputs(torch.float32, 19)
    for x in xs:
        x[~mask] = float("nan")
        x.requires_grad_()
    state.requires_grad_()
    y, final = wkv7(*xs, state, mask, checkpoint_size=7)
    final.square().sum().backward()
    assert torch.isfinite(y).all() and torch.isfinite(final).all()
    for x in xs:
        assert torch.isfinite(x.grad).all()
        assert torch.count_nonzero(x.grad[~mask]) == 0


def test_mask_is_equivalent_to_removing_padding():
    xs, state, mask = inputs(torch.bfloat16, 31, B=1)
    mask[0, 3:21:2] = True
    y, final = wkv7(*xs, state, mask)
    compact, compact_state = wkv7(*(x[:, mask[0]] for x in xs), state)
    torch.testing.assert_close(y[:, mask[0]], compact, rtol=0, atol=0)
    torch.testing.assert_close(final, compact_state, rtol=0, atol=0)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("batch,length", [(2, 17), (48, 1), (96, 1)])
def test_large_batch_dispatch_backward_and_decode(dtype, batch, length):
    xs, state, mask = inputs(dtype, length, B=batch, H=32)
    actual_inputs = [x.requires_grad_() for x in (*xs, state)]
    reference_inputs = [x.detach().double().requires_grad_() for x in actual_inputs]
    y, s = wkv7(*actual_inputs, mask)
    yr, sr = wkv7_reference(*reference_inputs, mask)
    tol = 0.008 if dtype == torch.bfloat16 else 0.001 if dtype == torch.float16 else 2e-5
    torch.testing.assert_close(y.double(), yr, atol=2e-6, rtol=tol)
    torch.testing.assert_close(s.double(), sr, atol=2e-6, rtol=2e-5)
    dy, ds = torch.randn_like(y), torch.randn_like(s)
    ga = torch.autograd.grad((y, s), actual_inputs, (dy, ds))
    gr = torch.autograd.grad((yr, sr), reference_inputs, (dy.double(), ds.double()))
    for i, (g, ref) in enumerate(zip(ga, gr)):
        assert (g.double() - ref).abs().max() < (2e-5 if i == 6 else tol) * max(
            ref.abs().max().item(), 1e-2
        )
    with torch.no_grad():
        current = state
        decoded = []
        for t in range(length):
            out, current = wkv7(*(x[:, t : t + 1] for x in xs), current, mask[:, t : t + 1])
            decoded.append(out)
        torch.testing.assert_close(torch.cat(decoded, 1), y, atol=2e-6, rtol=tol)
        torch.testing.assert_close(current, s, atol=2e-6, rtol=2e-5)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_head_batch_dispatch_is_bitwise_invariant(dtype):
    # B*H=64 uses warp rows; duplicating the batch crosses into shared tiles.
    xs, state, mask = inputs(dtype, 65, B=1, H=64)
    mask[:] = True
    mask[:, 4::7] = False
    for x in xs:
        x[~mask] = float("nan")
        x.requires_grad_()
    y, final = wkv7(*xs, state, mask)
    with torch.no_grad():
        batched, states = wkv7(*(x.repeat(2, 1, 1, 1) for x in xs),
                              state.repeat(2, 1, 1, 1), mask.repeat(2, 1))
    for i in range(2):
        torch.testing.assert_close(batched[i:i+1], y, atol=0, rtol=0)
        torch.testing.assert_close(states[i:i+1], final, atol=0, rtol=0)


@pytest.mark.parametrize("checkpoint_size", [1, 7, 64, 256])
def test_checkpoint_size_preserves_fp32_gradients(checkpoint_size):
    xs, state, mask = inputs(torch.float32, 65)
    values = [x.requires_grad_() for x in (*xs, state)]
    y, s = wkv7(*values, mask, checkpoint_size=checkpoint_size)
    base, bs = wkv7(*values, mask, checkpoint_size=16)
    dy, ds = torch.randn_like(y), torch.randn_like(s)
    grads = torch.autograd.grad((y, s), values, (dy, ds))
    baseline = torch.autograd.grad((base, bs), values, (dy, ds))
    torch.testing.assert_close(y, base, atol=0, rtol=0)
    torch.testing.assert_close(s, bs, atol=0, rtol=0)
    for actual, expected in zip(grads, baseline):
        torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-5)


@pytest.mark.parametrize("batch,heads", [(1, 1), (16, 32)])
def test_state_axes_are_key_then_value(batch, heads):
    """An asymmetric sparse recurrence independently fixes the public state axes."""
    import math

    xs = [torch.zeros(batch, 1, heads, 64, device="cuda") for _ in range(6)]
    r, w, k, v, a, b = xs
    r[..., 2] = 1
    k[..., 7] = 2
    v[..., 11] = 6
    initial = torch.zeros(batch, heads, 64, 64, device="cuda")
    initial[..., 2, 5] = 0.75
    initial.requires_grad_()
    y, final = wkv7(*xs, initial)
    decay = math.exp(-math.exp(-0.5) * 0.5)
    expected_y = torch.zeros_like(y)
    expected_y[..., 5] = 0.75 * decay
    expected_s = torch.zeros_like(final)
    expected_s[..., 2, 5] = 0.75 * decay
    expected_s[..., 7, 11] = 12
    torch.testing.assert_close(y, expected_y)
    torch.testing.assert_close(final, expected_s)
    (grad,) = torch.autograd.grad(y[..., 5].sum(), initial)
    expected_grad = torch.zeros_like(initial)
    expected_grad[..., 2, 5] = decay
    torch.testing.assert_close(grad, expected_grad)


@pytest.mark.parametrize("batch,tokens", [(48, 1), (96, 7), (48, 17)])
def test_unused_final_state_adjoint(batch, tokens):
    """The empty-adjoint bridge must retain the full initial-state gradient."""
    xs, state, mask = inputs(torch.float32, tokens, B=batch, H=2)
    leaves = [x.requires_grad_() for x in (*xs, state)]
    y, final = wkv7(*leaves, mask)
    dy = torch.randn_like(y)
    actual = torch.autograd.grad(y, leaves, dy)
    y, final = wkv7(*leaves, mask)
    ds = torch.zeros_like(final)
    expected = torch.autograd.grad((y, final), leaves, (dy, ds))
    for a, e in zip(actual, expected):
        torch.testing.assert_close(a, e, atol=0, rtol=0)
    assert torch.count_nonzero(ds) == 0
    assert actual[-1].shape == state.shape


def test_unaligned_vectors_backward_matches_aligned():
    """Contiguous views need not have the 16-byte alignment required by cp.async."""
    xs, state, mask = inputs(torch.float32, 17, B=1, H=2)
    mask.fill_(True)
    unaligned = []
    for x in xs:
        view = torch.empty(x.numel() + 1, device="cuda")[1:].view_as(x)
        view.copy_(x)
        assert view.is_contiguous() and view.data_ptr() % 16
        unaligned.append(view.requires_grad_())
    aligned = [x.detach().clone().requires_grad_() for x in xs]
    y, s = wkv7(*unaligned, state, mask)
    expected, final = wkv7(*aligned, state, mask)
    torch.testing.assert_close(y, expected, atol=0, rtol=0)
    torch.testing.assert_close(s, final, atol=0, rtol=0)
    dy, ds = torch.randn_like(y), torch.randn_like(s)
    actual_grads = torch.autograd.grad((y, s), unaligned, (dy, ds))
    expected_grads = torch.autograd.grad((expected, final), aligned, (dy, ds))
    for actual, reference in zip(actual_grads, expected_grads):
        torch.testing.assert_close(actual, reference, atol=2e-6, rtol=2e-5)
