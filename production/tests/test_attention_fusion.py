import pytest
import torch
from conftest import tiny_weights

from rwkv_native import FastRWKV7


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("batch", [1, 4, 48])
@pytest.mark.parametrize("tokens", [1, 7])
def test_attention_pack_matches_forward_and_gradients(dtype, batch, tokens):
    torch.manual_seed(902)
    weights = tiny_weights()
    plain = FastRWKV7(weights, dtype=dtype, backend="fixed", fused_attention=False)
    fused = FastRWKV7(weights, dtype=dtype, backend="fixed", fused_attention=True)
    ids = (
        torch.arange(batch, device="cuda")[:, None] + torch.arange(tokens, device="cuda")[None] + 7
    ) % 20
    mask = (torch.arange(batch, device="cuda") % 3 != 1)[:, None] if tokens == 1 else None
    y, s = plain(ids, attention_mask=mask)
    z, t = fused(ids, attention_mask=mask)
    torch.testing.assert_close(z, y, atol=0, rtol=0)
    for name in ("attention", "channel", "wkv"):
        for a, b in zip(getattr(s, name), getattr(t, name)):
            torch.testing.assert_close(a, b, atol=0, rtol=0)
    dy = torch.randn_like(y) * 0.01
    ga = torch.autograd.grad(y, tuple(plain.parameters()), dy, allow_unused=True)
    gb = torch.autograd.grad(z, tuple(fused.parameters()), dy, allow_unused=True)
    for a, b in zip(ga, gb):
        if a is None or b is None:
            assert a is None and b is None
            continue
        assert torch.isfinite(b).all()
        torch.testing.assert_close(a, b, atol=2e-5, rtol=0.02 if dtype == torch.bfloat16 else 0.003)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_attention_prefill_matches_decode_padding_and_reordering(dtype):
    from test_consistent_backend import states_close

    from rwkv_native.model import State

    torch.manual_seed(903)
    model = FastRWKV7(tiny_weights(), dtype=dtype, backend="fixed")
    ids = (
        torch.arange(48, device="cuda")[:, None] + torch.arange(7, device="cuda")[None] + 11
    ) % 128
    initial = model.initial_state(48)
    for name in ("attention", "channel", "wkv"):
        for x in getattr(initial, name):
            x.normal_(0, 0.01)
    with torch.no_grad():
        full, final = model(ids, initial)
        state = initial
        outputs = []
        for t in range(7):
            y, state = model(ids[:, t : t + 1], state)
            outputs.append(y)
        torch.testing.assert_close(torch.cat(outputs, 1), full, atol=0, rtol=0)
        states_close(state, final)
        order = torch.randperm(48, device="cuda")
        shuffled = State(
            *[[x[order] for x in getattr(initial, g)] for g in ("attention", "channel", "wkv")]
        )
        y, state = model(ids[order], shuffled)
        torch.testing.assert_close(y, full[order], atol=0, rtol=0)
        states_close(state, final, order)
        padded = torch.nn.functional.pad(ids, (5, 3), value=-999)
        mask = torch.nn.functional.pad(torch.ones_like(ids, dtype=torch.bool), (5, 3), value=False)
        y, state = model(padded, initial, mask)
        torch.testing.assert_close(y[mask], full.flatten(0, 1), atol=0, rtol=0)
        states_close(state, final)
