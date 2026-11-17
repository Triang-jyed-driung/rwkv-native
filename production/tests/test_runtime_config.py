"""Public execution options and their observable numerical/autograd contracts."""

import pytest
import torch
from conftest import tiny_weights, states_close

from rwkv_native import FastRWKV7, RuntimeConfig, wkv7
from rwkv_native.ops.gemm import matmul


@pytest.mark.parametrize("kwargs", [
    {"precision": "fp32"}, {"consistency": "invalid"},
    {"precision": "fp16", "dtype": torch.bfloat16}, {"dtype": torch.float32},
])
def test_invalid_options_fail_before_loading_weights(kwargs):
    with pytest.raises(ValueError):
        FastRWKV7({}, **kwargs)


@pytest.mark.parametrize("precision", ["fp16", "bf16"])
@pytest.mark.parametrize("consistency", ["strict", "none"])
@pytest.mark.parametrize("masked", [False, True])
def test_grad_switch_precision_and_state(precision, consistency, masked):
    weights = tiny_weights()
    train = FastRWKV7(weights, precision=precision, consistency=consistency)
    ids = torch.arange(14, device="cuda").reshape(2, 7)
    mask = None
    if masked:
        mask = torch.ones_like(ids, dtype=torch.bool)
        mask[:, 2] = False
    state = train.initial_state(2)
    for group in (state.attention, state.channel, state.wkv):
        for value in group:
            value.normal_(0, 0.01).requires_grad_()
    snapshots = [[x.detach().clone() for x in g] for g in (state.attention, state.channel, state.wkv)]
    y, final = train(ids, state, mask)
    with torch.no_grad():
        z, other = train(ids, state, mask)
    with torch.inference_mode():
        iz, istate = train(ids, state, mask)
    torch.testing.assert_close(iz, z, atol=0, rtol=0)
    states_close(istate, other)
    assert y.requires_grad and not z.requires_grad
    assert train.weight("emb.weight").dtype == train.config.dtype
    assert all(not x.requires_grad and x.dtype == torch.float32
               for g in (other.attention, other.channel, other.wkv) for x in g)
    if masked:
        assert not z[~mask].count_nonzero()
    if consistency == "strict":
        torch.testing.assert_close(z, y, atol=0, rtol=0)
        states_close(other, final)
        parts, current = [], state
        with torch.no_grad():
            for t in range(ids.shape[1]):
                out, current = train(ids[:, t:t+1], current, mask[:, t:t+1] if masked else None)
                parts.append(out)
        torch.testing.assert_close(torch.cat(parts, 1), z, atol=0, rtol=0)
        states_close(current, other)
    else:
        assert set(train._projection_policy.backends.values()) == {"torch"}
        torch.testing.assert_close(z, y, atol=2e-5, rtol=2e-5)
    for original, saved in zip((state.attention, state.channel, state.wkv), snapshots):
        for x, snapshot in zip(original, saved):
            torch.testing.assert_close(x, snapshot, atol=0, rtol=0)
    (y.square().mean() + sum(x.square().mean() for x in final.wkv)).backward()
    assert train.weight("head.weight").grad is not None
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in train.parameters())
    with torch.no_grad():
        features, _ = train.forward_features(ids, state, mask, checkpoint_blocks=True)
        assert not features.requires_grad
        assert not train.project(features.requires_grad_(), train.weight("head.weight").T).requires_grad
    # eval() alone is not an inference execution switch.
    train.eval()
    assert train.project(features, train.weight("head.weight").T).requires_grad


def test_dtype_alias_conversion_and_fixed_override():
    model = FastRWKV7(tiny_weights(), dtype=torch.bfloat16, consistency="none", backend="fixed")
    assert model.config == RuntimeConfig("bf16", "none")
    assert set(model._projection_policy.backends.values()) == {"fixed"}
    model.to(dtype=torch.float16)
    assert model.config.precision == "fp16"
    assert FastRWKV7(tiny_weights()).config == RuntimeConfig()


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_unrestricted_projection_forward_and_gradients(dtype):
    torch.manual_seed(103)
    x = torch.randn(17, 64, device="cuda", requires_grad=True)
    w = (torch.randn(64, 32, device="cuda") * 0.1).to(dtype).requires_grad_()
    y = matmul(x, w, backend="torch")
    operand = x.detach().to(dtype)
    ref = operand.double() @ w.detach().double()
    torch.testing.assert_close(y.double(), ref, atol=2e-5, rtol=2e-5)
    dy = torch.randn_like(y) * 1e-4
    dx, dw = torch.autograd.grad(y, (x, w), dy)
    torch.testing.assert_close(dx.double(), dy.double() @ w.detach().double().T,
                               atol=2e-8, rtol=2e-4)
    torch.testing.assert_close(dw, (operand.double().T @ dy.double()).to(dtype),
                               atol=2e-6, rtol=0.02)


def test_no_grad_wkv_never_requests_checkpoints(monkeypatch):
    # Inputs may retain requires_grad inside no_grad (e.g. caller-provided state).
    import importlib
    module = importlib.import_module("rwkv_native.ops.wkv")
    real = module.extension("core")
    saved = []

    class Spy:
        def forward(self, *args):
            saved.append(args[-1])
            return real.forward(*args)

    monkeypatch.setattr(module, "extension", lambda kind: Spy())
    xs = [torch.zeros(1, 17, 1, 64, device="cuda", requires_grad=True) for _ in range(6)]
    state = torch.zeros(1, 1, 64, 64, device="cuda", requires_grad=True)
    with torch.no_grad():
        y, final = wkv7(*xs, state)
    assert saved == [False]
    assert not y.requires_grad and not final.requires_grad
