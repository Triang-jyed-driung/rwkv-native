"""Masking must remove both state updates and their gradient contributions."""
import pytest
import torch

from conftest import tiny_weights
from rwkv_native import FastRWKV7, State, wkv7
from wkv_reference import wkv7_reference


GROUPS = ("attention", "channel", "wkv")


def assert_rmse(actual, expected, relative_limit, label):
    actual, expected = actual.detach().double(), expected.detach().double()
    assert torch.isfinite(actual).all() and torch.isfinite(expected).all(), label
    error = (actual - expected).square().mean().sqrt().item()
    scale = expected.square().mean().sqrt().item()
    assert error <= relative_limit * scale, (label, error, scale, relative_limit)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("batch,heads,tokens", [(1, 40, 17), (3, 64, 65), (2, 2, 257)])
@pytest.mark.parametrize("layout", ["padding", "holes"])
def test_wkv_masked_backward_against_compacted_fp64(dtype, batch, heads, tokens, layout):
    torch.manual_seed(817)
    shape = (batch, tokens, heads, 64)
    xs = [torch.randn(shape, device="cuda", dtype=dtype) * 0.1 for _ in range(6)]
    xs[4] = -torch.nn.functional.normalize(xs[4].float(), dim=-1).to(dtype)
    xs[5] = -xs[4] * torch.sigmoid(xs[5])
    xs = [x.detach().requires_grad_() for x in xs]
    initial = (torch.randn(batch, heads, 64, 64, device="cuda") * 0.01).requires_grad_()
    before = initial.detach().clone()
    mask = torch.ones(batch, tokens, device="cuda", dtype=torch.bool)
    for row in range(batch):
        mask[row, :3 + row] = False
        mask[row, -2 - row:] = False
        if layout == "holes":
            mask[row, 6 + row::5] = False
    if batch > 1:
        mask[-1] = False
    output, final = wkv7(*xs, initial, mask)
    dy, ds = torch.randn_like(output), torch.randn_like(final)
    # dy is deliberately nonzero at masked positions too.
    grads = torch.autograd.grad((output, final), (*xs, initial), (dy, ds))
    for grad in grads[:6]:
        assert torch.count_nonzero(grad[~mask]) == 0
    assert torch.count_nonzero(output[~mask]) == 0
    assert torch.equal(initial, before)
    tolerance = {torch.float32: 2e-5, torch.float16: 0.001, torch.bfloat16: 0.008}[dtype]
    for row in range(batch):
        valid = mask[row]
        if not valid.any():
            assert torch.equal(final[row], initial[row])
            assert torch.equal(grads[-1][row], ds[row])
            continue
        # The oracle sees no mask at all: physically remove every masked token.
        compact = [x[row:row+1, valid].detach().double().requires_grad_() for x in xs]
        state = initial[row:row+1].detach().double().requires_grad_()
        reference, reference_final = wkv7_reference(*compact, state)
        expected = torch.autograd.grad(
            (reference, reference_final), (*compact, state),
            (dy[row:row+1, valid].double(), ds[row:row+1].double()),
        )
        assert_rmse(output[row:row+1, valid], reference, tolerance, "valid output")
        assert_rmse(final[row:row+1], reference_final, 2e-5, "final state")
        for name, actual, target in zip("rwkvab", grads[:6], expected[:6]):
            assert_rmse(actual[row:row+1, valid], target, tolerance, name)
        assert_rmse(grads[-1][row:row+1], expected[-1], 2e-5, "initial-state gradient")


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("loss_kind", ["output_only", "state_only"])
def test_all_masked_wkv_backward_zero_inputs_and_identity_state(dtype, loss_kind):
    torch.manual_seed(818)
    xs = [torch.randn(3, 17, 64, 64, device="cuda", dtype=dtype).requires_grad_()
          for _ in range(6)]
    initial = torch.randn(3, 64, 64, 64, device="cuda", requires_grad=True)
    mask = torch.zeros(3, 17, device="cuda", dtype=torch.bool)
    output, final = wkv7(*xs, initial, mask)
    ds = torch.randn_like(final)
    if loss_kind == "state_only":
        grads = torch.autograd.grad(final, (*xs, initial), ds)
        expected_state = ds
    else:
        grads = torch.autograd.grad(output, (*xs, initial), torch.randn_like(output))
        expected_state = torch.zeros_like(ds)
    for grad in grads[:6]:
        assert torch.count_nonzero(grad) == 0
    assert torch.equal(grads[-1], expected_state)
    assert torch.equal(final, initial)
    assert torch.count_nonzero(output) == 0


def state_values(state):
    return [value for name in GROUPS for value in getattr(state, name)]


@pytest.mark.parametrize("width", [2560, 4096])
@pytest.mark.parametrize("precision", ["fp16", "bf16"])
@pytest.mark.parametrize("consistency", ["strict", "none"])
@pytest.mark.parametrize("layout", ["padding", "holes"])
def test_model_masking_preserves_parameter_and_initial_state_gradients(
        width, precision, consistency, layout):
    torch.manual_seed(819)
    model = FastRWKV7(tiny_weights(width), precision=precision, consistency=consistency)
    ids = torch.randint(1, 32, (2, 17), device="cuda")
    initial = model.initial_state(2)
    for value in state_values(initial):
        value.normal_(0, 0.01).requires_grad_()
    original_state = [value.detach().clone() for value in state_values(initial)]
    dy = torch.randn(2, 17, 128, device="cuda") * 0.001
    ds = [torch.randn_like(value) * 0.001 for value in state_values(initial)]
    named = list(model.named_parameters())
    leaves = [value for _, value in named] + state_values(initial)
    names = [name for name, _ in named] + [f"initial_state.{i}" for i in range(len(ds))]

    def gradients(tokens, mask, incoming):
        logits, final = model(tokens, initial, mask)
        values = torch.autograd.grad((logits, *state_values(final)), leaves,
                                     (incoming, *ds), allow_unused=True)
        return [None if value is None else value.detach().cpu() for value in values]

    reference = gradients(ids, None, dy)
    # Both variants change the physical sequence length and checkpoint placement.
    positions = (torch.arange(17, device="cuda") + 3 if layout == "padding"
                 else torch.arange(17, device="cuda") * 2 + 3)
    length = int(positions[-1]) + 6
    padded = torch.full((2, length), 127, device="cuda", dtype=ids.dtype)
    mask = torch.zeros_like(padded, dtype=torch.bool)
    padded[:, positions] = ids
    mask[:, positions] = True
    incoming = torch.randn(2, length, 128, device="cuda") * 0.001
    incoming[:, positions] = dy
    actual = gradients(padded, mask, incoming)
    tolerance = 0.003 if precision == "fp16" else 0.02
    for name, a, b in zip(names, actual, reference):
        if a is None or b is None:
            assert a is None and b is None, name
        else:
            assert_rmse(a, b, tolerance, name)
    embedding = actual[names.index("weights.emb__weight")]
    assert torch.count_nonzero(embedding[0]) == 0
    assert torch.count_nonzero(embedding[127]) == 0
    for value, original in zip(state_values(initial), original_state):
        assert torch.equal(value, original)
