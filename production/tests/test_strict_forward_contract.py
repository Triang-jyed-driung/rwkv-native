"""The public strict contract applies to forward values, not backward ordering."""
import pytest
import torch

from conftest import tiny_weights
from rwkv_native import FastRWKV7, State


GROUPS = ("attention", "channel", "wkv")


def snapshot(output):
    logits, state = output
    return logits.detach().cpu(), State(*[
        [value.detach().cpu() for value in getattr(state, name)] for name in GROUPS
    ])


def exact(actual, expected, label):
    assert actual.shape == expected.shape and actual.dtype == expected.dtype, label
    # Include the FP32 bit patterns, rather than accepting a numerical tolerance.
    assert torch.equal(actual.contiguous().view(torch.uint8),
                       expected.contiguous().view(torch.uint8)), label


def exact_state(actual, expected, rows=None):
    for name in GROUPS:
        for layer, (a, b) in enumerate(zip(getattr(actual, name), getattr(expected, name))):
            exact(a, b if rows is None else b[rows], f"{name}, layer {layer}")


def select_state(state, rows):
    return State(*[[value[rows] for value in getattr(state, name)] for name in GROUPS])


def initial_state(model, batch):
    state = model.initial_state(batch)
    for name in GROUPS:
        for value in getattr(state, name):
            value.normal_(0, 0.01)
    return state


def run(model, ids, state, grad_enabled, mask=None):
    with torch.set_grad_enabled(grad_enabled):
        output = model(ids, state, mask)
        assert output[0].requires_grad == grad_enabled
        return snapshot(output)


@pytest.fixture(scope="module", params=[
    (2560, "fp16"), (2560, "bf16"), (4096, "fp16"), (4096, "bf16")
], ids=lambda item: f"c{item[0]}-{item[1]}")
def model(request):
    width, precision = request.param
    weights = tiny_weights(width)
    ranks = (96, 96, 320, 64) if width == 2560 else (128, 128, 480, 96)
    for layer in range(2):
        for name, rank in zip("wagv", ranks):
            weights[f"blocks.{layer}.att.{name}1"] = torch.randn(width, rank) * 0.02
            weights[f"blocks.{layer}.att.{name}2"] = torch.randn(rank, width) * 0.02
    # Exercise the normal automatic dispatch, including real checkpoint ranks.
    return FastRWKV7(weights, precision=precision, consistency="strict")


@pytest.mark.parametrize("tokens", [17, 65])
def test_same_sequence_at_every_batch_position(model, tokens):
    torch.manual_seed(121)
    ids = torch.randint(128, (5, tokens), device="cuda")
    state = initial_state(model, 5)
    # Distinct companions prevent a cross-sample mixing bug from being hidden
    # by a batch containing only identical sequences and identical states.
    for row in (2, 4):
        ids[row].copy_(ids[0])
        for name in GROUPS:
            for value in getattr(state, name):
                value[row].copy_(value[0])
    before = snapshot((ids, state))[1]
    solo = run(model, ids[:1], select_state(state, slice(0, 1)), False)
    baseline = run(model, ids, state, False)
    order = torch.tensor([1, 0, 3, 2, 4], device="cuda")
    cpu_order = order.cpu()
    for grad_enabled in (False, True):
        actual = run(model, ids, state, grad_enabled)
        exact(actual[0], baseline[0], "train/inference batch logits")
        exact_state(actual[1], baseline[1])
        for row in (0, 2, 4):
            exact(actual[0][row:row+1], solo[0], f"batch position {row}")
            exact_state(select_state(actual[1], slice(row, row+1)), solo[1])
        moved = run(model, ids[order], select_state(state, order), grad_enabled)
        exact(moved[0], baseline[0][cpu_order], "batch permutation logits")
        exact_state(moved[1], baseline[1], cpu_order)
        for row in (1, 3, 4):
            exact(moved[0][row:row+1], solo[0], f"moved batch position {row}")
            exact_state(select_state(moved[1], slice(row, row+1)), solo[1])
    exact_state(snapshot((ids, state))[1], before)


@pytest.mark.parametrize("tokens", [17, 65])
@pytest.mark.parametrize("batch", [1, 3])
def test_single_token_n_times_matches_one_n_token_forward(model, tokens, batch):
    torch.manual_seed(122)
    ids = torch.randint(128, (batch, tokens), device="cuda")
    initial = initial_state(model, batch)
    before = snapshot((ids, initial))[1]
    reference = run(model, ids, initial, False)
    for grad_enabled in (False, True):
        whole = run(model, ids, initial, grad_enabled)
        exact(whole[0], reference[0], "whole train/inference logits")
        exact_state(whole[1], reference[1])
        state, parts = initial, []
        with torch.set_grad_enabled(grad_enabled):
            for step in range(tokens):
                logits, state = model(ids[:, step:step+1], state)
                assert logits.requires_grad == grad_enabled
                parts.append(logits.detach().cpu())
                # This checks forward equivalence, not truncated/full BPTT.
                state = State(*[[value.detach() for value in getattr(state, name)]
                                for name in GROUPS])
        exact(torch.cat(parts, dim=1), reference[0], "T=1 repeated versus T=n")
        exact_state(snapshot((logits, state))[1], reference[1])
    exact_state(snapshot((ids, initial))[1], before)


@pytest.mark.parametrize("prefix,tokens,suffix", [(3, 17, 5), (7, 65, 9), (5, 257, 11)])
def test_unmasked_sequence_matches_prefix_and_suffix_masking(model, prefix, tokens, suffix):
    torch.manual_seed(123)
    ids = torch.randint(128, (3, tokens), device="cuda")
    initial = initial_state(model, 3)
    before = snapshot((ids, initial))[1]
    reference = run(model, ids, initial, False)  # Actually omit attention_mask.
    padded = torch.full((3, prefix+tokens+suffix), 127, device="cuda", dtype=ids.dtype)
    padded[:, prefix:prefix+tokens] = ids
    mask = torch.zeros_like(padded, dtype=torch.bool)
    mask[:, prefix:prefix+tokens] = True
    for grad_enabled in (False, True):
        plain = run(model, ids, initial, grad_enabled)
        actual = run(model, padded, initial, grad_enabled, mask)
        exact(plain[0], reference[0], "unmasked train/inference logits")
        exact_state(plain[1], reference[1])
        exact(actual[0][:, prefix:prefix+tokens], reference[0], "valid tokens after padding")
        assert torch.count_nonzero(actual[0][~mask.cpu()]) == 0
        exact_state(actual[1], reference[1])
    exact_state(snapshot((ids, initial))[1], before)
