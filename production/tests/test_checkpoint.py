"""Checkpointing must preserve features, recurrent state and parameter gradients."""

import pytest
import torch
from conftest import tiny_weights, states_close

from rwkv_native import FastRWKV7


@pytest.mark.parametrize("masked", [False, True])
def test_features_and_block_checkpoint(masked):
    model = FastRWKV7(tiny_weights(), backend="fixed")
    ids = torch.arange(14, device="cuda").reshape(2, 7)
    mask = torch.ones_like(ids, dtype=torch.bool) if masked else None
    if masked:
        mask[:, 1::3] = False
    initial = model.initial_state(2)
    for group in (initial.attention, initial.channel, initial.wkv):
        for value in group:
            value.normal_(0, 0.01)
    with torch.no_grad():
        expected, state = model(ids, initial, mask)
    results = []
    for checkpoint in (False, True):
        model.zero_grad(set_to_none=True)
        features, final = model.forward_features(ids, initial, mask, checkpoint_blocks=checkpoint)
        logits = model.project(features, model.weight("head.weight").T)
        if mask is not None:
            logits = logits.masked_fill(~mask[..., None], 0)
        torch.testing.assert_close(logits, expected, atol=0, rtol=0)
        states_close(final, state)
        loss = logits.square().mean() + sum(
            value.square().mean()
            for group in (final.attention, final.channel, final.wkv)
            for value in group
        )
        loss.backward()
        results.append(
            {name: p.grad.clone() for name, p in model.named_parameters() if p.grad is not None}
        )
    assert results[0].keys() == results[1].keys()
    for name in results[0]:
        torch.testing.assert_close(results[0][name], results[1][name], atol=0, rtol=0)
