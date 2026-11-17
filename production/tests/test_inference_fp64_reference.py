from pathlib import Path
import sys

import torch
from torch.nn import functional as F

from conftest import tiny_weights

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'tools'))
from inference_fp64_check import reference_forward


def test_reference_batch_samples_have_independent_recurrent_states():
    torch.set_num_threads(1)
    weights = tiny_weights()
    ids = torch.tensor([[1, 7, 11], [9, 2, 5]])
    together, states = reference_forward(weights, ids, 2)
    for sample in range(2):
        separate, own_states = reference_forward(weights, ids[sample:sample+1], 2)
        torch.testing.assert_close(together[sample:sample+1], separate, rtol=1e-12, atol=1e-12)
        for group in states:
            for actual, expected in zip(states[group], own_states[group]):
                torch.testing.assert_close(actual[sample:sample+1], expected, rtol=1e-12, atol=1e-12)


def test_zero_attention_and_ffn_outputs_leave_only_embedding_normalization():
    weights = tiny_weights()
    for layer in range(2):
        weights[f'blocks.{layer}.att.output.weight'].zero_()
        weights[f'blocks.{layer}.ffn.value.weight'].zero_()
        weights[f'blocks.{layer}.att.value.weight'].zero_()
    ids = torch.tensor([[1, 2, 3], [4, 5, 6]])
    hidden, states = reference_forward(weights, ids, 2)
    embedding = weights['emb.weight'][ids].double()
    expected = F.layer_norm(F.layer_norm(embedding, (64,), eps=1e-5), (64,), eps=1e-5)
    torch.testing.assert_close(hidden, expected, rtol=0, atol=0)
    assert all(torch.count_nonzero(value).item() == 0 for value in states['wkv'])
